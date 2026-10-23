"""Isaac Lab / PhysX bridge. Import after AppLauncher starts Kit."""

import math
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from .core import Layout


def tensor(x):
    return x if isinstance(x, torch.Tensor) else x.torch


def from_physx(state):
    """Isaac Lab 3 state is xyzw; internal/replay contract is wxyz."""
    result = state.clone()
    result[..., 3:7] = state[..., [6, 3, 4, 5]]
    return result


def actuator_gains(actuator):
    """Honor supplier PD gains; use the frequency rule only when both are absent."""
    if ("stiffness" in actuator) != ("damping" in actuator):
        raise ValueError("Provide both stiffness and damping")
    inertia = float(actuator["armature"])
    stiffness = float(actuator.get("stiffness", inertia * (20 * math.pi) ** 2))
    damping = float(actuator.get("damping", 4 * inertia * 20 * math.pi))
    if not all(math.isfinite(v) for v in (inertia, stiffness, damping)):
        raise ValueError("Nonfinite actuator parameters")
    if inertia < 0 or stiffness <= 0 or damping < 0:
        raise ValueError("Armature/damping must be nonnegative and stiffness positive")
    return stiffness, damping


def stage_physx_asset(source, destination):
    """Exclude source scene planes; PhysX supplies one shared ground at z=0.

    Robot bodies, joints and geometry stay unchanged. The original UMR asset
    remains authoritative for kinematics and is never rewritten.
    """
    tree = ET.parse(source)
    root = tree.getroot()
    world = root.find("worldbody")
    if world is not None:
        for plane in list(world.findall('geom[@type="plane"]')):
            world.remove(plane)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    for attribute in ("meshdir", "texturedir"):
        compiler.set(attribute, str((source.parent / compiler.get(attribute, "")).resolve()))
    for include in root.iter("include"):
        include.set("file", str((source.parent / include.attrib["file"]).resolve()))
    tree.write(destination, encoding="unicode")
    return destination


class PhysX:
    def __init__(self, cfg, names, num_envs, device, randomize=False):
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab import sim
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

        asset = Path(os.environ[cfg["asset_env"]]).resolve()
        if not asset.is_file():
            raise FileNotFoundError(asset)
        # Match Isaac Lab's 4096-environment humanoid example. Defaults (2**21)
        # overflow aggregate pairs at this scale and silently drop interactions.
        self.sim = sim.SimulationContext(sim.SimulationCfg(
            dt=0.005, device=device,
            physics=PhysxCfg(
                gpu_found_lost_pairs_capacity=2**23,
                gpu_total_aggregate_pairs_capacity=2**23,
            ),
        ))
        acts = {}
        for i, a in enumerate(cfg["actuators"]):
            inertia = a["armature"]
            stiffness, damping = actuator_gains(a)
            acts[str(i)] = ImplicitActuatorCfg(
                joint_names_expr=[a["pattern"]],
                stiffness=stiffness,
                damping=damping,
                armature=inertia,
                effort_limit_sim=a["effort"],
                velocity_limit_sim=a["velocity"],
            )
        scene_cfg = InteractiveSceneCfg(
            num_envs=num_envs, env_spacing=4.0, replicate_physics=True
        )
        scene_cfg.ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim.GroundPlaneCfg(
                physics_material=sim.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
                    friction_combine_mode="multiply"
                )
            ),
        )
        spawn_options = {
            "asset_path": str(asset),
            "fix_base": False,
            "physics_material": sim.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
                    friction_combine_mode="multiply"
            ),
            "rigid_props": sim.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=1.0
            ),
            "articulation_props": sim.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        }
        if cfg.get("asset_format", "urdf") == "mjcf":
            self._asset_directory = tempfile.TemporaryDirectory(prefix="supermimic-physx-")
            imported_asset = stage_physx_asset(asset, Path(self._asset_directory.name) / asset.name)
            spawn_options["asset_path"] = str(imported_asset)
            spawn = sim.MjcfFileCfg(
                **spawn_options, self_collision=True, collision_from_visuals=False
            )
        else:
            spawn = sim.UrdfFileCfg(
                **spawn_options,
                merge_fixed_joints=True,
                joint_drive=sim.UrdfFileCfg.JointDriveCfg(
                    gains=sim.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=0.0, damping=0.0
                    )
                ),
            )
        scene_cfg.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=spawn,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.8), joint_pos=cfg["default_pose"]
            ),
            actuators=acts,
        )
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()
        self.robot = self.scene["robot"]
        if not type(self.robot).__module__.startswith("isaaclab_physx."):
            raise RuntimeError(f"Expected PhysX backend, got {type(self.robot)}")
        self.device = device
        jnames, bnames = names
        if set(jnames) != set(self.robot.joint_names):
            raise ValueError(
                f"Joint mismatch: motion={jnames}, PhysX={self.robot.joint_names}"
            )
        self.jids = [self.robot.joint_names.index(j) for j in jnames]
        self.bids = [self.robot.body_names.index(b) for b in bnames]
        self.layout = Layout(len(jnames), len(bnames))
        self.default = tensor(self.robot.data.default_joint_pos)[0, self.jids].clone()
        self.limits = tensor(self.robot.data.soft_joint_pos_limits)[
            0, self.jids
        ].clone()
        self.scale = torch.empty(len(jnames), device=device)
        for j, name in enumerate(jnames):
            matches = [a for a in cfg["actuators"] if re.fullmatch(a["pattern"], name)]
            if len(matches) != 1:
                raise ValueError(f"Nonunique actuator mapping: {name}")
            a = matches[0]
            stiffness, _ = actuator_gains(a)
            self.scale[j] = 0.25 * a["effort"] / stiffness
        self.randomize = randomize
        self.context = torch.zeros((num_envs, 1 + len(jnames) + 3), device=device)
        self.context[:, 0] = 1
        if randomize:
            self.torso_id = self.robot.body_names.index(cfg["torso_body"])
            self.nominal_com = tensor(self.robot.data.body_com_pose_b)[:, self.torso_id:self.torso_id + 1].clone()
            # Reuse a finite pool as Isaac Lab does: PhysX allows 64k materials.
            self.friction_buckets = 0.3 + 0.9 * torch.rand(64, device=device)
        self.scene.update(0.005)

    def state(self):
        d = self.robot.data
        root = from_physx(tensor(d.root_link_state_w))
        root[:, :3] -= tensor(self.scene.env_origins)
        bodies = from_physx(tensor(d.body_link_state_w)[:, self.bids])
        bodies[..., :3] -= tensor(self.scene.env_origins)[:, None, :]
        return torch.cat(
            (
                root,
                tensor(d.joint_pos)[:, self.jids],
                tensor(d.joint_vel)[:, self.jids],
                bodies.flatten(1),
            ),
            1,
        )

    def apply_context(self, ids, context):
        """Set absolute episode parameters; repeated calls do not accumulate COM shifts."""
        import warp as wp

        self.context[ids] = context
        materials = wp.to_torch(self.robot.root_view.get_material_properties())
        cpu_ids = ids.to(device="cpu", dtype=torch.int32)
        materials[cpu_ids, :, :2] = context[:, 0].cpu()[:, None, None]
        self.robot.root_view.set_material_properties(
            wp.from_torch(materials, dtype=wp.float32), wp.from_torch(cpu_ids, dtype=wp.int32)
        )
        com = self.nominal_com[ids].clone()
        com[:, 0, :3] += context[:, -3:]
        self.robot.set_coms_index(coms=com, body_ids=[self.torso_id], env_ids=ids)

    def randomization_state(self):
        return {"context": self.context, "friction_buckets": self.friction_buckets}

    def restore_randomization(self, saved):
        self.friction_buckets = saved["friction_buckets"].clone()
        self.apply_context(torch.arange(len(self.context), device=self.device), saved["context"])

    def verify_context(self):
        """Read actual PhysX model properties, rather than trusting requested values."""
        import warp as wp

        materials = wp.to_torch(self.robot.root_view.get_material_properties())
        expected = self.context[:, 0].cpu()[:, None, None].expand_as(materials[..., :2])
        torch.testing.assert_close(materials[..., :2], expected)
        actual = wp.to_torch(self.robot.root_view.get_coms())[:, self.torso_id, :3].to(self.device)
        expected_com = self.nominal_com[:, 0, :3] + self.context[:, -3:]
        torch.testing.assert_close(actual, expected_com)
        return {"material_readback": True, "com_readback": True, "context_dim": self.context.shape[1]}

    def reset(self, ids, state, resample_context=True):
        if self.randomize and resample_context:
            context = self.context.new_empty((len(ids), self.context.shape[1]))
            context[:, 0] = self.friction_buckets[torch.randint(64, (len(ids),), device=self.device)]
            context[:, 1:-3] = (2 * torch.rand_like(context[:, 1:-3]) - 1) * 0.01
            context[:, -3:] = (2 * torch.rand_like(context[:, -3:]) - 1) * context.new_tensor([0.025, 0.05, 0.05])
            self.apply_context(ids, context)
        root, q, v, _ = self.layout.parts(state)
        root = root.clone()
        root[:, :3] += tensor(self.scene.env_origins)[ids]
        root[:, 3:7] = root[:, [4, 5, 6, 3]]
        jp = tensor(self.robot.data.default_joint_pos)[ids].clone()
        jv = torch.zeros_like(jp)
        jp[:, self.jids] = q
        jv[:, self.jids] = v
        self.robot.write_joint_state_to_sim(jp, jv, env_ids=ids)
        self.robot.write_root_link_pose_to_sim_index(root_pose=root[:, :7], env_ids=ids)
        # The installed Lab 3 link-velocity setter forwards its link buffer to
        # PhysX's COM API. Use the explicit COM setter with rigid-body conversion.
        offset = tensor(self.robot.data.root_com_pose_w)[ids, :3] - root[:, :3]
        com_velocity = root[:, 7:13].clone()
        com_velocity[:, :3] += torch.cross(root[:, 10:13], offset, dim=-1)
        self.robot.write_root_com_velocity_to_sim_index(
            root_velocity=com_velocity, env_ids=ids
        )
        self.robot.reset(ids)
        self.scene.update(0.005)

    def step(self, action):
        self.robot.set_joint_position_target(
            self.default + self.context[:, 1:-3] + self.scale * action, joint_ids=self.jids
        )
        for _ in range(4):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(0.005)
        state = self.state()
        if not torch.isfinite(state).all():
            raise FloatingPointError("Nonfinite PhysX state")
        return state
