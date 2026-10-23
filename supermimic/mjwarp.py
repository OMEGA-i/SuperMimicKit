"""Native MuJoCo Warp simulation with zero-copy PyTorch state access."""
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from .core import Layout, rotate_inverse
from .physx import actuator_gains, stage_physx_asset


def build_model(cfg, names):
    import mujoco as mj

    source = Path(os.environ[cfg['asset_env']]).resolve()
    with tempfile.TemporaryDirectory(prefix='supermimic-mjwarp-') as temp:
        path = stage_physx_asset(source, Path(temp) / source.name)
        tree = ET.parse(path)
        root = tree.getroot()
        world = root.find('worldbody')
        ET.SubElement(world, 'geom', name='supermimic_ground', type='plane', size='0 0 .1',
                      friction='1 .005 .0001', contype='1', conaffinity='1')
        for old in list(root.findall('actuator')):
            root.remove(old)
        actuators = ET.SubElement(root, 'actuator')
        params = []
        for name in names:
            matches = [a for a in cfg['actuators'] if re.fullmatch(a['pattern'], name)]
            if len(matches) != 1:
                raise ValueError(f'Nonunique actuator mapping: {name}')
            a = matches[0]
            kp, kd = actuator_gains(a)
            ET.SubElement(actuators, 'general', name='sm_' + name, joint=name,
                          gaintype='fixed', biastype='affine', dyntype='none',
                          gear='1 0 0 0 0 0', gainprm=str(kp), biasprm=f'0 {-kp} {-kd}',
                          forcelimited='true', forcerange=f"{-a['effort']} {a['effort']}",
                          ctrllimited='false')
            params.append(a)
        tree.write(path, encoding='unicode')
        model = mj.MjModel.from_xml_path(str(path))
    model.geom_friction[:, 0] = 1.0
    model.opt.timestep = .005
    model.opt.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    solver = cfg.get('mujoco_solver', 'NEWTON')
    if solver not in ('NEWTON', 'CG'):
        raise ValueError(f'Unsupported MuJoCo solver: {solver}')
    model.opt.solver = getattr(mj.mjtSolver, 'mjSOL_' + solver)
    if not cfg.get('mujoco_multiccd', True):
        model.opt.disableflags |= mj.mjtDisableBit.mjDSBL_MULTICCD
    # Observations come directly from state; MJCF sensor outputs are unused.
    model.opt.disableflags |= mj.mjtDisableBit.mjDSBL_SENSOR
    # PD damping is supplied by the actuators, not duplicated in passive joints.
    for name, a in zip(names, params):
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        did = model.jnt_dofadr[jid]
        model.dof_armature[did] = a['armature']
        model.dof_damping[did] = 0
    mj.mj_setConst(model, mj.MjData(model))
    return model, params


class MJWarp:
    def __init__(self, cfg, names, num_envs, device, randomize=False):
        import mujoco as mj
        import mujoco_warp as mw
        import warp as wp

        if not str(device).startswith('cuda'):
            raise ValueError('MuJoCo Warp training requires CUDA')
        wp.init()
        torch.cuda.set_device(device)
        self.wp, self.mw, self.device = wp, mw, device
        previous_stream = torch.cuda.current_stream(device)
        self.torch_stream = torch.cuda.Stream(device=device)
        self.torch_stream.wait_stream(previous_stream)
        torch.cuda.set_stream(self.torch_stream)
        self.stream = wp.stream_from_torch(self.torch_stream)
        self.randomize = randomize
        jnames, bnames = names
        self.cpu_model, params = build_model(cfg, jnames)
        m = self.cpu_model
        joint_ids = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n) for n in jnames]
        if m.nq != len(jnames) + 7 or m.nv != len(jnames) + 6 or min(joint_ids) < 0:
            raise ValueError('Expected one free base and the named hinge joints')
        self.qids = m.jnt_qposadr[joint_ids].tolist()
        self.vids = m.jnt_dofadr[joint_ids].tolist()
        self.bids = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, b) for b in bnames]
        if min(self.bids) < 0:
            raise ValueError('Unknown tracked body')
        self.root_bid = int(m.jnt_bodyid[0])
        self.layout = Layout(len(jnames), len(bnames))
        self.default = torch.zeros(len(jnames), device=device)
        for j, name in enumerate(jnames):
            for pattern, value in cfg['default_pose'].items():
                if re.fullmatch(pattern, name):
                    self.default[j] = value
        limits = m.jnt_range[joint_ids].copy()
        limits[~m.jnt_limited[joint_ids].astype(bool)] = [-np.inf, np.inf]
        self.limits = torch.as_tensor(limits, dtype=torch.float32, device=device)
        self.scale = torch.tensor([.25 * a['effort'] / actuator_gains(a)[0] for a in params], device=device)
        self.context = torch.zeros((num_envs, len(jnames) + 4), device=device)
        self.context[:, 0] = 1
        with wp.ScopedDevice(device), wp.ScopedStream(self.stream):
            self.model = mw.put_model(m)
            self.data = mw.make_data(m, nworld=num_envs, nconmax=128, njmax=512)
            self.torso_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, cfg['torso_body'])
            self.nominal_com = torch.tensor(m.body_ipos[self.torso_id], dtype=torch.float32, device=device)
            # These outputs of set_const_0 must also have per-world storage.
            for name in ('body_ipos', 'geom_friction', 'dof_invweight0', 'body_invweight0',
                         'actuator_acc0', 'tendon_length0', 'tendon_invweight0',
                         'cam_pos0', 'cam_poscom0', 'cam_mat0', 'light_pos0',
                         'light_poscom0', 'light_dir0'):
                arr = getattr(self.model, name)
                if arr.size:
                    value = wp.to_torch(arr).repeat(num_envs, *([1] * (wp.to_torch(arr).ndim - 1))).contiguous()
                    setattr(self.model, name, wp.from_torch(value, dtype=arr.dtype))
            arr = self.model.stat.meaninertia
            self.model.stat.meaninertia = wp.from_torch(wp.to_torch(arr).repeat(num_envs), dtype=arr.dtype)
            mw.set_const_0(self.model, self.data)
            mw.forward(self.model, self.data)
            # Compile before capture, then reuse graphs for all simulation calls.
            mw.step(self.model, self.data)
            with wp.ScopedCapture() as capture:
                for _ in range(4):
                    mw.step(self.model, self.data)
                mw.forward(self.model, self.data)
            self.step_graph = capture.graph
            with wp.ScopedCapture() as capture:
                mw.forward(self.model, self.data)
            self.forward_graph = capture.graph
            with wp.ScopedCapture() as capture:
                mw.set_const_0(self.model, self.data)
            self.constants_graph = capture.graph
        self.rootids = torch.tensor(m.body_rootid, device=device, dtype=torch.long)

    def view(self, name):
        return self.wp.to_torch(getattr(self.data, name))

    def body_states(self):
        pos = self.view('xpos')
        velocity = self.view('cvel')
        offset = pos - self.view('subtree_com')[:, self.rootids]
        linear = velocity[..., 3:] + torch.cross(velocity[..., :3], offset, dim=-1)
        return torch.cat((pos, self.view('xquat'), linear, velocity[..., :3]), -1)

    def state(self):
        b = self.body_states()
        return torch.cat((b[:, self.root_bid], self.view('qpos')[:, self.qids],
                          self.view('qvel')[:, self.vids], b[:, self.bids].flatten(1)), -1)

    def apply_context(self, ids, context):
        self.context[ids] = context
        friction = self.wp.to_torch(self.model.geom_friction)
        friction[ids, :, 0] = context[:, :1]
        com = self.wp.to_torch(self.model.body_ipos)
        com[ids, self.torso_id] = self.nominal_com + context[:, -3:]
        with self.wp.ScopedStream(self.stream):
            self.wp.capture_launch(self.constants_graph)

    def randomization_state(self):
        return {'context': self.context}

    def restore_randomization(self, saved):
        self.apply_context(torch.arange(len(self.context), device=self.device), saved['context'])

    def verify_context(self):
        friction = self.wp.to_torch(self.model.geom_friction)[..., 0]
        torch.testing.assert_close(friction, self.context[:, :1].expand_as(friction))
        com = self.wp.to_torch(self.model.body_ipos)[:, self.torso_id]
        torch.testing.assert_close(com, self.nominal_com + self.context[:, -3:])
        return {'material_readback': True, 'com_readback': True, 'context_dim': self.context.shape[1]}

    def reset(self, ids, state, resample_context=True):
        if self.randomize and resample_context:
            context = torch.rand((len(ids), self.context.shape[1]), device=self.device)
            context[:, 0] = .3 + .9 * context[:, 0]
            context[:, 1:-3] = (2 * context[:, 1:-3] - 1) * .01
            context[:, -3:] = (2 * context[:, -3:] - 1) * context.new_tensor([.025, .05, .05])
            self.apply_context(ids, context)
        root, q, v, _ = self.layout.parts(state)
        self.view('qpos')[ids, :7] = root[:, :7]
        self.view('qpos')[ids[:, None], self.qids] = q
        self.view('qvel')[ids, :3] = root[:, 7:10]
        self.view('qvel')[ids, 3:6] = rotate_inverse(root[:, 3:7], root[:, 10:13])
        self.view('qvel')[ids[:, None], self.vids] = v
        self.view('qacc_warmstart')[ids] = 0
        self.view('ctrl')[ids] = self.default + self.context[ids, 1:-3]
        with self.wp.ScopedStream(self.stream):
            self.wp.capture_launch(self.forward_graph)

    def step(self, action):
        debug = os.environ.get('SUPERMIMIC_DEBUG_DIR')
        if debug:
            before = {k:self.view(k).clone() for k in ('qpos', 'qvel', 'qacc_warmstart')}
        self.view('ctrl').copy_(self.default + self.context[:, 1:-3] + self.scale * action)
        with self.wp.ScopedStream(self.stream):
            self.wp.capture_launch(self.step_graph)
        state = self.state()
        if not torch.isfinite(state).all():
            if debug:
                bad = (~torch.isfinite(state).all(-1)).nonzero().flatten()
                report = {k:v[bad].cpu() for k,v in before.items()}
                report.update(context=self.context[bad].cpu(), action=action[bad].cpu(),
                              after_state=state[bad].cpu(), ids=bad.cpu())
                for name in ('qacc', 'qpos', 'qvel', 'ncon', 'nefc'):
                    value = getattr(self.data, name, None)
                    if value is not None:
                        report['after_' + name] = self.wp.to_torch(value).cpu()
                path = Path(debug)
                path.mkdir(parents=True, exist_ok=True)
                torch.save(report, path / 'mjwarp_failure.pt')
            raise FloatingPointError('Nonfinite MuJoCo Warp state')
        return state
