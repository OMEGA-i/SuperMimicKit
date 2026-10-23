"""Render a trusted UMR trajectory with metric framing and save kinematic metrics."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco as mj
import numpy as np


def render(source, output, start=0, end=None):
    with np.load(source, allow_pickle=False) as result:
        all_poses = result["qpos"]
        if (
            start < 0
            or start >= len(all_poses)
            or (end is not None and (end <= start or end > len(all_poses)))
        ):
            raise ValueError("Invalid frame interval")
        poses = all_poses[start:end]
        fps = float(result["fps"].item())
        robot_xml = str(result["robot_xml"].item())
        object_poses = None
        if "source_format" in result and str(result["source_format"].item()).startswith("grail"):
            from supermimic.scene_pair import load_pair, playback_model
            _, scene_xml, object_poses = load_pair(source)
            object_poses = object_poses[start:end]
            model = playback_model(robot_xml, scene_xml)
        else:
            model = mj.MjModel.from_xml_path(robot_xml)
    if object_poses is not None:
        poses = np.concatenate([poses, object_poses], axis=1)
    if poses.shape[1] != model.nq or not np.isfinite(poses).all():
        raise ValueError("Invalid trajectory")
    data = mj.MjData(model)
    model.vis.global_.offwidth = 480
    model.vis.global_.offheight = 480
    camera = mj.MjvCamera()
    camera.distance = 4
    camera.elevation = 0
    snapshots = []
    chosen = set(np.linspace(0, len(poses) - 1, 4, dtype=int))
    output.mkdir(parents=True, exist_ok=False)
    with (
        mj.Renderer(model, height=480, width=480) as renderer,
        imageio.get_writer(output / "rollout.mp4", fps=fps) as writer,
    ):
        for t, q in enumerate(poses):
            data.qpos[:] = q
            mj.mj_forward(model, data)
            camera.lookat[:] = [*q[:2], 1.25]
            panels = []
            for azimuth in [0, 90]:
                camera.azimuth = azimuth
                renderer.update_scene(data, camera=camera)
                for view in renderer.scene.camera:
                    view.orthographic = 1
                    view.frustum_bottom = -1.25
                    view.frustum_top = 1.25
                    view.frustum_near = 0.01
                    view.frustum_far = 50
                for z in np.arange(0, 2.51, 0.5):
                    scene = renderer.scene
                    geom = scene.geoms[scene.ngeom]
                    mj.mjv_initGeom(
                        geom,
                        mj.mjtGeom.mjGEOM_LINE,
                        np.zeros(3),
                        np.zeros(3),
                        np.eye(3).ravel(),
                        np.array([0.5, 0.5, 0.5, 1.0]),
                    )
                    line_start = (
                        np.array([q[0] - 0.8, q[1] - 1.25, z])
                        if azimuth == 0
                        else np.array([q[0] - 1.25, q[1] - 0.8, z])
                    )
                    line_end = (
                        line_start + np.array([0, 2.5, 0])
                        if azimuth == 0
                        else line_start + np.array([2.5, 0, 0])
                    )
                    mj.mjv_connector(
                        geom, mj.mjtGeom.mjGEOM_LINE, 1.0, line_start, line_end
                    )
                    scene.ngeom += 1
                panels.append(renderer.render().copy())
            frame = np.concatenate(panels, axis=1)
            writer.append_data(frame)
            if t in chosen:
                snapshots.append(frame)
    imageio.imwrite(output / "contact_sheet.png", np.concatenate(snapshots, axis=0))
    hinges = np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_HINGE)
    limited = hinges[model.jnt_limited[hinges].astype(bool)]
    values = poses[:, model.jnt_qposadr[limited]]
    violation = np.maximum(
        model.jnt_range[limited, 0] - values, values - model.jnt_range[limited, 1]
    ).clip(0)
    metrics = {
        "frames": len(poses),
        "source_frame_start": start,
        "source_frame_end_exclusive": start + len(poses),
        "fps": fps,
        "finite": True,
        "max_joint_limit_violation_rad": float(violation.max(initial=0)),
        "max_joint_step_rad": float(
            np.abs(np.diff(poses[:, model.jnt_qposadr[hinges]], axis=0)).max(initial=0)
        ),
        "vertical_frame_m": 2.5,
        "grid_spacing_m": 0.5,
        "visual_qc": "pending_review",
        "physics_qc": "not_run",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int)
    a = p.parse_args()
    render(a.source, a.output, a.start, a.end)
