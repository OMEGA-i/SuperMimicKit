"""UMR result -> 50 Hz named, scalar-first, world-frame training motion."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def resample(qpos, fps, target=50):
    src = np.arange(len(qpos)) / fps
    dst = np.arange(int(np.floor(src[-1] * target)) + 1) / target
    out = np.stack([np.interp(dst, src, qpos[:, i]) for i in range(qpos.shape[1])], -1)
    out[:, 3:7] = Slerp(src, Rotation.from_quat(qpos[:, 3:7][:, [1, 2, 3, 0]]))(
        dst
    ).as_quat()[:, [3, 0, 1, 2]]
    return out


def convert(source, output, xml, bodies, robot):
    """Consume trusted, locally generated UMR output using its exact MJCF."""
    import mujoco as mj

    model = mj.MjModel.from_xml_path(str(xml))
    data = mj.MjData(model)
    # Upstream UMR writes robot_joint_names as an object array. Only use results
    # produced by the pinned local UMR pipeline, never arbitrary downloaded NPZs.
    with np.load(source, allow_pickle=True) as result:
        qpos = np.asarray(result["qpos"], dtype=np.float64)
        fps = float(np.asarray(result["fps"]).item())
        actual_robot = str(np.asarray(result["robot_name"]).item())
        names = [str(n) for n in result["robot_joint_names"].tolist()]
        frame_ids = np.asarray(result["frame_ids"])
        scene_required = str(result.get("source_format", "").item() if "source_format" in result else "").startswith("grail")
    if scene_required:
        from .scene_pair import load_pair
        load_pair(source)  # Fail before writing a motion if its scene is missing/stale.
    if actual_robot != robot:
        raise ValueError(f"UMR robot mismatch: {actual_robot} != {robot}")
    if qpos.ndim != 2 or len(qpos) < 2 or qpos.shape[1] != model.nq:
        raise ValueError("UMR qpos shape does not match the floating-base MJCF")
    if not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Nonfinite UMR result or invalid fps")
    if frame_ids.shape != (len(qpos),) or not np.all(np.diff(frame_ids) > 0):
        raise ValueError("UMR frame_ids must be strictly increasing")
    if not np.all(np.diff(frame_ids) == np.diff(frame_ids)[0]):
        raise ValueError("Nonuniform UMR frame spacing is unsupported")
    if model.jnt_type[0] != mj.mjtJoint.mjJNT_FREE:
        raise ValueError("Expected a free root as the first MJCF joint")
    if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=-1), 1, atol=1e-4):
        raise ValueError("UMR root quaternion is not unit length")
    qpos = resample(qpos, fps)
    if len(qpos) < 2:
        raise ValueError("Clip duration is shorter than one 50 Hz interval")
    joints = [
        j for j in range(model.njnt) if model.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE
    ]
    if model.nq != len(joints) + 7:
        raise ValueError("Expected one free root and scalar hinge joints")
    joint_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in joints]
    if names != joint_names:
        raise ValueError("UMR joint order differs from the supplied MJCF")
    body_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, b) for b in bodies]
    missing = [name for name, bid in zip(bodies, body_ids) if bid < 0]
    if missing:
        raise ValueError(f"Unknown tracked bodies: {missing}")
    qvel = np.zeros((len(qpos), model.nv))
    for t in range(len(qpos) - 1):
        mj.mj_differentiatePos(model, qvel[t], 0.02, qpos[t], qpos[t + 1])
    qvel[-1] = qvel[-2]
    states = []
    for q, v in zip(qpos, qvel):
        data.qpos[:] = q
        data.qvel[:] = v
        mj.mj_forward(model, data)
        bv = []
        for bid in body_ids:
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros_like(jacp)
            mj.mj_jacBody(model, data, jacp, jacr, bid)
            bv.append(np.r_[data.xpos[bid], data.xquat[bid], jacp @ v, jacr @ v])
        # Free-joint rotational qvel is local; convert to world for Isaac Lab contract.
        omega = Rotation.from_quat(q[3:7][[1, 2, 3, 0]]).apply(v[3:6])
        states.append(np.r_[q[:7], v[:3], omega, q[7:], v[6:], np.array(bv).ravel()])
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "motion.npz",
        state=np.array(states, dtype=np.float32),
        fps=50,
        joint_names=np.array(joint_names),
        body_names=np.array(bodies),
    )
    meta = {
        "body_names": bodies,
        "motion_sha256": hashlib.sha256((output / "motion.npz").read_bytes()).hexdigest(),
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "robot": robot,
        "retargeter": "UMR",
        "xml": str(xml),
        "xml_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
        "source_fps": fps,
        "fps": 50,
        "quaternion": "wxyz",
        "velocity_frame": "world",
        "frames": len(states),
        "retarget_qc": "NOT_VISUALLY_VALIDATED",
    }
    if scene_required:
        from .scene_pair import export_pair
        meta.update(export_pair(source, output, len(states)))
    (output / "metadata.json").write_text(json.dumps(meta, indent=2))


def export_dataset(source, output, xml, bodies, robot, resume=False):
    """Export a snapshot of completed UMR clips; never read in-flight results."""
    manifest_path = source / "dataset.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "expected_clips": manifest["expected_clips"],
        "source_status": manifest["status"],
        "clips": [],
    }
    xml_hash = hashlib.sha256(xml.read_bytes()).hexdigest()
    seen = set()
    for entry in manifest["clips"]:
        if entry["status"] != "completed":
            continue
        relative = Path(entry["directory"])
        if relative.is_absolute() or ".." in relative.parts or relative in seen:
            raise ValueError("Invalid or duplicate dataset clip path")
        seen.add(relative)
        src, dst = source / relative / "retarget.npz", output / relative
        source_hash = hashlib.sha256(src.read_bytes()).hexdigest()
        if source_hash != entry["result_sha256"]:
            raise ValueError(f"Retarget result hash mismatch: {src}")
        if dst.exists():
            if not resume:
                raise FileExistsError(dst)
            meta = json.loads((dst / "metadata.json").read_text())
            expected = {"source_sha256": source_hash, "xml_sha256": xml_hash,
                        "robot": robot, "body_names": bodies, "fps": 50}
            if any(meta.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Existing export contract mismatch: {dst}")
            if meta.get("motion_sha256") != hashlib.sha256((dst / "motion.npz").read_bytes()).hexdigest():
                raise ValueError(f"Existing motion hash mismatch: {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Publish complete clip directories only; interrupted exports can retry.
            with tempfile.TemporaryDirectory(prefix=".export-", dir=dst.parent) as temp:
                staged = Path(temp) / "clip"
                convert(src, staged, xml, bodies, robot)
                staged.rename(dst)
            meta = json.loads((dst / "metadata.json").read_text())
        report["clips"].append({"directory": str(relative), "frames": meta["frames"],
                                "motion_sha256": meta["motion_sha256"]})
        report["status"] = "complete" if len(report["clips"]) == report["expected_clips"] else "partial"
        temp = output / "export.json.tmp"
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(output / "export.json")
        print(f"Exported {len(report['clips'])}/{report['expected_clips']}: {relative}", flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--xml", type=Path, required=True)
    p.add_argument("--dataset", action="store_true", help="Read source/dataset.json completed clips")
    p.add_argument("--resume", action="store_true", help="Reuse verified dataset exports")
    args = p.parse_args()
    if args.resume and not args.dataset:
        p.error("--resume requires --dataset")
    cfg = json.loads(args.config.read_text())
    if args.dataset:
        export_dataset(args.source, args.output, args.xml, cfg["tracked_bodies"], cfg["robot"], args.resume)
        return
    files = (
        sorted(args.source.rglob("*.npz")) if args.source.is_dir() else [args.source]
    )
    if not files:
        raise ValueError("No UMR result NPZ files")
    for source in files:
        rel = (
            source.relative_to(args.source).with_suffix("")
            if args.source.is_dir()
            else Path(source.stem)
        )
        convert(
            source, args.output / rel, args.xml, cfg["tracked_bodies"], cfg["robot"]
        )


if __name__ == "__main__":
    main()
