"""Numeric and foot-to-plane checks for completed UMR dataset results."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco as mj
import numpy as np


def check(source, feet):
    with np.load(source, allow_pickle=False) as result:
        qpos = result["qpos"]
        fps = float(result["fps"])
        xml = Path(str(result["robot_xml"].item()))
    spec = mj.MjSpec.from_file(str(xml))
    spec.worldbody.add_geom(
        name="qc_ground_plane",
        type=mj.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.1],
        pos=[0, 0, 0],
    )
    model = spec.compile()
    if qpos.ndim != 2 or qpos.shape != (len(qpos), model.nq) or len(qpos) < 2:
        raise ValueError("Invalid qpos shape")
    if not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Nonfinite trajectory or invalid FPS")
    ground = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "qc_ground_plane")
    groups = []
    for name in feet:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if bid < 1:
            raise ValueError(f"Missing foot body: {name}")
        descendants = {bid}
        for body in range(bid + 1, model.nbody):
            if int(model.body_parentid[body]) in descendants:
                descendants.add(body)
        geoms = [
            g
            for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) in descendants
            and (model.geom_contype[g] or model.geom_conaffinity[g])
        ]
        if not geoms:
            raise ValueError(f"No collision geometry for foot: {name}")
        groups.append(geoms)
    data = mj.MjData(model)
    distances = np.empty((len(qpos), len(feet)))
    nearest = np.empty(distances.shape, dtype=np.int32)
    minimum_body_z = float("inf")
    for t, pose in enumerate(qpos):
        data.qpos[:] = pose
        mj.mj_forward(model, data)
        minimum_body_z = min(minimum_body_z, float(data.xpos[1:, 2].min()))
        for index, geoms in enumerate(groups):
            values = [
                mj.mj_geomDistance(model, data, g, ground, 100.0, None) for g in geoms
            ]
            local = int(np.argmin(values))
            distances[t, index] = values[local]
            nearest[t, index] = geoms[local]
    limited = np.flatnonzero(
        (model.jnt_type == mj.mjtJoint.mjJNT_HINGE) & model.jnt_limited
    )
    angles = qpos[:, model.jnt_qposadr[limited]]
    violation = np.maximum(
        model.jnt_range[limited, 0] - angles, angles - model.jnt_range[limited, 1]
    ).clip(0)
    worst = []
    for index in range(len(feet)):
        frame = int(distances[:, index].argmin())
        mask = distances[:, index] < -0.01
        edges = np.diff(np.r_[False, mask, False].astype(np.int8))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        geom = int(nearest[frame, index])
        worst.append(
            {
                "frame": frame,
                "time_s": frame / fps,
                "geom_id": geom,
                "geom_name": mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom),
                "distance_m": float(distances[frame, index]),
                "below_minus_1cm_frames": int(mask.sum()),
                "longest_below_minus_1cm_s": float(
                    (ends - starts).max(initial=0) / fps
                ),
            }
        )
    return {
        "worst_foot_frames": worst,
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "xml": str(xml),
        "frames": len(qpos),
        "fps": fps,
        "finite": True,
        "max_joint_limit_violation_rad": float(violation.max(initial=0)),
        "minimum_body_origin_z_m": minimum_body_z,
        "foot_bodies": feet,
        "foot_collision_geoms": groups,
        "foot_distance_min_m": distances.min(axis=0).tolist(),
        "foot_distance_max_m": distances.max(axis=0).tolist(),
        "foot_distance_quantiles_m": np.quantile(
            distances, [0, 0.01, 0.5, 0.99, 1], axis=0
        ).tolist(),
        "both_feet_above_2cm_fraction": float((distances > 0.02).all(axis=1).mean()),
        "dynamic_feasibility": "not_tested",
        "note": "Distances use collision geometry at z=0; airborne frames can be valid jumps. No automatic filtering.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True, help="UMR dataset.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--foot-bodies", nargs=2, required=True)
    p.add_argument("--clip", help="Optional exact clip directory from dataset.json")
    a = p.parse_args()
    manifest = json.loads(a.dataset.read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for clip in manifest["clips"]:
        if clip["status"] != "completed" or (a.clip and clip["directory"] != a.clip):
            continue
        source = a.dataset.parent / clip["directory"] / "retarget.npz"
        if hashlib.sha256(source.read_bytes()).hexdigest() != clip["result_sha256"]:
            raise ValueError(f"Result hash mismatch: {source}")
        result = check(source, a.foot_bodies)
        reports.append(result)
        (a.output / "metrics.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(
            source.parent.name,
            result["max_joint_limit_violation_rad"],
            result["foot_distance_min_m"],
            flush=True,
        )
    if not reports:
        raise ValueError("No completed retarget results")


if __name__ == "__main__":
    main()
