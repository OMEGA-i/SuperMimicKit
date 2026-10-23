"""
MIT License

Copyright (c) 2025 Joao Pedro Araujo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Adapted from https://github.com/jaraujo98/lafan_to_smplx
# Uses our BVH reader; no robot retarget package is required for conversion.
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from supermimic.lafan import load_bvh

lafan_frame_offsets = {
    "world": R.from_euler("x", 0),
    "Hips": R.from_euler("z", -np.pi / 2) * R.from_euler("y", -np.pi / 2),
    "LeftUpLeg": R.from_euler("z", np.pi / 2) * R.from_euler("y", np.pi / 2),
    "LeftLeg": R.from_euler("z", np.pi / 2) * R.from_euler("y", np.pi / 2),
    "LeftFoot": R.from_euler("z", 0.37117860986509) * R.from_euler("y", np.pi / 2),
    "LeftToe": R.from_euler("y", np.pi / 2),
    "RightUpLeg": R.from_euler("z", np.pi / 2) * R.from_euler("y", np.pi / 2),
    "RightLeg": R.from_euler("z", np.pi / 2) * R.from_euler("y", np.pi / 2),
    "RightFoot": R.from_euler("z", 0.37117860986509) * R.from_euler("y", np.pi / 2),
    "RightToe": R.from_euler("y", np.pi / 2),
    "Spine": R.from_euler("z", -np.pi / 2) * R.from_euler("y", -np.pi / 2),
    "Spine1": R.from_euler("z", -np.pi / 2) * R.from_euler("y", -np.pi / 2),
    "Spine2": R.from_euler("z", -np.pi / 2) * R.from_euler("y", -np.pi / 2),
    "Neck": R.from_euler("z", -np.pi / 2) * R.from_euler("y", -np.pi / 2),
    "Head": R.from_euler("z", -np.pi / 2) * R.from_euler("y", -np.pi / 2),
    "LeftShoulder": R.from_euler("x", np.pi / 2),
    "LeftArm": R.from_euler("x", np.pi / 2),
    "LeftForeArm": R.from_euler("x", np.pi / 2),
    "LeftHand": R.from_euler("x", np.pi / 2),
    "RightShoulder": R.from_euler("z", np.pi) * R.from_euler("x", -np.pi / 2),
    "RightArm": R.from_euler("z", np.pi) * R.from_euler("x", -np.pi / 2),
    "RightForeArm": R.from_euler("z", np.pi) * R.from_euler("x", -np.pi / 2),
    "RightHand": R.from_euler("z", np.pi) * R.from_euler("x", -np.pi / 2),
}

smplx_to_lafan_map = {
    "left_hip": "LeftUpLeg",
    "right_hip": "RightUpLeg",
    "spine1": "Spine",
    "left_knee": "LeftLeg",
    "right_knee": "RightLeg",
    "spine2": "Spine1",
    "left_ankle": "LeftFoot",
    "right_ankle": "RightFoot",
    "spine3": "Spine2",
    "left_foot": "LeftToe",
    "right_foot": "RightToe",
    "neck": "Neck",
    "left_collar": "LeftShoulder",
    "right_collar": "RightShoulder",
    "head": "Head",
    "left_shoulder": "LeftArm",
    "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm",
    "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand",
    "right_wrist": "RightHand",
}


BETAS = [
    1.4775,
    0.6674,
    -1.1742,
    0.4731,
    1.2984,
    -0.2159,
    1.5276,
    -0.3152,
    -0.6441,
    -0.2986,
    0.5089,
    -0.6354,
    0.3321,
    -0.1099,
    -0.3060,
    -0.7330,
]


def convert(source, output, model_path, max_frames=0):
    import smplx
    import torch
    from smplx.joint_names import JOINT_NAMES

    if max_frames < 0:
        raise ValueError("max_frames must be nonnegative")
    if output.exists():
        raise FileExistsError(output)
    names, parents, positions, quaternions, fps = load_bvh(source)
    if len(names) != 22 or set(names) != set(lafan_frame_offsets) - {"world"}:
        raise ValueError("Expected the original 22-joint LaFAN skeleton")
    if max_frames:
        positions = positions[:max_frames]
        quaternions = quaternions[:max_frames]
    count = len(positions)
    model = smplx.create(
        str(model_path),
        model_type="smplx",
        ext="npz" if (model_path / "smplx/SMPLX_NEUTRAL.npz").is_file() else "pkl",
        gender="neutral",
        use_pca=False,
        num_betas=16,
        betas=torch.tensor([BETAS]),
    )
    orientations = {
        name: R.from_quat(quaternions[:, j], scalar_first=True)
        * lafan_frame_offsets[name]
        for j, name in enumerate(names)
    }
    poses = np.zeros((count, 55, 3), dtype=np.float32)
    poses[:, 0] = orientations["Hips"].as_rotvec()
    for j, human in enumerate(JOINT_NAMES[1:22], start=1):
        bone = smplx_to_lafan_map[human]
        parent = names[parents[names.index(bone)]]
        poses[:, j] = (orientations[parent].inv() * orientations[bone]).as_rotvec()
    translation = np.empty((count, 3), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, count, 128):
            end = min(start + 128, count)
            result = model(
                global_orient=torch.from_numpy(poses[start:end, 0]),
                body_pose=torch.from_numpy(poses[start:end, 1:22].reshape(-1, 63)),
            )
            translation[start:end] = (
                positions[start:end].mean(axis=1)
                - result.joints[:, :22].mean(dim=1).numpy()
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    np.savez_compressed(
        output,
        poses=poses,
        trans=translation,
        betas=np.asarray(BETAS, dtype=np.float32),
        gender="neutral",
        mocap_frame_rate=fps,
        output_up="z",
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "converter_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "fps": fps,
                "frames": count,
                "conversion": "LaFAN local rotation mapping with centroid alignment",
                "qc": "not visually validated",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Parent directory containing smplx/SMPLX_NEUTRAL model",
    )
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if a.threads < 1 or a.max_frames < 0:
        p.error("threads must be positive and max-frames nonnegative")
    import torch

    torch.set_num_threads(a.threads)
    if a.source.is_dir():
        sources = sorted(a.source.rglob("*.bvh"))
        if not sources:
            p.error("No BVH files in source directory")
        a.output.mkdir(parents=True, exist_ok=False)
        manifest = {"status": "running", "expected_clips": len(sources), "clips": []}
        model_file = a.model_path / "smplx/SMPLX_NEUTRAL.npz"
        if not model_file.is_file():
            model_file = model_file.with_suffix(".pkl")
        manifest["model_sha256"] = hashlib.sha256(model_file.read_bytes()).hexdigest()
        manifest["source_root"] = str(a.source.resolve())
        report = a.output / "conversion.json"
        for source in sources:
            relative = source.relative_to(a.source).with_suffix(".npz")
            entry = {"source": str(source), "output": str(relative)}
            try:
                convert(source, a.output / relative, a.model_path, a.max_frames)
                entry.update(
                    status="completed",
                    **json.loads(
                        (a.output / relative).with_suffix(".json").read_text()
                    ),
                )
            except Exception as error:  # noqa: BLE001 -- record each failed clip and continue the dataset
                entry.update(status="failed", error=repr(error))
            manifest["clips"].append(entry)
            temporary = report.with_suffix(".tmp")
            temporary.write_text(json.dumps(manifest, indent=2) + "\n")
            temporary.replace(report)
            print(
                f"{len(manifest['clips'])}/{len(sources)} {relative}: {entry['status']}",
                flush=True,
            )
        manifest["status"] = (
            "completed"
            if all(x["status"] == "completed" for x in manifest["clips"])
            else "failed"
        )
        report.write_text(json.dumps(manifest, indent=2) + "\n")
        if manifest["status"] != "completed":
            raise RuntimeError(f"Some clips failed; inspect {report}")
    else:
        convert(a.source, a.output, a.model_path, a.max_frames)
