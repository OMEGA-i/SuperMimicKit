import json

import numpy as np
import pytest

pytest.importorskip("mujoco")
from supermimic.retarget import convert


def fixture(tmp_path):
    xml = tmp_path / "robot.xml"
    xml.write_text(
        """<mujoco><worldbody><body name="root"><freejoint/><geom type="sphere" size=".1" mass="1"/><body name="tip" pos="0 0 .2"><joint name="hinge" axis="0 1 0" range="-1 1"/><geom type="sphere" size=".05" mass=".1"/></body></body></worldbody></mujoco>"""
    )
    q = np.zeros((4, 8))
    q[:, 0] = np.arange(4) * 0.02
    q[:, 2] = 1
    q[:, 3] = 1
    result = tmp_path / "result.npz"
    fields = {
        "qpos": q,
        "fps": np.array([50.0]),
        "robot_name": np.array("fixture"),
        "robot_joint_names": np.array(["hinge"], dtype=object),
        "frame_ids": np.arange(4),
    }
    np.savez(result, **fields)
    return xml, result, fields


def test_umr_export_fk_and_world_velocity(tmp_path):
    xml, result, _ = fixture(tmp_path)
    output = tmp_path / "converted"
    convert(result, output, xml, ["root", "tip"], "fixture")
    with np.load(output / "motion.npz") as motion:
        states = motion["state"]
        np.testing.assert_allclose(states[:, 7], 1, atol=1e-6)
        np.testing.assert_allclose(states[:, 15 + 13 + 2], 1.2, atol=1e-6)
        np.testing.assert_allclose(states[:, 15 + 13 + 7], 1, atol=1e-6)
        assert motion["joint_names"].tolist() == ["hinge"]
    assert json.loads((output / "metadata.json").read_text())["retargeter"] == "UMR"


@pytest.mark.parametrize(
    "field,value",
    [
        ("robot_name", np.array("wrong")),
        ("robot_joint_names", np.array(["wrong"], dtype=object)),
        ("frame_ids", np.array([0, 1, 3, 4])),
        ("fps", np.array([0.0])),
    ],
)
def test_umr_export_rejects_mismatched_contract(tmp_path, field, value):
    xml, result, fields = fixture(tmp_path)
    fields[field] = value
    np.savez(result, **fields)
    with pytest.raises(ValueError):
        convert(result, tmp_path / "converted", xml, ["root", "tip"], "fixture")
    assert not (tmp_path / "converted").exists()


def test_dataset_snapshot_resume_and_integrity(tmp_path):
    import hashlib
    from supermimic.retarget import export_dataset

    xml, result, _ = fixture(tmp_path)
    source = tmp_path / "dataset"
    clip = source / "walk"
    clip.mkdir(parents=True)
    result.rename(clip / "retarget.npz")
    manifest = {"status": "running", "expected_clips": 2, "clips": [
        {"status": "completed", "directory": "walk", "result_sha256": hashlib.sha256((clip / "retarget.npz").read_bytes()).hexdigest()},
        {"status": "running", "directory": "unfinished"},
    ]}
    (source / "dataset.json").write_text(json.dumps(manifest))
    output = tmp_path / "export"
    report = export_dataset(source, output, xml, ["root", "tip"], "fixture")
    assert report["status"] == "partial" and len(report["clips"]) == 1
    assert not (output / "unfinished").exists()
    mtime = (output / "walk/motion.npz").stat().st_mtime_ns
    export_dataset(source, output, xml, ["root", "tip"], "fixture", resume=True)
    assert (output / "walk/motion.npz").stat().st_mtime_ns == mtime
    with pytest.raises(ValueError, match="contract mismatch"):
        export_dataset(source, output, xml, ["tip", "root"], "fixture", resume=True)
    (output / "walk/motion.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="motion hash mismatch"):
        export_dataset(source, output, xml, ["root", "tip"], "fixture", resume=True)
