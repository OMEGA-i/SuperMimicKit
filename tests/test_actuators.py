import math

import pytest

from supermimic.physx import actuator_gains


def test_zero_armature_supplier_gains():
    assert actuator_gains({"armature": 0, "stiffness": 10, "damping": 0.5}) == (10, 0.5)


def test_frequency_rule_unchanged():
    assert actuator_gains({"armature": 0.01}) == pytest.approx(
        (0.01 * (20 * math.pi) ** 2, 0.04 * 20 * math.pi)
    )


@pytest.mark.parametrize(
    "cfg",
    [
        {"armature": 0},
        {"armature": 0.1, "stiffness": 10},
        {"armature": 0.1, "stiffness": float("nan"), "damping": 1},
    ],
)
def test_invalid_gains(cfg):
    with pytest.raises(ValueError):
        actuator_gains(cfg)
