import itertools

import pytest

from supermimic.train import learning_rate


def test_paper_schedule_and_constant_tail():
    assert learning_rate(0) == pytest.approx(2e-4)
    assert learning_rate(125_000) == pytest.approx(2e-4 * 0.515)
    assert learning_rate(250_000) == pytest.approx(6e-6)
    assert learning_rate(320_000) == pytest.approx(6e-6)


def test_resume_uses_global_iteration():
    full = [learning_rate(i, decay_iterations=10) for i in range(16)]
    resumed = [learning_rate(i, decay_iterations=10) for i in range(7, 16)]
    assert full[7:] == resumed
    assert all(a >= b for a, b in itertools.pairwise(full))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base": float("nan")},
        {"base": 0},
        {"decay_iterations": 0},
        {"floor": -1},
        {"floor": 2},
    ],
)
def test_invalid_schedule(kwargs):
    with pytest.raises(ValueError):
        learning_rate(0, **kwargs)
