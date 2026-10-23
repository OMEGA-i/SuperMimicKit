import pytest
from supermimic.precision import configure_precision


@pytest.mark.parametrize("mode", ["fp32", "tf32", "bf16"])
def test_precision_switches_are_explicit(mode, monkeypatch):
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    try:
        result = configure_precision(mode)
        assert result["mode"] == mode
        assert result["storage"] == "float32"
        if result["api"] == "fp32_precision":
            assert result["matmul"] == ("tf32" if mode == "tf32" else "ieee")
            assert result["cudnn_conv"] == result["matmul"]
        else:
            assert result["matmul"] == (mode == "tf32")
    finally:
        configure_precision("fp32")


def test_tf32_override_is_not_silently_ignored(monkeypatch):
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    with pytest.raises(RuntimeError, match="NVIDIA_TF32_OVERRIDE"):
        configure_precision("tf32")
