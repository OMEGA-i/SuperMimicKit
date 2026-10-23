"""Explicit FP32 storage / FP32, TF32 or BF16 network compute modes."""

import os
import torch


def configure_precision(mode="fp32"):
    if mode not in ("fp32", "tf32", "bf16"):
        raise ValueError(f"Unknown precision: {mode}")
    if mode == "tf32" and os.environ.get("NVIDIA_TF32_OVERRIDE") == "0":
        raise RuntimeError(
            "NVIDIA_TF32_OVERRIDE=0 disables the requested TF32 mode; unset it before launch"
        )
    compute = "tf32" if mode == "tf32" else "ieee"
    # Do not mix PyTorch's new per-backend API with the legacy allow_tf32 API.
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = compute
        torch.backends.cudnn.fp32_precision = compute
        torch.backends.cudnn.conv.fp32_precision = compute
        torch.backends.cudnn.rnn.fp32_precision = compute
        return {
            "mode": mode,
            "storage": "float32",
            "api": "fp32_precision",
            "matmul": torch.backends.cuda.matmul.fp32_precision,
            "cudnn_conv": torch.backends.cudnn.conv.fp32_precision,
            "cudnn_rnn": torch.backends.cudnn.rnn.fp32_precision,
        }
    torch.set_float32_matmul_precision("high" if mode == "tf32" else "highest")
    torch.backends.cuda.matmul.allow_tf32 = mode == "tf32"
    torch.backends.cudnn.allow_tf32 = mode == "tf32"
    return {
        "mode": mode,
        "storage": "float32",
        "api": "allow_tf32",
        "matmul": torch.backends.cuda.matmul.allow_tf32,
        "cudnn": torch.backends.cudnn.allow_tf32,
    }
