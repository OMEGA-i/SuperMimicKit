"""Exercise compiled MLP forward/backward and record actual CUDA kernel names."""

import argparse
import json
import torch
from supermimic.precision import configure_precision
from supermimic.core_v2 import run_mlp

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--device", default="cuda:0")
a = p.parse_args()
torch.cuda.set_device(a.device)
results = []
for mode in ("fp32", "tf32", "bf16"):
    settings = configure_precision(mode)
    torch.manual_seed(42)
    net = torch.nn.Sequential(
        torch.nn.Linear(512, 1024), torch.nn.SiLU(), torch.nn.Linear(1024, 512)
    ).to(a.device)
    x = torch.randn(1024, 512, device=a.device, requires_grad=True)
    amp = "bf16" if mode == "bf16" else "none"
    f = torch.compile(lambda z: run_mlp(net, z, amp).square().mean(), fullgraph=True)
    for _ in range(2):
        net.zero_grad(set_to_none=True)
        x.grad = None
        loss = f(x)
        loss.backward()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        net.zero_grad(set_to_none=True)
        x.grad = None
        loss = f(x)
        loss.backward()
        torch.cuda.synchronize()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())
    kernels = sorted(
        {
            e.name
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        }
    )
    results.append(
        {
            "settings": settings,
            "loss": float(loss),
            "parameter_dtype": str(next(net.parameters()).dtype),
            "finite_gradients": True,
            "cuda_kernels": kernels,
        }
    )
with open(a.output, "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
