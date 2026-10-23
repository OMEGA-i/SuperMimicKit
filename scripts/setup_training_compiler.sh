#!/usr/bin/env bash
# Run on a fresh Ubuntu training pod before launching compiled workloads.
set -euo pipefail
sudo -n apt-get update
sudo -n apt-get install -y build-essential
export CC=${CC:-$(command -v gcc)} CXX=${CXX:-$(command -v g++)}
.venv/bin/python - <<'PY'
import torch
from supermimic.compiler import check_compiler
check_compiler()
f = torch.compile(lambda x: (x.sin() * x).sum(), fullgraph=True)
x = torch.randn(1024, device='cuda', requires_grad=True)
f(x).backward()
torch.cuda.synchronize()
assert torch.isfinite(x.grad).all()
print('CUDA/Triton forward and backward compilation passed')
PY
