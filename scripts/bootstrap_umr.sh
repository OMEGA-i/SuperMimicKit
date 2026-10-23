#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/apply_umr_overlay.py
if [[ ! -f third_party/UMR/requirements-umr.txt ]]; then
  echo 'Initialize the UMR submodule first.' >&2
  exit 1
fi
if [[ -e .venv-umr ]]; then
  echo '.venv-umr already exists; refusing to replace an existing environment.' >&2
  exit 1
fi
uv venv --python 3.12 .venv-umr
uv pip install --python .venv-umr/bin/python torch==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv-umr/bin/python -r third_party/UMR/requirements-umr.txt
.venv-umr/bin/python -c 'import torch, mujoco, warp, smplx; print(torch.__version__, mujoco.__version__, torch.cuda.is_available())'
