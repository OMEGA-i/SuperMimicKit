#!/usr/bin/env bash
# Clone an installed Isaac Sim runtime into this project's dedicated environment.
# Never install packages into the source runtime. Run from the SuperMimic checkout.
set -euo pipefail
source_runtime=${1:?Pass the existing Isaac runtime venv directory}
[[ -f pyproject.toml && -d supermimic ]] || { echo 'Run inside SuperMimic'; exit 1; }
[[ -x "$source_runtime/bin/python" ]] || { echo 'Missing source Python'; exit 1; }
[[ ! -e .venv ]] || { echo '.venv already exists; refusing overwrite'; exit 1; }
cp -a --reflink=auto "$source_runtime" .venv
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -c 'import sys, torch; print(sys.prefix, torch.__version__)'
