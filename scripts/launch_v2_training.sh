#!/usr/bin/env bash
# One process / GPU / embodiment. Run from the checked-out project on the pod.
set -euo pipefail
robot=${1:?robot: g1 h2 a3}
gpu=${2:?GPU index}
run_name=${3:?unique run name}
case "$robot" in g1|h2|a3) ;; *) exit 2 ;; esac
artifact_root=${SUPERMIMIC_OUTPUT_ROOT:-outputs}
mkdir -p "$artifact_root"
export SUPERMIMIC_ROBOT_ASSETS="${SUPERMIMIC_ROBOT_ASSETS:-$PWD/assets/robots}"
export SUPERMIMIC_G1_MJCF="$SUPERMIMIC_ROBOT_ASSETS/g1/robot.xml"
export SUPERMIMIC_H2_MJCF="$SUPERMIMIC_ROBOT_ASSETS/h2/robot.xml"
export SUPERMIMIC_A3_MJCF="$SUPERMIMIC_ROBOT_ASSETS/a3/robot.xml"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CC=${CC:-/usr/bin/gcc} CXX=${CXX:-/usr/bin/g++}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-8}
output="$artifact_root/$run_name"
if [[ -e "$output" || -e "$output.pid" ]]; then echo "Run already exists: $output" >&2; exit 1; fi
nohup .venv/bin/python -u -m supermimic.train \
  --config "configs/$robot.json" --motions "datasets/lafan/motions/$robot" \
  --motion-split train40 --device "cuda:$gpu" --output "$output" \
  --precision "${4:-fp32}" --num-envs 4096 --iterations 320000 --checkpoint-every 500 --compile "${@:5}" \
  > "$output.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$output.pid"
printf '%s GPU %s PID %s output %s\n' "$robot" "$gpu" "$pid" "$output"
