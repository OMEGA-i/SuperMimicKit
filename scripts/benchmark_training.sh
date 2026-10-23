#!/usr/bin/env bash
# Bounded, production-size short run. Never resumes or overwrites training.
set -euo pipefail
robot=${1:?robot}; gpu=${2:?gpu}; mode=${3:?fp32, tf32, bf16 or legacy}; name=${4:?unique name}
case "$robot" in g1|h2|a3) ;; *) exit 2;; esac
root=${SUPERMIMIC_OUTPUT_ROOT:-outputs}
mkdir -p "$root"
source_root=$PWD
export SUPERMIMIC_ROBOT_ASSETS="${SUPERMIMIC_ROBOT_ASSETS:-$PWD/assets/robots}"
export SUPERMIMIC_G1_MJCF="$SUPERMIMIC_ROBOT_ASSETS/g1/robot.xml"
export SUPERMIMIC_H2_MJCF="$SUPERMIMIC_ROBOT_ASSETS/h2/robot.xml"
export SUPERMIMIC_A3_MJCF="$SUPERMIMIC_ROBOT_ASSETS/a3/robot.xml"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CC=${CC:-/usr/bin/gcc} CXX=${CXX:-/usr/bin/g++}
extra=()
case "$mode" in fp32|tf32|bf16) extra=(--precision "$mode");; legacy) extra=(--precision tf32 --legacy-execution);; *) exit 2;; esac
exec "$source_root/.venv/bin/python" -u -m supermimic.train \
 --config "configs/$robot.json" --motions "datasets/lafan/motions/$robot" --output "$root/$name" \
 --device "cuda:$gpu" --num-envs 4096 --iterations "${BENCH_ITERATIONS:-16}" --replay-steps 64 \
 --world-only-iterations 1 --compile --profile-iteration "${PROFILE_ITERATION:--1}" "${extra[@]}" "${@:5}"
