#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-13,17,23}"
EPISODES="${EPISODES:-100}"
STEPS="${STEPS:-128}"
RUN_ROOT="${RUN_ROOT:-outputs/gpu_preflight_mixed_v2_peragent_joint_logq_best_mappo_maddpg}"
CONFIG="${CONFIG:-trisatflow/configs/satedgesim_trace_mixed_v2_peragent_joint_logq_best.yaml}"
N_LEO="${N_LEO:-16}"
UPPER="${UPPER:-mappo}"
LOWER="${LOWER:-maddpg}"

mkdir -p "$RUN_ROOT"
LOG_PATH="$RUN_ROOT/train_preflight.log"

echo "[preflight] training start: run_root=$RUN_ROOT seeds=$SEEDS episodes=$EPISODES steps=$STEPS"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python scripts/sweep_algorithm_combinations.py \
  --config "$CONFIG" \
  --upper "$UPPER" \
  --lower "$LOWER" \
  --episodes "$EPISODES" \
  --steps "$STEPS" \
  --n-leo "$N_LEO" \
  --seeds "$SEEDS" \
  --device "$DEVICE" \
  --output-root "$RUN_ROOT" | tee "$LOG_PATH"

echo "[preflight] done: $RUN_ROOT"
