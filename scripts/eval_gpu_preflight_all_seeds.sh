#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEEDS="${SEEDS:-13,17,23}"
RUN_ROOT="${RUN_ROOT:-outputs/gpu_preflight_mixed_v2_peragent_joint_logq_best_mappo_maddpg}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-outputs/summary_gpu_preflight_mixed_v2_mappo_maddpg.json}"

IFS=',' read -r -a seed_arr <<< "$SEEDS"
for seed in "${seed_arr[@]}"; do
  seed_trimmed="$(echo "$seed" | xargs)"
  [ -n "$seed_trimmed" ] || continue
  echo "[eval-all] running seed=$seed_trimmed"
  SEED="$seed_trimmed" RUN_ROOT="$RUN_ROOT" bash scripts/eval_gpu_preflight_seed.sh
done

echo "[eval-all] summarizing seeds=$SEEDS"
python scripts/summarize_gpu_preflight.py \
  --run-root "$RUN_ROOT" \
  --seeds "$SEEDS" \
  --output "$SUMMARY_OUTPUT"

echo "[eval-all] done: $SUMMARY_OUTPUT"
