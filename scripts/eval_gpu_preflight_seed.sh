#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${SEED:-13}"
RUN_ROOT="${RUN_ROOT:-outputs/gpu_preflight_mixed_v2_peragent_joint_logq_best_mappo_maddpg}"
UPPER="${UPPER:-mappo}"
LOWER="${LOWER:-maddpg}"
TRACE="${TRACE:-traces/satedgesim_real_dense_mixed_v2_seed13.jsonl}"
N_LEO="${N_LEO:-16}"
NUM_STATES="${NUM_STATES:-8192}"
TIE_EPS="${TIE_EPS:-0.05}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8088}"
DEVICES_COUNT="${DEVICES_COUNT:-20}"
MAX_DECISIONS="${MAX_DECISIONS:-500}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-30}"
SCENARIO_PROFILE="${SCENARIO_PROFILE:-mixed_cost_landscape_v2}"
TASK_SOURCE_MODE="${TASK_SOURCE_MODE:-round_robin_leo}"
DEVICE="${DEVICE:-cpu}"

RUN_DIR="$RUN_ROOT/seed_${SEED}/upper_${UPPER}__lower_${LOWER}"
EVAL_DIR="$RUN_DIR/eval"
CHECKPOINT="$RUN_DIR/checkpoint.pt"
METRICS="$RUN_DIR/metrics.csv"

mkdir -p "$EVAL_DIR"
test -f "$CHECKPOINT"
test -f "$METRICS"

echo "[eval][seed=$SEED] 1/9 policy health"
python scripts/check_four_tier_policy_health.py \
  --metrics "$METRICS" \
  --tail-window 10 \
  --min-feasibility 0.95 | tee "$EVAL_DIR/policy_health.json"

echo "[eval][seed=$SEED] 2/9 state-conditioned diagnosis"
python scripts/diagnose_state_conditioned_policy.py \
  --checkpoint "$CHECKPOINT" \
  --trace "$TRACE" \
  --n-leo "$N_LEO" \
  --num-states "$NUM_STATES" \
  --output "$EVAL_DIR/state_conditioned.json"

echo "[eval][seed=$SEED] 3/9 eval modes"
python scripts/inspect_policy_eval_modes.py \
  --checkpoint "$CHECKPOINT" \
  --trace "$TRACE" \
  --n-leo "$N_LEO" \
  --num-states "$NUM_STATES" \
  --tie-eps "$TIE_EPS" \
  --output "$EVAL_DIR/eval_modes.json"

echo "[eval][seed=$SEED] 4/9 regret evaluation"
python scripts/evaluate_policy_regret.py \
  --checkpoint "$CHECKPOINT" \
  --trace "$TRACE" \
  --n-leo "$N_LEO" \
  --num-states "$NUM_STATES" \
  --tie-eps "$TIE_EPS" \
  --eval-modes raw_argmax,stochastic_eval,margin_cost_tiebreak,cost_greedy_baseline \
  --stochastic-seed "$SEED" \
  --output "$EVAL_DIR/regret.json"

echo "[eval][seed=$SEED] 5/9 replay raw_argmax"
python scripts/replay_on_satedgesim.py \
  --base-url "$BASE_URL" \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE" \
  --devices-count "$DEVICES_COUNT" \
  --seed "$SEED" \
  --max-decisions "$MAX_DECISIONS" \
  --eval-mode raw_argmax \
  --tie-eps "$TIE_EPS" \
  --scenario-profile "$SCENARIO_PROFILE" \
  --task-source-mode "$TASK_SOURCE_MODE" \
  --request-timeout "$REQUEST_TIMEOUT" \
  --output-dir "$RUN_DIR/replay_raw_argmax"

echo "[eval][seed=$SEED] 6/9 replay stochastic_eval"
python scripts/replay_on_satedgesim.py \
  --base-url "$BASE_URL" \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE" \
  --devices-count "$DEVICES_COUNT" \
  --seed "$SEED" \
  --max-decisions "$MAX_DECISIONS" \
  --eval-mode stochastic_eval \
  --stochastic-seed "$SEED" \
  --tie-eps "$TIE_EPS" \
  --scenario-profile "$SCENARIO_PROFILE" \
  --task-source-mode "$TASK_SOURCE_MODE" \
  --request-timeout "$REQUEST_TIMEOUT" \
  --output-dir "$RUN_DIR/replay_stochastic_eval"

echo "[eval][seed=$SEED] 7/9 replay margin_cost_tiebreak"
python scripts/replay_on_satedgesim.py \
  --base-url "$BASE_URL" \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE" \
  --devices-count "$DEVICES_COUNT" \
  --seed "$SEED" \
  --max-decisions "$MAX_DECISIONS" \
  --eval-mode margin_cost_tiebreak \
  --stochastic-seed "$SEED" \
  --tie-eps "$TIE_EPS" \
  --scenario-profile "$SCENARIO_PROFILE" \
  --task-source-mode "$TASK_SOURCE_MODE" \
  --request-timeout "$REQUEST_TIMEOUT" \
  --output-dir "$RUN_DIR/replay_margin_cost_tiebreak"

echo "[eval][seed=$SEED] 8/9 summarize replay"
python scripts/summarize_satedgesim_replay.py \
  --input-dir "$RUN_DIR/replay_raw_argmax" \
  --output "$RUN_DIR/replay_raw_argmax/summary_compact.json"
python scripts/summarize_satedgesim_replay.py \
  --input-dir "$RUN_DIR/replay_stochastic_eval" \
  --output "$RUN_DIR/replay_stochastic_eval/summary_compact.json"
python scripts/summarize_satedgesim_replay.py \
  --input-dir "$RUN_DIR/replay_margin_cost_tiebreak" \
  --output "$RUN_DIR/replay_margin_cost_tiebreak/summary_compact.json"

echo "[eval][seed=$SEED] 9/9 readiness"
python scripts/check_preflight_cuda_readiness.py \
  --metrics "$METRICS" \
  --state-conditioned-diagnosis "$EVAL_DIR/state_conditioned.json" \
  --eval-modes "$EVAL_DIR/eval_modes.json" \
  --regret "$EVAL_DIR/regret.json" \
  --raw-replay-summary "$RUN_DIR/replay_raw_argmax/summary_compact.json" \
  --stochastic-replay-summary "$RUN_DIR/replay_stochastic_eval/summary_compact.json" \
  --output "$EVAL_DIR/readiness.json"

echo "[eval][seed=$SEED] done"
