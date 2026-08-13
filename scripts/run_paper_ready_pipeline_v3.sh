#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_paper_ready_pipeline_v3.sh <mode> [options]

Modes:
  preflight-offline      Audit configs, prior smoke gates, trace bank, and reporting inputs without SatEdgeSim.
  preflight-satedgesim   Audit SatEdgeSim Maven compile, REST, /version provenance, receipts, and binding.
  build-traces           Build actual + controlled-stress paper v3 trace banks from a live SatEdgeSim bridge.
  dry-run                CPU-sized end-to-end dry-run under outputs/paper_ready_v3/dry_run.
  formal-main            RL vs RL formal run; requires both preflights.
  formal-rules           Rule-baseline formal run; requires both preflights.
  formal-ablation        Ablation run; actual and stress outputs remain separate.
  formal-learning        Learning-baseline formal run.
  formal-replay          Sequential online replay; requires SatEdgeSim preflight and bound lower actions.
  formal-report          Aggregate, test, and export paper tables/figures.

Common options:
  --device cpu|cuda|auto
  --base-url URL
  --satedgesim-root PATH
  --trace-root PATH
  --config PATH
  --output-root PATH
EOF
}

MODE="${1:-}"
if [[ -z "$MODE" || "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  usage
  exit 0
fi
shift

CONFIG="${CONFIG:-trisatflow/configs/paper/satedgesim_trace_mixed_v3_safe.yaml}"
TRACE_ROOT="${TRACE_ROOT:-traces/paper_v3}"
PRIMARY_TRACE_ROOT="${PRIMARY_TRACE_ROOT:-$TRACE_ROOT/actual_projection}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/paper_ready_v3}"
DEVICE="${DEVICE:-cuda}"
RULE_DEVICE="${RULE_DEVICE:-cpu}"
SATEDGE_BASE_URL="${SATEDGE_BASE_URL:-http://127.0.0.1:8088}"
SATEDGESIM_ROOT="${SATEDGESIM_ROOT:-}"
N_LEO="${N_LEO:-12}"
STEPS="${STEPS:-128}"
TRAIN_SEEDS="${TRAIN_SEEDS:-13,21,42,57,73,89,97,109,127,149}"
VAL_SEEDS="${VAL_SEEDS:-101,131,151}"
TEST_SEEDS="${TEST_SEEDS:-202,303,404,505,606,707,808,909,1001,1103}"
UPPER_ALGOS="${UPPER_ALGOS:-mappo,ippo}"
LOWER_ALGOS="${LOWER_ALGOS:-maddpg,masac}"
RULE_BASELINES="${RULE_BASELINES:-local_only,neighbor_only,geo_only,ground_only,random_visible,min_delay_greedy,min_energy_greedy,queue_aware_greedy,mobility_risk_greedy,lyapunov_dpp_greedy}"
LEARNING_BASELINES="${LEARNING_BASELINES:-flat_ppo,flat_mappo,hierarchical_no_gnn}"
PRIMARY_SEMANTIC_CLASS="${PRIMARY_SEMANTIC_CLASS:-actual_physical_projection}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="$2"; shift 2 ;;
    --base-url)
      SATEDGE_BASE_URL="$2"; shift 2 ;;
    --satedgesim-root)
      SATEDGESIM_ROOT="$2"; shift 2 ;;
    --trace-root)
      TRACE_ROOT="$2"; PRIMARY_TRACE_ROOT="$TRACE_ROOT/actual_projection"; shift 2 ;;
    --config)
      CONFIG="$2"; shift 2 ;;
    --output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2 ;;
  esac
done

run() {
  echo "[paper-ready-v3] $*"
  "$@"
}

gate() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required gate artifact: $path" >&2
    exit 1
  fi
}

require_preflights() {
  gate "$OUTPUT_ROOT/preflight_offline/GATE_OK"
  gate "$OUTPUT_ROOT/preflight_satedgesim/GATE_OK"
}

require_satedgesim_root() {
  if [[ -z "$SATEDGESIM_ROOT" ]]; then
    echo "SATEDGESIM_ROOT or --satedgesim-root is required; no SatEdgeSim path is hardcoded by this pipeline." >&2
    exit 2
  fi
}

satedge_port_from_url() {
  python - "$SATEDGE_BASE_URL" <<'PY'
from urllib.parse import urlparse
import sys
parsed = urlparse(sys.argv[1])
print(parsed.port or (443 if parsed.scheme == "https" else 80))
PY
}

preflight_offline() {
  run python scripts/preflight_paper_ready_v3.py \
    --mode offline \
    --config "$CONFIG" \
    --trace-root "$TRACE_ROOT" \
    --output-dir "$OUTPUT_ROOT/preflight_offline"
}

preflight_satedgesim() {
  require_satedgesim_root
  run python scripts/preflight_paper_ready_v3.py \
    --mode satedgesim \
    --base-url "$SATEDGE_BASE_URL" \
    --satedgesim-root "$SATEDGESIM_ROOT" \
    --output-dir "$OUTPUT_ROOT/preflight_satedgesim"
}

build_traces() {
  require_satedgesim_root
  local port
  port="$(satedge_port_from_url)"
  run env \
    SATEDGESIM_ROOT="$SATEDGESIM_ROOT" \
    SATEDGE_BASE_URL="$SATEDGE_BASE_URL" \
    SATEDGE_PORT="$port" \
    TRACE_ROOT="$TRACE_ROOT" \
    STAGE8_OUT="$OUTPUT_ROOT/build_traces" \
    bash scripts/build_paper_trace_bank.sh
}

dry_run() {
  local out="$OUTPUT_ROOT/dry_run"
  rm -rf "$out"
  mkdir -p "$out/train"
  preflight_offline
  run timeout 180s python scripts/smoke_test.py \
    --config "$CONFIG" \
    --episodes 2 --steps 8 --n-leo 4 \
    --upper-algo mappo --lower-algo maddpg \
    --device "$DEVICE" \
    --output-dir "$out/train"
  run python scripts/export_paper_tables.py \
    --input-root "$out" \
    --output-dir "$out/tables" \
    --allow-smoke-small-n
  run python scripts/plot_paper_results.py \
    --input-root "$out" \
    --output-dir "$out/figures" \
    --allow-smoke-small-n
  printf 'GATE_OK\n' > "$out/GATE_OK"
  echo "PAPER_READY_V3_DRY_RUN_OK output_dir=$out"
}

formal_main() {
  require_preflights
  run python scripts/sweep_algorithm_combinations.py \
    --config "$CONFIG" \
    --upper "$UPPER_ALGOS" \
    --lower "$LOWER_ALGOS" \
    --episodes "${MAIN_EPISODES:-1000}" \
    --steps "$STEPS" \
    --n-leo "$N_LEO" \
    --train-seeds "$TRAIN_SEEDS" \
    --val-seeds "$VAL_SEEDS" \
    --test-seeds "$TEST_SEEDS" \
    --checkpoint-selection per_train_seed \
    --device "$DEVICE" \
    --output-root "$OUTPUT_ROOT/main_actual"
}

formal_rules() {
  require_preflights
  run python scripts/evaluate_rule_baselines.py \
    --config "$CONFIG" \
    --baselines "$RULE_BASELINES" \
    --seeds "$TEST_SEEDS" \
    --episodes "${RULE_EPISODES:-1000}" \
    --steps "$STEPS" \
    --n-leo "$N_LEO" \
    --device "$RULE_DEVICE" \
    --output-dir "$OUTPUT_ROOT/rules_actual"
}

formal_ablation() {
  require_preflights
  run python scripts/run_ablation_suite.py \
    --config-root trisatflow/configs/ablations \
    --ablations "${ABLATION_CFGS:-no_mask visibility_only completion_safe full_mask no_gnn static_gnn temporal_gnn no_cost_prior}" \
    --upper mappo \
    --lower maddpg \
    --episodes "${ABLATION_EPISODES:-1000}" \
    --steps "$STEPS" \
    --n-leo "$N_LEO" \
    --train-seeds "$TRAIN_SEEDS" \
    --val-seeds "$VAL_SEEDS" \
    --test-seeds "$TEST_SEEDS" \
    --device "$DEVICE" \
    --output-root "$OUTPUT_ROOT/ablation_actual"
  mkdir -p "$OUTPUT_ROOT/stress"
}

formal_learning() {
  require_preflights
  run python scripts/sweep_learning_baselines.py \
    --config "$CONFIG" \
    --baselines "$LEARNING_BASELINES" \
    --episodes "${LEARNING_EPISODES:-1000}" \
    --steps "$STEPS" \
    --n-leo "$N_LEO" \
    --train-seeds "$TRAIN_SEEDS" \
    --val-seeds "$VAL_SEEDS" \
    --test-seeds "$TEST_SEEDS" \
    --device "$DEVICE" \
    --output-root "$OUTPUT_ROOT/learning_baselines"
}

formal_replay() {
  require_satedgesim_root
  gate "$OUTPUT_ROOT/preflight_satedgesim/GATE_OK"
  gate "$OUTPUT_ROOT/preflight_satedgesim/SATEDGESIM_LOWER_ACTION_BINDING_OK"
  echo "formal-replay requires selected checkpoints from formal-main; run replay_on_satedgesim.py per checkpoint into $OUTPUT_ROOT/replay_actual" >&2
}

formal_report() {
  gate "$OUTPUT_ROOT/preflight_offline/GATE_OK"
  run python scripts/aggregate_results.py \
    --input-root "$OUTPUT_ROOT/main_actual" \
    --output "$OUTPUT_ROOT/main_actual_summary"
  run python scripts/statistical_tests.py \
    --input-root "$OUTPUT_ROOT/main_actual" \
    --output "$OUTPUT_ROOT/main_actual_summary/significance_tests.csv"
  run python scripts/export_paper_tables.py \
    --input-root "$OUTPUT_ROOT" \
    --primary-semantic-class "$PRIMARY_SEMANTIC_CLASS" \
    --formal \
    --output-dir "$OUTPUT_ROOT/report/tables"
  run python scripts/plot_paper_results.py \
    --input-root "$OUTPUT_ROOT" \
    --primary-semantic-class "$PRIMARY_SEMANTIC_CLASS" \
    --formal \
    --output-dir "$OUTPUT_ROOT/report/figures"
}

case "$MODE" in
  preflight-offline) preflight_offline ;;
  preflight-satedgesim) preflight_satedgesim ;;
  build-traces) build_traces ;;
  dry-run) dry_run ;;
  formal-main) formal_main ;;
  formal-rules) formal_rules ;;
  formal-ablation) formal_ablation ;;
  formal-learning) formal_learning ;;
  formal-replay) formal_replay ;;
  formal-report) formal_report ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage
    exit 2 ;;
esac
