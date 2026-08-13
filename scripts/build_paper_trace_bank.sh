#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SATEDGESIM_ROOT:-}" ]]; then
  echo "SATEDGESIM_ROOT must point to the SatEdgeSim root for Stage 8" >&2
  exit 2
fi

OUT="${STAGE8_OUT:-outputs/smoke/stage8}"
TRACE_ROOT="${TRACE_ROOT:-traces/paper_v3}"
SATEDGE_BASE_URL="${SATEDGE_BASE_URL:-http://127.0.0.1:8088}"
SATEDGE_PORT="${SATEDGE_PORT:-8088}"
PROJECTION_DECISIONS="${TRACE_BANK_PROJECTION_DECISIONS:-2000}"
SEQUENTIAL_DECISIONS="${TRACE_BANK_SEQUENTIAL_DECISIONS:-500}"
N_LEO="${TRACE_BANK_N_LEO:-12}"
DEVICES_COUNT="${TRACE_BANK_DEVICES_COUNT:-12}"
TIMEOUT_COMPILE="${TIMEOUT_COMPILE:-180s}"
TIMEOUT_REST="${TIMEOUT_REST:-180s}"
TIMEOUT_RECEIPT="${TIMEOUT_RECEIPT:-600s}"
TIMEOUT_EXPORT="${TIMEOUT_EXPORT:-1800s}"
TIMEOUT_AUDIT="${TIMEOUT_AUDIT:-300s}"

mkdir -p "$OUT" "$TRACE_ROOT"
find "$OUT" -maxdepth 1 -type f -delete
export SATEDGE_BASE_URL SATEDGE_PORT
export SATEDGESIM_SETTINGS_ROOT="${SATEDGESIM_SETTINGS_ROOT:-SatEdgeSim/settings/paper_v3_actual}"

timeout "$TIMEOUT_COMPILE" bash -c 'cd "$SATEDGESIM_ROOT" && mvn -q -DskipTests compile'
touch "$OUT/SATEDGESIM_MAVEN_COMPILE_OK"

bash scripts/start_satedgesim_for_paper_v3.sh "$SATEDGE_PORT" > "$OUT/satedgesim_server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

timeout "$TIMEOUT_REST" bash -c 'until curl -fsS "$SATEDGE_BASE_URL/health" >/dev/null; do sleep 1; done'
touch "$OUT/SATEDGESIM_REST_HEALTH_OK"
curl -fsS "$SATEDGE_BASE_URL/version" > "$OUT/version.json"
python - "$OUT/version.json" <<'PY'
import json
import sys
data=json.load(open(sys.argv[1], encoding="utf-8"))
required={"simulator_version","git_commit","rest_api_schema_version","state_schema_version","candidate_cost_estimator_version","lower_action_binding_version","settings_root","settings_sha256","build_time_utc"}
missing=sorted(k for k in required if not data.get(k))
if missing:
    raise SystemExit(f"missing /version fields: {missing}")
if str(data.get("git_commit")).lower() == "unknown":
    raise SystemExit("git_commit is unknown")
if str(data.get("settings_sha256")).startswith("MISSING:"):
    raise SystemExit("settings_sha256 indicates missing settings")
PY
touch "$OUT/SATEDGESIM_VERSION_PROVENANCE_OK"

timeout "$TIMEOUT_REST" python scripts/test_satedgesim_rest_contract.py \
  --base-url "$SATEDGE_BASE_URL" \
  --seed 13 \
  --devices-count "$DEVICES_COUNT" \
  --output "$OUT/rest_contract.json" > "$OUT/rest_contract.stdout"
touch "$OUT/SATEDGESIM_SESSION_LIFECYCLE_OK"

timeout "$TIMEOUT_RECEIPT" python scripts/stress_test_satedgesim_receipt_api.py \
  --base-url "$SATEDGE_BASE_URL" \
  --steps 100 --seed 13 \
  --devices-count "$DEVICES_COUNT" \
  --output "$OUT/receipt_api_stress.json" > "$OUT/receipt_api_stress.stdout"
python - "$OUT/receipt_api_stress.json" <<'PY'
import json
import sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
checks=[
    (d.get("num_http_timeout") == 0, "num_http_timeout"),
    (d.get("num_http_error") == 0, "num_http_error"),
    (d.get("receipt_accept_ratio", 0) >= 0.99, "receipt_accept_ratio"),
    (d.get("decision_id_match_ratio", 0) == 1.0, "decision_id_match_ratio"),
    (d.get("task_id_match_ratio", 0) == 1.0, "task_id_match_ratio"),
    (d.get("intent_execution_match_ratio", 0) >= 0.99, "intent_execution_match_ratio"),
]
bad=[name for ok,name in checks if not ok]
if bad:
    raise SystemExit(f"receipt stress thresholds failed: {bad}")
PY
touch "$OUT/SATEDGESIM_RECEIPT_API_STRESS_OK"

timeout "$TIMEOUT_RECEIPT" python scripts/test_satedgesim_decision_receipt.py \
  --base-url "$SATEDGE_BASE_URL" \
  --steps 100 --seed 13 \
  --devices-count "$DEVICES_COUNT" \
  --output "$OUT/decision_receipt.json" > "$OUT/decision_receipt.stdout"
python - "$OUT/decision_receipt.json" <<'PY'
import json
import sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
http_errors=d.get("http_timeout_count",0)+d.get("http_connection_error_count",0)
checks=[
    (http_errors == 0, "http_errors"),
    (d.get("receipt_accept_ratio", 0) >= 0.99, "receipt_accept_ratio"),
    (d.get("decision_id_match_ratio", 0) == 1.0, "decision_id_match_ratio"),
    (d.get("task_id_match_ratio", 0) == 1.0, "task_id_match_ratio"),
    (d.get("intent_execution_match_ratio", 0) >= 0.99, "intent_execution_match_ratio"),
]
bad=[name for ok,name in checks if not ok]
if bad:
    raise SystemExit(f"decision receipt thresholds failed: {bad}")
PY
touch "$OUT/SATEDGESIM_DECISION_RECEIPT_OK"

if timeout "$TIMEOUT_RECEIPT" python scripts/check_satedgesim_lower_action_binding.py \
  --base-url "$SATEDGE_BASE_URL" \
  --devices-count "$DEVICES_COUNT" \
  --output "$OUT/lower_action_binding.json" > "$OUT/lower_action_binding.stdout"; then
  touch "$OUT/SATEDGESIM_LOWER_ACTION_BINDING_OK"
else
  python - "$OUT/lower_action_binding.json" <<'PY'
import json
import sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
if d.get("status") != "STAGE_BLOCKED_FOR_FULL_HYBRID_CLAIM":
    raise SystemExit("lower action binding failed without explicit full-hybrid block")
PY
  touch "$OUT/SATEDGESIM_LOWER_ACTION_BINDING_BLOCKED_FULL_HYBRID"
fi

export_one() {
  local output="$1"
  shift
  mkdir -p "$(dirname "$output")"
  timeout "$TIMEOUT_EXPORT" python scripts/export_satedgesim_topology_trace.py \
    --base-url "$SATEDGE_BASE_URL" \
    --output "$output" \
    --n-leo "$N_LEO" --devices-count "$DEVICES_COUNT" \
    --success-profile paper_strict \
    --action-mask-mode completion_safe \
    --architecture full \
    --min-link-survival-margin-sec 0.5 \
    --clean-output-folder \
    "$@"
}

export_one "$TRACE_ROOT/actual_projection/train/seed_13.jsonl" \
  --max-decisions "$PROJECTION_DECISIONS" --seed 13 \
  --scenario-profile default --task-source-mode current \
  --trace-mode dense_projection --trace-semantic-class actual_physical_projection
export_one "$TRACE_ROOT/actual_projection/validation/seed_55.jsonl" \
  --max-decisions "$PROJECTION_DECISIONS" --seed 55 \
  --scenario-profile default --task-source-mode current \
  --trace-mode dense_projection --trace-semantic-class actual_physical_projection
export_one "$TRACE_ROOT/actual_projection/test/seed_89.jsonl" \
  --max-decisions "$PROJECTION_DECISIONS" --seed 89 \
  --scenario-profile default --task-source-mode current \
  --trace-mode dense_projection --trace-semantic-class actual_physical_projection
touch "$OUT/ACTUAL_TRACE_BANK_OK"

export_one "$TRACE_ROOT/actual_sequential_live/train/seed_13.jsonl" \
  --max-decisions "$SEQUENTIAL_DECISIONS" --seed 13 \
  --scenario-profile default --task-source-mode current \
  --trace-mode sequential_live --trace-semantic-class actual_physical_sequential_live
export_one "$TRACE_ROOT/actual_sequential_live/validation/seed_55.jsonl" \
  --max-decisions "$SEQUENTIAL_DECISIONS" --seed 55 \
  --scenario-profile default --task-source-mode current \
  --trace-mode sequential_live --trace-semantic-class actual_physical_sequential_live
export_one "$TRACE_ROOT/actual_sequential_live/test/seed_202.jsonl" \
  --max-decisions "$SEQUENTIAL_DECISIONS" --seed 202 \
  --scenario-profile default --task-source-mode current \
  --trace-mode sequential_live --trace-semantic-class actual_physical_sequential_live

export_one "$TRACE_ROOT/controlled_stress_projection/train/seed_13.jsonl" \
  --max-decisions "$PROJECTION_DECISIONS" --seed 13 \
  --scenario-profile mixed_cost_landscape_v2 --task-source-mode round_robin_leo \
  --trace-mode dense_projection --trace-semantic-class controlled_stress_projection
export_one "$TRACE_ROOT/controlled_stress_projection/validation/seed_55.jsonl" \
  --max-decisions "$PROJECTION_DECISIONS" --seed 55 \
  --scenario-profile mixed_cost_landscape_v2 --task-source-mode round_robin_leo \
  --trace-mode dense_projection --trace-semantic-class controlled_stress_projection
export_one "$TRACE_ROOT/controlled_stress_projection/test/seed_89.jsonl" \
  --max-decisions "$PROJECTION_DECISIONS" --seed 89 \
  --scenario-profile mixed_cost_landscape_v2 --task-source-mode round_robin_leo \
  --trace-mode dense_projection --trace-semantic-class controlled_stress_projection
touch "$OUT/CONTROLLED_STRESS_TRACE_BANK_OK"

timeout "$TIMEOUT_AUDIT" python scripts/audit_trace_bank.py \
  --trace-root "$TRACE_ROOT" \
  --require-disjoint-splits \
  --require-provenance \
  --paper-strict > "$OUT/trace_bank_audit.json"
touch "$OUT/TRACE_BANK_AUDIT_OK"
touch "$OUT/GATE_OK"
echo "STAGE8_GATE_OK artifact=$OUT/GATE_OK"
