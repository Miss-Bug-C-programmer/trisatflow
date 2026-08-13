#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?usage: run_stage_smoke_gate.sh <stage-id>}"
OUT="outputs/smoke/${STAGE}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-180s}"
POLICY_GATE_TIMEOUT="${POLICY_GATE_TIMEOUT:-120s}"
SMOKE_DEVICE="${SMOKE_DEVICE:-auto}"
rm -rf "$OUT"
mkdir -p "$OUT" "$OUT/train"

bash scripts/test_shards.sh "$OUT/tests"

timeout "$SMOKE_TIMEOUT" python scripts/smoke_test.py \
  --config trisatflow/configs/paper/satedgesim_trace_mixed_v3_safe.yaml \
  --episodes 2 --steps 8 --n-leo 4 \
  --upper-algo mappo --lower-algo maddpg \
  --device "$SMOKE_DEVICE" --output-dir "$OUT/train" 2>&1 | tee "$OUT/train/smoke_test.log"

if [[ "$STAGE" == "stage_09_policy_adaptivity" ]]; then
  mkdir -p "$OUT/train_stress"
  cat > "$OUT/controlled_stress_smoke.yaml" <<'YAML'
extends: ../../../trisatflow/configs/paper/satedgesim_trace_mixed_v3_safe.yaml
scenario:
  topology_trace_path: traces/paper_v3/controlled_stress_projection/train/seed_13.jsonl
  topology_trace_strict: true
  success_profile: paper_strict
  action_mask_mode: completion_safe
  action_mask_layer_mode: full
YAML
  timeout "$SMOKE_TIMEOUT" python scripts/smoke_test.py \
    --config "$OUT/controlled_stress_smoke.yaml" \
    --episodes 2 --steps 8 --n-leo 4 \
    --upper-algo mappo --lower-algo maddpg \
    --device "$SMOKE_DEVICE" --output-dir "$OUT/train_stress" 2>&1 | tee "$OUT/train_stress/smoke_test.log"
fi

python scripts/audit_stage_outputs.py \
  --stage "$STAGE" --input-root "$OUT"

if [[ "$STAGE" == "stage_09_policy_adaptivity" ]]; then
  timeout "$POLICY_GATE_TIMEOUT" python scripts/check_policy_adaptivity.py \
    --metrics "$OUT/train/metrics.csv" \
    --trace-root traces/paper_v3/actual_projection \
    --trace-semantic-class actual_physical_projection \
    --fail-on-deterministic-dominance \
    --output "$OUT/policy_adaptivity_actual.json"
  timeout "$POLICY_GATE_TIMEOUT" python scripts/check_policy_adaptivity.py \
    --metrics "$OUT/train_stress/metrics.csv" \
    --trace-root traces/paper_v3/controlled_stress_projection \
    --trace-semantic-class controlled_stress_projection \
    --fail-on-deterministic-dominance \
    --min-phase-action-divergence 0.01 \
    --output "$OUT/policy_adaptivity_controlled_stress.json"
fi

touch "$OUT/GATE_OK"
printf 'GATE_OK\n' > "$OUT/GATE_STATUS.txt"
