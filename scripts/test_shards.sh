#!/usr/bin/env bash
set -euo pipefail

OUT="${1:?usage: test_shards.sh <output-dir>}"
mkdir -p "$OUT"
STATUS_FILE="$OUT/shard_status.tsv"
printf 'shard\tstatus\tlog\n' > "$STATUS_FILE"

shards=(
  "tests/test_action_masks.py"
  "tests/test_units_and_metrics_schema.py"
  "tests/test_config_inheritance_and_contract.py"
  "tests/test_offline_baseline_adapter.py"
  "tests/test_checkpoint_selection_protocol.py"
  "tests/test_paper_ready_pipeline_v3.py"
)

failed=0
for shard in "${shards[@]}"; do
  name="$(basename "$shard" .py)"
  log="$OUT/${name}.log"
  if timeout 120s python -m pytest -q "$shard" >"$log" 2>&1; then
    status=0
  else
    status=$?
    failed=1
  fi
  printf '%s\t%s\t%s\n' "$shard" "$status" "$log" >> "$STATUS_FILE"
done

if [[ "$failed" -ne 0 ]]; then
  printf 'One or more test shards failed; see %s\n' "$OUT" >&2
  exit 1
fi
