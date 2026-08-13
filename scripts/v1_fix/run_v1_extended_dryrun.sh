#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cpu}"
SEEDS="${SEEDS:-13,17,23}"
MAX_DECISIONS="${MAX_DECISIONS:-500}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/v1_extended_dryrun}"

python scripts/run_experiment_matrix.py \
  --config trisatflow/configs/v1_fix/experiment_matrix_v1_extended.yaml \
  --dry-run \
  --device "${DEVICE}" \
  --seeds "${SEEDS}" \
  --max-decisions "${MAX_DECISIONS}" \
  --output-root "${OUTPUT_ROOT}"
