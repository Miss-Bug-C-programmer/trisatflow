#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/v1_core_cuda}"

python scripts/summarize_experiment_matrix.py \
  --input-root "${OUTPUT_ROOT}" \
  --output-csv "${OUTPUT_ROOT}/summary_matrix.csv" \
  --output-json "${OUTPUT_ROOT}/summary_matrix.json"

python scripts/v1_fix/export_paper_tables.py \
  --summary-json "${OUTPUT_ROOT}/summary_matrix.json" \
  --output-dir "${OUTPUT_ROOT}/paper_tables"

python scripts/v1_fix/export_figure_data.py \
  --summary-json "${OUTPUT_ROOT}/summary_matrix.json" \
  --output-dir "${OUTPUT_ROOT}/figure_data"
