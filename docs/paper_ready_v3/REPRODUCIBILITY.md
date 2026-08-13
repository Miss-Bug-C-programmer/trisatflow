# Paper-Ready V3 Reproducibility

This document is the auditable entry point for reproducing the paper-ready v3 pipeline.

## Required Preflights

Run the offline preflight first. It does not start SatEdgeSim and only reads repository configs, prior smoke gates, trace-bank manifests, and reporting inputs.

```bash
bash scripts/run_paper_ready_pipeline_v3.sh preflight-offline
```

Run the CPU dry-run before any formal GPU job.

```bash
bash scripts/run_paper_ready_pipeline_v3.sh dry-run --device cpu
```

Run the live SatEdgeSim preflight with an already running REST bridge. The pipeline does not hardcode a SatEdgeSim checkout path; pass it explicitly or set `SATEDGESIM_ROOT`.

```bash
export SATEDGE_BASE_URL="${SATEDGE_BASE_URL:-http://127.0.0.1:8088}"
export SATEDGESIM_ROOT=/path/to/SatEdgeSim-2.3.0

bash scripts/run_paper_ready_pipeline_v3.sh preflight-satedgesim \
  --base-url "$SATEDGE_BASE_URL" \
  --satedgesim-root "$SATEDGESIM_ROOT"
```

Formal GPU runs are blocked unless these artifacts exist:

```text
outputs/paper_ready_v3/preflight_offline/GATE_OK
outputs/paper_ready_v3/dry_run/GATE_OK
outputs/paper_ready_v3/preflight_satedgesim/GATE_OK
```

## Trace Banks

Build paper trace banks only from the live bridge:

```bash
bash scripts/run_paper_ready_pipeline_v3.sh build-traces \
  --base-url "$SATEDGE_BASE_URL" \
  --satedgesim-root "$SATEDGESIM_ROOT"
```

The required semantic classes are:

```text
traces/paper_v3/actual_projection           actual_physical_projection
traces/paper_v3/actual_sequential_live      actual_physical_sequential_live
traces/paper_v3/controlled_stress_projection controlled_stress_projection
```

Actual and controlled-stress results must not be aggregated together. Reporting uses `--primary-semantic-class actual_physical_projection` for main paper tables and figures.

## Stable Metrics

Paper tables and figures consume only normalized cost fields:

```text
final_normalized_system_cost
normalized_system_cost
```

Deprecated fields such as `mean_system_cost` are rejected by `trisatflow.reporting.input_validation`.

## Output Roots

Smoke and dry-run outputs:

```text
outputs/smoke/stage_13_pipeline/
outputs/paper_ready_v3/dry_run/
```

Formal outputs:

```text
outputs/paper_ready_v3/main_actual/
outputs/paper_ready_v3/rules_actual/
outputs/paper_ready_v3/learning_baselines/
outputs/paper_ready_v3/ablation_actual/
outputs/paper_ready_v3/stress/
outputs/paper_ready_v3/report/
```
