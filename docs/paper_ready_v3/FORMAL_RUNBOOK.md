# Paper-Ready V3 Formal Runbook

## 1. Offline Audit

```bash
bash scripts/run_paper_ready_pipeline_v3.sh preflight-offline
```

This checks prior `outputs/smoke/<stage>/GATE_OK` artifacts, the paper-safe config, trace-bank audit, and Stage 12 reporting input validation.

## 2. CPU Dry-Run

```bash
bash scripts/run_paper_ready_pipeline_v3.sh dry-run --device cpu
```

The dry-run writes only under `outputs/paper_ready_v3/dry_run` and may use smoke-size data. It must not write into the formal trace bank or formal result roots.

## 3. SatEdgeSim Live Preflight

Start the SatEdgeSim REST bridge separately, then run:

```bash
export SATEDGE_BASE_URL="${SATEDGE_BASE_URL:-http://127.0.0.1:8088}"
export SATEDGESIM_ROOT=/path/to/SatEdgeSim-2.3.0

bash scripts/run_paper_ready_pipeline_v3.sh preflight-satedgesim \
  --base-url "$SATEDGE_BASE_URL" \
  --satedgesim-root "$SATEDGESIM_ROOT"
```

The live preflight requires Maven compile, REST health, `/version` provenance, decision receipt checks, and lower-action binding audit. If lower action binding is blocked, SatEdgeSim online hybrid-action claims are not exportable.

## 4. Build Trace Banks

```bash
bash scripts/run_paper_ready_pipeline_v3.sh build-traces \
  --base-url "$SATEDGE_BASE_URL" \
  --satedgesim-root "$SATEDGESIM_ROOT"
```

Trace-bank audit must pass before formal runs.

## 5. Formal Main RL

```bash
bash scripts/run_paper_ready_pipeline_v3.sh formal-main --device cuda
```

Default formal minimums are at least 5 independent train seeds and at least 10 test trace seeds. Override seed lists only by environment variables, and keep train, validation, and test banks disjoint.

## 6. Rule Baselines

```bash
bash scripts/run_paper_ready_pipeline_v3.sh formal-rules --device cuda
```

Rule baselines must use the same paper-safe config, primary trace contract, steps, test seeds, and observation contract as the RL runs.

## 7. Ablations And Learning Baselines

```bash
bash scripts/run_paper_ready_pipeline_v3.sh formal-ablation --device cuda
bash scripts/run_paper_ready_pipeline_v3.sh formal-learning --device cuda
```

Actual and controlled-stress outputs remain separate. Controlled-stress results belong under `outputs/paper_ready_v3/stress/` and are not part of the main actual aggregation.

## 8. Replay And Reporting

```bash
bash scripts/run_paper_ready_pipeline_v3.sh formal-replay --device cuda \
  --base-url "$SATEDGE_BASE_URL" \
  --satedgesim-root "$SATEDGESIM_ROOT"

bash scripts/run_paper_ready_pipeline_v3.sh formal-report
```

Reporting exports CSV, LaTeX, PDF, and PNG files after input validation rejects mixed contracts, debug/oracle rows, placeholder baselines, deprecated metrics, and mixed semantic classes.

## Hard Blocks

Do not start formal GPU runs if any of these are true:

- A prior Stage `GATE_OK` artifact is missing.
- Offline preflight failed.
- SatEdgeSim live preflight failed.
- Maven compile failed.
- `/version` provenance is incomplete.
- Trace-bank audit failed.
- Actual and controlled-stress outputs are mixed.
- Policy adaptivity gate failed.
- Report inputs contain debug/oracle contracts.
- Rule baselines use a legacy evaluator.
- Placeholder literature baselines enter formal reporting.
- `lowerActionBindingVersion=unbound` is used for online hybrid-action conclusions.
