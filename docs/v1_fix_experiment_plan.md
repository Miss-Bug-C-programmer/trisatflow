# V1-Fix Experiment Plan

## Goal
Stabilize a submission-ready v1 experimental framework that supports three parallel tracks:

1. run immediately on experiment platform,
2. keep code iteration moving,
3. start manuscript drafting with reproducible tables/figure data.

## Profile Lock
- `mobility_aware_main_v1`
  - `success_profile=paper_strict`
  - `action_mask_mode=mobility_safe`
  - mobility fields retained in state/replay outputs
  - `MOBILITY_AWARE_PROFILE_STATUS = partial`
- `mobility_stress_visible_v1`
  - `success_profile=paper_strict`
  - `action_mask_mode=visible_only`
  - stress/robustness only
- `preflight_lenient_v1`
  - engineering-only validation profile
  - not used as primary paper result

## V1-Core Matrix
- profiles: `mobility_aware_main_v1`, `mobility_stress_visible_v1`
- architectures: `only_leo`, `leo_geo`, `leo_ground`, `full`
- baselines:
  - static: `local_only`, `neighbor_only`, `geo_only`, `ground_only`
  - heuristic: `random_visible`, `cost_greedy`, `mobility_risk_greedy`
  - TriSatFlow: `tri_mappo_maddpg`
- seeds: `13,17,23`
- max decisions: `500`

This subset is prioritized for paper-draft tables.

## V1-Extended Matrix
- includes: `hmadrl_maddqn_ddpg`, `weight_greedy_sweep`, `random_mobility_safe`, `round_robin_visible`, `tri_mappo_maddpg`
- intended for post-v1 strengthening and reviewer-facing supplement
- `hmadrl_maddqn_ddpg` is currently partial-training integration (replay facade + MADDQN module ready).

## TriSatFlow Checkpoint Binding
`tri_mappo_maddpg` is explicitly configured via:

```yaml
tri_mappo_maddpg:
  checkpoint_path: ...
  eval_mode: raw_argmax|stochastic_eval|margin_cost_tiebreak
  device: cpu|cuda
```

If `checkpoint_path` is missing, matrix marks `missing_checkpoint` and does not fallback.

## Paper Artifacts
- table exporter: `scripts/v1_fix/export_paper_tables.py`
- figure-data exporter: `scripts/v1_fix/export_figure_data.py`
- output roots contain:
  - `summary_matrix.csv/json`
  - `paper_tables/*.csv|*.md`
  - `figure_data/*.csv`

## Non-Blocking TODO
- full HMADRL training loop integration in matrix training path
- stronger mobility-aware profile calibration (contact/lifetime fidelity)
- optional weight-greedy parameter sweep automation

## Main Metrics Policy
- main: success/delay/mobility-failure/regret/execution reliability
- energy: tracked only as optional metric with `requires_manual_audit`
