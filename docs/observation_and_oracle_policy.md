# Observation And Oracle Policy

## Why default is safe

Primary experimental runs must avoid privileged leakage. In TriSatFlow, cost-prior and oracle-aligned signals can be useful for ablation/debug, but they are not part of the default observable state.

Default behavior:

- `observation.mode: safe_observable`
- `observation.include_oracle_cost: false`
- `observation.include_cost_prior_features: false`
- `reward.mode: physical_weighted`
- `reward.use_oracle_cost_components: false`
- `policy_regularization.enabled: false`
- `algo.policy_head: gnn_only`

This keeps the main policy on observable/estimated physical state only.

## Modes

- `safe_observable`
  - Main experiment default.
  - Cost-prior features are blocked.
  - Oracle-aligned reward is blocked.
  - Policy regularization is disabled by default.

- `cost_prior_ablation`
  - Enables privileged cost-prior features for ablation.
  - Oracle cost components remain disabled.

- `oracle_debug`
  - Debug-only mode.
  - Oracle/cost-prior signals may be exposed.
  - If trace/live row includes oracle fields, observation builder reads:
    - `local_oracle_normalized_cost`
    - `neighbor_oracle_normalized_cost`
    - `geo_oracle_normalized_cost`
    - `ground_oracle_normalized_cost`
  - Runtime prints warning; do not use as primary result setting.

## Feature access labels

`trisatflow/envs/obs_schema.py` labels each feature as one of:

- `observable`
- `estimated`
- `privileged`
- `oracle/debug-only`

Cost slots (`*_normalized_cost`) are treated as `oracle/debug-only` only when
`observation.mode=oracle_debug` and `include_oracle_cost=true`; otherwise they
stay `privileged`.

## Runtime warnings

The trainer prints explicit warnings when privileged features or oracle debug mode are enabled, and when a safe mode override disables privileged settings from legacy configs.

## Legacy config behavior

Older configs that implicitly relied on cost-prior/oracle are auto-mapped to the corresponding ablation/debug mode with warning (`legacy_auto_enabled=true`). New configs should set `observation.mode` explicitly.

## Sweep ablations

`scripts/sweep_algorithm_combinations.py` now supports:

- `--observation-ablation no-cost-prior`
- `--observation-ablation cost-prior-features-only`
- `--observation-ablation cost-prior-regularization`
- `--observation-ablation oracle-debug`
