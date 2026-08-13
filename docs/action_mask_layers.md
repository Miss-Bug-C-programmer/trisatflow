# Action Mask Layers (Round 11)

TriSatFlow upper-action masking is now decomposed into explicit layers in `trisatflow/envs/action_masks.py`.

## Layers

- `none`:
  no physics-aware filtering; only architecture constraints are applied.
- `visibility`:
  filters actions with no currently reachable target (link not visible / unavailable now).
- `completion_safe`:
  filters actions that are unlikely to finish under current observable/predicted link window constraints.
- `mobility_risk`:
  filters actions with high handover/mobility failure risk from current observable/estimated link stability.
- `full`:
  sequentially applies `visibility -> completion_safe -> mobility_risk`.

## Privileged-Info Policy

- `visibility`: uses current connectivity/mask observations.
- `completion_safe`: uses trace/runtime completion-safe flags and observable/predicted window margins only.
- `mobility_risk`: uses trace/runtime mobility-safe risk indicators only.

These layers do **not** use oracle future total cost/delay labels. Oracle-only signals remain isolated in oracle scripts/baselines.

## Config

```yaml
environment:
  action_mask:
    mode: full
    enable_visibility_mask: true
    enable_completion_safe_mask: true
    enable_mobility_risk_mask: true
```

Backwards compatibility:

- legacy `scenario.action_mask_mode` (`visible_only`, `mobility_safe`, `completion_safe`) is still supported.
- default `scenario.action_mask_layer_mode=legacy` preserves old behavior mapping.
