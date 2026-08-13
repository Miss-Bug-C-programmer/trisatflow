# Iteration Changelog

## Reviewer-driven changes

- Added scenario-level ablation switches in `ScenarioConfig`:
  - `enable_geo`
  - `enable_ground`
  - `enable_isl`
  - `enable_dynamic_skip_isl`
  - `enable_gnn`
  - `enable_lyapunov_reward`
  - `enable_cross_layer_feedback`
- Added `FeatureEncoder` as a same-interface no-message-passing baseline for the w/o-GNN ablation.
- Extended simulator behavior so disabled GEO/ground/ISL resources become infeasible, not silently available.
- Added a no-cross-layer-feedback upper reward path to test whether the two-layer design is actually coupled.
- Added `mean_virtual_delay_queue` to per-episode metrics and CSV outputs.
- Added rule-based baseline registry and evaluator:
  - random
  - local_only
  - neighbor_only
  - geo_only
  - ground_only
  - greedy_queue
  - greedy_delay
  - greedy_energy
  - greedy_weighted_cost
- Added `scripts/evaluate_rule_baselines.py` for persistent baseline CSV files.
- Added `scripts/run_ablation_suite.py` for persistent ablation CSV files.
- Added tests for no-ISL/no-GEO/no-ground and no-GNN ablation paths.

## Validation run in this environment

```bash
pytest -q tests/test_env_and_training.py
python scripts/smoke_test.py --episodes 2 --steps 4 --n-leo 4 --upper-algo mappo --lower-algo maddpg
python scripts/smoke_test.py --episodes 2 --steps 4 --n-leo 4 --upper-algo qmix --lower-algo masac
python scripts/evaluate_rule_baselines.py --episodes 2 --steps 4 --n-leo 4 --seeds 1 --baselines local_only,greedy_weighted_cost
python scripts/run_ablation_suite.py --episodes 1 --steps 3 --n-leo 4 --seeds 1 --ablations full,no_gnn,no_lyapunov,no_cross_layer_feedback,no_geo,no_isl
python scripts/sweep_algorithm_combinations.py --upper all --lower all --episodes 1 --steps 3 --n-leo 4 --seeds 3
```
