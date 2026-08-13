# Baseline Mapping

## Placeholder / Experimental
- TriSatFlow baseline id: `hmadrl_maddqn_ddpg`
- status: `experimental_placeholder` (default blocked; requires `--allow-placeholder-baselines`)

## Static Baselines
- `local_only`
- `neighbor_only`
- `geo_only`
- `ground_only`

## Paper-Ready Greedy/Heuristic Baselines
- `random_visible`
- `min_delay_greedy`
- `min_energy_greedy`
- `queue_aware_greedy`
- `mobility_risk_greedy`
- `lyapunov_dpp_greedy`

## RL Baseline
- `tri_mappo_maddpg`

## Debug / Compatibility (not paper-ready by default)
- `random_mobility_safe`
- `round_robin_visible`
- `weight_greedy`
- `cost_greedy` (alias of `min_delay_greedy`)
