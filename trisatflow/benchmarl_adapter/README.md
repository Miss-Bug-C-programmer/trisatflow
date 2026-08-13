# BenchMARL/TorchRL adapter notes (experimental)

The runnable prototype in `trisatflow/` mirrors BenchMARL's core separation:

- **Task / environment**: `trisatflow.envs.GeoLeoGroundEnv`
- **Model**: `trisatflow.models.TopologyEncoder`, equivalent in role to `benchmarl/models/gnn.py`
- **Upper discrete algorithms**: `mappo`, `ippo`, `iql`, `vdn`, `qmix`
- **Lower continuous algorithms**: `maddpg`, `iddpg`, `masac`, `isac`
- **Experiment config**: YAML configs under `trisatflow/configs/`

The algorithm names are intentionally aligned with BenchMARL 1.5.1 files included in this repo:

```text
benchmarl/algorithms/mappo.py
benchmarl/algorithms/ippo.py
benchmarl/algorithms/iql.py
benchmarl/algorithms/vdn.py
benchmarl/algorithms/qmix.py
benchmarl/algorithms/maddpg.py
benchmarl/algorithms/iddpg.py
benchmarl/algorithms/masac.py
benchmarl/algorithms/isac.py
```

## Status

- `trisatflow/benchmarl_adapter/torchrl_env.py` provides runnable adapters:
  - `TriSatFlowTorchRLEnv` (flat keys)
  - `TriSatFlowBenchMARLEnv` (grouped keys for BenchMARL task usage)
- BenchMARL task registry now includes `trisatflow/mixed_small`
  (`benchmarl/environments/trisatflow/common.py` + `benchmarl/conf/task/trisatflow/mixed_small.yaml`).
- The adapter remains **experimental** for paper-scale training, while native
  `HierarchicalTrainer` remains the primary audited training path.

## Why the native path remains dependency-light

1. The sandbox used to generate this prototype does not have `torchrl` / `tensordict` installed.
2. BenchMARL's standard experiment abstraction assumes one action spec per task, while TriSatFlow needs a coupled mixed-action hierarchy: upper discrete offloading + lower continuous resource allocation.
3. The cross-layer feedback, feasibility signals, and Lyapunov reward are easier to validate first in pure PyTorch.

## Current practical path (recommended for experiments)

Use:

```bash
python scripts/sweep_algorithm_combinations.py --upper all --lower all
```

This runs all supported algorithm combinations and persists:

```text
outputs/algorithm_sweep/sweep_summary.csv
outputs/algorithm_sweep/seed_<seed>/upper_<upper>__lower_<lower>/metrics.csv
```

## Current adapter capabilities

- Supports `reset`, `step`, `observation_spec`, `action_spec`, `reward_spec`, `done_spec`.
- Supports random and scripted TorchRL rollouts via `EnvBase.rollout`.
- Uses flat action keys:
  - `upper_action`: shape `[n_leo]`, categorical over 4 tiers
  - `lower_action`: shape `[n_leo, 3]`, bounded `[0, 1]`
- Supports grouped BenchMARL-style keys under `"leo"` with
  `("leo", "action", "upper_action")` and `("leo", "action", "lower_action")`.
- Exposes padded graph tensors with fixed specs:
  - `edge_index`: `[2, graph_max_edges]`
  - `edge_attr`: `[graph_max_edges, edge_feature_dim]`
  - `edge_mask`: `[graph_max_edges]`
  - `edge_count`: `[1]`

Current limitations:

1. Mixed hierarchical action structure is exposed as a composite action; not all
   BenchMARL algorithm configs are validated with this action layout yet.
2. Graph tensors are padded to a fixed maximum; extra edges are truncated if
   `graph_max_edges` is set too small.
3. It is a runnable integration bridge, but native hierarchical trainer remains
   the recommended path for paper-scale reproducible sweeps.

## Suggested full BenchMARL integration path on a GPU server

1. Install BenchMARL/TorchRL dependencies from the repository root.
2. Wrap `GeoLeoGroundEnv.reset/step` in a TorchRL `EnvBase` with a single agent group named `leo`.
3. Expose specs:
   - observation: `(leo, observation)` with shape `[n_leo, 12]`
   - discrete upper action: `(leo, upper_action)` with shape `[n_leo]`
   - continuous lower action: `(leo, lower_action)` with shape `[n_leo, 3]`
4. Either:
   - keep `HierarchicalTrainer` as the main mixed-action coordinator and swap the internal update classes, or
   - run two BenchMARL experiments per time step with a shared replay/trajectory object and explicit cross-layer reward handoff.

The first option is recommended for the paper prototype because it keeps the hierarchy, GNN state, feasibility feedback, and Lyapunov queue reward in one auditable training loop.
