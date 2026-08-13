# Experiment matrix for IEEE IoT-J-oriented validation

## Implemented algorithm combinations

The current TriSatFlow prototype supports direct upper/lower algorithm sweeps.

| Layer | Action type | Algorithms | BenchMARL family |
|---|---:|---|---|
| Upper global offloading | Discrete | `mappo` | MAPPO |
| Upper global offloading | Discrete | `ippo` | IPPO |
| Upper global offloading | Discrete | `iql` | IQL |
| Upper global offloading | Discrete | `vdn` | VDN |
| Upper global offloading | Discrete | `qmix` | QMIX |
| Lower resource allocation | Continuous | `maddpg` | MADDPG |
| Lower resource allocation | Continuous | `iddpg` | IDDPG |
| Lower resource allocation | Continuous | `masac` | MASAC |
| Lower resource allocation | Continuous | `isac` | ISAC |

## Combination sweep command

```bash
python scripts/sweep_algorithm_combinations.py \
  --config trisatflow/configs/small.yaml \
  --upper all \
  --lower all \
  --episodes 50 \
  --seeds 7,11,13,17,19 \
  --output-root outputs/iotj_algorithm_sweep
```

Outputs:

```text
outputs/iotj_algorithm_sweep/sweep_summary.csv
outputs/iotj_algorithm_sweep/seed_<seed>/upper_<upper>__lower_<lower>/metrics.csv
```

## CSV metrics emitted per episode

- `mean_delay`
- `mean_energy`
- `mean_queue`
- `mean_service`
- `mean_arrivals`
- `mean_system_cost`
- `mean_deadline_violation`
- `mean_feasibility`
- `mean_lyapunov_drift`
- `upper_local_ratio`
- `upper_neighbor_ratio`
- `upper_geo_ratio`
- `upper_ground_ratio`
- algorithm-specific losses and entropy/Q terms

## Sweep summary metrics

The sweep runner writes `sweep_summary.csv` with:

- upper/lower algorithm names
- seed
- scenario size
- final metrics
- tail-window mean metrics
- per-run metrics path
- run status and error message if a combination fails

## Recommended reviewer-facing comparisons

| Purpose | Upper algorithm | Lower algorithm | Reviewer-facing claim |
|---|---|---|---|
| Proposed default | MAPPO | MADDPG | Mixed-action hierarchy with CTDE and continuous control |
| Independent policy ablation | IPPO | MADDPG | Tests whether upper centralized critic helps |
| Independent value ablation | IQL | MADDPG | Tests whether cooperative upper value decomposition is needed |
| Additive value decomposition | VDN | MADDPG | Tests whether cooperative value factorization helps offloading |
| Monotonic value mixing | QMIX | MADDPG | Tests whether state-conditioned mixing improves cooperative offloading |
| Independent lower control | MAPPO | IDDPG | Tests whether lower-layer centralized critic matters |
| Entropy-regularized lower CTDE | MAPPO | MASAC | Tests stochastic continuous allocation robustness |
| Fully independent entropy baseline | IPPO | ISAC | Tests maximum decentralization |

## Structural ablations to add for the paper

| Ablation | Code knob | Expected reviewer question answered |
|---|---|---|
| w/o GEO | Make action 2 infeasible or remove it from action space | Is GEO actually useful despite propagation delay? |
| w/o Ground | Make action 3 infeasible or remove it from action space | Is ground cloud driving all gains? |
| w/o ISL | Remove neighbor action and ISL edges | Are LEO cooperation and topology modeling needed? |
| w/o GNN | Replace `TopologyEncoder` with per-node MLP | Does graph encoding matter? |
| w/o Lyapunov | Remove drift term from reward | Does queue-aware control improve long-term stability? |
| w/o cross-layer feedback | Remove feasibility/delay feedback from upper reward | Is the method more than A+B stacking? |
