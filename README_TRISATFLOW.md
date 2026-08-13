# TriSatFlow-BenchMARL prototype

This repository contains the original BenchMARL 1.5.1 code plus a runnable research prototype under `trisatflow/` for:

```text
GEO-LEO-Ground three-layer collaborative network
+ dynamic topology GNN encoder
+ upper-layer discrete global offloading MARL
+ lower-layer continuous resource allocation MARL
+ cross-layer reward and feasibility feedback
+ Lyapunov queue / long-term constraint handling
```

## Current algorithm sweep support

The lightweight TriSatFlow trainer supports the following BenchMARL-aligned algorithm families:

| Layer | Action type | Algorithms |
|---|---:|---|
| Upper global offloading | Discrete | `mappo`, `ippo`, `iql`, `vdn`, `qmix` |
| Lower resource allocation | Continuous | `maddpg`, `iddpg`, `masac`, `isac` |

The names map to the BenchMARL modules included in this repo:

```text
benchmarl.algorithms.mappo.Mappo
benchmarl.algorithms.ippo.Ippo
benchmarl.algorithms.iql.Iql
benchmarl.algorithms.vdn.Vdn
benchmarl.algorithms.qmix.Qmix
benchmarl.algorithms.maddpg.Maddpg
benchmarl.algorithms.iddpg.Iddpg
benchmarl.algorithms.masac.Masac
benchmarl.algorithms.isac.Isac
```

Because a full TorchRL/TensorDict stack is not always available, the `trisatflow/agents/*` implementations are compact dependency-light versions aligned with those families. They are designed for fast selection experiments before moving the final candidate into a full BenchMARL experiment.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch pyyaml pytest
pytest -q tests/test_env_and_training.py
python scripts/smoke_test.py --episodes 2 --steps 8 --n-leo 4
```

Expected terminal tail:

```text
SMOKE_TEST_OK upper=mappo lower=maddpg metrics_csv=outputs/smoke_test/metrics.csv checkpoint=outputs/smoke_test/smoke_checkpoint.pt
```

## Train one algorithm combination

```bash
python scripts/train_demo.py \
  --config trisatflow/configs/small.yaml \
  --upper-algo mappo \
  --lower-algo maddpg \
  --episodes 20 \
  --output-dir outputs/single_mappo_maddpg
```

Each run writes:

```text
outputs/<run>/metrics.csv
outputs/<run>/metrics.jsonl
outputs/<run>/checkpoint.pt
outputs/<run>/resolved_config.yaml
```

## Sweep upper/lower algorithm combinations

Run all supported upper/lower combinations:

```bash
python scripts/sweep_algorithm_combinations.py \
  --config trisatflow/configs/small.yaml \
  --upper all \
  --lower all \
  --episodes 50 \
  --seeds 7,11,13,17,19 \
  --output-root outputs/algorithm_sweep
```

The sweep-level summary is written to:

```text
outputs/algorithm_sweep/sweep_summary.csv
```

Every individual combination also has its own persistent `metrics.csv`.

## Rule-based baselines

```bash
python scripts/evaluate_rule_baselines.py \
  --config trisatflow/configs/small.yaml \
  --baselines all \
  --episodes 20 \
  --seeds 7,11,13,17,19 \
  --output-dir outputs/rule_baselines
```

Outputs:

```text
outputs/rule_baselines/baseline_episode_metrics.csv
outputs/rule_baselines/baseline_summary.csv
```

Supported rule baselines:

```text
random, local_only, neighbor_only, geo_only, ground_only,
greedy_queue, greedy_delay, greedy_energy, greedy_weighted_cost
```

## Ablation suite

```bash
python scripts/run_ablation_suite.py \
  --config trisatflow/configs/small.yaml \
  --upper mappo \
  --lower maddpg \
  --ablations all \
  --episodes 50 \
  --seeds 7,11,13,17,19 \
  --output-root outputs/ablation_suite
```

Ablation switches are stored in `ScenarioConfig` and saved to every run's `resolved_config.yaml`:

| Ablation | Scenario switch |
|---|---|
| w/o GNN | `enable_gnn=False` |
| w/o Lyapunov reward | `enable_lyapunov_reward=False` |
| w/o cross-layer feedback | `enable_cross_layer_feedback=False` |
| w/o GEO | `enable_geo=False` |
| w/o ground MEC | `enable_ground=False` |
| w/o ISL | `enable_isl=False` |
| ring ISL only | `enable_dynamic_skip_isl=False` |

Outputs:

```text
outputs/ablation_suite/ablation_runs.csv
outputs/ablation_suite/ablation_summary.csv
```

## Key files

```text
trisatflow/envs/geo_leo_ground_env.py       # GEO-LEO-Ground simulator + Lyapunov/cross-layer reward
trisatflow/models/gnn.py                    # TopologyEncoder + FeatureEncoder w/o-GNN ablation
trisatflow/models/policies.py               # policy / critic / Q / mixer modules
trisatflow/agents/hierarchical_trainer.py   # algorithm-combination trainer
trisatflow/agents/mappo_upper.py            # upper MAPPO
trisatflow/agents/upper_variants.py         # upper IPPO / IQL / VDN / QMIX
trisatflow/agents/maddpg_lower.py           # lower MADDPG
trisatflow/agents/lower_variants.py         # lower IDDPG / MASAC / ISAC
trisatflow/algorithms/registry.py           # supported algorithm registry
trisatflow/baselines/                       # rule-based baselines and evaluator
scripts/sweep_algorithm_combinations.py     # CSV-producing algorithm-combination runner
scripts/evaluate_rule_baselines.py          # CSV-producing rule-baseline runner
scripts/run_ablation_suite.py               # CSV-producing ablation runner
docs/IEEE_IOTJ_EXPERIMENT_CHECKLIST.md      # reviewer-facing experiment checklist
docs/ITERATION_CHANGELOG.md                 # changes made in this iteration
```

## Validation commands used for this iteration

```bash
pytest -q tests/test_env_and_training.py
python scripts/smoke_test.py --episodes 2 --steps 4 --n-leo 4 --upper-algo mappo --lower-algo maddpg
python scripts/smoke_test.py --episodes 2 --steps 4 --n-leo 4 --upper-algo qmix --lower-algo masac
python scripts/evaluate_rule_baselines.py --episodes 2 --steps 4 --n-leo 4 --seeds 1 --baselines local_only,greedy_weighted_cost
python scripts/run_ablation_suite.py --episodes 1 --steps 3 --n-leo 4 --seeds 1 --ablations full,no_gnn,no_lyapunov,no_cross_layer_feedback,no_geo,no_isl
python scripts/sweep_algorithm_combinations.py --upper all --lower all --episodes 1 --steps 3 --n-leo 4 --seeds 3
```

## Important interpretation note

Use the sweep results for algorithm selection and ablation evidence, not as final physical-simulator evidence. For an IEEE IoT-J submission, the selected algorithm pair should then be rerun with larger scenarios, multiple seeds, stronger optimization baselines, and preferably a higher-fidelity satellite mobility/link model or real ephemeris traces.
