# Literature Baseline Mapping

Stage 11 separates paper-ready learning baselines from literature placeholders.
All paper-ready baselines below use the paper-safe observation contract:

- `observation.mode: safe_observable`
- `observation.include_oracle_cost: false`
- `observation.include_cost_prior_features: false`
- `reward.mode: physical_weighted`
- `reward.use_oracle_cost_components: false`

The shared fairness contract is enforced by `scripts/sweep_learning_baselines.py`.
It fixes the trace bank, seed banks, episode budget, step budget, `n_leo`, reward
mode, and observation mode across learning baselines.

## Paper-Ready Learning Baselines

| Baseline id | Implementation | Literature role | Action-space mapping | Paper-ready status |
| --- | --- | --- | --- | --- |
| `flat_ppo` | `trisatflow.models.flat_hybrid_policy.FlatHybridPolicy` + `trisatflow.agents.flat_hybrid_trainer.FlatHybridTrainer` | Flat single-level PPO control baseline | One categorical head selects `local`, `neighbor`, `geo`, or `ground`; one continuous head emits CPU, bandwidth, and power fractions for the selected offloading action. | `paper_ready=true` |
| `flat_mappo` | Same flat hybrid actor with centralized value head | CTDE flat MAPPO-style learning baseline | Same four-way offloading action and continuous resource-allocation vector as `flat_ppo`; centralized critic is used only during training. | `paper_ready=true` |
| `hierarchical_no_gnn` | `HierarchicalTrainer` with `model.topology_encoder=no_gnn` | Hierarchical learning ablation without graph message passing | Same upper offloading and lower resource-allocation action spaces as TriSatFlow. | `paper_ready=true` |

These baselines are not rule policies. They train checkpoints and emit metrics
through the same trace/reward budget used by the main TriSatFlow runs.

## HMADRL / MADDQN-DDPG Mapping

| Item | Current mapping |
| --- | --- |
| Baseline id | `hmadrl_maddqn_ddpg` |
| Registry status | `type=placeholder`, `paper_ready=false` |
| Upper action mapping | Candidate MADDQN upper action maps to the same four abstract offloading tiers: `local`, `neighbor`, `geo`, `ground`. |
| Lower action mapping | Candidate DDPG lower action maps to normalized CPU, bandwidth, and transmit-power fractions. |
| Implemented components | Discrete value-learning and deterministic lower-control components exist in the lightweight trainer stack, but a full HMADRL literature-compatible training/checkpoint/evaluation pipeline is not yet complete. |
| Export policy | Formal exporters fail if this baseline appears in summary rows. It may only become paper-ready after full training, checkpoint replay, test-bank evaluation, and this mapping document are updated. |

The placeholder facade in `trisatflow/baselines/hmadrl_baseline.py` remains a
registry/replay compatibility object. It intentionally selects a random visible
fallback and is blocked from formal paper tables.
