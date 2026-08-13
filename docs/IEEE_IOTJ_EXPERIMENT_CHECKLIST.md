# IEEE IoT-J Experiment Checklist for TriSatFlow

This checklist is written from a strict reviewer perspective. Passing the smoke tests in this repository only proves that the prototype is executable. It does not prove that the method is ready for an IEEE IoT-J submission.

## Minimum experimental evidence expected

1. **Algorithm-selection sweep**
   - Upper discrete offloading: MAPPO, IPPO, IQL, VDN, QMIX.
   - Lower continuous allocation: MADDPG, IDDPG, MASAC, ISAC.
   - At least 3 seeds for development; preferably 5–10 seeds for reported results.
   - Report mean ± 95% confidence interval.

2. **Rule-based baselines**
   - Random.
   - Local-only.
   - Neighbor-only.
   - GEO-only.
   - Ground-only.
   - Greedy-delay.
   - Greedy-energy.
   - Greedy-weighted-cost.

3. **Ablations**
   - w/o GNN.
   - w/o Lyapunov drift-plus-penalty.
   - w/o cross-layer feedback.
   - w/o GEO.
   - w/o ground MEC.
   - w/o ISL.
   - ring-ISL only versus dynamic skip ISL.

4. **Load/scalability experiments**
   - Low/medium/high/burst task arrival rates.
   - Increasing LEO count: small, medium, large.
   - Report runtime decision latency, not only reward.

5. **Queue-stability evidence**
   - Long-horizon queue backlog curves.
   - Virtual deadline queue curves.
   - Deadline violation ratio.
   - Evidence that queues do not diverge under sustainable load.

## Commands added in this iteration

```bash
python scripts/evaluate_rule_baselines.py \
  --config trisatflow/configs/small.yaml \
  --baselines all \
  --episodes 20 \
  --seeds 7,11,13,17,19 \
  --output-dir outputs/rule_baselines
```

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

```bash
python scripts/sweep_algorithm_combinations.py \
  --config trisatflow/configs/small.yaml \
  --upper all \
  --lower all \
  --episodes 50 \
  --seeds 7,11,13,17,19 \
  --output-root outputs/algorithm_sweep
```

## Interpretation constraints

The current simulator is a controlled toy simulator. It is useful for algorithm debugging and early design selection. For the final paper, the chosen algorithm combination should be rerun under a higher-fidelity link/mobility model or real ephemeris traces. Otherwise, a reviewer may argue that the performance claim is simulator-specific.
