# Paper Claim Guard

This guard states what the paper may and may not claim from the current
reviewer-repair implementation. CPU smoke outputs are engineering validation
only; they are not formal experimental evidence.

## Safe Current Claims

- The code now supports a dimensioned physical mode that reports queue/service
  in cycles, task size in bits, compute in Hz/cycles, delay in seconds, and
  energy in Joules, while retaining legacy normalized mode.
- `normalized_system_cost` is dimensionless and comparable only within the same
  scenario profile and normalizer.
- The Lyapunov component is a Lyapunov-inspired reward shaping / queue
  regularizer, not a stability theorem.
- Rule baselines can be evaluated with explicit lower allocators:
  neutral, same learned when a checkpoint exists, optimized greedy, and
  oracle-like grid diagnostics.
- Main ablation variants must use `safe_observable` without cost-prior,
  oracle-cost, or future trace labels.
- Mask experiments must state `mask_source`. `oracle_trace` is an upper-bound
  diagnostic, while `predicted` and `measured` are deployable modes.
- SatEdgeSim replay summaries distinguish receipt acceptance, scheduling
  success, intent/execution match, no-fallback ratio, completion success, and
  energy source/unit.
- SatEdgeSim resource binding can currently be claimed only at the mode reported
  by metadata, such as candidate-level or resource-aware estimator-bound replay.
- Statistics must use train_seed/checkpoint_id as the primary independent unit.
  Test seeds and online seeds are repeated evaluations or clustered samples.
- P-DQN and flat hybrid actor-critic have tiny CPU update smoke evidence, and a
  small-scale grid oracle exists for oracle-gap diagnostics.

## Forbidden Claims Unless Future Evidence Enables Them

- Do not write: "IPPO+MADDPG is significantly best" unless the real full
  Holm-corrected pairwise tests support it.
- Do not write: "IPPO+MADDPG significantly outperforms all pairings" when the
  best-vs-runner-up Holm result is not significant.
- Do not write: "Lyapunov guarantees queue stability" without a formal drift
  theorem and proof metadata.
- Do not use finite `Qmax` buffer clamping as a proof that the learned policy
  keeps queues bounded.
- Do not write: "outperforms state-of-the-art hybrid-action RL" while Phase 10
  has only tiny smoke results.
- Do not mark P-DQN, flat hybrid AC, or attention candidate baselines
  `paper_ready=true` until full multi-seed experiments finish.
- Do not write: "SatEdgeSim validates the continuous lower allocator in native
  execution" when `native_scheduler_bound=false`.
- Do not write: "full hybrid closed-loop validation" for candidate-only or
  estimator-bound replay.
- Do not report `success_rate` when only scheduling or receipt evidence exists.
  Use `receipt_accept_ratio`, `scheduling_acceptance_rate`, or
  `completion_success_ratio` according to available evidence.
- Do not claim an energy advantage when `energy_source` is unknown or
  unavailable.
- Do not use diagnostic cost-prior results as the main ablation.
- Do not claim transfer across constellation sizes while trace manifest audit or
  transfer stress is blocked.

## Required Wording

For Table 3 algorithm pair conclusions, use:

> IPPO+MADDPG is selected as a mean-ranked reference pairing; the four pairings
> are statistically comparable under Holm-corrected pairwise tests.

For Lyapunov, use:

> Lyapunov-inspired reward shaping / queue regularizer.

For SatEdgeSim candidate-level replay, use:

> SatEdgeSim candidate-level action-mapping replay.

For SatEdgeSim estimator-bound replay, use:

> SatEdgeSim resource-aware estimator-bound replay.

For safe ablation, use:

> The main ablation fixes `observation_policy=safe_observable` and disables
> diagnostic cost-prior and oracle-cost features; no-mask results are interpreted
> as cost-safety operating points rather than pure performance improvements.

## Claim Enablement Conditions

| Claim | Required evidence before use |
|---|---|
| Significant best algorithm pairing | Full train_seed/checkpoint-level tests with Holm-significant best-vs-runner-up result and adequate sample size. |
| Physical metric validity | Metric schema with `unit`, `source`, `normalizer`, `comparable_scope`, plus physical-mode scenario audit. |
| Queue stability guarantee | Formal theoretical DPP mode with drift bound, assumptions, and proof metadata. |
| Strong-baseline superiority | Full multi-seed P-DQN/flat hybrid/optimized DPP/oracle-gap experiments under same mask/observation/lower fairness conditions. |
| Native SatEdgeSim continuous execution | `native_scheduler_bound=true`, completion receipts, and Java evidence that CPU/bandwidth/power affect native execution or accounting. |
| Deployment mask robustness | Predicted/measured mask stress under noise/staleness with oracle_trace separated as non-deployable upper bound. |
| Constellation transfer | Leakage-clean trace manifest plus 16/32/64 stress results. |

## Table Title Guards

- Table 4b should be titled around rule upper policy and lower allocator
  fairness, for example: "Rule upper policies under neutral, learned, optimized,
  and oracle-like lower allocators."
- Table 5 should use one of:
  - "SatEdgeSim candidate-level action-mapping replay"
  - "SatEdgeSim resource-aware estimator-bound replay"
  - "SatEdgeSim native-bound hybrid execution validation" only if
    `native_scheduler_bound=true` with completion evidence.
- Figure 10 should be rerun as safe-observable ablation and should not mix
  diagnostic cost-prior variants into the deployable main result.
