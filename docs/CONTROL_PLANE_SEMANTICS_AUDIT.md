# TriSatFlow Stage-2 Control-Plane Semantics Audit

This audit is the implementation contract for endogenous replanning. It
separates paper semantics from adapter capability and from diagnostic/oracle
evaluation. The controller must remain valid when the future stochastic
realisation is not available at decision time.

## Findings carried forward from Stage 1

| Finding | Stage-2 disposition |
|---|---|
| Full planner state was acquired before scope generation | Fixed in the proposed path: descriptors are built from cheap summaries; `PlannerState` is acquired only after positive VoC. |
| Fidelity was used as a handcrafted benefit multiplier | Removed from the proposed path. It is available only as `heuristic_fidelity_multiplier` ablation. |
| Hold cost, candidate benefit and candidate cost were conflated | `OutcomeEstimate` and `BenefitEstimate` separate hold outcome, candidate outcome, delay-interval cost and decision-plane cost. |
| Heterogeneous raw resources were directly added | Raw quantities remain separate and are priced by unit-specific coefficients. |
| Resource duals ignored realized consumption and physical duration | `ResourceBudgetState.update(cost, holding_time_sec=...)` uses current raw consumption rates. |
| Solver wall-clock was silently used as simulated delay | `solver_wallclock_sec` and `solver_simulated_latency_sec` are distinct. The wall-clock fallback is explicit ablation only. |
| Scope volume was recorded as migration volume | Configuration structural diff and apply receipt populate realized reconfiguration fields. |
| MARL scope/budget claims exceeded actual adapter capability | Hierarchical MARL reports no scope-aware acquisition or full inference restriction unless a configured provider implements it. |
| Viability acted as a final intervention label | Viability is a conservative screening gate; final selection is delay-aware VoC. |

## Decision-time information boundary

The proposed controller may consume:

- cheap monitor summaries, current persistent configuration and its age;
- deterministic/cached contact information exposed by the backend;
- typed uncertainty and lower-bound service-rate summaries;
- planner capability metadata and causal planning descriptors;
- current decision-resource prices and budgets.

It may not consume future arrivals, future queue trajectories, future channel
realisations, future remote load, post-decision rewards, or offline oracle
labels while selecting KEEP, scope, fidelity, budget or a planner.

Every monitor, descriptor, benefit estimate and planner-state acquisition carries
`future_stochastic_truth_used=False` in the proposed path. Oracle values remain
allowed only in explicitly marked offline evaluation modules.

## Selective acquisition contract

The sequence is:

1. `get_monitor_state` returns a low-cost `MonitorState`.
2. Viability screens the current configuration over a service horizon.
3. The controller generates candidate scopes and `PlanningDescriptor` objects
   without calling `get_planner_state`.
4. A common-horizon `BenefitEstimate` and decision-resource cost are produced
   for each planner/scope/budget candidate.
5. Only a positive VoC triggers heavy state acquisition and planner execution.

Scope-aware and budget-aware acquisition are true only when both the planner
and physical backend advertise the relevant capability. A legacy full-state
fallback is labeled `full_state_acquisition_compatibility`; disabling that
compatibility ablation rejects the run instead of claiming selective sensing.

## Delay and outcome contract

For monitor time `t`, common horizon `H` and modeled delay `δ`, the old
configuration remains active during `[t, t+δ]`; the candidate is evaluated on
`[t+δ, t+H]`. The benefit estimator records both intervals and uncertainty.

`solver_wallclock_sec` is a host measurement. It does not advance the physical
world. Physical delay is `enforced` only if the backend advances simulation time
and the before/after time or structured receipt verifies at least the requested
delta. A missing capability is reported or rejected when strict enforcement is
configured; Python sleep is never a physical substitute.

Post-delay validation is a separate backend-mediated check. A rejected or
unverifiable configuration is not counted as an accepted intervention.

## Reconfiguration accounting contract

`scope_normalized_volume` describes the requested eligible subset. It is not a
measurement of migration. The execution path computes `change_counts` from the
old and projected `PersistentConfiguration` and records predicted fields. A
structured backend apply receipt may overwrite these with measured bytes,
migration volume and changed binding counts; the discrepancy is retained in
metadata.

Decision-plane cost and data-plane utility remain separate. Cost summaries use
raw unit fields plus explicit prices; physical-second-normalized metrics are
reported separately from totals.

## Explicit ablations

The following controls are permitted only as labeled ablations:

- `always_high_fidelity`;
- `cost_blind_planner_selection`;
- `heuristic_fidelity_multiplier`;
- `mean_voc` or `lcb_voc`;
- `full_state_acquisition_compatibility`;
- fixed-period/state-change timing, solver-latency-only and reward-penalty delay.

None of these changes the proposed-paper semantics silently.

