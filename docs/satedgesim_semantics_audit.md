# SatEdgeSim Short-Term Replay Semantics Audit

Date: 2026-06-29

## Scope

This repair is intentionally limited to Python replay and summary semantics. It does not bind
TriSatFlow continuous lower actions to SatEdgeSim native VM MIPS, network bandwidth, or power
models.

Current validation mode:

- `satedgesim_validation_mode`: `candidate_level_discrete_replay`
- `continuous_resource_applied_to_native_scheduler`: `false`
- `cpu_share_effective`: `false`
- `bandwidth_share_effective`: `false`
- `tx_power_ratio_effective`: `false`

Claim guard:

> Continuous resource fields were accepted by API but not applied to native VM/network/power scheduling; interpret as candidate-level replay.

## Java Static Audit

Files inspected:

- `satedgeSimv2/SatEdgeSim/edu/weijunyong/satedgesim/server/RlAction.java`
- `satedgeSimv2/SatEdgeSim/edu/weijunyong/satedgesim/server/ExecutionReceipt.java`
- `satedgeSimv2/SatEdgeSim/edu/weijunyong/satedgesim/server/RlDecisionBridge.java`
- `satedgeSimv2/SatEdgeSim/edu/weijunyong/satedgesim/TasksOrchestration/ExternalRLOrchestrator.java`

Findings:

- `RlAction` stores `cpuShare`, `bandwidthShare`, and `txPowerRatio` and documents them as
  accepted REST-schema fields for a future custom CPU/network model.
- `ExternalRLOrchestrator.findVM` delegates only the VM-selection decision to `RlDecisionBridge`
  and keeps SatEdgeSim's original task lifecycle and network model unchanged.
- `ExecutionReceipt` separates `accepted`, `actionAccepted`, `executionScheduled`,
  `taskCompleted`, and `taskSucceeded`, so Python summaries must not treat action acceptance as
  task completion success.
- `energyDelta` is a receipt-level raw counter delta. It is not final per-task energy and is now
  reported as `receipt_energy_delta`.

No Java build was run in this phase. The `SatEdgeSim` directory in this checkout does not expose a
quick Gradle/Maven wrapper at its root, and the requested change is a Python-side semantics guard
rather than a Java scheduler modification.

## Summary Semantics

The Python summary now separates:

- `receipt_accept_ratio`: RL API accepted the submitted action receipt.
- `scheduling_success_ratio`: SatEdgeSim accepted candidate scheduling for the chosen target.
- `scheduling_acceptance_rate`: the scheduling-stage metric to use when completion evidence is
  unavailable.
- `intent_execution_match_ratio`: abstract action mapped to the intended executed tier.
- `no_fallback_ratio`: no fallback was used.
- `completion_success_ratio`: emitted only when completion receipt or final task-status metrics
  exist.

If completion evidence is unavailable, the summarizer suppresses `success_rate` and appends
`completion_receipt_unavailable_success_rate_suppressed` to `warnings`.

If completion evidence is available, `success_rate` and `success_ratio` are retained only as
deprecated compatibility aliases for `completion_success_ratio`, with
`deprecated_success_rate_alias=true`.

## Energy Semantics

The Python summary now reports:

- `final_cumulative_energy`: final SimLog/simulator counter when available.
- `receipt_energy_delta`: sum of receipt-level raw energy deltas when available.
- `energy_source`: `receipt_delta`, `simlog_final`, `task_final`, or `unavailable`.
- `energy_unit`: normalized to `J`, `normalized`, `simulator_counter`, or `unknown`.

When the source is `receipt_delta`, the output does not use names such as `final_task_energy`.

## Acceptable Claim

Current Table 5 title should be:

**SatEdgeSim candidate-level action-mapping replay**

This result can support whether a TriSatFlow upper-level discrete action maps to a feasible
SatEdgeSim candidate tier under the replay bridge. It cannot support full hybrid closed-loop
validation or any claim that continuous lower resources have been validated by SatEdgeSim native
VM/network/power scheduling.
