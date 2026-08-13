# SatEdgeSim-TriSatFlow Alignment

This document defines the paper-facing contract between SatEdgeSim REST decisions, exported topology traces, TriSatFlow trace replay, and CPU/GPU experiment configs.

## Four Actions

The abstract action space is fixed everywhere:

| id | name | meaning |
|---:|---|---|
| 0 | `local` | execute on the source LEO / source `EDGE_DEVICE` |
| 1 | `neighbor` | execute on another feasible LEO / non-source `EDGE_DEVICE` |
| 2 | `geo` | execute on GEO / `CLOUD` tier |
| 3 | `ground` | execute on ground station / `EDGE_DATACENTER` tier |

## SatEdgeSim `candidateVms`

Each REST `candidateVms[]` item is a concrete VM candidate. The fields used by TriSatFlow are:

- `logicalTier`: canonical tier label, one of `LOCAL`, `NEIGHBOR`, `GEO`, `GROUND`
- `abstractAction`: canonical action id `0..3`
- `abstractActionName`: canonical action name
- `isLocalToSource`, `isRemoteToSource`
- `isFeasible`, `infeasibleReason`
- `sourceDistance`, `propagationDelaySec`
- `estimatedTransmissionRateMbps`, `estimatedTransmissionDelaySec`
- `estimatedComputeCapacity`, `estimatedComputeDelaySec`
- `estimatedQueueLength`

TriSatFlow first trusts these explicit fields. Legacy datacenter-type inference remains only as a compatibility fallback and should be near zero in replay logs.

## `abstractActionMask`

`abstractActionMask = [local, neighbor, geo, ground]`.

Generation rule:

1. Iterate all feasible `candidateVms`.
2. Map each feasible candidate to one of the four abstract actions.
3. Mark that action as visible if at least one feasible candidate exists.

This mask is the source of truth for both training and replay validation. Visibility booleans in traces must be consistent with the mask.

## Dense Trace Schema

Real dense traces are exported to:

- `traces/satedgesim_real_dense_seed13.jsonl`

Synthetic traces must stay clearly named:

- `traces/satedgesim_synthetic_dense_seed13.jsonl`

Each JSONL row contains at least:

- `step`
- `leo_id`
- `local_visible`
- `neighbor_visible`
- `geo_visible`
- `ground_visible`
- `local_rate`
- `neighbor_rate`
- `geo_rate`
- `ground_rate`
- `neighbor_candidate_count`
- `geo_candidate_count`
- `ground_candidate_count`
- `neighbor_min_distance`
- `geo_min_distance`
- `ground_min_distance`
- `neighbor_best_queue`
- `geo_best_queue`
- `ground_best_queue`
- `neighbor_best_delay`
- `geo_best_delay`
- `ground_best_delay`
- `abstract_action_mask`
- `scenario_profile`
- `task_source_mode`
- `is_controlled_rl_scenario`

Real dense export uses source projection at each true SatEdgeSim decision instant: one blocked decision step produces `n_leo` projected source summaries. This is real topology data derived from the live SatEdgeSim state, not synthetic interpolation.

Receipt-fix refresh workflow for the paper trace:

1. restart or reset the Java REST server after receipt/cost-model changes
2. export a fresh dense trace, for example `traces/satedgesim_real_dense_mixed_receiptfix_seed13.jsonl`
3. validate dense coverage and visible ratios
4. compare the new trace against live observations before training
5. run oracle action analysis on the refreshed trace

Do not keep training on a pre-fix trace after changing live `delay/queue/rate` semantics.

## Balanced Four-Tier Profile

When the raw live SatEdgeSim scene does not expose enough `neighbor / geo / ground` opportunities for replay validation, the REST layer can run a controlled RL profile instead of pretending the default scene is balanced.

Supported `scenarioProfile` values:

- `default`
- `balanced_four_tier`
- `geo_favorable`
- `ground_favorable`
- `neighbor_favorable`
- `local_pressure`
- `remote_unavailable`

Supported `taskSourceMode` values:

- `current`
- `round_robin_leo`
- `random_leo`

`balanced_four_tier` contract:

- `local` remains executable
- `neighbor`, `geo`, and `ground` appear with non-trivial visibility
- `ground` is not allowed to be the only remote tier
- `geo` is not allowed to stay permanently invisible
- source tasks rotate across multiple LEOs under `round_robin_leo`
- delay / queue / rate / capacity vary by tier so one tier is not always trivially cheapest

Controlled RL profiles must be explicitly labeled in REST state and exported traces:

- `scenario_profile = balanced_four_tier`
- `task_source_mode = round_robin_leo`
- `is_controlled_rl_scenario = true`

This is different from the original SatEdgeSim default scene. Controlled profiles are valid for paper-facing RL experiments only when they are disclosed as controlled RL scenarios, not as untouched default SatEdgeSim scenarios.

## Strict Trace vs Debug Fallback

Paper configs:

- `scenario.topology_mode: satedgesim_trace`
- `scenario.topology_trace_repeat: true`
- `scenario.topology_trace_strict: true`
- `scenario.action_mask_enabled: true`

Strict mode means:

- missing `(step, leo_id)` is an error
- no analytic fallback is allowed
- `trace_missing_count` must be `0`
- `trace_fallback_count` must be `0`
- `trace_hit_ratio` must be `1.0`

Debug mode may set `topology_trace_strict: false`. In that case missing pairs fall back to the analytic topology and the counters record the damage.

## Replay Intent vs Execution

Replay logs must distinguish:

- policy action distribution: what the frozen policy asked for
- executed tier distribution: what SatEdgeSim actually executed
- visible opportunity distribution: what the environment made feasible

Acceptance checks:

- `intent_execution_match_ratio >= 0.99`
- `fallback_reason_distribution["none"] >= 0.99`
- warn if policy ratios and executed ratios differ by more than `1%`
- warn if only one action tier appears

Forced-action replay is the execution-side sanity check:

- `--force-upper-action local|neighbor|geo|ground`
- `--force-policy random_visible|round_robin_visible`

Its purpose is to prove the live SatEdgeSim action space can actually execute all four abstract actions when visible. If a forced action is not visible, replay must log `fallback_reason = forced_action_not_visible`; it must not silently substitute another tier.

How to distinguish scene coverage from policy collapse:

1. Run `diagnose_satedgesim_action_space.py`.
2. If `neighbor / geo / ground` visible ratios are near zero, classify `scene_coverage_insufficient`.
3. If all tiers are visible but `ground` has a persistent delay/cost advantage, classify `reward_or_cost_bias_ground_dominant`.
4. If visible forced actions map to the wrong executed tier, classify `mapper_bias`.
5. If training on balanced dense traces is diverse but checkpoint replay still chooses one tier, classify `train_replay_distribution_shift`, `learned_policy_ground_collapse`, or `insufficient_policy_training` depending on the training evidence.

## Decision Receipt Contract

Each blocked RL decision now has a unique, monotonic `decisionId`.

`GET /get_state` returns at least:

- `decisionId`
- `taskId`
- `sourceDeviceId`
- `sourceLeoId`
- `scenarioProfile`
- `taskSourceMode`
- `abstractActionMask`
- `candidateVms[]`

`POST /apply_action` must carry the current `decisionId` and `taskId`, plus the intended abstract action and the selected VM identity.

`/apply_action` returns an `ExecutionReceipt` immediately. Replay validation must treat this receipt as the source of truth instead of inferring execution from later state snapshots.

Receipt fields used by TriSatFlow:

- `accepted`
- `actionAccepted`
- `executionScheduled`
- `taskCompleted`
- `taskSucceeded`
- `decisionId`
- `taskId`
- `policyUpperAction`, `policyUpperActionName`
- `selectedVmId`, `selectedVmLogicalTier`, `selectedVmAbstractAction`
- `executedVmId`, `executedLogicalTier`, `executedAbstractAction`
- `intentExecutionMatch`
- `fallbackReason`
- `delay`
- `energyRawCounterBefore`, `energyRawCounterAfter`, `energyDelta`
- `success`
- `failureReason`
- `deadlineMiss`, `queueOverflow`, `vmUnavailable`, `linkUnavailable`
- `taskDropped`, `invalidAction`, `simulationFailure`
- `latencyExceeded`, `resourceExceeded`, `unknownFailure`
- `serverProcessingMs`

Rejection rules:

- stale `decisionId` -> `accepted=false`, `fallbackReason=stale_decision_id`
- mismatched `taskId` -> `accepted=false`, `fallbackReason=task_id_mismatch`
- invisible abstract action -> `accepted=false`, `fallbackReason=action_not_visible`

No silent fallback to another tier is allowed in these cases.

## Success Semantics

`success_ratio` in replay summaries is **task completion success ratio**, sourced from SatEdgeSim final metrics (`successRate`), not from one-step action acceptance.

- `actionAccepted`: REST action payload accepted by bridge.
- `executionScheduled`: task scheduling/execution was enqueued.
- `taskCompleted`: task already reached completion state at receipt time.
- `taskSucceeded`: completed task met success criteria.

Because replay is stepwise scheduling and task completion is asynchronous, many receipts are `taskCompleted=false` at decision time. These rows must be treated as `pending_task_completion`, not as task failures.

Summary contract:

- `scheduling_success_ratio`: action-side reliability (`receipt_accept_ratio`).
- `task_completion_success_ratio`: task-side reliability (`successRate`).
- `success_ratio`: alias of `task_completion_success_ratio`.

Low `task_completion_success_ratio` cannot be ignored even when `scheduling_success_ratio` is high.

## Failure Reason Contract

Replay summaries must publish task-level failure reasons:

- `failure_reason_distribution`
- `task_failure_reason_distribution`
- `receipt_failure_reason_distribution`

Current task-level mapping uses SatEdgeSim final counters:

- `tasksFailedLatency` -> `latency_deadline`
- `tasksFailedMobility` -> `mobility_link`
- `tasksFailedResourcesUnavailable` -> `resource_unavailable`
- `tasksFailedBecauseDeviceDead` -> `device_dead_or_dropped`

Decision-level failure breakdown by action/tier/phase is only valid when completion is synchronously attributable to that decision; otherwise it remains empty and should not be over-interpreted.

## Success Profiles

Replay/reset supports:

- `successProfile=default`
- `successProfile=preflight_lenient`
- `successProfile=paper_strict`

`preflight_lenient` is for engineering integration checks only. It relaxes completion strictness (for example deadline leniency) without changing action feasibility or policy inference semantics.

`paper_strict` and `default` preserve strict task constraints for reporting.

All replay artifacts must record `success_profile`; lenient results must never be reported as strict paper results.

## `/apply_action` HTTP Stability

`/apply_action` must validate the current decision, construct an `ExecutionReceipt`, and return JSON immediately.

The HTTP handler must not block on a long CloudSim advancement loop after the receipt is already known.

Server requirements:

- fixed thread-pool executor
- explicit JSON status codes for success and rejection
- lightweight `ExecutionReceipt` payload only
- receipt write path must flush and close the response body
- broken-pipe and timeout symptoms must be counted in server-side receipt stats

The main race that previously caused client timeouts was: decision `N` resolved inside the bridge, but the simulation thread entered decision `N+1` and overwrote the shared mutable receipt/current-state pointers before the waiting HTTP thread consumed decision `N`'s receipt. The fix is to cache receipts by `decisionId` and deliver the exact cached receipt to the matching `/apply_action` call.

## Why Replay Must Not Use `lastDecision`

`lastDecision` is retained only for compatibility and debugging.

It is not a valid paper-facing replay receipt because it is attached to later state snapshots and can be misread with the wrong `(decisionId, taskId)` pair. This is exactly how intent/execution accounting can drift even when forced single-tier replay is executable.

Correct replay order:

1. `GET /get_state`
2. read `decisionId`, `taskId`, `abstractActionMask`
3. choose policy action from the current state only
4. `POST /apply_action`
5. log the step from the returned `ExecutionReceipt`
6. fetch the next state

The next state must never be used to reinterpret the previous action.

## REST Timeout And Broken-Pipe Debugging

Debug endpoints:

- `GET /health`

## `mixed_cost_landscape_v2`

`mixed_cost_landscape_v2` replaces the old monolithic mixed profile with a phase-based controlled RL scene. The phase only changes workload and network conditions; it does not hardcode the oracle label.

Per REST state and per exported trace row now carry:

- `scenario_profile`
- `scenario_phase`
- `task_type`
- `traffic_phase`
- `cost_estimator_version`
- `queueEstimateSource`

Phase set:

- `local_favorable_phase`
- `neighbor_favorable_phase`
- `geo_favorable_phase`
- `ground_favorable_phase`
- `remote_congested_phase`
- `balanced_contention_phase`

Physical meaning:

- `local_favorable_phase`: small latency-sensitive tasks, low local queue, remote queue and propagation remain non-zero.
- `neighbor_favorable_phase`: fast peer LEO path, low neighbor queue, remote tiers still visible but not free.
- `geo_favorable_phase`: compute-heavy batch tasks with high GEO compute and realistic GEO propagation.
- `ground_favorable_phase`: ground service pipeline with strong ground rate and low ground queue, but GEO remains competitive.
- `remote_congested_phase`: periodic remote congestion so local or neighbor can become rational fallbacks.
- `balanced_contention_phase`: all four tiers keep visible trade-offs; no mechanical uniformity is forced.

## Candidate Cost Estimator

`CandidateCostEstimator` keeps the unified decomposition used by both live candidate views and dense source summaries:

- `propagation_delay`
- `transmission_delay`
- `compute_delay`
- `queue_delay`
- `total_delay = prop + tx + compute + queue`

Current version:

- `cost_estimator_version = v2_phase_calibrated_delay_queue`

The estimator is still shared across live state export and dense projection. `mixed_cost_landscape_v2` only changes phase-aware task demand scales and tier parameters; it does not inject oracle labels or post-hoc action ratios.

## Physical Plausibility Gate

Before training, run:

```bash
python scripts/check_scenario_physical_plausibility.py \
  --trace traces/satedgesim_real_dense_mixed_v2_seed13.jsonl \
  --output outputs/plausibility_mixed_v2_seed13.json
```

The gate checks at least:

- all delays are non-negative
- all rates are strictly positive
- all queues are non-negative
- GEO mean propagation delay stays above neighbor/local
- local propagation delay stays near zero
- every phase and every task type has non-trivial coverage
- no single phase dominates the whole trace
- no tier is always the minimum-delay winner
- queues and delays are not flattened to constants

## Oracle Gate

Run:

```bash
python scripts/oracle_action_analysis.py \
  --source trace \
  --trace traces/satedgesim_real_dense_mixed_v2_seed13.jsonl \
  --n-leo 16 \
  --num-states 8192 \
  --output outputs/oracle_mixed_v2_trace_analysis.json
```

Overall gate:

- `oracle_local_ratio >= 0.03`
- `oracle_neighbor_ratio >= 0.05`
- `oracle_geo_ratio >= 0.05`
- `oracle_ground_ratio >= 0.05`
- `max_oracle_action_ratio <= 0.80`

Phase-level sanity:

- the target tier in each favorable phase must be more common than in the overall oracle distribution
- favorable phases are not allowed to collapse to `100%` one-tier oracle selection
- `balanced_contention_phase` and `remote_congested_phase` are used to preserve mixed opportunity rather than to force uniform action counts

Hardcoding action ratios is explicitly disallowed because it would destroy the paper claim that the oracle is induced by real delay/capacity/queue trade-offs.

## `dense_projection` vs `sequential_live`

- `dense_projection`: paper-facing training trace. One true blocked SatEdgeSim decision is projected across all visible source LEOs. Use this to validate coverage, plausibility, and oracle diversity before training.
- `sequential_live`: live alignment trace for the actual current source only. Use this to check whether exported fields still match live REST observations under real queue evolution.

If `sequential_live` differs from live after receipt and estimator fixes, the remaining explanation must be concrete, such as sequential backlog evolution or reset warm-up effects. Training should not start on an unexplained mismatch.

## Preflight Training Entry

Preflight training is allowed only when all of the following hold:

1. dense trace validation passes
2. physical plausibility passes
3. overall oracle gate passes
4. phase-level oracle sanity passes
5. `sequential_live` compare has no unit mismatch and no unexplained estimator drift

Only after those checks does `trisatflow/configs/satedgesim_trace_mixed_v2.yaml` become the paper-facing preflight config. Training itself remains a separate step.
- `GET /debug/current_decision`
- `GET /debug/last_receipt`
- `GET /debug/receipt_stats`

`/debug/receipt_stats` should expose:

- `numApplyActionCalls`
- `numAccepted`
- `numRejected`
- `numTimeoutSuspected`
- `meanServerProcessingMs`
- `maxServerProcessingMs`
- `fallbackReasonDistribution`

TriSatFlow replay and receipt tests must also log:

- `http_status_code`
- `http_error`
- `server_processing_ms`
- `client_elapsed_ms`

If a client sees a timeout or broken pipe, classify it as transport instability first. Do not reinterpret it as an execution mismatch.

## Live Delay And Queue Features

Live `candidateVms[]` now expose:

- `estimatedTransmissionDelaySec`
- `estimatedComputeDelaySec`
- `estimatedTotalDelaySec`
- `estimatedQueueLength`
- `estimatedTransmissionRateMbps`
- `sourceDistance`
- `propagationDelaySec`
- `estimatedComputeCapacity`
- `queueEstimateSource`

`estimatedTotalDelaySec` is defined as:

- transmission delay
- plus propagation delay
- plus compute delay
- plus queue delay

`queueEstimateSource` meanings:

- `actual`: derived from current VM assignment progress using the live scheduler/history state
- `controlled_estimate`: deterministic scenario-profile estimate used only in controlled RL profiles

Trace export and live replay must use the same cost interpretation, otherwise the result must be classified as `trace_live_distribution_shift`.

Live source summaries and candidate-level fields use lower-camel names such as `localBestDelay` and `localBestQueue`. The shared observation builder must accept both lower-camel and legacy snake-case field names; otherwise live delay/queue values can be silently dropped and appear as zeros or constants during trace/live comparison.

## Candidate Cost Estimator

SatEdgeSim now uses a shared candidate cost estimator for both live `candidateVms[]` and dense source summaries.

Current version:

- `cost_estimator_version = v1_unified_delay_queue`

Fields:

- `estimatedTransmissionRateMbps`
- `sourceDistance`
- `propagationDelaySec`
- `estimatedTransmissionDelaySec`
- `estimatedComputeCapacity`
- `estimatedComputeDelaySec`
- `estimatedQueueLength`
- `estimatedQueueDelaySec`
- `estimatedTotalDelaySec`
- `queueEstimateSource`

Units:

- delay fields: seconds
- rate: Mbps
- compute capacity: MIPS
- task length: MI
- queue length: task count
- queue delay: `queue_length * expected_service_time`

Formulas:

- `transmission_delay = task_size_bits / link_rate_bps`
- `compute_delay = task_length_mi / vm_mips`
- `queue_delay = queue_length * expected_service_time`
- `total_delay = propagation_delay + transmission_delay + compute_delay + queue_delay`

All tiers use the same delay formula. Differences come from tier-specific link rate, compute capacity, propagation distance, and queue length.

## Dense Projection vs Sequential Live

Two real SatEdgeSim trace modes are supported:

- `dense_projection`
- `sequential_live`

`dense_projection`:

- one blocked global decision exports projected summaries for every LEO
- intended for strict trace training
- metadata should include:
  - `trace_generation_mode = dense_projection`
  - `cost_estimator_version`
  - `queueEstimateSource`

`sequential_live`:

- one blocked global decision exports only the current source LEO
- intended for trace/live alignment validation, not for multi-LEO strict-trace training
- metadata should include:
  - `trace_generation_mode = sequential_live`
  - `cost_estimator_version`
  - `queueEstimateSource`

Why both exist:

- training needs dense projected coverage at every global step
- live replay and receipt validation execute sequentially on the current source only
- therefore `sequential_live` is the correct baseline for checking whether the estimator and field mapping align with the live server

Required observation-distribution check before training:

- visible and mask fields must align
- live `delay` must not be all zeros or constants
- live `queue` must not be all zeros or constants
- if `local_delay` or `ground_delay` still shift, explain whether the cause is:
  - trace dense per-source projection vs online sequential live sampling
  - different cost formulas between exporter and `RlStateBuilder`
  - task-size / queue-update timing mismatch
  - unit mismatch
  - field-name mapping bug

Training must not proceed if the remaining shift is unexplained.

If `sequential_live` aligns with live but `dense_projection` does not, the remaining difference is attributable to dense source projection rather than to a receipt/mapping bug. This is acceptable only if the difference is documented and the oracle gate still passes.

## Deterministic Collapse vs Sampled Diversity

Training-time sampled action diversity is not enough to certify policy health.

Required distinction:

- deterministic argmax distribution: what checkpoint replay actually executes when evaluation is deterministic
- stochastic sampled distribution: what exploratory sampling produced during training

If sampled replay is diverse but deterministic argmax is nearly single-action, classify `deterministic_argmax_collapse`.

Training metrics must record:

- `eval_argmax_local_ratio`
- `eval_argmax_neighbor_ratio`
- `eval_argmax_geo_ratio`
- `eval_argmax_ground_ratio`
- `eval_argmax_remote_ratio`
- `eval_policy_entropy`

Interpretation:

- sampled action ratios describe exploratory behavior during training
- `eval_argmax_*` ratios describe the deterministic checkpoint policy
- if sampled ratios are diverse but `eval_argmax_*` collapses to one tier, the diversity came from stochastic sampling, not from a healthy deterministic policy

Required checkpoint replay acceptance:

- `intent_execution_match_ratio >= 0.99`
- `fallback_reason_distribution["none"] >= 0.99`
- policy and executed action ratios should differ by at most `0.01`
- if replay is still single-action, classify one of:
  - `oracle_ground_dominant`
  - `deterministic_argmax_collapse`
  - `trace_live_distribution_shift`
  - `insufficient_training`
  - `reward_normalization_failure`
  - `checkpoint_loading_error`

## Oracle Gate Before Training

Use `oracle_action_analysis.py` on the refreshed receipt-fix trace before CPU preflight.

Purpose:

- determine whether the trace itself exposes multi-action one-step opportunities
- distinguish cost-landscape collapse from policy collapse

Suggested gate for `mixed_cost_landscape`:

- `oracle_local_ratio >= 0.03`
- `oracle_neighbor_ratio >= 0.05`
- `oracle_geo_ratio >= 0.05`
- `oracle_ground_ratio >= 0.05`
- `max_oracle_action_ratio <= 0.80`

If this gate fails, fix `mixed_cost_landscape` first. Do not start preflight training.

Important restriction:

- do not hard-code action ratios
- do not relabel oracle actions by hand
- do not force a tier to be optimal every N steps

The four-action oracle diversity must emerge from the task mix and the cost landscape:

- local: small task, low local queue, remote congestion
- neighbor: good ISL rate, low neighbor queue
- GEO: compute-heavy and moderately delay-tolerant task, strong GEO compute
- ground: large-data or strong terrestrial compute phase with low ground congestion

## Mixed Cost Landscape Logic

`mixed_cost_landscape` is allowed to vary:

- tier visibility thresholds
- task-aware compute intensity effects
- task-aware data intensity effects
- queue lengths
- compute capacities
- link rates
- propagation-distance proxies

It is not allowed to directly encode the desired action histogram.

## Energy Audit

SatEdgeSim `SimLog.getMetricsSnapshot().energyConsumption` is not per-task energy.

Implementation audit:

- CPU energy path accumulates `(power in W / 3600) * seconds`, which is a cumulative `Wh` counter.
- Wireless energy path converts `J` to `Wh` before accumulation.
- Therefore the REST metric is a cumulative total-energy counter across all datacenters.
- Legacy `SimLog` labels saying `W` / `dBW` are misleading for the REST counter.

Replay summary policy:

- log `raw_energy_counter`
- compute per-decision delta as `energy_raw_delta`
- report `mean_energy_per_decision`
- report `mean_energy_raw_delta`
- report `energy_unit = "Wh"`

Paper note:

- energy is now treated as a cumulative-Wh counter delta, but because the legacy labels are inconsistent, it still requires a manual audit before becoming a headline paper metric.
- TriSatFlow training-side `mean_energy` is an internal normalized environment cost term, not SatEdgeSim physical energy.

## Config Use

Paper-facing configs:

- `trisatflow/configs/satedgesim_trace.yaml`
- `trisatflow/configs/satedgesim_trace_preflight.yaml`
- `trisatflow/configs/satedgesim_trace_balanced.yaml`

Debug-only patterns:

- any synthetic trace path
- any config with `topology_trace_strict: false`
- any run with non-zero trace fallback counters
- any replay with non-trivial mapper fallback
- controlled-profile runs that are used only to debug server semantics rather than to train/evaluate the paper policy

## Receipt API Stress Test

Use `scripts/stress_test_satedgesim_receipt_api.py` to validate the end-to-end REST receipt path before any training or checkpoint replay conclusion.

Required acceptance:

- `num_http_timeout = 0`
- `num_http_error = 0`
- `receipt_accept_ratio >= 0.99`
- `decision_id_match_ratio >= 0.99`
- `task_id_match_ratio >= 0.99`
- `intent_execution_match_ratio >= 0.99`
- `fallback_reason_distribution["none"] >= 0.99`

## Preflight Checklist

Before CUDA main experiments:

1. `export_satedgesim_topology_trace.py` produces a real dense trace with expected row count.
2. `validate_satedgesim_trace.py --require-dense` passes with non-zero GEO and ground visibility.
3. `stress_test_satedgesim_receipt_api.py` passes with zero HTTP timeouts and zero HTTP connection errors.
4. `test_satedgesim_decision_receipt.py` passes with receipt-level `decisionId/taskId/intentExecutionMatch >= 0.99`.
5. `round_robin_visible` replay summary shows `intent_execution_match_ratio >= 0.99`.
6. `round_robin_visible` replay summary shows `fallback_reason_distribution["none"] >= 0.99`.
7. Forced-action replay proves `local / neighbor / geo / ground` can all execute when visible.
8. Only after items 3-7 pass may mixed-profile CPU preflight training start.
9. The refreshed receipt-fix mixed trace passes the oracle gate above.
10. CPU preflight training finishes with `trace_hit_ratio = 1.0` and `trace_fallback_count = 0`.
11. `check_action_collapse.py` reports `ACTION_DIVERSITY_OK`.
12. Deterministic eval fields do not show unexplained single-action collapse, or the collapse is explicitly classified.
13. Mixed checkpoint replay does not collapse to a single action tier unless diagnosis clearly shows `oracle_dominant_action`, `deterministic_argmax_collapse`, `trace_live_distribution_shift`, or another named root cause.
14. Energy summary is interpreted as cumulative `Wh` counter deltas, not per-task instantaneous power.

CUDA main experiments are allowed only after the refreshed trace, oracle gate, CPU preflight, deterministic eval, and live checkpoint replay all pass.

## First-Round Training Closure (mixed_v2)

CPU preflight workflow (first closure round):

1. run strict-trace preflight training on CPU (`episodes=50`, `steps=64`)
2. verify strict trace counters are exact (`trace_hit_ratio=1.0`, `trace_fallback_count=0`)
3. run `check_four_tier_policy_health.py` with tail-window acceptance
4. run `inspect_training_deterministic_eval.py` and classify deterministic behavior
5. run `inspect_policy_action_distribution.py` on both `trace` and `live` sources
6. run `replay_on_satedgesim.py` + `summarize_satedgesim_replay.py` for 500 decisions

Deterministic eval meaning:

- deterministic eval is the checkpoint argmax behavior under evaluation mode
- it is the policy behavior that replay will execute if no stochastic sampling is used

Sampled diversity vs argmax diversity:

- sampled diversity describes exploration-time sampling spread
- argmax diversity describes deterministic policy preference
- sampled diversity is not evidence of deterministic policy diversity
- if sampled distribution is broad but argmax is single-action, classify `stochastic_diversity_only` + `deterministic_argmax_collapse`

Checkpoint live replay acceptance line:

- `intent_execution_match_ratio >= 0.99`
- `fallback_reason_distribution["none"] >= 0.99`
- `receipt_accept_ratio >= 0.99`
- `http_timeout_count = 0`
- `http_connection_error_count = 0`
- `|policy_ratio - executed_ratio| <= 0.01` for each action tier

CUDA preflight is allowed only when all of the following hold:

- strict-trace CPU preflight passes
- policy health has no blocker
- deterministic eval has no unexplained single-action collapse
- trace/live policy inspection has no severe distribution shift
- checkpoint replay passes the acceptance line above
- if replay remains single-action, the cause must be explicitly diagnosed (for example `oracle_dominant_action`) and consistent with oracle evidence

Energy metric guidance while `requires_manual_audit` remains:

- do not use SatEdgeSim replay energy as a headline optimization metric
- keep energy fields in logs for audit traceability
- use delay / success / receipt consistency as primary gating metrics for preflight conclusions

## MAPPO Collapse Diagnosis

Observed blocker in mixed_v2 CPU preflight:

- `deterministic_argmax_geo_collapse` while oracle remains multi-action
- typical pattern: sampled distribution is multi-action, deterministic argmax is single GEO

Diagnosis interpretation:

- this is a policy-update / value-learning / reward-scaling path issue until disproven
- it is not enough to increase entropy if argmax remains collapsed
- trace/live consistency and replay intent/execution consistency must be confirmed first

Required MAPPO audit tooling:

- `scripts/diagnose_mappo_upper_training.py`
- `scripts/test_mappo_update_sanity.py`
- `scripts/test_checkpoint_policy_consistency.py`
- `scripts/train_upper_oracle_imitation_debug.py`

MAPPO update sanity expectations:

- higher-advantage target action must increase probability
- masked action must stay near zero even if target advantage is high
- all sanity cases must pass before preflight conclusions

Checkpoint consistency expectations:

- `max_logit_diff_after_reload < 1e-6`
- `max_prob_diff_after_reload < 1e-6`
- argmax distribution from checkpoint test must match policy inspection output

Oracle imitation debug purpose:

- verify observation schema + policy architecture can represent multi-action oracle labels
- this is diagnostic only, not a paper training method
- if imitation is healthy but RL collapses, root cause is in RL update/reward/value path

Per-action reward/advantage logging (training metrics):

- `mean_reward_local_selected`, `mean_reward_neighbor_selected`, `mean_reward_geo_selected`, `mean_reward_ground_selected`
- `mean_cost_local_selected`, `mean_cost_neighbor_selected`, `mean_cost_geo_selected`, `mean_cost_ground_selected`
- `mean_advantage_local_selected`, `mean_advantage_neighbor_selected`, `mean_advantage_geo_selected`, `mean_advantage_ground_selected`

These fields are required to classify:

- `reward_advantage_bias`
- `mappo_update_bug`
- `insufficient_training`

CUDA preflight admission (100 episodes × 3 seeds) requires:

- MAPPO update sanity pass
- checkpoint consistency pass
- oracle imitation debug pass
- fixed 50-episode CPU preflight pass
- no unexplained deterministic collapse against oracle
- live replay acceptance line pass

## Mobility Reachability Contract (v2)

Decision-time visibility and execution-time reachability are different checks and must be reported separately.

- `visible_only`: action is available when at least one feasible candidate exists at decision time.
- `mobility_safe`: action is available only if candidate is visible and `estimatedTaskTransmissionTimeSec <= estimatedLinkLifetimeSec - min_link_survival_margin_sec`.
- `completion_safe`: action is available only if candidate is visible and `estimatedTaskCompletionTimeSec <= estimatedLinkLifetimeSec - min_link_survival_margin_sec`.

State and trace must expose all three masks in parallel:

- `abstractActionMaskVisible`
- `abstractActionMaskMobilitySafe`
- `abstractActionMaskCompletionSafe`
- active `abstractActionMask` selected by `scenario.action_mask_mode`

Per-candidate mobility fields required in REST state, trace rows, and replay decision logs:

- `linkAvailableNow`
- `estimatedLinkLifetimeSec`
- `estimatedTaskTransmissionTimeSec`
- `estimatedTaskComputeTimeSec`
- `estimatedTaskCompletionTimeSec`
- `linkSurvivalMarginSec`
- `linkSurvivalMarginToCompletionSec`
- `handoverRequired`
- `handoverAvailable`
- `mobilityRisk`
- `mobilityRiskSource` (`actual | controlled_estimate | unavailable`)

Interpretation:

- `visible_only` is allowed to have non-trivial mobility failures under strict mobility stress.
- `mobility_safe`/`completion_safe` are for exposing survivability constraints to the policy without changing cost definitions.
- do not silently hide remote actions; always report `remote_available_ratio`.

### Experiment Profiles

Profile A (`paper_strict_visible` / `mobility_stress`):

- `success_profile=paper_strict`
- `action_mask_mode=visible_only`
- keep real mobility failures as robustness signal
- not the default main-training profile

Profile B (`mobility_aware_main`):

- `success_profile=paper_strict`
- `action_mask_mode=mobility_safe` or `completion_safe`
- keep dynamic links and multi-action opportunities
- require higher `task_completion_success_ratio` than Profile A without collapsing to local-only

Profile C (`preflight_lenient`):

- `success_profile=preflight_lenient`
- for engineering integration only
- must never be used as primary paper result

### Reporting Rules

Replay summary must always report:

- `success_ratio` (`task_completion_success_ratio`)
- `scheduling_success_ratio`
- `mobility_link_failure_ratio`
- `latency_deadline_failure_ratio`
- mask mode and min margin
- per-tier visible/mobility-safe/completion-safe ratios

Low success is never ignored: if `success_ratio < 0.90`, warnings must include `low_success_ratio` and dominant failure reason.

## Baseline Taxonomy

The experiment matrix uses a baseline registry with unified action interface:

- static direction baselines: `local_only`, `neighbor_only`, `geo_only`, `ground_only`, `remote_only`
- architecture baselines (action-space constraints): `only_leo`, `leo_geo`, `leo_ground`, `full`
- heuristic baselines: `random_visible`, `random_mobility_safe`, `round_robin_visible`, `cost_greedy`, `weight_greedy`, `mobility_risk_greedy`
- DRL baselines: `hmadrl_maddqn_ddpg`, `tri_mappo_maddpg`

All baselines must return:

- `upper_action`, `lower_action`, `action_name`
- `decision_info` with `baseline_name`, `selection_reason`, `cost_rank`, `mobility_risk`, and fallback metadata

Invalid masked actions are forbidden. If target tier is unavailable, fallback policy must be explicit (`cost_greedy`, `random_visible`, or `local`).

### HMADRL-Style Baseline

`hmadrl_maddqn_ddpg` is implemented as a structure-equivalent baseline family:

- upper: MADDQN/DQN-style discrete offloading (epsilon-greedy, target network, TD loss, mask-aware Q-values)
- lower: DDPG-style continuous allocation

It intentionally does **not** use MAPPO clipped PPO objective, centralized per-agent MAPPO critic, or `cost_prior_ce`. This keeps it comparable to HMADRL-style hierarchical mixed-action DRL while preserving TriSatFlow interfaces.

## Metrics Taxonomy

Unified matrix summaries report:

- performance: delay/queue/system-cost/success/failure composition
- policy: per-tier ratios and selected-when-visible ratios
- state-conditioned: oracle agreement, regret, near-optimal hit rates
- execution consistency: intent/receipt/fallback/http metrics
- complexity: inference/training/replay runtime fields

Energy remains `requires_manual_audit`, so it is recorded as optional and must not be used as a headline optimization metric.

## Experiment Matrix

`scripts/run_experiment_matrix.py` runs profile × architecture × baseline × seed and writes:

- `outputs/matrix_.../profile_<P>/arch_<A>/baseline_<B>/seed_<S>/...`
- per-run replay logs + compact summary + regret output + readiness check
- global `summary_matrix.csv` and `summary_matrix.json`

`--smoke` runs a fixed minimal subset for CI/preflight:

- profile: `mobility_aware_main`
- architecture: `full`
- baselines: `random_visible`, `cost_greedy`, `tri_mappo_maddpg`
- seed: `13`

## Cross-Paper Alignment

For fair comparison with HMADRL/SGTO/SatEdgeSim/GEO-LEO hybrid offloading papers:

- keep the same four-tier offloading semantics (`local/neighbor/geo/ground`)
- separate action acceptance from task-completion success
- report both mobility-stress (`paper_strict_visible`) and mobility-aware main profiles
- keep architecture constraints explicit in training/evaluation metadata
- avoid hiding mobility failures via silent masking or lenient-profile substitution

## V1 Fix Experimental Framework

### Why v1-fix is needed

The current codebase has fixed execution consistency and observation-normalization mismatch, but mobility-driven task failures still dominate end-to-end success. v1-fix freezes a stable experiment contract so results can be generated consistently while remaining engineering gaps are tracked explicitly.

### What can be run now

- static + heuristic + TriSatFlow checkpoint baselines in matrix form
- profile ablation between mobility-aware and mobility-stress settings
- architecture ablation (`only_leo`, `leo_geo`, `leo_ground`, `full`)
- automatic summary matrix + paper table/figure-data export

### What remains TODO

- full HMADRL (`maddqn+ddpg`) training-loop integration in the matrix training path
- stronger mobility-aware profile calibration for contact/lifetime realism
- automated weight-greedy parameter sweep orchestration

### Baseline subset for paper draft

`v1_core` includes:

- static: `local_only`, `neighbor_only`, `geo_only`, `ground_only`
- heuristic: `random_visible`, `cost_greedy`, `mobility_risk_greedy`
- learned: `tri_mappo_maddpg`

This subset is intended for initial manuscript tables and discussion.

### Extended baseline plan

`v1_extended` carries follow-up baselines and sweeps:

- `hmadrl_maddqn_ddpg`
- `weight_greedy_sweep`
- `random_mobility_safe`
- `round_robin_visible`
- `tri_mappo_maddpg` under alternative eval modes

### Paper table / figure exporters

- `scripts/v1_fix/export_paper_tables.py`
- `scripts/v1_fix/export_figure_data.py`

Both exporters tolerate missing fields and emit `NA` instead of failing.

### Parallel workflow

1. experiment platform: run v1-core matrix and collect stable summary artifacts
2. code iteration: continue mobility-model and HMADRL integration improvements
3. manuscript drafting: fill outline/tables/figures from exported CSV/Markdown artifacts

## Reward Alignment And Preflight Reward Mode

`legacy_remote_biased` remains available for ablation, but it is no longer the default preflight reward because it can create reward-advantage mismatch against the SatEdgeSim candidate-cost landscape.

Observed legacy failure mode:

- action-level reward and MAPPO advantage can become systematically positive on remote tiers and negative on `local/neighbor`
- deterministic argmax can collapse to a single remote action even when oracle-cost opportunities are diverse
- root causes are additive shaping and penalty terms that are not guaranteed to preserve oracle cost ranking under all queue regimes

### Oracle-Aligned Reward

Use `reward.mode: oracle_aligned_cost` for preflight.

Contract:

- upper reward is `-normalized_system_cost`
- system cost is decomposed and logged per action: `delay_cost`, `queue_cost`, `transmission_cost`, `compute_cost`, `feasibility_penalty`
- optional `energy_cost` is disabled by default for this mode because energy still requires manual audit
- fixed tier bias terms (`remote_bonus`, `local/neighbor/geo/ground_penalty`) are zero by default
- `action_balance_bonus` and `selected_when_visible_bonus` are zero by default
- `use_lower_effect_in_upper_reward` is `false` by default so upper reward follows oracle-tier cost rather than lower-policy allocation noise

### Decomposition Logging

Every training step now exposes reward decomposition fields in `step.info` and writes per-action selected means into `metrics.csv`:

- reward/cost: `mean_reward_*_selected`, `mean_system_cost_*_selected`
- component costs: `mean_delay_cost_*_selected`, `mean_queue_cost_*_selected`, `mean_transmission_cost_*_selected`, `mean_compute_cost_*_selected`
- shaping terms: `mean_bonus_*_selected`, `mean_penalty_*_selected`, `mean_lower_effect_*_selected`
- policy-learning signal: `mean_advantage_*_selected`

This is the required evidence path for diagnosing why one tier is over- or under-rewarded.

### Why Legacy Is Ablation-Only

Legacy reward is kept for paper ablation and backward compatibility only. It should not be used as the preflight default when alignment checks fail, because reward shaping can dominate oracle-cost ordering and bias MAPPO advantage.

### CUDA Preflight Admission Gate

Before entering CUDA `100x3` preflight, all must pass on CPU oracle-aligned runs:

- reward-oracle alignment test passes (`agreement >= 0.60`, `spearman >= 0.50`)
- no unexplained fixed per-tier reward bias (no multi-order magnitude gap)
- deterministic eval does not show immediate single-action collapse
- trace/live inspection does not show severe distribution shift
- replay intent/execution checks remain healthy

If any gate fails, do not enter CUDA; fix reward/advantage alignment first.

## Deterministic Tie-Breaking Bias And Eval Modes

`deterministic_tie_breaking_bias` means the masked policy probabilities stay relatively close across feasible actions, but deterministic `argmax` still always chooses one action because of small stable ordering gaps.

This must be audited with margin statistics rather than hidden by sampled diversity.

### Eval Mode Definitions

- `raw_argmax`: pure greedy action from masked policy logits. This is the canonical deterministic RL policy output and must always be reported.
- `stochastic_eval`: sample from masked policy distribution with fixed seed. This is supplementary for stochastic behavior analysis, not a replacement for deterministic reporting.
- `margin_cost_tiebreak`: if `top1_prob - top2_prob <= tie_eps`, choose the lowest estimated state-visible cost among near-tie candidates; otherwise keep `top1`.
- `cost_greedy_baseline`: choose minimum estimated cost directly, without RL policy logits. This is a non-RL baseline only.

### Raw Argmax Must Stay Visible

Replay and inspection outputs must always include both:

- raw argmax distribution
- final policy distribution after eval mode logic

`margin_cost_tiebreak` results must never overwrite or hide `raw_argmax` results.

### Why Margin Cost Tie-Break Is Not Oracle Label

`margin_cost_tiebreak` uses only deployment-visible per-state cost features (delay/queue/rate/compute related candidate summaries), and does not read oracle labels or future outcomes. It is therefore an interpretable decision layer, not supervised oracle imitation.

### CUDA Readiness Classes

- `strong_pass`:
  - `raw_argmax` is not unexplained single-action
  - replay checks pass under raw argmax
  - readiness script outputs `strong_pass`
- `conditional_pass`:
  - `raw_argmax` is classified as `near_tie_argmax_artifact`
  - `stochastic_eval` and `margin_cost_tiebreak` are both replay-valid
  - `margin_cost_tiebreak` is not equivalent to `cost_greedy_baseline`
  - readiness script outputs `conditional_pass`
- `fail`:
  - `true_deterministic_collapse`, unresolved tie-bias class, or policy-cost evidence inconsistency
  - CUDA preflight is blocked

### IEEE IoT-J Reporting Requirement

When writing paper results, report all of:

- `raw_argmax`
- `stochastic_eval`
- `margin_cost_tiebreak`

If only `conditional_pass` is reached, present this explicitly as limitation plus ablation, and do not claim pure deterministic RL policy diversity.

## State-Conditioned Learning And Cost-Regularized Preflight

`true_deterministic_collapse` and `weak_state_conditioned_policy_learning` are different failure modes:

- `true_deterministic_collapse`: deterministic `raw_argmax` is single-action with non-trivial top1-top2 margin, so tie-break artifacts do not explain the collapse.
- `weak_state_conditioned_policy_learning`: policy probabilities show limited state/phase sensitivity and do not track oracle-cost ranking well enough per state.

Stochastic action ratio alone is not evidence of state-conditioned decision quality. A model can sample diverse actions from a near-constant prior while deterministic argmax remains collapsed.

Required diagnostics must include:

- per-state oracle agreement
- per-phase policy distribution and per-phase oracle agreement
- probability assigned to oracle action
- oracle-action rank under policy probability

### Cost-Rank KL Regularization

Preflight may enable a cost-aware stabilization term:

- `policy_regularization.enabled: true`
- `policy_regularization.mode: cost_rank_kl`
- `policy_regularization.weight`
- `policy_regularization.temperature`

Definition:

- build cost prior from deployment-visible state cost features:
  - `p_cost(a|s) = softmax(-cost(a|s)/temperature)`
- add actor regularization:
  - `KL(policy(.|s) || p_cost(.|s))`

This is not oracle-label imitation:

- no oracle action label is injected as supervised target
- no future information is used
- only current-state observable cost proxies are used

### Cost Features In Observation

`include_cost_features_in_obs: true` enables per-tier normalized cost features:

- `local_normalized_cost`
- `neighbor_normalized_cost`
- `geo_normalized_cost`
- `ground_normalized_cost`

These are deployment-time estimable quantities derived from delay/queue/rate style state features, not leaked oracle labels.

### Why CostReg Is A Preflight Candidate

When reward alignment is already fixed but policy still collapses in deterministic eval, cost-rank regularization can increase state-conditioned policy-cost consistency without hardcoding action balance or replacing RL with oracle imitation.

### CUDA Readiness With CostReg

- `strong_pass`:
  - raw argmax no unexplained single-action collapse
  - raw replay acceptance passes
  - state-conditioned diagnostics show clear phase/cost sensitivity
- `conditional_pass`:
  - raw argmax still biased but stochastic policy and cost-prior alignment are strong and replay-safe
  - report both raw and stochastic results in CUDA preflight
- `fail`:
  - deterministic single-action collapse remains and state-conditioned evidence is weak/inconsistent
  - do not enter CUDA

## Per-Agent Credit Assignment And State-Conditioned MAPPO

Why scalar team advantage can fail:

- with `rewards_agents[T,N] -> mean over N -> rewards[T]`, all agents share one scalar advantage
- actor update loses per-agent credit separation under heterogeneous local state/cost
- policy may converge to a global action prior with small fixed bias (for example GEO/ground), instead of learning state-conditioned offloading

Per-agent MAPPO change:

- `algorithm.upper.credit_assignment` supports:
  - `global_team` (legacy compatibility)
  - `per_agent` (preflight default candidate)
- in `per_agent` mode:
  - `rewards`, `values`, `returns`, `advantages` are all `[T,N]`
  - GAE is computed independently per agent
  - actor loss uses each agent's own advantage
  - value loss is averaged over agent-time elements
  - scalar-advantage expand-to-all-agents is removed

Centralized per-agent critic:

- use `CentralPerAgentValue` in `per_agent` mode
- critic remains centralized by consuming global graph context
- output is per agent value `V_i(s)` to support agent-specific credit assignment

Upper-only pretraining and neutral lower allocator:

- `lower_action_mode: neutral_allocator` provides fixed neutral lower action (cpu/bandwidth/power all ones)
- `lower_training_enabled: false` disables lower updates for upper-only diagnosis
- optional staged training via `upper_pretrain_episodes` + `joint_train_episodes`
- metrics record `train_phase` and `lower_mode`

Cost prior regularization update:

- keep `cost_rank_kl` (`KL(policy || prior)`) for ablation
- add `cost_prior_ce` as preferred stabilizer:
  - `L_aux = -sum_a prior(a|s) log pi(a|s)`
  - less mode-seeking than forward KL in this setup
- prior is still built from deployment-visible cost features only, not oracle labels

Trace-based observation normalization:

- `obs_normalization_mode: trace_p95` with `obs_normalization_path`
- replaces hardcoded scaling sensitivity by field-wise trace statistics
- diagnostics now report `feature_saturation_ratio_by_field` and warnings when ratio exceeds threshold

State-conditioned readiness gate (CPU before CUDA):

- require improvement on:
  - `logit_std_across_states_by_action_mean`
  - `prob_std_across_states_by_action_mean`
  - `mutual_information_phase_argmax`
  - `raw_argmax_oracle_agreement`
  - `prob_oracle_action_mean`
- if raw argmax remains 100 percent single-action and phase MI is zero, readiness stays `fail` even when stochastic ratios look diverse

## Weak State-Conditioning Root Cause And Signal-Flow Diagnostics

When upper-only per-agent MAPPO still shows weak state-conditioning, the required diagnosis path is:

- observation signal (`raw_obs` and normalized obs)
- encoder output (`node embedding`)
- policy hidden features
- logits and probabilities

`scripts/diagnose_policy_signal_flow.py` reports:

- per-stage variance and pairwise distance
- phase/oracle separability across stages
- cost-feature to logit/probability correlations
- saturation ratios by field

Interpretation rules:

- obs separable but embedding weak: `encoder_washes_out_state_signal`
- embedding separable but logits weak: `policy_head_washes_out_state_signal`
- logits separable with persistent fixed action offset: `logits_have_global_bias`
- delay/rate fields heavily clipped: `normalization_saturates_features`

## Hybrid GNN-Cost Policy Head

To increase state-conditioned action sensitivity, upper actor supports:

- `algorithm.upper.policy_head: gnn_only | hybrid_gnn_cost`

`hybrid_gnn_cost` keeps the GNN embedding branch and adds action-specific cost features:

- `visible`
- `normalized_cost`
- `delay`
- `queue`
- `rate`
- `compute_proxy`

Per-action logits are produced by a shared scorer over:

- agent embedding
- action-specific cost feature vector
- global pooled context

This uses only deployment-visible features and does not use oracle labels.

## Logit Centering And Action-Bias Regularization

Upper actor now supports:

- `algorithm.upper.logit_centering: true|false`
- `algorithm.upper.action_bias_regularization: <float>`

Mechanisms:

- logit centering subtracts per-state action-mean logit and removes common offset
- bias regularization penalizes batch-mean action-logit drift to reduce global fixed-bias collapse

Tracked metrics:

- `upper_action_bias_reg_loss`
- `mean_logit_local|neighbor|geo|ground`
- `std_logit_local|neighbor|geo|ground`

## Cost-Prior CE Sweep And Best Upper-Only Config

`scripts/sweep_cost_prior_regularization.py` scans:

- `weight in [0.05, 0.1, 0.2, 0.5]`
- `temperature in [0.25, 0.5, 1.0]`

with fixed upper-only per-agent setup:

- neutral lower allocator
- `hybrid_gnn_cost`
- `trace_p95` normalization
- `cost_prior_ce`

Selection is based on state-conditioned metrics (agreement/MI/std), not visual action balance.

Workflow relation:

- first pass `upperonly_best` gate
- only when `upperonly_best` reaches at least `conditional_pass`, proceed to `joint_best`
- if `upperonly_best` fails, do not enter joint/CUDA

## CUDA Admission Rule

CPU readiness remains mandatory:

- `strong_pass`: raw deterministic policy is state-conditioned and replay-safe
- `conditional_pass`: residual raw bias exists but state-conditioned evidence and replay checks remain acceptable with explicit ablation reporting
- `fail`: keep iterating upper policy learning; no CUDA

## Why `trace_p95` Still Saturated Delay

Even after moving away from hardcoded constants, plain `trace_p95` linear scaling can still saturate high-tail delay fields in mixed trace regimes:

- `normalized_delay = min(1, delay / p95)` collapses all `>p95` into the same value
- GEO/ground/neighbor delay tails have different scales; one shared linear cap weakens cross-phase contrast
- saturated delay features reduce policy confidence calibration even when argmax is already state-conditioned

## `trace_log_quantile` Normalization

Added mode:

- `scenario.obs_normalization_mode: trace_log_quantile`
- `scenario.obs_normalization_path: traces/obs_norm_mixed_v2_log_quantile.json`

Key transform:

- `norm(x) = log1p(x / scale) / log1p(p99 / scale)`, clipped to `[0,1]`
- per-field `scale` defaults to `p50`, with per-tier statistics
- applied to delay / queue / rate / cost features

New tooling:

- `scripts/fit_obs_normalization.py --mode trace_log_quantile`
- `scripts/check_obs_feature_saturation.py`

Diagnostics now include per-field saturation post-normalization and keep warnings for over-saturated fields.

## Confidence Calibration vs State-Conditioned Learning

State-conditioned learning and confidence calibration are separated:

- state-conditioned evidence:
  - non-single raw argmax
  - phase-sensitive argmax (`MI(phase,argmax)`)
  - cost-sensitive logits/probabilities
- confidence calibration:
  - `prob_oracle_action_mean`
  - oracle action rank under policy probability

A model can have good state-conditioned argmax but still weak calibrated probability mass on oracle action.

## `calibration_required` vs `fail`

Readiness states now include:

- `strong_pass`
- `conditional_pass`
- `calibration_required`
- `fail`

`calibration_required` is used when:

- raw argmax is already non-single and state-conditioned
- agreement/MI pass state-signal thresholds
- but `prob_oracle_action_mean` remains below calibration threshold

`fail` is reserved for collapse/insufficient state-signal cases (for example raw single-action + no phase MI).

## Entropy / Temperature / CE-Weight Sweep

Calibration sweep support in `scripts/sweep_cost_prior_regularization.py` now includes:

- `weight in [0.1, 0.2, 0.3]`
- `temperature in [0.5, 0.75, 1.0]`
- `entropy_coef in [0.005, 0.01, 0.02]`
- `entropy_schedule` (`constant` or `linear_decay`)

Selection policy remains state-conditioned-first:

- prioritize `raw_argmax_oracle_agreement`, `MI(phase,argmax)`, non-single raw argmax
- then use calibration metrics (`prob_oracle_action_mean`) as secondary tie-breakers

## Joint 50-Episode Admission

Joint training remains blocked until upper-only logq reaches at least `conditional_pass`.

If result is `calibration_required`:

- keep iterating upper-only calibration
- do not start joint stage
- do not enter CUDA preflight

## Why `prob_oracle_action_mean` Is Not A Sole Gate

`prob_oracle_action_mean` is now treated as a calibration reference, not as the only blocker:

- four-action baseline probability is `0.25`, so values around this level are not automatically invalid
- cost-prior targets are soft distributions and can have multiple near-optimal actions
- deployment readiness must be decided by executed cost/regret and replay reliability, not only probability sharpness

## Regret-Based Evaluation

Use `scripts/evaluate_policy_regret.py` for per-mode regret analysis on trace states:

- supported modes:
  - `raw_argmax`
  - `stochastic_eval`
  - `margin_cost_tiebreak`
  - `cost_greedy_baseline`
- per-state cost is oracle-aligned and computed for all 4 actions from deployment-visible components
- reports:
  - `mean_normalized_regret`, `median_normalized_regret`, `p90_normalized_regret`
  - `mean_cost_ratio`
  - near-optimal hit rates (`eps=0.01/0.05/0.10`)
  - per-phase and per-action regret/cost slices

## Near-Optimal Hit Rate

Near-optimal criterion:

- `action_cost <= oracle_cost * (1 + eps)`
- `eps in {0.01, 0.05, 0.10}`

This captures practical quality under near-tie states better than strict top-1 agreement alone.

## Upper-Only To Joint Admission

Upper-only readiness is judged with:

- non-single raw argmax
- phase MI
- oracle agreement
- regret and near-optimal hit rates
- replay acceptance

Admission policy:

- `strong_pass` or `conditional_pass`: enter joint CPU preflight
- `calibration_required`: allowed to enter joint CPU preflight, but CUDA blocked
- `fail`: do not enter joint

## Joint CPU Preflight Acceptance

Joint CPU preflight requires all of:

- `check_four_tier_policy_health` pass
- state-conditioned diagnostics exported
- eval-mode diagnostics exported
- regret evaluation exported
- replay acceptance (`raw/stochastic/margin`) with:
  - `intent_execution_match_ratio >= 0.99`
  - `fallback_reason_none_ratio >= 0.99`
  - `receipt_accept_ratio >= 0.99`
  - `policy_vs_executed_ratio_diff <= 0.01`
  - `http_timeout_count == 0`
  - `http_connection_error_count == 0`

## CUDA 100x3 Preflight Admission

CUDA preflight is allowed only after joint readiness is:

- `strong_pass`, or
- `conditional_pass`

If joint readiness is `calibration_required` or `fail`, keep CPU-side iteration and do not start CUDA seeds.

Readiness thresholds (cost-reg pipeline):

- `strong_pass`:
  - `raw_argmax_oracle_agreement >= 0.60`
  - `MI(phase,argmax) >= 0.30`
  - `prob_oracle_action_mean >= 0.35`
  - raw argmax distribution is non-single
  - replay acceptance passes
- `conditional_pass`:
  - `raw_argmax_oracle_agreement >= 0.55`
  - `MI(phase,argmax) >= 0.25`
  - `prob_oracle_action_mean >= 0.30`
  - raw argmax distribution is non-single
  - replay acceptance passes
  - report stochastic / margin / cost-greedy ablations
- `calibration_required`:
  - `raw_argmax_oracle_agreement >= 0.60`
  - `MI(phase,argmax) >= 0.30`
  - raw argmax distribution is non-single
  - but `prob_oracle_action_mean < 0.30`
- `fail`:
  - single-action collapse, or zero phase MI, or insufficient state-conditioned evidence

## GPU Preflight Protocol

### 1) Why Current Status Is `conditional_pass`

Joint CPU preflight already satisfies the critical deployment-safety gates:

- strict trace integrity (`trace_hit_ratio=1.0`, `trace_fallback_count=0`)
- replay reliability (`raw/stochastic/margin` acceptance passed)
- non-single action behavior in tail action ratios
- state-conditioned evidence and moderate raw regret (`mean_normalized_regret` in acceptable preflight range)

Residual gap is calibration strength, not hard safety failure, so current status remains `conditional_pass`.

### 2) Why CUDA Preflight Is Allowed

CUDA preflight is allowed because joint CPU preflight already passed admission-level safety checks and replay reliability checks. The next step is to verify reproducibility and stability across GPU seeds, not to change algorithm/reward/scenario semantics.

### 3) Why Final Main Experiments Are Still Blocked

`conditional_pass` is sufficient for CUDA preflight and ablation expansion, but insufficient for direct paper-final claims. Main-experiment reporting still requires stronger multi-seed evidence and explicit mode-wise comparisons.

### 4) GPU Smoke (short run)

Run a short GPU smoke first:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_gpu_smoke_mixed_v2.sh
```

Default smoke command inside script:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sweep_algorithm_combinations.py \
  --config trisatflow/configs/satedgesim_trace_mixed_v2_peragent_joint_logq_best.yaml \
  --upper mappo \
  --lower maddpg \
  --episodes 10 \
  --steps 64 \
  --n-leo 16 \
  --seeds 13 \
  --output-root outputs/gpu_smoke_mixed_v2_peragent_joint_logq_best_mappo_maddpg
```

Smoke checks include:

- `metrics.csv` exists
- `checkpoint.pt` exists
- `trace_hit_ratio=1.0`
- `trace_fallback_count=0`
- `mean_feasibility>=0.95`
- no CUDA OOM / device mismatch signatures in logs
- no NaN/Inf in key reward/loss fields
- per-episode elapsed time bounded by `MAX_SEC_PER_EP` (default `120` sec/ep)

### 5) GPU 100 Episodes x 3 Seeds Preflight

Use one unified run root only:

- `outputs/gpu_preflight_mixed_v2_peragent_joint_logq_best_mappo_maddpg`

Training command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sweep_algorithm_combinations.py \
  --config trisatflow/configs/satedgesim_trace_mixed_v2_peragent_joint_logq_best.yaml \
  --upper mappo \
  --lower maddpg \
  --episodes 100 \
  --steps 128 \
  --n-leo 16 \
  --seeds 13,17,23 \
  --output-root outputs/gpu_preflight_mixed_v2_peragent_joint_logq_best_mappo_maddpg
```

or:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_gpu_preflight_mixed_v2_mappo_maddpg.sh
```

### 6) Per-Seed Acceptance Commands

For each seed:

```bash
SEED=13 RUN_ROOT=outputs/gpu_preflight_mixed_v2_peragent_joint_logq_best_mappo_maddpg \
  bash scripts/eval_gpu_preflight_seed.sh
```

The script runs:

1. policy health
2. state-conditioned diagnosis
3. eval modes
4. regret evaluation
5. replay `raw_argmax`
6. replay `stochastic_eval`
7. replay `margin_cost_tiebreak`
8. replay summaries (`summary_compact.json`)
9. readiness

Per-seed path convention:

- `outputs/gpu_preflight_mixed_v2_peragent_joint_logq_best_mappo_maddpg/seed_<SEED>/upper_mappo__lower_maddpg/`

### 7) Multi-Seed Summary

Run all seeds then summarize:

```bash
SEEDS=13,17,23 RUN_ROOT=outputs/gpu_preflight_mixed_v2_peragent_joint_logq_best_mappo_maddpg \
  SUMMARY_OUTPUT=outputs/summary_gpu_preflight_mixed_v2_mappo_maddpg.json \
  bash scripts/eval_gpu_preflight_all_seeds.sh
```

Summary script:

```bash
python scripts/summarize_gpu_preflight.py \
  --run-root outputs/gpu_preflight_mixed_v2_peragent_joint_logq_best_mappo_maddpg \
  --seeds 13,17,23 \
  --output outputs/summary_gpu_preflight_mixed_v2_mappo_maddpg.json
```

### 8) Pass / Fail Criteria

Per seed minimum gate:

- `trace_hit_ratio=1.0`
- `trace_fallback_count=0`
- `mean_feasibility>=0.95`
- replay `raw/stochastic/margin` all satisfy:
  - `intent_execution_match_ratio>=0.99`
  - `fallback_reason_none_ratio>=0.99`
  - `http_timeout_count=0`
  - `http_connection_error_count=0`
  - `policy_vs_executed_ratio_diff<=0.01`
- readiness must not be `fail`

Across three seeds:

- at least `2/3` seeds are `conditional_pass` or `strong_pass`
- no replay fail in any seed
- no trace fallback in any seed
- no obvious divergence in system cost / delay / queue
- no unexplained single-action collapse

If only `conditional_pass` is reached:

- proceed to longer GPU preflight and ablations
- do not present as final paper main result
- report `raw/stochastic/margin/cost_greedy` comparisons explicitly

### 9) Energy `requires_manual_audit`

Energy remains `requires_manual_audit` in this stage and is excluded from preflight pass/fail gating. Energy can be logged, but should not be used as primary acceptance evidence before manual audit completion.

### 10) Next-Stage Expansion (After This Preflight Only)

Only after this CUDA preflight passes the above gates:

- extend episodes/seeds for longer GPU preflight
- add ablation matrix expansion under the same scenario/reward contract
- then promote to formal main-experiment plan

Do not jump directly to large-scale final runs before preflight acceptance is complete.

## Train-Live Normalization And Replay Alignment (2026-05-27)

- `replay_on_satedgesim.py` and all live/trace policy inspection scripts must use checkpoint normalization, not legacy default.
- `FrozenTriSatFlowPolicy` now loads and stores:
  - `obs_normalization_mode`
  - `obs_normalization_path`
  - `obs_normalization_stats`
  - `obs_normalization_loaded`
- For checkpoints with `obs_normalization_mode=trace_log_quantile`, missing normalization file is a fatal error (no silent legacy fallback).
- Replay outputs (`decision_log.csv`, `summary.json`, compact summary) include:
  - `obs_normalization_mode`
  - `obs_normalization_path`
  - `obs_normalization_loaded`
  - `obs_feature_dim`

### Device CLI Override

- Training scripts support `--device cpu|cuda|cuda:0|auto`.
- `--device` overrides config device.
- `--device auto` resolves to `cuda` when available, else `cpu`.
- If CUDA is requested but unavailable, trainer falls back to CPU with warning.
- `resolved_config.yaml` and metrics rows record:
  - `requested_device`
  - `actual_device`
  - `device_fallback_reason`

### GPU Preflight Eval Scripts

- Unified scripts:
  - `scripts/run_gpu_smoke_mixed_v2.sh`
  - `scripts/run_gpu_preflight_mixed_v2_mappo_maddpg.sh`
  - `scripts/eval_gpu_preflight_seed.sh`
  - `scripts/eval_gpu_preflight_all_seeds.sh`
  - `scripts/summarize_gpu_preflight.py`
- `eval_gpu_preflight_seed.sh` uses replay default `MAX_DECISIONS=500`.
- Eval scripts support `DEVICE=cpu` for CPU-only environment replay.
- Training scripts support `DEVICE=cuda` for GPU training preflight.

### Replay Warning Semantics

- New summary warnings include:
  - `single_action_dominance`
  - `visible_but_never_selected_*`
  - `low_success_ratio`
  - `raw_argmax_ground_dominance`
  - `neighbor_unused_when_visible`
  - `geo_unused_when_visible`
  - `obs_normalization_missing`
- Warnings affect replay readiness status but do not imply execution-link failure.

### Success/Failure And Optional Module Notes

- Replay summaries keep reporting `success_ratio` and `fallback_reason_distribution` as primary execution-quality indicators.
- Checkpoint load reporting distinguishes:
  - `loaded_required_modules`
  - `skipped_optional_modules`
  - `missing_required_modules`
- `lower_critic` / `lower_target_critic` optional mismatch warnings are inference-safe when required modules (`encoder`, `upper_actor`, `lower_actor`) are loaded.
