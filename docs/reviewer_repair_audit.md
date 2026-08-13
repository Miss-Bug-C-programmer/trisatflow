# Reviewer Repair Audit

Date: 2026-06-29

Scope:
- TriSatFlow: `D:\research\experiment\6-DRL_satellite\trisatflow`
- SatEdgeSim: `D:\research\experiment\6-DRL_satellite\satedgeSimv2`
- CPU-only validation environment: `D:\conda_envs\receiversync-viz`

No large training or SatEdgeSim live replay was run. The new smoke script uses `n_leo=4`, `episode_len=4`, no training, and no live server.

## Code Locations Audited

TriSatFlow:
- `trisatflow/envs/geo_leo_ground_env.py`
- `trisatflow/envs/action_masks.py`
- `trisatflow/envs/obs_builder.py`
- `trisatflow/envs/topology_trace.py`
- `trisatflow/envs/physical_metrics.py`
- `trisatflow/agents/hierarchical_trainer.py`
- `trisatflow/agents/maddpg_lower.py`
- `trisatflow/agents/lower_variants.py`
- `trisatflow/baselines/registry.py`
- `trisatflow/baselines/static_policies.py`
- `trisatflow/baselines/heuristic_policies.py`
- `scripts/replay_on_satedgesim.py`
- `scripts/replay_baseline_on_satedgesim.py`
- `scripts/summarize_satedgesim_replay.py`

SatEdgeSim actual paths found:
- `SatEdgeSim/edu/weijunyong/satedgesim/server/RlAction.java`
- `SatEdgeSim/edu/weijunyong/satedgesim/server/ExecutionReceipt.java`
- `SatEdgeSim/edu/weijunyong/satedgesim/server/RlDecisionBridge.java`
- `SatEdgeSim/edu/weijunyong/satedgesim/TasksOrchestration/ExternalRLOrchestrator.java`

The requested `SatEdgeSim/simulation/rl/...` and `SatEdgeSim/orchestrator/...` paths do not exist in this checkout.

## Unit Semantics

Current TriSatFlow environment semantics:
- `queue`: internal task/work queue length. `ScenarioConfig` comments call queue/workload "normalized work units"; `physical_metrics.py` exports the same tensor as `physical_queue_length_tasks` and only derives `physical_queue_cycles` by multiplying `queue_cycles_per_unit`.
- `service`: same internal queue/work units per step. In `GeoLeoGroundEnv.step`, service is `min(prev_queue, target_cpu * feasible)`.
- `delay`: internal delay units. `physical_delay_s = delay_units * delay_s_per_unit`; default `delay_s_per_unit=1.0`, so default exported delay is numerically seconds but still model-estimated.
- `energy`: internal energy units from `0.02 * cpu_alloc^2 + tx_power * tx_delay`. `physical_energy_j = energy_units * energy_j_per_unit`; default `energy_j_per_unit=1.0`.

Current SatEdgeSim replay semantics:
- Receipt `delay` and estimated delay fields are seconds-style fields (`estimatedTotalDelaySec`, `deadline`).
- Receipt energy fields are raw cumulative energy counters and deltas from SatEdgeSim's energy model. Replay summaries currently label the raw unit as `Wh` and set `energy_audit_status = requires_manual_audit`.
- `physical_metrics.energy_delta_from_cumulative_wh` defines the explicit conversion rule `Wh * 3600 -> J`, but replay summaries do not yet fully normalize all SatEdgeSim energy outputs into joules.

## Lyapunov Reward Status

Lyapunov is not a native scheduler constraint or a proof of stability in the current code. It directly changes the training reward when enabled:
- `lyapunov_cost = drift + lyapunov_v * immediate_cost`
- `realized_upper_cost = lyapunov_cost if enable_lyapunov_reward else immediate_cost`
- `upper_reward = -realized_upper_cost.detach()` in the non-oracle path when cross-layer feedback is enabled.

So it is more than a cosmetic logging field, but paper wording should treat it as reward shaping/objective design inside the TriSatFlow simulator, not as a demonstrated queue-stability theorem or an enforced SatEdgeSim scheduler mechanism.

## Baseline Lower Action

All static/rule baseline decisions go through `finalize_baseline_decision`, which returns:

```json
"lower_action": [1.0, 1.0, 1.0]
```

This means rule baselines use full CPU share, full bandwidth share, and full transmit-power ratio in the shared action schema. They do not run a learned lower allocator.

Checkpoint replay can emit learned lower actions through `FrozenTriSatFlowPolicy.act`, and `replay_on_satedgesim.py` writes those values to `cpuShare`, `bandwidthShare`, and `txPowerRatio`.

## Action Mask and Trace Oracle Fields

Action masks do use trace/exported oracle-style fields when present:
- `TopologyTraceProvider` parses `abstract_action_mask_visible`, `abstract_action_mask_completion_safe`, `abstract_action_mask_mobility_safe`, and `abstract_action_mask_final`.
- `ActionMaskDiagnostics` uses `mask_field_presence` to decide whether a trace field is present.
- `build_upper_action_mask` prioritizes trace visible/completion/mobility masks over analytic visibility whenever the trace row is provided and the field is present.

Implication: action-mask experiments using SatEdgeSim traces should be described as using exported availability/safety fields. Claims about purely observable policy information must distinguish the observation tensor from the mask oracle/path constraints.

## Encoder Detach / Coupling

Current trainer behavior:
- In the rollout loop, the lower actor receives `embed.detach()` from the upper/shared encoder.
- During lower-agent update, `shared_frozen` detaches embeddings in `_train_embed`.
- `shared_joint` and `separate` create an encoder optimizer and do not detach in `_train_embed`; `separate` also owns a target encoder.
- `separate` action selection recomputes lower embeddings from obs/graph when those inputs are available.

So the action-time path is detached from the upper rollout embedding, while the update-time path depends on `algo.encoder_mode`.

## SatEdgeSim Continuous Action Binding

`RlAction.java` accepts `cpuShare`, `bandwidthShare`, and `txPowerRatio`, but the Java comment states these fields are accepted for a unified action schema and can be wired later.

Actual scheduling path:
- `ExternalRLOrchestrator.findVM` asks `RlDecisionBridge` for a VM index.
- `RlDecisionBridge` validates and returns `targetVmIndex`.
- No audited Java scheduler/network/energy path consumes `cpuShare`, `bandwidthShare`, or `txPowerRatio`.

Therefore SatEdgeSim live replay currently validates upper offloading intent and VM selection, not native continuous resource-control effects.

## Replay Metrics Semantics

- `receipt_accept_ratio`: fraction of decisions where SatEdgeSim accepted the action and scheduled execution. This is an action-validity/scheduling-handshake metric.
- `intent_execution_match_ratio`: fraction where the executed abstract action equals the policy intended abstract action. This detects bridge mapping/fallback mismatch.
- `task_completion_success_ratio` / `success_ratio`: task outcome metric from final SatEdgeSim metrics or receipt task-success fields. This is not equivalent to action acceptance; an accepted action may later fail latency, mobility, or resource constraints.

## Paper Claim Readiness

Currently supportable claims:
- TriSatFlow has a four-tier abstract action model: local, neighbor, GEO, ground.
- Dynamic masks can combine architecture filters, visibility, completion-safety, and mobility-safety fields.
- Rule/static baselines are implemented and registered with paper-ready metadata for the formal baseline names.
- CPU-only small smoke execution can import the audited modules, create a small environment, step it, call a rule baseline, and write a JSON summary.
- SatEdgeSim replay bridge can accept abstract upper actions, map them to VM candidates, return execution receipts, and summarize acceptance/mismatch/task outcome metrics.

Claims requiring qualification:
- Lyapunov behavior: supported as a reward/objective term, not as a formal stability guarantee.
- Physical units: TriSatFlow exports seconds/joules/tasks through scale factors, but the core simulator still uses normalized internal units unless scale factors and trace semantics are tightly controlled.
- SatEdgeSim energy: current replay outputs still need a completed Wh-to-J audit before joule-level cross-method claims.
- Observation purity: safe observations avoid oracle cost features, but action masks may still use exported trace safety/availability fields.

Currently unsupported claims:
- Learned lower `cpuShare`, `bandwidthShare`, or `txPowerRatio` improve native SatEdgeSim scheduling, latency, or energy. Those fields are not wired into the native scheduler.
- End-to-end SatEdgeSim validation of continuous MADDPG resource allocation.
- Large-scale statistical performance superiority. This audit ran no large episodes, no multi-seed experiments, and no live SatEdgeSim replay.
- Formal queue stability or theoretical Lyapunov optimality.

## Validation Run

Smoke command run:

```powershell
cd D:\research\experiment\6-DRL_satellite\trisatflow
conda run -p D:\conda_envs\receiversync-viz python scripts\run_cpu_smoke_audit.py
D:\conda_envs\receiversync-viz\python.exe scripts\run_cpu_smoke_audit.py
```

Result: passed. Output:

```text
outputs/reviewer_repair/audit_smoke/summary.json
```

Full pytest attempt:

```powershell
conda run -p D:\conda_envs\receiversync-viz python -m pytest -q
```

Result: blocked before running tests because legacy `test/conftest.py` imports BenchMARL dependencies and the environment lacks `tensordict`.

Related test subset:

```powershell
D:\conda_envs\receiversync-viz\python.exe -m pytest -q --basetemp D:\research\experiment\6-DRL_satellite\trisatflow\.pytest_tmp_audit tests\test_env_and_training.py tests\test_action_masks.py tests\test_baseline_registry_paper_ready.py tests\test_observation_oracle_policy.py tests\test_lower_encoder_modes.py tests\test_lower_action_binding_gate.py tests\test_units_and_metrics_schema.py
```

Result: `34 passed in 4.95s`.

Notes:
- A direct subset run without `--basetemp` hit Windows temp permission issues under `C:\Users\...\AppData\Local\Temp\pytest-of-...`.
- One import-time failure was fixed in `scripts/aggregate_results.py` by lazy-loading `scipy` and `statistical_tests` only after deprecated metric validation. This is an import/dependency fix, not an algorithm behavior change.

Skipped tests:
- Full legacy `test/` BenchMARL suite was not run because `tensordict` is missing from the specified conda environment.
- No live SatEdgeSim REST replay was run because the task requested CPU smoke only and no server startup/large replay.
- No large experiments were run.

## Files Changed

- Added `scripts/run_cpu_smoke_audit.py`
- Added `docs/reviewer_repair_audit.md`
- Updated `scripts/aggregate_results.py` with a lazy import dependency fix

## Next Repair Phases Readiness

Phase 1 can start, with boundaries:
- Ready for import/path cleanup, small unit tests, mask/metric semantics hardening, and SatEdgeSim receipt/summary schema cleanup.
- Not ready for claims about continuous lower-action effects in native SatEdgeSim until `cpuShare`, `bandwidthShare`, and `txPowerRatio` are wired into CPU/network/energy scheduling and covered by targeted tests.
- Not ready for final paper performance claims until the energy-unit audit, observation/mask oracle policy, and large multi-seed protocol are completed separately.
