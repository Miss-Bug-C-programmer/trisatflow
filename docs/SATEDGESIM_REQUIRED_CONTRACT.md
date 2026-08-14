# SatEdgeSim v22 Contract Audit

This document records the local read-only audit of
`D:\research\experiment\6-DRL_satellite\satedgeSimv2-github`. The adapter must
report capabilities at runtime; missing features are not simulated in Python.

## Current endpoints observed

The v22 checkout documents and exposes the existing RL lifecycle:

```text
GET  /health
GET  /version
POST /reset
GET  /get_state
POST /step
POST /apply_action
GET  /get_metrics
POST /close
GET  /topology/current
POST /topology/contact_plan
GET  /configuration/viability
```

`/get_state` is the current blocking RL decision state and includes candidate
VMs, action masks and detailed state. `/step`/`/apply_action` are execution
dispatch operations. The local SatEdgeSim documentation states that the Java
simulation thread blocks in the external RL orchestrator while Python supplies
the next action.

## Capability matrix

| Required capability | v22 status from local checkout | Adapter behaviour |
|---|---|---|
| Low-cost monitor endpoint/state | **REQUIRED / MISSING** | `/get_monitor_state` is probed. If absent, `/get_state` is marked `compatibility_preflight`, `monitor_is_true_cheap=False`, and bytes/queries/latency metadata are recorded. |
| Heavy planner-state endpoint | **REQUIRED / MISSING** as a dedicated endpoint | `/get_planner_state` is probed. Fallback uses the already acquired full state only after escalation and keeps fallback metadata. |
| Deterministic contact forecast | **AVAILABLE**: `POST /topology/contact_plan` | Used as topology/contact source; the documented forecast does not expose future stochastic workload/queue/channel state. |
| Current physical topology | **AVAILABLE**: `GET /topology/current` | Exposed by `get_current_topology`. |
| Report-only configuration viability | **AVAILABLE**: `GET /configuration/viability` | It is not treated as persistent configuration viability; the outer controller composes its own report. |
| Execution action dispatch | **AVAILABLE**: `/step`, `/apply_action` | Used by `dispatch_under_configuration`; dispatch is counted separately from replanning. |
| Persistent configuration apply | **REQUIRED / MISSING** | `/configuration/apply` is probed; absent capability raises rather than silently claiming persistence. |
| Execution under a persistent current configuration | **REQUIRED / MISSING** | Requires a native server contract that materializes each task from the current Π_k without implying a replan. |
| Explicit physical decision-delay / advance semantics | **REQUIRED / MISSING** | `/advance_world` is probed. No Python sleep is used; unsupported runs report `physical_delay_enforced=False`. |
| Current simulation time | **AVAILABLE**: documented `simulationTime` in `/get_state` | Adapter accepts the documented key and compatible aliases. |
| Cumulative metrics | **AVAILABLE**: `GET /get_metrics` | Stored as backend/data-plane metadata when used. |
| Mid-transfer contact enforcement capability flag | **REQUIRED / MISSING** | Adapter currently reports `supports_mid_transfer_contact_enforcement=False` until native semantics are exposed. |

## Required native extensions for paper-authoritative experiments

The final SatEdgeSim integration needs a true cheap monitor that does not first
enumerate all candidate VMs or detailed graph/resource state, plus a separate
heavy planner-state acquisition path. It also needs a persistent configuration
application/materialization contract, an explicit physical time advance during
planner delay, and a flag describing whether contact loss is enforced during an
in-flight transfer.

At minimum, the native contract should expose:

```text
GET  /get_monitor_state
GET  /get_planner_state            (scope/budget-aware if possible)
POST /configuration/apply
POST /configuration/dispatch       (execute under current Π_k)
POST /advance_world                (physical decision delay)
GET  /capabilities                 (or equivalent capability payload)
```

The payloads must include current simulation time, configuration id/version,
server capability metadata and sufficient receipt/metrics fields to distinguish
dispatch, replanning and configuration changes.

## Authority and fallback rules

`backend_source`, `topology_source`, `monitor_state_source` and
`physical_delay_enforced` are required run metadata. A trace, analytic model or
full-state compatibility fallback is never relabeled as authoritative SatEdgeSim
physical mode. The existing `GeoLeoGroundEnv` remains available through the
explicit non-authoritative legacy adapter for tests and legacy baselines.
