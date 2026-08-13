# SatEdgeSim Resource Binding Status

## Current Mode

Current implemented mode: `resource_aware_estimator_bound`.

The continuous resource fields now enter an auditable estimator path:

- `cpuShare` changes effective MIPS and estimated compute delay.
- `bandwidthShare` changes effective bandwidth and estimated transmission delay.
- `txPowerRatio` changes estimated tx power and estimated tx energy.

They do not yet bind native CloudSim VM scheduler, native network bandwidth allocation, or native tx power accounting.

Required summary flags:

- `continuous_resource_binding_mode=resource_aware_estimator_bound`
- `continuous_resource_applied=true`
- `estimator_bound=true`
- `native_scheduler_bound=false`
- `full_hybrid_closed_loop_claim_allowed=false`

## Paper-Safe Claims

Allowed:

> SatEdgeSim replay is upgraded from candidate-only action mapping to resource-aware estimator-bound replay, where lower-level continuous actions affect candidate ranking/admission estimates.

Allowed:

> Scheduling receipt, completion receipt, scheduling acceptance, completion success, and energy source are reported separately.

Allowed Table 5 title:

> SatEdgeSim resource-aware estimator-bound replay

## Forbidden Claims

Do not write:

- "full hybrid closed-loop validation"
- "lower continuous allocator is validated by SatEdgeSim native scheduler"
- "cpuShare changes native VM MIPS"
- "bandwidthShare changes native network bandwidth"
- "txPowerRatio changes native power accounting"
- "receipt acceptance equals task success"
- "intent_execution_match_ratio equals completion success"
- "energy advantage" when `energy_source=unknown` or only receipt delta is available

## Next Repair Needed For Native Binding

To set `native_scheduler_bound=true`, SatEdgeSim must implement and test at least CPU and bandwidth native binding in the actual task execution and network transfer paths, and completion receipts must demonstrate the changed native metrics.

