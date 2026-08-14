# Algorithm Semantics v2

This note gives the implementable semantics for the second-phase outer
controller. It is intentionally narrower than a claim of a complete
SatEdgeSim-native planner.

## State and intervention event

At monitor time `t_k`, the controller observes

`M_k = (q_k, w_k, d_k, l_k, c_k, u_k, h_k)`

where queue/workload/deadline/load/contact summaries, typed uncertainty and
the monitor acquisition record are available. The persistent configuration
`Γ_k` remains active until a validated `Γ_{k+1}` is applied. Monitor epochs and
intervention epochs are control counters; they do not imply a fixed physical
slot duration.

The viability estimator is a screening mechanism. With lower-bound service
rate `r^-`, effective contact/service horizon `H_eff` and safety fraction `ρ`,
its cumulative service certificate is

`S^-_k = r^-_k H_eff (1 - ρ)`.

The report escalates when service, contact, deadline, uncertainty or current
performance screening is unsafe. It does not itself claim that intervention is
optimal.

## Candidate descriptors and common-horizon VoC

For candidate `x = (Ω, f, b, p)`, the controller first forms a causal
descriptor `D_x` without enumerating full planner state. It contains scope
cardinality, estimated acquisition/compute quantities, planner family/fidelity,
capability flags and an uncertainty summary.

Let `H` be the configured evaluation horizon and `δ_x` the modeled decision
delay. Both hold and candidate outcomes use the same absolute end time
`T = t_k + H`:

`J_hold = J(Γ_k, [t_k, T])`

`J_x = J(Γ_k, [t_k, t_k + δ_x]) + J(Γ_x, [t_k + δ_x, T])`.

The causal estimate is

`B_x = J_hold - J_x`,

with uncertainty `σ_x` and optional lower-confidence score

`B_x^LCB = B_x - β σ_x`.

The executable estimator uses monitor summaries and descriptor priors only;
`Γ_x` is represented by the descriptor's expected improvement rather than by a
future rollout. Future stochastic truth is therefore not required or consumed.

The decision value is

`VoC_x = Score(B_x) - C_decision(x) - C_recfg(x)`,

where `Score` is the mean or the explicitly selected LCB ablation. KEEP is
selected when every candidate has non-positive value. KEEP is not added to the
inner MARL action space.

## Resource accounting

Raw resource fields are not added across incompatible units. The implementation
keeps bytes, seconds, energy proxies, compute proxies, changed assignment /
resource / route counts and migration volume distinct, then applies explicit
unit prices. The decision cost is

`C_decision = C_obs + C_sync + C_solve + C_signal`,

and intervention cost is

`C_intervention = C_decision + C_recfg`.

The realized reconfiguration term is computed from the structural difference
between `Γ_k` and `Γ_{k+1}` and may be refined by an authoritative apply
receipt. Requested scope volume is a separate reporting field.

## Physical delay and SMDP transition

Host solver wall-clock is recorded as measurement only. Simulated delay comes
from explicit modeled components (`solver_simulated_latency_sec`, monitor/sync/
signal latency) or the wall-clock-as-simulated ablation. When physical
enforcement is required, the backend must expose a physical advance operation;
the controller verifies before/after world time or a structured receipt.

An accepted intervention yields an outer transition

`(S_k, x_k, R_k, S_{k+1}, Δ_k)`

with `Δ_k` equal to observed physical holding time. Power-law discounting uses
`γ^{Δ_k}` under the configured time-based SMDP mode.

## What is not claimed

The hierarchical MAPPO/MADDPG adapter remains an adapter around a configured
trainer/checkpoint. It does not create a new trained agent, infer unavailable
scope-aware computation, or convert implementation labels into theoretical
benefit. SatEdgeSim runs are authoritative only when the runtime capability
matrix exposes the required cheap monitor, persistent configuration apply,
post-delay validation and verifiable physical advance contracts.

