# TriSatFlow prototype design notes

## What is implemented

- GEO-LEO-Ground three-layer normalized simulator.
- Dynamic LEO topology with ring ISLs and temporary skip links.
- Pure PyTorch GraphSAGE-style topology encoder.
- Upper-layer MAPPO-style discrete global offloading.
- Lower-layer MADDPG-style continuous resource allocation.
- Cross-layer coupling:
  - upper action conditions the lower actor;
  - lower feasibility, realized delay, energy, queue and violation shape the upper reward;
  - lower centralized critic receives upper and lower actions jointly.
- Lyapunov drift-plus-penalty queue reward.
- Baseline scripts for random, local-only and greedy queue policies.

## Action definition

Upper action per LEO:

```text
0 = Local LEO
1 = Neighbor LEO
2 = GEO satellite cloud / coordinator
3 = Ground MEC / cloud gateway
```

Lower action per LEO:

```text
[cpu_fraction, bandwidth_fraction, transmit_power_fraction]
```

## Reviewer-facing rationale

The prototype avoids using MADDPG for discrete offloading. MAPPO is used for the discrete upper layer; MADDPG is used for continuous lower-layer resource allocation. This matches the mixed-action nature of the problem and avoids the common reviewer concern that discrete actions are being artificially relaxed without explanation.

## What should be improved before a full IEEE IoT-J submission

1. Replace normalized orbital visibility with ephemeris-driven traces.
2. Add more baselines: QMIX/MADDQN upper, MATD3/MASAC lower, single-level PPO/SAC, no-GNN and no-Lyapunov ablations.
3. Run larger scenarios: 20-50 LEO for main results, 100+ for scalability.
4. Add service-cache, task-type and optional DAG dependencies.
5. Report convergence, runtime decision latency, queue stability and GEO utilization.
