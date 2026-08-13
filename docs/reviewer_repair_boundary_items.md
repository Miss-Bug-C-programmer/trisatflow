# Reviewer Repair Boundary Items

This document records the boundary between current CPU smoke evidence and
claims that require full experiments or stronger simulator evidence. It is a
claim-control artifact, not a performance result.

Machine-readable artifacts:

- `outputs/reviewer_repair/boundary_items/boundary_items.json`
- `outputs/reviewer_repair/boundary_items/boundary_items.csv`
- `outputs/reviewer_repair/boundary_items/summary.json`

## Summary

- Boundary items: 12
- Critical boundaries: 6
- Formal experiment results available: false
- Unrestricted paper claims allowed: false
- Final CPU smoke gates currently pass, but smoke evidence remains bounded to
  imports, metadata guards, tiny updates, and replay semantics.

## Critical Boundaries

| ID | Area | Current boundary | Forbidden until stronger evidence |
|---|---|---|---|
| B00 | CPU smoke scope | CPU smoke validates code paths and guards only. | Treating smoke results as formal performance results. |
| B02 | Lyapunov semantics | Lyapunov component is reward shaping / queue regularization. | Claiming queue stability guarantees without theorem metadata. |
| B06 | SatEdgeSim resource binding | Current validation is candidate-level or resource-aware estimator-bound according to metadata. | Claiming native VM/network/power execution binding when `native_scheduler_bound=false`. |
| B07 | Receipt/completion/energy | Receipt, scheduling, completion, and energy source must remain separate. | Treating receipt acceptance or intent match as task success. |
| B08 | Statistics | Algorithm-pair wording must follow checkpoint-level Holm-corrected tests. | Claiming significant best without Holm-supported full-seed evidence. |
| B09 | Strong baselines | P-DQN/flat hybrid have tiny update smoke only. | Claiming state-of-the-art hybrid RL superiority before full multi-seed baselines. |

## High-Impact Boundaries

| ID | Area | Current boundary | Unlock condition |
|---|---|---|---|
| B01 | Physical metrics | Physical and normalized metrics are schema-separated. | Run physical-vs-normalized ranking audit on final scenarios. |
| B03 | Lower allocator fairness | Rule baselines must state lower allocator. | Run Table 4b rule-upper x lower-allocator matrix. |
| B04 | Safe observable ablation | Main ablation must stay under `safe_observable` with no diagnostic cost prior. | Rerun Figure 10 full matrix with training seeds. |
| B05 | Mask deployability | `oracle_trace` mask is diagnostic upper bound, not deployable. | Run predicted/measured mask noise and staleness stress on full checkpoints. |
| B10 | Trace leakage and transfer | Current active trace bank passes manifest/leakage audit; transfer remains pending. | Run 16/32/64 checkpoint transfer/stress experiments. |
| B11 | Encoder semantics | Encoder wording must match diagnostics for the reported run. | Run diagnostics on final training checkpoints and cite actual `encoder_mode`. |

## Required Paper Discipline

- Do not use CPU smoke outputs as final experiment results.
- Do not collapse normalized cost, physical delay, and physical energy into one
  unit system.
- Do not describe Lyapunov reward shaping as a stability theorem.
- Do not use diagnostic cost-prior ablation as the deployable main ablation.
- Do not use oracle masks as deployment evidence.
- Do not claim native SatEdgeSim continuous-resource validation unless native
  scheduler binding is verified with completion evidence.
- Do not claim transfer across constellation sizes until full 16/32/64
  experiments pass.

## Evidence Standard

Every stronger claim must satisfy its corresponding unlock condition in
`outputs/reviewer_repair/boundary_items/boundary_items.json`. If an item remains
in a smoke-only or full-pending state, the manuscript must use the allowed
claim wording and avoid the forbidden claim wording.
