# Reviewer Repair Final Status

This document is a quality-gate map for reviewer-repair work. CPU smoke outputs
verify code paths and metadata guards only; they are not formal paper results.

## Integration Evidence

- Final smoke runner: `scripts/run_reviewer_repair_cpu_smoke_all.py`
- Quality gates: `scripts/check_reviewer_repair_quality_gates.py`
- Final outputs: `outputs/reviewer_repair/final_cpu_smoke/`
- Full experiment plan: `docs/gpu_full_experiment_commands.md`
- Paper claim guard: `docs/paper_claim_guard.md`

## Latest CPU Integration Result

- Stage status: 12 passed, 0 blocked, 0 failed.
- P9R fixed: trace manifest build now reports `manifest_build_status=ok` with 20 active records and 0 incomplete records; split audit reports `audit_status=passed` and `leakage_risk=none`.
- P9R caveat: this resolves manifest/leakage audit blocking for the current trace bank, but constellation-size transfer claims still require the planned 16/32/64 full transfer experiments.
- P11R fixed: SatEdgeSim now compiles through `mvn -q -DskipTests compile`, and `RlResourceBindingSmoke` passes in the final integration run.
- Quality gates: 10 passed, 0 blocked, 0 failed.
- Overall smoke status: all CPU quality gates pass. This still does not convert CPU smoke into formal experiment evidence, and it does not allow native closed-loop claims unless `native_scheduler_bound=true` with completion evidence.

## Reviewer Issue Matrix

| Issue | code_changes | tests | smoke_outputs | full_experiment_needed | current_claim_allowed | current_claim_forbidden | paper_text_revision_needed |
|---|---|---|---|---|---|---|---|
| R1 novelty/system integration claim | Final smoke and claim gates integrate all repair phases; SatEdgeSim resource binding status is separated from TriSatFlow training. | Stage smoke plus quality gates. | `final_cpu_smoke/summary.json`, `quality_gates.json`. | Full end-to-end experiment matrix E1-E13. | The system now has auditable components for hybrid action, masks, physical metrics, and replay semantics. | Broad full-stack closed-loop validation claim before native binding and full seeds. | Tone down novelty to auditable framework plus validated smoke paths. |
| R2 physical units | `envs/physical_model.py`, workload conversion, metric schema wrappers, physical mode config. | `tests/test_dimensional_model.py`, `tests/test_physical_metric_schema.py`. | `physical_model/summary.json`. | E1 physical vs normalized ranking audit. | Physical mode reports seconds, Joules, cycles with source/scope. | Directly comparing normalized and physical cost across profiles. | Define Eq. (5)-(7) units and normalized objective separately. |
| R3 Lyapunov stability | Queue cap modes, Lyapunov diagnostics, reward metadata. | `tests/test_lyapunov_diagnostics.py`. | `lyapunov_dpp/summary.json`. | Formal theorem only if theoretical DPP is added. | Lyapunov-inspired reward shaping / queue regularizer. | Guarantees queue stability or finite buffer as proof. | Replace stability guarantee wording with queue-regularized objective. |
| R4 strong baselines | Optimized DPP, P-DQN, flat hybrid AC, attention candidate policy, grid oracle. | `tests/test_optimized_dpp_baseline.py`, `tests/test_strong_baseline_training.py`, `tests/test_small_scale_oracle.py`. | `lyapunov_dpp/summary.json`, `strong_baselines/*`. | E3 strong learning baselines, E4 DPP, E5 oracle gap. | P-DQN/flat hybrid have tiny real update smoke; oracle gap harness exists. | State-of-the-art hybrid RL superiority from tiny smoke. | Present strong baselines as planned/full-experiment requirement until completed. |
| R5 baseline lower fairness | `baselines/lower_allocators.py`, `fair_wrappers.py`, fairness evaluator. | `tests/test_lower_allocator_fairness.py`. | `lower_fairness/neutral`, `lower_fairness/optimized_greedy`. | E2 Table 4b rule upper x lower allocator. | Rule baselines can be evaluated under explicit lower allocator controls. | Attributing Table 4 gaps only to upper policy. | Add lower allocator column and fairness protocol. |
| R6 statistics | `analysis/statistical_schema.py`, `analysis/statistical_tests.py`, statistical runners and guards. | `tests/test_statistics_protocol.py`. | `statistics/summary.json`, `pairwise_tests.csv`, `claim_guard.json`. | E6 8-10 training seed robustness. | Mean-ranked reference and Holm-corrected comparable-pair language. | Significant best if Holm non-significant. | Use checkpoint/train_seed as statistical unit and include effect sizes/CI. |
| R7 SatEdgeSim continuous resources | Java resource profile, estimator, binding mode, smoke; Python summary binding metadata. | `RlResourceBindingSmoke`, `tests/test_satedgesim_metric_semantics.py`. | `satedgesim_semantics/summary.json`. | E10 candidate vs estimator vs native-bound replay. | Estimator-bound resource-aware replay when binding mode says so. | Native continuous allocator validation unless native scheduler bound is true. | Rename Table 5 according to binding mode. |
| R8 SatEdgeSim success/energy | Summary semantics split receipt, scheduling, completion, energy source. | `tests/test_satedgesim_metric_semantics.py`. | `satedgesim_semantics/summary.json`. | E11 completion/energy audit with real logs. | Scheduling acceptance and completion success are separate. | `success_rate` without completion receipt. | Define receipt acceptance vs task completion. |
| R9 receipt/completion/energy source | `RlCompletionReceipt`, scheduling vs completion stages, energy source fields. | Java smoke and Python semantic tests. | `satedgesim_semantics/summary.json`. | E11 final cumulative energy validation. | Energy can be reported only with source/unit. | Energy advantage when source is unknown. | Add energy source footnote and suppress unsupported energy claims. |
| R10 ablation cost-prior confound | Safe observable configs, obs guard, safe ablation runner, Pareto plotter. | `tests/test_safe_observable_ablation.py`. | `safe_ablation/summary.json`, `safe_ablation/figures`. | E7 safe observable Figure 10 rerun. | Main ablation is deployable safe_observable. | Using diagnostic cost-prior result as main ablation. | Figure 10 must be rerun under safe_observable. |
| R11 Table/citation issue | Final docs and claim guard identify table title/wording requirements. | Quality gates. | `final_cpu_smoke/claim_guard.json`. | Paper editorial pass. | Tables can state exact validation mode. | Ambiguous table titles implying stronger evidence. | Update captions, citation/context language. |
| R12 normalized vs physical metrics | Metric schema includes unit/source/normalizer/comparable_scope. | `tests/test_physical_metric_schema.py`. | `physical_model/summary.json`. | E1 ranking audit under physical mode. | Normalized cost is dimensionless and same-profile only. | Physical Joules/seconds and normalized cost as same unit. | Add metric schema to Methods/Appendix. |
| R13 hierarchical training/encoder detach | Encoder mode config, gradient diagnostics, cadence diagnostics. | `tests/test_encoder_gradient_diagnostics.py`. | `encoder_diagnostics/summary.json`, `gradient_report.csv`. | E12 diagnostics on formal checkpoints. | Encoder gradient path can be stated per `encoder_mode`. | Jointly trained shared encoder unless lower gradient evidence is positive. | Align text with `shared_upper_only` or `shared_joint` evidence. |
| R14 mask deployability | Mask source, predictors, noise/staleness injection, metadata. | `tests/test_mask_source_and_noise.py`. | `mask_noise/summary.json`. | E8 mask prediction/noise stress. | Predicted/measured masks are deployable variants. | Oracle trace mask as deployable main result. | Separate oracle upper-bound mask from predicted deployable mask. |
| R15 GNN transfer/stress | Trace manifest/audit tools, stress configs, transfer limitation docs. | `tests/test_trace_split_audit.py`, `tests/test_stress_config_smoke.py`. | `trace_stress/*`. | E9 16/32/64 transfer and leakage-clean manifest. | Current trace bank manifest/leakage audit passes; stress harness exists. | Transfer/generalization claim before full 16/32/64 transfer evaluation. | Add limitation and require completed transfer matrix. |

## Author Question Matrix

| Question | code_changes | tests | smoke_outputs | full_experiment_needed | current_claim_allowed | current_claim_forbidden | paper_text_revision_needed |
|---|---|---|---|---|---|---|---|
| Q1 strong RL baselines | P-DQN, flat hybrid AC, optimized DPP, grid oracle. | Strong baseline and oracle tests. | `strong_baselines/*`. | E3-E5. | Tiny training update implemented. | SOTA comparison claim. | Add full-baseline experiment plan/results before strong claim. |
| Q2 receipt vs scheduling | Python summaries and Java receipt stages. | SatEdgeSim semantic tests. | `satedgesim_semantics/summary.json`. | E11. | Receipt acceptance is API/scheduling evidence. | Receipt acceptance equals task success. | Rename metrics. |
| Q3 energy normalization | Energy source/unit fields. | SatEdgeSim semantic tests. | `satedgesim_semantics/summary.json`. | E11. | Energy source must be declared. | Energy advantage from unknown source. | Add energy-source table note. |
| Q4 mask oracle | Mask source and noise metadata. | Mask tests. | `mask_noise/summary.json`. | E8. | Predicted mask stress can be reported after full run. | Oracle mask as deployable. | Split oracle/predicted mask sections. |
| Q5 physical dimensions | Dimensioned physical model. | Physical tests. | `physical_model/summary.json`. | E1. | Cycles/bits/Hz/bps/J/s are explicit. | Mixed unit equations. | Rewrite Eq. (5)-(7) definitions. |
| Q6 Lyapunov proof | Semantics downgraded. | Lyapunov diagnostics tests. | `lyapunov_dpp/summary.json`. | Formal theorem optional. | Reward shaping/regularizer. | Stability theorem. | Remove theorem-like wording. |
| Q7 statistics | Checkpoint-level tests and claim guard. | Statistics tests. | `statistics/*`. | E6. | Statistically comparable when Holm non-significant. | Significant best without support. | Use safe Table 3 wording. |
| Q8 continuous resources in baselines | Lower allocator wrappers. | Lower fairness tests. | `lower_fairness/*`. | E2. | Lower allocator stated explicitly. | Default lower allocator hidden. | Add Table 4b design. |
| Q9 diagnostic cost-prior ablation | Safe observable ablation suite. | Safe ablation tests. | `safe_ablation/*`. | E7. | Main ablation deployable. | Diagnostic cost prior as main. | Reframe no_mask as cost-safety operating point. |
| Q10 training non-stationarity | Cadence/gradient diagnostics. | Encoder diagnostics tests. | `encoder_diagnostics/*`. | E12. | Update cadence and gradient paths auditable. | Unqualified shared encoder learning claim. | Add training semantics section. |
| Q11 trace split/leakage | Trace manifest and split audit. | Trace audit tests. | `trace_stress/*`. | E13. | Current active trace bank has complete manifest metadata and passed split audit. | Extending leakage-free claim to un-audited future traces. | Add data governance appendix. |
| Q12 SatEdgeSim closed-loop | Java binding mode and estimator smoke. | Java smoke and semantic tests. | `satedgesim_semantics/*`. | E10-E11. | Estimator-bound/candidate-level replay, depending on mode. | Full native hybrid execution unless native bound is verified. | Change Table 5 title and discussion. |

## Current Readiness

- CPU smoke integration is ready to run with `scripts/run_reviewer_repair_cpu_smoke_all.py`.
- Quality gates are ready to run with `scripts/check_reviewer_repair_quality_gates.py`.
- Phase readiness is claim-specific: physical metrics, Lyapunov semantics, lower fairness controls, safe ablation guards, mask semantics, statistics guards, encoder diagnostics, strong-baseline tiny updates, and SatEdgeSim estimator-bound semantics have smoke evidence.
- Trace split/leakage audit is passed for the current active trace bank; transfer claims remain pending full 16/32/64 experiments.
- Native SatEdgeSim continuous scheduler binding remains not claimed unless a future run sets `native_scheduler_bound=true` with completion evidence.
