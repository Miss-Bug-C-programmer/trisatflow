# GPU/HPC Full Experiment Commands

These commands are templates for formal experiments. CPU smoke outputs under
`outputs/reviewer_repair` are engineering evidence only and must not be cited as
final performance results.

Use the project root:

```powershell
cd D:\research\experiment\6-DRL_satellite\trisatflow
conda activate D:\conda_envs\receiversync-viz
$env:PYTHONPATH = "$PWD;$PWD\trisatflow"
```

## E1 Physical vs Normalized Ranking Audit

- requires_gpu: false
- expected_runtime_level: medium
- seeds: scenario seeds plus all final train checkpoints
- output_dir: `outputs/full_repair/E1_physical_ranking`
- expected_table_or_figure: metric schema table and physical-vs-normalized ranking audit
- reviewer_issue_addressed: R2, R12, Q5
- paper_claim_enabled_if_passed: dimensioned delay/energy/queue metrics are internally consistent

```powershell
python scripts/run_cpu_smoke_physical_model.py
python scripts/summarize_experiment_matrix.py --input outputs/paper_ready_v3 --output-dir outputs/full_repair/E1_physical_ranking --metric-mode dual
```

## E2 Lower Allocator Fairness

- requires_gpu: false for rule baselines, true if same learned lower checkpoints are evaluated
- expected_runtime_level: medium
- seeds: all final scenario seeds
- output_dir: `outputs/full_repair/E2_lower_fairness`
- expected_table_or_figure: Table 4b rule upper x neutral/same_lower/optimized/oracle lower
- reviewer_issue_addressed: R5, Q8
- paper_claim_enabled_if_passed: upper-policy gains separated from lower allocator effects

```powershell
python scripts/evaluate_baseline_lower_fairness.py --baselines geo_only,ground_only,random_visible --lower-allocator neutral --episodes 20 --steps 128 --device cpu --output-dir outputs/full_repair/E2_lower_fairness/neutral
python scripts/evaluate_baseline_lower_fairness.py --baselines geo_only,ground_only,random_visible --lower-allocator optimized_greedy --episodes 20 --steps 128 --device cpu --output-dir outputs/full_repair/E2_lower_fairness/optimized_greedy
python scripts/evaluate_baseline_lower_fairness.py --baselines geo_only,ground_only,random_visible --lower-allocator same_learned --checkpoint outputs/paper_ready_v3/checkpoints/lower.pt --episodes 20 --steps 128 --device cuda --output-dir outputs/full_repair/E2_lower_fairness/same_learned
```

## E3 Strong Learning Baselines

- requires_gpu: true
- expected_runtime_level: high
- seeds: 8-10 training seeds, matched test seeds
- output_dir: `outputs/full_repair/E3_strong_baselines`
- expected_table_or_figure: strong baseline comparison table
- reviewer_issue_addressed: R4, Q1
- paper_claim_enabled_if_passed: comparison against implemented hybrid-action RL baselines

```powershell
python scripts/train_strong_baseline_tiny.py --baseline pdqn_hybrid --episodes 800 --steps 128 --n-leo 16 --device cuda --output-dir outputs/full_repair/E3_strong_baselines/pdqn_seed${SEED}
python scripts/train_strong_baseline_tiny.py --baseline flat_hybrid_ac --episodes 800 --steps 128 --n-leo 16 --device cuda --output-dir outputs/full_repair/E3_strong_baselines/flat_seed${SEED}
python scripts/run_strong_baselines.py --baselines pdqn_hybrid,flat_hybrid_ac,small_scale_grid_oracle --episodes 50 --steps 128 --n-leo 16 --device cuda --output-dir outputs/full_repair/E3_strong_baselines/eval
```

## E4 Optimized DPP Comparison

- requires_gpu: false
- expected_runtime_level: medium
- seeds: all final scenario seeds
- output_dir: `outputs/full_repair/E4_optimized_dpp`
- expected_table_or_figure: DPP comparison table
- reviewer_issue_addressed: R3, R4, Q6
- paper_claim_enabled_if_passed: optimized greedy DPP is a stronger heuristic comparator

```powershell
python scripts/run_cpu_smoke_lyapunov_dpp.py
python scripts/evaluate_baseline_lower_fairness.py --baselines optimized_lyapunov_dpp --lower-allocator optimized_greedy --episodes 50 --steps 128 --device cpu --output-dir outputs/full_repair/E4_optimized_dpp
```

## E5 Small-Scale Oracle Gap

- requires_gpu: false
- expected_runtime_level: medium-high due enumeration
- seeds: all small-scale scenario seeds
- output_dir: `outputs/full_repair/E5_oracle_gap`
- expected_table_or_figure: oracle gap table
- reviewer_issue_addressed: R4, Q1
- paper_claim_enabled_if_passed: finite small-scale gap to grid oracle, not MINLP optimality

```powershell
python scripts/evaluate_oracle_gap.py --episodes 20 --steps 8 --n-leo 4 --device cpu --output-dir outputs/full_repair/E5_oracle_gap
```

## E6 8-10 Training Seeds Statistical Robustness

- requires_gpu: true for producing checkpoints, false for statistics
- expected_runtime_level: high
- seeds: 8-10 independent train_seed/checkpoint_id values
- output_dir: `outputs/full_repair/E6_statistics`
- expected_table_or_figure: Table 3 with Holm p-values, effect sizes, cluster bootstrap CI
- reviewer_issue_addressed: R6, Q7
- paper_claim_enabled_if_passed: statistically supported differences only where Holm-corrected tests pass

```powershell
python scripts/run_statistical_tests.py --input outputs/paper_ready_v3/experiment_matrix.csv --split offline --output-dir outputs/full_repair/E6_statistics
```

## E7 Safe Observable Ablation

- requires_gpu: true
- expected_runtime_level: high
- seeds: 8-10 train seeds per variant
- output_dir: `outputs/full_repair/E7_safe_ablation`
- expected_table_or_figure: rerun Figure 10 and cost-safety Pareto scatter
- reviewer_issue_addressed: R10, Q9
- paper_claim_enabled_if_passed: deployable GNN/mask/Lyapunov/cross-layer ablation conclusions

```powershell
python scripts/run_safe_ablation_suite.py --episodes 800 --steps 128 --device cuda --variants safe_observable_full,safe_no_mask,safe_no_gnn,safe_static_gnn,safe_no_lyapunov,safe_no_cross_layer --output-dir outputs/full_repair/E7_safe_ablation
python scripts/plot_safe_ablation_pareto.py --input outputs/full_repair/E7_safe_ablation/summary.json --output-dir outputs/full_repair/E7_safe_ablation/figures
```

## E8 Mask Predicted/Noise Stress

- requires_gpu: true for trained policies, false for rule-policy diagnostics
- expected_runtime_level: medium-high
- seeds: all final checkpoints and scenario seeds
- output_dir: `outputs/full_repair/E8_mask_noise`
- expected_table_or_figure: mask prediction/noise stress figure
- reviewer_issue_addressed: R14, Q4
- paper_claim_enabled_if_passed: deployable predicted-mask robustness under stated noise/staleness

```powershell
python scripts/run_mask_noise_stress.py --mask-source measured,predicted,oracle_trace --noise-levels 0,0.25,0.5,1.0 --episodes 50 --steps 128 --device cuda --output-dir outputs/full_repair/E8_mask_noise
```

## E9 Transfer/Stress 16/32/64

- requires_gpu: true
- expected_runtime_level: high
- seeds: all transfer train/test seeds
- output_dir: `outputs/full_repair/E9_transfer_stress`
- expected_table_or_figure: transfer/stress table for 16/32/64
- reviewer_issue_addressed: R15, Q11
- paper_claim_enabled_if_passed: constellation-size transfer only if manifest audit also passes

```powershell
python scripts/build_trace_manifest.py --project-root . --output-dir outputs/full_repair/E9_transfer_stress/manifests
python scripts/audit_trace_splits.py --manifest-dir outputs/full_repair/E9_transfer_stress/manifests --output-dir outputs/full_repair/E9_transfer_stress/audit
python scripts/run_stress_suite.py --stress-configs scale_16,scale_32,scale_64,domain_shift,mask_noise_mild,mask_noise_severe --policy checkpoint --checkpoint outputs/paper_ready_v3/checkpoints/best.pt --episodes 50 --steps 128 --device cuda --output-dir outputs/full_repair/E9_transfer_stress/results
```

## E10 SatEdgeSim Candidate vs Estimator vs Native-Bound Replay

- requires_gpu: false for replay, true if generating checkpoints
- expected_runtime_level: medium
- seeds: max decisions per checkpoint and online seeds
- output_dir: `outputs/full_repair/E10_satedgesim_binding`
- expected_table_or_figure: Table 5 with binding-mode-specific title
- reviewer_issue_addressed: R7, R12, Q12
- paper_claim_enabled_if_passed: candidate-level or estimator-bound replay, or native-bound only if native evidence passes

```powershell
python scripts/replay_on_satedgesim.py --max-decisions 1000 --output-dir outputs/full_repair/E10_satedgesim_binding/replay
python scripts/summarize_satedgesim_replay.py --input-dir outputs/full_repair/E10_satedgesim_binding/replay --output outputs/full_repair/E10_satedgesim_binding/summary.json
```

## E11 Receipt/Completion/Energy Audit

- requires_gpu: false
- expected_runtime_level: medium
- seeds: all replay online seeds
- output_dir: `outputs/full_repair/E11_receipt_completion_energy`
- expected_table_or_figure: receipt/completion/energy audit appendix
- reviewer_issue_addressed: R8, R9, Q2, Q3
- paper_claim_enabled_if_passed: task success and energy claims with completion receipt and known energy source

```powershell
python scripts/run_cpu_smoke_satedgesim_summary_semantics.py --output-dir outputs/full_repair/E11_receipt_completion_energy/fixtures
python scripts/summarize_satedgesim_replay.py --input-dir outputs/full_repair/E10_satedgesim_binding/replay --output outputs/full_repair/E11_receipt_completion_energy/summary.json
```

## E12 Encoder Sharing Diagnostics

- requires_gpu: true for final checkpoints, false for smoke
- expected_runtime_level: medium
- seeds: all algorithm-pair checkpoints
- output_dir: `outputs/full_repair/E12_encoder_diagnostics`
- expected_table_or_figure: encoder gradient/cadence diagnostic appendix
- reviewer_issue_addressed: R13, Q10
- paper_claim_enabled_if_passed: exact encoder-mode wording backed by gradient diagnostics

```powershell
python scripts/run_cpu_smoke_encoder_diagnostics.py --device cuda --output-dir outputs/full_repair/E12_encoder_diagnostics
```

## E13 Trace Split/Leakage Audit

- requires_gpu: false
- expected_runtime_level: low-medium
- seeds: all train/validation/test trace files
- output_dir: `outputs/full_repair/E13_trace_audit`
- expected_table_or_figure: data governance appendix
- reviewer_issue_addressed: R15, Q11
- paper_claim_enabled_if_passed: leakage-audited split statement

```powershell
python scripts/build_trace_manifest.py --project-root . --output-dir outputs/full_repair/E13_trace_audit/manifests
python scripts/audit_trace_splits.py --manifest-dir outputs/full_repair/E13_trace_audit/manifests --output-dir outputs/full_repair/E13_trace_audit/audit
```
