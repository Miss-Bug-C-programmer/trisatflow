# GPU Transfer and Stress Commands

These commands are for full experiments on a GPU machine. They should not be run on the current CPU-only workstation.

## Trace Manifest and Leakage Audit

```powershell
cd D:\research\experiment\6-DRL_satellite\trisatflow
conda activate D:\conda_envs\receiversync-viz
$env:PYTHONPATH = "$PWD;$PWD\trisatflow"
python scripts/build_trace_manifest.py --project-root . --output-dir outputs/reviewer_repair/trace_stress/manifests
python scripts/audit_trace_splits.py --manifest-dir outputs/reviewer_repair/trace_stress/manifests --output-dir outputs/reviewer_repair/trace_stress/audit
```

## CPU Smoke Stress

```powershell
python scripts/run_stress_suite.py --stress-configs trisatflow/configs/stress/scale_16.yaml --policy random_visible --episodes 1 --steps 4 --device cpu --output-dir outputs/reviewer_repair/trace_stress/stress_smoke
```

## Full Stress Matrix

```powershell
python scripts/run_stress_suite.py --stress-configs trisatflow/configs/stress/scale_16.yaml,trisatflow/configs/stress/scale_32.yaml,trisatflow/configs/stress/scale_64.yaml,trisatflow/configs/stress/isl_sparse.yaml,trisatflow/configs/stress/isl_dense.yaml,trisatflow/configs/stress/gateway_low_visibility.yaml,trisatflow/configs/stress/gateway_high_visibility.yaml,trisatflow/configs/stress/burst_low.yaml,trisatflow/configs/stress/burst_high.yaml,trisatflow/configs/stress/deadline_tight.yaml,trisatflow/configs/stress/deadline_loose.yaml,trisatflow/configs/stress/mask_noise_mild.yaml,trisatflow/configs/stress/mask_noise_severe.yaml,trisatflow/configs/stress/domain_shift_satedgesim.yaml --policy checkpoint --checkpoint outputs/paper_ready_v3/main_actual/train/seed_13/upper_ippo__lower_maddpg/best.pt --episodes 10 --steps 64 --device cuda --output-dir outputs/paper_ready_v3/transfer_stress/full_checkpoint
```

## Transfer Plot Summary

```powershell
python scripts/plot_transfer_results.py --input outputs/paper_ready_v3/transfer_stress/full_checkpoint/stress_results.csv --output-dir outputs/paper_ready_v3/transfer_stress/full_checkpoint/figures
```

## Required Reporting Guard

Only write "inductive transfer smoke supported" if checkpoint forward and evaluation succeed for `scale_32.yaml` and `scale_64.yaml` without fixed-size blocker rows in `transfer_blockers.json`.

