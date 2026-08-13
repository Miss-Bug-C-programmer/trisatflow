# GPU Strong Baseline Commands

Run these only on a GPU machine. The current workstation is CPU-only and should use the tiny smoke commands.

## P-DQN Hybrid

```powershell
cd D:\research\experiment\6-DRL_satellite\trisatflow
conda activate D:\conda_envs\receiversync-viz
$env:PYTHONPATH = "$PWD;$PWD\trisatflow"
python scripts/train_strong_baseline_tiny.py --baseline pdqn_hybrid --episodes 800 --steps 64 --n-leo 16 --device cuda --output-dir outputs/strong_baselines/full/pdqn_hybrid/seed_13
```

Repeat for all planned training seeds, then evaluate with the same trace split, mask source, observation policy, and lower-action semantics as TriSatFlow.

## Flat Hybrid Actor-Critic

```powershell
python scripts/train_strong_baseline_tiny.py --baseline flat_hybrid_ac --episodes 800 --steps 64 --n-leo 16 --device cuda --output-dir outputs/strong_baselines/full/flat_hybrid_ac/seed_13
```

## Strong Baseline Evaluation

```powershell
python scripts/run_strong_baselines.py --baselines pdqn_hybrid,flat_hybrid_ac,small_scale_grid_oracle --episodes 20 --steps 64 --n-leo 16 --device cuda --output-dir outputs/strong_baselines/full/eval
```

## Oracle Gap Diagnostics

Use `small_scale_grid_oracle` only for small-scale diagnostic settings. It is not a MINLP solver.

```powershell
python scripts/evaluate_oracle_gap.py --episodes 20 --steps 32 --n-leo 4 --device cuda --output-dir outputs/strong_baselines/full/oracle_gap_n4
```

## Required Reporting Guard

Do not mark any strong baseline as `paper_ready=true` until full multi-seed results and Holm-corrected statistics are available.

