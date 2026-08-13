from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_oracle_upper_bound_small_scale_runs_and_exports(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out = tmp_path / "oracle_out"
    cmd = [
        sys.executable,
        "scripts/run_oracle_upper_bound.py",
        "--config",
        "trisatflow/configs/small.yaml",
        "--n-leo",
        "2",
        "--steps",
        "4",
        "--seed",
        "13",
        "--output-root",
        str(out),
    ]
    p = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr

    summary = out / "oracle_summary.csv"
    assert summary.exists()
    rows = _read_csv(summary)
    assert len(rows) == 1
    row = rows[0]
    assert row.get("baseline") == "oracle_upper_bound_bruteforce_1step"
    assert str(row.get("uses_privileged_info", "")).lower() in {"true", "1"}
    assert "oracle_cost" in row
    assert "oracle_delay" in row
    assert "oracle_energy" in row
    assert "oracle_feasibility" in row


def test_aggregate_can_merge_oracle_and_rl_rows(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_root = tmp_path / "merge_root"
    input_root.mkdir(parents=True, exist_ok=True)

    sweep = input_root / "sweep_summary.csv"
    with sweep.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "status",
                "phase",
                "seed",
                "upper_algo",
                "lower_algo",
                "baseline",
                "observation_ablation",
                "final_mean_system_cost",
                "checkpoint",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "status": "ok",
                "phase": "test",
                "seed": "13",
                "upper_algo": "mappo",
                "lower_algo": "maddpg",
                "baseline": "",
                "observation_ablation": "",
                "final_mean_system_cost": "10.0",
                "checkpoint": "ckpt.pt",
            }
        )

    oracle = input_root / "oracle_summary.csv"
    with oracle.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "status",
                "phase",
                "seed",
                "upper_algo",
                "lower_algo",
                "baseline",
                "observation_ablation",
                "final_mean_system_cost",
                "oracle_cost",
                "uses_privileged_info",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "status": "ok",
                "phase": "test",
                "seed": "13",
                "upper_algo": "oracle",
                "lower_algo": "oracle",
                "baseline": "oracle_upper_bound_bruteforce_1step",
                "observation_ablation": "",
                "final_mean_system_cost": "8.0",
                "oracle_cost": "8.0",
                "uses_privileged_info": "true",
            }
        )

    out = input_root / "summary"
    cmd = [
        sys.executable,
        "scripts/aggregate_results.py",
        "--input-root",
        str(input_root),
        "--output",
        str(out),
        "--phase",
        "test",
    ]
    p = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr

    alg = _read_csv(out / "summary_by_algorithm.csv")
    labels = {(r.get("upper_algo"), r.get("lower_algo"), r.get("baseline")) for r in alg}
    assert ("mappo", "maddpg", "") in labels
    assert ("oracle", "oracle", "oracle_upper_bound_bruteforce_1step") in labels
