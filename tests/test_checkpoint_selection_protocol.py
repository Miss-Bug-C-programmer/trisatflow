from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import aggregate_results  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_per_train_seed_sweep_keeps_independent_checkpoints_and_test_rows(tmp_path: Path) -> None:
    out_root = tmp_path / "seed_protocol_sweep"
    cmd = [
        sys.executable,
        "scripts/sweep_algorithm_combinations.py",
        "--config",
        "trisatflow/configs/small.yaml",
        "--upper",
        "mappo",
        "--lower",
        "maddpg",
        "--episodes",
        "1",
        "--steps",
        "2",
        "--n-leo",
        "4",
        "--train-seeds",
        "13,42",
        "--val-seeds",
        "101",
        "--test-seeds",
        "202,303",
        "--checkpoint-selection",
        "per_train_seed",
        "--device",
        "cpu",
        "--output-root",
        str(out_root),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stderr

    for seed in (13, 42):
        assert (out_root / "train" / f"seed_{seed}" / "upper_mappo__lower_maddpg" / "checkpoint.pt").is_file()

    rows = _read_csv(out_root / "sweep_summary.csv")
    test_rows = [row for row in rows if row["phase"] == "test"]
    assert len(test_rows) == 4
    assert {row["train_seed"] for row in test_rows} == {"13", "42"}
    assert {row["eval_seed"] for row in test_rows} == {"202", "303"}
    assert all(row["checkpoint_selection_mode"] == "per_train_seed" for row in test_rows)
    assert all(row["protocol_role"] == "test" for row in test_rows)
    assert all(row["checkpoint_id"] for row in test_rows)
    assert all(row["experiment_contract_sha256"] for row in test_rows)

    protocol = out_root / "protocol_mappo_maddpg.json"
    assert protocol.is_file()


def test_aggregate_collapses_test_bank_before_train_seed_statistics() -> None:
    raw_rows = [
        {
            "status": "ok",
            "phase": "test",
            "train_seed": "13",
            "eval_seed": "202",
            "upper_algo": "mappo",
            "lower_algo": "maddpg",
            "final_normalized_system_cost": "10.0",
            "checkpoint_selection_mode": "per_train_seed",
        },
        {
            "status": "ok",
            "phase": "test",
            "train_seed": "13",
            "eval_seed": "303",
            "upper_algo": "mappo",
            "lower_algo": "maddpg",
            "final_normalized_system_cost": "14.0",
            "checkpoint_selection_mode": "per_train_seed",
        },
        {
            "status": "ok",
            "phase": "test",
            "train_seed": "42",
            "eval_seed": "202",
            "upper_algo": "mappo",
            "lower_algo": "maddpg",
            "final_normalized_system_cost": "20.0",
            "checkpoint_selection_mode": "per_train_seed",
        },
        {
            "status": "ok",
            "phase": "test",
            "train_seed": "42",
            "eval_seed": "303",
            "upper_algo": "mappo",
            "lower_algo": "maddpg",
            "final_normalized_system_cost": "24.0",
            "checkpoint_selection_mode": "per_train_seed",
        },
    ]
    rows = aggregate_results._normalize_summary_rows(raw_rows, "final_normalized_system_cost")

    assert len(rows) == 2
    assert {row["seed"] for row in rows} == {13, 42}
    assert {row["metric"] for row in rows} == {12.0, 22.0}
    assert all(row["eval_seed_bank"] == "202,303" for row in rows)


def test_paper_mode_rejects_global_checkpoint_selection(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/sweep_algorithm_combinations.py",
        "--config",
        "trisatflow/configs/paper/satedgesim_trace_mixed_v3_safe.yaml",
        "--upper",
        "mappo",
        "--lower",
        "maddpg",
        "--train-seeds",
        "13",
        "--val-seeds",
        "101",
        "--test-seeds",
        "202",
        "--checkpoint-selection",
        "best_val_global",
        "--output-root",
        str(tmp_path / "out"),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, timeout=30)

    assert result.returncode != 0
    assert "best_val_global is not allowed for paper config" in (result.stdout + result.stderr)


def test_aggregate_rejects_inconsistent_test_seed_bank_across_algorithms() -> None:
    raw_rows = [
        {
            "status": "ok",
            "phase": "test",
            "train_seed": "13",
            "eval_seed": "202",
            "upper_algo": "mappo",
            "lower_algo": "maddpg",
            "final_normalized_system_cost": "10.0",
        },
        {
            "status": "ok",
            "phase": "test",
            "train_seed": "13",
            "eval_seed": "303",
            "upper_algo": "ippo",
            "lower_algo": "maddpg",
            "final_normalized_system_cost": "11.0",
        },
    ]

    try:
        aggregate_results._normalize_summary_rows(raw_rows, "final_normalized_system_cost")
    except ValueError as exc:
        assert "inconsistent test seed bank" in str(exc)
    else:
        raise AssertionError("inconsistent test seed bank should fail")
