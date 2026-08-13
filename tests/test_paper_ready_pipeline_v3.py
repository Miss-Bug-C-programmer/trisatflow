from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from trisatflow.reporting.input_validation import ReportingInputError, load_reporting_input


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_summary(root: Path, rows: list[dict[str, object]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (root / "sweep_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _row(seed: int, semantic: str) -> dict[str, object]:
    return {
        "status": "ok",
        "phase": "test",
        "seed": seed,
        "train_seed": seed,
        "eval_seed": 202,
        "eval_seed_bank": "202,303",
        "n_eval_seeds": 2,
        "upper_algo": "mappo",
        "lower_algo": "maddpg",
        "baseline": "",
        "experiment_contract_sha256": "contract-a",
        "metric_schema_version": "3.0",
        "trace_semantic_class": semantic,
        "observation_mode": "safe_observable",
        "include_oracle_cost": 0,
        "include_cost_prior_features": 0,
        "final_normalized_system_cost": 1.0 + 0.01 * seed,
    }


def test_pipeline_v3_help_lists_required_modes() -> None:
    result = subprocess.run(
        ["bash", "scripts/run_paper_ready_pipeline_v3.sh", "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=True,
    )

    for mode in (
        "preflight-offline",
        "preflight-satedgesim",
        "build-traces",
        "dry-run",
        "formal-main",
        "formal-rules",
        "formal-ablation",
        "formal-learning",
        "formal-replay",
        "formal-report",
    ):
        assert mode in result.stdout


def test_legacy_pipeline_forwards_v3_dry_run_safely() -> None:
    script = (REPO_ROOT / "scripts" / "run_paper_ready_pipeline.sh").read_text(encoding="utf-8")

    assert "deprecated" in script
    assert "run_paper_ready_pipeline_v3.sh" in script
    assert "dry-run --device" in script
    assert "formal-main" in script


def test_satedgesim_preflight_requires_explicit_root(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/preflight_paper_ready_v3.py",
            "--mode",
            "satedgesim",
            "--base-url",
            "http://127.0.0.1:9",
            "--satedgesim-root",
            "",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert result.returncode != 0
    assert "SATEDGESIM_ROOT" in result.stderr
    assert not (tmp_path / "GATE_OK").exists()


def test_primary_semantic_class_rejects_mixed_report_inputs(tmp_path: Path) -> None:
    rows = [_row(seed, "actual_physical_projection") for seed in (13, 42, 57)]
    rows.append(_row(73, "controlled_stress_projection"))
    _write_summary(tmp_path, rows)

    with pytest.raises(ReportingInputError, match="mixed or non-primary trace_semantic_class"):
        load_reporting_input(tmp_path, primary_semantic_class="actual_physical_projection")


def test_primary_semantic_class_accepts_single_class(tmp_path: Path) -> None:
    _write_summary(tmp_path, [_row(seed, "actual_physical_projection") for seed in (13, 42, 57)])

    report = load_reporting_input(tmp_path, primary_semantic_class="actual_physical_projection")

    assert report.contract_sha256 == "contract-a"
    assert {row["trace_semantic_class"] for row in report.rows} == {"actual_physical_projection"}
