from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_smoke_summary(root: Path) -> Path:
    run_dir = root / "profile_main" / "arch_full" / "baseline_geo_only" / "seed_13" / "replay"
    run_dir.mkdir(parents=True)
    summary = {
        "status": "ok",
        "baseline_name": "geo_only",
        "outputs_are_smoke_only": True,
        "tiny_results_are_not_paper_results": True,
        "formal_claim_allowed": False,
        "paper_ready": False,
        "actual_decisions": 1,
    }
    path = run_dir / "summary_compact.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def test_collector_rejects_smoke_summary_by_default(tmp_path: Path) -> None:
    _write_smoke_summary(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_experiment_matrix.py",
            "--input-root",
            str(tmp_path),
            "--output-csv",
            str(tmp_path / "summary_matrix.csv"),
            "--output-json",
            str(tmp_path / "summary_matrix.json"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "formal collector rejected diagnostic/smoke input" in (result.stdout + result.stderr)


def test_allow_diagnostic_inputs_requires_diagnostic_output_name(tmp_path: Path) -> None:
    _write_smoke_summary(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_experiment_matrix.py",
            "--input-root",
            str(tmp_path),
            "--output-csv",
            str(tmp_path / "summary_matrix.csv"),
            "--output-json",
            str(tmp_path / "summary_matrix.json"),
            "--allow-diagnostic-inputs",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "diagnostic inputs require an output path containing 'diagnostic'" in (result.stdout + result.stderr)


def test_allow_diagnostic_inputs_marks_output_rows(tmp_path: Path) -> None:
    _write_smoke_summary(tmp_path)
    out_csv = tmp_path / "diagnostic_summary_matrix.csv"
    out_json = tmp_path / "diagnostic_summary_matrix.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_experiment_matrix.py",
            "--input-root",
            str(tmp_path),
            "--output-csv",
            str(out_csv),
            "--output-json",
            str(out_json),
            "--allow-diagnostic-inputs",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    rows = list(csv.DictReader(out_csv.open("r", encoding="utf-8", newline="")))
    assert rows
    assert rows[0]["diagnostic_input"] == "True"
    assert "outputs_are_smoke_only=true" in rows[0]["diagnostic_reasons"]
