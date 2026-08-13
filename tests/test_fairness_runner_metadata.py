from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_formal_fairness_runner_smoke_metadata_complete(tmp_path: Path) -> None:
    output_dir = tmp_path / "diagnostic_lower_fairness_smoke"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_baseline_lower_fairness_formal.py",
            "--config",
            "trisatflow/configs/small.yaml",
            "--baselines",
            "geo_only",
            "--lower-allocator",
            "neutral",
            "--episodes",
            "1",
            "--steps",
            "2",
            "--train-seeds",
            "13",
            "--eval-seeds",
            "101",
            "--output-dir",
            str(output_dir),
            "--smoke",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_mode"] == "smoke"
    assert summary["outputs_are_smoke_only"] is True
    assert summary["allocator_mode"] == "neutral"
    assert summary["formal_claim_allowed"] is False
    assert summary["num_training_seeds"] == 1
    assert summary["num_eval_seeds"] == 1
    assert isinstance(summary["config_sha256"], str) and len(summary["config_sha256"]) == 64
    assert isinstance(summary["git_commit"], str) and summary["git_commit"]


def test_smoke_runner_refuses_formal_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "formal_lower_fairness_smoke"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_baseline_lower_fairness_formal.py",
            "--config",
            "trisatflow/configs/small.yaml",
            "--baselines",
            "geo_only",
            "--lower-allocator",
            "neutral",
            "--episodes",
            "1",
            "--steps",
            "2",
            "--output-dir",
            str(output_dir),
            "--smoke",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "smoke fairness runner refuses to write into formal" in (result.stdout + result.stderr)
