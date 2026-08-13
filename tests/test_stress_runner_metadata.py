from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_formal_stress_matrix import main as stress_main
from trisatflow.reporting.formal_input_guard import FormalInputError, validate_summary_tree


STRESS_DIR = Path("trisatflow/configs/stress")


def _write_matrix(path: Path, configs: list[str]) -> None:
    path.write_text(
        "\n".join(["stress_configs:", *[f"  - {item}" for item in configs]]) + "\n",
        encoding="utf-8",
    )


def test_smoke_stress_runner_writes_non_formal_metadata(tmp_path: Path) -> None:
    matrix = tmp_path / "stress_matrix.yaml"
    _write_matrix(matrix, ["trisatflow/configs/stress/scale_16.yaml"])
    out = tmp_path / "stress_smoke"

    rc = stress_main(
        [
            "--matrix-config",
            str(matrix),
            "--run-mode",
            "smoke",
            "--output-dir",
            str(out),
            "--max-episodes",
            "1",
            "--max-steps",
            "1",
        ]
    )

    assert rc == 0
    summary = json.loads((out / "stress_summary.json").read_text(encoding="utf-8"))
    assert summary["run_mode"] == "smoke"
    assert summary["outputs_are_smoke_only"] is True
    assert summary["formal_claim_allowed"] is False
    assert summary["paper_ready"] is False
    assert summary["stress_rows"] == 1
    assert (out / "stress_results.csv").is_file()


def test_formal_runner_rejects_diagnostic_oracle_input(tmp_path: Path) -> None:
    cfg = tmp_path / "oracle_stress.yaml"
    cfg.write_text(
        "\n".join(
            [
                "output_dir: outputs/stress_oracle_debug",
                "scenario:",
                "  n_leo: 4",
                "  episode_len: 1",
                "  mask_source: oracle_trace",
                "  physical:",
                "    enabled: true",
                "experiment:",
                "  diagnostic_oracle_allowed: true",
                "stress:",
                "  stress_name: oracle_debug",
                "  train_n_leo: 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"variable_n_transfer_supported": True}), encoding="utf-8")
    manifest = tmp_path / "trace_manifest.json"
    manifest.write_text(json.dumps({"trace_id": "unit"}), encoding="utf-8")

    rc = stress_main(
        [
            "--config",
            str(cfg),
            "--checkpoint",
            str(checkpoint),
            "--trace-manifest",
            str(manifest),
            "--run-mode",
            "formal",
            "--output-dir",
            str(tmp_path / "formal_out"),
        ]
    )

    assert rc == 2


def test_collector_rejects_smoke_stress_output_as_formal(tmp_path: Path) -> None:
    matrix = tmp_path / "stress_matrix.yaml"
    _write_matrix(matrix, ["trisatflow/configs/stress/scale_16.yaml"])
    out = tmp_path / "stress_smoke"
    assert stress_main(
        [
            "--matrix-config",
            str(matrix),
            "--run-mode",
            "smoke",
            "--output-dir",
            str(out),
            "--max-episodes",
            "1",
            "--max-steps",
            "1",
        ]
    ) == 0

    with pytest.raises(FormalInputError, match="outputs_are_smoke_only=true"):
        validate_summary_tree(out)
