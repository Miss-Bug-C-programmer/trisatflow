from __future__ import annotations

import json
from pathlib import Path

from scripts.run_formal_stress_matrix import checkpoint_supports_variable_n, main as stress_main


def test_variable_n_incompatible_checkpoint_metadata_fails_fast(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"variable_n_transfer_supported": False}), encoding="utf-8")
    manifest = tmp_path / "trace_manifest.json"
    manifest.write_text(json.dumps({"trace_id": "unit"}), encoding="utf-8")

    rc = stress_main(
        [
            "--config",
            "trisatflow/configs/stress/scale_32.yaml",
            "--checkpoint",
            str(checkpoint),
            "--trace-manifest",
            str(manifest),
            "--run-mode",
            "formal",
            "--output-dir",
            str(tmp_path / "formal_transfer"),
        ]
    )

    assert rc == 2


def test_variable_n_support_requires_explicit_checkpoint_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({}), encoding="utf-8")

    assert checkpoint_supports_variable_n(checkpoint, train_n_leo=16, test_n_leo=16) is True
    assert checkpoint_supports_variable_n(checkpoint, train_n_leo=16, test_n_leo=32) is False


def test_variable_n_supported_metadata_allows_guard(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"variable_n_transfer_supported": True}), encoding="utf-8")

    assert checkpoint_supports_variable_n(checkpoint, train_n_leo=16, test_n_leo=32) is True
