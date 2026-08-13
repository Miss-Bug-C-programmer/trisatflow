from __future__ import annotations

import pytest

from trisatflow.data.trace_manifest import validate_trace_manifest_usage


def _manifest(*, split: str, allowed_usage: list[str], trace_id: str = "trace") -> dict:
    return {
        "trace_id": trace_id,
        "split": split,
        "sha256": "b" * 64,
        "generator_version": "unit-generator-v1",
        "scenario_profile": "unit",
        "seed": 7,
        "created_at": "2026-01-01T00:00:00Z",
        "source": "satedgesim_export",
        "allowed_usage": allowed_usage,
    }


def test_formal_eval_cannot_read_train_split_manifest() -> None:
    train = _manifest(split="train", allowed_usage=["train", "eval"], trace_id="train-trace")

    with pytest.raises(ValueError, match="formal eval cannot read train split trace"):
        validate_trace_manifest_usage([train], usage="eval", run_mode="formal")


def test_formal_replay_cannot_read_train_split_manifest() -> None:
    train = _manifest(split="train", allowed_usage=["train", "replay"], trace_id="train-trace")

    with pytest.raises(ValueError, match="formal replay cannot read train split trace"):
        validate_trace_manifest_usage([train], usage="replay", run_mode="formal")


def test_formal_eval_accepts_test_split_with_eval_usage() -> None:
    test = _manifest(split="test", allowed_usage=["eval"], trace_id="test-trace")

    metadata = validate_trace_manifest_usage([test], usage="eval", run_mode="formal")

    assert metadata["formal_claim_allowed"] is True
    assert metadata["outputs_are_smoke_only"] is False


def test_smoke_usage_can_report_non_formal_metadata_for_train_eval_fixture() -> None:
    train = _manifest(split="train", allowed_usage=["train", "eval"], trace_id="smoke-train-trace")

    metadata = validate_trace_manifest_usage([train], usage="eval", run_mode="smoke")

    assert metadata["formal_claim_allowed"] is False
    assert metadata["outputs_are_smoke_only"] is True
    assert metadata["trace_manifest_errors"]
