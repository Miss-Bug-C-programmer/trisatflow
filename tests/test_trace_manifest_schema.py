from __future__ import annotations

from trisatflow.data.trace_manifest import validate_trace_manifest_schema


def _manifest(**overrides):
    payload = {
        "trace_id": "unit-trace",
        "split": "test",
        "sha256": "a" * 64,
        "generator_version": "unit-generator-v1",
        "scenario_profile": "unit",
        "seed": 123,
        "created_at": "2026-01-01T00:00:00Z",
        "source": "satedgesim_export",
        "allowed_usage": ["eval"],
    }
    payload.update(overrides)
    return payload


def test_formal_trace_manifest_requires_sha256_split_and_source() -> None:
    for field in ("sha256", "split", "source"):
        payload = _manifest()
        payload.pop(field)

        errors = validate_trace_manifest_schema(payload, strict_required=True)

        assert any(field in message for message in errors)


def test_formal_trace_manifest_schema_accepts_complete_target_record() -> None:
    errors = validate_trace_manifest_schema(_manifest(), strict_required=True)

    assert errors == []


def test_formal_trace_manifest_requires_allowed_usage() -> None:
    payload = _manifest()
    payload.pop("allowed_usage")

    errors = validate_trace_manifest_schema(payload, strict_required=True)

    assert any("allowed_usage" in message for message in errors)
