from __future__ import annotations

import json

from scripts.audit_trace_splits import audit_manifest_dir
from scripts.build_trace_manifest import build_record
from trisatflow.data.trace_manifest import audit_manifest_records, normalize_manifest_record, validate_manifest_record


def _record(**overrides):
    base = {
        "trace_id": "trace/train.jsonl",
        "trace_path": "trace/train.jsonl",
        "source": "satedgesim_export",
        "generation_seed": 13,
        "n_leo": 16,
        "n_geo": 1,
        "n_ground": 1,
        "duration_s": 8.0,
        "slot_duration_s": 1.0,
        "scenario_profile": "default",
        "split": "train",
        "sha256": "a" * 64,
        "content_fingerprint": "sha256:" + "a" * 64,
        "contains_oracle_fields": False,
        "oracle_field_names": [],
        "safe_observable_excludes_oracle_fields": True,
        "generator_config_hash": "g",
        "created_at": "2026-01-01T00:00:00Z",
        "notes": "unit test manifest",
        "same_distribution_non_overlapping": True,
    }
    base.update(overrides)
    return normalize_manifest_record(base)


def test_missing_manifest_fails() -> None:
    summary = audit_manifest_records([])
    assert summary["audit_status"] == "failed_missing_manifest"
    assert summary["leakage_risk"] == "high"


def test_duplicate_sha_detected_as_leakage_risk() -> None:
    train = _record(split="train", trace_id="train", sha256="b" * 64)
    test = _record(split="test", trace_id="test", sha256="b" * 64, generation_seed=89)
    summary = audit_manifest_records([train, test])
    assert summary["leakage_risk"] == "high"
    assert any(issue["code"] == "duplicate_sha256_cross_split" for issue in summary["issues"])


def test_oracle_fields_not_allowed_into_safe_observable() -> None:
    bad = _record(
        contains_oracle_fields=True,
        oracle_field_names=["completion_safe", "link_lifetime_s"],
        safe_observable_excludes_oracle_fields=False,
    )
    assert any("safe_observable_excludes" in err for err in validate_manifest_record(bad))
    summary = audit_manifest_records([bad])
    assert any(issue["code"] == "oracle_leakage_risk" for issue in summary["issues"])


def test_oracle_fields_can_exist_when_safe_observable_excludes_them() -> None:
    good = _record(
        contains_oracle_fields=True,
        oracle_field_names=["completion_safe"],
        safe_observable_excludes_oracle_fields=True,
    )
    summary = audit_manifest_records([good])
    assert not any(issue["code"] == "oracle_leakage_risk" for issue in summary["issues"])


def test_audit_uses_active_manifest_files_and_ignores_stale_records(tmp_path) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    active = _record(trace_id="active/train.jsonl", trace_path="active/train.jsonl", sha256="c" * 64)
    stale = {"trace_id": "stale/tmp.jsonl", "trace_path": "stale/tmp.jsonl"}
    (manifest_dir / "active.json").write_text(json.dumps(active), encoding="utf-8")
    (manifest_dir / "stale.json").write_text(json.dumps(stale), encoding="utf-8")
    (manifest_dir / "manifest_build_summary.json").write_text(
        json.dumps({"manifest_build_status": "ok", "active_manifest_files": ["active.json"]}),
        encoding="utf-8",
    )

    summary = audit_manifest_dir(manifest_dir, tmp_path / "audit")
    assert summary["audit_status"] == "passed"
    assert summary["stale_manifest_files_ignored"] == ["stale.json"]
    assert not summary["issues"]


def test_build_record_derives_dimension_metadata_from_coverage_and_jsonl(tmp_path) -> None:
    project_root = tmp_path
    trace_path = project_root / "traces" / "derived_seed7.jsonl"
    trace_path.parent.mkdir(parents=True)
    rows = [
        {"step": 0, "leo_id": 0, "simulation_time": 10.0, "trace_origin": "satedgesim", "scenario_profile": "derived", "trace_generation_mode": "dense_projection"},
        {"step": 0, "leo_id": 1, "simulation_time": 10.0, "trace_origin": "satedgesim", "scenario_profile": "derived", "trace_generation_mode": "dense_projection"},
        {"step": 1, "leo_id": 0, "simulation_time": 11.0, "trace_origin": "satedgesim", "scenario_profile": "derived", "trace_generation_mode": "dense_projection"},
        {"step": 1, "leo_id": 1, "simulation_time": 11.0, "trace_origin": "satedgesim", "scenario_profile": "derived", "trace_generation_mode": "dense_projection"},
    ]
    trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (trace_path.with_name(trace_path.name + ".coverage.json")).write_text(
        json.dumps({"num_decision_steps": 2, "devices_count_used": 2, "scenario_profile": "derived"}),
        encoding="utf-8",
    )

    record = build_record(trace_path, project_root=project_root)
    assert record["generation_seed"] == 7
    assert record["n_leo"] == 2
    assert record["duration_s"] == 2.0
    assert record["time_start_s"] == 10.0
    assert record["time_end_s"] == 12.0
    assert record["metadata_derivation"]["coverage_json"] is True
    assert record["metadata_derivation"]["jsonl_scanned"] is True
