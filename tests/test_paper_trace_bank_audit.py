from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT = REPO_ROOT / "scripts" / "audit_trace_bank.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(semantic: str, mode: str, dense_mode: str, *, controlled: bool = False) -> list[dict[str, object]]:
    source = "controlled_estimate" if controlled else "live"
    return [
        {
            "step": 0,
            "leo_id": 0,
            "phase_id": "phase_a",
            "trace_origin": "satedgesim",
            "synthetic": False,
            "trace_semantic_class": semantic,
            "trace_generation_mode": mode,
            "dense_projection_mode": dense_mode,
            "success_profile": "paper_strict",
            "action_mask_mode": "completion_safe",
            "queue_estimate_source": source,
            "mobility_risk_source": source,
            "abstract_action_mask_visible": [1, 1, 1, 1],
            "abstract_action_mask_completion_safe": [1, 0, 1, 1],
            "abstract_action_mask_mobility_safe": [1, 0, 0, 1],
            "abstract_action_mask_final": [1, 0, 1, 1],
        },
        {
            "step": 1,
            "leo_id": 0,
            "phase_id": "phase_b",
            "trace_origin": "satedgesim",
            "synthetic": False,
            "trace_semantic_class": semantic,
            "trace_generation_mode": mode,
            "dense_projection_mode": dense_mode,
            "success_profile": "paper_strict",
            "action_mask_mode": "completion_safe",
            "queue_estimate_source": source,
            "mobility_risk_source": source,
            "abstract_action_mask_visible": [1, 1, 1, 1],
            "abstract_action_mask_completion_safe": [1, 1, 0, 1],
            "abstract_action_mask_mobility_safe": [1, 1, 0, 0],
            "abstract_action_mask_final": [1, 1, 0, 1],
        },
    ]


def _write_trace(root: Path, rel: str, rows: list[dict[str, object]], *, semantic: str, mode: str, dense_mode: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row, trace_fixture_tag=rel) for row in rows]
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    coverage_status = "DENSE_TRACE_OK" if mode == "dense_projection" else "SEQUENTIAL_TRACE_OK"
    coverage = {
        "status": coverage_status,
        "dense_coverage_ratio": 1.0,
        "num_decision_steps": 2,
        "num_rows": len(rows),
    }
    path.with_suffix(path.suffix + ".coverage.json").write_text(json.dumps(coverage), encoding="utf-8")
    source = str(rows[0]["queue_estimate_source"])
    manifest = {
        "trace_sha256": _sha(path),
        "trace_semantic_class": semantic,
        "trace_origin": "satedgesim",
        "synthetic": False,
        "source_simulator_commit": "abc123",
        "simulator_version": "SatEdgeSim-2.3.0-test",
        "rest_api_schema_version": "paper_v3_rest_v1",
        "state_schema_version": "paper_v3_state_v1",
        "settings_sha256": "settingshash",
        "exporter_version": "paper_v3_exporter_v1",
        "seed": 13,
        "scenario_parameters": {"devices_count": 12},
        "scenario_profile": "mixed_cost_landscape_v2" if "controlled" in semantic else "default",
        "task_source_mode": "round_robin_leo" if "controlled" in semantic else "current",
        "success_profile": "paper_strict",
        "action_mask_mode": "completion_safe",
        "min_link_survival_margin_sec": 0.5,
        "architecture": "full",
        "n_leo": 12,
        "num_steps": 2,
        "trace_generation_mode": mode,
        "dense_projection_mode": dense_mode,
        "candidate_cost_estimator_version": "v1_unified_delay_queue",
        "lower_action_binding_version": "vm_network_power_binding_v1",
        "energy_counter_unit": "Wh",
        "energy_counter_semantics": "cumulative_total_across_all_datacenters",
        "queue_estimate_source": source,
        "mobility_risk_source": source,
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_minimal_bank(root: Path) -> None:
    for split, suffix in (("train", "a"), ("validation", "b"), ("test", "c")):
        _write_trace(
            root,
            f"actual_projection/{split}/seed_{suffix}.jsonl",
            _rows("actual_physical_projection", "dense_projection", "source_projection"),
            semantic="actual_physical_projection",
            mode="dense_projection",
            dense_mode="source_projection",
        )
    _write_trace(
        root,
        "actual_sequential_live/test/seed_seq.jsonl",
        _rows("actual_physical_sequential_live", "sequential_live", "none"),
        semantic="actual_physical_sequential_live",
        mode="sequential_live",
        dense_mode="none",
    )
    _write_trace(
        root,
        "controlled_stress_projection/train/seed_stress.jsonl",
        _rows("controlled_stress_projection", "dense_projection", "source_projection", controlled=True),
        semantic="controlled_stress_projection",
        mode="dense_projection",
        dense_mode="source_projection",
    )


def _run_audit(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(AUDIT),
            "--trace-root",
            str(root),
            "--require-disjoint-splits",
            "--require-provenance",
            "--paper-strict",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_paper_trace_bank_audit_accepts_minimal_valid_bank(tmp_path: Path) -> None:
    _write_minimal_bank(tmp_path)
    result = _run_audit(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "TRACE_BANK_AUDIT_OK"
    assert (tmp_path / "index.json").is_file()


def test_paper_trace_bank_audit_rejects_split_overlap(tmp_path: Path) -> None:
    _write_minimal_bank(tmp_path)
    train = tmp_path / "actual_projection/train/seed_a.jsonl"
    validation = tmp_path / "actual_projection/validation/seed_b.jsonl"
    validation.write_text(train.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = json.loads(validation.with_suffix(validation.suffix + ".manifest.json").read_text(encoding="utf-8"))
    manifest["trace_sha256"] = _sha(validation)
    validation.with_suffix(validation.suffix + ".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = _run_audit(tmp_path)
    assert result.returncode != 0
    assert "split_sha256_overlap:train:validation" in result.stdout


def test_paper_trace_bank_audit_rejects_synthetic_actual(tmp_path: Path) -> None:
    _write_minimal_bank(tmp_path)
    trace = tmp_path / "actual_projection/train/seed_a.jsonl"
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    rows[0]["synthetic"] = True
    trace.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    manifest_path = trace.with_suffix(trace.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["trace_sha256"] = _sha(trace)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = _run_audit(tmp_path)
    assert result.returncode != 0
    assert "synthetic_trace_in_paper_bank" in result.stdout
