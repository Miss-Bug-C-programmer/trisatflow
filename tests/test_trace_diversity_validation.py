from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from trisatflow.envs.action_masks import build_upper_action_mask
from trisatflow.envs.topology_trace import TopologyTraceProvider
from scripts.export_satedgesim_topology_trace import _coverage_status


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "scripts" / "validate_satedgesim_trace.py"
DYNAMIC_TRACE = REPO_ROOT / "tests" / "fixtures" / "traces" / "dynamic_layered_trace.jsonl"
ALL_OPEN_TRACE = REPO_ROOT / "tests" / "fixtures" / "traces" / "all_open_trace.jsonl"
MISSING_LAYER_TRACE = REPO_ROOT / "tests" / "fixtures" / "traces" / "missing_layer_trace.jsonl"


def _run_validator(trace: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), "--trace", str(trace), "--paper-strict", *extra],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def _run_validator_raw(trace: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), "--trace", str(trace), *extra],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_dynamic_layered_trace_passes_paper_strict() -> None:
    result = _run_validator(DYNAMIC_TRACE)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    assert summary["status"] == "TRACE_VALIDATION_OK"
    assert summary["unique_final_masks"] >= 2
    assert summary["neighbor_transition_count"] >= 1
    assert summary["geo_transition_count"] >= 1
    assert summary["ground_transition_count"] >= 1
    assert summary["visibility_prune_ratio"] > 0.0
    assert summary["completion_prune_ratio"] > 0.0
    assert summary["mobility_prune_ratio"] > 0.0
    assert summary["final_layer_mismatch_count"] == 0


def test_all_open_trace_fails_paper_strict() -> None:
    result = _run_validator(ALL_OPEN_TRACE)
    assert result.returncode != 0
    summary = json.loads(result.stdout)
    assert summary["fully_open_visible_ratio"] == 1.0
    assert summary["fully_open_final_ratio"] == 1.0
    assert any("fully_open" in failure for failure in summary["failures"])


def test_missing_layer_fixture_fails_paper_strict() -> None:
    result = _run_validator(MISSING_LAYER_TRACE)
    assert result.returncode != 0
    summary = json.loads(result.stdout)
    assert summary["fallback_due_missing_field_ratio"] == 1.0
    assert any("missing_explicit_layered_mask_rows" in failure for failure in summary["failures"])


def test_visible_mobility_and_completion_modes_select_expected_final_layer(tmp_path: Path) -> None:
    trace = tmp_path / "mask_modes.jsonl"
    _write_jsonl(
        trace,
        [
            {
                "step": 0,
                "leo_id": 0,
                "local_visible": True,
                "neighbor_visible": True,
                "geo_visible": True,
                "ground_visible": True,
                "abstract_action_mask_visible": [1, 1, 1, 1],
                "abstract_action_mask_completion_safe": [1, 0, 1, 1],
                "abstract_action_mask_mobility_safe": [1, 0, 1, 0],
                "abstract_action_mask_final": [1, 1, 1, 1],
                "mask_field_presence": {"visible": True, "completion_safe": True, "mobility_safe": True, "final": True},
                "action_mask_mode": "visible_only",
                "phase_id": "phase_a",
            },
            {
                "step": 1,
                "leo_id": 0,
                "local_visible": True,
                "neighbor_visible": True,
                "geo_visible": True,
                "ground_visible": True,
                "abstract_action_mask_visible": [1, 1, 1, 1],
                "abstract_action_mask_completion_safe": [1, 0, 0, 1],
                "abstract_action_mask_mobility_safe": [1, 0, 1, 1],
                "abstract_action_mask_final": [1, 0, 1, 1],
                "mask_field_presence": {"visible": True, "completion_safe": True, "mobility_safe": True, "final": True},
                "action_mask_mode": "mobility_safe",
                "phase_id": "phase_a",
            },
            {
                "step": 2,
                "leo_id": 0,
                "local_visible": True,
                "neighbor_visible": True,
                "geo_visible": True,
                "ground_visible": True,
                "abstract_action_mask_visible": [1, 1, 1, 1],
                "abstract_action_mask_completion_safe": [1, 0, 0, 1],
                "abstract_action_mask_mobility_safe": [1, 0, 1, 1],
                "abstract_action_mask_final": [1, 0, 0, 1],
                "mask_field_presence": {"visible": True, "completion_safe": True, "mobility_safe": True, "final": True},
                "action_mask_mode": "completion_safe",
                "phase_id": "phase_b",
            },
        ],
    )
    result = _run_validator_raw(trace, "--min-rows", "1", "--min-remote-visible-ratio", "0")
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    assert summary["mask_visible_contradictions"] == 0
    assert summary["final_layer_mismatch_count"] == 0
    assert summary["mobility_not_subset_visible_count"] == 0
    assert summary["completion_relation_violation_count"] == 0


def test_synthetic_trace_fails_paper_strict(tmp_path: Path) -> None:
    trace = tmp_path / "synthetic.jsonl"
    row = json.loads(DYNAMIC_TRACE.read_text(encoding="utf-8").splitlines()[0])
    row["trace_origin"] = "synthetic"
    row["synthetic"] = True
    row["trace_semantic_class"] = "synthetic_debug"
    row["success_profile"] = "synthetic"
    row["dense_projection_mode"] = "none"
    _write_jsonl(trace, [row])
    result = _run_validator(trace)
    assert result.returncode != 0
    summary = json.loads(result.stdout)
    assert summary["synthetic_ratio"] == 1.0
    assert any("trace_origin_not_satedgesim" in failure for failure in summary["failures"])


def test_explicit_remote_prune_to_local_is_not_missing_fallback(tmp_path: Path) -> None:
    trace = tmp_path / "local_only_pruned.jsonl"
    _write_jsonl(
        trace,
        [
            {
                "step": 0,
                "leo_id": 0,
                "abstract_action_mask_visible": [1, 1, 1, 1],
                "abstract_action_mask_completion_safe": [1, 0, 0, 0],
                "abstract_action_mask_mobility_safe": [1, 0, 0, 0],
                "abstract_action_mask_final": [1, 0, 0, 0],
                "mask_field_presence": {"visible": True, "completion_safe": True, "mobility_safe": True, "final": True},
            }
        ],
    )
    provider = TopologyTraceProvider(trace, n_leo=1, device=torch.device("cpu"), repeat=False, strict=True)
    snapshot = provider.snapshot(0)
    diag = build_upper_action_mask(
        visibility_mask=torch.tensor([[True, True, True, True]]),
        architecture_mask=torch.tensor([True, True, True, True]),
        trace_snapshot=snapshot,
        action_mask_enabled=True,
        mode="full",
        legacy_mode="completion_safe",
        enable_visibility_mask=True,
        enable_completion_safe_mask=True,
        enable_mobility_risk_mask=True,
        local_action_index=0,
    )
    assert diag.final_mask.tolist() == [[True, False, False, False]]
    assert float(diag.fallback_due_missing_field_count.item()) == 0.0


def test_missing_mask_fields_are_diagnosed_in_debug_fallback(tmp_path: Path) -> None:
    trace = tmp_path / "visible_only.jsonl"
    _write_jsonl(
        trace,
        [
            {
                "step": 0,
                "leo_id": 0,
                "abstract_action_mask_visible": [1, 1, 0, 0],
            }
        ],
    )
    provider = TopologyTraceProvider(trace, n_leo=1, device=torch.device("cpu"), repeat=False, strict=False)
    snapshot = provider.snapshot(0)
    diag = build_upper_action_mask(
        visibility_mask=torch.tensor([[True, True, True, True]]),
        architecture_mask=torch.tensor([True, True, True, True]),
        trace_snapshot=snapshot,
        action_mask_enabled=True,
        mode="full",
        legacy_mode="completion_safe",
        enable_visibility_mask=True,
        enable_completion_safe_mask=True,
        enable_mobility_risk_mask=True,
        local_action_index=0,
    )
    assert provider.stats()["missing_mask_field_count"] == 3
    assert float(diag.fallback_due_missing_field_count.item()) == 2.0


def test_dense_and_sequential_coverage_status_names_are_distinct() -> None:
    assert _coverage_status(
        trace_mode="dense_projection",
        dense_supported=True,
        sparse_steps=0,
        missing_pairs=0,
        num_rows=4,
    ) == "DENSE_TRACE_OK"
    assert _coverage_status(
        trace_mode="sequential_live",
        dense_supported=False,
        sparse_steps=0,
        missing_pairs=0,
        num_rows=4,
    ) == "SEQUENTIAL_TRACE_OK"
    assert _coverage_status(
        trace_mode="sequential_live",
        dense_supported=False,
        sparse_steps=0,
        missing_pairs=1,
        num_rows=4,
    ) == "SEQUENTIAL_TRACE_INCOMPLETE"
