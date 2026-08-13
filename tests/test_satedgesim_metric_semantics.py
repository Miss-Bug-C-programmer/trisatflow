from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_cpu_smoke_satedgesim_summary_semantics import run_smoke  # noqa: E402


def test_receipt_only_does_not_emit_completion_success_rate() -> None:
    summary = run_smoke()
    case_a = summary["cases"][0]["summary"]

    assert case_a["satedgesim_validation_mode"] == "candidate_level_discrete_replay"
    assert case_a["continuous_resource_binding_mode"] == "candidate_only"
    assert case_a["continuous_resource_applied"] is False
    assert case_a["native_scheduler_bound"] is False
    assert case_a["estimator_bound"] is False
    assert case_a["full_hybrid_closed_loop_claim_allowed"] is False
    assert case_a["continuous_resource_applied_to_native_scheduler"] is False
    assert case_a["cpu_share_effective"] is False
    assert case_a["bandwidth_share_effective"] is False
    assert case_a["tx_power_ratio_effective"] is False
    assert "success_rate" not in case_a
    assert "completion_success_ratio" not in case_a
    assert case_a["scheduling_acceptance_rate"] == 1.0
    assert case_a["receipt_accept_ratio"] == 1.0


def test_completion_success_is_explicit_and_deprecated_alias_is_marked() -> None:
    summary = run_smoke()
    case_b = summary["cases"][1]["summary"]

    assert case_b["completion_success_available"] is True
    assert case_b["completion_success_ratio"] == 1.0
    assert case_b["task_completion_success_ratio"] == 1.0
    assert case_b["success_rate"] == 1.0
    assert case_b["deprecated_success_rate_alias"] is True
    assert case_b["continuous_resource_binding_mode"] == "resource_aware_estimator_bound"
    assert case_b["continuous_resource_applied"] is True
    assert case_b["estimator_bound"] is True
    assert case_b["native_scheduler_bound"] is False
    assert case_b["full_hybrid_closed_loop_claim_allowed"] is False


def test_energy_sources_are_not_conflated() -> None:
    summary = run_smoke()
    case_c = summary["cases"][2]["summary"]

    assert case_c["energy_source"] == "simlog_final"
    assert case_c["energy_unit"] == "simulator_counter"
    assert case_c["final_cumulative_energy"] == 12.5
    assert case_c["receipt_energy_delta"] == 0.25
    assert "final_task_energy" not in case_c


def test_smoke_summary_records_table_title_and_checks() -> None:
    summary = run_smoke()

    assert summary["table5_title_suggestion"] == "SatEdgeSim resource-aware estimator-bound replay"
    assert all(summary["semantic_checks"].values())
