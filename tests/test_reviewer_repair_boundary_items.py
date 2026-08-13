from __future__ import annotations

from scripts.generate_reviewer_repair_boundary_items import BoundaryItem, build_boundary_items


def test_boundary_items_have_engineering_schema() -> None:
    items = build_boundary_items()
    assert items
    assert all(isinstance(item, BoundaryItem) for item in items)
    required_ids = {"B00", "B02", "B06", "B08", "B09", "B10"}
    assert required_ids.issubset({item.item_id for item in items})
    for item in items:
        assert item.allowed_claim
        assert item.forbidden_claim
        assert item.unlock_condition
        assert item.evidence_files
        assert item.severity in {"low", "medium", "high", "critical"}


def test_cpu_smoke_is_not_promoted_to_formal_results() -> None:
    item = next(item for item in build_boundary_items() if item.item_id == "B00")
    assert "formal" in item.forbidden_claim.lower()
    assert "full" in item.unlock_condition.lower()


def test_satedgesim_native_claim_remains_guarded_without_native_bound() -> None:
    item = next(item for item in build_boundary_items() if item.item_id == "B06")
    assert "native_scheduler_bound=false" in item.forbidden_claim or "native_scheduler_bound=false" in item.machine_check
    assert "completion" in item.unlock_condition.lower()


def test_strong_baseline_boundary_requires_full_experiment() -> None:
    item = next(item for item in build_boundary_items() if item.item_id == "B09")
    assert "tiny" in item.current_status
    assert "multi-seed" in item.unlock_condition.lower()
