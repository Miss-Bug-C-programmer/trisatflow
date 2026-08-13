from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from satedgesim_semantics import (  # noqa: E402
    completion_observed_ratio,
    completion_success_ratio,
    has_completion_evidence,
    join_completion_receipt,
)


def test_join_completion_receipt_by_decision_id() -> None:
    scheduling_row = {
        "receipt_stage": "scheduling",
        "receipt_decision_id": 3,
        "receipt_task_id": 99,
        "actionAccepted": True,
        "executionScheduled": True,
        "taskCompleted": None,
        "taskSucceeded": None,
    }
    completion = {
        "receiptStage": "completion",
        "decisionId": 3,
        "taskId": 99,
        "taskCompleted": True,
        "taskSucceeded": False,
        "failureReason": "DEADLINE_EXCEEDED",
        "simulationTime": 42.0,
    }

    joined = join_completion_receipt(scheduling_row, [completion])

    assert joined["completion_observed"] is True
    assert joined["taskCompleted"] is True
    assert joined["taskSucceeded"] is False
    assert joined["success"] is False
    assert joined["failureReason"] == "DEADLINE_EXCEEDED"
    assert joined["completion_simulation_time"] == 42.0
    assert has_completion_evidence([joined], {}, {}) is True
    assert completion_success_ratio([joined], {}, {}) == 0.0


def test_missing_completion_is_not_failure() -> None:
    scheduling_row = {
        "receipt_stage": "scheduling",
        "receipt_decision_id": 4,
        "receipt_task_id": 100,
        "actionAccepted": True,
        "executionScheduled": True,
        "taskCompleted": None,
        "taskSucceeded": None,
        "success": None,
        "failureReason": "pending_task_completion",
    }

    joined = join_completion_receipt(scheduling_row, [])

    assert joined["completion_observed"] is False
    assert joined["taskCompleted"] is None
    assert joined["taskSucceeded"] is None
    assert joined["success"] is None
    assert has_completion_evidence([joined], {}, {}) is False
    assert completion_success_ratio([joined], {}, {}) is None
    assert completion_observed_ratio([joined]) == 0.0


def test_second_join_does_not_clear_existing_completion() -> None:
    already_joined = {
        "receipt_stage": "scheduling",
        "receipt_decision_id": 5,
        "receipt_task_id": 101,
        "completion_observed": True,
        "taskCompleted": True,
        "taskSucceeded": True,
        "success": True,
    }

    joined = join_completion_receipt(already_joined, [])

    assert joined["completion_observed"] is True
    assert joined["taskCompleted"] is True
    assert joined["taskSucceeded"] is True
    assert completion_success_ratio([joined], {}, {}) == 1.0
