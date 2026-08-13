from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from satedgesim_semantics import (  # noqa: E402
    completion_observed_ratio,
    completion_success_ratio,
    has_completion_evidence,
    is_completion_receipt,
    normalize_scheduling_receipt,
)


def test_scheduling_receipt_keeps_completion_unknown() -> None:
    receipt = normalize_scheduling_receipt(
        {
            "receiptStage": "scheduling",
            "accepted": True,
            "actionAccepted": True,
            "executionScheduled": True,
            "taskCompleted": False,
            "taskSucceeded": False,
            "success": False,
            "decisionId": 7,
            "taskId": 11,
        }
    )

    assert receipt["taskCompleted"] is None
    assert receipt["taskSucceeded"] is None
    assert receipt["success"] is None
    assert not is_completion_receipt(receipt)
    assert has_completion_evidence([receipt], {}, {}) is False
    assert completion_success_ratio([receipt], {}, {}) is None
    assert completion_observed_ratio([receipt]) == 0.0


def test_completion_receipt_is_required_for_task_success_ratio() -> None:
    row = {
        "receiptStage": "completion",
        "decisionId": 7,
        "taskId": 11,
        "taskCompleted": True,
        "taskSucceeded": True,
        "failureReason": "none",
    }

    assert is_completion_receipt(row)
    assert has_completion_evidence([row], {}, {}) is True
    assert completion_success_ratio([row], {}, {}) == 1.0
    assert completion_observed_ratio([row]) == 1.0


def test_deprecated_success_ratio_is_not_completion_evidence() -> None:
    summary = {"success_ratio": 1.0}

    assert has_completion_evidence([], summary, {}) is False
    assert completion_success_ratio([], summary, {}) is None
