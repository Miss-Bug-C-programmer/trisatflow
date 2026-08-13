from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from satedgesim_semantics import (  # noqa: E402
    require_native_scheduler_bound_for_formal_claim,
    resource_binding_semantics,
)


def test_estimator_bound_metadata_is_not_full_closed_loop() -> None:
    rows = [
        {
            "continuous_resource_binding_mode": "resource_aware_estimator_bound",
            "continuous_resource_applied": True,
            "native_scheduler_bound": False,
            "estimator_bound": True,
        }
    ]

    info = resource_binding_semantics(rows, {}, {})

    assert info["resource_binding_mode"] == "resource_aware_estimator_bound"
    assert info["continuous_resource_binding_mode"] == "resource_aware_estimator_bound"
    assert info["native_scheduler_bound"] is False
    assert info["estimator_bound"] is True
    assert info["lower_continuous_allocator_validated_by_satedgesim"] is False
    assert info["full_hybrid_closed_loop_claim_allowed"] is False




def test_native_binding_metadata_allows_formal_claim_with_evidence() -> None:
    rows = [
        {
            "continuous_resource_binding_mode": "native_scheduler_bound",
            "continuous_resource_applied": True,
            "native_scheduler_bound": True,
            "native_binding_applied": True,
            "native_cpu_mips_bound": True,
            "native_network_bandwidth_bound": True,
            "native_tx_power_bound": True,
        }
    ]

    info = resource_binding_semantics(rows, {"native_scheduler_bound": True}, {})

    assert info["resource_binding_mode"] == "native_scheduler_bound"
    assert info["native_scheduler_bound"] is True
    assert info["native_binding_evidence"] is True
    assert info["lower_continuous_allocator_validated_by_satedgesim"] is True
    assert info["full_hybrid_closed_loop_claim_allowed"] is True
    require_native_scheduler_bound_for_formal_claim(rows, {"native_scheduler_bound": True}, {})


def test_formal_full_hybrid_claim_fails_without_native_binding() -> None:
    rows = [{"continuous_resource_binding_mode": "resource_aware_estimator_bound", "native_scheduler_bound": False}]

    with pytest.raises(ValueError, match="formal SatEdgeSim lower continuous validation requires native_scheduler_bound=true"):
        require_native_scheduler_bound_for_formal_claim(rows, {}, {})


def test_summarizer_require_native_scheduler_bound_rejects_estimator(tmp_path: Path) -> None:
    case_dir = tmp_path / "estimator_bound_case"
    case_dir.mkdir(parents=True)
    with (case_dir / "decision_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "receipt_accepted",
                "actionAccepted",
                "executionScheduled",
                "continuous_resource_binding_mode",
                "continuous_resource_applied",
                "native_scheduler_bound",
                "estimator_bound",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "step": 0,
                "receipt_accepted": 1,
                "actionAccepted": True,
                "executionScheduled": True,
                "continuous_resource_binding_mode": "resource_aware_estimator_bound",
                "continuous_resource_applied": True,
                "native_scheduler_bound": False,
                "estimator_bound": True,
            }
        )
    (case_dir / "summary.json").write_text(json.dumps({"status": "SMOKE"}), encoding="utf-8")
    (case_dir / "final_metrics.json").write_text(
        json.dumps({"energyConsumption": 1.0, "energyCounterUnit": "Wh"}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_satedgesim_replay.py",
            "--input-dir",
            str(case_dir),
            "--require-native-scheduler-bound",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "formal SatEdgeSim lower continuous validation requires native_scheduler_bound=true" in (result.stdout + result.stderr)
