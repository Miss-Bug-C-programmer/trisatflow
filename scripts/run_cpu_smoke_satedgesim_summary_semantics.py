from __future__ import annotations

import csv
import json
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "reviewer_repair" / "satedgesim_semantics"

FIELDNAMES = [
    "step",
    "policy_upper_action",
    "policy_upper_action_name",
    "final_policy_action",
    "executed_abstract_action",
    "executed_abstract_action_name",
    "executedLogicalTier",
    "abstract_action_mask",
    "abstract_action_mask_visible",
    "receipt_accepted",
    "actionAccepted",
    "executionScheduled",
    "intent_execution_match",
    "fallback_reason",
    "taskCompleted",
    "taskSucceeded",
    "success",
    "energy_raw_delta",
    "energy_unit",
    "continuous_resource_binding_mode",
    "continuous_resource_applied",
    "native_scheduler_bound",
    "estimator_bound",
]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_case(case_dir: Path, rows: Iterable[Mapping[str, Any]], final_metrics: Mapping[str, Any] | None = None) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    with (case_dir / "decision_log.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})
    _write_json(case_dir / "summary.json", {"status": "FIXTURE_ONLY"})
    _write_json(case_dir / "final_metrics.json", final_metrics or {})


def _run_summarizer(case_dir: Path) -> Dict[str, Any]:
    output_path = case_dir / "summary_replay.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "summarize_satedgesim_replay.py"),
        "--input-dir",
        str(case_dir),
        "--output",
        str(output_path),
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, capture_output=True, text=True)
    return json.loads(output_path.read_text(encoding="utf-8"))


def run_smoke(output_root: Path = OUTPUT_ROOT) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    cases: List[Dict[str, Any]] = []

    case_a = output_root / "case_a_receipt_only"
    _write_case(
        case_a,
        [
            {
                "step": 0,
                "policy_upper_action": 2,
                "policy_upper_action_name": "GEO",
                "final_policy_action": 2,
                "executed_abstract_action": 2,
                "executed_abstract_action_name": "GEO",
                "executedLogicalTier": "GEO",
                "abstract_action_mask": "[1,1,1,1]",
                "abstract_action_mask_visible": "[1,1,1,1]",
                "receipt_accepted": 1,
                "actionAccepted": True,
                "executionScheduled": True,
                "intent_execution_match": 1,
                "fallback_reason": "none",
                "energy_raw_delta": 0.25,
                "energy_unit": "Wh",
                "continuous_resource_binding_mode": "candidate_only",
                "continuous_resource_applied": False,
                "native_scheduler_bound": False,
                "estimator_bound": False,
            }
        ],
    )
    cases.append({"case": "accepted_receipt_no_completion", "summary": _run_summarizer(case_a)})

    case_b = output_root / "case_b_completion"
    _write_case(
        case_b,
        [
            {
                "step": 0,
                "policy_upper_action": 3,
                "policy_upper_action_name": "GROUND",
                "final_policy_action": 3,
                "executed_abstract_action": 3,
                "executed_abstract_action_name": "GROUND",
                "executedLogicalTier": "GROUND",
                "abstract_action_mask": "[1,1,1,1]",
                "abstract_action_mask_visible": "[1,1,1,1]",
                "receipt_accepted": 1,
                "actionAccepted": True,
                "executionScheduled": True,
                "intent_execution_match": 1,
                "fallback_reason": "none",
                "taskCompleted": True,
                "taskSucceeded": True,
                "success": True,
                "energy_raw_delta": 0.4,
                "energy_unit": "J",
                "continuous_resource_binding_mode": "resource_aware_estimator_bound",
                "continuous_resource_applied": True,
                "native_scheduler_bound": False,
                "estimator_bound": True,
            }
        ],
        {"successRate": 1.0, "tasksSent": 1, "tasksFailed": 0, "energyCounterUnit": "J"},
    )
    cases.append({"case": "accepted_receipt_with_completion", "summary": _run_summarizer(case_b)})

    case_c = output_root / "case_c_energy_sources"
    _write_case(
        case_c,
        [
            {
                "step": 0,
                "policy_upper_action": 1,
                "policy_upper_action_name": "NEIGHBOR",
                "final_policy_action": 1,
                "executed_abstract_action": 1,
                "executed_abstract_action_name": "NEIGHBOR",
                "executedLogicalTier": "EDGE",
                "abstract_action_mask": "[1,1,0,0]",
                "abstract_action_mask_visible": "[1,1,0,0]",
                "receipt_accepted": 1,
                "actionAccepted": True,
                "executionScheduled": True,
                "intent_execution_match": 1,
                "fallback_reason": "none",
                "energy_raw_delta": 0.25,
                "energy_unit": "Wh",
                "continuous_resource_binding_mode": "resource_aware_estimator_bound",
                "continuous_resource_applied": True,
                "native_scheduler_bound": False,
                "estimator_bound": True,
            }
        ],
        {"energyConsumption": 12.5, "energyCounterUnit": "Wh"},
    )
    cases.append({"case": "receipt_delta_and_final_cumulative_energy", "summary": _run_summarizer(case_c)})

    aggregate = {
        "status": "ok",
        "satedgesim_validation_mode": "resource_aware_estimator_bound_replay",
        "table5_title_suggestion": "SatEdgeSim resource-aware estimator-bound replay",
        "cases": cases,
        "semantic_checks": {
            "case_a_success_rate_suppressed": "success_rate" not in cases[0]["summary"],
            "case_a_scheduling_acceptance_rate_present": "scheduling_acceptance_rate" in cases[0]["summary"],
            "case_b_completion_success_present": "completion_success_ratio" in cases[1]["summary"],
            "case_b_estimator_bound_not_native": (
                cases[1]["summary"].get("estimator_bound") is True
                and cases[1]["summary"].get("native_scheduler_bound") is False
                and cases[1]["summary"].get("full_hybrid_closed_loop_claim_allowed") is False
            ),
            "case_c_keeps_receipt_delta_and_final_cumulative": (
                cases[2]["summary"].get("receipt_energy_delta") is not None
                and cases[2]["summary"].get("final_cumulative_energy") is not None
            ),
        },
    }
    _write_json(output_root / "summary.json", aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    summary = run_smoke(args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
