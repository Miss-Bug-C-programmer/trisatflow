from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.satedgesim_eval.action_mapper import abstract_action_mask_from_state, map_upper_to_target_vm_with_trace
from trisatflow.satedgesim_eval.client import SatEdgeSimClient, SatEdgeSimClientError

ACTION_NAMES = ["LOCAL", "NEIGHBOR", "GEO", "GROUND"]
TERMINAL_STATUSES = {"FINISHED", "CLOSED", "FAILED", "ERROR"}


def _wait_for_action_state(client: SatEdgeSimClient, poll_sleep_sec: float, max_polls: int = 600) -> Dict[str, Any]:
    state = client.get_state()
    polls = 0
    while state.get("status") not in {"WAITING_FOR_ACTION"} and state.get("status") not in TERMINAL_STATUSES and polls < max_polls:
        time.sleep(poll_sleep_sec)
        state = client.get_state()
        polls += 1
    return state


def _round_robin_visible_action(mask: List[int], cursor: int) -> int:
    for offset in range(4):
        idx = (cursor + offset) % 4
        if idx < len(mask) and bool(mask[idx]):
            return idx
    return 0


def _to_int(value: Any, default: int = -1) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SatEdgeSim decisionId/taskId execution receipts.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8088")
    parser.add_argument("--scenario-profile", type=str, default="mixed_cost_landscape")
    parser.add_argument("--task-source-mode", type=str, default="round_robin_leo")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--devices-count", type=int, default=20)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--poll-sleep-sec", type=float, default=0.05)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--retry", type=int, default=0)
    args = parser.parse_args()

    client = SatEdgeSimClient(args.base_url, timeout=args.request_timeout)
    client.ensure_healthy()
    state = client.reset(
        devices_count=args.devices_count,
        seed=args.seed,
        wait_for_first_decision=True,
        extra={
            "scenarioProfile": args.scenario_profile,
            "taskSourceMode": args.task_source_mode,
            "maxDecisions": int(args.steps),
        },
    )

    cursor = 0
    rows: List[Dict[str, Any]] = []
    for step in range(args.steps):
        state = _wait_for_action_state(client, args.poll_sleep_sec)
        if state.get("status") in TERMINAL_STATUSES:
            break
        if state.get("status") != "WAITING_FOR_ACTION":
            break

        mask = abstract_action_mask_from_state(state)
        action_index = _round_robin_visible_action(mask, cursor)
        cursor = (cursor + 1) % 4
        target_vm_index, mapper_trace = map_upper_to_target_vm_with_trace(state, action_index, require_visible=True)
        task = dict(state.get("task") or {})
        action = {
            "decisionId": int(state.get("decisionId", state.get("requestId", -1))),
            "requestId": int(state.get("decisionId", state.get("requestId", -1))),
            "taskId": int(task.get("id", state.get("taskId", -1))),
            "policyUpperAction": int(action_index),
            "policyUpperActionName": ACTION_NAMES[action_index],
            "abstractAction": int(action_index),
            "abstractActionName": ACTION_NAMES[action_index],
            "targetVmIndex": int(target_vm_index),
            "targetVmId": _to_int(mapper_trace.get("selected_vm_id"), -1),
            "selectedVmId": _to_int(mapper_trace.get("selected_vm_id"), -1),
            "cpuShare": 1.0,
            "bandwidthShare": 1.0,
            "txPowerRatio": 1.0,
            "queuePriority": 1.0,
            "extra": {"testCase": "decision_receipt"},
        }
        receipt: Dict[str, Any] | None = None
        http_error = ""
        client_elapsed_ms = None
        for attempt in range(args.retry + 1):
            t0 = time.perf_counter()
            try:
                receipt = client.apply_action(action)
                client_elapsed_ms = (time.perf_counter() - t0) * 1000.0
                break
            except SatEdgeSimClientError as exc:
                client_elapsed_ms = (time.perf_counter() - t0) * 1000.0
                http_error = str(exc)
                if attempt >= args.retry:
                    receipt = {
                        "accepted": False,
                        "decisionId": action["decisionId"],
                        "taskId": action["taskId"],
                        "executedAbstractAction": -1,
                        "intentExecutionMatch": False,
                        "fallbackReason": exc.error_type,
                        "serverProcessingMs": None,
                        "_httpStatusCode": exc.status_code,
                    }
                else:
                    time.sleep(args.poll_sleep_sec)
        rows.append(
            {
                "step": step,
                "stateDecisionId": action["decisionId"],
                "receiptDecisionId": receipt.get("decisionId") if receipt else None,
                "stateTaskId": action["taskId"],
                "receiptTaskId": receipt.get("taskId") if receipt else None,
                "submittedAction": action_index,
                "selectedVmId": action["selectedVmId"],
                "receiptExecutedAbstractAction": receipt.get("executedAbstractAction") if receipt else None,
                "accepted": bool(receipt and receipt.get("accepted", False)),
                "intentExecutionMatch": bool(receipt and receipt.get("intentExecutionMatch", False)),
                "fallbackReason": str((receipt or {}).get("fallbackReason") or "none"),
                "serverProcessingMs": (receipt or {}).get("serverProcessingMs"),
                "clientElapsedMs": client_elapsed_ms,
                "httpStatusCode": (receipt or {}).get("_httpStatusCode"),
                "httpError": http_error,
            }
        )

    tested = len(rows)
    decision_match = sum(1 for row in rows if _to_int(row["stateDecisionId"]) == _to_int(row["receiptDecisionId"]))
    task_match = sum(1 for row in rows if _to_int(row["stateTaskId"]) == _to_int(row["receiptTaskId"]))
    intent_match = sum(1 for row in rows if bool(row["intentExecutionMatch"]))
    fallback_counter = Counter(str(row["fallbackReason"]) for row in rows)
    http_timeout_count = sum(1 for row in rows if row["fallbackReason"] == "http_timeout")
    http_connection_error_count = sum(1 for row in rows if row["fallbackReason"] == "http_connection_error")
    accepted_count = sum(1 for row in rows if bool(row["accepted"]))
    result = {
        "tested_decisions": tested,
        "decision_id_match_ratio": decision_match / max(1, tested),
        "task_id_match_ratio": task_match / max(1, tested),
        "intent_execution_match_ratio": intent_match / max(1, tested),
        "receipt_accept_ratio": accepted_count / max(1, tested),
        "http_timeout_count": http_timeout_count,
        "http_connection_error_count": http_connection_error_count,
        "mean_server_processing_ms": sum(float(row["serverProcessingMs"]) for row in rows if row["serverProcessingMs"] is not None) / max(1, sum(1 for row in rows if row["serverProcessingMs"] is not None)),
        "max_server_processing_ms": max([float(row["serverProcessingMs"]) for row in rows if row["serverProcessingMs"] is not None] or [0.0]),
        "mean_client_elapsed_ms": sum(float(row["clientElapsedMs"]) for row in rows if row["clientElapsedMs"] is not None) / max(1, sum(1 for row in rows if row["clientElapsedMs"] is not None)),
        "max_client_elapsed_ms": max([float(row["clientElapsedMs"]) for row in rows if row["clientElapsedMs"] is not None] or [0.0]),
        "fallback_reason_distribution": {key: value / max(1, tested) for key, value in sorted(fallback_counter.items())},
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
