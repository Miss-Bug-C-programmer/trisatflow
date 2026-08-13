#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.satedgesim_eval.action_mapper import abstract_action_mask_from_state, map_upper_to_target_vm_with_trace
from trisatflow.satedgesim_eval.client import SatEdgeSimClient

ACTION_NAMES = ["LOCAL", "NEIGHBOR", "GEO", "GROUND"]
TERMINAL_STATUSES = {"FINISHED", "CLOSED", "FAILED", "ERROR"}
REQUIRED_VERSION_FIELDS = {
    "simulator_version",
    "git_commit",
    "rest_api_schema_version",
    "state_schema_version",
    "candidate_cost_estimator_version",
    "lower_action_binding_version",
    "settings_root",
    "settings_sha256",
    "build_time_utc",
}


def _wait_for_action_state(client: SatEdgeSimClient, *, poll_sleep_sec: float, deadline_sec: float) -> Dict[str, Any]:
    deadline = time.time() + deadline_sec
    state = client.get_state()
    while state.get("status") != "WAITING_FOR_ACTION" and state.get("status") not in TERMINAL_STATUSES:
        if time.time() >= deadline:
            raise TimeoutError(f"timed out waiting for WAITING_FOR_ACTION; last_status={state.get('status')}")
        time.sleep(poll_sleep_sec)
        state = client.get_state()
    return state


def _action_from_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    mask = abstract_action_mask_from_state(dict(state))
    action_index = next((idx for idx, bit in enumerate(mask[:4]) if bool(bit)), 0)
    target_vm_index, mapper_trace = map_upper_to_target_vm_with_trace(dict(state), action_index, require_visible=True)
    task = state.get("task") if isinstance(state.get("task"), Mapping) else {}
    task_id = task.get("id", state.get("taskId"))
    return {
        "decisionId": int(state.get("decisionId", state.get("requestId", -1))),
        "requestId": int(state.get("decisionId", state.get("requestId", -1))),
        "taskId": int(task_id),
        "policyUpperAction": int(action_index),
        "policyUpperActionName": ACTION_NAMES[action_index],
        "abstractAction": int(action_index),
        "abstractActionName": ACTION_NAMES[action_index],
        "targetVmIndex": int(target_vm_index),
        "targetVmId": int(mapper_trace.get("selected_vm_id", -1)),
        "selectedVmId": int(mapper_trace.get("selected_vm_id", -1)),
        "cpuShare": 1.0,
        "bandwidthShare": 1.0,
        "txPowerRatio": 1.0,
        "extra": {"testCase": "stage8_rest_contract"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the SatEdgeSim REST contract required by paper-ready v3.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--devices-count", type=int, default=12)
    parser.add_argument("--scenario-profile", default="default")
    parser.add_argument("--task-source-mode", default="current")
    parser.add_argument("--success-profile", default="paper_strict")
    parser.add_argument("--action-mask-mode", default="completion_safe")
    parser.add_argument("--wait-timeout-ms", type=int, default=30000)
    parser.add_argument("--poll-sleep-sec", type=float, default=0.05)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    client = SatEdgeSimClient(args.base_url, timeout=max(10.0, args.wait_timeout_ms / 1000.0 + 5.0))
    result: Dict[str, Any] = {"checks": {}}
    health = client.ensure_healthy()
    result["checks"]["health_before"] = health

    version = client.version()
    missing = sorted(field for field in REQUIRED_VERSION_FIELDS if not version.get(field))
    if missing:
        raise RuntimeError(f"/version missing fields: {missing}")
    result["checks"]["version"] = version

    state = client.reset(
        devices_count=args.devices_count,
        seed=args.seed,
        wait_for_first_decision=True,
        wait_timeout_ms=args.wait_timeout_ms,
        extra={
            "scenarioProfile": args.scenario_profile,
            "taskSourceMode": args.task_source_mode,
            "successProfile": args.success_profile,
            "actionMaskMode": args.action_mask_mode,
            "maxDecisions": 4,
        },
    )
    result["checks"]["reset_status"] = state.get("status")
    state = _wait_for_action_state(client, poll_sleep_sec=args.poll_sleep_sec, deadline_sec=args.wait_timeout_ms / 1000.0)
    result["checks"]["get_state_status"] = state.get("status")

    action = _action_from_state(state)
    receipt = client.apply_action(action)
    if not bool(receipt.get("accepted")):
        raise RuntimeError(f"/apply_action receipt was not accepted: {receipt}")
    result["checks"]["apply_action"] = receipt

    state = _wait_for_action_state(client, poll_sleep_sec=args.poll_sleep_sec, deadline_sec=args.wait_timeout_ms / 1000.0)
    step_action = _action_from_state(state)
    stepped = client.step(step_action, wait_timeout_ms=args.wait_timeout_ms)
    if stepped.get("status") not in {"WAITING_FOR_ACTION", "FINISHED", "RUNNING"}:
        raise RuntimeError(f"/step returned unexpected status: {stepped.get('status')}")
    result["checks"]["step_status"] = stepped.get("status")

    metrics = client.get_metrics()
    result["checks"]["metrics"] = metrics
    close_payload = client.close()
    result["checks"]["close"] = close_payload
    health_after = client.ensure_healthy()
    result["checks"]["health_after_close"] = health_after
    result["status"] = "SATEDGESIM_REST_CONTRACT_OK"

    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
