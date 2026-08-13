from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.satedgesim_eval.action_mapper import abstract_action_mask_from_state, map_upper_to_target_vm_with_trace
from trisatflow.satedgesim_eval.client import SatEdgeSimClient, SatEdgeSimClientError
from trisatflow.satedgesim_eval.state_adapter import (
    is_controlled_rl_scenario_from_state,
    scenario_profile_from_state,
    source_leo_id_from_state,
    task_source_mode_from_state,
)

ACTION_NAMES = ["local", "neighbor", "geo", "ground"]
TERMINAL_STATUSES = {"FINISHED", "CLOSED", "FAILED", "ERROR"}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_is_feasible(vm: Dict[str, Any], mask: List[Any], idx: int) -> bool:
    if "isFeasible" in vm:
        return bool(vm.get("isFeasible"))
    if idx < len(mask):
        return bool(mask[idx])
    if "feasible" in vm:
        return bool(vm.get("feasible"))
    return True


def _candidate_action(vm: Dict[str, Any], task: Dict[str, Any]) -> int:
    try:
        action = int(vm.get("abstractAction"))
        if 0 <= action <= 3:
            return action
    except (TypeError, ValueError):
        pass
    logical = str(vm.get("logicalTier") or "").strip().lower()
    if logical == "local":
        return 0
    if logical == "neighbor":
        return 1
    if logical in {"geo", "cloud"}:
        return 2
    if logical in {"ground", "edge"}:
        return 3
    source_dc = task.get("sourceDatacenterId")
    if source_dc is not None and str(vm.get("datacenterId")) == str(source_dc):
        return 0
    return -1


def _wait_for_decision(client: SatEdgeSimClient, poll_sleep_sec: float, max_polls: int = 300) -> Dict[str, Any]:
    state = client.get_state()
    polls = 0
    while state.get("status") == "RUNNING" and polls < max_polls:
        time.sleep(poll_sleep_sec)
        state = client.get_state()
        polls += 1
    return state


def _advance_action(state: Dict[str, Any], step: int) -> Dict[str, Any]:
    mask = abstract_action_mask_from_state(state)
    for offset in range(4):
        action = (step + offset) % 4
        if action < len(mask) and bool(mask[action]):
            target_vm_index, _ = map_upper_to_target_vm_with_trace(state, action, require_visible=True)
            return {
                "requestId": int(state.get("requestId", -1)),
                "targetVmIndex": int(target_vm_index),
                "abstractAction": int(action),
                "abstractActionName": ACTION_NAMES[action].upper(),
                "cpuShare": 1.0,
                "bandwidthShare": 1.0,
                "txPowerRatio": 1.0,
                "queuePriority": 1.0,
                "extra": {"diagnosticAdvance": True},
            }
    return {
        "requestId": int(state.get("requestId", -1)),
        "targetVmIndex": -1,
        "abstractAction": 0,
        "abstractActionName": "LOCAL",
        "cpuShare": 1.0,
        "bandwidthShare": 1.0,
        "txPowerRatio": 1.0,
        "queuePriority": 1.0,
        "extra": {"diagnosticAdvance": True},
    }


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _classify_root_cause(
    visible_ratio: List[float],
    delay_mean: List[float],
    mapper_mismatch_count: int,
) -> str:
    local_ratio, neighbor_ratio, geo_ratio, ground_ratio = visible_ratio
    if ground_ratio >= 0.95 and local_ratio >= 0.95 and neighbor_ratio <= 0.05 and geo_ratio <= 0.05:
        return "scene_coverage_insufficient"
    if mapper_mismatch_count > 0:
        return "mapper_bias"
    if min(neighbor_ratio, geo_ratio, ground_ratio) >= 0.20 and delay_mean[3] > 0.0:
        alternatives = [delay_mean[idx] for idx in (1, 2) if delay_mean[idx] > 0.0]
        if alternatives and delay_mean[3] <= min(alternatives) * 0.90:
            return "reward_or_cost_bias_ground_dominant"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose SatEdgeSim live action-space coverage and ground-only risk.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8088")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--devices-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--scenario-profile", type=str, default="default")
    parser.add_argument("--task-source-mode", type=str, default="current")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--poll-sleep-sec", type=float, default=0.05)
    parser.add_argument("--wait-timeout-ms", type=int, default=30000)
    args = parser.parse_args()

    client = SatEdgeSimClient(args.base_url)
    client.ensure_healthy()
    state = client.reset(
        devices_count=args.devices_count,
        seed=args.seed,
        wait_for_first_decision=True,
        wait_timeout_ms=args.wait_timeout_ms,
        extra={
            "scenarioProfile": args.scenario_profile,
            "taskSourceMode": args.task_source_mode,
            "maxDecisions": int(args.samples),
        },
    )

    sampled_states = 0
    visible_counts = [0, 0, 0, 0]
    candidate_count_sum = [0.0, 0.0, 0.0, 0.0]
    delay_values: List[List[float]] = [[], [], [], []]
    queue_values: List[List[float]] = [[], [], [], []]
    rate_values: List[List[float]] = [[], [], [], []]
    distance_values: List[List[float]] = [[], [], [], []]
    capacity_values: List[List[float]] = [[], [], [], []]
    mask_distribution: Counter[str] = Counter()
    source_ids: Counter[int] = Counter()
    scenario_profiles: Counter[str] = Counter()
    source_modes: Counter[str] = Counter()
    mapper_mismatch_count = 0
    per_state_dominant: Counter[str] = Counter()
    fallback_distribution: Counter[str] = Counter()
    controlled_flags: Counter[bool] = Counter()

    try:
        while sampled_states < args.samples:
            if state.get("status") in TERMINAL_STATUSES:
                break
            if state.get("status") != "WAITING_FOR_ACTION":
                state = _wait_for_decision(client, args.poll_sleep_sec)
                continue
            state = client.get_state()
            if state.get("status") != "WAITING_FOR_ACTION":
                continue

            task = dict(state.get("task") or {})
            mask = abstract_action_mask_from_state(state)
            mask_key = "".join(str(int(bool(x))) for x in mask[:4])
            mask_distribution[mask_key] += 1
            for idx in range(4):
                if idx < len(mask) and bool(mask[idx]):
                    visible_counts[idx] += 1

            source_ids[int(source_leo_id_from_state(state))] += 1
            scenario_profiles[scenario_profile_from_state(state)] += 1
            source_modes[task_source_mode_from_state(state)] += 1
            controlled_flags[bool(is_controlled_rl_scenario_from_state(state))] += 1

            action_mask = list(state.get("actionMask") or [])
            candidates = list(state.get("candidateVms") or [])
            tier_counts = [0, 0, 0, 0]
            for idx, vm in enumerate(candidates):
                action = _candidate_action(vm, task)
                if not (0 <= action <= 3):
                    continue
                if not _candidate_is_feasible(vm, action_mask, idx):
                    continue
                tier_counts[action] += 1
                candidate_count_sum[action] += 1.0
                delay_values[action].append(
                    _to_float(vm.get("estimatedTransmissionDelaySec"), 0.0)
                    + _to_float(vm.get("estimatedComputeDelaySec"), 0.0)
                )
                queue_values[action].append(_to_float(vm.get("estimatedQueueLength", vm.get("assignedTasks", 0.0)), 0.0))
                rate_values[action].append(_to_float(vm.get("estimatedTransmissionRateMbps", vm.get("bw", 0.0)), 0.0))
                distance_values[action].append(_to_float(vm.get("sourceDistance", vm.get("distanceToSource", 0.0)), 0.0))
                capacity_values[action].append(
                    _to_float(vm.get("estimatedComputeCapacity"), 0.0)
                    or (_to_float(vm.get("mips"), 0.0) * max(1.0, _to_float(vm.get("pesNumber"), 1.0)))
                )

            dominant_idx = max(range(4), key=lambda tier: tier_counts[tier])
            per_state_dominant[ACTION_NAMES[dominant_idx]] += 1

            for action in range(4):
                if action < len(mask) and bool(mask[action]):
                    _, mapper_trace = map_upper_to_target_vm_with_trace(state, action, require_visible=True)
                    selected = mapper_trace.get("selected_abstract_action")
                    if selected is None or int(selected) != action:
                        mapper_mismatch_count += 1
                    fallback_distribution[str(mapper_trace.get("fallback_reason") or "none")] += 1

            sampled_states += 1
            state = client.step(_advance_action(state, sampled_states), wait_timeout_ms=args.wait_timeout_ms)
    finally:
        try:
            client.close()
        except SatEdgeSimClientError:
            pass

    visible_ratio = [visible_counts[idx] / max(1, sampled_states) for idx in range(4)]
    delay_mean = [_mean(delay_values[idx]) for idx in range(4)]
    queue_mean = [_mean(queue_values[idx]) for idx in range(4)]
    rate_mean = [_mean(rate_values[idx]) for idx in range(4)]
    distance_mean = [_mean(distance_values[idx]) for idx in range(4)]
    capacity_mean = [_mean(capacity_values[idx]) for idx in range(4)]
    likely_reason = _classify_root_cause(visible_ratio, delay_mean, mapper_mismatch_count)

    output = {
        "num_states": sampled_states,
        "scenario_profile_distribution": dict(scenario_profiles),
        "task_source_mode_distribution": dict(source_modes),
        "source_leo_distribution": dict(source_ids),
        "is_controlled_rl_scenario": controlled_flags.most_common(1)[0][0] if controlled_flags else False,
        "local_available_ratio": visible_ratio[0],
        "neighbor_available_ratio": visible_ratio[1],
        "geo_available_ratio": visible_ratio[2],
        "ground_available_ratio": visible_ratio[3],
        "remote_available_ratio": 0.0,
        "local_candidate_count_mean": candidate_count_sum[0] / max(1, sampled_states),
        "neighbor_candidate_count_mean": candidate_count_sum[1] / max(1, sampled_states),
        "geo_candidate_count_mean": candidate_count_sum[2] / max(1, sampled_states),
        "ground_candidate_count_mean": candidate_count_sum[3] / max(1, sampled_states),
        "local_delay_mean": delay_mean[0],
        "neighbor_delay_mean": delay_mean[1],
        "geo_delay_mean": delay_mean[2],
        "ground_delay_mean": delay_mean[3],
        "local_queue_mean": queue_mean[0],
        "neighbor_queue_mean": queue_mean[1],
        "geo_queue_mean": queue_mean[2],
        "ground_queue_mean": queue_mean[3],
        "local_rate_mean": rate_mean[0],
        "neighbor_rate_mean": rate_mean[1],
        "geo_rate_mean": rate_mean[2],
        "ground_rate_mean": rate_mean[3],
        "local_distance_mean": distance_mean[0],
        "neighbor_distance_mean": distance_mean[1],
        "geo_distance_mean": distance_mean[2],
        "ground_distance_mean": distance_mean[3],
        "local_compute_capacity_mean": capacity_mean[0],
        "neighbor_compute_capacity_mean": capacity_mean[1],
        "geo_compute_capacity_mean": capacity_mean[2],
        "ground_compute_capacity_mean": capacity_mean[3],
        "mask_distribution": dict(mask_distribution),
        "dominant_available_tier": per_state_dominant.most_common(1)[0][0] if per_state_dominant else "unknown",
        "mapper_mismatch_count": mapper_mismatch_count,
        "mapper_fallback_distribution": dict(fallback_distribution),
        "likely_ground_only_reason": likely_reason,
        "root_cause_candidates": [
            "scene_coverage_insufficient",
            "reward_or_cost_bias_ground_dominant",
            "mapper_bias",
            "train_replay_distribution_shift",
            "learned_policy_ground_collapse",
            "insufficient_policy_training",
            "unknown",
        ],
    }
    remote_rows = 0
    for mask_key, count in mask_distribution.items():
        if len(mask_key) >= 4 and ("1" in mask_key[1:4]):
            remote_rows += count
    output["remote_available_ratio"] = remote_rows / max(1, sampled_states)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
