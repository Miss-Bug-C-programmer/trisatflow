from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _dist(counter: Counter, denom: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in sorted(counter.items()):
        out[str(key)] = float(value / max(1, denom))
    return out


def _norm_reason(row: Dict[str, str]) -> str:
    reason = str(row.get("failureReason") or "").strip()
    if reason:
        return reason
    fb = str(row.get("fallback_reason") or "none").strip()
    if fb and fb != "none":
        return fb
    return "unknown_failure"


def _mean(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose replay success/failure structure from decision log and summary.")
    parser.add_argument("--decision-log", type=str, required=True)
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    rows = _read_csv(Path(args.decision_log))
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))

    total = len(rows)
    successes = 0
    failures = 0
    pending = 0
    reason_counter = Counter()
    failure_by_action = Counter()
    failure_by_tier = Counter()
    failure_by_phase = Counter()
    failure_by_task_type = Counter()
    success_by_action = Counter()
    success_by_tier = Counter()
    success_by_phase = Counter()
    fail_delay: List[float] = []
    succ_delay: List[float] = []
    fail_queue: List[float] = []
    succ_queue: List[float] = []
    fail_rate: List[float] = []
    succ_rate: List[float] = []
    deadline_miss = 0
    queue_overflow = 0
    vm_unavailable = 0
    link_unavailable = 0
    task_dropped = 0
    latency_exceeded = 0
    resource_exceeded = 0
    unknown_failure = 0

    for row in rows:
        completed = _to_bool(row.get("taskCompleted"), False)
        success = _to_bool(row.get("taskSucceeded", row.get("success")), False)
        reason = _norm_reason(row)
        if not completed and reason == "pending_task_completion":
            pending += 1
            continue
        action = str(row.get("policyUpperActionName") or row.get("policy_upper_action_name") or "UNKNOWN")
        tier = str(row.get("executedLogicalTier") or row.get("executed_logical_tier") or "UNKNOWN")
        phase = str(row.get("scenario_phase") or "unknown")
        task_type = str(row.get("task_type") or "unknown")
        delay = _to_float(row.get("delay", row.get("estimatedTotalDelaySec")), 0.0)
        queue = _to_float(row.get("queueLength", row.get("estimatedQueueLength")), 0.0)
        rate = _to_float(row.get("estimatedTransmissionRateMbps"), 0.0)
        if success:
            successes += 1
            success_by_action[action] += 1
            success_by_tier[tier] += 1
            success_by_phase[phase] += 1
            succ_delay.append(delay)
            succ_queue.append(queue)
            succ_rate.append(rate)
        else:
            failures += 1
            reason_counter[reason] += 1
            failure_by_action[action] += 1
            failure_by_tier[tier] += 1
            failure_by_phase[phase] += 1
            failure_by_task_type[task_type] += 1
            fail_delay.append(delay)
            fail_queue.append(queue)
            fail_rate.append(rate)
            deadline_miss += int(_to_bool(row.get("deadlineMiss"), False))
            queue_overflow += int(_to_bool(row.get("queueOverflow"), False))
            vm_unavailable += int(_to_bool(row.get("vmUnavailable"), False))
            link_unavailable += int(_to_bool(row.get("linkUnavailable"), False))
            task_dropped += int(_to_bool(row.get("taskDropped"), False))
            latency_exceeded += int(_to_bool(row.get("latencyExceeded"), False))
            resource_exceeded += int(_to_bool(row.get("resourceExceeded"), False))
            unknown_failure += int(_to_bool(row.get("unknownFailure"), False))

    success_ratio = _to_float(
        summary.get("task_completion_success_ratio", summary.get("success_ratio")),
        0.0,
    )
    scheduling_success_ratio = _to_float(summary.get("scheduling_success_ratio", summary.get("receipt_accept_ratio")), 0.0)
    task_failure_reason_dist = summary.get("task_failure_reason_distribution", {})
    if isinstance(task_failure_reason_dist, dict) and task_failure_reason_dist:
        # task-level failure distribution is normalized by tasksSent and aligns with success_ratio semantics.
        reason_counter = Counter({str(k): float(v) for k, v in task_failure_reason_dist.items()})
    failure_total = max(1, failures)

    deadline_ratio = deadline_miss / failure_total
    queue_ratio = queue_overflow / failure_total
    vm_ratio = vm_unavailable / failure_total
    link_ratio = link_unavailable / failure_total
    task_drop_ratio = task_dropped / failure_total
    latency_ratio = latency_exceeded / failure_total
    resource_ratio = resource_exceeded / failure_total
    unknown_ratio = unknown_failure / failure_total

    diagnosis = "unknown"
    if scheduling_success_ratio >= 0.99 and success_ratio < 0.90:
        task_latency = _to_float(task_failure_reason_dist.get("latency_deadline"))
        task_mobility = _to_float(task_failure_reason_dist.get("mobility_link"))
        task_resource = _to_float(task_failure_reason_dist.get("resource_unavailable"))
        if task_failure_reason_dist:
            if max(deadline_ratio, latency_ratio, task_latency) >= 0.30 and task_latency >= task_mobility:
                diagnosis = "deadline_too_strict"
            elif max(link_ratio, task_mobility) >= 0.25:
                diagnosis = "link_unavailable_or_rate_low"
            elif max(vm_ratio, resource_ratio, task_resource) >= 0.25:
                diagnosis = "resource_capacity_insufficient"
            elif queue_ratio >= 0.35:
                diagnosis = "queue_overflow_dominant"
            else:
                diagnosis = "unknown"
        elif pending >= int(0.8 * max(1, total)):
            diagnosis = "receipt_success_mapping_bug"
        elif max(deadline_ratio, latency_ratio, _to_float(task_failure_reason_dist.get("latency_deadline"))) >= 0.45:
            diagnosis = "deadline_too_strict"
        elif queue_ratio >= 0.45:
            diagnosis = "queue_overflow_dominant"
        elif max(vm_ratio, resource_ratio, _to_float(task_failure_reason_dist.get("resource_unavailable"))) >= 0.45:
            diagnosis = "resource_capacity_insufficient"
        elif (
            max(link_ratio, _to_float(task_failure_reason_dist.get("mobility_link"))) >= 0.30
            or (_mean(fail_rate) > 0.0 and _mean(succ_rate) > 0.0 and _mean(fail_rate) < 0.7 * _mean(succ_rate))
        ):
            diagnosis = "link_unavailable_or_rate_low"
        elif sum(reason_counter.values()) > 0 and max(reason_counter.values()) / max(1, failures) < 0.20 and unknown_ratio >= 0.5:
            diagnosis = "satedgesim_success_semantics_bug"
        else:
            # compare failure concentration on tier/action vs their overall usage
            total_action = Counter()
            total_tier = Counter()
            for row in rows:
                total_action[str(row.get("policyUpperActionName") or row.get("policy_upper_action_name") or "UNKNOWN")] += 1
                total_tier[str(row.get("executedLogicalTier") or row.get("executed_logical_tier") or "UNKNOWN")] += 1
            high_risk_tier = False
            for tier, fcnt in failure_by_tier.items():
                usage = total_tier.get(tier, 0)
                if usage > 0 and (fcnt / usage) >= 0.50 and fcnt >= max(10, int(0.20 * failures)):
                    high_risk_tier = True
                    break
            if high_risk_tier:
                diagnosis = "policy_selects_high_failure_tier"
            else:
                diagnosis = "unknown"

    payload: Dict[str, Any] = {
        "decision_log": args.decision_log,
        "summary": args.summary,
        "num_rows": total,
        "num_pending_rows": pending,
        "num_success_rows": successes,
        "num_failure_rows": failures,
        "task_completion_success_ratio": success_ratio,
        "scheduling_success_ratio": scheduling_success_ratio,
        "failure_reason_distribution": summary.get("failure_reason_distribution", _dist(reason_counter, failures)),
        "task_failure_reason_distribution": task_failure_reason_dist,
        "receipt_failure_reason_distribution": summary.get("receipt_failure_reason_distribution", _dist(reason_counter, failures)),
        "failure_by_action": _dist(failure_by_action, failures),
        "failure_by_executed_tier": _dist(failure_by_tier, failures),
        "failure_by_scenario_phase": _dist(failure_by_phase, failures),
        "failure_by_task_type": _dist(failure_by_task_type, failures),
        "success_by_action": _dist(success_by_action, successes),
        "success_by_executed_tier": _dist(success_by_tier, successes),
        "success_by_phase": _dist(success_by_phase, successes),
        "mean_delay_success": _mean(succ_delay),
        "mean_delay_failure": _mean(fail_delay),
        "mean_queue_success": _mean(succ_queue),
        "mean_queue_failure": _mean(fail_queue),
        "mean_rate_success": _mean(succ_rate),
        "mean_rate_failure": _mean(fail_rate),
        "deadline_miss_ratio": deadline_ratio,
        "queue_overflow_ratio": queue_ratio,
        "vm_unavailable_ratio": vm_ratio,
        "link_unavailable_ratio": link_ratio,
        "task_dropped_ratio": task_drop_ratio,
        "latency_exceeded_ratio": latency_ratio,
        "resource_exceeded_ratio": resource_ratio,
        "unknown_failure_ratio": unknown_ratio,
        "diagnosis": diagnosis,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
