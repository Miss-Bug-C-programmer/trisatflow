from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satedgesim_semantics import (
    completion_success_ratio,
    energy_semantics,
    has_completion_evidence,
    resource_binding_semantics,
    validation_metadata,
)
from trisatflow.baselines.registry import (
    ACTION_NAMES,
    apply_architecture_filter,
    baseline_metadata,
    baseline_names,
    build_baseline_policy,
    extract_candidate_info,
    normalize_architecture,
    state_action_mask,
)
from trisatflow.experiment_profiles import get_profile, profile_metadata
from trisatflow.satedgesim_eval.action_mapper import map_upper_to_target_vm_with_trace
from trisatflow.satedgesim_eval.client import SatEdgeSimClient, SatEdgeSimClientError

TERMINAL = {"FINISHED", "CLOSED", "FAILED", "ERROR"}


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v in (None, ""):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _wait_action_state(client: SatEdgeSimClient, poll_sleep_sec: float) -> Dict[str, Any]:
    state = client.get_state()
    while state.get("status") == "RUNNING":
        time.sleep(poll_sleep_sec)
        state = client.get_state()
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay SatEdgeSim using baseline registry policies.")
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--architecture", type=str, default="full", choices=["only_leo", "leo_geo", "leo_ground", "full"])
    parser.add_argument(
        "--profile",
        type=str,
        default="mobility_aware_main",
        choices=[
            "mobility_aware_main",
            "mobility_stress_visible",
            "preflight_lenient",
            "mobility_aware_main_v1",
            "mobility_stress_visible_v1",
            "preflight_lenient_v1",
        ],
    )
    parser.add_argument("--fallback-policy", type=str, default="cost_greedy", choices=["cost_greedy", "random_visible", "local"])
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8088")
    parser.add_argument("--scenario-profile", type=str, default="mixed_cost_landscape_v2")
    parser.add_argument("--task-source-mode", type=str, default="round_robin_leo")
    parser.add_argument("--success-profile", type=str, default="")
    parser.add_argument("--action-mask-mode", type=str, default="")
    parser.add_argument("--min-link-survival-margin-sec", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--devices-count", type=int, default=20)
    parser.add_argument("--max-decisions", type=int, default=500)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--wait-timeout-ms", type=int, default=30000)
    parser.add_argument(
        "--continuous-resource-binding-mode",
        type=str,
        default="resource_aware_estimator_bound",
        choices=["candidate_only", "resource_aware_estimator_bound", "native_scheduler_bound"],
        help="How SatEdgeSim should bind cpuShare/bandwidthShare/txPowerRatio.",
    )
    parser.add_argument("--poll-sleep-sec", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--allow-placeholder-baselines", action="store_true")
    parser.add_argument("--allow-non-paper-ready-baselines", action="store_true")
    args = parser.parse_args()

    if args.baseline not in baseline_names():
        raise ValueError(f"unsupported baseline={args.baseline!r}; choose from {baseline_names()}")
    meta = baseline_metadata(args.baseline)
    if meta.type == "placeholder" and not bool(args.allow_placeholder_baselines):
        raise ValueError(
            f"baseline={args.baseline!r} is placeholder and blocked by default; "
            "pass --allow-placeholder-baselines to force run"
        )
    if (not meta.paper_ready) and (meta.type != "placeholder") and not bool(args.allow_non_paper_ready_baselines):
        raise ValueError(
            f"baseline={args.baseline!r} is non-paper-ready and blocked by default; "
            "pass --allow-non-paper-ready-baselines to run debug baselines"
        )
    architecture = normalize_architecture(args.architecture)
    profile = get_profile(args.profile)
    success_profile = args.success_profile or profile.success_profile
    action_mask_mode = args.action_mask_mode or profile.action_mask_mode

    rng = random.Random(args.seed)
    policy = build_baseline_policy(args.baseline)
    if hasattr(policy, "fallback_policy"):
        setattr(policy, "fallback_policy", str(args.fallback_policy))

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    client = SatEdgeSimClient(args.base_url, timeout=args.request_timeout)
    health = client.ensure_healthy()
    state = client.reset(
        devices_count=args.devices_count,
        seed=args.seed,
        wait_for_first_decision=True,
        wait_timeout_ms=args.wait_timeout_ms,
        extra={
            "scenarioProfile": args.scenario_profile,
            "taskSourceMode": args.task_source_mode,
            "successProfile": success_profile,
            "actionMaskMode": action_mask_mode,
            "minLinkSurvivalMarginSec": max(0.0, float(args.min_link_survival_margin_sec)),
            "maxDecisions": int(args.max_decisions),
        },
    )

    rows: List[Dict[str, Any]] = []
    start = time.time()

    try:
        for step in range(args.max_decisions):
            if state.get("status") in TERMINAL:
                break
            if state.get("status") != "WAITING_FOR_ACTION":
                state = _wait_action_state(client, args.poll_sleep_sec)
                continue
            state = client.get_state()
            if state.get("status") in TERMINAL:
                break
            if state.get("status") != "WAITING_FOR_ACTION":
                continue

            raw_mask = state_action_mask(state, action_mask_mode=action_mask_mode)
            masked = apply_architecture_filter(raw_mask, architecture)
            candidate_info = extract_candidate_info(state)

            decision = policy.select_action(
                obs=None,
                state=state,
                mask=masked,
                candidate_info=candidate_info,
                rng=rng,
            )
            upper_action = int(decision["upper_action"])
            mapper_idx, mapper_trace = map_upper_to_target_vm_with_trace(state, upper_action, require_visible=False)

            if mapper_idx < 0:
                # Last-resort fallback to local for bridge compatibility.
                upper_action = 0
                mapper_idx, mapper_trace = map_upper_to_target_vm_with_trace(state, upper_action, require_visible=False)
                decision["decision_info"]["fallback_used"] = True
                decision["decision_info"]["fallback_reason"] = "mapper_no_candidate"

            task = state.get("task") or {}
            decision_id = int(state.get("decisionId", state.get("requestId", -1)))
            task_id = int(task.get("id", state.get("taskId", -1)))

            payload = {
                "decisionId": decision_id,
                "requestId": decision_id,
                "taskId": task_id,
                "targetVmIndex": int(mapper_idx),
                "targetVmId": int(_to_float(mapper_trace.get("selected_vm_id"), -1)),
                "selectedVmId": int(_to_float(mapper_trace.get("selected_vm_id"), -1)),
                "policyUpperAction": int(upper_action),
                "policyUpperActionName": ACTION_NAMES[int(upper_action)].upper(),
                "abstractAction": int(upper_action),
                "abstractActionName": ACTION_NAMES[int(upper_action)].upper(),
                "cpuShare": float(decision.get("lower_action", [1.0, 1.0, 1.0])[0]),
                "bandwidthShare": float(decision.get("lower_action", [1.0, 1.0, 1.0])[1]),
                "txPowerRatio": float(decision.get("lower_action", [1.0, 1.0, 1.0])[2]),
                "queuePriority": 1.0,
                "extra": {
                    "baselineName": args.baseline,
                    "architecture": architecture,
                    "profileName": profile.profile_name,
                    "selectionReason": decision.get("decision_info", {}).get("selection_reason"),
                    "continuous_resource_binding_mode": args.continuous_resource_binding_mode,
                    "bindingMode": args.continuous_resource_binding_mode,
                },
            }
            try:
                receipt = client.apply_action(payload)
            except SatEdgeSimClientError as exc:
                receipt = {
                    "accepted": False,
                    "actionAccepted": False,
                    "executionScheduled": False,
                    "taskCompleted": False,
                    "taskSucceeded": False,
                    "decisionId": decision_id,
                    "taskId": task_id,
                    "executedAbstractAction": -1,
                    "executedLogicalTier": "",
                    "fallbackReason": exc.error_type,
                    "failureReason": exc.error_type,
                }

            row = {
                "step": int(step),
                "baseline_name": args.baseline,
                "architecture": architecture,
                "profile_name": profile.profile_name,
                "action_mask_mode": action_mask_mode,
                "success_profile": success_profile,
                "policy_upper_action": int(upper_action),
                "policyUpperActionName": ACTION_NAMES[int(upper_action)].upper(),
                "final_policy_action": int(upper_action),
                "final_policy_action_name": ACTION_NAMES[int(upper_action)].upper(),
                "abstract_action_mask": json.dumps([int(x) for x in masked]),
                "selected_vm_id": mapper_trace.get("selected_vm_id"),
                "selected_vm_logical_tier": mapper_trace.get("selected_logical_tier"),
                "selected_vm_abstract_action": mapper_trace.get("selected_abstract_action"),
                "selected_level": mapper_trace.get("selected_level"),
                "selected_candidate_index": mapper_trace.get("selected_candidate_index"),
                "selected_distance": mapper_trace.get("selected_distance"),
                "selected_queue": mapper_trace.get("selected_queue"),
                "selected_capacity": mapper_trace.get("selected_capacity"),
                "selected_rate_mbps": mapper_trace.get("selected_rate_mbps"),
                "selected_prop_delay_sec": mapper_trace.get("selected_prop_delay_sec"),
                "decision_info": json.dumps(decision.get("decision_info") or {}, ensure_ascii=False),
                "selection_reason": (decision.get("decision_info") or {}).get("selection_reason"),
                "cost_rank": (decision.get("decision_info") or {}).get("cost_rank"),
                "mobilityRisk": (decision.get("decision_info") or {}).get("mobility_risk"),
                "estimated_cost": (decision.get("decision_info") or {}).get("estimated_cost"),
                "fallback_reason": receipt.get("fallbackReason", (decision.get("decision_info") or {}).get("fallback_reason", "none")),
                "receipt_accepted": int(bool(receipt.get("accepted", False))),
                "intent_execution_match": int(bool(receipt.get("intentExecutionMatch", False))),
                "executed_abstract_action": receipt.get("executedAbstractAction"),
                "executed_abstract_action_name": ACTION_NAMES[int(receipt.get("executedAbstractAction"))].upper() if str(receipt.get("executedAbstractAction", "")).isdigit() and 0 <= int(receipt.get("executedAbstractAction")) < 4 else "",
                "executed_logical_tier": receipt.get("executedLogicalTier"),
                "delay": receipt.get("delay"),
                "energy_source": receipt.get("energySource", "receipt_delta"),
                "energy_unit": receipt.get("energyUnit", ""),
                "continuous_resource_binding_mode": receipt.get("continuous_resource_binding_mode", receipt.get("continuousResourceBindingMode")),
                "continuous_resource_applied": receipt.get("continuous_resource_applied", receipt.get("continuousResourceApplied")),
                "native_scheduler_bound": receipt.get("native_scheduler_bound", receipt.get("nativeSchedulerBound")),
                "estimator_bound": receipt.get("estimator_bound", receipt.get("estimatorBound")),
                "full_hybrid_closed_loop_claim_allowed": receipt.get(
                    "full_hybrid_closed_loop_claim_allowed",
                    receipt.get("fullHybridClosedLoopClaimAllowed"),
                ),
                "estimatorExpectedDelaySec": receipt.get("estimatorExpectedDelaySec"),
                "estimatorExpectedEnergyJ": receipt.get("estimatorExpectedEnergyJ"),
                "native_binding_applied": receipt.get("native_binding_applied", receipt.get("nativeBindingApplied")),
                "native_cpu_mips_bound": receipt.get("native_cpu_mips_bound", receipt.get("nativeCpuMipsBound")),
                "native_network_bandwidth_bound": receipt.get("native_network_bandwidth_bound", receipt.get("nativeNetworkBandwidthBound")),
                "native_tx_power_bound": receipt.get("native_tx_power_bound", receipt.get("nativeTxPowerBound")),
                "native_base_mips": receipt.get("native_base_mips", receipt.get("nativeBaseMips")),
                "native_applied_mips": receipt.get("native_applied_mips", receipt.get("nativeAppliedMips")),
                "native_cpu_share": receipt.get("native_cpu_share", receipt.get("nativeCpuShare")),
                "native_bandwidth_share": receipt.get("native_bandwidth_share", receipt.get("nativeBandwidthShare")),
                "native_tx_power_ratio": receipt.get("native_tx_power_ratio", receipt.get("nativeTxPowerRatio")),
                "native_cpu_binding_scope": receipt.get("native_cpu_binding_scope", receipt.get("nativeCpuBindingScope")),
                "native_network_binding_scope": receipt.get("native_network_binding_scope", receipt.get("nativeNetworkBindingScope")),
                "native_tx_power_binding_scope": receipt.get("native_tx_power_binding_scope", receipt.get("nativeTxPowerBindingScope")),
                "deadline": receipt.get("deadline"),
                "queueLength": receipt.get("queueLength"),
                "success": receipt.get("taskSucceeded", receipt.get("success")),
                "failureReason": receipt.get("failureReason"),
                "deadlineMiss": receipt.get("deadlineMiss"),
                "queueOverflow": receipt.get("queueOverflow"),
                "vmUnavailable": receipt.get("vmUnavailable"),
                "linkUnavailable": receipt.get("linkUnavailable"),
                "taskDropped": receipt.get("taskDropped"),
                "latencyExceeded": receipt.get("latencyExceeded"),
                "resourceExceeded": receipt.get("resourceExceeded"),
                "linkAvailableNow": receipt.get("linkAvailableNow"),
                "estimatedLinkLifetimeSec": receipt.get("estimatedLinkLifetimeSec"),
                "estimatedTaskTransmissionTimeSec": receipt.get("estimatedTaskTransmissionTimeSec"),
                "estimatedTaskComputeTimeSec": receipt.get("estimatedTaskComputeTimeSec"),
                "estimatedTaskCompletionTimeSec": receipt.get("estimatedTaskCompletionTimeSec"),
                "linkSurvivalMarginSec": receipt.get("linkSurvivalMarginSec"),
                "linkSurvivalMarginToCompletionSec": receipt.get("linkSurvivalMarginToCompletionSec"),
                "handoverRequired": receipt.get("handoverRequired"),
                "handoverAvailable": receipt.get("handoverAvailable"),
                "mobilityRiskSource": receipt.get("mobilityRiskSource"),
            }
            rows.append(row)
            state = client.get_state()
        final_metrics = client.get_metrics()
    finally:
        try:
            close_resp = client.close()
        except Exception as exc:  # noqa: BLE001
            close_resp = {"status": "CLOSE_FAILED", "message": str(exc)}

    total = len(rows)
    receipt_accept_count = sum(int(bool(row.get("receipt_accepted", False))) for row in rows)
    scheduling_success_count = sum(
        int(bool(row.get("executionScheduled", row.get("actionAccepted", row.get("receipt_accepted", False)))))
        for row in rows
    )
    intent_execution_match_count = sum(int(bool(row.get("intent_execution_match", False))) for row in rows)
    no_fallback_count = sum(1 for row in rows if str(row.get("fallback_reason") or "none") == "none")
    energy_info = energy_semantics(rows, {}, final_metrics)
    binding_info = resource_binding_semantics(rows, {}, final_metrics)
    completion_available = has_completion_evidence(rows, {}, final_metrics)
    completion_ratio = completion_success_ratio(rows, {}, final_metrics) if completion_available else None

    summary = {
        **validation_metadata(),
        **binding_info,
        "status": "SATEDGESIM_BASELINE_REPLAY_OK",
        "baseline_name": args.baseline,
        "baseline_metadata": meta.to_dict(),
        "architecture": architecture,
        "profile_name": profile.profile_name,
        "action_mask_mode": action_mask_mode,
        "success_profile": success_profile,
        "profile_metadata": profile_metadata(profile.profile_name),
        "base_url": args.base_url,
        "seed": args.seed,
        "devices_count": args.devices_count,
        "max_decisions": args.max_decisions,
        "actual_decisions": len(rows),
        "elapsed_sec": time.time() - start,
        "health": health,
        "close_response": close_resp,
        "final_metrics": final_metrics,
        "receipt_accept_ratio": receipt_accept_count / max(1, total),
        "receipt_accept_ratio_semantics": "RL API accepted action receipt",
        "scheduling_success_ratio": scheduling_success_count / max(1, total),
        "scheduling_acceptance_rate": scheduling_success_count / max(1, total),
        "scheduling_success_ratio_semantics": "SatEdgeSim accepted candidate scheduling",
        "scheduling_acceptance_rate_semantics": "candidate scheduling acceptance when completion evidence is unavailable",
        "intent_execution_match_ratio": intent_execution_match_count / max(1, total),
        "intent_execution_match_ratio_semantics": "abstract policy action mapped to intended executed tier",
        "no_fallback_ratio": no_fallback_count / max(1, total),
        "completion_success_available": completion_available,
        "completion_receipt_available": completion_available,
        "deprecated_success_rate_alias": bool(completion_available),
        "success_ratio_semantics": "deprecated alias for completion_success_ratio" if completion_available else "suppressed_without_completion_evidence",
        "raw_energy_counter_final": energy_info["final_cumulative_energy"],
        "final_cumulative_energy": energy_info["final_cumulative_energy"],
        "receipt_energy_delta": energy_info["receipt_energy_delta"],
        "energy_source": energy_info["energy_source"],
        "energy_unit": energy_info["energy_unit"],
        "energy_semantics": energy_info["energy_semantics"],
        "energy_audit_status": "requires_manual_audit",
    }
    if completion_available:
        summary["completion_success_ratio"] = completion_ratio
        summary["task_completion_success_ratio"] = completion_ratio
        summary["success_ratio"] = completion_ratio
        summary["success_rate"] = completion_ratio

    _write_csv(outdir / "decision_log.csv", rows)
    _write_json(outdir / "summary.json", summary)
    _write_json(outdir / "final_metrics.json", final_metrics)

    # Reuse unified replay summarizer for compact output and warnings.
    cmd = [
        sys.executable,
        "scripts/summarize_satedgesim_replay.py",
        "--input-dir",
        str(outdir),
        "--output",
        str(outdir / "summary_compact.json"),
    ]
    subprocess.run(cmd, check=True)

    print(f"SATEDGESIM_BASELINE_REPLAY_OK baseline={args.baseline} decisions={len(rows)} output={outdir}")


if __name__ == "__main__":
    main()
