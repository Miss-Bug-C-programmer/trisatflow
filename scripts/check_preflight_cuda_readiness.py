from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _tail_mean(rows: List[Dict[str, str]], key: str, tail: int) -> float:
    if not rows:
        return 0.0
    scope = rows[-tail:] if tail > 0 else rows
    vals = [_to_float(row.get(key), 0.0) for row in scope]
    return float(sum(vals) / max(1, len(vals)))


def _dist_l1(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = sorted(set(a) | set(b))
    return float(sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys))


def _load_json(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _regret_overall(regret_payload: Dict[str, Any], mode: str) -> Dict[str, float]:
    mode_results = regret_payload.get("mode_results") or {}
    mode_block = mode_results.get(mode) or {}
    overall = mode_block.get("overall") or {}
    return {
        "mean_normalized_regret": _to_float(overall.get("mean_normalized_regret"), 0.0),
        "near_optimal_hit_rate_05": _to_float(overall.get("near_optimal_hit_rate_05"), 0.0),
        "near_optimal_hit_rate_10": _to_float(overall.get("near_optimal_hit_rate_10"), 0.0),
        "selected_oracle_agreement": _to_float(overall.get("selected_oracle_agreement"), 0.0),
    }


def _acceptance_ok(summary: Dict[str, Any]) -> bool:
    acc = dict(summary.get("acceptance") or {})
    required = [
        "intent_execution_match_ratio_ge_0_99",
        "fallback_reason_none_ratio_ge_0_99",
        "receipt_accept_ratio_ge_0_99",
        "policy_vs_executed_ratio_diff_le_0_01",
        "http_timeout_eq_0",
        "http_connection_error_eq_0",
    ]
    return all(bool(acc.get(k, False)) for k in required)


def _legacy_readiness(
    metrics_rows: List[Dict[str, str]],
    tie_diag: Dict[str, Any],
    eval_modes: Dict[str, Any],
) -> Dict[str, Any]:
    trace_hit = _tail_mean(metrics_rows, "trace_hit_ratio", 5)
    trace_fallback = _tail_mean(metrics_rows, "trace_fallback_count", 5)
    feasibility = _tail_mean(metrics_rows, "mean_feasibility", 5)

    tie_class = str(tie_diag.get("classification") or "unresolved")
    near_tie_005 = _to_float(tie_diag.get("near_tie_ratio_eps_0_05"), 0.0)

    raw_dist = dict(eval_modes.get("raw_argmax_distribution") or {})
    stochastic_dist = dict(eval_modes.get("stochastic_eval_distribution") or {})
    margin_dist = dict(eval_modes.get("margin_cost_tiebreak_distribution") or {})
    cost_dist = dict(eval_modes.get("cost_greedy_baseline_distribution") or {})

    raw_oracle = _to_float(eval_modes.get("raw_argmax_vs_oracle_agreement"), 0.0)
    stochastic_oracle = _to_float(eval_modes.get("stochastic_eval_vs_oracle_agreement"), 0.0)
    margin_oracle = _to_float(eval_modes.get("margin_cost_tiebreak_vs_oracle_agreement"), 0.0)
    cost_oracle = _to_float(eval_modes.get("cost_greedy_vs_oracle_agreement"), 0.0)
    tie_break_applied = _to_float(eval_modes.get("tie_break_applied_ratio"), 0.0)
    stochastic_dominant = max((float(v) for v in stochastic_dist.values()), default=0.0)

    raw_dominant = max((float(v) for v in raw_dist.values()), default=0.0)
    margin_cost_l1 = _dist_l1(margin_dist, cost_dist)
    margin_cost_oracle_gap = abs(margin_oracle - cost_oracle)
    margin_equivalent_cost = margin_cost_l1 < 0.02 and margin_cost_oracle_gap < 0.01

    reasons: List[str] = []
    requirements_for_cuda: List[str] = []

    if trace_hit < 1.0 - 1.0e-9:
        reasons.append(f"trace_hit_ratio={trace_hit:.6f} < 1.0")
    if trace_fallback > 0.0:
        reasons.append(f"trace_fallback_count_tail={trace_fallback:.6f} > 0")
    if feasibility < 0.95:
        reasons.append(f"mean_feasibility_tail={feasibility:.6f} < 0.95")

    readiness = "fail"
    if reasons:
        requirements_for_cuda.append("fix_trace_or_feasibility_blockers")
    else:
        if tie_class == "true_deterministic_collapse":
            reasons.append("true_deterministic_collapse")
            requirements_for_cuda.append("fix_policy_collapse_before_cuda")
        elif tie_class == "checkpoint_or_logit_bias":
            reasons.append("checkpoint_or_logit_bias")
            requirements_for_cuda.append("fix_state_insensitive_logit_bias_before_cuda")
        elif raw_dominant <= 0.98 and raw_oracle >= 0.60:
            readiness = "strong_pass"
            reasons.append("raw_argmax_not_single_action_and_oracle_agreement_reasonable")
            requirements_for_cuda.append("run_cuda_and_report_raw_argmax")
        elif tie_class == "near_tie_argmax_artifact":
            if margin_equivalent_cost:
                reasons.append("margin_cost_tiebreak_equivalent_to_cost_greedy")
                requirements_for_cuda.append("improve_policy_signal_before_cuda")
            elif margin_oracle >= raw_oracle + 0.20 and near_tie_005 >= 0.50 and tie_break_applied > 0.0 and stochastic_dominant < 0.90:
                readiness = "conditional_pass"
                reasons.append("near_tie_argmax_artifact_with_cost_interpretable_decision_layer")
                requirements_for_cuda.append("report_raw_argmax_stochastic_margin_tiebreak_triplet")
                requirements_for_cuda.append("treat_raw_argmax_single_action_as_limitation")
            else:
                reasons.append("near_tie_artifact_but_eval_mode_evidence_insufficient")
                requirements_for_cuda.append("collect_stronger_eval_mode_evidence")
        else:
            reasons.append("unresolved_tie_bias_classification")
            requirements_for_cuda.append("resolve_tie_bias_classification")

    evidence = {
        "tie_classification": tie_class,
        "near_tie_ratio_eps_0_05": near_tie_005,
        "raw_argmax_dominant_ratio": raw_dominant,
        "raw_argmax_vs_oracle_agreement": raw_oracle,
        "stochastic_eval_vs_oracle_agreement": stochastic_oracle,
        "margin_cost_tiebreak_vs_oracle_agreement": margin_oracle,
        "cost_greedy_vs_oracle_agreement": cost_oracle,
        "tie_break_applied_ratio": tie_break_applied,
        "margin_vs_cost_distribution_l1": margin_cost_l1,
        "margin_vs_cost_oracle_gap": margin_cost_oracle_gap,
        "trace_hit_ratio_tail": trace_hit,
        "trace_fallback_count_tail": trace_fallback,
        "mean_feasibility_tail": feasibility,
    }
    return {"readiness": readiness, "reasons": reasons, "requirements_for_cuda": requirements_for_cuda, "evidence": evidence}


def _costreg_readiness(
    metrics_rows: List[Dict[str, str]],
    state_diag: Dict[str, Any],
    eval_modes: Dict[str, Any],
    regret_eval: Dict[str, Any],
    raw_replay: Dict[str, Any],
    stochastic_replay: Dict[str, Any],
) -> Dict[str, Any]:
    trace_hit = _tail_mean(metrics_rows, "trace_hit_ratio", 10)
    trace_fallback = _tail_mean(metrics_rows, "trace_fallback_count", 10)
    feasibility = _tail_mean(metrics_rows, "mean_feasibility", 10)
    state_class = str(state_diag.get("classification", "unknown"))
    raw_oracle_agreement = _to_float(state_diag.get("raw_argmax_oracle_agreement"), 0.0)
    stochastic_oracle_agreement = _to_float(state_diag.get("stochastic_oracle_agreement"), 0.0)
    prob_oracle_action_mean = _to_float(state_diag.get("prob_oracle_action_mean"), 0.0)
    mi_phase = _to_float((state_diag.get("state_sensitivity") or {}).get("mutual_information_phase_argmax"), 0.0)
    mi_oracle = _to_float((state_diag.get("state_sensitivity") or {}).get("mutual_information_oracle_argmax"), 0.0)
    corr_neg_cost_prob = _to_float((state_diag.get("state_sensitivity") or {}).get("correlation_between_negative_cost_and_policy_prob"), 0.0)
    logit_std_map = dict(state_diag.get("logit_std_across_states_by_action") or {})
    prob_std_map = dict(state_diag.get("prob_std_across_states_by_action") or {})
    logit_std_mean = _to_float(sum(float(v) for v in logit_std_map.values()) / max(1, len(logit_std_map)), 0.0)
    prob_std_mean = _to_float(sum(float(v) for v in prob_std_map.values()) / max(1, len(prob_std_map)), 0.0)
    policy_cost_prior_agreement = _tail_mean(metrics_rows, "policy_cost_prior_agreement", 10)
    raw_dist = dict(state_diag.get("raw_argmax_distribution") or {})
    raw_dominant = max((float(v) for v in raw_dist.values()), default=0.0)
    raw_regret = _regret_overall(regret_eval, "raw_argmax")
    stochastic_regret = _regret_overall(regret_eval, "stochastic_eval")
    margin_regret = _regret_overall(regret_eval, "margin_cost_tiebreak")
    cost_greedy_regret = _regret_overall(regret_eval, "cost_greedy_baseline")

    raw_replay_ok = _acceptance_ok(raw_replay) if raw_replay else True
    stochastic_replay_ok = _acceptance_ok(stochastic_replay) if stochastic_replay else True
    raw_replay_geo = _to_float(raw_replay.get("raw_argmax_geo_ratio", 0.0), 0.0)

    reasons: List[str] = []
    requirements_for_cuda: List[str] = []

    if trace_hit < 1.0 - 1.0e-9:
        reasons.append(f"trace_hit_ratio={trace_hit:.6f} < 1.0")
    if trace_fallback > 0.0:
        reasons.append(f"trace_fallback_count_tail={trace_fallback:.6f} > 0")
    if feasibility < 0.95:
        reasons.append(f"mean_feasibility_tail={feasibility:.6f} < 0.95")
    if raw_replay and not raw_replay_ok:
        reasons.append("raw_replay_acceptance_failed")
    if stochastic_replay and not stochastic_replay_ok:
        reasons.append("stochastic_replay_acceptance_failed")

    readiness = "fail"
    if reasons:
        requirements_for_cuda.append("fix_trace_or_replay_acceptance_blockers")
    else:
        state_conditioned = (
            state_class in {"policy_matches_oracle_conditionally", "state_conditioned_but_argmax_biased", "weak_state_conditioning", "state_signal_flow_ok"}
            or mi_phase > 0.02
            or (corr_neg_cost_prob > 0.25 and logit_std_mean > 0.01 and prob_std_mean > 0.002)
        )
        # "non-single" means no single action dominates all states.
        raw_not_single = raw_dominant < 0.999
        if raw_replay:
            raw_not_single = raw_not_single and (raw_replay_geo <= 0.98)
        replay_ok = raw_replay_ok and stochastic_replay_ok
        raw_mean_regret = _to_float(raw_regret.get("mean_normalized_regret"), 0.0)
        raw_hit05 = _to_float(raw_regret.get("near_optimal_hit_rate_05"), 0.0)
        raw_hit10 = _to_float(raw_regret.get("near_optimal_hit_rate_10"), 0.0)
        stochastic_mean_regret = _to_float(stochastic_regret.get("mean_normalized_regret"), 0.0)
        stochastic_hit10 = _to_float(stochastic_regret.get("near_optimal_hit_rate_10"), 0.0)
        margin_mean_regret = _to_float(margin_regret.get("mean_normalized_regret"), 0.0)
        margin_hit10 = _to_float(margin_regret.get("near_optimal_hit_rate_10"), 0.0)
        alt_regret_better = (
            min(stochastic_mean_regret, margin_mean_regret) <= raw_mean_regret - 0.02
            or max(stochastic_hit10, margin_hit10) >= raw_hit10 + 0.05
        )
        state_signal_good = raw_not_single and mi_phase >= 0.25 and raw_oracle_agreement >= 0.55 and state_conditioned

        if (
            raw_not_single
            and mi_phase >= 0.30
            and raw_oracle_agreement >= 0.55
            and raw_mean_regret <= 0.10
            and raw_hit05 >= 0.60
            and replay_ok
        ):
            readiness = "strong_pass"
            reasons.append("raw_argmax_regret_and_state_conditioning_pass")
            requirements_for_cuda.append("run_cuda_and_report_raw_argmax")
        else:
            if (
                raw_not_single
                and mi_phase >= 0.25
                and raw_mean_regret <= 0.20
                and raw_hit10 >= 0.60
                and alt_regret_better
                and replay_ok
            ):
                readiness = "conditional_pass"
                reasons.append("raw_regret_moderate_and_alternative_eval_modes_improve_regret")
                requirements_for_cuda.append("report_raw_stochastic_margin_costgreedy_ablations")
                requirements_for_cuda.append("treat_raw_argmax_bias_as_limitation")
            elif (
                state_signal_good
                and raw_mean_regret <= 0.25
                and raw_hit10 >= 0.55
                and prob_oracle_action_mean < 0.30
            ):
                readiness = "calibration_required"
                reasons.append("state_conditioned_and_regret_near_pass_but_probability_confidence_weak")
                requirements_for_cuda.append("allow_joint_cpu_preflight_but_block_cuda")
            else:
                if not raw_not_single:
                    reasons.append("raw_argmax_single_action_collapse")
                    requirements_for_cuda.append("improve_state_conditioned_learning_before_cuda")
                elif mi_phase <= 1.0e-8:
                    reasons.append("mutual_information_phase_argmax_is_zero")
                    requirements_for_cuda.append("improve_state_conditioned_learning_before_cuda")
                elif raw_mean_regret > 0.25:
                    reasons.append("raw_argmax_mean_regret_too_high")
                    requirements_for_cuda.append("improve_regret_before_cuda")
                elif raw_hit10 < 0.55:
                    reasons.append("raw_argmax_near_optimal_hit_rate_too_low")
                    requirements_for_cuda.append("improve_near_optimal_hit_rate_before_cuda")
                elif raw_oracle_agreement < 0.55:
                    reasons.append("raw_argmax_oracle_agreement_too_low")
                    requirements_for_cuda.append("improve_policy_signal_before_cuda")
                else:
                    reasons.append("insufficient_state_conditioned_evidence")
                    requirements_for_cuda.append("improve_policy_cost_prior_alignment_before_cuda")

    evidence = {
        "state_conditioned_classification": state_class,
        "raw_argmax_oracle_agreement": raw_oracle_agreement,
        "stochastic_oracle_agreement": stochastic_oracle_agreement,
        "prob_oracle_action_mean": prob_oracle_action_mean,
        "mutual_information_phase_argmax": mi_phase,
        "correlation_between_negative_cost_and_policy_prob": corr_neg_cost_prob,
        "logit_std_across_states_by_action_mean": logit_std_mean,
        "prob_std_across_states_by_action_mean": prob_std_mean,
        "mutual_information_oracle_argmax": mi_oracle,
        "policy_cost_prior_agreement_tail": policy_cost_prior_agreement,
        "raw_argmax_dominant_ratio": raw_dominant,
        "raw_argmax_regret": raw_regret,
        "stochastic_eval_regret": stochastic_regret,
        "margin_cost_tiebreak_regret": margin_regret,
        "cost_greedy_baseline_regret": cost_greedy_regret,
        "trace_hit_ratio_tail": trace_hit,
        "trace_fallback_count_tail": trace_fallback,
        "mean_feasibility_tail": feasibility,
        "raw_replay_acceptance_ok": raw_replay_ok,
        "stochastic_replay_acceptance_ok": stochastic_replay_ok,
        "raw_replay_geo_ratio": raw_replay_geo,
        "eval_modes_raw_oracle_agreement": _to_float(eval_modes.get("raw_argmax_vs_oracle_agreement"), 0.0),
        "eval_modes_stochastic_oracle_agreement": _to_float(eval_modes.get("stochastic_eval_vs_oracle_agreement"), 0.0),
    }
    return {"readiness": readiness, "reasons": reasons, "requirements_for_cuda": requirements_for_cuda, "evidence": evidence}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check CUDA preflight readiness from diagnostics and replay summaries.")
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--argmax-tie-diagnosis", type=str, default="")
    parser.add_argument("--state-conditioned-diagnosis", type=str, default="")
    parser.add_argument("--eval-modes", type=str, required=True)
    parser.add_argument("--regret", type=str, default="")
    parser.add_argument("--raw-replay-summary", type=str, default="")
    parser.add_argument("--stochastic-replay-summary", type=str, default="")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    metrics_rows = list(csv.DictReader(Path(args.metrics).open("r", encoding="utf-8", newline="")))
    tie_diag = _load_json(args.argmax_tie_diagnosis)
    state_diag = _load_json(args.state_conditioned_diagnosis)
    eval_modes = _load_json(args.eval_modes)
    regret_eval = _load_json(args.regret)
    raw_replay = _load_json(args.raw_replay_summary)
    stochastic_replay = _load_json(args.stochastic_replay_summary)

    use_costreg = bool(args.state_conditioned_diagnosis)
    if use_costreg:
        result = _costreg_readiness(metrics_rows, state_diag, eval_modes, regret_eval, raw_replay, stochastic_replay)
    else:
        result = _legacy_readiness(metrics_rows, tie_diag, eval_modes)

    out = {
        "metrics": args.metrics,
        "argmax_tie_diagnosis": args.argmax_tie_diagnosis,
        "state_conditioned_diagnosis": args.state_conditioned_diagnosis,
        "eval_modes": args.eval_modes,
        "regret": args.regret,
        "raw_replay_summary": args.raw_replay_summary,
        "stochastic_replay_summary": args.stochastic_replay_summary,
        "readiness": result["readiness"],
        "reasons": result["reasons"],
        "requirements_for_cuda": result["requirements_for_cuda"],
        "evidence": result["evidence"],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
