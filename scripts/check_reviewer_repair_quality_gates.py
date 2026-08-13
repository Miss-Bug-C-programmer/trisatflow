"""Check reviewer-repair quality gates from CPU smoke artifacts.

The gates intentionally fail or block when evidence is absent. They do not
upgrade CPU smoke results into formal paper claims.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEWER_ROOT = REPO_ROOT / "outputs" / "reviewer_repair"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive audit path
        return {"_json_error": str(exc)}


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _gate(gate_id: str, name: str) -> dict[str, Any]:
    return {"gate_id": gate_id, "name": name, "status": "passed", "evidence": [], "blockers": [], "failures": []}


def _set_blocked(gate: dict[str, Any], reason: str) -> None:
    if gate["status"] != "failed":
        gate["status"] = "blocked"
    gate["blockers"].append(reason)


def _set_failed(gate: dict[str, Any], reason: str) -> None:
    gate["status"] = "failed"
    gate["failures"].append(reason)


def _flatten_dicts(obj: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        rows.append(obj)
        for value in obj.values():
            rows.extend(_flatten_dicts(value))
    elif isinstance(obj, list):
        for item in obj:
            rows.extend(_flatten_dicts(item))
    return rows


def gate_physical() -> dict[str, Any]:
    gate = _gate("A", "Physical metric units and mode separation")
    summary = _load_json(REVIEWER_ROOT / "physical_model" / "summary.json")
    if not summary:
        _set_blocked(gate, "physical_model/summary.json is missing")
        return gate
    examples = summary.get("physical_metric_examples") or []
    required = {"metric", "value", "unit", "source", "normalizer", "comparable_scope"}
    missing = [ex.get("metric", "<unknown>") for ex in examples if not required.issubset(set(ex.keys()))]
    if not examples or missing:
        _set_failed(gate, f"physical metric schema missing required keys for: {missing or 'all examples'}")
    if not summary.get("legacy_mode_ok") or not summary.get("physical_mode_ok"):
        _set_failed(gate, "legacy and physical smoke modes are not both marked ok")
    if not summary.get("units_present"):
        _set_failed(gate, "units_present is false or missing")
    gate["evidence"].append("outputs/reviewer_repair/physical_model/summary.json")
    return gate


def gate_lyapunov() -> dict[str, Any]:
    gate = _gate("B", "Lyapunov semantics and stability claim guard")
    summary = _load_json(REVIEWER_ROOT / "lyapunov_dpp" / "summary.json")
    if not summary:
        _set_blocked(gate, "lyapunov_dpp/summary.json is missing")
        return gate
    rows = _flatten_dicts(summary)
    for row in rows:
        if row.get("queue_stability_claim_allowed") is True and not row.get("stability_theorem_metadata"):
            _set_failed(gate, "queue_stability_claim_allowed=true without theorem/proof metadata")
        if row.get("queue_cap_mode") == "finite_buffer" and row.get("used_as_stability_proof") is True:
            _set_failed(gate, "finite buffer boundedness is marked as a stability proof")
    gate["evidence"].append("outputs/reviewer_repair/lyapunov_dpp/summary.json")
    return gate


def gate_lower_fairness() -> dict[str, Any]:
    gate = _gate("C", "Lower allocator fairness metadata")
    paths = [
        REVIEWER_ROOT / "lower_fairness" / "neutral" / "summary.json",
        REVIEWER_ROOT / "lower_fairness" / "optimized_greedy" / "summary.json",
        REVIEWER_ROOT / "lower_fairness" / "summary.json",
    ]
    found = False
    for path in paths:
        data = _load_json(path)
        if not data:
            continue
        found = True
        rows = data.get("rows") if isinstance(data.get("rows"), list) else _flatten_dicts(data)
        if not any("lower_allocator" in row or "lower_allocator_name" in row for row in rows):
            _set_failed(gate, f"{path} lacks lower_allocator fields")
        gate["evidence"].append(str(path))
    if not found:
        _set_blocked(gate, "no lower_fairness summary.json found")
    return gate


def gate_safe_ablation() -> dict[str, Any]:
    gate = _gate("D", "Safe observable ablation deployability")
    summary = _load_json(REVIEWER_ROOT / "safe_ablation" / "summary.json")
    if not summary:
        _set_blocked(gate, "safe_ablation/summary.json is missing")
        return gate
    rows = summary.get("rows") or []
    if not rows:
        _set_failed(gate, "safe ablation summary has no rows")
    for row in rows:
        if row.get("main_ablation_deployable", True) and row.get("uses_cost_prior") is True:
            _set_failed(gate, f"main ablation row uses cost prior: {row.get('variant')}")
        if row.get("main_ablation_deployable", True) and row.get("uses_oracle_cost") is True:
            _set_failed(gate, f"main ablation row uses oracle cost: {row.get('variant')}")
        if row.get("diagnostic_only") and row.get("deployable") is True:
            _set_failed(gate, f"diagnostic row marked deployable: {row.get('variant')}")
    gate["evidence"].append("outputs/reviewer_repair/safe_ablation/summary.json")
    return gate


def gate_mask() -> dict[str, Any]:
    gate = _gate("E", "Mask source deployability")
    summary = _load_json(REVIEWER_ROOT / "mask_noise" / "summary.json")
    if not summary:
        _set_blocked(gate, "mask_noise/summary.json is missing")
        return gate
    rows = summary.get("settings") or summary.get("rows") or _flatten_dicts(summary)
    for row in rows:
        if "mask_source" not in row:
            continue
        if not row.get("mask_source"):
            _set_failed(gate, "mask_source is empty")
        if row.get("mask_source") == "oracle_trace" and row.get("deployable") is True:
            _set_failed(gate, "oracle_trace mask marked deployable=true")
    if not any("mask_source" in row for row in rows):
        _set_failed(gate, "no mask_source metadata found")
    gate["evidence"].append("outputs/reviewer_repair/mask_noise/summary.json")
    return gate


def gate_satedgesim_semantics() -> dict[str, Any]:
    gate = _gate("F", "SatEdgeSim receipt/completion/energy semantics")
    summary = _load_json(REVIEWER_ROOT / "satedgesim_semantics" / "summary.json")
    if not summary:
        _set_blocked(gate, "satedgesim_semantics/summary.json is missing")
        return gate
    rows = _flatten_dicts(summary)
    for row in rows:
        if row.get("completion_receipt_available") is False and "success_rate" in row:
            _set_failed(gate, "success_rate is present despite completion_receipt_available=false")
        if row.get("native_scheduler_bound") is False and row.get("full_hybrid_closed_loop_claim_allowed") is True:
            _set_failed(gate, "full hybrid closed-loop claim allowed while native_scheduler_bound=false")
        if row.get("energy_source") in {"unknown", "unavailable"} and row.get("energy_advantage_claim_allowed") is True:
            _set_failed(gate, "energy advantage claim allowed with unknown/unavailable energy source")
    gate["evidence"].append("outputs/reviewer_repair/satedgesim_semantics/summary.json")
    return gate


def gate_statistics() -> dict[str, Any]:
    gate = _gate("G", "Statistics protocol and Holm claim guard")
    pairwise = _load_csv(REVIEWER_ROOT / "statistics" / "pairwise_tests.csv")
    claim_guard = _load_json(REVIEWER_ROOT / "statistics" / "claim_guard.json")
    if not pairwise:
        _set_blocked(gate, "statistics/pairwise_tests.csv is missing or empty")
    else:
        if "n_effective_pairs" not in pairwise[0]:
            _set_failed(gate, "pairwise_tests.csv lacks n_effective_pairs")
    if not claim_guard:
        _set_blocked(gate, "statistics/claim_guard.json is missing")
    else:
        text = json.dumps(claim_guard, ensure_ascii=False).lower()
        if claim_guard.get("holm_significant_best") is False and "significantly best" in text:
            _set_failed(gate, "claim_guard contains significant-best wording despite Holm non-significance")
    gate["evidence"].extend(["outputs/reviewer_repair/statistics/pairwise_tests.csv", "outputs/reviewer_repair/statistics/claim_guard.json"])
    return gate


def gate_encoder() -> dict[str, Any]:
    gate = _gate("H", "Encoder gradient diagnostics")
    summary = _load_json(REVIEWER_ROOT / "encoder_diagnostics" / "summary.json")
    report = _load_csv(REVIEWER_ROOT / "encoder_diagnostics" / "gradient_report.csv")
    if not summary:
        _set_blocked(gate, "encoder_diagnostics/summary.json is missing")
    if not report:
        _set_blocked(gate, "encoder_diagnostics/gradient_report.csv is missing")
        return gate
    for row in report:
        if row.get("encoder_mode") == "shared_joint":
            try:
                lower_grad = float(row.get("shared_encoder_grad_norm_from_lower") or 0.0)
            except ValueError:
                lower_grad = 0.0
            if lower_grad <= 0.0:
                _set_failed(gate, "shared_joint row lacks positive lower shared-encoder gradient")
        if row.get("action_collection_detach_confused_with_training_detach") in {"true", "True", "1"}:
            _set_failed(gate, "action collection detach is flagged as confused with training detach")
    gate["evidence"].extend(["outputs/reviewer_repair/encoder_diagnostics/summary.json", "outputs/reviewer_repair/encoder_diagnostics/gradient_report.csv"])
    return gate


def gate_trace() -> dict[str, Any]:
    gate = _gate("I", "Trace leakage and transfer readiness")
    manifest = _load_json(REVIEWER_ROOT / "trace_stress" / "manifests" / "manifest_build_summary.json")
    audit = _load_json(REVIEWER_ROOT / "trace_stress" / "audit" / "audit_summary.json")
    stress = _load_json(REVIEWER_ROOT / "trace_stress" / "stress_smoke" / "stress_summary.json")
    if not manifest:
        _set_blocked(gate, "trace manifest summary missing")
    if audit.get("audit_status") in {"passed", "ok"} and not manifest:
        _set_failed(gate, "trace audit passed while manifest is missing")
    manifest_status = str(manifest.get("manifest_build_status") or manifest.get("status") or "")
    audit_status = str(audit.get("audit_status") or audit.get("status") or "")
    if "failed" in manifest_status or "incomplete" in manifest_status or "failed" in audit_status or "incomplete" in audit_status:
        _set_blocked(gate, f"trace governance blocked: manifest={manifest_status or 'missing'} audit={audit_status or 'missing'}")
    if stress.get("claim_constellation_transfer") is True and gate["status"] == "blocked":
        _set_failed(gate, "transfer claim enabled while trace/transfer gate is blocked")
    gate["evidence"].extend(
        [
            "outputs/reviewer_repair/trace_stress/manifests/manifest_build_summary.json",
            "outputs/reviewer_repair/trace_stress/audit/audit_summary.json",
            "outputs/reviewer_repair/trace_stress/stress_smoke/stress_summary.json",
        ]
    )
    return gate


def gate_strong_baseline() -> dict[str, Any]:
    gate = _gate("J", "Strong baseline training and paper-readiness guard")
    summary = _load_json(REVIEWER_ROOT / "strong_baselines" / "eval" / "strong_baseline_summary.json")
    pdqn = _load_json(REVIEWER_ROOT / "strong_baselines" / "pdqn_tiny" / "training_smoke.json")
    flat = _load_json(REVIEWER_ROOT / "strong_baselines" / "flat_hybrid_tiny" / "training_smoke.json")
    if not summary:
        _set_blocked(gate, "strong_baselines/eval/strong_baseline_summary.json is missing")
    rows = summary.get("rows") or summary.get("methods") or _flatten_dicts(summary)
    for row in rows:
        if row.get("paper_ready") is True and row.get("update_implemented") is not True:
            _set_failed(gate, f"paper_ready=true without update_implemented for {row.get('method')}")
        if row.get("paper_ready") is True and row.get("smoke_training_passed") is not True:
            _set_failed(gate, f"paper_ready=true without smoke_training_passed for {row.get('method')}")
    for name, data in {"pdqn_hybrid": pdqn, "flat_hybrid_ac": flat}.items():
        if not data:
            _set_blocked(gate, f"{name} tiny training_smoke.json missing")
        elif data.get("update_performed") is False and data.get("training_complete_for_smoke") is not True:
            _set_failed(gate, f"{name} tiny training did not perform an update")
    gate["evidence"].extend(
        [
            "outputs/reviewer_repair/strong_baselines/eval/strong_baseline_summary.json",
            "outputs/reviewer_repair/strong_baselines/pdqn_tiny/training_smoke.json",
            "outputs/reviewer_repair/strong_baselines/flat_hybrid_tiny/training_smoke.json",
        ]
    )
    return gate


def build_claim_guard(gates: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [gate for gate in gates if gate["status"] == "failed"]
    blocked = [gate for gate in gates if gate["status"] == "blocked"]
    forbidden = [
        "CPU smoke results are formal experiment results",
        "IPPO+MADDPG is significantly best unless Holm-corrected tests support it",
        "Lyapunov reward guarantees queue stability",
        "oracle_trace mask is deployable",
        "SatEdgeSim validates native continuous resource execution when native_scheduler_bound=false",
        "strong baselines prove state-of-the-art superiority before full multi-seed runs",
        "constellation-size transfer is supported while trace/stress gate is blocked",
    ]
    allowed = [
        "CPU smoke validates code paths and metadata guards only",
        "physical-mode metrics may be reported with units/source/comparable_scope",
        "Lyapunov component may be described as reward shaping/queue regularization",
        "SatEdgeSim results may use candidate-level or estimator-bound replay titles according to binding metadata",
        "algorithm-pair conclusions must follow Holm-corrected checkpoint-level statistics",
    ]
    return {
        "overall_claim_status": "blocked_or_failed" if failed or blocked else "claim_guard_passed_for_smoke_scope",
        "formal_results_available": False,
        "failed_gates": [gate["gate_id"] for gate in failed],
        "blocked_gates": [gate["gate_id"] for gate in blocked],
        "allowed_claims": allowed,
        "forbidden_claims": forbidden,
        "required_wording": {
            "statistics": "IPPO+MADDPG is selected as a mean-ranked reference pairing; the four pairings are statistically comparable under Holm-corrected pairwise tests.",
            "satedgesim": "SatEdgeSim resource-aware estimator-bound replay" ,
            "lyapunov": "Lyapunov-inspired reward shaping / queue regularizer",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check reviewer-repair quality gates.")
    parser.add_argument("--input", type=Path, required=True, help="Final smoke output directory containing stage_status.csv.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    stage_status = _load_csv(args.input / "stage_status.csv")
    gates = [
        gate_physical(),
        gate_lyapunov(),
        gate_lower_fairness(),
        gate_safe_ablation(),
        gate_mask(),
        gate_satedgesim_semantics(),
        gate_statistics(),
        gate_encoder(),
        gate_trace(),
        gate_strong_baseline(),
    ]
    stage_failures = [row for row in stage_status if row.get("status") in {"failed", "not_implemented"}]
    stage_blockers = [row for row in stage_status if row.get("status") == "blocked"]
    if stage_failures:
        gate = _gate("STAGE", "Stage execution status")
        for row in stage_failures:
            _set_failed(gate, f"{row.get('stage_id')} {row.get('stage_name')}: {row.get('exception') or row.get('blockers')}")
        gates.append(gate)
    if stage_blockers:
        gate = _gate("STAGE_BLOCKED", "Stage blocked status")
        for row in stage_blockers:
            _set_blocked(gate, f"{row.get('stage_id')} {row.get('stage_name')}: {row.get('blockers')}")
        gates.append(gate)

    counts: dict[str, int] = {}
    for gate in gates:
        counts[gate["status"]] = counts.get(gate["status"], 0) + 1
    overall = "passed" if set(counts) <= {"passed"} else "failed_or_blocked"
    quality = {
        "status": overall,
        "cpu_only": True,
        "formal_experiment_results": False,
        "stage_status_input": str(args.input / "stage_status.csv"),
        "gate_counts": counts,
        "gates": gates,
    }
    claim_guard = build_claim_guard(gates)
    _write_json(args.output_dir / "quality_gates.json", quality)
    _write_json(args.output_dir / "claim_guard.json", claim_guard)
    print(json.dumps({"status": overall, "gate_counts": counts, "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
