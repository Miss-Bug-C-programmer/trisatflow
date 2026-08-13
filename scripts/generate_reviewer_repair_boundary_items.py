"""Generate reviewer-repair boundary inventory.

The output is a claim-boundary artifact, not an experiment result. It records
what the current CPU smoke evidence supports, what remains forbidden, and what
full-scale evidence would unlock stronger wording.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEWER_ROOT = REPO_ROOT / "outputs" / "reviewer_repair"


@dataclass(frozen=True)
class BoundaryItem:
    item_id: str
    area: str
    reviewer_issues: str
    current_status: str
    evidence_files: str
    allowed_claim: str
    forbidden_claim: str
    unlock_condition: str
    paper_section: str
    severity: str
    machine_check: str


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive report path
        return {"_json_error": str(exc)}


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def build_boundary_items(repo_root: Path = REPO_ROOT) -> list[BoundaryItem]:
    reviewer_root = repo_root / "outputs" / "reviewer_repair"
    final_summary = _load_json(reviewer_root / "final_cpu_smoke" / "summary.json")
    final_guard = _load_json(reviewer_root / "final_cpu_smoke" / "claim_guard.json")
    physical = _load_json(reviewer_root / "physical_model" / "summary.json")
    lyapunov = _load_json(reviewer_root / "lyapunov_dpp" / "summary.json")
    safe_ablation = _load_json(reviewer_root / "safe_ablation" / "summary.json")
    mask_noise = _load_json(reviewer_root / "mask_noise" / "summary.json")
    satedgesim = _load_json(reviewer_root / "satedgesim_semantics" / "summary.json")
    statistics_guard = _load_json(reviewer_root / "statistics" / "claim_guard.json")
    strong = _load_json(reviewer_root / "strong_baselines" / "eval" / "strong_baseline_summary.json")
    trace_audit = _load_json(reviewer_root / "trace_stress" / "audit" / "audit_summary.json")
    trace_stress = _load_json(reviewer_root / "trace_stress" / "stress_smoke" / "stress_summary.json")

    stage_counts = final_summary.get("stage_counts") or {}
    all_smoke_passed = final_summary.get("status") == "passed" and int(stage_counts.get("passed", 0)) >= 12
    native_bound = any(row.get("native_scheduler_bound") is True for row in _flatten_dicts(satedgesim))
    completion_rows = [row for row in _flatten_dicts(satedgesim) if "completion_receipt_available" in row]
    completion_available = any(row.get("completion_receipt_available") is True for row in completion_rows)
    trace_audit_passed = trace_audit.get("audit_status") == "passed" and trace_audit.get("leakage_risk") == "none"

    return [
        BoundaryItem(
            "B00",
            "CPU smoke scope",
            "All",
            "smoke_passed_formal_pending" if all_smoke_passed else "smoke_not_clean",
            "outputs/reviewer_repair/final_cpu_smoke/summary.json; outputs/reviewer_repair/final_cpu_smoke/quality_gates.json",
            "CPU smoke validates code paths, metadata guards, and tiny update/replay checks.",
            "CPU smoke results are formal performance results or paper-ready multi-seed evidence.",
            "Run the full GPU/HPC matrix in docs/gpu_full_experiment_commands.md and regenerate statistical summaries.",
            "Global Methods / Experimental Protocol",
            "critical",
            f"formal_experiment_results={_bool_text(final_summary.get('formal_experiment_results'))}; paper_claim_ready={_bool_text(final_summary.get('paper_claim_ready'))}",
        ),
        BoundaryItem(
            "B01",
            "Physical and normalized metrics",
            "R2,R12,Q5",
            "schema_smoke_passed" if physical.get("units_present") else "schema_missing",
            "outputs/reviewer_repair/physical_model/summary.json",
            "Physical metrics may be reported with explicit unit/source/normalizer/comparable_scope.",
            "Normalized cost is directly comparable across scenario profiles or equivalent to Joules/seconds.",
            "Run E1 physical-vs-normalized ranking audit on final checkpoints and scenarios.",
            "System Model / Metrics",
            "high",
            f"legacy_mode_ok={_bool_text(physical.get('legacy_mode_ok'))}; physical_mode_ok={_bool_text(physical.get('physical_mode_ok'))}; units_present={_bool_text(physical.get('units_present'))}",
        ),
        BoundaryItem(
            "B02",
            "Lyapunov semantics",
            "R3,Q6",
            "reward_shaping_only",
            "outputs/reviewer_repair/lyapunov_dpp/summary.json",
            "Lyapunov-inspired reward shaping / queue regularizer.",
            "The policy guarantees queue stability or finite-buffer boundedness proves stability.",
            "Provide a formal theoretical DPP mode with drift upper bound, assumptions, and proof metadata.",
            "Optimization Objective / Stability Discussion",
            "critical",
            f"queue_stability_claim_allowed={_bool_text(lyapunov.get('queue_stability_claim_allowed'))}; semantics={lyapunov.get('lyapunov_semantics')}",
        ),
        BoundaryItem(
            "B03",
            "Lower allocator fairness",
            "R5,Q8",
            "fairness_controls_available",
            "outputs/reviewer_repair/lower_fairness/neutral/summary.json; outputs/reviewer_repair/lower_fairness/optimized_greedy/summary.json",
            "Rule baselines can be compared under explicit lower allocator controls.",
            "Rule-baseline gaps are solely due to upper offloading policy when lower allocator differs or is hidden.",
            "Run full Table 4b rule-upper x neutral/same_lower/optimized/oracle lower matrix.",
            "Baselines / Ablation",
            "high",
            "lower_allocator field required in every rule baseline summary",
        ),
        BoundaryItem(
            "B04",
            "Safe observable ablation",
            "R10,Q9",
            "deployable_smoke_passed" if safe_ablation.get("main_ablation_deployable") else "deployability_blocked",
            "outputs/reviewer_repair/safe_ablation/summary.json",
            "Main ablation fixes safe_observable and disables diagnostic cost-prior/oracle-cost fields.",
            "Diagnostic cost-prior variants are deployable main ablation results.",
            "Rerun Figure 10 full matrix under safe_observable across training seeds.",
            "Ablation Study",
            "high",
            f"main_ablation_deployable={_bool_text(safe_ablation.get('main_ablation_deployable'))}; uses_cost_prior={_bool_text(safe_ablation.get('uses_cost_prior'))}",
        ),
        BoundaryItem(
            "B05",
            "Mask deployability",
            "R14,Q4",
            "mask_source_metadata_available",
            "outputs/reviewer_repair/mask_noise/summary.json",
            "Predicted/measured masks are deployable modes when metadata says so; oracle_trace is an upper-bound diagnostic.",
            "oracle_trace mask is deployable or can be mixed into main deployment claims.",
            "Run E8 predicted/measured mask noise and staleness stress on full checkpoints.",
            "Action Mask / Deployment",
            "high",
            f"mask_summary_available={_bool_text(bool(mask_noise))}",
        ),
        BoundaryItem(
            "B06",
            "SatEdgeSim resource binding",
            "R7,Q1,Q12",
            "estimator_bound_not_native" if not native_bound else "native_bound_evidence_available",
            "outputs/reviewer_repair/satedgesim_semantics/summary.json; ../satedgeSimv2/docs/rl_resource_binding_audit.md",
            "SatEdgeSim resource-aware estimator-bound replay when metadata reports estimator_bound=true.",
            "SatEdgeSim validates continuous lower allocator in native VM/network/power execution when native_scheduler_bound=false.",
            "Implement and verify native scheduler binding with completion receipts and native energy/accounting evidence.",
            "SatEdgeSim Validation",
            "critical",
            f"native_scheduler_bound={_bool_text(native_bound)}; validation_mode={satedgesim.get('satedgesim_validation_mode')}",
        ),
        BoundaryItem(
            "B07",
            "Receipt/completion/energy interpretation",
            "R8,R9,Q2,Q3",
            "completion_split_guarded" if completion_rows else "completion_metadata_missing",
            "outputs/reviewer_repair/satedgesim_semantics/summary.json",
            "Report receipt_accept_ratio, scheduling_success_ratio, completion_success_ratio, and energy_source separately.",
            "intent_execution_match_ratio or receipt acceptance equals task success; unknown energy source supports energy advantage.",
            "Join real completion receipts or SimLog final task status and final cumulative energy for full replay.",
            "SatEdgeSim Metrics / Table 5",
            "critical",
            f"completion_receipt_available_any={_bool_text(completion_available)}",
        ),
        BoundaryItem(
            "B08",
            "Statistics and algorithm pair ranking",
            "R6,Q7",
            "mean_rank_reference_only" if statistics_guard.get("holm_significant_best") is False else "depends_on_full_statistics",
            "outputs/reviewer_repair/statistics/pairwise_tests.csv; outputs/reviewer_repair/statistics/claim_guard.json",
            "Use checkpoint/train_seed as the statistical unit and Holm-corrected claim wording.",
            "A method is significantly best when Holm-corrected full-seed tests do not support it.",
            "Run E6 8-10 independent training seeds and regenerate pairwise tests.",
            "Statistical Analysis / Table 3",
            "critical",
            f"claim_allowed={statistics_guard.get('claim_allowed')}; final_guard_status={final_guard.get('overall_claim_status')}",
        ),
        BoundaryItem(
            "B09",
            "Strong baselines",
            "R4,Q1",
            "tiny_update_only_full_pending",
            "outputs/reviewer_repair/strong_baselines/eval/strong_baseline_summary.json",
            "P-DQN and flat hybrid AC have real tiny CPU update smoke; grid oracle gap harness exists.",
            "The method outperforms state-of-the-art hybrid-action RL based on tiny smoke.",
            "Run E3/E5 full multi-seed strong-baseline training/evaluation and oracle-gap matrix.",
            "Baselines / Main Results",
            "critical",
            f"paper_ready_any={_bool_text(strong.get('paper_ready_any'))}; full_experiment_required={_bool_text(strong.get('full_experiment_required'))}",
        ),
        BoundaryItem(
            "B10",
            "Trace leakage and transfer",
            "R15,Q11",
            "trace_audit_passed_transfer_pending" if trace_audit_passed else "trace_audit_blocked",
            "outputs/reviewer_repair/trace_stress/manifests/manifest_build_summary.json; outputs/reviewer_repair/trace_stress/audit/audit_summary.json; outputs/reviewer_repair/trace_stress/stress_smoke/stress_summary.json",
            "The current active trace bank has complete manifest metadata and passed leakage audit.",
            "The learned policy transfers across constellation sizes before E9 16/32/64 checkpoint evaluation passes.",
            "Run E9 transfer/stress 16/32/64 with checkpoint shape verification and performance summaries.",
            "Data Governance / Transfer",
            "high",
            f"audit_status={trace_audit.get('audit_status')}; leakage_risk={trace_audit.get('leakage_risk')}; transfer_claim_supported={_bool_text(trace_stress.get('transfer_claim_supported'))}",
        ),
        BoundaryItem(
            "B11",
            "Encoder training semantics",
            "R13,Q10",
            "diagnostics_smoke_passed",
            "outputs/reviewer_repair/encoder_diagnostics/summary.json; outputs/reviewer_repair/encoder_diagnostics/gradient_report.csv",
            "Encoder mode and lower-gradient path can be stated according to diagnostics.",
            "Shared encoder is jointly trained by both levels unless shared_joint lower gradient evidence exists for the reported run.",
            "Run E12 diagnostics on final training checkpoints and cite the actual encoder_mode.",
            "Training Semantics",
            "high",
            "action collection detach and training detach must remain separate in reports",
        ),
    ]


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


def write_outputs(items: list[BoundaryItem], output_dir: Path, docs_path: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in items]
    json_path = output_dir / "boundary_items.json"
    csv_path = output_dir / "boundary_items.csv"
    summary_path = output_dir / "summary.json"
    json_path.write_text(json.dumps({"items": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    critical = [item for item in items if item.severity == "critical"]
    summary = {
        "status": "ok",
        "formal_experiment_results": False,
        "num_boundary_items": len(items),
        "critical_boundary_count": len(critical),
        "boundary_items_json": str(json_path),
        "boundary_items_csv": str(csv_path),
        "docs_path": str(docs_path),
        "paper_ready_unrestricted_claims": False,
        "warnings": [],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = render_markdown(items, summary)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        docs_path.write_text(markdown, encoding="utf-8")
    except PermissionError as exc:
        fallback = output_dir / "reviewer_repair_boundary_items.md"
        fallback.write_text(markdown, encoding="utf-8")
        summary["docs_path"] = str(fallback)
        summary["warnings"].append(f"docs_path_write_failed: {exc}")
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def render_markdown(items: list[BoundaryItem], summary: dict[str, Any]) -> str:
    lines = [
        "# Reviewer Repair Boundary Items",
        "",
        "This document records the boundary between current CPU smoke evidence and claims that require full experiments or stronger simulator evidence. It is a claim-control artifact, not a performance result.",
        "",
        "## Summary",
        "",
        f"- Boundary items: {summary['num_boundary_items']}",
        f"- Critical boundaries: {summary['critical_boundary_count']}",
        "- Formal experiment results available: false",
        "- Unrestricted paper claims allowed: false",
        "",
        "## Boundary Inventory",
        "",
        "| ID | Area | Status | Allowed Claim | Forbidden Claim | Unlock Condition |",
        "|---|---|---|---|---|---|",
    ]
    for item in items:
        lines.append(
            f"| {item.item_id} | {item.area} | {item.current_status} | {item.allowed_claim} | {item.forbidden_claim} | {item.unlock_condition} |"
        )
    lines.extend(
        [
            "",
            "## Evidence Standard",
            "",
            "- CPU smoke can validate imports, metadata guards, tiny updates, and replay semantics.",
            "- CPU smoke cannot establish formal multi-seed performance, native scheduler binding, statistical superiority, or constellation transfer.",
            "- Stronger claims are enabled only by the `unlock_condition` of the corresponding boundary item.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate reviewer-repair claim boundary inventory.")
    parser.add_argument("--output-dir", type=Path, default=REVIEWER_ROOT / "boundary_items")
    parser.add_argument("--docs-path", type=Path, default=None, help="Optional markdown output path. Defaults to output-dir.")
    args = parser.parse_args()
    items = build_boundary_items(REPO_ROOT)
    docs_path = args.docs_path or (args.output_dir / "reviewer_repair_boundary_items.md")
    summary = write_outputs(items, args.output_dir, docs_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
