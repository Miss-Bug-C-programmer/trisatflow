from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List
import math

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.evaluation.metrics import unified_metrics_from_summary


def _tri_field(payload: Dict[str, Any], summary_path: Path, key: str, default: str = "NA") -> str:
    value = payload.get(key)
    if value not in (None, ""):
        return str(value)
    run_meta_path = summary_path.parent.parent / "run_metadata.json"
    if run_meta_path.exists():
        try:
            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
            tri_meta = run_meta.get("tri_mappo_maddpg", {})
            mapped = {
                "checkpoint": "tri_checkpoint_path",
                "eval_mode": "tri_eval_mode",
                "obs_normalization_mode": "tri_obs_normalization_mode",
                "action_mask_mode": "tri_action_mask_mode",
            }
            tri_value = tri_meta.get(mapped.get(key, ""))
            if tri_value not in (None, ""):
                return str(tri_value)
        except Exception:
            pass
    return default




def _finite(value: Any) -> Any:
    try:
        if value in (None, "", "NA"):
            return "NA"
        x = float(value)
        if not math.isfinite(x):
            return "NA"
        return x
    except (TypeError, ValueError):
        return "NA"


def _extract_regret_metrics(regret_path: Path, eval_mode: str = "raw_argmax") -> Dict[str, Any]:
    if not regret_path.exists():
        return {
            "regret_eval_status": "missing",
            "regret_eval_path": str(regret_path),
            "normalized_regret": "NA",
            "near_optimal_hit_rate_05": "NA",
            "near_optimal_hit_rate_10": "NA",
            "raw_argmax_oracle_agreement": "NA",
        }
    try:
        payload = json.loads(regret_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"regret_eval_status": "unreadable", "regret_eval_error": repr(exc), "regret_eval_path": str(regret_path)}
    status = str(payload.get("status") or "ok")
    if status == "skipped":
        return {"regret_eval_status": "skipped", "regret_eval_path": str(regret_path)}
    mode_results = payload.get("mode_results") or {}
    mode_payload = mode_results.get(eval_mode) or mode_results.get("raw_argmax") or {}
    overall = mode_payload.get("overall") if isinstance(mode_payload, dict) else {}
    if not isinstance(overall, dict):
        overall = {}
    return {
        "regret_eval_status": status,
        "regret_eval_path": str(regret_path),
        "regret_eval_mode": str(eval_mode),
        "normalized_regret": _finite(overall.get("mean_normalized_regret")),
        "near_optimal_hit_rate_05": _finite(overall.get("near_optimal_hit_rate_05")),
        "near_optimal_hit_rate_10": _finite(overall.get("near_optimal_hit_rate_10")),
        "raw_argmax_oracle_agreement": _finite(overall.get("selected_oracle_agreement")),
        "mean_selected_cost": _finite(overall.get("mean_selected_cost")),
        "mean_oracle_cost": _finite(overall.get("mean_oracle_cost")),
        "mean_cost_ratio": _finite(overall.get("mean_cost_ratio")),
        "regret_obs_feature_dim": payload.get("obs_feature_dim", "NA"),
        "regret_include_failure_risk": payload.get("include_failure_risk", "NA"),
    }


def _parse_token(parts: List[str], prefix: str) -> str:
    for part in parts:
        if part.startswith(prefix):
            return part[len(prefix):]
    return ""


def _checkpoint_id(payload: Dict[str, Any], summary_path: Path, seed: str) -> str:
    for key in ("checkpoint_id", "tri_checkpoint_id"):
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    checkpoint = payload.get("checkpoint") or payload.get("tri_checkpoint_path")
    if checkpoint not in (None, ""):
        return str(checkpoint)
    baseline = payload.get("baseline") or summary_path.parent.name
    return f"{baseline}::seed_{seed or 'unknown'}"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment matrix outputs into CSV/JSON.")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--allow-diagnostic-inputs", action="store_true")
    args = parser.parse_args()

    root = Path(args.input_root)
    rows: List[Dict[str, Any]] = []

    for summary_path in sorted(root.rglob("summary_compact.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        diagnostic_reasons = []
        for key in ("outputs_are_smoke_only", "tiny_results_are_not_paper_results"):
            if bool(payload.get(key, False)):
                diagnostic_reasons.append(f"{key}=true")
        if payload.get("formal_claim_allowed") is False:
            diagnostic_reasons.append("formal_claim_allowed=false")
        if payload.get("paper_ready") is False:
            diagnostic_reasons.append("paper_ready=false")
        if diagnostic_reasons and not args.allow_diagnostic_inputs:
            raise SystemExit("formal collector rejected diagnostic/smoke input: " + ", ".join(diagnostic_reasons))
        metrics = unified_metrics_from_summary(payload)
        run_dir = summary_path.parent.parent
        regret_eval_mode = str(payload.get("eval_mode") or _tri_field(payload, summary_path, "eval_mode", "raw_argmax"))
        regret_metrics = _extract_regret_metrics(run_dir / "regret_eval.json", regret_eval_mode)
        # Regret metrics are produced by scripts/evaluate_policy_regret.py, not
        # by replay summary_compact.json. Merge them here and preserve NA for
        # non-finite values such as Infinity.
        metrics.update({k: v for k, v in regret_metrics.items() if v not in (None, "")})
        rel_dir = summary_path.parent.relative_to(root)
        parts = list(rel_dir.parts)
        profile = ""
        arch = ""
        baseline = ""
        seed = ""
        for p in parts:
            if p.startswith("profile_"):
                profile = p[len("profile_"):]
            elif p.startswith("arch_"):
                arch = p[len("arch_"):]
            elif p.startswith("baseline_"):
                baseline = p[len("baseline_"):]
            elif p.startswith("seed_"):
                seed = p[len("seed_"):]
        train_seed = str(payload.get("train_seed") or seed or _parse_token(parts, "train_seed_"))
        test_seed = str(payload.get("test_seed") or _parse_token(parts, "test_seed_"))
        online_seed = str(payload.get("online_seed") or _parse_token(parts, "online_seed_"))
        checkpoint_id = _checkpoint_id(payload, summary_path, train_seed)
        statistical_unit = (
            "train_seed_checkpoint_cluster"
            if online_seed or str(payload.get("satedgesim_validation_mode", "")).strip()
            else "train_seed_checkpoint"
        )
        row = {
            "profile": profile,
            "architecture": arch,
            "baseline": baseline,
            "seed": seed,
            "method": baseline or str(payload.get("method", "")),
            "train_seed": train_seed,
            "checkpoint_id": checkpoint_id,
            "test_seed": test_seed,
            "online_seed": online_seed,
            "split": "online" if online_seed else "offline",
            "statistical_unit": statistical_unit,
            "source_file": str(summary_path),
            "summary_path": str(summary_path),
            "diagnostic_input": bool(diagnostic_reasons),
            "diagnostic_reasons": ";".join(diagnostic_reasons),
            "warnings": json.dumps(payload.get("warnings", []), ensure_ascii=False),
            "readiness": payload.get("readiness", "unknown"),
            "tri_checkpoint_path": _tri_field(payload, summary_path, "checkpoint") if baseline == "tri_mappo_maddpg" else "NA",
            "tri_eval_mode": _tri_field(payload, summary_path, "eval_mode") if baseline == "tri_mappo_maddpg" else "NA",
            "tri_obs_normalization_mode": _tri_field(payload, summary_path, "obs_normalization_mode") if baseline == "tri_mappo_maddpg" else "NA",
            "tri_action_mask_mode": _tri_field(payload, summary_path, "action_mask_mode") if baseline == "tri_mappo_maddpg" else "NA",
            **metrics,
        }
        rows.append(row)

    out_csv = Path(args.output_csv) if args.output_csv else root / "summary_matrix.csv"
    out_json = Path(args.output_json) if args.output_json else root / "summary_matrix.json"
    if args.allow_diagnostic_inputs and rows:
        out_names = f"{out_csv.name} {out_json.name}".lower()
        if any(bool(row.get("diagnostic_input", False)) for row in rows) and "diagnostic" not in out_names:
            raise SystemExit("diagnostic inputs require an output path containing 'diagnostic'")
    _write_csv(out_csv, rows)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"MATRIX_SUMMARY_OK rows={len(rows)} csv={out_csv} json={out_json}")


if __name__ == "__main__":
    main()
