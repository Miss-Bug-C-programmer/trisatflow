from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trisatflow.baselines.registry import assert_no_placeholder_baselines


def _to_float(v: Any) -> float | None:
    try:
        if v in (None, "", "NA"):
            return None
        value = float(v)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


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


def _write_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No data\n", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "NA")) for k in keys) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean_std(values: List[float | None]) -> Tuple[Any, Any]:
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return "NA", "NA"
    if len(cleaned) == 1:
        return cleaned[0], 0.0
    return mean(cleaned), pstdev(cleaned)


def _group(rows: List[Dict[str, Any]], keys: Iterable[str]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    out: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[tuple(row.get(k, "NA") for k in keys)].append(row)
    return out


def _aggregate_main(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for (profile, arch, baseline), group_rows in sorted(_group(rows, ["profile", "architecture", "baseline"]).items()):
        success_mean, success_std = _mean_std([_to_float(r.get("task_success_ratio")) for r in group_rows])
        delay_mean, delay_std = _mean_std([_to_float(r.get("mean_delay")) for r in group_rows])
        out.append(
            {
                "profile": profile,
                "architecture": arch,
                "baseline": baseline,
                "success_ratio_mean": success_mean,
                "success_ratio_std": success_std,
                "mean_delay_mean": delay_mean,
                "mean_delay_std": delay_std,
                "mobility_failure_mean": _mean_std([_to_float(r.get("mobility_link_failure_ratio")) for r in group_rows])[0],
                "deadline_failure_mean": _mean_std([_to_float(r.get("deadline_failure_ratio")) for r in group_rows])[0],
                "mean_system_cost_mean": _mean_std([_to_float(r.get("mean_system_cost")) for r in group_rows])[0],
                "normalized_regret_mean": _mean_std([_to_float(r.get("normalized_regret")) for r in group_rows])[0],
                "near_optimal_hit_rate_05_mean": _mean_std([_to_float(r.get("near_optimal_hit_rate_05")) for r in group_rows])[0],
                "load_balance_index_mean": _mean_std([_to_float(r.get("load_balance_index")) for r in group_rows])[0],
                "mean_inference_ms": _mean_std([_to_float(r.get("mean_inference_ms")) for r in group_rows])[0],
                "energy_note": "requires_manual_audit",
            }
        )
    return out


def _aggregate_simple(rows: List[Dict[str, Any]], keys: List[str], label: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for group_key, group_rows in sorted(_group(rows, keys).items()):
        row = {keys[i]: group_key[i] for i in range(len(keys))}
        row[f"{label}_success_ratio_mean"] = _mean_std([_to_float(r.get("task_success_ratio")) for r in group_rows])[0]
        row[f"{label}_mean_delay_mean"] = _mean_std([_to_float(r.get("mean_delay")) for r in group_rows])[0]
        row[f"{label}_mobility_failure_mean"] = _mean_std([_to_float(r.get("mobility_link_failure_ratio")) for r in group_rows])[0]
        out.append(row)
    return out


def _action_distribution(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for (profile, arch, baseline), group_rows in sorted(_group(rows, ["profile", "architecture", "baseline"]).items()):
        out.append(
            {
                "profile": profile,
                "architecture": arch,
                "baseline": baseline,
                "local_ratio": _mean_std([_to_float(r.get("upper_local_ratio")) for r in group_rows])[0],
                "neighbor_ratio": _mean_std([_to_float(r.get("upper_neighbor_ratio")) for r in group_rows])[0],
                "geo_ratio": _mean_std([_to_float(r.get("upper_geo_ratio")) for r in group_rows])[0],
                "ground_ratio": _mean_std([_to_float(r.get("upper_ground_ratio")) for r in group_rows])[0],
                "remote_ratio": _mean_std([_to_float(r.get("remote_ratio")) for r in group_rows])[0],
            }
        )
    return out


def _execution_reliability(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for (profile, arch, baseline), group_rows in sorted(_group(rows, ["profile", "architecture", "baseline"]).items()):
        out.append(
            {
                "profile": profile,
                "architecture": arch,
                "baseline": baseline,
                "intent_execution_match_ratio": _mean_std([_to_float(r.get("intent_execution_match_ratio")) for r in group_rows])[0],
                "receipt_accept_ratio": _mean_std([_to_float(r.get("receipt_accept_ratio")) for r in group_rows])[0],
                "fallback_none_ratio": _mean_std([_to_float(r.get("fallback_none_ratio")) for r in group_rows])[0],
                "policy_executed_diff": _mean_std([_to_float(r.get("policy_executed_ratio_diff")) for r in group_rows])[0],
                "http_timeout_count": _mean_std([_to_float(r.get("http_timeout_count")) for r in group_rows])[0],
                "http_connection_error_count": _mean_std([_to_float(r.get("http_connection_error_count")) for r in group_rows])[0],
            }
        )
    return out


def _dump(name: str, rows: List[Dict[str, Any]], outdir: Path) -> None:
    _write_csv(outdir / f"{name}.csv", rows)
    _write_md(outdir / f"{name}.md", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-ready v1 tables from summary_matrix.json")
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    summary_rows = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    assert_no_placeholder_baselines(summary_rows, context="export_paper_tables")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    table_main = _aggregate_main(summary_rows)
    table_arch = _aggregate_simple(summary_rows, ["profile", "architecture"], "arch")
    table_profile = _aggregate_simple(summary_rows, ["profile"], "profile")
    table_action = _action_distribution(summary_rows)
    table_exec = _execution_reliability(summary_rows)

    _dump("table_main_baseline_comparison", table_main, outdir)
    _dump("table_architecture_ablation", table_arch, outdir)
    _dump("table_profile_ablation", table_profile, outdir)
    _dump("table_action_distribution", table_action, outdir)
    _dump("table_execution_reliability", table_exec, outdir)

    manifest = {
        "status": "OK",
        "summary_json": args.summary_json,
        "tables": [
            "table_main_baseline_comparison",
            "table_architecture_ablation",
            "table_profile_ablation",
            "table_action_distribution",
            "table_execution_reliability",
        ],
        "energy_policy": "requires_manual_audit_not_in_main_table",
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"V1_PAPER_TABLES_OK output_dir={outdir}")


if __name__ == "__main__":
    main()
