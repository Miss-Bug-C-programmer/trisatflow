from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.reporting.input_validation import ReportingInputError, load_reporting_input, write_csv


TABLE_SPECS = {
    "table_ii_hierarchical_rl_component_selection": "Table II: hierarchical RL component selection",
    "table_iii_best_rl_vs_rule_based_baselines": "Table III: best RL vs rule-based baselines",
    "table_iv_learning_based_and_literature_baselines": "Table IV: learning-based and literature baselines",
    "table_s1_complete_raw_seed_level_results": "Table S1: complete raw seed-level results",
    "table_s2_statistical_tests_holm": "Table S2: statistical tests with Holm correction",
}

LEARNING_BASELINES = {"flat_ppo", "flat_mappo", "hierarchical_no_gnn"}
RULE_TYPES = {"static", "heuristic", "optimization"}


def _method_type(row: Dict[str, Any]) -> str:
    baseline = str(row.get("baseline", "") or "").strip()
    if baseline in LEARNING_BASELINES:
        return "learning"
    if baseline:
        try:
            from trisatflow.baselines.registry import baseline_metadata

            meta = baseline_metadata(baseline)
            return str(meta.type)
        except Exception:
            return "baseline"
    if str(row.get("method", "")).startswith("flat_"):
        return "learning"
    return "rl"


def _group_by_method(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("method", "unknown"))].append(row)
    return grouped


def _stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "mean": "", "std": ""}
    return {
        "n": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def _summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for method, group in sorted(_group_by_method(rows).items()):
        values = [float(row["metric_value"]) for row in group]
        delays = [_to_float(row.get("mean_delay_s")) for row in group]
        energies = [_to_float(row.get("mean_energy_j")) for row in group]
        remote = [_to_float(row.get("upper_remote_ratio")) for row in group]
        cost_stats = _stats(values)
        out.append(
            {
                "method": method,
                "method_type": _method_type(group[0]),
                "phase_set": ",".join(sorted({str(row.get("phase", "")) for row in group})),
                "n_rows": len(group),
                "n_independent_train_seeds": len({int(row.get("train_seed", 0)) for row in group}),
                "normalized_system_cost_mean": cost_stats["mean"],
                "normalized_system_cost_std": cost_stats["std"],
                "mean_delay_s": _mean_clean(delays),
                "mean_energy_j": _mean_clean(energies),
                "upper_remote_ratio": _mean_clean(remote),
            }
        )
    return out


def _component_selection(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = [row for row in _summary_rows(rows) if row["method_type"] == "rl" or "gnn" in row["method"]]
    return candidates or _summary_rows(rows)


def _best_rl_vs_rules(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = _summary_rows(rows)
    rl = [row for row in summaries if row["method_type"] in {"rl", "learning"}]
    rules = [row for row in summaries if row["method_type"] in RULE_TYPES]
    selected: List[Dict[str, Any]] = []
    if rl:
        selected.append(min(rl, key=lambda row: _sort_metric(row["normalized_system_cost_mean"])))
    selected.extend(rules)
    return selected or summaries


def _learning_and_literature(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = _summary_rows(rows)
    selected = [row for row in summaries if row["method"] in LEARNING_BASELINES or row["method_type"] == "learning"]
    return selected or summaries


def _raw_seed_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = [
        "method",
        "phase",
        "train_seed",
        "eval_seed",
        "eval_seed_bank",
        "metric_value",
        "mean_delay_s",
        "mean_energy_j",
        "upper_local_ratio",
        "upper_neighbor_ratio",
        "upper_geo_ratio",
        "upper_ground_ratio",
        "experiment_contract_sha256",
        "metric_schema_version",
    ]
    return [{key: row.get(key, "") for key in keys} for row in rows]


def _stat_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    wanted = [
        "phase",
        "method_a",
        "method_b",
        "metric",
        "n_independent_train_seeds",
        "mean_difference",
        "relative_difference_pct",
        "p_value_raw",
        "p_value_holm",
        "status",
    ]
    return [{key: row.get(key, "") for key in wanted} for row in rows]


def _write_latex_table(path: Path, rows: List[Dict[str, Any]], *, caption: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("% No data\n", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{_tex(caption)}}}",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\hline",
        " & ".join(_tex(col) for col in columns) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_tex(_format_value(row.get(col, ""))) for col in columns) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_table_pair(output_dir: Path, name: str, rows: List[Dict[str, Any]]) -> Dict[str, str]:
    csv_path = output_dir / f"{name}.csv"
    tex_path = output_dir / f"{name}.tex"
    write_csv(csv_path, rows)
    _write_latex_table(tex_path, rows, caption=TABLE_SPECS[name])
    return {"csv": str(csv_path), "tex": str(tex_path)}


def _to_float(value: Any) -> float | None:
    try:
        if value in (None, "", "NA"):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean_clean(values: List[float | None]) -> Any:
    clean = [float(v) for v in values if v is not None]
    return mean(clean) if clean else ""


def _sort_metric(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("inf")


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def _tex(value: Any) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export audited paper-ready CSV and LaTeX tables.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-smoke-small-n", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument(
        "--primary-semantic-class",
        default="",
        help="Require every reporting row to come from this trace_semantic_class.",
    )
    args = parser.parse_args()

    try:
        report = load_reporting_input(
            args.input_root,
            allow_smoke_small_n=bool(args.allow_smoke_small_n),
            formal=bool(args.formal),
            primary_semantic_class=str(args.primary_semantic_class),
        )
    except ReportingInputError as exc:
        raise SystemExit(f"export_paper_tables input validation failed: {exc}") from exc

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "table_ii_hierarchical_rl_component_selection": _write_table_pair(
            out,
            "table_ii_hierarchical_rl_component_selection",
            _component_selection(report.rows),
        ),
        "table_iii_best_rl_vs_rule_based_baselines": _write_table_pair(
            out,
            "table_iii_best_rl_vs_rule_based_baselines",
            _best_rl_vs_rules(report.rows),
        ),
        "table_iv_learning_based_and_literature_baselines": _write_table_pair(
            out,
            "table_iv_learning_based_and_literature_baselines",
            _learning_and_literature(report.rows),
        ),
        "table_s1_complete_raw_seed_level_results": _write_table_pair(
            out,
            "table_s1_complete_raw_seed_level_results",
            _raw_seed_rows(report.rows),
        ),
        "table_s2_statistical_tests_holm": _write_table_pair(
            out,
            "table_s2_statistical_tests_holm",
            _stat_rows(report.significance_rows),
        ),
    }
    manifest = {
        "status": "ok",
        "input_root": str(report.input_root),
        "contract_sha256": report.contract_sha256,
        "metric_schema_version": report.metric_schema_version,
        "primary_semantic_class": str(args.primary_semantic_class),
        "smoke_mode": bool(report.smoke_mode),
        "tables": artifacts,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"PAPER_TABLES_OK output_dir={out} tables={len(artifacts)}")


if __name__ == "__main__":
    main()
