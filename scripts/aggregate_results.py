from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.baselines.registry import assert_no_placeholder_baselines

DEPRECATED_METRIC_ALIASES = {"mean_system_cost", "final_mean_system_cost", "tail_mean_system_cost"}


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _merge_rows_preserve_order(a: List[Dict[str, str]], b: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not a:
        return list(b)
    if not b:
        return list(a)
    out = list(a)
    out.extend(b)
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: Any) -> float | None:
    try:
        if value in (None, "", "NA"):
            return None
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _phase_priority(available: Iterable[str]) -> str:
    available_set = {str(x).strip().lower() for x in available}
    for phase in ("test", "val", "train"):
        if phase in available_set:
            return phase
    return "train"


def _safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _sample_std(values: List[float], mean_v: float) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    var = sum((v - mean_v) ** 2 for v in values) / (n - 1)
    return float(math.sqrt(max(0.0, var)))


def _stats(values: List[float]) -> Dict[str, float]:
    n = len(values)
    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "standard_error": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "n_seeds": 0.0,
        }
    mean_v = _safe_mean(values)
    std_v = _sample_std(values, mean_v)
    stderr = std_v / math.sqrt(n) if n > 0 else float("nan")
    if n > 1 and math.isfinite(stderr) and stderr > 0.0:
        try:
            from scipy.stats import t  # type: ignore

            critical = float(t.ppf(0.975, df=n - 1))
        except ModuleNotFoundError:
            critical_by_df = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
            critical = critical_by_df.get(n - 1, 1.96)
        ci_half = critical * stderr
    elif math.isfinite(stderr):
        ci_half = 0.0
    else:
        ci_half = float("nan")
    return {
        "mean": float(mean_v),
        "std": float(std_v),
        "standard_error": float(stderr),
        "ci95_low": float(mean_v - ci_half),
        "ci95_high": float(mean_v + ci_half),
        "n_seeds": float(n),
    }


def _normalize_summary_rows(raw_rows: List[Dict[str, str]], metric: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metric_candidates = _metric_candidates(metric)
    for row in raw_rows:
        status = str(row.get("status", "ok") or "ok").strip().lower()
        if status not in {"ok", "success", ""}:
            continue
        metric_value = None
        for candidate in metric_candidates:
            metric_value = _to_float(row.get(candidate))
            if metric_value is not None:
                break
        if metric_value is None:
            continue
        phase = str(row.get("phase", "train") or "train").strip().lower()
        train_seed_raw = row.get("train_seed", "")
        seed_raw = train_seed_raw if str(train_seed_raw).strip() else row.get("seed", "")
        try:
            seed = int(seed_raw)
        except Exception:
            continue
        eval_seed = None
        eval_seed_raw = row.get("eval_seed", "")
        if str(eval_seed_raw).strip():
            try:
                eval_seed = int(eval_seed_raw)
            except Exception:
                eval_seed = None
        rows.append(
            {
                "phase": phase,
                "seed": seed,
                "train_seed": seed,
                "eval_seed": eval_seed,
                "metric": float(metric_value),
                "upper_algo": str(row.get("upper_algo", "") or "").strip(),
                "lower_algo": str(row.get("lower_algo", "") or "").strip(),
                "baseline": str(row.get("baseline", "") or "").strip(),
                "observation_ablation": str(row.get("observation_ablation", "") or "").strip(),
                "checkpoint": str(row.get("selected_checkpoint", "") or row.get("checkpoint", "") or "").strip(),
                "output_dir": str(row.get("output_dir", "") or "").strip(),
                "checkpoint_selection_mode": str(row.get("checkpoint_selection_mode", "") or "").strip(),
            }
        )
    return _collapse_test_seed_bank(rows)


def _method_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("upper_algo", "")),
        str(row.get("lower_algo", "")),
        str(row.get("baseline", "")),
    )


def _collapse_test_seed_bank(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bank_by_method: Dict[Tuple[str, str, str], set[int]] = {}
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    out: List[Dict[str, Any]] = []
    for row in rows:
        is_seed_protocol_test = row.get("phase") == "test" and row.get("eval_seed") is not None
        if not is_seed_protocol_test:
            out.append(row)
            continue
        bank_by_method.setdefault(_method_key(row), set()).add(int(row["eval_seed"]))
        key = (
            row["phase"],
            row["seed"],
            row["upper_algo"],
            row["lower_algo"],
            row["baseline"],
            row["observation_ablation"],
            row.get("checkpoint_selection_mode", ""),
        )
        grouped.setdefault(key, []).append(row)

    banks = {method: tuple(sorted(bank)) for method, bank in bank_by_method.items()}
    if len(set(banks.values())) > 1:
        raise ValueError(f"inconsistent test seed bank across algorithms: {banks}")

    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        first = dict(group_rows[0])
        values = [float(row["metric"]) for row in group_rows]
        first["metric"] = float(sum(values) / len(values))
        first["eval_seed_bank"] = ",".join(str(s) for s in sorted({int(row["eval_seed"]) for row in group_rows}))
        first["n_eval_seeds"] = len(group_rows)
        first["eval_seed"] = ""
        out.append(first)
    return out


def _parse_algo_seed_from_path(path: Path) -> Tuple[str, str, int | None]:
    upper = ""
    lower = ""
    seed = None
    for part in path.parts:
        if part.startswith("seed_"):
            try:
                seed = int(part.split("seed_", 1)[1])
            except Exception:
                seed = None
        if part.startswith("upper_") and "__lower_" in part:
            m = re.match(r"upper_(.+)__lower_(.+)", part)
            if m:
                upper, lower = m.group(1), m.group(2)
    return upper, lower, seed


def _collect_rows_from_metrics(input_root: Path, metric: str) -> List[Dict[str, Any]]:
    metric_candidates = _metric_candidates(metric)
    rows: List[Dict[str, Any]] = []
    for metrics_path in sorted(input_root.rglob("metrics.csv")):
        data_rows = _read_csv_rows(metrics_path)
        if not data_rows:
            continue
        last = data_rows[-1]
        value = None
        for key in metric_candidates:
            value = _to_float(last.get(key))
            if value is not None:
                break
        if value is None:
            continue
        upper, lower, seed = _parse_algo_seed_from_path(metrics_path)
        if seed is None:
            continue
        rows.append(
            {
                "phase": "train",
                "seed": int(seed),
                "metric": float(value),
                "upper_algo": upper,
                "lower_algo": lower,
                "baseline": "",
                "observation_ablation": "",
                "checkpoint": str(metrics_path.parent / "checkpoint.pt"),
                "output_dir": str(metrics_path.parent),
            }
        )
    return rows


def _metric_candidates(metric: str) -> List[str]:
    candidates = [metric]
    if metric.startswith("final_"):
        candidates.append(metric[len("final_"):])
    if metric.startswith("tail_"):
        candidates.append(metric[len("tail_"):])
    if metric.endswith("_mean"):
        candidates.append(metric[:-len("_mean")])
    if metric == "final_normalized_system_cost":
        candidates.append("normalized_system_cost")
    return list(dict.fromkeys(candidates))


def _reject_deprecated_metric_alias(metric: str, *, allow: bool) -> None:
    if allow:
        return
    if metric in DEPRECATED_METRIC_ALIASES:
        raise SystemExit(
            f"metric {metric!r} is deprecated and ambiguous for paper tables; "
            "use 'final_normalized_system_cost' or pass --allow-deprecated-metric-alias for legacy audits"
        )


def _best_checkpoint(rows: List[Dict[str, Any]]) -> str:
    candidates = [r for r in rows if r.get("checkpoint")]
    if not candidates:
        return ""
    best = min(candidates, key=lambda r: float(r.get("metric", 1.0e18)))
    return str(best.get("checkpoint", ""))


def _aggregate(
    rows: List[Dict[str, Any]],
    *,
    metric_name: str,
    group_keys: List[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in group_keys)
        grouped.setdefault(key, []).append(row)

    out_rows: List[Dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items(), key=lambda kv: kv[0]):
        per_seed: Dict[int, float] = {}
        for row in group_rows:
            seed = int(row["seed"])
            per_seed[seed] = float(row["metric"])
        values = [float(per_seed[s]) for s in sorted(per_seed.keys())]
        stats = _stats(values)
        out = {k: v for k, v in zip(group_keys, key)}
        out.update(
            {
                "metric": metric_name,
                "n_seeds": int(stats["n_seeds"]),
                "mean": stats["mean"],
                "std": stats["std"],
                "standard_error": stats["standard_error"],
                "ci95_low": stats["ci95_low"],
                "ci95_high": stats["ci95_high"],
                "best_checkpoint": _best_checkpoint(group_rows),
            }
        )
        out_rows.append(out)
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed sweep outputs and export paper-ready summary tables.")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--summary-csv", type=str, default="")
    parser.add_argument("--oracle-csv", type=str, default="", help="Optional oracle summary CSV to merge (default: <input-root>/oracle_summary.csv).")
    parser.add_argument("--output", type=str, required=True, help="Output directory for summary tables.")
    parser.add_argument("--metric", type=str, default="final_normalized_system_cost")
    parser.add_argument("--phase", type=str, default="auto", choices=["auto", "train", "val", "test"])
    parser.add_argument("--significance-method", type=str, default="paired_t", choices=["paired_t", "wilcoxon"])
    parser.add_argument("--practical-threshold-pct", type=float, default=5.0)
    parser.add_argument("--allow-deprecated-metric-alias", action="store_true")
    args = parser.parse_args()
    _reject_deprecated_metric_alias(args.metric, allow=bool(args.allow_deprecated_metric_alias))

    input_root = Path(args.input_root)
    summary_csv = Path(args.summary_csv) if args.summary_csv else input_root / "sweep_summary.csv"
    oracle_csv = Path(args.oracle_csv) if args.oracle_csv else input_root / "oracle_summary.csv"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = _read_csv_rows(summary_csv)
    oracle_rows = _read_csv_rows(oracle_csv)
    merged_rows = _merge_rows_preserve_order(raw_rows, oracle_rows)
    if merged_rows:
        rows = _normalize_summary_rows(merged_rows, args.metric)
        source = f"summary_csv:{summary_csv};oracle_csv:{oracle_csv if oracle_rows else 'missing'}"
    else:
        rows = _collect_rows_from_metrics(input_root, args.metric)
        source = f"metrics_scan:{input_root}"

    if not rows:
        _write_csv(output_dir / "summary_by_algorithm.csv", [])
        _write_csv(output_dir / "summary_by_ablation.csv", [])
        _write_csv(output_dir / "significance_tests.csv", [])
        print(f"AGGREGATE_OK rows=0 source={source} output={output_dir}")
        return

    assert_no_placeholder_baselines(rows, context="aggregate_results")

    phase = args.phase
    if phase == "auto":
        phase = _phase_priority(row["phase"] for row in rows)
    rows = [row for row in rows if row["phase"] == phase]

    summary_by_algorithm = _aggregate(
        rows,
        metric_name=args.metric,
        group_keys=["phase", "upper_algo", "lower_algo", "baseline", "observation_ablation"],
    )
    summary_by_ablation = _aggregate(
        rows,
        metric_name=args.metric,
        group_keys=["phase", "baseline", "observation_ablation"],
    )

    significance_input = []
    for row in rows:
        significance_input.append(
            {
                "phase": row["phase"],
                "seed": row["seed"],
                "value": row["metric"],
                "upper_algo": row["upper_algo"],
                "lower_algo": row["lower_algo"],
                "baseline": row["baseline"],
                "observation_ablation": row["observation_ablation"],
            }
        )
    from statistical_tests import build_significance_rows

    significance_rows = build_significance_rows(
        significance_input,
        test_method=args.significance_method,
        metric=args.metric,
        practical_threshold_pct=float(args.practical_threshold_pct),
    )

    _write_csv(output_dir / "summary_by_algorithm.csv", summary_by_algorithm)
    _write_csv(output_dir / "summary_by_ablation.csv", summary_by_ablation)
    _write_csv(output_dir / "significance_tests.csv", significance_rows)

    print(
        f"AGGREGATE_OK rows={len(rows)} phase={phase} metric={args.metric} source={source} "
        f"algorithm_csv={output_dir / 'summary_by_algorithm.csv'} "
        f"ablation_csv={output_dir / 'summary_by_ablation.csv'} "
        f"significance_csv={output_dir / 'significance_tests.csv'}"
    )


if __name__ == "__main__":
    main()
