from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from scipy.stats import t, ttest_rel, wilcoxon  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - minimal CPU env fallback
    class _TFallback:
        @staticmethod
        def ppf(_q: float, df: int) -> float:
            table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
            return table.get(int(df), 1.96)

    class _Result:
        def __init__(self, statistic: float, pvalue: float) -> None:
            self.statistic = statistic
            self.pvalue = pvalue

    def ttest_rel(values_a, values_b):  # type: ignore
        diffs = [float(a) - float(b) for a, b in zip(values_a, values_b)]
        n = len(diffs)
        mean_diff = sum(diffs) / max(1, n)
        if n <= 1:
            return _Result(0.0, 1.0)
        std = math.sqrt(max(0.0, sum((x - mean_diff) ** 2 for x in diffs) / (n - 1)))
        if std <= 0.0:
            if abs(mean_diff) <= 1.0e-12:
                return _Result(0.0, 1.0)
            return _Result(math.copysign(float("inf"), mean_diff), 0.0)
        stat = mean_diff / (std / math.sqrt(n))
        return _Result(stat, math.erfc(abs(stat) / math.sqrt(2.0)))

    def wilcoxon(values_a, values_b, zero_method="wilcox", alternative="two-sided"):  # type: ignore
        nonzero = [float(a) - float(b) for a, b in zip(values_a, values_b) if abs(float(a) - float(b)) > 1.0e-12]
        if not nonzero:
            raise ValueError("zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.")
        positives = sum(1 for d in nonzero if d > 0)
        negatives = len(nonzero) - positives
        stat = min(positives, negatives)
        pvalue = min(1.0, 2.0 * sum(math.comb(len(nonzero), k) for k in range(0, stat + 1)) / (2 ** len(nonzero)))
        return _Result(float(stat), float(pvalue))

    t = _TFallback()

from trisatflow.analysis.statistical_tests import (
    bootstrap_ci,
    cliffs_delta,
    cluster_bootstrap_ci,
    cohen_dz_from_pairs,
    holm_bonferroni,
    paired_bootstrap_ci,
    paired_t_test,
    wilcoxon_signed_rank,
)

DEPRECATED_COMPATIBILITY_ENTRYPOINT = True

SIGNIFICANCE_FIELDNAMES = [
    "phase",
    "baseline",
    "observation_ablation",
    "metric",
    "method_a",
    "method_b",
    "algorithm_a",
    "algorithm_b",
    "test_method",
    "n_independent_train_seeds",
    "n_pairs",
    "shared_seeds",
    "mean_difference",
    "mean_diff_a_minus_b",
    "relative_difference_pct",
    "std_diff",
    "stderr_diff",
    "ci95_low",
    "ci95_high",
    "t_statistic",
    "statistic",
    "p_value_raw",
    "p_value_raw_float",
    "p_value",
    "p_value_holm",
    "p_value_holm_float",
    "cohens_dz",
    "practical_threshold_pct",
    "practically_significant",
    "significant_p_lt_0_05",
    "status",
]


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SIGNIFICANCE_FIELDNAMES)
            writer.writeheader()
        return
    keys: List[str] = [key for key in SIGNIFICANCE_FIELDNAMES if any(key in row for row in rows)]
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


def _sample_std(values: List[float], mean_v: float) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    var = sum((x - mean_v) ** 2 for x in values) / (n - 1)
    return float(math.sqrt(max(var, 0.0)))


def _student_t_ci(mean_v: float, stderr: float, n: int) -> tuple[float, float]:
    if n <= 1 or not math.isfinite(stderr) or stderr <= 0.0:
        return float(mean_v), float(mean_v)
    critical = float(t.ppf(0.975, df=n - 1))
    ci_half = critical * stderr
    return float(mean_v - ci_half), float(mean_v + ci_half)


def _format_p_value(p_value: float | None) -> str:
    if p_value is None or not math.isfinite(float(p_value)):
        return ""
    p = float(p_value)
    if p <= 0.0:
        return "<1e-300"
    if p < 1.0e-4:
        return f"{p:.3e}"
    return f"{p:.6f}"


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values in original order."""
    if not p_values:
        return []
    cleaned = [1.0 if (not math.isfinite(float(p)) or float(p) < 0.0) else min(1.0, float(p)) for p in p_values]
    order = sorted(range(len(cleaned)), key=lambda idx: cleaned[idx])
    adjusted = [1.0 for _ in cleaned]
    running_max = 0.0
    m = len(cleaned)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * cleaned[idx])
        running_max = max(running_max, candidate)
        adjusted[idx] = running_max
    return adjusted


def _paired_t_test(values_a: List[float], values_b: List[float]) -> Dict[str, float]:
    n = len(values_a)
    if n != len(values_b):
        raise ValueError("paired_t_test expects equal-length vectors")
    diffs = [a - b for a, b in zip(values_a, values_b)]
    mean_diff = sum(diffs) / max(1, n)
    if n < 2:
        return {
            "n_pairs": float(n),
            "mean_diff": float(mean_diff),
            "std_diff": 0.0,
            "stderr_diff": 0.0,
            "t_stat": 0.0,
            "p_value": 1.0,
            "cohens_dz": 0.0,
            "ci95_low": float(mean_diff),
            "ci95_high": float(mean_diff),
            "status": "insufficient_pairs",
        }

    std_diff = _sample_std(diffs, mean_diff)
    stderr = std_diff / math.sqrt(n) if n > 0 else 0.0
    if stderr <= 0.0:
        if mean_diff == 0.0:
            t_stat = 0.0
            p_value = 1.0
            cohens_dz = 0.0
        else:
            t_stat = math.copysign(float("inf"), mean_diff)
            p_value = 0.0
            cohens_dz = math.copysign(float("inf"), mean_diff)
    else:
        result = ttest_rel(values_a, values_b)
        t_stat = float(result.statistic)
        p_value = float(result.pvalue)
        cohens_dz = mean_diff / std_diff if std_diff > 0.0 else 0.0
    ci95_low, ci95_high = _student_t_ci(mean_diff, stderr, n)
    return {
        "n_pairs": float(n),
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "stderr_diff": float(stderr),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_dz": float(cohens_dz),
        "ci95_low": float(ci95_low),
        "ci95_high": float(ci95_high),
        "status": "ok",
    }


def _wilcoxon_test(values_a: List[float], values_b: List[float]) -> Dict[str, float]:
    diffs = [a - b for a, b in zip(values_a, values_b)]
    n = len(diffs)
    mean_diff = sum(diffs) / max(1, n)
    std_diff = _sample_std(diffs, mean_diff)
    stderr = std_diff / math.sqrt(max(1, n))
    ci95_low, ci95_high = _student_t_ci(mean_diff, stderr, n)

    if n < 1:
        return {
            "n_pairs": 0.0,
            "mean_diff": 0.0,
            "std_diff": 0.0,
            "stderr_diff": 0.0,
            "w_stat": 0.0,
            "p_value": 1.0,
            "cohens_dz": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "status": "insufficient_pairs",
        }

    try:
        w = wilcoxon(values_a, values_b, zero_method="wilcox", alternative="two-sided")
        p_value = float(getattr(w, "pvalue", 1.0))
        w_stat = float(getattr(w, "statistic", 0.0))
        status = "ok"
    except ValueError as exc:
        p_value = 1.0 if all(abs(x) <= 1.0e-12 for x in diffs) else 0.0
        w_stat = 0.0
        status = f"degenerate:{exc.__class__.__name__}"
    return {
        "n_pairs": float(n),
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "stderr_diff": float(stderr),
        "w_stat": w_stat,
        "p_value": p_value,
        "cohens_dz": float(mean_diff / std_diff) if std_diff > 0.0 else 0.0,
        "ci95_low": float(ci95_low),
        "ci95_high": float(ci95_high),
        "status": status,
    }


def _phase_priority(available: Iterable[str]) -> str:
    available_set = {str(x).strip().lower() for x in available}
    for phase in ("test", "val", "train"):
        if phase in available_set:
            return phase
    return "train"


def _algorithm_label(row: Dict[str, Any]) -> str:
    upper = str(row.get("upper_algo", "")).strip()
    lower = str(row.get("lower_algo", "")).strip()
    baseline = str(row.get("baseline", "")).strip()
    if upper or lower:
        return f"{upper}__{lower}"
    if baseline:
        return baseline
    return "unknown"


def _normalize_rows(raw_rows: List[Dict[str, str]], metric: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metric_candidates = _metric_candidates(metric)
    for row in raw_rows:
        value = None
        for candidate in metric_candidates:
            value = _to_float(row.get(candidate))
            if value is not None:
                break
        if value is None:
            continue
        phase = str(row.get("phase", "train") or "train").strip().lower()
        train_seed_raw = row.get("train_seed", "")
        seed_val = train_seed_raw if str(train_seed_raw).strip() else row.get("seed", "")
        try:
            seed = int(seed_val)
        except Exception:
            continue
        eval_seed = None
        eval_seed_raw = row.get("eval_seed", "")
        if str(eval_seed_raw).strip():
            try:
                eval_seed = int(eval_seed_raw)
            except Exception:
                eval_seed = None
        status = str(row.get("status", "ok") or "ok").strip().lower()
        if status not in {"ok", "success", ""}:
            continue
        rows.append(
            {
                "seed": seed,
                "train_seed": seed,
                "eval_seed": eval_seed,
                "phase": phase,
                "value": value,
                "upper_algo": str(row.get("upper_algo", "") or "").strip(),
                "lower_algo": str(row.get("lower_algo", "") or "").strip(),
                "baseline": str(row.get("baseline", "") or "").strip(),
                "observation_ablation": str(row.get("observation_ablation", "") or "").strip(),
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
        values = [float(row["value"]) for row in group_rows]
        first["value"] = float(sum(values) / len(values))
        first["eval_seed_bank"] = ",".join(str(s) for s in sorted({int(row["eval_seed"]) for row in group_rows}))
        first["n_eval_seeds"] = len(group_rows)
        first["eval_seed"] = None
        out.append(first)
    return out


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


def build_significance_rows(
    rows: List[Dict[str, Any]],
    *,
    test_method: str = "paired_t",
    metric: str = "normalized_system_cost",
    practical_threshold_pct: float = 5.0,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, Dict[int, List[float]]]] = {}
    for row in rows:
        cohort = (
            str(row.get("phase", "train")),
            str(row.get("baseline", "")),
            str(row.get("observation_ablation", "")),
        )
        algo = _algorithm_label(row)
        grouped.setdefault(cohort, {}).setdefault(algo, {}).setdefault(int(row["seed"]), []).append(float(row["value"]))

    out_rows: List[Dict[str, Any]] = []
    for (phase, baseline, ablation), algo_seed_lists in sorted(grouped.items()):
        algo_seed_values = {
            algo: {seed: sum(values) / len(values) for seed, values in seed_lists.items() if values}
            for algo, seed_lists in algo_seed_lists.items()
        }
        algos = sorted(algo_seed_values.keys())
        for i in range(len(algos)):
            for j in range(i + 1, len(algos)):
                algo_a = algos[i]
                algo_b = algos[j]
                seeds_a = algo_seed_values[algo_a]
                seeds_b = algo_seed_values[algo_b]
                shared = sorted(set(seeds_a.keys()) & set(seeds_b.keys()))
                if len(shared) < 2:
                    out_rows.append(
                        {
                            "phase": phase,
                            "baseline": baseline,
                            "observation_ablation": ablation,
                            "metric": metric,
                            "method_a": algo_a,
                            "method_b": algo_b,
                            "algorithm_a": algo_a,
                            "algorithm_b": algo_b,
                            "test_method": test_method,
                            "n_independent_train_seeds": len(shared),
                            "n_pairs": len(shared),
                            "shared_seeds": ",".join(str(s) for s in shared),
                            "status": "insufficient_pairs",
                            "mean_difference": "",
                            "relative_difference_pct": "",
                            "ci95_low": "",
                            "ci95_high": "",
                            "t_statistic": "",
                            "p_value_raw": "",
                            "p_value_holm": "",
                            "cohens_dz": "",
                            "practical_threshold_pct": float(practical_threshold_pct),
                            "practically_significant": "",
                            "significant_p_lt_0_05": "",
                        }
                    )
                    continue

                values_a = [float(seeds_a[s]) for s in shared]
                values_b = [float(seeds_b[s]) for s in shared]
                if test_method == "wilcoxon":
                    stats = _wilcoxon_test(values_a, values_b)
                else:
                    stats = _paired_t_test(values_a, values_b)
                p_value_raw = float(stats.get("p_value", 1.0))
                status = str(stats.get("status", "ok"))
                mean_b = sum(values_b) / max(1, len(values_b))
                mean_diff = float(stats.get("mean_diff", 0.0))
                if abs(mean_b) <= 1.0e-12:
                    relative_diff = 0.0 if abs(mean_diff) <= 1.0e-12 else math.copysign(float("inf"), mean_diff)
                else:
                    relative_diff = 100.0 * mean_diff / abs(mean_b)
                out_rows.append(
                    {
                        "phase": phase,
                        "baseline": baseline,
                        "observation_ablation": ablation,
                        "metric": metric,
                        "method_a": algo_a,
                        "method_b": algo_b,
                        "algorithm_a": algo_a,
                        "algorithm_b": algo_b,
                        "test_method": test_method,
                        "n_independent_train_seeds": int(stats.get("n_pairs", len(shared))),
                        "n_pairs": int(stats.get("n_pairs", len(shared))),
                        "shared_seeds": ",".join(str(s) for s in shared),
                        "mean_difference": mean_diff,
                        "mean_diff_a_minus_b": mean_diff,
                        "relative_difference_pct": float(relative_diff),
                        "std_diff": float(stats.get("std_diff", 0.0)),
                        "stderr_diff": float(stats.get("stderr_diff", 0.0)),
                        "ci95_low": float(stats.get("ci95_low", 0.0)),
                        "ci95_high": float(stats.get("ci95_high", 0.0)),
                        "t_statistic": float(stats.get("t_stat", stats.get("w_stat", 0.0))),
                        "statistic": float(stats.get("t_stat", stats.get("w_stat", 0.0))),
                        "p_value_raw": _format_p_value(p_value_raw),
                        "p_value_raw_float": p_value_raw,
                        "p_value": p_value_raw,
                        "p_value_holm": _format_p_value(p_value_raw),
                        "p_value_holm_float": p_value_raw,
                        "cohens_dz": float(stats.get("cohens_dz", 0.0)),
                        "practical_threshold_pct": float(practical_threshold_pct),
                        "practically_significant": bool(abs(relative_diff) >= float(practical_threshold_pct)),
                        "significant_p_lt_0_05": bool(status == "ok" and p_value_raw < 0.05),
                        "status": status,
                    }
                )
    _apply_holm_by_family(out_rows)
    return out_rows


def _apply_holm_by_family(rows: List[Dict[str, Any]]) -> None:
    families: Dict[Tuple[str, str, str, str], List[int]] = {}
    for idx, row in enumerate(rows):
        if str(row.get("status", "")) != "ok":
            continue
        family = (
            str(row.get("phase", "")),
            str(row.get("baseline", "")),
            str(row.get("observation_ablation", "")),
            str(row.get("metric", "")),
        )
        families.setdefault(family, []).append(idx)
    for indices in families.values():
        raw = [float(rows[idx].get("p_value_raw_float", 1.0)) for idx in indices]
        adjusted = holm_adjust(raw)
        for idx, p_holm in zip(indices, adjusted):
            rows[idx]["p_value_holm"] = _format_p_value(float(p_holm))
            rows[idx]["p_value_holm_float"] = float(p_holm)
            rows[idx]["significant_p_lt_0_05"] = bool(float(p_holm) < 0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired significance tests across algorithm groups in sweep summary.")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--summary-csv", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--metric", type=str, default="final_normalized_system_cost")
    parser.add_argument("--phase", type=str, default="auto", choices=["auto", "train", "val", "test"])
    parser.add_argument("--method", type=str, default="paired_t", choices=["paired_t", "wilcoxon"])
    parser.add_argument("--practical-threshold-pct", type=float, default=5.0)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    summary_csv = Path(args.summary_csv) if args.summary_csv else input_root / "sweep_summary.csv"
    output = Path(args.output) if args.output else input_root / "significance_tests.csv"

    raw_rows = _read_csv_rows(summary_csv)
    rows = _normalize_rows(raw_rows, args.metric)
    if not rows:
        _write_csv(output, [])
        print(f"SIGNIFICANCE_OK rows=0 output={output}")
        return

    phase = args.phase
    if phase == "auto":
        phase = _phase_priority(row["phase"] for row in rows)
    rows = [row for row in rows if row["phase"] == phase]

    out_rows = build_significance_rows(
        rows,
        test_method=args.method,
        metric=args.metric,
        practical_threshold_pct=float(args.practical_threshold_pct),
    )
    _write_csv(output, out_rows)
    print(f"SIGNIFICANCE_OK rows={len(out_rows)} phase={phase} metric={args.metric} method={args.method} output={output}")


if __name__ == "__main__":
    main()
