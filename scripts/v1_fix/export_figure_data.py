from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _to_float(v: Any) -> float | None:
    try:
        if v in (None, "", "NA"):
            return None
        value = float(v)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _group(rows: List[Dict[str, Any]], keys: Iterable[str]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    out: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[tuple(row.get(k, "NA") for k in keys)].append(row)
    return out


def _mean(vals: List[float | None]) -> float | str:
    cleaned = [float(v) for v in vals if v is not None]
    if not cleaned:
        return "NA"
    return sum(cleaned) / len(cleaned)


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
    parser = argparse.ArgumentParser(description="Export v1 figure data from summary_matrix.json")
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig_success: List[Dict[str, Any]] = []
    for (profile, baseline), g in sorted(_group(rows, ["profile", "baseline"]).items()):
        fig_success.append({
            "profile": profile,
            "baseline": baseline,
            "success_ratio": _mean([_to_float(r.get("task_success_ratio")) for r in g]),
        })

    fig_delay: List[Dict[str, Any]] = []
    for (profile, baseline), g in sorted(_group(rows, ["profile", "baseline"]).items()):
        fig_delay.append({
            "profile": profile,
            "baseline": baseline,
            "mean_delay": _mean([_to_float(r.get("mean_delay")) for r in g]),
        })

    fig_mobility: List[Dict[str, Any]] = []
    for (profile, architecture), g in sorted(_group(rows, ["profile", "architecture"]).items()):
        fig_mobility.append({
            "profile": profile,
            "architecture": architecture,
            "mobility_link_failure_ratio": _mean([_to_float(r.get("mobility_link_failure_ratio")) for r in g]),
        })

    fig_arch: List[Dict[str, Any]] = []
    for (profile, architecture, baseline), g in sorted(_group(rows, ["profile", "architecture", "baseline"]).items()):
        fig_arch.append({
            "profile": profile,
            "architecture": architecture,
            "baseline": baseline,
            "success_ratio": _mean([_to_float(r.get("task_success_ratio")) for r in g]),
            "mean_delay": _mean([_to_float(r.get("mean_delay")) for r in g]),
        })

    fig_action: List[Dict[str, Any]] = []
    for (profile, baseline), g in sorted(_group(rows, ["profile", "baseline"]).items()):
        fig_action.append({
            "profile": profile,
            "baseline": baseline,
            "local_ratio": _mean([_to_float(r.get("upper_local_ratio")) for r in g]),
            "neighbor_ratio": _mean([_to_float(r.get("upper_neighbor_ratio")) for r in g]),
            "geo_ratio": _mean([_to_float(r.get("upper_geo_ratio")) for r in g]),
            "ground_ratio": _mean([_to_float(r.get("upper_ground_ratio")) for r in g]),
            "remote_ratio": _mean([_to_float(r.get("remote_ratio")) for r in g]),
        })

    fig_regret: List[Dict[str, Any]] = []
    for (profile, baseline), g in sorted(_group(rows, ["profile", "baseline"]).items()):
        fig_regret.append({
            "profile": profile,
            "baseline": baseline,
            "normalized_regret": _mean([_to_float(r.get("normalized_regret")) for r in g]),
            "near_optimal_hit_rate_05": _mean([_to_float(r.get("near_optimal_hit_rate_05")) for r in g]),
        })

    _write_csv(outdir / "fig_success_vs_baseline.csv", fig_success)
    _write_csv(outdir / "fig_delay_vs_baseline.csv", fig_delay)
    _write_csv(outdir / "fig_mobility_failure_vs_profile.csv", fig_mobility)
    _write_csv(outdir / "fig_architecture_ablation.csv", fig_arch)
    _write_csv(outdir / "fig_action_distribution.csv", fig_action)
    _write_csv(outdir / "fig_regret_vs_baseline.csv", fig_regret)

    manifest = {
        "status": "OK",
        "files": [
            "fig_success_vs_baseline.csv",
            "fig_delay_vs_baseline.csv",
            "fig_mobility_failure_vs_profile.csv",
            "fig_architecture_ablation.csv",
            "fig_action_distribution.csv",
            "fig_regret_vs_baseline.csv",
        ],
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"V1_FIGURE_DATA_OK output_dir={outdir}")


if __name__ == "__main__":
    main()
