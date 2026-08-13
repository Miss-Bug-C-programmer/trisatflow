from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.envs.obs_builder import canonical_row


FIELDS = [
    "local_delay",
    "neighbor_delay",
    "geo_delay",
    "ground_delay",
    "local_queue",
    "neighbor_queue",
    "geo_queue",
    "ground_queue",
    "local_rate",
    "neighbor_rate",
    "geo_rate",
    "ground_rate",
    "local_normalized_cost",
    "neighbor_normalized_cost",
    "geo_normalized_cost",
    "ground_normalized_cost",
]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _q(t: torch.Tensor, p: float) -> float:
    if t.numel() == 0:
        return 0.0
    return float(torch.quantile(t, torch.tensor(p, dtype=torch.float32)))


def _normalize_linear(value: float, p95: float) -> float:
    return float(max(0.0, min(1.0, value / max(1.0e-6, p95))))


def _normalize_log_quantile(value: float, scale: float, denom: float) -> float:
    if denom <= 1.0e-9:
        return 0.0
    return float(max(0.0, min(1.0, math.log1p(max(0.0, value) / scale) / denom)))


def _fit_field(values: List[float], *, mode: str) -> Dict[str, float]:
    v = [max(0.0, float(x)) for x in values]
    t = torch.tensor(v if v else [0.0], dtype=torch.float32)
    p50 = _q(t, 0.50)
    p90 = _q(t, 0.90)
    p95 = _q(t, 0.95)
    p99 = _q(t, 0.99)
    vmax = float(t.max())
    out: Dict[str, float] = {
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "max": vmax,
    }
    if mode == "trace_log_quantile":
        scale = max(1.0e-6, p50)
        denom = math.log1p(max(scale + 1.0e-6, p99) / scale)
        sat = 0.0
        if v:
            sat = float(sum(1.0 for x in v if _normalize_log_quantile(x, scale, denom) >= 0.999) / len(v))
        out["scale"] = float(scale)
        out["denom"] = float(denom)
        out["saturation_ratio_after_normalization"] = sat
    else:
        sat = 0.0
        if v:
            sat = float(sum(1.0 for x in v if _normalize_linear(x, p95) >= 0.999) / len(v))
        out["saturation_ratio_after_normalization"] = sat
    return out


def _raw_tier_cost(row: Mapping[str, Any], tier: str) -> float:
    delay = max(0.0, _to_float(row.get(f"{tier}_delay"), 0.0))
    queue = max(0.0, _to_float(row.get(f"{tier}_queue"), 0.0))
    rate = max(1.0e-6, _to_float(row.get(f"{tier}_rate"), 0.0))
    tx = 0.0 if tier == "local" else (1.0 / rate)
    compute = max(0.0, delay - tx)
    return float(delay + 0.5 * queue + 0.2 * tx + 0.2 * compute)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit observation normalization stats from SatEdgeSim trace.")
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--mode", type=str, default="trace_p95", choices=["trace_p95", "trace_log_quantile"])
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    rows: List[Mapping[str, Any]] = []
    with Path(args.trace).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    canonical_rows = [canonical_row(row) for row in rows]
    field_values: Dict[str, List[float]] = {field: [] for field in FIELDS}
    for row in canonical_rows:
        for field in FIELDS:
            if field.endswith("_normalized_cost"):
                tier = field.replace("_normalized_cost", "")
                field_values[field].append(_raw_tier_cost(row, tier))
            else:
                field_values[field].append(_to_float(row.get(field), 0.0))

    stats = {field: _fit_field(vals, mode=args.mode) for field, vals in field_values.items()}
    payload = {
        "trace": args.trace,
        "mode": args.mode,
        "num_rows": len(canonical_rows),
        "fields": stats,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
