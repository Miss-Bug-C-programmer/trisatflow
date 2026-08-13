from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.envs.obs_builder import build_shared_observation
from trisatflow.envs.obs_schema import FIELD_NAMES


VISIBLE_FIELD_MAP = {
    "local_rate": "local_visible",
    "neighbor_rate": "neighbor_visible",
    "geo_rate": "geo_visible",
    "ground_rate": "ground_visible",
    "local_delay": "local_visible",
    "neighbor_delay": "neighbor_visible",
    "geo_delay": "geo_visible",
    "ground_delay": "ground_visible",
    "local_queue": "local_visible",
    "neighbor_queue": "neighbor_visible",
    "geo_queue": "geo_visible",
    "ground_queue": "ground_visible",
    "local_normalized_cost": "local_visible",
    "neighbor_normalized_cost": "neighbor_visible",
    "geo_normalized_cost": "geo_visible",
    "ground_normalized_cost": "ground_visible",
}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _load_rows(path: str) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Check feature saturation ratio after observation normalization.")
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--normalization", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--max-saturation-ratio", type=float, default=0.10)
    args = parser.parse_args()

    norm_payload = json.loads(Path(args.normalization).read_text(encoding="utf-8"))
    norm_mode = str(norm_payload.get("mode") or "trace_p95")
    norm_stats = dict(norm_payload.get("fields") or {})

    rows = _load_rows(args.trace)
    batch = build_shared_observation(
        rows,
        source_index=0,
        node_feature_dim=20,
        normalization_mode=norm_mode,
        normalization_stats=norm_stats,
    )
    obs = batch.obs.detach().cpu()

    saturation_all: Dict[str, float] = {}
    saturation_visible_only: Dict[str, float] = {}
    warnings: List[str] = []
    for idx, field in enumerate(FIELD_NAMES[: obs.shape[1]]):
        vals = obs[:, idx]
        sat_all = float((vals >= args.threshold).float().mean().item())
        saturation_all[field] = sat_all
        vis_field = VISIBLE_FIELD_MAP.get(field)
        if vis_field is None:
            saturation_visible_only[field] = sat_all
        else:
            vis_idx = FIELD_NAMES.index(vis_field)
            vis_mask = obs[:, vis_idx] > 0.5
            if bool(vis_mask.any()):
                sat_vis = float((vals[vis_mask] >= args.threshold).float().mean().item())
            else:
                sat_vis = 0.0
            saturation_visible_only[field] = sat_vis

    for tier in ("local", "neighbor", "geo", "ground"):
        key = f"{tier}_delay"
        if saturation_visible_only.get(key, 0.0) > float(args.max_saturation_ratio):
            warnings.append(f"{key}_saturation_visible_only_gt_{args.max_saturation_ratio:.2f}")
    for tier in ("local", "neighbor", "geo", "ground"):
        key = f"{tier}_normalized_cost"
        if saturation_visible_only.get(key, 0.0) > float(args.max_saturation_ratio):
            warnings.append(f"{key}_saturation_visible_only_gt_{args.max_saturation_ratio:.2f}")

    out = {
        "trace": args.trace,
        "normalization": args.normalization,
        "normalization_mode": norm_mode,
        "num_rows": int(obs.shape[0]),
        "threshold": float(args.threshold),
        "max_saturation_ratio": float(args.max_saturation_ratio),
        "saturation_ratio_by_field_all": saturation_all,
        "saturation_ratio_by_field_visible_only": saturation_visible_only,
        "warnings": warnings,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
