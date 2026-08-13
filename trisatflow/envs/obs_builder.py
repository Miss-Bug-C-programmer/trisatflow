from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

from trisatflow.envs.obs_schema import (
    ACTION_GEO,
    ACTION_GROUND,
    ACTION_LOCAL,
    ACTION_NEIGHBOR,
    FIELD_NAMES,
    IDX_GEO_DELAY,
    IDX_GEO_QUEUE,
    IDX_GEO_RATE,
    IDX_GEO_VISIBLE,
    IDX_GROUND_DELAY,
    IDX_GROUND_QUEUE,
    IDX_GROUND_RATE,
    IDX_GROUND_VISIBLE,
    IDX_LOCAL_DELAY,
    IDX_LOCAL_QUEUE,
    IDX_LOCAL_RATE,
    IDX_LOCAL_VISIBLE,
    IDX_LOCAL_NORMALIZED_COST,
    IDX_LOCAL_COMPLETION_SAFE,
    IDX_NEIGHBOR_COMPLETION_SAFE,
    IDX_GEO_COMPLETION_SAFE,
    IDX_GROUND_COMPLETION_SAFE,
    IDX_LOCAL_MOBILITY_RISK,
    IDX_NEIGHBOR_MOBILITY_RISK,
    IDX_GEO_MOBILITY_RISK,
    IDX_GROUND_MOBILITY_RISK,
    IDX_LOCAL_LINK_LIFETIME,
    IDX_NEIGHBOR_LINK_LIFETIME,
    IDX_GEO_LINK_LIFETIME,
    IDX_GROUND_LINK_LIFETIME,
    IDX_LOCAL_LINK_MARGIN_TO_COMPLETION,
    IDX_NEIGHBOR_LINK_MARGIN_TO_COMPLETION,
    IDX_GEO_LINK_MARGIN_TO_COMPLETION,
    IDX_GROUND_LINK_MARGIN_TO_COMPLETION,
    IDX_LOCAL_HANDOVER_REQUIRED,
    IDX_NEIGHBOR_HANDOVER_REQUIRED,
    IDX_GEO_HANDOVER_REQUIRED,
    IDX_GROUND_HANDOVER_REQUIRED,
    IDX_NEIGHBOR_DELAY,
    IDX_NEIGHBOR_NORMALIZED_COST,
    IDX_NEIGHBOR_QUEUE,
    IDX_NEIGHBOR_RATE,
    IDX_NEIGHBOR_VISIBLE,
    LEGACY_NODE_FEATURE_DIM,
    SHARED_NODE_FEATURE_DIM_WITH_COST,
    SHARED_NODE_FEATURE_DIM_WITH_MOBILITY,
    SHARED_NODE_FEATURE_DIM,
    IDX_GEO_NORMALIZED_COST,
    IDX_GROUND_NORMALIZED_COST,
)

RATE_NORMALIZER = {
    ACTION_LOCAL: 1000.0,
    ACTION_NEIGHBOR: 800.0,
    ACTION_GEO: 400.0,
    ACTION_GROUND: 400.0,
}
DELAY_NORMALIZER = 0.1
QUEUE_NORMALIZER = 80.0
LINK_LIFETIME_NORMALIZER = 120.0
LINK_MARGIN_NORMALIZER = 60.0

TRACE_NORMALIZATION_MODES = {"trace_p95", "trace_log_quantile"}


def load_observation_normalization_stats(
    mode: str = "legacy",
    path: str = "",
    *,
    strict: bool = False,
) -> tuple[str, str, Dict[str, Any] | None, bool]:
    resolved_mode = str(mode or "legacy").strip().lower()
    raw_path = str(path or "").strip()
    requires_stats = resolved_mode in TRACE_NORMALIZATION_MODES
    if not requires_stats:
        return resolved_mode, raw_path, None, False

    if not raw_path:
        if strict or resolved_mode == "trace_log_quantile":
            raise FileNotFoundError(
                f"observation normalization mode '{resolved_mode}' requires obs_normalization_path, got empty path"
            )
        return resolved_mode, raw_path, None, False

    candidate_paths = []
    explicit = Path(raw_path)
    candidate_paths.append(explicit)
    if not explicit.is_absolute():
        candidate_paths.append(Path.cwd() / explicit)
        candidate_paths.append(Path(__file__).resolve().parents[2] / explicit)

    resolved_path = None
    for candidate in candidate_paths:
        if candidate.exists():
            resolved_path = candidate
            break
    if resolved_path is None:
        if strict or resolved_mode == "trace_log_quantile":
            raise FileNotFoundError(
                f"observation normalization file not found for mode '{resolved_mode}': {raw_path}"
            )
        return resolved_mode, raw_path, None, False

    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        if strict or resolved_mode == "trace_log_quantile":
            raise ValueError(
                f"observation normalization file must contain a JSON object: {resolved_path}"
            )
        return resolved_mode, str(resolved_path), None, False
    stats = dict(payload.get("fields") or payload)
    return resolved_mode, str(resolved_path), stats, True


def _norm_ref(
    field: str,
    *,
    mode: str,
    stats: Mapping[str, Any] | None,
    legacy_default: float,
) -> float:
    if mode == "trace_p95" and isinstance(stats, Mapping):
        info = stats.get(field)
        if isinstance(info, Mapping):
            p95 = _to_float(info.get("p95"), legacy_default)
            if p95 > 0.0:
                return p95
    return legacy_default


def _norm_info(field: str, *, stats: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(stats, Mapping):
        return {}
    info = stats.get(field)
    if isinstance(info, Mapping):
        return info
    return {}


def _normalize_feature(
    field: str,
    value: float,
    *,
    mode: str,
    stats: Mapping[str, Any] | None,
    legacy_default: float,
) -> float:
    x = max(0.0, float(value))
    if mode == "trace_log_quantile":
        info = _norm_info(field, stats=stats)
        p50 = max(1.0e-6, _to_float(info.get("p50"), legacy_default))
        p99 = max(p50 + 1.0e-6, _to_float(info.get("p99"), p50 * 10.0))
        scale = max(1.0e-6, _to_float(info.get("scale"), p50))
        denom = _to_float(info.get("denom"), 0.0)
        if denom <= 0.0:
            denom = math.log1p(p99 / scale)
        if denom <= 1.0e-9:
            return 0.0
        return float(max(0.0, min(1.0, math.log1p(x / scale) / denom)))

    ref = _norm_ref(field, mode=mode, stats=stats, legacy_default=legacy_default)
    return float(max(0.0, min(1.0, x / max(1.0e-6, ref))))


def _normalized_costs_from_canonical(canonical: Mapping[str, Any]) -> Dict[str, float]:
    names = ("local", "neighbor", "geo", "ground")
    cost_map: Dict[str, float] = {}
    for name in names:
        visible = bool(_to_float(canonical.get(f"{name}_visible"), 0.0) > 0.5)
        if not visible:
            cost_map[name] = 1.0
            continue
        delay = max(0.0, _to_float(canonical.get(f"{name}_delay"), 0.0))
        queue = max(0.0, _to_float(canonical.get(f"{name}_queue"), 0.0))
        rate = max(1.0e-6, _to_float(canonical.get(f"{name}_rate"), 0.0))
        tx_proxy = 0.0 if name == "local" else 1.0 / rate
        compute_proxy = max(0.0, delay - tx_proxy)
        cost_map[name] = delay + 0.5 * queue + 0.2 * tx_proxy + 0.2 * compute_proxy

    visible_costs = [cost_map[name] for name in names if bool(_to_float(canonical.get(f"{name}_visible"), 0.0) > 0.5)]
    if not visible_costs:
        return {f"{name}_normalized_cost": 1.0 for name in names}
    c_min = min(visible_costs)
    c_max = max(visible_costs)
    denom = max(1.0e-6, c_max - c_min)
    out = {}
    for name in names:
        if not bool(_to_float(canonical.get(f"{name}_visible"), 0.0) > 0.5):
            out[f"{name}_normalized_cost"] = 1.0
        else:
            out[f"{name}_normalized_cost"] = float(max(0.0, min(1.0, (cost_map[name] - c_min) / denom)))
    return out


def _raw_tier_cost(canonical: Mapping[str, Any], tier: str) -> float:
    delay = max(0.0, _to_float(canonical.get(f"{tier}_delay"), 0.0))
    queue = max(0.0, _to_float(canonical.get(f"{tier}_queue"), 0.0))
    rate = max(1.0e-6, _to_float(canonical.get(f"{tier}_rate"), 0.0))
    tx = 0.0 if tier == "local" else (1.0 / rate)
    compute = max(0.0, delay - tx)
    return float(delay + 0.5 * queue + 0.2 * tx + 0.2 * compute)


def _oracle_normalized_cost_from_row(row: Mapping[str, Any], tier: str) -> float | None:
    camel_base = tier[0].upper() + tier[1:]
    for key in (
        f"{tier}_oracle_normalized_cost",
        f"{tier}_oracle_cost",
        f"{tier}OracleNormalizedCost",
        f"{camel_base}OracleNormalizedCost",
        f"{camel_base}OracleCost",
    ):
        raw = row.get(key)
        if raw not in (None, ""):
            return max(0.0, _to_float(raw, 0.0))
    return None


@dataclass
class SharedObservationBatch:
    obs: torch.Tensor
    raw_rows: List[Dict[str, float]]
    mask: torch.Tensor
    leo_ids: List[int]
    source_index: int


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if value in (None, ""):
        return default
    return bool(value)


def _visible(row: Mapping[str, Any], prefix: str) -> float:
    camel = prefix + "Visible"
    return 1.0 if _to_bool(row.get(f"{prefix}_visible", row.get(camel)), False) else 0.0


def _rate(row: Mapping[str, Any], prefix: str) -> float:
    camel = prefix + "Rate"
    return max(0.0, _to_float(row.get(f"{prefix}_rate", row.get(camel)), 0.0))


def _delay(row: Mapping[str, Any], prefix: str) -> float | None:
    camel_base = prefix[0].upper() + prefix[1:]
    for key in (
        f"{prefix}_delay",
        f"{prefix}_best_delay",
        prefix + "Delay",
        prefix + "BestDelay",
        f"{camel_base}Delay",
        f"{camel_base}BestDelay",
    ):
        if row.get(key) not in (None, ""):
            return max(0.0, _to_float(row.get(key), 0.0))
    return None


def _queue(row: Mapping[str, Any], prefix: str) -> float | None:
    camel_base = prefix[0].upper() + prefix[1:]
    for key in (
        f"{prefix}_queue",
        f"{prefix}_best_queue",
        prefix + "Queue",
        prefix + "BestQueue",
        f"{camel_base}Queue",
        f"{camel_base}BestQueue",
    ):
        if row.get(key) not in (None, ""):
            return max(0.0, _to_float(row.get(key), 0.0))
    return None




def _completion_safe(row: Mapping[str, Any], prefix: str, visible: float) -> float:
    camel_base = prefix[0].upper() + prefix[1:]
    for key in (
        f"{prefix}_completion_safe",
        f"{prefix}_completionSafe",
        f"{prefix}CompletionSafe",
        f"{camel_base}CompletionSafe",
    ):
        if row.get(key) not in (None, ""):
            return 1.0 if _to_bool(row.get(key), False) else 0.0
    return 1.0 if visible > 0.5 else 0.0


def _mobility_risk(row: Mapping[str, Any], prefix: str, visible: float) -> float:
    camel_base = prefix[0].upper() + prefix[1:]
    for key in (
        f"{prefix}_mobility_risk",
        f"{prefix}_mobility_risk_mean",
        f"{prefix}MobilityRisk",
        f"{prefix}MobilityRiskMean",
        f"{camel_base}MobilityRisk",
        f"{camel_base}MobilityRiskMean",
    ):
        if row.get(key) not in (None, ""):
            return max(0.0, min(1.0, _to_float(row.get(key), 1.0)))
    return 0.0 if visible > 0.5 and prefix == "local" else (0.0 if visible > 0.5 else 1.0)


def _link_lifetime(row: Mapping[str, Any], prefix: str) -> float:
    camel_base = prefix[0].upper() + prefix[1:]
    for key in (
        f"{prefix}_link_lifetime_sec",
        f"{prefix}_best_link_lifetime_sec",
        f"{prefix}LinkLifetimeSec",
        f"{camel_base}LinkLifetimeSec",
        f"{camel_base}BestLinkLifetimeSec",
    ):
        if row.get(key) not in (None, ""):
            return max(0.0, _to_float(row.get(key), 0.0))
    return 0.0


def _link_margin_to_completion(row: Mapping[str, Any], prefix: str) -> float:
    camel_base = prefix[0].upper() + prefix[1:]
    for key in (
        f"{prefix}_link_survival_margin_to_completion_sec",
        f"{prefix}_best_link_survival_margin_to_completion_sec",
        f"{prefix}_link_survival_margin_sec",
        f"{prefix}_best_link_survival_margin_sec",
        f"{prefix}LinkSurvivalMarginToCompletionSec",
        f"{camel_base}LinkSurvivalMarginToCompletionSec",
        f"{camel_base}BestLinkSurvivalMarginToCompletionSec",
        f"{camel_base}BestLinkSurvivalMarginSec",
    ):
        if row.get(key) not in (None, ""):
            return _to_float(row.get(key), 0.0)
    return 0.0


def _handover_required(row: Mapping[str, Any], prefix: str) -> float:
    camel_base = prefix[0].upper() + prefix[1:]
    for key in (
        f"{prefix}_handover_required",
        f"{prefix}HandoverRequired",
        f"{camel_base}HandoverRequired",
    ):
        if row.get(key) not in (None, ""):
            return 1.0 if _to_bool(row.get(key), False) else 0.0
    return 0.0

def _impute_missing_local(remote_values: Iterable[float], *, default: float, scale: float = 1.0) -> float:
    values = [float(v) for v in remote_values if v is not None and float(v) > 0.0]
    if not values:
        return default
    values.sort()
    return max(0.0, scale * values[len(values) // 2])


def canonical_row(row: Mapping[str, Any]) -> Dict[str, float]:
    neighbor_delay = _delay(row, "neighbor")
    geo_delay = _delay(row, "geo")
    ground_delay = _delay(row, "ground")
    neighbor_queue = _queue(row, "neighbor")
    geo_queue = _queue(row, "geo")
    ground_queue = _queue(row, "ground")

    local_delay = _delay(row, "local")
    if local_delay is None:
        local_delay = _impute_missing_local([neighbor_delay, geo_delay, ground_delay], default=0.02, scale=0.85)
    local_queue = _queue(row, "local")
    if local_queue is None:
        local_queue = _impute_missing_local([neighbor_queue, geo_queue, ground_queue], default=1.0, scale=0.85)

    result = {
        "local_visible": _visible(row, "local"),
        "neighbor_visible": _visible(row, "neighbor"),
        "geo_visible": _visible(row, "geo"),
        "ground_visible": _visible(row, "ground"),
        "local_rate": _rate(row, "local"),
        "neighbor_rate": _rate(row, "neighbor"),
        "geo_rate": _rate(row, "geo"),
        "ground_rate": _rate(row, "ground"),
        "local_delay": float(local_delay),
        "neighbor_delay": float(neighbor_delay or 0.0),
        "geo_delay": float(geo_delay or 0.0),
        "ground_delay": float(ground_delay or 0.0),
        "local_queue": float(local_queue),
        "neighbor_queue": float(neighbor_queue or 0.0),
        "geo_queue": float(geo_queue or 0.0),
        "ground_queue": float(ground_queue or 0.0),
    }
    for tier in ("local", "neighbor", "geo", "ground"):
        visible = float(result[f"{tier}_visible"])
        result[f"{tier}_completion_safe"] = _completion_safe(row, tier, visible)
        result[f"{tier}_mobility_risk"] = _mobility_risk(row, tier, visible)
        result[f"{tier}_link_lifetime_sec"] = _link_lifetime(row, tier)
        result[f"{tier}_link_survival_margin_to_completion_sec"] = _link_margin_to_completion(row, tier)
        result[f"{tier}_handover_required"] = _handover_required(row, tier)
    result.update(_normalized_costs_from_canonical(result))
    return result


def build_shared_observation(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_index: int = 0,
    node_feature_dim: int = SHARED_NODE_FEATURE_DIM,
    device: torch.device | None = None,
    normalization_mode: str = "legacy",
    normalization_stats: Mapping[str, Any] | None = None,
    access_mode: str = "safe_observable",
    include_cost_prior_features: bool = True,
    include_oracle_cost: bool = False,
) -> SharedObservationBatch:
    mode = str(access_mode or "safe_observable").strip().lower()
    allow_cost_prior = bool(include_cost_prior_features)
    allow_oracle_cost = bool(include_oracle_cost) and mode == "oracle_debug"
    if mode == "safe_observable":
        if bool(include_cost_prior_features) or bool(include_oracle_cost):
            warnings.warn(
                "safe_observable disallows cost-prior/oracle observation features; privileged fields were disabled.",
                UserWarning,
                stacklevel=2,
            )
        allow_cost_prior = False
        allow_oracle_cost = False
    canonical = [canonical_row(row) for row in rows]
    obs = torch.zeros((len(canonical), SHARED_NODE_FEATURE_DIM_WITH_MOBILITY), dtype=torch.float32)
    leo_ids: List[int] = []
    for i, row in enumerate(canonical):
        leo_ids.append(int(_to_float(rows[i].get("leo_id", rows[i].get("sourceDeviceId", i)), i)))
        obs[i, IDX_LOCAL_VISIBLE] = row["local_visible"]
        obs[i, IDX_NEIGHBOR_VISIBLE] = row["neighbor_visible"]
        obs[i, IDX_GEO_VISIBLE] = row["geo_visible"]
        obs[i, IDX_GROUND_VISIBLE] = row["ground_visible"]
        obs[i, IDX_LOCAL_RATE] = _normalize_feature(
            "local_rate",
            row["local_rate"],
            mode=normalization_mode,
            stats=normalization_stats,
            legacy_default=RATE_NORMALIZER[ACTION_LOCAL],
        )
        obs[i, IDX_NEIGHBOR_RATE] = _normalize_feature(
            "neighbor_rate",
            row["neighbor_rate"],
            mode=normalization_mode,
            stats=normalization_stats,
            legacy_default=RATE_NORMALIZER[ACTION_NEIGHBOR],
        )
        obs[i, IDX_GEO_RATE] = _normalize_feature(
            "geo_rate",
            row["geo_rate"],
            mode=normalization_mode,
            stats=normalization_stats,
            legacy_default=RATE_NORMALIZER[ACTION_GEO],
        )
        obs[i, IDX_GROUND_RATE] = _normalize_feature(
            "ground_rate",
            row["ground_rate"],
            mode=normalization_mode,
            stats=normalization_stats,
            legacy_default=RATE_NORMALIZER[ACTION_GROUND],
        )
        obs[i, IDX_LOCAL_DELAY] = _normalize_feature(
            "local_delay",
            row["local_delay"],
            mode=normalization_mode,
            stats=normalization_stats,
            legacy_default=DELAY_NORMALIZER,
        )
        obs[i, IDX_NEIGHBOR_DELAY] = _normalize_feature(
            "neighbor_delay",
            row["neighbor_delay"],
            mode=normalization_mode,
            stats=normalization_stats,
            legacy_default=DELAY_NORMALIZER,
        )
        obs[i, IDX_GEO_DELAY] = _normalize_feature(
            "geo_delay",
            row["geo_delay"],
            mode=normalization_mode,
            stats=normalization_stats,
            legacy_default=DELAY_NORMALIZER,
        )
        obs[i, IDX_GROUND_DELAY] = _normalize_feature(
            "ground_delay",
            row["ground_delay"],
            mode=normalization_mode,
            stats=normalization_stats,
            legacy_default=DELAY_NORMALIZER,
        )
        obs[i, IDX_LOCAL_QUEUE] = _normalize_feature(
            "local_queue",
            row["local_queue"],
            mode=normalization_mode,
            stats=normalization_stats,
            legacy_default=QUEUE_NORMALIZER,
        )
        obs[i, IDX_NEIGHBOR_QUEUE] = _normalize_feature(
            "neighbor_queue",
            row["neighbor_queue"],
            mode=normalization_mode,
            stats=normalization_stats,
            legacy_default=QUEUE_NORMALIZER,
        )
        obs[i, IDX_GEO_QUEUE] = _normalize_feature(
            "geo_queue",
            row["geo_queue"],
            mode=normalization_mode,
            stats=normalization_stats,
            legacy_default=QUEUE_NORMALIZER,
        )
        obs[i, IDX_GROUND_QUEUE] = _normalize_feature(
            "ground_queue",
            row["ground_queue"],
            mode=normalization_mode,
            stats=normalization_stats,
            legacy_default=QUEUE_NORMALIZER,
        )
        if allow_cost_prior:
            local_cost_value = _raw_tier_cost(row, "local")
            neighbor_cost_value = _raw_tier_cost(row, "neighbor")
            geo_cost_value = _raw_tier_cost(row, "geo")
            ground_cost_value = _raw_tier_cost(row, "ground")
            if allow_oracle_cost:
                local_oracle = _oracle_normalized_cost_from_row(rows[i], "local")
                neighbor_oracle = _oracle_normalized_cost_from_row(rows[i], "neighbor")
                geo_oracle = _oracle_normalized_cost_from_row(rows[i], "geo")
                ground_oracle = _oracle_normalized_cost_from_row(rows[i], "ground")
                if local_oracle is not None:
                    local_cost_value = local_oracle
                if neighbor_oracle is not None:
                    neighbor_cost_value = neighbor_oracle
                if geo_oracle is not None:
                    geo_cost_value = geo_oracle
                if ground_oracle is not None:
                    ground_cost_value = ground_oracle
            obs[i, IDX_LOCAL_NORMALIZED_COST] = _normalize_feature(
                "local_normalized_cost",
                local_cost_value,
                mode=normalization_mode,
                stats=normalization_stats,
                legacy_default=1.0,
            )
            obs[i, IDX_NEIGHBOR_NORMALIZED_COST] = _normalize_feature(
                "neighbor_normalized_cost",
                neighbor_cost_value,
                mode=normalization_mode,
                stats=normalization_stats,
                legacy_default=1.0,
            )
            obs[i, IDX_GEO_NORMALIZED_COST] = _normalize_feature(
                "geo_normalized_cost",
                geo_cost_value,
                mode=normalization_mode,
                stats=normalization_stats,
                legacy_default=1.0,
            )
            obs[i, IDX_GROUND_NORMALIZED_COST] = _normalize_feature(
                "ground_normalized_cost",
                ground_cost_value,
                mode=normalization_mode,
                stats=normalization_stats,
                legacy_default=1.0,
            )
        else:
            obs[i, IDX_LOCAL_NORMALIZED_COST] = 0.0
            obs[i, IDX_NEIGHBOR_NORMALIZED_COST] = 0.0
            obs[i, IDX_GEO_NORMALIZED_COST] = 0.0
            obs[i, IDX_GROUND_NORMALIZED_COST] = 0.0
        obs[i, IDX_LOCAL_COMPLETION_SAFE] = row["local_completion_safe"]
        obs[i, IDX_NEIGHBOR_COMPLETION_SAFE] = row["neighbor_completion_safe"]
        obs[i, IDX_GEO_COMPLETION_SAFE] = row["geo_completion_safe"]
        obs[i, IDX_GROUND_COMPLETION_SAFE] = row["ground_completion_safe"]
        obs[i, IDX_LOCAL_MOBILITY_RISK] = max(0.0, min(1.0, row["local_mobility_risk"]))
        obs[i, IDX_NEIGHBOR_MOBILITY_RISK] = max(0.0, min(1.0, row["neighbor_mobility_risk"]))
        obs[i, IDX_GEO_MOBILITY_RISK] = max(0.0, min(1.0, row["geo_mobility_risk"]))
        obs[i, IDX_GROUND_MOBILITY_RISK] = max(0.0, min(1.0, row["ground_mobility_risk"]))
        obs[i, IDX_LOCAL_LINK_LIFETIME] = max(0.0, min(1.0, row["local_link_lifetime_sec"] / LINK_LIFETIME_NORMALIZER))
        obs[i, IDX_NEIGHBOR_LINK_LIFETIME] = max(0.0, min(1.0, row["neighbor_link_lifetime_sec"] / LINK_LIFETIME_NORMALIZER))
        obs[i, IDX_GEO_LINK_LIFETIME] = max(0.0, min(1.0, row["geo_link_lifetime_sec"] / LINK_LIFETIME_NORMALIZER))
        obs[i, IDX_GROUND_LINK_LIFETIME] = max(0.0, min(1.0, row["ground_link_lifetime_sec"] / LINK_LIFETIME_NORMALIZER))
        obs[i, IDX_LOCAL_LINK_MARGIN_TO_COMPLETION] = max(0.0, min(1.0, row["local_link_survival_margin_to_completion_sec"] / LINK_MARGIN_NORMALIZER))
        obs[i, IDX_NEIGHBOR_LINK_MARGIN_TO_COMPLETION] = max(0.0, min(1.0, row["neighbor_link_survival_margin_to_completion_sec"] / LINK_MARGIN_NORMALIZER))
        obs[i, IDX_GEO_LINK_MARGIN_TO_COMPLETION] = max(0.0, min(1.0, row["geo_link_survival_margin_to_completion_sec"] / LINK_MARGIN_NORMALIZER))
        obs[i, IDX_GROUND_LINK_MARGIN_TO_COMPLETION] = max(0.0, min(1.0, row["ground_link_survival_margin_to_completion_sec"] / LINK_MARGIN_NORMALIZER))
        obs[i, IDX_LOCAL_HANDOVER_REQUIRED] = row["local_handover_required"]
        obs[i, IDX_NEIGHBOR_HANDOVER_REQUIRED] = row["neighbor_handover_required"]
        obs[i, IDX_GEO_HANDOVER_REQUIRED] = row["geo_handover_required"]
        obs[i, IDX_GROUND_HANDOVER_REQUIRED] = row["ground_handover_required"]
    if device is not None:
        obs = obs.to(device)
    projected = project_observation(obs, target_dim=node_feature_dim)
    mask = torch.stack(
        [
            obs[:, IDX_LOCAL_VISIBLE] > 0.5,
            obs[:, IDX_NEIGHBOR_VISIBLE] > 0.5,
            obs[:, IDX_GEO_VISIBLE] > 0.5,
            obs[:, IDX_GROUND_VISIBLE] > 0.5,
        ],
        dim=-1,
    )
    if device is not None:
        mask = mask.to(device)
    return SharedObservationBatch(
        obs=projected,
        raw_rows=canonical,
        mask=mask,
        leo_ids=leo_ids,
        source_index=int(max(0, min(source_index, max(0, len(canonical) - 1)))),
    )


def project_observation(obs: torch.Tensor, *, target_dim: int) -> torch.Tensor:
    if target_dim == SHARED_NODE_FEATURE_DIM_WITH_MOBILITY:
        return obs
    if target_dim == SHARED_NODE_FEATURE_DIM_WITH_COST:
        return obs[:, :SHARED_NODE_FEATURE_DIM_WITH_COST]
    if target_dim == SHARED_NODE_FEATURE_DIM:
        return obs[:, :SHARED_NODE_FEATURE_DIM]
    if target_dim > SHARED_NODE_FEATURE_DIM and target_dim < SHARED_NODE_FEATURE_DIM_WITH_COST:
        return obs[:, :target_dim]
    if target_dim > SHARED_NODE_FEATURE_DIM_WITH_COST and target_dim < SHARED_NODE_FEATURE_DIM_WITH_MOBILITY:
        return obs[:, :target_dim]
    if target_dim == LEGACY_NODE_FEATURE_DIM:
        legacy = torch.zeros((obs.shape[0], LEGACY_NODE_FEATURE_DIM), dtype=obs.dtype, device=obs.device)
        legacy[:, 0] = obs[:, IDX_LOCAL_QUEUE]
        legacy[:, 1] = obs[:, IDX_LOCAL_DELAY]
        legacy[:, 2] = torch.ones(obs.shape[0], dtype=obs.dtype, device=obs.device)
        legacy[:, 3] = obs[:, [IDX_LOCAL_RATE, IDX_NEIGHBOR_RATE, IDX_GEO_RATE, IDX_GROUND_RATE]].mean(dim=-1)
        legacy[:, 4] = 1.0 - obs[:, [IDX_LOCAL_DELAY, IDX_NEIGHBOR_DELAY, IDX_GEO_DELAY, IDX_GROUND_DELAY]].mean(dim=-1)
        legacy[:, 5] = obs[:, IDX_LOCAL_VISIBLE]
        legacy[:, 6] = obs[:, IDX_NEIGHBOR_VISIBLE]
        legacy[:, 7] = obs[:, IDX_NEIGHBOR_RATE]
        legacy[:, 8] = obs[:, IDX_GEO_VISIBLE]
        legacy[:, 9] = obs[:, IDX_GEO_RATE]
        legacy[:, 10] = obs[:, IDX_GROUND_VISIBLE]
        legacy[:, 11] = obs[:, IDX_GROUND_RATE]
        return legacy.clamp(0.0, 1.0)
    if target_dim < SHARED_NODE_FEATURE_DIM:
        return obs[:, :target_dim]
    pad = torch.zeros((obs.shape[0], target_dim - SHARED_NODE_FEATURE_DIM_WITH_MOBILITY), dtype=obs.dtype, device=obs.device)
    return torch.cat([obs, pad], dim=-1)


def dense_rows_from_state(state: Mapping[str, Any], *, n_leo: int | None = None) -> List[Dict[str, Any]]:
    summaries = list(state.get("denseSourceSummaries") or [])
    rows: List[Dict[str, Any]] = []
    for i, summary in enumerate(summaries):
        row = dict(summary)
        row["leo_id"] = int(_to_float(summary.get("sourceDeviceId"), i))
        rows.append(row)
    rows.sort(key=lambda item: int(_to_float(item.get("leo_id"), 0.0)))
    if n_leo is not None and len(rows) >= n_leo:
        rows = rows[:n_leo]
    return rows


def ring_edge_index(n_nodes: int, *, device: torch.device | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_nodes <= 1:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 4), dtype=torch.float32, device=device)
        return edge_index, edge_attr
    src: List[int] = []
    dst: List[int] = []
    attrs: List[List[float]] = []
    for i in range(n_nodes):
        for j in ((i - 1) % n_nodes, (i + 1) % n_nodes):
            src.append(i)
            dst.append(j)
            attrs.append([1.0, 0.1, 1.0, 0.0])
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_attr = torch.tensor(attrs, dtype=torch.float32, device=device)
    return edge_index, edge_attr


def field_stats_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[float]]:
    out = {name: [] for name in FIELD_NAMES}
    for row in rows:
        canonical = canonical_row(row)
        for key, value in canonical.items():
            out[key].append(float(value))
    return out
