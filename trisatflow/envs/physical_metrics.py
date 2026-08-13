from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch

from trisatflow.envs.units import UnitScaleConfig

METRIC_SCHEMA_VERSION = "3.0"
DELAY_SEMANTICS = {
    "physical_seconds_actual",
    "physical_seconds_controlled_estimate",
    "normalized_score",
    "legacy_unknown",
}
ENERGY_CONVERSION_RULE_WH_TO_J = "wh_to_j_x3600"

METRIC_UNITS = {
    "mean_deadline_exceedance": "seconds",
    "mean_deadline_violation_ratio": "ratio",
    "mean_delay_s": "seconds",
    "p95_delay_s": "seconds",
    "mean_energy_j": "joules",
    "mean_queue_length_tasks": "tasks",
    "normalized_system_cost": "dimensionless",
}

METRIC_RECORD_SCHEMA_VERSION = "1.0"

METRIC_DESCRIPTORS: Dict[str, Dict[str, object]] = {
    "mean_delay_s": {
        "unit": "s",
        "source": "physical_delay_seconds",
        "normalizer": None,
        "comparable_scope": "same_physical_scenario_or_dimensioned_trace",
    },
    "p95_delay_s": {
        "unit": "s",
        "source": "physical_delay_seconds",
        "normalizer": None,
        "comparable_scope": "same_physical_scenario_or_dimensioned_trace",
    },
    "mean_energy_j": {
        "unit": "J",
        "source": "physical_energy_joules",
        "normalizer": None,
        "comparable_scope": "same_physical_scenario_or_dimensioned_trace",
    },
    "mean_queue_length_tasks": {
        "unit": "tasks",
        "source": "legacy_task_count_queue",
        "normalizer": None,
        "comparable_scope": "legacy_normalized_env_only",
    },
    "mean_queue_cycles": {
        "unit": "cycles",
        "source": "dimensioned_cycle_backlog",
        "normalizer": None,
        "comparable_scope": "same_physical_scenario_or_dimensioned_trace",
    },
    "normalized_system_cost": {
        "unit": "dimensionless",
        "source": "legacy_normalized_objective",
        "normalizer": {
            "delay": "deadline_threshold",
            "queue": "max_queue_or_queue_cap_cycles",
            "energy": "energy_scale",
        },
        "comparable_scope": "same_scenario_profile_only",
    },
}

DEPRECATED_METRIC_ALIASES = {
    "mean_deadline_violation": {
        "replacement": "mean_deadline_exceedance",
        "reason": "legacy field was an exceedance magnitude, not a 0/1 violation ratio",
    },
    "mean_system_cost": {
        "replacement": "normalized_system_cost",
        "reason": "legacy field mixes reward-mode-specific training cost semantics",
    },
    "mean_queue_cycles": {
        "replacement": "mean_queue_length_tasks",
        "reason": "queue state is a task-count/work-queue length, not CPU cycles",
    },
}


@dataclass(frozen=True)
class StepMetricBundle:
    """Split per-step metrics into physical / normalized / reward views."""

    physical_delay_s: torch.Tensor
    physical_energy_j: torch.Tensor
    physical_queue_cycles: torch.Tensor
    physical_queue_length_tasks: torch.Tensor
    normalized_system_cost: torch.Tensor
    reward: torch.Tensor
    legacy_trace_delay_score: torch.Tensor


def build_step_metric_bundle(
    *,
    delay_units: torch.Tensor,
    energy_units: torch.Tensor,
    queue_units: torch.Tensor,
    normalized_cost: torch.Tensor,
    reward: torch.Tensor,
    trace_delay_anomaly_mask: torch.Tensor,
    units: UnitScaleConfig,
    queue_unit: str = "tasks",
) -> StepMetricBundle:
    delay_s = delay_units * float(units.delay_s_per_unit)
    energy_j = energy_units * float(units.energy_j_per_unit)
    if str(queue_unit).strip().lower() == "cycles":
        queue_cycles = queue_units
        queue_length_tasks = torch.zeros_like(queue_units)
    else:
        queue_cycles = queue_units * float(units.queue_cycles_per_unit)
        queue_length_tasks = queue_units

    legacy_trace_delay_score = torch.zeros_like(delay_s)
    safe_delay_s = delay_s
    if trace_delay_anomaly_mask.numel() == delay_s.numel() and trace_delay_anomaly_mask.any():
        safe_delay_s = torch.where(trace_delay_anomaly_mask, torch.zeros_like(delay_s), delay_s)
        legacy_trace_delay_score = torch.where(trace_delay_anomaly_mask, delay_s, torch.zeros_like(delay_s))

    return StepMetricBundle(
        physical_delay_s=safe_delay_s,
        physical_energy_j=energy_j,
        physical_queue_cycles=queue_cycles,
        physical_queue_length_tasks=queue_length_tasks,
        normalized_system_cost=normalized_cost,
        reward=reward,
        legacy_trace_delay_score=legacy_trace_delay_score,
    )


def step_bundle_to_info(bundle: StepMetricBundle) -> Dict[str, torch.Tensor]:
    return {
        "physical_delay_s": bundle.physical_delay_s,
        "physical_energy_j": bundle.physical_energy_j,
        "physical_queue_cycles": bundle.physical_queue_cycles,
        "physical_queue_length_tasks": bundle.physical_queue_length_tasks,
        "normalized_system_cost": bundle.normalized_system_cost,
        "normalized_training_cost": bundle.normalized_system_cost,
        "reward": bundle.reward,
        "reward_mean": bundle.reward.mean(),
        "legacy_trace_delay_score": bundle.legacy_trace_delay_score,
    }


def metric_descriptor(metric: str, *, normalized_source: str = "legacy_normalized_objective") -> Dict[str, object]:
    desc = dict(METRIC_DESCRIPTORS.get(metric) or {})
    if not desc:
        desc = {
            "unit": "unknown",
            "source": "unspecified",
            "normalizer": None,
            "comparable_scope": "unspecified",
        }
    if metric == "normalized_system_cost":
        desc["unit"] = "dimensionless"
        desc["source"] = normalized_source
        desc["comparable_scope"] = "same_scenario_profile_only"
    return desc


def build_metric_record(
    metric: str,
    value: Any,
    *,
    normalized_source: str = "legacy_normalized_objective",
) -> Dict[str, object]:
    desc = metric_descriptor(metric, normalized_source=normalized_source)
    return {
        "metric": metric,
        "value": float(value),
        "unit": desc["unit"],
        "source": desc["source"],
        "normalizer": desc["normalizer"],
        "comparable_scope": desc["comparable_scope"],
    }


def build_metric_records(
    values: Mapping[str, Any],
    *,
    normalized_source: str = "legacy_normalized_objective",
) -> list[Dict[str, object]]:
    records: list[Dict[str, object]] = []
    for metric, value in values.items():
        if value in (None, ""):
            continue
        try:
            records.append(build_metric_record(metric, value, normalized_source=normalized_source))
        except (TypeError, ValueError):
            continue
    return records


def energy_delta_from_cumulative_wh(
    raw_energy_counter_wh: float,
    previous_raw_energy_counter_wh: float,
) -> Dict[str, float | str]:
    """Convert SatEdgeSim cumulative Wh counters into per-step Wh/J deltas."""

    raw = float(raw_energy_counter_wh)
    previous = float(previous_raw_energy_counter_wh)
    delta_wh = max(0.0, raw - previous)
    return {
        "raw_energy_counter_wh": raw,
        "previous_raw_energy_counter_wh": previous,
        "step_energy_delta_wh": delta_wh,
        "step_energy_delta_j": delta_wh * 3600.0,
        "energy_conversion_rule": ENERGY_CONVERSION_RULE_WH_TO_J,
    }


def normalize_delay_semantic(value: Any, *, default: str = "legacy_unknown") -> str:
    text = str(value or "").strip()
    return text if text in DELAY_SEMANTICS else default


def infer_trace_delay_semantic(row: Mapping[str, Any]) -> str:
    explicit = normalize_delay_semantic(row.get("delay_semantic"), default="")
    if explicit:
        return explicit
    semantic_class = str(row.get("trace_semantic_class") or "").strip().lower()
    queue_source = str(row.get("queue_estimate_source", row.get("queueEstimateSource", ""))).strip().lower()
    controlled = bool(row.get("is_controlled_rl_scenario", False)) or "controlled" in semantic_class or "controlled" in queue_source
    if controlled:
        return "physical_seconds_controlled_estimate"
    if str(row.get("trace_origin", "")).strip().lower() == "satedgesim":
        return "physical_seconds_actual"
    return "legacy_unknown"


def is_paper_safe_delay_semantic(delay_semantic: str) -> bool:
    return normalize_delay_semantic(delay_semantic) in {
        "physical_seconds_actual",
        "physical_seconds_controlled_estimate",
    }


def metric_schema_manifest(cfg: object) -> Dict[str, object]:
    """Return paper-facing metric schema metadata for manifests."""
    scenario = getattr(cfg, "scenario", cfg)
    reward = getattr(cfg, "reward", None)
    physical = getattr(scenario, "physical", getattr(cfg, "physical", None))
    physical_enabled = bool(getattr(physical, "enabled", False))
    normalized_source = "affine_normalized_offline_objective" if physical_enabled else "legacy_normalized_objective"
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "metric_record_schema_version": METRIC_RECORD_SCHEMA_VERSION,
        "metric_units": dict(METRIC_UNITS),
        "metric_descriptors": {
            name: metric_descriptor(name, normalized_source=normalized_source)
            for name in sorted(METRIC_DESCRIPTORS)
        },
        "delay_unit": "seconds",
        "delay_semantic": str(getattr(scenario, "delay_semantic", "physical_seconds_actual")),
        "allowed_delay_semantics": sorted(DELAY_SEMANTICS),
        "energy_unit": "joules",
        "energy_counter_unit": "Wh",
        "energy_conversion_rule": ENERGY_CONVERSION_RULE_WH_TO_J,
        "energy_counter_semantic": "cumulative_wh_requires_step_delta_conversion",
        "queue_unit": str(getattr(physical, "queue_unit", "cycles")) if physical_enabled else "tasks",
        "physical_model_enabled": physical_enabled,
        "physical_metric_mode": str(getattr(physical, "metric_mode", "normalized" if not physical_enabled else "dual")),
        "cost_normalization": {
            "normalized_system_cost_unit": "dimensionless",
            "normalized_system_cost_source": normalized_source,
            "normalized_system_cost_comparable_scope": "same_scenario_profile_only",
            "enabled": bool(getattr(reward, "cost_normalization_enabled", False)),
            "per_tier": bool(getattr(reward, "per_tier_cost_normalization", False)),
            "delay_scale": float(max(getattr(scenario, "deadline_threshold", 1.0), 1.0e-6)),
            "queue_scale": float(max(getattr(physical, "queue_cap_cycles", getattr(scenario, "max_queue", 1.0)), 1.0))
            if physical_enabled
            else float(max(getattr(scenario, "max_queue", 1.0), 1.0)),
            "energy_scale": float(max(getattr(scenario, "leo_energy_init", 1.0), 1.0)),
            "components": {
                "delay": float(getattr(reward, "delay_weight", getattr(reward, "delay", 0.0))),
                "queue": float(getattr(reward, "queue_weight", getattr(reward, "queue", 0.0))),
                "transmission": float(getattr(reward, "transmission_weight", 0.0)),
                "compute": float(getattr(reward, "compute_weight", 0.0)),
                "energy": float(getattr(reward, "energy", 0.0)),
                "feasibility": float(getattr(reward, "feasibility_weight", getattr(reward, "infeasible", 0.0))),
                "failure_risk": float(getattr(reward, "failure_penalty_weight", 0.0)),
            },
        },
        "reward_weights": {
            key: float(getattr(reward, key, 0.0))
            for key in (
                "delay",
                "energy",
                "queue",
                "violation",
                "infeasible",
                "delay_weight",
                "queue_weight",
                "transmission_weight",
                "compute_weight",
                "feasibility_weight",
            )
        },
        "trace_delay_source_semantic": (
            "trace delay fields are physical seconds only when below "
            "trace_delay_anomaly_threshold_s; larger values are exported as "
            "legacy_trace_delay_score and excluded from *_delay_s metrics"
        ),
        "trace_delay_anomaly_threshold_s": float(getattr(scenario, "trace_delay_anomaly_threshold_s", 1.0e3)),
        "deprecated_metric_aliases": dict(DEPRECATED_METRIC_ALIASES),
    }
