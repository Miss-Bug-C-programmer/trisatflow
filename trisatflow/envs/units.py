from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch

# Physical fields must carry explicit unit suffixes.
PHYSICAL_SUFFIXES = ("_s", "_sec", "_ms", "_mbps", "_ghz", "_j", "_bits", "_cycles", "_w")
# Normalized/training fields should never pretend to be physical units.
NORMALIZED_TOKEN_CANDIDATES = ("normalized", "norm", "reward", "cost", "score")


@dataclass(frozen=True)
class UnitScaleConfig:
    """Lightweight conversion factors from internal env units to physical units.

    Defaults are 1.0 to preserve previous numerical behavior.
    """

    delay_s_per_unit: float = 1.0
    energy_j_per_unit: float = 1.0
    queue_cycles_per_unit: float = 1.0
    cpu_ghz_per_unit: float = 1.0
    rate_mbps_per_unit: float = 1.0
    bandwidth_mbps_per_unit: float = 1.0
    power_w_per_unit: float = 1.0
    task_size_bits_per_unit: float = 1.0
    workload_cycles_per_unit: float = 1.0


@dataclass(frozen=True)
class TraceDelayInterpretation:
    """Policy for trace delay safety checks.

    If delay is unusually large for physical seconds, we treat it as a legacy
    normalized trace score and keep it out of physical-delay reporting.
    """

    anomaly_threshold_s: float = 1.0e3
    treat_anomaly_as_legacy_score: bool = True


def has_physical_unit_suffix(field_name: str) -> bool:
    name = str(field_name or "").strip().lower()
    return any(name.endswith(suffix) for suffix in PHYSICAL_SUFFIXES)


def is_normalized_or_training_field(field_name: str) -> bool:
    name = str(field_name or "").strip().lower()
    return any(token in name for token in NORMALIZED_TOKEN_CANDIDATES)


def is_misnamed_normalized_field(field_name: str) -> bool:
    """Normalized/training field should not be named like physical unit."""
    return is_normalized_or_training_field(field_name) and has_physical_unit_suffix(field_name)


def validate_metric_field_names(fields: Sequence[str]) -> list[str]:
    violations: list[str] = []
    for field in fields:
        if is_misnamed_normalized_field(field):
            violations.append(str(field))
    return violations


def detect_trace_delay_anomaly(
    delay_values_s: torch.Tensor,
    *,
    interpretation: TraceDelayInterpretation,
) -> torch.Tensor:
    """Return bool mask where delay looks too large for physical seconds."""
    if delay_values_s.numel() == 0:
        return torch.zeros_like(delay_values_s, dtype=torch.bool)
    threshold = max(1.0e-6, float(interpretation.anomaly_threshold_s))
    return delay_values_s > threshold


def classify_legacy_trace_delay_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    interpretation: TraceDelayInterpretation,
) -> int:
    """Count rows whose delay appears to be normalized/legacy score."""
    anomaly_count = 0
    threshold = max(1.0e-6, float(interpretation.anomaly_threshold_s))
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for key in (
            "local_delay",
            "neighbor_delay",
            "geo_delay",
            "ground_delay",
            "local_total_delay",
            "neighbor_total_delay",
            "geo_total_delay",
            "ground_total_delay",
        ):
            value = row.get(key)
            try:
                x = float(value)
            except (TypeError, ValueError):
                continue
            if x > threshold:
                anomaly_count += 1
                break
    return anomaly_count
