"""Decision-plane records kept separate from data-plane utility."""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any, Iterable


class ControlMetrics:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.configuration_lifetimes: list[float] = []
        self.num_dispatches = 0
        self.num_replans = 0
        self.num_configuration_changes = 0
        self.stale_plan_rejection = 0
        self.keep_count = 0
        self._oracle_evaluation_counts = {
            "unnecessary_replanning": 0,
            "missed_intervention": 0,
            "false_safe": 0,
            "false_alarm": 0,
            "total": 0,
        }

    def record(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.records.append(dict(payload))
        action = str(payload.get("control_action", ""))
        if action == "KEEP":
            self.keep_count += 1
        if action == "INTERVENE":
            self.num_replans += 1
        if action == "DISPATCH":
            self.num_dispatches += 1
        if payload.get("configuration_changed"):
            self.num_configuration_changes += 1
        if payload.get("stale_plan_rejection"):
            self.stale_plan_rejection += 1
        # Oracle labels are accepted only through an explicit offline-evaluation
        # field.  The controller never creates this field from future truth.
        if payload.get("oracle_evaluation_only") and isinstance(payload.get("oracle_evaluation_label"), str):
            label = str(payload["oracle_evaluation_label"])
            if label in self._oracle_evaluation_counts:
                self._oracle_evaluation_counts[label] += 1
            self._oracle_evaluation_counts["total"] += 1
        return payload

    def record_oracle_evaluation(self, label: str) -> None:
        """Record an offline label without exposing it to controller logic."""

        if label not in self._oracle_evaluation_counts:
            raise ValueError(f"Unknown oracle evaluation label: {label}")
        self._oracle_evaluation_counts[label] += 1
        self._oracle_evaluation_counts["total"] += 1

    def add_lifetime(self, lifetime: float) -> None:
        if lifetime >= 0.0:
            self.configuration_lifetimes.append(float(lifetime))

    def summary(self) -> dict[str, Any]:
        lifetimes = sorted(self.configuration_lifetimes)
        physical_seconds = sum(
            max(0.0, float(record.get("actual_decision_delay_sec", 0.0) or 0.0))
            for record in self.records
        )
        physical_seconds = max(physical_seconds, max((float(record.get("physical_time_sec", 0.0)) for record in self.records), default=0.0), 1.0e-9)
        decision_energy = sum(float(record.get("decision_energy", 0.0)) for record in self.records)
        control_bytes = sum(float(record.get("control_plane_bytes", 0.0)) for record in self.records)
        decision_compute = sum(float(record.get("decision_compute", record.get("solve_cost", 0.0))) for record in self.records)
        return {
            "num_dispatches": self.num_dispatches,
            "num_replans": self.num_replans,
            "num_configuration_changes": self.num_configuration_changes,
            "stale_plan_rejection": self.stale_plan_rejection,
            "keep_ratio": self.keep_count / max(1, len(self.records)),
            "replanning_frequency": self.num_replans / max(1, len(self.records)),
            "mean_configuration_lifetime": sum(lifetimes) / len(lifetimes) if lifetimes else 0.0,
            "median_configuration_lifetime": median(lifetimes) if lifetimes else 0.0,
            "p95_configuration_lifetime": _percentile(lifetimes, 0.95),
            "scope_size_distribution": dict(Counter(int(record.get("scope_cardinality", 0)) for record in self.records)),
            "planner_distribution": dict(Counter(str(record.get("planner_name", "none")) for record in self.records)),
            "fidelity_distribution": dict(Counter(str(record.get("planner_fidelity", "none")) for record in self.records)),
            "data_plane_utility": sum(float(record.get("data_plane_utility", 0.0)) for record in self.records),
            "decision_plane_cost": sum(float(record.get("decision_cost", 0.0)) for record in self.records),
            "realized_end_to_end_utility": sum(float(record.get("end_to_end_utility", 0.0)) for record in self.records),
            "decision_energy": decision_energy,
            "control_bytes": control_bytes,
            "decision_compute": decision_compute,
            "physical_seconds_observed": physical_seconds,
            "decision_energy_per_physical_sec": decision_energy / physical_seconds,
            "control_bytes_per_physical_sec": control_bytes / physical_seconds,
            "decision_compute_per_physical_sec": decision_compute / physical_seconds,
            "replanning_rate_per_physical_sec": self.num_replans / physical_seconds,
            "total_reconfiguration_volume": sum(float(record.get("reconfiguration_volume", 0.0)) for record in self.records),
            "unnecessary_replanning_ratio": self._oracle_ratio("unnecessary_replanning"),
            "missed_intervention_ratio": self._oracle_ratio("missed_intervention"),
            "false_safe": self._oracle_evaluation_counts["false_safe"],
            "false_alarm": self._oracle_evaluation_counts["false_alarm"],
            "oracle_evaluation_only": True,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"records": list(self.records), "summary": self.summary()}

    def _oracle_ratio(self, label: str) -> float:
        total = self._oracle_evaluation_counts["total"]
        return self._oracle_evaluation_counts[label] / max(1, total)


def _percentile(values: Iterable[float], q: float) -> float:
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[index]
