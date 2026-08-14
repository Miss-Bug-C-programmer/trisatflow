"""First-class decision-plane resource accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class DecisionCostBreakdown:
    observation_bytes: int = 0
    observation_latency_sec: float = 0.0
    observation_energy: float = 0.0
    sync_bytes: int = 0
    sync_latency_sec: float = 0.0
    sync_energy: float = 0.0
    solver_wallclock_sec: float = 0.0
    solver_simulated_latency_sec: float = 0.0
    solver_compute_proxy: float = 0.0
    solver_energy_proxy: float = 0.0
    signal_bytes: int = 0
    signal_latency_sec: float = 0.0
    signal_energy: float = 0.0
    reconfiguration_bytes: int = 0
    migration_volume: float = 0.0
    num_changed_assignments: int = 0
    num_changed_resources: int = 0
    num_changed_routes: int = 0
    actual_reconfiguration_bytes: int = 0
    actual_migration_volume: float = 0.0
    obs_price: float = 1.0
    sync_price: float = 1.0
    solve_price: float = 1.0
    signal_price: float = 1.0
    reconfiguration_price: float = 1.0
    observation_byte_price: float | None = None
    observation_latency_price: float | None = None
    observation_energy_price: float | None = None
    sync_byte_price: float | None = None
    sync_latency_price: float | None = None
    sync_energy_price: float | None = None
    solve_wallclock_price: float | None = None
    solve_latency_price: float | None = None
    solve_compute_price: float | None = None
    solve_energy_price: float | None = None
    signal_byte_price: float | None = None
    signal_latency_price: float | None = None
    signal_energy_price: float | None = None
    reconfiguration_byte_price: float | None = None
    reconfiguration_volume_price: float | None = None
    reconfiguration_assignment_price: float | None = None
    reconfiguration_resource_price: float | None = None
    reconfiguration_route_price: float | None = None
    units: dict[str, str] = field(
        default_factory=lambda: {
            "observation_bytes": "byte",
            "observation_latency_sec": "sec",
            "observation_energy": "joule_proxy",
            "sync_bytes": "byte",
            "sync_latency_sec": "sec",
            "sync_energy": "joule_proxy",
            "solver_wallclock_sec": "sec",
            "solver_simulated_latency_sec": "sec",
            "solver_compute_proxy": "compute_proxy",
            "solver_energy_proxy": "joule_proxy",
            "signal_bytes": "byte",
            "signal_latency_sec": "sec",
            "signal_energy": "joule_proxy",
            "reconfiguration_bytes": "byte",
            "migration_volume": "volume_proxy",
        }
    )
    price_provenance: str = "default_unit_prices"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def obs_cost(self) -> float:
        return (
            self._price(self.observation_byte_price, self.obs_price) * float(self.observation_bytes)
            + self._price(self.observation_latency_price, self.obs_price) * self.observation_latency_sec
            + self._price(self.observation_energy_price, self.obs_price) * self.observation_energy
        )

    @property
    def sync_cost(self) -> float:
        return (
            self._price(self.sync_byte_price, self.sync_price) * float(self.sync_bytes)
            + self._price(self.sync_latency_price, self.sync_price) * self.sync_latency_sec
            + self._price(self.sync_energy_price, self.sync_price) * self.sync_energy
        )

    @property
    def solve_cost(self) -> float:
        return (
            self._price(self.solve_wallclock_price, self.solve_price) * self.solver_wallclock_sec
            + self._price(self.solve_latency_price, self.solve_price) * self.solver_simulated_latency_sec
            + self._price(self.solve_compute_price, self.solve_price) * self.solver_compute_proxy
            + self._price(self.solve_energy_price, self.solve_price) * self.solver_energy_proxy
        )

    @property
    def signal_cost(self) -> float:
        return (
            self._price(self.signal_byte_price, self.signal_price) * float(self.signal_bytes)
            + self._price(self.signal_latency_price, self.signal_price) * self.signal_latency_sec
            + self._price(self.signal_energy_price, self.signal_price) * self.signal_energy
        )

    @property
    def reconfiguration_cost(self) -> float:
        measured = (self.metadata or {}).get("reconfiguration_receipt_status") == "verified"
        bytes_value = self.actual_reconfiguration_bytes if measured else self.reconfiguration_bytes
        volume_value = self.actual_migration_volume if measured else self.migration_volume
        return (
            self._price(self.reconfiguration_byte_price, self.reconfiguration_price) * float(bytes_value)
            + self._price(self.reconfiguration_volume_price, self.reconfiguration_price) * volume_value
            + self._price(self.reconfiguration_assignment_price, self.reconfiguration_price) * self.num_changed_assignments
            + self._price(self.reconfiguration_resource_price, self.reconfiguration_price) * self.num_changed_resources
            + self._price(self.reconfiguration_route_price, self.reconfiguration_price) * self.num_changed_routes
        )

    @staticmethod
    def _price(specific: float | None, legacy: float) -> float:
        return float(legacy if specific is None else specific)

    @property
    def decision_cost(self) -> float:
        return self.obs_cost + self.sync_cost + self.solve_cost + self.signal_cost

    @property
    def intervention_cost(self) -> float:
        return self.decision_cost + self.reconfiguration_cost

    # Paper notation aliases retained alongside descriptive Python names.
    @property
    def C_obs(self) -> float:
        return self.obs_cost

    @property
    def C_sync(self) -> float:
        return self.sync_cost

    @property
    def C_solve(self) -> float:
        return self.solve_cost

    @property
    def C_signal(self) -> float:
        return self.signal_cost

    @property
    def C_recfg(self) -> float:
        return self.reconfiguration_cost

    @property
    def C_decision(self) -> float:
        return self.decision_cost

    @property
    def C_intervention(self) -> float:
        return self.intervention_cost

    def add(self, other: "DecisionCostBreakdown") -> "DecisionCostBreakdown":
        result = DecisionCostBreakdown()
        raw_fields = (
            "observation_bytes", "observation_latency_sec", "observation_energy", "sync_bytes", "sync_latency_sec", "sync_energy",
            "solver_wallclock_sec", "solver_simulated_latency_sec", "solver_compute_proxy", "solver_energy_proxy",
            "signal_bytes", "signal_latency_sec", "signal_energy", "reconfiguration_bytes", "migration_volume",
            "num_changed_assignments", "num_changed_resources", "num_changed_routes", "actual_reconfiguration_bytes", "actual_migration_volume",
        )
        for name in raw_fields:
            setattr(result, name, getattr(self, name) + getattr(other, name))
        for name in (
            "obs_price", "sync_price", "solve_price", "signal_price", "reconfiguration_price",
            "observation_byte_price", "observation_latency_price", "observation_energy_price",
            "sync_byte_price", "sync_latency_price", "sync_energy_price", "solve_wallclock_price",
            "solve_latency_price", "solve_compute_price", "solve_energy_price", "signal_byte_price",
            "signal_latency_price", "signal_energy_price", "reconfiguration_byte_price",
            "reconfiguration_volume_price", "reconfiguration_assignment_price", "reconfiguration_resource_price",
            "reconfiguration_route_price", "price_provenance",
        ):
            setattr(result, name, getattr(self, name))
        result.units = {**self.units, **other.units}
        result.metadata = {**self.metadata, **other.metadata}
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "DecisionCostBreakdown":
        if not value:
            return cls()
        aliases = {
            "C_obs": "observation_bytes",
            "C_sync": "sync_bytes",
            "C_solve": "solver_compute_proxy",
            "C_signal": "signal_bytes",
            "C_recfg": "migration_volume",
        }
        payload = {aliases.get(key, key): item for key, item in value.items()}
        valid = set(cls.__dataclass_fields__)
        return cls(**{key: item for key, item in payload.items() if key in valid})

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "obs_cost": self.obs_cost,
                "sync_cost": self.sync_cost,
                "solve_cost": self.solve_cost,
                "signal_cost": self.signal_cost,
                "decision_cost": self.decision_cost,
                "reconfiguration_cost": self.reconfiguration_cost,
                "intervention_cost": self.intervention_cost,
                "price_provenance": self.price_provenance,
                "units": dict(self.units),
            }
        )
        return payload


@dataclass
class ResourceBudgetState:
    average_decision_energy_budget: float | None = None
    average_control_bytes_budget: float | None = None
    average_decision_compute_budget: float | None = None
    dual_energy: float = 0.0
    dual_bytes: float = 0.0
    dual_compute: float = 0.0
    step_size: float = 0.01
    elapsed_physical_time_sec: float = 0.0
    last_consumption: dict[str, float] = field(default_factory=dict)

    def update(self, cost: DecisionCostBreakdown, *, holding_time_sec: float = 1.0) -> "ResourceBudgetState":
        """Dual ascent using this intervention's actual raw consumption rate."""

        duration = max(float(holding_time_sec), 1.0e-9)
        energy = float(cost.observation_energy + cost.sync_energy + cost.solver_energy_proxy + cost.signal_energy)
        measured = (cost.metadata or {}).get("reconfiguration_receipt_status") == "verified"
        reconfiguration_bytes = cost.actual_reconfiguration_bytes if measured else cost.reconfiguration_bytes
        control_bytes = float(cost.observation_bytes + cost.sync_bytes + cost.signal_bytes + reconfiguration_bytes)
        compute = float(cost.solver_compute_proxy)
        self.last_consumption = {
            "decision_energy": energy,
            "control_bytes": control_bytes,
            "decision_compute": compute,
            "duration_sec": duration,
            "energy_rate": energy / duration,
            "bytes_rate": control_bytes / duration,
            "compute_rate": compute / duration,
        }
        self.elapsed_physical_time_sec += duration
        if self.average_decision_energy_budget is not None:
            self.dual_energy = max(0.0, self.dual_energy + self.step_size * (energy / duration - self.average_decision_energy_budget))
        if self.average_control_bytes_budget is not None:
            self.dual_bytes = max(0.0, self.dual_bytes + self.step_size * (control_bytes / duration - self.average_control_bytes_budget))
        if self.average_decision_compute_budget is not None:
            self.dual_compute = max(0.0, self.dual_compute + self.step_size * (compute / duration - self.average_decision_compute_budget))
        return self


def cost_from_monitor(monitor_state: Any) -> DecisionCostBreakdown:
    acquisition = getattr(monitor_state, "acquisition", None)
    if acquisition is None:
        return DecisionCostBreakdown()
    return DecisionCostBreakdown(
        observation_bytes=int(getattr(acquisition, "obs_bytes", 0)),
        observation_latency_sec=float(getattr(acquisition, "latency_sec", 0.0)),
    )
