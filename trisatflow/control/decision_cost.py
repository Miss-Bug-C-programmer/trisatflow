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
    obs_price: float = 1.0
    sync_price: float = 1.0
    solve_price: float = 1.0
    signal_price: float = 1.0
    reconfiguration_price: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def obs_cost(self) -> float:
        return self.obs_price * (float(self.observation_bytes) + self.observation_latency_sec + self.observation_energy)

    @property
    def sync_cost(self) -> float:
        return self.sync_price * (float(self.sync_bytes) + self.sync_latency_sec + self.sync_energy)

    @property
    def solve_cost(self) -> float:
        return self.solve_price * (
            self.solver_wallclock_sec + self.solver_simulated_latency_sec + self.solver_compute_proxy + self.solver_energy_proxy
        )

    @property
    def signal_cost(self) -> float:
        return self.signal_price * (float(self.signal_bytes) + self.signal_latency_sec + self.signal_energy)

    @property
    def reconfiguration_cost(self) -> float:
        return self.reconfiguration_price * (
            float(self.reconfiguration_bytes) + self.migration_volume + self.num_changed_assignments + self.num_changed_resources
        )

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
        for name in self.__dataclass_fields__:
            if name == "metadata":
                continue
            left = getattr(self, name)
            right = getattr(other, name)
            setattr(result, name, left + right if isinstance(left, (int, float)) else left)
        result.obs_price = self.obs_price
        result.sync_price = self.sync_price
        result.solve_price = self.solve_price
        result.signal_price = self.signal_price
        result.reconfiguration_price = self.reconfiguration_price
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

    def update(self, cost: DecisionCostBreakdown) -> "ResourceBudgetState":
        def projected(current: float, budget: float | None) -> float:
            if budget is None:
                return current
            return max(0.0, current + self.step_size * (budget - current))

        self.dual_energy = projected(self.dual_energy, self.average_decision_energy_budget)
        self.dual_bytes = projected(self.dual_bytes, self.average_control_bytes_budget)
        self.dual_compute = projected(self.dual_compute, self.average_decision_compute_budget)
        return self


def cost_from_monitor(monitor_state: Any) -> DecisionCostBreakdown:
    acquisition = getattr(monitor_state, "acquisition", None)
    if acquisition is None:
        return DecisionCostBreakdown()
    return DecisionCostBreakdown(
        observation_bytes=int(getattr(acquisition, "obs_bytes", 0)),
        observation_latency_sec=float(getattr(acquisition, "latency_sec", 0.0)),
    )
