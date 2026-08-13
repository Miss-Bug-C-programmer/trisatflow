from __future__ import annotations

import random
from typing import Any, Dict, Mapping, Sequence

from trisatflow.baselines.registry import ACTION_NAMES, _cost_rank, _to_float


GRID_LOW = (0.25, 0.5, 0.75, 1.0)
GRID_HIGH = tuple(round(0.1 * idx, 1) for idx in range(1, 11))


class OptimizedLyapunovDppPolicy:
    """Finite-grid Lyapunov-inspired DPP baseline.

    This is an empirical optimization baseline. It estimates one-slot
    drift-plus-penalty over feasible abstract actions and resource shares, but
    does not implement a theoretical drift upper bound or stability proof.
    """

    name = "optimized_lyapunov_dpp"

    def __init__(
        self,
        *,
        dpp_v: float = 1.0,
        grid_mode: str = "grid_low",
        delay_weight: float = 1.0,
        energy_weight: float = 0.05,
        violation_weight: float = 1.0,
        queue_weight: float = 1.0,
        risk_weight: float = 0.2,
        fallback_policy: str = "cost_greedy",
    ) -> None:
        self.dpp_v = float(max(0.0, dpp_v))
        self.grid_mode = str(grid_mode or "grid_low").strip().lower()
        self.grid = GRID_HIGH if self.grid_mode == "grid_high" else GRID_LOW
        self.delay_weight = float(max(0.0, delay_weight))
        self.energy_weight = float(max(0.0, energy_weight))
        self.violation_weight = float(max(0.0, violation_weight))
        self.queue_weight = float(max(0.0, queue_weight))
        self.risk_weight = float(max(0.0, risk_weight))
        self.fallback_policy = str(fallback_policy)

    def select_action(
        self,
        obs: Any,
        state: Mapping[str, Any],
        mask: Sequence[int],
        candidate_info: Mapping[int, Mapping[str, Any]],
        rng: random.Random,
    ) -> Dict[str, Any]:
        del obs, rng
        feasible_actions = [
            int(action)
            for action in range(len(mask))
            if action < 4 and bool(mask[action]) and bool(candidate_info.get(action, {}).get("is_available", True))
        ]
        if not feasible_actions:
            feasible_actions = [0]

        best: tuple[float, int, list[float], dict[str, float]] | None = None
        for action in feasible_actions:
            info = candidate_info.get(action, {})
            for bw_share in self.grid:
                for tx_power_ratio in self.grid:
                    for cpu_share in self.grid:
                        objective, components = self._objective(
                            action=action,
                            state=state,
                            info=info,
                            cpu_share=float(cpu_share),
                            bw_share=float(bw_share),
                            tx_power_ratio=float(tx_power_ratio),
                        )
                        lower_action = [float(cpu_share), float(bw_share), float(tx_power_ratio)]
                        item = (objective, action, lower_action, components)
                        if best is None or (item[0], item[1], item[2]) < (best[0], best[1], best[2]):
                            best = item

        assert best is not None
        objective, selected_action, lower_action, components = best
        info = dict(candidate_info.get(selected_action) or {})
        decision_info: Dict[str, Any] = {
            "baseline_name": self.name,
            "requested_action": int(selected_action),
            "selected_action": int(selected_action),
            "selected_tier": ACTION_NAMES[selected_action],
            "target_tier": ACTION_NAMES[selected_action],
            "selection_reason": "optimized_lyapunov_dpp_grid_search",
            "fallback_used": False,
            "fallback_reason": "none",
            "cost_rank": _cost_rank(candidate_info, selected_action),
            "mobility_risk": _to_float(info.get("mobility_risk"), 1.0),
            "estimated_cost": _to_float(info.get("estimated_cost"), float("inf")),
            "estimated_delay_s": _to_float(info.get("estimated_delay"), 0.0),
            "estimated_queue": _to_float(info.get("estimated_queue"), 0.0),
            "estimated_energy_j": _to_float(info.get("estimated_energy_j"), 0.0),
            "selected_vm_id": int(_to_float(info.get("selected_vm_id"), -1)),
            "selected_candidate_index": int(_to_float(info.get("selected_candidate_index"), -1)),
            "objective": float(objective),
            "dpp_v": self.dpp_v,
            "grid_mode": self.grid_mode,
            "lower_action_order": "cpu_share,bandwidth_share,tx_power_ratio",
            "lyapunov_semantics": "reward_shaping_no_stability_theorem",
            "queue_stability_claim_allowed": False,
            "uses_oracle": False,
            "uses_privileged_info": False,
        }
        decision_info.update(components)
        return {
            "upper_action": int(selected_action),
            "lower_action": lower_action,
            "action_name": ACTION_NAMES[selected_action],
            "decision_info": decision_info,
        }

    def _objective(
        self,
        *,
        action: int,
        state: Mapping[str, Any],
        info: Mapping[str, Any],
        cpu_share: float,
        bw_share: float,
        tx_power_ratio: float,
    ) -> tuple[float, dict[str, float]]:
        local_queue = max(0.0, _to_float(state.get("local_queue", state.get("queue", 0.0)), 0.0))
        target_queue = max(0.0, _to_float(info.get("estimated_queue"), 0.0))
        backlog = max(local_queue, target_queue)
        base_delay = max(0.0, _to_float(info.get("estimated_delay"), _to_float(info.get("estimated_cost"), 0.0)))
        base_energy = max(0.0, _to_float(info.get("estimated_energy_j"), 0.0))
        mobility_risk = max(0.0, min(1.0, _to_float(info.get("mobility_risk"), 0.0)))
        deadline = max(1.0e-6, _to_float(state.get("deadline_threshold", state.get("deadlineThreshold", 1.0)), 1.0))
        rate_mbps = max(1.0e-6, _to_float(info.get("rate_mbps", info.get("estimated_rate_mbps", 1.0)), 1.0))

        service_proxy = cpu_share * (1.0 + 0.01 * min(rate_mbps, 1000.0))
        next_queue = max(0.0, backlog - service_proxy)
        estimated_drift = 0.5 * (next_queue * next_queue - backlog * backlog)

        if action == 0:
            resource_delay = base_delay / max(cpu_share, 1.0e-6)
            resource_energy = base_energy + 0.01 * cpu_share * cpu_share
        else:
            tx_delay = base_delay / max(bw_share, 1.0e-6)
            compute_delay = base_delay / max(cpu_share, 1.0e-6)
            resource_delay = 0.5 * tx_delay + 0.5 * compute_delay
            resource_energy = base_energy + tx_power_ratio * tx_delay + 0.01 * cpu_share * cpu_share
        violation = max(0.0, resource_delay - deadline)
        penalty = (
            self.delay_weight * resource_delay
            + self.energy_weight * resource_energy
            + self.violation_weight * violation
            + self.queue_weight * next_queue
            + self.risk_weight * mobility_risk
        )
        objective = estimated_drift + self.dpp_v * penalty
        return float(objective), {
            "estimated_drift": float(estimated_drift),
            "estimated_penalty": float(penalty),
            "estimated_next_queue": float(next_queue),
            "estimated_resource_delay_s": float(resource_delay),
            "estimated_resource_energy_j": float(resource_energy),
            "estimated_violation_s": float(violation),
        }


class LyapunovDppOptimizedPolicy(OptimizedLyapunovDppPolicy):
    name = "lyapunov_dpp_optimized"
