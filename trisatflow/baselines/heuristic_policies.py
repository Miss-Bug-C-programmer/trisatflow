from __future__ import annotations

import random
from typing import Any, Dict, Mapping, Sequence

from trisatflow.baselines.registry import finalize_baseline_decision, state_action_mask


class RandomVisiblePolicy:
    name = "random_visible"

    def __init__(self, fallback_policy: str = "cost_greedy") -> None:
        self.fallback_policy = str(fallback_policy)

    def select_action(self, obs: Any, state: Mapping[str, Any], mask: Sequence[int], candidate_info: Mapping[int, Mapping[str, Any]], rng: random.Random) -> Dict[str, Any]:
        visible = [idx for idx in range(4) if idx < len(mask) and bool(mask[idx])]
        requested = int(rng.choice(visible)) if visible else 0
        return finalize_baseline_decision(
            baseline_name=self.name,
            requested_action=requested,
            mask=mask,
            candidate_info=candidate_info,
            rng=rng,
            fallback_policy=self.fallback_policy,
            selection_reason="random_visible",
            extra_info={"uses_oracle": False, "uses_privileged_info": False},
        )


class RandomMobilitySafePolicy:
    name = "random_mobility_safe"

    def __init__(self, fallback_policy: str = "cost_greedy") -> None:
        self.fallback_policy = str(fallback_policy)

    def select_action(self, obs: Any, state: Mapping[str, Any], mask: Sequence[int], candidate_info: Mapping[int, Mapping[str, Any]], rng: random.Random) -> Dict[str, Any]:
        mobility_mask = state_action_mask(state, action_mask_mode="mobility_safe")
        mobility_visible = [idx for idx in range(4) if idx < len(mobility_mask) and bool(mobility_mask[idx])]
        if mobility_visible:
            requested = int(rng.choice(mobility_visible))
            reason = "random_mobility_safe"
        else:
            visible = [idx for idx in range(4) if idx < len(mask) and bool(mask[idx])]
            requested = int(rng.choice(visible)) if visible else 0
            reason = "mobility_safe_empty_fallback_visible"
        return finalize_baseline_decision(
            baseline_name=self.name,
            requested_action=requested,
            mask=mask,
            candidate_info=candidate_info,
            rng=rng,
            fallback_policy=self.fallback_policy,
            selection_reason=reason,
            extra_info={"uses_oracle": False, "uses_privileged_info": False},
        )


class RoundRobinVisiblePolicy:
    name = "round_robin_visible"

    def __init__(self, fallback_policy: str = "cost_greedy") -> None:
        self.fallback_policy = str(fallback_policy)
        self.cursor = 0

    def select_action(self, obs: Any, state: Mapping[str, Any], mask: Sequence[int], candidate_info: Mapping[int, Mapping[str, Any]], rng: random.Random) -> Dict[str, Any]:
        requested = 0
        for offset in range(4):
            idx = (self.cursor + offset) % 4
            if idx < len(mask) and bool(mask[idx]):
                requested = idx
                break
        self.cursor = (self.cursor + 1) % 4
        return finalize_baseline_decision(
            baseline_name=self.name,
            requested_action=int(requested),
            mask=mask,
            candidate_info=candidate_info,
            rng=rng,
            fallback_policy=self.fallback_policy,
            selection_reason="round_robin_visible",
            extra_info={"rr_cursor_next": self.cursor, "uses_oracle": False, "uses_privileged_info": False},
        )


class MinDelayGreedyPolicy:
    name = "min_delay_greedy"

    def __init__(self, fallback_policy: str = "cost_greedy") -> None:
        self.fallback_policy = str(fallback_policy)

    def select_action(self, obs: Any, state: Mapping[str, Any], mask: Sequence[int], candidate_info: Mapping[int, Mapping[str, Any]], rng: random.Random) -> Dict[str, Any]:
        visible = [idx for idx in range(4) if idx < len(mask) and bool(mask[idx])]
        requested = min(
            visible,
            key=lambda a: (
                float(candidate_info.get(a, {}).get("estimated_delay", 1.0e18)),
                float(candidate_info.get(a, {}).get("estimated_queue", 1.0e18)),
                float(candidate_info.get(a, {}).get("mobility_risk", 1.0)),
                a,
            ),
            default=0,
        )
        return finalize_baseline_decision(
            baseline_name=self.name,
            requested_action=int(requested),
            mask=mask,
            candidate_info=candidate_info,
            rng=rng,
            fallback_policy=self.fallback_policy,
            selection_reason="min_estimated_delay",
            extra_info={"uses_oracle": False, "uses_privileged_info": False},
        )


class MinEnergyGreedyPolicy:
    name = "min_energy_greedy"

    def __init__(self, fallback_policy: str = "cost_greedy") -> None:
        self.fallback_policy = str(fallback_policy)

    def select_action(self, obs: Any, state: Mapping[str, Any], mask: Sequence[int], candidate_info: Mapping[int, Mapping[str, Any]], rng: random.Random) -> Dict[str, Any]:
        visible = [idx for idx in range(4) if idx < len(mask) and bool(mask[idx])]
        requested = min(
            visible,
            key=lambda a: (
                float(candidate_info.get(a, {}).get("estimated_energy_j", 1.0e18)),
                float(candidate_info.get(a, {}).get("estimated_delay", 1.0e18)),
                float(candidate_info.get(a, {}).get("estimated_queue", 1.0e18)),
                a,
            ),
            default=0,
        )
        return finalize_baseline_decision(
            baseline_name=self.name,
            requested_action=int(requested),
            mask=mask,
            candidate_info=candidate_info,
            rng=rng,
            fallback_policy=self.fallback_policy,
            selection_reason="min_estimated_energy",
            extra_info={"uses_oracle": False, "uses_privileged_info": False},
        )


class QueueAwareGreedyPolicy:
    name = "queue_aware_greedy"

    def __init__(self, queue_weight: float = 1.0, delay_weight: float = 0.5, risk_weight: float = 0.25, fallback_policy: str = "cost_greedy") -> None:
        self.queue_weight = float(max(0.0, queue_weight))
        self.delay_weight = float(max(0.0, delay_weight))
        self.risk_weight = float(max(0.0, risk_weight))
        self.fallback_policy = str(fallback_policy)

    def _score(self, info: Mapping[str, Any]) -> float:
        queue = float(info.get("estimated_queue", 0.0))
        delay = float(info.get("estimated_delay", 0.0))
        risk = float(info.get("mobility_risk", 1.0))
        return float(self.queue_weight * queue + self.delay_weight * delay + self.risk_weight * risk)

    def select_action(self, obs: Any, state: Mapping[str, Any], mask: Sequence[int], candidate_info: Mapping[int, Mapping[str, Any]], rng: random.Random) -> Dict[str, Any]:
        visible = [idx for idx in range(4) if idx < len(mask) and bool(mask[idx])]
        requested = min(visible, key=lambda a: (self._score(candidate_info.get(a, {})), a), default=0)
        return finalize_baseline_decision(
            baseline_name=self.name,
            requested_action=int(requested),
            mask=mask,
            candidate_info=candidate_info,
            rng=rng,
            fallback_policy=self.fallback_policy,
            selection_reason="queue_aware_weighted_greedy",
            extra_info={
                "queue_weight": self.queue_weight,
                "delay_weight": self.delay_weight,
                "risk_weight": self.risk_weight,
                "uses_oracle": False,
                "uses_privileged_info": False,
            },
        )


class MobilityRiskGreedyPolicy:
    name = "mobility_risk_greedy"

    def __init__(self, fallback_policy: str = "cost_greedy", risk_tie_eps: float = 0.05) -> None:
        self.fallback_policy = str(fallback_policy)
        self.risk_tie_eps = float(max(0.0, risk_tie_eps))

    def select_action(self, obs: Any, state: Mapping[str, Any], mask: Sequence[int], candidate_info: Mapping[int, Mapping[str, Any]], rng: random.Random) -> Dict[str, Any]:
        visible = [idx for idx in range(4) if idx < len(mask) and bool(mask[idx])]
        if visible:
            risk_min = min(float(candidate_info.get(a, {}).get("mobility_risk", 1.0)) for a in visible)
            shortlist = [
                a for a in visible
                if float(candidate_info.get(a, {}).get("mobility_risk", 1.0)) <= (risk_min + self.risk_tie_eps)
            ]
            requested = min(shortlist, key=lambda a: float(candidate_info.get(a, {}).get("estimated_delay", 1.0e18)))
        else:
            requested = 0
        return finalize_baseline_decision(
            baseline_name=self.name,
            requested_action=int(requested),
            mask=mask,
            candidate_info=candidate_info,
            rng=rng,
            fallback_policy=self.fallback_policy,
            selection_reason="min_mobility_risk_then_delay",
            extra_info={"risk_tie_eps": self.risk_tie_eps, "uses_oracle": False, "uses_privileged_info": False},
        )


class LyapunovDppGreedyPolicy:
    name = "lyapunov_dpp_greedy"

    def __init__(self, dpp_v: float = 1.0, queue_weight: float = 1.0, risk_weight: float = 0.2, fallback_policy: str = "cost_greedy") -> None:
        self.dpp_v = float(max(0.0, dpp_v))
        self.queue_weight = float(max(0.0, queue_weight))
        self.risk_weight = float(max(0.0, risk_weight))
        self.fallback_policy = str(fallback_policy)

    def _dpp_score(self, info: Mapping[str, Any]) -> float:
        delay = float(info.get("estimated_delay", 0.0))
        queue = float(info.get("estimated_queue", 0.0))
        risk = float(info.get("mobility_risk", 1.0))
        return float(self.dpp_v * delay + self.queue_weight * queue + self.risk_weight * risk)

    def select_action(self, obs: Any, state: Mapping[str, Any], mask: Sequence[int], candidate_info: Mapping[int, Mapping[str, Any]], rng: random.Random) -> Dict[str, Any]:
        visible = [idx for idx in range(4) if idx < len(mask) and bool(mask[idx])]
        requested = min(visible, key=lambda a: (self._dpp_score(candidate_info.get(a, {})), a), default=0)
        return finalize_baseline_decision(
            baseline_name=self.name,
            requested_action=int(requested),
            mask=mask,
            candidate_info=candidate_info,
            rng=rng,
            fallback_policy=self.fallback_policy,
            selection_reason="lyapunov_dpp_greedy",
            extra_info={
                "dpp_v": self.dpp_v,
                "queue_weight": self.queue_weight,
                "risk_weight": self.risk_weight,
                "uses_oracle": False,
                "uses_privileged_info": False,
            },
        )


# Backward-compatible alias class names.
class CostGreedyPolicy(MinDelayGreedyPolicy):
    name = "cost_greedy"


class WeightGreedyPolicy(QueueAwareGreedyPolicy):
    name = "weight_greedy"
