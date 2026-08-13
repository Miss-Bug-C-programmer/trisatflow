from __future__ import annotations

import random
from typing import Any, Dict, Mapping, Sequence

from trisatflow.baselines.registry import ACTION_NAMES, finalize_baseline_decision


class _StaticTierPolicy:
    name = "static"
    target_action = 0
    fallback_policy = "cost_greedy"

    def select_action(
        self,
        obs: Any,
        state: Mapping[str, Any],
        mask: Sequence[int],
        candidate_info: Mapping[int, Mapping[str, Any]],
        rng: random.Random,
    ) -> Dict[str, Any]:
        return finalize_baseline_decision(
            baseline_name=self.name,
            requested_action=int(self.target_action),
            mask=mask,
            candidate_info=candidate_info,
            rng=rng,
            fallback_policy=self.fallback_policy,
            selection_reason=f"static_target_{ACTION_NAMES[int(self.target_action)]}",
            extra_info={
                "target_tier": ACTION_NAMES[int(self.target_action)],
                "uses_oracle": False,
                "uses_privileged_info": False,
            },
        )


class LocalOnlyPolicy(_StaticTierPolicy):
    name = "local_only"
    target_action = 0


class NeighborOnlyPolicy(_StaticTierPolicy):
    name = "neighbor_only"
    target_action = 1


class GeoOnlyPolicy(_StaticTierPolicy):
    name = "geo_only"
    target_action = 2


class GroundOnlyPolicy(_StaticTierPolicy):
    name = "ground_only"
    target_action = 3


class RemoteOnlyPolicy:
    name = "remote_only"

    def __init__(self, fallback_policy: str = "cost_greedy") -> None:
        self.fallback_policy = str(fallback_policy)

    def select_action(
        self,
        obs: Any,
        state: Mapping[str, Any],
        mask: Sequence[int],
        candidate_info: Mapping[int, Mapping[str, Any]],
        rng: random.Random,
    ) -> Dict[str, Any]:
        remote_actions = [a for a in (2, 3) if a < len(mask) and bool(mask[a])]
        if remote_actions:
            remote_actions = sorted(
                remote_actions,
                key=lambda a: (
                    float(candidate_info.get(a, {}).get("mobility_risk", 1.0)),
                    float(candidate_info.get(a, {}).get("estimated_cost", 1.0e18)),
                    a,
                ),
            )
            requested = int(remote_actions[0])
            reason = "remote_only_lowest_risk_cost"
        else:
            requested = 2
            reason = "remote_unavailable_fallback"
        return finalize_baseline_decision(
            baseline_name=self.name,
            requested_action=requested,
            mask=mask,
            candidate_info=candidate_info,
            rng=rng,
            fallback_policy=self.fallback_policy,
            selection_reason=reason,
            extra_info={"target_tier": "remote", "uses_oracle": False, "uses_privileged_info": False},
        )
