from __future__ import annotations

import random
from typing import Any, Dict, Mapping, Sequence

from trisatflow.baselines.lower_allocators import (
    LowerAllocator,
    allocator_to_env_lower_action,
    lower_allocator_metadata,
)
from trisatflow.baselines.registry import BaselinePolicy


class LowerAllocatorWrappedBaseline:
    def __init__(self, baseline: BaselinePolicy, allocator: LowerAllocator) -> None:
        self.baseline = baseline
        self.allocator = allocator
        self.name = getattr(baseline, "name", "baseline")

    def select_action(
        self,
        obs: Any,
        state: Mapping[str, Any],
        mask: Sequence[int],
        candidate_info: Mapping[int, Mapping[str, Any]],
        rng: random.Random,
    ) -> Dict[str, Any]:
        decision = dict(self.baseline.select_action(obs, state, mask, candidate_info, rng))
        upper_action = int(decision.get("upper_action", 0))
        allocator_action = self.allocator.allocate(obs, state, upper_action, candidate_info)
        decision["lower_action"] = allocator_to_env_lower_action(allocator_action)
        info = dict(decision.get("decision_info") or {})
        info.update(lower_allocator_metadata(self.allocator))
        info["lower_action_allocator_order_values"] = [float(v) for v in allocator_action.tolist()]
        info["lower_action_env_order_values"] = list(decision["lower_action"])
        decision["decision_info"] = info
        return decision


def wrap_baseline_with_lower_allocator(
    baseline: BaselinePolicy,
    allocator: LowerAllocator,
) -> LowerAllocatorWrappedBaseline:
    return LowerAllocatorWrappedBaseline(baseline, allocator)
