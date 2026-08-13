from __future__ import annotations

import random
from typing import Any, Dict, Mapping, Sequence

from trisatflow.baselines.registry import finalize_baseline_decision


class HMADRLMaddqnDdpgBaseline:
    """HMADRL-style placeholder baseline facade.

    This object exposes the baseline registry interface for replay/matrix flows.
    Full training logic (MADDQN upper + DDPG lower) is wired through
    `trisatflow/agents/maddqn_upper.py` and dedicated config/scripts.
    """

    name = "hmadrl_maddqn_ddpg"
    paper_ready = False
    placeholder = True

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
        visible = [idx for idx in range(4) if idx < len(mask) and bool(mask[idx])]
        requested = int(rng.choice(visible)) if visible else 0
        return finalize_baseline_decision(
            baseline_name=self.name,
            requested_action=requested,
            mask=mask,
            candidate_info=candidate_info,
            rng=rng,
            fallback_policy=self.fallback_policy,
            selection_reason="experimental_placeholder_random_visible",
            extra_info={
                "hmadrl_status": "experimental_placeholder",
                "upper_algo": "maddqn",
                "lower_algo": "ddpg",
                "uses_mappo": False,
                "uses_cost_prior_ce": False,
                "uses_oracle": False,
                "uses_privileged_info": False,
            },
        )


class TriMappoMaddpgBaseline:
    """TriSatFlow baseline facade for matrix/replay bookkeeping.

    Actual policy inference for this baseline should use checkpoint replay via
    `scripts/replay_on_satedgesim.py`.
    """

    name = "tri_mappo_maddpg"
    paper_ready = True
    placeholder = False

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
        del obs, state, mask, candidate_info, rng
        raise RuntimeError(
            "TriMappoMaddpgBaseline requires a trained checkpoint; use checkpoint replay "
            "instead of registry fallback for formal or diagnostic evaluation"
        )
