from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from trisatflow.envs import GeoLeoGroundEnv
from trisatflow.envs.obs_schema import (
    IDX_GEO_RATE,
    IDX_GROUND_RATE,
    IDX_LOCAL_QUEUE,
    IDX_NEIGHBOR_RATE,
    LEGACY_IDX_GEO_RATE,
    LEGACY_IDX_GROUND_RATE,
    LEGACY_IDX_NEIGHBOR_RATE,
    LEGACY_NODE_FEATURE_DIM,
    SHARED_NODE_FEATURE_DIM,
    upper_action_mask_from_legacy_obs,
    upper_action_mask_from_shared_obs,
)


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    description: str


def _validate_obs(obs: torch.Tensor) -> None:
    if obs.dim() != 2:
        raise ValueError(f"Expected baseline observation [n_agents, feature_dim], got {tuple(obs.shape)}")
    if obs.shape[-1] < LEGACY_NODE_FEATURE_DIM:
        raise ValueError(
            f"Baseline observation feature_dim={obs.shape[-1]} is unsupported; "
            f"expected legacy >= {LEGACY_NODE_FEATURE_DIM} or shared >= {SHARED_NODE_FEATURE_DIM}."
        )


def _uses_shared_schema(obs: torch.Tensor) -> bool:
    _validate_obs(obs)
    return obs.shape[-1] >= SHARED_NODE_FEATURE_DIM


def _action_mask(obs: torch.Tensor) -> torch.Tensor:
    if _uses_shared_schema(obs):
        return upper_action_mask_from_shared_obs(obs)
    return upper_action_mask_from_legacy_obs(obs)


def _observable_features(obs: torch.Tensor):
    """Return the observable features consumed by the offline rule baselines.

    The repository supports both the historical 12-dimensional layout and the
    current shared tier-summary layout.  The previous implementation always used
    legacy offsets.  That silently read delays as visibility flags under the
    shared schema and invalidated the greedy-baseline comparison.
    """

    mask = _action_mask(obs)
    if _uses_shared_schema(obs):
        queue = obs[:, IDX_LOCAL_QUEUE]
        neighbor_rate = obs[:, IDX_NEIGHBOR_RATE]
        geo_rate = obs[:, IDX_GEO_RATE]
        ground_rate = obs[:, IDX_GROUND_RATE]
        # The shared schema intentionally does not expose residual battery as an
        # observation feature.  Keep the weighted rule observable-only instead
        # of fabricating a privileged energy signal.
        energy_left = torch.ones_like(queue)
    else:
        queue = obs[:, 0]
        energy_left = obs[:, 2]
        neighbor_rate = obs[:, LEGACY_IDX_NEIGHBOR_RATE]
        geo_rate = obs[:, LEGACY_IDX_GEO_RATE]
        ground_rate = obs[:, LEGACY_IDX_GROUND_RATE]
    return queue, energy_left, neighbor_rate, geo_rate, ground_rate, mask


def _local_fallback(mask: torch.Tensor) -> torch.Tensor:
    """Choose local when available, otherwise the first executable action."""

    local = torch.full((mask.shape[0],), GeoLeoGroundEnv.ACTION_LOCAL, dtype=torch.long, device=mask.device)
    first_valid = mask.float().argmax(dim=-1).long()
    return torch.where(mask[:, GeoLeoGroundEnv.ACTION_LOCAL], local, first_valid)


def _apply_mask(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if scores.shape != mask.shape:
        raise ValueError(f"scores/mask mismatch: scores={tuple(scores.shape)} mask={tuple(mask.shape)}")
    safe_mask = mask.clone()
    empty = ~safe_mask.any(dim=-1)
    if empty.any():
        safe_mask[empty, GeoLeoGroundEnv.ACTION_LOCAL] = True
    return scores.masked_fill(~safe_mask, float("inf"))


class RandomPolicy:
    """Uniform random policy over executable actions only."""

    def __call__(self, obs: torch.Tensor):
        mask = _action_mask(obs)
        probs = mask.float()
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0)
        upper = torch.multinomial(probs, num_samples=1).squeeze(-1)
        lower = torch.rand(obs.shape[0], GeoLeoGroundEnv.LOWER_ACTION_DIM, device=obs.device)
        return upper, lower


class ConstantTargetPolicy:
    """Static-tier policy with an executable local fallback.

    A masked MDP cannot execute an unavailable direction.  The old evaluator
    forced invalid actions into the environment, while the online SatEdgeSim
    baseline registry used fallback semantics.  The offline and online static
    baselines now follow the same behavior.
    """

    def __init__(self, target: int, resource_level: float = 0.75):
        self.target = int(target)
        self.resource_level = float(resource_level)

    def __call__(self, obs: torch.Tensor):
        mask = _action_mask(obs)
        upper = torch.full((obs.shape[0],), self.target, dtype=torch.long, device=obs.device)
        fallback = _local_fallback(mask)
        upper = torch.where(mask[:, self.target], upper, fallback)
        lower = torch.ones(obs.shape[0], GeoLeoGroundEnv.LOWER_ACTION_DIM, device=obs.device) * self.resource_level
        return upper, lower


class LocalOnlyPolicy(ConstantTargetPolicy):
    def __init__(self):
        super().__init__(GeoLeoGroundEnv.ACTION_LOCAL, resource_level=0.75)


class NeighborOnlyPolicy(ConstantTargetPolicy):
    def __init__(self):
        super().__init__(GeoLeoGroundEnv.ACTION_NEIGHBOR, resource_level=0.75)


class GeoOnlyPolicy(ConstantTargetPolicy):
    def __init__(self):
        super().__init__(GeoLeoGroundEnv.ACTION_GEO, resource_level=0.75)


class GroundOnlyPolicy(ConstantTargetPolicy):
    def __init__(self):
        super().__init__(GeoLeoGroundEnv.ACTION_GROUND, resource_level=0.75)


class GreedyQueuePolicy:
    """Queue-pressure rule that only selects executable actions."""

    def __call__(self, obs: torch.Tensor):
        queue, _, _, _, _, mask = _observable_features(obs)
        upper = _local_fallback(mask)
        neighbor = torch.full_like(upper, GeoLeoGroundEnv.ACTION_NEIGHBOR)
        geo = torch.full_like(upper, GeoLeoGroundEnv.ACTION_GEO)
        ground = torch.full_like(upper, GeoLeoGroundEnv.ACTION_GROUND)

        choose_neighbor = (queue > 0.25) & mask[:, GeoLeoGroundEnv.ACTION_NEIGHBOR]
        upper = torch.where(choose_neighbor, neighbor, upper)
        choose_geo = (queue > 0.35) & mask[:, GeoLeoGroundEnv.ACTION_GEO]
        upper = torch.where(choose_geo, geo, upper)
        choose_ground = (queue > 0.55) & mask[:, GeoLeoGroundEnv.ACTION_GROUND]
        upper = torch.where(choose_ground, ground, upper)
        lower = torch.stack(
            [
                torch.clamp(0.4 + queue, 0.1, 1.0),
                torch.clamp(0.5 + queue, 0.1, 1.0),
                torch.clamp(0.4 + queue, 0.1, 1.0),
            ],
            dim=-1,
        )
        return upper, lower


class GreedyDelayPolicy:
    """Approximate delay-first rule from observation-level link summaries."""

    def __call__(self, obs: torch.Tensor):
        queue, _, neighbor_rate, geo_rate, ground_rate, mask = _observable_features(obs)
        lower = torch.ones(obs.shape[0], GeoLeoGroundEnv.LOWER_ACTION_DIM, device=obs.device) * 0.9
        local_score = 0.30 + queue
        neighbor_score = 0.45 + 0.7 * queue - neighbor_rate
        geo_score = 1.00 + 0.5 * queue - 0.5 * geo_rate
        ground_score = 0.65 + 0.4 * queue - 0.7 * ground_rate
        scores = _apply_mask(torch.stack([local_score, neighbor_score, geo_score, ground_score], dim=-1), mask)
        return torch.argmin(scores, dim=-1), lower


class GreedyEnergyPolicy:
    """Energy-first rule: stay local unless backlog is high and ground is executable."""

    def __call__(self, obs: torch.Tensor):
        queue, _, _, _, _, mask = _observable_features(obs)
        upper = _local_fallback(mask)
        ground = torch.full_like(upper, GeoLeoGroundEnv.ACTION_GROUND)
        upper = torch.where((queue > 0.75) & mask[:, GeoLeoGroundEnv.ACTION_GROUND], ground, upper)
        lower = torch.stack(
            [
                torch.clamp(0.25 + 0.45 * queue, 0.15, 0.8),
                torch.clamp(0.25 + 0.35 * queue, 0.15, 0.75),
                torch.clamp(0.10 + 0.25 * queue, 0.05, 0.55),
            ],
            dim=-1,
        )
        return upper, lower


class GreedyWeightedCostPolicy:
    """Balanced observable-only weighted delay/queue/link-quality rule."""

    def __call__(self, obs: torch.Tensor):
        queue, energy_left, neighbor_rate, geo_rate, ground_rate, mask = _observable_features(obs)
        local = 0.5 * queue + 0.25 * (1.0 - energy_left)
        neighbor = 0.45 * queue + 0.15 - 0.2 * neighbor_rate
        geo = 0.35 * queue + 0.55 - 0.1 * geo_rate
        ground = 0.30 * queue + 0.35 - 0.2 * ground_rate
        scores = _apply_mask(torch.stack([local, neighbor, geo, ground], dim=-1), mask)
        upper = torch.argmin(scores, dim=-1)
        lower = torch.stack(
            [
                torch.clamp(0.35 + 0.55 * queue, 0.2, 1.0),
                torch.clamp(0.35 + 0.45 * queue, 0.2, 1.0),
                torch.clamp(0.20 + 0.40 * queue, 0.05, 0.85),
            ],
            dim=-1,
        )
        return upper, lower


def baseline_registry() -> Dict[str, object]:
    return {
        "random": RandomPolicy(),
        "local_only": LocalOnlyPolicy(),
        "neighbor_only": NeighborOnlyPolicy(),
        "geo_only": GeoOnlyPolicy(),
        "ground_only": GroundOnlyPolicy(),
        "greedy_queue": GreedyQueuePolicy(),
        "greedy_delay": GreedyDelayPolicy(),
        "greedy_energy": GreedyEnergyPolicy(),
        "greedy_weighted_cost": GreedyWeightedCostPolicy(),
    }
