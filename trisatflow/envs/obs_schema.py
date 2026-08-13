from __future__ import annotations

from typing import List

import torch

ACTION_LOCAL = 0
ACTION_NEIGHBOR = 1
ACTION_GEO = 2
ACTION_GROUND = 3
N_UPPER_ACTIONS = 4
ACTION_NAMES = ["local", "neighbor", "geo", "ground"]
TIER_NAMES = ACTION_NAMES

# Shared tier-summary schema used by trace export, live SatEdgeSim replay, and
# new training runs.
IDX_LOCAL_VISIBLE = 0
IDX_NEIGHBOR_VISIBLE = 1
IDX_GEO_VISIBLE = 2
IDX_GROUND_VISIBLE = 3

IDX_LOCAL_RATE = 4
IDX_NEIGHBOR_RATE = 5
IDX_GEO_RATE = 6
IDX_GROUND_RATE = 7

IDX_LOCAL_DELAY = 8
IDX_NEIGHBOR_DELAY = 9
IDX_GEO_DELAY = 10
IDX_GROUND_DELAY = 11

IDX_LOCAL_QUEUE = 12
IDX_NEIGHBOR_QUEUE = 13
IDX_GEO_QUEUE = 14
IDX_GROUND_QUEUE = 15

IDX_LOCAL_NORMALIZED_COST = 16
IDX_NEIGHBOR_NORMALIZED_COST = 17
IDX_GEO_NORMALIZED_COST = 18
IDX_GROUND_NORMALIZED_COST = 19

# Completion-aware mobility features. These are appended after the original
# 20-dim cost schema, so legacy 12/16/20-dim checkpoints remain loadable.
IDX_LOCAL_COMPLETION_SAFE = 20
IDX_NEIGHBOR_COMPLETION_SAFE = 21
IDX_GEO_COMPLETION_SAFE = 22
IDX_GROUND_COMPLETION_SAFE = 23

IDX_LOCAL_MOBILITY_RISK = 24
IDX_NEIGHBOR_MOBILITY_RISK = 25
IDX_GEO_MOBILITY_RISK = 26
IDX_GROUND_MOBILITY_RISK = 27

IDX_LOCAL_LINK_LIFETIME = 28
IDX_NEIGHBOR_LINK_LIFETIME = 29
IDX_GEO_LINK_LIFETIME = 30
IDX_GROUND_LINK_LIFETIME = 31

IDX_LOCAL_LINK_MARGIN_TO_COMPLETION = 32
IDX_NEIGHBOR_LINK_MARGIN_TO_COMPLETION = 33
IDX_GEO_LINK_MARGIN_TO_COMPLETION = 34
IDX_GROUND_LINK_MARGIN_TO_COMPLETION = 35

IDX_LOCAL_HANDOVER_REQUIRED = 36
IDX_NEIGHBOR_HANDOVER_REQUIRED = 37
IDX_GEO_HANDOVER_REQUIRED = 38
IDX_GROUND_HANDOVER_REQUIRED = 39

SHARED_NODE_FEATURE_DIM = 16
SHARED_NODE_FEATURE_DIM_WITH_COST = 20
SHARED_NODE_FEATURE_DIM_WITH_MOBILITY = 40

FIELD_NAMES: List[str] = [
    "local_visible",
    "neighbor_visible",
    "geo_visible",
    "ground_visible",
    "local_rate",
    "neighbor_rate",
    "geo_rate",
    "ground_rate",
    "local_delay",
    "neighbor_delay",
    "geo_delay",
    "ground_delay",
    "local_queue",
    "neighbor_queue",
    "geo_queue",
    "ground_queue",
    "local_normalized_cost",
    "neighbor_normalized_cost",
    "geo_normalized_cost",
    "ground_normalized_cost",
    "local_completion_safe",
    "neighbor_completion_safe",
    "geo_completion_safe",
    "ground_completion_safe",
    "local_mobility_risk",
    "neighbor_mobility_risk",
    "geo_mobility_risk",
    "ground_mobility_risk",
    "local_link_lifetime_sec",
    "neighbor_link_lifetime_sec",
    "geo_link_lifetime_sec",
    "ground_link_lifetime_sec",
    "local_link_survival_margin_to_completion_sec",
    "neighbor_link_survival_margin_to_completion_sec",
    "geo_link_survival_margin_to_completion_sec",
    "ground_link_survival_margin_to_completion_sec",
    "local_handover_required",
    "neighbor_handover_required",
    "geo_handover_required",
    "ground_handover_required",
]

# Feature access policy for experiment safety audits.
FEATURE_ACCESS_CLASS = {
    "local_visible": "observable",
    "neighbor_visible": "observable",
    "geo_visible": "observable",
    "ground_visible": "observable",
    "local_rate": "estimated",
    "neighbor_rate": "estimated",
    "geo_rate": "estimated",
    "ground_rate": "estimated",
    "local_delay": "estimated",
    "neighbor_delay": "estimated",
    "geo_delay": "estimated",
    "ground_delay": "estimated",
    "local_queue": "estimated",
    "neighbor_queue": "estimated",
    "geo_queue": "estimated",
    "ground_queue": "estimated",
    "local_normalized_cost": "privileged",
    "neighbor_normalized_cost": "privileged",
    "geo_normalized_cost": "privileged",
    "ground_normalized_cost": "privileged",
    "local_completion_safe": "estimated",
    "neighbor_completion_safe": "estimated",
    "geo_completion_safe": "estimated",
    "ground_completion_safe": "estimated",
    "local_mobility_risk": "estimated",
    "neighbor_mobility_risk": "estimated",
    "geo_mobility_risk": "estimated",
    "ground_mobility_risk": "estimated",
    "local_link_lifetime_sec": "estimated",
    "neighbor_link_lifetime_sec": "estimated",
    "geo_link_lifetime_sec": "estimated",
    "ground_link_lifetime_sec": "estimated",
    "local_link_survival_margin_to_completion_sec": "estimated",
    "neighbor_link_survival_margin_to_completion_sec": "estimated",
    "geo_link_survival_margin_to_completion_sec": "estimated",
    "ground_link_survival_margin_to_completion_sec": "estimated",
    "local_handover_required": "estimated",
    "neighbor_handover_required": "estimated",
    "geo_handover_required": "estimated",
    "ground_handover_required": "estimated",
}

ORACLE_SWITCHABLE_COST_FIELDS = {
    "local_normalized_cost",
    "neighbor_normalized_cost",
    "geo_normalized_cost",
    "ground_normalized_cost",
}


def feature_access_class(
    field_name: str,
    *,
    access_mode: str = "safe_observable",
    include_oracle_cost: bool = False,
) -> str:
    base = str(FEATURE_ACCESS_CLASS.get(field_name, "unknown"))
    mode = str(access_mode or "safe_observable").strip().lower()
    if (
        field_name in ORACLE_SWITCHABLE_COST_FIELDS
        and mode == "oracle_debug"
        and bool(include_oracle_cost)
    ):
        return "oracle/debug-only"
    return base

# Legacy 12-dim layout retained only for backward-compatible checkpoint loading.
LEGACY_NODE_FEATURE_DIM = 12
LEGACY_IDX_LOCAL_VISIBLE = 5
LEGACY_IDX_NEIGHBOR_VISIBLE = 6
LEGACY_IDX_NEIGHBOR_RATE = 7
LEGACY_IDX_GEO_VISIBLE = 8
LEGACY_IDX_GEO_RATE = 9
LEGACY_IDX_GROUND_VISIBLE = 10
LEGACY_IDX_GROUND_RATE = 11


def upper_action_mask_from_shared_obs(obs: torch.Tensor) -> torch.Tensor:
    if obs.dim() != 2 or obs.shape[-1] < SHARED_NODE_FEATURE_DIM:
        raise ValueError(f"Expected shared obs with shape [n_agents, >=16], got {tuple(obs.shape)}")
    local = obs[:, IDX_LOCAL_VISIBLE] > 0.5
    neighbor = obs[:, IDX_NEIGHBOR_VISIBLE] > 0.5
    geo = obs[:, IDX_GEO_VISIBLE] > 0.5
    ground = obs[:, IDX_GROUND_VISIBLE] > 0.5
    mask = torch.stack([local, neighbor, geo, ground], dim=-1)
    empty = ~mask.any(dim=-1)
    if empty.any():
        mask[empty, ACTION_LOCAL] = True
    return mask


def upper_action_mask_from_legacy_obs(obs: torch.Tensor) -> torch.Tensor:
    if obs.dim() != 2 or obs.shape[-1] < LEGACY_NODE_FEATURE_DIM:
        raise ValueError(f"Expected legacy obs with shape [n_agents, >=12], got {tuple(obs.shape)}")
    local = obs[:, LEGACY_IDX_LOCAL_VISIBLE] > 0.5
    neighbor = obs[:, LEGACY_IDX_NEIGHBOR_VISIBLE] > 0.5
    geo = obs[:, LEGACY_IDX_GEO_VISIBLE] > 0.5
    ground = obs[:, LEGACY_IDX_GROUND_VISIBLE] > 0.5
    mask = torch.stack([local, neighbor, geo, ground], dim=-1)
    empty = ~mask.any(dim=-1)
    if empty.any():
        mask[empty, ACTION_LOCAL] = True
    return mask
