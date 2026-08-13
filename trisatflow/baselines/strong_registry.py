from __future__ import annotations

from typing import Any, Dict

from trisatflow.agents.attention_candidate_policy import AttentionCandidatePolicy
from trisatflow.agents.flat_hybrid_actor_critic import FlatHybridActorCriticAgent
from trisatflow.agents.hybrid_pdqn import HybridPDQNAgent
from trisatflow.oracles.small_scale_grid_oracle import SmallScaleGridOracle


STRONG_BASELINE_NAMES = {
    "pdqn_hybrid": HybridPDQNAgent,
    "flat_hybrid_ac": FlatHybridActorCriticAgent,
    "attention_candidate": AttentionCandidatePolicy,
    "small_scale_grid_oracle": SmallScaleGridOracle,
    "grid_oracle": SmallScaleGridOracle,
}


def _strong_meta(
    *,
    key: str,
    family: str,
    trainable: bool,
    update_implemented: bool,
    continuous_action_supported: bool = True,
    is_placeholder: bool = False,
    status: str = "",
    oracle_name_guard: str = "",
) -> Dict[str, Any]:
    return {
        "method": key,
        "baseline_name": key if key != "grid_oracle" else "small_scale_grid_oracle",
        "baseline_family": family,
        "implemented": bool(update_implemented and not is_placeholder),
        "trainable": bool(trainable),
        "requires_checkpoint": bool(trainable),
        "checkpoint_loaded": False,
        "paper_ready": False,
        "is_placeholder": bool(is_placeholder),
        "allows_formal_eval": False,
        "fallback_policy": None,
        "update_implemented": bool(update_implemented),
        "mask_supported": True,
        "action_mask_supported": True,
        "continuous_action_supported": bool(continuous_action_supported),
        "smoke_training_passed": False,
        "full_experiment_required": True,
        **({"status": status} if status else {}),
        **({"oracle_name_guard": oracle_name_guard} if oracle_name_guard else {}),
    }


def strong_baseline_metadata(name: str) -> Dict[str, Any]:
    key = str(name).strip().lower()
    if key == "pdqn_hybrid":
        return _strong_meta(key=key, family="pdqn_hybrid", trainable=True, update_implemented=True)
    if key == "flat_hybrid_ac":
        return _strong_meta(key=key, family="flat_hybrid_actor_critic", trainable=True, update_implemented=True)
    if key == "attention_candidate":
        return _strong_meta(
            key=key,
            family="attention_candidate",
            trainable=True,
            update_implemented=False,
            is_placeholder=True,
            status="forward_select_only_future_baseline_candidate",
        )
    if key in {"small_scale_grid_oracle", "grid_oracle"}:
        return _strong_meta(
            key="small_scale_grid_oracle",
            family="small_scale_grid_oracle",
            trainable=False,
            update_implemented=False,
            is_placeholder=True,
            oracle_name_guard="grid_oracle_not_minlp",
        )
    raise KeyError(f"Unknown strong baseline: {name}")


def list_strong_baselines() -> Dict[str, Dict[str, Any]]:
    return {name: strong_baseline_metadata(name) for name in ("pdqn_hybrid", "flat_hybrid_ac", "attention_candidate", "small_scale_grid_oracle")}

