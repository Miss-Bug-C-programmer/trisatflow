from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch


GRADIENT_FIELDS = [
    "encoder_mode",
    "upper_actor_grad_norm",
    "upper_critic_grad_norm",
    "shared_encoder_grad_norm_from_upper",
    "shared_encoder_grad_norm_from_lower",
    "lower_actor_grad_norm",
    "lower_critic_grad_norm",
    "separate_lower_encoder_grad_norm",
    "upper_policy_kl",
    "upper_action_entropy",
    "lower_action_variance",
    "lower_action_sensitivity_to_upper_action",
    "update_step",
    "lower_allocator_not_conditioned_effectively",
    "unavailable_fields",
]


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().cpu())
    return float(total ** 0.5)


@torch.no_grad()
def lower_action_sensitivity_to_upper_action(
    lower_agent: Any,
    embed: torch.Tensor,
    *,
    obs: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
    edge_attr: torch.Tensor | None = None,
    n_upper_actions: int = 4,
) -> Dict[str, Any]:
    actions: List[torch.Tensor] = []
    for action_id in range(n_upper_actions):
        upper = torch.full((embed.shape[0],), action_id, dtype=torch.long, device=embed.device)
        action = lower_agent.act(
            embed,
            upper,
            explore=False,
            obs=obs,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        actions.append(action.detach())
    diffs = []
    for i in range(len(actions)):
        for j in range(i + 1, len(actions)):
            diffs.append(torch.mean(torch.abs(actions[i] - actions[j])))
    sensitivity = float(torch.stack(diffs).mean().cpu()) if diffs else 0.0
    variance = float(torch.stack(actions, dim=0).var(unbiased=False).mean().cpu()) if actions else 0.0
    return {
        "lower_action_sensitivity_to_upper_action": sensitivity,
        "lower_action_variance": variance,
        "lower_allocator_not_conditioned_effectively": bool(sensitivity < 1.0e-6),
    }


def build_gradient_report(
    *,
    trainer: Any,
    upper_losses: Dict[str, Any],
    lower_losses: Dict[str, Any],
    update_step: int,
    sensitivity: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    unavailable: Dict[str, str] = {}
    upper_actor = getattr(getattr(trainer, "upper_agent", None), "actor", None)
    upper_critic = getattr(getattr(trainer, "upper_agent", None), "critic", None)
    if upper_critic is None:
        upper_critic = getattr(getattr(trainer, "upper_agent", None), "value", None)
    encoder = getattr(trainer, "encoder", None)

    if upper_actor is None:
        unavailable["upper_actor_grad_norm"] = "upper agent has no actor module"
    if upper_critic is None:
        unavailable["upper_critic_grad_norm"] = "upper agent has no critic/value module"
    if encoder is None:
        unavailable["shared_encoder_grad_norm_from_upper"] = "trainer has no shared encoder"

    row: Dict[str, Any] = {
        "encoder_mode": str(getattr(trainer, "_lower_encoder_mode", lambda: "unknown")()),
        "upper_actor_grad_norm": grad_norm(upper_actor.parameters()) if upper_actor is not None else 0.0,
        "upper_critic_grad_norm": grad_norm(upper_critic.parameters()) if upper_critic is not None else 0.0,
        "shared_encoder_grad_norm_from_upper": float(upper_losses.get("shared_encoder_grad_norm_from_upper", 0.0)),
        "shared_encoder_grad_norm_from_lower": float(lower_losses.get("shared_encoder_grad_norm_from_lower", lower_losses.get("lower_encoder_grad_norm", 0.0))),
        "lower_actor_grad_norm": float(lower_losses.get("lower_actor_grad_norm", 0.0)),
        "lower_critic_grad_norm": float(lower_losses.get("lower_critic_grad_norm", 0.0)),
        "separate_lower_encoder_grad_norm": float(lower_losses.get("separate_lower_encoder_grad_norm", 0.0)),
        "upper_policy_kl": float(upper_losses.get("upper_approx_kl", upper_losses.get("approx_kl", 0.0))),
        "upper_action_entropy": float(upper_losses.get("upper_entropy", upper_losses.get("entropy", 0.0))),
        "lower_action_variance": float((sensitivity or {}).get("lower_action_variance", 0.0)),
        "lower_action_sensitivity_to_upper_action": float((sensitivity or {}).get("lower_action_sensitivity_to_upper_action", 0.0)),
        "lower_allocator_not_conditioned_effectively": bool((sensitivity or {}).get("lower_allocator_not_conditioned_effectively", False)),
        "update_step": int(update_step),
        "unavailable_fields": unavailable,
    }
    return row
