from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from trisatflow.envs.obs_schema import (
    IDX_GEO_DELAY,
    IDX_GEO_NORMALIZED_COST,
    IDX_GEO_QUEUE,
    IDX_GEO_RATE,
    IDX_GEO_VISIBLE,
    IDX_GROUND_DELAY,
    IDX_GROUND_NORMALIZED_COST,
    IDX_GROUND_QUEUE,
    IDX_GROUND_RATE,
    IDX_GROUND_VISIBLE,
    IDX_LOCAL_DELAY,
    IDX_LOCAL_NORMALIZED_COST,
    IDX_LOCAL_QUEUE,
    IDX_LOCAL_RATE,
    IDX_LOCAL_VISIBLE,
    IDX_NEIGHBOR_DELAY,
    IDX_NEIGHBOR_NORMALIZED_COST,
    IDX_NEIGHBOR_QUEUE,
    IDX_NEIGHBOR_RATE,
    IDX_NEIGHBOR_VISIBLE,
    LEGACY_NODE_FEATURE_DIM,
    SHARED_NODE_FEATURE_DIM,
    SHARED_NODE_FEATURE_DIM_WITH_COST,
    upper_action_mask_from_legacy_obs,
    upper_action_mask_from_shared_obs,
)


def masked_policy_logits_and_probs(logits: torch.Tensor, action_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply feasibility mask and return masked logits with normalized probs."""
    mask = action_mask.to(device=logits.device, dtype=torch.bool)
    if mask.shape != logits.shape:
        raise ValueError(f"action_mask shape {tuple(mask.shape)} does not match logits shape {tuple(logits.shape)}")
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min / 4)
    probs = torch.softmax(masked, dim=-1)
    return masked, probs



def upper_action_mask_from_obs(obs: torch.Tensor) -> torch.Tensor:
    """Return per-agent feasible [local, neighbor, geo, ground] mask.

    New shared-trace/live observations use the 16-dim tier-summary schema.
    Legacy 12-dim checkpoints still use the original layout and are supported
    for backward-compatible replay diagnostics.
    """
    if obs.dim() != 2:
        raise ValueError(f"Expected obs with shape [n_agents, features], got {tuple(obs.shape)}")
    if obs.shape[-1] >= SHARED_NODE_FEATURE_DIM:
        return upper_action_mask_from_shared_obs(obs)
    if obs.shape[-1] >= LEGACY_NODE_FEATURE_DIM:
        return upper_action_mask_from_legacy_obs(obs)
    raise ValueError(f"Expected obs with shape [n_agents, >=12], got {tuple(obs.shape)}")

def _mlp(in_dim: int, hidden_dim: int, out_dim: int, final_activation: nn.Module | None = None) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)]
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class UpperMAPPOPolicy(nn.Module):
    """Decentralized categorical actor for discrete global offloading."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        n_actions: int = 4,
        *,
        policy_head: str = "gnn_only",
        logit_centering: bool = False,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_actions = int(n_actions)
        self.policy_head = str(policy_head or "gnn_only").strip().lower()
        self.logit_centering = bool(logit_centering)
        self.net = _mlp(embed_dim, hidden_dim, n_actions)
        self._hybrid_in_dim = embed_dim * 2 + 6
        if self.policy_head == "hybrid_gnn_cost":
            self.hybrid_hidden = nn.Sequential(
                nn.Linear(self._hybrid_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.hybrid_out = nn.Linear(hidden_dim, 1)
        else:
            self.hybrid_hidden = None
            self.hybrid_out = None

    @staticmethod
    def _normalized_cost_from_obs(obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] >= SHARED_NODE_FEATURE_DIM_WITH_COST:
            return torch.stack(
                [
                    obs[:, IDX_LOCAL_NORMALIZED_COST],
                    obs[:, IDX_NEIGHBOR_NORMALIZED_COST],
                    obs[:, IDX_GEO_NORMALIZED_COST],
                    obs[:, IDX_GROUND_NORMALIZED_COST],
                ],
                dim=-1,
            )
        if obs.shape[-1] >= SHARED_NODE_FEATURE_DIM:
            rates = torch.stack(
                [obs[:, IDX_LOCAL_RATE], obs[:, IDX_NEIGHBOR_RATE], obs[:, IDX_GEO_RATE], obs[:, IDX_GROUND_RATE]],
                dim=-1,
            )
            delays = torch.stack(
                [obs[:, IDX_LOCAL_DELAY], obs[:, IDX_NEIGHBOR_DELAY], obs[:, IDX_GEO_DELAY], obs[:, IDX_GROUND_DELAY]],
                dim=-1,
            )
            queues = torch.stack(
                [obs[:, IDX_LOCAL_QUEUE], obs[:, IDX_NEIGHBOR_QUEUE], obs[:, IDX_GEO_QUEUE], obs[:, IDX_GROUND_QUEUE]],
                dim=-1,
            )
            visible = torch.stack(
                [
                    obs[:, IDX_LOCAL_VISIBLE] > 0.5,
                    obs[:, IDX_NEIGHBOR_VISIBLE] > 0.5,
                    obs[:, IDX_GEO_VISIBLE] > 0.5,
                    obs[:, IDX_GROUND_VISIBLE] > 0.5,
                ],
                dim=-1,
            )
            tx = torch.zeros_like(rates)
            tx[:, 1:] = 1.0 / rates[:, 1:].clamp_min(1.0e-6)
            compute = torch.relu(delays - tx)
            raw = delays + 0.5 * queues + 0.2 * tx + 0.2 * compute
            raw = raw.masked_fill(~visible, torch.finfo(raw.dtype).max / 4)
            row_min = raw.min(dim=-1, keepdim=True).values
            row_max = raw.max(dim=-1, keepdim=True).values
            norm = (raw - row_min) / (row_max - row_min).clamp_min(1.0e-6)
            return torch.where(visible, norm, torch.ones_like(norm))
        # Legacy fallback.
        return torch.ones((obs.shape[0], 4), dtype=obs.dtype, device=obs.device)

    @classmethod
    def _action_cost_features(cls, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] >= SHARED_NODE_FEATURE_DIM:
            visible = torch.stack(
                [obs[:, IDX_LOCAL_VISIBLE], obs[:, IDX_NEIGHBOR_VISIBLE], obs[:, IDX_GEO_VISIBLE], obs[:, IDX_GROUND_VISIBLE]],
                dim=-1,
            )
            rates = torch.stack(
                [obs[:, IDX_LOCAL_RATE], obs[:, IDX_NEIGHBOR_RATE], obs[:, IDX_GEO_RATE], obs[:, IDX_GROUND_RATE]],
                dim=-1,
            )
            delays = torch.stack(
                [obs[:, IDX_LOCAL_DELAY], obs[:, IDX_NEIGHBOR_DELAY], obs[:, IDX_GEO_DELAY], obs[:, IDX_GROUND_DELAY]],
                dim=-1,
            )
            queues = torch.stack(
                [obs[:, IDX_LOCAL_QUEUE], obs[:, IDX_NEIGHBOR_QUEUE], obs[:, IDX_GEO_QUEUE], obs[:, IDX_GROUND_QUEUE]],
                dim=-1,
            )
            norm_cost = cls._normalized_cost_from_obs(obs)
        else:
            # Legacy observations expose visibility and partial rate only.
            visible = torch.ones((obs.shape[0], 4), dtype=obs.dtype, device=obs.device)
            rates = torch.ones((obs.shape[0], 4), dtype=obs.dtype, device=obs.device)
            delays = torch.zeros((obs.shape[0], 4), dtype=obs.dtype, device=obs.device)
            queues = torch.zeros((obs.shape[0], 4), dtype=obs.dtype, device=obs.device)
            norm_cost = torch.ones((obs.shape[0], 4), dtype=obs.dtype, device=obs.device)
        tx = torch.zeros_like(rates)
        tx[:, 1:] = 1.0 / rates[:, 1:].clamp_min(1.0e-6)
        compute_proxy = torch.relu(delays - tx)
        # [N, A, 6] -> [visible, normalized_cost, delay, queue, rate, compute_proxy]
        return torch.stack([visible, norm_cost, delays, queues, rates, compute_proxy], dim=-1)

    def compute_logits(
        self,
        node_embed: torch.Tensor,
        *,
        obs: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        if self.policy_head == "hybrid_gnn_cost":
            if obs is None:
                raise ValueError("hybrid_gnn_cost policy head requires obs tensor")
            if self.hybrid_hidden is None or self.hybrid_out is None:
                raise RuntimeError("hybrid_gnn_cost head is not initialized")
            action_feat = self._action_cost_features(obs).to(node_embed.device, node_embed.dtype)
            global_ctx = node_embed.mean(dim=0, keepdim=True).expand_as(node_embed)
            agent_ctx = node_embed.unsqueeze(1).expand(-1, self.n_actions, -1)
            global_ctx_ex = global_ctx.unsqueeze(1).expand(-1, self.n_actions, -1)
            hybrid_in = torch.cat([agent_ctx, action_feat, global_ctx_ex], dim=-1)
            hidden = self.hybrid_hidden(hybrid_in.reshape(-1, hybrid_in.shape[-1]))
            logits = self.hybrid_out(hidden).reshape(node_embed.shape[0], self.n_actions)
            hidden_agent = hidden.reshape(node_embed.shape[0], self.n_actions, self.hidden_dim).mean(dim=1)
            details = {
                "policy_hidden": hidden_agent,
                "action_features": action_feat,
                "global_context": global_ctx,
                "policy_head": torch.tensor(1, dtype=torch.int64, device=node_embed.device),
            }
        else:
            # Keep compatibility with legacy scripts that directly call actor.net.
            h1 = F.relu(self.net[0](node_embed))
            h2 = F.relu(self.net[2](h1))
            logits = self.net[4](h2)
            details = {
                "policy_hidden": h2,
                "action_features": torch.zeros((node_embed.shape[0], self.n_actions, 6), dtype=node_embed.dtype, device=node_embed.device),
                "global_context": node_embed.mean(dim=0, keepdim=True).expand_as(node_embed),
                "policy_head": torch.tensor(0, dtype=torch.int64, device=node_embed.device),
            }
        if self.logit_centering:
            logits = logits - logits.mean(dim=-1, keepdim=True)
        if return_details:
            return logits, details
        return logits

    def forward(
        self,
        node_embed: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        obs: torch.Tensor | None = None,
    ) -> torch.distributions.Categorical:
        logits = self.compute_logits(node_embed, obs=obs)
        if action_mask is not None:
            logits, _ = masked_policy_logits_and_probs(logits, action_mask)
        return torch.distributions.Categorical(logits=logits)


class CentralValue(nn.Module):
    """Centralized critic used by MAPPO. It sees the global graph embedding."""

    def __init__(self, embed_dim: int, n_agents: int, hidden_dim: int):
        super().__init__()
        self.net = _mlp(embed_dim * n_agents, hidden_dim, 1)

    def forward(self, node_embed: torch.Tensor) -> torch.Tensor:
        flat = node_embed.reshape(1, -1)
        return self.net(flat).squeeze(0).squeeze(-1)


class CentralPerAgentValue(nn.Module):
    """Centralized per-agent critic.

    Each agent value is conditioned on local embedding and a global pooled
    context, while keeping actor execution decentralized.
    """

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = _mlp(embed_dim * 2, hidden_dim, 1)

    def forward(self, node_embed: torch.Tensor) -> torch.Tensor:
        if node_embed.ndim != 2:
            raise ValueError(f"Expected node_embed [N, D], got {tuple(node_embed.shape)}")
        global_context = node_embed.mean(dim=0, keepdim=True).expand(node_embed.shape[0], -1)
        feat = torch.cat([node_embed, global_context], dim=-1)
        return self.net(feat).squeeze(-1)


class AgentValue(nn.Module):
    """Decentralized value head for IPPO-style upper-layer training."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = _mlp(embed_dim, hidden_dim, 1)

    def forward(self, node_embed: torch.Tensor) -> torch.Tensor:
        return self.net(node_embed).squeeze(-1)


class UpperQNetwork(nn.Module):
    """Per-agent Q network for IQL/VDN/QMIX-style discrete offloading."""

    def __init__(self, embed_dim: int, hidden_dim: int, n_actions: int = 4):
        super().__init__()
        self.net = _mlp(embed_dim, hidden_dim, n_actions)

    def forward(self, node_embed: torch.Tensor) -> torch.Tensor:
        return self.net(node_embed)


class QMixer(nn.Module):
    """Small monotonic mixer for a dependency-light QMIX approximation.

    It is intentionally compact: per-agent chosen Q values are mixed by
    positive state-conditioned weights plus a state-conditioned bias.
    """

    def __init__(self, embed_dim: int, n_agents: int, hidden_dim: int):
        super().__init__()
        self.n_agents = n_agents
        self.hyper_w = _mlp(embed_dim * n_agents, hidden_dim, n_agents)
        self.hyper_b = _mlp(embed_dim * n_agents, hidden_dim, 1)

    def forward(self, chosen_q: torch.Tensor, node_embed: torch.Tensor) -> torch.Tensor:
        flat = node_embed.reshape(1, -1)
        w = F.softplus(self.hyper_w(flat)).squeeze(0) + 1e-6
        b = self.hyper_b(flat).squeeze()
        return (w * chosen_q).sum() + b


class LowerActor(nn.Module):
    """Continuous deterministic resource actor conditioned on high-level offloading action."""

    def __init__(self, embed_dim: int, n_upper_actions: int, hidden_dim: int, action_dim: int = 3):
        super().__init__()
        self.n_upper_actions = n_upper_actions
        self.net = _mlp(embed_dim + n_upper_actions, hidden_dim, action_dim, nn.Sigmoid())

    def forward(self, node_embed: torch.Tensor, upper_action: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(upper_action.long(), num_classes=self.n_upper_actions).float().to(node_embed.device)
        return self.net(torch.cat([node_embed, one_hot], dim=-1))


class StochasticLowerActor(nn.Module):
    """Squashed Gaussian actor for SAC-style continuous resource allocation.

    The sampled action is in [0, 1], matching the simulator's normalized
    [CPU, bandwidth, transmit power] resource fractions.
    """

    def __init__(self, embed_dim: int, n_upper_actions: int, hidden_dim: int, action_dim: int = 3):
        super().__init__()
        self.n_upper_actions = n_upper_actions
        self.action_dim = action_dim
        self.trunk = _mlp(embed_dim + n_upper_actions, hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def _params(self, node_embed: torch.Tensor, upper_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        one_hot = F.one_hot(upper_action.long(), num_classes=self.n_upper_actions).float().to(node_embed.device)
        h = self.trunk(torch.cat([node_embed, one_hot], dim=-1))
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(-5.0, 2.0)
        return mu, log_std

    def sample(self, node_embed: torch.Tensor, upper_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self._params(node_embed, upper_action)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        raw = dist.rsample()
        squashed = torch.tanh(raw)
        action = 0.5 * (squashed + 1.0)
        # tanh correction and affine scale correction. Constant scale term is
        # included for completeness; it does not affect gradients materially.
        log_prob = dist.log_prob(raw) - torch.log(1.0 - squashed.pow(2) + 1e-6) - torch.log(torch.tensor(2.0, device=node_embed.device))
        return action.clamp(0.0, 1.0), log_prob.sum(dim=-1)

    @torch.no_grad()
    def mean_action(self, node_embed: torch.Tensor, upper_action: torch.Tensor) -> torch.Tensor:
        mu, _ = self._params(node_embed, upper_action)
        return (0.5 * (torch.tanh(mu) + 1.0)).clamp(0.0, 1.0)


class LowerCritic(nn.Module):
    """Centralized critic for MADDPG/MASAC-style continuous allocation.

    Input includes all node embeddings, all upper actions and all lower actions,
    giving the lower layer explicit CTDE-style coupling with upper-layer
    decisions.
    """

    def __init__(self, embed_dim: int, n_agents: int, n_upper_actions: int, lower_action_dim: int, hidden_dim: int):
        super().__init__()
        self.n_agents = int(n_agents)
        in_dim = n_agents * (embed_dim + n_upper_actions + lower_action_dim)
        self.n_upper_actions = n_upper_actions
        self.net = _mlp(in_dim, hidden_dim, 1)

    def forward(self, node_embed: torch.Tensor, upper_action: torch.Tensor, lower_action: torch.Tensor) -> torch.Tensor:
        upper_one_hot = F.one_hot(upper_action.long(), num_classes=self.n_upper_actions).float().to(node_embed.device)
        x = torch.cat([node_embed, upper_one_hot, lower_action], dim=-1).reshape(1, -1)
        return self.net(x).squeeze(0).squeeze(-1)


class LocalLowerCritic(nn.Module):
    """Per-agent critic for IDDPG/ISAC lower-layer ablations."""

    def __init__(self, embed_dim: int, n_upper_actions: int, lower_action_dim: int, hidden_dim: int):
        super().__init__()
        self.n_upper_actions = n_upper_actions
        self.net = _mlp(embed_dim + n_upper_actions + lower_action_dim, hidden_dim, 1)

    def forward(self, node_embed: torch.Tensor, upper_action: torch.Tensor, lower_action: torch.Tensor) -> torch.Tensor:
        upper_one_hot = F.one_hot(upper_action.long(), num_classes=self.n_upper_actions).float().to(node_embed.device)
        x = torch.cat([node_embed, upper_one_hot, lower_action], dim=-1)
        return self.net(x).squeeze(-1)
