from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from trisatflow.models.policies import masked_policy_logits_and_probs


class FlatHybridPolicy(nn.Module):
    """Single-level hybrid policy for learning-baseline comparisons.

    The actor shares one encoded state representation for both the discrete
    offloading choice and the continuous resource allocation vector.  This keeps
    flat baselines on the same observation contract as TriSatFlow while removing
    the hierarchical upper/lower split.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        n_actions: int = 4,
        resource_dim: int = 3,
        *,
        centralized_value: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_actions = int(n_actions)
        self.resource_dim = int(resource_dim)
        self.centralized_value = bool(centralized_value)
        self.actor_body = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.offload_head = nn.Linear(self.hidden_dim, self.n_actions)
        self.resource_mean_head = nn.Linear(self.hidden_dim, self.resource_dim)
        self.resource_log_std = nn.Parameter(torch.full((self.resource_dim,), -0.8))
        self.local_value = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.central_value = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def actor_features(self, embed: torch.Tensor) -> torch.Tensor:
        return self.actor_body(embed)

    def distributions(
        self,
        embed: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[torch.distributions.Categorical, torch.distributions.Normal, torch.Tensor]:
        h = self.actor_features(embed)
        logits = self.offload_head(h)
        masked_logits, _probs = masked_policy_logits_and_probs(logits, action_mask)
        offload_dist = torch.distributions.Categorical(logits=masked_logits)
        resource_mean = torch.tanh(self.resource_mean_head(h))
        resource_std = self.resource_log_std.exp().clamp(1.0e-3, 2.0).expand_as(resource_mean)
        resource_dist = torch.distributions.Normal(resource_mean, resource_std)
        return offload_dist, resource_dist, masked_logits

    def value(self, embed: torch.Tensor) -> torch.Tensor:
        if self.centralized_value:
            pooled = embed.mean(dim=0, keepdim=True)
            value = self.central_value(pooled).squeeze(-1).expand(embed.shape[0])
            return value
        return self.local_value(embed).squeeze(-1)

    @staticmethod
    def resource_action(raw_resource: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(raw_resource).clamp(0.0, 1.0)

    @torch.no_grad()
    def deterministic_actions(self, embed: torch.Tensor, action_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.actor_features(embed)
        logits = self.offload_head(h)
        masked_logits, _probs = masked_policy_logits_and_probs(logits, action_mask)
        offload = torch.argmax(masked_logits, dim=-1)
        lower = self.resource_action(torch.tanh(self.resource_mean_head(h)))
        return offload, lower
