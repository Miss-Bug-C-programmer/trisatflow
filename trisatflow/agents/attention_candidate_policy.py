from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn
import torch.nn.functional as F


class AttentionCandidatePolicy(nn.Module):
    """Candidate-level attention policy for local/neighbor/GEO/ground tokens.

    This module has forward/select_action and is intentionally marked not
    paper-ready until a full RL update protocol is implemented and evaluated.
    """

    baseline_family = "attention_candidate"

    def __init__(self, obs_dim: int, n_actions: int = 4, param_dim: int = 3, hidden_dim: int = 64, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.param_dim = int(param_dim)
        self.device = torch.device(device)
        self.query = nn.Linear(obs_dim, hidden_dim)
        self.action_embedding = nn.Embedding(n_actions, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.param_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, param_dim))
        self.to(self.device)

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "method": "attention_candidate",
            "baseline_family": self.baseline_family,
            "trainable": True,
            "update_implemented": False,
            "mask_supported": True,
            "action_mask_supported": True,
            "continuous_action_supported": True,
            "paper_ready": False,
            "smoke_training_passed": False,
            "full_experiment_required": True,
            "status": "forward_select_only_future_baseline_candidate",
        }

    def forward(self, obs: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        obs = obs.to(self.device).float()
        batch = obs.shape[0]
        q = self.query(obs)
        action_ids = torch.arange(self.n_actions, device=self.device)
        tokens = self.key(self.action_embedding(action_ids)).unsqueeze(0).expand(batch, -1, -1)
        logits = (tokens * q.unsqueeze(1)).sum(dim=-1) / max(tokens.shape[-1] ** 0.5, 1.0)
        if mask is not None:
            logits = logits.masked_fill(~mask.to(self.device).bool(), -1.0e9)
        q_rep = q.unsqueeze(1).expand(-1, self.n_actions, -1)
        params = torch.sigmoid(self.param_head(torch.cat([q_rep, tokens], dim=-1)))
        return logits, params

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, params = self.forward(obs, mask)
        action = torch.argmax(logits, dim=-1)
        lower = params.gather(1, action.view(-1, 1, 1).expand(-1, 1, self.param_dim)).squeeze(1)
        return action, lower.clamp(0.0, 1.0)

    def supervised_update(self, obs: torch.Tensor, mask: torch.Tensor, target_action: torch.Tensor) -> Dict[str, float]:
        logits, _ = self.forward(obs, mask)
        loss = F.cross_entropy(logits, target_action.to(self.device).long())
        loss.backward()
        return {"supervised_loss": float(loss.detach().cpu().item())}

