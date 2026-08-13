from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MADDQNUpperNetwork(nn.Module):
    """Small per-agent Q network for HMADRL-style discrete upper control."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_actions: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [N, D] -> q: [N, 4]
        return self.net(obs)


@dataclass
class MADDQNConfig:
    gamma: float = 0.99
    lr: float = 3.0e-4
    tau: float = 0.01
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10000


class MADDQNUpperAgent:
    """Minimal MADDQN-style upper agent with mask-aware epsilon-greedy act."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_actions: int = 4, cfg: MADDQNConfig | None = None, device: str = "cpu") -> None:
        self.cfg = cfg or MADDQNConfig()
        self.device = torch.device(device)
        self.n_actions = int(n_actions)
        self.online_q = MADDQNUpperNetwork(input_dim, hidden_dim, n_actions).to(self.device)
        self.target_q = MADDQNUpperNetwork(input_dim, hidden_dim, n_actions).to(self.device)
        self.target_q.load_state_dict(self.online_q.state_dict())
        self.optimizer = torch.optim.Adam(self.online_q.parameters(), lr=float(self.cfg.lr))
        self.steps = 0

    def _epsilon(self) -> float:
        span = max(1, int(self.cfg.epsilon_decay_steps))
        frac = min(1.0, float(self.steps) / float(span))
        return float(self.cfg.epsilon_start + (self.cfg.epsilon_end - self.cfg.epsilon_start) * frac)

    def act(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None, explore: bool = True) -> Dict[str, Any]:
        self.steps += 1
        obs = obs.to(self.device)
        q = self.online_q(obs)
        if action_mask is None:
            action_mask = torch.ones_like(q, dtype=torch.bool)
        else:
            action_mask = action_mask.to(device=self.device, dtype=torch.bool)
        q_masked = q.masked_fill(~action_mask, -1.0e9)
        greedy = torch.argmax(q_masked, dim=-1)
        if not explore:
            return {"action": greedy, "q_values": q, "epsilon": 0.0}

        eps = self._epsilon()
        rand_flag = torch.rand(greedy.shape[0], device=self.device) < eps
        rand_action = []
        for i in range(greedy.shape[0]):
            valid = torch.nonzero(action_mask[i], as_tuple=False).view(-1)
            if valid.numel() == 0:
                rand_action.append(torch.tensor(0, device=self.device, dtype=torch.long))
            else:
                pick = valid[torch.randint(0, valid.numel(), (1,), device=self.device)]
                rand_action.append(pick.view(()))
        rand_action_t = torch.stack(rand_action)
        action = torch.where(rand_flag, rand_action_t, greedy)
        return {"action": action, "q_values": q, "epsilon": eps}

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        action = batch["action"].to(self.device).long()
        reward = batch["reward"].to(self.device).float()
        done = batch["done"].to(self.device).float()
        next_mask = batch.get("next_mask")
        if next_mask is None:
            next_mask = torch.ones((next_obs.shape[0], self.n_actions), device=self.device, dtype=torch.bool)
        else:
            next_mask = next_mask.to(self.device, dtype=torch.bool)

        q = self.online_q(obs).gather(1, action.view(-1, 1)).squeeze(-1)
        with torch.no_grad():
            next_q_online = self.online_q(next_obs).masked_fill(~next_mask, -1.0e9)
            next_action = torch.argmax(next_q_online, dim=-1, keepdim=True)
            next_q_target = self.target_q(next_obs).gather(1, next_action).squeeze(-1)
            target = reward + float(self.cfg.gamma) * (1.0 - done) * next_q_target

        loss = F.mse_loss(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_q.parameters(), 5.0)
        self.optimizer.step()
        self.soft_update()
        return {"td_loss": float(loss.item()), "q_mean": float(q.mean().item()), "target_mean": float(target.mean().item())}

    def soft_update(self) -> None:
        tau = float(self.cfg.tau)
        with torch.no_grad():
            for p_t, p_o in zip(self.target_q.parameters(), self.online_q.parameters()):
                p_t.data.mul_(1.0 - tau).add_(tau * p_o.data)
