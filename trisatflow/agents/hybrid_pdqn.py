from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

import torch
from torch import nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def _grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(param.grad.detach().pow(2).sum().cpu().item())
    return float(total ** 0.5)


class ParameterActor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 4, param_dim: int = 3, hidden_dim: int = 128) -> None:
        super().__init__()
        self.n_actions = int(n_actions)
        self.param_dim = int(param_dim)
        self.net = _mlp(obs_dim, hidden_dim, self.n_actions * self.param_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(obs)).view(obs.shape[0], self.n_actions, self.param_dim)


class HybridQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 4, param_dim: int = 3, hidden_dim: int = 128) -> None:
        super().__init__()
        self.n_actions = int(n_actions)
        self.net = _mlp(obs_dim + n_actions + param_dim, hidden_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(action.long().clamp(0, self.n_actions - 1), num_classes=self.n_actions).float()
        return self.net(torch.cat([obs.float(), one_hot, params.float()], dim=-1)).squeeze(-1)

    def all_actions(self, obs: torch.Tensor, params_by_action: torch.Tensor) -> torch.Tensor:
        values: List[torch.Tensor] = []
        for action in range(self.n_actions):
            a = torch.full((obs.shape[0],), action, dtype=torch.long, device=obs.device)
            values.append(self.forward(obs, a, params_by_action[:, action, :]))
        return torch.stack(values, dim=-1)


@dataclass
class HybridBatch:
    obs: torch.Tensor
    mask: torch.Tensor
    action: torch.Tensor
    lower_action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    next_mask: torch.Tensor
    done: torch.Tensor


class HybridReplayBuffer:
    def __init__(self, capacity: int = 10000, device: str | torch.device = "cpu") -> None:
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.rows: List[Dict[str, torch.Tensor]] = []

    def __len__(self) -> int:
        return len(self.rows)

    def add_batch(
        self,
        *,
        obs: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor,
        lower_action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        next_mask: torch.Tensor,
        done: bool | torch.Tensor,
    ) -> None:
        done_tensor = torch.as_tensor(done, dtype=torch.float32, device=obs.device).expand(obs.shape[0])
        for idx in range(obs.shape[0]):
            self.rows.append(
                {
                    "obs": obs[idx].detach().cpu(),
                    "mask": mask[idx].detach().cpu().float(),
                    "action": action[idx].detach().cpu().long().view(()),
                    "lower_action": lower_action[idx].detach().cpu().float(),
                    "reward": reward[idx].detach().cpu().float().view(()),
                    "next_obs": next_obs[idx].detach().cpu(),
                    "next_mask": next_mask[idx].detach().cpu().float(),
                    "done": done_tensor[idx].detach().cpu().float().view(()),
                }
            )
        if len(self.rows) > self.capacity:
            self.rows = self.rows[-self.capacity :]

    def sample(self, batch_size: int) -> HybridBatch:
        if not self.rows:
            raise ValueError("Cannot sample from an empty HybridReplayBuffer")
        rows = random.sample(self.rows, k=min(int(batch_size), len(self.rows)))
        return HybridBatch(
            obs=torch.stack([r["obs"] for r in rows]).to(self.device),
            mask=torch.stack([r["mask"] for r in rows]).to(self.device),
            action=torch.stack([r["action"] for r in rows]).to(self.device),
            lower_action=torch.stack([r["lower_action"] for r in rows]).to(self.device),
            reward=torch.stack([r["reward"] for r in rows]).to(self.device),
            next_obs=torch.stack([r["next_obs"] for r in rows]).to(self.device),
            next_mask=torch.stack([r["next_mask"] for r in rows]).to(self.device),
            done=torch.stack([r["done"] for r in rows]).to(self.device),
        )


class HybridPDQNAgent:
    baseline_family = "pdqn_hybrid"

    def __init__(
        self,
        obs_dim: int,
        *,
        n_actions: int = 4,
        param_dim: int = 3,
        hidden_dim: int = 128,
        gamma: float = 0.95,
        tau: float = 0.02,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        device: str | torch.device = "cpu",
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.param_dim = int(param_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.device = torch.device(device)
        self.actor = ParameterActor(obs_dim, n_actions, param_dim, hidden_dim).to(self.device)
        self.critic = HybridQNetwork(obs_dim, n_actions, param_dim, hidden_dim).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "method": "pdqn_hybrid",
            "baseline_family": self.baseline_family,
            "trainable": True,
            "update_implemented": True,
            "mask_supported": True,
            "action_mask_supported": True,
            "continuous_action_supported": True,
            "paper_ready": False,
            "smoke_training_passed": False,
            "full_experiment_required": True,
            "env_lower_action_order": "cpu_share,bandwidth_share,tx_power_ratio",
        }

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor, mask: torch.Tensor, epsilon: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
        obs = obs.to(self.device).float()
        mask = mask.to(self.device).bool()
        params = self.actor(obs)
        q_values = self.critic.all_actions(obs, params)
        q_values = q_values.masked_fill(~mask, -1.0e9)
        greedy = q_values.argmax(dim=-1)
        if epsilon > 0.0:
            random_actions = []
            for row in mask.detach().cpu().tolist():
                feasible = [idx for idx, bit in enumerate(row) if bit]
                random_actions.append(random.choice(feasible) if feasible else 0)
            random_tensor = torch.tensor(random_actions, dtype=torch.long, device=self.device)
            choose_random = torch.rand(obs.shape[0], device=self.device) < float(epsilon)
            action = torch.where(choose_random, random_tensor, greedy)
        else:
            action = greedy
        lower = params.gather(1, action.view(-1, 1, 1).expand(-1, 1, self.param_dim)).squeeze(1).clamp(0.0, 1.0)
        return action, lower

    def update(self, batch: HybridBatch | Mapping[str, torch.Tensor]) -> Dict[str, float]:
        if isinstance(batch, Mapping):
            batch = HybridBatch(**{k: v.to(self.device) for k, v in batch.items()})  # type: ignore[arg-type]
        obs = batch.obs.to(self.device).float()
        mask = batch.mask.to(self.device).bool()
        action = batch.action.to(self.device).long()
        lower = batch.lower_action.to(self.device).float().clamp(0.0, 1.0)
        reward = batch.reward.to(self.device).float()
        next_obs = batch.next_obs.to(self.device).float()
        next_mask = batch.next_mask.to(self.device).bool()
        done = batch.done.to(self.device).float()

        with torch.no_grad():
            next_params = self.target_actor(next_obs)
            next_q = self.target_critic.all_actions(next_obs, next_params).masked_fill(~next_mask, -1.0e9)
            target = reward + self.gamma * (1.0 - done) * next_q.max(dim=-1).values

        pred = self.critic(obs, action, lower)
        critic_loss = F.mse_loss(pred, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = _grad_norm(self.critic.parameters())
        self.critic_opt.step()

        for p in self.critic.parameters():
            p.requires_grad_(False)
        params = self.actor(obs)
        actor_q = self.critic.all_actions(obs, params).masked_fill(~mask, -1.0e9)
        actor_loss = -actor_q.max(dim=-1).values.mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = _grad_norm(self.actor.parameters())
        self.actor_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "grad_norm": float((critic_grad * critic_grad + actor_grad * actor_grad) ** 0.5),
            "critic_grad_norm": float(critic_grad),
            "actor_grad_norm": float(actor_grad),
        }

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        with torch.no_grad():
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(source_param.data, alpha=self.tau)

    def save(self, path: str) -> None:
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "param_dim": self.param_dim,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, *, device: str | torch.device = "cpu") -> "HybridPDQNAgent":
        payload = torch.load(path, map_location=device)
        agent = cls(int(payload["obs_dim"]), n_actions=int(payload.get("n_actions", 4)), param_dim=int(payload.get("param_dim", 3)), device=device)
        agent.actor.load_state_dict(payload["actor"])
        agent.critic.load_state_dict(payload["critic"])
        agent.target_actor.load_state_dict(payload.get("target_actor", payload["actor"]))
        agent.target_critic.load_state_dict(payload.get("target_critic", payload["critic"]))
        return agent

