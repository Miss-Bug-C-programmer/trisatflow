from __future__ import annotations

import copy
import random
from typing import Any, Dict, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from trisatflow.agents.hybrid_pdqn import HybridBatch, HybridQNetwork, _grad_norm, _mlp


class FlatHybridActor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 4, param_dim: int = 3, hidden_dim: int = 128) -> None:
        super().__init__()
        self.n_actions = int(n_actions)
        self.param_dim = int(param_dim)
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.logits = nn.Linear(hidden_dim, n_actions)
        self.params = nn.Linear(hidden_dim, n_actions * param_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(obs.float())
        logits = self.logits(h)
        params = torch.sigmoid(self.params(h)).view(obs.shape[0], self.n_actions, self.param_dim)
        return logits, params


class FlatHybridActorCriticAgent:
    baseline_family = "flat_hybrid_actor_critic"

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
        entropy_coef: float = 0.01,
        device: str | torch.device = "cpu",
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.param_dim = int(param_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.entropy_coef = float(entropy_coef)
        self.device = torch.device(device)
        self.actor = FlatHybridActor(obs_dim, n_actions, param_dim, hidden_dim).to(self.device)
        self.critic = HybridQNetwork(obs_dim, n_actions, param_dim, hidden_dim).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "method": "flat_hybrid_ac",
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
    def select_action(self, obs: torch.Tensor, mask: torch.Tensor, epsilon: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        obs = obs.to(self.device).float()
        mask = mask.to(self.device).bool()
        logits, params = self.actor(obs)
        masked_logits = logits.masked_fill(~mask, -1.0e9)
        probs = torch.softmax(masked_logits, dim=-1)
        if epsilon > 0.0:
            random_actions = []
            for row in mask.detach().cpu().tolist():
                feasible = [idx for idx, bit in enumerate(row) if bit]
                random_actions.append(random.choice(feasible) if feasible else 0)
            random_tensor = torch.tensor(random_actions, dtype=torch.long, device=self.device)
            sampled = torch.distributions.Categorical(probs=probs).sample()
            choose_random = torch.rand(obs.shape[0], device=self.device) < float(epsilon)
            action = torch.where(choose_random, random_tensor, sampled)
        else:
            action = torch.distributions.Categorical(probs=probs).sample()
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
            next_logits, next_params = self.target_actor(next_obs)
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
        logits, params = self.actor(obs)
        masked_logits = logits.masked_fill(~mask, -1.0e9)
        dist = torch.distributions.Categorical(logits=masked_logits)
        selected_params = params.gather(1, action.view(-1, 1, 1).expand(-1, 1, self.param_dim)).squeeze(1)
        q_selected = self.critic(obs, action, selected_params)
        with torch.no_grad():
            advantage = target - pred.detach()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        actor_loss = -(log_prob * advantage).mean() - 0.1 * q_selected.mean() - self.entropy_coef * entropy
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
    def load(cls, path: str, *, device: str | torch.device = "cpu") -> "FlatHybridActorCriticAgent":
        payload = torch.load(path, map_location=device)
        agent = cls(int(payload["obs_dim"]), n_actions=int(payload.get("n_actions", 4)), param_dim=int(payload.get("param_dim", 3)), device=device)
        agent.actor.load_state_dict(payload["actor"])
        agent.critic.load_state_dict(payload["critic"])
        agent.target_actor.load_state_dict(payload.get("target_actor", payload["actor"]))
        agent.target_critic.load_state_dict(payload.get("target_critic", payload["critic"]))
        return agent

