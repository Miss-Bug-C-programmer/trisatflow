from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from trisatflow.agents.replay import ReplayBuffer
from trisatflow.config import AlgoConfig
from trisatflow.models import FeatureEncoder, LowerActor, LowerCritic, TopologyEncoder


class LowerMADDPGAgent:
    """MADDPG-style lower continuous resource allocation agent."""

    def __init__(
        self,
        encoder: TopologyEncoder | FeatureEncoder,
        actor: LowerActor,
        critic: LowerCritic,
        target_actor: LowerActor,
        target_critic: LowerCritic,
        cfg: AlgoConfig,
        device: torch.device,
        *,
        target_encoder: TopologyEncoder | FeatureEncoder | None = None,
        encoder_mode: str | None = None,
    ):
        self.encoder = encoder
        self.actor = actor
        self.critic = critic
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.cfg = cfg
        self.device = device
        mode = self._canonical_encoder_mode(encoder_mode or getattr(cfg, "encoder_mode", "shared_upper_only"))
        self.encoder_mode = mode
        self.lower_observation_mode = str(getattr(cfg, "lower_observation_mode", "shared_embedding") or "shared_embedding").strip().lower()
        if self.lower_observation_mode not in {"shared_embedding", "raw_obs_reencode"}:
            self.lower_observation_mode = "shared_embedding"
        self.stop_gradient_to_encoder_from_lower = bool(getattr(cfg, "stop_gradient_to_encoder_from_lower", self.encoder_mode == "shared_upper_only"))
        self.target_encoder = target_encoder if self.encoder_mode == "separate_lower_encoder" else None
        self.joint_encoder_loss_weight = float(getattr(cfg, "joint_encoder_loss_weight", 0.5) or 0.5)

        self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()), lr=cfg.lower_lr)
        self.critic_optimizer = torch.optim.Adam(list(self.critic.parameters()), lr=cfg.critic_lr)
        self.encoder_optimizer = None
        if self.encoder_mode == "separate_lower_encoder" or (
            self.encoder_mode == "shared_joint" and not self.stop_gradient_to_encoder_from_lower
        ):
            self.encoder_optimizer = torch.optim.Adam(list(self.encoder.parameters()), lr=float(getattr(cfg, "encoder_lr", cfg.lower_lr)))

        self._hard_update(self.target_actor, self.actor)
        self._hard_update(self.target_critic, self.critic)
        if self.target_encoder is not None:
            self._hard_update(self.target_encoder, self.encoder)

    @torch.no_grad()
    def act(
        self,
        embed: torch.Tensor,
        upper_action: torch.Tensor,
        explore: bool = True,
        *,
        obs: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        act_embed = self._act_embed(embed, obs=obs, edge_index=edge_index, edge_attr=edge_attr)
        action = self.actor(act_embed, upper_action)
        if explore:
            action = action + self.cfg.exploration_noise * torch.randn_like(action)
        return action.clamp(0.0, 1.0)

    def update(self, replay: ReplayBuffer) -> Dict[str, float]:
        if len(replay) < max(self.cfg.lower_warmup, self.cfg.lower_batch_size):
            return {
                "lower_actor_loss": 0.0,
                "lower_critic_loss": 0.0,
                "lower_encoder_grad_norm": 0.0,
                "shared_encoder_grad_norm_from_lower": 0.0,
                "separate_lower_encoder_grad_norm": 0.0,
                "lower_actor_grad_norm": 0.0,
                "lower_critic_grad_norm": 0.0,
                "lower_encoder_mode": self.encoder_mode,
                "lower_observation_mode": self.lower_observation_mode,
                "lower_training_stop_gradient_to_encoder": bool(self.stop_gradient_to_encoder_from_lower),
                "lower_actor_conditioned_on_upper_action": True,
                "lower_critic_input_schema": "flatten(all_agent_embedding, upper_action_onehot, lower_action)",
                "lower_critic_n_agents_fixed_at_init": int(getattr(self.critic, "n_agents", 0) or 0),
                "lower_q_mean": 0.0,
                "lower_q_target_mean": 0.0,
            }
        batch = replay.sample(self.cfg.lower_batch_size, self.device)
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        upper_action = batch["upper_action"].long()
        lower_action = batch["lower_action"].float()
        reward = batch["reward"].float().mean(dim=-1)
        done = batch["done"].float().view(-1)

        q_values = []
        target_q_values = []
        for b in range(obs.shape[0]):
            edge_index = batch["edge_index"][b].long()
            edge_attr = batch["edge_attr"][b].float()
            next_edge_index = batch["next_edge_index"][b].long()
            next_edge_attr = batch["next_edge_attr"][b].float()

            embed = self._train_embed(obs[b], edge_index, edge_attr)
            next_embed = self._target_embed(next_obs[b], next_edge_index, next_edge_attr)
            q_values.append(self.critic(embed, upper_action[b], lower_action[b]))
            with torch.no_grad():
                next_lower = self.target_actor(next_embed, upper_action[b])
                target_q = reward[b] + self.cfg.gamma * (1.0 - done[b]) * self.target_critic(next_embed, upper_action[b], next_lower)
                target_q_values.append(target_q)

        q = torch.stack(q_values)
        target_q = torch.stack(target_q_values).detach()
        critic_loss = F.mse_loss(q, target_q)

        self._zero_lower_optimizers(include_encoder=self._encoder_trainable())
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        enc_grad_norm_critic = self._clip_encoder_grad()
        self.critic_optimizer.step()
        self._step_encoder_optimizer()

        # Recompute actor loss after critic update, while freezing critic params.
        for param in self.critic.parameters():
            param.requires_grad_(False)
        actor_q_values = []
        for b in range(obs.shape[0]):
            edge_index = batch["edge_index"][b].long()
            edge_attr = batch["edge_attr"][b].float()
            embed = self._train_embed(obs[b], edge_index, edge_attr)
            current_lower = self.actor(embed, upper_action[b])
            actor_q_values.append(self.critic(embed, upper_action[b], current_lower))
        actor_loss = -torch.stack(actor_q_values).mean()
        encoder_actor_term = actor_loss * float(self.joint_encoder_loss_weight)

        self._zero_lower_optimizers(include_encoder=self._encoder_trainable())
        actor_loss.backward(retain_graph=self._encoder_trainable())
        if self._encoder_trainable():
            self._zero_encoder_grads_only()
            encoder_actor_term.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        enc_grad_norm_actor = self._clip_encoder_grad()
        self.actor_optimizer.step()
        self._step_encoder_optimizer()
        for param in self.critic.parameters():
            param.requires_grad_(True)

        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        if self.target_encoder is not None:
            self._soft_update(self.target_encoder, self.encoder)

        return {
            "lower_actor_loss": float(actor_loss.detach().cpu()),
            "lower_critic_loss": float(critic_loss.detach().cpu()),
            "lower_encoder_grad_norm": float(max(enc_grad_norm_critic, enc_grad_norm_actor)),
            "shared_encoder_grad_norm_from_lower": float(max(enc_grad_norm_critic, enc_grad_norm_actor)) if self.encoder_mode == "shared_joint" else 0.0,
            "separate_lower_encoder_grad_norm": float(max(enc_grad_norm_critic, enc_grad_norm_actor)) if self.encoder_mode == "separate_lower_encoder" else 0.0,
            "lower_actor_grad_norm": float(self._grad_norm(self.actor.parameters())),
            "lower_critic_grad_norm": float(self._grad_norm(self.critic.parameters())),
            "lower_encoder_mode": self.encoder_mode,
            "lower_observation_mode": self.lower_observation_mode,
            "lower_training_stop_gradient_to_encoder": bool(self.stop_gradient_to_encoder_from_lower),
            "lower_actor_conditioned_on_upper_action": True,
            "lower_critic_input_schema": "flatten(all_agent_embedding, upper_action_onehot, lower_action)",
            "lower_critic_n_agents_fixed_at_init": int(getattr(self.critic, "n_agents", 0) or 0),
            "lower_q_mean": float(q.detach().mean().cpu()),
            "lower_q_target_mean": float(target_q.detach().mean().cpu()),
        }

    def _encoder_trainable(self) -> bool:
        return (
            self.encoder_optimizer is not None
            and self.encoder_mode in {"shared_joint", "separate_lower_encoder"}
            and (self.encoder_mode == "separate_lower_encoder" or not self.stop_gradient_to_encoder_from_lower)
        )

    def _train_embed(self, obs: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        try:
            embed = self.encoder(obs, edge_index, edge_attr, update_state=False)
        except TypeError:
            embed = self.encoder(obs, edge_index, edge_attr)
        if self.encoder_mode == "shared_upper_only" or (
            self.encoder_mode == "shared_joint" and self.stop_gradient_to_encoder_from_lower
        ):
            return embed.detach()
        return embed

    @torch.no_grad()
    def _target_embed(self, obs: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if self.target_encoder is not None:
            try:
                return self.target_encoder(obs, edge_index, edge_attr, update_state=False)
            except TypeError:
                return self.target_encoder(obs, edge_index, edge_attr)
        try:
            return self.encoder(obs, edge_index, edge_attr, update_state=False)
        except TypeError:
            return self.encoder(obs, edge_index, edge_attr)

    @torch.no_grad()
    def _act_embed(
        self,
        fallback_embed: torch.Tensor,
        *,
        obs: torch.Tensor | None,
        edge_index: torch.Tensor | None,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.encoder_mode != "separate_lower_encoder" and self.lower_observation_mode != "raw_obs_reencode":
            return fallback_embed
        if obs is None or edge_index is None or edge_attr is None:
            return fallback_embed
        try:
            return self.encoder(obs, edge_index, edge_attr, update_state=True)
        except TypeError:
            return self.encoder(obs, edge_index, edge_attr)

    def _zero_lower_optimizers(self, *, include_encoder: bool) -> None:
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        if include_encoder and self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()

    def _zero_encoder_grads_only(self) -> None:
        for p in self.encoder.parameters():
            p.grad = None

    def _clip_encoder_grad(self) -> float:
        if not self._encoder_trainable():
            return 0.0
        return float(nn.utils.clip_grad_norm_(self.encoder.parameters(), 5.0).detach().cpu())

    def _step_encoder_optimizer(self) -> None:
        if self._encoder_trainable() and self.encoder_optimizer is not None:
            self.encoder_optimizer.step()

    @staticmethod
    def _canonical_encoder_mode(mode: object) -> str:
        raw = str(mode or "shared_upper_only").strip().lower()
        aliases = {
            "shared_frozen": "shared_upper_only",
            "shared_upper_only": "shared_upper_only",
            "shared_joint": "shared_joint",
            "separate": "separate_lower_encoder",
            "separate_lower_encoder": "separate_lower_encoder",
        }
        return aliases.get(raw, "shared_upper_only")

    @staticmethod
    def _grad_norm(parameters) -> float:
        total = 0.0
        for p in parameters:
            if p.grad is None:
                continue
            total += float(p.grad.detach().pow(2).sum().cpu())
        return float(total ** 0.5)

    def _soft_update(self, target: nn.Module, src: nn.Module) -> None:
        with torch.no_grad():
            for target_param, src_param in zip(target.parameters(), src.parameters()):
                target_param.data.mul_(1.0 - self.cfg.tau).add_(src_param.data, alpha=self.cfg.tau)

    @staticmethod
    def _hard_update(target: nn.Module, src: nn.Module) -> None:
        target.load_state_dict(src.state_dict())
