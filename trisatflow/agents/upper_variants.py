from __future__ import annotations

import math
from typing import Dict, Literal

import torch
import torch.nn.functional as F
from torch import nn

from trisatflow.agents.replay import ReplayBuffer, RolloutBuffer
from trisatflow.config import AlgoConfig
from trisatflow.models import AgentValue, QMixer, TopologyEncoder, UpperMAPPOPolicy, UpperQNetwork, upper_action_mask_from_obs


class UpperIPPOAgent:
    """IPPO-style upper layer for discrete global offloading.

    This keeps the same decentralized categorical policy as MAPPO but replaces
    the central value critic with per-agent value heads. It is a useful
    independent-policy ablation against MAPPO's centralized critic.
    """

    is_off_policy = False

    def __init__(self, encoder: TopologyEncoder, actor: UpperMAPPOPolicy, value: AgentValue, cfg: AlgoConfig, device: torch.device):
        self.encoder = encoder
        self.actor = actor
        self.value = value
        self.cfg = cfg
        self.device = device
        self.update_step = 0
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.value.parameters()),
            lr=cfg.upper_lr,
        )

    def _encode_obs(
        self,
        obs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        *,
        update_state: bool,
    ) -> torch.Tensor:
        try:
            return self.encoder(obs, edge_index, edge_attr, update_state=update_state)
        except TypeError:
            return self.encoder(obs, edge_index, edge_attr)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, episode: int = 0):
        embed = self._encode_obs(obs, edge_index, edge_attr, update_state=True)
        action_mask = upper_action_mask_from_obs(obs)
        dist = self.actor(embed, action_mask=action_mask, obs=obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value(embed)
        return action, log_prob, value, embed

    def evaluate_actions(self, obs, edge_index, edge_attr, actions):
        embed = self._encode_obs(obs, edge_index, edge_attr, update_state=False)
        action_mask = upper_action_mask_from_obs(obs)
        dist = self.actor(embed, action_mask=action_mask, obs=obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value(embed)
        return log_prob, entropy, values

    def update(self, buffer: RolloutBuffer, replay: ReplayBuffer | None = None) -> Dict[str, float]:
        if len(buffer) == 0:
            return {
                "upper_loss": 0.0,
                "upper_actor_loss": 0.0,
                "upper_policy_loss": 0.0,
                "upper_value_loss": 0.0,
                "upper_entropy": 0.0,
                "upper_approx_kl": 0.0,
                "upper_clip_fraction": 0.0,
                "upper_explained_variance": 0.0,
                "upper_ppo_epochs_ran": 0.0,
                "upper_num_minibatches": 0.0,
                "upper_early_stop_kl": 0.0,
            }
        rewards = torch.stack(buffer.reward).to(self.device)  # [T, N]
        old_values = torch.stack(buffer.value).to(self.device)
        if old_values.ndim == 1:
            old_values = old_values.unsqueeze(-1).expand(-1, rewards.shape[1])
        dones = torch.tensor(buffer.done, dtype=torch.float32, device=self.device)
        normalize_adv = self._advantage_normalization_enabled() if self._ppo_update_mode() == "standard_ppo" else True
        returns, advantages = self._gae(rewards, old_values, dones, normalize=normalize_adv)
        value_loss_scale = self._value_loss_scale(returns)
        old_log_probs = torch.stack(buffer.log_prob).detach().to(self.device)  # [T, N]

        prev_modes = (self.encoder.training, self.actor.training, self.value.training)
        self.encoder.train(True)
        self.actor.train(True)
        self.value.train(True)
        try:
            if self._ppo_update_mode() == "legacy_compact":
                summary = self._legacy_update(buffer, old_log_probs, old_values, returns, advantages, value_loss_scale)
            else:
                summary = self._standard_update(buffer, old_log_probs, old_values, returns, advantages, value_loss_scale)
        finally:
            self.encoder.train(prev_modes[0])
            self.actor.train(prev_modes[1])
            self.value.train(prev_modes[2])

        with torch.no_grad():
            self.encoder.train(False)
            self.actor.train(False)
            self.value.train(False)
            pred_values = []
            ratio_terms = []
            for idx in range(len(buffer)):
                obs = buffer.obs[idx].to(self.device)
                edge_index = buffer.edge_index[idx].to(self.device)
                edge_attr = buffer.edge_attr[idx].to(self.device)
                action = buffer.upper_action[idx].to(self.device)
                new_log_prob, _entropy, values = self.evaluate_actions(obs, edge_index, edge_attr, action)
                pred_values.append(values)
                ratio_terms.append(torch.exp(new_log_prob - old_log_probs[idx]))
            pred_flat = torch.cat(pred_values, dim=0) if pred_values else torch.zeros((0,), device=self.device)
            return_flat = returns.reshape(-1)
            var_y = torch.var(return_flat, unbiased=False) if return_flat.numel() > 0 else torch.tensor(0.0, device=self.device)
            if float(var_y.detach().cpu()) <= 1.0e-12:
                explained = 0.0
            else:
                explained = float((1.0 - torch.var(return_flat - pred_flat, unbiased=False) / var_y).detach().cpu())
            raw_value_loss = float(torch.mean((return_flat.double() - pred_flat.double()).pow(2)).detach().cpu()) if return_flat.numel() > 0 else 0.0
            ratio_mean = float(torch.cat(ratio_terms, dim=0).mean().detach().cpu()) if ratio_terms else 0.0
            ratio_std = float(torch.cat(ratio_terms, dim=0).std(unbiased=False).detach().cpu()) if ratio_terms else 0.0
        self.encoder.train(prev_modes[0])
        self.actor.train(prev_modes[1])
        self.value.train(prev_modes[2])

        out = {
            "upper_loss": float(summary["upper_loss"]),
            "upper_actor_loss": float(summary["upper_policy_loss"]),
            "upper_policy_loss": float(summary["upper_policy_loss"]),
            "upper_value_loss": float(summary["upper_value_loss"]),
            "upper_value_loss_raw": raw_value_loss,
            "upper_value_loss_scale": float(value_loss_scale),
            "upper_entropy": float(summary["upper_entropy"]),
            "upper_approx_kl": float(summary["upper_approx_kl"]),
            "upper_clip_fraction": float(summary["upper_clip_fraction"]),
            "upper_grad_norm": float(summary["upper_grad_norm"]),
            "upper_explained_variance": explained,
            "upper_ppo_epochs_ran": float(summary["upper_ppo_epochs_ran"]),
            "upper_num_minibatches": float(summary["upper_num_minibatches"]),
            "upper_early_stop_kl": float(summary["upper_early_stop_kl"]),
            "upper_credit_assignment": "per_agent",
            "upper_critic_scope": "decentralized_per_agent",
            "upper_advantage_mean": float(advantages.mean().detach().cpu()),
            "upper_advantage_std": float(advantages.std(unbiased=False).detach().cpu()),
            "upper_return_mean": float(returns.mean().detach().cpu()),
            "upper_return_std": float(returns.std(unbiased=False).detach().cpu()),
            "upper_value_mean": float(old_values.mean().detach().cpu()),
            "upper_value_std": float(old_values.std(unbiased=False).detach().cpu()),
            "upper_ratio_mean": ratio_mean,
            "upper_ratio_std": ratio_std,
            "upper_policy_head": str(getattr(self.actor, "policy_head", "gnn_only")),
            "policy_loss": float(summary["upper_policy_loss"]),
            "value_loss": float(summary["upper_value_loss"]),
            "value_loss_raw": raw_value_loss,
            "entropy": float(summary["upper_entropy"]),
            "approx_kl": float(summary["upper_approx_kl"]),
            "clip_fraction": float(summary["upper_clip_fraction"]),
            "explained_variance": explained,
        }
        return out

    def _legacy_update(
        self,
        buffer: RolloutBuffer,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        value_loss_scale: float,
    ) -> Dict[str, float]:
        losses = []
        actor_losses = []
        value_losses = []
        entropy_terms = []
        approx_kl_terms = []
        clip_fraction_terms = []

        for idx in range(len(buffer)):
            obs = buffer.obs[idx].to(self.device)
            edge_index = buffer.edge_index[idx].to(self.device)
            edge_attr = buffer.edge_attr[idx].to(self.device)
            action = buffer.upper_action[idx].to(self.device)
            new_log_prob, entropy, values = self.evaluate_actions(obs, edge_index, edge_attr, action)
            ratio = torch.exp(new_log_prob - old_log_probs[idx])
            adv = advantages[idx].detach()
            clipped = torch.clamp(ratio, 1.0 - self._clip_param(), 1.0 + self._clip_param()) * adv
            actor_loss = -torch.minimum(ratio * adv, clipped).mean()
            value_loss = (((values - returns[idx].detach()) / value_loss_scale) ** 2).mean()
            entropy_mean = entropy.mean()
            loss = actor_loss + self._value_loss_coef() * value_loss - self._entropy_coef() * entropy_mean
            losses.append(loss)
            actor_losses.append(actor_loss.detach())
            value_losses.append(value_loss.detach())
            entropy_terms.append(entropy_mean.detach())
            approx_kl_terms.append((old_log_probs[idx] - new_log_prob).mean().detach())
            clip_fraction_terms.append((torch.abs(ratio - 1.0) > self._clip_param()).float().mean().detach())

        total_loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.value.parameters()), self._max_grad_norm())
        self.optimizer.step()
        self.update_step += 1
        return {
            "upper_loss": float(total_loss.detach().cpu()),
            "upper_policy_loss": float(torch.stack(actor_losses).mean().cpu()),
            "upper_value_loss": float(torch.stack(value_losses).mean().cpu()),
            "upper_entropy": float(torch.stack(entropy_terms).mean().cpu()),
            "upper_approx_kl": float(torch.stack(approx_kl_terms).mean().cpu()),
            "upper_clip_fraction": float(torch.stack(clip_fraction_terms).mean().cpu()),
            "upper_grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            "upper_ppo_epochs_ran": 1.0,
            "upper_num_minibatches": float(len(buffer)),
            "upper_early_stop_kl": 0.0,
        }

    def _standard_update(
        self,
        buffer: RolloutBuffer,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        value_loss_scale: float,
    ) -> Dict[str, float]:
        steps = len(buffer)
        n_agents = int(old_log_probs.shape[1])
        mb_size = self._minibatch_size()
        steps_per_mb = max(1, min(steps, int(math.ceil(float(mb_size) / float(max(1, n_agents)))))) if mb_size > 0 else steps
        epochs = self._ppo_epochs()

        loss_terms = []
        policy_terms = []
        value_terms = []
        entropy_terms = []
        kl_terms = []
        clip_terms = []
        grad_terms = []
        early_stop = False
        minibatches = 0
        epochs_ran = 0

        for ep in range(epochs):
            epochs_ran = ep + 1
            perm = torch.randperm(steps, device=self.device)
            for start in range(0, steps, steps_per_mb):
                mb_idx = perm[start : start + steps_per_mb].tolist()
                if not mb_idx:
                    continue
                mb_losses = []
                mb_policies = []
                mb_values = []
                mb_entropy = []
                old_lp_cat = []
                new_lp_cat = []
                ratio_cat = []
                for idx in mb_idx:
                    obs = buffer.obs[idx].to(self.device)
                    edge_index = buffer.edge_index[idx].to(self.device)
                    edge_attr = buffer.edge_attr[idx].to(self.device)
                    action = buffer.upper_action[idx].to(self.device)
                    new_log_prob, entropy, values = self.evaluate_actions(obs, edge_index, edge_attr, action)
                    ratio = torch.exp(new_log_prob - old_log_probs[idx])
                    adv = advantages[idx].detach()
                    clipped = torch.clamp(ratio, 1.0 - self._clip_param(), 1.0 + self._clip_param()) * adv
                    actor_loss = -torch.minimum(ratio * adv, clipped).mean()

                    value_target = returns[idx].detach()
                    value_unclipped = ((values - value_target) / value_loss_scale).pow(2)
                    value_clipped_pred = old_values[idx] + torch.clamp(values - old_values[idx], -self._value_clip_param(), self._value_clip_param())
                    value_clipped = ((value_clipped_pred - value_target) / value_loss_scale).pow(2)
                    value_loss = 0.5 * torch.maximum(value_unclipped, value_clipped).mean()

                    entropy_mean = entropy.mean()
                    loss = actor_loss + self._value_loss_coef() * value_loss - self._entropy_coef() * entropy_mean
                    mb_losses.append(loss)
                    mb_policies.append(actor_loss.detach())
                    mb_values.append(value_loss.detach())
                    mb_entropy.append(entropy_mean.detach())
                    old_lp_cat.append(old_log_probs[idx].detach())
                    new_lp_cat.append(new_log_prob.detach())
                    ratio_cat.append(ratio.detach())

                if not mb_losses:
                    continue
                loss = torch.stack(mb_losses).mean()
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.value.parameters()), self._max_grad_norm())
                self.optimizer.step()
                self.update_step += 1
                minibatches += 1

                old_cat = torch.cat(old_lp_cat, dim=0)
                new_cat = torch.cat(new_lp_cat, dim=0)
                ratio_cat_t = torch.cat(ratio_cat, dim=0)
                approx_kl = (old_cat - new_cat).mean().detach()
                clip_frac = (torch.abs(ratio_cat_t - 1.0) > self._clip_param()).float().mean().detach()

                loss_terms.append(loss.detach())
                policy_terms.append(torch.stack(mb_policies).mean())
                value_terms.append(torch.stack(mb_values).mean())
                entropy_terms.append(torch.stack(mb_entropy).mean())
                kl_terms.append(approx_kl)
                clip_terms.append(clip_frac)
                grad_terms.append(torch.as_tensor(grad_norm).detach())

                if self._target_kl() > 0.0 and float(approx_kl.cpu()) > self._target_kl():
                    early_stop = True
                    break
            if early_stop:
                break

        if minibatches == 0:
            return {
                "upper_loss": 0.0,
                "upper_policy_loss": 0.0,
                "upper_value_loss": 0.0,
                "upper_entropy": 0.0,
                "upper_approx_kl": 0.0,
                "upper_clip_fraction": 0.0,
                "upper_grad_norm": 0.0,
                "upper_ppo_epochs_ran": float(epochs_ran),
                "upper_num_minibatches": 0.0,
                "upper_early_stop_kl": float(early_stop),
            }
        return {
            "upper_loss": float(torch.stack(loss_terms).mean().cpu()),
            "upper_policy_loss": float(torch.stack(policy_terms).mean().cpu()),
            "upper_value_loss": float(torch.stack(value_terms).mean().cpu()),
            "upper_entropy": float(torch.stack(entropy_terms).mean().cpu()),
            "upper_approx_kl": float(torch.stack(kl_terms).mean().cpu()),
            "upper_clip_fraction": float(torch.stack(clip_terms).mean().cpu()),
            "upper_grad_norm": float(torch.stack(grad_terms).mean().cpu()),
            "upper_ppo_epochs_ran": float(epochs_ran),
            "upper_num_minibatches": float(minibatches),
            "upper_early_stop_kl": float(early_stop),
        }

    def _entropy_coef(self) -> float:
        schedule = str(self.cfg.entropy_coef_schedule or "").strip().lower()
        if schedule != "linear_decay":
            return float(self.cfg.entropy_coef)
        frac = min(1.0, self.update_step / max(1, self.cfg.epsilon_decay_episodes))
        return float(self.cfg.entropy_coef * (1.0 - 0.8 * frac))

    def _gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, *, normalize: bool):
        # Support both [T] and [T, N] inputs so debug/eval paths for
        # independent critics do not crash on singleton-agent views.
        squeeze_last_dim = rewards.dim() == 1
        if squeeze_last_dim:
            rewards = rewards.unsqueeze(-1)
            values = values.unsqueeze(-1)
            dones = dones.unsqueeze(-1)

        returns = torch.zeros_like(rewards)
        adv = torch.zeros_like(rewards)
        next_value = torch.zeros((rewards.shape[1],), device=self.device)
        next_adv = torch.zeros((rewards.shape[1],), device=self.device)
        for t in reversed(range(rewards.shape[0])):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_value * mask - values[t]
            next_adv = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * next_adv
            adv[t] = next_adv
            returns[t] = adv[t] + values[t]
            next_value = values[t]
        if normalize:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)
        if squeeze_last_dim:
            returns = returns.squeeze(-1)
            adv = adv.squeeze(-1)
        return returns, adv

    def _ppo_update_mode(self) -> str:
        mode = str(getattr(self.cfg, "ppo_update_mode", "standard_ppo") or "standard_ppo").strip().lower()
        if mode not in {"standard_ppo", "legacy_compact"}:
            return "standard_ppo"
        return mode

    def _ppo_epochs(self) -> int:
        return max(1, int(getattr(self.cfg, "ppo_epochs", 1) or 1))

    def _minibatch_size(self) -> int:
        return max(0, int(getattr(self.cfg, "minibatch_size", 0) or 0))

    def _clip_param(self) -> float:
        return float(getattr(self.cfg, "clip_param", getattr(self.cfg, "ppo_clip", 0.2)))

    def _value_clip_param(self) -> float:
        return float(getattr(self.cfg, "value_clip_param", self._clip_param()))

    def _value_loss_coef(self) -> float:
        return float(getattr(self.cfg, "value_loss_coef", getattr(self.cfg, "value_coef", 0.5)))

    def _max_grad_norm(self) -> float:
        return float(getattr(self.cfg, "max_grad_norm", 5.0) or 5.0)

    def _target_kl(self) -> float:
        return float(getattr(self.cfg, "target_kl", 0.0) or 0.0)

    def _advantage_normalization_enabled(self) -> bool:
        return bool(getattr(self.cfg, "advantage_normalization", True))

    def _value_loss_scale(self, returns: torch.Tensor) -> float:
        mode = str(getattr(self.cfg, "value_loss_rescale_mode", "batch_std") or "batch_std").strip().lower()
        eps = float(getattr(self.cfg, "value_loss_rescale_eps", 1.0e-6) or 1.0e-6)
        if mode == "none":
            return 1.0
        if returns.numel() == 0:
            return 1.0
        if mode == "batch_rms":
            scale = float(torch.sqrt(torch.mean(returns.pow(2))).detach().cpu())
        else:
            scale = float(torch.std(returns, unbiased=False).detach().cpu())
        return max(eps, scale)


class UpperValueDecompositionAgent:
    """IQL/VDN/QMIX-style upper discrete offloading agent.

    This is a dependency-light implementation aligned with the BenchMARL
    families. It is not a replacement for the full BenchMARL+TorchRL trainer,
    but it lets the TriSatFlow prototype run algorithm-combination sweeps in
    environments where TorchRL/TensorDict are unavailable.
    """

    is_off_policy = True

    def __init__(
        self,
        mode: Literal["iql", "vdn", "qmix"],
        encoder: TopologyEncoder,
        q_net: UpperQNetwork,
        target_encoder: TopologyEncoder,
        target_q_net: UpperQNetwork,
        cfg: AlgoConfig,
        device: torch.device,
        mixer: QMixer | None = None,
        target_mixer: QMixer | None = None,
    ):
        self.mode = mode
        self.encoder = encoder
        self.q_net = q_net
        self.target_encoder = target_encoder
        self.target_q_net = target_q_net
        self.mixer = mixer
        self.target_mixer = target_mixer
        self.cfg = cfg
        self.device = device
        params = list(self.encoder.parameters()) + list(self.q_net.parameters())
        if self.mixer is not None:
            params += list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=cfg.upper_lr)
        self._hard_update(self.target_encoder, self.encoder)
        self._hard_update(self.target_q_net, self.q_net)
        if self.mixer is not None and self.target_mixer is not None:
            self._hard_update(self.target_mixer, self.mixer)

    def _encode_obs(
        self,
        obs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        *,
        update_state: bool,
    ) -> torch.Tensor:
        try:
            return self.encoder(obs, edge_index, edge_attr, update_state=update_state)
        except TypeError:
            return self.encoder(obs, edge_index, edge_attr)

    def _encode_target_obs(self, obs: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        try:
            return self.target_encoder(obs, edge_index, edge_attr, update_state=False)
        except TypeError:
            return self.target_encoder(obs, edge_index, edge_attr)

    def epsilon(self, episode: int) -> float:
        frac = min(1.0, max(0.0, episode / max(1, self.cfg.epsilon_decay_episodes)))
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, episode: int = 0):
        embed = self._encode_obs(obs, edge_index, edge_attr, update_state=True)
        q = self.q_net(embed)
        action_mask = upper_action_mask_from_obs(obs)
        masked_q = q.masked_fill(~action_mask, torch.finfo(q.dtype).min / 4)
        greedy = masked_q.argmax(dim=-1)
        eps = self.epsilon(episode)
        random_dist = torch.distributions.Categorical(probs=action_mask.float() / action_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0))
        random_actions = random_dist.sample()
        explore_mask = torch.rand(greedy.shape, device=obs.device) < eps
        action = torch.where(explore_mask, random_actions, greedy)
        chosen_q = q.gather(-1, action.view(-1, 1)).squeeze(-1)
        pseudo_log_prob = torch.zeros_like(chosen_q)
        pseudo_value = chosen_q.mean().detach()
        return action, pseudo_log_prob, pseudo_value, embed

    def update(self, buffer: RolloutBuffer | None = None, replay: ReplayBuffer | None = None) -> Dict[str, float]:
        if replay is None or len(replay) < max(self.cfg.upper_warmup, self.cfg.upper_batch_size):
            return {"upper_loss": 0.0, "upper_q_loss": 0.0, "upper_epsilon": self.cfg.epsilon_start}
        batch = replay.sample(self.cfg.upper_batch_size, self.device)
        losses = []
        for b in range(batch["obs"].shape[0]):
            obs = batch["obs"][b]
            next_obs = batch["next_obs"][b]
            edge_index = batch["edge_index"][b].long()
            edge_attr = batch["edge_attr"][b].float()
            next_edge_index = batch["next_edge_index"][b].long()
            next_edge_attr = batch["next_edge_attr"][b].float()
            action = batch["upper_action"][b].long()
            reward_agents = batch["upper_reward"][b].float()
            done = batch["done"][b].float().view(())

            embed = self._encode_obs(obs, edge_index, edge_attr, update_state=False)
            q = self.q_net(embed)
            chosen_q = q.gather(-1, action.view(-1, 1)).squeeze(-1)
            with torch.no_grad():
                next_embed = self._encode_target_obs(next_obs, next_edge_index, next_edge_attr)
                next_mask = upper_action_mask_from_obs(next_obs)
                next_q_all = self.target_q_net(next_embed)
                next_q = next_q_all.masked_fill(~next_mask, torch.finfo(next_q_all.dtype).min / 4).max(dim=-1).values
                target_agents = reward_agents + self.cfg.gamma * (1.0 - done) * next_q

            if self.mode == "iql":
                losses.append(F.mse_loss(chosen_q, target_agents.detach()))
            else:
                reward_total = reward_agents.mean()
                if self.mode == "vdn":
                    total_q = chosen_q.sum()
                    with torch.no_grad():
                        total_target_q = reward_total + self.cfg.gamma * (1.0 - done) * next_q.sum()
                else:
                    assert self.mixer is not None and self.target_mixer is not None
                    total_q = self.mixer(chosen_q, embed)
                    with torch.no_grad():
                        total_target_q = reward_total + self.cfg.gamma * (1.0 - done) * self.target_mixer(next_q, next_embed)
                losses.append(F.mse_loss(total_q, total_target_q.detach()))

        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        params = list(self.encoder.parameters()) + list(self.q_net.parameters())
        if self.mixer is not None:
            params += list(self.mixer.parameters())
        nn.utils.clip_grad_norm_(params, 5.0)
        self.optimizer.step()
        self._soft_update(self.target_encoder, self.encoder)
        self._soft_update(self.target_q_net, self.q_net)
        if self.mixer is not None and self.target_mixer is not None:
            self._soft_update(self.target_mixer, self.mixer)
        return {"upper_loss": float(loss.detach().cpu()), "upper_q_loss": float(loss.detach().cpu())}

    def _soft_update(self, target: nn.Module, src: nn.Module) -> None:
        with torch.no_grad():
            for target_param, src_param in zip(target.parameters(), src.parameters()):
                target_param.data.mul_(1.0 - self.cfg.tau).add_(src_param.data, alpha=self.cfg.tau)

    @staticmethod
    def _hard_update(target: nn.Module, src: nn.Module) -> None:
        target.load_state_dict(src.state_dict())
