from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from trisatflow.agents.replay import RolloutBuffer
from trisatflow.config import AlgoConfig, PolicyRegularizationConfig
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
    SHARED_NODE_FEATURE_DIM_WITH_COST,
)
from trisatflow.models import TopologyEncoder, UpperMAPPOPolicy, upper_action_mask_from_obs
from trisatflow.models.policies import masked_policy_logits_and_probs


class UpperMAPPOAgent:
    """Small MAPPO updater for the upper discrete offloading layer."""

    is_off_policy = False

    def __init__(
        self,
        encoder: TopologyEncoder,
        actor: UpperMAPPOPolicy,
        critic: nn.Module,
        cfg: AlgoConfig,
        policy_regularization: PolicyRegularizationConfig,
        device: torch.device,
    ):
        self.encoder = encoder
        self.actor = actor
        self.critic = critic
        self.cfg = cfg
        self.reg_cfg = policy_regularization
        self.device = device
        self.update_step = 0
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.upper_lr,
        )
        self.last_update_diagnostics: Dict[str, float] = {}

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
        value = self.critic(embed)
        return action, log_prob, value, embed

    def evaluate_actions(self, obs, edge_index, edge_attr, actions):
        embed = self._encode_obs(obs, edge_index, edge_attr, update_state=False)
        action_mask = upper_action_mask_from_obs(obs)
        logits, actor_details = self.actor.compute_logits(embed, obs=obs, return_details=True)
        masked_logits, probs = masked_policy_logits_and_probs(logits, action_mask)
        dist = torch.distributions.Categorical(probs=probs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.critic(embed)
        return log_prob, entropy, value, probs, action_mask, logits, actor_details

    def _build_cost_prior(self, obs: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] >= SHARED_NODE_FEATURE_DIM_WITH_COST:
            costs = torch.stack(
                [
                    obs[:, IDX_LOCAL_NORMALIZED_COST],
                    obs[:, IDX_NEIGHBOR_NORMALIZED_COST],
                    obs[:, IDX_GEO_NORMALIZED_COST],
                    obs[:, IDX_GROUND_NORMALIZED_COST],
                ],
                dim=-1,
            ).to(self.device)
        else:
            # Backward-compatible fallback when the observation does not carry
            # explicit normalized per-tier cost features.
            visible = torch.stack(
                [
                    obs[:, IDX_LOCAL_VISIBLE],
                    obs[:, IDX_NEIGHBOR_VISIBLE],
                    obs[:, IDX_GEO_VISIBLE],
                    obs[:, IDX_GROUND_VISIBLE],
                ],
                dim=-1,
            ).to(self.device)
            rates = torch.stack(
                [
                    obs[:, IDX_LOCAL_RATE],
                    obs[:, IDX_NEIGHBOR_RATE],
                    obs[:, IDX_GEO_RATE],
                    obs[:, IDX_GROUND_RATE],
                ],
                dim=-1,
            ).to(self.device)
            delays = torch.stack(
                [
                    obs[:, IDX_LOCAL_DELAY],
                    obs[:, IDX_NEIGHBOR_DELAY],
                    obs[:, IDX_GEO_DELAY],
                    obs[:, IDX_GROUND_DELAY],
                ],
                dim=-1,
            ).to(self.device)
            queues = torch.stack(
                [
                    obs[:, IDX_LOCAL_QUEUE],
                    obs[:, IDX_NEIGHBOR_QUEUE],
                    obs[:, IDX_GEO_QUEUE],
                    obs[:, IDX_GROUND_QUEUE],
                ],
                dim=-1,
            ).to(self.device)
            tx_proxy = torch.zeros_like(rates)
            tx_proxy[:, 1:] = 1.0 / rates[:, 1:].clamp_min(1.0e-6)
            compute_proxy = torch.relu(delays - tx_proxy)
            raw = delays + 0.5 * queues + 0.2 * tx_proxy + 0.2 * compute_proxy
            raw = raw.masked_fill(visible <= 0.5, torch.finfo(raw.dtype).max / 4)
            row_min = raw.min(dim=-1, keepdim=True).values
            row_max = raw.max(dim=-1, keepdim=True).values
            denom = (row_max - row_min).clamp_min(1.0e-6)
            costs = torch.where(visible > 0.5, (raw - row_min) / denom, torch.ones_like(raw))

        masked_cost = costs.masked_fill(~action_mask.bool(), torch.finfo(costs.dtype).max / 4)
        logits = -masked_cost / max(1.0e-6, float(self.reg_cfg.temperature))
        logits = logits.masked_fill(~action_mask.bool(), torch.finfo(costs.dtype).min / 4)
        return torch.softmax(logits, dim=-1)

    def update(self, buffer: RolloutBuffer, replay=None) -> Dict[str, float]:
        if len(buffer) == 0:
            return {
                "upper_loss": 0.0,
                "upper_actor_loss": 0.0,
                "upper_policy_loss": 0.0,
                "upper_value_loss": 0.0,
                "upper_entropy": 0.0,
                "upper_approx_kl": 0.0,
                "upper_clip_fraction": 0.0,
                "upper_grad_norm": 0.0,
                "upper_explained_variance": 0.0,
            }

        rewards_agents = torch.stack(buffer.reward).to(self.device)  # [T, N]
        n_agents = int(rewards_agents.shape[1])
        stacked_values = torch.stack(buffer.value).to(self.device)
        if stacked_values.ndim == 1:
            stacked_values = stacked_values.unsqueeze(-1)
        values_agents = self._to_agent_tensor(stacked_values, n_agents=n_agents)
        dones = torch.tensor(buffer.done, dtype=torch.float32, device=self.device)

        credit_mode = self._credit_mode()
        if credit_mode == "per_agent":
            rewards_for_gae = rewards_agents
            values_for_gae = values_agents
        else:
            rewards_for_gae = rewards_agents.mean(dim=-1)
            values_for_gae = values_agents.mean(dim=-1)

        normalize_adv = self._advantage_normalization_enabled() if self._ppo_update_mode() == "standard_ppo" else True
        returns, advantages = self._gae(rewards_for_gae, values_for_gae, dones, normalize=normalize_adv)
        if credit_mode == "per_agent":
            advantages_agents = self._to_agent_tensor(advantages, n_agents=n_agents)
            returns_agents = self._to_agent_tensor(returns, n_agents=n_agents)
        else:
            advantages_agents = advantages.unsqueeze(-1).expand(advantages.shape[0], n_agents)
            returns_agents = returns.unsqueeze(-1).expand(returns.shape[0], n_agents)
        value_loss_scale = self._value_loss_scale(returns_agents)
        old_log_probs = torch.stack(buffer.log_prob).detach().to(self.device)  # [T, N]
        old_values_agents = self._to_agent_tensor(values_agents, n_agents=n_agents).detach()

        prev_modes = (self.encoder.training, self.actor.training, self.critic.training)
        self.encoder.train(True)
        self.actor.train(True)
        self.critic.train(True)
        try:
            if self._ppo_update_mode() == "legacy_compact":
                train_summary = self._update_legacy_compact(
                    buffer,
                    old_log_probs,
                    old_values_agents,
                    returns_agents,
                    advantages_agents,
                    value_loss_scale,
                )
            else:
                train_summary = self._update_standard_ppo(
                    buffer,
                    old_log_probs,
                    old_values_agents,
                    returns_agents,
                    advantages_agents,
                    value_loss_scale,
                )
        finally:
            self.encoder.train(prev_modes[0])
            self.actor.train(prev_modes[1])
            self.critic.train(prev_modes[2])

        update_stats = self._collect_post_update_diagnostics(
            buffer=buffer,
            rewards_agents=rewards_agents,
            old_log_probs=old_log_probs,
            old_values_agents=old_values_agents,
            returns_agents=returns_agents,
            advantages_agents=advantages_agents,
            credit_mode=credit_mode,
            train_summary=train_summary,
            value_loss_scale=value_loss_scale,
        )
        self.last_update_diagnostics = dict(update_stats)
        return update_stats

    def _update_legacy_compact(
        self,
        buffer: RolloutBuffer,
        old_log_probs: torch.Tensor,
        old_values_agents: torch.Tensor,
        returns_agents: torch.Tensor,
        advantages_agents: torch.Tensor,
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
            new_log_prob_agents, entropy_agents, value_raw, probs, action_mask, logits_raw, _details = self.evaluate_actions(
                obs, edge_index, edge_attr, action
            )
            value_agents = self._to_agent_vector(value_raw, n_agents=action.shape[0])
            ratio_agents = torch.exp(new_log_prob_agents - old_log_probs[idx])
            adv_agents = advantages_agents[idx].detach()
            clipped = torch.clamp(ratio_agents, 1.0 - self._clip_param(), 1.0 + self._clip_param()) * adv_agents
            actor_loss = -torch.minimum(ratio_agents * adv_agents, clipped).mean()
            value_delta = (value_agents - returns_agents[idx].detach()) / value_loss_scale
            value_loss = value_delta.pow(2).mean()
            entropy = entropy_agents.mean()
            reg_loss, _ce_loss, _prior_kl, _oracle_idx = self._policy_regularization_loss(
                probs=probs,
                obs=obs,
                action_mask=action_mask,
                idx=idx,
                buffer=buffer,
            )
            action_bias_reg = self._action_bias_regularization(logits_raw)
            loss = actor_loss + self._value_loss_coef() * value_loss - self._entropy_coef() * entropy + reg_loss + action_bias_reg

            losses.append(loss)
            actor_losses.append(actor_loss.detach())
            value_losses.append(value_loss.detach())
            entropy_terms.append(entropy.detach())
            approx_kl_terms.append((old_log_probs[idx] - new_log_prob_agents).mean().detach())
            clip_fraction_terms.append((torch.abs(ratio_agents - 1.0) > self._clip_param()).float().mean().detach())

        total_loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters()),
            self._max_grad_norm(),
        )
        self.optimizer.step()
        self.update_step += 1

        return {
            "upper_loss": float(total_loss.detach().cpu()),
            "upper_policy_loss": float(torch.stack(actor_losses).mean().cpu()) if actor_losses else 0.0,
            "upper_value_loss": float(torch.stack(value_losses).mean().cpu()) if value_losses else 0.0,
            "upper_entropy": float(torch.stack(entropy_terms).mean().cpu()) if entropy_terms else 0.0,
            "upper_approx_kl": float(torch.stack(approx_kl_terms).mean().cpu()) if approx_kl_terms else 0.0,
            "upper_clip_fraction": float(torch.stack(clip_fraction_terms).mean().cpu()) if clip_fraction_terms else 0.0,
            "upper_grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            "upper_ppo_epochs_ran": 1.0,
            "upper_num_minibatches": float(len(buffer)),
            "upper_early_stop_kl": 0.0,
        }

    def _update_standard_ppo(
        self,
        buffer: RolloutBuffer,
        old_log_probs: torch.Tensor,
        old_values_agents: torch.Tensor,
        returns_agents: torch.Tensor,
        advantages_agents: torch.Tensor,
        value_loss_scale: float,
    ) -> Dict[str, float]:
        steps = len(buffer)
        n_agents = int(old_log_probs.shape[1])
        ppo_epochs = self._ppo_epochs()
        steps_per_minibatch = self._steps_per_minibatch(total_steps=steps, n_agents=n_agents)

        total_loss_terms = []
        policy_loss_terms = []
        value_loss_terms = []
        entropy_terms = []
        approx_kl_terms = []
        clip_fraction_terms = []
        grad_norm_terms = []

        early_stop = False
        epochs_ran = 0
        minibatches_ran = 0

        for epoch in range(ppo_epochs):
            epochs_ran = epoch + 1
            perm = torch.randperm(steps, device=self.device)
            for start in range(0, steps, steps_per_minibatch):
                mb_idx = perm[start : start + steps_per_minibatch].tolist()
                if not mb_idx:
                    continue

                loss_terms = []
                actor_terms = []
                value_terms = []
                entropy_mb_terms = []
                old_lp_cat = []
                new_lp_cat = []
                ratio_cat = []

                for idx in mb_idx:
                    obs = buffer.obs[idx].to(self.device)
                    edge_index = buffer.edge_index[idx].to(self.device)
                    edge_attr = buffer.edge_attr[idx].to(self.device)
                    action = buffer.upper_action[idx].to(self.device)

                    new_lp, entropy_agents, value_raw, probs, action_mask, logits_raw, _details = self.evaluate_actions(
                        obs, edge_index, edge_attr, action
                    )
                    value_agents = self._to_agent_vector(value_raw, n_agents=action.shape[0])
                    old_lp = old_log_probs[idx]
                    ratio = torch.exp(new_lp - old_lp)
                    adv = advantages_agents[idx].detach()

                    unclipped_policy = ratio * adv
                    clipped_policy = torch.clamp(ratio, 1.0 - self._clip_param(), 1.0 + self._clip_param()) * adv
                    actor_loss = -torch.minimum(unclipped_policy, clipped_policy).mean()

                    old_value = old_values_agents[idx]
                    value_target = returns_agents[idx].detach()
                    value_unclipped = ((value_agents - value_target) / value_loss_scale).pow(2)
                    value_clipped_pred = old_value + torch.clamp(
                        value_agents - old_value,
                        -self._value_clip_param(),
                        self._value_clip_param(),
                    )
                    value_clipped = ((value_clipped_pred - value_target) / value_loss_scale).pow(2)
                    value_loss = 0.5 * torch.maximum(value_unclipped, value_clipped).mean()

                    entropy = entropy_agents.mean()
                    reg_loss, _ce_loss, _prior_kl, _oracle_idx = self._policy_regularization_loss(
                        probs=probs,
                        obs=obs,
                        action_mask=action_mask,
                        idx=idx,
                        buffer=buffer,
                    )
                    action_bias_reg = self._action_bias_regularization(logits_raw)
                    loss = (
                        actor_loss
                        + self._value_loss_coef() * value_loss
                        - self._entropy_coef() * entropy
                        + reg_loss
                        + action_bias_reg
                    )

                    loss_terms.append(loss)
                    actor_terms.append(actor_loss.detach())
                    value_terms.append(value_loss.detach())
                    entropy_mb_terms.append(entropy.detach())
                    old_lp_cat.append(old_lp.detach())
                    new_lp_cat.append(new_lp.detach())
                    ratio_cat.append(ratio.detach())

                if not loss_terms:
                    continue

                mb_loss = torch.stack(loss_terms).mean()
                self.optimizer.zero_grad()
                mb_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters()),
                    self._max_grad_norm(),
                )
                self.optimizer.step()
                self.update_step += 1
                minibatches_ran += 1

                old_lp_batch = torch.cat(old_lp_cat, dim=0)
                new_lp_batch = torch.cat(new_lp_cat, dim=0)
                ratio_batch = torch.cat(ratio_cat, dim=0)
                approx_kl = (old_lp_batch - new_lp_batch).mean().detach()
                clip_fraction = (torch.abs(ratio_batch - 1.0) > self._clip_param()).float().mean().detach()

                total_loss_terms.append(mb_loss.detach())
                policy_loss_terms.append(torch.stack(actor_terms).mean())
                value_loss_terms.append(torch.stack(value_terms).mean())
                entropy_terms.append(torch.stack(entropy_mb_terms).mean())
                approx_kl_terms.append(approx_kl)
                clip_fraction_terms.append(clip_fraction)
                grad_norm_terms.append(torch.as_tensor(grad_norm).detach())

                if self._target_kl() > 0.0 and float(approx_kl.cpu()) > self._target_kl():
                    early_stop = True
                    break

            if early_stop:
                break

        if minibatches_ran == 0:
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
            "upper_loss": float(torch.stack(total_loss_terms).mean().cpu()),
            "upper_policy_loss": float(torch.stack(policy_loss_terms).mean().cpu()),
            "upper_value_loss": float(torch.stack(value_loss_terms).mean().cpu()),
            "upper_entropy": float(torch.stack(entropy_terms).mean().cpu()),
            "upper_approx_kl": float(torch.stack(approx_kl_terms).mean().cpu()),
            "upper_clip_fraction": float(torch.stack(clip_fraction_terms).mean().cpu()),
            "upper_grad_norm": float(torch.stack(grad_norm_terms).mean().cpu()),
            "upper_ppo_epochs_ran": float(epochs_ran),
            "upper_num_minibatches": float(minibatches_ran),
            "upper_early_stop_kl": float(early_stop),
        }

    def _collect_post_update_diagnostics(
        self,
        *,
        buffer: RolloutBuffer,
        rewards_agents: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values_agents: torch.Tensor,
        returns_agents: torch.Tensor,
        advantages_agents: torch.Tensor,
        credit_mode: str,
        train_summary: Dict[str, float],
        value_loss_scale: float,
    ) -> Dict[str, float]:
        entropy_terms = []
        approx_kl_terms = []
        clip_fraction_terms = []
        ratio_mean_terms = []
        ratio_std_terms = []
        old_logprob_mean_terms = []
        new_logprob_mean_terms = []
        logit_mean_terms: List[torch.Tensor] = []
        logit_std_terms: List[torch.Tensor] = []
        value_pred_terms = []

        reg_losses = []
        ce_terms = []
        prior_kl_terms = []
        action_bias_terms = []

        action_count = torch.zeros((4,), dtype=torch.float32, device=self.device)
        action_reward_sum = torch.zeros((4,), dtype=torch.float32, device=self.device)
        action_adv_sum = torch.zeros((4,), dtype=torch.float32, device=self.device)
        action_return_sum = torch.zeros((4,), dtype=torch.float32, device=self.device)
        action_value_sum = torch.zeros((4,), dtype=torch.float32, device=self.device)

        oracle_prob_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        oracle_agree_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        oracle_count = torch.zeros((), dtype=torch.float32, device=self.device)
        oracle_adv_values: List[torch.Tensor] = []
        oracle_match_values: List[torch.Tensor] = []
        phase_alignment_terms: Dict[str, List[float]] = {}

        prev_modes = (self.encoder.training, self.actor.training, self.critic.training)
        self.encoder.train(False)
        self.actor.train(False)
        self.critic.train(False)
        try:
            with torch.no_grad():
                for idx in range(len(buffer)):
                    obs = buffer.obs[idx].to(self.device)
                    edge_index = buffer.edge_index[idx].to(self.device)
                    edge_attr = buffer.edge_attr[idx].to(self.device)
                    action = buffer.upper_action[idx].to(self.device)

                    new_lp, entropy_agents, value_raw, probs, action_mask, logits_raw, _details = self.evaluate_actions(
                        obs, edge_index, edge_attr, action
                    )
                    value_agents = self._to_agent_vector(value_raw, n_agents=action.shape[0])
                    value_pred_terms.append(value_agents.detach())

                    ratio = torch.exp(new_lp - old_log_probs[idx])
                    entropy_terms.append(entropy_agents.mean().detach())
                    approx_kl_terms.append((old_log_probs[idx] - new_lp).mean().detach())
                    clip_fraction_terms.append((torch.abs(ratio - 1.0) > self._clip_param()).float().mean().detach())
                    ratio_mean_terms.append(ratio.mean().detach())
                    ratio_std_terms.append(ratio.std(unbiased=False).detach())
                    old_logprob_mean_terms.append(old_log_probs[idx].mean().detach())
                    new_logprob_mean_terms.append(new_lp.mean().detach())
                    logit_mean_terms.append(logits_raw.mean(dim=0).detach())
                    logit_std_terms.append(logits_raw.std(dim=0, unbiased=False).detach())

                    reg_loss, ce_loss, prior_kl, oracle_idx = self._policy_regularization_loss(
                        probs=probs,
                        obs=obs,
                        action_mask=action_mask,
                        idx=idx,
                        buffer=buffer,
                    )
                    reg_losses.append(reg_loss.detach())
                    ce_terms.append(ce_loss.detach())
                    prior_kl_terms.append(prior_kl.detach())
                    action_bias_terms.append(self._action_bias_regularization(logits_raw).detach())

                    if oracle_idx is not None:
                        policy_probs = probs.clamp_min(1.0e-9)
                        oracle_prob_sum += policy_probs.gather(1, oracle_idx.view(-1, 1)).sum()
                        oracle_agree_sum += (torch.argmax(policy_probs, dim=-1) == oracle_idx).float().sum()
                        oracle_count += float(policy_probs.shape[0])
                        oracle_adv_values.append(advantages_agents[idx].detach())
                        oracle_match_values.append((action == oracle_idx).float().detach())

                    reward_agents = rewards_agents[idx]
                    adv_agents = advantages_agents[idx].detach()
                    ret_agents = returns_agents[idx].detach()
                    for action_idx in range(4):
                        mask = action == action_idx
                        count = mask.float().sum()
                        if count.item() <= 0:
                            continue
                        action_count[action_idx] += count
                        action_reward_sum[action_idx] += reward_agents[mask].sum().detach()
                        action_adv_sum[action_idx] += adv_agents[mask].sum().detach()
                        action_return_sum[action_idx] += ret_agents[mask].sum().detach()
                        action_value_sum[action_idx] += value_agents[mask].sum().detach()

                    if idx < len(buffer.scenario_phase):
                        phase_labels = buffer.scenario_phase[idx]
                        for aid, phase_name in enumerate(phase_labels):
                            phase_alignment_terms.setdefault(str(phase_name), []).append(float(adv_agents[aid].detach().cpu()))
        finally:
            self.encoder.train(prev_modes[0])
            self.actor.train(prev_modes[1])
            self.critic.train(prev_modes[2])

        value_pred_flat = torch.cat(value_pred_terms, dim=0) if value_pred_terms else torch.zeros((0,), device=self.device)
        returns_flat = returns_agents.reshape(-1)
        explained_variance = self._explained_variance(value_pred_flat, returns_flat)
        raw_value_loss = float(torch.mean((value_pred_flat.double() - returns_flat.double()).pow(2)).detach().cpu()) if value_pred_flat.numel() > 0 else 0.0

        action_count_safe = torch.clamp(action_count, min=1.0)
        action_count_cpu = action_count.detach().cpu()
        has_local = float(action_count_cpu[0].item()) > 0.0
        has_neighbor = float(action_count_cpu[1].item()) > 0.0
        has_geo = float(action_count_cpu[2].item()) > 0.0
        has_ground = float(action_count_cpu[3].item()) > 0.0

        advantage_agent_std = float(advantages_agents.std(unbiased=False).detach().cpu())
        advantage_agent_snr = float(
            advantages_agents.abs().mean().detach().cpu() / (advantages_agents.std(unbiased=False).detach().cpu() + 1.0e-6)
        )

        if oracle_adv_values and oracle_match_values:
            adv_all = torch.cat([t.reshape(-1) for t in oracle_adv_values], dim=0).float()
            match_all = torch.cat([t.reshape(-1) for t in oracle_match_values], dim=0).float()
            if adv_all.numel() > 1 and match_all.std(unbiased=False).item() > 0.0:
                a = adv_all - adv_all.mean()
                b = match_all - match_all.mean()
                denom = torch.sqrt((a.pow(2).sum() * b.pow(2).sum()).clamp_min(1.0e-12))
                advantage_oracle_alignment = float((a * b).sum() / denom)
            else:
                advantage_oracle_alignment = 0.0
        else:
            advantage_oracle_alignment = 0.0

        phase_adv_values = [abs(sum(v) / max(1, len(v))) for _, v in phase_alignment_terms.items() if v]
        phase_advantage_alignment = float(sum(phase_adv_values) / max(1, len(phase_adv_values)))

        update_stats = {
            "upper_loss": float(train_summary.get("upper_loss", 0.0)),
            "upper_actor_loss": float(train_summary.get("upper_policy_loss", 0.0)),
            "upper_policy_loss": float(train_summary.get("upper_policy_loss", 0.0)),
            "upper_value_loss": float(train_summary.get("upper_value_loss", 0.0)),
            "upper_value_loss_raw": raw_value_loss,
            "upper_value_loss_scale": float(value_loss_scale),
            "upper_entropy": float(train_summary.get("upper_entropy", 0.0)),
            "upper_approx_kl": float(train_summary.get("upper_approx_kl", 0.0)),
            "upper_clip_fraction": float(train_summary.get("upper_clip_fraction", 0.0)),
            "upper_grad_norm": float(train_summary.get("upper_grad_norm", 0.0)),
            "upper_ppo_epochs_ran": float(train_summary.get("upper_ppo_epochs_ran", 0.0)),
            "upper_num_minibatches": float(train_summary.get("upper_num_minibatches", 0.0)),
            "upper_early_stop_kl": float(train_summary.get("upper_early_stop_kl", 0.0)),
            "upper_explained_variance": explained_variance,
            "upper_credit_assignment": credit_mode,
            "upper_policy_head": str(getattr(self.actor, "policy_head", "gnn_only")),
            "upper_critic_scope": "centralized_per_agent" if credit_mode == "per_agent" else "centralized_global_team",
            "upper_advantage_mean": float(advantages_agents.mean().detach().cpu()),
            "upper_advantage_std": float(advantages_agents.std(unbiased=False).detach().cpu()),
            "upper_advantage_agent_std": advantage_agent_std,
            "upper_advantage_agent_snr": advantage_agent_snr,
            "upper_return_mean": float(returns_agents.mean().detach().cpu()),
            "upper_return_std": float(returns_agents.std(unbiased=False).detach().cpu()),
            "upper_value_mean": float(old_values_agents.mean().detach().cpu()),
            "upper_value_std": float(old_values_agents.std(unbiased=False).detach().cpu()),
            "upper_ratio_mean": float(torch.stack(ratio_mean_terms).mean().cpu()) if ratio_mean_terms else 0.0,
            "upper_ratio_std": float(torch.stack(ratio_std_terms).mean().cpu()) if ratio_std_terms else 0.0,
            "upper_old_logprob_mean": float(torch.stack(old_logprob_mean_terms).mean().cpu()) if old_logprob_mean_terms else 0.0,
            "upper_new_logprob_mean": float(torch.stack(new_logprob_mean_terms).mean().cpu()) if new_logprob_mean_terms else 0.0,
            "upper_cost_rank_kl_loss": float(torch.stack(reg_losses).mean().cpu()) if reg_losses else 0.0,
            "upper_cost_prior_ce_loss": float(torch.stack(ce_terms).mean().cpu()) if ce_terms else 0.0,
            "upper_policy_cost_prior_kl": float(torch.stack(prior_kl_terms).mean().cpu()) if prior_kl_terms else 0.0,
            "prob_oracle_action_mean": float((oracle_prob_sum / oracle_count.clamp_min(1.0)).detach().cpu()),
            "policy_cost_prior_agreement": float((oracle_agree_sum / oracle_count.clamp_min(1.0)).detach().cpu()),
            "upper_action_bias_reg_loss": float(torch.stack(action_bias_terms).mean().cpu()) if action_bias_terms else 0.0,
            "advantage_oracle_alignment": advantage_oracle_alignment,
            "phase_advantage_alignment": phase_advantage_alignment,
            "mean_logit_local": float(torch.stack(logit_mean_terms).mean(dim=0)[0].cpu()) if logit_mean_terms else 0.0,
            "mean_logit_neighbor": float(torch.stack(logit_mean_terms).mean(dim=0)[1].cpu()) if logit_mean_terms else 0.0,
            "mean_logit_geo": float(torch.stack(logit_mean_terms).mean(dim=0)[2].cpu()) if logit_mean_terms else 0.0,
            "mean_logit_ground": float(torch.stack(logit_mean_terms).mean(dim=0)[3].cpu()) if logit_mean_terms else 0.0,
            "std_logit_local": float(torch.stack(logit_std_terms).mean(dim=0)[0].cpu()) if logit_std_terms else 0.0,
            "std_logit_neighbor": float(torch.stack(logit_std_terms).mean(dim=0)[1].cpu()) if logit_std_terms else 0.0,
            "std_logit_geo": float(torch.stack(logit_std_terms).mean(dim=0)[2].cpu()) if logit_std_terms else 0.0,
            "std_logit_ground": float(torch.stack(logit_std_terms).mean(dim=0)[3].cpu()) if logit_std_terms else 0.0,
            "mean_advantage_local_selected": float((action_adv_sum[0] / action_count_safe[0]).detach().cpu()) if has_local else 0.0,
            "mean_advantage_neighbor_selected": float((action_adv_sum[1] / action_count_safe[1]).detach().cpu()) if has_neighbor else 0.0,
            "mean_advantage_geo_selected": float((action_adv_sum[2] / action_count_safe[2]).detach().cpu()) if has_geo else 0.0,
            "mean_advantage_ground_selected": float((action_adv_sum[3] / action_count_safe[3]).detach().cpu()) if has_ground else 0.0,
            "mean_reward_local_selected": float((action_reward_sum[0] / action_count_safe[0]).detach().cpu()) if has_local else 0.0,
            "mean_reward_neighbor_selected": float((action_reward_sum[1] / action_count_safe[1]).detach().cpu()) if has_neighbor else 0.0,
            "mean_reward_geo_selected": float((action_reward_sum[2] / action_count_safe[2]).detach().cpu()) if has_geo else 0.0,
            "mean_reward_ground_selected": float((action_reward_sum[3] / action_count_safe[3]).detach().cpu()) if has_ground else 0.0,
            "mean_return_local_selected": float((action_return_sum[0] / action_count_safe[0]).detach().cpu()) if has_local else 0.0,
            "mean_return_neighbor_selected": float((action_return_sum[1] / action_count_safe[1]).detach().cpu()) if has_neighbor else 0.0,
            "mean_return_geo_selected": float((action_return_sum[2] / action_count_safe[2]).detach().cpu()) if has_geo else 0.0,
            "mean_return_ground_selected": float((action_return_sum[3] / action_count_safe[3]).detach().cpu()) if has_ground else 0.0,
            "mean_value_local_selected": float((action_value_sum[0] / action_count_safe[0]).detach().cpu()) if has_local else 0.0,
            "mean_value_neighbor_selected": float((action_value_sum[1] / action_count_safe[1]).detach().cpu()) if has_neighbor else 0.0,
            "mean_value_geo_selected": float((action_value_sum[2] / action_count_safe[2]).detach().cpu()) if has_geo else 0.0,
            "mean_value_ground_selected": float((action_value_sum[3] / action_count_safe[3]).detach().cpu()) if has_ground else 0.0,
        }

        update_stats.update(
            {
                "policy_loss": update_stats["upper_policy_loss"],
                "value_loss": update_stats["upper_value_loss"],
                "value_loss_raw": update_stats["upper_value_loss_raw"],
                "entropy": update_stats["upper_entropy"],
                "approx_kl": update_stats["upper_approx_kl"],
                "clip_fraction": update_stats["upper_clip_fraction"],
                "explained_variance": update_stats["upper_explained_variance"],
            }
        )
        return update_stats

    def _policy_regularization_loss(
        self,
        *,
        probs: torch.Tensor,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        idx: int,
        buffer: RolloutBuffer,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        reg_loss = torch.zeros((), device=self.device)
        ce_loss = torch.zeros((), device=self.device)
        prior_kl = torch.zeros((), device=self.device)
        oracle_idx = None
        if not bool(self.reg_cfg.enabled):
            return reg_loss, ce_loss, prior_kl, oracle_idx

        if idx < len(buffer.cost_prior):
            cost_prior = buffer.cost_prior[idx].to(self.device)
        else:
            cost_prior = self._build_cost_prior(obs, action_mask)
        prior = cost_prior.clamp_min(1.0e-9)
        prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
        policy_probs = probs.clamp_min(1.0e-9)
        prior_kl = (policy_probs * (policy_probs.log() - prior.log())).sum(dim=-1).mean()
        mode = str(self.reg_cfg.mode or "").strip().lower()
        if mode == "cost_rank_kl":
            reg_loss = float(self.reg_cfg.weight) * prior_kl
        elif mode == "cost_prior_ce":
            ce_loss = -(prior * policy_probs.log()).sum(dim=-1).mean()
            reg_loss = float(self.reg_cfg.weight) * ce_loss
        oracle_idx = torch.argmax(prior, dim=-1)
        return reg_loss, ce_loss, prior_kl, oracle_idx

    def _action_bias_regularization(self, logits_raw: torch.Tensor) -> torch.Tensor:
        action_bias_reg = torch.zeros((), device=self.device)
        bias_weight = float(getattr(self.cfg, "action_bias_regularization", 0.0) or 0.0)
        if bias_weight <= 0.0:
            return action_bias_reg
        mean_logits = logits_raw.mean(dim=0)
        return bias_weight * mean_logits.pow(2).mean()

    def _entropy_coef(self) -> float:
        schedule = str(self.cfg.entropy_coef_schedule or "").strip().lower()
        if schedule != "linear_decay":
            return float(self.cfg.entropy_coef)
        frac = min(1.0, self.update_step / max(1, self.cfg.epsilon_decay_episodes))
        return float(self.cfg.entropy_coef * (1.0 - 0.8 * frac))

    def _gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, *, normalize: bool):
        rewards = rewards.to(self.device)
        values = values.to(self.device)
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)
        if values.ndim == 1:
            values = values.unsqueeze(-1)
        returns = torch.zeros_like(rewards)
        adv = torch.zeros_like(rewards)
        next_value = torch.zeros((rewards.shape[1],), device=self.device)
        next_adv = torch.zeros((rewards.shape[1],), device=self.device)
        for t in reversed(range(rewards.shape[0])):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_value * mask - values[t]
            next_adv = delta + self.cfg.gae_lambda * self.cfg.gamma * mask * next_adv
            adv[t] = next_adv
            returns[t] = adv[t] + values[t]
            next_value = values[t]
        if normalize:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)
        if adv.shape[-1] == 1:
            return returns.squeeze(-1), adv.squeeze(-1)
        return returns, adv

    def _explained_variance(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if pred.numel() == 0 or target.numel() == 0:
            return 0.0
        y = target.reshape(-1)
        y_pred = pred.reshape(-1)
        var_y = torch.var(y, unbiased=False)
        if float(var_y.detach().cpu()) < 1.0e-12:
            return 0.0
        return float((1.0 - torch.var(y - y_pred, unbiased=False) / var_y).detach().cpu())

    def _steps_per_minibatch(self, *, total_steps: int, n_agents: int) -> int:
        minibatch_size = self._minibatch_size()
        if minibatch_size <= 0:
            return max(1, total_steps)
        if n_agents <= 0:
            return max(1, total_steps)
        steps_per = int(math.ceil(float(minibatch_size) / float(n_agents)))
        return max(1, min(total_steps, steps_per))

    def _credit_mode(self) -> str:
        return str(getattr(self.cfg, "credit_assignment", "global_team") or "global_team").strip().lower()

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

    @staticmethod
    def _to_agent_tensor(tensor: torch.Tensor, *, n_agents: int) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor.view(1, 1).expand(1, n_agents)
        if tensor.ndim == 1:
            if tensor.shape[0] == n_agents:
                return tensor.unsqueeze(0)
            return tensor.unsqueeze(-1).expand(tensor.shape[0], n_agents)
        return tensor

    @staticmethod
    def _to_agent_vector(tensor: torch.Tensor, *, n_agents: int) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor.view(1).expand(n_agents)
        if tensor.ndim == 1:
            if tensor.shape[0] == n_agents:
                return tensor
            if tensor.shape[0] == 1:
                return tensor.expand(n_agents)
            return tensor[:n_agents]
        if tensor.ndim == 2:
            if tensor.shape[0] == 1 and tensor.shape[1] == n_agents:
                return tensor.squeeze(0)
            if tensor.shape[1] == 1 and tensor.shape[0] == n_agents:
                return tensor.squeeze(-1)
            return tensor.reshape(-1)[:n_agents]
        return tensor.reshape(-1)[:n_agents]
