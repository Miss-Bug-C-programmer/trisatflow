from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from trisatflow.config import TrainConfig, save_config
from trisatflow.envs import GeoLeoGroundEnv
from trisatflow.envs.physical_metrics import METRIC_SCHEMA_VERSION
from trisatflow.experiment_contracts import assert_paper_safe
from trisatflow.models import FeatureEncoder, TemporalTopologyEncoder, TopologyEncoder, upper_action_mask_from_obs
from trisatflow.models.flat_hybrid_policy import FlatHybridPolicy


class FlatHybridTrainer:
    """Dependency-light PPO trainer for flat hybrid learning baselines."""

    def __init__(self, config: TrainConfig, *, baseline_name: str) -> None:
        self.cfg = config
        self.baseline_name = str(baseline_name)
        if self.baseline_name not in {"flat_ppo", "flat_mappo"}:
            raise ValueError("FlatHybridTrainer supports flat_ppo and flat_mappo")
        if self.cfg.steps_per_episode is not None:
            self.cfg.scenario.episode_len = int(self.cfg.steps_per_episode)
        assert_paper_safe(self.cfg)
        self.device = self._resolve_device()
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        torch.manual_seed(int(self.cfg.scenario.seed))
        self.env = GeoLeoGroundEnv(self.cfg.scenario, self.cfg.reward, self.device)
        self.encoder = self._build_encoder().to(self.device)
        self.policy = FlatHybridPolicy(
            int(self.cfg.algo.gnn_hidden_dim),
            int(self.cfg.algo.policy_hidden_dim),
            GeoLeoGroundEnv.N_UPPER_ACTIONS,
            GeoLeoGroundEnv.LOWER_ACTION_DIM,
            centralized_value=self.baseline_name == "flat_mappo",
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.policy.parameters()),
            lr=float(self.cfg.algo.upper_lr),
        )

    def _resolve_device(self) -> torch.device:
        requested = str(self.cfg.device or "cpu").strip()
        resolved = requested
        fallback = ""
        if requested.lower() == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
            if resolved == "cpu":
                fallback = "requested device 'auto' resolved to cpu because torch.cuda.is_available() is False"
        try:
            device = torch.device(resolved)
        except Exception:
            device = torch.device("cpu")
            fallback = f"requested device '{requested}' is invalid; falling back to cpu"
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
            fallback = f"requested device '{requested}' but torch.cuda.is_available() is False; falling back to cpu"
        self.cfg.requested_device = requested
        self.cfg.actual_device = str(device)
        self.cfg.device_fallback_reason = fallback
        self.cfg.device = str(device)
        if fallback:
            print(f"[FlatHybridTrainer] {fallback}")
        return device

    def _encoder_mode(self) -> str:
        mode = str(getattr(self.cfg.model, "topology_encoder", "static_gnn") or "static_gnn").strip().lower()
        if mode not in {"no_gnn", "static_gnn", "temporal_gnn"}:
            mode = "static_gnn"
        return mode

    def _build_encoder(self):
        mode = self._encoder_mode()
        if mode == "no_gnn":
            return FeatureEncoder(self.cfg.scenario.node_feature_dim, self.cfg.scenario.edge_feature_dim, self.cfg.algo.gnn_hidden_dim)
        base = TopologyEncoder(self.cfg.scenario.node_feature_dim, self.cfg.scenario.edge_feature_dim, self.cfg.algo.gnn_hidden_dim)
        if mode != "temporal_gnn":
            return base
        temporal = self.cfg.model.temporal
        return TemporalTopologyEncoder(
            base,
            base_dim=self.cfg.algo.gnn_hidden_dim,
            history_len=int(temporal.history_len),
            temporal_hidden_dim=int(temporal.hidden_dim),
        )

    @staticmethod
    def _reset_encoder_state(encoder: object) -> None:
        reset = getattr(encoder, "reset_temporal_state", None)
        if callable(reset):
            reset()

    def _encode(self, obs: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, *, update_state: bool) -> torch.Tensor:
        try:
            return self.encoder(obs, edge_index, edge_attr, update_state=update_state)
        except TypeError:
            return self.encoder(obs, edge_index, edge_attr)

    def train(self) -> List[Dict[str, Any]]:
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(self.cfg, output_dir / "resolved_config.yaml")
        history: List[Dict[str, Any]] = []
        for episode in range(1, int(self.cfg.total_episodes) + 1):
            rollout = self._collect_rollout(explore=True)
            losses = self._ppo_update(rollout)
            summary = self._summarize(episode, rollout["infos"])
            summary.update(losses)
            summary.update(
                {
                    "baseline": self.baseline_name,
                    "upper_algo": self.baseline_name,
                    "lower_algo": "flat_resource_head",
                    "observation_mode": str(self.cfg.observation.mode),
                    "include_oracle_cost": float(bool(self.cfg.observation.include_oracle_cost)),
                    "include_cost_prior_features": float(bool(self.cfg.observation.include_cost_prior_features)),
                    "requested_device": self.cfg.requested_device,
                    "actual_device": self.cfg.actual_device,
                    "device_fallback_reason": self.cfg.device_fallback_reason,
                }
            )
            history.append(summary)
            self._write_metrics(output_dir, history)
            if episode % int(self.cfg.log_interval) == 0:
                print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return history

    @torch.no_grad()
    def evaluate(self, *, seed: int | None = None, episodes: int = 1) -> Dict[str, Any]:
        old_seed = int(self.cfg.scenario.seed)
        if seed is not None:
            self.cfg.scenario.seed = int(seed)
            self.env = GeoLeoGroundEnv(self.cfg.scenario, self.cfg.reward, self.device)
        rows: List[Dict[str, Any]] = []
        try:
            for episode in range(1, int(episodes) + 1):
                rollout = self._collect_rollout(explore=False)
                rows.append(self._summarize(episode, rollout["infos"]))
        finally:
            if seed is not None:
                self.cfg.scenario.seed = old_seed
                self.env = GeoLeoGroundEnv(self.cfg.scenario, self.cfg.reward, self.device)
        return self._mean_rows(rows)

    def _collect_rollout(self, *, explore: bool) -> Dict[str, Any]:
        self._reset_encoder_state(self.encoder)
        obs, edge_index, edge_attr = self.env.reset()
        data: Dict[str, Any] = {k: [] for k in ("obs", "edge_index", "edge_attr", "action", "raw_resource", "log_prob", "value", "reward", "done")}
        data["infos"] = []
        done = False
        while not done:
            embed = self._encode(obs, edge_index, edge_attr, update_state=True)
            action_mask = upper_action_mask_from_obs(obs)
            if explore:
                offload_dist, resource_dist, _ = self.policy.distributions(embed, action_mask)
                action = offload_dist.sample()
                raw_resource = resource_dist.sample()
                lower_action = self.policy.resource_action(raw_resource)
                log_prob = offload_dist.log_prob(action) + resource_dist.log_prob(raw_resource).sum(dim=-1)
            else:
                action, lower_action = self.policy.deterministic_actions(embed, action_mask)
                raw_resource = torch.logit(lower_action.clamp(1.0e-6, 1.0 - 1.0e-6))
                log_prob = torch.zeros(action.shape, dtype=obs.dtype, device=self.device)
            value = self.policy.value(embed)
            step = self.env.step(action, lower_action)
            for key, value_item in (
                ("obs", obs),
                ("edge_index", edge_index),
                ("edge_attr", edge_attr),
                ("action", action),
                ("raw_resource", raw_resource),
                ("log_prob", log_prob),
                ("value", value),
                ("reward", step.upper_reward),
            ):
                data[key].append(value_item.detach().cpu())
            data["done"].append(bool(step.done))
            data["infos"].append(step.info)
            obs, edge_index, edge_attr, done = step.obs, step.edge_index, step.edge_attr, step.done
        return data

    def _ppo_update(self, rollout: Dict[str, Any]) -> Dict[str, float]:
        rewards = torch.stack(rollout["reward"]).to(self.device)
        old_values = torch.stack(rollout["value"]).to(self.device)
        old_log_prob = torch.stack(rollout["log_prob"]).to(self.device)
        dones = torch.tensor(rollout["done"], dtype=torch.float32, device=self.device)
        returns, advantages = self._gae(rewards, old_values, dones)
        old_values = old_values.detach()
        old_log_prob = old_log_prob.detach()
        advantages = advantages.detach()
        returns = returns.detach()
        epochs = max(1, min(int(self.cfg.algo.ppo_epochs), 4))
        clip = float(self.cfg.algo.clip_param)
        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        for _ in range(epochs):
            losses = []
            value_terms = []
            entropy_terms = []
            for i in range(len(rollout["obs"])):
                obs = rollout["obs"][i].to(self.device)
                edge_index = rollout["edge_index"][i].to(self.device)
                edge_attr = rollout["edge_attr"][i].to(self.device)
                action = rollout["action"][i].to(self.device)
                raw_resource = rollout["raw_resource"][i].to(self.device)
                embed = self._encode(obs, edge_index, edge_attr, update_state=False)
                action_mask = upper_action_mask_from_obs(obs)
                offload_dist, resource_dist, _ = self.policy.distributions(embed, action_mask)
                new_log_prob = offload_dist.log_prob(action) + resource_dist.log_prob(raw_resource).sum(dim=-1)
                ratio = torch.exp(new_log_prob - old_log_prob[i])
                adv = advantages[i]
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value = self.policy.value(embed)
                value_loss = F.mse_loss(value, returns[i])
                entropy = offload_dist.entropy().mean() + resource_dist.entropy().sum(dim=-1).mean()
                losses.append(policy_loss)
                value_terms.append(value_loss)
                entropy_terms.append(entropy)
            loss_policy = torch.stack(losses).mean()
            loss_value = torch.stack(value_terms).mean()
            entropy_mean = torch.stack(entropy_terms).mean()
            loss = loss_policy + float(self.cfg.algo.value_loss_coef) * loss_value - float(self.cfg.algo.entropy_coef) * entropy_mean
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.policy.parameters()),
                float(self.cfg.algo.max_grad_norm),
            )
            self.optimizer.step()
            policy_losses.append(float(loss_policy.detach().cpu()))
            value_losses.append(float(loss_value.detach().cpu()))
            entropies.append(float(entropy_mean.detach().cpu()))
        return {
            "policy_loss": sum(policy_losses) / len(policy_losses),
            "value_loss": sum(value_losses) / len(value_losses),
            "entropy": sum(entropies) / len(entropies),
            "grad_norm": float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm),
        }

    def _gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        next_value = torch.zeros_like(values[0])
        next_adv = torch.zeros_like(values[0])
        gamma = float(self.cfg.algo.gamma)
        lam = float(self.cfg.algo.gae_lambda)
        for t in reversed(range(rewards.shape[0])):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * nonterminal - values[t]
            next_adv = delta + gamma * lam * nonterminal * next_adv
            advantages[t] = next_adv
            returns[t] = advantages[t] + values[t]
            next_value = values[t]
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False).clamp_min(1.0e-6)
        advantages = (advantages - adv_mean) / adv_std
        return returns, advantages

    def _summarize(self, episode: int, infos: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        def cat(key: str) -> torch.Tensor:
            return torch.cat([info[key].detach().cpu().view(-1).float() for info in infos if key in info])

        def mean(key: str, fallback: str | None = None) -> float:
            if any(key in info for info in infos):
                values = cat(key)
            elif fallback is not None:
                values = cat(fallback)
            else:
                return 0.0
            return float(values.mean()) if values.numel() else 0.0

        actions = cat("upper_action").long()
        hist = torch.bincount(actions, minlength=GeoLeoGroundEnv.N_UPPER_ACTIONS).float()
        hist = hist / hist.sum().clamp_min(1.0)
        return {
            "episode": int(episode),
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "mean_delay": mean("delay"),
            "mean_energy": mean("energy"),
            "mean_queue": mean("queue"),
            "mean_delay_s": mean("physical_delay_s", "delay"),
            "mean_energy_j": mean("physical_energy_j", "energy"),
            "normalized_system_cost": mean("normalized_system_cost", "system_cost"),
            "reward_mean": mean("reward", "upper_reward"),
            "mean_system_cost": mean("system_cost"),
            "mean_feasibility": mean("feasible"),
            "upper_local_ratio": float(hist[0]),
            "upper_neighbor_ratio": float(hist[1]),
            "upper_geo_ratio": float(hist[2]),
            "upper_ground_ratio": float(hist[3]),
            "upper_remote_ratio": float(hist[1:].sum()),
        }

    @staticmethod
    def _mean_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {}
        out: Dict[str, Any] = {}
        for key in rows[0]:
            vals = [row.get(key) for row in rows]
            numeric = [float(v) for v in vals if isinstance(v, (int, float))]
            out[key] = sum(numeric) / len(numeric) if len(numeric) == len(vals) else vals[-1]
        return out

    def _write_metrics(self, output_dir: Path, rows: List[Dict[str, Any]]) -> None:
        (output_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "baseline": self.baseline_name,
                "encoder": self.encoder.state_dict(),
                "policy": self.policy.state_dict(),
                "config": self.cfg,
            },
            path,
        )
