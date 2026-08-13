from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import csv
import json
import math
import torch

from trisatflow.agents.lower_variants import LowerIDDPGAgent, LowerSACAgent
from trisatflow.agents.maddpg_lower import LowerMADDPGAgent
from trisatflow.agents.mappo_upper import UpperMAPPOAgent
from trisatflow.agents.replay import ReplayBuffer, RolloutBuffer
from trisatflow.agents.upper_variants import UpperIPPOAgent, UpperValueDecompositionAgent
from trisatflow.config import TrainConfig, save_config
from trisatflow.encoder_modes import apply_encoder_mode_to_algo, checkpoint_encoder_metadata
from trisatflow.diagnostics.gradient_flow import build_gradient_report, grad_norm, lower_action_sensitivity_to_upper_action
from trisatflow.diagnostics.training_cadence import cadence_report
from trisatflow.envs import GeoLeoGroundEnv
from trisatflow.envs.physical_metrics import METRIC_SCHEMA_VERSION
from trisatflow.envs.obs_schema import (
    IDX_GEO_DELAY,
    IDX_GEO_NORMALIZED_COST,
    IDX_GEO_QUEUE,
    IDX_GEO_RATE,
    IDX_GROUND_DELAY,
    IDX_GROUND_NORMALIZED_COST,
    IDX_GROUND_QUEUE,
    IDX_GROUND_RATE,
    IDX_LOCAL_DELAY,
    IDX_LOCAL_NORMALIZED_COST,
    IDX_LOCAL_QUEUE,
    IDX_LOCAL_RATE,
    IDX_NEIGHBOR_DELAY,
    IDX_NEIGHBOR_NORMALIZED_COST,
    IDX_NEIGHBOR_QUEUE,
    IDX_NEIGHBOR_RATE,
    SHARED_NODE_FEATURE_DIM,
    SHARED_NODE_FEATURE_DIM_WITH_COST,
)
from trisatflow.models import (
    AgentValue,
    CentralPerAgentValue,
    CentralValue,
    LocalLowerCritic,
    LowerActor,
    LowerCritic,
    FeatureEncoder,
    QMixer,
    StochasticLowerActor,
    TemporalTopologyEncoder,
    TopologyEncoder,
    UpperMAPPOPolicy,
    UpperQNetwork,
    upper_action_mask_from_obs,
)
from trisatflow.algorithms import validate_algorithm_choice

SAFE_OBSERVATION_MODE = "safe_observable"
COST_PRIOR_ABLATION_MODE = "cost_prior_ablation"
ORACLE_DEBUG_MODE = "oracle_debug"


class HierarchicalTrainer:
    """Coordinates GNN + upper discrete MARL + lower continuous MARL training.

    Supported dependency-light combinations:

    * upper discrete offloading: MAPPO, IPPO, IQL, VDN, QMIX
    * lower continuous resource allocation: MADDPG, IDDPG, MASAC, ISAC

    The names are aligned with BenchMARL's algorithms. The implementation here
    is intentionally compact so algorithm-combination sweeps can run in the
    TriSatFlow prototype without requiring the full TorchRL runtime.
    """

    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.encoder_mode_semantics = apply_encoder_mode_to_algo(self.cfg.algo, warn=False)
        if self.cfg.steps_per_episode is not None:
            self.cfg.scenario.episode_len = int(self.cfg.steps_per_episode)
        if hasattr(self.cfg, "physical"):
            self.cfg.scenario.physical = self.cfg.physical
        self._apply_observation_and_oracle_policy()
        validate_algorithm_choice(config.algo.upper_algo, config.algo.lower_algo)
        requested_raw = str(config.device or "cpu").strip()
        resolved_raw = requested_raw
        fallback_reason = ""
        if requested_raw.lower() == "auto":
            if torch.cuda.is_available():
                resolved_raw = "cuda"
            else:
                resolved_raw = "cpu"
                fallback_reason = "requested device 'auto' resolved to cpu because torch.cuda.is_available() is False"
        try:
            requested_device = torch.device(resolved_raw)
        except Exception:
            requested_device = torch.device("cpu")
            fallback_reason = f"requested device '{requested_raw}' is invalid; falling back to cpu"
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
            fallback_reason = f"requested device '{requested_raw}' but torch.cuda.is_available() is False; falling back to cpu"
        else:
            self.device = requested_device
        self.requested_device = requested_raw
        self.actual_device = str(self.device)
        self.device_fallback_reason = fallback_reason
        self.cfg.requested_device = self.requested_device
        self.cfg.actual_device = self.actual_device
        self.cfg.device_fallback_reason = self.device_fallback_reason
        self.cfg.device = self.actual_device
        if self.device_fallback_reason:
            print(f"[HierarchicalTrainer] {self.device_fallback_reason}")
        if self.device.type == "cpu":
            # The lightweight per-step Python/PyTorch loops are faster and more
            # stable on CPU with a single intra-op thread, especially when a
            # CUDA config falls back to CPU in CI or laptop smoke tests.
            torch.set_num_threads(1)
        torch.manual_seed(config.scenario.seed)
        self.env = GeoLeoGroundEnv(config.scenario, config.reward, self.device)
        self.encoder = self._build_topology_encoder(shared=True)
        if not bool(self.encoder_mode_semantics.upper_encoder_trainable):
            for param in self.encoder.parameters():
                param.requires_grad_(False)
        self.upper_agent = self._build_upper_agent()
        self.lower_agent = self._build_lower_agent()
        self.replay = ReplayBuffer(capacity=50000)
        self.upper_update_count = 0
        self.lower_update_count = 0
        self.env_steps_since_upper_update = 0
        self.env_steps_since_lower_update = 0
        self.gradient_diagnostics_history: List[Dict[str, object]] = []
        self._cost_prior_features_enabled = bool(getattr(self.cfg.observation, "include_cost_prior_features", False))
        self._oracle_debug_enabled = str(getattr(self.cfg.observation, "mode", SAFE_OBSERVATION_MODE)).strip().lower() == ORACLE_DEBUG_MODE
        self._cost_prior_diagnostics_enabled = self._cost_prior_features_enabled or self._oracle_debug_enabled or bool(self.cfg.policy_regularization.enabled)

    def _topology_encoder_mode(self) -> str:
        raw = str(getattr(getattr(self.cfg, "model", None), "topology_encoder", "") or "").strip().lower()
        if raw in {"no_gnn", "static_gnn", "temporal_gnn"}:
            return raw
        # Backward compatibility: legacy configs only toggled scenario.enable_gnn.
        return "static_gnn" if bool(getattr(self.cfg.scenario, "enable_gnn", True)) else "no_gnn"

    def _temporal_enabled(self) -> bool:
        model = getattr(self.cfg, "model", None)
        temporal = getattr(model, "temporal", None)
        if temporal is None:
            return False
        return bool(getattr(temporal, "enabled", False)) and self._topology_encoder_mode() == "temporal_gnn"

    def _build_topology_encoder(self, *, shared: bool) -> TopologyEncoder | FeatureEncoder | TemporalTopologyEncoder:
        mode = self._topology_encoder_mode()
        if mode == "no_gnn":
            return FeatureEncoder(
                node_dim=self.cfg.scenario.node_feature_dim,
                edge_dim=self.cfg.scenario.edge_feature_dim,
                hidden_dim=self.cfg.algo.gnn_hidden_dim,
            ).to(self.device)

        base = TopologyEncoder(
            node_dim=self.cfg.scenario.node_feature_dim,
            edge_dim=self.cfg.scenario.edge_feature_dim,
            hidden_dim=self.cfg.algo.gnn_hidden_dim,
        ).to(self.device)
        if mode != "temporal_gnn":
            return base
        temporal_cfg = getattr(getattr(self.cfg, "model", None), "temporal", None)
        history_len = int(getattr(temporal_cfg, "history_len", 4) or 4)
        hidden_dim = int(getattr(temporal_cfg, "hidden_dim", 128) or 128)
        return TemporalTopologyEncoder(
            base,
            base_dim=self.cfg.algo.gnn_hidden_dim,
            history_len=history_len,
            temporal_hidden_dim=hidden_dim,
        ).to(self.device)

    @staticmethod
    def _reset_encoder_state(encoder: object) -> None:
        reset_fn = getattr(encoder, "reset_temporal_state", None)
        if callable(reset_fn):
            reset_fn()

    def _reset_temporal_states(self) -> None:
        self._reset_encoder_state(self.encoder)
        self._reset_encoder_state(getattr(self.lower_agent, "encoder", None))
        self._reset_encoder_state(getattr(self.lower_agent, "target_encoder", None))

    def _apply_observation_and_oracle_policy(self) -> None:
        obs_cfg = self.cfg.observation
        mode = str(getattr(obs_cfg, "mode", SAFE_OBSERVATION_MODE) or SAFE_OBSERVATION_MODE).strip().lower()
        if mode not in {SAFE_OBSERVATION_MODE, COST_PRIOR_ABLATION_MODE, ORACLE_DEBUG_MODE}:
            raise ValueError(
                f"Unsupported observation.mode={mode!r}; expected one of "
                f"{SAFE_OBSERVATION_MODE!r}, {COST_PRIOR_ABLATION_MODE!r}, {ORACLE_DEBUG_MODE!r}"
            )
        obs_cfg.mode = mode
        obs_cfg.include_oracle_cost = bool(getattr(obs_cfg, "include_oracle_cost", False))
        obs_cfg.include_cost_prior_features = bool(getattr(obs_cfg, "include_cost_prior_features", False))

        if bool(getattr(obs_cfg, "legacy_auto_enabled", False)):
            print(
                "[WARNING][observation-policy] legacy config auto-mapped to privileged mode; "
                "please set observation.mode explicitly."
            )

        if mode == SAFE_OBSERVATION_MODE:
            if obs_cfg.include_oracle_cost or obs_cfg.include_cost_prior_features:
                print("[WARNING][observation-policy] safe_observable overrides include_oracle_cost/include_cost_prior_features to False.")
            obs_cfg.include_oracle_cost = False
            obs_cfg.include_cost_prior_features = False
            if str(self.cfg.reward.mode).strip().lower() == "oracle_aligned_cost":
                print("[WARNING][observation-policy] safe_observable disallows oracle-aligned reward; fallback to reward.mode=physical_weighted.")
                self.cfg.reward.mode = "physical_weighted"
            self.cfg.reward.use_oracle_cost_components = False
            if self.cfg.policy_regularization.enabled:
                print("[WARNING][observation-policy] safe_observable disables policy_regularization by default.")
            self.cfg.policy_regularization.enabled = False
            self.cfg.policy_regularization.mode = "none"
            if str(self.cfg.algo.policy_head).strip().lower() == "hybrid_gnn_cost":
                print("[WARNING][observation-policy] safe_observable switches policy_head from hybrid_gnn_cost to gnn_only.")
                self.cfg.algo.policy_head = "gnn_only"
        elif mode == COST_PRIOR_ABLATION_MODE:
            obs_cfg.include_cost_prior_features = True
            obs_cfg.include_oracle_cost = False
            self.cfg.reward.use_oracle_cost_components = False
            if str(self.cfg.reward.mode).strip().lower() == "oracle_aligned_cost":
                print("[WARNING][observation-policy] cost_prior_ablation does not permit oracle reward; fallback to reward.mode=physical_weighted.")
                self.cfg.reward.mode = "physical_weighted"
            print("[WARNING][observation-policy] cost_prior_ablation enabled (privileged cost-prior features are exposed).")
        else:
            # ORACLE_DEBUG_MODE
            obs_cfg.include_oracle_cost = True
            obs_cfg.include_cost_prior_features = True
            print("[WARNING][observation-policy] oracle_debug enabled: oracle/privileged signals are exposed; do not use for primary results.")

        self.cfg.scenario.observation_access_mode = mode
        self.cfg.scenario.observation_include_oracle_cost = bool(obs_cfg.include_oracle_cost)
        self.cfg.scenario.observation_include_cost_prior_features = bool(obs_cfg.include_cost_prior_features)
        self.cfg.scenario.include_cost_features_in_obs = bool(obs_cfg.include_cost_prior_features)
        if self.cfg.scenario.include_cost_features_in_obs and self.cfg.scenario.node_feature_dim < SHARED_NODE_FEATURE_DIM_WITH_COST:
            self.cfg.scenario.node_feature_dim = SHARED_NODE_FEATURE_DIM_WITH_COST

    def _build_upper_agent(self):
        upper = self.cfg.algo.upper_algo
        cfg = self.cfg.algo
        n_actions = GeoLeoGroundEnv.N_UPPER_ACTIONS
        if upper == "mappo":
            actor = UpperMAPPOPolicy(
                cfg.gnn_hidden_dim,
                cfg.policy_hidden_dim,
                n_actions,
                policy_head=str(getattr(cfg, "policy_head", "gnn_only") or "gnn_only"),
                logit_centering=bool(getattr(cfg, "logit_centering", False)),
            ).to(self.device)
            credit_mode = str(getattr(cfg, "credit_assignment", "global_team") or "global_team").strip().lower()
            if credit_mode == "per_agent":
                critic = CentralPerAgentValue(cfg.gnn_hidden_dim, cfg.policy_hidden_dim).to(self.device)
            else:
                critic = CentralValue(cfg.gnn_hidden_dim, self.env.n_agents, cfg.policy_hidden_dim).to(self.device)
            return UpperMAPPOAgent(self.encoder, actor, critic, cfg, self.cfg.policy_regularization, self.device)
        if upper == "ippo":
            actor = UpperMAPPOPolicy(
                cfg.gnn_hidden_dim,
                cfg.policy_hidden_dim,
                n_actions,
                policy_head=str(getattr(cfg, "policy_head", "gnn_only") or "gnn_only"),
                logit_centering=bool(getattr(cfg, "logit_centering", False)),
            ).to(self.device)
            value = AgentValue(cfg.gnn_hidden_dim, cfg.policy_hidden_dim).to(self.device)
            return UpperIPPOAgent(self.encoder, actor, value, cfg, self.device)
        if upper in {"iql", "vdn", "qmix"}:
            q_net = UpperQNetwork(cfg.gnn_hidden_dim, cfg.policy_hidden_dim, n_actions).to(self.device)
            target_encoder = self._build_topology_encoder(shared=False)
            target_q_net = UpperQNetwork(cfg.gnn_hidden_dim, cfg.policy_hidden_dim, n_actions).to(self.device)
            mixer = target_mixer = None
            if upper == "qmix":
                mixer = QMixer(cfg.gnn_hidden_dim, self.env.n_agents, cfg.policy_hidden_dim).to(self.device)
                target_mixer = QMixer(cfg.gnn_hidden_dim, self.env.n_agents, cfg.policy_hidden_dim).to(self.device)
            return UpperValueDecompositionAgent(
                upper, self.encoder, q_net, target_encoder, target_q_net, cfg, self.device, mixer=mixer, target_mixer=target_mixer
            )
        raise ValueError(f"Unknown upper algorithm: {upper}")

    def _build_lower_agent(self):
        lower = self.cfg.algo.lower_algo
        cfg = self.cfg.algo
        lower_encoder_mode = self._lower_encoder_mode()
        lower_encoder, lower_target_encoder = self._build_lower_encoder_bundle(mode=lower_encoder_mode)
        if lower == "maddpg":
            actor = LowerActor(cfg.gnn_hidden_dim, GeoLeoGroundEnv.N_UPPER_ACTIONS, cfg.policy_hidden_dim, GeoLeoGroundEnv.LOWER_ACTION_DIM).to(self.device)
            critic = LowerCritic(
                cfg.gnn_hidden_dim,
                self.env.n_agents,
                GeoLeoGroundEnv.N_UPPER_ACTIONS,
                GeoLeoGroundEnv.LOWER_ACTION_DIM,
                cfg.policy_hidden_dim,
            ).to(self.device)
            target_actor = LowerActor(cfg.gnn_hidden_dim, GeoLeoGroundEnv.N_UPPER_ACTIONS, cfg.policy_hidden_dim, GeoLeoGroundEnv.LOWER_ACTION_DIM).to(self.device)
            target_critic = LowerCritic(
                cfg.gnn_hidden_dim,
                self.env.n_agents,
                GeoLeoGroundEnv.N_UPPER_ACTIONS,
                GeoLeoGroundEnv.LOWER_ACTION_DIM,
                cfg.policy_hidden_dim,
            ).to(self.device)
            return LowerMADDPGAgent(
                lower_encoder,
                actor,
                critic,
                target_actor,
                target_critic,
                cfg,
                self.device,
                target_encoder=lower_target_encoder,
                encoder_mode=lower_encoder_mode,
            )
        if lower == "iddpg":
            actor = LowerActor(cfg.gnn_hidden_dim, GeoLeoGroundEnv.N_UPPER_ACTIONS, cfg.policy_hidden_dim, GeoLeoGroundEnv.LOWER_ACTION_DIM).to(self.device)
            critic = LocalLowerCritic(cfg.gnn_hidden_dim, GeoLeoGroundEnv.N_UPPER_ACTIONS, GeoLeoGroundEnv.LOWER_ACTION_DIM, cfg.policy_hidden_dim).to(self.device)
            target_actor = LowerActor(cfg.gnn_hidden_dim, GeoLeoGroundEnv.N_UPPER_ACTIONS, cfg.policy_hidden_dim, GeoLeoGroundEnv.LOWER_ACTION_DIM).to(self.device)
            target_critic = LocalLowerCritic(cfg.gnn_hidden_dim, GeoLeoGroundEnv.N_UPPER_ACTIONS, GeoLeoGroundEnv.LOWER_ACTION_DIM, cfg.policy_hidden_dim).to(self.device)
            return LowerIDDPGAgent(
                lower_encoder,
                actor,
                critic,
                target_actor,
                target_critic,
                cfg,
                self.device,
                target_encoder=lower_target_encoder,
                encoder_mode=lower_encoder_mode,
            )
        if lower in {"masac", "isac"}:
            actor = StochasticLowerActor(cfg.gnn_hidden_dim, GeoLeoGroundEnv.N_UPPER_ACTIONS, cfg.policy_hidden_dim, GeoLeoGroundEnv.LOWER_ACTION_DIM).to(self.device)
            if lower == "masac":
                critic = LowerCritic(
                    cfg.gnn_hidden_dim,
                    self.env.n_agents,
                    GeoLeoGroundEnv.N_UPPER_ACTIONS,
                    GeoLeoGroundEnv.LOWER_ACTION_DIM,
                    cfg.policy_hidden_dim,
                ).to(self.device)
                target_critic = LowerCritic(
                    cfg.gnn_hidden_dim,
                    self.env.n_agents,
                    GeoLeoGroundEnv.N_UPPER_ACTIONS,
                    GeoLeoGroundEnv.LOWER_ACTION_DIM,
                    cfg.policy_hidden_dim,
                ).to(self.device)
            else:
                critic = LocalLowerCritic(cfg.gnn_hidden_dim, GeoLeoGroundEnv.N_UPPER_ACTIONS, GeoLeoGroundEnv.LOWER_ACTION_DIM, cfg.policy_hidden_dim).to(self.device)
                target_critic = LocalLowerCritic(cfg.gnn_hidden_dim, GeoLeoGroundEnv.N_UPPER_ACTIONS, GeoLeoGroundEnv.LOWER_ACTION_DIM, cfg.policy_hidden_dim).to(self.device)
            return LowerSACAgent(
                lower,
                lower_encoder,
                actor,
                critic,
                target_critic,
                cfg,
                self.device,
                target_encoder=lower_target_encoder,
                encoder_mode=lower_encoder_mode,
            )
        raise ValueError(f"Unknown lower algorithm: {lower}")

    def _lower_encoder_mode(self) -> str:
        mode = str(getattr(self.cfg.algo, "encoder_mode", "shared_upper_detached_lower") or "shared_upper_detached_lower").strip().lower()
        aliases = {
            "shared_frozen": "shared_upper_only",
            "shared_upper_detached_lower": "shared_upper_only",
            "shared_upper_only": "shared_upper_only",
            "shared_joint": "shared_joint",
            "separate": "separate_lower_encoder",
            "separate_lower_encoder": "separate_lower_encoder",
        }
        return aliases.get(mode, "shared_upper_only")

    def _build_lower_encoder_bundle(self, *, mode: str):
        if mode in {"shared_upper_only", "shared_joint"}:
            return self.encoder, None
        lower_encoder = self._build_topology_encoder(shared=False)
        target_encoder = self._build_topology_encoder(shared=False)
        return lower_encoder, target_encoder

    def train(self) -> List[Dict[str, float]]:
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(self.cfg, output_dir / "resolved_config.yaml")
        metrics_history: List[Dict[str, float]] = []
        for episode in range(1, self.cfg.total_episodes + 1):
            train_phase = self._episode_train_phase(episode)
            rollout = RolloutBuffer()
            self._reset_temporal_states()
            obs, edge_index, edge_attr = self.env.reset()
            ep_infos: List[Dict[str, torch.Tensor]] = []
            done = False
            while not done:
                decision_step = int(self.env.t)
                upper_action, log_prob, value, embed = self.upper_agent.act(obs, edge_index, edge_attr, episode=episode)
                if self._use_neutral_lower(train_phase):
                    lower_action = self._neutral_lower_action(obs.shape[0])
                    lower_mode = "neutral_allocator"
                else:
                    lower_embed = embed.detach() if bool(getattr(self.cfg.algo, "detach_embedding_during_action_collection", True)) else embed
                    lower_action = self.lower_agent.act(
                        lower_embed,
                        upper_action,
                        explore=True,
                        obs=obs,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                    )
                    lower_mode = "learned"
                pre_step_mask = upper_action_mask_from_obs(obs).detach().cpu()
                pre_step_prior = None
                pre_step_oracle = None
                if self._cost_prior_diagnostics_enabled:
                    pre_step_prior = self._cost_prior_from_obs(obs.detach(), pre_step_mask.to(obs.device)).detach().cpu()
                    pre_step_oracle = torch.argmax(pre_step_prior, dim=-1).detach().cpu()
                pre_step_probs = self._policy_probs(obs, edge_index, edge_attr).detach().cpu()
                step = self.env.step(upper_action, lower_action)
                self.env_steps_since_upper_update += 1
                self.env_steps_since_lower_update += 1
                rollout.obs.append(obs.detach().cpu())
                rollout.edge_index.append(edge_index.detach().cpu())
                rollout.edge_attr.append(edge_attr.detach().cpu())
                rollout.upper_action.append(upper_action.detach().cpu())
                rollout.log_prob.append(log_prob.detach().cpu())
                rollout.value.append(value.detach().cpu())
                rollout.reward.append(step.upper_reward.detach().cpu())
                rollout.done.append(step.done)
                if pre_step_prior is not None and pre_step_oracle is not None:
                    rollout.cost_prior.append(pre_step_prior)
                    rollout.oracle_action.append(pre_step_oracle)
                rollout.old_action_probs.append(pre_step_probs)
                rollout.step_index.append(decision_step)
                scenario_phase, task_type = self._trace_labels_for_step(decision_step)
                rollout.scenario_phase.append(scenario_phase)
                rollout.task_type.append(task_type)
                rollout.extras.append(
                    {
                        "train_phase": train_phase,
                        "lower_mode": lower_mode,
                        "lower_action_collection_embed_detached": bool(getattr(self.cfg.algo, "detach_embedding_during_action_collection", True)),
                    }
                )
                self.replay.add(
                    obs=obs,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    upper_action=upper_action,
                    lower_action=lower_action,
                    reward=step.lower_reward,
                    upper_reward=step.upper_reward,
                    next_obs=step.obs,
                    next_edge_index=step.edge_index,
                    next_edge_attr=step.edge_attr,
                    done=torch.tensor(float(step.done), device=self.device),
                )
                ep_infos.append(step.info)
                obs, edge_index, edge_attr, done = step.obs, step.edge_index, step.edge_attr, step.done

            upper_losses = self._maybe_update_upper(episode, rollout)
            shared_encoder_grad_from_upper = grad_norm(self.encoder.parameters())
            upper_losses["shared_encoder_grad_norm_from_upper"] = float(shared_encoder_grad_from_upper)
            lower_losses = self._maybe_update_lower(episode, train_phase)
            sensitivity = self._lower_sensitivity_snapshot(obs, edge_index, edge_attr)
            gradient_row = build_gradient_report(
                trainer=self,
                upper_losses=upper_losses,
                lower_losses=lower_losses,
                update_step=len(metrics_history) + 1,
                sensitivity=sensitivity,
            )
            self.gradient_diagnostics_history.append(gradient_row)
            self._write_rollout_debug(output_dir, episode, rollout, upper_losses)
            summary = self._summarize_episode(episode, ep_infos)
            summary.update(self._deterministic_eval_summary(rollout))
            summary.update(upper_losses)
            summary.update(lower_losses)
            summary.update(self._training_semantics_summary())
            summary.update(
                cadence_report(
                    upper_update_count=self.upper_update_count,
                    lower_update_count=self.lower_update_count,
                    env_steps_since_upper_update=self.env_steps_since_upper_update,
                    env_steps_since_lower_update=self.env_steps_since_lower_update,
                    replay_buffer_size=len(self.replay),
                    rollout_buffer_size=len(rollout),
                    upper_update_every=int(getattr(self.cfg.algo, "upper_update_every", 1) or 1),
                    lower_update_every=int(getattr(self.cfg.algo, "lower_update_every", 1) or 1),
                    lower_updates_per_upper_update=int(getattr(self.cfg.algo, "lower_updates_per_upper_update", 1) or 1),
                )
            )
            summary.update({k: v for k, v in gradient_row.items() if k != "unavailable_fields"})
            summary["gradient_unavailable_fields"] = json.dumps(gradient_row.get("unavailable_fields", {}), ensure_ascii=False)
            summary["upper_algo"] = self.cfg.algo.upper_algo
            summary["lower_algo"] = self.cfg.algo.lower_algo
            summary["train_phase"] = train_phase
            summary["lower_mode"] = "neutral_allocator" if self._use_neutral_lower(train_phase) else "learned"
            summary["lower_encoder_mode"] = self._lower_encoder_mode()
            summary["observation_mode"] = str(self.cfg.observation.mode)
            summary["include_oracle_cost"] = float(bool(self.cfg.observation.include_oracle_cost))
            summary["include_cost_prior_features"] = float(bool(self.cfg.observation.include_cost_prior_features))
            summary["temporal_encoder_enabled"] = float(self._temporal_enabled())
            summary["history_len"] = float(
                int(getattr(getattr(getattr(self.cfg, "model", None), "temporal", None), "history_len", 1) or 1)
            )
            summary["requested_device"] = self.requested_device
            summary["actual_device"] = self.actual_device
            summary["device_fallback_reason"] = self.device_fallback_reason
            metrics_history.append(summary)
            self._write_metrics_files(output_dir, metrics_history)
            self._write_gradient_diagnostics(output_dir)
            if episode % self.cfg.log_interval == 0:
                print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return metrics_history

    def _write_metrics_files(self, output_dir: Path, metrics_history: List[Dict[str, float]]) -> None:
        with open(output_dir / "metrics.jsonl", "w", encoding="utf-8") as f:
            for item in metrics_history:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        keys: List[str] = []
        for item in metrics_history:
            for key in item.keys():
                if key not in keys:
                    keys.append(key)
        preferred = [
            "episode", "metric_schema_version", "upper_algo", "lower_algo", "train_phase", "lower_mode", "lower_encoder_mode", "observation_mode", "include_oracle_cost", "include_cost_prior_features", "temporal_encoder_enabled", "history_len", "requested_device", "actual_device", "device_fallback_reason", "normalized_system_cost", "mean_delay_s", "p95_delay_s", "mean_energy_j", "mean_queue_length_tasks",
            "mean_deadline_exceedance", "mean_deadline_violation_ratio", "reward_mean", "mean_system_cost", "mean_delay", "mean_energy", "mean_queue",
            "mean_queue_cycles", "mean_leo_queue", "mean_geo_queue", "mean_ground_queue", "max_remote_queue", "queue_stability_metric",
            "mean_deadline_violation", "mean_feasibility", "mean_lyapunov_drift", "mean_offload_gain",
            "mean_local_queue_pressure", "mean_mask_size", "neighbor_visible_ratio", "geo_visible_ratio",
            "ground_visible_ratio", "remote_available_ratio", "trace_missing_count", "trace_fallback_count",
            "trace_hit_ratio", "upper_local_ratio", "upper_neighbor_ratio",
            "upper_geo_ratio", "upper_ground_ratio", "upper_remote_ratio", "neighbor_selected_when_visible_ratio",
            "geo_selected_when_visible_ratio", "ground_selected_when_visible_ratio", "remote_selected_when_visible_ratio",
            "mean_reward_local_selected", "mean_reward_neighbor_selected", "mean_reward_geo_selected", "mean_reward_ground_selected",
            "mean_system_cost_local_selected", "mean_system_cost_neighbor_selected", "mean_system_cost_geo_selected", "mean_system_cost_ground_selected",
            "mean_delay_cost_local_selected", "mean_delay_cost_neighbor_selected", "mean_delay_cost_geo_selected", "mean_delay_cost_ground_selected",
            "mean_queue_cost_local_selected", "mean_queue_cost_neighbor_selected", "mean_queue_cost_geo_selected", "mean_queue_cost_ground_selected",
            "mean_transmission_cost_local_selected", "mean_transmission_cost_neighbor_selected", "mean_transmission_cost_geo_selected", "mean_transmission_cost_ground_selected",
            "mean_compute_cost_local_selected", "mean_compute_cost_neighbor_selected", "mean_compute_cost_geo_selected", "mean_compute_cost_ground_selected",
            "mean_bonus_local_selected", "mean_bonus_neighbor_selected", "mean_bonus_geo_selected", "mean_bonus_ground_selected",
            "mean_penalty_local_selected", "mean_penalty_neighbor_selected", "mean_penalty_geo_selected", "mean_penalty_ground_selected",
            "mean_lower_effect_local_selected", "mean_lower_effect_neighbor_selected", "mean_lower_effect_geo_selected", "mean_lower_effect_ground_selected",
            "mean_cost_local_selected", "mean_cost_neighbor_selected", "mean_cost_geo_selected", "mean_cost_ground_selected",
            "mean_advantage_local_selected", "mean_advantage_neighbor_selected", "mean_advantage_geo_selected", "mean_advantage_ground_selected",
            "eval_argmax_local_ratio", "eval_argmax_neighbor_ratio", "eval_argmax_geo_ratio",
            "eval_argmax_ground_ratio", "eval_argmax_remote_ratio", "eval_policy_entropy", "eval_warning",
            "policy_loss", "value_loss", "value_loss_raw", "entropy", "approx_kl", "clip_fraction", "explained_variance",
            "upper_policy_loss", "upper_value_loss", "upper_entropy", "upper_approx_kl", "upper_clip_fraction",
            "upper_value_loss_raw", "upper_value_loss_scale",
            "upper_grad_norm", "upper_explained_variance", "upper_ppo_epochs_ran", "upper_num_minibatches", "upper_early_stop_kl",
            "upper_advantage_mean", "upper_advantage_std", "upper_return_mean", "upper_return_std",
            "upper_credit_assignment", "upper_policy_head", "upper_advantage_agent_std", "upper_advantage_agent_snr",
            "upper_critic_scope",
            "advantage_oracle_alignment", "phase_advantage_alignment",
            "upper_value_mean", "upper_value_std", "upper_ratio_mean", "upper_ratio_std",
            "upper_old_logprob_mean", "upper_new_logprob_mean",
            "upper_cost_rank_kl_loss", "upper_cost_prior_ce_loss", "upper_policy_cost_prior_kl",
            "prob_oracle_action_mean", "policy_cost_prior_agreement",
            "upper_action_bias_reg_loss",
            "mean_logit_local", "mean_logit_neighbor", "mean_logit_geo", "mean_logit_ground",
            "std_logit_local", "std_logit_neighbor", "std_logit_geo", "std_logit_ground",
            "lower_actor_loss", "lower_critic_loss", "lower_encoder_grad_norm", "lower_q_mean", "lower_q_target_mean",
        ]
        ordered = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]
        with open(output_dir / "metrics.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ordered)
            writer.writeheader()
            writer.writerows(metrics_history)

    def _summarize_episode(self, episode: int, infos: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        def mean_key(key: str) -> float:
            return float(torch.stack([info[key].float().mean().detach().cpu() for info in infos]).mean())

        def mean_key_optional(key: str, fallback_key: str | None = None) -> float:
            present = [info[key].float().mean().detach().cpu() for info in infos if key in info]
            if present:
                return float(torch.stack(present).mean())
            if fallback_key is not None:
                return mean_key(fallback_key)
            return 0.0

        def cat_key(key: str) -> torch.Tensor:
            return torch.cat([info[key].view(-1).detach().cpu() for info in infos]).float()

        def cat_key_optional(key: str, fallback_key: str | None = None) -> torch.Tensor:
            present = [info[key].view(-1).detach().cpu() for info in infos if key in info]
            if present:
                return torch.cat(present).float()
            if fallback_key is not None:
                return cat_key(fallback_key)
            return torch.zeros((0,), dtype=torch.float32)

        def conditional_ratio(selected_key: str, available_key: str) -> float:
            selected = cat_key(selected_key)
            available = cat_key(available_key)
            denom = available.sum().clamp_min(1.0)
            return float(selected.sum() / denom)

        def conditional_mean(metric_key: str, selected_key: str) -> float:
            metric = cat_key(metric_key)
            selected = cat_key(selected_key)
            denom = selected.sum().clamp_min(1.0)
            return float((metric * selected).sum() / denom)

        actions = torch.cat([info["upper_action"].view(-1).detach().cpu() for info in infos])
        hist = torch.bincount(actions, minlength=GeoLeoGroundEnv.N_UPPER_ACTIONS).float()
        hist = hist / hist.sum().clamp_min(1.0)
        neighbor_available = cat_key("upper_mask_neighbor")
        geo_available = cat_key("upper_mask_geo")
        ground_available = cat_key("upper_mask_ground")
        remote_available = ((neighbor_available + geo_available + ground_available) > 0).float()
        remote_actions = ((actions != GeoLeoGroundEnv.ACTION_LOCAL).float()).mean()
        summary = {
            "episode": episode,
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "mean_delay": mean_key("delay"),
            "mean_energy": mean_key("energy"),
            "mean_queue": mean_key("queue"),
            "mean_delay_s": mean_key_optional("physical_delay_s", "delay"),
            "p95_delay_s": float(torch.quantile(cat_key_optional("physical_delay_s", "delay"), 0.95).item()),
            "mean_energy_j": mean_key_optional("physical_energy_j", "energy"),
            "mean_queue_length_tasks": mean_key_optional("physical_queue_length_tasks", "queue"),
            "mean_queue_cycles": mean_key_optional("physical_queue_cycles", "queue"),
            "mean_leo_queue": mean_key_optional("leo_queue", "queue"),
            "mean_geo_queue": mean_key_optional("geo_queue", "queue"),
            "mean_ground_queue": mean_key_optional("ground_queue", "queue"),
            "max_remote_queue": float(cat_key_optional("max_remote_queue", "queue").max().item()) if infos else 0.0,
            "queue_stability_metric": mean_key_optional("queue_stability_metric"),
            "normalized_system_cost": mean_key_optional("normalized_system_cost", "normalized_cost"),
            "reward_mean": mean_key_optional("reward", "upper_reward"),
            "mean_service": mean_key("service"),
            "mean_arrivals": mean_key("arrivals"),
            "mean_system_cost": mean_key("system_cost"),
            "mean_deadline_exceedance": mean_key("deadline_exceedance"),
            "mean_deadline_violation_ratio": mean_key("deadline_violation_flag"),
            "mean_deadline_violation": mean_key("deadline_exceedance"),
            "mean_feasibility": mean_key("feasible"),
            "mean_lyapunov_drift": mean_key("lyapunov_drift"),
            "mean_virtual_delay_queue": mean_key("virtual_delay_queue"),
            "mean_offload_gain": mean_key("offload_gain"),
            "mean_local_queue_pressure": mean_key("local_queue_pressure"),
            "mean_remote_feasible_bonus": mean_key("remote_feasible_bonus"),
            "mean_mask_size": mean_key("upper_mask_size"),
            "invalid_action_ratio": mean_key("invalid_action_ratio"),
            "masked_action_ratio": mean_key("masked_action_ratio"),
            "visibility_mask_ratio": mean_key("visibility_mask_ratio"),
            "completion_mask_ratio": mean_key("completion_mask_ratio"),
            "mobility_mask_ratio": mean_key("mobility_mask_ratio"),
            "mask_source_code": mean_key_optional("mask_source_code"),
            "uses_oracle_trace_mask": mean_key_optional("uses_oracle_trace_mask"),
            "mask_deployable": mean_key_optional("mask_deployable"),
            "mask_false_positive_rate_observed": mean_key_optional("mask_false_positive_rate_observed"),
            "mask_false_negative_rate_observed": mean_key_optional("mask_false_negative_rate_observed"),
            "mask_predictor_fallback_ratio": mean_key_optional("mask_predictor_fallback"),
            "mask_fallback_due_empty_ratio": mean_key_optional("mask_fallback_due_empty"),
            "mask_staleness_slots": mean_key_optional("mask_staleness_slots"),
            "link_lifetime_noise_std_s": mean_key_optional("link_lifetime_noise_std_s"),
            "completion_time_noise_std_s": mean_key_optional("completion_time_noise_std_s"),
            "configured_mask_false_positive_rate": mean_key_optional("configured_mask_false_positive_rate"),
            "configured_mask_false_negative_rate": mean_key_optional("configured_mask_false_negative_rate"),
            "mean_action_mask_raw_count": mean_key("action_mask_raw_count"),
            "mean_action_mask_after_visibility_count": mean_key("action_mask_after_visibility_count"),
            "mean_action_mask_after_completion_safe_count": mean_key("action_mask_after_completion_safe_count"),
            "mean_action_mask_after_mobility_risk_count": mean_key("action_mask_after_mobility_risk_count"),
            "mean_action_mask_final_valid_count": mean_key("action_mask_final_valid_count"),
            "neighbor_visible_ratio": float(neighbor_available.mean()),
            "geo_visible_ratio": float(geo_available.mean()),
            "ground_visible_ratio": float(ground_available.mean()),
            "remote_available_ratio": float(remote_available.mean()),
            "neighbor_selected_when_visible_ratio": conditional_ratio("selected_neighbor_when_visible", "upper_mask_neighbor"),
            "geo_selected_when_visible_ratio": conditional_ratio("selected_geo_when_visible", "upper_mask_geo"),
            "ground_selected_when_visible_ratio": conditional_ratio("selected_ground_when_visible", "upper_mask_ground"),
            "remote_selected_when_visible_ratio": conditional_ratio("selected_remote_when_visible", "upper_mask_remote"),
            "mean_reward_local_selected": conditional_mean("upper_reward", "selected_local"),
            "mean_reward_neighbor_selected": conditional_mean("upper_reward", "selected_neighbor"),
            "mean_reward_geo_selected": conditional_mean("upper_reward", "selected_geo"),
            "mean_reward_ground_selected": conditional_mean("upper_reward", "selected_ground"),
            "mean_system_cost_local_selected": conditional_mean("system_cost", "selected_local"),
            "mean_system_cost_neighbor_selected": conditional_mean("system_cost", "selected_neighbor"),
            "mean_system_cost_geo_selected": conditional_mean("system_cost", "selected_geo"),
            "mean_system_cost_ground_selected": conditional_mean("system_cost", "selected_ground"),
            "mean_delay_cost_local_selected": conditional_mean("delay_cost", "selected_local"),
            "mean_delay_cost_neighbor_selected": conditional_mean("delay_cost", "selected_neighbor"),
            "mean_delay_cost_geo_selected": conditional_mean("delay_cost", "selected_geo"),
            "mean_delay_cost_ground_selected": conditional_mean("delay_cost", "selected_ground"),
            "mean_queue_cost_local_selected": conditional_mean("queue_cost", "selected_local"),
            "mean_queue_cost_neighbor_selected": conditional_mean("queue_cost", "selected_neighbor"),
            "mean_queue_cost_geo_selected": conditional_mean("queue_cost", "selected_geo"),
            "mean_queue_cost_ground_selected": conditional_mean("queue_cost", "selected_ground"),
            "mean_transmission_cost_local_selected": conditional_mean("transmission_cost", "selected_local"),
            "mean_transmission_cost_neighbor_selected": conditional_mean("transmission_cost", "selected_neighbor"),
            "mean_transmission_cost_geo_selected": conditional_mean("transmission_cost", "selected_geo"),
            "mean_transmission_cost_ground_selected": conditional_mean("transmission_cost", "selected_ground"),
            "mean_compute_cost_local_selected": conditional_mean("compute_cost", "selected_local"),
            "mean_compute_cost_neighbor_selected": conditional_mean("compute_cost", "selected_neighbor"),
            "mean_compute_cost_geo_selected": conditional_mean("compute_cost", "selected_geo"),
            "mean_compute_cost_ground_selected": conditional_mean("compute_cost", "selected_ground"),
            "mean_bonus_local_selected": conditional_mean("bonus_total", "selected_local"),
            "mean_bonus_neighbor_selected": conditional_mean("bonus_total", "selected_neighbor"),
            "mean_bonus_geo_selected": conditional_mean("bonus_total", "selected_geo"),
            "mean_bonus_ground_selected": conditional_mean("bonus_total", "selected_ground"),
            "mean_penalty_local_selected": conditional_mean("penalty_total", "selected_local"),
            "mean_penalty_neighbor_selected": conditional_mean("penalty_total", "selected_neighbor"),
            "mean_penalty_geo_selected": conditional_mean("penalty_total", "selected_geo"),
            "mean_penalty_ground_selected": conditional_mean("penalty_total", "selected_ground"),
            "mean_lower_effect_local_selected": conditional_mean("lower_allocation_effect", "selected_local"),
            "mean_lower_effect_neighbor_selected": conditional_mean("lower_allocation_effect", "selected_neighbor"),
            "mean_lower_effect_geo_selected": conditional_mean("lower_allocation_effect", "selected_geo"),
            "mean_lower_effect_ground_selected": conditional_mean("lower_allocation_effect", "selected_ground"),
            "mean_cost_local_selected": conditional_mean("upper_cost", "selected_local"),
            "mean_cost_neighbor_selected": conditional_mean("upper_cost", "selected_neighbor"),
            "mean_cost_geo_selected": conditional_mean("upper_cost", "selected_geo"),
            "mean_cost_ground_selected": conditional_mean("upper_cost", "selected_ground"),
            "upper_local_ratio": float(hist[0]),
            "upper_neighbor_ratio": float(hist[1]),
            "upper_geo_ratio": float(hist[2]),
            "upper_ground_ratio": float(hist[3]),
            "upper_remote_ratio": float(remote_actions),
        }
        summary.update(self.env.trace_stats())
        return summary

    def _deterministic_eval_summary(self, rollout: RolloutBuffer) -> Dict[str, float | str]:
        if len(rollout) == 0:
            return {
                "eval_argmax_local_ratio": 0.0,
                "eval_argmax_neighbor_ratio": 0.0,
                "eval_argmax_geo_ratio": 0.0,
                "eval_argmax_ground_ratio": 0.0,
                "eval_argmax_remote_ratio": 0.0,
                "eval_policy_entropy": 0.0,
                "eval_warning": "",
            }
        argmax_hist = torch.zeros(GeoLeoGroundEnv.N_UPPER_ACTIONS, dtype=torch.float32)
        entropy_terms = []
        for idx in range(len(rollout)):
            obs = rollout.obs[idx].to(self.device)
            edge_index = rollout.edge_index[idx].to(self.device)
            edge_attr = rollout.edge_attr[idx].to(self.device)
            diagnostics = self._inspect_policy(obs, edge_index, edge_attr)
            argmax_hist += torch.bincount(
                diagnostics["argmax_action"].detach().cpu(),
                minlength=GeoLeoGroundEnv.N_UPPER_ACTIONS,
            ).float()
            entropy_terms.append(diagnostics["entropy"].mean().detach().cpu())
        argmax_hist = argmax_hist / argmax_hist.sum().clamp_min(1.0)
        sampled_remote = 0.0
        if hasattr(rollout, "upper_action") and rollout.upper_action:
            actions = torch.cat([item.view(-1).detach().cpu() for item in rollout.upper_action]).long()
            sampled_remote = float((actions != GeoLeoGroundEnv.ACTION_LOCAL).float().mean())
        eval_warning = ""
        if float(argmax_hist[GeoLeoGroundEnv.ACTION_GROUND]) > 0.98 and sampled_remote > 0.05:
            eval_warning = "deterministic_policy_single_action_dominance"
        return {
            "eval_argmax_local_ratio": float(argmax_hist[0]),
            "eval_argmax_neighbor_ratio": float(argmax_hist[1]),
            "eval_argmax_geo_ratio": float(argmax_hist[2]),
            "eval_argmax_ground_ratio": float(argmax_hist[3]),
            "eval_argmax_remote_ratio": float(argmax_hist[1:].sum()),
            "eval_policy_entropy": float(torch.stack(entropy_terms).mean()) if entropy_terms else 0.0,
            "eval_warning": eval_warning,
        }

    def _inspect_policy(self, obs: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Dict[str, torch.Tensor]:
        try:
            embed = self.encoder(obs, edge_index, edge_attr, update_state=False)
        except TypeError:
            embed = self.encoder(obs, edge_index, edge_attr)
        mask = upper_action_mask_from_obs(obs)
        if hasattr(self.upper_agent, "actor"):
            logits = self.upper_agent.actor.compute_logits(embed, obs=obs)
        elif hasattr(self.upper_agent, "q_net"):
            logits = self.upper_agent.q_net(embed)
        else:
            raise RuntimeError(f"Unsupported upper agent for deterministic eval: {type(self.upper_agent).__name__}")
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min / 4)
        probs = torch.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        return {
            "argmax_action": probs.argmax(dim=-1),
            "entropy": dist.entropy(),
        }

    def _policy_probs(self, obs: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            try:
                embed = self.encoder(obs, edge_index, edge_attr, update_state=False)
            except TypeError:
                embed = self.encoder(obs, edge_index, edge_attr)
            mask = upper_action_mask_from_obs(obs)
            if hasattr(self.upper_agent, "actor"):
                logits = self.upper_agent.actor.compute_logits(embed, obs=obs)
            elif hasattr(self.upper_agent, "q_net"):
                logits = self.upper_agent.q_net(embed)
            else:
                raise RuntimeError(f"Unsupported upper agent for policy probs: {type(self.upper_agent).__name__}")
            masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min / 4)
            return torch.softmax(masked_logits, dim=-1)

    def _cost_prior_from_obs(self, obs: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        mask = action_mask.to(device=obs.device, dtype=torch.bool)
        if obs.shape[-1] >= SHARED_NODE_FEATURE_DIM_WITH_COST:
            costs = torch.stack(
                [
                    obs[:, IDX_LOCAL_NORMALIZED_COST],
                    obs[:, IDX_NEIGHBOR_NORMALIZED_COST],
                    obs[:, IDX_GEO_NORMALIZED_COST],
                    obs[:, IDX_GROUND_NORMALIZED_COST],
                ],
                dim=-1,
            )
        elif obs.shape[-1] >= SHARED_NODE_FEATURE_DIM:
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
            tx = torch.zeros_like(rates)
            tx[:, 1:] = 1.0 / rates[:, 1:].clamp_min(1.0e-6)
            compute = torch.relu(delays - tx)
            raw = delays + 0.5 * queues + 0.2 * tx + 0.2 * compute
            raw = raw.masked_fill(~mask, float("inf"))
            row_min = torch.where(torch.isfinite(raw), raw, torch.zeros_like(raw)).min(dim=-1, keepdim=True).values
            row_max = torch.where(torch.isfinite(raw), raw, torch.zeros_like(raw)).max(dim=-1, keepdim=True).values
            denom = (row_max - row_min).clamp_min(1.0e-6)
            costs = torch.where(torch.isfinite(raw), (raw - row_min) / denom, torch.ones_like(raw))
        else:
            # Legacy 12-dim observations do not include per-tier queue/cost
            # features; keep cost prior well-defined as uniform over feasible actions.
            prior = mask.float()
            return prior / prior.sum(dim=-1, keepdim=True).clamp_min(1.0)
        masked_cost = costs.masked_fill(~mask, torch.finfo(costs.dtype).max / 4)
        logits = -masked_cost / max(1.0e-6, float(self.cfg.policy_regularization.temperature))
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min / 4)
        return torch.softmax(logits, dim=-1)

    def _trace_labels_for_step(self, step: int) -> tuple[List[str], List[str]]:
        n = self.cfg.scenario.n_leo
        scenario_phase = ["unknown"] * n
        task_type = ["unknown"] * n
        provider = getattr(self.env, "_trace_provider", None)
        if provider is None:
            return scenario_phase, task_type
        step_rows = getattr(provider, "_by_step", {}).get(int(step), {})
        for leo in range(n):
            row = step_rows.get(leo)
            if not isinstance(row, dict):
                continue
            scenario_phase[leo] = str(row.get("scenario_phase", row.get("scenarioPhase", "unknown")))
            task_type[leo] = str(row.get("task_type", row.get("taskType", "unknown")))
        return scenario_phase, task_type

    def _write_rollout_debug(
        self,
        output_dir: Path,
        episode: int,
        rollout: RolloutBuffer,
        upper_losses: Dict[str, float],
    ) -> None:
        if len(rollout) == 0:
            return
        if not hasattr(self.upper_agent, "_gae") or not hasattr(self.upper_agent, "evaluate_actions"):
            return
        rewards_agents = torch.stack(rollout.reward).to(self.device)  # [T, N]
        stacked_values = torch.stack(rollout.value).to(self.device)
        if stacked_values.ndim == 1:
            stacked_values = stacked_values.unsqueeze(-1)
        values = self._to_agent_tensor(stacked_values, n_agents=rewards_agents.shape[1])
        dones = torch.tensor(rollout.done, dtype=torch.float32, device=self.device)
        credit_mode = str(getattr(self.cfg.algo, "credit_assignment", "global_team") or "global_team").strip().lower()
        if credit_mode == "per_agent":
            rewards_for_gae = rewards_agents
        else:
            rewards_for_gae = rewards_agents.mean(dim=-1)
            values = values.mean(dim=-1)
        try:
            returns, advantages = self.upper_agent._gae(rewards_for_gae, values, dones, normalize=True)  # type: ignore[attr-defined]
        except TypeError:
            returns, advantages = self.upper_agent._gae(rewards_for_gae, values, dones)  # type: ignore[attr-defined]
        if credit_mode == "per_agent":
            returns_agents = self._to_agent_tensor(returns, n_agents=rewards_agents.shape[1])
            adv_agents = self._to_agent_tensor(advantages, n_agents=rewards_agents.shape[1])
        else:
            returns_agents = returns.unsqueeze(-1).expand(returns.shape[0], rewards_agents.shape[1])
            adv_agents = advantages.unsqueeze(-1).expand(advantages.shape[0], rewards_agents.shape[1])
        rows: List[Dict[str, object]] = []
        for t in range(len(rollout)):
            obs = rollout.obs[t].to(self.device)
            edge_index = rollout.edge_index[t].to(self.device)
            edge_attr = rollout.edge_attr[t].to(self.device)
            action = rollout.upper_action[t].to(self.device).long()
            old_lp = rollout.log_prob[t].to(self.device)
            old_probs = rollout.old_action_probs[t].to(self.device) if t < len(rollout.old_action_probs) else self._policy_probs(obs, edge_index, edge_attr)
            eval_out = self.upper_agent.evaluate_actions(  # type: ignore[attr-defined]
                obs, edge_index, edge_attr, action
            )
            if isinstance(eval_out, tuple) and len(eval_out) >= 4:
                new_lp = eval_out[0]
                value_new = eval_out[2]
                new_probs = eval_out[3]
            elif isinstance(eval_out, tuple) and len(eval_out) == 3:
                new_lp = eval_out[0]
                value_new = eval_out[2]
                new_probs = self._policy_probs(obs, edge_index, edge_attr)
            else:
                raise RuntimeError(
                    f"Unexpected evaluate_actions output for {type(self.upper_agent).__name__}: {type(eval_out)}"
                )
            value_agents = self._to_agent_tensor(value_new.detach(), n_agents=obs.shape[0]).view(-1)
            oracle_action = None
            cost_prior = None
            if self._cost_prior_diagnostics_enabled:
                if t < len(rollout.oracle_action):
                    oracle_action = rollout.oracle_action[t].to(self.device).long()
                if t < len(rollout.cost_prior):
                    cost_prior = rollout.cost_prior[t].to(self.device)
                elif oracle_action is not None:
                    cost_prior = self._cost_prior_from_obs(obs, upper_action_mask_from_obs(obs))
                if oracle_action is None and cost_prior is not None:
                    oracle_action = torch.argmax(cost_prior, dim=-1)
            phase_labels = rollout.scenario_phase[t] if t < len(rollout.scenario_phase) else ["unknown"] * obs.shape[0]
            task_labels = rollout.task_type[t] if t < len(rollout.task_type) else ["unknown"] * obs.shape[0]
            step_idx = int(rollout.step_index[t]) if t < len(rollout.step_index) else t
            extra = rollout.extras[t] if t < len(rollout.extras) and isinstance(rollout.extras[t], dict) else {}
            for agent_id in range(obs.shape[0]):
                prior_cost = torch.full((4,), float("nan"), dtype=torch.float32, device=self.device)
                oracle_rank = -1
                oracle_action_idx = -1
                policy_oracle_prob = 0.0
                if cost_prior is not None:
                    prior_cost = cost_prior[agent_id].clamp_min(1.0e-9)
                if oracle_action is not None and cost_prior is not None:
                    oracle_action_idx = int(oracle_action[agent_id].item())
                    rank = torch.argsort(prior_cost)
                    oracle_rank = int((rank == oracle_action[agent_id]).nonzero(as_tuple=False)[0].item() + 1)
                    policy_oracle_prob = float(new_probs[agent_id, oracle_action[agent_id]].detach().cpu())
                rows.append(
                    {
                        "episode": episode,
                        "step": step_idx,
                        "agent_id": int(agent_id),
                        "scenario_phase": str(phase_labels[agent_id]) if agent_id < len(phase_labels) else "unknown",
                        "task_type": str(task_labels[agent_id]) if agent_id < len(task_labels) else "unknown",
                        "selected_action": int(action[agent_id].item()),
                        "oracle_action": oracle_action_idx,
                        "reward": float(rewards_agents[t, agent_id].detach().cpu()),
                        "return": float(returns_agents[t, agent_id].detach().cpu()),
                        "value": float(value_agents[agent_id].detach().cpu()),
                        "advantage": float(adv_agents[t, agent_id].detach().cpu()),
                        "old_logprob": float(old_lp[agent_id].detach().cpu()),
                        "new_logprob": float(new_lp[agent_id].detach().cpu()),
                        "prob_local": float(new_probs[agent_id, 0].detach().cpu()),
                        "prob_neighbor": float(new_probs[agent_id, 1].detach().cpu()),
                        "prob_geo": float(new_probs[agent_id, 2].detach().cpu()),
                        "prob_ground": float(new_probs[agent_id, 3].detach().cpu()),
                        "old_prob_local": float(old_probs[agent_id, 0].detach().cpu()),
                        "old_prob_neighbor": float(old_probs[agent_id, 1].detach().cpu()),
                        "old_prob_geo": float(old_probs[agent_id, 2].detach().cpu()),
                        "old_prob_ground": float(old_probs[agent_id, 3].detach().cpu()),
                        "cost_local": float(prior_cost[0].detach().cpu()),
                        "cost_neighbor": float(prior_cost[1].detach().cpu()),
                        "cost_geo": float(prior_cost[2].detach().cpu()),
                        "cost_ground": float(prior_cost[3].detach().cpu()),
                        "oracle_action_rank": oracle_rank,
                        "policy_oracle_prob": policy_oracle_prob,
                        "policy_cost_prior_kl": float(upper_losses.get("upper_policy_cost_prior_kl", 0.0)),
                        "upper_credit_assignment": credit_mode,
                        "train_phase": str(extra.get("train_phase", "")),
                        "lower_mode": str(extra.get("lower_mode", "")),
                    }
                )
        out_path = output_dir / "rollout_debug.csv"
        if not rows:
            return
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _maybe_update_upper(self, episode: int, rollout: RolloutBuffer) -> Dict[str, float]:
        every = max(1, int(getattr(self.cfg.algo, "upper_update_every", 1) or 1))
        if episode % every != 0:
            return {
                "upper_loss": 0.0,
                "upper_actor_loss": 0.0,
                "upper_policy_loss": 0.0,
                "upper_value_loss": 0.0,
                "upper_entropy": 0.0,
                "upper_approx_kl": 0.0,
                "upper_grad_norm": 0.0,
                "upper_update_skipped_by_cadence": True,
            }
        out = self.upper_agent.update(rollout, self.replay)
        self.upper_update_count += 1
        self.env_steps_since_upper_update = 0
        out["upper_update_skipped_by_cadence"] = False
        return out

    def _maybe_update_lower(self, episode: int, train_phase: str) -> Dict[str, float]:
        if not self._lower_training_enabled(train_phase):
            return self._empty_lower_update("lower_training_disabled")
        every = max(1, int(getattr(self.cfg.algo, "lower_update_every", 1) or 1))
        if episode % every != 0:
            return self._empty_lower_update("lower_update_skipped_by_cadence")
        updates = max(1, int(getattr(self.cfg.algo, "lower_updates_per_upper_update", 1) or 1))
        last: Dict[str, float] = {}
        for _ in range(updates):
            last = self.lower_agent.update(self.replay)
            self.lower_update_count += 1
        self.env_steps_since_lower_update = 0
        last["lower_updates_this_step"] = float(updates)
        return last

    def _empty_lower_update(self, reason: str) -> Dict[str, float]:
        return {
            "lower_actor_loss": 0.0,
            "lower_critic_loss": 0.0,
            "lower_encoder_grad_norm": 0.0,
            "shared_encoder_grad_norm_from_lower": 0.0,
            "separate_lower_encoder_grad_norm": 0.0,
            "lower_actor_grad_norm": 0.0,
            "lower_critic_grad_norm": 0.0,
            "lower_q_mean": 0.0,
            "lower_q_target_mean": 0.0,
            "lower_update_skip_reason": reason,
            "lower_updates_this_step": 0.0,
        }

    def _training_semantics_summary(self) -> Dict[str, object]:
        mode = self._lower_encoder_mode()
        stop_lower = bool(getattr(self.cfg.algo, "stop_gradient_to_encoder_from_lower", mode == "shared_upper_only"))
        return {
            "encoder_mode": mode,
            "lower_observation_mode": str(getattr(self.cfg.algo, "lower_observation_mode", "shared_embedding")),
            "stop_gradient_to_encoder_from_lower": stop_lower,
            "detach_embedding_during_action_collection": bool(getattr(self.cfg.algo, "detach_embedding_during_action_collection", True)),
            "lower_action_collection_embed_detached": bool(getattr(self.cfg.algo, "detach_embedding_during_action_collection", True)),
            "freeze_upper_during_lower_update": bool(getattr(self.cfg.algo, "freeze_upper_during_lower_update", True)),
            "separate_lower_encoder": bool(mode == "separate_lower_encoder"),
            "lower_encoder_grad_expected": bool(mode in {"shared_joint", "separate_lower_encoder"} and not stop_lower),
            "training_update_detach_semantics": (
                "lower_loss_updates_shared_encoder"
                if mode == "shared_joint" and not stop_lower
                else "lower_loss_does_not_update_shared_encoder"
                if mode != "separate_lower_encoder"
                else "lower_loss_updates_separate_lower_encoder"
            ),
            "action_collection_detach_semantics": "collection_no_grad_only_not_training_update_detach",
            "lower_actor_input_schema": "node_embedding + one_hot(upper_action)",
            "lower_critic_input_schema": "flatten(all_agent_embedding, upper_action_onehot, lower_action)",
            "credit_assignment_note": "lower action is conditioned on upper action; lower critic observes upper and lower actions",
        }

    def _lower_sensitivity_snapshot(self, obs: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Dict[str, object]:
        try:
            with torch.no_grad():
                try:
                    embed = self.encoder(obs, edge_index, edge_attr, update_state=False)
                except TypeError:
                    embed = self.encoder(obs, edge_index, edge_attr)
            return lower_action_sensitivity_to_upper_action(
                self.lower_agent,
                embed.detach(),
                obs=obs,
                edge_index=edge_index,
                edge_attr=edge_attr,
                n_upper_actions=GeoLeoGroundEnv.N_UPPER_ACTIONS,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "lower_action_sensitivity_to_upper_action": 0.0,
                "lower_action_variance": 0.0,
                "lower_allocator_not_conditioned_effectively": True,
                "unavailable_reason": repr(exc),
            }

    def _write_gradient_diagnostics(self, output_dir: Path) -> None:
        if not self.gradient_diagnostics_history:
            return
        path = output_dir / "gradient_report.csv"
        keys: List[str] = []
        for row in self.gradient_diagnostics_history:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.gradient_diagnostics_history:
                clean = dict(row)
                if isinstance(clean.get("unavailable_fields"), dict):
                    clean["unavailable_fields"] = json.dumps(clean["unavailable_fields"], ensure_ascii=False)
                writer.writerow(clean)

    @staticmethod
    def _to_agent_tensor(tensor: torch.Tensor, *, n_agents: int) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor.view(1, 1).expand(1, n_agents)
        if tensor.ndim == 1:
            if tensor.shape[0] == n_agents:
                return tensor.unsqueeze(0)
            return tensor.unsqueeze(-1).expand(tensor.shape[0], n_agents)
        return tensor

    def _episode_train_phase(self, episode: int) -> str:
        upper_pre = int(max(0, getattr(self.cfg, "upper_pretrain_episodes", 0)))
        if upper_pre > 0 and episode <= upper_pre:
            return "upper_pretrain"
        return "joint_train"

    def _use_neutral_lower(self, train_phase: str) -> bool:
        mode = str(getattr(self.cfg, "lower_action_mode", "learned") or "learned").strip().lower()
        if mode == "neutral_allocator":
            return True
        return train_phase == "upper_pretrain" and int(max(0, getattr(self.cfg, "upper_pretrain_episodes", 0))) > 0

    def _lower_training_enabled(self, train_phase: str) -> bool:
        if not bool(getattr(self.cfg, "lower_training_enabled", True)):
            return False
        return train_phase != "upper_pretrain"

    def _neutral_lower_action(self, n_agents: int) -> torch.Tensor:
        return torch.ones((n_agents, GeoLeoGroundEnv.LOWER_ACTION_DIM), dtype=torch.float32, device=self.device)

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.cfg),
            "encoder": self.encoder.state_dict(),
            "upper_agent_class": self.upper_agent.__class__.__name__,
            "lower_agent_class": self.lower_agent.__class__.__name__,
            "encoder_mode_metadata": checkpoint_encoder_metadata(self.cfg.algo),
        }
        for name in ["actor", "critic", "value", "q_net", "mixer", "target_q_net", "target_mixer"]:
            module = getattr(self.upper_agent, name, None)
            if module is not None and hasattr(module, "state_dict"):
                payload[f"upper_{name}"] = module.state_dict()
        for name in ["encoder", "target_encoder", "actor", "critic", "target_actor", "target_critic"]:
            module = getattr(self.lower_agent, name, None)
            if module is not None and hasattr(module, "state_dict"):
                payload[f"lower_{name}"] = module.state_dict()
        torch.save(payload, path)
