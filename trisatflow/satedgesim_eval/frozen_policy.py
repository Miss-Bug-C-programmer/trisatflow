from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Tuple

import torch

from trisatflow.config import (
    AlgoConfig,
    EvaluationConfig,
    ExperimentConfig,
    PolicyRegularizationConfig,
    RewardWeights,
    ScenarioConfig,
    TrainConfig,
)
from trisatflow.encoder_modes import validate_checkpoint_encoder_metadata
from trisatflow.envs import GeoLeoGroundEnv
from trisatflow.envs.obs_builder import canonical_row, load_observation_normalization_stats
from trisatflow.envs.obs_schema import ACTION_NAMES
from trisatflow.models import (
    CentralPerAgentValue,
    CentralValue,
    FeatureEncoder,
    LocalLowerCritic,
    LowerActor,
    LowerCritic,
    StochasticLowerActor,
    TopologyEncoder,
    UpperMAPPOPolicy,
    UpperQNetwork,
    upper_action_mask_from_obs,
)

EVAL_MODES = {"raw_argmax", "stochastic_eval", "margin_cost_tiebreak", "cost_greedy_baseline"}


def _dataclass_from_dict(cls, data: Dict[str, Any]):
    if not isinstance(data, dict):
        return cls()
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in allowed})


def _train_config_from_checkpoint(raw: Dict[str, Any], *, device: str) -> TrainConfig:
    if not isinstance(raw, dict):
        raw = {}
    return TrainConfig(
        total_episodes=int(raw.get("total_episodes", 1) or 1),
        log_interval=int(raw.get("log_interval", 1) or 1),
        device=device,
        output_dir=str(raw.get("output_dir", "outputs/satedgesim_replay")),
        scenario=_dataclass_from_dict(ScenarioConfig, raw.get("scenario", {})),
        reward=_dataclass_from_dict(RewardWeights, raw.get("reward", {})),
        policy_regularization=_dataclass_from_dict(PolicyRegularizationConfig, raw.get("policy_regularization", {})),
        algo=_dataclass_from_dict(AlgoConfig, raw.get("algo", {})),
        evaluation=_dataclass_from_dict(EvaluationConfig, raw.get("evaluation", {})),
        experiment=_dataclass_from_dict(ExperimentConfig, raw.get("experiment", {})),
    )


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _visible(canonical: Mapping[str, Any], action_idx: int) -> bool:
    names = ["local", "neighbor", "geo", "ground"]
    name = names[int(max(0, min(3, action_idx)))]
    return bool(_to_float(canonical.get(f"{name}_visible", 0.0), 0.0) > 0.5)


def _action_costs_from_canonical_row(
    canonical: Mapping[str, Any],
    reward: RewardWeights,
) -> List[float]:
    costs: List[float] = []
    for idx, name in enumerate(["local", "neighbor", "geo", "ground"]):
        if not _visible(canonical, idx):
            costs.append(float("inf"))
            continue
        delay = max(0.0, _to_float(canonical.get(f"{name}_delay"), 0.0))
        queue = max(0.0, _to_float(canonical.get(f"{name}_queue"), 0.0))
        rate = max(1.0e-9, _to_float(canonical.get(f"{name}_rate"), 0.0))
        transmission = 0.0 if idx == GeoLeoGroundEnv.ACTION_LOCAL else (1.0 / rate)
        compute = max(0.0, delay - transmission)
        costs.append(
            reward.delay_weight * delay
            + reward.queue_weight * queue
            + reward.transmission_weight * transmission
            + reward.compute_weight * compute
        )
    return costs


class FrozenTriSatFlowPolicy:
    """Evaluation-only checkpoint runner.

    This class intentionally avoids HierarchicalTrainer so replay does not
    construct a training environment, optimizer step path, or replay buffer.
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")

        requested_device = device
        actual_device = device
        self.device_fallback_reason: str | None = None
        if device.startswith("cuda") and not torch.cuda.is_available():
            actual_device = "cpu"
            self.device_fallback_reason = (
                f"requested device '{requested_device}' but torch.cuda.is_available() is False; falling back to cpu"
            )
            print(f"[FrozenTriSatFlowPolicy] {self.device_fallback_reason}")

        self.requested_device = requested_device
        self.device = torch.device(actual_device)
        payload = torch.load(self.checkpoint_path, map_location=self.device)
        if "config" not in payload:
            raise KeyError(f"checkpoint has no 'config' field: {self.checkpoint_path}")

        self.cfg = _train_config_from_checkpoint(payload["config"], device=str(self.device))
        self.encoder_mode_metadata = validate_checkpoint_encoder_metadata(
            payload,
            self.cfg.algo,
            formal=bool(getattr(getattr(self.cfg, "experiment", None), "paper_ready", False)),
        )
        self.obs_normalization_mode = str(getattr(self.cfg.scenario, "obs_normalization_mode", "legacy") or "legacy").strip().lower()
        self.obs_normalization_path = str(getattr(self.cfg.scenario, "obs_normalization_path", "") or "").strip()
        strict_obs_norm = self.obs_normalization_mode == "trace_log_quantile"
        (
            self.obs_normalization_mode,
            resolved_norm_path,
            self.obs_normalization_stats,
            self.obs_normalization_loaded,
        ) = load_observation_normalization_stats(
            self.obs_normalization_mode,
            self.obs_normalization_path,
            strict=strict_obs_norm,
        )
        self.obs_normalization_path = resolved_norm_path or self.obs_normalization_path
        self.cfg.scenario.obs_normalization_mode = self.obs_normalization_mode
        self.cfg.scenario.obs_normalization_path = self.obs_normalization_path

        self.loaded_required_modules: List[str] = []
        self.loaded_optional_modules: List[str] = []
        self.skipped_optional_modules: List[str] = []
        self.missing_required_modules: List[str] = []

        required_modules = ["encoder", "upper_actor", "lower_actor"]
        for key in required_modules:
            if key not in payload:
                self.missing_required_modules.append(key)
        if self.missing_required_modules:
            missing = ", ".join(self.missing_required_modules)
            raise KeyError(f"checkpoint is missing required module state_dict(s): {missing}")

        self.encoder = self._build_encoder().to(self.device)
        self.upper_actor = self._build_upper_actor().to(self.device)
        self.lower_actor = self._build_lower_actor().to(self.device)
        self.optional_modules = self._build_optional_modules()

        self._load_module(self.encoder, payload, "encoder", strict=True)
        self.loaded_required_modules.append("encoder")
        self._load_module(self.upper_actor, payload, "upper_actor", strict=True)
        self.loaded_required_modules.append("upper_actor")
        self._load_module(self.lower_actor, payload, "lower_actor", strict=True)
        self.loaded_required_modules.append("lower_actor")
        for key, module in self.optional_modules.items():
            loaded = self._load_module(module, payload, key, strict=False)
            if loaded:
                self.loaded_optional_modules.append(key)
            else:
                self.skipped_optional_modules.append(key)

        self._eval_all()

    def _build_encoder(self) -> torch.nn.Module:
        encoder_cls = TopologyEncoder if self.cfg.scenario.enable_gnn else FeatureEncoder
        return encoder_cls(
            node_dim=self.cfg.scenario.node_feature_dim,
            edge_dim=self.cfg.scenario.edge_feature_dim,
            hidden_dim=self.cfg.algo.gnn_hidden_dim,
        )

    def _build_upper_actor(self) -> torch.nn.Module:
        upper = self.cfg.algo.upper_algo
        if upper in {"mappo", "ippo"}:
            return UpperMAPPOPolicy(
                self.cfg.algo.gnn_hidden_dim,
                self.cfg.algo.policy_hidden_dim,
                GeoLeoGroundEnv.N_UPPER_ACTIONS,
                policy_head=str(getattr(self.cfg.algo, "policy_head", "gnn_only") or "gnn_only"),
                logit_centering=bool(getattr(self.cfg.algo, "logit_centering", False)),
            )
        if upper in {"iql", "vdn", "qmix"}:
            return UpperQNetwork(
                self.cfg.algo.gnn_hidden_dim,
                self.cfg.algo.policy_hidden_dim,
                GeoLeoGroundEnv.N_UPPER_ACTIONS,
            )
        raise ValueError(f"unsupported upper algorithm for replay: {upper}")

    def _build_lower_actor(self) -> torch.nn.Module:
        lower = self.cfg.algo.lower_algo
        if lower in {"maddpg", "iddpg"}:
            return LowerActor(
                self.cfg.algo.gnn_hidden_dim,
                GeoLeoGroundEnv.N_UPPER_ACTIONS,
                self.cfg.algo.policy_hidden_dim,
                GeoLeoGroundEnv.LOWER_ACTION_DIM,
            )
        if lower in {"masac", "isac"}:
            return StochasticLowerActor(
                self.cfg.algo.gnn_hidden_dim,
                GeoLeoGroundEnv.N_UPPER_ACTIONS,
                self.cfg.algo.policy_hidden_dim,
                GeoLeoGroundEnv.LOWER_ACTION_DIM,
            )
        raise ValueError(f"unsupported lower algorithm for replay: {lower}")

    def _build_optional_modules(self) -> Dict[str, torch.nn.Module]:
        modules: Dict[str, torch.nn.Module] = {}
        n_agents = max(
            1,
            self.cfg.scenario.n_leo
            + (1 if self.cfg.scenario.enable_geo else 0)
            + (1 if self.cfg.scenario.enable_ground else 0),
        )
        upper = self.cfg.algo.upper_algo
        lower = self.cfg.algo.lower_algo
        if upper == "mappo":
            credit = str(getattr(self.cfg.algo, "credit_assignment", "global_team") or "global_team").strip().lower()
            if credit == "per_agent":
                modules["upper_critic"] = CentralPerAgentValue(self.cfg.algo.gnn_hidden_dim, self.cfg.algo.policy_hidden_dim).to(self.device)
            else:
                modules["upper_critic"] = CentralValue(self.cfg.algo.gnn_hidden_dim, n_agents, self.cfg.algo.policy_hidden_dim).to(self.device)
        if lower == "maddpg":
            modules["lower_critic"] = LowerCritic(
                self.cfg.algo.gnn_hidden_dim,
                n_agents,
                GeoLeoGroundEnv.N_UPPER_ACTIONS,
                GeoLeoGroundEnv.LOWER_ACTION_DIM,
                self.cfg.algo.policy_hidden_dim,
            ).to(self.device)
            modules["lower_target_actor"] = LowerActor(
                self.cfg.algo.gnn_hidden_dim,
                GeoLeoGroundEnv.N_UPPER_ACTIONS,
                self.cfg.algo.policy_hidden_dim,
                GeoLeoGroundEnv.LOWER_ACTION_DIM,
            ).to(self.device)
            modules["lower_target_critic"] = LowerCritic(
                self.cfg.algo.gnn_hidden_dim,
                n_agents,
                GeoLeoGroundEnv.N_UPPER_ACTIONS,
                GeoLeoGroundEnv.LOWER_ACTION_DIM,
                self.cfg.algo.policy_hidden_dim,
            ).to(self.device)
        elif lower == "iddpg":
            modules["lower_critic"] = LocalLowerCritic(
                self.cfg.algo.gnn_hidden_dim,
                GeoLeoGroundEnv.N_UPPER_ACTIONS,
                GeoLeoGroundEnv.LOWER_ACTION_DIM,
                self.cfg.algo.policy_hidden_dim,
            ).to(self.device)
        elif lower in {"masac", "isac"}:
            critic_cls = LowerCritic if lower == "masac" else LocalLowerCritic
            if lower == "masac":
                modules["lower_critic"] = critic_cls(
                    self.cfg.algo.gnn_hidden_dim,
                    n_agents,
                    GeoLeoGroundEnv.N_UPPER_ACTIONS,
                    GeoLeoGroundEnv.LOWER_ACTION_DIM,
                    self.cfg.algo.policy_hidden_dim,
                ).to(self.device)
            else:
                modules["lower_critic"] = critic_cls(
                    self.cfg.algo.gnn_hidden_dim,
                    GeoLeoGroundEnv.N_UPPER_ACTIONS,
                    GeoLeoGroundEnv.LOWER_ACTION_DIM,
                    self.cfg.algo.policy_hidden_dim,
                ).to(self.device)
        return modules

    @staticmethod
    def _load_module(module: torch.nn.Module, payload: Dict[str, Any], key: str, *, strict: bool) -> bool:
        if key not in payload:
            if strict:
                raise KeyError(f"checkpoint is missing required module state_dict: {key}")
            return False
        try:
            module.load_state_dict(payload[key], strict=strict)
            return True
        except RuntimeError:
            if strict:
                raise
            print(f"[FrozenTriSatFlowPolicy] skipping optional module '{key}' due to shape mismatch")
            return False

    def _eval_all(self) -> None:
        self.encoder.eval()
        self.upper_actor.eval()
        self.lower_actor.eval()
        for module in self.optional_modules.values():
            module.eval()

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        *,
        source_index: int = 0,
        deterministic: bool = True,
        eval_mode: str | None = None,
        tie_break_eps: float | None = None,
        stochastic_seed: int | None = None,
        raw_rows: List[Mapping[str, Any]] | None = None,
    ) -> Tuple[int, List[float], Dict[str, Any]]:
        obs = obs.to(self.device).float()
        edge_index = edge_index.to(self.device).long()
        edge_attr = edge_attr.to(self.device).float()

        if obs.ndim != 2:
            raise ValueError(f"obs must have shape [N, F], got {tuple(obs.shape)}")
        if obs.shape[0] == 0:
            raise ValueError("obs has zero nodes")

        source_index = int(max(0, min(source_index, obs.shape[0] - 1)))
        embed = self.encoder(obs, edge_index, edge_attr)
        diagnostics = self.inspect_upper_policy(obs, edge_index, edge_attr)
        resolved_mode = (eval_mode or ("raw_argmax" if deterministic else "stochastic_eval")).strip().lower()
        if resolved_mode not in EVAL_MODES:
            raise ValueError(f"unsupported eval_mode={resolved_mode!r}; expected one of {sorted(EVAL_MODES)}")
        eps = float(tie_break_eps if tie_break_eps is not None else self.cfg.evaluation.tie_break_eps)
        rng = random.Random(
            int(
                stochastic_seed
                if stochastic_seed is not None
                else self.cfg.evaluation.stochastic_seed + max(0, source_index)
            )
        )
        selection = self.select_action_from_diagnostics(
            diagnostics,
            source_index=source_index,
            raw_rows=raw_rows,
            eval_mode=resolved_mode,
            tie_break_eps=eps,
            rng=rng,
        )
        upper_action = int(selection["final_action"])
        upper_all = diagnostics["argmax_action"].clone()
        upper_all[source_index] = upper_action
        lower_all = self._lower_actions(embed, upper_all, deterministic=deterministic)
        lower_action_tensor = lower_all[source_index].detach().cpu().float().clamp(0.0, 1.0)
        lower_action = [float(x) for x in lower_action_tensor.tolist()]
        if len(lower_action) < 3:
            lower_action = (lower_action + [1.0, 1.0, 1.0])[:3]

        debug = {
            "requested_device": self.requested_device,
            "device": str(self.device),
            "device_fallback_reason": self.device_fallback_reason,
            "upper_algo": self.cfg.algo.upper_algo,
            "lower_algo": self.cfg.algo.lower_algo,
            "num_policy_nodes": int(obs.shape[0]),
            "source_index": source_index,
            "eval_mode": resolved_mode,
            "tie_break_eps": eps,
            "raw_argmax_action": int(selection["raw_argmax_action"]),
            "raw_argmax_action_name": ACTION_NAMES[int(selection["raw_argmax_action"])].upper(),
            "final_action": int(selection["final_action"]),
            "final_action_name": ACTION_NAMES[int(selection["final_action"])].upper(),
            "tie_break_applied": bool(selection["tie_break_applied"]),
            "tie_break_candidate_actions": [int(i) for i in selection["tie_break_candidate_actions"]],
            "tie_break_candidate_action_names": [ACTION_NAMES[int(i)].upper() for i in selection["tie_break_candidate_actions"]],
            "selected_by_policy_prob": float(selection["selected_by_policy_prob"]),
            "selected_by_cost_rank": int(selection["selected_by_cost_rank"]),
            "cost_by_action": [float(v) for v in selection["cost_by_action"]],
        }
        probs = diagnostics["probs"][source_index].detach().cpu().float().tolist()
        while len(probs) < GeoLeoGroundEnv.N_UPPER_ACTIONS:
            probs.append(0.0)
        debug["upper_probs"] = [float(x) for x in probs[: GeoLeoGroundEnv.N_UPPER_ACTIONS]]
        debug["upper_prob_local"] = debug["upper_probs"][0]
        debug["upper_prob_neighbor"] = debug["upper_probs"][1]
        debug["upper_prob_geo"] = debug["upper_probs"][2]
        debug["upper_prob_ground"] = debug["upper_probs"][3]
        return upper_action, lower_action[:3], debug

    @torch.no_grad()
    def inspect_upper_policy(
        self,
        obs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        obs = obs.to(self.device).float()
        edge_index = edge_index.to(self.device).long()
        edge_attr = edge_attr.to(self.device).float()
        embed = self.encoder(obs, edge_index, edge_attr)
        action_mask = upper_action_mask_from_obs(obs)
        if isinstance(self.upper_actor, UpperMAPPOPolicy):
            logits, details = self.upper_actor.compute_logits(embed, obs=obs, return_details=True)
        else:
            logits = self.upper_actor(embed)
            details = {
                "policy_hidden": embed,
                "action_features": torch.zeros((embed.shape[0], 4, 6), dtype=embed.dtype, device=embed.device),
                "global_context": embed.mean(dim=0, keepdim=True).expand_as(embed),
            }
        masked_logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min / 4)
        probs = torch.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        return {
            "mask": action_mask,
            "obs": obs,
            "embed": embed,
            "policy_hidden": details.get("policy_hidden", embed),
            "action_features": details.get("action_features"),
            "global_context": details.get("global_context"),
            "logits": logits,
            "masked_logits": masked_logits,
            "probs": probs,
            "argmax_action": probs.argmax(dim=-1),
            "sampled_action": dist.sample(),
            "entropy": dist.entropy(),
        }

    def _upper_actions(
        self,
        embed: torch.Tensor,
        *,
        action_mask: torch.Tensor,
        deterministic: bool,
        obs: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(self.upper_actor, UpperMAPPOPolicy):
            dist = self.upper_actor(embed, action_mask=action_mask, obs=obs)
            if deterministic:
                return dist.probs.argmax(dim=-1), dist.probs
            return dist.sample(), dist.probs
        q_values = self.upper_actor(embed)
        if q_values.ndim != 2:
            raise ValueError(f"unexpected upper Q tensor shape: {tuple(q_values.shape)}")
        mask = action_mask.to(device=q_values.device, dtype=torch.bool)
        masked_q = q_values.masked_fill(~mask, torch.finfo(q_values.dtype).min / 4)
        probs = torch.softmax(masked_q, dim=-1)
        if deterministic:
            return masked_q.argmax(dim=-1), probs
        return torch.distributions.Categorical(logits=masked_q).sample(), probs

    def _lower_actions(self, embed: torch.Tensor, upper_action: torch.Tensor, *, deterministic: bool) -> torch.Tensor:
        if isinstance(self.lower_actor, StochasticLowerActor):
            if deterministic:
                return self.lower_actor.mean_action(embed, upper_action).clamp(0.0, 1.0)
            action, _ = self.lower_actor.sample(embed, upper_action)
            return action.clamp(0.0, 1.0)
        return self.lower_actor(embed, upper_action).clamp(0.0, 1.0)

    @torch.no_grad()
    def select_action_from_diagnostics(
        self,
        diagnostics: Dict[str, torch.Tensor],
        *,
        source_index: int,
        raw_rows: List[Mapping[str, Any]] | None,
        eval_mode: str,
        tie_break_eps: float,
        rng: random.Random,
    ) -> Dict[str, Any]:
        probs = diagnostics["probs"][source_index].detach().cpu()
        mask = diagnostics["mask"][source_index].detach().cpu().bool()
        raw_argmax = int(torch.argmax(probs).item())
        feasible = [idx for idx in range(min(GeoLeoGroundEnv.N_UPPER_ACTIONS, probs.numel())) if bool(mask[idx])]
        if not feasible:
            feasible = [GeoLeoGroundEnv.ACTION_LOCAL]
        top_probs, top_idx = torch.sort(probs[mask], descending=True)
        top1_prob = float(top_probs[0].item()) if top_probs.numel() > 0 else 0.0
        top2_prob = float(top_probs[1].item()) if top_probs.numel() > 1 else top1_prob
        raw_row = raw_rows[source_index] if raw_rows and 0 <= source_index < len(raw_rows) else {}
        canonical = canonical_row(raw_row) if raw_row else {"local_visible": 1.0}
        cost_by_action = _action_costs_from_canonical_row(canonical, self.cfg.reward)
        finite_costs = [(idx, cost_by_action[idx]) for idx in feasible if torch.isfinite(torch.tensor(cost_by_action[idx]))]
        if not finite_costs:
            finite_costs = [(idx, 0.0) for idx in feasible]
        finite_costs.sort(key=lambda item: item[1])
        cost_rank = {idx: rank + 1 for rank, (idx, _) in enumerate(finite_costs)}
        cost_best = int(finite_costs[0][0])

        final_action = raw_argmax
        tie_break_applied = False
        tie_break_candidates: List[int] = []
        if eval_mode == "raw_argmax":
            final_action = raw_argmax
        elif eval_mode == "stochastic_eval":
            weights = [max(0.0, float(probs[idx].item())) for idx in feasible]
            total = sum(weights)
            if total <= 1.0e-12:
                final_action = raw_argmax
            else:
                normed = [w / total for w in weights]
                final_action = int(rng.choices(feasible, weights=normed, k=1)[0])
        elif eval_mode == "cost_greedy_baseline":
            final_action = cost_best
        elif eval_mode == "margin_cost_tiebreak":
            sorted_idx = torch.argsort(probs, descending=True)
            top1 = int(sorted_idx[0].item())
            tie_break_candidates = [top1]
            for k in range(1, int(sorted_idx.numel())):
                idx = int(sorted_idx[k].item())
                if not bool(mask[idx]):
                    continue
                if (top1_prob - float(probs[idx].item())) <= tie_break_eps + 1.0e-12:
                    tie_break_candidates.append(idx)
                else:
                    break
            tie_break_candidates = [idx for idx in tie_break_candidates if idx in feasible]
            if not tie_break_candidates:
                tie_break_candidates = [top1]
            if (top1_prob - top2_prob) <= tie_break_eps + 1.0e-12:
                tie_break_applied = True
                final_action = min(
                    tie_break_candidates,
                    key=lambda idx: (cost_by_action[idx], -float(probs[idx].item()), idx),
                )
            else:
                final_action = top1
        else:
            raise ValueError(f"unsupported eval_mode={eval_mode!r}")
        final_action = int(final_action)
        return {
            "raw_argmax_action": raw_argmax,
            "final_action": final_action,
            "top1_prob": top1_prob,
            "top2_prob": top2_prob,
            "top1_prob_margin": float(top1_prob - top2_prob),
            "tie_break_applied": bool(tie_break_applied),
            "tie_break_candidate_actions": tie_break_candidates,
            "selected_by_policy_prob": float(probs[final_action].item()),
            "selected_by_cost_rank": int(cost_rank.get(final_action, len(finite_costs) + 1)),
            "cost_by_action": cost_by_action,
        }
