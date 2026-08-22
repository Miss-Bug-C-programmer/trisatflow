from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import math
import torch
import torch.nn.functional as F

from trisatflow.config import RewardWeights, ScenarioConfig
from trisatflow.envs.action_masks import ActionMaskDiagnostics, build_upper_action_mask
from trisatflow.envs.obs_builder import build_shared_observation, load_observation_normalization_stats
from trisatflow.envs.obs_schema import (
    IDX_GEO_RATE,
    IDX_GROUND_RATE,
    IDX_LOCAL_QUEUE,
    IDX_LOCAL_RATE,
    IDX_NEIGHBOR_RATE,
    SHARED_NODE_FEATURE_DIM,
)
from trisatflow.envs.physical_metrics import build_step_metric_bundle, step_bundle_to_info
from trisatflow.envs.physical_model import PhysicalStepOutput, compute_physical_step
from trisatflow.envs.task_workload import sample_task_workload_batch
from trisatflow.envs.topology_trace import TopologyTraceProvider, TraceTopologySnapshot
from trisatflow.envs.units import TraceDelayInterpretation, UnitScaleConfig
from trisatflow.envs.mask_noise import apply_prediction_noise
from trisatflow.envs.mask_predictors import predict_masks_from_observables, predict_masks_from_physical_observables
from trisatflow.baselines.registry import apply_architecture_filter, normalize_architecture


@dataclass
class StepOutput:
    obs: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    upper_reward: torch.Tensor
    lower_reward: torch.Tensor
    done: bool
    info: Dict[str, torch.Tensor]


class GeoLeoGroundEnv:
    """GEO-LEO-Ground computation-offloading simulator.

    The upper layer chooses one of four executable offloading directions:
        0 local LEO, 1 neighboring LEO, 2 GEO/cloud tier, 3 ground/edge tier.

    The environment exposes a dynamic action mask through the observation. GEO
    and ground are retained as first-class actions, but they are feasible only
    when the current topology provides an available candidate. This mirrors the
    candidate-list semantics of SatEdgeSim: a policy should learn where to send
    a task among currently reachable local/neighbor/GEO/ground targets, rather
    than learn to recover from selecting links that do not exist.
    """

    ACTION_LOCAL = 0
    ACTION_NEIGHBOR = 1
    ACTION_GEO = 2
    ACTION_GROUND = 3
    N_UPPER_ACTIONS = 4
    LOWER_ACTION_DIM = 3

    def __init__(
        self,
        scenario: ScenarioConfig | None = None,
        reward_weights: RewardWeights | None = None,
        device: str | torch.device = "cpu",
    ):
        self.cfg = scenario or ScenarioConfig()
        self.weights = reward_weights or RewardWeights()
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.cfg.seed)
        self.t = 0
        self.queue = torch.zeros(self.cfg.n_leo, device=self.device)
        self.leo_queue = self.queue
        self.geo_queue = torch.zeros(self.cfg.n_leo, device=self.device)
        self.ground_queue = torch.zeros(self.cfg.n_leo, device=self.device)
        self.ground_station_queue = torch.zeros(self.cfg.n_leo, device=self.device)
        self.virtual_delay_queue = torch.zeros(self.cfg.n_leo, device=self.device)
        self.energy = torch.full((self.cfg.n_leo,), self.cfg.leo_energy_init, device=self.device)
        self.cumulative_total_system_energy_j = torch.zeros(self.cfg.n_leo, device=self.device)
        self.phase = torch.linspace(0, 2 * math.pi, self.cfg.n_leo + 1, device=self.device)[:-1]
        self.last_arrivals = torch.zeros(self.cfg.n_leo, device=self.device)
        self.last_service = torch.zeros(self.cfg.n_leo, device=self.device)
        self.last_task_bits = torch.ones(self.cfg.n_leo, device=self.device)
        self.last_cycles_per_bit = torch.ones(self.cfg.n_leo, device=self.device)
        self.episode_action_counts = torch.zeros((self.cfg.n_leo, self.N_UPPER_ACTIONS), device=self.device)
        self.last_metrics: Dict[str, torch.Tensor] = {}
        self._trace_snapshot_cache_step: int | None = None
        self._trace_snapshot_cache_value: TraceTopologySnapshot | None = None
        self._trace_provider: TopologyTraceProvider | None = None
        self._action_mask_cache_step: int | None = None
        self._action_mask_cache_value: ActionMaskDiagnostics | None = None
        # Topology-derived tensors depend only on the slot and immutable
        # scenario configuration.  Reuse them across repeated evaluation
        # episodes instead of rebuilding small graphs and masks millions of
        # times.  Training semantics are unchanged because queue-dependent
        # quantities are not cached here.
        self._graph_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._neighbor_rate_cache: Dict[int, torch.Tensor] = {}
        self._geo_access_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = {}
        self._ground_access_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = {}
        self._action_mask_details_cache: Dict[int, ActionMaskDiagnostics] = {}
        self._mask_prediction_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        if self.cfg.topology_mode == "satedgesim_trace" and self.cfg.topology_trace_path:
            self._trace_provider = TopologyTraceProvider(
                self.cfg.topology_trace_path,
                n_leo=self.cfg.n_leo,
                device=self.device,
                repeat=self.cfg.topology_trace_repeat,
                strict=self.cfg.topology_trace_strict,
            )
        elif self.cfg.topology_mode not in {"analytic", "satedgesim_trace"}:
            raise ValueError(f"Unsupported topology_mode={self.cfg.topology_mode!r}; use 'analytic' or 'satedgesim_trace'.")
        self._obs_norm_stats: Dict[str, Dict[str, float]] = {}
        self._obs_norm_mode = str(getattr(self.cfg, "obs_normalization_mode", "legacy") or "legacy").strip().lower()
        norm_path = str(getattr(self.cfg, "obs_normalization_path", "") or "").strip()
        strict_norm = self._obs_norm_mode == "trace_log_quantile"
        _, resolved_norm_path, stats, _ = load_observation_normalization_stats(
            self._obs_norm_mode,
            norm_path,
            strict=strict_norm,
        )
        if resolved_norm_path:
            self.cfg.obs_normalization_path = resolved_norm_path
        if isinstance(stats, dict):
            self._obs_norm_stats = dict(stats)
        self._unit_scale = UnitScaleConfig(
            delay_s_per_unit=float(getattr(self.cfg, "delay_s_per_unit", 1.0)),
            energy_j_per_unit=float(getattr(self.cfg, "energy_j_per_unit", 1.0)),
            queue_cycles_per_unit=float(getattr(self.cfg, "queue_cycles_per_unit", 1.0)),
            cpu_ghz_per_unit=float(getattr(self.cfg, "cpu_ghz_per_unit", 1.0)),
            rate_mbps_per_unit=float(getattr(self.cfg, "rate_mbps_per_unit", 1.0)),
            bandwidth_mbps_per_unit=float(getattr(self.cfg, "bandwidth_mbps_per_unit", 1.0)),
            power_w_per_unit=float(getattr(self.cfg, "power_w_per_unit", 1.0)),
            task_size_bits_per_unit=float(getattr(self.cfg, "task_size_bits_per_unit", 1.0)),
            workload_cycles_per_unit=float(getattr(self.cfg, "workload_cycles_per_unit", 1.0)),
        )
        self._trace_delay_interpretation = TraceDelayInterpretation(
            anomaly_threshold_s=float(getattr(self.cfg, "trace_delay_anomaly_threshold_s", 1.0e3)),
            treat_anomaly_as_legacy_score=bool(getattr(self.cfg, "trace_treat_large_delay_as_legacy_score", True)),
        )

    @property
    def n_agents(self) -> int:
        return self.cfg.n_leo

    @property
    def node_feature_dim(self) -> int:
        return self.cfg.node_feature_dim

    def reset(self, *, rule_baseline_observation: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.t = 0
        self._trace_snapshot_cache_step = None
        self._trace_snapshot_cache_value = None
        self._action_mask_cache_step = None
        self._action_mask_cache_value = None
        self._mask_prediction_cache.clear()
        if self._trace_provider is not None:
            self._trace_provider.reset_counters()
        self.queue = torch.zeros(self.cfg.n_leo, device=self.device)
        self.leo_queue = self.queue
        self.geo_queue = torch.zeros(self.cfg.n_leo, device=self.device)
        self.ground_queue = torch.zeros(self.cfg.n_leo, device=self.device)
        self.ground_station_queue = torch.zeros(self.cfg.n_leo, device=self.device)
        self.virtual_delay_queue = torch.zeros(self.cfg.n_leo, device=self.device)
        self.energy = torch.full((self.cfg.n_leo,), self.cfg.leo_energy_init, device=self.device)
        self.cumulative_total_system_energy_j = torch.zeros(self.cfg.n_leo, device=self.device)
        self.last_arrivals = self._sample_arrivals()
        self.queue = self.last_arrivals.clone().clamp_min(0.0)
        if self._queue_cap_mode() == "finite_buffer":
            self.queue = torch.clamp(self.queue, max=self._queue_cap_value())
        self.last_service.zero_()
        self.episode_action_counts.zero_()
        if rule_baseline_observation:
            edge_index, edge_attr = self._empty_graph()
            return self._get_rule_baseline_obs(), edge_index, edge_attr
        return self._get_obs_graph()

    def step(
        self,
        upper_action: torch.Tensor,
        lower_action: torch.Tensor,
        *,
        minimal_info: bool = False,
    ) -> StepOutput:
        if bool(getattr(self.cfg, "formal_claim_required", False)) and not self._physical_enabled():
            raise RuntimeError("formal/paper-ready experiments require scenario.physical.enabled=true")

        upper_action = upper_action.to(self.device).long().view(self.cfg.n_leo)
        lower_action = lower_action.to(self.device).float().view(self.cfg.n_leo, self.LOWER_ACTION_DIM)
        lower_action = lower_action.clamp(0.0, 1.0)

        prev_queue = self.queue.clone()
        prev_virtual_queue = self.virtual_delay_queue.clone()

        cpu_frac = lower_action[:, 0].clamp(0.05, 1.0)
        bw_frac = lower_action[:, 1].clamp(0.05, 1.0)
        power_frac = lower_action[:, 2].clamp(0.02, 1.0)
        cpu_alloc = cpu_frac * self.cfg.leo_cpu_capacity
        bw_alloc = bw_frac * self.cfg.bandwidth_max
        tx_power = power_frac * self.cfg.tx_power_max

        edge_index, _ = self._build_graph()
        neighbor = self._select_neighbor(edge_index)
        neighbor_queue = self.queue[neighbor]

        target_cpu = torch.where(
            upper_action == self.ACTION_LOCAL,
            cpu_alloc,
            torch.where(
                upper_action == self.ACTION_NEIGHBOR,
                0.85 * self.cfg.leo_cpu_capacity * torch.ones_like(cpu_alloc),
                torch.where(
                    upper_action == self.ACTION_GEO,
                    self.cfg.geo_cpu_capacity * torch.ones_like(cpu_alloc),
                    self.cfg.ground_cpu_capacity * torch.ones_like(cpu_alloc),
                ),
            ),
        )
        link_rate, prop_delay, feasible = self._target_link_terms(upper_action, bw_alloc, tx_power)

        physical_step: PhysicalStepOutput | None = None
        if self._physical_enabled():
            p = self._physical_cfg()
            target_cpu_hz = self._physical_target_cpu_hz(upper_action)
            link_rate_bps = self._physical_link_rate_bps(upper_action)
            target_queue_cycles = self._physical_target_queue_cycles(upper_action, neighbor)
            physical_step = compute_physical_step(
                backlog_cycles=prev_queue,
                task_bits=self.last_task_bits,
                cycles_per_bit=self.last_cycles_per_bit,
                cpu_share=cpu_frac,
                bw_share=bw_frac,
                tx_power_ratio=power_frac,
                target_cpu_hz=target_cpu_hz,
                link_rate_bps=link_rate_bps,
                propagation_delay_s=prop_delay,
                target_queue_cycles=target_queue_cycles,
                feasible=feasible,
                slot_duration_s=float(p.slot_duration_s),
                max_tx_power_w=float(p.max_tx_power_w),
                kappa=float(p.kappa),
                action=upper_action,
                local_action_index=self.ACTION_LOCAL,
                compute_energy_model=str(p.compute_energy_model),
            )
            cpu_alloc = cpu_frac * float(p.leo_cpu_hz)
            bw_alloc = bw_frac * float(max(p.local_rate_bps, p.isl_base_rate_bps, p.geo_base_rate_bps, p.ground_base_rate_bps))
            tx_power = power_frac * float(p.max_tx_power_w)
            target_cpu = (cpu_frac * target_cpu_hz).clamp_min(1.0e-12)
            link_rate = (bw_frac * link_rate_bps).clamp_min(1.0e-12)
            service = physical_step.served_cycles
            queueing_delay = physical_step.queueing_delay_s
            tx_delay = physical_step.tx_delay_s
            comp_delay = physical_step.compute_delay_s
            target_queue_delay = physical_step.target_queue_delay_s
            delay = physical_step.e2e_delay_s
            energy = physical_step.total_energy_j
        else:
            service = torch.minimum(prev_queue, target_cpu * feasible.float())
            queueing_delay = prev_queue / (target_cpu + 1e-6)
            tx_delay = torch.where(upper_action == self.ACTION_LOCAL, torch.zeros_like(link_rate), service / (link_rate + 1e-6))
            comp_delay = service / (target_cpu + 1e-6)
            target_queue_delay = torch.where(
                upper_action == self.ACTION_NEIGHBOR,
                neighbor_queue / (self.cfg.leo_cpu_capacity + 1e-6),
                torch.where(
                    upper_action == self.ACTION_GEO,
                    self.queue.mean() / (self.cfg.geo_cpu_capacity + 1e-6),
                    torch.where(
                        upper_action == self.ACTION_GROUND,
                        self.queue.mean() / (self.cfg.ground_cpu_capacity + 1e-6),
                        torch.zeros_like(queueing_delay),
                    ),
                ),
            )
            delay = queueing_delay + tx_delay + comp_delay + prop_delay + target_queue_delay
            energy = 0.02 * cpu_alloc.pow(2) + tx_power * tx_delay
        infeasible_penalty = (~feasible).float()
        deadline_exceedance = F.relu(delay - self.cfg.deadline_threshold)
        deadline_violation_flag = (delay > self.cfg.deadline_threshold).float()

        local_service_cf = torch.minimum(prev_queue, cpu_alloc)
        local_delay_cf = (prev_queue / (cpu_alloc + 1e-6)) + (local_service_cf / (cpu_alloc + 1e-6)) + self.cfg.local_prop_delay
        remote_selected = (upper_action != self.ACTION_LOCAL).float()
        local_selected = (upper_action == self.ACTION_LOCAL).float()
        mobility_risk_selected = torch.zeros_like(delay)
        completion_safe_selected = torch.ones_like(delay)
        handover_required_selected = torch.zeros_like(delay)
        link_margin_selected = torch.zeros_like(delay)
        trace_for_risk = self._trace_snapshot(self.t)
        if trace_for_risk is not None:
            chosen = upper_action.view(-1, 1).clamp(0, self.N_UPPER_ACTIONS - 1)
            risk_stack = torch.stack(
                [trace_for_risk.local_mobility_risk, trace_for_risk.neighbor_mobility_risk, trace_for_risk.geo_mobility_risk, trace_for_risk.ground_mobility_risk],
                dim=-1,
            )
            completion_stack = trace_for_risk.abstract_action_mask_completion_safe.float()
            handover_stack = torch.stack(
                [trace_for_risk.local_handover_required, trace_for_risk.neighbor_handover_required, trace_for_risk.geo_handover_required, trace_for_risk.ground_handover_required],
                dim=-1,
            )
            margin_stack = torch.stack(
                [trace_for_risk.local_link_margin_to_completion, trace_for_risk.neighbor_link_margin_to_completion, trace_for_risk.geo_link_margin_to_completion, trace_for_risk.ground_link_margin_to_completion],
                dim=-1,
            )
            provided = trace_for_risk.provided.float()
            mobility_risk_selected = torch.where(provided > 0.5, risk_stack.gather(1, chosen).squeeze(-1), mobility_risk_selected)
            completion_safe_selected = torch.where(provided > 0.5, completion_stack.gather(1, chosen).squeeze(-1), completion_safe_selected)
            handover_required_selected = torch.where(provided > 0.5, handover_stack.gather(1, chosen).squeeze(-1), handover_required_selected)
            link_margin_selected = torch.where(provided > 0.5, margin_stack.gather(1, chosen).squeeze(-1), link_margin_selected)
        completion_failure = remote_selected * (1.0 - completion_safe_selected.clamp(0.0, 1.0))
        negative_margin_penalty = remote_selected * F.relu(-link_margin_selected) / max(float(getattr(self.cfg, "deadline_threshold", 1.0)), 1.0e-6)
        mobility_failure_risk = remote_selected * (mobility_risk_selected.clamp(0.0, 1.0) + handover_required_selected.clamp(0.0, 1.0)) + completion_failure + negative_margin_penalty
        failure_risk_cost = (self.weights.failure_penalty_weight * mobility_failure_risk) if self.weights.include_failure_risk else torch.zeros_like(delay)
        offload_gain = F.relu(local_delay_cf - delay) * remote_selected * feasible.float()
        if self._physical_enabled():
            local_queue_delay_for_pressure = prev_queue / (float(self._physical_cfg().leo_cpu_hz) + 1.0e-12)
        else:
            local_queue_delay_for_pressure = prev_queue / (self.cfg.leo_cpu_capacity + 1e-6)
        local_queue_pressure = F.relu(local_queue_delay_for_pressure - self.cfg.deadline_threshold) * local_selected
        remote_feasible_bonus = remote_selected * feasible.float()
        selected_when_visible_bonus = feasible.float()
        balance_bonus = torch.zeros_like(delay)
        if self.weights.action_balance_bonus > 0.0:
            counts = self.episode_action_counts.gather(1, upper_action.view(-1, 1)).squeeze(-1)
            balance_bonus = 1.0 / torch.sqrt(1.0 + counts)

        arrivals = self._sample_arrivals()
        queue_cap = self._queue_cap_value()
        queue_cap_mode = self._queue_cap_mode()
        if physical_step is not None:
            # In the physical simulator, a feasible remote offload removes the
            # admitted workload from the source LEO queue and transfers any
            # unserved remote residual to the corresponding remote queue below.
            # This preserves the source/remote queue distinction without
            # changing the legacy normalized debug path.
            remote_selected_f = (upper_action != self.ACTION_LOCAL).float()
            failed_remote_f = remote_selected_f * (1.0 - feasible.float())
            local_residual_cycles = (prev_queue - service).clamp_min(0.0) * local_selected
            failed_remote_residual_cycles = prev_queue * failed_remote_f
            source_residual_cycles = local_residual_cycles + failed_remote_residual_cycles
            unclamped_queue = torch.clamp(source_residual_cycles + arrivals, min=0.0)
        else:
            unclamped_queue = torch.clamp(prev_queue - service + arrivals, min=0.0)
        overflow_amount = F.relu(unclamped_queue - queue_cap)
        overflow_flag = (overflow_amount > 0.0).float()
        new_queue = unclamped_queue
        if queue_cap_mode == "finite_buffer":
            new_queue = torch.clamp(new_queue, max=queue_cap)
        self.virtual_delay_queue = F.relu(prev_virtual_queue + delay - self.cfg.deadline_threshold)
        self.queue = new_queue
        self.leo_queue = self.queue
        if physical_step is not None:
            geo_selected_f = (upper_action == self.ACTION_GEO).float()
            ground_selected_f = (upper_action == self.ACTION_GROUND).float()
            remote_residual_cycles = (prev_queue - service).clamp_min(0.0) * feasible.float()
            self.geo_queue = (self.geo_queue + remote_residual_cycles * geo_selected_f).clamp_min(0.0)
            self.ground_queue = (self.ground_queue + remote_residual_cycles * ground_selected_f).clamp_min(0.0)
            self.ground_station_queue = self.ground_queue.clone()
            leo_source_energy_j = physical_step.source_tx_energy_j + physical_step.source_local_compute_energy_j
            self.energy = torch.clamp(self.energy - leo_source_energy_j, min=0.0)
            self.cumulative_total_system_energy_j = self.cumulative_total_system_energy_j + physical_step.total_energy_j.detach()
        else:
            self.energy = torch.clamp(self.energy - energy, min=0.0)
        self.last_arrivals = arrivals
        self.last_service = service
        self.episode_action_counts.scatter_add_(
            1,
            upper_action.view(-1, 1),
            torch.ones((self.cfg.n_leo, 1), dtype=self.episode_action_counts.dtype, device=self.device),
        )
        self.t += 1

        lyapunov_prev = 0.5 * (prev_queue.pow(2) + prev_virtual_queue.pow(2))
        lyapunov_next = 0.5 * (self.queue.pow(2) + self.virtual_delay_queue.pow(2))
        drift = lyapunov_next - lyapunov_prev
        load_balance = self.queue.std(unbiased=False).expand_as(delay)
        selected_local = (upper_action == self.ACTION_LOCAL).float()
        selected_neighbor = (upper_action == self.ACTION_NEIGHBOR).float()
        selected_geo = (upper_action == self.ACTION_GEO).float()
        selected_ground = (upper_action == self.ACTION_GROUND).float()
        selected_ref = torch.ones_like(delay)
        if self.weights.per_tier_cost_normalization:
            tier_ref_local = self.cfg.local_prop_delay + (1.0 / max(self.cfg.leo_cpu_capacity, 1.0e-6))
            tier_ref_neighbor = self.cfg.isl_prop_delay + (1.0 / max(self.cfg.leo_cpu_capacity, 1.0e-6))
            tier_ref_geo = self.cfg.geo_prop_delay + (1.0 / max(self.cfg.geo_cpu_capacity, 1.0e-6))
            tier_ref_ground = self.cfg.ground_prop_delay + (1.0 / max(self.cfg.ground_cpu_capacity, 1.0e-6))
            tier_refs = torch.tensor(
                [tier_ref_local, tier_ref_neighbor, tier_ref_geo, tier_ref_ground],
                dtype=delay.dtype,
                device=self.device,
            ).clamp_min(1.0e-6)
            selected_ref = tier_refs.gather(0, upper_action.clamp(0, self.N_UPPER_ACTIONS - 1))

        delay_scale = max(self.cfg.deadline_threshold, 1.0e-6)
        queue_scale = max(self.cfg.max_queue, 1.0)
        energy_scale = max(self.cfg.leo_energy_init, 1.0)
        if self._physical_enabled():
            p = self._physical_cfg()
            queue_scale = max(float(p.queue_cap_cycles), 1.0)
            energy_scale = max(
                float(p.max_tx_power_w) * float(p.slot_duration_s)
                + float(p.kappa) * float(p.queue_cap_cycles) * float(p.leo_cpu_hz) ** 2,
                1.0e-12,
            )

        legacy_delay_term = delay
        legacy_energy_term = energy
        legacy_queue_term = self.queue
        legacy_violation_term = deadline_exceedance
        legacy_load_balance_term = load_balance
        normalize_legacy_terms = self.weights.cost_normalization_enabled or self._physical_enabled()
        if normalize_legacy_terms:
            legacy_delay_term = delay / delay_scale
            legacy_energy_term = energy / energy_scale
            legacy_queue_term = self.queue / queue_scale
            legacy_violation_term = deadline_exceedance / delay_scale
            legacy_load_balance_term = load_balance / queue_scale
            if self.weights.per_tier_cost_normalization:
                legacy_delay_term = legacy_delay_term / selected_ref
                legacy_queue_term = legacy_queue_term / selected_ref

        dynamic_local_penalty = self.weights.local_queue_penalty * selected_local * queueing_delay
        dynamic_neighbor_penalty = self.weights.neighbor_link_penalty * selected_neighbor / (link_rate + 1.0e-6)
        dynamic_geo_penalty = self.weights.geo_delay_penalty * selected_geo * prop_delay
        dynamic_ground_penalty = self.weights.ground_congestion_penalty * selected_ground * target_queue_delay
        fixed_local_penalty = self.weights.local_penalty * selected_local
        fixed_neighbor_penalty = self.weights.neighbor_penalty * selected_neighbor
        fixed_geo_penalty = self.weights.geo_penalty * selected_geo
        fixed_ground_penalty = self.weights.ground_penalty * selected_ground
        local_penalty_term = dynamic_local_penalty + fixed_local_penalty
        neighbor_penalty_term = dynamic_neighbor_penalty + fixed_neighbor_penalty
        geo_penalty_term = dynamic_geo_penalty + fixed_geo_penalty
        ground_penalty_term = dynamic_ground_penalty + fixed_ground_penalty
        total_penalty = local_penalty_term + neighbor_penalty_term + geo_penalty_term + ground_penalty_term
        remote_bonus_coef = self.weights.remote_bonus if abs(self.weights.remote_bonus) > 0.0 else self.weights.remote_feasible_bonus
        remote_bonus_term = remote_bonus_coef * remote_feasible_bonus
        action_balance_bonus_term = self.weights.action_balance_bonus * balance_bonus
        selected_visible_bonus_term = self.weights.selected_when_visible_bonus * selected_when_visible_bonus

        reward_mode = str(getattr(self.weights, "mode", "legacy_remote_biased") or "legacy_remote_biased").strip().lower()
        use_oracle_mode = reward_mode == "oracle_aligned_cost"
        raw_cost = (
            self.weights.delay_weight * prop_delay
            + self.weights.queue_weight * (queueing_delay + target_queue_delay)
            + self.weights.transmission_weight * tx_delay
            + self.weights.compute_weight * comp_delay
            + self.weights.feasibility_weight * infeasible_penalty
        )
        if self.weights.include_energy:
            raw_cost = raw_cost + self.weights.energy * energy

        trace = self._trace_snapshot(self.t - 1)
        chosen_upper = upper_action.view(-1, 1)
        trace_selected = trace is not None and self.weights.use_oracle_cost_components
        if trace_selected and trace is not None:
            provided = trace.provided.float()
            prop_stack = torch.stack(
                [trace.local_prop_delay, trace.neighbor_prop_delay, trace.geo_prop_delay, trace.ground_prop_delay], dim=-1
            )
            tx_stack = torch.stack(
                [trace.local_tx_delay, trace.neighbor_tx_delay, trace.geo_tx_delay, trace.ground_tx_delay], dim=-1
            )
            compute_stack = torch.stack(
                [trace.local_compute_delay, trace.neighbor_compute_delay, trace.geo_compute_delay, trace.ground_compute_delay], dim=-1
            )
            queue_delay_stack = torch.stack(
                [trace.local_queue_delay, trace.neighbor_queue_delay, trace.geo_queue_delay, trace.ground_queue_delay], dim=-1
            )
            queue_stack = torch.stack([trace.local_queue, trace.neighbor_queue, trace.geo_queue, trace.ground_queue], dim=-1)
            prop_component = torch.where(
                provided > 0.5, prop_stack.gather(1, chosen_upper).squeeze(-1), prop_delay
            )
            tx_component = torch.where(
                provided > 0.5, tx_stack.gather(1, chosen_upper).squeeze(-1), tx_delay
            )
            compute_component = torch.where(
                provided > 0.5, compute_stack.gather(1, chosen_upper).squeeze(-1), comp_delay
            )
            queue_component = torch.where(
                provided > 0.5, queue_delay_stack.gather(1, chosen_upper).squeeze(-1), (queueing_delay + target_queue_delay)
            )
            queue_length_component = torch.where(
                provided > 0.5, queue_stack.gather(1, chosen_upper).squeeze(-1), prev_queue
            )
            feasible_actions = self._upper_action_mask_at_step(self.t - 1).float()
            total_delay_stack = prop_stack + tx_stack + compute_stack + queue_delay_stack
            prop_ref = (prop_stack * feasible_actions).sum(dim=-1) / feasible_actions.sum(dim=-1).clamp_min(1.0)
            tx_ref = (tx_stack * feasible_actions).sum(dim=-1) / feasible_actions.sum(dim=-1).clamp_min(1.0)
            compute_ref = (compute_stack * feasible_actions).sum(dim=-1) / feasible_actions.sum(dim=-1).clamp_min(1.0)
            queue_ref = (queue_delay_stack * feasible_actions).sum(dim=-1) / feasible_actions.sum(dim=-1).clamp_min(1.0)
            queue_len_ref = (queue_stack * feasible_actions).sum(dim=-1) / feasible_actions.sum(dim=-1).clamp_min(1.0)
            total_delay_ref = (total_delay_stack * feasible_actions).sum(dim=-1) / feasible_actions.sum(dim=-1).clamp_min(1.0)
        else:
            prop_component = prop_delay
            tx_component = tx_delay
            compute_component = comp_delay
            queue_component = queueing_delay + target_queue_delay
            queue_length_component = prev_queue
            prop_ref = torch.full_like(prop_component, delay_scale)
            tx_ref = torch.full_like(tx_component, delay_scale)
            compute_ref = torch.full_like(compute_component, delay_scale)
            queue_ref = torch.full_like(queue_component, delay_scale)
            queue_len_ref = torch.full_like(queue_length_component, queue_scale)
            total_delay_ref = torch.full_like(queue_component, delay_scale)

        total_delay_component = prop_component + tx_component + compute_component + queue_component

        if self.weights.cost_normalization_enabled:
            if self.weights.per_tier_cost_normalization:
                norm_delay_component = prop_component / prop_ref.clamp_min(1.0e-6)
                norm_tx_component = tx_component / tx_ref.clamp_min(1.0e-6)
                norm_compute_component = compute_component / compute_ref.clamp_min(1.0e-6)
                norm_queue_component = queue_component / queue_ref.clamp_min(1.0e-6)
                norm_queue_length_component = queue_length_component / queue_len_ref.clamp_min(1.0e-6)
                norm_total_delay_component = total_delay_component / total_delay_ref.clamp_min(1.0e-6)
            else:
                norm_delay_component = prop_component / delay_scale
                norm_tx_component = tx_component / delay_scale
                norm_compute_component = compute_component / delay_scale
                norm_queue_component = queue_component / delay_scale
                norm_queue_length_component = queue_length_component / queue_scale
                norm_total_delay_component = total_delay_component / delay_scale
            norm_energy_component = energy / energy_scale
        else:
            norm_delay_component = prop_component
            norm_tx_component = tx_component
            norm_compute_component = compute_component
            norm_queue_component = queue_component
            norm_queue_length_component = queue_length_component
            norm_total_delay_component = total_delay_component
            norm_energy_component = energy
        queue_share = queue_component / total_delay_component.clamp_min(1.0e-6)
        tx_share = tx_component / total_delay_component.clamp_min(1.0e-6)
        compute_share = compute_component / total_delay_component.clamp_min(1.0e-6)
        delay_cost = self.weights.delay_weight * norm_total_delay_component
        queue_cost = self.weights.queue_weight * (0.75 * queue_share + 0.25 * norm_queue_length_component)
        transmission_cost = self.weights.transmission_weight * tx_share
        compute_cost = self.weights.compute_weight * compute_share
        energy_cost = self.weights.energy * norm_energy_component if self.weights.include_energy else torch.zeros_like(delay_cost)
        feasibility_penalty = self.weights.feasibility_weight * infeasible_penalty
        normalized_cost = delay_cost + queue_cost + transmission_cost + compute_cost + feasibility_penalty + energy_cost + failure_risk_cost

        neutral_cpu = torch.ones_like(cpu_frac)
        neutral_bw = torch.ones_like(bw_frac)
        neutral_power = torch.ones_like(power_frac)
        neutral_cpu_alloc = neutral_cpu * self.cfg.leo_cpu_capacity
        neutral_bw_alloc = neutral_bw * self.cfg.bandwidth_max
        neutral_tx_power = neutral_power * self.cfg.tx_power_max
        neutral_target_cpu = torch.where(
            upper_action == self.ACTION_LOCAL,
            neutral_cpu_alloc,
            torch.where(
                upper_action == self.ACTION_NEIGHBOR,
                0.85 * self.cfg.leo_cpu_capacity * torch.ones_like(neutral_cpu_alloc),
                torch.where(
                    upper_action == self.ACTION_GEO,
                    self.cfg.geo_cpu_capacity * torch.ones_like(neutral_cpu_alloc),
                    self.cfg.ground_cpu_capacity * torch.ones_like(neutral_cpu_alloc),
                ),
            ),
        )
        neutral_link_rate, neutral_prop_delay, neutral_feasible = self._target_link_terms(upper_action, neutral_bw_alloc, neutral_tx_power)
        neutral_service = torch.minimum(prev_queue, neutral_target_cpu * neutral_feasible.float())
        neutral_queueing = prev_queue / (neutral_target_cpu + 1.0e-6)
        neutral_tx = torch.where(upper_action == self.ACTION_LOCAL, torch.zeros_like(neutral_link_rate), neutral_service / (neutral_link_rate + 1.0e-6))
        neutral_compute = neutral_service / (neutral_target_cpu + 1.0e-6)
        neutral_target_queue = torch.where(
            upper_action == self.ACTION_NEIGHBOR,
            neighbor_queue / (self.cfg.leo_cpu_capacity + 1.0e-6),
            torch.where(
                upper_action == self.ACTION_GEO,
                self.queue.mean() / (self.cfg.geo_cpu_capacity + 1.0e-6),
                torch.where(
                    upper_action == self.ACTION_GROUND,
                    self.queue.mean() / (self.cfg.ground_cpu_capacity + 1.0e-6),
                    torch.zeros_like(neutral_queueing),
                ),
            ),
        )
        neutral_total_delay = neutral_prop_delay + neutral_tx + neutral_compute + neutral_queueing + neutral_target_queue
        lower_effect = delay - neutral_total_delay
        if self.weights.cost_normalization_enabled:
            lower_effect = lower_effect / delay_scale

        legacy_delay_cost = self.weights.delay * legacy_delay_term
        legacy_queue_cost = self.weights.queue * legacy_queue_term
        legacy_energy_cost = self.weights.energy * legacy_energy_term
        legacy_violation_cost = self.weights.violation * legacy_violation_term
        legacy_feasibility_cost = self.weights.infeasible * infeasible_penalty
        legacy_balance_cost = self.weights.load_balance * legacy_load_balance_term
        legacy_pressure_cost = self.weights.local_queue_pressure * local_queue_pressure
        legacy_shaping_gain = self.weights.offload_gain * offload_gain
        immediate_cost = (
            legacy_delay_cost
            + legacy_queue_cost
            + legacy_energy_cost
            + legacy_violation_cost
            + legacy_feasibility_cost
            + failure_risk_cost
            + legacy_balance_cost
            + legacy_pressure_cost
            + total_penalty
            - legacy_shaping_gain
            - remote_bonus_term
            - selected_visible_bonus_term
            - action_balance_bonus_term
        )
        lyapunov_cost = drift + self.weights.lyapunov_v * immediate_cost
        realized_upper_cost = lyapunov_cost if self.cfg.enable_lyapunov_reward else immediate_cost
        proxy_cost = (
            self.weights.delay * (prev_queue / (target_cpu + 1e-6) + prop_delay) / max(self.cfg.deadline_threshold, 1.0e-6)
            + self.weights.queue * (prev_queue / max(self.cfg.max_queue, 1.0))
            + self.weights.infeasible * infeasible_penalty
            + failure_risk_cost
            + self.weights.local_queue_pressure * local_queue_pressure
            + total_penalty
            - self.weights.offload_gain * offload_gain
            - remote_bonus_term
            - selected_visible_bonus_term
            - action_balance_bonus_term
        )

        if use_oracle_mode:
            oracle_system_cost = normalized_cost + total_penalty - remote_bonus_term - selected_visible_bonus_term - action_balance_bonus_term
            if self.weights.use_lower_effect_in_upper_reward:
                oracle_system_cost = oracle_system_cost + lower_effect
            realized_upper_cost = oracle_system_cost
            upper_reward = -oracle_system_cost.detach()
            system_cost = oracle_system_cost
            raw_cost = (
                self.weights.delay_weight * total_delay_component
                + self.weights.queue_weight * (0.75 * queue_share + 0.25 * queue_length_component / queue_scale)
                + self.weights.transmission_weight * tx_share
                + self.weights.compute_weight * compute_share
                + self.weights.feasibility_weight * infeasible_penalty
                + (self.weights.energy * energy if self.weights.include_energy else torch.zeros_like(delay))
                + failure_risk_cost
            )
        else:
            if self.cfg.enable_cross_layer_feedback:
                upper_reward = -realized_upper_cost.detach()
            else:
                upper_reward = -proxy_cost.detach()
            system_cost = immediate_cost
            delay_cost = legacy_delay_cost
            queue_cost = legacy_queue_cost
            transmission_cost = self.weights.delay * tx_delay
            compute_cost = self.weights.delay * comp_delay
            energy_cost = legacy_energy_cost
            feasibility_penalty = legacy_feasibility_cost
            normalized_cost = immediate_cost
            raw_cost = (
                self.weights.delay * delay
                + self.weights.energy * energy
                + self.weights.queue * self.queue
                + self.weights.violation * deadline_exceedance
                + self.weights.infeasible * infeasible_penalty
                + failure_risk_cost
            )

        trace_delay_anomaly_mask = torch.zeros_like(delay, dtype=torch.bool)
        if trace is not None and self._trace_delay_interpretation.treat_anomaly_as_legacy_score:
            trace_delay_by_tier = torch.stack(
                [trace.local_delay, trace.neighbor_delay, trace.geo_delay, trace.ground_delay],
                dim=-1,
            )
            selected_trace_delay = trace_delay_by_tier.gather(1, chosen_upper).squeeze(-1)
            scaled_trace_delay = selected_trace_delay * float(self._unit_scale.delay_s_per_unit)
            trace_delay_anomaly_mask = trace.provided & (scaled_trace_delay > float(self._trace_delay_interpretation.anomaly_threshold_s))

        metric_bundle = build_step_metric_bundle(
            delay_units=delay.detach(),
            energy_units=energy.detach(),
            queue_units=self.queue.detach(),
            normalized_cost=normalized_cost.detach(),
            reward=upper_reward.detach(),
            trace_delay_anomaly_mask=trace_delay_anomaly_mask.detach(),
            units=self._unit_scale,
            queue_unit="cycles" if self._physical_enabled() else "tasks",
        )

        lower_reward = -(
            self.weights.delay * delay
            + self.weights.energy * energy
            + 0.5 * self.weights.violation * deadline_exceedance
            + self.weights.infeasible * infeasible_penalty
            + failure_risk_cost
            + 0.02 * cpu_frac.pow(2)
            + 0.02 * bw_frac.pow(2)
            + 0.02 * power_frac.pow(2)
        ).detach()

        if minimal_info:
            # Rule-baseline evaluation needs only a compact metrics subset.
            # Avoid constructing and cloning the full training diagnostics map
            # for millions of non-learning environment steps.
            compact_mask_details = self._upper_action_mask_details_at_step(self.t - 1)
            obs = self._get_rule_baseline_obs()
            edge_index_next, edge_attr_next = self._empty_graph()
            done = self.t >= self.cfg.episode_len
            info = {
                "delay": delay.detach(),
                "energy": energy.detach(),
                "queue": self.queue.detach(),
                "physical_delay_s": metric_bundle.physical_delay_s.detach(),
                "physical_energy_j": metric_bundle.physical_energy_j.detach(),
                "physical_queue_cycles": metric_bundle.physical_queue_cycles.detach(),
                "physical_queue_length_tasks": metric_bundle.physical_queue_length_tasks.detach(),
                "normalized_system_cost": metric_bundle.normalized_system_cost.detach(),
                "service": service.detach(),
                "arrivals": arrivals.detach(),
                "deadline_exceedance": deadline_exceedance.detach(),
                "deadline_violation_flag": deadline_violation_flag.detach(),
                "deadline_violation": deadline_exceedance.detach(),
                "feasible": feasible.float().detach(),
                "mobility_failure_risk": mobility_failure_risk.detach(),
                "lyapunov_drift": drift.detach(),
                "lyapunov_reward_shaping_cost": lyapunov_cost.detach(),
                "queue_regularizer_metric": (arrivals - service).detach(),
                "queue_cap_value": torch.full_like(delay, float(queue_cap)),
                "queue_cap_mode_is_unbounded_eval": torch.full_like(delay, 1.0 if queue_cap_mode == "unbounded_eval" else 0.0),
                "queue_stability_claim_allowed": torch.zeros_like(delay),
                "lyapunov_semantics_code": torch.ones_like(delay),
                "finite_buffer_overflow_count": overflow_flag.detach() if queue_cap_mode == "finite_buffer" else torch.zeros_like(delay),
                "overflow_risk": overflow_flag.detach(),
                "virtual_delay_queue": self.virtual_delay_queue.detach(),
                "upper_action": upper_action.detach(),
                "system_cost": system_cost.detach(),
                "mask_source_code": torch.full_like(delay, float({"measured": 0, "predicted": 1, "oracle_trace": 2}.get(compact_mask_details.mask_source, 1))),
                "uses_oracle_trace_mask": torch.full_like(delay, 1.0 if compact_mask_details.uses_oracle_trace_mask else 0.0),
                "mask_deployable": torch.full_like(delay, 1.0 if compact_mask_details.deployable else 0.0),
                "mask_false_positive_rate_observed": compact_mask_details.mask_false_positive_rate_observed.detach(),
                "mask_false_negative_rate_observed": compact_mask_details.mask_false_negative_rate_observed.detach(),
                "mask_predictor_fallback": compact_mask_details.predictor_fallback.detach(),
                "mask_fallback_due_empty": compact_mask_details.fallback_due_empty_mask_count.detach(),
                "mask_staleness_slots": torch.full_like(delay, float(compact_mask_details.mask_staleness_slots)),
                "link_lifetime_noise_std_s": torch.full_like(delay, float(compact_mask_details.link_lifetime_noise_std_s)),
                "completion_time_noise_std_s": torch.full_like(delay, float(compact_mask_details.completion_time_noise_std_s)),
                "configured_mask_false_positive_rate": torch.full_like(delay, float(compact_mask_details.mask_false_positive_rate)),
                "configured_mask_false_negative_rate": torch.full_like(delay, float(compact_mask_details.mask_false_negative_rate)),
            }
            info["simulator_semantics"] = ("physical_dimensioned" if self._physical_enabled() else "legacy_normalized_debug")  # type: ignore[assignment]
            info["mask_source"] = compact_mask_details.mask_source  # type: ignore[assignment]
            info["lyapunov_semantics"] = "reward_shaping_no_stability_theorem"  # type: ignore[assignment]
            info["lyapunov_claim_mode"] = str(getattr(self.cfg, "lyapunov_claim_mode", "inspired_reward"))  # type: ignore[assignment]
            info["queue_cap_mode"] = queue_cap_mode  # type: ignore[assignment]
            if physical_step is not None:
                info.update(
                    {
                        "served_bits": physical_step.served_bits.detach(),
                        "compute_energy_j": physical_step.compute_energy_j.detach(),
                        "tx_energy_j": physical_step.tx_energy_j.detach(),
                        "queue_unit_is_cycles": torch.ones((), device=self.device),
                    }
                )
            diagnostic_oracle_allowed = bool(getattr(self.cfg, "diagnostic_oracle_allowed", False))
            info["diagnostic_oracle_allowed"] = diagnostic_oracle_allowed  # type: ignore[assignment]
            info["formal_claim_allowed"] = bool(
                self._physical_enabled() and not compact_mask_details.uses_oracle_trace_mask and not diagnostic_oracle_allowed
            )  # type: ignore[assignment]
            info["mask_predictor_units"] = self._mask_predictor_units_for_step(self.t - 1)  # type: ignore[assignment]
            return StepOutput(obs, edge_index_next, edge_attr_next, upper_reward, lower_reward, done, info)

        # The mask is computed after state advance for the next observation;
        # old_mask records what the policy actually faced for this decision.
        old_mask_details = self._upper_action_mask_details_at_step(self.t - 1)
        old_mask = old_mask_details.final_mask
        chosen_old_valid = old_mask.gather(1, upper_action.clamp(0, self.N_UPPER_ACTIONS - 1).view(-1, 1)).squeeze(-1)
        invalid_action_ratio = (~chosen_old_valid).float()
        obs, edge_index_next, edge_attr_next = self._get_obs_graph()
        done = self.t >= self.cfg.episode_len
        neighbor_idx_for_metrics = self._select_neighbor(self._build_graph()[0])
        neighbor_queue_metric = self.queue[neighbor_idx_for_metrics]
        geo_queue_metric = self.geo_queue if self._physical_enabled() else self.queue.mean().expand_as(self.queue)
        ground_queue_metric = self.ground_queue if self._physical_enabled() else self.queue.mean().expand_as(self.queue)
        ground_station_queue_metric = self.ground_station_queue if self._physical_enabled() else ground_queue_metric
        self.last_metrics = {
            "delay": delay.detach(),
            "energy": energy.detach(),
            "queue": self.queue.detach(),
            # Explicit multi-tier queue diagnostics for paper-scale analysis.
            "leo_queue": self.queue.detach(),
            "neighbor_queue_tier": neighbor_queue_metric.detach(),
            "geo_queue": geo_queue_metric.detach(),
            "ground_queue": ground_queue_metric.detach(),
            "ground_station_queue": ground_station_queue_metric.detach(),
            "max_remote_queue": torch.stack([neighbor_queue_metric, geo_queue_metric, ground_queue_metric], dim=-1).max(dim=-1).values.detach(),
            "queue_stability_metric": (arrivals - service).detach(),
            "queue_regularizer_metric": (arrivals - service).detach(),
            "queue_cap_value": torch.full_like(delay, float(queue_cap)),
            "queue_cap_mode_is_unbounded_eval": torch.full_like(delay, 1.0 if queue_cap_mode == "unbounded_eval" else 0.0),
            "queue_stability_claim_allowed": torch.zeros_like(delay),
            "lyapunov_semantics_code": torch.ones_like(delay),
            "finite_buffer_overflow_count": overflow_flag.detach() if queue_cap_mode == "finite_buffer" else torch.zeros_like(delay),
            "overflow_risk": overflow_flag.detach(),
            "physical_delay_s": metric_bundle.physical_delay_s.detach(),
            "physical_energy_j": metric_bundle.physical_energy_j.detach(),
            "physical_queue_cycles": metric_bundle.physical_queue_cycles.detach(),
            "physical_queue_length_tasks": metric_bundle.physical_queue_length_tasks.detach(),
            "service": service.detach(),
            "arrivals": arrivals.detach(),
            "deadline_exceedance": deadline_exceedance.detach(),
            "deadline_violation_flag": deadline_violation_flag.detach(),
            "deadline_violation": deadline_exceedance.detach(),
            "feasible": feasible.float().detach(),
            "completion_safe_selected": completion_safe_selected.detach(),
            "mobility_risk_selected": mobility_risk_selected.detach(),
            "handover_required_selected": handover_required_selected.detach(),
            "link_margin_selected": link_margin_selected.detach(),
            "mobility_failure_risk": mobility_failure_risk.detach(),
            "failure_risk_cost": failure_risk_cost.detach(),
            "lyapunov_drift": drift.detach(),
            "lyapunov_reward_shaping_cost": lyapunov_cost.detach(),
            "virtual_delay_queue": self.virtual_delay_queue.detach(),
            "upper_action": upper_action.detach(),
            "offload_gain": offload_gain.detach(),
            "local_queue_pressure": local_queue_pressure.detach(),
            "remote_feasible_bonus": remote_feasible_bonus.detach(),
            "selected_when_visible_bonus": selected_when_visible_bonus.detach(),
            "action_balance_bonus": balance_bonus.detach(),
            "upper_mask_local": old_mask[:, self.ACTION_LOCAL].float().detach(),
            "upper_mask_neighbor": old_mask[:, self.ACTION_NEIGHBOR].float().detach(),
            "upper_mask_geo": old_mask[:, self.ACTION_GEO].float().detach(),
            "upper_mask_ground": old_mask[:, self.ACTION_GROUND].float().detach(),
            "upper_mask_size": old_mask.float().sum(dim=-1).detach(),
            "upper_mask_remote": old_mask[:, 1:].any(dim=-1).float().detach(),
            "action_mask_raw_count": old_mask_details.raw_count.detach(),
            "action_mask_after_visibility_count": old_mask_details.visibility_count.detach(),
            "action_mask_after_completion_safe_count": old_mask_details.completion_safe_count.detach(),
            "action_mask_after_mobility_risk_count": old_mask_details.mobility_risk_count.detach(),
            "action_mask_final_valid_count": old_mask_details.final_count.detach(),
            "invalid_action_ratio": invalid_action_ratio.detach(),
            "masked_action_ratio": old_mask_details.masked_action_ratio.detach(),
            "visibility_mask_ratio": old_mask_details.visibility_mask_ratio.detach(),
            "completion_mask_ratio": old_mask_details.completion_mask_ratio.detach(),
            "mobility_mask_ratio": old_mask_details.mobility_mask_ratio.detach(),
            "mask_source_code": torch.full_like(delay, float({"measured": 0, "predicted": 1, "oracle_trace": 2}.get(old_mask_details.mask_source, 1))),
            "uses_oracle_trace_mask": torch.full_like(delay, 1.0 if old_mask_details.uses_oracle_trace_mask else 0.0),
            "mask_deployable": torch.full_like(delay, 1.0 if old_mask_details.deployable else 0.0),
            "mask_false_positive_rate_observed": old_mask_details.mask_false_positive_rate_observed.detach(),
            "mask_false_negative_rate_observed": old_mask_details.mask_false_negative_rate_observed.detach(),
            "mask_predictor_fallback": old_mask_details.predictor_fallback.detach(),
            "mask_fallback_due_empty": old_mask_details.fallback_due_empty_mask_count.detach(),
            "mask_staleness_slots": torch.full_like(delay, float(old_mask_details.mask_staleness_slots)),
            "link_lifetime_noise_std_s": torch.full_like(delay, float(old_mask_details.link_lifetime_noise_std_s)),
            "completion_time_noise_std_s": torch.full_like(delay, float(old_mask_details.completion_time_noise_std_s)),
            "configured_mask_false_positive_rate": torch.full_like(delay, float(old_mask_details.mask_false_positive_rate)),
            "configured_mask_false_negative_rate": torch.full_like(delay, float(old_mask_details.mask_false_negative_rate)),
            "selected_neighbor_when_visible": ((upper_action == self.ACTION_NEIGHBOR) & old_mask[:, self.ACTION_NEIGHBOR]).float().detach(),
            "selected_geo_when_visible": ((upper_action == self.ACTION_GEO) & old_mask[:, self.ACTION_GEO]).float().detach(),
            "selected_ground_when_visible": ((upper_action == self.ACTION_GROUND) & old_mask[:, self.ACTION_GROUND]).float().detach(),
            "selected_remote_when_visible": ((upper_action != self.ACTION_LOCAL) & old_mask.gather(1, upper_action.clamp(0, 3).view(-1, 1)).squeeze(-1)).float().detach(),
            "selected_local": (upper_action == self.ACTION_LOCAL).float().detach(),
            "selected_neighbor": (upper_action == self.ACTION_NEIGHBOR).float().detach(),
            "selected_geo": (upper_action == self.ACTION_GEO).float().detach(),
            "selected_ground": (upper_action == self.ACTION_GROUND).float().detach(),
            "reward_total": upper_reward.detach(),
            "system_cost": system_cost.detach(),
            "delay_cost": delay_cost.detach(),
            "queue_cost": queue_cost.detach(),
            "transmission_cost": transmission_cost.detach(),
            "compute_cost": compute_cost.detach(),
            "energy_cost": energy_cost.detach(),
            "feasibility_penalty": feasibility_penalty.detach(),
            "remote_bonus": remote_bonus_term.detach(),
            "local_penalty": local_penalty_term.detach(),
            "neighbor_penalty": neighbor_penalty_term.detach(),
            "geo_penalty": geo_penalty_term.detach(),
            "ground_penalty": ground_penalty_term.detach(),
            "penalty_total": total_penalty.detach(),
            "bonus_total": (remote_bonus_term + selected_visible_bonus_term + action_balance_bonus_term).detach(),
            "action_balance_bonus_term": action_balance_bonus_term.detach(),
            "selected_when_visible_bonus_term": selected_visible_bonus_term.detach(),
            "lower_allocation_effect": lower_effect.detach(),
            "normalized_cost": normalized_cost.detach(),
            "normalized_system_cost": metric_bundle.normalized_system_cost.detach(),
            "normalized_training_cost": metric_bundle.normalized_system_cost.detach(),
            "raw_cost": raw_cost.detach(),
            "reward": metric_bundle.reward.detach(),
            "reward_mean": metric_bundle.reward.mean().detach().view(1),
            "legacy_trace_delay_score": metric_bundle.legacy_trace_delay_score.detach(),
            "trace_delay_anomaly_flag": trace_delay_anomaly_mask.float().detach(),
        }
        if physical_step is not None:
            self.last_metrics.update(
                {
                    "served_cycles": physical_step.served_cycles.detach(),
                    "served_bits": physical_step.served_bits.detach(),
                    "queueing_delay_s": physical_step.queueing_delay_s.detach(),
                    "tx_delay_s": physical_step.tx_delay_s.detach(),
                    "compute_delay_s": physical_step.compute_delay_s.detach(),
                    "target_queue_delay_s": physical_step.target_queue_delay_s.detach(),
                    "compute_energy_j": physical_step.compute_energy_j.detach(),
                    "tx_energy_j": physical_step.tx_energy_j.detach(),
                    "leo_tx_energy_j": physical_step.source_tx_energy_j.detach(),
                    "leo_local_compute_energy_j": physical_step.source_local_compute_energy_j.detach(),
                    "leo_remote_compute_energy_j": torch.zeros_like(physical_step.total_energy_j).detach(),
                    "geo_compute_energy_j": torch.where((upper_action == self.ACTION_GEO), physical_step.remote_compute_energy_j, torch.zeros_like(physical_step.total_energy_j)).detach(),
                    "ground_compute_energy_j": torch.where((upper_action == self.ACTION_GROUND), physical_step.remote_compute_energy_j, torch.zeros_like(physical_step.total_energy_j)).detach(),
                    "network_energy_j": physical_step.network_energy_j.detach(),
                    "total_system_energy_j": physical_step.total_energy_j.detach(),
                    "cumulative_total_system_energy_j": self.cumulative_total_system_energy_j.detach(),
                    "queue_unit_is_cycles": torch.ones((), device=self.device),
                }
            )
        if not bool(getattr(self.cfg, "export_physical_metrics", True)):
            self.last_metrics.pop("physical_delay_s", None)
            self.last_metrics.pop("physical_energy_j", None)
            self.last_metrics.pop("physical_queue_cycles", None)
            self.last_metrics.pop("physical_queue_length_tasks", None)
            self.last_metrics.pop("normalized_system_cost", None)
            self.last_metrics.pop("normalized_training_cost", None)
            self.last_metrics.pop("legacy_trace_delay_score", None)
            self.last_metrics.pop("trace_delay_anomaly_flag", None)
        trace_stats = self.trace_stats()
        self.last_metrics.update(
            {
                "action_mask_mode_resolved": torch.tensor(float({"none": 0, "visibility": 1, "completion_safe": 2, "mobility_risk": 3, "full": 4}.get(old_mask_details.mode, 1)), device=self.device),
                "trace_missing_count": torch.tensor(float(trace_stats["trace_missing_count"]), device=self.device),
                "trace_fallback_count": torch.tensor(float(trace_stats["trace_fallback_count"]), device=self.device),
                "trace_hit_ratio": torch.tensor(float(trace_stats["trace_hit_ratio"]), device=self.device),
            }
        )
        info = {k: v.clone() for k, v in self.last_metrics.items()}
        if bool(getattr(self.cfg, "export_physical_metrics", True)):
            info.update({k: v.clone() for k, v in step_bundle_to_info(metric_bundle).items()})
        info["system_cost"] = system_cost.detach()
        info["upper_reward"] = upper_reward.detach()
        info["upper_cost"] = realized_upper_cost.detach()
        info["mean_reward"] = upper_reward.mean().detach().view(1)
        info["lyapunov_semantics"] = "reward_shaping_no_stability_theorem"  # type: ignore[assignment]
        info["lyapunov_claim_mode"] = str(getattr(self.cfg, "lyapunov_claim_mode", "inspired_reward"))  # type: ignore[assignment]
        info["queue_cap_mode"] = queue_cap_mode  # type: ignore[assignment]
        info["mask_source"] = old_mask_details.mask_source  # type: ignore[assignment]
        info["simulator_semantics"] = ("physical_dimensioned" if self._physical_enabled() else "legacy_normalized_debug")  # type: ignore[assignment]
        diagnostic_oracle_allowed = bool(getattr(self.cfg, "diagnostic_oracle_allowed", False))
        uses_oracle_trace_mask = bool(old_mask_details.uses_oracle_trace_mask)
        info["diagnostic_oracle_allowed"] = diagnostic_oracle_allowed  # type: ignore[assignment]
        info["formal_claim_allowed"] = bool(
            self._physical_enabled() and not uses_oracle_trace_mask and not diagnostic_oracle_allowed
        )  # type: ignore[assignment]
        info["mask_predictor_units"] = self._mask_predictor_units_for_step(self.t - 1)  # type: ignore[assignment]
        return StepOutput(obs, edge_index_next, edge_attr_next, upper_reward, lower_reward, done, info)

    def _empty_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty((2, 0), dtype=torch.long, device=self.device),
            torch.empty((0, self.cfg.edge_feature_dim), dtype=torch.float32, device=self.device),
        )

    def _normalize_rule_feature(self, field: str, values: torch.Tensor, *, legacy_default: float) -> torch.Tensor:
        values = values.float().clamp_min(0.0)
        if self._obs_norm_mode == "trace_log_quantile":
            info = self._obs_norm_stats.get(field) if isinstance(self._obs_norm_stats, dict) else None
            info = info if isinstance(info, dict) else {}
            p50 = max(1.0e-6, float(info.get("p50", legacy_default)))
            p99 = max(p50 + 1.0e-6, float(info.get("p99", p50 * 10.0)))
            scale = max(1.0e-6, float(info.get("scale", p50)))
            denom = float(info.get("denom", 0.0))
            if denom <= 0.0:
                denom = math.log1p(p99 / scale)
            if denom <= 1.0e-9:
                return torch.zeros_like(values)
            return (torch.log1p(values / scale) / denom).clamp(0.0, 1.0)
        ref = float(legacy_default)
        if self._obs_norm_mode == "trace_p95":
            info = self._obs_norm_stats.get(field) if isinstance(self._obs_norm_stats, dict) else None
            if isinstance(info, dict):
                p95 = float(info.get("p95", ref))
                if p95 > 0.0:
                    ref = p95
        return (values / max(1.0e-6, ref)).clamp(0.0, 1.0)

    def _get_rule_baseline_obs(self) -> torch.Tensor:
        """Build the observable subset required by offline rule baselines.

        The training observation builder intentionally exports a rich schema and
        canonical row diagnostics.  Offline rules consume only executable masks,
        tier rates, and local queue pressure.  Construct those fields directly
        to avoid Python scalar extraction and row re-parsing on every evaluation
        step.  For legacy schemas, retain the general builder.
        """

        if self.cfg.node_feature_dim < SHARED_NODE_FEATURE_DIM:
            return self._get_obs_graph()[0]

        action_mask = self._upper_action_mask_at_step(self.t)
        one = torch.ones(self.cfg.n_leo, device=self.device)
        neighbor_rate = self._neighbor_rate().max(dim=-1).values
        geo_rate = self._geo_rate(one, one)
        ground_rate = self._ground_rate(one, one)
        local_queue = self.queue
        trace = self._trace_snapshot(self.t)
        if trace is not None:
            neighbor_rate = torch.where(trace.provided, trace.neighbor_rate, neighbor_rate)
            # Match the canonical observation builder exactly: the exported
            # trace rates are observation-level summaries, not rates after the
            # policy's resource-allocation multiplier.
            geo_rate = torch.where(trace.provided, trace.geo_rate, geo_rate)
            ground_rate = torch.where(trace.provided, trace.ground_rate, ground_rate)
            local_queue = torch.where(trace.provided & (trace.local_queue > 0.0), trace.local_queue, local_queue)

        obs = torch.zeros((self.cfg.n_leo, self.cfg.node_feature_dim), dtype=torch.float32, device=self.device)
        obs[:, :4] = action_mask.float()
        obs[:, IDX_LOCAL_RATE] = self._normalize_rule_feature(
            "local_rate", torch.full_like(local_queue, 1000.0), legacy_default=1000.0
        )
        obs[:, IDX_NEIGHBOR_RATE] = self._normalize_rule_feature("neighbor_rate", neighbor_rate, legacy_default=800.0)
        obs[:, IDX_GEO_RATE] = self._normalize_rule_feature("geo_rate", geo_rate, legacy_default=400.0)
        obs[:, IDX_GROUND_RATE] = self._normalize_rule_feature("ground_rate", ground_rate, legacy_default=400.0)
        obs[:, IDX_LOCAL_QUEUE] = self._normalize_rule_feature("local_queue", local_queue, legacy_default=80.0)
        return obs

    def baseline_contexts(self) -> List[Dict[str, object]]:
        """Return per-LEO observable-only contexts for formal offline baselines."""

        obs = self._get_rule_baseline_obs()
        final_mask = self._upper_action_mask_at_step(self.t)
        rows = self._shared_obs_rows()
        contexts: List[Dict[str, object]] = []
        for leo, row in enumerate(rows):
            mask = [bool(final_mask[leo, action].item()) for action in range(self.N_UPPER_ACTIONS)]
            candidate_info = self._baseline_candidate_info(row, mask)
            state = {
                "leo_id": int(leo),
                "step": int(self.t),
                "abstractActionMask": [1 if item else 0 for item in mask],
                "abstractActionMaskVisible": [1 if item else 0 for item in mask],
                "abstractActionMaskMobilitySafe": [
                    1 if float(row[f"{tier}_mobility_risk"]) <= 0.5 else 0
                    for tier in ("local", "neighbor", "geo", "ground")
                ],
                "abstractActionMaskCompletionSafe": [
                    1 if float(row[f"{tier}_completion_safe"]) > 0.5 else 0
                    for tier in ("local", "neighbor", "geo", "ground")
                ],
                "local_queue": float(row["local_queue"]),
                "neighbor_queue": float(row["neighbor_queue"]),
                "geo_queue": float(row["geo_queue"]),
                "ground_queue": float(row["ground_queue"]),
            }
            contexts.append(
                {
                    "obs": obs[leo].detach().clone(),
                    "state": state,
                    "mask": mask,
                    "candidate_info": candidate_info,
                }
            )
        return contexts

    def _baseline_candidate_info(self, row: Dict[str, float], mask: List[bool]) -> Dict[int, Dict[str, object]]:
        out: Dict[int, Dict[str, object]] = {}
        for action, tier in enumerate(("local", "neighbor", "geo", "ground")):
            delay = float(row[f"{tier}_delay"])
            queue = float(row[f"{tier}_queue"])
            risk = float(row[f"{tier}_mobility_risk"])
            rate = float(row[f"{tier}_rate"])
            completion_safe = bool(float(row[f"{tier}_completion_safe"]) > 0.5)
            handover_required = bool(float(row[f"{tier}_handover_required"]) > 0.5)
            estimated_energy = 0.0 if action == self.ACTION_LOCAL else max(0.0, delay / max(rate, 1.0e-6))
            out[action] = {
                "action": action,
                "tier": tier,
                "is_available": bool(mask[action]),
                "estimated_cost": float(delay + 0.5 * queue + 0.2 * risk + 0.05 * estimated_energy),
                "estimated_delay": delay,
                "estimated_queue": queue,
                "estimated_energy_j": float(estimated_energy),
                "mobility_risk": risk,
                "link_lifetime_sec": float(row[f"{tier}_link_lifetime_sec"]),
                "link_survival_margin_to_completion_sec": float(row[f"{tier}_link_survival_margin_to_completion_sec"]),
                "handover_required": handover_required,
                "completion_safe": completion_safe,
                "rate_mbps": rate,
                "selected_vm_id": -1,
                "selected_candidate_index": -1,
            }
        return out

    def _get_obs_graph(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_index, edge_attr = self._build_graph()
        rows = self._shared_obs_rows()
        batch = build_shared_observation(
            rows,
            source_index=0,
            node_feature_dim=self.cfg.node_feature_dim,
            device=self.device,
            normalization_mode=self._obs_norm_mode,
            normalization_stats=self._obs_norm_stats,
            access_mode=str(getattr(self.cfg, "observation_access_mode", "safe_observable")),
            include_cost_prior_features=bool(getattr(self.cfg, "observation_include_cost_prior_features", False)),
            include_oracle_cost=bool(getattr(self.cfg, "observation_include_oracle_cost", False)),
        )
        return batch.obs.float(), edge_index, edge_attr

    def _shared_obs_rows(self) -> List[Dict[str, float]]:
        action_mask = self._upper_action_mask_at_step(self.t)
        neighbor_rate = self._neighbor_rate().max(dim=-1).values
        neighbor = self._select_neighbor(self._build_graph()[0])
        neighbor_queue = self.queue[neighbor]
        one = torch.ones(self.cfg.n_leo, device=self.device)
        geo_rate = self._geo_rate(one, one)
        ground_rate = self._ground_rate(one, one)
        local_delay = (self.queue / (self.cfg.leo_cpu_capacity + 1.0e-6)) + self.cfg.local_prop_delay
        neighbor_delay = (
            self.queue / (neighbor_rate + 1.0e-6)
            + self.cfg.isl_prop_delay
            + neighbor_queue / (self.cfg.leo_cpu_capacity + 1.0e-6)
        )
        geo_queue = self.queue.mean().expand_as(self.queue)
        ground_queue = self.queue.mean().expand_as(self.queue)
        geo_delay = (
            self.queue / (geo_rate + 1.0e-6)
            + self.cfg.geo_prop_delay
            + geo_queue / (self.cfg.geo_cpu_capacity + 1.0e-6)
        )
        ground_delay = (
            self.queue / (ground_rate + 1.0e-6)
            + self.cfg.ground_prop_delay
            + ground_queue / (self.cfg.ground_cpu_capacity + 1.0e-6)
        )

        trace = self._trace_snapshot(self.t)
        local_completion_safe = action_mask[:, self.ACTION_LOCAL].float()
        neighbor_completion_safe = action_mask[:, self.ACTION_NEIGHBOR].float()
        geo_completion_safe = action_mask[:, self.ACTION_GEO].float()
        ground_completion_safe = action_mask[:, self.ACTION_GROUND].float()
        local_mobility_risk = torch.zeros_like(self.queue)
        neighbor_mobility_risk = torch.where(action_mask[:, self.ACTION_NEIGHBOR], torch.zeros_like(self.queue), torch.ones_like(self.queue))
        geo_mobility_risk = torch.where(action_mask[:, self.ACTION_GEO], torch.zeros_like(self.queue), torch.ones_like(self.queue))
        ground_mobility_risk = torch.where(action_mask[:, self.ACTION_GROUND], torch.zeros_like(self.queue), torch.ones_like(self.queue))
        local_link_lifetime = torch.zeros_like(self.queue)
        neighbor_link_lifetime = torch.zeros_like(self.queue)
        geo_link_lifetime = torch.zeros_like(self.queue)
        ground_link_lifetime = torch.zeros_like(self.queue)
        local_link_margin = torch.zeros_like(self.queue)
        neighbor_link_margin = torch.zeros_like(self.queue)
        geo_link_margin = torch.zeros_like(self.queue)
        ground_link_margin = torch.zeros_like(self.queue)
        local_handover_required = torch.zeros_like(self.queue)
        neighbor_handover_required = torch.zeros_like(self.queue)
        geo_handover_required = torch.zeros_like(self.queue)
        ground_handover_required = torch.zeros_like(self.queue)
        if trace is not None:
            neighbor_rate = torch.where(trace.provided, trace.neighbor_rate, neighbor_rate)
            geo_rate = torch.where(trace.provided, trace.geo_rate, geo_rate)
            ground_rate = torch.where(trace.provided, trace.ground_rate, ground_rate)
            neighbor_delay = torch.where(trace.provided & (trace.neighbor_delay > 0.0), trace.neighbor_delay, neighbor_delay)
            geo_delay = torch.where(trace.provided & (trace.geo_delay > 0.0), trace.geo_delay, geo_delay)
            ground_delay = torch.where(trace.provided & (trace.ground_delay > 0.0), trace.ground_delay, ground_delay)
            local_delay = torch.where(trace.provided & (trace.local_delay > 0.0), trace.local_delay, local_delay)
            neighbor_queue = torch.where(trace.provided & (trace.neighbor_queue > 0.0), trace.neighbor_queue, neighbor_queue)
            geo_queue = torch.where(trace.provided & (trace.geo_queue > 0.0), trace.geo_queue, geo_queue)
            ground_queue = torch.where(trace.provided & (trace.ground_queue > 0.0), trace.ground_queue, ground_queue)
            local_queue = torch.where(trace.provided & (trace.local_queue > 0.0), trace.local_queue, self.queue)
            local_completion_safe = torch.where(trace.provided, trace.abstract_action_mask_completion_safe[:, self.ACTION_LOCAL].float(), local_completion_safe)
            neighbor_completion_safe = torch.where(trace.provided, trace.abstract_action_mask_completion_safe[:, self.ACTION_NEIGHBOR].float(), neighbor_completion_safe)
            geo_completion_safe = torch.where(trace.provided, trace.abstract_action_mask_completion_safe[:, self.ACTION_GEO].float(), geo_completion_safe)
            ground_completion_safe = torch.where(trace.provided, trace.abstract_action_mask_completion_safe[:, self.ACTION_GROUND].float(), ground_completion_safe)
            local_mobility_risk = torch.where(trace.provided, trace.local_mobility_risk, local_mobility_risk)
            neighbor_mobility_risk = torch.where(trace.provided, trace.neighbor_mobility_risk, neighbor_mobility_risk)
            geo_mobility_risk = torch.where(trace.provided, trace.geo_mobility_risk, geo_mobility_risk)
            ground_mobility_risk = torch.where(trace.provided, trace.ground_mobility_risk, ground_mobility_risk)
            local_link_lifetime = torch.where(trace.provided, trace.local_link_lifetime, local_link_lifetime)
            neighbor_link_lifetime = torch.where(trace.provided, trace.neighbor_link_lifetime, neighbor_link_lifetime)
            geo_link_lifetime = torch.where(trace.provided, trace.geo_link_lifetime, geo_link_lifetime)
            ground_link_lifetime = torch.where(trace.provided, trace.ground_link_lifetime, ground_link_lifetime)
            local_link_margin = torch.where(trace.provided, trace.local_link_margin_to_completion, local_link_margin)
            neighbor_link_margin = torch.where(trace.provided, trace.neighbor_link_margin_to_completion, neighbor_link_margin)
            geo_link_margin = torch.where(trace.provided, trace.geo_link_margin_to_completion, geo_link_margin)
            ground_link_margin = torch.where(trace.provided, trace.ground_link_margin_to_completion, ground_link_margin)
            local_handover_required = torch.where(trace.provided, trace.local_handover_required, local_handover_required)
            neighbor_handover_required = torch.where(trace.provided, trace.neighbor_handover_required, neighbor_handover_required)
            geo_handover_required = torch.where(trace.provided, trace.geo_handover_required, geo_handover_required)
            ground_handover_required = torch.where(trace.provided, trace.ground_handover_required, ground_handover_required)
        else:
            local_queue = self.queue

        rows: List[Dict[str, float]] = []
        for leo in range(self.cfg.n_leo):
            rows.append(
                {
                    "leo_id": leo,
                    "local_visible": float(action_mask[leo, self.ACTION_LOCAL].item()),
                    "neighbor_visible": float(action_mask[leo, self.ACTION_NEIGHBOR].item()),
                    "geo_visible": float(action_mask[leo, self.ACTION_GEO].item()),
                    "ground_visible": float(action_mask[leo, self.ACTION_GROUND].item()),
                    "local_rate": 1000.0,
                    "neighbor_rate": float(neighbor_rate[leo].item()),
                    "geo_rate": float(geo_rate[leo].item()),
                    "ground_rate": float(ground_rate[leo].item()),
                    "local_delay": float(local_delay[leo].item()),
                    "neighbor_delay": float(neighbor_delay[leo].item()),
                    "geo_delay": float(geo_delay[leo].item()),
                    "ground_delay": float(ground_delay[leo].item()),
                    "local_queue": float(local_queue[leo].item()),
                    "neighbor_queue": float(neighbor_queue[leo].item()),
                    "geo_queue": float(geo_queue[leo].item()),
                    "ground_queue": float(ground_queue[leo].item()),
                    "local_completion_safe": float(local_completion_safe[leo].item()),
                    "neighbor_completion_safe": float(neighbor_completion_safe[leo].item()),
                    "geo_completion_safe": float(geo_completion_safe[leo].item()),
                    "ground_completion_safe": float(ground_completion_safe[leo].item()),
                    "local_mobility_risk": float(local_mobility_risk[leo].item()),
                    "neighbor_mobility_risk": float(neighbor_mobility_risk[leo].item()),
                    "geo_mobility_risk": float(geo_mobility_risk[leo].item()),
                    "ground_mobility_risk": float(ground_mobility_risk[leo].item()),
                    "local_link_lifetime_sec": float(local_link_lifetime[leo].item()),
                    "neighbor_link_lifetime_sec": float(neighbor_link_lifetime[leo].item()),
                    "geo_link_lifetime_sec": float(geo_link_lifetime[leo].item()),
                    "ground_link_lifetime_sec": float(ground_link_lifetime[leo].item()),
                    "local_link_survival_margin_to_completion_sec": float(local_link_margin[leo].item()),
                    "neighbor_link_survival_margin_to_completion_sec": float(neighbor_link_margin[leo].item()),
                    "geo_link_survival_margin_to_completion_sec": float(geo_link_margin[leo].item()),
                    "ground_link_survival_margin_to_completion_sec": float(ground_link_margin[leo].item()),
                    "local_handover_required": float(local_handover_required[leo].item()),
                    "neighbor_handover_required": float(neighbor_handover_required[leo].item()),
                    "geo_handover_required": float(geo_handover_required[leo].item()),
                    "ground_handover_required": float(ground_handover_required[leo].item()),
                }
            )
        return rows

    def _sample_arrivals(self) -> torch.Tensor:
        rate = torch.full((self.cfg.n_leo,), self.cfg.arrival_rate, device=self.device)
        burst_mask = torch.rand(self.cfg.n_leo, device=self.device, generator=self.generator) < self.cfg.burst_prob
        rate = torch.where(burst_mask, rate * self.cfg.burst_multiplier, rate)
        task_count = torch.poisson(rate, generator=self.generator)
        if not self._physical_enabled():
            return task_count
        p = self._physical_cfg()
        workload = sample_task_workload_batch(
            task_count=task_count,
            task_size_bits_mean=float(p.task_size_bits_mean),
            task_size_bits_std=float(p.task_size_bits_std),
            cycles_per_bit_mean=float(p.cycles_per_bit_mean),
            cycles_per_bit_std=float(p.cycles_per_bit_std),
            generator=self.generator,
        )
        self.last_task_bits = workload.task_bits.to(self.device)
        self.last_cycles_per_bit = workload.cycles_per_bit.to(self.device)
        return workload.arrival_cycles.to(self.device)

    def _physical_enabled(self) -> bool:
        return bool(getattr(self._physical_cfg(), "enabled", False))

    def _physical_cfg(self):
        return getattr(self.cfg, "physical", None)

    def _queue_cap_mode(self) -> str:
        mode = str(getattr(self.cfg, "queue_cap_mode", "finite_buffer") or "finite_buffer").strip().lower()
        if mode not in {"finite_buffer", "unbounded_eval"}:
            mode = "finite_buffer"
        physical = self._physical_cfg()
        if self._physical_enabled() and bool(getattr(physical, "unbounded_queue_eval", False)):
            return "unbounded_eval"
        return mode

    def _queue_cap_value(self) -> float:
        if self._physical_enabled():
            return float(getattr(self._physical_cfg(), "queue_cap_cycles", self.cfg.max_queue))
        return float(getattr(self.cfg, "max_queue", 0.0))

    def _physical_target_cpu_hz(self, upper_action: torch.Tensor) -> torch.Tensor:
        p = self._physical_cfg()
        leo = torch.full((self.cfg.n_leo,), float(p.leo_cpu_hz), dtype=torch.float32, device=self.device)
        geo = torch.full_like(leo, float(p.geo_cpu_hz))
        ground = torch.full_like(leo, float(p.ground_cpu_hz))
        return torch.where(
            upper_action == self.ACTION_GEO,
            geo,
            torch.where(upper_action == self.ACTION_GROUND, ground, leo),
        )

    def _physical_link_rate_bps(self, upper_action: torch.Tensor) -> torch.Tensor:
        p = self._physical_cfg()
        local = torch.full((self.cfg.n_leo,), float(p.local_rate_bps), dtype=torch.float32, device=self.device)
        neighbor_quality = self._neighbor_rate().max(dim=-1).values / max(float(self.cfg.isl_base_rate), 1.0e-12)
        neighbor = neighbor_quality.clamp_min(0.0) * float(p.isl_base_rate_bps)
        geo_visible, geo_quality, trace_geo_rate = self._geo_access(self.t)
        ground_visible, ground_quality, trace_ground_rate = self._ground_access(self.t)
        geo = geo_visible.float() * (0.35 + 0.65 * geo_quality.clamp(0.0, 1.0)) * float(p.geo_base_rate_bps)
        ground = ground_visible.float() * (0.35 + 0.65 * ground_quality.clamp(0.0, 1.0)) * float(p.ground_base_rate_bps)
        if trace_geo_rate is not None:
            geo = torch.where(trace_geo_rate > 0.0, trace_geo_rate * 1.0e6, geo)
        if trace_ground_rate is not None:
            ground = torch.where(trace_ground_rate > 0.0, trace_ground_rate * 1.0e6, ground)
        trace = self._trace_snapshot(self.t)
        if trace is not None:
            neighbor = torch.where(trace.provided & (trace.neighbor_rate > 0.0), trace.neighbor_rate * 1.0e6, neighbor)
        return torch.where(
            upper_action == self.ACTION_LOCAL,
            local,
            torch.where(
                upper_action == self.ACTION_NEIGHBOR,
                neighbor,
                torch.where(upper_action == self.ACTION_GEO, geo, ground),
            ),
        ).clamp_min(1.0)

    def _physical_target_queue_cycles(self, upper_action: torch.Tensor, neighbor: torch.Tensor) -> torch.Tensor:
        neighbor_queue = self.queue[neighbor]
        zeros = torch.zeros_like(self.queue)
        return torch.where(
            upper_action == self.ACTION_NEIGHBOR,
            neighbor_queue,
            torch.where(
                upper_action == self.ACTION_GEO,
                self.geo_queue,
                torch.where(upper_action == self.ACTION_GROUND, self.ground_queue, zeros),
            ),
        )

    def _build_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        cached = self._graph_cache.get(int(self.t))
        if cached is not None:
            return cached
        n = self.cfg.n_leo
        src = []
        dst = []
        attrs = []
        if not self.cfg.enable_isl:
            result = (
                torch.empty((2, 0), dtype=torch.long, device=self.device),
                torch.empty((0, self.cfg.edge_feature_dim), dtype=torch.float32, device=self.device),
            )
            self._graph_cache[int(self.t)] = result
            return result
        phase = (self.phase + self.t * self.cfg.orbit_speed) % (2 * math.pi)

        def add_edge(i: int, j: int, rate: torch.Tensor, delay_scale: float, active: torch.Tensor, edge_type: float) -> None:
            src.append(i)
            dst.append(j)
            attrs.append(torch.stack([
                rate / self.cfg.isl_base_rate,
                torch.as_tensor(delay_scale, device=self.device, dtype=rate.dtype),
                active.to(rate.dtype),
                torch.as_tensor(edge_type, device=self.device, dtype=rate.dtype),
            ]))

        for i in range(n):
            for j in ((i - 1) % n, (i + 1) % n):
                delta = self._angular_distance(phase[i], phase[j])
                rate = self.cfg.isl_base_rate * (0.65 + 0.35 * torch.cos(delta).abs())
                add_edge(i, j, rate, self.cfg.isl_prop_delay / self.cfg.geo_prop_delay, torch.ones((), device=self.device), 0.0)

            if self.cfg.enable_dynamic_skip_isl:
                for j in ((i - 2) % n, (i + 2) % n):
                    score = torch.sin(phase[i] + phase[j] + 0.3 * self.t)
                    active = (score > self.cfg.visibility_threshold).float()
                    rate = self.cfg.isl_base_rate * (0.05 + active * (0.35 + 0.25 * score.clamp_min(0.0)))
                    add_edge(i, j, rate, 1.5 * self.cfg.isl_prop_delay / self.cfg.geo_prop_delay, active, 1.0)

        edge_index = torch.tensor([src, dst], dtype=torch.long, device=self.device)
        edge_attr = torch.stack(attrs, dim=0).float()
        result = (edge_index, edge_attr)
        self._graph_cache[int(self.t)] = result
        return result

    def _select_neighbor(self, edge_index: torch.Tensor) -> torch.Tensor:
        n = self.cfg.n_leo
        neighbor = torch.arange(n, device=self.device)
        for i in range(n):
            candidates = edge_index[1, edge_index[0] == i]
            if candidates.numel() == 0:
                neighbor[i] = i
            else:
                q = self.queue[candidates]
                neighbor[i] = candidates[torch.argmin(q)]
        return neighbor

    def _neighbor_rate(self, step: int | None = None) -> torch.Tensor:
        n = self.cfg.n_leo
        step = self.t if step is None else int(step)
        cached = self._neighbor_rate_cache.get(int(step))
        if cached is not None:
            return cached
        if not self.cfg.enable_isl:
            result = torch.zeros(n, 2, device=self.device)
            self._neighbor_rate_cache[int(step)] = result
            return result
        phase = (self.phase + step * self.cfg.orbit_speed) % (2 * math.pi)
        rates = []
        for offset in (-1, 1):
            idx = (torch.arange(n, device=self.device) + offset) % n
            delta = self._angular_distance(phase, phase[idx])
            rates.append(self.cfg.isl_base_rate * (0.65 + 0.35 * torch.cos(delta).abs()))
        result = torch.stack(rates, dim=-1)
        self._neighbor_rate_cache[int(step)] = result
        return result

    def _upper_action_mask_at_step(self, step: int) -> torch.Tensor:
        return self._upper_action_mask_details_at_step(step).final_mask

    def _upper_action_mask_details_at_step(self, step: int) -> ActionMaskDiagnostics:
        step = int(step)
        cached = self._action_mask_details_cache.get(step)
        if cached is not None:
            return cached
        if self._action_mask_cache_step == step and self._action_mask_cache_value is not None:
            return self._action_mask_cache_value
        local = torch.ones(self.cfg.n_leo, dtype=torch.bool, device=self.device)
        neighbor = self.cfg.enable_isl and (self._neighbor_rate(step).max(dim=-1).values > self.cfg.mask_min_rate)
        if not torch.is_tensor(neighbor):
            neighbor = torch.full((self.cfg.n_leo,), bool(neighbor), dtype=torch.bool, device=self.device)
        geo_visible, _, _ = self._geo_access(step)
        ground_visible, _, _ = self._ground_access(step)
        visibility_mask = torch.stack([local, neighbor.bool(), geo_visible.bool(), ground_visible.bool()], dim=-1)
        enabled = torch.tensor(
            [True, self.cfg.enable_isl, self.cfg.enable_geo, self.cfg.enable_ground],
            dtype=torch.bool,
            device=self.device,
        ).view(1, 4)
        architecture = normalize_architecture(getattr(self.cfg, "action_space_architecture", "full"))
        arch_mask = torch.tensor(apply_architecture_filter([1, 1, 1, 1], architecture), dtype=torch.bool, device=self.device).view(1, 4)
        enabled = enabled & arch_mask
        trace = self._trace_snapshot(step)
        mask_source = str(getattr(self.cfg, "mask_source", "predicted") or "predicted").strip().lower()
        if mask_source not in {"measured", "predicted", "oracle_trace"}:
            mask_source = "predicted"
        prediction = self._predicted_action_mask_sources(step, visibility_mask)
        diag = build_upper_action_mask(
            visibility_mask=visibility_mask,
            architecture_mask=enabled,
            trace_snapshot=trace,
            action_mask_enabled=bool(self.cfg.action_mask_enabled),
            mode=str(getattr(self.cfg, "action_mask_layer_mode", "legacy")),
            legacy_mode=str(getattr(self.cfg, "action_mask_mode", "visible_only")),
            enable_visibility_mask=bool(getattr(self.cfg, "enable_visibility_mask", True)),
            enable_completion_safe_mask=bool(getattr(self.cfg, "enable_completion_safe_mask", True)),
            enable_mobility_risk_mask=bool(getattr(self.cfg, "enable_mobility_risk_mask", True)),
            local_action_index=self.ACTION_LOCAL,
            mask_source=mask_source,
            predicted_completion_safe_mask=prediction[0],
            predicted_mobility_safe_mask=prediction[1],
            predictor_fallback=prediction[2],
            link_lifetime_noise_std_s=float(getattr(self.cfg, "link_lifetime_noise_std_s", 0.0)),
            completion_time_noise_std_s=float(getattr(self.cfg, "completion_time_noise_std_s", 0.0)),
            mask_false_positive_rate=float(getattr(self.cfg, "mask_false_positive_rate", 0.0)),
            mask_false_negative_rate=float(getattr(self.cfg, "mask_false_negative_rate", 0.0)),
            mask_staleness_slots=int(getattr(self.cfg, "mask_staleness_slots", 0)),
            mask_false_positive_rate_observed=prediction[3],
            mask_false_negative_rate_observed=prediction[4],
        )
        self._action_mask_cache_step = step
        self._action_mask_cache_value = diag
        self._action_mask_details_cache[step] = diag
        return diag

    def _predicted_action_mask_sources(self, step: int, visibility_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        step = int(step)
        stale = max(0, int(getattr(self.cfg, "mask_staleness_slots", 0) or 0))
        if stale > 0 and (step - stale) in self._mask_prediction_cache:
            return self._mask_prediction_cache[step - stale]

        vis = visibility_mask.bool()
        horizon = float(getattr(self.cfg, "mask_prediction_horizon_s", getattr(self.cfg, "deadline_threshold", 1.0)) or 1.0)
        min_margin = float(getattr(self.cfg, "min_link_survival_margin_sec", 0.0) or 0.0)
        source = str(getattr(self.cfg, "mask_source", "predicted") or "predicted").strip().lower()
        if source == "measured":
            completion = vis.clone()
            mobility = vis.clone()
            fallback = torch.zeros(self.cfg.n_leo, dtype=torch.float32, device=self.device)
            observed_fp = torch.zeros_like(fallback)
            observed_fn = torch.zeros_like(fallback)
            value = (completion.bool(), mobility.bool(), fallback.float(), observed_fp.float(), observed_fn.float())
            self._mask_prediction_cache[step] = value
            return value

        if self._physical_enabled():
            p = self._physical_cfg()
            local_rate_bps = torch.full((self.cfg.n_leo,), float(p.local_rate_bps), dtype=torch.float32, device=self.device)
            neighbor_quality = (
                self._neighbor_rate(step).max(dim=-1).values / max(float(getattr(self.cfg, "isl_base_rate", 1.0)), 1.0e-12)
                if self.cfg.enable_isl
                else torch.zeros(self.cfg.n_leo, dtype=torch.float32, device=self.device)
            )
            neighbor_rate_bps = neighbor_quality.clamp_min(0.0) * float(p.isl_base_rate_bps)
            geo_visible, geo_quality, trace_geo_rate = self._geo_access(step)
            ground_visible, ground_quality, trace_ground_rate = self._ground_access(step)
            geo_rate_bps = geo_visible.float() * (0.35 + 0.65 * geo_quality.clamp(0.0, 1.0)) * float(p.geo_base_rate_bps)
            ground_rate_bps = ground_visible.float() * (0.35 + 0.65 * ground_quality.clamp(0.0, 1.0)) * float(p.ground_base_rate_bps)
            if trace_geo_rate is not None:
                geo_rate_bps = torch.where(trace_geo_rate > 0.0, trace_geo_rate * 1.0e6, geo_rate_bps)
            if trace_ground_rate is not None:
                ground_rate_bps = torch.where(trace_ground_rate > 0.0, trace_ground_rate * 1.0e6, ground_rate_bps)

            trace = self._trace_snapshot(step)
            if trace is not None:
                neighbor_rate_bps = torch.where(
                    trace.provided & (trace.neighbor_rate > 0.0),
                    trace.neighbor_rate * 1.0e6,
                    neighbor_rate_bps,
                )
            rates_bps = torch.stack([local_rate_bps, neighbor_rate_bps, geo_rate_bps, ground_rate_bps], dim=-1).clamp_min(0.0)
            delays = torch.stack(
                [
                    torch.full_like(local_rate_bps, float(getattr(self.cfg, "local_prop_delay", 0.0))),
                    torch.full_like(local_rate_bps, float(getattr(self.cfg, "isl_prop_delay", 0.0))),
                    torch.full_like(local_rate_bps, float(getattr(self.cfg, "geo_prop_delay", 0.0))),
                    torch.full_like(local_rate_bps, float(getattr(self.cfg, "ground_prop_delay", 0.0))),
                ],
                dim=-1,
            )
            local_cpu_hz = torch.full_like(local_rate_bps, float(p.leo_cpu_hz))
            neighbor_cpu_hz = torch.full_like(local_rate_bps, float(p.leo_cpu_hz))
            geo_cpu_hz = torch.full_like(local_rate_bps, float(p.geo_cpu_hz))
            ground_cpu_hz = torch.full_like(local_rate_bps, float(p.ground_cpu_hz))
            target_cpu_hz = torch.stack([local_cpu_hz, neighbor_cpu_hz, geo_cpu_hz, ground_cpu_hz], dim=-1).clamp_min(1.0)
            edge_index, _ = self._build_graph()
            neighbor = self._select_neighbor(edge_index)
            neighbor_queue = self.queue[neighbor]
            zeros = torch.zeros_like(self.queue)
            target_queue_cycles = torch.stack([zeros, neighbor_queue, self.geo_queue, self.ground_queue], dim=-1).clamp_min(0.0)
            link_lifetime_s = None
            if trace is not None:
                estimated = torch.zeros((self.cfg.n_leo, self.N_UPPER_ACTIONS), dtype=torch.float32, device=self.device)
                estimated[:, self.ACTION_LOCAL] = horizon * 2.0
                estimated[:, self.ACTION_NEIGHBOR] = trace.neighbor_link_lifetime
                estimated[:, self.ACTION_GEO] = trace.geo_link_lifetime
                estimated[:, self.ACTION_GROUND] = trace.ground_link_lifetime
                link_lifetime_s = estimated
            pred = predict_masks_from_physical_observables(
                visibility_mask=vis,
                backlog_cycles=self.queue,
                cycles_per_bit=self.last_cycles_per_bit,
                rate_bps_by_action=rates_bps,
                propagation_delay_s_by_action=delays,
                target_cpu_hz_by_action=target_cpu_hz,
                target_queue_cycles_by_action=target_queue_cycles,
                link_lifetime_s_by_action=link_lifetime_s,
                horizon_s=horizon,
                min_link_survival_margin_s=min_margin,
                cpu_share_for_feasibility=float(getattr(self.cfg, "mask_feasibility_cpu_share", 1.0)),
                bw_share_for_feasibility=float(getattr(self.cfg, "mask_feasibility_bw_share", 1.0)),
                local_action_index=self.ACTION_LOCAL,
            )
        else:
            neighbor_rate = self._neighbor_rate(step).max(dim=-1).values if self.cfg.enable_isl else torch.zeros(self.cfg.n_leo, device=self.device)
            geo_visible, geo_quality, trace_geo_rate = self._geo_access(step)
            ground_visible, ground_quality, trace_ground_rate = self._ground_access(step)
            local_rate = torch.full((self.cfg.n_leo,), max(float(getattr(self.cfg, "leo_cpu_capacity", 1.0)), 1.0), dtype=torch.float32, device=self.device)
            geo_rate = geo_visible.float() * (0.35 + 0.65 * geo_quality.clamp(0.0, 1.0)) * float(getattr(self.cfg, "geo_base_rate", 1.0))
            ground_rate = ground_visible.float() * (0.35 + 0.65 * ground_quality.clamp(0.0, 1.0)) * float(getattr(self.cfg, "ground_base_rate", 1.0))
            if trace_geo_rate is not None:
                geo_rate = torch.where(trace_geo_rate > 0.0, trace_geo_rate, geo_rate)
            if trace_ground_rate is not None:
                ground_rate = torch.where(trace_ground_rate > 0.0, trace_ground_rate, ground_rate)
            rates = torch.stack([local_rate, neighbor_rate, geo_rate, ground_rate], dim=-1).clamp_min(0.0)
            delays = torch.stack(
                [
                    torch.full_like(local_rate, float(getattr(self.cfg, "local_prop_delay", 0.0))),
                    torch.full_like(local_rate, float(getattr(self.cfg, "isl_prop_delay", 0.0))),
                    torch.full_like(local_rate, float(getattr(self.cfg, "geo_prop_delay", 0.0))),
                    torch.full_like(local_rate, float(getattr(self.cfg, "ground_prop_delay", 0.0))),
                ],
                dim=-1,
            )
            pred = predict_masks_from_observables(
                visibility_mask=vis,
                queue=self.queue,
                rate_by_action=rates,
                delay_by_action=delays,
                horizon_s=horizon,
                min_link_survival_margin_s=min_margin,
            )

        noisy = apply_prediction_noise(
            completion_time_s=pred.completion_time_s,
            link_lifetime_s=pred.link_lifetime_s,
            completion_safe_mask=pred.completion_safe_mask,
            mobility_safe_mask=pred.mobility_safe_mask,
            raw_mask=vis,
            horizon_s=horizon,
            min_link_survival_margin_s=min_margin,
            link_lifetime_noise_std_s=float(getattr(self.cfg, "link_lifetime_noise_std_s", 0.0)),
            completion_time_noise_std_s=float(getattr(self.cfg, "completion_time_noise_std_s", 0.0)),
            mask_false_positive_rate=float(getattr(self.cfg, "mask_false_positive_rate", 0.0)),
            mask_false_negative_rate=float(getattr(self.cfg, "mask_false_negative_rate", 0.0)),
            generator=self.generator,
        )
        completion = noisy.completion_safe_mask
        mobility = noisy.mobility_safe_mask
        fallback = pred.predictor_fallback
        observed_fp = noisy.observed_false_positive_rate
        observed_fn = noisy.observed_false_negative_rate
        value = (completion.bool(), mobility.bool(), fallback.float(), observed_fp.float(), observed_fn.float())
        self._mask_prediction_cache[step] = value
        return value

    def _mask_predictor_units_for_step(self, step: int) -> str:
        del step
        return "physical_seconds" if self._physical_enabled() else "legacy_normalized_debug"

    def _trace_snapshot(self, step: int) -> TraceTopologySnapshot | None:
        if self._trace_provider is None:
            return None
        step = int(step)
        if self._trace_snapshot_cache_step == step and self._trace_snapshot_cache_value is not None:
            return self._trace_snapshot_cache_value
        snapshot = self._trace_provider.snapshot(step)
        self._trace_snapshot_cache_step = step
        self._trace_snapshot_cache_value = snapshot
        return snapshot

    def trace_stats(self) -> Dict[str, float]:
        if self._trace_provider is None:
            return {
                "trace_missing_count": 0.0,
                "trace_fallback_count": 0.0,
                "trace_hit_ratio": 1.0,
            }
        stats = self._trace_provider.stats()
        return {
            "trace_missing_count": float(stats.get("trace_pair_missing", 0)),
            "trace_fallback_count": float(stats.get("trace_fallback_count", 0)),
            "trace_hit_ratio": float(stats.get("trace_hit_ratio", 0.0)),
        }

    def _geo_access(self, step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        step = int(step)
        cached = self._geo_access_cache.get(step)
        if cached is not None:
            return cached
        if not self.cfg.enable_geo:
            zero = torch.zeros(self.cfg.n_leo, device=self.device)
            result = (zero.bool(), zero, None)
            self._geo_access_cache[step] = result
            return result
        visible, quality = self._analytic_geo_access(step)
        trace = self._trace_snapshot(step)
        trace_rate = None
        if trace is not None:
            provided = trace.provided
            visible = torch.where(provided, trace.abstract_action_mask_visible[:, self.ACTION_GEO], visible)
            trace_quality = torch.clamp(trace.geo_rate / (self.cfg.geo_base_rate + 1e-6), min=0.0, max=1.0)
            quality = torch.where(provided, trace_quality, quality)
            trace_rate = torch.where(provided, trace.geo_rate, torch.zeros_like(trace.geo_rate))
        result = (visible.bool(), quality.float(), trace_rate)
        self._geo_access_cache[step] = result
        return result

    def _ground_access(self, step: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        step = int(step)
        cached = self._ground_access_cache.get(step)
        if cached is not None:
            return cached
        if not self.cfg.enable_ground:
            zero = torch.zeros(self.cfg.n_leo, device=self.device)
            result = (zero.bool(), zero, None)
            self._ground_access_cache[step] = result
            return result
        visible, quality = self._analytic_ground_access(step)
        trace = self._trace_snapshot(step)
        trace_rate = None
        if trace is not None:
            provided = trace.provided
            visible = torch.where(provided, trace.abstract_action_mask_visible[:, self.ACTION_GROUND], visible)
            trace_quality = torch.clamp(trace.ground_rate / (self.cfg.ground_base_rate + 1e-6), min=0.0, max=1.0)
            quality = torch.where(provided, trace_quality, quality)
            trace_rate = torch.where(provided, trace.ground_rate, torch.zeros_like(trace.ground_rate))
        result = (visible.bool(), quality.float(), trace_rate)
        self._ground_access_cache[step] = result
        return result

    def _analytic_geo_access(self, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        phase = (self.phase + step * self.cfg.orbit_speed) % (2 * math.pi)
        centers = torch.linspace(0, 2 * math.pi, max(1, self.cfg.n_geo) + 1, device=self.device)[:-1] + 0.7
        distances = torch.stack([self._angular_distance(phase, c) for c in centers], dim=-1)
        min_dist = distances.min(dim=-1).values
        width = max(1e-6, float(self.cfg.geo_coverage_width_rad))
        visible = min_dist <= width
        quality = torch.clamp((width - min_dist) / width, min=0.0, max=1.0)
        return visible, quality

    def _analytic_ground_access(self, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        phase = (self.phase + step * self.cfg.orbit_speed) % (2 * math.pi)
        centers = torch.linspace(0, 2 * math.pi, max(1, self.cfg.n_ground) + 1, device=self.device)[:-1]
        centers = (centers + self.cfg.gateway_drift_rate * step) % (2 * math.pi)
        distances = torch.stack([self._angular_distance(phase, c) for c in centers], dim=-1)
        min_dist = distances.min(dim=-1).values
        width = max(1e-6, float(self.cfg.ground_coverage_width_rad))
        visible = min_dist <= width
        quality = torch.clamp((width - min_dist) / width, min=0.0, max=1.0)
        return visible, quality

    def _visibility_geo(self) -> torch.Tensor:
        visible, _, _ = self._geo_access(self.t)
        return visible

    def _visibility_ground(self) -> torch.Tensor:
        visible, _, _ = self._ground_access(self.t)
        return visible

    def _geo_rate(self, bw_alloc: torch.Tensor, tx_power: torch.Tensor) -> torch.Tensor:
        visible, quality, trace_rate = self._geo_access(self.t)
        congestion = 1.0 / (1.0 + self.cfg.geo_backhaul_congestion * self.queue.mean() / (self.cfg.max_queue + 1e-6))
        analytic = (
            visible.float()
            * self.cfg.geo_base_rate
            * (0.35 + 0.65 * quality)
            * torch.log2(1.0 + 2.5 * tx_power)
            * (bw_alloc / self.cfg.bandwidth_max)
            * congestion
        )
        if trace_rate is not None:
            traced = trace_rate.to(self.device) * torch.log2(1.0 + 2.5 * tx_power) * (bw_alloc / self.cfg.bandwidth_max)
            analytic = torch.where(trace_rate > 0.0, traced, analytic)
        return analytic

    def _ground_rate(self, bw_alloc: torch.Tensor, tx_power: torch.Tensor) -> torch.Tensor:
        visible, quality, trace_rate = self._ground_access(self.t)
        congestion = 1.0 / (1.0 + self.cfg.ground_backhaul_congestion * self.queue.mean() / (self.cfg.max_queue + 1e-6))
        analytic = (
            visible.float()
            * self.cfg.ground_base_rate
            * (0.35 + 0.65 * quality)
            * torch.log2(1.0 + 3.0 * tx_power)
            * (bw_alloc / self.cfg.bandwidth_max)
            * congestion
        )
        if trace_rate is not None:
            traced = trace_rate.to(self.device) * torch.log2(1.0 + 3.0 * tx_power) * (bw_alloc / self.cfg.bandwidth_max)
            analytic = torch.where(trace_rate > 0.0, traced, analytic)
        return analytic

    def _target_link_terms(self, upper_action: torch.Tensor, bw_alloc: torch.Tensor, tx_power: torch.Tensor):
        local_rate = torch.full((self.cfg.n_leo,), 1e6, device=self.device)
        neigh_rate = self._neighbor_rate().max(dim=-1).values * torch.log2(1.0 + 2.0 * tx_power) * (bw_alloc / self.cfg.bandwidth_max)
        trace = self._trace_snapshot(self.t)
        if trace is not None:
            neigh_rate = torch.where(trace.provided, trace.neighbor_rate, neigh_rate)
        geo_rate = self._geo_rate(bw_alloc, tx_power)
        ground_rate = self._ground_rate(bw_alloc, tx_power)
        link_rate = torch.where(
            upper_action == self.ACTION_LOCAL,
            local_rate,
            torch.where(
                upper_action == self.ACTION_NEIGHBOR,
                neigh_rate,
                torch.where(upper_action == self.ACTION_GEO, geo_rate, ground_rate),
            ),
        )
        prop = torch.where(
            upper_action == self.ACTION_LOCAL,
            torch.full_like(link_rate, self.cfg.local_prop_delay),
            torch.where(
                upper_action == self.ACTION_NEIGHBOR,
                torch.full_like(link_rate, self.cfg.isl_prop_delay),
                torch.where(
                    upper_action == self.ACTION_GEO,
                    torch.full_like(link_rate, self.cfg.geo_prop_delay),
                    torch.full_like(link_rate, self.cfg.ground_prop_delay),
                ),
            ),
        )
        current_mask = self._upper_action_mask_at_step(self.t)
        feasible = current_mask.gather(1, upper_action.view(-1, 1)).squeeze(-1).bool()
        return link_rate, prop, feasible

    @staticmethod
    def _angular_distance(a: torch.Tensor, b: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(b):
            b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
        return torch.abs(torch.atan2(torch.sin(a - b), torch.cos(a - b)))
