from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class MaskPrediction:
    completion_time_s: torch.Tensor
    link_lifetime_s: torch.Tensor
    completion_safe_mask: torch.Tensor
    mobility_safe_mask: torch.Tensor
    predictor_fallback: torch.Tensor


def estimate_completion_time(
    *,
    visibility_mask: torch.Tensor,
    queue: torch.Tensor,
    rate_by_action: torch.Tensor,
    delay_by_action: torch.Tensor,
    horizon_s: float,
) -> torch.Tensor:
    """Legacy normalized completion proxy.

    This function is retained for non-physical/debug configurations where the
    queue and rate tensors are both normalized legacy quantities.  Physical
    configurations must use :func:`estimate_completion_time_physical` so cycles,
    bits, hertz, bit/s, and seconds are not mixed.
    """

    vis = visibility_mask.bool()
    q = queue.float().view(-1, 1)
    rates = rate_by_action.float().clamp_min(1.0e-6)
    delays = delay_by_action.float().clamp_min(0.0)
    service_proxy = q / rates
    completion = delays + service_proxy
    fallback = torch.full_like(completion, float(max(horizon_s, 1.0)))
    return torch.where(vis, completion, fallback)


def estimate_completion_time_physical(
    *,
    visibility_mask: torch.Tensor,
    backlog_cycles: torch.Tensor,
    cycles_per_bit: torch.Tensor,
    rate_bps_by_action: torch.Tensor,
    propagation_delay_s_by_action: torch.Tensor,
    target_cpu_hz_by_action: torch.Tensor,
    target_queue_cycles_by_action: torch.Tensor,
    cpu_share_for_feasibility: float = 1.0,
    bw_share_for_feasibility: float = 1.0,
    local_action_index: int = 0,
    horizon_s: float = 1.0,
) -> torch.Tensor:
    """Dimensionally consistent completion-time estimate for mask prediction.

    The completion mask is an upper-layer feasibility prior.  It should answer
    whether an action can plausibly complete before link breakage under an
    admissible lower-level resource allocation; it must not mix physical queue
    cycles with legacy Mbps/unitless rates.  Queue/backlog are CPU cycles,
    cycles_per_bit converts cycles to payload bits, CPU is cycles/s, rates are
    bit/s, and the result is seconds.
    """

    vis = visibility_mask.bool()
    backlog = backlog_cycles.float().clamp_min(0.0).view(-1, 1)
    cpb = cycles_per_bit.float().clamp_min(1.0).view(-1, 1)
    rates = rate_bps_by_action.float().clamp_min(1.0)
    prop = propagation_delay_s_by_action.float().clamp_min(0.0)
    cpu = target_cpu_hz_by_action.float().clamp_min(1.0)
    target_queue = target_queue_cycles_by_action.float().clamp_min(0.0)
    cpu_share = max(1.0e-6, min(1.0, float(cpu_share_for_feasibility)))
    bw_share = max(1.0e-6, min(1.0, float(bw_share_for_feasibility)))

    effective_cpu = cpu * cpu_share
    effective_rate = rates * bw_share
    bits = backlog / cpb
    tx_time = bits / effective_rate
    compute_time = backlog / effective_cpu
    target_wait_time = target_queue / effective_cpu
    completion = prop + compute_time + target_wait_time + tx_time

    if 0 <= int(local_action_index) < completion.shape[1]:
        local_idx = int(local_action_index)
        # Local execution has no radio transfer; completion is governed by the
        # source backlog and local CPU only.
        completion[:, local_idx] = prop[:, local_idx] + compute_time[:, local_idx]

    fallback = torch.full_like(completion, float(max(horizon_s, 1.0)))
    return torch.where(vis, completion, fallback)


def estimate_link_lifetime(
    *,
    visibility_mask: torch.Tensor,
    rate_by_action: torch.Tensor,
    horizon_s: float,
) -> torch.Tensor:
    vis = visibility_mask.bool()
    rates = rate_by_action.float().clamp_min(0.0)
    norm = rates / rates.max(dim=-1, keepdim=True).values.clamp_min(1.0e-6)
    lifetime = float(max(horizon_s, 1.0)) * (0.75 + 1.25 * norm)
    return torch.where(vis, lifetime, torch.zeros_like(lifetime))


def estimate_link_lifetime_from_remote_rates(
    *,
    visibility_mask: torch.Tensor,
    rate_by_action: torch.Tensor,
    horizon_s: float,
    local_action_index: int = 0,
) -> torch.Tensor:
    """Predict link lifetime from observable remote-link quality.

    Local CPU capacity is not a radio-link quality.  Excluding the local column
    avoids the physical-mode bug where a large local_rate_bps compresses all
    remote lifetimes toward the minimum and over-prunes satellite/ground links.
    """

    vis = visibility_mask.bool()
    rates = rate_by_action.float().clamp_min(0.0)
    remote_rates = rates.clone()
    local_idx = int(local_action_index)
    if 0 <= local_idx < remote_rates.shape[1]:
        remote_rates[:, local_idx] = 0.0
    remote_max = remote_rates.max(dim=-1, keepdim=True).values.clamp_min(1.0e-6)
    norm = remote_rates / remote_max
    lifetime = float(max(horizon_s, 1.0)) * (0.75 + 1.25 * norm)
    if 0 <= local_idx < lifetime.shape[1]:
        # Local execution is not constrained by radio link survival.
        lifetime[:, local_idx] = float(max(horizon_s, 1.0)) * 2.0
    return torch.where(vis, lifetime, torch.zeros_like(lifetime))


def predict_masks_from_observables(
    *,
    visibility_mask: torch.Tensor,
    queue: torch.Tensor,
    rate_by_action: torch.Tensor | None,
    delay_by_action: torch.Tensor | None,
    horizon_s: float,
    min_link_survival_margin_s: float = 0.0,
) -> MaskPrediction:
    vis = visibility_mask.bool()
    fallback = torch.zeros(vis.shape[0], dtype=torch.float32, device=vis.device)
    if rate_by_action is None:
        rate_by_action = vis.float()
        fallback += 1.0
    if delay_by_action is None:
        delay_by_action = torch.where(vis, torch.ones_like(vis, dtype=torch.float32), torch.full_like(vis, float(horizon_s), dtype=torch.float32))
        fallback += 1.0
    completion_time = estimate_completion_time(
        visibility_mask=vis,
        queue=queue,
        rate_by_action=rate_by_action,
        delay_by_action=delay_by_action,
        horizon_s=horizon_s,
    )
    link_lifetime = estimate_link_lifetime(
        visibility_mask=vis,
        rate_by_action=rate_by_action,
        horizon_s=horizon_s,
    )
    margin = float(min_link_survival_margin_s)
    completion_safe = vis & (completion_time <= link_lifetime.clamp_min(1.0e-6))
    mobility_safe = vis & ((link_lifetime - completion_time) >= margin)
    # Local execution has no link handover risk in this abstraction.
    if vis.shape[1] > 0:
        completion_safe[:, 0] = vis[:, 0]
        mobility_safe[:, 0] = vis[:, 0]
    return MaskPrediction(
        completion_time_s=completion_time,
        link_lifetime_s=link_lifetime,
        completion_safe_mask=completion_safe,
        mobility_safe_mask=mobility_safe,
        predictor_fallback=(fallback > 0).float(),
    )


def predict_masks_from_physical_observables(
    *,
    visibility_mask: torch.Tensor,
    backlog_cycles: torch.Tensor,
    cycles_per_bit: torch.Tensor,
    rate_bps_by_action: torch.Tensor,
    propagation_delay_s_by_action: torch.Tensor,
    target_cpu_hz_by_action: torch.Tensor,
    target_queue_cycles_by_action: torch.Tensor,
    link_lifetime_s_by_action: torch.Tensor | None,
    horizon_s: float,
    min_link_survival_margin_s: float = 0.0,
    cpu_share_for_feasibility: float = 1.0,
    bw_share_for_feasibility: float = 1.0,
    local_action_index: int = 0,
) -> MaskPrediction:
    vis = visibility_mask.bool()
    completion_time = estimate_completion_time_physical(
        visibility_mask=vis,
        backlog_cycles=backlog_cycles,
        cycles_per_bit=cycles_per_bit,
        rate_bps_by_action=rate_bps_by_action,
        propagation_delay_s_by_action=propagation_delay_s_by_action,
        target_cpu_hz_by_action=target_cpu_hz_by_action,
        target_queue_cycles_by_action=target_queue_cycles_by_action,
        cpu_share_for_feasibility=cpu_share_for_feasibility,
        bw_share_for_feasibility=bw_share_for_feasibility,
        local_action_index=local_action_index,
        horizon_s=horizon_s,
    )
    if link_lifetime_s_by_action is None:
        link_lifetime = estimate_link_lifetime_from_remote_rates(
            visibility_mask=vis,
            rate_by_action=rate_bps_by_action,
            horizon_s=horizon_s,
            local_action_index=local_action_index,
        )
    else:
        link_lifetime = torch.where(
            vis,
            link_lifetime_s_by_action.float().clamp_min(0.0),
            torch.zeros_like(completion_time),
        )
    margin = float(min_link_survival_margin_s)
    completion_safe = vis & (completion_time <= link_lifetime.clamp_min(1.0e-6))
    mobility_safe = vis & ((link_lifetime - completion_time) >= margin)
    if vis.shape[1] > 0:
        local_idx = int(local_action_index)
        if 0 <= local_idx < vis.shape[1]:
            completion_safe[:, local_idx] = vis[:, local_idx]
            mobility_safe[:, local_idx] = vis[:, local_idx]
    return MaskPrediction(
        completion_time_s=completion_time,
        link_lifetime_s=link_lifetime,
        completion_safe_mask=completion_safe,
        mobility_safe_mask=mobility_safe,
        predictor_fallback=torch.zeros(vis.shape[0], dtype=torch.float32, device=vis.device),
    )


def estimate_completion_time_from_candidate_info(obs: Any, candidate_info: Any) -> float:
    del obs
    if not isinstance(candidate_info, dict):
        return 1.0
    for key in ("estimated_delay", "estimatedTotalDelaySec", "estimatedCompletionTimeSec"):
        if key in candidate_info:
            try:
                return max(0.0, float(candidate_info[key]))
            except (TypeError, ValueError):
                return 1.0
    return 1.0


def estimate_link_lifetime_from_candidate_info(obs: Any, candidate_info: Any) -> float:
    del obs
    if not isinstance(candidate_info, dict):
        return 1.0
    for key in ("link_lifetime_sec", "linkLifetimeSec", "link_survival_margin_to_completion_sec"):
        if key in candidate_info:
            try:
                return max(0.0, float(candidate_info[key]))
            except (TypeError, ValueError):
                return 1.0
    return 1.0
