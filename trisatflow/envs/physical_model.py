from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PhysicalStepOutput:
    effective_cpu_hz: torch.Tensor
    effective_link_bps: torch.Tensor
    service_capacity_cycles: torch.Tensor
    served_cycles: torch.Tensor
    served_bits: torch.Tensor
    waiting_cycles: torch.Tensor
    queueing_delay_s: torch.Tensor
    tx_delay_s: torch.Tensor
    compute_delay_s: torch.Tensor
    propagation_delay_s: torch.Tensor
    target_queue_delay_s: torch.Tensor
    e2e_delay_s: torch.Tensor
    compute_energy_j: torch.Tensor
    tx_energy_j: torch.Tensor
    source_tx_energy_j: torch.Tensor
    source_local_compute_energy_j: torch.Tensor
    remote_compute_energy_j: torch.Tensor
    network_energy_j: torch.Tensor
    total_energy_j: torch.Tensor


def compute_physical_step(
    *,
    backlog_cycles: torch.Tensor,
    task_bits: torch.Tensor,
    cycles_per_bit: torch.Tensor,
    cpu_share: torch.Tensor,
    bw_share: torch.Tensor,
    tx_power_ratio: torch.Tensor,
    target_cpu_hz: torch.Tensor,
    link_rate_bps: torch.Tensor,
    propagation_delay_s: torch.Tensor,
    target_queue_cycles: torch.Tensor,
    feasible: torch.Tensor,
    slot_duration_s: float,
    max_tx_power_w: float,
    kappa: float,
    action: torch.Tensor,
    local_action_index: int = 0,
    compute_energy_model: str = "kappa_cycles_f2",
) -> PhysicalStepOutput:
    """Compute one dimensioned service/delay/energy step.

    Queue/backlog/service are CPU cycles. Rates are bit/s, CPU capacities are
    cycles/s (Hz), delays are seconds, and energy is Joules.
    """

    del task_bits  # Workload size is represented by backlog cycles here.
    backlog = backlog_cycles.float().clamp_min(0.0)
    cpb = cycles_per_bit.float().clamp_min(1.0e-12)
    cpu = cpu_share.float().clamp(0.0, 1.0)
    bw = bw_share.float().clamp(0.0, 1.0)
    power_ratio = tx_power_ratio.float().clamp(0.0, 1.0)
    target_cpu = target_cpu_hz.float().clamp_min(1.0e-12)
    link_rate = link_rate_bps.float().clamp_min(1.0e-12)
    prop = propagation_delay_s.float().clamp_min(0.0)
    target_queue = target_queue_cycles.float().clamp_min(0.0)
    feasible_f = feasible.float()
    action_long = action.long()

    effective_cpu_hz = (cpu * target_cpu).clamp_min(1.0e-12)
    service_capacity_cycles = effective_cpu_hz * float(slot_duration_s)
    served_cycles = torch.minimum(backlog, service_capacity_cycles) * feasible_f
    served_bits = served_cycles / cpb

    is_local = action_long == int(local_action_index)
    effective_link_bps = (bw * link_rate).clamp_min(1.0e-12)
    remote_tx_delay = served_bits / effective_link_bps
    tx_delay_s = torch.where(is_local, torch.zeros_like(remote_tx_delay), remote_tx_delay)
    compute_delay_s = served_cycles / effective_cpu_hz
    waiting_cycles = (backlog - served_cycles).clamp_min(0.0)
    queueing_delay_s = waiting_cycles / effective_cpu_hz
    target_queue_delay_s = target_queue / effective_cpu_hz
    target_queue_delay_s = torch.where(is_local, torch.zeros_like(target_queue_delay_s), target_queue_delay_s)
    e2e_delay_s = queueing_delay_s + tx_delay_s + compute_delay_s + prop + target_queue_delay_s

    tx_power_w = power_ratio * float(max_tx_power_w)
    tx_energy_j = tx_power_w * tx_delay_s
    tx_energy_j = torch.where(is_local, torch.zeros_like(tx_energy_j), tx_energy_j)

    model = str(compute_energy_model or "kappa_cycles_f2").strip().lower()
    if model == "kappa_f3_time":
        active_time_s = compute_delay_s
        compute_energy_j = float(kappa) * effective_cpu_hz.pow(3) * active_time_s
    elif model == "kappa_cycles_f2":
        compute_energy_j = float(kappa) * served_cycles * effective_cpu_hz.pow(2)
    else:
        raise ValueError(
            f"unsupported compute_energy_model={compute_energy_model!r}; "
            "expected 'kappa_cycles_f2' or 'kappa_f3_time'"
        )
    compute_energy_j = compute_energy_j.clamp_min(0.0)
    source_tx_energy_j = tx_energy_j.clamp_min(0.0)
    source_local_compute_energy_j = torch.where(is_local, compute_energy_j, torch.zeros_like(compute_energy_j))
    remote_compute_energy_j = torch.where(is_local, torch.zeros_like(compute_energy_j), compute_energy_j)
    network_energy_j = torch.zeros_like(compute_energy_j)
    total_energy_j = (source_tx_energy_j + source_local_compute_energy_j + remote_compute_energy_j + network_energy_j).clamp_min(0.0)

    return PhysicalStepOutput(
        effective_cpu_hz=effective_cpu_hz,
        effective_link_bps=effective_link_bps,
        service_capacity_cycles=service_capacity_cycles,
        served_cycles=served_cycles,
        served_bits=served_bits,
        waiting_cycles=waiting_cycles,
        queueing_delay_s=queueing_delay_s,
        tx_delay_s=tx_delay_s,
        compute_delay_s=compute_delay_s,
        propagation_delay_s=prop,
        target_queue_delay_s=target_queue_delay_s,
        e2e_delay_s=e2e_delay_s,
        compute_energy_j=compute_energy_j,
        tx_energy_j=tx_energy_j,
        source_tx_energy_j=source_tx_energy_j,
        source_local_compute_energy_j=source_local_compute_energy_j,
        remote_compute_energy_j=remote_compute_energy_j,
        network_energy_j=network_energy_j,
        total_energy_j=total_energy_j,
    )
