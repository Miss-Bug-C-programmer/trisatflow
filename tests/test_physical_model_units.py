from __future__ import annotations

import torch

from trisatflow.envs.physical_model import compute_physical_step


def _physical_step(*, cpu_share: float, bw_share: float):
    return compute_physical_step(
        backlog_cycles=torch.tensor([4.0e9]),
        task_bits=torch.tensor([4.0e6]),
        cycles_per_bit=torch.tensor([1000.0]),
        cpu_share=torch.tensor([cpu_share]),
        bw_share=torch.tensor([bw_share]),
        tx_power_ratio=torch.tensor([0.5]),
        target_cpu_hz=torch.tensor([8.0e9]),
        link_rate_bps=torch.tensor([20.0e6]),
        propagation_delay_s=torch.tensor([0.2]),
        target_queue_cycles=torch.tensor([0.0]),
        feasible=torch.tensor([True]),
        slot_duration_s=1.0,
        max_tx_power_w=2.0,
        kappa=1.0e-28,
        action=torch.tensor([1]),
        local_action_index=0,
    )


def test_cpu_share_increases_service_capacity_without_increasing_compute_delay() -> None:
    low = _physical_step(cpu_share=0.25, bw_share=0.5)
    high = _physical_step(cpu_share=0.75, bw_share=0.5)

    assert float(high.effective_cpu_hz.item()) > float(low.effective_cpu_hz.item())
    assert float(high.service_capacity_cycles.item()) > float(low.service_capacity_cycles.item())
    assert float(high.served_cycles.item()) >= float(low.served_cycles.item())
    assert float(high.compute_delay_s.item()) <= float(low.compute_delay_s.item()) + 1.0e-9


def test_bandwidth_share_increases_link_capacity_without_increasing_tx_delay() -> None:
    low = _physical_step(cpu_share=0.5, bw_share=0.25)
    high = _physical_step(cpu_share=0.5, bw_share=0.75)

    assert float(high.effective_link_bps.item()) > float(low.effective_link_bps.item())
    assert float(high.tx_delay_s.item()) <= float(low.tx_delay_s.item())


def test_queueing_delay_does_not_double_count_served_workload() -> None:
    out = _physical_step(cpu_share=0.5, bw_share=0.5)
    expected_waiting_cycles = (out.served_cycles.new_tensor([4.0e9]) - out.served_cycles).clamp_min(0.0)
    expected_queueing = expected_waiting_cycles / out.effective_cpu_hz

    assert torch.allclose(out.queueing_delay_s, expected_queueing)
    assert torch.allclose(
        out.e2e_delay_s,
        out.queueing_delay_s + out.tx_delay_s + out.compute_delay_s + out.propagation_delay_s + out.target_queue_delay_s,
    )


def test_energy_output_contains_split_unit_fields() -> None:
    out = _physical_step(cpu_share=0.5, bw_share=0.5)

    for field in (
        "source_tx_energy_j",
        "source_local_compute_energy_j",
        "remote_compute_energy_j",
        "network_energy_j",
        "total_energy_j",
    ):
        value = getattr(out, field)
        assert value.shape == torch.Size([1])
        assert float(value.item()) >= 0.0
    assert torch.allclose(
        out.total_energy_j,
        out.source_tx_energy_j
        + out.source_local_compute_energy_j
        + out.remote_compute_energy_j
        + out.network_energy_j,
    )
