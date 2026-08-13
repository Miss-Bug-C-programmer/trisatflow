from __future__ import annotations

import torch

from trisatflow.envs.physical_model import compute_physical_step


def _base(**overrides):
    args = {
        "backlog_cycles": torch.tensor([100.0]),
        "task_bits": torch.tensor([10.0]),
        "cycles_per_bit": torch.tensor([10.0]),
        "cpu_share": torch.tensor([1.0]),
        "bw_share": torch.tensor([1.0]),
        "tx_power_ratio": torch.tensor([0.5]),
        "target_cpu_hz": torch.tensor([50.0]),
        "link_rate_bps": torch.tensor([10.0]),
        "propagation_delay_s": torch.tensor([0.1]),
        "target_queue_cycles": torch.tensor([0.0]),
        "feasible": torch.tensor([1.0]),
        "slot_duration_s": 1.0,
        "max_tx_power_w": 2.0,
        "kappa": 1.0e-6,
        "action": torch.tensor([1]),
    }
    args.update(overrides)
    return compute_physical_step(**args)


def test_slot_duration_doubles_service_until_backlog_saturation() -> None:
    one = _base(slot_duration_s=1.0)
    two = _base(slot_duration_s=2.0)
    saturated = _base(slot_duration_s=4.0)

    assert torch.allclose(one.served_cycles, torch.tensor([50.0]))
    assert torch.allclose(two.served_cycles, torch.tensor([100.0]))
    assert torch.allclose(saturated.served_cycles, torch.tensor([100.0]))


def test_cycles_per_bit_doubled_halves_bits_and_tx_delay() -> None:
    base = _base(cycles_per_bit=torch.tensor([10.0]))
    doubled = _base(cycles_per_bit=torch.tensor([20.0]))

    assert torch.allclose(doubled.served_bits, base.served_bits / 2.0)
    assert torch.allclose(doubled.tx_delay_s, base.tx_delay_s / 2.0)


def test_cpu_doubled_halves_compute_delay_for_unsaturated_service() -> None:
    slow = _base(backlog_cycles=torch.tensor([100.0]), target_cpu_hz=torch.tensor([50.0]), slot_duration_s=10.0)
    fast = _base(backlog_cycles=torch.tensor([100.0]), target_cpu_hz=torch.tensor([100.0]), slot_duration_s=10.0)

    assert torch.allclose(slow.served_cycles, fast.served_cycles)
    assert torch.allclose(fast.compute_delay_s, slow.compute_delay_s / 2.0)


def test_local_action_has_zero_tx_delay_and_energy() -> None:
    local = _base(action=torch.tensor([0]))

    assert float(local.tx_delay_s.item()) == 0.0
    assert float(local.tx_energy_j.item()) == 0.0


def test_energy_nonnegative_and_remote_tx_energy_monotonic_in_power() -> None:
    low = _base(tx_power_ratio=torch.tensor([0.25]))
    high = _base(tx_power_ratio=torch.tensor([0.75]))

    assert float(low.total_energy_j.item()) >= 0.0
    assert float(high.total_energy_j.item()) >= 0.0
    assert float(high.tx_energy_j.item()) > float(low.tx_energy_j.item())
