from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TaskWorkloadBatch:
    task_count: torch.Tensor
    task_bits: torch.Tensor
    cycles_per_bit: torch.Tensor
    arrival_cycles: torch.Tensor


def sample_task_workload_batch(
    *,
    task_count: torch.Tensor,
    task_size_bits_mean: float,
    task_size_bits_std: float,
    cycles_per_bit_mean: float,
    cycles_per_bit_std: float,
    generator: torch.Generator | None = None,
) -> TaskWorkloadBatch:
    """Sample positive task sizes and CPU intensities, then convert to cycles."""

    device = task_count.device
    dtype = torch.float32
    count = task_count.to(device=device, dtype=dtype).clamp_min(0.0)
    shape = count.shape

    bits_mean = torch.full(shape, float(task_size_bits_mean), dtype=dtype, device=device)
    cpb_mean = torch.full(shape, float(cycles_per_bit_mean), dtype=dtype, device=device)
    if float(task_size_bits_std) > 0.0:
        task_bits = bits_mean + float(task_size_bits_std) * torch.randn(shape, dtype=dtype, device=device, generator=generator)
    else:
        task_bits = bits_mean
    if float(cycles_per_bit_std) > 0.0:
        cycles_per_bit = cpb_mean + float(cycles_per_bit_std) * torch.randn(shape, dtype=dtype, device=device, generator=generator)
    else:
        cycles_per_bit = cpb_mean

    task_bits = task_bits.clamp_min(1.0)
    cycles_per_bit = cycles_per_bit.clamp_min(1.0)
    arrival_cycles = count * task_bits * cycles_per_bit
    return TaskWorkloadBatch(
        task_count=count,
        task_bits=task_bits,
        cycles_per_bit=cycles_per_bit,
        arrival_cycles=arrival_cycles,
    )
