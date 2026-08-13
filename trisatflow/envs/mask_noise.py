from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MaskNoiseResult:
    completion_safe_mask: torch.Tensor
    mobility_safe_mask: torch.Tensor
    false_positive_mask: torch.Tensor
    false_negative_mask: torch.Tensor
    observed_false_positive_rate: torch.Tensor
    observed_false_negative_rate: torch.Tensor


def apply_prediction_noise(
    *,
    completion_time_s: torch.Tensor,
    link_lifetime_s: torch.Tensor,
    completion_safe_mask: torch.Tensor,
    mobility_safe_mask: torch.Tensor,
    raw_mask: torch.Tensor,
    horizon_s: float,
    min_link_survival_margin_s: float = 0.0,
    link_lifetime_noise_std_s: float = 0.0,
    completion_time_noise_std_s: float = 0.0,
    mask_false_positive_rate: float = 0.0,
    mask_false_negative_rate: float = 0.0,
    generator: torch.Generator | None = None,
) -> MaskNoiseResult:
    raw = raw_mask.bool()
    noisy_completion_time = completion_time_s.float()
    noisy_link_lifetime = link_lifetime_s.float()
    if float(completion_time_noise_std_s) > 0.0:
        noisy_completion_time = noisy_completion_time + torch.randn(
            noisy_completion_time.shape,
            generator=generator,
            device=noisy_completion_time.device,
            dtype=noisy_completion_time.dtype,
        ) * float(completion_time_noise_std_s)
    if float(link_lifetime_noise_std_s) > 0.0:
        noisy_link_lifetime = noisy_link_lifetime + torch.randn(
            noisy_link_lifetime.shape,
            generator=generator,
            device=noisy_link_lifetime.device,
            dtype=noisy_link_lifetime.dtype,
        ) * float(link_lifetime_noise_std_s)

    completion = raw & (noisy_completion_time.clamp_min(0.0) <= noisy_link_lifetime.clamp_min(0.0))
    mobility = raw & ((noisy_link_lifetime.clamp_min(0.0) - noisy_completion_time.clamp_min(0.0)) >= float(min_link_survival_margin_s))
    if raw.shape[1] > 0:
        completion[:, 0] = raw[:, 0]
        mobility[:, 0] = raw[:, 0]
    completion = completion | (completion_safe_mask.bool() & raw & (completion_time_noise_std_s == 0.0) & (link_lifetime_noise_std_s == 0.0))
    mobility = mobility | (mobility_safe_mask.bool() & raw & (link_lifetime_noise_std_s == 0.0))

    before = completion & mobility
    false_positive = torch.zeros_like(before)
    false_negative = torch.zeros_like(before)
    fp_rate = max(0.0, min(1.0, float(mask_false_positive_rate)))
    fn_rate = max(0.0, min(1.0, float(mask_false_negative_rate)))
    if fp_rate > 0.0:
        fp_draw = torch.rand(before.shape, generator=generator, device=before.device) < fp_rate
        false_positive = fp_draw & raw & ~before
        completion = completion | false_positive
        mobility = mobility | false_positive
    if fn_rate > 0.0:
        fn_draw = torch.rand(before.shape, generator=generator, device=before.device) < fn_rate
        false_negative = fn_draw & raw & before
        completion = completion & ~false_negative
        mobility = mobility & ~false_negative

    denom = raw.float().sum(dim=-1).clamp_min(1.0)
    observed_fp = false_positive.float().sum(dim=-1) / denom
    observed_fn = false_negative.float().sum(dim=-1) / denom
    return MaskNoiseResult(
        completion_safe_mask=completion & raw,
        mobility_safe_mask=mobility & raw,
        false_positive_mask=false_positive,
        false_negative_mask=false_negative,
        observed_false_positive_rate=observed_fp,
        observed_false_negative_rate=observed_fn,
    )
