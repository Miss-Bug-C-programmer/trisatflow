from __future__ import annotations

from dataclasses import dataclass

import torch

from trisatflow.envs.topology_trace import TraceTopologySnapshot

_MODE_SYNONYMS = {
    "none": "none",
    "no_mask": "none",
    "off": "none",
    "visibility": "visibility",
    "visible_only": "visibility",
    "visibility_only": "visibility",
    "completion_safe": "completion_safe",
    "completion": "completion_safe",
    "mobility_risk": "mobility_risk",
    "mobility_safe": "mobility_risk",
    "full": "full",
    "full_mask": "full",
    "legacy": "legacy",
}


@dataclass
class ActionMaskDiagnostics:
    mode: str
    raw_mask: torch.Tensor
    visibility_mask: torch.Tensor
    completion_safe_mask: torch.Tensor
    mobility_risk_mask: torch.Tensor
    final_mask: torch.Tensor
    raw_count: torch.Tensor
    visibility_count: torch.Tensor
    completion_safe_count: torch.Tensor
    mobility_risk_count: torch.Tensor
    final_count: torch.Tensor
    masked_action_ratio: torch.Tensor
    visibility_mask_ratio: torch.Tensor
    completion_mask_ratio: torch.Tensor
    mobility_mask_ratio: torch.Tensor
    fallback_due_missing_field_count: torch.Tensor
    fallback_due_missing_field_ratio: torch.Tensor
    mask_source: str
    uses_oracle_trace_mask: bool
    deployable: bool
    link_lifetime_noise_std_s: float
    completion_time_noise_std_s: float
    mask_false_positive_rate: float
    mask_false_negative_rate: float
    mask_staleness_slots: int
    mask_false_positive_rate_observed: torch.Tensor
    mask_false_negative_rate_observed: torch.Tensor
    predictor_fallback: torch.Tensor
    fallback_due_empty_mask_count: torch.Tensor

    @property
    def raw_visibility_mask(self) -> torch.Tensor:
        return self.visibility_mask

    @property
    def final_action_mask(self) -> torch.Tensor:
        return self.final_mask

    @property
    def predicted_safety_mask(self) -> torch.Tensor:
        return self.completion_safe_mask & self.mobility_risk_mask


def resolve_action_mask_mode(mode: str | None, *, legacy_mode: str | None = "visible_only") -> str:
    normalized = _MODE_SYNONYMS.get(str(mode or "legacy").strip().lower(), "legacy")
    if normalized != "legacy":
        return normalized
    legacy = _MODE_SYNONYMS.get(str(legacy_mode or "visible_only").strip().lower(), "visibility")
    if legacy in {"none", "visibility", "completion_safe", "mobility_risk", "full"}:
        return legacy
    return "visibility"


def build_upper_action_mask(
    *,
    visibility_mask: torch.Tensor,
    architecture_mask: torch.Tensor,
    trace_snapshot: TraceTopologySnapshot | None,
    action_mask_enabled: bool,
    mode: str | None,
    legacy_mode: str | None,
    enable_visibility_mask: bool,
    enable_completion_safe_mask: bool,
    enable_mobility_risk_mask: bool,
    local_action_index: int = 0,
    mask_source: str = "predicted",
    predicted_completion_safe_mask: torch.Tensor | None = None,
    predicted_mobility_safe_mask: torch.Tensor | None = None,
    predictor_fallback: torch.Tensor | None = None,
    link_lifetime_noise_std_s: float = 0.0,
    completion_time_noise_std_s: float = 0.0,
    mask_false_positive_rate: float = 0.0,
    mask_false_negative_rate: float = 0.0,
    mask_staleness_slots: int = 0,
    mask_false_positive_rate_observed: torch.Tensor | None = None,
    mask_false_negative_rate_observed: torch.Tensor | None = None,
) -> ActionMaskDiagnostics:
    vis = visibility_mask.bool()
    arch = architecture_mask.bool()
    if arch.dim() == 1:
        arch = arch.view(1, -1)
    if arch.shape[0] == 1 and vis.shape[0] > 1:
        arch = arch.expand(vis.shape[0], vis.shape[1])
    if arch.shape != vis.shape:
        raise ValueError(f"architecture_mask shape={tuple(arch.shape)} does not match visibility_mask shape={tuple(vis.shape)}")

    raw = arch.clone()
    visibility = raw.clone()
    completion = visibility.clone()
    mobility = completion.clone()
    fallback_due_missing_field = torch.zeros(vis.shape[0], dtype=torch.float32, device=vis.device)
    source = str(mask_source or "predicted").strip().lower()
    if source not in {"measured", "predicted", "oracle_trace"}:
        source = "predicted"
    uses_oracle_trace = source == "oracle_trace"
    deployable = not uses_oracle_trace

    resolved_mode = resolve_action_mask_mode(mode, legacy_mode=legacy_mode)
    if not bool(action_mask_enabled):
        final, empty_fallback = _ensure_non_empty(raw.clone(), raw_mask=raw, local_action_index=local_action_index)
        return _build_diagnostics(
            mode=resolved_mode,
            raw=raw,
            visibility=visibility,
            completion=completion,
            mobility=mobility,
            final=final,
            mask_source=source,
            uses_oracle_trace_mask=uses_oracle_trace,
            deployable=deployable,
            link_lifetime_noise_std_s=link_lifetime_noise_std_s,
            completion_time_noise_std_s=completion_time_noise_std_s,
            mask_false_positive_rate=mask_false_positive_rate,
            mask_false_negative_rate=mask_false_negative_rate,
            mask_staleness_slots=mask_staleness_slots,
            mask_false_positive_rate_observed=mask_false_positive_rate_observed,
            mask_false_negative_rate_observed=mask_false_negative_rate_observed,
            predictor_fallback=predictor_fallback,
            fallback_due_empty_mask_count=empty_fallback,
        )

    apply_visibility = resolved_mode in {"visibility", "completion_safe", "mobility_risk", "full"}
    apply_completion = resolved_mode in {"completion_safe", "full"}
    apply_mobility = resolved_mode in {"mobility_risk", "full"}

    if not bool(enable_visibility_mask):
        apply_visibility = False
    if not bool(enable_completion_safe_mask):
        apply_completion = False
    if not bool(enable_mobility_risk_mask):
        apply_mobility = False

    if apply_visibility:
        vis_source = vis
        if source == "oracle_trace" and trace_snapshot is not None:
            provided = trace_snapshot.provided.bool().view(-1, 1)
            vis_trace = trace_snapshot.abstract_action_mask_visible.bool()
            visible_present = _trace_mask_field_present(trace_snapshot, 0, vis).view(-1, 1)
            fallback_due_missing_field += (provided.squeeze(-1) & ~visible_present.squeeze(-1)).float()
            vis_source = torch.where(provided & visible_present, vis_trace, vis_source)
        visibility = raw & vis_source
    else:
        visibility = raw.clone()

    if apply_completion:
        completion_source = visibility.clone()
        if source == "oracle_trace" and trace_snapshot is not None:
            provided = trace_snapshot.provided.bool().view(-1, 1)
            completion_trace = trace_snapshot.abstract_action_mask_completion_safe.bool()
            completion_present = _trace_mask_field_present(trace_snapshot, 1, vis).view(-1, 1)
            fallback_due_missing_field += (provided.squeeze(-1) & ~completion_present.squeeze(-1)).float()
            completion_source = torch.where(provided & completion_present, completion_trace, completion_source)
        elif source == "predicted" and predicted_completion_safe_mask is not None:
            completion_source = predicted_completion_safe_mask.bool()
        completion = visibility & completion_source
    else:
        completion = visibility.clone()

    if apply_mobility:
        mobility_source = completion.clone()
        if source == "oracle_trace" and trace_snapshot is not None:
            provided = trace_snapshot.provided.bool().view(-1, 1)
            mobility_trace = trace_snapshot.abstract_action_mask_mobility_safe.bool()
            mobility_present = _trace_mask_field_present(trace_snapshot, 2, vis).view(-1, 1)
            fallback_due_missing_field += (provided.squeeze(-1) & ~mobility_present.squeeze(-1)).float()
            mobility_source = torch.where(provided & mobility_present, mobility_trace, mobility_source)
        elif source == "predicted" and predicted_mobility_safe_mask is not None:
            mobility_source = predicted_mobility_safe_mask.bool()
        mobility = completion & mobility_source
    else:
        mobility = completion.clone()

    final, empty_fallback = _ensure_non_empty(mobility.clone(), raw_mask=raw, local_action_index=local_action_index)
    return _build_diagnostics(
        mode=resolved_mode,
        raw=raw,
        visibility=visibility,
        completion=completion,
        mobility=mobility,
        final=final,
        fallback_due_missing_field=fallback_due_missing_field,
        mask_source=source,
        uses_oracle_trace_mask=uses_oracle_trace,
        deployable=deployable,
        link_lifetime_noise_std_s=link_lifetime_noise_std_s,
        completion_time_noise_std_s=completion_time_noise_std_s,
        mask_false_positive_rate=mask_false_positive_rate,
        mask_false_negative_rate=mask_false_negative_rate,
        mask_staleness_slots=mask_staleness_slots,
        mask_false_positive_rate_observed=mask_false_positive_rate_observed,
        mask_false_negative_rate_observed=mask_false_negative_rate_observed,
        predictor_fallback=predictor_fallback,
        fallback_due_empty_mask_count=empty_fallback,
    )


def _ensure_non_empty(mask: torch.Tensor, *, raw_mask: torch.Tensor, local_action_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    safe = mask.bool()
    raw = raw_mask.bool()
    empty = ~safe.any(dim=-1)
    if not empty.any():
        return safe, torch.zeros(safe.shape[0], dtype=torch.float32, device=safe.device)
    fixed = safe.clone()
    for idx in torch.nonzero(empty, as_tuple=False).view(-1).tolist():
        fallback = torch.zeros_like(fixed[idx], dtype=torch.bool)
        local_ok = 0 <= int(local_action_index) < int(fallback.numel()) and bool(raw[idx, int(local_action_index)].item())
        if local_ok:
            fallback[int(local_action_index)] = True
        else:
            valid = torch.nonzero(raw[idx], as_tuple=False).view(-1)
            if valid.numel() > 0:
                fallback[int(valid[0].item())] = True
            elif 0 <= int(local_action_index) < int(fallback.numel()):
                fallback[int(local_action_index)] = True
            else:
                fallback[0] = True
        fixed[idx] = fallback
    return fixed, empty.float()


def _build_diagnostics(
    *,
    mode: str,
    raw: torch.Tensor,
    visibility: torch.Tensor,
    completion: torch.Tensor,
    mobility: torch.Tensor,
    final: torch.Tensor,
    fallback_due_missing_field: torch.Tensor | None = None,
    mask_source: str = "predicted",
    uses_oracle_trace_mask: bool = False,
    deployable: bool = True,
    link_lifetime_noise_std_s: float = 0.0,
    completion_time_noise_std_s: float = 0.0,
    mask_false_positive_rate: float = 0.0,
    mask_false_negative_rate: float = 0.0,
    mask_staleness_slots: int = 0,
    mask_false_positive_rate_observed: torch.Tensor | None = None,
    mask_false_negative_rate_observed: torch.Tensor | None = None,
    predictor_fallback: torch.Tensor | None = None,
    fallback_due_empty_mask_count: torch.Tensor | None = None,
) -> ActionMaskDiagnostics:
    raw_count = raw.float().sum(dim=-1)
    visibility_count = visibility.float().sum(dim=-1)
    completion_count = completion.float().sum(dim=-1)
    mobility_count = mobility.float().sum(dim=-1)
    final_count = final.float().sum(dim=-1)
    denom = raw_count.clamp_min(1.0)
    masked_ratio = (raw_count - final_count).clamp_min(0.0) / denom
    visibility_ratio = (raw_count - visibility_count).clamp_min(0.0) / denom
    completion_ratio = (visibility_count - completion_count).clamp_min(0.0) / denom
    mobility_ratio = (completion_count - mobility_count).clamp_min(0.0) / denom
    if fallback_due_missing_field is None:
        fallback_due_missing_field = torch.zeros_like(final_count)
    fallback_due_missing_field = fallback_due_missing_field.float()
    fallback_due_missing_field_ratio = (fallback_due_missing_field > 0).float()
    if mask_false_positive_rate_observed is None:
        mask_false_positive_rate_observed = torch.zeros_like(final_count)
    if mask_false_negative_rate_observed is None:
        mask_false_negative_rate_observed = torch.zeros_like(final_count)
    if predictor_fallback is None:
        predictor_fallback = torch.zeros_like(final_count)
    if fallback_due_empty_mask_count is None:
        fallback_due_empty_mask_count = torch.zeros_like(final_count)
    return ActionMaskDiagnostics(
        mode=mode,
        raw_mask=raw,
        visibility_mask=visibility,
        completion_safe_mask=completion,
        mobility_risk_mask=mobility,
        final_mask=final,
        raw_count=raw_count,
        visibility_count=visibility_count,
        completion_safe_count=completion_count,
        mobility_risk_count=mobility_count,
        final_count=final_count,
        masked_action_ratio=masked_ratio,
        visibility_mask_ratio=visibility_ratio,
        completion_mask_ratio=completion_ratio,
        mobility_mask_ratio=mobility_ratio,
        fallback_due_missing_field_count=fallback_due_missing_field,
        fallback_due_missing_field_ratio=fallback_due_missing_field_ratio,
        mask_source=mask_source,
        uses_oracle_trace_mask=bool(uses_oracle_trace_mask),
        deployable=bool(deployable),
        link_lifetime_noise_std_s=float(link_lifetime_noise_std_s),
        completion_time_noise_std_s=float(completion_time_noise_std_s),
        mask_false_positive_rate=float(mask_false_positive_rate),
        mask_false_negative_rate=float(mask_false_negative_rate),
        mask_staleness_slots=int(mask_staleness_slots),
        mask_false_positive_rate_observed=mask_false_positive_rate_observed.float(),
        mask_false_negative_rate_observed=mask_false_negative_rate_observed.float(),
        predictor_fallback=predictor_fallback.float(),
        fallback_due_empty_mask_count=fallback_due_empty_mask_count.float(),
    )


def _trace_mask_field_present(trace_snapshot: TraceTopologySnapshot, field_index: int, reference: torch.Tensor) -> torch.Tensor:
    presence = getattr(trace_snapshot, "mask_field_presence", None)
    if presence is None:
        return torch.zeros(reference.shape[0], dtype=torch.bool, device=reference.device)
    presence = presence.bool()
    if presence.dim() == 1:
        return presence
    if presence.shape[0] != reference.shape[0] or presence.shape[1] <= field_index:
        raise ValueError(
            f"trace mask_field_presence shape={tuple(presence.shape)} cannot describe mask shape={tuple(reference.shape)}"
        )
    return presence[:, field_index]
