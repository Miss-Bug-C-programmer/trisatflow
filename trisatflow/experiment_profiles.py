from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class ExperimentProfile:
    profile_name: str
    action_mask_mode: str
    success_profile: str
    mobility_risk_enabled: bool
    intended_use: str
    mobility_aware_profile_status: str = "n/a"
    min_link_survival_margin_sec: float = 0.0


def profile_registry() -> Dict[str, ExperimentProfile]:
    return {
        "mobility_aware_main": ExperimentProfile(
            profile_name="mobility_aware_main",
            action_mask_mode="completion_safe",
            success_profile="paper_strict",
            mobility_risk_enabled=True,
            intended_use="main_evaluation_candidate",
            mobility_aware_profile_status="completion_aware",
            min_link_survival_margin_sec=1.0,
        ),
        "mobility_stress_visible": ExperimentProfile(
            profile_name="mobility_stress_visible",
            action_mask_mode="visible_only",
            success_profile="paper_strict",
            mobility_risk_enabled=True,
            intended_use="robustness_stress_profile",
            mobility_aware_profile_status="n/a",
        ),
        "preflight_lenient": ExperimentProfile(
            profile_name="preflight_lenient",
            action_mask_mode="visible_only",
            success_profile="preflight_lenient",
            mobility_risk_enabled=True,
            intended_use="engineering_integration_only",
            mobility_aware_profile_status="n/a",
        ),
        "mobility_aware_main_v1": ExperimentProfile(
            profile_name="mobility_aware_main_v1",
            action_mask_mode="completion_safe",
            success_profile="paper_strict",
            mobility_risk_enabled=True,
            intended_use="v1_main_evaluation_candidate",
            mobility_aware_profile_status="completion_aware",
            min_link_survival_margin_sec=1.0,
        ),
        "mobility_stress_visible_v1": ExperimentProfile(
            profile_name="mobility_stress_visible_v1",
            action_mask_mode="visible_only",
            success_profile="paper_strict",
            mobility_risk_enabled=True,
            intended_use="v1_robustness_stress_profile",
            mobility_aware_profile_status="n/a",
        ),
        "preflight_lenient_v1": ExperimentProfile(
            profile_name="preflight_lenient_v1",
            action_mask_mode="visible_only",
            success_profile="preflight_lenient",
            mobility_risk_enabled=True,
            intended_use="v1_engineering_integration_only",
            mobility_aware_profile_status="n/a",
        ),
    }


def get_profile(name: str) -> ExperimentProfile:
    key = str(name or "").strip().lower()
    reg = profile_registry()
    if key not in reg:
        raise ValueError(f"unsupported profile={name!r}; choose from {sorted(reg)}")
    return reg[key]


def profile_metadata(name: str) -> Dict[str, Any]:
    return asdict(get_profile(name))
