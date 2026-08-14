"""Decision-plane control primitives for endogenous replanning.

The legacy slotwise environment and trainers remain in their original modules.
This package adds the outer control loop without changing the inner MARL action
space.
"""

from trisatflow.control.types import (
    ClockState,
    ControllerContext,
    FeasibilityStatus,
    MonitorAcquisitionMetadata,
    MonitorState,
    PlannerCapabilities,
    PlannerState,
    PlanningBudget,
    PlannerFidelity,
    PlannerResult,
    SMDPTransition,
)
from trisatflow.control.persistent_configuration import PersistentConfiguration
from trisatflow.control.scope import ReconfigurationScope, ScopeGenerator
from trisatflow.control.viability import ConservativeViabilityEstimator, ViabilityReport
from trisatflow.control.controller import ControlDecision, EndogenousReplanningController
from trisatflow.control.config import ControllerConfig
from trisatflow.control.decision_cost import DecisionCostBreakdown
from trisatflow.control.decision_delay import DecisionDelayBreakdown, DecisionDelayModel, PostDelayRevalidator
from trisatflow.control.monitor import CheapConfigurationMonitor

__all__ = [
    "ClockState",
    "CheapConfigurationMonitor",
    "ConservativeViabilityEstimator",
    "ControlDecision",
    "ControllerContext",
    "ControllerConfig",
    "DecisionCostBreakdown",
    "DecisionDelayBreakdown",
    "DecisionDelayModel",
    "EndogenousReplanningController",
    "FeasibilityStatus",
    "MonitorAcquisitionMetadata",
    "MonitorState",
    "PersistentConfiguration",
    "PlannerCapabilities",
    "PlannerFidelity",
    "PlannerResult",
    "PlannerState",
    "PlanningBudget",
    "ReconfigurationScope",
    "ScopeGenerator",
    "PostDelayRevalidator",
    "SMDPTransition",
    "ViabilityReport",
]
