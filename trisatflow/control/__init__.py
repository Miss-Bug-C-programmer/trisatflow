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
    PlanningDescriptor,
    SMDPTransition,
)
from trisatflow.control.benefit import (
    BenefitEstimate,
    BenefitEstimator,
    ConservativeAnalyticalBenefitEstimator,
    CostToGoEstimate,
    OutcomeEstimate,
    PlannerPerformanceProfile,
)
from trisatflow.control.persistent_configuration import PersistentConfiguration
from trisatflow.control.scope import ConstraintViolation, ReconfigurationScope, ScopeGenerator, ViolationProvenance
from trisatflow.control.viability import ConservativeViabilityEstimator, SoftPerformanceRisk, ViabilityCertificate, ViabilityReport
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
    "BenefitEstimate",
    "BenefitEstimator",
    "ConservativeAnalyticalBenefitEstimator",
    "CostToGoEstimate",
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
    "PlanningDescriptor",
    "PlannerState",
    "PlanningBudget",
    "ReconfigurationScope",
    "ScopeGenerator",
    "ViolationProvenance",
    "ConstraintViolation",
    "PostDelayRevalidator",
    "SMDPTransition",
    "OutcomeEstimate",
    "PlannerPerformanceProfile",
    "ViabilityReport",
    "ViabilityCertificate",
    "SoftPerformanceRisk",
]
