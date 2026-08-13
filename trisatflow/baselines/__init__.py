from trisatflow.baselines.registry import (
    ACTION_INDEX,
    ACTION_NAMES,
    ARCHITECTURE_ALLOWED_ACTIONS,
    BaselineMetadata,
    FORMAL_BASELINE_NAMES,
    LEGACY_BASELINE_ALIASES,
    apply_architecture_filter,
    baseline_metadata,
    baseline_metadata_json,
    baseline_metadata_registry,
    baseline_names,
    build_baseline_policy,
    extract_candidate_info,
    paper_ready_baseline_names,
    state_action_mask,
)
from trisatflow.baselines.offline_adapter import offline_baseline_registry


def evaluate_named_baselines(*args, **kwargs):
    from trisatflow.baselines.evaluator import evaluate_named_baselines as _impl
    return _impl(*args, **kwargs)


def evaluate_policy(*args, **kwargs):
    from trisatflow.baselines.evaluator import evaluate_policy as _impl
    return _impl(*args, **kwargs)


def write_csv(*args, **kwargs):
    from trisatflow.baselines.evaluator import write_csv as _impl
    return _impl(*args, **kwargs)


def baseline_registry(*args, **kwargs):
    return offline_baseline_registry(*args, **kwargs)


__all__ = [
    "ACTION_INDEX",
    "ACTION_NAMES",
    "ARCHITECTURE_ALLOWED_ACTIONS",
    "BaselineMetadata",
    "FORMAL_BASELINE_NAMES",
    "LEGACY_BASELINE_ALIASES",
    "apply_architecture_filter",
    "baseline_metadata",
    "baseline_metadata_json",
    "baseline_metadata_registry",
    "baseline_names",
    "build_baseline_policy",
    "extract_candidate_info",
    "paper_ready_baseline_names",
    "state_action_mask",
    "evaluate_named_baselines",
    "evaluate_policy",
    "write_csv",
    "baseline_registry",
]
