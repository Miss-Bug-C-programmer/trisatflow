from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class AlgorithmChoice:
    name: str
    action_type: str
    benchmarl_module: str
    implemented_in_lightweight_trainer: bool
    reviewer_note: str


def supported_algorithm_matrix() -> Dict[str, List[AlgorithmChoice]]:
    """Algorithm candidates aligned with BenchMARL's available families.

    The dependency-light TriSatFlow trainer now implements multiple algorithm
    families so that algorithm-combination sweeps can run without a full
    TorchRL/TensorDict install. The class names map to BenchMARL 1.5.1 modules
    included in this repository.
    """

    return {
        "upper_discrete_offloading": [
            AlgorithmChoice(
                name="mappo",
                action_type="discrete",
                benchmarl_module="benchmarl.algorithms.mappo.Mappo",
                implemented_in_lightweight_trainer=True,
                reviewer_note="CTDE policy-gradient baseline; default upper-layer choice.",
            ),
            AlgorithmChoice(
                name="ippo",
                action_type="discrete",
                benchmarl_module="benchmarl.algorithms.ippo.Ippo",
                implemented_in_lightweight_trainer=True,
                reviewer_note="Independent PPO ablation for decentralized offloading policies.",
            ),
            AlgorithmChoice(
                name="iql",
                action_type="discrete",
                benchmarl_module="benchmarl.algorithms.iql.Iql",
                implemented_in_lightweight_trainer=True,
                reviewer_note="Independent value-learning ablation for discrete offloading.",
            ),
            AlgorithmChoice(
                name="vdn",
                action_type="discrete",
                benchmarl_module="benchmarl.algorithms.vdn.Vdn",
                implemented_in_lightweight_trainer=True,
                reviewer_note="Value-decomposition baseline for cooperative offloading.",
            ),
            AlgorithmChoice(
                name="qmix",
                action_type="discrete",
                benchmarl_module="benchmarl.algorithms.qmix.Qmix",
                implemented_in_lightweight_trainer=True,
                reviewer_note="Monotonic value-mixing baseline for cooperative offloading.",
            ),
        ],
        "lower_continuous_resource": [
            AlgorithmChoice(
                name="maddpg",
                action_type="continuous",
                benchmarl_module="benchmarl.algorithms.maddpg.Maddpg",
                implemented_in_lightweight_trainer=True,
                reviewer_note="CTDE deterministic continuous control; default lower-layer choice.",
            ),
            AlgorithmChoice(
                name="iddpg",
                action_type="continuous",
                benchmarl_module="benchmarl.algorithms.iddpg.Iddpg",
                implemented_in_lightweight_trainer=True,
                reviewer_note="Independent deterministic continuous-control ablation.",
            ),
            AlgorithmChoice(
                name="masac",
                action_type="continuous",
                benchmarl_module="benchmarl.algorithms.masac.Masac",
                implemented_in_lightweight_trainer=True,
                reviewer_note="Entropy-regularized CTDE continuous control.",
            ),
            AlgorithmChoice(
                name="isac",
                action_type="continuous",
                benchmarl_module="benchmarl.algorithms.isac.Isac",
                implemented_in_lightweight_trainer=True,
                reviewer_note="Independent entropy-regularized continuous-control ablation.",
            ),
        ],
        "flat_hybrid_learning_baselines": [
            AlgorithmChoice(
                name="flat_ppo",
                action_type="hybrid_discrete_continuous",
                benchmarl_module="benchmarl.algorithms.ippo.Ippo",
                implemented_in_lightweight_trainer=True,
                reviewer_note="Single-level PPO baseline with discrete offloading and continuous resource heads.",
            ),
            AlgorithmChoice(
                name="flat_mappo",
                action_type="hybrid_discrete_continuous",
                benchmarl_module="benchmarl.algorithms.mappo.Mappo",
                implemented_in_lightweight_trainer=True,
                reviewer_note="Single-level MAPPO-style baseline with centralized value and flat hybrid actor.",
            ),
            AlgorithmChoice(
                name="hierarchical_no_gnn",
                action_type="hierarchical_hybrid",
                benchmarl_module="trisatflow.agents.hierarchical_trainer.HierarchicalTrainer",
                implemented_in_lightweight_trainer=True,
                reviewer_note="TriSatFlow hierarchy with message passing removed; controls for topology encoder value.",
            ),
        ],
    }


def upper_algorithm_names() -> List[str]:
    return [item.name for item in supported_algorithm_matrix()["upper_discrete_offloading"]]


def lower_algorithm_names() -> List[str]:
    return [item.name for item in supported_algorithm_matrix()["lower_continuous_resource"]]


def learning_baseline_names() -> List[str]:
    return [item.name for item in supported_algorithm_matrix()["flat_hybrid_learning_baselines"]]


def validate_algorithm_choice(upper: str, lower: str) -> None:
    upper_names = set(upper_algorithm_names())
    lower_names = set(lower_algorithm_names())
    if upper not in upper_names:
        raise ValueError(f"Unsupported upper algorithm {upper!r}; choose from {sorted(upper_names)}")
    if lower not in lower_names:
        raise ValueError(f"Unsupported lower algorithm {lower!r}; choose from {sorted(lower_names)}")
