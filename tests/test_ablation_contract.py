from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from trisatflow.agents import HierarchicalTrainer
from trisatflow.config import load_config
from trisatflow.experiment_contracts import (
    assert_paper_safe,
    assert_same_contract,
    contract_diff_paths,
    resolve_contract,
)
from trisatflow.models import FeatureEncoder, TemporalTopologyEncoder, TopologyEncoder


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "trisatflow" / "configs" / "paper" / "satedgesim_trace_mixed_v3_safe.yaml"
ABLATION_ROOT = REPO_ROOT / "trisatflow" / "configs" / "ablations"
REQUIRED_ABLATIONS = {
    "no_mask",
    "visibility_only",
    "completion_safe",
    "full_mask",
    "no_gnn",
    "static_gnn",
    "temporal_gnn",
    "no_cost_prior",
    "safe_observable_full",
    "safe_no_gnn",
    "safe_static_gnn",
    "safe_no_mask",
    "safe_visibility_only",
    "safe_completion_safe",
    "safe_no_lyapunov",
    "safe_no_cross_layer",
    "diagnostic_cost_prior_only",
}
DIAGNOSTIC_ONLY_ABLATIONS = {"diagnostic_cost_prior_only"}


def _payload(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    assert isinstance(data, dict)
    return data


def _spec(path: Path) -> tuple[str, list[str]]:
    data = _payload(path)
    ablation = data.get("ablation")
    assert isinstance(ablation, dict), path
    name = str(ablation.get("name", "")).strip()
    allowed = ablation.get("allowed_contract_diff_paths")
    assert isinstance(allowed, list), path
    assert all(isinstance(item, str) for item in allowed), path
    return name, list(allowed)


def test_all_required_ablations_extend_paper_safe_base() -> None:
    configs = {path.stem: path for path in ABLATION_ROOT.glob("*.yaml")}
    assert REQUIRED_ABLATIONS <= set(configs)

    for name in REQUIRED_ABLATIONS:
        path = configs[name]
        payload = _payload(path)
        spec_name, _allowed = _spec(path)
        cfg = load_config(path)

        assert spec_name == name
        assert payload["extends"] == "../paper/satedgesim_trace_mixed_v3_safe.yaml"
        assert BASE_CONFIG.resolve().as_posix() in cfg.config_source_chain
        assert cfg.config_source_chain[-1] == path.resolve().as_posix()
        if name not in DIAGNOSTIC_ONLY_ABLATIONS:
            assert_paper_safe(cfg)


def test_ablation_contract_diffs_are_whitelisted_and_do_not_change_trace_reward_or_observation() -> None:
    base_cfg = load_config(BASE_CONFIG)
    base_contract = resolve_contract(base_cfg, "stage10-test-trace-sha")

    for path in sorted(ABLATION_ROOT.glob("*.yaml")):
        name, allowed = _spec(path)
        if name in DIAGNOSTIC_ONLY_ABLATIONS:
            continue
        cfg = load_config(path)
        contract = resolve_contract(cfg, "stage10-test-trace-sha")
        diffs = contract_diff_paths(base_contract, contract)

        assert_same_contract(base_contract, contract, allowed)
        assert not [diff for diff in diffs if diff.startswith("trace.")]
        assert not [diff for diff in diffs if diff.startswith("reward.")]
        assert not [diff for diff in diffs if diff.startswith("observation.")]
        assert cfg.scenario.topology_trace_path == base_cfg.scenario.topology_trace_path
        assert cfg.scenario.success_profile == "paper_strict"
        assert cfg.observation.mode == "safe_observable"
        assert cfg.observation.include_oracle_cost is False
        assert cfg.observation.include_cost_prior_features is False
        assert cfg.reward.use_oracle_cost_components is False
        assert name == path.stem


def test_action_mask_ablations_resolve_to_target_layer_modes() -> None:
    expected = {
        "no_mask": ("none", False, False, False, False),
        "visibility_only": ("visibility", True, True, False, False),
        "completion_safe": ("completion_safe", True, True, True, False),
        "full_mask": ("full", True, True, True, True),
    }
    for name, (mode, enabled, visibility, completion, mobility) in expected.items():
        cfg = load_config(ABLATION_ROOT / f"{name}.yaml")
        assert cfg.scenario.action_mask_layer_mode == mode
        assert cfg.scenario.action_mask_enabled is enabled
        assert cfg.scenario.enable_visibility_mask is visibility
        assert cfg.scenario.enable_completion_safe_mask is completion
        assert cfg.scenario.enable_mobility_risk_mask is mobility


def test_encoder_ablations_enter_distinct_encoder_paths(tmp_path: Path) -> None:
    cases = {
        "no_gnn": FeatureEncoder,
        "static_gnn": TopologyEncoder,
        "temporal_gnn": TemporalTopologyEncoder,
    }
    for name, cls in cases.items():
        cfg = load_config(ABLATION_ROOT / f"{name}.yaml")
        cfg.device = "cpu"
        cfg.output_dir = str(tmp_path / name)
        cfg.scenario.topology_mode = "analytic"
        cfg.scenario.n_leo = 4
        cfg.scenario.episode_len = 2
        cfg.steps_per_episode = 2

        trainer = HierarchicalTrainer(cfg)

        assert cfg.model.topology_encoder == name
        assert isinstance(trainer.encoder, cls)
        if name == "temporal_gnn":
            assert cfg.model.temporal.enabled is True
        else:
            assert cfg.model.temporal.enabled is False


def test_ablation_suite_cli_rejects_legacy_partial_config_mode() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_ablation_suite.py"),
            "--config",
            str(BASE_CONFIG),
            "--smoke",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert result.returncode != 0
    assert "use --config-root" in result.stderr
