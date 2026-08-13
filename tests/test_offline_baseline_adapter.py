from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from trisatflow.baselines.offline_adapter import (
    FORMAL_BASELINE_NAMES,
    OfflineBaselineAdapter,
    build_offline_baseline_policy,
)
from trisatflow.config import load_config
from trisatflow.envs import GeoLeoGroundEnv


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_SAFE_CONFIG = REPO_ROOT / "trisatflow" / "configs" / "paper" / "satedgesim_trace_mixed_v3_safe.yaml"


def _small_env() -> GeoLeoGroundEnv:
    cfg = load_config(PAPER_SAFE_CONFIG)
    cfg.scenario.n_leo = 4
    cfg.scenario.episode_len = 3
    cfg.scenario.seed = 202
    return GeoLeoGroundEnv(cfg.scenario, cfg.reward, "cpu")


def _assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            text = str(key).lower()
            assert "oracle" not in text
            assert "future" not in text
            assert "hindsight" not in text
            _assert_no_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_forbidden_keys(item)


def test_formal_policies_select_only_final_mask_actions() -> None:
    env = _small_env()
    env.reset(rule_baseline_observation=True)

    for name in FORMAL_BASELINE_NAMES:
        adapter = OfflineBaselineAdapter(build_offline_baseline_policy(name), rng=random.Random(7))
        batch = adapter.select_actions(env)
        contexts = env.baseline_contexts()
        for action, ctx in zip(batch.upper_action.tolist(), contexts):
            assert ctx["mask"][int(action)] is True


def test_static_unavailable_preference_falls_back_and_counts() -> None:
    cfg = load_config(PAPER_SAFE_CONFIG)
    cfg.scenario.n_leo = 4
    cfg.scenario.episode_len = 2
    cfg.scenario.enable_geo = False
    env = GeoLeoGroundEnv(cfg.scenario, cfg.reward, "cpu")
    env.reset(rule_baseline_observation=True)

    adapter = OfflineBaselineAdapter(build_offline_baseline_policy("geo_only"), rng=random.Random(11))
    batch = adapter.select_actions(env)

    assert all(info["requested_action"] == 2 for info in batch.decision_info)
    assert all(info["fallback_used"] for info in batch.decision_info)
    assert all(info["invalid_attempt"] for info in batch.decision_info)
    assert adapter.stats.fallback_count == cfg.scenario.n_leo
    assert adapter.stats.invalid_attempt_count == cfg.scenario.n_leo


def test_random_visible_does_not_select_masked_action() -> None:
    cfg = load_config(PAPER_SAFE_CONFIG)
    cfg.scenario.n_leo = 4
    cfg.scenario.episode_len = 2
    cfg.scenario.action_space_architecture = "only_leo"
    env = GeoLeoGroundEnv(cfg.scenario, cfg.reward, "cpu")
    env.reset(rule_baseline_observation=True)

    adapter = OfflineBaselineAdapter(build_offline_baseline_policy("random_visible"), rng=random.Random(123))
    for _ in range(10):
        batch = adapter.select_actions(env)
        contexts = env.baseline_contexts()
        for action, ctx in zip(batch.upper_action.tolist(), contexts):
            assert ctx["mask"][int(action)] is True
            assert int(action) in {0, 1}


def test_baseline_contexts_exclude_oracle_future_hindsight_fields() -> None:
    env = _small_env()
    env.reset(rule_baseline_observation=True)

    contexts = env.baseline_contexts()

    assert contexts
    for ctx in contexts:
        assert set(ctx) == {"obs", "state", "mask", "candidate_info"}
        assert sorted(ctx["candidate_info"]) == [0, 1, 2, 3]
        _assert_no_forbidden_keys(ctx["state"])
        _assert_no_forbidden_keys(ctx["candidate_info"])


def test_baseline_manifest_matches_rl_smoke_manifest_fingerprint(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    baseline_dir = tmp_path / "baselines"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "smoke_test.py"),
            "--config",
            str(PAPER_SAFE_CONFIG),
            "--episodes",
            "1",
            "--steps",
            "2",
            "--n-leo",
            "2",
            "--device",
            "cpu",
            "--output-dir",
            str(train_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=120,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_rule_baselines.py"),
            "--config",
            str(PAPER_SAFE_CONFIG),
            "--baselines",
            "local_only",
            "--seeds",
            "202",
            "--episodes",
            "1",
            "--steps",
            "2",
            "--n-leo",
            "2",
            "--device",
            "cpu",
            "--output-dir",
            str(baseline_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=120,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "audit_experiment_contract.py"),
            "--compare",
            str(train_dir / "manifest.json"),
            str(baseline_dir / "manifest.json"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )

    train_manifest = json.loads((train_dir / "manifest.json").read_text(encoding="utf-8"))
    baseline_manifest = json.loads((baseline_dir / "manifest.json").read_text(encoding="utf-8"))
    assert train_manifest["experiment_contract_sha256"] == baseline_manifest["experiment_contract_sha256"]
