from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from trisatflow.config import load_config, save_config
from trisatflow.experiment_contracts import (
    assert_paper_safe,
    contract_sha256,
    file_sha256,
    resolve_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_SAFE_CONFIG = REPO_ROOT / "trisatflow" / "configs" / "paper" / "satedgesim_trace_mixed_v3_safe.yaml"
DEBUG_CONFIG = REPO_ROOT / "trisatflow" / "configs" / "debug" / "satedgesim_trace_mixed_v3_oracle_debug.yaml"


def test_relative_extends_deep_merges_mappings_and_replaces_lists(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    child = child_dir / "safe.yaml"
    base.write_text(
        "\n".join(
            [
                "training:",
                "  episodes: 7",
                "  steps_per_episode: 9",
                "scenario:",
                "  n_leo: 6",
                "  seed: 13",
                "reward:",
                "  mode: physical_weighted",
                "  delay: 1.0",
                "  queue: 0.2",
                "experiment:",
                "  split:",
                "    train_seeds: [1, 2]",
            ]
        ),
        encoding="utf-8",
    )
    child.write_text(
        "\n".join(
            [
                "extends: ../base.yaml",
                "scenario:",
                "  seed: 21",
                "reward:",
                "  queue: 0.4",
                "experiment:",
                "  split:",
                "    train_seeds: [3]",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(child)

    assert cfg.total_episodes == 7
    assert cfg.scenario.episode_len == 9
    assert cfg.scenario.n_leo == 6
    assert cfg.scenario.seed == 21
    assert cfg.reward.delay == 1.0
    assert cfg.reward.queue == 0.4
    assert cfg.experiment.split.train_seeds == [3]
    assert cfg.config_source_chain == [base.resolve().as_posix(), child.resolve().as_posix()]


def test_extends_cycle_detection(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("extends: b.yaml\n", encoding="utf-8")
    b.write_text("extends: a.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Cyclic config extends chain"):
        load_config(a)


def test_paper_safe_config_rejects_oracle_fields() -> None:
    safe = load_config(PAPER_SAFE_CONFIG)
    assert_paper_safe(safe)
    assert safe.observation.mode == "safe_observable"
    assert safe.observation.include_oracle_cost is False
    assert safe.observation.include_cost_prior_features is False
    assert safe.reward.use_oracle_cost_components is False

    debug = load_config(DEBUG_CONFIG)
    with pytest.raises(ValueError, match="not paper-safe"):
        assert_paper_safe(debug)


def test_debug_config_and_paper_config_contract_fingerprints_differ() -> None:
    safe = load_config(PAPER_SAFE_CONFIG)
    debug = load_config(DEBUG_CONFIG)

    safe_contract = resolve_contract(safe, file_sha256(REPO_ROOT / safe.scenario.topology_trace_path))
    debug_contract = resolve_contract(debug, file_sha256(REPO_ROOT / debug.scenario.topology_trace_path))

    assert contract_sha256(safe_contract) != contract_sha256(debug_contract)


def test_resolved_config_can_be_saved_with_source_chain(tmp_path: Path) -> None:
    cfg = load_config(PAPER_SAFE_CONFIG)
    out = tmp_path / "resolved_config.yaml"

    save_config(cfg, out)
    payload = yaml.safe_load(out.read_text(encoding="utf-8"))

    assert payload["config_source_chain"] == cfg.config_source_chain
    assert payload["observation"]["mode"] == "safe_observable"
    assert payload["reward"]["mode"] == "physical_weighted"


def test_trace_hash_change_changes_contract_fingerprint(tmp_path: Path) -> None:
    cfg = load_config(PAPER_SAFE_CONFIG)
    trace_a = tmp_path / "trace_a.jsonl"
    trace_b = tmp_path / "trace_b.jsonl"
    trace_a.write_text('{"step": 0, "visible": true}\n', encoding="utf-8")
    trace_b.write_text('{"step": 0, "visible": false}\n', encoding="utf-8")

    digest_a = contract_sha256(resolve_contract(cfg, file_sha256(trace_a)))
    digest_b = contract_sha256(resolve_contract(cfg, file_sha256(trace_b)))

    assert digest_a != digest_b


def test_contract_audit_cli_accepts_paper_safe_config() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "audit_experiment_contract.py"),
            "--config",
            str(PAPER_SAFE_CONFIG),
            "--require-paper-safe",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )

    assert "AUDIT_EXPERIMENT_CONTRACT_OK" in result.stdout
    assert "experiment_contract_sha256=" in result.stdout
