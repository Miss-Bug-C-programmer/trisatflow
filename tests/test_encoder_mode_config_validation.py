from __future__ import annotations

import pytest

from trisatflow.config import AlgoConfig, TrainConfig, load_config
from trisatflow.config_validation import validate_train_config
from trisatflow.encoder_modes import canonicalize_encoder_mode


def test_deprecated_separate_alias_is_migrated_by_loader(tmp_path) -> None:
    config = tmp_path / "legacy_encoder_alias.yaml"
    config.write_text(
        "\n".join(
            [
                "algo:",
                "  encoder_mode: separate",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.warns(UserWarning, match="migrated to 'separate_lower_encoder'"):
        cfg = load_config(config)

    assert cfg.algo.encoder_mode == "separate_lower_encoder"
    assert cfg.algo.stop_gradient_to_encoder_from_lower is False


def test_deprecated_shared_alias_is_migrated_by_loader(tmp_path) -> None:
    config = tmp_path / "ambiguous_shared_alias.yaml"
    config.write_text(
        "\n".join(
            [
                "algo:",
                "  encoder_mode: shared",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.warns(UserWarning, match="migrated to 'shared_upper_detached_lower'"):
        cfg = load_config(config)

    assert cfg.algo.encoder_mode == "shared_upper_detached_lower"
    assert cfg.algo.stop_gradient_to_encoder_from_lower is True


def test_unknown_encoder_mode_fails_instead_of_defaulting() -> None:
    with pytest.raises(ValueError, match="Unsupported algo.encoder_mode"):
        canonicalize_encoder_mode("shared_magic")


def test_direct_config_validation_requires_canonical_encoder_mode() -> None:
    cfg = TrainConfig(algo=AlgoConfig(encoder_mode="separate"))

    with pytest.raises(ValueError, match="algo.encoder_mode"):
        validate_train_config(cfg)


def test_shared_joint_config_sets_lower_encoder_trainable_flag() -> None:
    cfg = TrainConfig(algo=AlgoConfig(encoder_mode="shared_joint"))
    # Direct construction is normalized by the trainer/checkpoint path; the
    # validator accepts the canonical enum without alias migration.
    validate_train_config(cfg)
    assert cfg.algo.encoder_mode == "shared_joint"
