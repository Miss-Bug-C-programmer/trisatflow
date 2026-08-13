from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping
import warnings


CANONICAL_ENCODER_MODES = {
    "shared_upper_detached_lower",
    "shared_joint",
    "separate_lower_encoder",
    "shared_frozen",
}

DEPRECATED_ENCODER_MODE_ALIASES = {
    "shared_upper_only": "shared_upper_detached_lower",
    "shared_detached": "shared_upper_detached_lower",
    "shared": "shared_upper_detached_lower",
    "separate": "separate_lower_encoder",
}


@dataclass(frozen=True)
class EncoderModeSemantics:
    encoder_mode: str
    lower_embed_detached: bool
    lower_encoder_trainable: bool
    upper_encoder_trainable: bool
    uses_separate_lower_encoder: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonicalize_encoder_mode(mode: object, *, warn: bool = False) -> str:
    raw = str(mode or "shared_upper_detached_lower").strip().lower()
    if raw in CANONICAL_ENCODER_MODES:
        return raw
    if raw in DEPRECATED_ENCODER_MODE_ALIASES:
        canonical = DEPRECATED_ENCODER_MODE_ALIASES[raw]
        if warn:
            warnings.warn(
                f"[DEPRECATED] algo.encoder_mode={raw!r} has been migrated to {canonical!r}; "
                "use a canonical encoder mode in new configs",
                UserWarning,
                stacklevel=3,
            )
        return canonical
    raise ValueError(
        f"Unsupported algo.encoder_mode={raw!r}; expected one of {sorted(CANONICAL_ENCODER_MODES)}"
    )


def encoder_mode_semantics(mode: object) -> EncoderModeSemantics:
    canonical = canonicalize_encoder_mode(mode)
    if canonical == "shared_upper_detached_lower":
        return EncoderModeSemantics(
            encoder_mode=canonical,
            lower_embed_detached=True,
            lower_encoder_trainable=False,
            upper_encoder_trainable=True,
            uses_separate_lower_encoder=False,
        )
    if canonical == "shared_joint":
        return EncoderModeSemantics(
            encoder_mode=canonical,
            lower_embed_detached=False,
            lower_encoder_trainable=True,
            upper_encoder_trainable=True,
            uses_separate_lower_encoder=False,
        )
    if canonical == "separate_lower_encoder":
        return EncoderModeSemantics(
            encoder_mode=canonical,
            lower_embed_detached=False,
            lower_encoder_trainable=True,
            upper_encoder_trainable=True,
            uses_separate_lower_encoder=True,
        )
    if canonical == "shared_frozen":
        return EncoderModeSemantics(
            encoder_mode=canonical,
            lower_embed_detached=True,
            lower_encoder_trainable=False,
            upper_encoder_trainable=False,
            uses_separate_lower_encoder=False,
        )
    raise AssertionError(f"unreachable encoder mode: {canonical}")


def apply_encoder_mode_to_algo(algo: Any, *, warn: bool = False) -> EncoderModeSemantics:
    semantics = encoder_mode_semantics(canonicalize_encoder_mode(getattr(algo, "encoder_mode", None), warn=warn))
    algo.encoder_mode = semantics.encoder_mode
    algo.stop_gradient_to_encoder_from_lower = bool(semantics.lower_embed_detached)
    return semantics


def checkpoint_encoder_metadata(algo: Any) -> dict[str, Any]:
    return encoder_mode_semantics(getattr(algo, "encoder_mode", None)).to_dict()


def validate_checkpoint_encoder_metadata(
    payload: Mapping[str, Any],
    algo: Any,
    *,
    formal: bool,
) -> dict[str, Any]:
    expected = checkpoint_encoder_metadata(algo)
    actual = payload.get("encoder_mode_metadata")
    if actual is None:
        if formal:
            raise ValueError(
                "formal eval requires checkpoint encoder_mode_metadata; "
                "re-save the checkpoint with encoder mode metadata"
            )
        return expected
    if not isinstance(actual, Mapping):
        raise ValueError("checkpoint encoder_mode_metadata must be a mapping")
    missing = [key for key in expected if key not in actual]
    if missing:
        raise ValueError(f"checkpoint encoder_mode_metadata missing required field(s): {missing}")
    conflicts = {
        key: {"checkpoint": actual.get(key), "config": expected[key]}
        for key in expected
        if actual.get(key) != expected[key]
    }
    if conflicts:
        raise ValueError(
            "checkpoint encoder_mode_metadata conflicts with config: "
            f"{conflicts}"
        )
    return dict(actual)
