from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trisatflow.agents.hierarchical_trainer import HierarchicalTrainer
from trisatflow.config import AlgoConfig, ScenarioConfig, TrainConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _cfg(mode: str, output_dir: Path, device: str) -> TrainConfig:
    stop_gradient = mode == "shared_upper_only"
    return TrainConfig(
        total_episodes=1,
        steps_per_episode=6,
        lower_training_enabled=True,
        lower_action_mode="learned",
        log_interval=99,
        device=device,
        output_dir=str(output_dir),
        scenario=ScenarioConfig(n_leo=4, episode_len=6, seed=19, enable_gnn=False),
        algo=AlgoConfig(
            upper_algo="mappo",
            lower_algo="maddpg",
            encoder_mode=mode,
            lower_observation_mode="shared_embedding",
            stop_gradient_to_encoder_from_lower=stop_gradient,
            detach_embedding_during_action_collection=True,
            upper_update_every=1,
            lower_update_every=1,
            lower_updates_per_upper_update=1,
            freeze_upper_during_lower_update=True,
            log_gradient_diagnostics=True,
            gradient_diagnostics_interval=1,
            gnn_hidden_dim=8,
            policy_hidden_dim=16,
            ppo_epochs=1,
            minibatch_size=16,
            lower_batch_size=2,
            lower_warmup=1,
            upper_lr=1.0e-3,
            lower_lr=1.0e-3,
            critic_lr=1.0e-3,
            encoder_lr=1.0e-3,
            exploration_noise=0.02,
        ),
    )


def _run_mode(mode: str, output_root: Path, device: str) -> Dict[str, Any]:
    run_dir = output_root / mode
    trainer = HierarchicalTrainer(_cfg(mode, run_dir, device))
    history = trainer.train()
    gradient_rows = _read_csv(run_dir / "gradient_report.csv")
    last = history[-1] if history else {}
    shared_lower_grad = float(last.get("shared_encoder_grad_norm_from_lower", 0.0) or 0.0)
    separate_grad = float(last.get("separate_lower_encoder_grad_norm", 0.0) or 0.0)
    return {
        "mode": mode,
        "ok": bool(history and gradient_rows),
        "output_dir": str(run_dir),
        "shared_encoder_grad_norm_from_lower": shared_lower_grad,
        "separate_lower_encoder_grad_norm": separate_grad,
        "lower_action_sensitivity_to_upper_action": float(last.get("lower_action_sensitivity_to_upper_action", 0.0) or 0.0),
        "lower_allocator_not_conditioned_effectively": bool(last.get("lower_allocator_not_conditioned_effectively", False)),
        "lower_action_collection_embed_detached": bool(last.get("lower_action_collection_embed_detached", False)),
        "training_update_detach_semantics": str(last.get("training_update_detach_semantics", "")),
        "gradient_rows": gradient_rows,
    }


def _write_docs(path: Path, summary: Dict[str, Any]) -> None:
    shared_upper = next((m for m in summary["modes"] if m["mode"] == "shared_upper_only"), {})
    shared_joint = next((m for m in summary["modes"] if m["mode"] == "shared_joint"), {})
    text = [
        "# Encoder Training Semantics",
        "",
        "This audit separates action collection detach/no_grad from training-update detach.",
        "",
        "## Current Default",
        "",
        "`encoder_mode=shared_upper_only` is the compatibility default. In this mode the lower allocator consumes the shared embedding, but lower loss does not update the shared encoder.",
        "",
        f"- shared_upper_only lower shared-encoder grad: `{shared_upper.get('shared_encoder_grad_norm_from_lower', 0.0)}`",
        f"- shared_joint lower shared-encoder grad: `{shared_joint.get('shared_encoder_grad_norm_from_lower', 0.0)}`",
        "",
        "## Safe Paper Wording",
        "",
        "If using `shared_upper_only`:",
        "",
        "the encoder is learned by the upper actor-critic and reused as a fixed representation by the lower allocator during lower updates.",
        "",
        "Only if `shared_joint` passes the gradient diagnostic may the paper say:",
        "",
        "shared encoder is jointly trained by both decision levels.",
        "",
        "## Credit Assignment",
        "",
        "The lower actor input is `node_embedding + one_hot(upper_action)`. The lower critic input is `flatten(all_agent_embedding, upper_action_onehot, lower_action)`, so lower continuous allocation is explicitly conditioned on the upper discrete offloading decision.",
        "",
        "## Caveat",
        "",
        "The current MADDPG centralized critic is initialized with a fixed number of agents for the active environment. This is recorded in diagnostics and should not be described as agent-count invariant without a separate pooling critic implementation.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU smoke for hierarchical encoder gradient diagnostics.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "outputs" / "reviewer_repair" / "encoder_diagnostics"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = [_run_mode("shared_upper_only", output_dir, args.device), _run_mode("shared_joint", output_dir, args.device)]
    rows: List[Dict[str, Any]] = []
    for mode in modes:
        for row in mode.pop("gradient_rows", []):
            row["mode"] = mode["mode"]
            rows.append(row)
    _write_csv(output_dir / "gradient_report.csv", rows)
    summary = {
        "status": "ok",
        "device": args.device,
        "modes": modes,
        "shared_upper_only_ok": bool(modes[0]["ok"] and modes[0]["shared_encoder_grad_norm_from_lower"] == 0.0),
        "shared_joint_ok": bool(modes[1]["ok"] and modes[1]["shared_encoder_grad_norm_from_lower"] > 0.0),
        "action_collection_detach_not_training_detach": bool(
            modes[1]["lower_action_collection_embed_detached"]
            and modes[1]["shared_encoder_grad_norm_from_lower"] > 0.0
        ),
        "docs": str(REPO_ROOT / "docs" / "encoder_training_semantics.md"),
    }
    if not summary["shared_joint_ok"]:
        summary["status"] = "blocked"
        summary["blocker"] = "encoder_mode=shared_joint did not produce shared encoder gradient from lower update"
    docs_path = REPO_ROOT / "docs" / "encoder_training_semantics.md"
    try:
        _write_docs(docs_path, summary)
        summary["docs_write_status"] = "ok"
    except PermissionError as exc:
        summary["docs_write_status"] = "permission_denied"
        summary["docs_write_warning"] = str(exc)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
