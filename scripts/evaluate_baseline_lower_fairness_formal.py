from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_baseline_lower_fairness import (  # noqa: E402
    _parse_values,
    _write_outputs,
    evaluate_one,
)
from trisatflow.config import load_config  # noqa: E402


def _parse_ints(text: str, default: list[int]) -> list[int]:
    values = [int(item.strip()) for item in str(text or "").split(",") if item.strip()]
    return values or list(default)


def _split_names(text: str) -> list[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal rule-baseline lower allocator fairness runner.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--baselines", default="geo_only,ground_only,random_visible")
    parser.add_argument("--lower-allocator", required=True, choices=["neutral", "same_learned", "optimized_greedy", "oracle_grid"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--neutral-values", default="", help="Allocator order: bandwidth_share,tx_power_ratio,cpu_share.")
    parser.add_argument("--train-seeds", default="13,17,23")
    parser.add_argument("--eval-seeds", default="101,103,107")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true", help="Run a bounded local smoke without formal claims.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    run_mode = "smoke" if bool(args.smoke) else "formal"
    if run_mode == "smoke":
        out_text = output_dir.name.lower()
        if "formal" in out_text or "paper_ready" in out_text:
            raise ValueError("smoke fairness runner refuses to write into formal or paper_ready output directories")
        if int(args.episodes) > 2 or int(args.steps) > 8:
            raise ValueError("--smoke lower fairness run requires episodes<=2 and steps<=8")

    cfg = load_config(args.config)
    baselines = _split_names(args.baselines)
    train_seeds = _parse_ints(args.train_seeds, [13])
    eval_seeds = _parse_ints(args.eval_seeds, [101])
    if run_mode == "smoke":
        train_seeds = train_seeds[:1]
        eval_seeds = eval_seeds[:1]
    neutral_values = _parse_values(args.neutral_values)

    rows = []
    for train_seed in train_seeds:
        for eval_seed in eval_seeds:
            for baseline in baselines:
                row = evaluate_one(
                    cfg=load_config(args.config),
                    baseline_name=baseline,
                    lower_allocator_name=args.lower_allocator,
                    checkpoint=args.checkpoint or None,
                    neutral_values=neutral_values,
                    episodes=int(args.episodes),
                    steps=int(args.steps),
                    device=str(args.device),
                    seed=int(eval_seed),
                    formal=run_mode == "formal",
                )
                row["train_seed"] = int(train_seed)
                row["eval_seed"] = int(eval_seed)
                row["run_mode"] = run_mode
                rows.append(row)

    _write_outputs(
        rows,
        output_dir,
        run_mode=run_mode,
        cfg=cfg,
        num_training_seeds=len(train_seeds),
        num_eval_seeds=len(eval_seeds),
        update_root=False,
    )
    print(json.dumps({"run_mode": run_mode, "rows": len(rows), "output_dir": str(output_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
