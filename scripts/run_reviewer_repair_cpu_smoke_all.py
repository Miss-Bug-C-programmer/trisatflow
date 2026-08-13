"""Run reviewer-repair CPU smoke checks and write stage-level evidence.

This is an integration auditor, not a paper experiment runner. A stage can be
passed, failed, blocked, skipped, or not_implemented; blocked stages are kept
explicit so downstream claim guards cannot silently treat them as evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
SATEDGESIM_ROOT = WORKSPACE_ROOT / "satedgeSimv2"
PY = sys.executable


@dataclass
class CommandSpec:
    args: list[str]
    cwd: Path = REPO_ROOT
    timeout_s: int = 240
    allow_failure: bool = False

    @property
    def display(self) -> str:
        return " ".join(str(a) for a in self.args)


@dataclass
class StageSpec:
    stage_id: str
    stage_name: str
    required_for_paper_claim: str
    commands: list[CommandSpec]
    output_dir: Path
    tests_run: list[str] = field(default_factory=list)
    claim_allowed: str = ""
    claim_forbidden: str = ""
    next_action: str = ""
    post_check: Callable[[dict[str, Any]], None] | None = None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive audit path
        return {"_json_error": str(exc)}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_command(cmd: CommandSpec, log_path: Path, env: dict[str, str]) -> dict[str, Any]:
    started = time.time()
    result: dict[str, Any] = {
        "command": cmd.display,
        "cwd": str(cmd.cwd),
        "timeout_s": cmd.timeout_s,
        "allow_failure": cmd.allow_failure,
        "log_path": str(log_path),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            cmd.args,
            cwd=str(cmd.cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=cmd.timeout_s,
        )
        elapsed = time.time() - started
        log_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
        result.update(
            {
                "returncode": proc.returncode,
                "elapsed_s": round(elapsed, 3),
                "ok": proc.returncode == 0 or cmd.allow_failure,
            }
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - started
        output = (exc.stdout or "") + "\n[TIMEOUT]\n"
        log_path.write_text(output, encoding="utf-8", errors="replace")
        result.update({"returncode": None, "elapsed_s": round(elapsed, 3), "ok": False, "timeout": True})
    except FileNotFoundError as exc:
        result.update({"returncode": None, "elapsed_s": 0.0, "ok": False, "exception": str(exc)})
        log_path.write_text(str(exc), encoding="utf-8", errors="replace")
    return result


def _trace_post_check(record: dict[str, Any]) -> None:
    manifest = _load_json(REPO_ROOT / "outputs" / "reviewer_repair" / "trace_stress" / "manifests" / "manifest_build_summary.json")
    audit = _load_json(REPO_ROOT / "outputs" / "reviewer_repair" / "trace_stress" / "audit" / "audit_summary.json")
    blockers: list[str] = []
    manifest_status = str(manifest.get("manifest_build_status") or manifest.get("status") or "")
    audit_status = str(audit.get("audit_status") or audit.get("status") or "")
    leakage_risk = str(audit.get("leakage_risk") or "")
    if "failed" in manifest_status or "incomplete" in manifest_status:
        blockers.append(f"manifest_build_status={manifest_status}")
    if "failed" in audit_status or "incomplete" in audit_status or leakage_risk == "high":
        blockers.append(f"audit_status={audit_status or 'missing'} leakage_risk={leakage_risk or 'unknown'}")
    if blockers:
        record["status"] = "blocked"
        record["blockers"] = blockers
        record["claim_allowed"] = "trace tooling and scale-16 stress smoke exist, but split/leakage audit is blocked by manifest metadata quality"
        record["claim_forbidden"] = "constellation-size transfer or leakage-free trace split claims"
        record["next_action"] = "Complete trace metadata manifest, rerun leakage audit, then run 16/32/64 transfer stress."


def _satedgesim_binding_post_check(record: dict[str, Any]) -> None:
    command_results = record.get("command_results") or []
    for result in command_results:
        exception = str(result.get("exception") or "")
        log_path = Path(str(result.get("log_path") or ""))
        log_text = ""
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if "No such file" in exception or "WinError 2" in exception or "找不到" in log_text:
            record["status"] = "blocked"
            record["blockers"] = ["Maven executable is not available in this shell; Java binding smoke could not be recompiled here"]
            record["exception"] = ""
            record["claim_allowed"] = "Existing SatEdgeSim estimator-bound code and Python semantic smoke remain auditable, but this integration run did not recompile Java."
            record["claim_forbidden"] = "Native scheduler binding or full hybrid closed-loop claim from this final CPU smoke."
            record["next_action"] = "Install or expose Maven on PATH, then rerun P11R and RlResourceBindingSmoke."
            return


def _stage_specs(device: str, output_dir: Path) -> list[StageSpec]:
    out = REPO_ROOT / "outputs" / "reviewer_repair"
    pytest = [PY, "-m", "pytest"]
    def mvn_args(*args: str) -> list[str]:
        if os.name == "nt":
            return [os.environ.get("ComSpec", "cmd.exe"), "/c", "mvn", *args]
        return ["mvn", *args]

    return [
        StageSpec(
            "P0",
            "Prompt 0 audit smoke",
            "Reviewer repair baseline audit evidence",
            [CommandSpec([PY, "scripts/run_cpu_smoke_audit.py"])],
            out / "audit_smoke",
            ["scripts/run_cpu_smoke_audit.py"],
            "Audit smoke can be cited as code-path inventory evidence only.",
            "Formal validation or performance claims.",
            "Keep docs/reviewer_repair_audit.md updated as later repairs change semantics.",
        ),
        StageSpec(
            "P1",
            "Physical model smoke",
            "R2/R12 physical unit semantics",
            [
                CommandSpec(pytest + ["tests/test_dimensional_model.py", "tests/test_physical_metric_schema.py", "-q"]),
                CommandSpec([PY, "scripts/run_cpu_smoke_physical_model.py"]),
            ],
            out / "physical_model",
            ["tests/test_dimensional_model.py", "tests/test_physical_metric_schema.py"],
            "Dimensioned physical-mode smoke and schema evidence.",
            "Claiming normalized and physical costs are directly comparable across scenario profiles.",
            "Run full physical-vs-normalized ranking audit on final scenarios.",
        ),
        StageSpec(
            "P2",
            "Lyapunov/DPP smoke",
            "R3/R4 Lyapunov semantics and stronger DPP baseline",
            [
                CommandSpec(pytest + ["tests/test_lyapunov_diagnostics.py", "tests/test_optimized_dpp_baseline.py", "-q"]),
                CommandSpec([PY, "scripts/run_cpu_smoke_lyapunov_dpp.py"]),
            ],
            out / "lyapunov_dpp",
            ["tests/test_lyapunov_diagnostics.py", "tests/test_optimized_dpp_baseline.py"],
            "Lyapunov-inspired queue regularizer and optimized DPP smoke.",
            "Queue-stability theorem or finite-buffer boundedness proof.",
            "If theory is desired, add a formal drift bound; otherwise keep paper wording as reward shaping.",
        ),
        StageSpec(
            "P3",
            "Lower allocator fairness smoke",
            "R5/R8 rule baseline lower-resource fairness",
            [
                CommandSpec(pytest + ["tests/test_lower_allocator_fairness.py", "-q"]),
                CommandSpec(
                    [
                        PY,
                        "scripts/evaluate_baseline_lower_fairness.py",
                        "--lower-allocator",
                        "neutral",
                        "--episodes",
                        "1",
                        "--steps",
                        "4",
                        "--device",
                        device,
                        "--output-dir",
                        str(out / "lower_fairness" / "neutral"),
                    ]
                ),
                CommandSpec(
                    [
                        PY,
                        "scripts/evaluate_baseline_lower_fairness.py",
                        "--lower-allocator",
                        "optimized_greedy",
                        "--episodes",
                        "1",
                        "--steps",
                        "4",
                        "--device",
                        device,
                        "--output-dir",
                        str(out / "lower_fairness" / "optimized_greedy"),
                    ]
                ),
            ],
            out / "lower_fairness",
            ["tests/test_lower_allocator_fairness.py"],
            "Rule upper policies can be compared under explicit lower allocator controls.",
            "Attributing all Table 4 gaps to upper offloading decisions without lower allocator controls.",
            "Run Table 4b rule-upper by lower-allocator matrix with full seeds.",
        ),
        StageSpec(
            "P4",
            "Safe observable ablation smoke",
            "R10 cost-prior confound control",
            [
                CommandSpec(pytest + ["tests/test_safe_observable_ablation.py", "-q"]),
                CommandSpec(
                    [
                        PY,
                        "scripts/run_safe_ablation_suite.py",
                        "--episodes",
                        "1",
                        "--steps",
                        "4",
                        "--device",
                        device,
                        "--variants",
                        "safe_observable_full,safe_no_mask,safe_no_gnn,safe_no_lyapunov",
                        "--output-dir",
                        str(out / "safe_ablation"),
                    ]
                ),
                CommandSpec(
                    [
                        PY,
                        "scripts/plot_safe_ablation_pareto.py",
                        "--input",
                        str(out / "safe_ablation" / "summary.json"),
                        "--output-dir",
                        str(out / "safe_ablation" / "figures"),
                    ]
                ),
            ],
            out / "safe_ablation",
            ["tests/test_safe_observable_ablation.py"],
            "Deployable safe_observable ablation smoke and cost-safety Pareto guard.",
            "Using diagnostic cost-prior ablation as the main deployable ablation.",
            "Rerun Figure 10 matrix under safe_observable with full training seeds.",
        ),
        StageSpec(
            "P5",
            "Mask source/noise smoke",
            "R14 mask deployability and prediction stress",
            [
                CommandSpec(pytest + ["tests/test_mask_source_and_noise.py", "-q"]),
                CommandSpec(
                    [
                        PY,
                        "scripts/run_mask_noise_stress.py",
                        "--mask-source",
                        "predicted",
                        "--noise-levels",
                        "0,1.0",
                        "--episodes",
                        "1",
                        "--steps",
                        "4",
                        "--device",
                        device,
                        "--output-dir",
                        str(out / "mask_noise"),
                    ]
                ),
            ],
            out / "mask_noise",
            ["tests/test_mask_source_and_noise.py"],
            "Predicted/measured/oracle mask semantics are separated in metadata.",
            "Treating oracle_trace masks as deployable main experiments.",
            "Run predicted mask noise/staleness sweeps on full scenarios.",
        ),
        StageSpec(
            "P6",
            "SatEdgeSim metric semantics smoke",
            "R8/R9 receipt, scheduling, completion, and energy semantics",
            [
                CommandSpec(pytest + ["tests/test_satedgesim_metric_semantics.py", "-q"]),
                CommandSpec([PY, "scripts/run_cpu_smoke_satedgesim_summary_semantics.py", "--output-dir", str(out / "satedgesim_semantics")]),
            ],
            out / "satedgesim_semantics",
            ["tests/test_satedgesim_metric_semantics.py"],
            "Candidate/estimator replay summaries distinguish receipt, scheduling, completion, and energy source.",
            "Calling receipt acceptance task success or using unknown energy source for energy claims.",
            "Use completion receipts or SimLog joins for task success claims.",
        ),
        StageSpec(
            "P7R",
            "Statistics protocol smoke",
            "R6 checkpoint-level statistical unit and claim guard",
            [
                CommandSpec(pytest + ["tests/test_statistics_protocol.py", "-q"]),
                CommandSpec([PY, "scripts/run_cpu_smoke_statistics.py", "--output-dir", str(out / "statistics")]),
            ],
            out / "statistics",
            ["tests/test_statistics_protocol.py"],
            "Holm-corrected checkpoint-level statistics and automatic claim downgrade.",
            "Treating test_seed/online_seed as independent training samples.",
            "Run full 8-10 checkpoint seed protocol before significance claims.",
        ),
        StageSpec(
            "P8R",
            "Encoder diagnostics smoke",
            "R13 hierarchical encoder gradient semantics",
            [
                CommandSpec(pytest + ["tests/test_encoder_gradient_diagnostics.py", "-q"]),
                CommandSpec([PY, "scripts/run_cpu_smoke_encoder_diagnostics.py", "--device", device, "--output-dir", str(out / "encoder_diagnostics")]),
            ],
            out / "encoder_diagnostics",
            ["tests/test_encoder_gradient_diagnostics.py"],
            "Upper/lower gradient-path and cadence semantics are auditable.",
            "Claiming shared_joint encoder training unless lower gradient diagnostics pass.",
            "Keep paper wording aligned with actual encoder_mode used in full experiments.",
        ),
        StageSpec(
            "P9R",
            "Trace split/leakage and stress smoke",
            "R15 trace governance and transfer/stress claims",
            [
                CommandSpec(pytest + ["tests/test_trace_split_audit.py", "tests/test_stress_config_smoke.py", "-q"]),
                CommandSpec([PY, "scripts/build_trace_manifest.py", "--project-root", ".", "--output-dir", str(out / "trace_stress" / "manifests")], allow_failure=True),
                CommandSpec(
                    [
                        PY,
                        "scripts/audit_trace_splits.py",
                        "--manifest-dir",
                        str(out / "trace_stress" / "manifests"),
                        "--output-dir",
                        str(out / "trace_stress" / "audit"),
                    ],
                    allow_failure=True,
                ),
                CommandSpec(
                    [
                        PY,
                        "scripts/run_stress_suite.py",
                        "--stress-configs",
                        "trisatflow/configs/stress/scale_16.yaml",
                        "--policy",
                        "random_visible",
                        "--episodes",
                        "1",
                        "--steps",
                        "4",
                        "--device",
                        device,
                        "--output-dir",
                        str(out / "trace_stress" / "stress_smoke"),
                    ]
                ),
            ],
            out / "trace_stress",
            ["tests/test_trace_split_audit.py", "tests/test_stress_config_smoke.py"],
            "Stress harness exists; trace manifest audit must be completed before transfer claims.",
            "Leakage-free split or constellation transfer claims while manifest audit is blocked.",
            "Complete trace metadata and run 16/32/64 transfer experiments.",
            post_check=_trace_post_check,
        ),
        StageSpec(
            "P10R",
            "Strong baseline tiny training and oracle smoke",
            "R4/Q1 strong hybrid baselines and oracle gap",
            [
                CommandSpec(pytest + ["tests/test_strong_baseline_training.py", "tests/test_small_scale_oracle.py", "-q"], timeout_s=300),
                CommandSpec(
                    [
                        PY,
                        "scripts/train_strong_baseline_tiny.py",
                        "--baseline",
                        "pdqn_hybrid",
                        "--episodes",
                        "2",
                        "--steps",
                        "4",
                        "--n-leo",
                        "4",
                        "--device",
                        device,
                        "--output-dir",
                        str(out / "strong_baselines" / "pdqn_tiny"),
                    ],
                    timeout_s=300,
                ),
                CommandSpec(
                    [
                        PY,
                        "scripts/train_strong_baseline_tiny.py",
                        "--baseline",
                        "flat_hybrid_ac",
                        "--episodes",
                        "2",
                        "--steps",
                        "4",
                        "--n-leo",
                        "4",
                        "--device",
                        device,
                        "--output-dir",
                        str(out / "strong_baselines" / "flat_hybrid_tiny"),
                    ],
                    timeout_s=300,
                ),
                CommandSpec(
                    [
                        PY,
                        "scripts/run_strong_baselines.py",
                        "--baselines",
                        "pdqn_hybrid,flat_hybrid_ac,small_scale_grid_oracle",
                        "--episodes",
                        "1",
                        "--steps",
                        "4",
                        "--n-leo",
                        "4",
                        "--device",
                        device,
                        "--output-dir",
                        str(out / "strong_baselines" / "eval"),
                    ],
                    timeout_s=300,
                ),
                CommandSpec(
                    [
                        PY,
                        "scripts/evaluate_oracle_gap.py",
                        "--episodes",
                        "1",
                        "--steps",
                        "4",
                        "--n-leo",
                        "4",
                        "--device",
                        device,
                        "--output-dir",
                        str(out / "strong_baselines" / "oracle_gap"),
                    ],
                    timeout_s=300,
                ),
            ],
            out / "strong_baselines",
            ["tests/test_strong_baseline_training.py", "tests/test_small_scale_oracle.py"],
            "P-DQN and flat hybrid have real tiny update smoke; grid oracle computes small-scale gap.",
            "Claiming state-of-the-art superiority from tiny smoke or marking paper_ready before full seeds.",
            "Run full GPU strong-baseline suite and small-scale oracle gap matrix.",
        ),
        StageSpec(
            "P11R",
            "SatEdgeSim resource binding smoke",
            "R7/R8/R9 continuous resource binding semantics",
            [
                CommandSpec(mvn_args("-q", "-DskipTests", "compile"), cwd=SATEDGESIM_ROOT, timeout_s=300),
                CommandSpec(mvn_args("-q", "exec:java", "-Dexec.mainClass=edu.weijunyong.satedgesim.server.RlResourceBindingSmoke"), cwd=SATEDGESIM_ROOT, timeout_s=240),
                CommandSpec(pytest + ["tests/test_satedgesim_metric_semantics.py", "-q"]),
                CommandSpec([PY, "scripts/run_cpu_smoke_satedgesim_summary_semantics.py", "--output-dir", str(out / "satedgesim_semantics")]),
            ],
            out / "satedgesim_semantics",
            ["tests/test_satedgesim_metric_semantics.py", "RlResourceBindingSmoke"],
            "Estimator-bound resource-aware replay can be claimed if smoke passes.",
            "Full hybrid native execution validation unless native_scheduler_bound=true with completion evidence.",
            "Implement and verify native VM/network/power binding if full closed-loop claim is needed.",
            post_check=_satedgesim_binding_post_check,
        ),
    ]


def run_all(device: str, output_dir: Path, skip_commands: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    entries = [str(REPO_ROOT), str(REPO_ROOT / "trisatflow")]
    if existing_pythonpath:
        entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(entries)

    stage_records: list[dict[str, Any]] = []
    for spec in _stage_specs(device=device, output_dir=output_dir):
        stage_started = time.time()
        record: dict[str, Any] = {
            "stage_id": spec.stage_id,
            "stage_name": spec.stage_name,
            "status": "skipped" if skip_commands else "passed",
            "required_for_paper_claim": spec.required_for_paper_claim,
            "cpu_smoke_command": " && ".join(cmd.display for cmd in spec.commands),
            "output_dir": str(spec.output_dir),
            "tests_run": ";".join(spec.tests_run),
            "tests_passed": False if not skip_commands else None,
            "blockers": [],
            "exception": "",
            "claim_allowed": spec.claim_allowed,
            "claim_forbidden": spec.claim_forbidden,
            "next_action": spec.next_action,
            "command_results": [],
        }

        missing_inputs = [cmd.args[1] for cmd in spec.commands if len(cmd.args) > 1 and str(cmd.args[1]).endswith(".py") and not (cmd.cwd / cmd.args[1]).exists()]
        if missing_inputs:
            record["status"] = "not_implemented"
            record["blockers"] = [f"missing command file: {path}" for path in missing_inputs]
            stage_records.append(record)
            continue

        if not skip_commands:
            for index, cmd in enumerate(spec.commands):
                log_path = logs_dir / f"{spec.stage_id}_{index + 1}.log"
                result = _run_command(cmd, log_path, env)
                record["command_results"].append(result)
                if not result.get("ok", False):
                    record["status"] = "failed"
                    record["exception"] = f"command failed: {result.get('command')}"
                    break

            test_results = [
                result
                for result in record["command_results"]
                if "pytest" in result.get("command", "") or "-m pytest" in result.get("command", "")
            ]
            record["tests_passed"] = bool(test_results) and all(result.get("returncode") == 0 for result in test_results)

        if spec.post_check is not None:
            spec.post_check(record)

        record["elapsed_s"] = round(time.time() - stage_started, 3)
        if isinstance(record.get("blockers"), list):
            record["blockers"] = "; ".join(str(item) for item in record["blockers"])
        stage_records.append(record)

    csv_path = output_dir / "stage_status.csv"
    json_path = output_dir / "summary.json"
    fieldnames = [
        "stage_id",
        "stage_name",
        "status",
        "required_for_paper_claim",
        "cpu_smoke_command",
        "output_dir",
        "tests_run",
        "tests_passed",
        "blockers",
        "exception",
        "claim_allowed",
        "claim_forbidden",
        "next_action",
        "elapsed_s",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in stage_records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})

    counts: dict[str, int] = {}
    for record in stage_records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    overall = "passed" if set(counts) <= {"passed"} else "passed_with_failures_or_blockers"
    summary = {
        "status": overall,
        "cpu_only": True,
        "formal_experiment_results": False,
        "output_dir": str(output_dir),
        "stage_status_csv": str(csv_path),
        "stage_counts": counts,
        "paper_claim_ready": False,
        "stage_status": stage_records,
    }
    _write_json(json_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all reviewer-repair CPU smoke checks.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "reviewer_repair" / "final_cpu_smoke")
    parser.add_argument("--skip-commands", action="store_true", help="Only materialize stage status from existing evidence.")
    args = parser.parse_args()
    if args.device != "cpu":
        print("This integration smoke is CPU-only; use --device cpu.", file=sys.stderr)
        return 2
    summary = run_all(device=args.device, output_dir=args.output_dir, skip_commands=args.skip_commands)
    print(json.dumps({"status": summary["status"], "stage_counts": summary["stage_counts"], "output_dir": summary["output_dir"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
