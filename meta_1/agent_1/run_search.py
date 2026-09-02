"""Run the assigned Phase 3 search without touching the held-out test split."""

from __future__ import annotations

import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

from automodel.model import BASELINES
from automodel.pipeline import I80PredictionEvaluator
from automodel.search_utils import fit_candidate


ROOT = Path(__file__).resolve().parent
BASELINE_ITEMS = tuple(BASELINES.items())
ATTEMPTS = {
    1: {
        "expression": "exp(a*rho)",
        "tree_nodes": 4,
        "parameter_names": ("a",),
    },
    2: {
        "expression": "exp(a*rho + b*rho^2)",
        "tree_nodes": 10,
        "parameter_names": ("a", "b"),
    },
    3: {
        "expression": "exp(a*rho + b*rho^2 + c*rho^3)",
        "tree_nodes": 16,
        "parameter_names": ("a", "b", "c"),
    },
}


def load_correction(attempt: int):
    model_path = ROOT / f"attempt_{attempt}" / "model.py"
    spec = importlib.util.spec_from_file_location(f"agent_1_attempt_{attempt}", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.correction


def format_parameters(names, values) -> str:
    return ", ".join(f"{name}={value:.10g}" for name, value in zip(names, values))


def write_reports(attempt: int, records: list[dict], elapsed: float) -> None:
    attempt_dir = ROOT / f"attempt_{attempt}"
    config = ATTEMPTS[attempt]
    successful = [record for record in records if record["status"] == "completed"]
    payload = {
        "attempt": attempt,
        "candidate": config,
        "fit_protocol": {
            "dataset": "I80",
            "problem": "prediction",
            "training_split": "full train",
            "evaluation_split": "full validation",
            "test_split_accessed": False,
            "optimizer": "scipy.optimize.minimize/Powell",
            "restarts": 2,
            "max_evaluations_per_restart": 30,
            "bounds_per_parameter": [-4.0, 4.0],
            "seed_formula": f"{1100 + attempt * 10} + baseline_index",
            "baseline_order": [key for key, _ in BASELINE_ITEMS],
        },
        "records": records,
        "attempt_runtime_seconds": elapsed,
        "attempt_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "completed_baselines": len(successful),
        "failed_baselines": len(records) - len(successful),
    }
    (attempt_dir / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    training_lines = [
        f"# Attempt {attempt} training",
        "",
        f"Expression: `{config['expression']}` ({config['tree_nodes']} tree nodes).",
        "",
        "All values below use the full I80 prediction training split. Fits used",
        "two Powell starts, 30 evaluations per start, and parameter bounds `[-4, 4]`.",
        "",
        "| Baseline | Constants | rho error | velocity error | data error | Feasible | Evaluations | Optimizer status | Fit runtime (s) | Peak RSS (MB) |",
        "|---|---|---:|---:|---:|:---:|---:|---|---:|---:|",
    ]
    evaluation_lines = [
        f"# Attempt {attempt} evaluation",
        "",
        "All values below use the full I80 prediction validation split. The test",
        "split was not accessed. Fitness is validation data error plus the stated",
        "`0.01 * tree_nodes` complexity penalty.",
        "",
        "| Baseline | Expression | Constants | rho error | velocity error | data error | fitness | Finite | Runtime (s) | Peak RSS (MB) |",
        "|---|---|---|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for record in records:
        if record["status"] != "completed":
            message = record["error"].replace("|", "\\|")
            training_lines.append(f"| {record['baseline']} | FAILURE: {message} | | | | no | | | | |")
            evaluation_lines.append(f"| {record['baseline']} | FAILURE: {message} | | | | | | | | |")
            continue
        fit = record["fit"]
        train = fit["train"]
        validation = fit["validation"]
        constants = format_parameters(fit["parameter_names"], fit["parameters"])
        status = f"{fit['optimizer_success']}: {fit['optimizer_message']}".replace("|", "\\|")
        training_lines.append(
            f"| {record['baseline']} | {constants} | {train['rho_error']:.6f} | "
            f"{train['velocity_error']:.6f} | {train['data_error']:.6f} | "
            f"{'yes' if fit['feasible'] else 'no'} | {fit['optimizer_evaluations']} | "
            f"{status} | {fit['fit_runtime_seconds']:.3f} | {train['peak_rss_mb']:.1f} |"
        )
        evaluation_lines.append(
            f"| {record['baseline']} | `{fit['expression']}` | {constants} | "
            f"{validation['rho_error']:.6f} | {validation['velocity_error']:.6f} | "
            f"{validation['data_error']:.6f} | {fit['validation_fitness']:.6f} | "
            f"{'yes' if validation['finite'] else 'no'} | {validation['runtime_seconds']:.3f} | "
            f"{validation['peak_rss_mb']:.1f} |"
        )
    training_lines.extend(["", f"Attempt wall time: {elapsed:.3f} seconds.", ""])
    evaluation_lines.extend(["", f"Complexity penalty: {0.01 * config['tree_nodes']:.2f}.", ""])
    (attempt_dir / "training.md").write_text("\n".join(training_lines), encoding="utf-8")
    (attempt_dir / "evaluation.md").write_text("\n".join(evaluation_lines), encoding="utf-8")


def main() -> None:
    evaluator = I80PredictionEvaluator()
    for attempt, config in ATTEMPTS.items():
        correction = load_correction(attempt)
        records: list[dict] = []
        attempt_start = time.perf_counter()
        for baseline_index, (_, baseline) in enumerate(BASELINE_ITEMS):
            seed = 1100 + attempt * 10 + baseline_index
            print(
                f"attempt={attempt} baseline={baseline.name} seed={seed}",
                flush=True,
            )
            try:
                fit = fit_candidate(
                    evaluator=evaluator,
                    baseline=baseline,
                    correction=correction,
                    expression=config["expression"],
                    tree_nodes=config["tree_nodes"],
                    parameter_names=config["parameter_names"],
                    bounds=[(-4.0, 4.0)] * len(config["parameter_names"]),
                    seed=seed,
                    restarts=2,
                    max_evaluations=30,
                )
                records.append(
                    {"baseline_key": _, "baseline": baseline.name, "status": "completed", "fit": fit.to_dict()}
                )
                print(
                    f"completed validation_data_error={fit.validation.data_error:.8f} "
                    f"fitness={fit.validation_fitness:.8f}",
                    flush=True,
                )
            except Exception as exc:  # retain auditable failure and continue coverage
                records.append(
                    {
                        "baseline_key": _,
                        "baseline": baseline.name,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "seed": seed,
                    }
                )
                print(f"FAILED {type(exc).__name__}: {exc}", flush=True)
            write_reports(attempt, records, time.perf_counter() - attempt_start)


if __name__ == "__main__":
    main()
