"""Fit direct DEC convolution candidates on train and score validation only."""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

from automodel.model import BASELINES
from automodel.pipeline import I80PredictionEvaluator
from automodel.search_utils import fit_candidate, is_nonlocal_feasible


ROOT = Path(__file__).resolve().parent
BASELINE_ITEMS = tuple(BASELINES.items())
ATTEMPTS = {
    1: {
        "expression": "exp(a*conv_1(rho,ones))",
        "tree_nodes": 6,
        "parameter_names": ("a",),
        "bounds": ((-500.0, 500.0),),
    },
    2: {
        "expression": "exp(a*conv_3(rho,ones))",
        "tree_nodes": 6,
        "parameter_names": ("a",),
        "bounds": ((-250.0, 250.0),),
    },
    3: {
        "expression": "exp(a*conv_1(rho,ones)+b*conv_3(rho,ones))",
        "tree_nodes": 12,
        "parameter_names": ("a", "b"),
        "bounds": ((-500.0, 500.0), (-250.0, 250.0)),
    },
}


def load_correction(attempt: int):
    model_path = ROOT / f"attempt_{attempt}" / "model.py"
    spec = importlib.util.spec_from_file_location(
        f"meta_3_agent_1_attempt_{attempt}", model_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.correction


def format_parameters(names, values) -> str:
    return ", ".join(
        f"{name}={value:.10g}" for name, value in zip(names, values)
    )


def write_reports(attempt: int, records: list[dict], elapsed: float) -> None:
    attempt_dir = ROOT / f"attempt_{attempt}"
    config = ATTEMPTS[attempt]
    completed = [record for record in records if record["status"] == "completed"]
    bounds_json = [list(pair) for pair in config["bounds"]]
    payload = {
        "meta_iteration": 3,
        "agent": 1,
        "attempt": attempt,
        "candidate": {
            "expression": config["expression"],
            "tree_nodes": config["tree_nodes"],
            "parameter_names": list(config["parameter_names"]),
            "nonlocal": True,
            "convolution_implementation": "dctkit.dec.cochain.convolution",
            "kernel": "P0 ones",
            "speed_bound": "row_abs_jacobian_sum",
        },
        "fit_protocol": {
            "dataset": "I80",
            "problem": "prediction",
            "training_split": "full train (times 0-63)",
            "evaluation_split": "full validation (times 64-107)",
            "test_evaluated": False,
            "optimizer": "scipy.optimize.minimize/Powell via automodel.search_utils.fit_candidate",
            "objective": "training E_data",
            "restarts": 2,
            "max_evaluations_per_restart": 45,
            "parameter_bounds": bounds_json,
            "seed_formula": f"{7100 + attempt * 10} + baseline_index",
            "baseline_order": [key for key, _ in BASELINE_ITEMS],
            "feasibility_check": "automodel.search_utils.is_nonlocal_feasible",
            "finite_full_simulation_gate": True,
            "nonlocal": True,
            "speed_bound": "row_abs_jacobian_sum",
            "device": "CPU",
            "pythonpath": "current checkout: <repo>/src:<repo>",
        },
        "records": records,
        "attempt_runtime_seconds": elapsed,
        "attempt_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "completed_baselines": len(completed),
        "failed_baselines": len(records) - len(completed),
        "test_evaluated": False,
        "nonlocal": True,
        "speed_bound": "row_abs_jacobian_sum",
    }
    (attempt_dir / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    training_lines = [
        f"# Attempt {attempt} training",
        "",
        f"Expression: `{config['expression']}` ({config['tree_nodes']} GP tree nodes).",
        "",
        "All values use the complete I80 prediction training split (times 0-63).",
        "Each fit used two deterministic Powell starts with at most 45 function",
        f"evaluations per start. Bounds: `{bounds_json}`.",
        "Nonlocal feasibility used homogeneous states; full simulations supplied",
        "the finite gate. The flux speed bound was `row_abs_jacobian_sum`.",
        "",
        "| Baseline | Parameters | E_rho | E_v | E_data | Finite/feasible | Evaluations | Restarts | Optimizer status | Fit runtime (s) | Split runtime (s) | Peak RSS (MB) |",
        "|---|---|---:|---:|---:|:---:|---:|---:|---|---:|---:|---:|",
    ]
    evaluation_lines = [
        f"# Attempt {attempt} evaluation",
        "",
        "All values use the complete I80 prediction validation split (times 64-107).",
        "The held-out test split was not evaluated. Fitness is validation",
        "`E_data + 0.01 * tree_nodes`; lower is better. This is a nonlocal",
        "convolution candidate using the `row_abs_jacobian_sum` speed bound.",
        "",
        "| Baseline | Expression | Parameters | E_rho | E_v | E_data | Fitness | Finite/feasible | Runtime (s) | Peak RSS (MB) |",
        "|---|---|---|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for record in records:
        if record["status"] != "completed":
            message = record["error"].replace("|", "\\|")
            training_lines.append(
                f"| {record['baseline']} | FAILURE: {message} | | | | no | | | | | | |"
            )
            evaluation_lines.append(
                f"| {record['baseline']} | FAILURE: {message} | | | | | | no | | |"
            )
            continue
        fit = record["fit"]
        train = fit["train"]
        validation = fit["validation"]
        params = format_parameters(fit["parameter_names"], fit["parameters"])
        status = (
            f"{fit['optimizer_success']}: {fit['optimizer_message']}"
        ).replace("|", "\\|")
        training_lines.append(
            f"| {record['baseline']} | {params} | {train['rho_error']:.6f} | "
            f"{train['velocity_error']:.6f} | {train['data_error']:.6f} | "
            f"{'yes' if fit['feasible'] else 'no'} | "
            f"{fit['optimizer_evaluations']} | {fit['restarts']} | {status} | "
            f"{fit['fit_runtime_seconds']:.3f} | {train['runtime_seconds']:.3f} | "
            f"{train['peak_rss_mb']:.1f} |"
        )
        evaluation_lines.append(
            f"| {record['baseline']} | `{fit['expression']}` | {params} | "
            f"{validation['rho_error']:.6f} | {validation['velocity_error']:.6f} | "
            f"{validation['data_error']:.6f} | {fit['validation_fitness']:.6f} | "
            f"{'yes' if validation['finite'] and fit['feasible'] else 'no'} | "
            f"{validation['runtime_seconds']:.3f} | "
            f"{validation['peak_rss_mb']:.1f} |"
        )
    training_lines.extend(["", f"Attempt wall time: {elapsed:.3f} seconds.", ""])
    evaluation_lines.extend(
        [
            "",
            f"Complexity penalty: {0.01 * config['tree_nodes']:.2f}.",
            "`test_evaluated=false`; `nonlocal=true`;",
            "`speed_bound=row_abs_jacobian_sum`.",
            "",
        ]
    )
    (attempt_dir / "training.md").write_text(
        "\n".join(training_lines), encoding="utf-8"
    )
    (attempt_dir / "evaluation.md").write_text(
        "\n".join(evaluation_lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="reload each results.json and refit only failed baseline records",
    )
    args = parser.parse_args()
    evaluator = I80PredictionEvaluator()
    for attempt, config in ATTEMPTS.items():
        correction = load_correction(attempt)
        result_path = ROOT / f"attempt_{attempt}" / "results.json"
        prior_elapsed = 0.0
        if args.retry_failed and result_path.exists():
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            records = list(previous["records"])
            prior_elapsed = float(previous.get("attempt_runtime_seconds", 0.0))
            selected = [
                (index, key, baseline)
                for index, (key, baseline) in enumerate(BASELINE_ITEMS)
                if records[index]["status"] != "completed"
            ]
        else:
            records = []
            selected = [
                (index, key, baseline)
                for index, (key, baseline) in enumerate(BASELINE_ITEMS)
            ]
        attempt_start = time.perf_counter()
        for baseline_index, baseline_key, baseline in selected:
            seed = 7100 + attempt * 10 + baseline_index
            print(
                f"attempt={attempt} baseline={baseline.name} seed={seed}", flush=True
            )
            try:
                fit = fit_candidate(
                    evaluator=evaluator,
                    baseline=baseline,
                    correction=correction,
                    expression=config["expression"],
                    tree_nodes=config["tree_nodes"],
                    parameter_names=config["parameter_names"],
                    bounds=config["bounds"],
                    seed=seed,
                    restarts=2,
                    max_evaluations=45,
                    feasibility_check=is_nonlocal_feasible,
                )
                record = {
                    "baseline_key": baseline_key,
                    "baseline": baseline.name,
                    "status": "completed",
                    "fit": fit.to_dict(),
                    "nonlocal": True,
                    "speed_bound": "row_abs_jacobian_sum",
                    "test_evaluated": False,
                }
                if args.retry_failed:
                    records[baseline_index] = record
                else:
                    records.append(record)
                print(
                    f"completed train_E_data={fit.train.data_error:.8f} "
                    f"validation_E_data={fit.validation.data_error:.8f} "
                    f"fitness={fit.validation_fitness:.8f}",
                    flush=True,
                )
            except Exception as exc:
                record = {
                    "baseline_key": baseline_key,
                    "baseline": baseline.name,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "seed": seed,
                    "nonlocal": True,
                    "speed_bound": "row_abs_jacobian_sum",
                    "test_evaluated": False,
                }
                if args.retry_failed:
                    records[baseline_index] = record
                else:
                    records.append(record)
                print(f"FAILED {type(exc).__name__}: {exc}", flush=True)
            write_reports(
                attempt,
                records,
                prior_elapsed + time.perf_counter() - attempt_start,
            )


if __name__ == "__main__":
    main()
