"""Run meta-2 agent-3 direct-multiplier attempts without test access."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
import numpy as np
from dctkit.dec import cochain as C
from scipy.optimize import minimize

from automodel.model import BASELINES, Baseline
from automodel.pipeline import I80PredictionEvaluator
from automodel.search_utils import physical_density_limit


ROOT = Path(__file__).resolve().parent
BASELINE_ITEMS = tuple(BASELINES.items())
ATTEMPTS = {
    1: {
        "expression": "1 + a*rho",
        "tree_nodes": 5,
        "parameter_names": ("a",),
    },
    2: {
        "expression": "(1 + a*rho)/(1 + b*rho)",
        "tree_nodes": 11,
        "parameter_names": ("a", "b"),
    },
    3: {
        "expression": "(1 + a*rho + b*rho^2)/(1 + c*rho)",
        "tree_nodes": 17,
        "parameter_names": ("a", "b", "c"),
    },
}
BOUNDS = (-0.9, 4.0)
RESTARTS = 2
MAX_EVALUATIONS = 45


def load_model(attempt: int):
    path = ROOT / f"attempt_{attempt}" / "model.py"
    spec = importlib.util.spec_from_file_location(f"meta2_agent3_attempt{attempt}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def feasibility_details(
    evaluator: I80PredictionEvaluator,
    baseline: Baseline,
    model,
    parameters: Sequence[float],
) -> dict:
    """Audit positivity, singularity avoidance, and velocity monotonicity.

    Multiplier and denominator checks span the normalized observed density range
    [0, 1]. Corrected velocity is checked only on the calibrated baseline's
    physical support, as required by CONTEXT.md.
    """

    parameters = tuple(float(value) for value in parameters)
    if not np.all(np.isfinite(parameters)):
        return {"feasible": False, "reason": "non-finite parameter"}

    observed_x = jnp.linspace(0.0, 1.0, 79)
    observed_rho = C.CochainP0(evaluator.S, observed_x)
    multiplier = model.correction(observed_rho, parameters).coeffs

    denominator = jnp.ones_like(observed_x)
    if hasattr(model, "denominator"):
        denominator = model.denominator(observed_x, parameters)

    upper = physical_density_limit(baseline)
    physical_x = jnp.linspace(max(1e-5, upper / 790.0), upper, 79)
    physical_rho = C.CochainP0(evaluator.S, physical_x)
    physical_multiplier = model.correction(physical_rho, parameters).coeffs
    velocity = baseline.velocity(physical_x, *baseline.coefficients) * physical_multiplier
    velocity_differences = jnp.diff(velocity)
    tolerance = 1e-7

    finite = bool(
        jnp.all(jnp.isfinite(multiplier))
        & jnp.all(jnp.isfinite(denominator))
        & jnp.all(jnp.isfinite(velocity))
    )
    positive_multiplier = bool(jnp.all(multiplier > 0.0))
    nonsingular_denominator = bool(jnp.all(denominator > 1e-6))
    nonnegative_velocity = bool(jnp.all(velocity >= -tolerance))
    nonincreasing_velocity = bool(jnp.all(velocity_differences <= tolerance))
    feasible = (
        finite
        and positive_multiplier
        and nonsingular_denominator
        and nonnegative_velocity
        and nonincreasing_velocity
    )
    failed = [
        label
        for label, passed in (
            ("finite", finite),
            ("positive_multiplier", positive_multiplier),
            ("nonsingular_positive_denominator", nonsingular_denominator),
            ("nonnegative_velocity", nonnegative_velocity),
            ("nonincreasing_velocity", nonincreasing_velocity),
        )
        if not passed
    ]
    return {
        "feasible": feasible,
        "reason": "passed" if feasible else ", ".join(failed),
        "checks": {
            "finite": finite,
            "positive_multiplier_observed_domain": positive_multiplier,
            "nonsingular_positive_denominator_observed_domain": nonsingular_denominator,
            "nonnegative_velocity_physical_domain": nonnegative_velocity,
            "nonincreasing_velocity_physical_domain": nonincreasing_velocity,
        },
        "observed_density_domain": [0.0, 1.0],
        "physical_density_domain": [float(physical_x[0]), float(upper)],
        "minimum_multiplier": float(jnp.min(multiplier)),
        "minimum_denominator": float(jnp.min(denominator)),
        "minimum_corrected_velocity": float(jnp.min(velocity)),
        "maximum_velocity_forward_difference": float(jnp.max(velocity_differences)),
        "samples": 79,
        "tolerance": tolerance,
    }


def fit_candidate(
    evaluator: I80PredictionEvaluator,
    baseline: Baseline,
    model,
    config: dict,
    seed: int,
) -> dict:
    """Fit on full train and score only full validation with explicit checks."""

    parameter_names = tuple(config["parameter_names"])
    bounds = [BOUNDS] * len(parameter_names)
    rng = np.random.default_rng(seed)
    starts = [np.zeros(len(bounds), dtype=float)]
    starts.append(np.asarray([rng.uniform(*bound) for bound in bounds]))
    objective_evaluations = 0
    infeasible_evaluations = 0
    restart_reports = []
    best = None
    fit_start = time.perf_counter()

    def objective(values: np.ndarray) -> float:
        nonlocal objective_evaluations, infeasible_evaluations
        objective_evaluations += 1
        feasibility = feasibility_details(evaluator, baseline, model, values)
        if not feasibility["feasible"]:
            infeasible_evaluations += 1
            return 100.0
        result = evaluator.evaluate(
            baseline,
            model.correction,
            tuple(values),
            config["expression"],
            "train",
        )
        return result.data_error if result.finite else 100.0

    for restart_index, initial in enumerate(starts):
        restart_start = time.perf_counter()
        result = minimize(
            objective,
            initial,
            method="Powell",
            bounds=bounds,
            options={"maxfev": MAX_EVALUATIONS, "xtol": 2e-3, "ftol": 2e-3},
        )
        restart_reports.append(
            {
                "restart": restart_index,
                "initial_parameters": [float(value) for value in initial],
                "final_parameters": [float(value) for value in result.x],
                "training_objective": float(result.fun),
                "success": bool(result.success),
                "message": str(result.message),
                "evaluations": int(result.nfev),
                "runtime_seconds": time.perf_counter() - restart_start,
            }
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result

    assert best is not None
    parameters = tuple(float(value) for value in best.x)
    feasibility = feasibility_details(evaluator, baseline, model, parameters)
    train = evaluator.evaluate(
        baseline, model.correction, parameters, config["expression"], "train"
    )
    validation = evaluator.evaluate(
        baseline, model.correction, parameters, config["expression"], "validation"
    )
    return {
        "expression": config["expression"],
        "tree_nodes": config["tree_nodes"],
        "parameter_names": list(parameter_names),
        "parameters": list(parameters),
        "train": asdict(train),
        "validation": asdict(validation),
        "validation_fitness": validation.data_error + 0.01 * config["tree_nodes"],
        "feasible": bool(feasibility["feasible"] and train.finite and validation.finite),
        "feasibility": feasibility,
        "optimizer": {
            "name": "scipy.optimize.minimize/Powell",
            "success": bool(best.success),
            "message": str(best.message),
            "objective_evaluations": objective_evaluations,
            "infeasible_objective_evaluations": infeasible_evaluations,
            "restarts": RESTARTS,
            "restart_reports": restart_reports,
        },
        "seed": seed,
        "fit_runtime_seconds": time.perf_counter() - fit_start,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "test_evaluated": False,
    }


def format_parameters(names, values) -> str:
    return ", ".join(f"{name}={value:.10g}" for name, value in zip(names, values))


def write_reports(attempt: int, records: dict, elapsed: float) -> None:
    attempt_dir = ROOT / f"attempt_{attempt}"
    config = ATTEMPTS[attempt]
    payload = {
        "attempt": attempt,
        "candidate": config,
        "protocol": {
            "dataset": "I80",
            "problem": "prediction",
            "fit_split": "full train (times 0-63)",
            "selection_split": "full validation (times 64-107)",
            "test_evaluated": False,
            "optimizer": "scipy.optimize.minimize/Powell",
            "bounds_each_parameter": list(BOUNDS),
            "restarts": RESTARTS,
            "max_evaluations_per_restart": MAX_EVALUATIONS,
            "seed_formula": "6100 + 10*attempt + baseline_index",
            "baseline_order": [key for key, _ in BASELINE_ITEMS],
            "fitness": "validation.data_error + 0.01*tree_nodes",
            "feasibility": (
                "finite positive multiplier and positive denominator on rho in [0,1]; "
                "finite nonnegative non-increasing corrected velocity on physical domain"
            ),
        },
        "baselines": records,
        "attempt_runtime_seconds": elapsed,
        "attempt_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    (attempt_dir / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    training_lines = [
        f"# Attempt {attempt} training",
        "",
        f"Expression: `{config['expression']}` ({config['tree_nodes']} tree nodes).",
        "",
        "All values use the full I80 prediction training split (times 0--63).",
        "Two deterministic Powell starts used 45 evaluations per start and",
        "parameter bounds `[-0.9, 4.0]`. Feasibility was checked before every",
        "simulation objective call.",
        "",
        "| Baseline | Constants | rho error | velocity error | data error | Feasible | Evaluations | Infeasible rejected | Optimizer status | Fit runtime (s) | Peak RSS (MB) |",
        "|---|---|---:|---:|---:|:---:|---:|---:|---|---:|---:|",
    ]
    evaluation_lines = [
        f"# Attempt {attempt} evaluation",
        "",
        "All values use the full I80 prediction validation split (times 64--107).",
        "The held-out test split was not evaluated. Fitness is validation data",
        "error plus `0.01 * tree_nodes`.",
        "",
        "| Baseline | Expression | Constants | rho error | velocity error | data error | fitness | Feasible | Validation runtime (s) | Peak RSS (MB) |",
        "|---|---|---|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for key, baseline in BASELINE_ITEMS:
        record = records.get(key)
        if record is None:
            continue
        if record["status"] != "completed":
            message = record["error"].replace("|", "\\|")
            training_lines.append(f"| {baseline.name} | FAILURE: {message} | | | | no | | | | | |")
            evaluation_lines.append(f"| {baseline.name} | FAILURE: {message} | | | | | | no | | |")
            continue
        fit = record["fit"]
        train = fit["train"]
        validation = fit["validation"]
        optimizer = fit["optimizer"]
        constants = format_parameters(fit["parameter_names"], fit["parameters"])
        status = f"{optimizer['success']}: {optimizer['message']}".replace("|", "\\|")
        training_lines.append(
            f"| {baseline.name} | {constants} | {train['rho_error']:.6f} | "
            f"{train['velocity_error']:.6f} | {train['data_error']:.6f} | "
            f"{'yes' if fit['feasible'] else 'no'} | {optimizer['objective_evaluations']} | "
            f"{optimizer['infeasible_objective_evaluations']} | {status} | "
            f"{fit['fit_runtime_seconds']:.3f} | {fit['peak_rss_mb']:.1f} |"
        )
        evaluation_lines.append(
            f"| {baseline.name} | `{fit['expression']}` | {constants} | "
            f"{validation['rho_error']:.6f} | {validation['velocity_error']:.6f} | "
            f"{validation['data_error']:.6f} | {fit['validation_fitness']:.6f} | "
            f"{'yes' if fit['feasible'] else 'no'} | {validation['runtime_seconds']:.3f} | "
            f"{validation['peak_rss_mb']:.1f} |"
        )
    training_lines.extend(["", f"Attempt wall time: {elapsed:.3f} seconds.", ""])
    evaluation_lines.extend(
        ["", f"Complexity penalty: {0.01 * config['tree_nodes']:.2f}.", "", "`test_evaluated = false`.", ""]
    )
    (attempt_dir / "training.md").write_text("\n".join(training_lines), encoding="utf-8")
    (attempt_dir / "evaluation.md").write_text("\n".join(evaluation_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt", type=int, choices=ATTEMPTS)
    args = parser.parse_args()
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("Set JAX_PLATFORMS=cpu")

    attempt = args.attempt
    config = ATTEMPTS[attempt]
    model = load_model(attempt)
    evaluator = I80PredictionEvaluator()
    records = {}
    attempt_start = time.perf_counter()
    for baseline_index, (key, baseline) in enumerate(BASELINE_ITEMS):
        seed = 6100 + attempt * 10 + baseline_index
        print(f"attempt={attempt} baseline={baseline.name} seed={seed}", flush=True)
        try:
            fit = fit_candidate(evaluator, baseline, model, config, seed)
            records[key] = {"status": "completed", "fit": fit}
            print(
                f"completed train={fit['train']['data_error']:.8f} "
                f"validation={fit['validation']['data_error']:.8f} "
                f"fitness={fit['validation_fitness']:.8f} feasible={fit['feasible']}",
                flush=True,
            )
        except Exception as exc:
            records[key] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "seed": seed,
                "test_evaluated": False,
            }
            print(f"FAILED {type(exc).__name__}: {exc}", flush=True)
        write_reports(attempt, records, time.perf_counter() - attempt_start)


if __name__ == "__main__":
    main()
