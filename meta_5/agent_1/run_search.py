"""Synchronous nonlinear convolution-contrast search on train/validation only."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import math
import resource
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from dctkit.dec import cochain as C

from automodel.model import BASELINES
from automodel.pipeline import I80PredictionEvaluator
from automodel.search_utils import (
    FitResult,
    fit_candidate,
    is_nonlocal_feasible,
    physical_density_limit,
)
from meta_5.agent_1 import common


ROOT = Path(__file__).resolve().parent
BASELINE_ITEMS = tuple(BASELINES.items())
RESTARTS = 2
MAX_EVALUATIONS = 60
SPEED_BOUND = (
    "row-wise sum(abs(full flux Jacobian)); includes all off-diagonal "
    "Hodge/convolution coupling"
)


def fit_with_zero_fallback(
    *, evaluator, baseline, correction, expression, tree_nodes,
    parameter_names, bounds, seed
) -> FitResult:
    """Run the common fit and explicitly retain its finite zero start.

    Bounded Powell may return an invalid boundary point without retaining the
    supplied initial point. Since zero is an intentional start and exactly
    recovers the fixed incumbent, compare it explicitly and keep it whenever
    the returned endpoint is invalid or has worse training error.
    """

    start = time.perf_counter()
    fit = fit_candidate(
        evaluator=evaluator,
        baseline=baseline,
        correction=correction,
        expression=expression,
        tree_nodes=tree_nodes,
        parameter_names=parameter_names,
        bounds=bounds,
        seed=seed,
        restarts=RESTARTS,
        max_evaluations=MAX_EVALUATIONS,
        feasibility_check=is_nonlocal_feasible,
    )
    zero = tuple(0.0 for _ in parameter_names)
    zero_feasible = is_nonlocal_feasible(evaluator, baseline, correction, zero)
    zero_train = evaluator.evaluate(baseline, correction, zero, expression, "train")
    zero_validation = evaluator.evaluate(
        baseline, correction, zero, expression, "validation"
    )
    fit_valid = bool(
        fit.feasible
        and fit.train.finite
        and fit.validation.finite
        and math.isfinite(fit.train.data_error)
        and math.isfinite(fit.validation.data_error)
    )
    zero_valid = bool(
        zero_feasible
        and zero_train.finite
        and zero_validation.finite
        and math.isfinite(zero_train.data_error)
        and math.isfinite(zero_validation.data_error)
    )
    if zero_valid and (not fit_valid or zero_train.data_error < fit.train.data_error):
        reason = "invalid endpoint" if not fit_valid else "endpoint worse than zero start"
        fit = replace(
            fit,
            parameters=zero,
            train=zero_train,
            validation=zero_validation,
            validation_fitness=zero_validation.data_error + 0.01 * tree_nodes,
            feasible=True,
            optimizer_success=False,
            optimizer_message=(
                f"{fit.optimizer_message}; retained explicit finite zero start "
                f"because Powell returned {reason}"
            ),
        )
    return replace(fit, fit_runtime_seconds=time.perf_counter() - start)


def load_model(attempt: int):
    path = ROOT / f"attempt_{attempt}" / "model.py"
    name = f"meta5_agent1_attempt{attempt}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def feasibility_diagnostics(evaluator, baseline, correction, parameters) -> dict:
    upper = physical_density_limit(baseline)
    densities = jnp.linspace(max(1e-5, upper / 640.0), upper, 64)
    center = evaluator.S.num_nodes // 2
    all_multipliers = []
    center_multipliers = []
    velocities = []
    for density in densities:
        rho = C.CochainP0(evaluator.S, density * jnp.ones(evaluator.S.num_nodes))
        multiplier = correction(rho, parameters).coeffs.flatten()
        all_multipliers.append(multiplier)
        center_multiplier = multiplier[center]
        center_multipliers.append(center_multiplier)
        base_velocity = baseline.velocity(
            jnp.asarray([density]), *baseline.coefficients
        )[0]
        velocities.append(base_velocity * center_multiplier)
    all_multipliers = jnp.stack(all_multipliers)
    center_multipliers = jnp.asarray(center_multipliers)
    velocities = jnp.asarray(velocities)
    velocity_differences = jnp.diff(velocities)
    tolerance = 1e-7
    finite = bool(
        jnp.all(jnp.isfinite(all_multipliers))
        & jnp.all(jnp.isfinite(velocities))
    )
    positive = bool(jnp.all(all_multipliers > 0))
    nonnegative = bool(jnp.all(velocities >= -tolerance))
    nonincreasing = bool(jnp.all(velocity_differences <= tolerance))
    return {
        "method": "64 homogeneous fields; central node constitutive audit",
        "samples": 64,
        "physical_density_domain": [float(densities[0]), float(upper)],
        "finite_multiplier_and_velocity": finite,
        "positive_multiplier_all_nodes": positive,
        "nonnegative_central_corrected_velocity": nonnegative,
        "nonincreasing_central_corrected_velocity": nonincreasing,
        "minimum_multiplier_all_nodes": float(jnp.min(all_multipliers)),
        "maximum_multiplier_all_nodes": float(jnp.max(all_multipliers)),
        "minimum_center_multiplier": float(jnp.min(center_multipliers)),
        "maximum_center_multiplier": float(jnp.max(center_multipliers)),
        "minimum_corrected_velocity": float(jnp.min(velocities)),
        "maximum_velocity_forward_difference": float(jnp.max(velocity_differences)),
        "tolerance": tolerance,
        "passed": bool(finite and positive and nonnegative and nonincreasing),
    }


def constants_text(fit: dict) -> str:
    return ", ".join(
        f"{name}={value:.10g}"
        for name, value in zip(fit["parameter_names"], fit["parameters"])
    )


def write_reports(
    attempt: int, model, records: list[dict], elapsed: float
) -> None:
    attempt_dir = ROOT / f"attempt_{attempt}"
    completed = [record["fit"] for record in records if record["status"] == "completed"]
    aggregate_runtime = sum(fit["fit_runtime_seconds"] for fit in completed)
    peak_rss = max(
        (
            split["peak_rss_mb"]
            for fit in completed
            for split in (fit["train"], fit["validation"])
        ),
        default=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    payload = {
        "meta_iteration": 5,
        "agent": 1,
        "attempt": attempt,
        "candidate": {
            "expression_template": model.TEMPLATE,
            "parameter_names": list(model.PARAMETER_NAMES),
            "tree_node_counting": (
                "global incumbent total nodes + product root + complete nonlinear "
                "increment; repeated symbolic contrast subtrees counted separately"
            ),
            "incumbent_tree_nodes": common.INCUMBENT_NODES,
            "incremental_nodes": model.ADDED_NODES,
            "tree_nodes_by_baseline": {
                key: nodes + model.ADDED_NODES
                for key, nodes in common.INCUMBENT_NODES.items()
            },
            "incumbent_coefficients_fixed": True,
            "nonlocal": True,
            "bound_note": getattr(model, "BOUND_NOTE", None),
        },
        "fit_protocol": {
            "dataset": "I80",
            "problem": "prediction",
            "training_split": "full train (times 0-63)",
            "evaluation_split": "full validation (times 64-107)",
            "test_evaluated": False,
            "optimizer": "scipy.optimize.minimize/Powell via fit_candidate",
            "restarts": RESTARTS,
            "max_evaluations_per_restart": MAX_EVALUATIONS,
            "parameter_bounds": [list(pair) for pair in model.BOUNDS],
            "seed_formula": f"{12100 + attempt * 10} + baseline_index",
            "baseline_order": [key for key, _ in BASELINE_ITEMS],
            "feasibility_check": "automodel.search_utils.is_nonlocal_feasible",
            "wave_speed_bound": SPEED_BOUND,
            "convolution_implementation": "exact dctkit.dec.cochain.convolution",
            "hodge_implementation": "repaired dctkit DEC star composition for IDM incumbent",
            "execution": "synchronous",
            "device": str(jax.devices()[0]),
            "pythonpath": "current checkout: <repo>/src:<repo>",
            "fitness": "validation E_data + 0.01*total_tree_nodes",
        },
        "global_incumbent_validation": common.INCUMBENT_VALIDATION,
        "records": records,
        "attempt_runtime_seconds": aggregate_runtime,
        "last_report_write_process_elapsed_seconds": elapsed,
        "attempt_peak_rss_mb": peak_rss,
        "completed_baselines": sum(r["status"] == "completed" for r in records),
        "failed_baselines": sum(r["status"] != "completed" for r in records),
        "feasible_baselines": sum(
            r["status"] == "completed" and r["fit"]["feasible"] for r in records
        ),
        "nonlocal": True,
        "test_evaluated": False,
    }
    (attempt_dir / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    training = [
        f"# Attempt {attempt} training",
        "",
        f"Template: `{model.TEMPLATE}`.",
        "",
        "All coefficients of each global incumbent were fixed. Only the new",
        f"coefficient(s) `{', '.join(model.PARAMETER_NAMES)}` were fitted on full train",
        f"times 0--63 with two deterministic Powell starts, at most {MAX_EVALUATIONS}",
        f"evaluations per start, and bounds `{[list(pair) for pair in model.BOUNDS]}`.",
        f"Nonlocal speed bound: `{SPEED_BOUND}`.",
        "",
        "| Baseline | Full expression | Nodes | New constants | E_rho | E_v | E_data | Feasible | Evaluations | Seed | Fit time (s) | RSS (MB) |",
        "|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|",
    ]
    evaluation = [
        f"# Attempt {attempt} evaluation",
        "",
        "Full validation times 64--107 were used. Test times 108--179 were not",
        "evaluated. Fitness is `E_data + 0.01*total_tree_nodes`, including the",
        f"fixed incumbent. Nonlocal speed bound: `{SPEED_BOUND}`.",
        "",
        "| Baseline | Full expression | New constants | E_rho | E_v | E_data | Fitness | Incumbent fitness | Delta | Finite/feasible | Runtime (s) | RSS (MB) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for record in records:
        if record["status"] != "completed":
            error = record["error"].replace("|", "\\|")
            training.append(
                f"| {record['baseline']} | FAILURE: {error} | | | | | | no | | {record['seed']} | | |"
            )
            evaluation.append(
                f"| {record['baseline']} | FAILURE: {error} | | | | | | | | | no | | |"
            )
            continue
        fit = record["fit"]
        train = fit["train"]
        validation = fit["validation"]
        constants = constants_text(fit)
        expression = fit["expression"].replace("|", "\\|")
        incumbent = record["incumbent_validation"]
        delta = fit["validation_fitness"] - incumbent["fitness"]
        training.append(
            f"| {record['baseline']} | `{expression}` | {fit['tree_nodes']} | {constants} | "
            f"{train['rho_error']:.6f} | {train['velocity_error']:.6f} | "
            f"{train['data_error']:.6f} | {'yes' if fit['feasible'] else 'no'} | "
            f"{fit['optimizer_evaluations']} | {fit['seed']} | "
            f"{fit['fit_runtime_seconds']:.3f} | {train['peak_rss_mb']:.1f} |"
        )
        evaluation.append(
            f"| {record['baseline']} | `{expression}` | {constants} | "
            f"{validation['rho_error']:.6f} | {validation['velocity_error']:.6f} | "
            f"{validation['data_error']:.6f} | {fit['validation_fitness']:.6f} | "
            f"{incumbent['fitness']:.6f} | {delta:+.6f} | "
            f"{'yes' if fit['feasible'] and validation['finite'] else 'no'} | "
            f"{validation['runtime_seconds']:.3f} | {validation['peak_rss_mb']:.1f} |"
        )
    training.extend(
        ["", f"Aggregate completed fit runtime: {aggregate_runtime:.3f} seconds.", "`test_evaluated = false`.", ""]
    )
    evaluation.extend(
        ["", "All candidates use exact `C.convolution` and are `nonlocal = true`.", "`test_evaluated = false`.", ""]
    )
    (attempt_dir / "training.md").write_text("\n".join(training), encoding="utf-8")
    (attempt_dir / "evaluation.md").write_text("\n".join(evaluation), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--baseline", choices=[key for key, _ in BASELINE_ITEMS])
    args = parser.parse_args()
    model = load_model(args.attempt)
    evaluator = I80PredictionEvaluator()
    result_path = ROOT / f"attempt_{args.attempt}" / "results.json"
    records = []
    if args.baseline is not None and result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        records = [
            record
            for record in previous.get("records", [])
            if record["baseline_key"] != args.baseline
        ]
    attempt_start = time.perf_counter()
    for baseline_index, (baseline_key, baseline) in enumerate(BASELINE_ITEMS):
        if args.baseline is not None and baseline_key != args.baseline:
            continue
        seed = 12100 + args.attempt * 10 + baseline_index
        expression = model.expression(baseline_key)
        tree_nodes = common.INCUMBENT_NODES[baseline_key] + model.ADDED_NODES
        correction = model.make_correction(baseline_key)
        print(
            f"attempt={args.attempt} baseline={baseline.name} seed={seed} nodes={tree_nodes}",
            flush=True,
        )
        try:
            fit = fit_with_zero_fallback(
                evaluator=evaluator,
                baseline=baseline,
                correction=correction,
                expression=expression,
                tree_nodes=tree_nodes,
                parameter_names=model.PARAMETER_NAMES,
                bounds=model.BOUNDS,
                seed=seed,
            )
            diagnostics = feasibility_diagnostics(
                evaluator, baseline, correction, fit.parameters
            )
            incumbent = common.INCUMBENT_VALIDATION[baseline_key]
            records.append(
                {
                    "baseline_key": baseline_key,
                    "baseline": baseline.name,
                    "status": "completed",
                    "fit": fit.to_dict(),
                    "incumbent_validation": incumbent,
                    "validation_fitness_delta_vs_global_incumbent": (
                        fit.validation_fitness - incumbent["fitness"]
                    ),
                    "diagnostics": {
                        "homogeneous_feasibility": diagnostics,
                        "wave_speed_bound": SPEED_BOUND,
                        "simulation_finite_train": fit.train.finite,
                        "simulation_finite_validation": fit.validation.finite,
                        "hodge_primitives_in_fixed_idm_incumbent": [
                            "St_oneP0",
                            "SquareD1",
                            "St_oneD1",
                        ],
                    },
                    "nonlocal": True,
                    "test_evaluated": False,
                }
            )
            print(
                f"completed train={fit.train.data_error:.8f} "
                f"validation={fit.validation.data_error:.8f} "
                f"fitness={fit.validation_fitness:.8f}",
                flush=True,
            )
        except Exception as exc:
            records.append(
                {
                    "baseline_key": baseline_key,
                    "baseline": baseline.name,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "seed": seed,
                    "nonlocal": True,
                    "test_evaluated": False,
                }
            )
            print(f"FAILED {type(exc).__name__}: {exc}", flush=True)
        order = {key: index for index, (key, _) in enumerate(BASELINE_ITEMS)}
        records.sort(key=lambda record: order[record["baseline_key"]])
        write_reports(
            args.attempt, model, records, time.perf_counter() - attempt_start
        )


if __name__ == "__main__":
    main()
