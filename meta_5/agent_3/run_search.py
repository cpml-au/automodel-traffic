"""Meta-5 agent 3 mixed repaired-Hodge/convolution search, train/validation only."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from dctkit.dec import cochain as C
from deap import gp
from scipy.optimize import minimize

from automodel.model import BASELINES
from automodel.pipeline import I80PredictionEvaluator
from automodel.search_utils import FitResult, is_nonlocal_feasible, physical_density_limit
from automodel.typed_primitives import build_traffic_pset
from meta_5.agent_3 import common


ROOT = Path(__file__).resolve().parent
BASELINE_ITEMS = tuple(BASELINES.items())
RESTARTS = 2
MAX_EVALUATIONS = 60
SPEED_BOUND = (
    "row-wise sum(abs(full flux Jacobian)); includes all off-diagonal "
    "Hodge/convolution coupling"
)


def fit_candidate_reconditioned(
    *, evaluator, baseline, correction, expression, tree_nodes,
    parameter_names, bounds, parameter_scales, seed,
):
    """Fit in dimensionless Powell coordinates and return effective constants."""

    parameter_names = tuple(parameter_names)
    bounds = tuple(tuple(pair) for pair in bounds)
    scales = np.asarray(parameter_scales, dtype=float)
    scaled_bounds = tuple(
        (low / scale, high / scale)
        for (low, high), scale in zip(bounds, scales)
    )
    rng = np.random.default_rng(seed)
    starts = [np.zeros(len(bounds))]
    for _ in range(max(0, RESTARTS - 1)):
        starts.append(
            np.asarray([rng.uniform(low, high) for low, high in scaled_bounds])
        )
    evaluations = 0
    started = time.perf_counter()
    best = None
    best_seen_value = float("inf")
    best_seen_parameters = None

    def effective(values):
        return np.asarray(values, dtype=float) * scales

    def objective(values):
        nonlocal evaluations, best_seen_value, best_seen_parameters
        evaluations += 1
        parameters = effective(values)
        if not is_nonlocal_feasible(evaluator, baseline, correction, parameters):
            return 100.0
        result = evaluator.evaluate(
            baseline, correction, tuple(parameters), expression, "train"
        )
        value = result.data_error if result.finite else 100.0
        if result.finite and value < best_seen_value:
            best_seen_value = value
            best_seen_parameters = tuple(float(x) for x in parameters)
        return value

    for initial in starts:
        # Explicitly audit every start. Bounded Powell may otherwise begin at an
        # interior line-search point and never evaluate the supplied zero start.
        objective(initial)
        fit = minimize(
            objective,
            initial,
            method="Powell",
            bounds=scaled_bounds,
            options={"maxfev": MAX_EVALUATIONS - 1, "xtol": 2e-3, "ftol": 2e-3},
        )
        if best is None or float(fit.fun) < float(best.fun):
            best = fit
    assert best is not None
    no_feasible_point = best_seen_parameters is None
    # Powell can stop mid-coordinate at maxfev, leaving result.x at an unevaluated
    # or rejected endpoint. Use the best actually evaluated feasible parameters.
    # If the whole structure is infeasible (notably Weidmann when C vanishes on
    # homogeneous states), evaluate and record the zero/incumbent fallback.
    parameters = (
        tuple(0.0 for _ in parameter_names)
        if no_feasible_point
        else best_seen_parameters
    )
    feasible = is_nonlocal_feasible(evaluator, baseline, correction, parameters)
    train = evaluator.evaluate(baseline, correction, parameters, expression, "train")
    validation = evaluator.evaluate(
        baseline, correction, parameters, expression, "validation"
    )
    return FitResult(
        expression=expression,
        tree_nodes=tree_nodes,
        parameter_names=parameter_names,
        parameters=parameters,
        train=train,
        validation=validation,
        validation_fitness=validation.data_error + 0.01 * tree_nodes,
        feasible=feasible and train.finite and validation.finite,
        optimizer_success=bool(best.success) and not no_feasible_point,
        optimizer_message=(
            "No finite feasible point found; zero/incumbent fallback evaluated."
            if no_feasible_point
            else str(best.message)
        ),
        optimizer_evaluations=evaluations,
        restarts=RESTARTS,
        seed=seed,
        fit_runtime_seconds=time.perf_counter() - started,
    )


def load_model(attempt: int):
    path = ROOT / f"attempt_{attempt}" / "model.py"
    spec = importlib.util.spec_from_file_location(f"meta5_agent3_attempt{attempt}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_grammar(model) -> tuple[list[str], int]:
    pset, _ = build_traffic_pset()
    pset.addTerminal(0.0, float, "a")
    pset.addTerminal(0.0, float, "b")
    pset.addTerminal(3.0, float, "three")
    pset.addTerminal(79.0, float, "seventy_nine")
    registered = [
        name
        for name in ("St_oneP0", "St_oneP1", "St_oneD0", "St_oneD1")
        if name in pset.mapping
    ]
    if len(registered) != 4:
        raise RuntimeError(f"incomplete repaired Hodge family: {registered}")
    tree = gp.PrimitiveTree.from_string(model.TYPED_FACTOR, pset)
    gp.compile(tree, pset)
    expected = model.ADDED_NODES - 1
    if len(tree) != expected:
        raise RuntimeError(
            f"attempt {model.ATTEMPT}: typed factor has {len(tree)} nodes, expected {expected}"
        )
    return registered, len(tree)


def feasibility_diagnostics(evaluator, baseline, correction, parameters) -> dict:
    upper = physical_density_limit(baseline)
    densities = jnp.linspace(max(1e-5, upper / 640.0), upper, 64)
    center = evaluator.S.num_nodes // 2
    multipliers = []
    centers = []
    velocities = []
    for density in densities:
        rho = C.CochainP0(evaluator.S, density * jnp.ones(evaluator.S.num_nodes))
        values = correction(rho, parameters).coeffs.flatten()
        multipliers.append(values)
        centers.append(values[center])
        velocities.append(
            baseline.velocity(jnp.asarray([density]), *baseline.coefficients)[0]
            * values[center]
        )
    multipliers = jnp.stack(multipliers)
    centers = jnp.asarray(centers)
    velocities = jnp.asarray(velocities)
    differences = jnp.diff(velocities)
    tolerance = 1e-7
    finite = bool(jnp.all(jnp.isfinite(multipliers)) & jnp.all(jnp.isfinite(velocities)))
    positive = bool(jnp.all(multipliers > 0))
    nonnegative = bool(jnp.all(velocities >= -tolerance))
    nonincreasing = bool(jnp.all(differences <= tolerance))
    return {
        "method": "64 homogeneous fields; central node constitutive audit",
        "samples": 64,
        "physical_density_domain": [float(densities[0]), float(upper)],
        "finite_multiplier_and_velocity": finite,
        "positive_multiplier_all_nodes": positive,
        "nonnegative_central_corrected_velocity": nonnegative,
        "nonincreasing_central_corrected_velocity": nonincreasing,
        "minimum_multiplier_all_nodes": float(jnp.min(multipliers)),
        "maximum_multiplier_all_nodes": float(jnp.max(multipliers)),
        "minimum_center_multiplier": float(jnp.min(centers)),
        "maximum_center_multiplier": float(jnp.max(centers)),
        "minimum_corrected_velocity": float(jnp.min(velocities)),
        "maximum_velocity_forward_difference": float(jnp.max(differences)),
        "tolerance": tolerance,
        "passed": finite and positive and nonnegative and nonincreasing,
    }


def format_parameters(names, values):
    return ", ".join(f"{name}={value:.10g}" for name, value in zip(names, values))


def write_reports(model, records, registered, factor_nodes, elapsed):
    attempt_dir = ROOT / f"attempt_{model.ATTEMPT}"
    completed = [r["fit"] for r in records if r["status"] == "completed"]
    payload = {
        "meta_iteration": 5,
        "agent": 3,
        "attempt": model.ATTEMPT,
        "candidate": {
            "symbolic_factor": model.SYMBOLIC_FACTOR,
            "typed_gp_factor": model.TYPED_FACTOR,
            "typed_gp_factor_nodes": factor_nodes,
            "attachment_node": "CMulP0",
            "added_nodes_including_attachment": model.ADDED_NODES,
            "tree_node_counting": (
                "fixed global incumbent nodes from final_candidates.json plus one "
                "CMulP0 attachment and the complete typed factor; repeated feature "
                "subtrees are counted separately"
            ),
            "incumbent_nodes": common.INCUMBENT_NODES,
            "total_nodes_by_baseline": {
                key: common.INCUMBENT_NODES[key] + model.ADDED_NODES
                for key, _ in BASELINE_ITEMS
            },
            "incumbent_coefficients_fixed": True,
            "incumbent_source": "automodel/final_candidates.json",
            "hodge_primitives_used": list(model.HODGE_PRIMITIVES),
            "hodge_primitives_registered": registered,
            "nonlocal": True,
        },
        "fit_protocol": {
            "dataset": "I80",
            "problem": "prediction",
            "training_split": "full train times 0-63 (4,800 rows)",
            "evaluation_split": "full validation times 64-107 (3,300 rows)",
            "test_evaluated": False,
            "optimizer": "scipy.optimize.minimize/Powell via fit_candidate",
            "objective": "full training E_data",
            "restarts": RESTARTS,
            "max_evaluations_per_restart": MAX_EVALUATIONS,
            "parameter_bounds": [list(bound) for bound in model.BOUNDS],
            "parameter_scales": list(model.PARAMETER_SCALES),
            "optimizer_internal_bounds": [
                [bound[0] / scale, bound[1] / scale]
                for bound, scale in zip(model.BOUNDS, model.PARAMETER_SCALES)
            ],
            "seed_formula": f"{13100 + model.ATTEMPT * 10} + baseline_index",
            "baseline_order": [key for key, _ in BASELINE_ITEMS],
            "feasibility_check": "automodel.search_utils.is_nonlocal_feasible",
            "wave_speed_bound": SPEED_BOUND,
            "fitness": "validation E_data + 0.01*full tree nodes",
        },
        "global_incumbent_validation_fitness": common.INCUMBENT_FITNESS,
        "records": records,
        "attempt_fit_runtime_seconds": sum(f["fit_runtime_seconds"] for f in completed),
        "process_elapsed_seconds_at_last_write": elapsed,
        "attempt_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "completed_baselines": len(completed),
        "failed_baselines": len(records) - len(completed),
        "execution": {
            "checkout": str(Path.cwd()),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "jax_platform_name": os.environ.get("JAX_PLATFORMS", ""),
            "device": "CPU",
            "synchronous": True,
        },
        "nonlocal": True,
        "test_evaluated": False,
    }
    (attempt_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    training = [
        f"# Attempt {model.ATTEMPT} training",
        "",
        "All global incumbent coefficients were held fixed. Only the new constants",
        f"used two deterministic Powell restarts and at most {MAX_EVALUATIONS}",
        "evaluations per restart on full train times 0--63.",
        "",
        "| FD | Full expression | Nodes | Parameters | E_rho | E_v | E_data | Optimizer | Evals | Feasible | Runtime (s) | RSS (MB) | Seed |",
        "|---|---|---:|---|---:|---:|---:|---|---:|:---:|---:|---:|---:|",
    ]
    evaluation = [
        f"# Attempt {model.ATTEMPT} evaluation",
        "",
        "Full validation times 64--107 only. Fitness is",
        "`E_data + 0.01*full_tree_nodes`; lower is better.",
        f"Wave-speed bound: `{SPEED_BOUND}`.",
        "",
        "| FD | Parameters | E_rho | E_v | E_data | Fitness | Incumbent fitness | Change | Finite/feasible | Runtime (s) | RSS (MB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for record in records:
        if record["status"] != "completed":
            err = record["error"].replace("|", "/")
            training.append(f"| {record['baseline']} | FAILED: {err} | | | | | | | | no | | | {record['seed']} |")
            evaluation.append(f"| {record['baseline']} | FAILED: {err} | | | | | | | no | | |")
            continue
        fit = record["fit"]
        train = fit["train"]
        val = fit["validation"]
        params = format_parameters(fit["parameter_names"], fit["parameters"])
        message = fit["optimizer_message"].replace("|", "/")
        training.append(
            f"| {record['baseline']} | `{fit['expression']}` | {fit['tree_nodes']} | {params} | "
            f"{train['rho_error']:.6f} | {train['velocity_error']:.6f} | {train['data_error']:.6f} | "
            f"{'success' if fit['optimizer_success'] else 'stopped'}: {message} | "
            f"{fit['optimizer_evaluations']} | {'yes' if fit['feasible'] else 'no'} | "
            f"{fit['fit_runtime_seconds']:.3f} | {train['peak_rss_mb']:.1f} | {fit['seed']} |"
        )
        evaluation.append(
            f"| {record['baseline']} | {params} | {val['rho_error']:.6f} | "
            f"{val['velocity_error']:.6f} | {val['data_error']:.6f} | "
            f"{fit['validation_fitness']:.6f} | {record['incumbent_fitness']:.6f} | "
            f"{record['fitness_change_from_incumbent']:+.6f} | "
            f"{'yes' if fit['feasible'] and val['finite'] else 'no'} | "
            f"{val['runtime_seconds']:.3f} | {val['peak_rss_mb']:.1f} |"
        )
    training.extend(["", "`test_evaluated = false`.", ""])
    evaluation.extend([
        "",
        f"Registered repaired Hodge variants: `{', '.join(registered)}`.",
        "All candidates use `is_nonlocal_feasible`.",
        "`test_evaluated = false`.",
        "",
    ])
    (attempt_dir / "training.md").write_text("\n".join(training), encoding="utf-8")
    (attempt_dir / "evaluation.md").write_text("\n".join(evaluation), encoding="utf-8")


def run_attempt(attempt: int, only_baseline: str | None = None):
    model = load_model(attempt)
    registered, factor_nodes = validate_grammar(model)
    evaluator = I80PredictionEvaluator()
    result_path = ROOT / f"attempt_{attempt}" / "results.json"
    records = []
    if only_baseline is not None and result_path.exists():
        old = json.loads(result_path.read_text(encoding="utf-8"))
        records = [r for r in old.get("records", []) if r["baseline_key"] != only_baseline]
    started = time.perf_counter()
    for baseline_index, (baseline_key, baseline) in enumerate(BASELINE_ITEMS):
        if only_baseline is not None and baseline_key != only_baseline:
            continue
        seed = 13100 + attempt * 10 + baseline_index
        correction = model.make_correction(baseline_key)
        expression = model.expression(baseline_key)
        nodes = common.INCUMBENT_NODES[baseline_key] + model.ADDED_NODES
        print(f"attempt={attempt} baseline={baseline.name} seed={seed} nodes={nodes}", flush=True)
        try:
            fit = fit_candidate_reconditioned(
                evaluator=evaluator,
                baseline=baseline,
                correction=correction,
                expression=expression,
                tree_nodes=nodes,
                parameter_names=model.PARAMETER_NAMES,
                bounds=model.BOUNDS,
                parameter_scales=model.PARAMETER_SCALES,
                seed=seed,
            )
            diagnostics = feasibility_diagnostics(evaluator, baseline, correction, fit.parameters)
            records.append({
                "baseline_key": baseline_key,
                "baseline": baseline.name,
                "status": "completed",
                "fit": fit.to_dict(),
                "incumbent_fitness": common.INCUMBENT_FITNESS[baseline_key],
                "fitness_change_from_incumbent": fit.validation_fitness - common.INCUMBENT_FITNESS[baseline_key],
                "diagnostics": {
                    "homogeneous_feasibility": diagnostics,
                    "wave_speed_bound": SPEED_BOUND,
                    "simulation_finite_train": fit.train.finite,
                    "simulation_finite_validation": fit.validation.finite,
                },
                "nonlocal": True,
                "test_evaluated": False,
            })
            print(
                f"completed baseline={baseline.name} parameters={fit.parameters} "
                f"train={fit.train.data_error:.6f} validation={fit.validation.data_error:.6f} "
                f"fitness={fit.validation_fitness:.6f} "
                f"delta={fit.validation_fitness-common.INCUMBENT_FITNESS[baseline_key]:+.6f} "
                f"feasible={fit.feasible}",
                flush=True,
            )
        except Exception as exc:
            records.append({
                "baseline_key": baseline_key,
                "baseline": baseline.name,
                "status": "failed",
                "seed": seed,
                "error": f"{type(exc).__name__}: {exc}",
                "test_evaluated": False,
            })
            print(f"failed baseline={baseline.name}: {type(exc).__name__}: {exc}", flush=True)
        order = {key: index for index, (key, _) in enumerate(BASELINE_ITEMS)}
        records.sort(key=lambda r: order[r["baseline_key"]])
        write_reports(model, records, registered, factor_nodes, time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--baseline", choices=[key for key, _ in BASELINE_ITEMS])
    args = parser.parse_args()
    run_attempt(args.attempt, args.baseline)


if __name__ == "__main__":
    main()
