"""Run meta-4 incumbent-plus-alternative-convolution searches, never test."""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    fit_candidate,
    is_nonlocal_feasible,
    physical_density_limit,
)


ROOT = Path(__file__).resolve().parent
BASELINE_ITEMS = tuple(BASELINES.items())
INCUMBENT_NODES = {
    "greenshields": 25,
    "idm": 14,
    "weidmann": 10,
    "triangular": 1,
    "del_castillo": 21,
}
INCUMBENT_VALIDATION = {
    "greenshields": {"data_error": 6.703732, "fitness": 6.953732},
    "idm": {"data_error": 5.277018, "fitness": 5.417018},
    "weidmann": {"data_error": 6.222293, "fitness": 6.322293},
    "triangular": {"data_error": 6.447489, "fitness": 6.457489},
    "del_castillo": {"data_error": 5.520221, "fitness": 5.730221},
}
ATTEMPTS = {
    1: {"contrast": 2, "template": "g_inc*exp(a*(conv_3-2*conv_1))"},
    2: {"contrast": 4, "template": "g_inc*exp(a*(conv_3-4*conv_1))"},
    3: {"contrast": 3, "template": "g_inc*exp(a*(conv_3-3*conv_1))"},
}
PARAMETER_NAMES = ("a",)
ADDED_NODES = 13
BOUNDS = (-300.0, 300.0)
RESTARTS = 2
MAX_EVALUATIONS = 60
SPEED_BOUND = "row-wise sum(abs(full flux Jacobian)); includes off-diagonal coupling"


def load_model(attempt: int):
    path = ROOT / f"attempt_{attempt}" / "model.py"
    name = f"meta4_agent1_attempt{attempt}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def feasibility_diagnostics(evaluator, baseline, correction, parameters) -> dict:
    """Fully report the homogeneous-state audit used by the fit gate."""

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
        "method": "64 homogeneous fields; central node for constitutive velocity",
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


def write_reports(attempt: int, records: list[dict], elapsed: float) -> None:
    attempt_dir = ROOT / f"attempt_{attempt}"
    config = ATTEMPTS[attempt]
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
        "meta_iteration": 4,
        "agent": 1,
        "attempt": attempt,
        "candidate": {
            "expression_template": config["template"],
            "parameter_names": list(PARAMETER_NAMES),
            "contrast_subtraction": config["contrast"],
            "tree_node_counting": (
                "fixed incumbent total nodes + product root + complete 12-node "
                "incremental exponential contrast; no common-subexpression folding"
            ),
            "incumbent_tree_nodes": INCUMBENT_NODES,
            "incremental_nodes": ADDED_NODES,
            "tree_nodes_by_baseline": {
                key: nodes + ADDED_NODES for key, nodes in INCUMBENT_NODES.items()
            },
            "incumbent_coefficients_fixed": True,
            "nonlocal": True,
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
            "parameter_bounds": [list(BOUNDS)],
            "seed_formula": f"{10100 + attempt * 10} + baseline_index",
            "baseline_order": [key for key, _ in BASELINE_ITEMS],
            "feasibility_check": "automodel.search_utils.is_nonlocal_feasible",
            "wave_speed_bound": SPEED_BOUND,
            "convolution_implementation": "exact dctkit.dec.cochain.convolution",
            "execution": "synchronous",
            "device": str(jax.devices()[0]),
            "pythonpath": "current checkout: <repo>/src:<repo>",
            "fitness": "validation E_data + 0.01*total_tree_nodes",
        },
        "meta3_incumbent_validation": INCUMBENT_VALIDATION,
        "records": records,
        "attempt_runtime_seconds": aggregate_runtime,
        "last_report_write_process_elapsed_seconds": elapsed,
        "attempt_peak_rss_mb": peak_rss,
        "completed_baselines": sum(r["status"] == "completed" for r in records),
        "failed_baselines": sum(r["status"] != "completed" for r in records),
        "nonlocal": True,
        "test_evaluated": False,
    }
    (attempt_dir / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    training = [
        f"# Attempt {attempt} training",
        "",
        f"Template: `{config['template']}`.",
        "",
        "All per-FD meta-3 incumbent coefficients were held fixed. Only `a` was",
        "fitted on full I80 training times 0--63. Two deterministic Powell starts",
        f"used at most {MAX_EVALUATIONS} evaluations each with bounds `{list(BOUNDS)}`.",
        f"Convolution is exact `C.convolution`; speed bound is `{SPEED_BOUND}`.",
        "",
        "| Baseline | Full expression | Nodes | New constant | E_rho | E_v | E_data | Feasible | Evaluations | Seed | Fit time (s) | RSS (MB) |",
        "|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|",
    ]
    evaluation = [
        f"# Attempt {attempt} evaluation",
        "",
        "Validation uses full I80 times 64--107. The held-out test interval was",
        "not evaluated. Fitness is `E_data + 0.01*total_tree_nodes`, including the",
        f"fixed incumbent. Nonlocal speed uses `{SPEED_BOUND}`.",
        "",
        "| Baseline | Full expression | New constant | E_rho | E_v | E_data | Fitness | Meta-3 fitness | Delta | Finite/feasible | Runtime (s) | RSS (MB) |",
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
        constant = constants_text(fit)
        expression = fit["expression"].replace("|", "\\|")
        incumbent = record["incumbent_validation"]
        delta = fit["validation_fitness"] - incumbent["fitness"]
        training.append(
            f"| {record['baseline']} | `{expression}` | {fit['tree_nodes']} | {constant} | "
            f"{train['rho_error']:.6f} | {train['velocity_error']:.6f} | "
            f"{train['data_error']:.6f} | {'yes' if fit['feasible'] else 'no'} | "
            f"{fit['optimizer_evaluations']} | {fit['seed']} | "
            f"{fit['fit_runtime_seconds']:.3f} | {train['peak_rss_mb']:.1f} |"
        )
        evaluation.append(
            f"| {record['baseline']} | `{expression}` | {constant} | "
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
        ["", "All candidates are `nonlocal = true` and use exact `C.convolution`.", "`test_evaluated = false`.", ""]
    )
    (attempt_dir / "training.md").write_text("\n".join(training), encoding="utf-8")
    (attempt_dir / "evaluation.md").write_text("\n".join(evaluation), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, choices=sorted(ATTEMPTS))
    parser.add_argument("--baseline", choices=[key for key, _ in BASELINE_ITEMS])
    args = parser.parse_args()
    evaluator = I80PredictionEvaluator()
    for attempt in sorted(ATTEMPTS):
        if args.attempt is not None and attempt != args.attempt:
            continue
        model = load_model(attempt)
        result_path = ROOT / f"attempt_{attempt}" / "results.json"
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
            seed = 10100 + attempt * 10 + baseline_index
            expression = model.expression(baseline_key)
            tree_nodes = INCUMBENT_NODES[baseline_key] + ADDED_NODES
            correction = model.make_correction(baseline_key)
            print(
                f"attempt={attempt} baseline={baseline.name} seed={seed} nodes={tree_nodes}",
                flush=True,
            )
            try:
                fit = fit_candidate(
                    evaluator=evaluator,
                    baseline=baseline,
                    correction=correction,
                    expression=expression,
                    tree_nodes=tree_nodes,
                    parameter_names=PARAMETER_NAMES,
                    bounds=(BOUNDS,),
                    seed=seed,
                    restarts=RESTARTS,
                    max_evaluations=MAX_EVALUATIONS,
                    feasibility_check=is_nonlocal_feasible,
                )
                diagnostics = feasibility_diagnostics(
                    evaluator, baseline, correction, fit.parameters
                )
                records.append(
                    {
                        "baseline_key": baseline_key,
                        "baseline": baseline.name,
                        "status": "completed",
                        "fit": fit.to_dict(),
                        "incumbent_validation": INCUMBENT_VALIDATION[baseline_key],
                        "validation_fitness_delta_vs_meta3_incumbent": (
                            fit.validation_fitness
                            - INCUMBENT_VALIDATION[baseline_key]["fitness"]
                        ),
                        "diagnostics": {
                            "homogeneous_feasibility": diagnostics,
                            "wave_speed_bound": SPEED_BOUND,
                            "simulation_finite_train": fit.train.finite,
                            "simulation_finite_validation": fit.validation.finite,
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
            write_reports(attempt, records, time.perf_counter() - attempt_start)


if __name__ == "__main__":
    main()
