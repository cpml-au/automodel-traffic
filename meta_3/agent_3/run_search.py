"""Run incumbent-plus-convolution searches without accessing the test split."""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

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
    "greenshields": 12,
    "idm": 1,
    "weidmann": 10,
    "triangular": 1,
    "del_castillo": 2,
}
ATTEMPTS = {
    1: {
        "template": "g_inc*exp(a*conv_3(rho,ones))",
        "parameter_names": ("a",),
        # Full tree: incumbent, product root, and six-node exponential factor.
        "added_nodes": 7,
    },
    2: {
        "template": "g_inc*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))",
        "parameter_names": ("a",),
        # Full tree adds product root and a twelve-node exponential factor.
        "added_nodes": 13,
    },
    3: {
        "template": (
            "g_inc*exp(a*conv_3(rho,ones)"
            "+b*(conv_3(rho,ones)-3*conv_1(rho,ones)))"
        ),
        "parameter_names": ("a", "b"),
        # Full tree adds product root and an eighteen-node exponential factor.
        "added_nodes": 19,
    },
}
BOUNDS = (-300.0, 300.0)
RESTARTS = 2
MAX_EVALUATIONS = 45
SPEED_BOUND = "row-wise sum(abs(full flux Jacobian)); includes off-diagonal coupling"


def load_model(attempt: int):
    path = ROOT / f"attempt_{attempt}" / "model.py"
    spec = importlib.util.spec_from_file_location(
        f"meta3_agent3_attempt{attempt}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def feasibility_diagnostics(evaluator, baseline, correction, parameters) -> dict:
    """Report the homogeneous-state audit used by is_nonlocal_feasible."""

    upper = physical_density_limit(baseline)
    densities = jnp.linspace(max(1e-5, upper / 640.0), upper, 64)
    center = evaluator.S.num_nodes // 2
    center_multipliers = []
    all_multipliers = []
    velocities = []
    for density in densities:
        rho = C.CochainP0(
            evaluator.S, density * jnp.ones(evaluator.S.num_nodes)
        )
        multiplier = correction(rho, parameters).coeffs.flatten()
        all_multipliers.append(multiplier)
        center_multiplier = multiplier[center]
        center_multipliers.append(center_multiplier)
        velocities.append(
            baseline.velocity(
                jnp.asarray([density]), *baseline.coefficients
            )[0]
            * center_multiplier
        )
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
    nonnegative_velocity = bool(jnp.all(velocities >= -tolerance))
    nonincreasing_velocity = bool(jnp.all(velocity_differences <= tolerance))
    return {
        "method": "64 homogeneous fields; central node for constitutive velocity",
        "samples": 64,
        "physical_density_domain": [float(densities[0]), float(upper)],
        "finite_multiplier_and_velocity": finite,
        "positive_multiplier_all_nodes": positive,
        "nonnegative_central_corrected_velocity": nonnegative_velocity,
        "nonincreasing_central_corrected_velocity": nonincreasing_velocity,
        "minimum_multiplier_all_nodes": float(jnp.min(all_multipliers)),
        "maximum_multiplier_all_nodes": float(jnp.max(all_multipliers)),
        "minimum_center_multiplier": float(jnp.min(center_multipliers)),
        "maximum_center_multiplier": float(jnp.max(center_multipliers)),
        "minimum_corrected_velocity": float(jnp.min(velocities)),
        "maximum_velocity_forward_difference": float(jnp.max(velocity_differences)),
        "tolerance": tolerance,
        "passed": bool(
            finite and positive and nonnegative_velocity and nonincreasing_velocity
        ),
    }


def format_parameters(names, values) -> str:
    return ", ".join(f"{name}={value:.10g}" for name, value in zip(names, values))


def write_reports(attempt: int, records: list[dict], elapsed: float) -> None:
    attempt_dir = ROOT / f"attempt_{attempt}"
    config = ATTEMPTS[attempt]
    completed_fits = [
        record["fit"] for record in records if record["status"] == "completed"
    ]
    aggregate_fit_runtime = sum(
        fit["fit_runtime_seconds"] for fit in completed_fits
    )
    measured_peak_rss = max(
        (
            split["peak_rss_mb"]
            for fit in completed_fits
            for split in (fit["train"], fit["validation"])
        ),
        default=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    payload = {
        "meta_iteration": 3,
        "agent": 3,
        "attempt": attempt,
        "candidate": {
            "expression_template": config["template"],
            "parameter_names": list(config["parameter_names"]),
            "tree_node_counting": (
                "incumbent tree nodes + product root + complete incremental factor; "
                "repeated convolution subtrees are counted separately"
            ),
            "tree_nodes_by_baseline": {
                key: INCUMBENT_NODES[key] + config["added_nodes"]
                for key, _ in BASELINE_ITEMS
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
            "parameter_bounds": [list(BOUNDS)] * len(config["parameter_names"]),
            "seed_formula": f"{9100 + attempt * 10} + baseline_index",
            "baseline_order": [key for key, _ in BASELINE_ITEMS],
            "feasibility_check": "automodel.search_utils.is_nonlocal_feasible",
            "wave_speed_bound": SPEED_BOUND,
            "device": "CPU",
            "pythonpath": "current checkout: <repo>/src:<repo>",
            "fitness": "validation E_data + 0.01*total_tree_nodes",
        },
        "records": records,
        "attempt_runtime_seconds": aggregate_fit_runtime,
        "attempt_runtime_measurement": (
            "sum of completed per-baseline fit runtimes; valid after merged IDM rerun"
        ),
        "last_report_write_process_elapsed_seconds": elapsed,
        "attempt_peak_rss_mb": measured_peak_rss,
        "completed_baselines": sum(r["status"] == "completed" for r in records),
        "failed_baselines": sum(r["status"] != "completed" for r in records),
        "nonlocal": True,
        "test_evaluated": False,
    }
    (attempt_dir / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    training_lines = [
        f"# Attempt {attempt} training",
        "",
        f"Template: `{config['template']}`.",
        "",
        "Every incumbent coefficient was held fixed. Only the new convolution",
        f"coefficient(s) `{', '.join(config['parameter_names'])}` were fitted on full",
        "I80 training times 0--63. Two deterministic Powell starts used at most",
        f"{MAX_EVALUATIONS} evaluations each with bounds `{list(BOUNDS)}`.",
        "",
        "| Baseline | Full expression | Nodes | New constants | E_rho | E_v | E_data | Feasible | Evaluations | Seed | Fit time (s) | RSS (MB) |",
        "|---|---|---:|---|---:|---:|---:|:---:|---:|---:|---:|---:|",
    ]
    evaluation_lines = [
        f"# Attempt {attempt} evaluation",
        "",
        "Validation uses full I80 times 64--107. The held-out test interval was",
        "not evaluated. Fitness is `E_data + 0.01*total_tree_nodes`.",
        f"The nonlocal solver speed is bounded by `{SPEED_BOUND}`.",
        "",
        "| Baseline | Full expression | New constants | E_rho | E_v | E_data | Fitness | Finite/feasible | Runtime (s) | RSS (MB) |",
        "|---|---|---|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for record in records:
        if record["status"] != "completed":
            error = record["error"].replace("|", "\\|")
            training_lines.append(
                f"| {record['baseline']} | FAILURE: {error} | | | | | | no | | {record['seed']} | | |"
            )
            evaluation_lines.append(
                f"| {record['baseline']} | FAILURE: {error} | | | | | | | no | | |"
            )
            continue
        fit = record["fit"]
        train = fit["train"]
        validation = fit["validation"]
        constants = format_parameters(fit["parameter_names"], fit["parameters"])
        expression = fit["expression"].replace("|", "\\|")
        training_lines.append(
            f"| {record['baseline']} | `{expression}` | {fit['tree_nodes']} | "
            f"{constants} | {train['rho_error']:.6f} | {train['velocity_error']:.6f} | "
            f"{train['data_error']:.6f} | {'yes' if fit['feasible'] else 'no'} | "
            f"{fit['optimizer_evaluations']} | {fit['seed']} | "
            f"{fit['fit_runtime_seconds']:.3f} | {train['peak_rss_mb']:.1f} |"
        )
        evaluation_lines.append(
            f"| {record['baseline']} | `{expression}` | {constants} | "
            f"{validation['rho_error']:.6f} | {validation['velocity_error']:.6f} | "
            f"{validation['data_error']:.6f} | {fit['validation_fitness']:.6f} | "
            f"{'yes' if fit['feasible'] and validation['finite'] else 'no'} | "
            f"{validation['runtime_seconds']:.3f} | {validation['peak_rss_mb']:.1f} |"
        )
    training_lines.extend(
        [
            "",
            f"Aggregate completed fit runtime: {aggregate_fit_runtime:.3f} seconds.",
            "`test_evaluated = false`.",
            "",
        ]
    )
    evaluation_lines.extend(
        ["", "All candidates are labeled `nonlocal = true`.", "`test_evaluated = false`.", ""]
    )
    (attempt_dir / "training.md").write_text(
        "\n".join(training_lines), encoding="utf-8"
    )
    (attempt_dir / "evaluation.md").write_text(
        "\n".join(evaluation_lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, choices=sorted(ATTEMPTS))
    parser.add_argument("--baseline", choices=[key for key, _ in BASELINE_ITEMS])
    args = parser.parse_args()
    evaluator = I80PredictionEvaluator()
    for attempt, config in ATTEMPTS.items():
        if args.attempt is not None and attempt != args.attempt:
            continue
        model = load_model(attempt)
        result_path = ROOT / f"attempt_{attempt}" / "results.json"
        records: list[dict] = []
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
            seed = 9100 + attempt * 10 + baseline_index
            expression = model.expression(baseline_key)
            tree_nodes = INCUMBENT_NODES[baseline_key] + config["added_nodes"]
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
                    parameter_names=config["parameter_names"],
                    bounds=(BOUNDS,) * len(config["parameter_names"]),
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
