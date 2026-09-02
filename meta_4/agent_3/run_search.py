"""Run meta-4 protected-square-root controls without accessing I80 test data."""

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
from automodel.search_utils import fit_candidate, is_nonlocal_feasible, physical_density_limit


ROOT = Path(__file__).resolve().parent
BASELINE_ITEMS = tuple(BASELINES.items())
INCUMBENT_NODES = {
    "greenshields": 25,
    "idm": 14,
    "weidmann": 10,
    "triangular": 1,
    "del_castillo": 21,
}
META3_FITNESS = {
    "greenshields": 6.953732490539551,
    "idm": 5.417018070220947,
    "weidmann": 6.322293,
    "triangular": 6.457489,
    "del_castillo": 5.73022123336792,
}
ATTEMPTS = {
    1: {
        "template": "g_inc*exp(a*SqrtP0(rho))",
        "parameter_names": ("a",),
        "added_nodes": 6,
    },
    2: {
        "template": "g_inc*exp(a*SqrtP0(rho)+b*rho)",
        "parameter_names": ("a", "b"),
        "added_nodes": 10,
    },
    3: {
        "template": "g_inc*exp(a*SqrtP0(rho)+b*rho^2)",
        "parameter_names": ("a", "b"),
        "added_nodes": 12,
    },
}
BOUNDS = (-5.0, 5.0)
RESTARTS = 2
MAX_EVALUATIONS = 60
SPEED_BOUND = "row-wise sum(abs(full flux Jacobian)); includes off-diagonal coupling"


def load_model(attempt: int):
    path = ROOT / f"attempt_{attempt}" / "model.py"
    spec = importlib.util.spec_from_file_location(f"meta4_agent3_attempt{attempt}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def feasibility_diagnostics(evaluator, baseline, correction, parameters) -> dict:
    upper = physical_density_limit(baseline)
    densities = jnp.linspace(max(1e-5, upper / 640.0), upper, 64)
    center = evaluator.S.num_nodes // 2
    center_multipliers = []
    all_multipliers = []
    velocities = []
    for density in densities:
        rho = C.CochainP0(evaluator.S, density * jnp.ones(evaluator.S.num_nodes))
        multiplier = correction(rho, parameters).coeffs.flatten()
        all_multipliers.append(multiplier)
        center_multipliers.append(multiplier[center])
        velocities.append(
            baseline.velocity(jnp.asarray([density]), *baseline.coefficients)[0]
            * multiplier[center]
        )
    all_multipliers = jnp.stack(all_multipliers)
    center_multipliers = jnp.asarray(center_multipliers)
    velocities = jnp.asarray(velocities)
    differences = jnp.diff(velocities)
    tolerance = 1e-7
    finite = bool(
        jnp.all(jnp.isfinite(all_multipliers)) & jnp.all(jnp.isfinite(velocities))
    )
    positive = bool(jnp.all(all_multipliers > 0))
    nonnegative_velocity = bool(jnp.all(velocities >= -tolerance))
    nonincreasing_velocity = bool(jnp.all(differences <= tolerance))
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
        "maximum_velocity_forward_difference": float(jnp.max(differences)),
        "tolerance": tolerance,
        "passed": bool(finite and positive and nonnegative_velocity and nonincreasing_velocity),
    }


def format_parameters(names, values) -> str:
    return ", ".join(f"{name}={value:.10g}" for name, value in zip(names, values))


def write_reports(attempt: int, records: list[dict], elapsed: float) -> None:
    attempt_dir = ROOT / f"attempt_{attempt}"
    config = ATTEMPTS[attempt]
    completed = [r["fit"] for r in records if r["status"] == "completed"]
    aggregate_runtime = sum(f["fit_runtime_seconds"] for f in completed)
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
        "agent": 3,
        "attempt": attempt,
        "candidate": {
            "expression_template": config["template"],
            "parameter_names": list(config["parameter_names"]),
            "protected_sqrt_implementation": "jnp.sqrt(jnp.maximum(rho, 0))",
            "gp_equivalent": "SqrtP0(rho)",
            "tree_node_counting": (
                "complete fixed meta-3 incumbent tree plus product root and complete "
                "incremental exponential factor"
            ),
            "incumbent_nodes_by_baseline": INCUMBENT_NODES,
            "tree_nodes_by_baseline": {
                key: INCUMBENT_NODES[key] + config["added_nodes"]
                for key, _ in BASELINE_ITEMS
            },
            "incumbent_coefficients_fixed": True,
            "nonlocal_by_baseline": {
                "greenshields": True,
                "idm": True,
                "weidmann": False,
                "triangular": False,
                "del_castillo": True,
            },
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
            "seed_formula": f"{12100 + attempt * 10} + baseline_index",
            "baseline_order": [key for key, _ in BASELINE_ITEMS],
            "feasibility_check": "automodel.search_utils.is_nonlocal_feasible",
            "wave_speed_bound": SPEED_BOUND,
            "device": "CPU",
            "execution": "synchronous; current checkout",
            "pythonpath": "current checkout: <repo>/src:<repo>",
            "fitness": "validation E_data + 0.01*total_tree_nodes",
        },
        "meta3_incumbent_validation_fitness": META3_FITNESS,
        "records": records,
        "attempt_runtime_seconds": aggregate_runtime,
        "attempt_process_elapsed_seconds": elapsed,
        "attempt_peak_rss_mb": peak_rss,
        "completed_baselines": sum(r["status"] == "completed" for r in records),
        "failed_baselines": sum(r["status"] != "completed" for r in records),
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
        "All meta-3 incumbent coefficients were fixed. The protected square root",
        "was executed as `jnp.sqrt(jnp.maximum(rho, 0))`, corresponding to GP",
        "`SqrtP0(rho)`. New constants were fitted on full I80 train times 0--63",
        f"using two Powell restarts, at most {MAX_EVALUATIONS} evaluations per",
        "restart, and bounds `[-5, 5]`.",
        "",
        "| Baseline | Full expression | Nodes | Parameters | E_rho | E_v | E_data | Feasible | Success | Evaluations | Seed | Fit time (s) | RSS (MB) |",
        "|---|---|---:|---|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|",
    ]
    evaluation = [
        f"# Attempt {attempt} evaluation",
        "",
        "Validation uses full I80 times 64--107. Test times 108--179 were not",
        "evaluated. Fitness is `E_data + 0.01*total_tree_nodes`.",
        "",
        "| Baseline | Full expression | Parameters | E_rho | E_v | E_data | Fitness | Meta-3 fitness | Change | Finite/feasible | Runtime (s) | RSS (MB) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for record in records:
        if record["status"] != "completed":
            error = record["error"].replace("|", "\\|")
            training.append(f"| {record['baseline']} | FAILURE: {error} | | | | | | no | no | | {record['seed']} | | |")
            evaluation.append(f"| {record['baseline']} | FAILURE: {error} | | | | | | | | | no | | |")
            continue
        fit = record["fit"]
        train = fit["train"]
        validation = fit["validation"]
        constants = format_parameters(fit["parameter_names"], fit["parameters"])
        expression = fit["expression"].replace("|", "\\|")
        delta = fit["validation_fitness"] - META3_FITNESS[record["baseline_key"]]
        training.append(
            f"| {record['baseline']} | `{expression}` | {fit['tree_nodes']} | {constants} | "
            f"{train['rho_error']:.6f} | {train['velocity_error']:.6f} | {train['data_error']:.6f} | "
            f"{'yes' if fit['feasible'] else 'no'} | {'yes' if fit['optimizer_success'] else 'no'} | "
            f"{fit['optimizer_evaluations']} | {fit['seed']} | {fit['fit_runtime_seconds']:.3f} | "
            f"{train['peak_rss_mb']:.1f} |"
        )
        evaluation.append(
            f"| {record['baseline']} | `{expression}` | {constants} | "
            f"{validation['rho_error']:.6f} | {validation['velocity_error']:.6f} | "
            f"{validation['data_error']:.6f} | {fit['validation_fitness']:.6f} | "
            f"{META3_FITNESS[record['baseline_key']]:.6f} | {delta:+.6f} | "
            f"{'yes' if fit['feasible'] and validation['finite'] else 'no'} | "
            f"{validation['runtime_seconds']:.3f} | {validation['peak_rss_mb']:.1f} |"
        )
    training.extend(["", f"Aggregate fit runtime: {aggregate_runtime:.3f} seconds.", "`test_evaluated = false`.", ""])
    evaluation.extend(["", f"Wave-speed bound: `{SPEED_BOUND}`.", "`test_evaluated = false`.", ""])
    (attempt_dir / "training.md").write_text("\n".join(training), encoding="utf-8")
    (attempt_dir / "evaluation.md").write_text("\n".join(evaluation), encoding="utf-8")


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
            prior = json.loads(result_path.read_text(encoding="utf-8"))
            records = [r for r in prior.get("records", []) if r["baseline_key"] != args.baseline]
        attempt_start = time.perf_counter()
        for baseline_index, (baseline_key, baseline) in enumerate(BASELINE_ITEMS):
            if args.baseline is not None and baseline_key != args.baseline:
                continue
            seed = 12100 + attempt * 10 + baseline_index
            expression = model.expression(baseline_key)
            tree_nodes = INCUMBENT_NODES[baseline_key] + config["added_nodes"]
            correction = model.make_correction(baseline_key)
            print(f"attempt={attempt} baseline={baseline.name} seed={seed} nodes={tree_nodes}", flush=True)
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
                fit_dict = fit.to_dict()
                records.append(
                    {
                        "baseline_key": baseline_key,
                        "baseline": baseline.name,
                        "status": "completed",
                        "fit": fit_dict,
                        "diagnostics": {
                            "homogeneous_feasibility": feasibility_diagnostics(
                                evaluator, baseline, correction, fit.parameters
                            ),
                            "protected_sqrt": {
                                "implementation": "jnp.sqrt(jnp.maximum(rho, 0))",
                                "gp_equivalent": "SqrtP0(rho)",
                            },
                            "wave_speed_bound": SPEED_BOUND,
                            "simulation_finite_train": fit.train.finite,
                            "simulation_finite_validation": fit.validation.finite,
                        },
                        "meta3_incumbent_validation_fitness": META3_FITNESS[baseline_key],
                        "fitness_change_vs_meta3": fit.validation_fitness - META3_FITNESS[baseline_key],
                        "nonlocal_incumbent": baseline_key in {"greenshields", "idm", "del_castillo"},
                        "test_evaluated": False,
                    }
                )
                print(
                    f"completed baseline={baseline.name} parameters={fit.parameters} "
                    f"train={fit.train.data_error:.6f} validation={fit.validation.data_error:.6f} "
                    f"fitness={fit.validation_fitness:.6f}",
                    flush=True,
                )
            except Exception as exc:
                records.append(
                    {
                        "baseline_key": baseline_key,
                        "baseline": baseline.name,
                        "status": "failed",
                        "seed": seed,
                        "error": f"{type(exc).__name__}: {exc}",
                        "test_evaluated": False,
                    }
                )
                print(f"failed baseline={baseline.name}: {type(exc).__name__}: {exc}", flush=True)
            order = {key: index for index, (key, _) in enumerate(BASELINE_ITEMS)}
            records.sort(key=lambda record: order[record["baseline_key"]])
            write_reports(attempt, records, time.perf_counter() - attempt_start)


if __name__ == "__main__":
    main()
