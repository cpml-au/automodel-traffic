"""Run meta-5 density-gated convolution searches, train/validation only."""

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
from dctkit.dec import cochain as C
from deap import gp

from automodel.model import BASELINES
from automodel.pipeline import I80PredictionEvaluator
from automodel.search_utils import fit_candidate, is_nonlocal_feasible, physical_density_limit
from automodel.typed_primitives import build_traffic_pset
from meta_5.agent_2 import common


ROOT = Path(__file__).resolve().parent
BASELINE_ITEMS = tuple(BASELINES.items())
RESTARTS = 2
MAX_EVALUATIONS = 60
SPEED_BOUND = "row-wise sum(abs(full flux Jacobian)); includes off-diagonal convolution coupling"


def load_model(attempt: int):
    path = ROOT / f"attempt_{attempt}" / "model.py"
    spec = importlib.util.spec_from_file_location(f"meta5_agent2_attempt{attempt}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_grammar(models):
    pset, _ = build_traffic_pset()
    pset.addTerminal(0.0, float, "a")
    pset.addTerminal(0.0, float, "b")
    pset.addTerminal(3.0, float, "three")
    counts = {}
    for model in models:
        tree = gp.PrimitiveTree.from_string(model.TYPED_FACTOR, pset)
        gp.compile(tree, pset)
        expected = model.ADDED_NODES - 1
        if len(tree) != expected:
            raise RuntimeError(
                f"attempt {model.ATTEMPT}: compiled {len(tree)} factor nodes, expected {expected}"
            )
        counts[model.ATTEMPT] = len(tree)
    return counts


def feasibility_diagnostics(evaluator, baseline, correction, parameters):
    upper = physical_density_limit(baseline)
    densities = jnp.linspace(max(1e-5, upper / 640.0), upper, 64)
    center = evaluator.S.num_nodes // 2
    all_multipliers, center_multipliers, velocities = [], [], []
    for density in densities:
        rho = C.CochainP0(evaluator.S, density * jnp.ones(evaluator.S.num_nodes))
        multiplier = correction(rho, parameters).coeffs.flatten()
        all_multipliers.append(multiplier)
        center_multipliers.append(multiplier[center])
        base_velocity = baseline.velocity(jnp.asarray([density]), *baseline.coefficients)[0]
        velocities.append(base_velocity * multiplier[center])
    all_multipliers = jnp.stack(all_multipliers)
    center_multipliers = jnp.asarray(center_multipliers)
    velocities = jnp.asarray(velocities)
    differences = jnp.diff(velocities)
    tolerance = 1e-7
    finite = bool(jnp.all(jnp.isfinite(all_multipliers)) & jnp.all(jnp.isfinite(velocities)))
    positive = bool(jnp.all(all_multipliers > 0))
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
        "minimum_multiplier_all_nodes": float(jnp.min(all_multipliers)),
        "maximum_multiplier_all_nodes": float(jnp.max(all_multipliers)),
        "minimum_center_multiplier": float(jnp.min(center_multipliers)),
        "maximum_center_multiplier": float(jnp.max(center_multipliers)),
        "minimum_corrected_velocity": float(jnp.min(velocities)),
        "maximum_velocity_forward_difference": float(jnp.max(differences)),
        "tolerance": tolerance,
        "passed": finite and positive and nonnegative and nonincreasing,
    }


def parameters_text(names, values):
    return ", ".join(f"{name}={value:.10g}" for name, value in zip(names, values))


def write_reports(model, factor_nodes, records, elapsed):
    attempt_dir = ROOT / f"attempt_{model.ATTEMPT}"
    completed = [r["fit"] for r in records if r["status"] == "completed"]
    payload = {
        "meta_iteration": 5,
        "agent": 2,
        "attempt": model.ATTEMPT,
        "candidate": {
            "symbolic_factor": model.SYMBOLIC_FACTOR,
            "typed_gp_factor": model.TYPED_FACTOR,
            "typed_gp_factor_nodes": factor_nodes,
            "attachment_node": "CMulP0",
            "added_nodes_including_attachment": model.ADDED_NODES,
            "incumbent_nodes": common.INCUMBENT_NODES,
            "total_nodes_by_baseline": {
                key: common.INCUMBENT_NODES[key] + model.ADDED_NODES
                for key, _ in BASELINE_ITEMS
            },
            "tree_node_counting": "all incumbent nodes plus attachment and complete typed factor; repeated C subtrees counted separately",
            "incumbent_coefficients_fixed": True,
            "dec_implementation": ["C.convolution", "C.cochain_mul"],
            "nonlocal": True,
        },
        "fit_protocol": {
            "dataset": "I80",
            "problem": "prediction",
            "training_split": "full train times 0-63 (4,800 rows)",
            "validation_split": "full validation times 64-107 (3,300 rows)",
            "test_split": "not evaluated (times 108-179)",
            "test_evaluated": False,
            "optimizer": "scipy.optimize.minimize/Powell via fit_candidate",
            "objective": "training E_data",
            "restarts": RESTARTS,
            "max_evaluations_per_restart": MAX_EVALUATIONS,
            "parameter_bounds": [list(x) for x in model.BOUNDS],
            "seed_formula": f"{13100 + model.ATTEMPT * 10}+baseline_index",
            "baseline_order": [key for key, _ in BASELINE_ITEMS],
            "selection": "validation E_data + 0.01*total_tree_nodes",
            "feasibility_check": "automodel.search_utils.is_nonlocal_feasible",
            "wave_speed_bound": SPEED_BOUND,
        },
        "records": records,
        "attempt_fit_runtime_seconds": sum(x["fit_runtime_seconds"] for x in completed),
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
        },
        "test_evaluated": False,
    }
    (attempt_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    train_lines = [
        f"# Attempt {model.ATTEMPT} training", "",
        f"Factor: `{model.SYMBOLIC_FACTOR}`. Full train times 0--63; incumbent constants fixed.", "",
        "| FD | Full expression | Nodes | New parameters | E_rho | E_v | E_data | Evals | Optimizer | Feasible | Fit time (s) | RSS (MB) | Seed |",
        "|---|---|---:|---|---:|---:|---:|---:|---|:---:|---:|---:|---:|",
    ]
    eval_lines = [
        f"# Attempt {model.ATTEMPT} evaluation", "",
        "Full validation times 64--107; fitness includes every tree node. Test was not evaluated.",
        f"Nonlocal speed bound: `{SPEED_BOUND}`.", "",
        "| FD | Parameters | E_rho | E_v | E_data | Fitness | Incumbent fitness | Change | Finite/feasible | Runtime (s) | RSS (MB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for record in records:
        if record["status"] != "completed":
            error = record["error"].replace("|", "/")
            train_lines.append(f"| {record['baseline']} | FAILED: {error} | | | | | | | | no | | | {record['seed']} |")
            eval_lines.append(f"| {record['baseline']} | FAILED: {error} | | | | | | | no | | |")
            continue
        fit = record["fit"]
        train, validation = fit["train"], fit["validation"]
        params = parameters_text(fit["parameter_names"], fit["parameters"])
        optimizer = ("success" if fit["optimizer_success"] else "stopped") + ": " + fit["optimizer_message"].replace("|", "/")
        train_lines.append(
            f"| {record['baseline']} | `{fit['expression']}` | {fit['tree_nodes']} | {params} | "
            f"{train['rho_error']:.6f} | {train['velocity_error']:.6f} | {train['data_error']:.6f} | "
            f"{fit['optimizer_evaluations']} | {optimizer} | {'yes' if fit['feasible'] else 'no'} | "
            f"{fit['fit_runtime_seconds']:.3f} | {train['peak_rss_mb']:.1f} | {fit['seed']} |"
        )
        eval_lines.append(
            f"| {record['baseline']} | {params} | {validation['rho_error']:.6f} | "
            f"{validation['velocity_error']:.6f} | {validation['data_error']:.6f} | "
            f"{fit['validation_fitness']:.6f} | {record['incumbent_fitness']:.6f} | "
            f"{record['fitness_change_from_incumbent']:+.6f} | "
            f"{'yes' if fit['feasible'] and validation['finite'] else 'no'} | "
            f"{validation['runtime_seconds']:.3f} | {validation['peak_rss_mb']:.1f} |"
        )
    train_lines.extend(["", "`test_evaluated = false`.", ""])
    eval_lines.extend(["", "`test_evaluated = false`.", ""])
    (attempt_dir / "training.md").write_text("\n".join(train_lines), encoding="utf-8")
    (attempt_dir / "evaluation.md").write_text("\n".join(eval_lines), encoding="utf-8")


def write_summary():
    rows = []
    for attempt in (1, 2, 3):
        path = ROOT / f"attempt_{attempt}" / "results.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.extend((attempt, r) for r in payload["records"] if r["status"] == "completed" and r["fit"]["feasible"])
    lines = [
        "# Meta-5 agent 2 summary", "",
        "Density-gated DEC convolution contrasts were appended to every fixed global meta-4 incumbent.",
        "All fits used full train and full validation only; the held-out test was never evaluated.", "",
        "| FD | Best attempt | Expression | Parameters | Nodes | Validation E_data | Fitness | Incumbent fitness | Change |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    winners = {}
    for key, baseline in BASELINE_ITEMS:
        candidates = [(a, r) for a, r in rows if r["baseline_key"] == key]
        if not candidates:
            lines.append(
                f"| {baseline.name} | -- | no feasible candidate; retain incumbent | -- | "
                f"{common.INCUMBENT_NODES[key]} | -- | {common.INCUMBENT_FITNESS[key]:.6f} | "
                f"{common.INCUMBENT_FITNESS[key]:.6f} | +0.000000 |"
            )
            continue
        attempt, row = min(candidates, key=lambda x: x[1]["fit"]["validation_fitness"])
        winners[key] = row
        fit = row["fit"]
        lines.append(
            f"| {baseline.name} | {attempt} | `{fit['expression']}` | "
            f"{parameters_text(fit['parameter_names'], fit['parameters'])} | {fit['tree_nodes']} | "
            f"{fit['validation']['data_error']:.6f} | {fit['validation_fitness']:.6f} | "
            f"{row['incumbent_fitness']:.6f} | {row['fitness_change_from_incumbent']:+.6f} |"
        )
    improved = [key for key, row in winners.items() if row["fitness_change_from_incumbent"] < 0]
    lines.extend([
        "", f"Lineage winners over the incumbent: {', '.join(improved) if improved else 'none'}.",
        f"Nonlocal speed bound: `{SPEED_BOUND}`.",
        "Selection required homogeneous nonlocal feasibility and finite full simulations.",
        "Ten candidates passed; five pathological Weidmann/Triangular/Del Castillo endpoints were rejected.",
        "`test_evaluated = false`.", "",
    ])
    (ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, choices=(1, 2, 3))
    parser.add_argument("--baseline", choices=[key for key, _ in BASELINE_ITEMS])
    args = parser.parse_args()
    models = [load_model(i) for i in (1, 2, 3)]
    factor_counts = validate_grammar(models)
    evaluator = I80PredictionEvaluator()
    order = {key: i for i, (key, _) in enumerate(BASELINE_ITEMS)}
    for model in models:
        if args.attempt is not None and model.ATTEMPT != args.attempt:
            continue
        result_path = ROOT / f"attempt_{model.ATTEMPT}" / "results.json"
        records = []
        if args.baseline is not None and result_path.exists():
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            records = [r for r in previous["records"] if r["baseline_key"] != args.baseline]
        started = time.perf_counter()
        for baseline_index, (baseline_key, baseline) in enumerate(BASELINE_ITEMS):
            if args.baseline is not None and baseline_key != args.baseline:
                continue
            seed = 13100 + model.ATTEMPT * 10 + baseline_index
            correction = model.make_correction(baseline_key)
            expression = model.expression(baseline_key)
            nodes = common.INCUMBENT_NODES[baseline_key] + model.ADDED_NODES
            print(f"attempt={model.ATTEMPT} baseline={baseline.name} seed={seed} nodes={nodes}", flush=True)
            try:
                fit = fit_candidate(
                    evaluator=evaluator, baseline=baseline, correction=correction,
                    expression=expression, tree_nodes=nodes,
                    parameter_names=model.PARAMETER_NAMES, bounds=model.BOUNDS,
                    seed=seed, restarts=RESTARTS, max_evaluations=MAX_EVALUATIONS,
                    feasibility_check=is_nonlocal_feasible,
                )
                diagnostics = feasibility_diagnostics(evaluator, baseline, correction, fit.parameters)
                records.append({
                    "baseline_key": baseline_key, "baseline": baseline.name,
                    "status": "completed", "fit": fit.to_dict(),
                    "incumbent_fitness": common.INCUMBENT_FITNESS[baseline_key],
                    "fitness_change_from_incumbent": fit.validation_fitness - common.INCUMBENT_FITNESS[baseline_key],
                    "diagnostics": {
                        "homogeneous_feasibility": diagnostics,
                        "wave_speed_bound": SPEED_BOUND,
                        "simulation_finite_train": fit.train.finite,
                        "simulation_finite_validation": fit.validation.finite,
                    },
                    "nonlocal": True, "test_evaluated": False,
                })
                print(
                    f"completed train={fit.train.data_error:.8f} validation={fit.validation.data_error:.8f} "
                    f"fitness={fit.validation_fitness:.8f} delta={fit.validation_fitness-common.INCUMBENT_FITNESS[baseline_key]:+.8f}",
                    flush=True,
                )
            except Exception as exc:
                records.append({
                    "baseline_key": baseline_key, "baseline": baseline.name,
                    "status": "failed", "error_type": type(exc).__name__,
                    "error": str(exc), "seed": seed, "nonlocal": True,
                    "test_evaluated": False,
                })
                print(f"FAILED {type(exc).__name__}: {exc}", flush=True)
            records.sort(key=lambda r: order[r["baseline_key"]])
            write_reports(model, factor_counts[model.ATTEMPT], records, time.perf_counter() - started)
        write_summary()


if __name__ == "__main__":
    main()
