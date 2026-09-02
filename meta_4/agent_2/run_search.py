"""Run meta-4 agent 2 Hodge/convolution searches without test access."""

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
from automodel.search_utils import (
    fit_candidate,
    is_nonlocal_feasible,
    physical_density_limit,
)
from meta_4.agent_2 import common
from tests.test_sr_primitives import build_traffic_pset


ROOT = Path(__file__).resolve().parent
BASELINE_ITEMS = tuple(BASELINES.items())
RESTARTS = 2
MAX_EVALUATIONS = 60
SPEED_BOUND = (
    "row-wise sum(abs(full flux Jacobian)); includes all off-diagonal "
    "Hodge/convolution coupling"
)


def load_model(attempt: int):
    path = ROOT / f"attempt_{attempt}" / "model.py"
    spec = importlib.util.spec_from_file_location(
        f"meta4_agent2_attempt{attempt}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_grammar(models: list[object]) -> tuple[list[str], dict[int, int]]:
    """Compile each typed Hodge/convolution factor in the repaired grammar."""

    pset, _ = build_traffic_pset()
    pset.addTerminal(0.0, float, "a")
    pset.addTerminal(0.0, float, "b")
    pset.addTerminal(3.0, float, "three")
    registered = [
        name
        for name in ("St_oneP0", "St_oneP1", "St_oneD0", "St_oneD1")
        if name in pset.mapping
    ]
    if len(registered) != 4:
        raise RuntimeError(f"incomplete repaired Hodge family: {registered}")
    node_counts = {}
    for model in models:
        tree = gp.PrimitiveTree.from_string(model.TYPED_FACTOR, pset)
        gp.compile(tree, pset)
        factor_nodes = model.ADDED_NODES - 1
        if len(tree) != factor_nodes:
            raise RuntimeError(
                f"attempt {model.ATTEMPT}: typed factor has {len(tree)} nodes, "
                f"expected {factor_nodes}"
            )
        node_counts[model.ATTEMPT] = len(tree)
    return registered, node_counts


def feasibility_diagnostics(evaluator, baseline, correction, parameters) -> dict:
    upper = physical_density_limit(baseline)
    densities = jnp.linspace(max(1e-5, upper / 640.0), upper, 64)
    center = evaluator.S.num_nodes // 2
    multipliers = []
    center_values = []
    velocities = []
    for density in densities:
        rho = C.CochainP0(
            evaluator.S, density * jnp.ones(evaluator.S.num_nodes)
        )
        values = correction(rho, parameters).coeffs.flatten()
        multipliers.append(values)
        center_values.append(values[center])
        velocities.append(
            baseline.velocity(jnp.asarray([density]), *baseline.coefficients)[0]
            * values[center]
        )
    multipliers = jnp.stack(multipliers)
    center_values = jnp.asarray(center_values)
    velocities = jnp.asarray(velocities)
    differences = jnp.diff(velocities)
    tolerance = 1e-7
    finite = bool(
        jnp.all(jnp.isfinite(multipliers))
        & jnp.all(jnp.isfinite(velocities))
    )
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
        "minimum_center_multiplier": float(jnp.min(center_values)),
        "maximum_center_multiplier": float(jnp.max(center_values)),
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
        "meta_iteration": 4,
        "agent": 2,
        "attempt": model.ATTEMPT,
        "candidate": {
            "symbolic_factor": model.SYMBOLIC_FACTOR,
            "typed_gp_factor": model.TYPED_FACTOR,
            "typed_gp_factor_nodes": factor_nodes,
            "attachment_node": "CMulP0",
            "added_nodes_including_attachment": model.ADDED_NODES,
            "tree_node_counting": (
                "fixed incumbent nodes + CMulP0 attachment + complete typed "
                "factor; repeated convolution subtrees counted separately"
            ),
            "incumbent_nodes": common.INCUMBENT_NODES,
            "total_nodes_by_baseline": {
                key: common.INCUMBENT_NODES[key] + model.ADDED_NODES
                for key, _ in BASELINE_ITEMS
            },
            "incumbent_coefficients_fixed": True,
            "hodge_primitives_used": list(model.HODGE_PRIMITIVES),
            "hodge_primitives_registered": registered,
            "implementation": [
                "dctkit.dec.cochain.star",
                "dctkit.dec.cochain.cochain_mul",
                "dctkit.dec.cochain.convolution",
            ],
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
            "parameter_bounds": [list(x) for x in model.BOUNDS],
            "seed_formula": f"{11100 + model.ATTEMPT * 10} + baseline_index",
            "baseline_order": [key for key, _ in BASELINE_ITEMS],
            "feasibility_check": "automodel.search_utils.is_nonlocal_feasible",
            "wave_speed_bound": SPEED_BOUND,
            "fitness": "validation E_data + 0.01*total_tree_nodes",
        },
        "records": records,
        "attempt_fit_runtime_seconds": sum(
            x["fit_runtime_seconds"] for x in completed
        ),
        "process_elapsed_seconds_at_last_write": elapsed,
        "attempt_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "completed_baselines": len(completed),
        "failed_baselines": len(records) - len(completed),
        "execution": {
            "checkout": str(Path.cwd()),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "jax_platform_name": os.environ.get("JAX_PLATFORMS", ""),
            "device": "CPU",
        },
        "nonlocal": True,
        "test_evaluated": False,
    }
    (attempt_dir / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    plan = [
        f"# Attempt {model.ATTEMPT} plan",
        "",
        f"Augment every fixed meta-3 incumbent with `{model.SYMBOLIC_FACTOR}`.",
        f"Exact typed factor: `{model.TYPED_FACTOR}` ({factor_nodes} nodes).",
        f"A `CMulP0` attachment makes {model.ADDED_NODES} added nodes; incumbent",
        "nodes are included separately in each total. The implementation uses",
        "`C.star`, `C.cochain_mul`, and `C.convolution`. Fit only the new constants",
        "on full train, select only on full validation, and never evaluate test.",
        "",
    ]
    (attempt_dir / "plan.md").write_text("\n".join(plan), encoding="utf-8")

    training = [
        f"# Attempt {model.ATTEMPT} training",
        "",
        "All incumbent coefficients were fixed. New constants used two deterministic",
        f"Powell restarts and at most {MAX_EVALUATIONS} evaluations per restart.",
        "",
        "| FD | Full expression | Nodes | Parameters | E_rho | E_v | E_data | Evals | Feasible | Runtime (s) | RSS (MB) | Seed |",
        "|---|---|---:|---|---:|---:|---:|---:|:---:|---:|---:|---:|",
    ]
    evaluation = [
        f"# Attempt {model.ATTEMPT} evaluation",
        "",
        "Full validation times 64--107; fitness is `E_data + 0.01*total_nodes`.",
        f"Nonlocal speed bound: `{SPEED_BOUND}`.",
        f"Registered Hodge names: `{', '.join(registered)}`.",
        "",
        "| FD | Parameters | E_rho | E_v | E_data | Fitness | Meta-3 fitness | Change | Finite/feasible | Runtime (s) | RSS (MB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for record in records:
        if record["status"] != "completed":
            err = record["error"].replace("|", "/")
            training.append(f"| {record['baseline']} | FAILED: {err} | | | | | | | no | | | {record['seed']} |")
            evaluation.append(f"| {record['baseline']} | FAILED: {err} | | | | | | | no | | |")
            continue
        fit = record["fit"]
        train = fit["train"]
        val = fit["validation"]
        params = format_parameters(fit["parameter_names"], fit["parameters"])
        training.append(
            f"| {record['baseline']} | `{fit['expression']}` | {fit['tree_nodes']} | "
            f"{params} | {train['rho_error']:.6f} | {train['velocity_error']:.6f} | "
            f"{train['data_error']:.6f} | {fit['optimizer_evaluations']} | "
            f"{'yes' if fit['feasible'] else 'no'} | {fit['fit_runtime_seconds']:.3f} | "
            f"{train['peak_rss_mb']:.1f} | {fit['seed']} |"
        )
        evaluation.append(
            f"| {record['baseline']} | {params} | {val['rho_error']:.6f} | "
            f"{val['velocity_error']:.6f} | {val['data_error']:.6f} | "
            f"{fit['validation_fitness']:.6f} | {record['meta3_fitness']:.6f} | "
            f"{record['fitness_change_from_meta3']:+.6f} | "
            f"{'yes' if fit['feasible'] and val['finite'] else 'no'} | "
            f"{val['runtime_seconds']:.3f} | {val['peak_rss_mb']:.1f} |"
        )
    training.extend(["", "`test_evaluated = false`.", ""])
    evaluation.extend(["", "All candidates use `is_nonlocal_feasible`.", "`test_evaluated = false`.", ""])
    (attempt_dir / "training.md").write_text("\n".join(training), encoding="utf-8")
    (attempt_dir / "evaluation.md").write_text("\n".join(evaluation), encoding="utf-8")


def write_summary():
    all_rows = []
    for attempt in (1, 2, 3):
        path = ROOT / f"attempt_{attempt}" / "results.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("records", []):
            if row["status"] == "completed":
                all_rows.append((attempt, row))
    lines = [
        "# Meta-4 agent 2 summary",
        "",
        "This lineage augmented each fixed meta-3 incumbent with repaired Hodge-star",
        "and convolution features. All comparisons use full validation only; test was",
        "not evaluated.",
        "",
        "| FD | Best attempt | Expression | Parameters | Validation E_data | Fitness | Meta-3 fitness | Change |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for key, baseline in BASELINE_ITEMS:
        rows = [(a, r) for a, r in all_rows if r["baseline_key"] == key]
        if not rows:
            continue
        attempt, row = min(rows, key=lambda x: x[1]["fit"]["validation_fitness"])
        fit = row["fit"]
        params = format_parameters(fit["parameter_names"], fit["parameters"])
        lines.append(
            f"| {baseline.name} | {attempt} | `{fit['expression']}` | {params} | "
            f"{fit['validation']['data_error']:.6f} | {fit['validation_fitness']:.6f} | "
            f"{row['meta3_fitness']:.6f} | {row['fitness_change_from_meta3']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "Lineage selection: retain the meta-3 incumbent for Greenshields,",
            "Weidmann, Triangular, and Del Castillo. Adopt attempt 2 for IDM,",
            "whose 32-node fitness is 5.048745 versus 5.417018 at meta 3",
            "(change -0.368274). Its Powell fit exhausted the 120-evaluation total",
            "budget, so the root review should independently re-evaluate it.",
            "Attempts 1 and 2 returned infeasible/nonfinite Weidmann endpoints and",
            "are rejected; their diagnostics remain in the result files.",
            "",
            "Registered live Hodge names: `St_oneP0`, `St_oneP1`, `St_oneD0`, `St_oneD1`.",
            f"Nonlocal speed bound: `{SPEED_BOUND}`.",
            "`test_evaluated = false`.",
            "",
        ]
    )
    (ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, choices=(1, 2, 3))
    parser.add_argument("--baseline", choices=[x[0] for x in BASELINE_ITEMS])
    args = parser.parse_args()
    models = [load_model(i) for i in (1, 2, 3)]
    registered, factor_counts = validate_grammar(models)
    evaluator = I80PredictionEvaluator()
    order = {key: i for i, (key, _) in enumerate(BASELINE_ITEMS)}
    for model in models:
        if args.attempt is not None and model.ATTEMPT != args.attempt:
            continue
        result_path = ROOT / f"attempt_{model.ATTEMPT}" / "results.json"
        records = []
        if args.baseline is not None and result_path.exists():
            old = json.loads(result_path.read_text(encoding="utf-8"))
            records = [
                row for row in old.get("records", [])
                if row["baseline_key"] != args.baseline
            ]
        started = time.perf_counter()
        for baseline_index, (baseline_key, baseline) in enumerate(BASELINE_ITEMS):
            if args.baseline is not None and baseline_key != args.baseline:
                continue
            seed = 11100 + model.ATTEMPT * 10 + baseline_index
            correction = model.make_correction(baseline_key)
            expression = model.expression(baseline_key)
            nodes = common.INCUMBENT_NODES[baseline_key] + model.ADDED_NODES
            print(
                f"attempt={model.ATTEMPT} baseline={baseline.name} "
                f"seed={seed} nodes={nodes}", flush=True
            )
            try:
                fit = fit_candidate(
                    evaluator=evaluator,
                    baseline=baseline,
                    correction=correction,
                    expression=expression,
                    tree_nodes=nodes,
                    parameter_names=model.PARAMETER_NAMES,
                    bounds=model.BOUNDS,
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
                        "meta3_fitness": common.META3_FITNESS[baseline_key],
                        "fitness_change_from_meta3": (
                            fit.validation_fitness
                            - common.META3_FITNESS[baseline_key]
                        ),
                        "diagnostics": {
                            "homogeneous_feasibility": diagnostics,
                            "wave_speed_bound": SPEED_BOUND,
                            "simulation_finite_train": fit.train.finite,
                            "simulation_finite_validation": fit.validation.finite,
                            "hodge_primitives_registered": registered,
                        },
                        "nonlocal": True,
                        "test_evaluated": False,
                    }
                )
                print(
                    f"completed train={fit.train.data_error:.8f} "
                    f"validation={fit.validation.data_error:.8f} "
                    f"fitness={fit.validation_fitness:.8f} "
                    f"delta={fit.validation_fitness-common.META3_FITNESS[baseline_key]:+.8f}",
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
            records.sort(key=lambda x: order[x["baseline_key"]])
            write_reports(
                model, records, registered, factor_counts[model.ATTEMPT],
                time.perf_counter() - started
            )
    write_summary()


if __name__ == "__main__":
    main()
