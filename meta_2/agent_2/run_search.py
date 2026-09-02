"""Run the meta-2 centered-exponential search on train/validation only."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

from automodel.model import BASELINES
from automodel.pipeline import I80PredictionEvaluator
from automodel.search_utils import fit_candidate, physical_density_limit


ROOT = Path(__file__).resolve().parent
BASELINE_ORDER = (
    "greenshields",
    "idm",
    "weidmann",
    "triangular",
    "del_castillo",
)


def load_model(attempt: int):
    path = ROOT / f"attempt_{attempt}" / "model.py"
    spec = importlib.util.spec_from_file_location(
        f"meta2_agent2_attempt{attempt}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric(value: float) -> str:
    return f"{value:.6f}"


def write_reports(attempt_dir: Path, payload: dict) -> None:
    results = payload["results"]
    training_lines = [
        f"# Attempt {payload['attempt']} training",
        "",
        "Constants were fit only on the complete I80 training split. Validation was "
        "scored only after fitting, and the held-out test split was never evaluated.",
        "",
        "| Baseline | Exact expression | Constants | E_rho | E_v | E_data | Fit evals | "
        "Optimizer | Fit runtime (s) | Peak RSS (MB) | Feasible |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for item in results:
        params = ", ".join(
            f"{name}={value:.10g}"
            for name, value in zip(item["parameter_names"], item["parameters"])
        )
        train = item["train"]
        training_lines.append(
            "| {baseline} | `{expression}` | {params} | {rho} | {velocity} | "
            "{data} | {evaluations} | {success}: {message} | {runtime:.3f} | "
            "{rss:.2f} | {feasible} |".format(
                baseline=item["baseline_key"],
                expression=item["expression"],
                params=params,
                rho=metric(train["rho_error"]),
                velocity=metric(train["velocity_error"]),
                data=metric(train["data_error"]),
                evaluations=item["optimizer_evaluations"],
                success="success" if item["optimizer_success"] else "stopped",
                message=item["optimizer_message"].replace("|", "/"),
                runtime=item["fit_runtime_seconds"],
                rss=train["peak_rss_mb"],
                feasible="yes" if item["feasible"] else "no",
            )
        )
    training_lines.extend(
        [
            "",
            "Optimizer: SciPy Powell with two deterministic restarts (zero/identity "
            "and one seeded uniform start), 45 function evaluations per restart, "
            "and bounds `[-5, 5]` for every coefficient. The fit-evaluation count is "
            "summed across restarts. Runtime includes fitting and final train/validation "
            "evaluations; RSS is the evaluator's process high-water mark.",
            "",
        ]
    )
    (attempt_dir / "training.md").write_text(
        "\n".join(training_lines), encoding="utf-8"
    )

    evaluation_lines = [
        f"# Attempt {payload['attempt']} evaluation",
        "",
        "Selection uses the complete I80 validation split and "
        "`E_fitness = E_data + 0.01 * tree_nodes`; lower is better.",
        "",
        "| Baseline | r* | Exact expression | Nodes | E_rho | E_v | E_data | "
        "E_fitness | Validation runtime (s) | Peak RSS (MB) | Finite | Feasible | Seed |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for item in results:
        validation = item["validation"]
        evaluation_lines.append(
            "| {baseline} | {rstar:.12g} | `{expression}` | {nodes} | {rho} | "
            "{velocity} | {data} | {fitness} | {runtime:.3f} | {rss:.2f} | "
            "{finite} | {feasible} | {seed} |".format(
                baseline=item["baseline_key"],
                rstar=item["center_r_star"],
                expression=item["expression"],
                nodes=item["tree_nodes"],
                rho=metric(validation["rho_error"]),
                velocity=metric(validation["velocity_error"]),
                data=metric(validation["data_error"]),
                fitness=metric(item["validation_fitness"]),
                runtime=validation["runtime_seconds"],
                rss=validation["peak_rss_mb"],
                finite="yes" if validation["finite"] else "no",
                feasible="yes" if item["feasible"] else "no",
                seed=item["seed"],
            )
        )
    failures = [
        item["baseline_key"]
        for item in results
        if not item["feasible"] or not item["validation"]["finite"]
    ]
    evaluation_lines.extend(
        [
            "",
            f"Failures: {', '.join(failures) if failures else 'none'}.",
            "",
            "Diagnostics: positivity/finiteness and non-negative, non-increasing "
            "corrected velocity were checked over 79 points on each FD's physical "
            "density domain by the shared feasibility routine.",
            "",
            "`test_evaluated = false`; no test prediction or score was computed.",
            "",
        ]
    )
    (attempt_dir / "evaluation.md").write_text(
        "\n".join(evaluation_lines), encoding="utf-8"
    )


def main() -> None:
    evaluator = I80PredictionEvaluator()
    for attempt in (1, 2, 3):
        attempt_dir = ROOT / f"attempt_{attempt}"
        model = load_model(attempt)
        run_start = time.perf_counter()
        results = []
        for baseline_index, baseline_key in enumerate(BASELINE_ORDER):
            baseline = BASELINES[baseline_key]
            physical_limit = physical_density_limit(baseline)
            r_star = 0.5 * physical_limit
            correction = model.make_correction(r_star)
            expression = model.expression(r_star)
            seed = 5100 + attempt * 10 + baseline_index
            fit = fit_candidate(
                evaluator=evaluator,
                baseline=baseline,
                correction=correction,
                expression=expression,
                tree_nodes=model.TREE_NODES,
                parameter_names=model.PARAMETER_NAMES,
                bounds=[(-5.0, 5.0)] * len(model.PARAMETER_NAMES),
                seed=seed,
                restarts=2,
                max_evaluations=45,
            )
            item = fit.to_dict()
            item["baseline_key"] = baseline_key
            item["physical_density_limit"] = physical_limit
            item["center_r_star"] = r_star
            results.append(item)
            print(
                f"attempt={attempt} baseline={baseline_key} train={fit.train.data_error:.6f} "
                f"validation={fit.validation.data_error:.6f} "
                f"fitness={fit.validation_fitness:.6f} feasible={fit.feasible}",
                flush=True,
            )

        payload = {
            "attempt": attempt,
            "family": "positive exponential polynomial centered at half the physical density limit",
            "symbolic_expression": model.SYMBOLIC_EXPRESSION,
            "parameter_names": list(model.PARAMETER_NAMES),
            "tree_nodes": model.TREE_NODES,
            "components_and_data": {
                "fixed_components": "I80 preprocessing, boundary/initial conditions, DEC mesh, Godunov solver, baseline FD and calibrated baseline coefficients",
                "tuned_components": "constants in the multiplicative correction only",
                "dataset": "I80 prediction",
                "training": "full chronological train times 0-63 (4,800 rows)",
                "validation": "full chronological validation times 64-107 (3,300 rows)",
                "test": "not evaluated",
                "center_definition": "r* = 0.5 * physical_density_limit(baseline)",
            },
            "optimizer": {
                "method": "scipy.optimize.minimize/Powell via automodel.search_utils.fit_candidate",
                "objective": "full training E_data",
                "bounds": [[-5.0, 5.0]] * len(model.PARAMETER_NAMES),
                "restarts": 2,
                "max_evaluations_per_restart": 45,
                "first_start": "zero vector (identity multiplier)",
                "second_start": "seeded uniform draw within bounds",
                "seed_formula": "5100 + attempt*10 + baseline_index",
            },
            "selection": {
                "split": "full validation",
                "fitness": "E_data + 0.01 * tree_nodes",
            },
            "diagnostics": {
                "feasibility": "shared 79-point positivity/finiteness and corrected-velocity non-negative/non-increasing check on physical density domain",
                "all_required_fits_completed": len(results) == len(BASELINE_ORDER),
            },
            "execution": {
                "checkout": str(Path.cwd()),
                "python": sys.executable,
                "python_version": platform.python_version(),
                "jax_platform_name": os.environ.get("JAX_PLATFORMS", ""),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "device": "CPU",
                "total_runtime_seconds": time.perf_counter() - run_start,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "test_evaluated": False,
            },
            "results": results,
        }
        (attempt_dir / "results.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        write_reports(attempt_dir, payload)


if __name__ == "__main__":
    main()
