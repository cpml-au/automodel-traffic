"""Run one prescribed jam-anchored Automodel candidate on all five baselines.

This Phase 3 harness deliberately exposes only the train and validation split
names accepted by :class:`I80PredictionEvaluator`.  It never requests the held-
out test split.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

from automodel.model import BASELINES
from automodel.pipeline import I80PredictionEvaluator
from automodel.search_utils import fit_candidate, physical_density_limit


BASELINE_ORDER = (
    "greenshields",
    "idm",
    "weidmann",
    "triangular",
    "del_castillo",
)


def load_model(path: Path):
    spec = importlib.util.spec_from_file_location(
        f"jam_anchored_attempt_{path.parent.name}", path
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
        "All correction constants were fit only against the complete I80 training split. "
        "The complete validation split was evaluated only after each fit. The held-out "
        "test split was never requested.",
        "",
        "| Baseline | Expression | Constants | E_rho | E_v | E_data | Fit evals | "
        "Optimizer | Runtime (s) | Peak RSS (MB) | Feasible |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for item in results:
        params = ", ".join(
            f"{name}={value:.10g}"
            for name, value in zip(item["parameter_names"], item["parameters"])
        )
        training = item["train"]
        training_lines.append(
            "| {baseline} | `{expression}` | {params} | {rho} | {velocity} | "
            "{data} | {evaluations} | {success}: {message} | {runtime:.3f} | "
            "{rss:.2f} | {feasible} |".format(
                baseline=item["baseline_key"],
                expression=item["expression"],
                params=params,
                rho=metric(training["rho_error"]),
                velocity=metric(training["velocity_error"]),
                data=metric(training["data_error"]),
                evaluations=item["optimizer_evaluations"],
                success="success" if item["optimizer_success"] else "stopped",
                message=item["optimizer_message"].replace("|", "/"),
                runtime=item["fit_runtime_seconds"],
                rss=training["peak_rss_mb"],
                feasible="yes" if item["feasible"] else "no",
            )
        )
    training_lines.extend(
        [
            "",
            "Optimizer: SciPy Powell, two deterministic restarts (the zero/identity "
            "start plus one seeded uniform start), 30 function evaluations per restart, "
            "and coefficient bounds `[-6, 6]`. The logged evaluation count is summed "
            "across both restarts. Runtime includes fitting and final train/validation "
            "evaluations; RSS is the process high-water mark reported by the evaluator.",
            "",
        ]
    )
    (attempt_dir / "training.md").write_text(
        "\n".join(training_lines), encoding="utf-8"
    )

    ranked = sorted(results, key=lambda item: item["validation_fitness"])
    evaluation_lines = [
        f"# Attempt {payload['attempt']} evaluation",
        "",
        "Selection uses the complete I80 validation split and "
        "`E_fitness = E_data + 0.01 * tree_nodes`. Lower is better.",
        "",
        "| Rank | Baseline | Expression | Nodes | E_rho | E_v | E_data | "
        "E_fitness | Finite | Feasible | Seed |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for rank, item in enumerate(ranked, 1):
        validation = item["validation"]
        evaluation_lines.append(
            "| {rank} | {baseline} | `{expression}` | {nodes} | {rho} | "
            "{velocity} | {data} | {fitness} | {finite} | {feasible} | {seed} |".format(
                rank=rank,
                baseline=item["baseline_key"],
                expression=item["expression"],
                nodes=item["tree_nodes"],
                rho=metric(validation["rho_error"]),
                velocity=metric(validation["velocity_error"]),
                data=metric(validation["data_error"]),
                fitness=metric(item["validation_fitness"]),
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
            "No test prediction or score was computed.",
            "",
        ]
    )
    (attempt_dir / "evaluation.md").write_text(
        "\n".join(evaluation_lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt_dir", type=Path)
    args = parser.parse_args()
    attempt_dir = args.attempt_dir.resolve()
    model = load_model(attempt_dir / "model.py")

    evaluator = I80PredictionEvaluator()
    run_start = time.perf_counter()
    results = []
    for baseline_index, baseline_key in enumerate(BASELINE_ORDER):
        baseline = BASELINES[baseline_key]
        r_j = physical_density_limit(baseline)
        correction = model.make_correction(r_j)
        expression = model.expression(r_j)
        seed = 2100 + model.ATTEMPT * 10 + baseline_index
        fit = fit_candidate(
            evaluator=evaluator,
            baseline=baseline,
            correction=correction,
            expression=expression,
            tree_nodes=model.TREE_NODES,
            parameter_names=model.PARAMETER_NAMES,
            bounds=[(-6.0, 6.0)] * len(model.PARAMETER_NAMES),
            seed=seed,
            restarts=2,
            max_evaluations=30,
        )
        item = fit.to_dict()
        item["baseline_key"] = baseline_key
        item["physical_density_limit_r_j"] = r_j
        results.append(item)
        print(
            f"attempt={model.ATTEMPT} baseline={baseline_key} "
            f"train={fit.train.data_error:.6f} "
            f"validation={fit.validation.data_error:.6f} "
            f"fitness={fit.validation_fitness:.6f} feasible={fit.feasible}",
            flush=True,
        )

    payload = {
        "attempt": model.ATTEMPT,
        "family": "positive jam-anchored exponential polynomial multiplier",
        "symbolic_expression": model.SYMBOLIC_EXPRESSION,
        "parameter_names": list(model.PARAMETER_NAMES),
        "tree_nodes": model.TREE_NODES,
        "optimizer": {
            "method": "scipy.optimize.minimize/Powell",
            "objective": "full training E_data",
            "bounds": [[-6.0, 6.0]] * len(model.PARAMETER_NAMES),
            "restarts": 2,
            "max_evaluations_per_restart": 30,
            "first_start": "zero vector (identity multiplier)",
            "second_start": "seeded uniform draw within bounds",
        },
        "selection": {
            "split": "full validation",
            "fitness": "E_data + 0.01 * tree_nodes",
        },
        "execution": {
            "checkout": str(Path.cwd()),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "jax_platform_name": os.environ.get("JAX_PLATFORMS", ""),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "total_runtime_seconds": time.perf_counter() - run_start,
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
