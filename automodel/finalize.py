"""Refit one frozen model and perform its single held-out Phase-4 check."""

from __future__ import annotations

import argparse
import json
import resource
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from automodel.final_models import FINAL_MODELS
from automodel.pipeline import I80PredictionEvaluator
from automodel.search_utils import is_feasible, is_nonlocal_feasible


ROOT = Path(__file__).resolve().parents[1]


def _decode(unit_values: np.ndarray, bounds) -> np.ndarray:
    lower = np.asarray([pair[0] for pair in bounds], dtype=float)
    upper = np.asarray([pair[1] for pair in bounds], dtype=float)
    return lower + 0.5 * (unit_values + 1.0) * (upper - lower)


def _encode(values: np.ndarray, bounds) -> np.ndarray:
    lower = np.asarray([pair[0] for pair in bounds], dtype=float)
    upper = np.asarray([pair[1] for pair in bounds], dtype=float)
    return 2.0 * (values - lower) / (upper - lower) - 1.0


def finalize(key: str, max_evaluations: int = 180) -> dict:
    model = FINAL_MODELS[key]
    evaluator = I80PredictionEvaluator()
    feasibility = is_nonlocal_feasible if model.nonlocal_correction else is_feasible
    cache = {}
    evaluations = 0
    started = time.perf_counter()

    def objective(unit_values: np.ndarray) -> float:
        nonlocal evaluations
        parameters = tuple(float(x) for x in _decode(unit_values, model.bounds))
        cache_key = tuple(round(x, 12) for x in parameters)
        if cache_key in cache:
            return cache[cache_key]
        evaluations += 1
        if not feasibility(evaluator, model.baseline, model.correction, parameters):
            cache[cache_key] = 100.0
            return 100.0
        result = evaluator.evaluate(
            model.baseline,
            model.correction,
            parameters,
            model.expression,
            "train_validation",
        )
        value = result.data_error if result.finite else 100.0
        cache[cache_key] = value
        return value

    current = np.asarray(model.initial_parameters, dtype=float)
    starts = [_encode(current, model.bounds), np.zeros(len(model.bounds))]
    # Explicitly retain feasible starts. This prevents a bounded Powell run from
    # replacing a valid incumbent with a pathological endpoint.
    candidates = [(objective(start), start.copy(), "explicit_start") for start in starts]
    optimizer_runs = []
    for start in starts:
        fit = minimize(
            objective,
            start,
            method="Powell",
            bounds=[(-1.0, 1.0)] * len(start),
            options={
                "maxfev": max_evaluations,
                "xtol": 1e-3,
                "ftol": 1e-3,
            },
        )
        candidates.append((float(fit.fun), np.asarray(fit.x), "powell"))
        optimizer_runs.append(
            {
                "success": bool(fit.success),
                "message": str(fit.message),
                "fun": float(fit.fun),
                "nfev": int(fit.nfev),
                "unit_parameters": [float(x) for x in fit.x],
            }
        )

    best_value, best_unit, selected_from = min(candidates, key=lambda item: item[0])
    parameters = tuple(float(x) for x in _decode(best_unit, model.bounds))
    feasible = feasibility(evaluator, model.baseline, model.correction, parameters)
    train_validation = evaluator.evaluate(
        model.baseline,
        model.correction,
        parameters,
        model.expression,
        "train_validation",
    )
    if not feasible or not train_validation.finite:
        raise RuntimeError(f"Final refit for {key} did not produce a feasible model")

    # This is the only test-set access in the script. Structure and coefficients
    # are frozen before this call, and the test result cannot affect fitting.
    test = evaluator.evaluate_test(
        model.baseline, model.correction, parameters, model.expression
    )
    elapsed = time.perf_counter() - started
    return {
        "baseline_key": key,
        "baseline": model.baseline.name,
        "frozen_source": model.source,
        "expression": model.expression,
        "tree_nodes": model.tree_nodes,
        "parameter_names": model.parameter_names,
        "selection_parameters": model.initial_parameters,
        "refit_parameters": parameters,
        "bounds": model.bounds,
        "optimizer": {
            "name": "scipy.optimize.minimize/Powell on scaled [-1,1] parameters",
            "starts": 2,
            "max_evaluations_per_start": max_evaluations,
            "objective_evaluations": evaluations,
            "selected_from": selected_from,
            "selected_objective": float(best_value),
            "runs": optimizer_runs,
        },
        "feasible": feasible,
        "train_validation": asdict(train_validation),
        "test": asdict(test),
        "test_evaluation_count": 1,
        "total_runtime_seconds": elapsed,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=sorted(FINAL_MODELS), required=True)
    parser.add_argument("--max-evaluations", type=int, default=180)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = finalize(args.baseline, args.max_evaluations)
    encoded = json.dumps(result, indent=2) + "\n"
    print(encoded, end="")
    output = args.output or ROOT / "automodel" / "phase4" / f"{args.baseline}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
