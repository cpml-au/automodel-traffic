"""Run one agent-3 attempt through the common split-safe Phase 3 protocol."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

from automodel.model import BASELINES
from automodel.pipeline import I80PredictionEvaluator
from automodel.search_utils import fit_candidate


ATTEMPTS = {
    1: {
        "expression": "exp(a*rho/(1+rho))",
        "tree_nodes": 8,
        "parameter_names": ("a",),
    },
    2: {
        "expression": "exp((a*rho+b*rho^2)/(1+rho))",
        "tree_nodes": 13,
        "parameter_names": ("a", "b"),
    },
    3: {
        "expression": "exp((a*rho+b*rho^2+c*rho^3)/(1+rho))",
        "tree_nodes": 20,
        "parameter_names": ("a", "b", "c"),
    },
}


def load_correction(path: Path):
    spec = importlib.util.spec_from_file_location(f"agent3_{path.parent.name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.correction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt", type=int, choices=ATTEMPTS)
    args = parser.parse_args()

    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("Set JAX_PLATFORMS=cpu for reproducible CPU evaluation")

    attempt_dir = Path(__file__).resolve().parent / f"attempt_{args.attempt}"
    config = ATTEMPTS[args.attempt]
    correction = load_correction(attempt_dir / "model.py")
    evaluator = I80PredictionEvaluator()
    reports = {}
    for baseline_index, (key, baseline) in enumerate(BASELINES.items()):
        seed = 3100 + args.attempt * 10 + baseline_index
        result = fit_candidate(
            evaluator=evaluator,
            baseline=baseline,
            correction=correction,
            expression=config["expression"],
            tree_nodes=config["tree_nodes"],
            parameter_names=config["parameter_names"],
            bounds=[(-5.0, 5.0)] * len(config["parameter_names"]),
            seed=seed,
            restarts=2,
            max_evaluations=30,
        )
        reports[key] = result.to_dict()
        # Persist after every baseline so a later failure cannot erase completed work.
        payload = {
            "attempt": args.attempt,
            "protocol": {
                "dataset": "I80",
                "problem": "prediction",
                "fit_split": "full train (times 0-63)",
                "selection_split": "full validation (times 64-107)",
                "test_evaluated": False,
                "optimizer": "scipy.optimize.minimize/Powell",
                "bounds": [-5.0, 5.0],
                "restarts": 2,
                "max_evaluations_per_restart": 30,
                "seed_formula": "3100 + 10*attempt + baseline_index",
                "fitness": "validation.data_error + 0.01*tree_nodes",
            },
            "expression": config["expression"],
            "tree_nodes": config["tree_nodes"],
            "baselines": reports,
        }
        (attempt_dir / "results.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
