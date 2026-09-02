"""Split-safe calibration and evaluation for multiplicative FD corrections."""

from __future__ import annotations

import argparse
import json
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import jax.numpy as jnp
import numpy as np
from dctkit.dec import cochain as C
from scipy.optimize import minimize_scalar

from automodel.model import BASELINES, Baseline, constant_multiplier
from sr_traffic.data.data import build_dataset, preprocess_data
from sr_traffic.sr.utils import solve
from sr_traffic.utils.flat import define_flats


@dataclass(frozen=True)
class Evaluation:
    baseline: str
    split: str
    expression: str
    parameters: tuple[float, ...]
    rho_error: float
    velocity_error: float
    data_error: float
    runtime_seconds: float
    peak_rss_mb: float
    finite: bool


class I80PredictionEvaluator:
    """Evaluate candidate corrections without exposing the held-out test split."""

    def __init__(self) -> None:
        self.data = preprocess_data("I80")
        train, validation, train_validation, test = build_dataset(
            self.data["t_sampled_circ"],
            self.data["S"],
            self.data["density"],
            self.data["v"],
            self.data["flow"],
            "prediction",
        )
        # Test is retained privately so Phases 1--3 cannot select it by name.
        self._splits = {
            "train": train,
            "validation": validation,
            "train_validation": train_validation,
        }
        self._test = test
        self.S = self.data["S"]
        zeros_p = C.CochainP0(self.S, jnp.zeros_like(self.data["vP0"][:, 0]))
        zeros_d = C.CochainD0(self.S, jnp.zeros_like(self.data["density"][:, 0]))
        all_flats = define_flats(self.S, zeros_p, zeros_d)
        self.flats = {
            "linear_left": all_flats["flat_linear_left_D"],
            "linear_right": all_flats["flat_linear_right_D"],
        }

    def subset(self, split: str, number_of_times: int | None) -> np.ndarray:
        X = self._splits[split]
        if number_of_times is None:
            return X
        times = np.unique(X[:, 0])[:number_of_times]
        return X[np.isin(X[:, 0], times)]

    def evaluate(
        self,
        baseline: Baseline,
        correction: Callable,
        parameters: Sequence[float],
        expression: str,
        split: str,
        number_of_times: int | None = None,
    ) -> Evaluation:
        X = self.subset(split, number_of_times)
        return self._evaluate_rows(
            baseline, correction, parameters, expression, split, X
        )

    def evaluate_test(
        self,
        baseline: Baseline,
        correction: Callable,
        parameters: Sequence[float],
        expression: str,
    ) -> Evaluation:
        """Evaluate one frozen Phase-4 model on the private held-out split.

        Keeping test access behind an explicit method makes accidental use
        during structural search conspicuous. Phase 3 callers can only select
        the public train/validation splits accepted by :meth:`evaluate`.
        """

        return self._evaluate_rows(
            baseline, correction, parameters, expression, "test", self._test
        )

    def _evaluate_rows(
        self,
        baseline: Baseline,
        correction: Callable,
        parameters: Sequence[float],
        expression: str,
        split: str,
        X: np.ndarray,
    ) -> Evaluation:

        def compiled_correction(rho, constants):
            return correction(rho, constants)

        ansatz = {
            "flux": baseline.flux,
            "v": baseline.velocity,
            "opt_coeffs": baseline.coefficients,
        }
        num_solver_times = int(X[-1, 0] * self.data["step"] + 1)
        start = time.perf_counter()
        total_error, fields = solve(
            compiled_correction,
            tuple(parameters),
            X,
            self.data["rho_bnd"],
            self.data["rho_0"],
            self.S,
            num_solver_times,
            self.data["delta_t_refined"],
            self.data["step"],
            self.flats,
            ansatz,
            "prediction",
        )
        total_error.block_until_ready()
        elapsed = time.perf_counter() - start

        time_indices = jnp.arange(X[0, 0], X[-1, 0] + 1, dtype=jnp.int64)
        rho_true = jnp.asarray(X[:, 1])
        velocity_true = jnp.asarray(X[:, 2])
        rho_model = fields["rho"][1:-3, time_indices * self.data["step"]].ravel("F")
        velocity_model = fields["v"][
            1:-3, time_indices * self.data["step"]
        ].ravel("F")
        rho_error = 100 * jnp.sum((rho_model - rho_true) ** 2) / jnp.sum(rho_true**2)
        velocity_error = 100 * jnp.sum(
            (velocity_model - velocity_true) ** 2
        ) / jnp.sum(velocity_true**2)
        finite = bool(
            jnp.isfinite(total_error)
            & jnp.all(jnp.isfinite(rho_model))
            & jnp.all(jnp.isfinite(velocity_model))
        )
        return Evaluation(
            baseline=baseline.name,
            split=split,
            expression=expression,
            parameters=tuple(float(value) for value in parameters),
            rho_error=float(rho_error),
            velocity_error=float(velocity_error),
            data_error=float(total_error),
            runtime_seconds=elapsed,
            peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            finite=finite,
        )


def fit_constant_multiplier(
    evaluator: I80PredictionEvaluator,
    baseline: Baseline,
    number_of_times: int | None,
) -> tuple[float, object]:
    """Fit the mock scale on training data with bounded scalar optimization."""

    def objective(scale: float) -> float:
        result = evaluator.evaluate(
            baseline,
            constant_multiplier,
            (scale,),
            "c0",
            "train",
            number_of_times,
        )
        return result.data_error if result.finite else 100.0

    fit = minimize_scalar(
        objective,
        bounds=(0.5, 1.5),
        method="bounded",
        options={"xatol": 1e-3, "maxiter": 20},
    )
    return float(fit.x), fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=sorted(BASELINES), default="triangular")
    parser.add_argument("--number-of-times", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    evaluator = I80PredictionEvaluator()
    baseline = BASELINES[args.baseline]
    run_start = time.perf_counter()
    scale, fit = fit_constant_multiplier(evaluator, baseline, args.number_of_times)
    training = evaluator.evaluate(
        baseline,
        constant_multiplier,
        (scale,),
        "c0",
        "train",
        args.number_of_times,
    )
    validation = evaluator.evaluate(
        baseline,
        constant_multiplier,
        (scale,),
        "c0",
        "validation",
        args.number_of_times,
    )
    report = {
        "model": "q_baseline(rho) * c0",
        "optimizer": {
            "name": "scipy.optimize.minimize_scalar/bounded",
            "bounds": [0.5, 1.5],
            "success": bool(fit.success),
            "message": str(fit.message),
            "evaluations": int(fit.nfev),
        },
        "training": asdict(training),
        "validation": asdict(validation),
        "total_runtime_seconds": time.perf_counter() - run_start,
    }
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
