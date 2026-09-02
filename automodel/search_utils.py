"""Shared, deterministic fitting utilities for Automodel Phase 3 agents."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import jax.numpy as jnp
import numpy as np
from dctkit.dec import cochain as C
from scipy.optimize import minimize

from automodel.model import Baseline
from automodel.pipeline import Evaluation, I80PredictionEvaluator


@dataclass(frozen=True)
class FitResult:
    expression: str
    tree_nodes: int
    parameter_names: tuple[str, ...]
    parameters: tuple[float, ...]
    train: Evaluation
    validation: Evaluation
    validation_fitness: float
    feasible: bool
    optimizer_success: bool
    optimizer_message: str
    optimizer_evaluations: int
    restarts: int
    seed: int
    fit_runtime_seconds: float

    def to_dict(self) -> dict:
        result = asdict(self)
        return result


def physical_density_limit(baseline: Baseline) -> float:
    """Return the calibrated upper end of a baseline's physical velocity domain."""

    if baseline.name in {"Greenshields", "Weidmann"}:
        return baseline.coefficients[1]
    if baseline.name == "Triangular":
        return 1.0 / baseline.coefficients[1]
    if baseline.name == "IDM":
        return 1.0 / (baseline.coefficients[0] + 1.0)
    if baseline.name == "Del Castillo":
        return baseline.coefficients[2]
    raise KeyError(baseline.name)


def is_feasible(
    evaluator: I80PredictionEvaluator,
    baseline: Baseline,
    correction: Callable,
    parameters: Sequence[float],
    samples: int = 79,
) -> bool:
    """Check positivity/finiteness and monotone velocity on the physical domain."""

    upper = physical_density_limit(baseline)
    rho_values = jnp.linspace(max(1e-5, upper / (samples * 10)), upper, samples)
    rho = C.CochainP0(evaluator.S, rho_values)
    multiplier = correction(rho, tuple(parameters)).coeffs
    velocity = baseline.velocity(rho_values, *baseline.coefficients) * multiplier
    tolerance = 1e-7
    return bool(
        jnp.all(jnp.isfinite(multiplier))
        & jnp.all(jnp.isfinite(velocity))
        & jnp.all(multiplier > 0)
        & jnp.all(velocity >= -tolerance)
        & jnp.all(jnp.diff(velocity) <= tolerance)
    )


def is_nonlocal_feasible(
    evaluator: I80PredictionEvaluator,
    baseline: Baseline,
    correction: Callable,
    parameters: Sequence[float],
    samples: int = 64,
) -> bool:
    """Check a spatial correction on homogeneous states away from boundaries.

    Convolution primitives use valid-mode padding at the downstream boundary, so
    applying the pointwise ramp check would mix density response with an intended
    boundary artifact. A homogeneous field isolates the constitutive response;
    the central node is unaffected by padding for the supported small kernels.
    Full train/validation simulations remain the second feasibility gate.
    """

    upper = physical_density_limit(baseline)
    rho_values = jnp.linspace(max(1e-5, upper / (samples * 10)), upper, samples)
    velocities = []
    for rho_value in rho_values:
        field = C.CochainP0(
            evaluator.S, rho_value * jnp.ones(evaluator.S.num_nodes)
        )
        multiplier = correction(field, tuple(parameters)).coeffs.flatten()
        if not bool(jnp.all(jnp.isfinite(multiplier)) & jnp.all(multiplier > 0)):
            return False
        center_multiplier = multiplier[evaluator.S.num_nodes // 2]
        # IDM's equilibrium inverse is vmapped and therefore requires a rank-one
        # input even for a single homogeneous density value.
        base_velocity = baseline.velocity(
            jnp.asarray([rho_value]), *baseline.coefficients
        )[0]
        velocity = base_velocity * center_multiplier
        velocities.append(velocity)
    velocities = jnp.asarray(velocities)
    tolerance = 1e-7
    return bool(
        jnp.all(jnp.isfinite(velocities))
        & jnp.all(velocities >= -tolerance)
        & jnp.all(jnp.diff(velocities) <= tolerance)
    )


def fit_candidate(
    evaluator: I80PredictionEvaluator,
    baseline: Baseline,
    correction: Callable,
    expression: str,
    tree_nodes: int,
    parameter_names: Sequence[str],
    bounds: Sequence[tuple[float, float]],
    seed: int,
    restarts: int = 2,
    max_evaluations: int = 30,
    feasibility_check: Callable | None = None,
) -> FitResult:
    """Tune one correction on train and score it once on validation.

    Powell is used because the JAX/Godunov objective contains discrete control
    flow. Restarts and their random seed are explicit so optimizer noise is
    auditable. The zero vector is always the first start, which recovers identity
    for the anchored/exponential families used by the search agents.
    """

    parameter_names = tuple(parameter_names)
    bounds = tuple(tuple(pair) for pair in bounds)
    rng = np.random.default_rng(seed)
    starts = [np.zeros(len(bounds))]
    for _ in range(max(0, restarts - 1)):
        starts.append(np.asarray([rng.uniform(low, high) for low, high in bounds]))

    evaluations = 0
    start_time = time.perf_counter()
    best = None
    check = feasibility_check or is_feasible

    def objective(values: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        if not check(evaluator, baseline, correction, values):
            return 100.0
        result = evaluator.evaluate(
            baseline,
            correction,
            tuple(values),
            expression,
            "train",
        )
        return result.data_error if result.finite else 100.0

    for initial in starts:
        fit = minimize(
            objective,
            initial,
            method="Powell",
            bounds=bounds,
            options={"maxfev": max_evaluations, "xtol": 2e-3, "ftol": 2e-3},
        )
        if best is None or float(fit.fun) < float(best.fun):
            best = fit

    assert best is not None
    parameters = tuple(float(value) for value in best.x)
    feasible = check(evaluator, baseline, correction, parameters)
    train = evaluator.evaluate(
        baseline, correction, parameters, expression, "train"
    )
    validation = evaluator.evaluate(
        baseline, correction, parameters, expression, "validation"
    )
    return FitResult(
        expression=expression,
        tree_nodes=tree_nodes,
        parameter_names=parameter_names,
        parameters=parameters,
        train=train,
        validation=validation,
        validation_fitness=validation.data_error + 0.01 * tree_nodes,
        feasible=feasible and train.finite and validation.finite,
        optimizer_success=bool(best.success),
        optimizer_message=str(best.message),
        optimizer_evaluations=evaluations,
        restarts=restarts,
        seed=seed,
        fit_runtime_seconds=time.perf_counter() - start_time,
    )
