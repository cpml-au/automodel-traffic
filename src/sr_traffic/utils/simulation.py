"""Simulation and scoring utilities for multiplicative traffic corrections."""

from __future__ import annotations

from functools import partial
from typing import Callable, Mapping, Sequence

import jax.numpy as jnp
import numpy.typing as npt
from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from jax import jacfwd

from sr_traffic.utils.godunov import body_fun, main_loop


def solve_correction(
    correction: Callable,
    constants: Sequence[float],
    observations: npt.NDArray,
    boundary_density: Mapping[str, npt.NDArray],
    initial_density: npt.NDArray,
    complex_: SimplicialComplex,
    num_time_points: int,
    delta_t: float,
    sampling_step: int,
    flats: Mapping[str, Callable],
    baseline_flux: Callable,
    baseline_coefficients: Sequence[float],
    task: str,
) -> tuple[float, dict[str, npt.NDArray]]:
    """Simulate a multiplicatively corrected flux and score its observations."""

    def multiplier(rho: C.CochainP0) -> C.CochainP0:
        return correction(rho, constants)

    def flux(rho: C.CochainP0) -> C.CochainP0:
        baseline = baseline_flux(rho, *baseline_coefficients)
        return C.cochain_mul(baseline, multiplier(rho))

    boundary_array = jnp.zeros((len(boundary_density), num_time_points))
    for index, values in boundary_density.items():
        boundary_array = boundary_array.at[int(index), :].set(values[:num_time_points])

    def flux_array(values: npt.NDArray) -> npt.NDArray:
        rho = C.CochainP0(complex_, values)
        return flux(rho).coeffs.flatten()

    flux_jacobian = jacfwd(flux_array)

    def flux_speed_array(values: npt.NDArray) -> npt.NDArray:
        # The row sum includes off-diagonal coupling from nonlocal corrections.
        return jnp.sum(jnp.abs(flux_jacobian(values.flatten())), axis=1)

    def flux_speed(rho: C.CochainP0) -> C.CochainP0:
        return C.CochainP0(complex_, flux_speed_array(rho.coeffs))

    single_iteration = partial(
        body_fun,
        complex_,
        boundary_array,
        flux,
        flux_speed,
        delta_t,
        0.0,
        flats,
    )
    rho_velocity_flux = main_loop(initial_density, single_iteration, num_time_points)
    density_steps = rho_velocity_flux[0][:, :, 0]
    velocity_steps = rho_velocity_flux[1][:, :, 0]
    flux_steps = rho_velocity_flux[2][:, :, 0]

    initial_primal_density = C.star(
        flats["linear_left"](C.CochainD0(complex_, initial_density))
    )
    initial_flux = flux(initial_primal_density).coeffs
    initial_velocity = initial_flux / initial_primal_density.coeffs

    rho_computed = jnp.vstack([initial_density, density_steps]).T
    velocity_computed = jnp.vstack([initial_velocity[:-1].ravel("F"), velocity_steps]).T
    flux_computed = jnp.vstack([initial_flux[:-1].ravel("F"), flux_steps]).T

    if task == "prediction":
        indices = jnp.arange(
            observations[0, 0], observations[-1, 0] + 1, dtype=jnp.int64
        )
        density_model = rho_computed[1:-3, indices * sampling_step].ravel("F")
        velocity_model = velocity_computed[1:-3, indices * sampling_step].ravel("F")
    elif task == "reconstruction":
        original_time_points = int((num_time_points - 1) / sampling_step) + 1
        num_positions = int(observations.shape[0] / original_time_points)
        indices = observations[:num_positions, 0].astype(jnp.int64)
        density_model = rho_computed[1:-3, ::sampling_step][indices].ravel("F")
        velocity_model = velocity_computed[1:-3, ::sampling_step][indices].ravel("F")
    else:
        raise ValueError(f"Unsupported task: {task}")

    density_true = observations[:, 1]
    velocity_true = observations[:, 2]
    density_error = (
        100 * jnp.sum((density_model - density_true) ** 2) / jnp.sum(density_true**2)
    )
    velocity_error = (
        100 * jnp.sum((velocity_model - velocity_true) ** 2) / jnp.sum(velocity_true**2)
    )
    total_error = 0.5 * (density_error + velocity_error)

    return total_error, {
        "rho": rho_computed,
        "v": velocity_computed,
        "f": flux_computed,
    }
