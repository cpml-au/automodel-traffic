"""Tests for local and convolution-aware Rusanov speed bounds."""

import jax.numpy as jnp
from dctkit.dec import cochain as C
from jax import jacfwd

from sr_traffic.data.data import preprocess_data
from sr_traffic.fd.diagrams import Greenshields_flux, define_flux_der
from automodel.model import BASELINES
from automodel.pipeline import I80PredictionEvaluator
from automodel.search_utils import is_nonlocal_feasible


def test_local_flux_speed_matches_absolute_analytic_derivative():
    data = preprocess_data("I80")
    complex_ = data["S"]
    v_max, rho_max = 0.55, 0.7
    rho_values = jnp.linspace(0.01, 0.6, complex_.num_nodes)
    rho = C.CochainP0(complex_, rho_values)

    speed = define_flux_der(complex_, Greenshields_flux)(
        rho, v_max, rho_max
    ).coeffs.flatten()
    expected = jnp.abs(v_max * (1.0 - 2.0 * rho_values / rho_max))

    assert bool(jnp.allclose(speed, expected, rtol=1e-5, atol=1e-6))


def test_convolution_speed_includes_off_diagonal_jacobian_entries():
    data = preprocess_data("I80")
    complex_ = data["S"]
    kernel = C.CochainP0(complex_, jnp.ones(complex_.num_nodes))

    def nonlocal_flux(rho):
        base = Greenshields_flux(rho, 0.55, 0.7)
        feature = C.convolution(rho, kernel, kernel_window=3)
        multiplier = C.CochainP0(complex_, jnp.exp(0.2 * feature.coeffs))
        return C.cochain_mul(base, multiplier)

    rho_values = jnp.linspace(0.01, 0.6, complex_.num_nodes)
    rho = C.CochainP0(complex_, rho_values)
    measured = define_flux_der(complex_, nonlocal_flux)(rho).coeffs.flatten()

    def array_flux(values):
        return nonlocal_flux(C.CochainP0(complex_, values)).coeffs.flatten()

    jacobian = jacfwd(array_flux)(rho_values)
    expected = jnp.sum(jnp.abs(jacobian), axis=1)
    off_diagonal = jacobian - jnp.diag(jnp.diag(jacobian))

    assert bool(jnp.any(jnp.abs(off_diagonal) > 0))
    assert bool(jnp.allclose(measured, expected, rtol=1e-5, atol=1e-6))


def test_identity_convolution_correction_passes_homogeneous_feasibility():
    evaluator = I80PredictionEvaluator()
    kernel = C.CochainP0(evaluator.S, jnp.ones(evaluator.S.num_nodes))

    def correction(rho, parameters):
        (coefficient,) = parameters
        feature = C.convolution(rho, kernel, kernel_window=3)
        return C.CochainP0(
            rho.complex, jnp.exp(coefficient * feature.coeffs)
        )

    for baseline in BASELINES.values():
        assert is_nonlocal_feasible(evaluator, baseline, correction, (0.0,))
