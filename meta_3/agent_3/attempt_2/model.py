"""Incumbent correction augmented by a convolution contrast."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


INCUMBENT_EXPRESSIONS = {
    "greenshields": "exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)",
    "idm": "1",
    "weidmann": "exp(0.136544052144*rho*(1-rho/0.80612097))",
    "triangular": "1",
    "del_castillo": "exp(0.0054952026563)",
}


def _incumbent_values(baseline_key: str, values):
    if baseline_key == "greenshields":
        return jnp.exp(
            0.02312297116054006
            - 0.08831739725970279 * values
            - 0.1500812210203765 * values**2
        )
    if baseline_key == "weidmann":
        return jnp.exp(
            0.1365440521436772 * values * (1.0 - values / 0.80612097)
        )
    if baseline_key == "del_castillo":
        return jnp.exp(0.0054952026563039776) * jnp.ones_like(values)
    if baseline_key in {"idm", "triangular"}:
        return jnp.ones_like(values)
    raise KeyError(baseline_key)


def make_correction(baseline_key: str) -> Callable:
    """Return ``g_inc*exp(a*(conv_3-3*conv_1))``."""

    def correction(rho: C.CochainP0, parameters: Sequence[float]) -> C.CochainP0:
        (a,) = parameters
        kernel = C.CochainP0(rho.complex, jnp.ones_like(rho.coeffs))
        conv_3 = C.convolution(rho, kernel, kernel_window=3).coeffs
        conv_1 = C.convolution(rho, kernel, kernel_window=1).coeffs
        contrast = conv_3 - 3.0 * conv_1
        values = _incumbent_values(baseline_key, rho.coeffs) * jnp.exp(a * contrast)
        return C.CochainP0(rho.complex, values)

    return correction


def expression(baseline_key: str) -> str:
    return (
        f"({INCUMBENT_EXPRESSIONS[baseline_key]})"
        "*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones)))"
    )
