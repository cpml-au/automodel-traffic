"""Fixed meta-3 incumbents with a protected square-root control factor."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


INCUMBENT_EXPRESSIONS = {
    "greenshields": (
        "exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)"
        "*exp(-293.458109123*(conv_3(rho,ones)-3*conv_1(rho,ones)))"
    ),
    "idm": "exp(-46.9914359751*(conv_3(rho,ones)-3*conv_1(rho,ones)))",
    "weidmann": "exp(0.136544052144*rho*(1-rho/0.80612097))",
    "triangular": "1",
    "del_castillo": (
        "exp(0.0054952026563)"
        "*exp(-2.12181160185*conv_3(rho,ones)"
        "-256.869229508*(conv_3(rho,ones)-3*conv_1(rho,ones)))"
    ),
}


def _incumbent_values(baseline_key: str, rho: C.CochainP0):
    values = rho.coeffs
    kernel = C.CochainP0(rho.complex, jnp.ones_like(values))

    if baseline_key == "greenshields":
        local = jnp.exp(
            0.02312297116054006
            - 0.08831739725970279 * values
            - 0.1500812210203765 * values**2
        )
        conv_3 = C.convolution(rho, kernel, kernel_window=3).coeffs
        conv_1 = C.convolution(rho, kernel, kernel_window=1).coeffs
        return local * jnp.exp(-293.4581091232274 * (conv_3 - 3.0 * conv_1))
    if baseline_key == "idm":
        conv_3 = C.convolution(rho, kernel, kernel_window=3).coeffs
        conv_1 = C.convolution(rho, kernel, kernel_window=1).coeffs
        return jnp.exp(-46.991435975062515 * (conv_3 - 3.0 * conv_1))
    if baseline_key == "weidmann":
        return jnp.exp(
            0.1365440521436772 * values * (1.0 - values / 0.80612097)
        )
    if baseline_key == "triangular":
        return jnp.ones_like(values)
    if baseline_key == "del_castillo":
        conv_3 = C.convolution(rho, kernel, kernel_window=3).coeffs
        conv_1 = C.convolution(rho, kernel, kernel_window=1).coeffs
        return jnp.exp(0.0054952026563039776) * jnp.exp(
            -2.1218116018455078 * conv_3
            - 256.86922950791177 * (conv_3 - 3.0 * conv_1)
        )
    raise KeyError(baseline_key)


def make_correction(baseline_key: str) -> Callable:
    """Return ``g_inc*exp(a*sqrt(max(rho,0)))``."""

    def correction(rho: C.CochainP0, parameters: Sequence[float]) -> C.CochainP0:
        (a,) = parameters
        protected_sqrt = jnp.sqrt(jnp.maximum(rho.coeffs, 0.0))
        values = _incumbent_values(baseline_key, rho) * jnp.exp(a * protected_sqrt)
        return C.CochainP0(rho.complex, values)

    return correction


def expression(baseline_key: str) -> str:
    return f"({INCUMBENT_EXPRESSIONS[baseline_key]})*exp(a*SqrtP0(rho))"
