"""Fixed meta-3 incumbents plus a ``conv_3 - 2*conv_1`` refinement."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


INCUMBENT_EXPRESSIONS = {
    "greenshields": (
        "exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)"
        "*exp(-293.4581091232274*(conv_3(rho,ones)-3*conv_1(rho,ones)))"
    ),
    "idm": "exp(-46.991435975062515*(conv_3(rho,ones)-3*conv_1(rho,ones)))",
    "weidmann": "exp(0.136544052144*rho*(1-rho/0.80612097))",
    "triangular": "1",
    "del_castillo": (
        "exp(0.0054952026563)"
        "*exp(-2.1218116018455078*conv_3(rho,ones)"
        "-256.86922950791177*(conv_3(rho,ones)-3*conv_1(rho,ones)))"
    ),
}


def _convolutions(rho: C.CochainP0) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate the exact DCTKit convolution primitive at windows 3 and 1."""

    kernel = C.CochainP0(rho.complex, jnp.ones_like(rho.coeffs))
    conv_3 = C.convolution(rho, kernel, kernel_window=3).coeffs
    conv_1 = C.convolution(rho, kernel, kernel_window=1).coeffs
    return conv_3, conv_1


def _incumbent_values(
    baseline_key: str,
    rho: C.CochainP0,
    conv_3: jnp.ndarray,
    conv_1: jnp.ndarray,
) -> jnp.ndarray:
    values = rho.coeffs
    contrast_3 = conv_3 - 3.0 * conv_1
    if baseline_key == "greenshields":
        pointwise = jnp.exp(
            0.02312297116054006
            - 0.08831739725970279 * values
            - 0.1500812210203765 * values**2
        )
        return pointwise * jnp.exp(-293.4581091232274 * contrast_3)
    if baseline_key == "idm":
        return jnp.exp(-46.991435975062515 * contrast_3)
    if baseline_key == "weidmann":
        return jnp.exp(
            0.1365440521436772 * values * (1.0 - values / 0.80612097)
        )
    if baseline_key == "triangular":
        return jnp.ones_like(values)
    if baseline_key == "del_castillo":
        return jnp.exp(0.0054952026563039776) * jnp.exp(
            -2.1218116018455078 * conv_3 - 256.86922950791177 * contrast_3
        )
    raise KeyError(baseline_key)


def make_correction(baseline_key: str) -> Callable:
    """Return the fixed incumbent times a fitted window-2 contrast factor."""

    def correction(rho: C.CochainP0, parameters: Sequence[float]) -> C.CochainP0:
        (a,) = parameters
        conv_3, conv_1 = _convolutions(rho)
        multiplier = _incumbent_values(baseline_key, rho, conv_3, conv_1) * jnp.exp(
            a * (conv_3 - 2.0 * conv_1)
        )
        return C.CochainP0(rho.complex, multiplier)

    return correction


def expression(baseline_key: str) -> str:
    return (
        f"({INCUMBENT_EXPRESSIONS[baseline_key]})"
        "*exp(a*(conv_3(rho,ones)-2*conv_1(rho,ones)))"
    )
