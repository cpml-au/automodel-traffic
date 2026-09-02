"""Fixed meta-3 incumbents and DEC features for meta-4 agent 2."""

from __future__ import annotations

import jax.numpy as jnp
from dctkit.dec import cochain as C


INCUMBENT_EXPRESSIONS = {
    "greenshields": (
        "exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)"
        "*exp(-293.458109123*(conv_3-3*conv_1))"
    ),
    "idm": "exp(-46.9914359751*(conv_3-3*conv_1))",
    "weidmann": "exp(0.136544052144*rho*(1-rho/0.80612097))",
    "triangular": "1",
    "del_castillo": (
        "exp(0.0054952026563)"
        "*exp(-2.12181160185*conv_3-256.869229508*(conv_3-3*conv_1))"
    ),
}

# Counts reproduced by the meta-3 root review. They include every repeated
# convolution subtree, coefficient terminal, and incumbent product root.
INCUMBENT_NODES = {
    "greenshields": 25,
    "idm": 14,
    "weidmann": 10,
    "triangular": 1,
    "del_castillo": 21,
}

META3_FITNESS = {
    "greenshields": 6.953732490539551,
    "idm": 5.417018070220947,
    "weidmann": 6.322293,
    "triangular": 6.457489,
    "del_castillo": 5.73022123336792,
}


def convolution_features(rho: C.CochainP0):
    """Return the exact one-/three-cell convolutions and level-free contrast."""

    ones = C.CochainP0(rho.complex, jnp.ones_like(rho.coeffs))
    conv_3 = C.convolution(rho, ones, kernel_window=3)
    conv_1 = C.convolution(rho, ones, kernel_window=1)
    three = C.CochainP0(rho.complex, 3.0 * jnp.ones_like(rho.coeffs))
    three_conv_1 = C.cochain_mul(conv_1, three)
    contrast = C.CochainP0(rho.complex, conv_3.coeffs - three_conv_1.coeffs)
    return conv_3, conv_1, contrast


def hlin(rho: C.CochainP0) -> C.CochainP0:
    """Typed feature ``St_oneD1(St_oneP0(rho))``."""

    return C.star(C.star(rho))


def hquad(rho: C.CochainP0) -> C.CochainP0:
    """Typed feature ``St_oneD1(SquareD1(St_oneP0(rho)))``."""

    dual = C.star(rho)
    dual_square = C.cochain_mul(dual, dual)
    return C.star(dual_square)


def incumbent(baseline_key: str, rho: C.CochainP0) -> C.CochainP0:
    """Evaluate one fixed meta-3 incumbent without refitting its constants."""

    values = rho.coeffs
    if baseline_key == "greenshields":
        local = C.CochainP0(
            rho.complex,
            jnp.exp(
                0.02312297116054006
                - 0.08831739725970279 * values
                - 0.1500812210203765 * values**2
            ),
        )
        _, _, contrast = convolution_features(rho)
        spatial = C.CochainP0(
            rho.complex, jnp.exp(-293.4581091232274 * contrast.coeffs)
        )
        return C.cochain_mul(local, spatial)
    if baseline_key == "idm":
        _, _, contrast = convolution_features(rho)
        return C.CochainP0(
            rho.complex, jnp.exp(-46.991435975062515 * contrast.coeffs)
        )
    if baseline_key == "weidmann":
        return C.CochainP0(
            rho.complex,
            jnp.exp(
                0.1365440521436772
                * values
                * (1.0 - values / 0.80612097)
            ),
        )
    if baseline_key == "triangular":
        return C.CochainP0(rho.complex, jnp.ones_like(values))
    if baseline_key == "del_castillo":
        conv_3, _, contrast = convolution_features(rho)
        local = C.CochainP0(
            rho.complex,
            jnp.exp(0.0054952026563039776) * jnp.ones_like(values),
        )
        spatial = C.CochainP0(
            rho.complex,
            jnp.exp(
                -2.1218116018455078 * conv_3.coeffs
                - 256.86922950791177 * contrast.coeffs
            ),
        )
        return C.cochain_mul(local, spatial)
    raise KeyError(baseline_key)


def scalar_cochain(rho: C.CochainP0, value: float) -> C.CochainP0:
    return C.CochainP0(rho.complex, value * jnp.ones_like(rho.coeffs))


def exp_cochain(rho: C.CochainP0, exponent: C.CochainP0) -> C.CochainP0:
    return C.CochainP0(rho.complex, jnp.exp(exponent.coeffs))
