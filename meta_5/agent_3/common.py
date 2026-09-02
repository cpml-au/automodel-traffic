"""Fixed global incumbents and repaired DEC features for meta-5 agent 3."""

from __future__ import annotations

import jax.numpy as jnp
from dctkit.dec import cochain as C


INCUMBENT_EXPRESSIONS = {
    "greenshields": (
        "exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)"
        "*exp(-293.458109123*C)"
    ),
    "idm": (
        "exp(-46.9914359751*C)"
        "*exp(-119537.388349*hquad*C)"
    ),
    "weidmann": "exp(0.136544052144*rho*(1-rho/0.80612097))",
    "triangular": "1",
    "del_castillo": (
        "exp(0.0054952026563)"
        "*exp(-2.12181160185*conv_3(rho,ones)-256.869229508*C)"
    ),
}
INCUMBENT_NODES = {
    "greenshields": 25,
    "idm": 32,
    "weidmann": 10,
    "triangular": 1,
    "del_castillo": 21,
}
INCUMBENT_FITNESS = {
    "greenshields": 6.953732490539551,
    "idm": 5.048744506835938,
    "weidmann": 6.322292900085449,
    "triangular": 6.457488784790039,
    "del_castillo": 5.73022123336792,
}


def convolution_features(rho: C.CochainP0):
    ones = C.CochainP0(rho.complex, jnp.ones_like(rho.coeffs))
    conv_3 = C.convolution(rho, ones, kernel_window=3)
    conv_1 = C.convolution(rho, ones, kernel_window=1)
    contrast = C.CochainP0(rho.complex, conv_3.coeffs - 3.0 * conv_1.coeffs)
    return conv_3, conv_1, contrast


def hlin(rho: C.CochainP0) -> C.CochainP0:
    """``St_oneD1(St_oneP0(rho))``."""

    return C.star(C.star(rho))


def hquad(rho: C.CochainP0) -> C.CochainP0:
    """``St_oneD1(SquareD1(St_oneP0(rho)))``."""

    dual = C.star(rho)
    return C.star(C.cochain_mul(dual, dual))


def incumbent(baseline_key: str, rho: C.CochainP0) -> C.CochainP0:
    values = rho.coeffs
    if baseline_key == "greenshields":
        local = jnp.exp(
            0.02312297116054006
            - 0.08831739725970279 * values
            - 0.1500812210203765 * values**2
        )
        _, _, contrast = convolution_features(rho)
        return C.CochainP0(
            rho.complex, local * jnp.exp(-293.4581091232274 * contrast.coeffs)
        )
    if baseline_key == "idm":
        _, _, contrast = convolution_features(rho)
        quadratic_gate = C.cochain_mul(hquad(rho), contrast)
        return C.CochainP0(
            rho.complex,
            jnp.exp(-46.991435975062515 * contrast.coeffs)
            * jnp.exp(-119537.38834933033 * quadratic_gate.coeffs),
        )
    if baseline_key == "weidmann":
        return C.CochainP0(
            rho.complex,
            jnp.exp(
                0.1365440521436772 * values * (1.0 - values / 0.80612097)
            ),
        )
    if baseline_key == "triangular":
        return C.CochainP0(rho.complex, jnp.ones_like(values))
    if baseline_key == "del_castillo":
        conv_3, _, contrast = convolution_features(rho)
        return C.CochainP0(
            rho.complex,
            jnp.exp(0.0054952026563039776)
            * jnp.exp(
                -2.1218116018455078 * conv_3.coeffs
                - 256.86922950791177 * contrast.coeffs
            ),
        )
    raise KeyError(baseline_key)


def scalar(rho: C.CochainP0, value: float) -> C.CochainP0:
    return C.CochainP0(rho.complex, value * jnp.ones_like(rho.coeffs))


def multiply(*cochains: C.CochainP0) -> C.CochainP0:
    result = cochains[0]
    for cochain in cochains[1:]:
        result = C.cochain_mul(result, cochain)
    return result


def attach_factor(baseline_key: str, rho: C.CochainP0, exponent) -> C.CochainP0:
    factor = C.CochainP0(rho.complex, jnp.exp(exponent.coeffs))
    return C.cochain_mul(incumbent(baseline_key, rho), factor)
