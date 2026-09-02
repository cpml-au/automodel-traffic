"""Exact fixed global incumbents and DEC contrast features for meta 5."""

from __future__ import annotations

import jax.numpy as jnp
from dctkit.dec import cochain as C


INCUMBENT_EXPRESSIONS = {
    "greenshields": (
        "exp(0.0231229711605-0.0883173972597*rho-0.15008122102*rho^2)"
        "*exp(-293.4581091232274*(conv_3(rho,ones)-3*conv_1(rho,ones)))"
    ),
    "idm": (
        "exp(-46.991435975062515*(conv_3(rho,ones)-3*conv_1(rho,ones)))"
        "*exp(-119537.38834933033*hquad(rho)"
        "*(conv_3(rho,ones)-3*conv_1(rho,ones)))"
    ),
    "weidmann": "exp(0.1365440521436772*rho*(1-rho/0.80612097))",
    "triangular": "1",
    "del_castillo": (
        "exp(0.0054952026563039776)"
        "*exp(-2.1218116018455078*conv_3(rho,ones)"
        "-256.86922950791177*(conv_3(rho,ones)-3*conv_1(rho,ones)))"
    ),
}

INCUMBENT_NODES = {
    "greenshields": 25,
    "idm": 32,
    "weidmann": 10,
    "triangular": 1,
    "del_castillo": 21,
}

INCUMBENT_VALIDATION = {
    "greenshields": {"data_error": 6.703732490539551, "fitness": 6.953732490539551},
    "idm": {"data_error": 4.7287445068359375, "fitness": 5.048744506835938},
    "weidmann": {"data_error": 6.222292900085449, "fitness": 6.322292900085449},
    "triangular": {"data_error": 6.447488784790039, "fitness": 6.457488784790039},
    "del_castillo": {"data_error": 5.52022123336792, "fitness": 5.73022123336792},
}


def convolution_features(rho: C.CochainP0):
    """Evaluate exact DCTKit window-3/window-1 convolution and their contrast."""

    ones = C.CochainP0(rho.complex, jnp.ones_like(rho.coeffs))
    conv_3 = C.convolution(rho, ones, kernel_window=3)
    conv_1 = C.convolution(rho, ones, kernel_window=1)
    contrast = C.CochainP0(
        rho.complex, conv_3.coeffs - 3.0 * conv_1.coeffs
    )
    return conv_3, conv_1, contrast


def hquad(rho: C.CochainP0) -> C.CochainP0:
    """Evaluate ``St_oneD1(SquareD1(St_oneP0(rho)))`` with repaired stars."""

    dual = C.star(rho)
    return C.star(C.cochain_mul(dual, dual))


def incumbent(
    baseline_key: str,
    rho: C.CochainP0,
    conv_3: C.CochainP0,
    contrast: C.CochainP0,
) -> C.CochainP0:
    """Evaluate the current global incumbent without refitting any coefficient."""

    values = rho.coeffs
    if baseline_key == "greenshields":
        local = jnp.exp(
            0.02312297116054006
            - 0.08831739725970279 * values
            - 0.1500812210203765 * values**2
        )
        multiplier = local * jnp.exp(-293.4581091232274 * contrast.coeffs)
    elif baseline_key == "idm":
        hodge_gate = hquad(rho).coeffs * contrast.coeffs
        multiplier = jnp.exp(-46.991435975062515 * contrast.coeffs) * jnp.exp(
            -119537.38834933033 * hodge_gate
        )
    elif baseline_key == "weidmann":
        multiplier = jnp.exp(
            0.1365440521436772 * values * (1.0 - values / 0.80612097)
        )
    elif baseline_key == "triangular":
        multiplier = jnp.ones_like(values)
    elif baseline_key == "del_castillo":
        multiplier = jnp.exp(0.0054952026563039776) * jnp.exp(
            -2.1218116018455078 * conv_3.coeffs
            - 256.86922950791177 * contrast.coeffs
        )
    else:
        raise KeyError(baseline_key)
    return C.CochainP0(rho.complex, multiplier)

