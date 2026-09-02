"""Global meta-4 incumbents and DEC features for meta-5 agent 2."""

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
        "*exp(-2.12181160185*conv_3-256.869229508*C)"
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


def scalar_cochain(rho: C.CochainP0, value: float) -> C.CochainP0:
    return C.CochainP0(rho.complex, value * jnp.ones_like(rho.coeffs))


def convolution_features(rho: C.CochainP0):
    """Return exact DEC convolutions and C=conv_3-3*conv_1."""

    ones = C.CochainP0(rho.complex, jnp.ones_like(rho.coeffs))
    conv_3 = C.convolution(rho, ones, kernel_window=3)
    conv_1 = C.convolution(rho, ones, kernel_window=1)
    scaled_conv_1 = C.cochain_mul(conv_1, scalar_cochain(rho, 3.0))
    contrast = C.CochainP0(
        rho.complex, conv_3.coeffs - scaled_conv_1.coeffs
    )
    return conv_3, conv_1, contrast


def hquad(rho: C.CochainP0) -> C.CochainP0:
    dual = C.star(rho)
    dual_square = C.cochain_mul(dual, dual)
    return C.star(dual_square)


def incumbent(baseline_key: str, rho: C.CochainP0) -> C.CochainP0:
    """Evaluate a fixed global incumbent from final_candidates.json."""

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
        base = C.CochainP0(
            rho.complex, jnp.exp(-46.991435975062515 * contrast.coeffs)
        )
        gate = C.cochain_mul(hquad(rho), contrast)
        hodge = C.CochainP0(
            rho.complex, jnp.exp(-119537.38834933033 * gate.coeffs)
        )
        return C.cochain_mul(base, hodge)
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
        exponent = (
            0.0054952026563039776
            - 2.1218116018455078 * conv_3.coeffs
            - 256.86922950791177 * contrast.coeffs
        )
        return C.CochainP0(rho.complex, jnp.exp(exponent))
    raise KeyError(baseline_key)


def rho_power(rho: C.CochainP0, power: int) -> C.CochainP0:
    if power == 1:
        return rho
    if power == 2:
        return C.cochain_mul(rho, rho)
    raise ValueError(power)


def density_gated_contrast(rho: C.CochainP0, power: int) -> C.CochainP0:
    _, _, contrast = convolution_features(rho)
    return C.cochain_mul(rho_power(rho, power), contrast)


def exp_cochain(rho: C.CochainP0, exponent: C.CochainP0) -> C.CochainP0:
    return C.CochainP0(rho.complex, jnp.exp(exponent.coeffs))
