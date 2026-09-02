"""Global incumbent times combined linear and quadratic contrast terms."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C

from meta_5.agent_1 import common


PARAMETER_NAMES = ("a", "b")
BOUNDS = ((-300.0, 300.0), (-20_000.0, 20_000.0))
TEMPLATE = "g_inc*exp(a*C+b*C^2), C=conv_3(rho,ones)-3*conv_1(rho,ones)"
# The symbolic a*C and b*Square(C) branches count repeated C subtrees. The
# complete exponential increment plus incumbent product root adds 26 nodes.
ADDED_NODES = 26


def make_correction(baseline_key: str):
    """Return fixed incumbent times ``exp(a*C+b*C^2)``."""

    def correction(rho: C.CochainP0, parameters: Sequence[float]):
        a, b = parameters
        conv_3, _, contrast = common.convolution_features(rho)
        incumbent = common.incumbent(baseline_key, rho, conv_3, contrast)
        exponent = a * contrast.coeffs + b * contrast.coeffs**2
        return C.CochainP0(
            rho.complex, incumbent.coeffs * jnp.exp(exponent)
        )

    return correction


def expression(baseline_key: str) -> str:
    contrast = "(conv_3(rho,ones)-3*conv_1(rho,ones))"
    return (
        f"({common.INCUMBENT_EXPRESSIONS[baseline_key]})"
        f"*exp(a*{contrast}+b*{contrast}^2)"
    )

