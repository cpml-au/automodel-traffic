"""Global incumbent times a sign-preserving cubic contrast response."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C

from meta_5.agent_1 import common


PARAMETER_NAMES = ("a",)
BOUNDS = ((-1_000_000.0, 1_000_000.0),)
TEMPLATE = "g_inc*exp(a*C^3), C=conv_3(rho,ones)-3*conv_1(rho,ones)"
# Cube is represented in the available grammar as C*SquareP0(C), with the C
# subtree counted twice. The full factor plus incumbent product adds 24 nodes.
ADDED_NODES = 24


def make_correction(baseline_key: str):
    """Return fixed incumbent times ``exp(a*C^3)``."""

    def correction(rho: C.CochainP0, parameters: Sequence[float]):
        (a,) = parameters
        conv_3, _, contrast = common.convolution_features(rho)
        incumbent = common.incumbent(baseline_key, rho, conv_3, contrast)
        factor = jnp.exp(a * contrast.coeffs * contrast.coeffs**2)
        return C.CochainP0(rho.complex, incumbent.coeffs * factor)

    return correction


def expression(baseline_key: str) -> str:
    return (
        f"({common.INCUMBENT_EXPRESSIONS[baseline_key]})"
        "*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^3)"
    )

