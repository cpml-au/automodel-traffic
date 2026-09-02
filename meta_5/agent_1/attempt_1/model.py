"""Global incumbent times an even quadratic convolution-contrast factor."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C

from meta_5.agent_1 import common


PARAMETER_NAMES = ("a",)
BOUNDS = ((-20_000.0, 20_000.0),)
TEMPLATE = "g_inc*exp(a*C^2), C=conv_3(rho,ones)-3*conv_1(rho,ones)"
# Increment: product root + exp + scalar multiply + coefficient + SquareP0 + C.
# C contains nine nodes. Total incremental contribution is 14 nodes.
ADDED_NODES = 14
BOUND_NOTE = (
    "A preflight range of [-1e6,1e6] overflowed exp(a*C^2) in full "
    "simulations for all FDs. The accepted synchronous run uses [-2e4,2e4]."
)


def make_correction(baseline_key: str):
    """Return fixed incumbent times ``exp(a*C^2)`` using exact convolution."""

    def correction(rho: C.CochainP0, parameters: Sequence[float]):
        (a,) = parameters
        conv_3, _, contrast = common.convolution_features(rho)
        incumbent = common.incumbent(baseline_key, rho, conv_3, contrast)
        factor = jnp.exp(a * contrast.coeffs**2)
        return C.CochainP0(rho.complex, incumbent.coeffs * factor)

    return correction


def expression(baseline_key: str) -> str:
    return (
        f"({common.INCUMBENT_EXPRESSIONS[baseline_key]})"
        "*exp(a*(conv_3(rho,ones)-3*conv_1(rho,ones))^2)"
    )
