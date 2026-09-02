"""Attempt 2: quadratic positive jam-anchored multiplier."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


ATTEMPT = 2
PARAMETER_NAMES = ("a", "b")
SYMBOLIC_EXPRESSION = "exp((a*rho+b*rho^2)*(1-rho/r_j))"
# GP-style count with one Square and the fixed r_j represented as a terminal.
TREE_NODES = 15


def make_correction(r_j: float) -> Callable:
    """Return ``g=exp((a*rho+b*rho^2)*(1-rho/r_j))`` for fixed ``r_j``."""

    def correction(
        rho: C.CochainP0, parameters: Sequence[float]
    ) -> C.CochainP0:
        a, b = parameters
        density = rho.coeffs
        exponent = (a * density + b * density**2) * (1.0 - density / r_j)
        return C.CochainP0(rho.complex, jnp.exp(exponent))

    return correction


def expression(r_j: float) -> str:
    return f"exp((a*rho+b*rho^2)*(1-rho/{r_j:.12g}))"
