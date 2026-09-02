"""Attempt 3: cubic positive jam-anchored multiplier."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


ATTEMPT = 3
PARAMETER_NAMES = ("a", "b", "c")
SYMBOLIC_EXPRESSION = "exp((a*rho+b*rho^2+c*rho^3)*(1-rho/r_j))"
# GP-style count with rho^3 represented by multiplying rho and Square(rho).
TREE_NODES = 21


def make_correction(r_j: float) -> Callable:
    """Return the prescribed cubic anchored multiplier for a fixed ``r_j``."""

    def correction(
        rho: C.CochainP0, parameters: Sequence[float]
    ) -> C.CochainP0:
        a, b, c = parameters
        density = rho.coeffs
        polynomial = a * density + b * density**2 + c * density**3
        exponent = polynomial * (1.0 - density / r_j)
        return C.CochainP0(rho.complex, jnp.exp(exponent))

    return correction


def expression(r_j: float) -> str:
    return f"exp((a*rho+b*rho^2+c*rho^3)*(1-rho/{r_j:.12g}))"
