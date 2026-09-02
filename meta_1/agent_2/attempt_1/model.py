"""Attempt 1: one-parameter positive jam-anchored multiplier."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


ATTEMPT = 1
PARAMETER_NAMES = ("a",)
SYMBOLIC_EXPRESSION = "exp(a*rho*(1-rho/r_j))"
# Explicit GP-style count: Exp, two Mul, Sub, Div, and five terminals.
TREE_NODES = 10


def make_correction(r_j: float) -> Callable:
    """Return ``g(rho; a)=exp(a*rho*(1-rho/r_j))`` for a fixed jam density.

    The exponential makes the multiplier strictly positive, while the final
    factor makes its exponent zero at ``rho=r_j`` and hence anchors ``g(r_j)=1``.
    """

    def correction(
        rho: C.CochainP0, parameters: Sequence[float]
    ) -> C.CochainP0:
        (a,) = parameters
        values = jnp.exp(a * rho.coeffs * (1.0 - rho.coeffs / r_j))
        return C.CochainP0(rho.complex, values)

    return correction


def expression(r_j: float) -> str:
    return f"exp(a*rho*(1-rho/{r_j:.12g}))"
