"""Attempt 1: positive exponential linear in centered density."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


ATTEMPT = 1
PARAMETER_NAMES = ("a",)
SYMBOLIC_EXPRESSION = "exp(a*(rho-r*))"
# Exp, Mul, Sub, and terminals a, rho, r*.
TREE_NODES = 6


def make_correction(r_star: float) -> Callable:
    """Create ``g(rho)=exp(a*(rho-r*))`` for a fixed FD center."""

    def correction(rho: C.CochainP0, parameters: Sequence[float]) -> C.CochainP0:
        (a,) = parameters
        x = rho.coeffs - r_star
        return C.CochainP0(rho.complex, jnp.exp(a * x))

    return correction


def expression(r_star: float) -> str:
    return f"exp(a*(rho-{r_star:.12g}))"
