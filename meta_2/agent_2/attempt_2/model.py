"""Attempt 2: positive exponential quadratic in centered density."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


ATTEMPT = 2
PARAMETER_NAMES = ("a", "b")
SYMBOLIC_EXPRESSION = "exp(a*(rho-r*) + b*(rho-r*)^2)"
# Exp and Add, plus centered linear (5 nodes) and quadratic (7 nodes) terms.
TREE_NODES = 14


def make_correction(r_star: float) -> Callable:
    """Create ``g(rho)=exp(a*x+b*x^2)``, with ``x=rho-r*``."""

    def correction(rho: C.CochainP0, parameters: Sequence[float]) -> C.CochainP0:
        a, b = parameters
        x = rho.coeffs - r_star
        return C.CochainP0(rho.complex, jnp.exp(a * x + b * jnp.square(x)))

    return correction


def expression(r_star: float) -> str:
    return f"exp(a*(rho-{r_star:.12g}) + b*(rho-{r_star:.12g})^2)"
