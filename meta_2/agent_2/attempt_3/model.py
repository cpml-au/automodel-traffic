"""Attempt 3: positive exponential cubic in centered density."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


ATTEMPT = 3
PARAMETER_NAMES = ("a", "b", "c")
SYMBOLIC_EXPRESSION = "exp(a*(rho-r*) + b*(rho-r*)^2 + c*(rho-r*)^3)"
# Exp, two Adds, centered linear (5), quadratic (7), and cubic (7) terms.
TREE_NODES = 22


def make_correction(r_star: float) -> Callable:
    """Create ``g(rho)=exp(a*x+b*x^2+c*x^3)``, with ``x=rho-r*``."""

    def correction(rho: C.CochainP0, parameters: Sequence[float]) -> C.CochainP0:
        a, b, c = parameters
        x = rho.coeffs - r_star
        x2 = jnp.square(x)
        return C.CochainP0(rho.complex, jnp.exp(a * x + b * x2 + c * x2 * x))

    return correction


def expression(r_star: float) -> str:
    return (
        f"exp(a*(rho-{r_star:.12g}) + b*(rho-{r_star:.12g})^2 + "
        f"c*(rho-{r_star:.12g})^3)"
    )
