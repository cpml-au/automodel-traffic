"""Saturating positive cubic correction for a fixed traffic FD."""

from typing import Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


def correction(
    rho: C.CochainP0, parameters: Sequence[float]
) -> C.CochainP0:
    """Return ``exp((a*rho+b*rho^2+c*rho^3)/(1+rho))`` pointwise."""

    a, b, c = parameters
    x = rho.coeffs
    x2 = jnp.square(x)
    exponent = (a * x + b * x2 + c * x2 * x) / (1.0 + x)
    return C.CochainP0(rho.complex, jnp.exp(exponent))
