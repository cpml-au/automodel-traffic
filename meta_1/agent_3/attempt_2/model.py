"""Saturating positive quadratic correction for a fixed traffic FD."""

from typing import Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


def correction(
    rho: C.CochainP0, parameters: Sequence[float]
) -> C.CochainP0:
    """Return ``exp((a*rho+b*rho^2)/(1+rho))`` pointwise."""

    a, b = parameters
    x = rho.coeffs
    exponent = (a * x + b * jnp.square(x)) / (1.0 + x)
    return C.CochainP0(rho.complex, jnp.exp(exponent))
