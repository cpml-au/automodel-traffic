"""Attempt 2: positive exponential-quadratic multiplier."""

from collections.abc import Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


def correction(rho: C.CochainP0, parameters: Sequence[float]) -> C.CochainP0:
    """Return ``g(rho) = exp(a * rho + b * rho**2)``."""

    a, b = parameters
    exponent = a * rho.coeffs + b * rho.coeffs**2
    return C.CochainP0(rho.complex, jnp.exp(exponent))
