"""Attempt 3: positive exponential-quadratic multiplier with an intercept."""

from collections.abc import Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


def correction(rho: C.CochainP0, parameters: Sequence[float]) -> C.CochainP0:
    """Return ``g(rho) = exp(c0 + a*rho + b*rho**2)``."""

    c0, a, b = parameters
    values = jnp.exp(c0 + a * rho.coeffs + b * rho.coeffs**2)
    return C.CochainP0(rho.complex, values)

