"""Attempt 3: positive exponential-cubic multiplier."""

from collections.abc import Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


def correction(rho: C.CochainP0, parameters: Sequence[float]) -> C.CochainP0:
    """Return ``g(rho) = exp(a*rho + b*rho**2 + c*rho**3)``."""

    a, b, c = parameters
    exponent = a * rho.coeffs + b * rho.coeffs**2 + c * rho.coeffs**3
    return C.CochainP0(rho.complex, jnp.exp(exponent))
