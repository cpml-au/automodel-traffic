"""Attempt 2: positive exponential-linear multiplier with an intercept."""

from collections.abc import Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


def correction(rho: C.CochainP0, parameters: Sequence[float]) -> C.CochainP0:
    """Return ``g(rho) = exp(c0 + a*rho)`` as a primal 0-cochain."""

    c0, a = parameters
    values = jnp.exp(c0 + a * rho.coeffs)
    return C.CochainP0(rho.complex, values)

