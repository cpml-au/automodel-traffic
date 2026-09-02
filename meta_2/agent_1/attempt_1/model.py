"""Attempt 1: positive global-scale multiplier with a log-scale intercept."""

from collections.abc import Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


def correction(rho: C.CochainP0, parameters: Sequence[float]) -> C.CochainP0:
    """Return ``g(rho) = exp(c0)`` as a primal 0-cochain."""

    (c0,) = parameters
    values = jnp.exp(c0) * jnp.ones_like(rho.coeffs)
    return C.CochainP0(rho.complex, values)

