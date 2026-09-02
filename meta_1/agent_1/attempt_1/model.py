"""Attempt 1: positive exponential-linear multiplier."""

from collections.abc import Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


def correction(rho: C.CochainP0, parameters: Sequence[float]) -> C.CochainP0:
    """Return ``g(rho) = exp(a * rho)`` as a primal 0-cochain."""

    (a,) = parameters
    return C.CochainP0(rho.complex, jnp.exp(a * rho.coeffs))
