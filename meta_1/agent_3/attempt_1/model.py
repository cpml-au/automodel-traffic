"""Saturating positive one-term correction for a fixed traffic FD."""

from typing import Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


def correction(
    rho: C.CochainP0, parameters: Sequence[float]
) -> C.CochainP0:
    """Return ``exp(a*rho/(1+rho))`` pointwise.

    The exponential makes the multiplier strictly positive, while the
    denominator limits the exponent's growth at high density.
    """

    (a,) = parameters
    x = rho.coeffs
    return C.CochainP0(rho.complex, jnp.exp(a * x / (1.0 + x)))
