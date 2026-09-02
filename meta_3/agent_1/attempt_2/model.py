"""Direct three-point DEC convolution multiplier for meta-iteration 3."""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


def correction(
    rho: C.CochainP0, parameters: Sequence[float]
) -> C.CochainP0:
    """Return ``exp(a * convolution(rho, ones, kernel_window=3))``."""

    (a,) = parameters
    ones = C.CochainP0(rho.complex, jnp.ones_like(rho.coeffs))
    feature = C.convolution(rho, ones, kernel_window=3)
    return C.CochainP0(rho.complex, jnp.exp(a * feature.coeffs))
