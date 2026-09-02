"""Combined one- and three-point DEC convolution multiplier."""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


def correction(
    rho: C.CochainP0, parameters: Sequence[float]
) -> C.CochainP0:
    """Return ``exp(a * conv_1(rho, ones) + b * conv_3(rho, ones))``."""

    a, b = parameters
    ones = C.CochainP0(rho.complex, jnp.ones_like(rho.coeffs))
    local_feature = C.convolution(rho, ones, kernel_window=1)
    neighborhood_feature = C.convolution(rho, ones, kernel_window=3)
    exponent = a * local_feature.coeffs + b * neighborhood_feature.coeffs
    return C.CochainP0(rho.complex, jnp.exp(exponent))
