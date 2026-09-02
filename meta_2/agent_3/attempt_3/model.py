"""Direct quadratic-over-linear pointwise correction for a fixed traffic FD."""

from typing import Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


def denominator(rho_coeffs, parameters: Sequence[float]):
    """Expose the denominator for the fitter's explicit singularity check."""

    _, _, c = parameters
    return 1.0 + c * rho_coeffs


def correction(
    rho: C.CochainP0, parameters: Sequence[float]
) -> C.CochainP0:
    """Return ``(1+a*rho+b*rho^2)/(1+c*rho)`` as a primal 0-cochain."""

    a, b, _ = parameters
    x = rho.coeffs
    numerator = 1.0 + a * x + b * jnp.square(x)
    return C.CochainP0(rho.complex, numerator / denominator(x, parameters))
