"""Direct linear-over-linear pointwise correction for a fixed traffic FD."""

from typing import Sequence

from dctkit.dec import cochain as C


def denominator(rho_coeffs, parameters: Sequence[float]):
    """Expose the denominator for the fitter's explicit singularity check."""

    _, b = parameters
    return 1.0 + b * rho_coeffs


def correction(
    rho: C.CochainP0, parameters: Sequence[float]
) -> C.CochainP0:
    """Return ``(1 + a*rho)/(1 + b*rho)`` as a primal 0-cochain."""

    a, _ = parameters
    x = rho.coeffs
    return C.CochainP0(rho.complex, (1.0 + a * x) / denominator(x, parameters))
