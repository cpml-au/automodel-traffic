"""Direct affine pointwise correction for a fixed traffic FD."""

from typing import Sequence

from dctkit.dec import cochain as C


def correction(
    rho: C.CochainP0, parameters: Sequence[float]
) -> C.CochainP0:
    """Return ``1 + a*rho`` as a primal 0-cochain."""

    (a,) = parameters
    return C.CochainP0(rho.complex, 1.0 + a * rho.coeffs)
