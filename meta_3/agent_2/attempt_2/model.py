"""Meta-3 Hodge-star attempt 2: square on the dual one-cochain."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


ATTEMPT = 2
PARAMETER_NAMES = ("a",)
BOUNDS = ((-2500.0, 2500.0),)
SYMBOLIC_EXPRESSION = "exp(a*St_oneD1(SquareD1(St_oneP0(rho))))"
TYPED_GP_EXPRESSION = (
    "ExpP0(MFP0(St_oneD1(SquareD1(St_oneP0(rho))), a))"
)
# ExpP0, MFP0, St_oneD1, SquareD1, St_oneP0, rho, a.
TREE_NODES = 7
HODGE_PRIMITIVES = ("St_oneP0", "St_oneD1")


def correction(
    rho: C.CochainP0, parameters: Sequence[float]
) -> C.CochainP0:
    """Return an exponential of the dual-square/back-star density feature."""

    (a,) = parameters
    dual_density = C.star(rho)
    dual_square = C.cochain_mul(dual_density, dual_density)
    feature = C.star(dual_square)
    scale = C.CochainP0(rho.complex, a * jnp.ones_like(feature.coeffs))
    exponent = C.cochain_mul(feature, scale)
    return C.CochainP0(rho.complex, jnp.exp(exponent.coeffs))
