"""Meta-3 Hodge-star attempt 1: double-star density feature."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


ATTEMPT = 1
PARAMETER_NAMES = ("a",)
BOUNDS = ((-5.0, 5.0),)
SYMBOLIC_EXPRESSION = "exp(a*St_oneD1(St_oneP0(rho)))"
TYPED_GP_EXPRESSION = "ExpP0(MFP0(St_oneD1(St_oneP0(rho)), a))"
# ExpP0, MFP0, St_oneD1, St_oneP0, rho, a.
TREE_NODES = 6
HODGE_PRIMITIVES = ("St_oneP0", "St_oneD1")


def correction(
    rho: C.CochainP0, parameters: Sequence[float]
) -> C.CochainP0:
    """Return ``exp(a * star(star(rho)))`` as a primal zero-cochain."""

    (a,) = parameters
    feature = C.star(C.star(rho))
    scale = C.CochainP0(rho.complex, a * jnp.ones_like(feature.coeffs))
    exponent = C.cochain_mul(feature, scale)
    return C.CochainP0(rho.complex, jnp.exp(exponent.coeffs))
