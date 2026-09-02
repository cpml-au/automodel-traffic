"""Meta-3 Hodge-star attempt 3: linear combination of both star features."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C


ATTEMPT = 3
PARAMETER_NAMES = ("a", "b")
BOUNDS = ((-5.0, 5.0), (-2500.0, 2500.0))
SYMBOLIC_EXPRESSION = (
    "exp(a*St_oneD1(St_oneP0(rho)) + "
    "b*St_oneD1(SquareD1(St_oneP0(rho))))"
)
TYPED_GP_EXPRESSION = (
    "ExpP0(AddCP0(MFP0(St_oneD1(St_oneP0(rho)), a), "
    "MFP0(St_oneD1(SquareD1(St_oneP0(rho))), b)))"
)
# ExpP0 + AddCP0 + two coefficient-feature products: 13 nodes total.
TREE_NODES = 13
HODGE_PRIMITIVES = ("St_oneP0", "St_oneD1")


def correction(
    rho: C.CochainP0, parameters: Sequence[float]
) -> C.CochainP0:
    """Return an exponential combining double-star and dual-square features."""

    a, b = parameters
    first = C.star(C.star(rho))
    dual_density = C.star(rho)
    second = C.star(C.cochain_mul(dual_density, dual_density))
    a_field = C.CochainP0(rho.complex, a * jnp.ones_like(first.coeffs))
    b_field = C.CochainP0(rho.complex, b * jnp.ones_like(second.coeffs))
    first_term = C.cochain_mul(first, a_field)
    second_term = C.cochain_mul(second, b_field)
    exponent = first_term.coeffs + second_term.coeffs
    return C.CochainP0(rho.complex, jnp.exp(exponent))
