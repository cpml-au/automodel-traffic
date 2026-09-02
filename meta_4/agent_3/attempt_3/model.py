"""Fixed meta-3 incumbents with square-root and quadratic density controls."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C

from meta_4.agent_3.attempt_1.model import INCUMBENT_EXPRESSIONS, _incumbent_values


def make_correction(baseline_key: str) -> Callable:
    """Return ``g_inc*exp(a*sqrt(max(rho,0))+b*rho^2)``."""

    def correction(rho: C.CochainP0, parameters: Sequence[float]) -> C.CochainP0:
        a, b = parameters
        protected_sqrt = jnp.sqrt(jnp.maximum(rho.coeffs, 0.0))
        exponent = a * protected_sqrt + b * rho.coeffs**2
        values = _incumbent_values(baseline_key, rho) * jnp.exp(exponent)
        return C.CochainP0(rho.complex, values)

    return correction


def expression(baseline_key: str) -> str:
    return (
        f"({INCUMBENT_EXPRESSIONS[baseline_key]})"
        "*exp(a*SqrtP0(rho)+b*rho^2)"
    )
