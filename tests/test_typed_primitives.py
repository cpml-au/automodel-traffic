"""Regression tests for the typed DEC primitive grammar used by Automodel."""

import jax.numpy as jnp
from dctkit.dec import cochain as C
from deap import gp

from automodel.typed_primitives import build_traffic_pset


def test_hodge_star_variants_are_registered_and_compilable():
    pset, complex_ = build_traffic_pset()

    assert "St_oneP0" in pset.mapping
    assert "St_oneD1" in pset.mapping

    expression = gp.PrimitiveTree.from_string("St_oneD1(St_oneP0(rho))", pset)
    function = gp.compile(expression, pset)
    rho = C.CochainP0(complex_, jnp.linspace(0.01, 0.5, complex_.num_nodes))
    result = function(rho)

    # DCTKit's star returns the generic Cochain runtime class; the GP wrapper's
    # typed return is represented by these primal/degree attributes.
    assert result.is_primal
    assert result.dim == 0
    assert result.coeffs.shape == rho.coeffs.shape
    assert bool(jnp.all(jnp.isfinite(result.coeffs)))
