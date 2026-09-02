"""Regression tests for the typed SR-Traffic DEC primitive grammar."""

from pathlib import Path

import jax.numpy as jnp
from dctkit.dec import cochain as C
from deap import gp
from flex.gp import primitives, util

from sr_traffic.data.data import preprocess_data
from sr_traffic.sr.primitives import add_new_primitives
from sr_traffic.utils.flat import define_flats


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "sr_traffic"
    / "sr"
    / "sr_traffic.yaml"
)


def build_traffic_pset():
    """Build the same typed primitive set used by the SR entry point."""

    _, config = util.load_config_data(CONFIG)
    data = preprocess_data("I80")
    complex_ = data["S"]
    zeros_p = C.CochainP0(complex_, jnp.zeros_like(data["vP0"][:, 0]))
    zeros_d = C.CochainD0(complex_, jnp.zeros_like(data["density"][:, 0]))
    all_flats = define_flats(complex_, zeros_p, zeros_d)

    pset = gp.PrimitiveSetTyped("MAIN", [C.CochainP0], C.CochainP0)
    add_new_primitives(pset, complex_, all_flats)
    pset.addTerminal(object, float, "c")
    pset.addTerminal(C.CochainP0(complex_, jnp.ones(complex_.num_nodes)), C.CochainP0, "ones")
    pset.renameArguments(ARG0="rho")
    pset = primitives.add_primitives_to_pset_from_dict(
        pset, config["gp"]["primitives"]
    )
    return pset, complex_


def test_hodge_star_variants_are_registered_and_compilable():
    pset, complex_ = build_traffic_pset()

    assert "St_oneP0" in pset.mapping
    assert "St_oneD1" in pset.mapping

    expression = gp.PrimitiveTree.from_string(
        "St_oneD1(St_oneP0(rho))", pset
    )
    function = gp.compile(expression, pset)
    rho = C.CochainP0(complex_, jnp.linspace(0.01, 0.5, complex_.num_nodes))
    result = function(rho)

    # DCTKit's star returns the generic Cochain runtime class; the GP wrapper's
    # typed return is represented by these primal/degree attributes.
    assert result.is_primal
    assert result.dim == 0
    assert result.coeffs.shape == rho.coeffs.shape
    assert bool(jnp.all(jnp.isfinite(result.coeffs)))
