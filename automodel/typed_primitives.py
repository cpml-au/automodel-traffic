"""Typed DEC primitive grammar used by the Automodel search records."""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from deap import gp
from flex.gp import primitives
from flex.gp.cochain_primitives import (
    CochainBasePrimitive,
    Complex,
    Dimension,
    Rank,
    generate_primitive_variants,
)
from jax import vmap

from sr_traffic.data.data import preprocess_data
from sr_traffic.utils.flat import define_flats

_SCALAR_DIMENSIONS = [Dimension.ZERO, Dimension.ONE]
_SCALAR_RANK = [Rank.SCALAR]
_PRIMITIVE_CONFIG = {
    "imports": {
        "flex.gp.cochain_primitives": ["coch_primitives"],
        "flex.gp.jax_primitives": ["jax_primitives"],
    },
    "used": [
        {"name": name, "dimension": None, "rank": None}
        for name in ("AddF", "SubF", "MulF", "Div", "SquareF", "SqrtF", "ExpF")
    ]
    + [
        {"name": name, "dimension": _SCALAR_DIMENSIONS, "rank": _SCALAR_RANK}
        for name in ("AddC", "SubC", "Square", "Sqrt", "MF", "Exp", "St_one", "CMul")
    ]
    + [
        {"name": "cob", "dimension": [Dimension.ZERO], "rank": _SCALAR_RANK},
        {"name": "del", "dimension": [Dimension.ONE], "rank": _SCALAR_RANK},
    ],
}


def _add_traffic_primitives(
    pset: gp.PrimitiveSetTyped,
    complex_: SimplicialComplex,
    all_flats: dict,
) -> None:
    modules = {"dctkit.dec": ["cochain"]}
    for window in (1, 3):
        base = CochainBasePrimitive(
            base_name=f"conv_{window}",
            base_fun=partial(C.convolution, kernel_window=window),
            input=["cochain.Cochain", "cochain.Cochain"],
            output="cochain.Cochain",
            att_input={
                "complex": (Complex.PRIMAL,),
                "dimension": (Dimension.ZERO,),
                "rank": (Rank.SCALAR,),
            },
            map_rule={
                "complex": lambda value: value,
                "dimension": lambda value: value,
                "rank": lambda value: value,
            },
        )
        variants = generate_primitive_variants(base, imports=modules)
        for name, primitive in variants.items():
            pset.addPrimitive(
                primitive.op,
                primitive.in_types,
                primitive.out_type,
                name=name,
            )

    def flat_left(values):
        return all_flats["flat_linear_left_P"](C.CochainP0(complex_, values)).coeffs

    def flat_right(values):
        return all_flats["flat_linear_right_P"](C.CochainP0(complex_, values)).coeffs

    flat_left_vmap = vmap(flat_left)
    flat_right_vmap = vmap(flat_right)

    def flat_left_primitive(cochain: C.CochainD0) -> C.CochainP1:
        coefficients = flat_left_vmap(cochain.coeffs.T)[:, :, 0].T
        return C.CochainP1(complex_, coefficients)

    def flat_right_primitive(cochain: C.CochainD0) -> C.CochainP1:
        coefficients = flat_right_vmap(cochain.coeffs.T)[:, :, 0].T
        return C.CochainP1(complex_, coefficients)

    pset.addPrimitive(
        flat_left_primitive,
        [C.CochainP0],
        C.CochainP1,
        name="flat_lin_leftP0",
    )
    pset.addPrimitive(
        flat_right_primitive,
        [C.CochainP0],
        C.CochainP1,
        name="flat_lin_rightP0",
    )


def build_traffic_pset() -> tuple[gp.PrimitiveSetTyped, SimplicialComplex]:
    """Build the typed primitive set used by Automodel's Hodge searches."""

    data = preprocess_data("I80")
    complex_ = data["S"]
    zeros_p = C.CochainP0(complex_, jnp.zeros_like(data["vP0"][:, 0]))
    zeros_d = C.CochainD0(complex_, jnp.zeros_like(data["density"][:, 0]))
    all_flats = define_flats(complex_, zeros_p, zeros_d)

    pset = gp.PrimitiveSetTyped("MAIN", [C.CochainP0], C.CochainP0)
    _add_traffic_primitives(pset, complex_, all_flats)
    pset.addTerminal(object, float, "c")
    pset.addTerminal(
        C.CochainP0(complex_, jnp.ones(complex_.num_nodes)),
        C.CochainP0,
        "ones",
    )
    pset.renameArguments(ARG0="rho")
    pset = primitives.add_primitives_to_pset_from_dict(pset, _PRIMITIVE_CONFIG)
    return pset, complex_
