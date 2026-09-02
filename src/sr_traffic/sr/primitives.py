from flex.gp.cochain_primitives import *
from flex.gp.jax_primitives import *
from functools import partial
from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from jax import vmap
from deap import gp
from typing import Dict


def add_new_primitives(
    pset: gp.PrimitiveSetTyped, S: SimplicialComplex, all_flats: Dict
):
    # Define the modules and functions needed to eval inputs and outputs
    modules_functions = {"dctkit.dec": ["cochain"]}

    new_primitives = []
    for i in (1, 3):
        conv_i = CochainBasePrimitive(
            base_name="conv_" + str(i),
            base_fun=partial(C.convolution, kernel_window=int(i)),
            input=["cochain.Cochain", "cochain.Cochain"],
            output="cochain.Cochain",
            att_input={
                # Only the primal zero-cochain variant is used.
                "complex": (Complex.PRIMAL,),
                "dimension": (Dimension.ZERO,),
                "rank": (Rank.SCALAR,),
            },
            map_rule={
                "complex": lambda x: x,
                "dimension": lambda x: x,
                "rank": lambda x: x,
            },
        )
        new_primitives.append(conv_i)

    new_generated_primitives = list(
        map(
            partial(generate_primitive_variants, imports=modules_functions),
            new_primitives,
        )
    )
    for new_primitive in new_generated_primitives:
        for primitive_name in new_primitive.keys():
            op = new_primitive[primitive_name].op
            in_types = new_primitive[primitive_name].in_types
            out_type = new_primitive[primitive_name].out_type
            pset.addPrimitive(op, in_types, out_type, name=primitive_name)

    def flat_lin_left_P_wrap(x):
        return all_flats["flat_linear_left_P"](C.CochainP0(S, x)).coeffs

    def flat_lin_right_P_wrap(x):
        return all_flats["flat_linear_right_P"](C.CochainP0(S, x)).coeffs

    flat_lin_left_P = vmap(flat_lin_left_P_wrap)
    flat_lin_right_P = vmap(flat_lin_right_P_wrap)

    def flat_primitive_lin_left_P(c: C.CochainD0):
        return C.CochainP1(S, flat_lin_left_P(c.coeffs.T)[:, :, 0].T)

    def flat_primitive_lin_right_P(c: C.CochainD0):
        return C.CochainP1(S, flat_lin_right_P(c.coeffs.T)[:, :, 0].T)

    pset.addPrimitive(
        flat_primitive_lin_left_P,
        [C.CochainP0],
        C.CochainP1,
        name="flat_lin_leftP0",
    )
    pset.addPrimitive(
        flat_primitive_lin_right_P,
        [C.CochainP0],
        C.CochainP1,
        name="flat_lin_rightP0",
    )
