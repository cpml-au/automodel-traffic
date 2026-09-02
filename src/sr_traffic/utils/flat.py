import jax.numpy as jnp
from jax import Array
import dctkit as dt
import dctkit.dec.cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.dec.flat import flat
from functools import partial
import numpy.typing as npt


def get_linear_left_interpolation(
    c: C.CochainP0 | C.CochainD0, sigma: C.CochainP0 | C.CochainD0
) -> Array:
    if c.is_primal:
        primal_volumes = c.complex.primal_volumes[1].reshape(-1, 1)
        coeffs = jnp.zeros(
            (c.complex.num_nodes - 1, c.coeffs.shape[1]), dtype=dt.float_dtype
        )
        coeffs = c.coeffs[1:] - 1 / 2 * primal_volumes * sigma.coeffs[1:]
        return coeffs

    dual_volumes = c.complex.dual_volumes[0].reshape(-1, 1)
    coeffs = jnp.zeros((c.complex.num_nodes, c.coeffs.shape[1]), dtype=dt.float_dtype)
    coeffs = coeffs.at[:-1].set(c.coeffs - 1 / 2 * dual_volumes[:-1] * sigma.coeffs)
    return coeffs


def get_linear_right_interpolation(
    c: C.CochainP0 | C.CochainD0, sigma: C.CochainP0 | C.CochainD0
) -> Array:
    if c.is_primal:
        primal_volumes = c.complex.primal_volumes[1].reshape(-1, 1)
        coeffs = jnp.zeros(
            (c.complex.num_nodes - 1, c.coeffs.shape[1]), dtype=dt.float_dtype
        )
        coeffs = c.coeffs[:-1] + 1 / 2 * primal_volumes * sigma.coeffs[:-1]
        return coeffs

    dual_volumes = c.complex.dual_volumes[0].reshape(-1, 1)
    coeffs = jnp.zeros((c.complex.num_nodes, c.coeffs.shape[1]), dtype=dt.float_dtype)
    coeffs = coeffs.at[1:].set(c.coeffs + 1 / 2 * dual_volumes[1:] * sigma.coeffs)
    return coeffs


def define_flats(S: SimplicialComplex, zeros_P: npt.NDArray, zeros_D: npt.NDArray):
    I_linear_left = get_linear_left_interpolation
    I_linear_right = get_linear_right_interpolation
    primal_edges = C.CochainP1(S, S.primal_edges_vectors)
    dual_edges = C.CochainD1(S, S.dual_edges_vectors)
    flat_linear_left_D = partial(
        flat,
        weights=None,
        edges=dual_edges,
        interp_func=I_linear_left,
        interp_func_args={"sigma": zeros_D},
    )
    flat_linear_right_D = partial(
        flat,
        weights=None,
        edges=dual_edges,
        interp_func=I_linear_right,
        interp_func_args={"sigma": zeros_D},
    )
    flat_linear_left_P = partial(
        flat,
        weights=None,
        edges=primal_edges,
        interp_func=I_linear_left,
        interp_func_args={"sigma": zeros_P},
    )
    flat_linear_right_P = partial(
        flat,
        weights=None,
        edges=primal_edges,
        interp_func=I_linear_right,
        interp_func_args={"sigma": zeros_P},
    )
    flats = {
        "flat_linear_left_D": flat_linear_left_D,
        "flat_linear_right_D": flat_linear_right_D,
        "flat_linear_left_P": flat_linear_left_P,
        "flat_linear_right_P": flat_linear_right_P,
    }

    return flats
