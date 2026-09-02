import jax.numpy as jnp
import dctkit.dec.cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from jax import vmap, lax, jacfwd
from functools import partial
import numpy.typing as npt
from typing import Callable, Sequence


def Greenshields_flux(rho: C.Cochain, v_max: float, rho_max: float):
    return C.Cochain(
        rho.dim,
        rho.is_primal,
        rho.complex,
        v_max * rho.coeffs * (1 - rho.coeffs / rho_max),
    )


def Weidmann_flux(rho: C.Cochain, v_max: float, rho_max: float, lambda_w: float):
    return C.Cochain(
        rho.dim,
        rho.is_primal,
        rho.complex,
        rho.coeffs * Weidmann_v(rho.coeffs, v_max, rho_max, lambda_w),
    )


def triangular_flux(rho: C.Cochain, V_0: float, l_eff: float, T: float):
    rho_critic = 1 / (V_0 * T + l_eff)
    free_traffic_idx = rho.coeffs <= rho_critic
    congested_traffic_idx = (rho.coeffs > rho_critic) * (rho.coeffs <= 1 / l_eff)
    flux_interm = jnp.where(
        congested_traffic_idx,
        1 / T * (1 - rho.coeffs * l_eff),
        jnp.zeros_like(rho.coeffs),
    )
    flux_coeffs = jnp.where(free_traffic_idx, V_0 * rho.coeffs, flux_interm)
    return C.Cochain(rho.dim, rho.is_primal, rho.complex, flux_coeffs)


def Greenshields_v(rho: npt.NDArray, v_max: float, rho_max: float):
    return v_max * (1 - rho / rho_max)


def Weidmann_v(rho: npt.NDArray, v_max: float, rho_max: float, lambda_w: float):
    return v_max * (1 - jnp.exp(-lambda_w * (1 / rho - 1 / rho_max)))


def triangular_v(rho: npt.NDArray, V_0: float, l_eff: float, T: float):
    rho_critic = 1 / (V_0 * T + l_eff)
    free_traffic_idx = rho <= rho_critic
    congested_traffic_idx = (rho > rho_critic) * (rho <= 1 / l_eff)
    flux_interm = jnp.where(
        congested_traffic_idx, 1 / T * (1 / rho - l_eff), jnp.zeros_like(rho)
    )
    v_coeffs = jnp.where(free_traffic_idx, V_0, flux_interm)
    return v_coeffs


def IDM_eq(
    s: npt.NDArray, v: npt.NDArray, s0: float, T: float, delta: float, v0: float
):
    return 1 - (v / v0) ** delta - ((s0 + v * T) / s) ** 2


@partial(vmap, in_axes=(0, None, None, None, None))
def inverse_IDM(s_target: npt.NDArray, s0: float, T: float, delta: float, v0: float):
    """Invert the equilibrium IDM spacing relation on ``0 <= v <= v0``.

    The previous unconstrained Newton iteration could step to negative velocity;
    fractional values of ``delta`` then produced NaNs.  On the physical interval
    the equilibrium equation is monotone, so fixed-iteration bisection is both
    robust and compatible with JAX transformations.
    """

    spacing = jnp.maximum(s_target, s0)

    def body_fun(_, bounds):
        lower, upper = bounds
        midpoint = 0.5 * (lower + upper)
        residual = IDM_eq(spacing, midpoint, s0, T, delta, v0)
        lower = jnp.where(residual > 0, midpoint, lower)
        upper = jnp.where(residual > 0, upper, midpoint)
        return lower, upper

    lower, upper = lax.fori_loop(0, 64, body_fun, (0.0, v0))
    velocity = 0.5 * (lower + upper)
    return jnp.where(s_target > s0, velocity, 0.0)


def IDM_v(rho: npt.NDArray, s0: float, T: float, delta: float, v0: float):
    """Evaluate the equilibrium IDM velocity as a function of density.

    ``inverse_IDM`` accepts the net vehicle spacing ``s`` rather than density.
    This wrapper performs the same ``s = 1 / rho - 1`` conversion used by the
    flux function and returns zero once the spacing reaches the minimum gap.
    Clipping density away from zero keeps the conversion finite.
    """

    rho_safe = jnp.maximum(rho, jnp.finfo(jnp.asarray(rho).dtype).eps)
    spacing = 1 / rho_safe - 1
    return inverse_IDM(spacing, s0, T, delta, v0)


def IDM_flux(rho: C.Cochain, s0: float, T: float, delta: float, v0: float):
    rho_coeffs = rho.coeffs.ravel()
    v = IDM_v(rho_coeffs, s0, T, delta, v0)
    return C.Cochain(rho.dim, rho.is_primal, rho.complex, rho_coeffs * v)


def del_castillo_v(
    rho: npt.NDArray, C_jam: float, V_max: float, rho_max: float, theta: float
):
    rho_norm = rho / rho_max
    a = V_max / C_jam
    v = (
        C_jam
        / rho_norm
        * (
            1
            + (a - 1) * rho_norm
            - ((a * rho_norm) ** theta + (1 - rho_norm) ** theta) ** (1 / theta)
        )
    )
    return v


def del_castillo_flux(
    rho: C.Cochain, C_jam: float, V_max: float, rho_max: float, theta: float
):
    v = del_castillo_v(rho.coeffs, C_jam, V_max, rho_max, theta)
    return C.Cochain(rho.dim, rho.is_primal, rho.complex, rho.coeffs * v)


def short_convolution_features(rho: C.CochainP0):
    """Return the frozen Automodel short-range DEC features.

    ``contrast`` cancels the density level on homogeneous interior nodes and
    responds to short spatial variations. The downstream boundary behavior is
    inherited from DCTKit's valid convolution padding.
    """

    ones = C.CochainP0(rho.complex, jnp.ones_like(rho.coeffs))
    conv_3 = C.convolution(rho, ones, kernel_window=3)
    conv_1 = C.convolution(rho, ones, kernel_window=1)
    contrast = C.CochainP0(
        rho.complex, conv_3.coeffs - 3.0 * conv_1.coeffs
    )
    return conv_3, conv_1, contrast


def quadratic_hodge_feature(rho: C.CochainP0) -> C.CochainP0:
    """Evaluate ``St_oneD1(SquareD1(St_oneP0(rho)))`` directly."""

    dual = C.star(rho)
    return C.star(C.cochain_mul(dual, dual))


def greenshields_correction_multiplier(
    rho: C.CochainP0, c0: float, a: float, b: float, d: float
) -> C.CochainP0:
    """Frozen Greenshields correction structure selected by Automodel."""

    _, _, contrast = short_convolution_features(rho)
    exponent = c0 + a * rho.coeffs + b * rho.coeffs**2 + d * contrast.coeffs
    return C.CochainP0(rho.complex, jnp.exp(exponent))


def idm_correction_multiplier(
    rho: C.CochainP0, a: float, b: float
) -> C.CochainP0:
    """Frozen IDM convolution/Hodge correction selected by Automodel."""

    _, _, contrast = short_convolution_features(rho)
    gated = quadratic_hodge_feature(rho).coeffs * contrast.coeffs
    return C.CochainP0(
        rho.complex, jnp.exp(a * contrast.coeffs + b * gated)
    )


def weidmann_correction_multiplier(
    rho: C.CochainP0, a: float, rho_max: float
) -> C.CochainP0:
    """Frozen jam-anchored Weidmann correction selected by Automodel."""

    exponent = a * rho.coeffs * (1.0 - rho.coeffs / rho_max)
    return C.CochainP0(rho.complex, jnp.exp(exponent))


def triangular_correction_multiplier(
    rho: C.CochainP0, a: float
) -> C.CochainP0:
    """Frozen density-gated convolution correction selected by Automodel."""

    _, _, contrast = short_convolution_features(rho)
    return C.CochainP0(
        rho.complex, jnp.exp(a * rho.coeffs * contrast.coeffs)
    )


def del_castillo_correction_multiplier(
    rho: C.CochainP0, c0: float, a: float, b: float
) -> C.CochainP0:
    """Frozen Del Castillo correction structure selected by Automodel."""

    conv_3, _, contrast = short_convolution_features(rho)
    exponent = c0 + a * conv_3.coeffs + b * contrast.coeffs
    return C.CochainP0(rho.complex, jnp.exp(exponent))


def _corrected_flux(
    rho: C.CochainP0,
    baseline_flux: Callable,
    baseline_coefficients: Sequence[float],
    multiplier: C.CochainP0,
) -> C.CochainP0:
    return C.cochain_mul(
        baseline_flux(rho, *baseline_coefficients), multiplier
    )


def Greenshields_corrected_flux(
    rho: C.CochainP0,
    v_max: float,
    rho_max: float,
    c0: float = 0.17774550192487548,
    a: float = -0.30750256510961904,
    b: float = -0.09003085592750004,
    d: float = -224.63557603202457,
) -> C.CochainP0:
    return _corrected_flux(
        rho,
        Greenshields_flux,
        (v_max, rho_max),
        greenshields_correction_multiplier(rho, c0, a, b, d),
    )


def IDM_corrected_flux(
    rho: C.CochainP0,
    s0: float,
    T: float,
    delta: float,
    v0: float,
    a: float = -46.99143597506256,
    b: float = -119537.38834933029,
) -> C.CochainP0:
    return _corrected_flux(
        rho,
        IDM_flux,
        (s0, T, delta, v0),
        idm_correction_multiplier(rho, a, b),
    )


def Weidmann_corrected_flux(
    rho: C.CochainP0,
    v_max: float,
    rho_max: float,
    lambda_w: float,
    a: float = 0.20910818226914563,
) -> C.CochainP0:
    return _corrected_flux(
        rho,
        Weidmann_flux,
        (v_max, rho_max, lambda_w),
        weidmann_correction_multiplier(rho, a, rho_max),
    )


def triangular_corrected_flux(
    rho: C.CochainP0,
    V_0: float,
    l_eff: float,
    T: float,
    a: float = -921.972502729021,
) -> C.CochainP0:
    return _corrected_flux(
        rho,
        triangular_flux,
        (V_0, l_eff, T),
        triangular_correction_multiplier(rho, a),
    )


def del_castillo_corrected_flux(
    rho: C.CochainP0,
    C_jam: float,
    V_max: float,
    rho_max: float,
    theta: float,
    c0: float = 0.07388543849777629,
    a: float = -3.328070298172179,
    b: float = -227.98310112971478,
) -> C.CochainP0:
    return _corrected_flux(
        rho,
        del_castillo_flux,
        (C_jam, V_max, rho_max, theta),
        del_castillo_correction_multiplier(rho, c0, a, b),
    )


def define_flux_der(S: SimplicialComplex, flux: Callable):
    """Build a Rusanov wave-speed bound for local or nonlocal fluxes.

    A pointwise fundamental diagram has a diagonal flux Jacobian, in which case
    the row-wise absolute sum is exactly ``abs(dq/drho)``. DEC convolutions and
    other spatial corrections introduce off-diagonal entries; summing their
    absolute values gives a conservative local Lipschitz bound instead of
    silently discarding the coupling.
    """

    def flux_wrap(rho_coeffs, *args):
        rho = C.CochainP0(S, rho_coeffs)
        return flux(rho, *args).coeffs.flatten()

    der = jacfwd(flux_wrap)

    def der_auto(rho, *args):
        jacobian = der(rho.coeffs.flatten(), *args)
        speed_bound = jnp.sum(jnp.abs(jacobian), axis=1)
        return C.CochainP0(rho.complex, speed_bound)

    return der_auto
