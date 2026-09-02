"""Checks for the five frozen Automodel correction structures."""

import jax.numpy as jnp
from dctkit.dec import cochain as C

from automodel.final_models import FINAL_MODELS
from sr_traffic.data.data import preprocess_data
from sr_traffic.fd import diagrams


CORRECTED_FLUXES = {
    "greenshields": diagrams.Greenshields_corrected_flux,
    "idm": diagrams.IDM_corrected_flux,
    "weidmann": diagrams.Weidmann_corrected_flux,
    "triangular": diagrams.triangular_corrected_flux,
    "del_castillo": diagrams.del_castillo_corrected_flux,
}

PHASE4_PARAMETERS = {
    "greenshields": (
        0.17774550192487548,
        -0.30750256510961904,
        -0.09003085592750004,
        -224.63557603202457,
    ),
    "idm": (-46.99143597506256, -119537.38834933029),
    "weidmann": (0.20910818226914563,),
    "triangular": (-921.972502729021,),
    "del_castillo": (
        0.07388543849777629,
        -3.328070298172179,
        -227.98310112971478,
    ),
}


def test_frozen_multipliers_match_corrected_flux_wrappers():
    complex_ = preprocess_data("I80")["S"]
    rho = C.CochainP0(
        complex_, jnp.linspace(0.01, 0.5, complex_.num_nodes)
    )

    for key, model in FINAL_MODELS.items():
        multiplier = model.correction(rho, model.initial_parameters)
        baseline = model.baseline.flux(rho, *model.baseline.coefficients)
        expected = C.cochain_mul(baseline, multiplier).coeffs
        measured = CORRECTED_FLUXES[key](
            rho, *model.baseline.coefficients, *model.initial_parameters
        ).coeffs

        assert multiplier.coeffs.shape == rho.coeffs.shape
        assert bool(jnp.all(jnp.isfinite(multiplier.coeffs)))
        assert bool(jnp.all(multiplier.coeffs > 0))
        assert bool(jnp.allclose(measured, expected, rtol=1e-6, atol=1e-7))


def test_corrected_flux_defaults_are_phase4_refits():
    complex_ = preprocess_data("I80")["S"]
    rho = C.CochainP0(
        complex_, jnp.linspace(0.01, 0.5, complex_.num_nodes)
    )

    for key, model in FINAL_MODELS.items():
        corrected_flux = CORRECTED_FLUXES[key]
        default = corrected_flux(rho, *model.baseline.coefficients).coeffs
        explicit = corrected_flux(
            rho, *model.baseline.coefficients, *PHASE4_PARAMETERS[key]
        ).coeffs
        assert bool(jnp.allclose(default, explicit, rtol=1e-7, atol=1e-8))
