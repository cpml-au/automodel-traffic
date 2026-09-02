"""Incumbent times a squared-density-gated convolution contrast."""

from __future__ import annotations

from collections.abc import Sequence

from dctkit.dec import cochain as C

from meta_5.agent_2 import common


ATTEMPT = 2
PARAMETER_NAMES = ("a",)
BOUNDS = ((-15000.0, 15000.0),)
SYMBOLIC_FACTOR = "exp(a*rho^2*C)"
TYPED_FACTOR = (
    "ExpP0(MFP0(CMulP0(SquareP0(rho),"
    "SubCP0(conv_3P0(rho,ones),MFP0(conv_1P0(rho,ones),three))),a))"
)
# 15 typed factor nodes plus the CMulP0 attachment to the incumbent.
ADDED_NODES = 16


def make_correction(baseline_key: str):
    def correction(rho: C.CochainP0, parameters: Sequence[float]):
        (a,) = parameters
        gated = common.density_gated_contrast(rho, 2)
        exponent = C.cochain_mul(gated, common.scalar_cochain(rho, a))
        return C.cochain_mul(
            common.incumbent(baseline_key, rho),
            common.exp_cochain(rho, exponent),
        )

    return correction


def expression(baseline_key: str) -> str:
    return f"({common.INCUMBENT_EXPRESSIONS[baseline_key]})*{SYMBOLIC_FACTOR}"
