"""Incumbent times linear- and squared-density contrast gates."""

from __future__ import annotations

from collections.abc import Sequence

from dctkit.dec import cochain as C

from meta_5.agent_2 import common


ATTEMPT = 3
PARAMETER_NAMES = ("a", "b")
BOUNDS = ((-3000.0, 3000.0), (-15000.0, 15000.0))
SYMBOLIC_FACTOR = "exp(a*rho*C+b*rho^2*C)"
TYPED_FACTOR = (
    "ExpP0(AddCP0("
    "MFP0(CMulP0(rho,SubCP0(conv_3P0(rho,ones),"
    "MFP0(conv_1P0(rho,ones),three))),a),"
    "MFP0(CMulP0(SquareP0(rho),SubCP0(conv_3P0(rho,ones),"
    "MFP0(conv_1P0(rho,ones),three))),b)))"
)
# 29 factor nodes, counting the repeated contrast twice, plus attachment.
ADDED_NODES = 30


def make_correction(baseline_key: str):
    def correction(rho: C.CochainP0, parameters: Sequence[float]):
        a, b = parameters
        linear = common.density_gated_contrast(rho, 1)
        quadratic = common.density_gated_contrast(rho, 2)
        linear_term = C.cochain_mul(
            linear, common.scalar_cochain(rho, a)
        )
        quadratic_term = C.cochain_mul(
            quadratic, common.scalar_cochain(rho, b)
        )
        exponent = C.CochainP0(
            rho.complex, linear_term.coeffs + quadratic_term.coeffs
        )
        return C.cochain_mul(
            common.incumbent(baseline_key, rho),
            common.exp_cochain(rho, exponent),
        )

    return correction


def expression(baseline_key: str) -> str:
    return f"({common.INCUMBENT_EXPRESSIONS[baseline_key]})*{SYMBOLIC_FACTOR}"
