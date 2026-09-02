"""Meta-3 incumbent times contrast plus quadratic-Hodge exponential terms."""

from __future__ import annotations

from collections.abc import Sequence

from dctkit.dec import cochain as C

from meta_4.agent_2 import common


ATTEMPT = 3
PARAMETER_NAMES = ("a", "b")
BOUNDS = ((-300.0, 300.0), (-2500.0, 2500.0))
SYMBOLIC_FACTOR = "exp(a*(conv_3-3*conv_1)+b*hquad)"
TYPED_FACTOR = (
    "ExpP0(AddCP0("
    "MFP0(SubCP0(conv_3P0(rho,ones),MFP0(conv_1P0(rho,ones),three)),a),"
    "MFP0(St_oneD1(SquareD1(St_oneP0(rho))),b)))"
)
# Factor has 19 nodes; one more CMulP0 attaches it to the fixed incumbent.
ADDED_NODES = 20
HODGE_PRIMITIVES = ("St_oneP0", "St_oneD1")


def make_correction(baseline_key: str):
    def correction(rho: C.CochainP0, parameters: Sequence[float]):
        a, b = parameters
        _, _, contrast = common.convolution_features(rho)
        contrast_term = C.cochain_mul(
            contrast, common.scalar_cochain(rho, a)
        )
        hquad_term = C.cochain_mul(
            common.hquad(rho), common.scalar_cochain(rho, b)
        )
        exponent = C.CochainP0(
            rho.complex, contrast_term.coeffs + hquad_term.coeffs
        )
        factor = common.exp_cochain(rho, exponent)
        return C.cochain_mul(common.incumbent(baseline_key, rho), factor)

    return correction


def expression(baseline_key: str) -> str:
    return f"({common.INCUMBENT_EXPRESSIONS[baseline_key]})*{SYMBOLIC_FACTOR}"
