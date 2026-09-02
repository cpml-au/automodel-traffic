"""Meta-3 incumbent times a quadratic-Hodge gated convolution contrast."""

from __future__ import annotations

from collections.abc import Sequence

from dctkit.dec import cochain as C

from meta_4.agent_2 import common


ATTEMPT = 2
PARAMETER_NAMES = ("a",)
BOUNDS = ((-1000000.0, 1000000.0),)
SYMBOLIC_FACTOR = "exp(a*hquad*(conv_3-3*conv_1))"
TYPED_FACTOR = (
    "ExpP0(CMulP0(MFP0(St_oneD1(SquareD1(St_oneP0(rho))),a),"
    "SubCP0(conv_3P0(rho,ones),MFP0(conv_1P0(rho,ones),three))))"
)
# Factor has 17 nodes; one more CMulP0 attaches it to the fixed incumbent.
ADDED_NODES = 18
HODGE_PRIMITIVES = ("St_oneP0", "St_oneD1")


def make_correction(baseline_key: str):
    def correction(rho: C.CochainP0, parameters: Sequence[float]):
        (a,) = parameters
        _, _, contrast = common.convolution_features(rho)
        gated = C.cochain_mul(common.hquad(rho), contrast)
        exponent = C.cochain_mul(gated, common.scalar_cochain(rho, a))
        factor = common.exp_cochain(rho, exponent)
        return C.cochain_mul(common.incumbent(baseline_key, rho), factor)

    return correction


def expression(baseline_key: str) -> str:
    return f"({common.INCUMBENT_EXPRESSIONS[baseline_key]})*{SYMBOLIC_FACTOR}"
