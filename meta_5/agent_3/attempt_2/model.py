"""Linear- and quadratic-Hodge gated convolution contrast refinement."""

from __future__ import annotations

from collections.abc import Sequence

from dctkit.dec import cochain as C

from meta_5.agent_3 import common


ATTEMPT = 2
PARAMETER_NAMES = ("a", "b")
BOUNDS = ((-2000.0, 2000.0), (-1000000.0, 1000000.0))
PARAMETER_SCALES = (500.0, 100000.0)
SYMBOLIC_FACTOR = "exp(a*hlin*C+b*hquad*C)"
TYPED_FACTOR = (
    "ExpP0(AddCP0("
    "MFP0(CMulP0(St_oneD1(St_oneP0(rho)),"
    "SubCP0(conv_3P0(rho,ones),MFP0(conv_1P0(rho,ones),three))),a),"
    "MFP0(CMulP0(St_oneD1(SquareD1(St_oneP0(rho))),"
    "SubCP0(conv_3P0(rho,ones),MFP0(conv_1P0(rho,ones),three))),b)))"
)
# Typed factor 33 nodes plus the CMulP0 attaching it to the incumbent.
ADDED_NODES = 34
HODGE_PRIMITIVES = ("St_oneP0", "St_oneD1", "SquareD1")


def make_correction(baseline_key: str):
    def correction(rho: C.CochainP0, parameters: Sequence[float]):
        a, b = parameters
        _, _, contrast = common.convolution_features(rho)
        linear_gate = common.multiply(common.hlin(rho), contrast)
        quadratic_gate = common.multiply(common.hquad(rho), contrast)
        first = C.cochain_mul(linear_gate, common.scalar(rho, a))
        second = C.cochain_mul(quadratic_gate, common.scalar(rho, b))
        exponent = C.CochainP0(rho.complex, first.coeffs + second.coeffs)
        return common.attach_factor(baseline_key, rho, exponent)

    return correction


def expression(baseline_key: str) -> str:
    return f"({common.INCUMBENT_EXPRESSIONS[baseline_key]})*{SYMBOLIC_FACTOR}"
