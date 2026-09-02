"""Ordinary and quadratic-Hodge gated convolution contrast refinement."""

from __future__ import annotations

from collections.abc import Sequence

from dctkit.dec import cochain as C

from meta_5.agent_3 import common


ATTEMPT = 1
PARAMETER_NAMES = ("a", "b")
BOUNDS = ((-500.0, 500.0), (-1000000.0, 1000000.0))
PARAMETER_SCALES = (100.0, 100000.0)
SYMBOLIC_FACTOR = "exp(a*C+b*hquad*C)"
TYPED_FACTOR = (
    "ExpP0(AddCP0("
    "MFP0(SubCP0(conv_3P0(rho,ones),MFP0(conv_1P0(rho,ones),three)),a),"
    "MFP0(CMulP0(St_oneD1(SquareD1(St_oneP0(rho))),"
    "SubCP0(conv_3P0(rho,ones),MFP0(conv_1P0(rho,ones),three))),b)))"
)
# Typed factor 29 nodes plus the CMulP0 attaching it to the incumbent.
ADDED_NODES = 30
HODGE_PRIMITIVES = ("St_oneP0", "St_oneD1", "SquareD1")


def make_correction(baseline_key: str):
    def correction(rho: C.CochainP0, parameters: Sequence[float]):
        a, b = parameters
        _, _, contrast = common.convolution_features(rho)
        ordinary = C.cochain_mul(contrast, common.scalar(rho, a))
        quadratic_gate = common.multiply(common.hquad(rho), contrast)
        gated = C.cochain_mul(quadratic_gate, common.scalar(rho, b))
        exponent = C.CochainP0(rho.complex, ordinary.coeffs + gated.coeffs)
        return common.attach_factor(baseline_key, rho, exponent)

    return correction


def expression(baseline_key: str) -> str:
    return f"({common.INCUMBENT_EXPRESSIONS[baseline_key]})*{SYMBOLIC_FACTOR}"
