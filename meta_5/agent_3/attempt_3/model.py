"""Homogeneous-scale-centered quadratic-Hodge contrast refinement."""

from __future__ import annotations

from collections.abc import Sequence

from dctkit.dec import cochain as C

from meta_5.agent_3 import common


ATTEMPT = 3
PARAMETER_NAMES = ("a",)
BOUNDS = ((-1000000.0, 1000000.0),)
PARAMETER_SCALES = (100000.0,)
SYMBOLIC_FACTOR = "exp(a*(79*hquad-rho^2)*C)"
TYPED_FACTOR = (
    "ExpP0(MFP0(CMulP0("
    "SubCP0(MFP0(St_oneD1(SquareD1(St_oneP0(rho))),seventy_nine),"
    "SquareP0(rho)),"
    "SubCP0(conv_3P0(rho,ones),MFP0(conv_1P0(rho,ones),three))),a))"
)
# Typed factor 22 nodes plus the CMulP0 attaching it to the incumbent.
ADDED_NODES = 23
HODGE_PRIMITIVES = ("St_oneP0", "St_oneD1", "SquareD1", "SquareP0")


def make_correction(baseline_key: str):
    def correction(rho: C.CochainP0, parameters: Sequence[float]):
        (a,) = parameters
        _, _, contrast = common.convolution_features(rho)
        quadratic = common.hquad(rho)
        centered = C.CochainP0(
            rho.complex, 79.0 * quadratic.coeffs - rho.coeffs**2
        )
        gate = common.multiply(centered, contrast)
        exponent = C.cochain_mul(gate, common.scalar(rho, a))
        return common.attach_factor(baseline_key, rho, exponent)

    return correction


def expression(baseline_key: str) -> str:
    return f"({common.INCUMBENT_EXPRESSIONS[baseline_key]})*{SYMBOLIC_FACTOR}"
