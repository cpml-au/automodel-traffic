"""Frozen Phase-4 model structures selected without test-set access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from dctkit.dec import cochain as C

from automodel.model import BASELINES, Baseline
from sr_traffic.fd import diagrams


@dataclass(frozen=True)
class FinalModel:
    key: str
    baseline: Baseline
    expression: str
    parameter_names: tuple[str, ...]
    initial_parameters: tuple[float, ...]
    bounds: tuple[tuple[float, float], ...]
    tree_nodes: int
    source: str
    nonlocal_correction: bool
    correction: Callable


def _greenshields(rho: C.CochainP0, parameters: Sequence[float]):
    return diagrams.greenshields_correction_multiplier(rho, *parameters)


def _idm(rho: C.CochainP0, parameters: Sequence[float]):
    return diagrams.idm_correction_multiplier(rho, *parameters)


def _weidmann(rho: C.CochainP0, parameters: Sequence[float]):
    (a,) = parameters
    rho_max = BASELINES["weidmann"].coefficients[1]
    return diagrams.weidmann_correction_multiplier(rho, a, rho_max)


def _triangular(rho: C.CochainP0, parameters: Sequence[float]):
    return diagrams.triangular_correction_multiplier(rho, *parameters)


def _del_castillo(rho: C.CochainP0, parameters: Sequence[float]):
    return diagrams.del_castillo_correction_multiplier(rho, *parameters)


FINAL_MODELS = {
    "greenshields": FinalModel(
        key="greenshields",
        baseline=BASELINES["greenshields"],
        expression=(
            "exp(c0+a*rho+b*rho^2+d*(conv_3(rho,ones)-3*conv_1(rho,ones)))"
        ),
        parameter_names=("c0", "rho", "rho_squared", "contrast"),
        initial_parameters=(
            0.02312297116054006,
            -0.08831739725970279,
            -0.1500812210203765,
            -293.4581091232274,
        ),
        bounds=((-1.0, 1.0), (-5.0, 5.0), (-5.0, 5.0), (-1000.0, 1000.0)),
        tree_nodes=25,
        source="meta_3/agent_3/attempt_2/model.py",
        nonlocal_correction=True,
        correction=_greenshields,
    ),
    "idm": FinalModel(
        key="idm",
        baseline=BASELINES["idm"],
        expression=(
            "exp(a*C+b*St_oneD1(SquareD1(St_oneP0(rho)))*C), "
            "C=conv_3(rho,ones)-3*conv_1(rho,ones)"
        ),
        parameter_names=("contrast", "quadratic_hodge_contrast"),
        initial_parameters=(-46.991435975062515, -119537.38834933033),
        bounds=((-300.0, 300.0), (-1000000.0, 1000000.0)),
        tree_nodes=32,
        source="meta_4/agent_2/attempt_2/model.py",
        nonlocal_correction=True,
        correction=_idm,
    ),
    "weidmann": FinalModel(
        key="weidmann",
        baseline=BASELINES["weidmann"],
        expression="exp(a*rho*(1-rho/rho_max))",
        parameter_names=("jam_anchored",),
        initial_parameters=(0.1365440521436772,),
        bounds=((-2.0, 2.0),),
        tree_nodes=10,
        source="meta_1/agent_2/attempt_1/model.py",
        nonlocal_correction=False,
        correction=_weidmann,
    ),
    "triangular": FinalModel(
        key="triangular",
        baseline=BASELINES["triangular"],
        expression=(
            "exp(a*rho*(conv_3(rho,ones)-3*conv_1(rho,ones)))"
        ),
        parameter_names=("density_contrast",),
        initial_parameters=(-921.9701997917449,),
        bounds=((-3000.0, 3000.0),),
        tree_nodes=16,
        source="meta_5/agent_2/attempt_1/model.py",
        nonlocal_correction=True,
        correction=_triangular,
    ),
    "del_castillo": FinalModel(
        key="del_castillo",
        baseline=BASELINES["del_castillo"],
        expression=(
            "exp(c0+a*conv_3(rho,ones)+b*(conv_3(rho,ones)-3*conv_1(rho,ones)))"
        ),
        parameter_names=("c0", "conv_3", "contrast"),
        initial_parameters=(
            0.0054952026563039776,
            -2.1218116018455078,
            -256.86922950791177,
        ),
        bounds=((-1.0, 1.0), (-20.0, 20.0), (-1000.0, 1000.0)),
        tree_nodes=21,
        source="meta_3/agent_3/attempt_3/model.py",
        nonlocal_correction=True,
        correction=_del_castillo,
    ),
}
