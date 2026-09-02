"""Simple multiplicative-correction models used by Automodel.

The production fundamental diagrams remain in :mod:`sr_traffic.fd.diagrams`.
This module provides the intentionally small Phase 2 model and a uniform model
registry for evaluating every basic diagram without modifying global config.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import jax.numpy as jnp
from dctkit.dec import cochain as C

from sr_traffic.fd import diagrams


@dataclass(frozen=True)
class Baseline:
    """A calibrated basic fundamental diagram used as a fixed ansatz."""

    name: str
    flux: Callable
    velocity: Callable
    coefficients: tuple[float, ...]


BASELINES = {
    "greenshields": Baseline(
        "Greenshields",
        diagrams.Greenshields_flux,
        diagrams.Greenshields_v,
        (0.54673127, 0.55995123),
    ),
    "idm": Baseline(
        "IDM",
        diagrams.IDM_flux,
        diagrams.IDM_v,
        (0.43936351, 0.93094344, 0.16251414, 0.61353022),
    ),
    "weidmann": Baseline(
        "Weidmann",
        diagrams.Weidmann_flux,
        diagrams.Weidmann_v,
        (0.63190729, 0.80612097, 0.24947817),
    ),
    "triangular": Baseline(
        "Triangular",
        diagrams.triangular_flux,
        diagrams.triangular_v,
        (0.37013956, 1.48964708, 6.59672108),
    ),
    "del_castillo": Baseline(
        "Del Castillo",
        diagrams.del_castillo_flux,
        diagrams.del_castillo_v,
        (0.31807369, 0.46732741, 0.61532169, 2.60100492),
    ),
}


def constant_multiplier(
    rho: C.CochainP0, parameters: Sequence[float] = (1.0,)
) -> C.CochainP0:
    """Return the Phase 2 mock correction ``g(rho; c) = c``.

    Keeping the parameter in an explicit sequence matches the callable signature
    used by the SR constant tuner. The default ``c=1`` exactly recovers the basic
    diagram.
    """

    (scale,) = parameters
    return C.CochainP0(rho.complex, scale * jnp.ones_like(rho.coeffs))

