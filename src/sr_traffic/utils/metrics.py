"""Evaluation metrics shared by traffic-model workflows."""

import numpy as np
import numpy.typing as npt


def relative_tts_error(
    true_density: npt.NDArray,
    model_density: npt.NDArray,
    time: npt.NDArray,
    position: npt.NDArray,
) -> float:
    """Return relative error in the space-time integral of density."""

    tts_true = np.trapezoid(
        np.trapezoid(true_density, time, axis=1), position, axis=0
    )
    tts_model = np.trapezoid(
        np.trapezoid(model_density, time, axis=1), position, axis=0
    )
    return float(np.abs((tts_model - tts_true) / tts_true))
