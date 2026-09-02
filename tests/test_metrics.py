"""Tests for shared traffic evaluation metrics."""

import numpy as np

from sr_traffic.utils.metrics import relative_tts_error


def test_relative_tts_error_integrates_density_over_space_and_time():
    time = np.array([0.0, 2.0])
    position = np.array([0.0, 1.0])
    true_density = 2.0 * np.ones((2, 2))
    model_density = 3.0 * np.ones((2, 2))

    assert relative_tts_error(true_density, model_density, time, position) == 0.5
