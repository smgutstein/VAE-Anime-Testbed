import numpy as np
import pytest

from VAE_Pareto_FID_KID_Plots import (
    chronological_colors,
    effective_kl_dimensions,
    minimization_frontier,
)


def test_effective_kl_dimensions_equal_distribution():
    assert effective_kl_dimensions([2.0, 2.0, 2.0, 2.0]) == pytest.approx(4.0)


def test_effective_kl_dimensions_concentrated_distribution():
    assert effective_kl_dimensions([4.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_effective_kl_dimensions_zero_distribution():
    assert effective_kl_dimensions([0.0, 0.0]) == 0.0


def test_minimization_frontier_returns_nondominated_points():
    x_values = np.array([3.0, 2.0, 1.0, 2.5])
    y_values = np.array([1.0, 3.0, 4.0, 2.0])
    assert minimization_frontier(x_values, y_values) == [2, 1, 3, 0]


def test_minimization_frontier_discards_dominated_point():
    x_values = np.array([1.0, 2.0, 3.0])
    y_values = np.array([1.0, 2.0, 0.5])
    assert minimization_frontier(x_values, y_values) == [0, 2]


def test_chronological_colors_run_from_light_to_full_color():
    colors = chronological_colors("#204080", 3, minimum_intensity=0.25)
    assert np.all(colors[0] > colors[-1])
    assert colors[-1] == pytest.approx([32 / 255, 64 / 255, 128 / 255])
