"""Regression tests for backend-independent robust-satisficing regret."""

import math

import numpy as np
import pytest
import torch

from basat.regret import RSRegret, finite_domain_optimal_fragility, rs_regret_statistics


def test_paper_equation_and_cumulative_default():
    metric = RSRegret(aspiration=-10.0, optimal_fragility=2.0, epsilon=0.5, p=1.0)
    rewards = np.array([-12.0, -10.0, -13.0])

    np.testing.assert_array_equal(metric.instantaneous(rewards), [1.0, 0.0, 2.0])
    np.testing.assert_array_equal(metric.evaluate(rewards), [1.0, 1.0, 3.0])
    np.testing.assert_array_equal(metric.evaluate(rewards, cumulative=False), [1.0, 0.0, 2.0])


def test_per_round_epsilon_and_torch_backend():
    metric = RSRegret(
        aspiration=10.0,
        optimal_fragility=2.0,
        epsilon=(0.0, 0.5, 1.0),
        p=2.0,
        cumulative=False,
    )
    rewards = torch.tensor([[9.0, 8.0, 3.0], [10.0, 9.0, 5.0]], dtype=torch.float32)
    actual = metric.evaluate(rewards)

    assert isinstance(actual, torch.Tensor)
    assert actual.dtype == torch.float32
    assert actual.device == rewards.device
    torch.testing.assert_close(
        actual,
        torch.tensor([[1.0, 1.0, 3.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="match the last dimension"):
        metric.evaluate(torch.ones(2, 4))


def test_finite_domain_true_fragility_and_zero_distance_guard():
    result = finite_domain_optimal_fragility(
        actions=[[0.0], [1.0], [2.0]],
        objective_values=[-1.0, 1.0, -3.0],
        aspiration=0.0,
        p=1.0,
    )
    assert result.per_center == (math.inf, 3.0, math.inf)
    assert result.optimal_fragility == 3.0
    assert result.optimal_indices == (1,)
    assert result.safe_indices == (1,)

    duplicate = finite_domain_optimal_fragility(
        actions=torch.tensor([[0.0], [0.0]], dtype=torch.float64),
        objective_values=torch.tensor([1.0, -1.0], dtype=torch.float64),
        aspiration=0.0,
    )
    assert duplicate.per_center == (math.inf, math.inf)
    assert duplicate.optimal_fragility == math.inf
    assert duplicate.optimal_indices == ()


def test_population_sd_over_two_and_plot_series_conversion():
    summary = rs_regret_statistics([[1.0, 2.0], [3.0, 4.0]], error_scale=0.5)
    assert summary.mean == (2.0, 3.0)
    assert summary.std == (1.0, 1.0)
    assert summary.error == (0.5, 0.5)
    assert summary.run_count == 2
    assert summary.round_count == 2

    series = summary.to_plot_series(color="red", linestyle="--")
    np.testing.assert_array_equal(series.values, [2.0, 3.0])
    np.testing.assert_array_equal(series.error, [0.5, 0.5])
    np.testing.assert_array_equal(series.x, [1, 2])
    assert series.color == "red"


@pytest.mark.parametrize("p", [float("inf"), -float("inf")])
def test_infinite_p_is_explicitly_rejected(p):
    with pytest.raises(ValueError, match="RS-2"):
        RSRegret(aspiration=0.0, optimal_fragility=1.0, epsilon=1.0, p=p)
    with pytest.raises(ValueError, match="RS-2"):
        finite_domain_optimal_fragility([[0.0]], [0.0], aspiration=0.0, p=p)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"aspiration": float("nan"), "optimal_fragility": 1.0, "epsilon": 1.0},
        {"aspiration": 0.0, "optimal_fragility": -1.0, "epsilon": 1.0},
        {"aspiration": 0.0, "optimal_fragility": 1.0, "epsilon": -1.0},
        {"aspiration": 0.0, "optimal_fragility": 1.0, "epsilon": []},
        {"aspiration": 0.0, "optimal_fragility": 1.0, "epsilon": 1.0, "p": 0.5},
    ],
)
def test_configuration_validation(kwargs):
    with pytest.raises((TypeError, ValueError)):
        RSRegret(**kwargs)


def test_statistics_reject_bad_shape_and_values():
    with pytest.raises(ValueError, match="runs-by-rounds"):
        rs_regret_statistics([1.0, 2.0])
    with pytest.raises(ValueError, match="nonnegative"):
        rs_regret_statistics([[0.0, -1.0]])
    with pytest.raises(ValueError, match="nonnegative"):
        rs_regret_statistics([[0.0]], error_scale=-0.5)
    with pytest.raises(TypeError, match="real-valued"):
        rs_regret_statistics([[True, False]])
