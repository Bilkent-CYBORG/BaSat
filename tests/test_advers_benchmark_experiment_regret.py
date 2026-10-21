"""Integration checks for the AdveRS benchmark and Experiment regret API."""

from __future__ import annotations

import json

import pytest
import torch

import basat
from basat.algorithms.algorithm import AlgorithmExceptionController


class _ScriptedAlgorithm(basat.Algorithm):
    """Minimal deterministic ask/tell learner for orchestration tests."""

    native_environments = frozenset({"stochastic"})
    ACQUISITION_POLICY = "scripted test coordinates"
    EXCEPTION_CLASS = AlgorithmExceptionController

    def __init__(self, *, bounds: basat.Bounds, seed: int = 1) -> None:
        super().__init__(bounds=bounds, seed=seed, dtype="float64", device="cpu")

    def _ask_impl(self, *, round_info):
        del round_info
        coordinate = (-1.0, 0.0, 1.0)[self.iteration % 3]
        return torch.tensor([coordinate], dtype=self.dtype, device=self.device)

    def _tell_impl(self, *, action, feedback, round_info) -> None:
        del action, feedback, round_info


def _environment(*, seed: int = 11, previous_data=None, noise_std: float = 4.0):
    bounds = basat.Bounds(lower=[-1.0], upper=[1.0])
    return basat.StochasticEnvironment(
        bounds=bounds,
        function=lambda x: x.squeeze() - 1.0,
        previous_data=previous_data,
        noise_std=noise_std,
        allow_repeated_actions=True,
        dtype="float64",
        device="cpu",
        seed=seed,
    )


def _metric() -> basat.RSRegret:
    return basat.RSRegret(
        aspiration=0.0,
        optimal_fragility=0.0,
        epsilon=0.0,
    )


def test_released_polynomial_grid_and_formula_are_explicit():
    objective = basat.AdveRSSynthetic()
    grid = objective.grid()
    values = objective(grid)

    assert grid.shape == (1024, 2)
    assert grid[0].tolist() == pytest.approx([-0.95, -0.45])
    assert grid[1, 1].item() == pytest.approx(-0.45)
    assert grid[32, 0].item() == pytest.approx(-0.95)
    assert values.min().item() == pytest.approx(-67.86243932812533)
    assert values.max().item() == pytest.approx(14.081681249585023)
    assert int(values.argmax()) == 924

    paper = basat.AdveRSSynthetic(formula="stableopt-paper")
    point = torch.tensor([1.0, 2.0], dtype=torch.float64)
    assert objective(point).item() == pytest.approx(-14.1)
    assert paper(point).item() == pytest.approx(-13.9)


def test_finite_grid_fragility_matches_released_protocol():
    objective = basat.AdveRSSynthetic()
    grid = objective.grid()
    result = basat.finite_domain_optimal_fragility(
        grid, objective(grid), aspiration=-10.0, p=1.0
    )

    assert result.optimal_fragility == pytest.approx(20.129238666994205)
    assert result.optimal_indices == (300,)
    assert grid[300].tolist() == pytest.approx([0.6564516129032258, 0.9580645161290324])


def test_experiment_regret_uses_latent_online_rows_and_excludes_warm_start(tmp_path):
    previous = basat.EnvironmentData(
        actions=torch.tensor([[0.25]], dtype=torch.float64),
        observations=torch.tensor([99.0], dtype=torch.float64),
    )
    environment = _environment(previous_data=previous)
    experiment = basat.Experiment(
        environment=environment,
        algorithm=_ScriptedAlgorithm(bounds=environment.bounds),
        output_directory=tmp_path / "latent",
    )

    result = experiment.run(max_evaluations=4, regret=_metric())

    assert result.evaluations == 4
    assert len(environment.history.records) == 4
    assert result.metadata["regret"]["initial_history_included"] is False
    assert result.metadata["regret"]["audit_sources"] == ["latent_observation"]
    assert result.metadata["regret"]["instantaneous"] == pytest.approx([2.0, 1.0, 0.0, 2.0])
    assert result.metadata["regret"]["values"] == pytest.approx([2.0, 3.0, 3.0, 5.0])
    assert experiment.regret_curves["_ScriptedAlgorithm"].tolist() == pytest.approx(
        [2.0, 3.0, 3.0, 5.0]
    )
    document = json.loads((tmp_path / "latent" / "result.json").read_text())
    assert document["algorithms"]["_ScriptedAlgorithm"]["result"]["metadata"][
        "regret"
    ]["values"] == pytest.approx([2.0, 3.0, 3.0, 5.0])


def test_resume_recomputes_same_curve_without_extra_evaluations(tmp_path):
    directory = tmp_path / "resume"
    first_environment = _environment(noise_std=0.0)
    first = basat.Experiment(
        environment=first_environment,
        algorithm=_ScriptedAlgorithm(bounds=first_environment.bounds),
        output_directory=directory,
    )
    assert first.run(max_evaluations=2, regret=_metric()).metadata["regret"][
        "values"
    ] == pytest.approx([2.0, 3.0])

    resumed_environment = _environment(noise_std=0.0)
    resumed = basat.Experiment(
        environment=resumed_environment,
        algorithm=_ScriptedAlgorithm(bounds=resumed_environment.bounds),
        output_directory=directory,
        resume=True,
    )
    result = resumed.run(max_evaluations=3, regret=_metric())
    assert result.metadata["regret"]["values"] == pytest.approx([2.0, 3.0, 3.0])
    assert len(resumed_environment.history.records) == 3

    repeated = resumed.run(max_evaluations=3, regret=_metric())
    assert repeated.metadata["regret"]["values"] == pytest.approx([2.0, 3.0, 3.0])
    assert len(resumed_environment.history.records) == 3


def test_trial_and_multi_algorithm_regret_are_kept_separate():
    trial_environment = _environment(noise_std=0.0)
    trials = basat.Experiment(
        environment=trial_environment,
        algorithm=_ScriptedAlgorithm(bounds=trial_environment.bounds),
    ).run(trials=2, max_evaluations=2, regret=_metric())
    assert trials.regret_statistics["_ScriptedAlgorithm"].run_count == 2
    assert trials.regret_statistics["_ScriptedAlgorithm"].mean == pytest.approx((2.0, 3.0))

    environment = _environment(noise_std=0.0)
    comparison = basat.Experiment(
        environment=environment,
        algorithm={
            "first": _ScriptedAlgorithm(bounds=environment.bounds, seed=1),
            "second": _ScriptedAlgorithm(bounds=environment.bounds, seed=2),
        },
    )
    result = comparison.run(
        max_evaluations=2,
        regret={"first": _metric(), "second": _metric()},
        plot=False,
    )
    assert set(result.results) == {"first", "second"}
    assert set(comparison.regret_curves) == {"first", "second"}
    assert comparison.regret_curves["first"].tolist() == pytest.approx([2.0, 3.0])


def test_regret_rejects_ambiguous_string_and_incomplete_mapping():
    environment = _environment(noise_std=0.0)
    experiment = basat.Experiment(
        environment=environment,
        algorithm=_ScriptedAlgorithm(bounds=environment.bounds),
    )
    with pytest.raises(basat.ExperimentConfigurationError, match="RSRegret"):
        experiment.run(max_evaluations=1, regret="RS")

    multi_environment = _environment(noise_std=0.0)
    multi = basat.Experiment(
        environment=multi_environment,
        algorithm={
            "first": _ScriptedAlgorithm(bounds=multi_environment.bounds, seed=1),
            "second": _ScriptedAlgorithm(bounds=multi_environment.bounds, seed=2),
        },
    )
    with pytest.raises(basat.ExperimentConfigurationError, match="exactly"):
        multi.run(max_evaluations=1, regret={"first": _metric()})


def test_plot_panel_exposes_source_legend_title_and_location():
    chart = basat.plot_panels(
        [
            basat.PlotPanel(
                {"method": basat.PlotSeries([1.0, 2.0])},
                legend_title="tau = -10, epsilon = 0.5",
                legend_location="upper left",
            )
        ],
        show=False,
    )
    legend = chart.axes[0, 0].get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == "tau = -10, epsilon = 0.5"
    assert legend._loc == 2
