"""Exact seed schedules preserve replay and ordinary trial compatibility."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import basat
from basat.algorithms.algorithm import AlgorithmExceptionController


class _SeededAlgorithm(basat.Algorithm):
    native_environments = frozenset({"stochastic"})
    ACQUISITION_POLICY = "uniform test proposal"
    EXCEPTION_CLASS = AlgorithmExceptionController

    def __init__(self, *, bounds: basat.Bounds, seed: int = 10) -> None:
        super().__init__(bounds=bounds, seed=seed, dtype="float64", device="cpu")

    def _ask_impl(self, *, round_info):
        del round_info
        return torch.rand((1,), generator=self.generator, dtype=self.dtype)

    def _tell_impl(self, *, action, feedback, round_info):
        del action, feedback, round_info


def _reward(x):
    return x.sum()


def _experiment(directory=None, *, resume=False, multi=False):
    bounds = basat.Bounds(lower=[0.0], upper=[1.0])
    environment = basat.StochasticEnvironment(
        bounds=bounds, function=_reward, noise_std=0.1, seed=50,
        allow_repeated_actions=True, dtype="float64", device="cpu",
    )
    first = _SeededAlgorithm(bounds=bounds, seed=10)
    algorithm = {"first": first, "second": _SeededAlgorithm(bounds=bounds, seed=20)} if multi else first
    return basat.Experiment(
        environment=environment, algorithm=algorithm,
        output_directory=directory, resume=resume,
    )


def _raw(experiment):
    return [
        {label: [record.observation.item() for record in env.history.records]
         for label, env in trial.source_environments.items()}
        for trial in experiment.trial_experiments
    ]


def test_sequence_seeds_initialize_algorithm_and_environment_and_manifest(tmp_path):
    experiment = _experiment(tmp_path / "scheduled")
    experiment.run(trials=2, trial_seeds=[2021, 42], max_rounds=2, plot=False)
    for child, seed in zip(experiment.trial_experiments, (2021, 42), strict=True):
        assert child.algorithm.seed == seed
        assert child.environment.seed == seed
    manifest = json.loads((tmp_path / "scheduled" / "trials.json").read_text())
    assert manifest["trial_seeds"] == {"_SeededAlgorithm": [2021, 42]}


def test_method_schedules_have_independent_exact_seeds():
    experiment = _experiment(multi=True)
    schedule = {"first": [2021, 2022], "second": [91, 92]}
    experiment.run(trials=2, trial_seeds=schedule, max_rounds=2, plot=False)
    for index, child in enumerate(experiment.trial_experiments):
        for label, algorithm in child.algorithms.items():
            assert algorithm.seed == schedule[label][index]
            assert child.source_environments[label].seed == schedule[label][index]
    assert _raw(experiment)[0]["first"] != _raw(experiment)[0]["second"]


def test_partial_resume_and_completed_noop_match_uninterrupted(tmp_path):
    schedule = [7, 8]
    expected = _experiment()
    expected.run(trials=2, trial_seeds=schedule, max_rounds=3, plot=False)

    partial = _experiment(tmp_path / "resume")
    partial.run(trials=2, trial_seeds=schedule, max_trials=1, max_rounds=1, plot=False)
    resumed = _experiment(tmp_path / "resume", resume=True)
    resumed.run(trials=2, trial_seeds=schedule, max_rounds=3, plot=False)
    assert _raw(resumed) == _raw(expected)
    resumed.run(trials=2, trial_seeds=schedule, max_rounds=3, plot=False)
    assert _raw(resumed) == _raw(expected)


def test_changed_or_removed_schedule_rejected_on_resume(tmp_path):
    first = _experiment(tmp_path / "immutable")
    first.run(trials=2, trial_seeds=[7, 8], max_rounds=1, plot=False)
    with pytest.raises(basat.ExperimentConfigurationError, match="cannot change"):
        first.run(trials=2, trial_seeds=[8, 7], max_rounds=1, plot=False)
    for schedule in ([8, 7], None):
        resumed = _experiment(tmp_path / "immutable", resume=True)
        with pytest.raises(basat.ExperimentConfigurationError, match="Saved trial settings differ"):
            resumed.run(trials=2, trial_seeds=schedule, max_rounds=1, plot=False)


@pytest.mark.parametrize("schedule", [
    [1], [1, 2, 3], [1, True], [1, -1], [1, 2**32], [1, 1.5], "12",
    {"wrong": [1, 2]}, {"_SeededAlgorithm": [1, 2], "extra": [3, 4]},
    np.array([[1, 2]]),
])
def test_invalid_schedule_rejected_before_trial_execution(schedule):
    experiment = _experiment()
    with pytest.raises(basat.ExperimentConfigurationError, match="trial_seeds"):
        experiment.run(trials=2, trial_seeds=schedule, max_rounds=1, plot=False)
    assert not experiment.environment.history.records
    assert experiment.algorithm.evaluations == 0


def test_ordinary_schedule_keeps_distinct_algorithm_and_environment_base_seeds(tmp_path):
    experiment = _experiment(tmp_path / "ordinary")
    experiment.run(trials=2, max_rounds=1, plot=False)
    assert [child.algorithm.seed for child in experiment.trial_experiments] == [10, 11]
    assert [child.environment.seed for child in experiment.trial_experiments] == [50, 51]
    manifest = json.loads((tmp_path / "ordinary" / "trials.json").read_text())
    assert "trial_seeds" not in manifest


def test_single_explicit_trial_and_numpy_integer_schedule():
    experiment = _experiment()
    experiment.run(trials=1, trial_seeds=np.array([23], dtype=np.int64), max_rounds=1, plot=False)
    assert experiment.trial_experiments[0].algorithm.seed == 23
    assert experiment.trial_experiments[0].environment.seed == 23
