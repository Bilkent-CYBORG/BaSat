"""Portable numerical/branch fixtures for the Intel CGP-UCB reconstruction.

This is not trajectory parity with the unpublished Intel preprocessing. The
community CGP-UCB source 9a5e6d4 supplies the finite-UCB algebra only; its toy
cardinality bug and observation-noise predictive variance are not adopted.
"""

import math

import numpy as np
import pytest
import scipy.linalg as la
import torch

import basat
from basat.algorithms.algorithm import AlgorithmStateError
from basat.environments.environment import EnvironmentConfigurationError


def _matern52(points, lengthscale):
    distance = np.sqrt(
        np.sum(
            ((points[:, None, :] - points[None, :, :]) / np.asarray(lengthscale)) ** 2,
            axis=-1,
        )
    )
    root = np.sqrt(5.0) * distance
    return (1.0 + root + root**2 / 3.0) * np.exp(-root)


def _tables(rows=3):
    actions = np.array([[0.0, 0.0], [1.0, 0.1], [2.0, 0.7], [3.0, 0.3], [4.0, 0.9], [5.0, 0.2]])
    # The same time-of-day repeats on a later chronological row.
    context_support = np.array([[0.0], [0.5]])
    contexts = context_support[np.arange(rows) % 2]
    values = 19.0 + np.sin(np.arange(rows)[:, None] * 0.3 + np.arange(6)[None, :])
    return actions, context_support, contexts, values


def _environment(rows=3, *, values=None):
    actions, _, contexts, default_values = _tables(rows)
    return basat.SequentialContextualTableEnvironment(
        actions=actions,
        contexts=contexts,
        observations=default_values if values is None else values,
        queries_per_context=5,
        observation_offset=19.0,
        noise_variance=0.05,
        dtype="float64",
        seed=2011,
    )


def _algorithm(rows=3, *, delta=0.1):
    actions, support, _, _ = _tables(rows)
    return basat.CGP_UCB(
        bounds=_environment(rows).bounds,
        config=basat.CGPUCBConfig(
            action_set=actions,
            action_kernel=basat.FiniteSupportKernel(actions, 1.7 * _matern52(actions, [1.2, 0.4])),
            context_kernel=basat.FiniteSupportKernel(support, _matern52(support, [0.7])),
            beta="finite-theory",
            delta=delta,
            finite_domain_cardinality=46 * 288,
            initial_points=1,
            batch_size=1,
            seed=2011,
            gp_backend="fixed_incremental",
            acquisition_optimizer=basat.AcquisitionOptimizerSpec(fit_maxiter=0),
            gp_config=basat.GaussianProcessConfig(
                mean="zero",
                noise_mode="fixed",
                noise_variance=0.05,
                standardize_outcomes=False,
                dtype="float64",
            ),
        ),
        dtype="float64",
        verbose=False,
        plot=False,
    )


def _experiment(directory=None, *, rows=3, resume=False, values=None, delta=0.1):
    return basat.Experiment(
        environment=_environment(rows, values=values),
        algorithm=_algorithm(rows, delta=delta),
        output_directory=directory,
        resume=resume,
        checkpoint_policy="periodic",
        save_every=120,
    )


def _history(experiment):
    return np.array(
        [
            [
                record.audit["context_row"],
                record.audit["action_index"],
                float(record.audit["stored_observation"]),
                float(record.observation),
            ]
            for record in experiment.environment.history.records
        ]
    )


@pytest.fixture(autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("t", [1, 2, 5, 6, 4320])
def test_finite_beta_uses_joint_point_count_and_sensor_query_index(t):
    learner = _algorithm()
    learner.iteration = t - 1
    expected = 2 * np.log(46 * 288 * t**2 * np.pi**2 / 0.6)
    assert learner._beta() == pytest.approx(expected, rel=0, abs=1e-13)
    # The community implementation uses input_space.size (twice its row
    # cardinality for two coordinates). Our configured cardinality is points.
    erroneous_scalar_count = 2 * np.log(2 * 46 * 288 * t**2 * np.pi**2 / 0.6)
    assert erroneous_scalar_count - learner._beta() == pytest.approx(2 * np.log(2))


def test_dense_latent_posterior_ucb_and_every_sequential_decision():
    experiment = _experiment()
    learner, environment = experiment.algorithm, experiment.environment
    actions, support, _, _ = _tables()
    action_cov = 1.7 * _matern52(actions, [1.2, 0.4])
    context_cov = _matern52(support, [0.7])
    sensor_rows, context_rows, observed = [], [], []
    assert learner.state.y_train.numel() == 0  # Calibration is not a warm start.
    for n in range(15):
        info = environment.before_action()
        available = info["candidate_dataset_indices"].numpy()
        phase = int(np.flatnonzero((support == info["context"].numpy()).all(1))[0])
        expected_beta = 2 * np.log(46 * 288 * (n + 1) ** 2 * np.pi**2 / 0.6)
        assert learner._beta() == pytest.approx(expected_beta)
        if n:
            matrix = (
                action_cov[np.ix_(sensor_rows, sensor_rows)]
                * context_cov[np.ix_(context_rows, context_rows)]
            )
            matrix += 0.05 * np.eye(n)
            cross = (
                action_cov[np.ix_(sensor_rows, available)] * context_cov[context_rows, phase, None]
            )
            chol = la.cho_factor(matrix, lower=True)
            expected_mean = cross.T @ la.cho_solve(chol, observed)
            expected_var = np.diag(action_cov)[available] * context_cov[phase, phase] - np.sum(
                cross * la.cho_solve(chol, cross), axis=0
            )
            query = learner._joint_input(info["candidate_actions"], info["context"])
            mean, std = learner.gp.predict(query)
            np.testing.assert_allclose(mean, expected_mean, rtol=0, atol=2e-12)
            np.testing.assert_allclose(std.square(), expected_var, rtol=0, atol=2e-12)
            _, noisy_std = learner.gp.predict(query, observation_noise=True)
            np.testing.assert_allclose(noisy_std.square() - std.square(), 0.05, rtol=0, atol=2e-12)
            scores = expected_mean + np.sqrt(expected_beta) * np.sqrt(expected_var)
            np.testing.assert_allclose(learner._ucb(query), scores, rtol=0, atol=2e-11)
            tolerance = 1e-12 + 1e-9 * max(1.0, abs(scores.max()))
            expected_action = int(available[scores >= scores.max() - tolerance].min())
        else:
            expected_action = 0
        experiment.run_one_step()
        record = environment.history.records[-1]
        selected = int(record.audit["action_index"])
        assert selected == expected_action
        sensor_rows.append(selected)
        context_rows.append(phase)
        observed.append(float(record.observation))
        assert learner.iteration == n + 1
        assert learner.state.y_train.numel() == n + 1
        assert learner.gp.X_train.shape[0] == n + 1  # tell has already conditioned.
        assert float(record.observation) == pytest.approx(
            float(record.audit["stored_observation"]) - 19
        )
        assert torch.all(learner.state.observation_noise_variance == 0.05)
        assert not learner.gp.jitter_adjustments.any()
    history = _history(experiment)
    for row in range(3):
        selected = history[history[:, 0] == row, 1]
        assert len(selected) == len(np.unique(selected)) == 5
    assert len(learner.state.context_history) == 15


def test_stable_tie_prefers_lowest_source_index_with_declared_tolerance():
    learner = _algorithm()
    scores = torch.tensor([1.0, 1.0 + 1e-10, 0.5], dtype=torch.float64)
    actions = torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    indices = torch.tensor([4, 9, 1])
    assert learner._stable_position(scores, actions, indices) == 0
    scores[1] = 1.0 + 1e-5
    assert learner._stable_position(scores, actions, indices) == 1


def test_finite_calibrated_policy_does_not_consume_random_draws():
    experiment = _experiment()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    learner_state = experiment.algorithm.generator.get_state().clone()
    environment_state = experiment.environment.generator.get_state().clone()
    experiment.run(max_rounds=15)
    after = np.random.get_state()
    assert after[0] == numpy_state[0]
    np.testing.assert_array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert torch.equal(experiment.algorithm.generator.get_state(), learner_state)
    assert torch.equal(experiment.environment.generator.get_state(), environment_state)


def test_hidden_response_table_is_not_revealed_or_used_before_query():
    first = _experiment()
    _, _, _, changed_values = _tables()
    changed_values[:, 1:] += 1000
    second = _experiment(values=changed_values)
    for experiment in [first, second]:
        assert experiment.algorithm.state.y_train.numel() == 0
        info = experiment.environment.before_action()
        assert set(info) == {"context", "candidate_actions", "candidate_dataset_indices"}
        experiment.run_one_step()
    # Only the common first sensor has been observed, so both posteriors and
    # next proposals must match despite the other hidden values differing.
    assert torch.equal(first.algorithm.state.y_train, second.algorithm.state.y_train)
    np.testing.assert_array_equal(_history(first), _history(second))
    for experiment in [first, second]:
        assert experiment.algorithm.state.y_train.tolist() == [0.0]
    proposal1 = first.algorithm.ask(round_info=first.environment.before_action())
    proposal2 = second.algorithm.ask(round_info=second.environment.before_action())
    assert torch.equal(proposal1, proposal2)


def test_periodic_120_crash_replay_final_tail_and_completed_noop(tmp_path, monkeypatch):
    reference = _experiment(rows=26)
    reference.run(max_rounds=127)
    directory = tmp_path / "periodic"
    interrupted = _experiment(directory, rows=26)
    for _ in range(123):
        interrupted.run_one_step()
    assert interrupted.completed_rounds == 123
    # No graceful run() return: only the cadence checkpoint at 120 is durable.
    resumed = _experiment(directory, rows=26, resume=True)
    assert resumed.completed_rounds == 120
    resumed.run(max_rounds=123)  # Successful return flushes a mid-context tail.
    tail = _experiment(directory, rows=26, resume=True)
    assert tail.completed_rounds == 123
    assert len(tail.environment.before_action()["candidate_actions"]) == 3
    tail.run(max_rounds=127)
    np.testing.assert_array_equal(_history(reference), _history(tail))
    assert tail.algorithm.state.y_train.numel() == 127
    assert tail.algorithm._beta() == pytest.approx(
        2 * math.log(46 * 288 * 128**2 * math.pi**2 / 0.6)
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed run attempted a new proposal, feedback or evaluation")

    before = _history(tail).copy()
    monkeypatch.setattr(tail.environment, "step", forbidden)
    monkeypatch.setattr(tail.algorithm, "ask", forbidden)
    monkeypatch.setattr(tail.algorithm, "tell", forbidden)
    assert tail.run(max_rounds=127).evaluations == 127
    np.testing.assert_array_equal(_history(tail), before)


def test_checkpoint_rejects_changed_data_and_confidence_configuration(tmp_path):
    directory = tmp_path / "identity"
    experiment = _experiment(directory)
    experiment.run(max_rounds=3)
    _, _, _, changed = _tables()
    changed[-1, -1] += 0.125
    with pytest.raises(EnvironmentConfigurationError, match=r"fingerprint|source"):
        _experiment(directory, resume=True, values=changed)
    with pytest.raises(AlgorithmStateError, match=r"configuration.*match"):
        _experiment(directory, resume=True, delta=0.2)
