"""Offline seven-context integration and interruption checks, using a tiny ragged table.

The fixture is synthetic and never downloads or runs the scientific campaign.
It exercises the same public data reader, calibration, environment, learner,
checkpoint and evaluation APIs as the independent MHC notebook cell.
"""

import copy
import csv
import hashlib
import io
import json
import math
import zipfile
from dataclasses import replace

import numpy as np
import pytest
import torch

import basat
from basat.algorithms.algorithm import AlgorithmStateError
from basat.environments.environment import EnvironmentConfigurationError


COUNTS = (2, 3, 1, 4, 2, 3, 2)


@pytest.fixture(scope="module")
def toy_benchmark(tmp_path_factory):
    directory = tmp_path_factory.mktemp("mhc-protocol-source")
    table = io.StringIO()
    writer = csv.writer(table, delimiter="\t", lineterminator="\n")
    writer.writerow(["species", "allele", "length", "cv", "sequence", "inequality", "ic50", "smm"])
    peptides = ("YYYYYYYYY", "AAAAAAAAA", "CCCCCCCCC", "GGGGGGGGG")
    # Interleave contexts, preserving a source order different from action
    # lexicographic order. Each context has a different measured support.
    for local in range(max(COUNTS)):
        for context, count in enumerate(COUNTS):
            if local < count:
                ic50 = 10 ** (1.3 + .19 * context + .27 * local + .07 * ((context + local) % 2))
                writer.writerow(["human", basat.MHCBindingBenchmark.SOURCE_ALLELES[context],
                                 "9", "0", peptides[local], "=", repr(ic50), "987654321"])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(basat.MHCBindingBenchmark.SOURCE_MEMBER, table.getvalue())
    content = buffer.getvalue()
    source = directory / "toy_mhc.zip"
    source.write_bytes(content)

    class ToyMHCBindingBenchmark(basat.MHCBindingBenchmark):
        SOURCE_SHA256 = hashlib.sha256(content).hexdigest()
        EXPECTED_COUNTS = COUNTS

    return ToyMHCBindingBenchmark(local_asset=source)


@pytest.fixture(scope="module")
def fitted(toy_benchmark, tmp_path_factory):
    sample = toy_benchmark.calibration_sample(seed=2021, size=sum(COUNTS))
    return basat.fit_contextual_rbf(
        sample.observations, action_coordinates=sample.actions,
        context_coordinates=sample.contexts, context_support=toy_benchmark.context_support,
        context_covariance=toy_benchmark.context_gram, initial_action_lengthscale=1.,
        initial_noise_variance=.1, noise_variance_bounds=(.02, 1.),
        cache_directory=tmp_path_factory.mktemp("mhc-protocol-fit"),
    )


def experiment(benchmark, fit, directory=None, *, resume=False, delta=.1, warm_start=False):
    dataset = benchmark.make_dataset(noise_variance=fit.noise_variance)
    previous_data = None
    if warm_start:
        # Mirrors author mode without downloading the real 4418-row asset:
        # one explicitly identified base observation, absent from the later
        # searchable pool, and no calibration observations inserted into GP.
        previous_data = basat.EnvironmentData(
            actions=dataset.actions[:1], contexts=dataset.contexts[:1],
            observations=dataset.observations[:1],
            observation_noise_variance=fit.noise_variance,
        )
        dataset = replace(dataset, actions=dataset.actions[1:], contexts=dataset.contexts[1:],
                          observations=dataset.observations[1:])
    environment = basat.ContextualEnvironment(
        bounds=benchmark.bounds, context_provider=benchmark.context_provider,
        dataset=dataset, previous_data=previous_data,
        noise_std=0., add_noise_to_data=False, allow_repeated_actions=True,
        data_row_selection="first", dtype="float64", device="cpu", seed=2021,
    )
    algorithm = basat.CGP_UCB(
        bounds=benchmark.bounds, maximize=True, dtype="float64", device="cpu",
        config=basat.CGPUCBConfig(
            action_set=benchmark.actions, action_kernel=fit.action_kernel,
            context_kernel=fit.context_kernel, kernel_combination="product",
            beta="finite-theory", delta=delta, finite_domain_cardinality=len(dataset.actions),
            initial_points=1, batch_size=1, seed=2021, gp_backend="fixed_incremental",
            finite_prediction="marginal", marginal_chunk_size=2,
            recommendation_candidates="latest_context",
            acquisition_optimizer=basat.AcquisitionOptimizerSpec(fit_maxiter=0),
            gp_config=basat.GaussianProcessConfig(
                mean="zero", noise_mode="fixed", noise_variance=fit.noise_variance,
                standardize_outcomes=False, dtype="float64", device="cpu"),
        ),
    )
    return basat.Experiment(
        environment=environment, algorithm=algorithm, output_directory=directory,
        resume=resume, checkpoint_policy="periodic", save_every=10,
    )


def trajectory(run):
    return np.array([[record.round_index, record.dataset_index, float(record.context[0]),
                      float(record.observation)] for record in run.environment.records])


@pytest.fixture(scope="module")
def completed_reference(toy_benchmark, fitted):
    run = experiment(toy_benchmark, fitted)
    run.run(max_rounds=350, max_evaluations=350, fail_fast=True)
    return run


def test_seven_ragged_contexts_keep_one_350_query_history_and_calibrated_noise(
    toy_benchmark, fitted, completed_reference
):
    run = completed_reference
    assert run.completed_rounds == run.algorithm.iteration == run.algorithm.evaluations == 350
    assert run.algorithm.state.actions_train.shape == (350, 45)
    assert run.algorithm.gp.X_train.shape == (350, 46)
    assert run.algorithm.state.context_history[49].tolist() == [0.]
    assert run.algorithm.state.context_history[50].tolist() == [1.]
    assert run.algorithm.state.context_history[299].tolist() == [5.]
    assert run.algorithm.state.context_history[300].tolist() == [6.]
    torch.testing.assert_close(run.algorithm.state.contexts_train, toy_benchmark.context_schedule,
                               rtol=0, atol=0)
    assert fitted.noise_variance > 0
    torch.testing.assert_close(run.algorithm.gp.observation_noise,
                               torch.full((350,), fitted.noise_variance, dtype=torch.float64),
                               rtol=0, atol=0)
    assert run.environment.add_noise_to_data is False
    rewards = toy_benchmark.dataset.observations
    selected = []
    for query, record in enumerate(run.environment.records):
        row = int(record.dataset_index)
        selected.append(row)
        assert int(record.context[0]) == query // 50
        assert torch.equal(record.action, toy_benchmark.actions[row])
        assert torch.equal(record.context, toy_benchmark.contexts[row])
        assert float(record.observation) == float(rewards[row])
        assert float(record.audit["stored_observation"]) == float(rewards[row])
        assert float(record.audit["realized_noise"]) == 0.
        assert float(record.observation_noise_variance) == fitted.noise_variance
        assert len(record.round_info["candidate_actions"]) == COUNTS[query // 50]
    # More evaluations than distinct pairs confirms repeats were not silently
    # removed, and calibration rows did not consume or enter the online history.
    assert len(set(selected)) <= sum(COUNTS) < 350
    torch.testing.assert_close(run.algorithm.state.y_train, rewards[selected], rtol=0, atol=0)
    statistics = basat.contextual_regret_statistics(
        run.environment, context_order=toy_benchmark.context_support, queries_per_context=50)
    assert statistics.complete
    assert statistics.per_context_query_counts == (50,) * 7
    assert len(statistics.maximum_simple_regret) == 50
    assert len(statistics.average_regret) == 350


def test_global_confidence_clock_cardinality_and_source_index_ties(toy_benchmark, fitted, monkeypatch):
    run = experiment(toy_benchmark, fitted)
    assert run.algorithm.state.y_train.numel() == 0
    for query in (1, 49, 50, 51, 100, 301, 350):
        run.algorithm.iteration = query - 1
        expected = 2 * math.log(sum(COUNTS) * query**2 * math.pi**2 / (6 * .1))
        assert run.algorithm._beta() == pytest.approx(expected, abs=1e-13)
    run.algorithm.iteration = 0
    run.run_one_step()
    info = run.environment.before_action()
    selected_before = run.environment.records[0].dataset_index
    assert int(info["candidate_dataset_indices"].min()) == selected_before
    assert len(info["candidate_actions"]) == COUNTS[0]  # Repeated first pair remains available.
    # Use equal scores to isolate stable source-row tie resolution; the global
    # action table contains rows for other contexts, which must remain excluded.
    monkeypatch.setattr(run.algorithm, "_score_available_actions",
                        lambda actions, context: torch.ones(len(actions), dtype=torch.float64))
    run.run_one_step()
    assert run.environment.records[-1].dataset_index == selected_before
    permitted = toy_benchmark.contexts[:, 0] == 0
    assert permitted[run.environment.records[-1].dataset_index]
    assert torch.equal(run.environment.records[-1].action,
                       info["candidate_actions"][torch.argmin(info["candidate_dataset_indices"])] )


class SimulatedCrash(RuntimeError):
    pass


@pytest.mark.parametrize("completed", [49, 50, 53])
def test_periodic_crash_after_proposal_rolls_back_then_crosses_task_boundary_exactly(
    toy_benchmark, fitted, completed_reference, tmp_path, monkeypatch, completed
):
    interrupted = experiment(toy_benchmark, fitted, tmp_path)
    for _ in range(completed):
        interrupted.run_one_step()
    manifest = json.loads((tmp_path / "latest.json").read_text())
    durable = completed // 10 * 10
    assert manifest["completed_rounds"] == durable

    def crash_after_proposal(action):
        assert interrupted.algorithm.last_ask is not None
        assert len(interrupted.environment.records) == completed
        assert int(interrupted.algorithm.state.current_context[0]) == completed // 50
        raise SimulatedCrash("after proposal, before offline observation")

    monkeypatch.setattr(interrupted.environment, "step", crash_after_proposal)
    with pytest.raises(SimulatedCrash):
        interrupted.run_one_step()
    restored = experiment(toy_benchmark, fitted, tmp_path, resume=True)
    assert restored.completed_rounds == durable
    assert restored.algorithm.last_ask is None
    restored.run(max_rounds=57, max_evaluations=57, fail_fast=True)
    np.testing.assert_array_equal(trajectory(restored), trajectory(completed_reference)[:57])
    assert restored.algorithm.state.contexts_train[49:52].reshape(-1).tolist() == [0., 1., 1.]


@pytest.mark.parametrize("completed", [49, 50])
def test_explicitly_durable_feedback_replays_once_without_reevaluation(
    toy_benchmark, fitted, completed_reference, tmp_path, monkeypatch, completed
):
    interrupted = experiment(toy_benchmark, fitted, tmp_path)
    interrupted.run(max_rounds=completed, max_evaluations=completed, fail_fast=True)

    def save_feedback_then_crash(**kwargs):
        assert interrupted.environment.round == completed + 1
        assert interrupted.algorithm.iteration == completed
        assert interrupted.algorithm.last_ask is not None
        interrupted.save_checkpoint(phase="pre_tell")
        raise SimulatedCrash("durable feedback precedes learner consumption")

    monkeypatch.setattr(interrupted.algorithm, "tell", save_feedback_then_crash)
    with pytest.raises(SimulatedCrash):
        interrupted.run_one_step()
    manifest = json.loads((tmp_path / "latest.json").read_text())
    assert manifest["phase"] == "pre_tell"
    assert manifest["completed_rounds"] == completed + 1
    assert manifest["consumed_records"] == completed

    def forbidden_step(*args, **kwargs):
        pytest.fail("Restoring durable feedback must not reevaluate the environment")

    with monkeypatch.context() as local:
        local.setattr(basat.ContextualEnvironment, "step", forbidden_step)
        restored = experiment(toy_benchmark, fitted, tmp_path, resume=True)
    assert restored.algorithm.iteration == completed + 1
    assert restored.algorithm_consumed_environment_records == completed + 1
    assert len(restored.environment.records) == completed + 1
    assert restored.algorithm.last_ask is None
    np.testing.assert_array_equal(trajectory(restored), trajectory(completed_reference)[:completed + 1])
    restored.run(max_rounds=57, max_evaluations=57, fail_fast=True)
    np.testing.assert_array_equal(trajectory(restored), trajectory(completed_reference)[:57])


def test_graceful_final_tail_is_durable_and_completed_resume_is_a_noop(
    toy_benchmark, fitted, completed_reference, tmp_path, monkeypatch
):
    run = experiment(toy_benchmark, fitted, tmp_path)
    run.run(max_rounds=53, max_evaluations=53, fail_fast=True)
    manifest = json.loads((tmp_path / "latest.json").read_text())
    assert manifest["completed_rounds"] == manifest["consumed_records"] == 53
    assert manifest["phase"] == "run_end"
    restored = experiment(toy_benchmark, fitted, tmp_path, resume=True)
    before = trajectory(restored).copy()
    np.testing.assert_array_equal(before, trajectory(completed_reference)[:53])

    def forbidden(*args, **kwargs):
        pytest.fail("A completed-budget replay must not ask, tell, evaluate, or recalibrate")

    monkeypatch.setattr(restored.algorithm, "ask", forbidden)
    monkeypatch.setattr(restored.algorithm, "tell", forbidden)
    monkeypatch.setattr(restored.environment, "step", forbidden)
    monkeypatch.setattr(basat, "fit_contextual_rbf", forbidden)
    result = restored.run(max_rounds=53, max_evaluations=53, fail_fast=True)
    assert result.evaluations == 53
    np.testing.assert_array_equal(trajectory(restored), before)


def test_resume_rejects_changed_calibration_noise_kernel_and_confidence_config(
    toy_benchmark, fitted, tmp_path
):
    run = experiment(toy_benchmark, fitted, tmp_path)
    run.run(max_rounds=3, max_evaluations=3, fail_fast=True)
    altered_noise = replace(fitted, noise_variance=fitted.noise_variance + .01)
    with pytest.raises(EnvironmentConfigurationError, match=r"fingerprint|source|configuration"):
        experiment(toy_benchmark, altered_noise, tmp_path, resume=True)
    kernel = copy.deepcopy(fitted.action_kernel)
    kernel.base_kernel.lengthscale = kernel.base_kernel.lengthscale.detach() * 1.1
    altered_kernel = replace(fitted, action_kernel=kernel)
    with pytest.raises(AlgorithmStateError, match=r"configuration.*match"):
        experiment(toy_benchmark, altered_kernel, tmp_path, resume=True)
    with pytest.raises(AlgorithmStateError, match=r"configuration.*match"):
        experiment(toy_benchmark, fitted, tmp_path, resume=True, delta=.2)


@pytest.fixture(scope="module")
def completed_warm_reference(toy_benchmark, fitted):
    run = experiment(toy_benchmark, fitted, warm_start=True)
    run.run(max_rounds=350, max_evaluations=350, fail_fast=True)
    return run


def test_author_base_warm_start_preserves_350_online_queries_and_metric_clock(
    toy_benchmark, fitted, completed_warm_reference,
):
    run = completed_warm_reference
    assert run.completed_rounds == run.algorithm.iteration == run.algorithm.evaluations == 350
    assert len(run.environment.records) == 350
    assert run.algorithm.gp.X_train.shape == (351, 46)
    assert run.algorithm.state.y_train.numel() == 351
    initial = run.environment.history.initial_data
    assert initial.number_of_observations == 1
    assert torch.equal(initial.actions[0], toy_benchmark.actions[0])
    assert torch.equal(initial.observations[0], toy_benchmark.dataset.observations[0])
    assert torch.equal(run.algorithm.state.actions_train[0], initial.actions[0])
    assert run.environment.dataset.actions.shape[0] == sum(COUNTS) - 1
    # The removed peptide can still exist in another context; exclusion is of
    # the precise measured action/context pair, not the peptide globally.
    online_pairs = torch.cat((run.environment.dataset.actions, run.environment.dataset.contexts), dim=1)
    base_pair = torch.cat((initial.actions[0], initial.contexts[0]))
    assert not torch.any(torch.all(online_pairs == base_pair, dim=1))
    assert [int(record.context[0]) for record in run.environment.records] == [i // 50 for i in range(350)]
    assert run.algorithm.state.contexts_train[49:53, 0].tolist() == [0., 0., 1., 1.]
    assert run.algorithm.gp.observation_noise.numel() == 351
    torch.testing.assert_close(
        run.algorithm.gp.observation_noise,
        torch.full((351,), fitted.noise_variance, dtype=torch.float64), rtol=0, atol=0)

    with pytest.raises(ValueError, match="warm-start data"):
        basat.contextual_regret_statistics(
            run.environment, context_order=toy_benchmark.context_support, queries_per_context=50)
    statistics = basat.contextual_regret_statistics(
        run.environment, context_order=toy_benchmark.context_support,
        queries_per_context=50, exclude_initial=True)
    assert statistics.complete
    assert statistics.initial_excluded_count == 1
    assert statistics.observed_queries == statistics.expected_queries == 350
    assert statistics.per_context_query_counts == (50,) * 7
    assert statistics.query_indices == tuple(range(1, 351))
    assert statistics.local_query_indices == tuple(range(1, 51)) * 7
    assert len(statistics.maximum_simple_regret) == 50
    assert len(statistics.average_regret) == 350
    # Base row is deliberately better than every remaining first-task reward.
    # It must not inflate the reference optimum or any online incumbent.
    assert initial.observations.item() > statistics.reference_values[0]
    assert statistics.reference_values[0] == pytest.approx(float(run.environment.records[0].observation))
    assert statistics.per_context_incumbent_values[0][0] == statistics.selected_rewards[0]
    assert statistics.per_context_simple_regret[0][0] == 0.
    np.testing.assert_allclose(
        statistics.average_regret,
        np.cumsum(statistics.instantaneous_regret) / np.arange(1, 351), atol=1e-14, rtol=0)


def test_author_base_does_not_advance_global_confidence_clock(toy_benchmark, fitted):
    run = experiment(toy_benchmark, fitted, warm_start=True)
    assert run.algorithm.state.y_train.numel() == 1
    assert run.algorithm.iteration == run.algorithm.evaluations == 0
    assert run.completed_rounds == 0
    assert len(run.environment.records) == 0
    cardinality = sum(COUNTS) - 1
    for completed in range(52):
        expected = 2 * math.log(cardinality * (completed + 1)**2 * math.pi**2 / (6 * .1))
        assert run.algorithm._beta() == pytest.approx(expected, abs=1e-13)
        assert run.algorithm.state.y_train.numel() == completed + 1
        run.run_one_step()
    assert run.environment.records[49].context.tolist() == [0.]
    assert run.environment.records[50].context.tolist() == [1.]


@pytest.mark.parametrize("crash_phase", ["pending_proposal", "durable_feedback"])
def test_author_base_interrupted_resume_crosses_task_boundary_and_completed_noop(
    toy_benchmark, fitted, completed_warm_reference, tmp_path, monkeypatch, crash_phase,
):
    run = experiment(toy_benchmark, fitted, tmp_path, warm_start=True)
    run.run(max_rounds=50, max_evaluations=50, fail_fast=True)

    if crash_phase == "pending_proposal":
        def crash(action):
            assert run.algorithm.last_ask is not None
            assert int(run.algorithm.state.current_context[0]) == 1
            assert len(run.environment.records) == 50
            raise SimulatedCrash("proposal before first second-task observation")

        monkeypatch.setattr(run.environment, "step", crash)
    else:
        def crash(**kwargs):
            assert len(run.environment.records) == 51
            assert run.algorithm.iteration == 50
            assert run.algorithm.state.y_train.numel() == 51
            run.save_checkpoint(phase="pre_tell")
            raise SimulatedCrash("first second-task feedback committed before tell")

        monkeypatch.setattr(run.algorithm, "tell", crash)
    with pytest.raises(SimulatedCrash):
        run.run_one_step()

    def forbidden(*args, **kwargs):
        pytest.fail("Durable feedback recovery/completed resume must perform no new work")

    with monkeypatch.context() as scoped:
        if crash_phase == "durable_feedback":
            scoped.setattr(basat.ContextualEnvironment, "step", forbidden)
        restored = experiment(toy_benchmark, fitted, tmp_path, resume=True, warm_start=True)
    expected_completed = 50 if crash_phase == "pending_proposal" else 51
    assert restored.completed_rounds == expected_completed
    assert restored.algorithm.state.y_train.numel() == expected_completed + 1
    assert restored.environment.history.initial_data.number_of_observations == 1
    restored.run(max_rounds=57, max_evaluations=57, fail_fast=True)
    np.testing.assert_array_equal(trajectory(restored), trajectory(completed_warm_reference)[:57])
    torch.testing.assert_close(
        restored.algorithm.state.y_train, completed_warm_reference.algorithm.state.y_train[:58],
        rtol=0, atol=0)
    manifest = json.loads((tmp_path / "latest.json").read_text())
    assert manifest["completed_rounds"] == manifest["consumed_records"] == 57
    no_op = experiment(toy_benchmark, fitted, tmp_path, resume=True, warm_start=True)
    before = trajectory(no_op).copy()
    monkeypatch.setattr(no_op.algorithm, "ask", forbidden)
    monkeypatch.setattr(no_op.algorithm, "tell", forbidden)
    monkeypatch.setattr(no_op.environment, "step", forbidden)
    monkeypatch.setattr(basat, "fit_contextual_rbf", forbidden)
    result = no_op.run(max_rounds=57, max_evaluations=57, fail_fast=True)
    assert result.evaluations == 57
    assert no_op.algorithm.state.y_train.numel() == 58
    np.testing.assert_array_equal(trajectory(no_op), before)
