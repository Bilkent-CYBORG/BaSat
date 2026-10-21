"""Source schedule, transaction recovery, and completed-result reuse checks."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import basat
from basat import _robos_protocol as source

LABELS = ("DRBO 3e", "DRBO e", "DRBO e/3", "RoBOS")


@pytest.fixture
def benchmark():
    root = Path(__file__).resolve().parents[1]
    candidates = (
        root / "results/reproducibility/E5/custom.npy",
        root / "results/reproducibility/robos2023_synthetic_source_v1/data/custom.npy",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        pytest.skip("The checksum-pinned RoBOS custom.npy local test asset is unavailable.")
    return basat.RoBOSSynthetic(local_asset=path)


def make_experiment(benchmark, output, *, rounds=2, resume=True, protocol=None):
    algorithms, environments = {}, {}
    for index, label in enumerate(LABELS):
        gp = basat.GaussianProcessConfig(
            kernel=basat.FixedRBFKernel(input_dimension=2, lengthscale=[.2, 5.], ard=True),
            mean="zero", noise_mode="fixed", noise_variance=.0004,
            standardize_outcomes=False, jitter=1e-10, dtype="float64", device="cpu",
        )
        shared = dict(
            context_kernel=basat.FixedRBFKernel(input_dimension=1, lengthscale=5.),
            action_kernel=basat.FixedRBFKernel(input_dimension=1, lengthscale=.2),
            gp_config=gp, gp_backend="fixed_gpy", gp_training_regularization=1e-8,
            beta=4., initial_points=3, seed=123, batch_size=1,
            feasibility_tolerance=1e-7, optimizer_tolerance=1e-8,
        )
        if index < 3:
            algorithms[label] = basat.DRBO(
                bounds=benchmark.bounds, dtype="float64", config=basat.DRBOConfig(
                    **shared, inner_solver="source_cvxpy", finite_action_tie_breaking="first",
                    action_optimizer=basat.ActionOptimizerSpec(fit_maxiter=0),
                ),
            )
        else:
            algorithms[label] = basat.RoBOS(
                bounds=benchmark.bounds, dtype="float64", config=basat.RoBOSConfig(
                    **shared, aspiration=.6, simplex_solver="source_cobyla",
                    allow_semidefinite_mmd=True, fit_maxiter=0,
                ),
            )
        environments[label] = basat.DistributionalEnvironment(
            bounds=benchmark.bounds, dataset=benchmark.dataset,
            context_support=benchmark.context_support,
            reference_distribution=benchmark.reference_distribution,
            sampling_distribution=benchmark.sampling_distribution,
            margin=(3 * benchmark.epsilon, benchmark.epsilon, benchmark.epsilon / 3, benchmark.epsilon)[index],
            noise_std=.02, add_noise_to_data=True, allow_repeated_actions=True,
            dtype="float64", seed=123,
        )
    if protocol is None:
        protocol = basat.RoBOSSourceProtocol(
            online_rounds=rounds, method_roles=dict(zip(LABELS, source.ROLES, strict=True)),
        )
    return basat.Experiment(
        environment=environments, algorithm=algorithms, output_directory=output,
        resume=resume, save_every=201, checkpoint_policy="transactional",
        evaluation_protocol=protocol,
    )


def run(experiment, benchmark, *, rounds=2, trials=2, max_trials=None):
    return experiment.run(
        trials=trials, max_trials=max_trials, max_rounds=rounds,
        regret=basat.RoBOSSourceRegret(aspiration=.6, epsilon=benchmark.epsilon),
        plot=False, show=False, progress=False, fail_fast=True,
    )


@pytest.fixture
def cheap_protocol(monkeypatch):
    """Keep native ask/tell/GP paths, replacing only expensive inner programs."""
    counters = {"drbo": 0, "fragility": 0}
    monkeypatch.setattr(source, "_require_runtime", source._runtime)

    def robust(self, upper, reference, margin, gram):
        del self, margin, gram
        counters["drbo"] += 1
        return float(reference @ upper), reference.clone(), {"simplex_sum": 0.}

    def fragility(self, upper, reference, gram):
        del gram
        counters["fragility"] += 1
        if self._source_solver_starts:
            self._source_solver_starts.pop(0)
        self._source_solver_diagnostics = {
            "solver": "test_only_deterministic", "success": True, "feasible": True,
            "global_optimum_certified": False,
        }
        return float(self.config.aspiration - reference @ upper), reference.clone(), {"simplex_sum": 0.}

    monkeypatch.setattr(basat.DRBO, "_worst_case_expected_ucb", robust)
    monkeypatch.setattr(basat.RoBOS, "_solve_fragility_source_cobyla", fragility)
    return counters


def _measured(result):
    """Compare scientific trajectories without unstable wall-clock metadata."""
    trials = []
    for row in result.records:
        methods = {}
        for label in LABELS:
            directory = Path(row["algorithms"][label]["output_directory"])
            manifest = json.loads((directory / "latest.json").read_text())
            checkpoint = torch.load(directory / manifest["checkpoint"], weights_only=False)
            methods[label] = basat.Experiment._jsonable(checkpoint["environment_state"]["history"])
        trials.append(methods)
    return trials, result.regret_runs


def test_source_plan_matches_released_draw_order_and_continuation(benchmark):
    policy = basat.RoBOSSourceProtocol(
        online_rounds=201, method_roles=dict(zip(LABELS, source.ROLES, strict=True)),
    )
    source_rng, actual_rng = np.random.RandomState(123), np.random.RandomState(123)
    probability = benchmark.sampling_distribution.numpy()
    previous_end = None
    for trial in range(2):
        plan = source.make_plan(policy, actual_rng, probability, action_count=8, trial_index=trial)
        assert plan["initial_indices"] == source_rng.randint(0, 400, 3).tolist()
        np.testing.assert_array_equal(plan["discarded_initial_normals"], source_rng.randn(3))
        uniforms = source_rng.rand(201)
        np.testing.assert_array_equal(plan["context_uniforms"], uniforms)
        assert plan["context_indices"] == [int(np.sum(probability.cumsum() <= u)) for u in uniforms]
        np.testing.assert_array_equal(plan["baseline_noise"], source_rng.randn(4, 201) * .02)
        for t in range(201):
            for action in range(8):
                for who in ("learner", "oracle"):
                    np.testing.assert_array_equal(
                        plan["starts"][t][who][action],
                        np.diff(np.r_[0., np.sort(source_rng.rand(49)), 1.]),
                    )
            assert plan["robos_noise"][t] == source_rng.randn(1, 1).item() * .02
        np.testing.assert_array_equal(plan["discarded_so_noise"], source_rng.randn(201) * .02)
        assert plan["end_rng"] == source._rng_state(source_rng)
        if previous_end is not None:
            assert plan["start_rng"] == previous_end
        previous_end = plan["end_rng"]
        source._check_plan(plan)


def test_small_trials_share_initialization_and_keep_expected_reward_hidden(benchmark, cheap_protocol, tmp_path):
    experiment = make_experiment(benchmark, tmp_path)
    result = run(experiment, benchmark)
    assert cheap_protocol["drbo"] > 0 and cheap_protocol["fragility"] > 0
    expected = basat.expected_values(benchmark.payoff_table, benchmark.sampling_distribution)
    for index, trial in enumerate(experiment.trial_experiments):
        initial_rows, context_rows = [], []
        plan = json.loads((tmp_path / "trials" / f"trial-{index + 1:03d}" / "random-plan.json").read_text())
        for label, item in trial.experiments.items():
            history = item.environment.history
            initial_rows.append(history.initial_data.observations.tolist())
            context_rows.append([record.context.tolist() for record in history.records])
            assert len(history.initial_data.observations) == 3
            assert item.algorithm.evaluations == 2
            assert item.algorithm.state.y_train.numel() == 5
            assert item.query_counts == {
                "initialization_queries": 0, "shared_initial_observations": 3,
                "online_queries": 2, "total_objective_queries": 2,
                "total_training_observations": 5,
            }
            for t, record in enumerate(history.records):
                selected = int(torch.nonzero(benchmark.actions[:, 0] == record.action.reshape(-1)[0])[0])
                assert float(record.audit["expected_observation"]) == pytest.approx(float(expected[selected]), abs=1e-15)
                assert record.audit["realized_context_index"] == plan["context_indices"][t]
                assert "expected_observation" not in record.feedback_extra
                assert "true_sampling_distribution" not in record.round_info
                assert "source_kappa" not in record.round_info
            report = result.results[label][index].metadata["regret"]
            assert report["reward_source"] == "expected_observation"
            assert report["global_optimum_certified"] is False
        assert initial_rows == [initial_rows[0]] * 4
        assert context_rows == [context_rows[0]] * 4


def test_completed_trials_do_not_ask_solve_refit_or_evaluate(benchmark, cheap_protocol, monkeypatch, tmp_path):
    direct = run(make_experiment(benchmark, tmp_path), benchmark)
    expected = _measured(direct)

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed source trials must perform no new learner or objective work.")

    monkeypatch.setattr(basat.DRBO, "ask", forbidden)
    monkeypatch.setattr(basat.RoBOS, "ask", forbidden)
    monkeypatch.setattr(basat.DRBO, "_worst_case_expected_ucb", forbidden)
    monkeypatch.setattr(basat.RoBOS, "evaluate_fragility", forbidden)
    monkeypatch.setattr(basat.FixedKernelGaussianProcess, "set_data", forbidden)
    monkeypatch.setattr(basat.DistributionalEnvironment, "_evaluate_source_scheduled", forbidden)
    monkeypatch.setattr(source, "_require_runtime", forbidden)
    resumed = run(make_experiment(benchmark, tmp_path), benchmark)
    assert _measured(resumed) == expected


@pytest.mark.parametrize("phase", ["pending_proposal", "pre_tell"])
def test_durable_pending_or_feedback_checkpoint_resumes_exactly(benchmark, cheap_protocol, monkeypatch, tmp_path, phase):
    direct = run(make_experiment(benchmark, tmp_path / "direct"), benchmark)
    direct_solves = dict(cheap_protocol)
    target = tmp_path / "resumed"
    original = basat.Experiment.save_checkpoint
    interrupted = False

    class PowerLoss(BaseException):
        pass

    def interrupt_after_commit(self, *, phase="post_tell", archive=False):
        nonlocal interrupted
        result = original(self, phase=phase, archive=archive)
        if not interrupted and phase == interrupt_phase and self.evaluation_protocol == "robos_2023":
            interrupted = True
            raise PowerLoss()
        return result

    interrupt_phase = phase
    monkeypatch.setattr(basat.Experiment, "save_checkpoint", interrupt_after_commit)
    with pytest.raises(PowerLoss):
        run(make_experiment(benchmark, target), benchmark)
    assert interrupted
    monkeypatch.setattr(basat.Experiment, "save_checkpoint", original)
    resumed = run(make_experiment(benchmark, target), benchmark)
    assert _measured(resumed) == _measured(direct)
    assert cheap_protocol == {name: 2 * count for name, count in direct_solves.items()}


def test_initialization_resumes_after_durable_shared_observation(benchmark, cheap_protocol, monkeypatch, tmp_path):
    direct = run(make_experiment(benchmark, tmp_path / "direct"), benchmark)
    target = tmp_path / "resumed"
    original = basat.Experiment._atomic_json_save
    interrupted = False

    class PowerLoss(BaseException):
        pass

    def interrupt_initialization(value, path):
        nonlocal interrupted
        result = original(value, path)
        if not interrupted and Path(path).name == "initialization.json" and len(value["observations"]) == 1:
            interrupted = True
            raise PowerLoss()
        return result

    monkeypatch.setattr(basat.Experiment, "_atomic_json_save", staticmethod(interrupt_initialization))
    with pytest.raises(PowerLoss):
        run(make_experiment(benchmark, target), benchmark)
    state = json.loads((target / "trials/trial-001/initialization.json").read_text())
    assert len(state["observations"]) == 1
    monkeypatch.setattr(basat.Experiment, "_atomic_json_save", staticmethod(original))
    resumed = run(make_experiment(benchmark, target), benchmark)
    assert _measured(resumed) == _measured(direct)


def test_metric_oracle_resumes_after_durable_round_commit(benchmark, cheap_protocol, monkeypatch, tmp_path):
    direct = run(make_experiment(benchmark, tmp_path / "direct"), benchmark)
    direct_solves = dict(cheap_protocol)
    target = tmp_path / "resumed"
    original = basat.Experiment._atomic_json_save
    interrupted = False

    class PowerLoss(BaseException):
        pass

    def interrupt_oracle(value, path):
        nonlocal interrupted
        result = original(value, path)
        if not interrupted and Path(path).name == "source-oracle.json" and len(value["values"]) == 1:
            interrupted = True
            raise PowerLoss()
        return result

    monkeypatch.setattr(basat.Experiment, "_atomic_json_save", staticmethod(interrupt_oracle))
    with pytest.raises(PowerLoss):
        run(make_experiment(benchmark, target), benchmark)
    state = json.loads((target / "trials/trial-001/source-oracle.json").read_text())
    assert len(state["values"]) == len(state["diagnostics"]) == 1
    monkeypatch.setattr(basat.Experiment, "_atomic_json_save", staticmethod(original))
    resumed = run(make_experiment(benchmark, target), benchmark)
    assert _measured(resumed) == _measured(direct)
    assert cheap_protocol == {name: 2 * count for name, count in direct_solves.items()}


def test_protocol_deepcopy_and_mutated_truth_fingerprint_rejected(benchmark, cheap_protocol, tmp_path):
    roles = dict(zip(LABELS, source.ROLES, strict=True))
    policy = basat.RoBOSSourceProtocol(online_rounds=2, method_roles=roles)
    roles["RoBOS"] = "invalid"
    assert policy.method_roles["RoBOS"] == "robos"
    cloned = copy.deepcopy(policy)
    assert cloned == policy and cloned.method_roles is not policy.method_roles
    run(make_experiment(benchmark, tmp_path, protocol=policy), benchmark)
    changed = make_experiment(benchmark, tmp_path, protocol=cloned)
    for environment in changed.source_environments.values():
        environment.dataset.observations[0] += .01
    with pytest.raises(basat.ExperimentCheckpointError, match="configuration changed"):
        run(changed, benchmark)


def test_full_budget_parameter_and_accounting_contract(benchmark, cheap_protocol, tmp_path):
    experiment = make_experiment(benchmark, tmp_path, rounds=201)
    metric = basat.RoBOSSourceRegret(aspiration=.6, epsilon=benchmark.epsilon)
    for online, total in ((200, None), (201, 201), (201, 200)):
        with pytest.raises(basat.ExperimentConfigurationError, match="budget"):
            experiment.run(trials=50, max_trials=1, max_rounds=online,
                           max_evaluations=total, regret=metric, plot=False)
    holder = SimpleNamespace(
        _trial_documents=[{} for _ in range(50)], _trial_count=50,
        _atomic_json_save=basat.Experiment._atomic_json_save, output_directory=tmp_path,
        elapsed_seconds=0., save=lambda: None,
    )
    source._save_summary(holder, experiment._robos_protocol)
    summary = json.loads((tmp_path / "source_summary.json").read_text())
    assert summary["online_queries"] == 40200
    assert summary["shared_initialization_queries"] == 150
    assert summary["initial_training_rows"] == 600
    assert summary["total_objective_queries"] == 40350


@pytest.mark.slow
def test_real_source_two_round_two_trial_smoke(benchmark, tmp_path):
    """Execute actual cold CVXPY and classical COBYLA in the dedicated kernel."""
    try:
        source._require_runtime()
    except basat.ExperimentConfigurationError:
        pytest.skip("Real historical smoke requires the dedicated RoBOS source kernel.")
    basat.set_seed(123, threads=1)
    experiment = make_experiment(benchmark, tmp_path)
    result = run(experiment, benchmark)
    assert result.planned_trials == 2
    assert len(result.records) == 2
    assert set(result.regret_runs) == set(LABELS)
    for label in LABELS:
        assert len(result.regret_runs[label]) == 2
        assert all(len(values) == 2 for values in result.regret_runs[label])
    summary = json.loads((tmp_path / "source_summary.json").read_text())
    assert summary["online_queries"] == 16
    assert summary["shared_initialization_queries"] == 6
    assert summary["initial_training_rows"] == 24
    assert summary["total_objective_queries"] == 22
