"""Released finite policies through the public transactional Experiment API."""
import json
from pathlib import Path

import pytest
import torch
import numpy as np

import basat

torch.set_num_threads(1)


def make(method=0, seed=123, directory=None, resume=False):
    benchmark = basat.AdveRSSynthetic()
    kernel = basat.make_kernel(
        basat.KernelConfig(name="polynomial", kwargs={"power": 6}),
        input_dimension=2, parameters={"offset": 2.84 / 1.11, "outputscale": 1.32 * 1.11**6},
        dtype="float64",
    )
    gp = basat.GaussianProcessConfig(
        kernel=kernel, mean="zero", noise_mode="fixed", noise_variance=1.,
        standardize_outcomes=False, dtype="float64", jitter=1e-10,
    )
    common = dict(bounds=benchmark.bounds, dtype="float64", verbose=False, plot=False)
    config = dict(beta=4., initial_points=10, fit_maxiter=0, gp_config=gp,
                  gp_backend="polynomial_features", source_protocol="advers_2025", seed=seed)
    if method < 3:
        algorithm = basat.AdveRS(**common, config=basat.AdveRSConfig(
            **config, variant=("RS1", "RS2", "RSG")[method], p=2. if method == 2 else 1.,
            aspiration=-10., finite_tie_break="first"))
    else:
        algorithm = basat.StableOPT(**common, config=basat.StableOPTConfig(
            **config, perturbation_radius=(.5, .25, 2.)[method - 3],
            query_policy="center", finite_tie_break="random"))
    env = basat.AdversarialEnvironment(
        bounds=benchmark.bounds, function=benchmark, candidate_actions=benchmark.grid(),
        attack=basat.LCBAttack(.5**.5 if method == 2 else .5),
        noise_std=1., dtype="float64", seed=seed, allow_repeated_actions=True,
    )
    return basat.Experiment(
        environment=env, algorithm=algorithm, evaluation_protocol="advers_2025",
        output_directory=directory, resume=resume, save_every=100,
    )


def trajectory(experiment):
    return [(r.audit["candidate_index"], r.audit["evaluated_candidate_index"],
             float(r.observation)) for r in experiment.environment.history.records]


def test_native_export_preserves_unbounded_fragility_diagnostic(tmp_path):
    experiment = make()
    experiment.run(max_rounds=1, plot=False, show=False)
    # Infinity denotes absence of a finite pessimistic certificate, not a bad
    # reward. Native Experiment export must preserve that source diagnostic.
    experiment.algorithm.state.pessimistic_scores.append(float("inf"))
    path = experiment.save(tmp_path / "native-results.json")
    report = json.loads(path.read_text())
    algorithm = next(iter(report["algorithms"].values()))
    assert algorithm["result"]["diagnostics"]["algorithm_specific"]["last_pessimistic_score"] == float("inf")
    assert all(np.isfinite(algorithm["observations"]))


@pytest.mark.parametrize("method", range(6))
def test_pinned_source_trajectory(method):
    fixture = json.loads((Path(__file__).parent / "fixtures/advers_2025_lcb.json").read_text())
    record = fixture["records"][method]
    experiment = make(method, record["seed"])
    assert experiment.environment.history.initial_data is None
    assert not experiment.algorithm._initialized
    result = experiment.run(max_rounds=5, plot=False, show=False)
    assert experiment._advers_bootstrap["indices"] == record["initial_indices"]
    observed = trajectory(experiment)
    assert [r[0] for r in observed] == record["proposed"]
    assert [r[1] for r in observed] == record["evaluated"]
    assert [r[2] for r in observed] == pytest.approx(record["observations"], abs=1e-10)
    assert result.metadata["query_counts"] == {
        "initialization_queries": 10, "online_queries": 5, "total_objective_queries": 15}


@pytest.mark.parametrize("method", [0, 4])
@pytest.mark.parametrize("phase", ["initial_design", "initial_observation", "initialized",
                                  "pending_proposal", "pre_tell", "post_tell"])
def test_interruption_resume_and_noop(tmp_path, monkeypatch, phase, method):
    seed = basat.derive_seed(2026, 1, method, 0)
    reference = make(method, seed)
    reference.run(max_rounds=4, plot=False, show=False)
    interrupted = make(method, seed, tmp_path)
    save = interrupted.save_checkpoint

    def crash(**kwargs):
        saved = save(**kwargs)
        if kwargs.get("phase") == phase:
            raise RuntimeError("injected crash after durable save")
        return saved

    monkeypatch.setattr(interrupted, "save_checkpoint", crash)
    with pytest.raises(RuntimeError, match="injected crash"):
        interrupted.run(max_rounds=4, plot=False, show=False)
    recovered = make(method, seed, tmp_path, resume=True)
    recovered.run(max_rounds=4, plot=False, show=False)
    assert trajectory(recovered) == trajectory(reference)
    assert recovered.algorithm.get_source_random_state() == reference.algorithm.get_source_random_state()
    before = recovered.query_counts
    monkeypatch.setattr(recovered.environment, "evaluate_function", lambda *args: pytest.fail("extra query"))
    recovered.run(max_rounds=4, plot=False, show=False)
    assert recovered.query_counts == before


def test_total_budget_includes_generated_initialization(tmp_path):
    experiment = make(directory=tmp_path)
    experiment.run(max_evaluations=4, plot=False, show=False)
    assert experiment.query_counts == {
        "initialization_queries": 4, "online_queries": 0, "total_objective_queries": 4}
    resumed = make(directory=tmp_path, resume=True)
    resumed.run(max_evaluations=12, plot=False, show=False)
    assert resumed.query_counts == {
        "initialization_queries": 10, "online_queries": 2, "total_objective_queries": 12}


def test_regret_uses_only_latent_online_rewards():
    experiment = make()
    metric = basat.RSRegret(aspiration=-10., optimal_fragility=20.129238666994205, epsilon=.5)
    result = experiment.run(max_rounds=3, regret=metric, plot=False, show=False)
    truth = [float(r.audit["latent_observation"]) for r in experiment.environment.history.records]
    assert len(result.metadata["regret"]["values"]) == 3
    assert result.metadata["regret"]["values"] == pytest.approx(metric.evaluate(truth).tolist())


def test_fixed_polynomial_posterior_matches_independent_dense_equations():
    experiment = make()
    gp = experiment.algorithm.gp
    train = np.array([[0., 0.], [.1, .2], [-.1, .15], [.3, -.2]])
    targets = np.array([1., -.5, 2., 1.5])
    points = np.array([[.05, .05], [.2, -.1]])
    gp.set_data(train, targets, np.ones(4))
    mean, std = gp.predict(torch.from_numpy(points))
    def kernel(a, b):
        return 1.32 * (1.11 * a @ b.T + 2.84)**6
    cross = kernel(train, points)
    covariance = kernel(train, train) + np.eye(4)
    expected_mean = cross.T @ np.linalg.solve(covariance, targets)
    expected_var = np.diag(kernel(points, points)) - np.sum(
        cross * np.linalg.solve(covariance, cross), axis=0)
    np.testing.assert_allclose(mean.numpy(), expected_mean, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(std.numpy()**2, expected_var, rtol=1e-7, atol=1e-8)
