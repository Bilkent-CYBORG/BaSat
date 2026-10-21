"""Exact contextual calibration, finite marginal prediction, and masked recommendations."""

import importlib
import json
from dataclasses import replace

import gpytorch
import numpy as np
import pytest
import torch
from scipy.optimize import OptimizeResult
from scipy.spatial.distance import cdist

import basat
from basat.models.ContextualCalibration import fit_contextual_rbf

module = importlib.import_module("basat.models.ContextualCalibration")


def arguments():
    return dict(
        observations=np.array([0.2, 1.1, -0.4, 0.4, 1.3, -0.2, 0.7, 1.6]),
        action_coordinates=np.array([[0.0], [0.8], [1.5], [0.0], [0.8], [1.5], [0.3], [0.8]]),
        context_coordinates=np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [2.0], [2.0]]),
        context_support=np.array([[0.0], [1.0], [2.0]]),
        context_covariance=np.array([[1., .3, .1], [.3, 1., .2], [.1, .2, 1.]]),
        initial_signal_variance=1.2,
    )


def fixed_optimizer(monkeypatch, *, success=True):
    calls = []

    def optimize(fun, x0, *, jac, method, bounds, options):
        value, gradient = fun(x0)
        calls.append(dict(jac=jac, method=method, options=options, gradient=gradient))
        return OptimizeResult(x=x0.copy(), fun=value, jac=gradient, success=success,
                              nit=1, message="fixture converged" if success else "iteration limit")

    monkeypatch.setattr(module, "minimize", optimize)
    return calls


def test_irregular_likelihood_and_analytic_gradient_match_independent_dense_calculation():
    args = arguments()
    x, y = args["action_coordinates"], args["observations"]
    ids = args["context_coordinates"].astype(int).ravel()
    distances = cdist(x, x, metric="sqeuclidean")
    gram = args["context_covariance"][np.ix_(ids, ids)]
    parameters = np.log([.7, 1.2, .1])
    actual, gradient = module._negative_log_likelihood(parameters, distances, gram, y)
    covariance = 1.2 * np.exp(-distances / (2 * .7**2)) * gram + .1 * np.eye(len(y))
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign == 1
    expected = .5 * (y @ np.linalg.solve(covariance, y) + logdet + len(y) * np.log(2 * np.pi))
    assert actual == pytest.approx(expected, abs=2e-12)
    finite_difference = []
    for coordinate in range(3):
        step = np.eye(3)[coordinate] * 1e-5
        plus = module._negative_log_likelihood(parameters + step, distances, gram, y)[0]
        minus = module._negative_log_likelihood(parameters - step, distances, gram, y)[0]
        finite_difference.append((plus - minus) / 2e-5)
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-8, atol=2e-8)


def test_calibration_keeps_zero_mean_and_returns_matching_fixed_kernels(monkeypatch):
    calls = fixed_optimizer(monkeypatch)
    args = arguments()
    fit = fit_contextual_rbf(**args)
    x = torch.tensor(args["action_coordinates"], dtype=torch.float64)
    contexts = torch.tensor(args["context_coordinates"], dtype=torch.float64)
    with torch.no_grad():
        covariance = (fit.action_kernel(x).to_dense() * fit.context_kernel(contexts).to_dense()).numpy()
    covariance += fit.noise_variance * np.eye(len(x))
    y = args["observations"]
    expected = .5 * (y @ np.linalg.solve(covariance, y)
                      + np.linalg.slogdet(covariance)[1] + len(y) * np.log(2 * np.pi))
    assert fit.observation_offset == 0
    assert fit.diagnostics["negative_log_marginal_likelihood"] == pytest.approx(expected, abs=2e-12)
    assert not any(p.requires_grad for p in fit.action_kernel.parameters())
    assert calls[0]["jac"] is True
    assert calls[0]["options"] == dict(maxiter=200, ftol=1e-10, maxls=40)
    args.pop("initial_signal_variance")
    automatic = fit_contextual_rbf(**args)
    assert automatic.parameters["signal_variance"] == pytest.approx(y.var(ddof=0))


def test_atomic_cache_reload_avoids_optimizer_and_rejects_changed_input(tmp_path, monkeypatch):
    calls = fixed_optimizer(monkeypatch)
    first = fit_contextual_rbf(**arguments(), cache_directory=tmp_path)
    before = (tmp_path / "contextual_rbf.json").read_bytes()
    second = fit_contextual_rbf(**arguments(), cache_directory=tmp_path)
    assert len(calls) == 1
    assert second.diagnostics["cache_hit"]
    assert first.parameters == second.parameters
    assert (tmp_path / "contextual_rbf.json").read_bytes() == before
    changed = arguments()
    changed["observations"][0] += .01
    with pytest.raises(ValueError, match="fingerprint does not match"):
        fit_contextual_rbf(**changed, cache_directory=tmp_path)
    with pytest.raises(ValueError, match="fingerprint does not match"):
        fit_contextual_rbf(**arguments(), maxiter=201, cache_directory=tmp_path)
    assert len(calls) == 1


@pytest.mark.parametrize("resign", [False, True])
def test_corrupted_cache_is_never_used_or_replaced(tmp_path, monkeypatch, resign):
    calls = fixed_optimizer(monkeypatch)
    fit_contextual_rbf(**arguments(), cache_directory=tmp_path)
    path = tmp_path / "contextual_rbf.json"
    artifact = json.loads(path.read_text())
    artifact["payload"]["parameters"]["noise_variance"] += .1
    if resign:
        artifact["sha256"] = module._digest(artifact["payload"])
    path.write_text(json.dumps(artifact))
    corrupted = path.read_bytes()
    with pytest.raises(ValueError, match="Invalid contextual RBF calibration cache"):
        fit_contextual_rbf(**arguments(), cache_directory=tmp_path)
    assert path.read_bytes() == corrupted
    assert len(calls) == 1


def test_failed_fit_stops_and_leaves_no_cache(tmp_path, monkeypatch):
    fixed_optimizer(monkeypatch, success=False)
    with pytest.raises(RuntimeError, match="calibration failed.*iteration limit"):
        fit_contextual_rbf(**arguments(), cache_directory=tmp_path)
    assert not (tmp_path / "contextual_rbf.json").exists()


def test_context_support_is_exact_and_gram_must_be_psd():
    args = arguments()
    args["context_coordinates"][0, 0] = .01
    with pytest.raises(ValueError, match="match context_support exactly"):
        fit_contextual_rbf(**args)
    args = arguments()
    args["context_covariance"][0, 0] = -1
    with pytest.raises(ValueError, match="positive semidefinite"):
        fit_contextual_rbf(**args)


def test_real_optimizer_and_cache_roundtrip(tmp_path, monkeypatch):
    fit = fit_contextual_rbf(**arguments(), cache_directory=tmp_path)
    assert fit.diagnostics["optimizer_success"]

    def forbidden(*args, **kwargs):
        pytest.fail("cached calibration must not optimize")

    monkeypatch.setattr(module, "minimize", forbidden)
    cached = fit_contextual_rbf(**arguments(), cache_directory=tmp_path)
    assert cached.parameters == fit.parameters


@pytest.mark.parametrize("arithmetic,conditioning", [("native", "incremental"), ("gpy", "full")])
def test_chunked_marginals_match_dense_joint_and_append_without_candidate_matrix(
    monkeypatch, arithmetic, conditioning
):
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).double()
    gp = basat.FixedKernelGaussianProcess(
        basat.GaussianProcessConfig(kernel=kernel, mean="zero", noise_mode="fixed",
                                   noise_variance=.07, standardize_outcomes=False, dtype="float64"),
        input_dimension=1, arithmetic=arithmetic, conditioning=conditioning,
    )
    train = np.array([[0.], [.4], [.9], [1.6]])
    y = np.array([.2, -.3, .7, .1])
    candidates = np.linspace(-.2, 2., 19)[:, None]
    gp.set_data(train[:3], y[:3], [.07, .12, .03])
    for append in (False, True):
        if append:
            gp.set_data(train, y, [.07, .12, .03, .2])
        expected_mean, expected_std = gp.predict(candidates)
        original_covariance = gp._covariance
        widths = []

        def guarded(x, z):
            assert len(x) == gp._count  # Only train-candidate cross covariance.
            assert len(z) <= 4
            widths.append(len(z))
            return original_covariance(x, z)

        with monkeypatch.context() as local:
            local.setattr(gp, "_covariance", guarded)
            mean, std = gp.predict_marginals(candidates, chunk_size=4)
            noisy_mean, noisy_std = gp.predict_marginals(candidates, chunk_size=4, observation_noise=True)
        assert max(widths) == 4
        np.testing.assert_allclose(mean, expected_mean, atol=3e-14, rtol=0)
        np.testing.assert_allclose(std, expected_std, atol=3e-14, rtol=0)
        np.testing.assert_allclose(noisy_mean, mean, atol=0, rtol=0)
        np.testing.assert_allclose(noisy_std.square() - std.square(), .07, atol=3e-15, rtol=0)


def learner(*, prediction="marginal", recommendation="latest_context"):
    actions = np.array([[0.], [1.], [2.]])
    return basat.CGP_UCB(
        bounds=basat.Bounds(lower=[0.], upper=[2.]), dtype="float64",
        config=basat.CGPUCBConfig(
            action_set=actions,
            action_kernel=gpytorch.kernels.RBFKernel().double(),
            context_kernel=basat.FiniteSupportKernel([[0.], [1.]], np.eye(2)),
            beta=2., gp_backend="fixed_incremental", finite_prediction=prediction,
            marginal_chunk_size=1, recommendation_candidates=recommendation,
            acquisition_optimizer=basat.AcquisitionOptimizerSpec(fit_maxiter=0),
            gp_config=basat.GaussianProcessConfig(
                mean="zero", noise_mode="fixed", noise_variance=.1,
                standardize_outcomes=False, dtype="float64"),
        ),
    )


def observed_first_round(algorithm):
    algorithm.initialize_from_environment(basat.EnvironmentData(
        actions=torch.empty((0, 1), dtype=torch.float64),
        observations=torch.empty(0, dtype=torch.float64)))
    info = dict(context=[1.], candidate_actions=[[1.], [2.]], candidate_dataset_indices=[1, 2])
    action = algorithm.ask(round_info=info)
    algorithm.tell(action=action, feedback=dict(
        observation=-1., context=[1.], evaluated_action=action, observation_noise_variance=.1))
    return action


def test_latest_context_recommendation_masks_missing_pairs_and_survives_checkpoint():
    algorithm = learner()
    observed_first_round(algorithm)
    recommended, value = algorithm._recommendation_impl()
    assert recommended.tolist() == [2.]
    restored = learner()
    restored.load_state_dict(algorithm.state_dict())
    restored_action, restored_value = restored._recommendation_impl()
    assert torch.equal(restored_action, recommended)
    assert restored_value == pytest.approx(value, abs=1e-14)
    np.testing.assert_array_equal(restored.state.latest_context_candidates, [[1.], [2.]])
    original = learner(recommendation="configured")
    observed_first_round(original)
    assert original._recommendation_impl()[0].tolist() == [0.]


def test_finite_marginal_ucb_matches_joint_without_calling_joint_prediction(monkeypatch):
    marginal, joint = learner(), learner(prediction="joint")
    for algorithm in (marginal, joint):
        observed_first_round(algorithm)

    def forbidden(*args, **kwargs):
        pytest.fail("marginal policy must not form the full candidate posterior")

    monkeypatch.setattr(marginal.gp, "predict", forbidden)
    info = dict(context=[1.], candidate_actions=[[1.], [2.]], candidate_dataset_indices=[1, 2])
    assert torch.equal(marginal.ask(round_info=info), joint.ask(round_info=info))
    assert marginal._last_acquisition_value == pytest.approx(joint._last_acquisition_value, abs=1e-14)


def test_latest_context_recommendation_retains_previously_queried_actions_only_in_that_context():
    algorithm = learner()
    observed_first_round(algorithm)
    action = algorithm.ask(round_info=dict(
        context=[1.], candidate_actions=[[2.]], candidate_dataset_indices=[2]))
    algorithm.tell(action=action, feedback=dict(
        observation=-3., context=[1.], evaluated_action=action, observation_noise_variance=.1))
    assert algorithm._recommendation_impl()[0].tolist() == [1.]
    action = algorithm.ask(round_info=dict(
        context=[0.], candidate_actions=[[0.]], candidate_dataset_indices=[0]))
    algorithm.tell(action=action, feedback=dict(
        observation=-5., context=[0.], evaluated_action=action, observation_noise_variance=.1))
    assert algorithm._recommendation_impl()[0].tolist() == [0.]
    np.testing.assert_array_equal(algorithm.state.latest_context_candidates, [[0.]])


def test_new_options_are_explicit_and_old_checkpoint_config_defaults_remain_defined():
    defaults = basat.CGPUCBConfig(action_set=[[0.], [1.]])
    assert defaults.finite_prediction == "joint"
    assert defaults.recommendation_candidates == "configured"
    assert basat.CGP_UCB.CHECKPOINT_ADDED_CONFIG_DEFAULTS["finite_prediction"] == "joint"
    with pytest.raises(ValueError, match="requires fixed_incremental"):
        basat.CGP_UCB(bounds=basat.Bounds(lower=[0.], upper=[1.]),
                      config=replace(defaults, finite_prediction="marginal"))
