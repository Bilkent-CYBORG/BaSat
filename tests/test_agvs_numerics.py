"""Independent posterior and proposal checks for the AG-VS source policy."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import gpytorch
import numpy as np
import pytest
import torch
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from basat.algorithms.Stochastic._agvs_source import (
    SourceBOGP,
    classical_ei,
    latin_hypercube,
    propose_bo,
)
from basat.algorithms.Stochastic._agvs_turbo import (
    SourceTurboGP,
    TurboProposalState,
    propose_turbo,
    update_turbo,
)


@pytest.fixture
def data():
    X = np.array([[.1, .2], [.3, .7], [.55, .4], [.8, .1], [.9, .8]])
    y = np.array([2.8, 3.4, 3.1, 2.5, 3.7])
    return X, y, np.tile([0., 1.], (2, 1))


@pytest.mark.parametrize("learn_noise", [False, True])
def test_bo_matches_independent_sklearn_raw_scale_reference(data, learn_noise):
    X, y, _ = data
    kernel = ConstantKernel(1.) * Matern(length_scale=1., nu=2.5)
    if learn_noise:
        kernel += WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 10.))
    reference = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-10 if learn_noise else 1e-4,
        normalize_y=False, n_restarts_optimizer=0, random_state=2021,
    ).fit(X, y)
    actual = SourceBOGP(learn_noise=learn_noise).fit(X, y)
    query = np.vstack([X, [[.15, .65], [.5, .5]]])
    for left, right in zip(actual.predict_numpy(query, return_std=True), reference.predict(query, return_std=True), strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(actual.model.kernel_.theta, reference.kernel_.theta)
    # Bounds are those of sklearn's raw-scale signal covariance, rather than
    # the separate bounded standardized TuRBO model.
    np.testing.assert_allclose(np.exp(actual.model.kernel_.bounds[:2]), np.tile([1e-5, 1e5], (2, 1)))


def test_bo_checkpoint_preserves_fitted_factors_without_fit_or_rng(data, monkeypatch):
    X, y, _ = data
    fitted = SourceBOGP().fit(X, y)
    state = fitted.state_dict()
    expected = fitted.predict_numpy(X, return_cov=True)
    monkeypatch.setattr(GaussianProcessRegressor, "fit", lambda *args, **kwargs: pytest.fail("Restore fitted an already-fitted GP."))
    restored = SourceBOGP.from_state_dict(state)
    for actual, reference in zip(restored.predict_numpy(X, return_cov=True), expected, strict=True):
        np.testing.assert_array_equal(actual, reference)


def test_warm_incremental_gp_matches_full_fixed_kernel_conditioning(data):
    X, y, _ = data
    initial = SourceBOGP().fit(X[:3], y[:3])
    extended = SourceBOGP().fit(X, y, warm_model=initial, hold_hyperparameters=True)
    reference = GaussianProcessRegressor(kernel=initial.model.kernel_, alpha=1e-4, optimizer=None).fit(X, y)
    assert extended.fit_mode == "incremental_fixed_kernel"
    for actual, expected in zip(extended.predict_numpy(X, return_std=True), reference.predict(X, return_std=True), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_lhs_full_design_is_stratified_and_uses_advancing_generator():
    bounds = np.tile([-2., 3.], (3, 1))
    generator = np.random.default_rng(2021)
    actual = latin_hypercube(bounds, 12, generator)
    reference_rng = np.random.default_rng(2021)
    columns = []
    for _ in range(3):
        columns.append((reference_rng.permutation(12) + reference_rng.random(12)) / 12.)
    np.testing.assert_array_equal(actual, -2. + 5. * np.column_stack(columns))
    for column in actual.T:
        np.testing.assert_array_equal(np.sort(np.floor((column + 2.) / 5. * 12.)), np.arange(12))
    assert generator.integers(0, 500000) == reference_rng.integers(0, 500000)


def test_ei_uses_posterior_incumbent_and_source_offset(data):
    X, y, _ = data
    model = SourceBOGP(noise_variance=.4).fit(X, y)
    means, stds = model.predict_numpy(X, return_std=True)
    incumbent = model.predict_numpy(X).max()
    assert abs(incumbent - y.max()) > 1e-3
    improvement = means - incumbent - 1e-4
    expected = improvement * norm.cdf(improvement / stds) + stds * norm.pdf(improvement / stds)
    np.testing.assert_allclose(classical_ei(model, X), expected, rtol=1e-14)


def test_independent_ei_uses_exact_1024_sobol_top_k_order(data):
    X, y, bounds = data
    model = SourceBOGP().fit(X, y)
    generator = np.random.default_rng(2021)
    actual, values = propose_bo(model, bounds, 3, generator)
    reference_rng = np.random.default_rng(2021)
    seed = int(reference_rng.integers(0, 500000))
    pool = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed).draw(1024).double().numpy()
    scores = classical_ei(model, pool)
    selected = np.argsort(scores)[-3:]
    np.testing.assert_array_equal(actual, pool[selected])
    np.testing.assert_array_equal(values, scores[selected])
    assert generator.integers(0, 500000) == reference_rng.integers(0, 500000)


@pytest.mark.parametrize("learn_noise", [False, True])
def test_turbo_uses_one_isotropic_gp_and_correct_noise_policy(data, learn_noise):
    X, y, bounds = data
    model = SourceTurboGP(bounds=bounds, maxiter=5, learn_noise=learn_noise).fit(X, y)
    gp = model.gp
    assert gp.covar_module.base_kernel.ard_num_dims is None
    assert gp.covar_module.base_kernel.lengthscale.numel() == 1
    constraint = gp.covar_module.base_kernel.raw_lengthscale_constraint
    assert float(constraint.lower_bound) == pytest.approx(.005)
    assert float(constraint.upper_bound) == pytest.approx(np.sqrt(2))
    raw_noise = gp.likelihood.noise_covar.raw_noise
    assert raw_noise.requires_grad is learn_noise
    if not learn_noise:
        assert model.fitted_noise_variance == pytest.approx(1e-4, rel=1e-12)
    query = np.array([[.2, .4], [.6, .7]])
    parameter = next(gp.parameters())
    with torch.no_grad():
        latent = gp(torch.as_tensor(query, dtype=parameter.dtype))
        expected_variance = (latent.variance + gp.likelihood.noise).numpy() * model.target_scale**2
    _, actual_std = model.predict_numpy(query, return_std=True)
    np.testing.assert_allclose(actual_std**2, expected_variance, rtol=1e-10)


def test_turbo_checkpoint_and_proposal_preserve_rng_ownership(data):
    X, y, bounds = data
    model = SourceTurboGP(bounds=bounds, maxiter=5).fit(X, y)
    restored = SourceTurboGP.from_state_dict(model.state_dict())
    global_state = torch.random.get_rng_state().clone()
    g1, g2 = torch.Generator().manual_seed(2021), torch.Generator().manual_seed(2021)
    first = propose_turbo(model, bounds, 3, np.random.RandomState(2021), g1, {}, TurboProposalState())
    second = propose_turbo(restored, bounds, 3, np.random.RandomState(2021), g2, {}, TurboProposalState())
    for actual, expected in zip(first, second, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert torch.equal(g1.get_state(), g2.get_state())
    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert not torch.equal(g1.get_state(), torch.Generator().manual_seed(2021).get_state())


@pytest.mark.parametrize("initial", [TurboProposalState(n_evals=48), TurboProposalState(length=.5**8)])
def test_turbo_ceiling_and_collapse_preserve_training_without_lhs(data, initial):
    X, y, bounds = data
    model = SourceTurboGP(bounds=bounds, maxiter=3).fit(X, y)
    original_model = model.gp
    state = copy.deepcopy(initial)
    propose_turbo(model, bounds, 3, np.random.RandomState(2), torch.Generator().manual_seed(2), {}, state)
    assert state.restart_count == 1
    assert state.n_evals == 0
    assert state.length == .8
    assert model.gp is original_model
    np.testing.assert_array_equal(model.X_train_, X)
    np.testing.assert_array_equal(model.y_train_, y)
    update_turbo(state, float(y.max()), [y.max(), y.max() - 1., y.max() - 2.], 2, 3)
    assert state.n_evals == 3


def test_turbo_numerical_retry_resets_warm_model_and_propagates_other_errors(data, monkeypatch):
    X, y, bounds = data
    previous = SourceTurboGP(bounds=bounds, maxiter=1).fit(X, y)
    model = SourceTurboGP(bounds=bounds, maxiter=1)
    attempts = []
    def fail_then_recover(inputs, targets, *, warm_model, jitter):
        attempts.append((inputs.dtype, warm_model, jitter))
        if len(attempts) < 3:
            raise RuntimeError("cholesky factorization failed")
        return previous.gp
    monkeypatch.setattr(model, "_train_once", fail_then_recover)
    model.fit(X, y, warm_model=previous)
    assert attempts == [(torch.float64, previous, 0.), (torch.float64, None, 0.), (torch.float64, None, .001)]
    assert model.cholesky_jitter == .001
    def unrelated(*args, **kwargs):
        raise ValueError("invalid configuration")
    monkeypatch.setattr(model, "_train_once", unrelated)
    with pytest.raises(ValueError, match="invalid configuration"):
        model.fit(X, y)


def test_turbo_fit_and_predict_ignore_ambient_botorch_defaults_and_restore_them():
    X = np.random.default_rng(2021).random((15, 30))
    y = np.sin(X[:, :3]).sum(axis=1)
    bounds = np.tile([0., 1.], (30, 1))
    models = []
    predictions = []
    for ambient in (False, True):
        with gpytorch.settings.fast_computations(
            covar_root_decomposition=ambient, log_prob=ambient, solves=ambient
        ), gpytorch.settings.max_cholesky_size(1), gpytorch.settings.max_eager_kernel_size(1), gpytorch.settings.cholesky_max_tries(7), gpytorch.settings.max_root_decomposition_size(7):
            model = SourceTurboGP(bounds=bounds, maxiter=5).fit(X, y)
            predictions.append(model.predict_numpy(X, return_std=True))
            models.append(model)
            assert gpytorch.settings.fast_computations.log_prob.on() is ambient
            assert gpytorch.settings.fast_computations.solves.on() is ambient
            assert gpytorch.settings.fast_computations.covar_root_decomposition.on() is ambient
            assert gpytorch.settings.max_cholesky_size.value() == 1
            assert gpytorch.settings.max_eager_kernel_size.value() == 1
            assert gpytorch.settings.cholesky_max_tries.value() == 7
            assert gpytorch.settings.max_root_decomposition_size.value() == 7
    for name, value in models[0].gp.state_dict().items():
        assert torch.equal(value, models[1].gp.state_dict()[name])
    for actual, expected in zip(*predictions, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert models[0].gp.train_inputs[0].stride() == (1, len(X))


def test_turbo_parameters_match_unmodified_zip_before_botorch_import():
    """A separate source process catches changes masked by shared imports.

    The supplied source's advanced column projection is Fortran-contiguous.
    The training layout and native GPyTorch defaults both affect floating-point
    GP parameters, even below the Cholesky threshold.
    """
    archive = Path(os.environ.get(
        "BASAT_AGVS_REFERENCE_ZIP",
        str(Path(__file__).resolve().parents[2] / "AISTATS2027_Anonymous_Code.zip"),
    ))
    if not archive.is_file():
        pytest.skip("The optional user-provided AG-VS source ZIP is unavailable.")
    X = np.random.default_rng(2021).random((15, 300))
    y = np.sin(X[:, :3]).sum(axis=1)
    source = r'''
import json, sys, types
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0, sys.argv[1] + "/AISTATS2027_Anonymous_Code")
from ag_vs.inner_optimizer.turbo.proposal import fit_turbo_node_gp
values = json.load(sys.stdin)
X, y = np.asarray(values["X"]), np.asarray(values["y"])
dimension = X.shape[1]
optimizer = types.SimpleNamespace(
    X=list(X), y=list(y), noise_variance=1e-4, learn_noise=False,
    turbo_min_cuda=1024, turbo_device="cpu",
    _node_bounds=lambda node: np.tile([0., 1.], (dimension, 1)))
node = types.SimpleNamespace(dims=list(range(dimension)), local_indices=list(range(len(X))))
model = fit_turbo_node_gp(optimizer, node)
means, stds = model.predict(X[:, list(range(dimension))], return_std=True)
print(json.dumps({
    "parameters": {key: value.detach().cpu().tolist() for key, value in model.gp.state_dict().items()},
    "stride": list(model.gp.train_inputs[0].stride()),
    "means": means.tolist(), "stds": stds.tolist()}))
'''
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[name] = "1"
    process = subprocess.run(
        [sys.executable, "-c", source, str(archive)], input=json.dumps({"X": X.tolist(), "y": y.tolist()}),
        text=True, capture_output=True, env=environment, check=True,
    )
    expected = json.loads(process.stdout)
    model = SourceTurboGP(bounds=np.tile([0., 1.], (300, 1))).fit(X, y)
    assert list(model.gp.train_inputs[0].stride()) == expected["stride"]
    for name, value in model.gp.state_dict().items():
        np.testing.assert_array_equal(value.detach().cpu().numpy(), expected["parameters"][name])
    mean, std = model.predict_numpy(X, return_std=True)
    np.testing.assert_array_equal(mean, expected["means"])
    np.testing.assert_array_equal(std, expected["stds"])
