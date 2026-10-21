"""Atomic calibration reuse must not change GP mathematics or mask bad fits."""

import hashlib
import importlib
import json

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from basat import fit_separable_matern

module = importlib.import_module("basat.models.SeparableCalibration")


def arguments(context_model="matern"):
    return dict(
        observations=np.array([[0.2, 1.1, -0.4], [0.4, 1.3, -0.2], [0.1, 0.9, -0.5],
                               [0.7, 1.6, 0.1]]),
        action_coordinates=np.array([[0.0], [0.8], [1.5]]),
        context_coordinates=np.array([[0.0], [0.5], [1.0], [1.5]]),
        context_support=np.array([[0.0], [0.25], [0.5], [0.75]]),
        context_model=context_model,
        initial_action_lengthscales=[0.7], initial_context_lengthscale=0.8,
        initial_signal_variance=1.2, initial_noise_variance=0.1,
        action_lengthscale_bounds=(0.1, 4.0), context_lengthscale_bounds=(0.1, 4.0),
        signal_variance_bounds=(0.01, 10.0), noise_variance_bounds=(0.001, 1.0),
        maxiter=150,
    )


def fixed_fit(monkeypatch, *, success=True, lower_boundary=False):
    """Keep fixed numerical inputs, independently testing the objective at them."""
    calls = []

    def minimize(fun, x0, *, method, bounds, options):
        x = np.array(x0, copy=True)
        if lower_boundary:
            x[0] = bounds[0][0]
        calls.append((method, options))
        return OptimizeResult(x=x, fun=fun(x), success=success, nit=3,
                              message="fixture converged" if success else "iteration limit")

    monkeypatch.setattr(module, "minimize", minimize)
    return calls


@pytest.mark.parametrize("context_model", ["matern", "independent", "pooled"])
def test_cached_reload_has_no_fit_and_same_covariance(tmp_path, monkeypatch, context_model):
    calls = fixed_fit(monkeypatch)
    kwargs = arguments(context_model)
    first = fit_separable_matern(**kwargs, cache_directory=tmp_path)
    second = fit_separable_matern(**kwargs, cache_directory=tmp_path)
    assert len(calls) == 1
    assert not first.diagnostics["cache_hit"]
    assert second.diagnostics["cache_hit"]
    assert first.parameters == second.parameters
    assert first.observation_offset == second.observation_offset
    assert first.noise_variance == second.noise_variance
    np.testing.assert_array_equal(first.action_kernel.covariance, second.action_kernel.covariance)
    np.testing.assert_array_equal(first.context_kernel.covariance, second.context_kernel.covariance)
    assert first.diagnostics["cache_fingerprint"] == second.diagnostics["cache_fingerprint"]


@pytest.mark.parametrize("change", ["observations", "action_coordinates", "context_coordinates",
                                    "context_support", "center", "maxiter", "ftol",
                                    "initial_context_lengthscale", "signal_variance_bounds"])
def test_cache_rejects_changed_inputs_or_settings(tmp_path, monkeypatch, change):
    calls = fixed_fit(monkeypatch)
    kwargs = arguments()
    fit_separable_matern(**kwargs, cache_directory=tmp_path)
    before = (tmp_path / "separable_matern.json").read_bytes()
    if change in {"observations", "action_coordinates", "context_coordinates", "context_support"}:
        kwargs[change] = kwargs[change].copy()
        kwargs[change].flat[0] += 0.01
    elif change == "center":
        kwargs[change] = False
    elif change == "maxiter":
        kwargs[change] = 151
    elif change == "ftol":
        kwargs[change] = 1e-9
    elif change == "initial_context_lengthscale":
        kwargs[change] = 0.9
    else:
        kwargs[change] = (0.01, 20.0)
    with pytest.raises(ValueError, match="fingerprint does not match"):
        fit_separable_matern(**kwargs, cache_directory=tmp_path)
    assert len(calls) == 1
    assert (tmp_path / "separable_matern.json").read_bytes() == before


@pytest.mark.parametrize("resign", [False, True])
def test_cache_rejects_corrupted_covariance(tmp_path, monkeypatch, resign):
    calls = fixed_fit(monkeypatch)
    kwargs = arguments()
    fit_separable_matern(**kwargs, cache_directory=tmp_path)
    path = tmp_path / "separable_matern.json"
    artifact = json.loads(path.read_text())
    artifact["payload"]["action_covariance"][0][0] += 0.1
    if resign:
        artifact["sha256"] = hashlib.sha256(json.dumps(
            artifact["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="Invalid separable calibration cache"):
        fit_separable_matern(**kwargs, cache_directory=tmp_path)
    assert len(calls) == 1


def test_failed_fits_are_not_cached(tmp_path, monkeypatch):
    calls = fixed_fit(monkeypatch, success=False)
    fit = fit_separable_matern(**arguments(), cache_directory=tmp_path)
    assert not fit.diagnostics["optimizer_success"]
    assert not (tmp_path / "separable_matern.json").exists()
    fit_separable_matern(**arguments(), cache_directory=tmp_path)
    assert len(calls) == 2


def test_uncached_behavior_and_boundary_diagnostics(monkeypatch):
    calls = fixed_fit(monkeypatch, lower_boundary=True)
    fit = fit_separable_matern(**arguments())
    fit_separable_matern(**arguments())
    assert len(calls) == 2
    assert "cache_hit" not in fit.diagnostics
    assert fit.diagnostics["boundary_parameters"] == [dict(
        parameter="action_lengthscale[0]", side="lower", value=pytest.approx(0.1), bound=0.1,
    )]


@pytest.mark.parametrize("context_model", ["matern", "independent", "pooled"])
def test_kronecker_nll_matches_dense_gaussian(monkeypatch, context_model):
    fixed_fit(monkeypatch)
    kwargs = arguments(context_model)
    fit = fit_separable_matern(**kwargs)
    covariance = np.kron(fit.context_kernel.covariance, fit.action_kernel.covariance)
    covariance += fit.noise_variance * np.eye(covariance.shape[0])
    centered = (kwargs["observations"] - kwargs["observations"].mean()).ravel()
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign == 1
    expected = 0.5 * (centered @ np.linalg.solve(covariance, centered) + logdet
                      + len(centered) * np.log(2 * np.pi))
    assert fit.diagnostics["negative_log_marginal_likelihood"] == pytest.approx(expected, abs=1e-11)


def test_real_optimizer_cache_round_trip(tmp_path, monkeypatch):
    first = fit_separable_matern(**arguments(), cache_directory=tmp_path)
    assert first.diagnostics["optimizer_success"], first.diagnostics

    def forbidden(*args, **kwargs):
        pytest.fail("cached calibration must not run an optimizer")

    monkeypatch.setattr(module, "minimize", forbidden)
    second = fit_separable_matern(**arguments(), cache_directory=tmp_path)
    assert first.parameters == second.parameters
    assert second.diagnostics["cache_hit"]


def test_malformed_cache_is_not_replaced(tmp_path, monkeypatch):
    calls = fixed_fit(monkeypatch)
    path = tmp_path / "separable_matern.json"
    path.write_text("{ incomplete")
    with pytest.raises(ValueError, match="Invalid separable calibration cache"):
        fit_separable_matern(**arguments(), cache_directory=tmp_path)
    assert not calls
    assert path.read_text() == "{ incomplete"
