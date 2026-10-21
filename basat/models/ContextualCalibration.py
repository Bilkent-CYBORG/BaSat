"""Exact offline RBF calibration from irregular action-context observations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gpytorch
import numpy as np
import scipy
import scipy.linalg as la
import torch
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from .FixedKernelGaussianProcess import FiniteSupportKernel

_CACHE_NAME = "contextual_rbf.json"
_CACHE_PROTOCOL = "basat-contextual-rbf-v1"
_PARAMETER_NAMES = ("action_lengthscale", "signal_variance", "noise_variance")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _array_identity(value: np.ndarray) -> dict[str, Any]:
    canonical = np.ascontiguousarray(value, dtype="<f8")
    return {"shape": list(canonical.shape), "dtype": "float64",
            "sha256": hashlib.sha256(canonical.tobytes()).hexdigest()}


def _negative_log_likelihood(
    log_parameters: np.ndarray, squared_distances: np.ndarray,
    context_gram: np.ndarray, targets: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return the exact zero-mean NLL and analytic log-parameter gradient."""
    lengthscale, signal, noise = np.exp(log_parameters)
    scaled_distances = squared_distances / lengthscale**2
    latent = signal * np.exp(-0.5 * scaled_distances) * context_gram
    covariance = latent.copy()
    covariance.flat[:: len(targets) + 1] += noise
    factor = la.cho_factor(covariance, lower=True, check_finite=False)
    alpha = la.cho_solve(factor, targets, check_finite=False)
    value = 0.5 * float(targets @ alpha + 2 * np.log(np.diag(factor[0])).sum()
                        + len(targets) * np.log(2 * np.pi))
    weight = la.cho_solve(factor, np.eye(len(targets)), check_finite=False)
    weight -= np.outer(alpha, alpha)
    gradient = 0.5 * np.array([
        np.sum(weight * latent * scaled_distances),
        np.sum(weight * latent), noise * np.trace(weight),
    ])
    return value, gradient


@dataclass(frozen=True)
class ContextualRBFCalibration:
    """Fixed covariance factors calibrated without retaining training rows.

    The action factor is an isotropic RBF including signal variance once.
    The supplied finite context covariance is unchanged. Observation offset
    is always zero: outcomes are neither centered nor standardized.
    """

    action_kernel: gpytorch.kernels.Kernel
    context_kernel: FiniteSupportKernel
    noise_variance: float
    observation_offset: float
    parameters: dict[str, float]
    diagnostics: dict[str, Any]


def fit_contextual_rbf(
    observations: Any,
    *,
    action_coordinates: Any,
    context_coordinates: Any,
    context_support: Any,
    context_covariance: Any,
    initial_action_lengthscale: float = 1.0,
    initial_signal_variance: float | None = None,
    initial_noise_variance: float = 0.1,
    action_lengthscale_bounds: tuple[float, float] = (1e-3, 1e3),
    signal_variance_bounds: tuple[float, float] = (1e-4, 100.0),
    noise_variance_bounds: tuple[float, float] = (1e-6, 100.0),
    maxiter: int = 200,
    ftol: float = 1e-10,
    maxls: int = 40,
    cache_directory: str | Path | None = None,
) -> ContextualRBFCalibration:
    """Fit three positive GP parameters by exact marginal likelihood.

    ``observations`` is a length-N vector; ``action_coordinates`` and
    ``context_coordinates`` are aligned (N,D) and (N,C) tables. Repeated
    actions or contexts and incomplete action-context grids are supported.
    Every context must match one unique row of ``context_support`` exactly.
    ``context_covariance`` is its fixed symmetric PSD covariance matrix.

    The caller selects calibration observations before calling this helper.
    No sampling, response transformation, posterior warm start, randomized
    restart, or synthetic noise occurs here. The mean is exactly zero.
    Initial signal variance defaults to the supplied observations' population
    variance; it must lie within the explicit optimization bounds. The three
    log parameters are action lengthscale, signal variance, and noise variance.

    A successful fit may be saved atomically under ``cache_directory``. Its
    fingerprint covers effective float64 inputs and all optimizer settings.
    A mismatched or corrupt cache raises ValueError; unsuccessful optimization
    raises RuntimeError and is never cached. Cached parameters and likelihood
    are revalidated without running the optimizer.
    """
    y = np.asarray(observations, dtype=np.float64)
    x = np.asarray(action_coordinates, dtype=np.float64)
    contexts = np.asarray(context_coordinates, dtype=np.float64)
    support = np.asarray(context_support, dtype=np.float64)
    gram = np.asarray(context_covariance, dtype=np.float64)
    if (y.ndim != 1 or not len(y) or x.ndim != 2 or contexts.ndim != 2
            or x.shape[0] != len(y) or contexts.shape[0] != len(y)
            or not x.shape[1] or not contexts.shape[1]
            or not all(np.isfinite(a).all() for a in (y, x, contexts))):
        raise ValueError("Supply finite aligned observations (N,), actions (N,D), and contexts (N,C).")
    if support.ndim != 2 or support.shape[1] != contexts.shape[1]:
        raise ValueError("context_support must have the same context dimension.")
    # Reuse the public kernel's exact support, symmetry, and PSD checks.
    context_kernel = FiniteSupportKernel(support, gram)
    matches = (contexts[:, None, :] == support[None, :, :]).all(axis=-1)
    if not matches.any(axis=1).all():
        raise ValueError("Every calibration context must match context_support exactly.")
    indices = matches.argmax(axis=1)
    paired_gram = gram[np.ix_(indices, indices)]
    if (isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 1
            or isinstance(maxls, bool) or not isinstance(maxls, int) or maxls < 1
            or isinstance(ftol, bool) or not np.isfinite(ftol) or ftol <= 0):
        raise ValueError("Optimizer limits and ftol must be positive.")
    initial = np.asarray([
        initial_action_lengthscale,
        float(y.var(ddof=0)) if initial_signal_variance is None else initial_signal_variance,
        initial_noise_variance,
    ], dtype=np.float64)
    bounds = np.asarray([action_lengthscale_bounds, signal_variance_bounds,
                         noise_variance_bounds], dtype=np.float64)
    if (initial.shape != (3,) or bounds.shape != (3, 2)
            or not np.isfinite(initial).all() or not np.isfinite(bounds).all()
            or (initial <= 0).any() or (bounds[:, 0] <= 0).any()
            or (bounds[:, 1] <= bounds[:, 0]).any()
            or (initial < bounds[:, 0]).any() or (initial > bounds[:, 1]).any()):
        raise ValueError("Initial parameters must be positive and inside ordered positive bounds.")
    identity = {
        "protocol": _CACHE_PROTOCOL,
        "inputs": {name: _array_identity(value) for name, value in (
            ("observations", y), ("action_coordinates", x),
            ("context_coordinates", contexts), ("context_support", support),
            ("context_covariance", gram),
        )},
        "settings": {"initial_parameters": initial.tolist(), "bounds": bounds.tolist(),
                     "maxiter": maxiter, "ftol": float(ftol), "maxls": maxls,
                     "mean": "zero", "outcome_transform": "none"},
    }
    fingerprint = _digest(identity)
    squared_distances = cdist(x, x, metric="sqeuclidean")

    def objective(log_parameters: np.ndarray) -> tuple[float, np.ndarray]:
        return _negative_log_likelihood(log_parameters, squared_distances, paired_gram, y)

    cache = None
    if cache_directory is not None:
        from ..workflow import ArtifactStore
        cache = ArtifactStore(str(cache_directory))
    if cache is not None and cache.exists(_CACHE_NAME):
        try:
            artifact = cache.read_json(_CACHE_NAME)
            payload = artifact["payload"]
            if artifact["sha256"] != _digest(payload):
                raise ValueError("artifact checksum does not match")
            if payload["fingerprint"] != fingerprint or payload["identity"] != identity:
                raise ValueError("input/configuration fingerprint does not match; use a fresh directory")
            parameters, diagnostics = payload["parameters"], payload["diagnostics"]
            if set(parameters) != set(_PARAMETER_NAMES) or diagnostics["optimizer_success"] is not True:
                raise ValueError("cached parameters or optimizer status are invalid")
            fitted = np.asarray([parameters[name] for name in _PARAMETER_NAMES], dtype=float)
            if (not np.isfinite(fitted).all() or (fitted <= 0).any()
                    or (fitted < bounds[:, 0] * (1 - 1e-12)).any()
                    or (fitted > bounds[:, 1] * (1 + 1e-12)).any()):
                raise ValueError("cached parameters are outside configured bounds")
            with threadpool_limits(limits=1):
                nll, _ = objective(np.log(fitted))
            if not np.isclose(nll, diagnostics["negative_log_marginal_likelihood"], rtol=1e-9, atol=1e-9):
                raise ValueError("cached likelihood does not match calibration inputs")
            diagnostics = {**diagnostics, "cache_hit": True}
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError, OSError) as error:
            raise ValueError(f"Invalid contextual RBF calibration cache: {error}") from error
    else:
        with threadpool_limits(limits=1):
            fit = minimize(objective, np.log(initial), jac=True, method="L-BFGS-B",
                           bounds=np.log(bounds),
                           options={"maxiter": maxiter, "ftol": ftol, "maxls": maxls})
        if not fit.success or not np.isfinite(fit.fun) or not np.isfinite(fit.x).all():
            raise RuntimeError(f"Contextual RBF calibration failed: {fit.message}")
        fitted = np.exp(fit.x)
        parameters = dict(zip(_PARAMETER_NAMES, map(float, fitted), strict=True))
        boundary_parameters = [
            {"parameter": name, "side": side, "value": float(value), "bound": float(bound)}
            for name, value, (lo, hi) in zip(_PARAMETER_NAMES, fitted, bounds, strict=True)
            for side, bound in (("lower", lo), ("upper", hi))
            if abs(float(np.log(value / bound))) <= 1e-6
        ]
        diagnostics = {
            "optimizer_success": True, "optimizer_message": str(fit.message),
            "iterations": int(fit.nit), "negative_log_marginal_likelihood": float(fit.fun),
            "calibration_rows": len(y), "initial_parameters": initial.tolist(),
            "parameter_bounds": bounds.tolist(), "maxiter": maxiter, "ftol": float(ftol),
            "maxls": maxls, "numerical_method": "exact Cholesky; analytic log-parameter gradient",
            "mean": "zero", "outcome_transform": "none", "boundary_log_tolerance": 1e-6,
            "boundary_parameters": boundary_parameters, "cache_hit": False,
            "cache_fingerprint": fingerprint, "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        }
        if cache is not None:
            payload = {"identity": identity, "fingerprint": fingerprint,
                       "parameters": parameters, "diagnostics": diagnostics}
            cache.write_json(_CACHE_NAME, {"payload": payload, "sha256": _digest(payload)})
    action_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).double()
    action_kernel.base_kernel.initialize(lengthscale=torch.tensor(fitted[0], dtype=torch.float64))
    action_kernel.initialize(outputscale=torch.tensor(fitted[1], dtype=torch.float64))
    action_kernel.requires_grad_(False)
    return ContextualRBFCalibration(action_kernel, context_kernel, float(fitted[2]),
                                   0.0, parameters, diagnostics)
