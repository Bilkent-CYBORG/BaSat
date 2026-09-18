"""Offline marginal-likelihood calibration on complete Cartesian grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from .FixedKernelGaussianProcess import FiniteSupportKernel


def _matern52(distance: np.ndarray) -> np.ndarray:
    r = np.sqrt(5.0) * distance
    return (1 + r + r * r / 3) * np.exp(-r)


@dataclass(frozen=True)
class SeparableCalibration:
    """Fixed calibrated covariance factors and scalar observation statistics.

    Attributes
    ----------
    action_kernel, context_kernel : FiniteSupportKernel
        Covariance factors on the supplied finite supports. Signal variance is
        included once, in the action factor.
    noise_variance, observation_offset : float
        Likelihood variance and global calibration-only centering constant.
    parameters, diagnostics : dict
        Fitted parameters and exact optimizer/protocol provenance.
    """

    action_kernel: FiniteSupportKernel
    context_kernel: FiniteSupportKernel
    noise_variance: float
    observation_offset: float
    parameters: dict[str, Any]
    diagnostics: dict[str, Any]


def fit_separable_matern(
    observations: Any,
    *,
    action_coordinates: Any,
    context_coordinates: Any,
    context_support: Any | None = None,
    context_model: Literal["matern", "independent", "pooled"] = "matern",
    initial_action_lengthscales: Any,
    initial_context_lengthscale: float,
    initial_signal_variance: float,
    initial_noise_variance: float,
    action_lengthscale_bounds: tuple[float, float],
    context_lengthscale_bounds: tuple[float, float],
    signal_variance_bounds: tuple[float, float],
    noise_variance_bounds: tuple[float, float],
    center: bool = True,
    maxiter: int = 150,
    ftol: float = 1e-10,
    maxls: int = 40,
) -> SeparableCalibration:
    """Fit a Matérn-5/2 grid GP by exact Gaussian marginal likelihood.

    Parameters
    ----------
    observations : array-like, shape (T, M)
        Complete calibration-only response table. No online data are needed.
    action_coordinates : array-like, shape (M, D)
        Physical action coordinates, with one learned lengthscale per column.
    context_coordinates : array-like, shape (T, C)
        Coordinates used for context distance. A circular embedding may be
        supplied explicitly by the user when contexts are periodic.
    context_support : array-like, optional
        Learner-visible context coordinates; defaults to context_coordinates.
        It must have T unique rows, in the same order as observations.
    context_model : {'matern', 'independent', 'pooled'}
        Matérn temporal factor, identity factor (independent context GPs), or
        all-ones factor (a single context-free GP). Each call fits its own
        spatial covariance and noise from the same calibration table.
    initial_action_lengthscales, initial_context_lengthscale,
    initial_signal_variance, initial_noise_variance : numeric
        Positive optimizer initialization in physical units. The unused
        context lengthscale is not optimized for independent/pooled models.
    action_lengthscale_bounds, context_lengthscale_bounds,
    signal_variance_bounds, noise_variance_bounds : (float, float)
        Explicit positive lower and upper optimization bounds.
    center : bool
        Subtract only the global mean of the supplied calibration table.
    maxiter, ftol, maxls : numeric
        L-BFGS-B settings. No randomized restarts or outcome-based tuning.

    Returns
    -------
    SeparableCalibration
        Fixed kernel factors, noise, offset, and optimizer diagnostics. The
        likelihood is evaluated with an exact Kronecker eigendecomposition,
        not a low-rank approximation. No posterior observations are retained.
    """
    y = np.asarray(observations, dtype=float)
    x = np.asarray(action_coordinates, dtype=float)
    t = np.asarray(context_coordinates, dtype=float)
    support = t if context_support is None else np.asarray(context_support, dtype=float)
    if (
        x.ndim != 2
        or t.ndim != 2
        or not len(x)
        or not len(t)
        or not x.shape[1]
        or not t.shape[1]
        or y.shape != (len(t), len(x))
    ):
        raise ValueError("observations must have shape (contexts, actions).")
    if (
        not isinstance(center, bool)
        or isinstance(maxiter, bool)
        or not isinstance(maxiter, int)
        or maxiter < 1
        or isinstance(maxls, bool)
        or not isinstance(maxls, int)
        or maxls < 1
        or not np.isfinite(ftol)
        or ftol <= 0
    ):
        raise ValueError("center must be Boolean; optimizer limits and ftol must be positive.")
    if (
        support.ndim != 2
        or len(support) != len(t)
        or not all(np.isfinite(a).all() for a in [x, t, support, y])
    ):
        raise ValueError("Calibration tables must have finite matching coordinates.")
    if len(np.unique(support, axis=0)) != len(support):
        raise ValueError("Calibration context_support rows must be unique.")
    if context_model not in {"matern", "independent", "pooled"}:
        raise ValueError("Unknown context covariance model.")
    initial_ls = np.asarray(initial_action_lengthscales, dtype=float).reshape(-1)
    if len(initial_ls) != x.shape[1]:
        raise ValueError("Supply one initial action lengthscale per coordinate.")
    bounds = [action_lengthscale_bounds] * x.shape[1]
    initial = initial_ls.tolist()
    if context_model == "matern":
        initial.append(initial_context_lengthscale)
        bounds.append(context_lengthscale_bounds)
    initial.extend([initial_signal_variance, initial_noise_variance])
    bounds.extend([signal_variance_bounds, noise_variance_bounds])
    if any(not np.isfinite(v) or v <= 0 for v in initial) or any(
        not np.isfinite(b).all() or b[0] <= 0 or b[1] <= b[0] for b in bounds
    ):
        raise ValueError("Initial parameters and ordered bounds must be finite and positive.")
    if any(not lo <= v <= hi for v, (lo, hi) in zip(initial, bounds, strict=True)):
        raise ValueError("Initial parameters must lie within their explicit bounds.")
    offset = float(y.mean()) if center else 0.0
    centered = y - offset
    temporal_distance = cdist(t, t)
    d = x.shape[1]

    def covariance(log_parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        p = np.exp(log_parameters)
        spatial = _matern52(cdist(x / p[:d], x / p[:d]))
        temporal = (
            _matern52(temporal_distance / p[d])
            if context_model == "matern"
            else np.eye(len(t))
            if context_model == "independent"
            else np.ones((len(t), len(t)))
        )
        return spatial, temporal, p[-2], p[-1]

    def nll(log_parameters: np.ndarray) -> float:
        spatial, temporal, amplitude, noise = covariance(log_parameters)
        es, us = la.eigh(spatial, check_finite=False)
        et, ut = la.eigh(temporal, check_finite=False)
        eigenvalues = amplitude * np.maximum(et[:, None], 0) * np.maximum(es[None, :], 0) + noise
        transformed = ut.T @ centered @ us
        return 0.5 * float(
            np.sum(np.log(eigenvalues) + transformed**2 / eigenvalues)
            + centered.size * np.log(2 * np.pi)
        )

    with threadpool_limits(limits=1):
        fit = minimize(
            nll,
            np.log(initial),
            method="L-BFGS-B",
            bounds=np.log(bounds),
            options={"maxiter": maxiter, "ftol": ftol, "maxls": maxls},
        )
    spatial, temporal, amplitude, noise = covariance(fit.x)
    p = np.exp(fit.x)
    parameters = {
        "action_lengthscales": p[:d].tolist(),
        "context_lengthscale": float(p[d]) if context_model == "matern" else None,
        "signal_variance": float(amplitude),
        "noise_variance": float(noise),
    }
    diagnostics = {
        "context_model": context_model,
        "calibration_rows": int(y.size),
        "negative_log_marginal_likelihood": float(fit.fun),
        "optimizer_success": bool(fit.success),
        "optimizer_message": str(fit.message),
        "iterations": int(fit.nit),
        "initial_parameters": initial,
        "parameter_bounds": bounds,
        "maxiter": maxiter,
        "ftol": ftol,
        "maxls": maxls,
        "numerical_method": "exact Kronecker eigendecomposition",
    }
    return SeparableCalibration(
        FiniteSupportKernel(x, amplitude * spatial),
        FiniteSupportKernel(support, temporal),
        float(noise),
        offset,
        parameters,
        diagnostics,
    )
