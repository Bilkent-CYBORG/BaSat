"""Offline marginal-likelihood calibration on complete Cartesian grids."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy
import scipy.linalg as la
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from .FixedKernelGaussianProcess import FiniteSupportKernel

_CACHE_PROTOCOL = "basat-separable-matern-v1"
_CACHE_NAME = "separable_matern.json"


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _array_identity(value: np.ndarray) -> dict[str, Any]:
    """Hash the effective float64 model inputs, independently of host endianness."""
    canonical = np.ascontiguousarray(value, dtype="<f8")
    return {"shape": list(canonical.shape), "dtype": "float64",
            "sha256": hashlib.sha256(canonical.tobytes()).hexdigest()}


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
    cache_directory: str | Path | None = None,
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
    cache_directory : str or Path, optional
        Opt-in directory for one atomic, checksummed JSON calibration artifact.
        A successful cached fit is reused only for identical effective float64
        inputs and scientific/optimizer settings. Changed or corrupt artifacts
        raise ValueError, rather than being overwritten or silently refitted.
        Use a fresh directory for a changed experiment. Unsuccessful fits are
        returned with their diagnostics but are never cached. No posterior
        observations are retained in the artifact.

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
    cache = None
    identity = None
    fingerprint = None
    if cache_directory is not None:
        # Import lazily: workflow's public helpers also import model classes.
        from ..workflow import ArtifactStore

        identity = {
            "protocol": _CACHE_PROTOCOL,
            "inputs": {name: _array_identity(value) for name, value in (
                ("observations", y), ("action_coordinates", x),
                ("context_coordinates", t), ("context_support", support),
            )},
            "settings": {
                "context_model": context_model,
                "initial_action_lengthscales": initial_ls.tolist(),
                "initial_context_lengthscale": float(initial_context_lengthscale),
                "initial_signal_variance": float(initial_signal_variance),
                "initial_noise_variance": float(initial_noise_variance),
                "action_lengthscale_bounds": list(action_lengthscale_bounds),
                "context_lengthscale_bounds": list(context_lengthscale_bounds),
                "signal_variance_bounds": list(signal_variance_bounds),
                "noise_variance_bounds": list(noise_variance_bounds),
                "center": center, "maxiter": maxiter, "ftol": float(ftol), "maxls": maxls,
            },
        }
        fingerprint = _json_digest(identity)
        cache = ArtifactStore(str(cache_directory))
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

    if cache is not None and cache.exists(_CACHE_NAME):
        try:
            artifact = cache.read_json(_CACHE_NAME)
            payload = artifact["payload"]
            if artifact["sha256"] != _json_digest(payload):
                raise ValueError("artifact checksum does not match")
            if payload["fingerprint"] != fingerprint or payload["identity"] != identity:
                raise ValueError("input/configuration fingerprint does not match; use a fresh directory")
            parameters = payload["parameters"]
            diagnostics = payload["diagnostics"]
            expected_keys = {"action_lengthscales", "context_lengthscale", "signal_variance", "noise_variance"}
            if set(parameters) != expected_keys or diagnostics["optimizer_success"] is not True:
                raise ValueError("cached parameters or convergence diagnostics are invalid")
            fitted = list(parameters["action_lengthscales"])
            if len(fitted) != d:
                raise ValueError("wrong number of action lengthscales")
            if context_model == "matern":
                fitted.append(parameters["context_lengthscale"])
            elif parameters["context_lengthscale"] is not None:
                raise ValueError("unexpected context lengthscale")
            fitted.extend([parameters["signal_variance"], parameters["noise_variance"]])
            p = np.asarray(fitted, dtype=float)
            if not np.isfinite(p).all() or (p <= 0).any() or any(
                v < lo * (1 - 1e-12) or v > hi * (1 + 1e-12)
                for v, (lo, hi) in zip(p, bounds, strict=True)
            ):
                raise ValueError("cached fitted parameters are outside the configured bounds")
            spatial, temporal, amplitude, noise = covariance(np.log(p))
            stored_spatial = np.asarray(payload["action_covariance"], dtype=float)
            stored_temporal = np.asarray(payload["context_covariance"], dtype=float)
            if (
                stored_spatial.shape != spatial.shape or stored_temporal.shape != temporal.shape
                or not np.allclose(stored_spatial, amplitude * spatial, rtol=5e-14, atol=5e-14)
                or not np.allclose(stored_temporal, temporal, rtol=5e-14, atol=5e-14)
                or payload["observation_offset"] != offset
                or payload["noise_variance"] != parameters["noise_variance"]
            ):
                raise ValueError("cached covariance or observation statistics do not match fitted parameters")
            with threadpool_limits(limits=1):
                if not np.isclose(nll(np.log(p)), diagnostics["negative_log_marginal_likelihood"],
                                  rtol=1e-8, atol=1e-8):
                    raise ValueError("cached likelihood does not match calibration inputs")
            return SeparableCalibration(
                FiniteSupportKernel(x, stored_spatial),
                FiniteSupportKernel(support, stored_temporal),
                float(parameters["noise_variance"]), offset, parameters,
                {**diagnostics, "cache_hit": True, "cache_fingerprint": fingerprint},
            )
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError, OSError) as error:
            raise ValueError(f"Invalid separable calibration cache: {error}") from error

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
    names = [f"action_lengthscale[{i}]" for i in range(d)]
    if context_model == "matern":
        names.append("context_lengthscale")
    names.extend(["signal_variance", "noise_variance"])
    # Bound proximity is measured in the same log coordinates optimized above.
    boundary_tolerance = 1e-6
    boundary_parameters = [
        {"parameter": name, "side": side, "value": float(value), "bound": float(bound)}
        for name, value, (lo, hi) in zip(names, p, bounds, strict=True)
        for side, bound in (("lower", lo), ("upper", hi))
        if abs(float(np.log(value / bound))) <= boundary_tolerance
    ]
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
        "boundary_log_tolerance": boundary_tolerance,
        "boundary_parameters": boundary_parameters,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }
    if cache is not None:
        diagnostics.update(cache_hit=False, cache_fingerprint=fingerprint)
    result = SeparableCalibration(
        FiniteSupportKernel(x, amplitude * spatial),
        FiniteSupportKernel(support, temporal),
        float(noise),
        offset,
        parameters,
        diagnostics,
    )
    if cache is not None and bool(fit.success) and np.isfinite(fit.fun):
        payload = {
            "identity": identity, "fingerprint": fingerprint,
            "parameters": parameters, "diagnostics": diagnostics,
            "action_covariance": (amplitude * spatial).tolist(),
            "context_covariance": temporal.tolist(),
            "noise_variance": float(noise), "observation_offset": offset,
        }
        cache.write_json(_CACHE_NAME, {"payload": payload, "sha256": _json_digest(payload)})
    return result
