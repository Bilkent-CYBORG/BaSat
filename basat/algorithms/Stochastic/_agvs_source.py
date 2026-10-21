"""AG-VS numerical policies from the supplied AISTATS 2027 source release.

The node GP and candidate policy deliberately belong to AG-VS rather than the
shared Basat GP/acquisition defaults. Adapted from the MIT-licensed anonymous
release; see ``_agvs_source_LICENSE.txt``. Models see observations only.
"""

from __future__ import annotations

import math
import warnings
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from torch.quasirandom import SobolEngine


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _dtype(value: torch.dtype | str) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    return {"float32": torch.float32, "float64": torch.float64}[value]


def latin_hypercube(bounds: Any, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw the release's full initial design with its advancing NumPy stream."""
    limits = _numpy(bounds)
    unit = np.zeros((int(n), len(limits)), dtype=float)
    if n <= 0:
        return unit
    for coordinate in range(len(limits)):
        permutation = rng.permutation(n)
        unit[:, coordinate] = (permutation + rng.random(n)) / float(n)
    return limits[:, 0] + unit * (limits[:, 1] - limits[:, 0])


class SourceBOGP:
    """Raw-coordinate, isotropic Matérn GP used by released AG-VS-BO.

    ``predict`` follows Basat's tensor-pair interface; ``predict_numpy`` exposes
    the sklearn posterior used by the released acquisition calculations.
    """

    def __init__(
        self,
        *,
        noise_variance: float = 1e-4,
        learn_noise: bool = False,
        maxiter: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | str = torch.float64,
    ) -> None:
        if not math.isfinite(noise_variance) or noise_variance < 0:
            raise ValueError("noise_variance must be finite and non-negative.")
        if maxiter is not None and (isinstance(maxiter, bool) or maxiter < 1):
            raise ValueError("maxiter must be a positive integer or None.")
        self.noise_variance = float(noise_variance)
        self.learn_noise = bool(learn_noise)
        self.maxiter = maxiter
        self.device, self.dtype = torch.device(device), _dtype(dtype)
        self.model: GaussianProcessRegressor | None = None
        self.fit_mode = "unfitted"

    def _new_model(self, *, kernel: Any = None) -> GaussianProcessRegressor:
        if kernel is None:
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            if self.learn_noise:
                kernel += WhiteKernel(
                    noise_level=max(self.noise_variance, 1e-8),
                    noise_level_bounds=(1e-8, 10.0),
                )
        optimizer: Any = "fmin_l_bfgs_b"
        if self.maxiter is not None:
            def optimize(objective, initial_theta, bounds):
                result = minimize(
                    objective, initial_theta, method="L-BFGS-B", jac=True,
                    bounds=bounds, options={"maxiter": self.maxiter},
                )
                return result.x, result.fun
            optimizer = optimize
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10 if self.learn_noise else self.noise_variance,
            normalize_y=False,
            n_restarts_optimizer=0,
            random_state=2021,
            optimizer=optimizer,
        )

    @property
    def X_train_(self) -> np.ndarray:
        return self._require_model().X_train_

    @property
    def y_train_(self) -> np.ndarray:
        return self._require_model().y_train_

    @property
    def lengthscales(self) -> np.ndarray:
        fitted = self._require_model()
        signal = fitted.kernel_.k1 if self.learn_noise else fitted.kernel_
        return np.full(self.X_train_.shape[1], float(signal.k2.length_scale))

    def _require_model(self) -> GaussianProcessRegressor:
        if self.model is None:
            raise RuntimeError("AG-VS BO GP has not been fitted.")
        return self.model

    def _incremental_fit(self, previous: SourceBOGP, X: np.ndarray, y: np.ndarray):
        source = previous._require_model()
        old_n = len(source.X_train_)
        if (
            old_n < 2 or old_n > len(X)
            or source.X_train_.shape[1:] != X.shape[1:]
            or not np.array_equal(source.X_train_, X[:old_n])
            or not np.array_equal(source.y_train_, y[:old_n])
            or previous.learn_noise != self.learn_noise
            or previous.noise_variance != self.noise_variance
        ):
            return None
        fitted = deepcopy(source)
        if old_n == len(X):
            return fitted
        try:
            added = X[old_n:]
            cross = source.kernel_(added, source.X_train_)
            solved = solve_triangular(source.L_, cross.T, lower=True, check_finite=False)
            schur = source.kernel_(added)
            schur.flat[:: len(added) + 1] += float(source.alpha)
            schur -= solved.T @ solved
            schur = 0.5 * (schur + schur.T)
            factor = np.zeros((len(X), len(X)), dtype=float)
            factor[:old_n, :old_n] = source.L_
            factor[old_n:, :old_n] = solved.T
            factor[old_n:, old_n:] = np.linalg.cholesky(schur)
            alpha = cho_solve((factor, True), y, check_finite=False)
        except (ValueError, TypeError, np.linalg.LinAlgError):
            return None
        fitted.X_train_, fitted.y_train_ = X.copy(), y.copy()
        fitted.L_, fitted.alpha_ = factor, alpha
        fitted.n_features_in_ = int(X.shape[1])
        fitted.log_marginal_likelihood_value_ = float(
            -0.5 * y.dot(alpha) - np.log(np.diag(factor)).sum()
            - 0.5 * len(X) * np.log(2.0 * np.pi)
        )
        return fitted

    def fit(
        self, X: Any, y: Any, *, warm_model: SourceBOGP | None = None,
        hold_hyperparameters: bool = False,
    ) -> SourceBOGP:
        inputs, targets = _numpy(X), _numpy(y).reshape(-1)
        if (
            inputs.ndim != 2 or len(inputs) < 1 or inputs.shape[1] < 1
            or len(inputs) != len(targets)
            or not np.isfinite(inputs).all() or not np.isfinite(targets).all()
        ):
            raise ValueError("AG-VS requires aligned finite node GP inputs and observations.")
        if warm_model is not None and hold_hyperparameters:
            self.model = self._incremental_fit(warm_model, inputs, targets)
            if self.model is not None:
                self.fit_mode = "incremental_fixed_kernel"
                return self
        kernel = None
        if warm_model is not None:
            kernel = clone(warm_model._require_model().kernel_)
        model = self._new_model(kernel=kernel)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(inputs, targets)
        self.model = model
        self.fit_mode = "warm_started_hyperparameter_fit" if kernel is not None else "full_hyperparameter_fit"
        return self

    def predict_numpy(self, X: Any, *, return_std: bool = False, return_cov: bool = False):
        return self._require_model().predict(_numpy(X), return_std=return_std, return_cov=return_cov)

    def predict(self, X: Any) -> tuple[torch.Tensor, torch.Tensor]:
        means, stds = self.predict_numpy(X, return_std=True)
        return (
            torch.as_tensor(means, dtype=self.dtype, device=self.device),
            torch.as_tensor(stds, dtype=self.dtype, device=self.device),
        )

    def state_dict(self) -> dict[str, Any]:
        """Save fitted factors, so restoring does not fit or advance any RNG."""
        model = self._require_model()
        return {
            "backend": "bo", "noise_variance": self.noise_variance,
            "learn_noise": self.learn_noise, "maxiter": self.maxiter,
            "fit_mode": self.fit_mode,
            "X_train": model.X_train_.copy(), "y_train": model.y_train_.copy(),
            "kernel_theta": model.kernel_.theta.copy(),
            "factor": model.L_.copy(), "alpha": model.alpha_.copy(),
            "log_marginal_likelihood": float(model.log_marginal_likelihood_value_),
        }

    @classmethod
    def from_state_dict(
        cls, state: dict[str, Any], *, device: torch.device | str = "cpu",
        dtype: torch.dtype | str = torch.float64,
    ) -> SourceBOGP:
        instance = cls(
            noise_variance=state["noise_variance"], learn_noise=state["learn_noise"],
            maxiter=state["maxiter"], device=device, dtype=dtype,
        )
        model = instance._new_model()
        model.kernel_ = clone(model.kernel)
        model.kernel_.theta = _numpy(state["kernel_theta"]).copy()
        model.X_train_, model.y_train_ = _numpy(state["X_train"]).copy(), _numpy(state["y_train"]).copy()
        model.L_, model.alpha_ = _numpy(state["factor"]).copy(), _numpy(state["alpha"]).copy()
        model.n_features_in_ = model.X_train_.shape[1]
        model._y_train_mean, model._y_train_std = np.array(0.0), np.array(1.0)
        model.log_marginal_likelihood_value_ = float(state["log_marginal_likelihood"])
        instance.model, instance.fit_mode = model, str(state["fit_mode"])
        return instance


def classical_ei(model: Any, candidates: Any, *, xi: float = 1e-4) -> np.ndarray:
    """Classical EI with the release's fitted-training-mean incumbent."""
    means, stds = model.predict_numpy(candidates, return_std=True)
    incumbent = float(np.max(model.predict_numpy(model.X_train_)))
    improvement = np.asarray(means).reshape(-1) - incumbent - float(xi)
    stds = np.asarray(stds).reshape(-1)
    values = np.zeros_like(improvement)
    positive = stds > 0
    z = np.zeros_like(improvement)
    z[positive] = improvement[positive] / stds[positive]
    values[positive] = improvement[positive] * norm.cdf(z[positive]) + stds[positive] * norm.pdf(z[positive])
    return values


def propose_bo(
    model: Any, bounds: Any, q: int, rng: np.random.Generator, *,
    joint: bool = False, mc_samples: int = 256, xi: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Released independent top-k EI, or its explicit greedy joint-q path."""
    limits = _numpy(bounds)
    if q < 1:
        return np.empty((0, len(limits))), np.empty(0)
    seed = int(rng.integers(0, 500000))
    unit = SobolEngine(len(limits), scramble=True, seed=seed).draw(max(1024, q)).to(torch.float64).numpy()
    candidates = limits[:, 0] + unit * (limits[:, 1] - limits[:, 0])
    pointwise = classical_ei(model, candidates, xi=xi)
    if not joint or q == 1:
        selected = np.argsort(pointwise)[-q:]
        return candidates[selected], pointwise[selected]
    means, stds = model.predict_numpy(candidates, return_std=True)
    means = np.asarray(means).reshape(-1)
    stds = np.maximum(np.asarray(stds).reshape(-1), 1e-12)
    selected = [int(np.nanargmax(pointwise))]
    scores = [float(pointwise[selected[0]])]
    best_f = float(np.max(model.y_train_))
    sample_count = max(32, int(mc_samples))
    while len(selected) < min(q, len(candidates)):
        chosen_x = candidates[np.asarray(selected, dtype=int)]
        chosen_mean, chosen_cov = model.predict_numpy(chosen_x, return_cov=True)
        chosen_mean = np.asarray(chosen_mean).reshape(-1)
        chosen_cov = np.asarray(chosen_cov)
        chosen_cov = 0.5 * (chosen_cov + chosen_cov.T)
        chosen_cov.flat[:: len(selected) + 1] += 1e-10
        chosen_samples = rng.multivariate_normal(chosen_mean, chosen_cov, size=sample_count, check_valid="ignore")
        fitted = model._require_model()
        cross_train = fitted.kernel_(candidates, fitted.X_train_)
        train_chosen = fitted.kernel_(fitted.X_train_, chosen_x)
        solved = np.linalg.solve(fitted.L_, train_chosen)
        solved = np.linalg.solve(fitted.L_.T, solved)
        cross_cov = fitted.kernel_(candidates, chosen_x) - cross_train @ solved
        weights = cross_cov @ np.linalg.pinv(chosen_cov, hermitian=True)
        conditional_mean = means[:, None] + ((chosen_samples - chosen_mean[None, :]) @ weights.T).T
        conditional_variance = np.maximum(stds**2 - np.einsum("ij,ij->i", weights, cross_cov), 1e-12)
        candidate_samples = conditional_mean + np.sqrt(conditional_variance)[:, None] * rng.standard_normal((len(candidates), sample_count))
        current_max = np.maximum(best_f, np.max(chosen_samples, axis=1))
        joint_values = np.mean(np.maximum(current_max[None, :], candidate_samples) - best_f, axis=1)
        joint_values[np.asarray(selected, dtype=int)] = -np.inf
        next_index = int(np.nanargmax(joint_values))
        selected.append(next_index)
        scores.append(float(joint_values[next_index]))
    return candidates[np.asarray(selected, dtype=int)], np.asarray(scores)


__all__ = ["SourceBOGP", "classical_ei", "latin_hypercube", "propose_bo"]
