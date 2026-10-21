"""AG-VS's shared isotropic TuRBO surrogate and persistent proposal state.

The supplied anonymous MIT release's AG-VS adapter defines the policy here;
see ``_agvs_source_LICENSE.txt``. This module owns no objective, observations,
or random stream. The surrounding algorithm supplies those explicitly.
"""

from __future__ import annotations

import math
import warnings
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from ._agvs_source import _dtype, _numpy


@contextmanager
def _native_settings(*, max_cholesky_size: int = 800):
    """Restore the native source settings within an importing BoTorch process."""
    with (
        gpytorch.settings.fast_computations(
            covar_root_decomposition=True, log_prob=True, solves=True
        ),
        gpytorch.settings.max_cholesky_size(max_cholesky_size),
        gpytorch.settings.max_eager_kernel_size(512),
        gpytorch.settings.cholesky_max_tries(3),
        gpytorch.settings.max_root_decomposition_size(100),
    ):
        yield


class _NodeExactGP(gpytorch.models.ExactGP):
    """One isotropic covariance for acquisition, scores and diagnostics."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor, likelihood: Any):
        super().__init__(X, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                lengthscale_constraint=gpytorch.constraints.Interval(0.005, math.sqrt(X.shape[1])),
            ),
            outputscale_constraint=gpytorch.constraints.Interval(0.05, 20.0),
        )

    def forward(self, X: torch.Tensor):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(X), self.covar_module(X))


class SourceTurboGP:
    """The released AG-VS-TuRBO model, presented through Basat's GP interface."""

    def __init__(
        self, *, bounds: Any, noise_variance: float = 1e-4,
        learn_noise: bool = False, maxiter: int | None = 50,
        device: torch.device | str = "cpu", dtype: torch.dtype | str = torch.float64,
        min_cuda: int = 1024,
    ) -> None:
        if not math.isfinite(noise_variance) or noise_variance <= 0:
            raise ValueError("TuRBO observation variance must be finite and positive.")
        self.bounds = _numpy(bounds).copy()
        self.noise_variance = float(noise_variance)
        self.learn_noise = bool(learn_noise)
        self.maxiter = 50 if maxiter is None else int(maxiter)
        if self.maxiter < 1:
            raise ValueError("TuRBO GP training steps must be positive.")
        self.device, self.dtype = torch.device(device), _dtype(dtype)
        self.min_cuda = int(min_cuda)
        self.gp: _NodeExactGP | None = None
        self.X_train_ = np.empty((0, len(self.bounds)))
        self.y_train_ = np.empty(0)
        self.target_median, self.target_scale = 0.0, 1.0
        self.cholesky_jitter = 0.0
        self.fit_mode = "unfitted"

    @property
    def model(self):
        return self.gp

    @property
    def lengthscales(self) -> np.ndarray:
        if self.gp is None:
            raise RuntimeError("AG-VS TuRBO GP has not been fitted.")
        value = float(self.gp.covar_module.base_kernel.lengthscale.detach().reshape(-1)[0].cpu())
        return np.full(self.X_train_.shape[1], value)

    @property
    def fitted_noise_variance(self) -> float:
        if self.gp is None:
            raise RuntimeError("AG-VS TuRBO GP has not been fitted.")
        return float(self.gp.likelihood.noise.detach().reshape(-1)[0].cpu()) * self.target_scale**2

    def _new_model(self, X: torch.Tensor, y: torch.Tensor) -> _NodeExactGP:
        if self.learn_noise:
            constraint = gpytorch.constraints.Interval(1e-8, 10.0)
        else:
            constraint = gpytorch.constraints.GreaterThan(0.0, transform=None)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=constraint).to(
            dtype=X.dtype, device=X.device
        )
        return _NodeExactGP(X, y, likelihood).to(dtype=X.dtype, device=X.device)

    def _train_once(
        self, X: torch.Tensor, y: torch.Tensor, *, warm_model: SourceTurboGP | None,
        jitter: float,
    ) -> _NodeExactGP:
        model = self._new_model(X, y)
        initial_noise = 1e-4 if self.learn_noise else self.noise_variance / self.target_scale**2
        if warm_model is not None and warm_model.gp is not None:
            signal = {key: value for key, value in warm_model.gp.state_dict().items() if not key.startswith("likelihood.")}
            model.load_state_dict(signal, strict=False)
            if self.learn_noise:
                old_noise = float(warm_model.gp.likelihood.noise.detach().reshape(-1)[0].cpu())
                initial_noise = min(max(old_noise, 1.0001e-8), 9.9999)
        else:
            model.initialize(**{
                "covar_module.outputscale": 1.0,
                "covar_module.base_kernel.lengthscale": 0.5,
                "likelihood.noise": initial_noise,
            })
        model.likelihood.noise = torch.as_tensor(initial_noise, dtype=X.dtype, device=X.device)
        model.likelihood.noise_covar.raw_noise.requires_grad_(self.learn_noise)
        model.train()
        model.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam([{"params": parameters}], lr=0.1)
        jitter_context = (
            gpytorch.settings.cholesky_jitter(float_value=jitter, double_value=jitter)
            if jitter > 0 else nullcontext()
        )
        # BoTorch changes these GPyTorch process defaults at import time. The
        # release runs its native ExactGP with all three enabled; even the
        # small-data Cholesky paths can otherwise differ by a few ulps, which
        # can change a later Thompson argmin. Keep the policy scoped here.
        with jitter_context, _native_settings():
            for _ in range(self.maxiter):
                optimizer.zero_grad()
                loss = -mll(model(X), y)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite AG-VS TuRBO GP loss.")
                loss.backward()
                if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
                    raise FloatingPointError("Non-finite AG-VS TuRBO GP gradient.")
                optimizer.step()
                if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
                    raise FloatingPointError("Non-finite AG-VS TuRBO GP parameter.")
        model.eval()
        model.likelihood.eval()
        return model

    @staticmethod
    def _numerical_failure(error: Exception) -> bool:
        if type(error).__name__.lower() in {"notpsderror", "nanerror", "_linalgerror", "linalgerror"}:
            return True
        tokens = ("cholesky", "not positive definite", "not p.d.", "notpsd", "singular", "symeig", "linalg", "non-finite", "nonfinite", "nan", "infinity")
        return isinstance(error, (RuntimeError, FloatingPointError)) and any(token in str(error).lower() for token in tokens)

    def fit(
        self, X: Any, y: Any, *, warm_model: SourceTurboGP | None = None,
        hold_hyperparameters: bool = False,
    ) -> SourceTurboGP:
        # The release builds X_local with NumPy's advanced column indexing,
        # which yields a Fortran-contiguous array. Its normalized torch view
        # keeps that column-major layout. GP reductions depend on the strides;
        # preserve them instead of converting the model inputs to row-major.
        self.X_train_ = np.array(_numpy(X), dtype=float, order="F", copy=True)
        self.y_train_ = _numpy(y).reshape(-1).copy()
        if (
            self.X_train_.ndim != 2 or len(self.X_train_) < 1
            or self.X_train_.shape != (len(self.y_train_), len(self.bounds))
            or not np.isfinite(self.X_train_).all() or not np.isfinite(self.y_train_).all()
        ):
            raise ValueError("AG-VS requires aligned finite TuRBO GP data.")
        unit = np.clip((self.X_train_ - self.bounds[:, 0]) / np.diff(self.bounds, axis=1).reshape(-1), 0, 1)
        costs = -self.y_train_
        self.target_median = float(np.median(costs))
        self.target_scale = float(np.std(costs))
        if self.target_scale < 1e-6:
            self.target_scale = 1.0
        targets = (costs - self.target_median) / self.target_scale
        train_device = torch.device("cpu") if len(unit) < self.min_cuda else self.device
        train_dtype = torch.float64 if len(unit) < self.min_cuda else torch.float32
        failures = []
        for stage, dtype, warm, jitter in (
            ("configured", train_dtype, warm_model, 0.0),
            ("float64", torch.float64, None, 0.0),
            ("float64+jitter", torch.float64, None, 1e-3),
        ):
            X_tensor = torch.as_tensor(unit, dtype=dtype, device=train_device)
            y_tensor = torch.as_tensor(targets, dtype=dtype, device=train_device)
            try:
                self.gp = self._train_once(X_tensor, y_tensor, warm_model=warm, jitter=jitter)
                self.cholesky_jitter = jitter
                self.fit_mode = "numerical_jitter_retry" if jitter else ("warm_started_hyperparameter_fit" if warm is not None else "full_hyperparameter_fit")
                return self
            except Exception as error:
                if not self._numerical_failure(error):
                    raise
                failures.append(f"{stage}: {type(error).__name__}: {error}")
        raise RuntimeError("AG-VS TuRBO numerical recovery failed: " + " | ".join(failures))

    def predict_numpy(self, X: Any, *, return_std: bool = False, return_cov: bool = False):
        if self.gp is None:
            raise RuntimeError("AG-VS TuRBO GP has not been fitted.")
        inputs = np.asfortranarray(_numpy(X))
        unit = np.clip((inputs - self.bounds[:, 0]) / np.diff(self.bounds, axis=1).reshape(-1), 0, 1)
        parameter = next(self.gp.parameters())
        query = torch.as_tensor(unit, dtype=parameter.dtype, device=parameter.device)
        with warnings.catch_warnings(), torch.no_grad(), gpytorch.settings.fast_pred_var(), _native_settings(), gpytorch.settings.cholesky_jitter(
            float_value=max(self.cholesky_jitter, 1e-6), double_value=max(self.cholesky_jitter, 1e-8)
        ):
            warnings.simplefilter("ignore", gpytorch.utils.warnings.GPInputWarning)
            posterior = self.gp.likelihood(self.gp(query))
            mean = -(self.target_median + self.target_scale * posterior.mean.detach().cpu().numpy())
            if return_cov:
                return mean, self.target_scale**2 * posterior.covariance_matrix.detach().cpu().numpy()
            if return_std:
                std = self.target_scale * posterior.variance.clamp_min(0).sqrt().detach().cpu().numpy()
                return mean, std
            return mean

    def predict(self, X: Any) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.predict_numpy(X, return_std=True)
        return torch.as_tensor(mean, dtype=self.dtype, device=self.device), torch.as_tensor(std, dtype=self.dtype, device=self.device)

    def state_dict(self) -> dict[str, Any]:
        if self.gp is None:
            raise RuntimeError("AG-VS TuRBO GP has not been fitted.")
        return {
            "backend": "turbo", "bounds": self.bounds.copy(),
            "noise_variance": self.noise_variance, "learn_noise": self.learn_noise,
            "maxiter": self.maxiter, "min_cuda": self.min_cuda,
            "X_train": self.X_train_.copy(), "y_train": self.y_train_.copy(),
            "target_median": self.target_median, "target_scale": self.target_scale,
            "cholesky_jitter": self.cholesky_jitter, "fit_mode": self.fit_mode,
            "model_state": {key: value.detach().cpu().clone() for key, value in self.gp.state_dict().items()},
            "training_X": self.gp.train_inputs[0].detach().cpu().clone(),
            "training_y": self.gp.train_targets.detach().cpu().clone(),
            "training_device": str(next(self.gp.parameters()).device),
        }

    @classmethod
    def from_state_dict(
        cls, state: dict[str, Any], *, device: torch.device | str = "cpu",
        dtype: torch.dtype | str = torch.float64,
    ) -> SourceTurboGP:
        instance = cls(
            bounds=state["bounds"], noise_variance=state["noise_variance"], learn_noise=state["learn_noise"],
            maxiter=state["maxiter"], min_cuda=state["min_cuda"], device=device, dtype=dtype,
        )
        for name in ("target_median", "target_scale", "cholesky_jitter", "fit_mode"):
            setattr(instance, name, state[name])
        instance.X_train_ = np.array(_numpy(state["X_train"]), dtype=float, order="F", copy=True)
        instance.y_train_ = _numpy(state["y_train"]).copy()
        training_device = torch.device("cpu") if state["training_device"] == "cpu" else torch.device(device)
        X = state["training_X"].to(device=training_device)
        y = state["training_y"].to(device=training_device)
        instance.gp = instance._new_model(X, y)
        instance.gp.load_state_dict(deepcopy(state["model_state"]))
        instance.gp.likelihood.noise_covar.raw_noise.requires_grad_(instance.learn_noise)
        instance.gp.eval()
        instance.gp.likelihood.eval()
        return instance


@dataclass
class TurboProposalState:
    length: float = 0.8
    success_count: int = 0
    failure_count: int = 0
    n_evals: int = 0
    restart_count: int = 0

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> TurboProposalState:
        return cls(**state)

    def restart(self) -> None:
        self.length = 0.8
        self.success_count = self.failure_count = self.n_evals = 0
        self.restart_count += 1


def propose_turbo(
    model: SourceTurboGP, bounds: Any, q: int,
    rng: np.random.Generator | np.random.RandomState,
    torch_generator: torch.Generator, sobol_engines: dict[int, SobolEngine],
    state: TurboProposalState, *, max_evals: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """One predictive Thompson batch; restarts reset only trust-region state."""
    if model.gp is None:
        raise RuntimeError("A fitted shared node GP is required for AG-VS TuRBO.")
    if q < 1 or q >= max_evals:
        raise ValueError("TuRBO batch size must be positive and less than its epoch limit.")
    if state.n_evals >= max_evals or state.n_evals + q > max_evals or state.length < 0.5**7:
        state.restart()
    limits = _numpy(bounds)
    dimension = len(limits)
    count = min(100 * dimension, 5000)
    unit_X = np.clip((model.X_train_ - limits[:, 0]) / np.diff(limits, axis=1).reshape(-1), 0, 1)
    costs = -model.y_train_
    center = unit_X[int(np.argmin(costs))][None, :]
    weights = model.lengthscales.copy()
    weights /= weights.mean()
    weights /= np.prod(np.power(weights, 1.0 / len(weights)))
    lower = np.clip(center - weights * state.length / 2, 0, 1)
    upper = np.clip(center + weights * state.length / 2, 0, 1)
    candidate_device = torch.device("cpu") if count < model.min_cuda else model.device
    candidate_dtype = torch.float64 if count < model.min_cuda else torch.float32
    model.gp = model.gp.to(dtype=candidate_dtype, device=candidate_device)
    model.gp.set_train_data(
        inputs=model.gp.train_inputs[0].to(dtype=candidate_dtype, device=candidate_device),
        targets=model.gp.train_targets.to(dtype=candidate_dtype, device=candidate_device), strict=False,
    )
    if dimension not in sobol_engines:
        sobol_engines[dimension] = SobolEngine(dimension, scramble=True, seed=torch_generator.initial_seed())
    perturbations = sobol_engines[dimension].draw(count).to(dtype=candidate_dtype, device=candidate_device).cpu().numpy()
    perturbations = lower + (upper - lower) * perturbations
    if isinstance(rng, np.random.RandomState):
        mask = rng.rand(count, dimension) <= min(20.0 / dimension, 1.0)
        empty = np.where(mask.sum(axis=1) == 0)[0]
        mask[empty, rng.randint(0, dimension, size=len(empty))] = True
    else:
        mask = rng.random((count, dimension)) <= min(20.0 / dimension, 1.0)
        empty = np.where(mask.sum(axis=1) == 0)[0]
        mask[empty, rng.integers(0, dimension, size=len(empty))] = True
    candidates = np.repeat(center, count, axis=0)
    candidates[mask] = perturbations[mask]
    query = torch.as_tensor(candidates, dtype=candidate_dtype, device=candidate_device)
    # GPyTorch distributions sample through the process generator. Scope that
    # detail and restore its state; the caller's explicit stream keeps advancing.
    devices = [candidate_device.index or 0] if candidate_device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        if candidate_device.type == "cpu":
            torch.random.set_rng_state(torch_generator.get_state())
        else:
            torch.cuda.set_rng_state(torch_generator.get_state(), candidate_device)
        with torch.no_grad(), _native_settings(max_cholesky_size=2000), gpytorch.settings.cholesky_jitter(
            float_value=max(model.cholesky_jitter, 1e-6), double_value=max(model.cholesky_jitter, 1e-8)
        ):
            samples = model.gp.likelihood(model.gp(query)).sample(torch.Size([q])).t().cpu().numpy()
        if candidate_device.type == "cpu":
            torch_generator.set_state(torch.random.get_rng_state())
        else:
            torch_generator.set_state(torch.cuda.get_rng_state(candidate_device))
    values = model.target_median + model.target_scale * samples
    positions = []
    scores = []
    working = values.copy()
    for column in range(q):
        position = int(np.argmin(working[:, column]))
        positions.append(position)
        scores.append(-float(np.min(values[:, column])))
        working[position, :] = np.inf
    points = limits[:, 0] + np.diff(limits, axis=1).reshape(-1) * candidates[positions]
    return points, np.asarray(scores)


def update_turbo(
    state: TurboProposalState, old_best: float, new_values: Any,
    dimension: int, batch_size: int,
) -> None:
    """Apply native relative success and trust-region counter updates once."""
    values = _numpy(new_values).reshape(-1)
    if len(values) < 1 or len(values) > batch_size:
        raise ValueError("TuRBO updates require one nonempty scheduler batch.")
    success = -float(np.max(values)) < -float(old_best) - 1e-3 * abs(float(old_best))
    if success:
        state.success_count += 1
        state.failure_count = 0
    else:
        state.success_count = 0
        state.failure_count += 1
    if state.success_count == 3:
        state.length = min(2 * state.length, 1.6)
        state.success_count = 0
    elif state.failure_count == math.ceil(max(4.0 / batch_size, dimension / batch_size)):
        state.length *= 0.5
        state.failure_count = 0
    state.n_evals += len(values)


__all__ = ["SourceTurboGP", "TurboProposalState", "propose_turbo", "update_turbo"]
