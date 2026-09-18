"""Exact fixed polynomial-kernel conditioning using its finite feature basis."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import gpytorch
import numpy as np
import scipy.linalg
import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from linear_operator.operators import (  # type: ignore[import-untyped]
    DiagLinearOperator,
    RootLinearOperator,
)

from .GaussianProcess import GaussianProcess, GaussianProcessConfig, ModelStateError


class PolynomialFeatureGaussianProcess(GaussianProcess):
    """Condition an exact polynomial GP by QR, with no rank approximation.

    The kernel must be ``outputscale * (x @ z + offset)**degree``. Only
    fixed-parameter, zero-mean, unstandardized CPU float64 models are accepted.
    There are exactly binomial(dimension + degree, degree) features. Positive
    observation variances are handled by whitening the design; this avoids
    subtracting two large ill-conditioned covariance matrices.

    Parameters
    ----------
    config : GaussianProcessConfig
        Explicit polynomial kernel and fixed observation variance.
    input_dimension : int
        Model coordinate count. The model never knows the objective.
    maximum_features : int
        Guard against an accidentally enormous combinatorial basis.
    """

    def __init__(self, config: GaussianProcessConfig, *, input_dimension: int,
                 maximum_features: int = 10000):
        super().__init__(config, input_dimension=input_dimension)
        if (self.dtype != torch.float64 or self.device.type != "cpu"
                or config.mean != "zero" or config.standardize_outcomes
                or config.noise_mode != "fixed"):
            raise ValueError("Polynomial feature inference requires zero-mean, fixed-noise, "
                             "unstandardized CPU float64 settings.")
        kernel = self._new_kernel().double()
        if kernel.active_dims is not None:
            raise ValueError("Expected a full-coordinate polynomial kernel.")
        amplitude = 1.0
        if isinstance(kernel, gpytorch.kernels.ScaleKernel):
            amplitude = float(kernel.outputscale.detach())
            kernel = kernel.base_kernel
        if not isinstance(kernel, gpytorch.kernels.PolynomialKernel) or kernel.active_dims is not None:
            raise ValueError("Expected a full-coordinate polynomial kernel.")
        power = kernel.power
        if (isinstance(power, bool) or not isinstance(power, Real)
                or not math.isfinite(power) or power < 0 or math.floor(power) != power):
            raise ValueError("Exact polynomial features require a non-negative integer degree.")
        degree, offset = int(power), float(kernel.offset.detach())
        count = math.comb(input_dimension + degree, degree)
        if count > maximum_features:
            raise ValueError(f"Polynomial needs {count} features; limit is {maximum_features}.")
        if amplitude <= 0 or offset <= 0 or not np.isfinite(amplitude + offset):
            raise ValueError("Polynomial outputscale and offset must be positive and finite.")

        def compositions(dimension, remaining, prefix=()):
            if dimension == 0:
                yield prefix
            else:
                for power in range(remaining + 1):
                    yield from compositions(dimension - 1, remaining - power, (*prefix, power))

        powers = list(compositions(input_dimension, degree))
        self.exponents = np.array(powers, dtype=int)
        self.coefficients = np.array([
            math.sqrt(amplitude * math.factorial(degree)
                      / math.prod(math.factorial(p) for p in row)
                      / math.factorial(degree - sum(row)) * offset ** (degree - sum(row)))
            for row in powers])
        self.feature_count = count
        self.identity = {"degree": degree, "offset": offset, "amplitude": amplitude,
                         "input_dimension": input_dimension}
        self.weights: Any = None
        self.covariance_root: Any = None

    def features(self, points: Any) -> np.ndarray:
        """Return the exact basis in the original model coordinates."""
        x = np.asarray(points, dtype=float).reshape(-1, self.input_dimension)
        return np.prod(x[:, None, :] ** self.exponents[None, :, :], axis=-1) * self.coefficients

    def set_data(self, X: Any, y: Any, noise_variance: Any = None) -> None:
        """Validate a complete table before replacing any fitted model state."""
        inputs, targets, noise = self._validate_data(X, y, noise_variance)
        if noise is None or not torch.all(noise > 0):
            raise ValueError("Exact feature conditioning needs positive observation variances.")
        self.X_train, self.y_train, self.observation_noise = (
            inputs.detach().clone(), targets.detach().clone(), noise.detach().clone())
        self.is_fit = False
        self.weights = self.covariance_root = None
        self.last_fit_diagnostics = {}

    def condition_only(self) -> dict[str, Any]:
        """Condition all saved observations using the configured fixed kernel."""
        if self.X_train is None or self.y_train is None or self.observation_noise is None:
            raise ValueError("Set training data before conditioning.")
        phi = self.features(self.X_train.numpy())
        scale = np.sqrt(self.observation_noise.numpy())
        design = np.vstack((np.eye(self.feature_count), phi / scale[:, None]))
        targets = np.r_[np.zeros(self.feature_count), self.y_train.numpy() / scale]
        q, r = np.linalg.qr(design, mode="reduced")
        self.weights = scipy.linalg.solve_triangular(r, q.T @ targets)
        self.covariance_root = scipy.linalg.solve_triangular(r, np.eye(self.feature_count))
        self.is_fit = True
        self.last_fit_diagnostics = {"method": "exact_polynomial_feature_QR",
                                     "feature_count": self.feature_count}
        return dict(self.last_fit_diagnostics)

    def fit(self, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("Polynomial feature backend is fixed-parameter; use fit_maxiter=0.")

    def predict(self, X: Any, *, observation_noise: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_fit:
            self.condition_only()
        points = self._prediction_inputs(X)
        phi = self.features(points.numpy())
        mean = phi @ self.weights
        variance = np.square(phi @ self.covariance_root).sum(1)
        if observation_noise:
            if self.config.noise_variance is None:
                raise ValueError("Test observation noise needs config.noise_variance.")
            variance += self.config.noise_variance
        shape = tuple(points.shape[:-1])
        return (torch.tensor(mean.reshape(shape), dtype=self.dtype),
                torch.tensor(np.sqrt(variance).reshape(shape), dtype=self.dtype))

    def posterior(self, X: Any, *, observation_noise: bool = False) -> Any:
        if not self.is_fit:
            self.condition_only()
        points = self._prediction_inputs(X)
        if points.ndim != 2:
            raise ValueError("Joint feature posterior expects a two-dimensional query table.")
        phi = self.features(points.numpy())
        root = phi @ self.covariance_root
        covariance: Any = RootLinearOperator(torch.tensor(root, dtype=self.dtype))
        if observation_noise:
            if self.config.noise_variance is None:
                raise ValueError("Test observation noise needs config.noise_variance.")
            covariance = covariance + DiagLinearOperator(torch.full(
                (len(points),), self.config.noise_variance, dtype=self.dtype))
        distribution = gpytorch.distributions.MultivariateNormal(
            torch.tensor(phi @ self.weights, dtype=self.dtype),
            covariance)
        return GPyTorchPosterior(distribution)

    def state_dict(self) -> dict[str, Any]:
        """Return a detached snapshot that cannot mutate the live fitted model."""
        return {
            "feature_version": 1, "identity": dict(self.identity),
            "X": None if self.X_train is None else self.X_train.detach().clone(),
            "y": None if self.y_train is None else self.y_train.detach().clone(),
            "noise": None if self.observation_noise is None else self.observation_noise.detach().clone(),
            "is_fit": self.is_fit,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("feature_version") != 1 or state.get("identity") != self.identity:
            raise ModelStateError("Polynomial feature model identity changed.")
        if state["X"] is None:
            self.reset()
            return
        self.set_data(state["X"], state["y"], state["noise"])
        if state["is_fit"]:
            self.condition_only()

    def reset(self) -> None:
        super().reset()
        self.weights = self.covariance_root = None
