"""Exact conditioning for calibrated, fixed covariance models.

The optional GPy full-conditioning operation order is adapted from GPy v1.10.0
inference/latent_function_inference/{exact_gaussian_inference,posterior}.py and
util/linalg.py. Copyright (c) 2012-2014, GPy authors. BSD-3-Clause terms are in
the adjacent FixedRBFKernel_GPy_LICENSE.txt. Native incremental conditioning
is independent of that compatibility path.
"""

from __future__ import annotations

import warnings
from typing import Any

import gpytorch
import numpy as np
import scipy.linalg as la
import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from linear_operator.operators import DenseLinearOperator  # type: ignore[import-untyped]

from ..warnings import BasatJitterWarning, BasatNumericalWarning
from .FixedRBFKernel import FixedRBFKernel
from .GaussianProcess import (
    GaussianProcess,
    GaussianProcessConfig,
    InvalidGaussianProcessConfigError,
    InvalidModelDataError,
    ModelNotFitError,
    ModelStateError,
)


class FiniteSupportKernel(gpytorch.kernels.Kernel):
    """Use a supplied covariance matrix on an exact finite set of coordinates.

    Parameters
    ----------
    support : array-like, shape (M, D)
        Unique permitted coordinates. Off-support inputs raise an error;
        this kernel never silently substitutes a nearest neighbour.
    covariance : array-like, shape (M, M)
        Finite symmetric positive-semidefinite covariance matrix.
    dtype, device : str or torch type
        Tensor precision and execution device.

    Examples
    --------
    >>> import basat
    >>> kernel = basat.FiniteSupportKernel([[0.0], [1.0]], [[1.0, 0.2], [0.2, 1.0]])
    """

    def __init__(
        self, support: Any, covariance: Any, *, dtype: Any = "float64", device: Any = "cpu"
    ) -> None:
        super().__init__()
        from .utils import resolve_device, resolve_dtype

        points = torch.as_tensor(support, dtype=resolve_dtype(dtype), device=resolve_device(device))
        matrix = torch.as_tensor(covariance, dtype=points.dtype, device=points.device)
        if (
            points.ndim != 2
            or not len(points)
            or not points.shape[1]
            or not torch.isfinite(points).all()
        ):
            raise ValueError("support must be a non-empty finite (M, D) table.")
        if len(torch.unique(points, dim=0)) != len(points):
            raise ValueError("support coordinates must be unique.")
        if matrix.shape != (len(points), len(points)) or not torch.isfinite(matrix).all():
            raise ValueError("covariance must be a finite (M, M) matrix.")
        tolerance = 100 * torch.finfo(points.dtype).eps * max(1.0, float(matrix.abs().max()))
        if not torch.allclose(matrix, matrix.T, rtol=0, atol=tolerance):
            raise ValueError("covariance must be symmetric.")
        if float(torch.linalg.eigvalsh(matrix).min()) < -tolerance * len(points):
            raise ValueError("covariance must be positive semidefinite.")
        self.register_buffer("support", points.detach().clone())
        self.register_buffer("covariance", matrix.detach().clone())

    def _indices(self, inputs: torch.Tensor) -> torch.Tensor:
        matches = (inputs.unsqueeze(-2) == self.support).all(dim=-1)
        if not bool(matches.any(dim=-1).all()):
            raise InvalidModelDataError(
                "FiniteSupportKernel received coordinates outside its support."
            )
        return matches.to(torch.long).argmax(dim=-1)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params: Any
    ) -> torch.Tensor:
        """Return covariances without interpolation or hidden coordinate rounding."""
        del params
        i, j = self._indices(x1), self._indices(x2)
        return self.covariance[i, j] if diag else self.covariance[i.unsqueeze(-1), j.unsqueeze(-2)]


class FixedKernelGaussianProcess(GaussianProcess):
    """Condition an exact zero-mean GP without hyperparameter optimization.

    Appending one observation uses a block-Cholesky update, and cached finite
    candidate moments receive an exact rank-one Gaussian update. All observed
    rows are retained. No inducing points, reward oracle, or approximate
    truncation are used. Changed historical inputs, targets *or noise* rebuild
    the factorization. Storage grows with the history, not a fixed run budget.

    Parameters
    ----------
    config : GaussianProcessConfig
        Must use ``mean='zero'``, ``noise_mode='fixed'``,
        ``standardize_outcomes=False``, ``dtype='float64'``, and CPU. Initialize
        kernel parameters before constructing this model; its private copy is
        frozen. Observation variances may be different for every row.
    input_dimension : int
        Number of model coordinates. No benchmark is supplied to the model.
    conditioning : {"incremental", "full"}, default "incremental"
        Update an appended row incrementally, or refactor the complete training
        table after every change. The latter preserves full-refit arithmetic.
    arithmetic : {"native", "gpy"}, default "native"
        ``"gpy"`` uses lower LAPACK solves, NumPy cross-kernel products and
        marginal variance reductions in GPy's operation order. It requires
        ``conditioning="full"`` and does not optimize hyperparameters.
    training_regularization : float, default 0.0
        Explicit nonnegative training-only diagonal, separate from likelihood
        noise and conditional jitter. For example, GPy exact inference uses
        ``1e-8``. Every new conditioning operation that adds this diagonal emits
        a Basat numerical warning; checkpoint reconstruction does not add it
        twice or report an old addition as a new event.

    Notes
    -----
    This inference backend does not expose a differentiable BoTorch model.
    Use finite candidate acquisition with :meth:`predict`, or the ordinary
    :class:`GaussianProcess` for fitted/continuous-gradient models. Jitter is
    added only after a failed positive-definiteness check and is reported by
    a Basat warning with its amount and location.

    Examples
    --------
    >>> import basat
    >>> config = basat.GaussianProcessConfig(mean='zero', noise_mode='fixed',
    ...     noise_variance=0.01, standardize_outcomes=False, dtype='float64')
    >>> gp = basat.FixedKernelGaussianProcess(config, input_dimension=1)
    >>> gp.set_data([[0.0], [1.0]], [0.0, 1.0])
    >>> mean, std = gp.predict([[0.5]])
    """

    def __init__(
        self,
        config: GaussianProcessConfig,
        *,
        input_dimension: int,
        conditioning: str = "incremental",
        arithmetic: str = "native",
        training_regularization: float = 0.0,
    ):
        super().__init__(config, input_dimension=input_dimension)
        if (
            config.mean != "zero"
            or config.noise_mode != "fixed"
            or config.standardize_outcomes
            or self.dtype != torch.float64
            or self.device.type != "cpu"
        ):
            raise InvalidGaussianProcessConfigError(
                "FixedKernelGaussianProcess requires zero mean, fixed noise, "
                "no outcome standardization, float64, and CPU."
            )
        if conditioning not in {"incremental", "full"} or arithmetic not in {"native", "gpy"}:
            raise InvalidGaussianProcessConfigError("Invalid fixed-kernel inference protocol.")
        if arithmetic == "gpy" and conditioning != "full":
            raise InvalidGaussianProcessConfigError("GPy arithmetic requires full conditioning.")
        if (
            isinstance(training_regularization, bool)
            or not isinstance(training_regularization, (int, float))
            or not np.isfinite(training_regularization)
            or training_regularization < 0
        ):
            raise InvalidGaussianProcessConfigError(
                "training_regularization must be a finite nonnegative scalar."
            )
        self.conditioning = conditioning
        self.arithmetic = arithmetic
        self.training_regularization = float(training_regularization)
        self._kernel = self._new_kernel().to(dtype=self.dtype, device=self.device)
        self._kernel.requires_grad_(False)
        self._kernel.eval()
        self._count = 0
        self._factor = np.empty((0, 0), dtype=np.float64, order="F")
        self._whitened = np.empty(0)
        self._jitter_adjustments = np.empty(0)
        self._alpha = np.empty((0, 1))
        self._cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    @property
    def jitter_adjustments(self) -> torch.Tensor:
        """Return numerical variance added to each training row, excluding noise.

        This is zero unless conditioning required numerical regularization.
        Checkpoints preserve these exact diagonal locations and amounts.
        """
        return torch.from_numpy(self._jitter_adjustments.copy())

    def _covariance(self, x: Any, z: Any) -> np.ndarray:
        with torch.no_grad():
            if isinstance(self._kernel, FixedRBFKernel) and x is z:
                return self._kernel(torch.as_tensor(x, dtype=self.dtype)).to_dense().numpy()
            return (
                self._kernel(
                    torch.as_tensor(x, dtype=self.dtype), torch.as_tensor(z, dtype=self.dtype)
                )
                .to_dense()
                .numpy()
            )

    def _reserve(self, count: int) -> None:
        if count <= len(self._whitened):
            return
        capacity = max(count, 64, int(len(self._whitened) * 1.25))
        factor = np.empty((capacity, capacity), dtype=np.float64, order="F")
        white = np.empty(capacity)
        factor[: self._count, : self._count] = self._factor[: self._count, : self._count]
        white[: self._count] = self._whitened[: self._count]
        self._factor, self._whitened = factor, white

    def _warn_jitter(self, amount: float, location: str) -> None:
        warnings.warn(
            f"Basat fixed-kernel GP: added jitter {amount:.6g} "
            f"(observation units squared) at {location} after a "
            "non-positive-definite covariance check.",
            BasatJitterWarning,
            stacklevel=3,
        )

    def set_data(self, X: Any, y: Any, noise_variance: Any = None) -> None:
        """Replace history or append an exact observation, preserving all rows."""
        self._set_data(X, y, noise_variance)

    def _set_data(
        self,
        X: Any,
        y: Any,
        noise_variance: Any,
        *,
        restored_jitter: np.ndarray | None = None,
        restoring: bool = False,
    ) -> None:
        x, target, noise = self._validate_data(X, y, noise_variance)
        assert noise is not None
        old, n = self._count, len(target)
        prefix = (
            old <= n
            and self.X_train is not None
            and self.y_train is not None
            and self.observation_noise is not None
            and torch.equal(x[:old], self.X_train)
            and torch.equal(target[:old], self.y_train)
            and torch.equal(noise[:old], self.observation_noise)
        )
        if n == old and prefix:
            return
        self._reserve(n)
        xx, yy, nn = x.numpy(), target.numpy(), noise.numpy()
        if self.training_regularization and not restoring:
            location = (
                f"appended observation {n}"
                if self.conditioning == "incremental" and old and n == old + 1 and prefix
                else f"full {n}-row covariance diagonal"
            )
            warnings.warn(
                "Basat fixed-kernel GP: added source-protocol training regularization "
                f"{self.training_regularization:.6g} (observation units squared) at {location}. "
                "This declared fixed diagonal is not observation noise or a response to a "
                "failed positive-definiteness check.",
                BasatNumericalWarning,
                stacklevel=3,
            )
        if self.conditioning == "incremental" and old and n == old + 1 and prefix:
            k = self._covariance(xx[:old], xx[-1:]).reshape(-1)
            w = la.solve_triangular(self._factor[:old, :old], k, lower=True, check_finite=False)
            innovation = float(
                self._covariance(xx[-1:], xx[-1:])[0, 0]
                + nn[-1]
                + self.training_regularization
                - w @ w
            )
            if not np.isfinite(innovation):
                raise InvalidModelDataError("Conditional covariance is not finite.")
            jitter = 0.0
            if innovation <= 0:
                jitter = max(self.config.jitter, -innovation + self.config.jitter)
                self._warn_jitter(jitter, f"appended observation {n}")
                innovation += jitter
            residual = yy[-1] - w @ self._whitened[:old]
            diagonal = np.sqrt(innovation)
            self._factor[old, :old], self._factor[old, old] = w, diagonal
            self._whitened[old] = residual / diagonal
            self._jitter_adjustments = np.append(self._jitter_adjustments, jitter)
            if self._cache is not None:
                points, mean, cov, projection = self._cache
                cross = self._covariance(points, xx[-1:]).ravel() - projection.T @ w
                self._cache = (
                    points,
                    mean + cross * residual / innovation,
                    cov - np.outer(cross, cross) / innovation,
                    np.vstack((projection, cross / diagonal)),
                )
        else:
            adjustments = np.zeros(n) if restored_jitter is None else restored_jitter.copy()
            matrix = self._covariance(xx, xx).copy()
            matrix.flat[:: n + 1] += nn + self.training_regularization + adjustments
            jitter = 0.0
            for attempt in range(self.config.max_cholesky_retries + 1):
                try:
                    stabilized = matrix if not jitter else matrix + jitter * np.eye(n)
                    if self.arithmetic == "gpy":
                        factor, info = la.lapack.dpotrf(np.ascontiguousarray(stabilized), lower=1)
                        if info:
                            raise la.LinAlgError(f"Lower Cholesky failed with status {info}.")
                    else:
                        factor = la.cholesky(stabilized, lower=True, check_finite=False)
                    break
                except la.LinAlgError:
                    if attempt == self.config.max_cholesky_retries:
                        raise
                    jitter = self.config.jitter * 10**attempt
                    self._warn_jitter(jitter, f"full {n}-row covariance diagonal")
            self._factor[:n, :n] = factor
            self._whitened[:n] = la.solve_triangular(factor, yy, lower=True, check_finite=False)
            if self.arithmetic == "gpy":
                self._alpha, info = la.lapack.dpotrs(
                    np.asfortranarray(factor), yy[:, None], lower=1
                )
                if info:
                    raise InvalidModelDataError(
                        f"Lower covariance solve failed with status {info}."
                    )
            self._jitter_adjustments = adjustments + jitter
            self._cache = None
        self.X_train, self.y_train, self.observation_noise = (
            x.clone(),
            target.clone(),
            noise.clone(),
        )
        self._count = n
        self.condition_only()

    def condition_only(self) -> dict[str, Any]:
        """Mark exact fixed-parameter conditioning ready; perform no optimization."""
        if not self._count:
            raise ModelNotFitError("Supply at least one training observation first.")
        self.is_fit = True
        self.last_fit_diagnostics = {
            "method": f"exact_{self.conditioning}_conditioning",
            "success": True,
            "training_rows": self._count,
            "optimized_hyperparameters": False,
            "arithmetic": self.arithmetic,
            "training_regularization": self.training_regularization,
        }
        return dict(self.last_fit_diagnostics)

    def fit(self, *, maxiter: int = 100, **optimizer_options: Any) -> dict[str, Any]:
        """Reject hyperparameter fitting; this model is explicitly fixed-kernel."""
        raise InvalidGaussianProcessConfigError(
            "Use condition_only(), not fit(), for a fixed-kernel GP."
        )

    def _moments(self, X: Any, observation_noise: bool) -> tuple[np.ndarray, np.ndarray]:
        if self.arithmetic == "gpy":
            return self._gpy_moments(X, observation_noise, full_covariance=True)
        if not self.is_fit:
            raise ModelNotFitError("Supply training data before posterior inference.")
        points = torch.as_tensor(X, dtype=self.dtype).detach().numpy()
        if (
            points.ndim != 2
            or points.shape[1] != self.input_dimension
            or not np.isfinite(points).all()
        ):
            raise InvalidModelDataError(
                "Prediction inputs must be finite (N, D_model) coordinates."
            )
        indices = None
        if self._cache is not None:
            previous, mean, cov, projection = self._cache
            matches = (points[:, None, :] == previous[None, :, :]).all(axis=-1)
            if matches.any(axis=1).all():
                indices = matches.argmax(axis=1)
        if indices is not None:
            mean, cov = mean[indices].copy(), cov[np.ix_(indices, indices)].copy()
        else:
            assert self.X_train is not None
            cross = self._covariance(self.X_train.numpy(), points)
            projection = la.solve_triangular(
                self._factor[: self._count, : self._count], cross, lower=True, check_finite=False
            )
            mean = projection.T @ self._whitened[: self._count]
            cov = self._covariance(points, points) - projection.T @ projection
            self._cache = points.copy(), mean.copy(), cov.copy(), projection
        if observation_noise:
            if self.config.noise_variance is None:
                raise InvalidModelDataError(
                    "Prediction noise requires a scalar configured noise_variance."
                )
            cov = cov + float(self.config.noise_variance) * np.eye(len(points))
        return mean, (cov + cov.T) / 2

    def _gpy_moments(
        self, X: Any, observation_noise: bool, *, full_covariance: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate fixed full conditioning in the declared source arithmetic."""
        if not self.is_fit:
            raise ModelNotFitError("Supply training data before posterior inference.")
        values = torch.as_tensor(X, dtype=self.dtype)
        if values.requires_grad:
            raise InvalidModelDataError("Fixed GPy arithmetic does not support input gradients.")
        points = values.numpy()
        if (
            points.ndim != 2
            or points.shape[1] != self.input_dimension
            or not np.isfinite(points).all()
        ):
            raise InvalidModelDataError(
                "Prediction inputs must be finite (N, D_model) coordinates."
            )
        assert self.X_train is not None
        cross = self._covariance(self.X_train.numpy(), points)
        mean = np.dot(cross.T, self._alpha).reshape(-1)
        projection, info = la.lapack.dtrtrs(
            np.asfortranarray(self._factor[: self._count, : self._count]), cross, lower=1
        )
        if info:
            raise InvalidModelDataError(f"Lower triangular solve failed with status {info}.")
        if full_covariance:
            product = la.blas.dsyrk(alpha=1.0, a=np.asfortranarray(projection.T), lower=0)
            upper = np.triu_indices(len(points), k=1)
            product.T[upper] = product[upper]
            uncertainty = self._covariance(points, points) - np.ascontiguousarray(product)
        else:
            with torch.no_grad():
                diagonal = self._kernel(values, diag=True).numpy()
            uncertainty = diagonal - np.square(projection).sum(0)
        if observation_noise:
            if self.config.noise_variance is None:
                raise InvalidModelDataError("Prediction noise requires scalar noise_variance.")
            uncertainty = uncertainty + float(self.config.noise_variance) * (
                np.eye(len(points)) if full_covariance else 1.0
            )
        return mean, uncertainty

    def predict(
        self, X: Any, *, observation_noise: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact posterior mean and standard deviation in observation units."""
        if self.arithmetic == "gpy":
            mean, variance = self._gpy_moments(X, observation_noise, full_covariance=False)
            if bool((variance < 0).any()):
                warnings.warn(
                    "Basat fixed-kernel GP: negative marginal variance from floating-point "
                    f"roundoff (minimum {float(variance.min()):.6g}) was clipped to zero.",
                    BasatNumericalWarning,
                    stacklevel=2,
                )
            return torch.from_numpy(mean), torch.from_numpy(np.sqrt(np.maximum(variance, 0)))
        mean, covariance = self._moments(X, observation_noise)
        return torch.from_numpy(mean.copy()), torch.from_numpy(
            np.sqrt(np.maximum(np.diag(covariance), 0))
        )

    def posterior(self, X: Any, *, observation_noise: bool = False) -> GPyTorchPosterior:
        """Return the fixed-kernel joint posterior; no input gradients are retained."""
        mean, covariance = self._moments(X, observation_noise)
        return GPyTorchPosterior(
            gpytorch.distributions.MultivariateNormal(
                torch.from_numpy(mean.copy()), DenseLinearOperator(torch.from_numpy(covariance))
            )
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore observations and their original numerical regularization.

        Previously reported jitter is not added or warned about a second time.
        Its saved row-specific diagonal is used when rebuilding the factor.
        """
        if (
            state.get("state_version") != self.STATE_VERSION
            or state.get("input_dimension") != self.input_dimension
        ):
            raise ModelStateError("Incompatible fixed-kernel GP checkpoint.")
        inference = state.get(
            "fixed_kernel_inference",
            {"conditioning": "incremental", "arithmetic": "native", "training_regularization": 0.0},
        )
        if inference != self._inference_settings():
            raise ModelStateError("Checkpoint fixed-kernel inference protocol differs.")
        stored_kernel = state.get("fixed_kernel_state")
        current_kernel = self._kernel.state_dict()
        if (
            not isinstance(stored_kernel, dict)
            or set(stored_kernel) != set(current_kernel)
            or any(
                not torch.equal(stored_kernel[key], value) for key, value in current_kernel.items()
            )
        ):
            raise ModelStateError("Checkpoint fixed covariance differs from this model's kernel.")
        raw_jitter = state.get("fixed_kernel_jitter")
        restored_jitter = None
        if raw_jitter is not None:
            try:
                restored_jitter = torch.as_tensor(raw_jitter, dtype=self.dtype).numpy().copy()
                count = 0 if state.get("X_train") is None else len(state["X_train"])
                if (
                    restored_jitter.shape != (count,)
                    or not np.isfinite(restored_jitter).all()
                    or (restored_jitter < 0).any()
                ):
                    raise ValueError("Invalid row-specific numerical variance.")
            except (TypeError, ValueError, RuntimeError) as error:
                raise ModelStateError("Invalid fixed-kernel jitter checkpoint.") from error
        self.reset()
        if state.get("X_train") is not None:
            self._set_data(
                state["X_train"],
                state["y_train"],
                state.get("observation_noise"),
                restored_jitter=restored_jitter,
                restoring=True,
            )

    def state_dict(self) -> dict[str, Any]:
        """Serialize observed data and exact fixed kernel identity, not an oracle."""
        state = super().state_dict()
        state["fixed_kernel_state"] = {
            key: value.detach().clone() for key, value in self._kernel.state_dict().items()
        }
        state["fixed_kernel_jitter"] = self.jitter_adjustments
        state["fixed_kernel_inference"] = self._inference_settings()
        return state

    def _inference_settings(self) -> dict[str, Any]:
        return dict(
            conditioning=self.conditioning,
            arithmetic=self.arithmetic,
            training_regularization=self.training_regularization,
        )

    def reset(self) -> None:
        """Clear the observed dataset, factorization, and candidate cache."""
        self._count = 0
        self._cache = None
        self._factor = np.empty((0, 0), dtype=np.float64, order="F")
        self._whitened = np.empty(0)
        self._jitter_adjustments = np.empty(0)
        self._alpha = np.empty((0, 1))
        self.X_train = self.y_train = self.observation_noise = None
        self.is_fit = False
        self.last_fit_diagnostics = {}
