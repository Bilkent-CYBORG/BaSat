"""Basat-owned source-parameterized RBF inference for good-action algorithms.

The model receives bounds and observations, never an objective function. Source
compatibility switches are explicit; the default uses fitted hyperparameters
and a conditional posterior. Numeric dependencies are internal to Basat.
"""

from __future__ import annotations

import copy
import hashlib
import math
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Literal

import gpytorch
import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.spatial.distance
import torch

from basat.warnings import BasatNumericalWarning, translate_backend_warnings

from .GaussianProcess import GaussianProcess, GaussianProcessConfig, ModelStateError
from .utils import Bounds


@dataclass(frozen=True)
class GoodActionGPConfig:
    """Explicit h/v RBF model used in the Cai et al. source experiment.

    The kernel is v*exp(-squared_unit_distance/h). Outcomes use population
    standardization and fixed regularization in standardized units. Refit
    cadence belongs to GoodActionConfig, not this model. Source-code
    compatibility switches deliberately differ from a correct conditional GP;
    they are never enabled implicitly.

    Notes
    -----
    ``apply_fitted_parameters=False`` and ``sampling_covariance='prior'``
    reproduce two released model-policy choices, not its entire numerical
    runner. They do not select its shared MT19937 stream, NumPy SVD sampler,
    float32 candidate arithmetic, deduplication, or stochastic local search.
    Do not interpret these flags alone as released-decision parity.

    Examples
    --------
    >>> model = basat.GoodActionGPConfig(
    ...     initial_bandwidth=.04, initial_amplitude=1.,
    ...     bandwidth_bounds=(.001, 1.), amplitude_bounds=(.05, 1.5),
    ...     regularizer=1e-8, fit_restarts=10, fit_maxfun=200)
    """
    initial_bandwidth: float = 0.04
    initial_amplitude: float = 1.0
    bandwidth_bounds: tuple[float, float] = (0.001, 1.0)
    amplitude_bounds: tuple[float, float] = (0.05, 1.5)
    regularizer: float = 1e-8
    fit_restarts: int = 10
    fit_maxfun: int = 200
    apply_fitted_parameters: bool = True
    sampling_covariance: Literal["posterior", "prior"] = "posterior"

    def __post_init__(self):
        for initial, pair in ((self.initial_bandwidth, self.bandwidth_bounds),
                              (self.initial_amplitude, self.amplitude_bounds)):
            if len(pair) != 2 or not all(math.isfinite(v) and v > 0 for v in pair) \
                    or not pair[0] <= initial <= pair[1]:
                raise ValueError("Positive initial parameters must lie within ordered bounds.")
        if not math.isfinite(self.regularizer) or self.regularizer <= 0:
            raise ValueError("regularizer must be finite and positive.")
        for count in (self.fit_restarts, self.fit_maxfun):
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("Fitting budgets must be positive integers.")
        if not isinstance(self.apply_fitted_parameters, bool):
            raise ValueError("apply_fitted_parameters must be Boolean.")
        if self.sampling_covariance not in {"posterior", "prior"}:
            raise ValueError("sampling_covariance must be posterior or prior.")


class _CertifiedMarginalCache:
    """Cache equivalent dense conditioning only after a numerical certificate.

    ``reference`` must be the original GPyTorch marginal routine, accepting a
    two-dimensional tensor and returning mean and standard deviation. Cache
    identity includes model identity, all data bytes, hyperparameters, outcome
    scaling, and likelihood noise, so replacing data with the same row count
    or changing fitted values cannot reuse stale conditioning.
    """

    condition_limit = 1e6
    mean_absolute_tolerance = 1e-9
    variance_absolute_tolerance = 1e-10

    def __init__(self):
        self.invalidate()

    def invalidate(self):
        self.key = None
        self.enabled = False
        self.diagnostics = {"enabled": False, "reason": "not_initialized"}

    def _moments(self, points):
        cross = self.amplitude * np.exp(
            -0.5 * scipy.spatial.distance.cdist(self.X, points, "sqeuclidean") / self.ell**2
        )
        solved = scipy.linalg.solve_triangular(self.factor, cross, lower=True, check_finite=False)
        mean = self.mean + cross.T @ self.alpha * self.std
        variance = (self.amplitude - np.square(solved).sum(0)) * self.std**2
        if np.any(variance < 0):
            warnings.warn(
                "Basat numerical event during normalized RBF GP marginals: "
                f"minimum raw variance {float(np.min(variance)):.3e}; rounding "
                "to the configured GPyTorch variance floor "
                f"{gpytorch.settings.min_variance.value(torch.float64):.3e}. "
                "No diagonal jitter was added.",
                BasatNumericalWarning,
                stacklevel=2,
            )
        variance = np.maximum(variance, gpytorch.settings.min_variance.value(torch.float64))
        return mean, variance

    def predict_or_none(self, gp, points, reference):
        if points.device.type != "cpu" or points.dtype != torch.float64:
            return None
        X = gp.X_train.detach().cpu().numpy()
        y = gp.y_train.detach().cpu().numpy()
        if X.ndim != 2 or X.shape[1] != 2 or points.ndim != 2 or points.shape[1] != 2:
            return None
        covariance = gp.model.covar_module
        if (
            not isinstance(gp.model.mean_module, gpytorch.means.ZeroMean)
            or not isinstance(covariance, gpytorch.kernels.ScaleKernel)
            or not isinstance(covariance.base_kernel, gpytorch.kernels.RBFKernel)
            or not isinstance(gp.likelihood, gpytorch.likelihoods.FixedNoiseGaussianLikelihood)
            or hasattr(gp.model, "input_transform")
        ):
            return None
        ell = covariance.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)
        noise = gp.likelihood.noise_covar.noise.detach().cpu().numpy()
        if len(ell) != 1 or not np.all(noise == gp.source_config.regularizer):
            return None
        amplitude = float(covariance.outputscale.detach())
        mean = float(gp.model.outcome_transform.means.reshape(-1)[0])
        std = float(gp.model.outcome_transform.stdvs.reshape(-1)[0])
        parameters = np.array([ell[0], amplitude, mean, std], dtype=np.float64)
        digest = hashlib.sha256()
        for array in (X, y, parameters, noise):
            digest.update(np.ascontiguousarray(array).tobytes())
        key = (id(gp.model), X.shape, y.shape, digest.digest())
        if key != self.key:
            self.key, self.enabled = key, False
            self.diagnostics = {
                "enabled": False,
                "condition_limit": self.condition_limit,
                "mean_absolute_tolerance": self.mean_absolute_tolerance,
                "variance_absolute_tolerance": self.variance_absolute_tolerance,
                "training_rows": len(y),
            }
            self.X, self.ell = X.copy(), float(ell[0])
            self.amplitude, self.mean, self.std = amplitude, mean, std
            matrix = amplitude * np.exp(
                -0.5 * scipy.spatial.distance.cdist(X, X, "sqeuclidean") / self.ell**2
            )
            matrix.flat[:: len(X) + 1] += gp.source_config.regularizer
            try:
                condition = float(np.linalg.cond(matrix))
                self.diagnostics["condition_number"] = condition
                if not np.isfinite(condition) or condition > self.condition_limit:
                    self.diagnostics["reason"] = "condition_limit"
                    return None
                self.factor = scipy.linalg.cholesky(matrix, lower=True, check_finite=True)
                self.alpha = scipy.linalg.cho_solve((self.factor, True), (y - mean) / std)
                # Deterministic checks include observed points (small predictive
                # variance), boundaries and off-design interior points. This
                # does not consume the algorithm, fit, or sampling RNG streams.
                indices = np.linspace(0, len(X) - 1, min(len(X), 8), dtype=int)
                fixed = np.array(
                    [
                        [-5.12, -5.12],
                        [-5.12, 5.12],
                        [5.12, -5.12],
                        [5.12, 5.12],
                        [0.0, 0.0],
                        [-3.71, 1.43],
                        [0.91, -2.17],
                        [2.61, 3.37],
                    ]
                )
                # The reference design is mapped to caller bounds; it is only a
                # numerical certificate and never an objective evaluation.
                fixed = gp.bounds.lower.numpy() + (fixed + 5.12) / 10.24 * gp.widths
                certificate_points = np.concatenate((X[indices], fixed))
                dense_mean, dense_variance = self._moments(certificate_points)
                actual_mean, actual_std = reference(
                    torch.tensor(certificate_points, dtype=torch.float64)
                )
                mean_error = float(np.max(np.abs(actual_mean.detach().numpy() - dense_mean)))
                variance_error = float(
                    np.max(np.abs(actual_std.detach().numpy() ** 2 - dense_variance))
                )
                self.diagnostics.update(
                    certificate_points=len(certificate_points),
                    maximum_mean_error=mean_error,
                    maximum_variance_error=variance_error,
                )
                if (
                    not np.isfinite(mean_error + variance_error)
                    or mean_error > self.mean_absolute_tolerance
                    or variance_error > self.variance_absolute_tolerance
                ):
                    self.diagnostics["reason"] = "marginal_certificate_failed"
                    return None
                self.enabled = True
                self.diagnostics.update(enabled=True, reason="certificate_passed")
            except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
                self.diagnostics["reason"] = type(exc).__name__
                return None
        if not self.enabled:
            return None
        mean, variance = self._moments(points.detach().cpu().numpy())
        if not np.isfinite(mean).all() or not np.isfinite(variance).all():
            self.enabled = False
            self.diagnostics.update(enabled=False, reason="nonfinite_prediction")
            return None
        return torch.from_numpy(mean), torch.from_numpy(np.sqrt(variance))


class GoodActionGaussianProcess(GaussianProcess):
    """SE GP with the source's h/v parameterization and paper-correct refits.

    The source uses k=v*exp(-squared_unit_distance/h), population outcome
    standardization and 1e-8 regularization in standardized units. The source
    runner discards the fitted pair; this adapter applies it, as the paper
    requires. Basat still handles conditioning and all four acquisitions.
    """

    def __init__(self, config: GoodActionGPConfig, *, bounds: Bounds,
                 seed: int = 1, dtype: str = "float64", device: str = "cpu"):
        if dtype != "float64" or device != "cpu":
            raise ValueError("The source-parameterized model currently requires CPU float64.")
        self.source_config = config
        self.bounds = bounds.to(dtype=torch.float64, device="cpu")
        self.widths = (self.bounds.upper - self.bounds.lower).numpy()
        self.fit_seed = seed
        same_width = bool(np.all(self.widths == self.widths[0]))
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(
            ard_num_dims=None if same_width else len(self.widths))).double()
        width = float(self.widths[0]) if same_width else torch.tensor(self.widths)
        kernel.base_kernel.lengthscale = width * math.sqrt(config.initial_bandwidth / 2)
        kernel.outputscale = config.initial_amplitude
        model_config = GaussianProcessConfig(
            kernel=kernel, mean="zero", noise_mode="fixed",
            noise_variance=config.regularizer, standardize_outcomes=True,
            jitter=config.regularizer, dtype=dtype, device=device)
        super().__init__(model_config, input_dimension=len(self.widths))
        self.fit_rng = np.random.default_rng(seed)
        self.marginal_cache = _CertifiedMarginalCache()
        if not config.apply_fitted_parameters or config.sampling_covariance != "posterior":
            warnings.warn(
                "Basat released-code GP compatibility: discarded fitted parameters "
                "and/or prior-covariance draws are explicitly enabled. This does not "
                "implement the paper's fitted conditional GP.", BasatNumericalWarning,
                stacklevel=2)

    def set_data(self, X, y, noise_variance=None):
        # Observations are noiseless. This 1e-8 is the stated model
        # regularizer, not environmental noise or emergency jitter.
        if noise_variance is not None:
            declared = torch.as_tensor(noise_variance)
            if not torch.all((declared == 0) | (declared == self.source_config.regularizer)):
                raise ValueError("The source RBF model requires noiseless observations; its "
                                 "regularizer is a separate standardized model variance.")
        super().set_data(X, y, None)
        self.marginal_cache.invalidate()
        assert self.model is not None and self.y_train is not None and self.likelihood is not None
        transform: Any = self.model.outcome_transform
        mean = self.y_train.mean()
        std = self.y_train.std(correction=0).clamp_min(1e-12)
        with torch.no_grad():
            transform.means.fill_(mean)
            transform.stdvs.fill_(std)
            transform._stdvs_sq.fill_(std.square())
        model: Any = self.model
        model.set_train_data(targets=(self.y_train - mean) / std, strict=False)
        self.likelihood.noise_covar.noise = torch.full_like(self.y_train, self.source_config.regularizer)

    def fit(self, *, maxiter=200, **kwargs):
        self._require_model()
        assert self.X_train is not None and self.model is not None
        if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 1:
            raise ValueError("maxiter must be a positive integer.")
        if kwargs:
            raise TypeError(f"Unsupported normalized-RBF fit arguments: {sorted(kwargs)}")
        X = self.X_train.detach().cpu().numpy() / self.widths
        y = self.model.train_targets.detach().cpu().numpy()
        distances = scipy.spatial.distance.cdist(X, X, "sqeuclidean")

        def objective(theta):
            h, v = theta
            kernel = v * np.exp(-distances / h)
            matrix = kernel + self.source_config.regularizer * np.eye(len(y))
            try:
                factor = scipy.linalg.cho_factor(matrix, lower=True)
                alpha = scipy.linalg.cho_solve(factor, y)
                inv = scipy.linalg.cho_solve(factor, np.eye(len(y)))
            except np.linalg.LinAlgError:
                return 1e100, np.zeros(2)
            loss = 0.5 * y @ alpha + np.log(np.diag(factor[0])).sum()
            difference = inv - np.outer(alpha, alpha)
            grad = 0.5 * np.array(
                [
                    np.sum(difference * kernel * distances / h**2),
                    np.sum(difference * kernel / v),
                ]
            )
            return float(loss), grad

        starts = self.fit_rng.uniform(
            [self.source_config.bandwidth_bounds[0], self.source_config.amplitude_bounds[0]],
            [self.source_config.bandwidth_bounds[1], self.source_config.amplitude_bounds[1]],
            size=(self.source_config.fit_restarts, 2))
        initial = min(starts, key=lambda theta: objective(theta)[0])
        fitted = scipy.optimize.minimize(
            objective,
            initial,
            jac=True,
            method="L-BFGS-B",
            bounds=[self.source_config.bandwidth_bounds, self.source_config.amplitude_bounds],
            options={"maxiter": maxiter, "maxfun": self.source_config.fit_maxfun},
        )
        theta = fitted.x if objective(fitted.x)[0] <= objective(initial)[0] else initial
        h, v = theta
        if self.source_config.apply_fitted_parameters:
            width = (float(self.widths[0]) if np.all(self.widths == self.widths[0])
                     else torch.tensor(self.widths))
            with torch.no_grad():
                self.model.covar_module.base_kernel.lengthscale = width * math.sqrt(h / 2.0)
                self.model.covar_module.outputscale = float(v)
        self.condition_only()
        self.last_fit_diagnostics.update(
            h=float(h), v=float(v), optimizer_success=bool(fitted.success),
            fitted_parameters_applied=self.source_config.apply_fitted_parameters
        )
        return dict(self.last_fit_diagnostics)

    def sample_posterior(self, X, num_samples=1, *, seed=1, observation_noise=False):
        # IID joint conditional draws (not the prior covariance used by the
        # released GP.sample implementation).
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError("num_samples must be a positive integer.")
        self._require_fit()
        assert self.model is not None
        with (
            torch.no_grad(),
            torch.random.fork_rng(),
            gpytorch.settings.max_cholesky_size(2048),
            gpytorch.settings.fast_computations(covar_root_decomposition=False),
            translate_backend_warnings(operation="normalized RBF GP conditional draw"),
        ):
            if seed is not None:
                torch.manual_seed(seed)
            if self.source_config.sampling_covariance == "prior":
                points = self._prediction_inputs(X)
                mean, _ = self.predict(points)
                transform: Any = self.model.outcome_transform
                scale = transform.stdvs.reshape(-1)[0]
                covariance = self.model.covar_module(points).to_dense() * scale.square()
                if observation_noise:
                    covariance = covariance + torch.eye(points.shape[-2], dtype=self.dtype,
                                                         device=points.device) * (
                        self.source_config.regularizer * scale.square())
                distribution = gpytorch.distributions.MultivariateNormal(mean, covariance)
                return distribution.rsample(torch.Size([num_samples]))
            posterior = self.posterior(X, observation_noise=observation_noise)
            return posterior.rsample(torch.Size([num_samples])).squeeze(-1)

    def predict(self, X, *, observation_noise=False):
        # Certified dense marginals accelerate deterministic PG/EG scores and
        # recommendation diagnostics. GS/STS joint samples remain unchanged.
        self._require_fit()
        points = self._prediction_inputs(X)
        if not observation_noise:
            dense = self.marginal_cache.predict_or_none(self, points, self._reference_marginals)
            if dense is not None:
                return dense
        return self._reference_marginals(points, observation_noise=observation_noise)

    def _reference_marginals(self, points, *, observation_noise=False):
        with (
            torch.no_grad(),
            translate_backend_warnings(operation="normalized RBF GP moments"),
        ):
            posterior = self.posterior(points.unsqueeze(-2), observation_noise=observation_noise)
            mean = posterior.mean.squeeze(-1).squeeze(-1)
            variance = posterior.variance.squeeze(-1).squeeze(-1).clamp_min(0.0)
        return mean, variance.sqrt()


    def state_dict(self):
        state = super().state_dict()
        state["source_gp_config"] = asdict(self.source_config)
        state["source_gp_bounds"] = [self.bounds.lower.tolist(), self.bounds.upper.tolist()]
        state["fit_rng_state"] = copy.deepcopy(self.fit_rng.bit_generator.state)
        return state

    def load_state_dict(self, state):
        if state.get("source_gp_config") != asdict(self.source_config):
            raise ModelStateError("Normalized RBF configuration changed.")
        if state.get("source_gp_bounds") != [self.bounds.lower.tolist(), self.bounds.upper.tolist()]:
            raise ModelStateError("Normalized RBF input bounds changed.")
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(state["fit_rng_state"])
        super().load_state_dict(state)
        self.fit_rng = rng
        self.marginal_cache.invalidate()

    def reset(self):
        super().reset()
        self.fit_rng = np.random.default_rng(self.fit_seed)
        self.marginal_cache.invalidate()
