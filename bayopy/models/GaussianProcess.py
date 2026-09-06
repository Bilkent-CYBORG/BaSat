"""Exact single-output Gaussian-process surrogate in explicit model coordinates."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood

from .Kernels import KernelConfig, build_kernel
from .utils import as_tensor, resolve_device, resolve_dtype


class GaussianProcessError(Exception):
    """Base class for BayOPy Gaussian-process failures."""


class InvalidGaussianProcessConfigError(GaussianProcessError, ValueError):
    """Raised when the GP configuration is internally inconsistent."""


class InvalidModelDataError(GaussianProcessError, ValueError):
    """Raised when training or prediction data violate ``D_model``."""


class ModelNotInitializedError(GaussianProcessError, RuntimeError):
    """Raised when an operation requires training data and a built model."""


class ModelNotFitError(GaussianProcessError, RuntimeError):
    """Raised when posterior inference is requested before fitting."""


class ModelFitError(GaussianProcessError, RuntimeError):
    """Raised when hyperparameter optimization fails."""


class ModelStateError(GaussianProcessError, ValueError):
    """Raised when checkpoint state is incompatible or corrupt."""


@dataclass(frozen=True, slots=True)
class GaussianProcessConfig:
    """Configuration of an exact BayOPy Gaussian process.

    Parameters
    ----------
    kernel:
        A declarative :class:`KernelConfig` or a custom GPyTorch kernel.
    mean:
        ``"zero"`` or ``"constant"``.
    noise_mode:
        ``"learned"`` estimates observation noise. ``"fixed"`` requires a
        scalar ``noise_variance`` or per-row values passed to :meth:`set_data`.
    noise_variance:
        Optional non-negative scalar variance.
    fit_method:
        ``"mle"`` ignores registered prior terms; ``"map"`` includes them.
    jitter:
        Positive diagonal stabilization added to every posterior covariance and
        used for guarded training factorizations.
    max_cholesky_retries:
        Number of permitted Cholesky attempts in GPyTorch.
    dtype, device:
        Canonical tensor precision and execution device.
    """

    kernel: KernelConfig | gpytorch.kernels.Kernel = field(default_factory=KernelConfig)
    mean: Literal["zero", "constant"] = "constant"
    noise_mode: Literal["fixed", "learned"] = "learned"
    noise_variance: float | None = None
    fit_method: Literal["mle", "map"] = "mle"
    jitter: float = 1e-6
    max_cholesky_retries: int = 5
    dtype: torch.dtype | str = torch.float32
    device: torch.device | str = "cpu"


class _ExactMarginalLogLikelihoodWithoutPriors(ExactMarginalLogLikelihood):
    """Exact marginal likelihood that omits registered MAP prior penalties."""

    def _add_other_terms(self, result: torch.Tensor, params: Any) -> torch.Tensor:
        """Add registered loss terms while deliberately excluding prior penalties."""

        for added_loss_term in self.model.added_loss_terms():
            result = result.add(added_loss_term.loss(*params))
        return result


class GaussianProcess:
    """Exact GP wrapper whose dimension is supplied by the owning algorithm.

    Parameters
    ----------
    config:
        Surrogate configuration.
    input_dimension:
        Explicit algorithm model dimension ``D_model``.  No environment or
        environment bounds are consulted.
    """

    STATE_VERSION = 1

    def __init__(self, config: GaussianProcessConfig, *, input_dimension: int):
        """Initialize an empty exact GP in an explicit ``D_model`` space."""

        if not isinstance(config, GaussianProcessConfig):
            raise InvalidGaussianProcessConfigError(
                "config must be a GaussianProcessConfig instance."
            )
        if (
            isinstance(input_dimension, bool)
            or not isinstance(input_dimension, int)
            or input_dimension < 1
        ):
            raise InvalidGaussianProcessConfigError("input_dimension must be a positive integer.")
        self._validate_config(config)
        self.config = config
        self.input_dimension = input_dimension
        self.dtype = resolve_dtype(config.dtype)
        self.device = resolve_device(config.device)
        self.X_train: torch.Tensor | None = None
        self.y_train: torch.Tensor | None = None
        self.observation_noise: torch.Tensor | None = None
        self.model: SingleTaskGP | None = None
        self.likelihood: gpytorch.likelihoods.Likelihood | None = None
        self.mll: ExactMarginalLogLikelihood | None = None
        self.is_fit = False
        self.last_fit_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _validate_config(config: GaussianProcessConfig) -> None:
        """Validate immutable configuration independently of training data."""

        if config.mean not in {"zero", "constant"}:
            raise InvalidGaussianProcessConfigError("mean must be 'zero' or 'constant'.")
        if config.noise_mode not in {"fixed", "learned"}:
            raise InvalidGaussianProcessConfigError("noise_mode must be 'fixed' or 'learned'.")
        if config.fit_method not in {"mle", "map"}:
            raise InvalidGaussianProcessConfigError("fit_method must be 'mle' or 'map'.")
        if not torch.isfinite(torch.tensor(float(config.jitter))) or config.jitter <= 0:
            raise InvalidGaussianProcessConfigError("jitter must be finite and positive.")
        if (
            isinstance(config.max_cholesky_retries, bool)
            or not isinstance(config.max_cholesky_retries, int)
            or config.max_cholesky_retries < 1
        ):
            raise InvalidGaussianProcessConfigError(
                "max_cholesky_retries must be a positive integer."
            )
        if config.noise_variance is not None:
            variance = torch.tensor(float(config.noise_variance))
            if not torch.isfinite(variance) or variance < 0:
                raise InvalidGaussianProcessConfigError(
                    "noise_variance must be finite and non-negative."
                )
        if not isinstance(config.kernel, (KernelConfig, gpytorch.kernels.Kernel)):
            raise InvalidGaussianProcessConfigError(
                "kernel must be a KernelConfig or GPyTorch Kernel."
            )

    @property
    def botorch_model(self) -> SingleTaskGP:
        """Return the constructed BoTorch model for acquisition functions."""

        self._require_model()
        assert self.model is not None
        return self.model

    def set_data(self, X: Any, y: Any, noise_variance: Any = None) -> None:
        """Replace model-space training data and rebuild the exact GP.

        Parameters
        ----------
        X:
            Training inputs of shape ``(N, D_model)``.
        y:
            Scalar observations with ``N`` elements.
        noise_variance:
            Optional scalar or length-``N`` vector of known variances.
        """

        inputs, targets, noise = self._validate_data(X, y, noise_variance)
        self.X_train = inputs
        self.y_train = targets
        self.observation_noise = noise
        self.is_fit = False
        self.last_fit_diagnostics = {}
        self._rebuild_model()

    def _validate_data(
        self, X: Any, y: Any, noise_variance: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Convert and validate one complete model-space training table."""

        inputs = as_tensor(X, dtype=self.dtype, device=self.device)
        targets = as_tensor(y, dtype=self.dtype, device=self.device).reshape(-1)
        if inputs.ndim != 2:
            raise InvalidModelDataError("X must have shape (N, D_model).")
        if inputs.shape[1] != self.input_dimension:
            raise InvalidModelDataError(
                f"Expected D_model={self.input_dimension}, got {inputs.shape[1]}."
            )
        if inputs.shape[0] == 0:
            raise InvalidModelDataError("At least one training row is required.")
        if inputs.shape[0] != targets.shape[0]:
            raise InvalidModelDataError("X and y row counts do not match.")
        if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
            raise InvalidModelDataError("Training data must be finite.")
        noise = self._prepare_noise(noise_variance, inputs.shape[0])
        if self.config.noise_mode == "fixed" and noise is None:
            raise InvalidModelDataError(
                "Fixed-noise mode requires config.noise_variance or noise_variance data."
            )
        return inputs, targets, noise

    def _prepare_noise(self, noise_variance: Any, row_count: int) -> torch.Tensor | None:
        """Resolve per-row known observation variances."""

        value = self.config.noise_variance if noise_variance is None else noise_variance
        if value is None:
            return None
        noise = as_tensor(value, dtype=self.dtype, device=self.device).reshape(-1)
        if noise.numel() == 1:
            noise = noise.expand(row_count).clone()
        if noise.shape != (row_count,):
            raise InvalidModelDataError(
                f"noise_variance must be scalar or have shape ({row_count},)."
            )
        if not torch.isfinite(noise).all() or (noise < 0).any():
            raise InvalidModelDataError("noise_variance must contain finite non-negative values.")
        return noise

    def _new_kernel(self) -> gpytorch.kernels.Kernel:
        """Construct an unshared covariance module for a rebuilt model."""

        if isinstance(self.config.kernel, KernelConfig):
            return build_kernel(self.config.kernel, input_dimension=self.input_dimension)
        return copy.deepcopy(self.config.kernel)

    def _rebuild_model(self) -> None:
        """Build BoTorch and GPyTorch objects for the current training table."""

        if self.X_train is None or self.y_train is None:
            self.model = None
            self.likelihood = None
            self.mll = None
            return
        train_y = self.y_train.unsqueeze(-1)
        effective_noise = (
            None
            if self.observation_noise is None
            else self.observation_noise.clamp_min(self.config.jitter)
        )
        likelihood = None
        train_yvar = None
        if self.config.noise_mode == "fixed":
            assert effective_noise is not None
            train_yvar = effective_noise.unsqueeze(-1)
        elif effective_noise is not None:
            # Known measurement variance and an additional learned noise term are
            # distinct sources of uncertainty, so neither is silently discarded.
            likelihood = FixedNoiseGaussianLikelihood(
                noise=effective_noise, learn_additional_noise=True
            )
        mean_module = ZeroMean() if self.config.mean == "zero" else ConstantMean()
        try:
            self.model = SingleTaskGP(
                self.X_train,
                train_y,
                train_Yvar=train_yvar,
                likelihood=likelihood,
                covar_module=self._new_kernel(),
                mean_module=mean_module,
                outcome_transform=None,
            ).to(dtype=self.dtype, device=self.device)
        except (RuntimeError, ValueError) as exc:
            raise InvalidModelDataError(f"Unable to construct Gaussian process: {exc}") from exc
        self.likelihood = self.model.likelihood
        self.mll = self._make_mll(self.config.fit_method)

    def _make_mll(self, method: Literal["mle", "map"]) -> ExactMarginalLogLikelihood:
        """Create an objective with or without registered prior penalties."""

        assert self.likelihood is not None and self.model is not None
        cls = (
            _ExactMarginalLogLikelihoodWithoutPriors
            if method == "mle"
            else ExactMarginalLogLikelihood
        )
        return cls(self.likelihood, self.model)

    def add_data(self, X: Any, y: Any, noise_variance: Any = None) -> None:
        """Append model-space rows and rebuild the exact GP.

        Parameters
        ----------
        X:
            One row ``(D_model,)`` or table ``(N, D_model)``.
        y:
            Scalar or length-``N`` finite targets.
        noise_variance:
            Optional non-negative scalar or length-``N`` known variances.
        """

        new_X = as_tensor(X, dtype=self.dtype, device=self.device)
        if new_X.ndim == 1:
            new_X = new_X.unsqueeze(0)
        new_y = as_tensor(y, dtype=self.dtype, device=self.device).reshape(-1)
        if self.X_train is None or self.y_train is None:
            self.set_data(new_X, new_y, noise_variance)
            return
        # Validate the appended block independently before combining it.
        new_X, new_y, new_noise = self._validate_data(new_X, new_y, noise_variance)
        old_noise = self.observation_noise
        if old_noise is None and new_noise is not None:
            if self.config.noise_variance is None:
                raise InvalidModelDataError(
                    "Cannot mix rows with and without known noise variance."
                )
            old_noise = self._prepare_noise(self.config.noise_variance, self.X_train.shape[0])
        if old_noise is not None and new_noise is None:
            if self.config.noise_variance is None:
                raise InvalidModelDataError(
                    "Cannot mix rows with and without known noise variance."
                )
            new_noise = self._prepare_noise(self.config.noise_variance, new_X.shape[0])
        combined_noise = (
            None
            if old_noise is None
            else torch.cat((old_noise, new_noise if new_noise is not None else old_noise[:0]))
        )
        self.set_data(
            torch.cat((self.X_train, new_X)),
            torch.cat((self.y_train, new_y)),
            combined_noise,
        )

    def fit(self, *, maxiter: int = 100, **optimizer_options: Any) -> dict[str, Any]:
        """Fit hyperparameters using the configured MLE or MAP objective.

        Parameters
        ----------
        maxiter:
            Positive optimizer iteration limit.
        **optimizer_options:
            Additional options forwarded to the BoTorch fitter.

        Returns
        -------
        dict[str, Any]
            Timing, method, convergence, and training-row diagnostics.
        """

        if self.config.fit_method == "mle":
            return self.fit_mle(maxiter=maxiter, **optimizer_options)
        return self.fit_map(maxiter=maxiter, **optimizer_options)

    def fit_mle(self, *, maxiter: int = 100, **optimizer_options: Any) -> dict[str, Any]:
        """Fit by maximizing marginal likelihood without registered priors.

        Parameters
        ----------
        maxiter:
            Positive optimizer iteration limit.
        **optimizer_options:
            Additional options forwarded to the BoTorch fitter.

        Returns
        -------
        dict[str, Any]
            Fit diagnostics.
        """

        return self._fit("mle", maxiter=maxiter, optimizer_options=optimizer_options)

    def fit_map(self, *, maxiter: int = 100, **optimizer_options: Any) -> dict[str, Any]:
        """Fit by maximizing marginal likelihood plus registered log priors.

        Parameters
        ----------
        maxiter:
            Positive optimizer iteration limit.
        **optimizer_options:
            Additional options forwarded to the BoTorch fitter.

        Returns
        -------
        dict[str, Any]
            Fit diagnostics.
        """

        return self._fit("map", maxiter=maxiter, optimizer_options=optimizer_options)

    def _fit(
        self,
        method: Literal["mle", "map"],
        *,
        maxiter: int,
        optimizer_options: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a guarded BoTorch hyperparameter fit."""

        self._require_model()
        if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 1:
            raise InvalidGaussianProcessConfigError("maxiter must be a positive integer.")
        assert self.model is not None and self.likelihood is not None
        self.mll = self._make_mll(method)
        options = {"maxiter": maxiter, **optimizer_options}
        started = time.perf_counter()
        try:
            self.model.train()
            self.likelihood.train()
            with (
                gpytorch.settings.cholesky_jitter(
                    float_value=self.config.jitter,
                    double_value=self.config.jitter,
                ),
                gpytorch.settings.cholesky_max_tries(self.config.max_cholesky_retries),
            ):
                fit_gpytorch_mll(self.mll, optimizer_kwargs={"options": options})
        except Exception as exc:
            self.is_fit = False
            self.last_fit_diagnostics = {
                "method": method,
                "success": False,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
            }
            raise ModelFitError(f"Gaussian-process {method.upper()} fit failed: {exc}") from exc
        self.model.eval()
        self.likelihood.eval()
        self.is_fit = True
        self.last_fit_diagnostics = {
            "method": method,
            "success": True,
            "elapsed_seconds": time.perf_counter() - started,
            "training_rows": int(self.X_train.shape[0]),  # type: ignore[union-attr]
            "input_dimension": self.input_dimension,
        }
        return dict(self.last_fit_diagnostics)

    def _require_model(self) -> None:
        """Require a built model and training table."""

        if self.model is None or self.X_train is None or self.y_train is None:
            raise ModelNotInitializedError("Call set_data() before this operation.")

    def _require_fit(self) -> None:
        """Require successfully optimized model hyperparameters."""

        self._require_model()
        if not self.is_fit:
            raise ModelNotFitError("Call fit(), fit_mle(), or fit_map() first.")

    def _prediction_inputs(self, X: Any) -> torch.Tensor:
        """Convert prediction inputs and enforce trailing ``D_model``."""

        inputs = as_tensor(X, dtype=self.dtype, device=self.device)
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        if inputs.ndim < 2 or inputs.shape[-1] != self.input_dimension:
            raise InvalidModelDataError(
                f"Prediction inputs must have trailing D_model={self.input_dimension}."
            )
        if not torch.isfinite(inputs).all():
            raise InvalidModelDataError("Prediction inputs must be finite.")
        return inputs

    def posterior(self, X: Any, *, observation_noise: bool = False) -> Any:
        """Return a positively stabilized posterior at model-space coordinates.

        Parameters
        ----------
        X:
            Points whose trailing dimension is exactly ``D_model``.
        observation_noise:
            Include the likelihood's observation noise when true.

        Returns
        -------
        Any
            BoTorch posterior preserving leading batch dimensions.
        """

        self._require_fit()
        inputs = self._prediction_inputs(X)
        assert self.model is not None
        with (
            gpytorch.settings.cholesky_jitter(
                float_value=self.config.jitter, double_value=self.config.jitter
            ),
            gpytorch.settings.cholesky_max_tries(self.config.max_cholesky_retries),
        ):
            posterior = self.model.posterior(inputs, observation_noise=observation_noise)
            if not isinstance(posterior, GPyTorchPosterior):
                raise InvalidModelDataError(
                    "Single-output Gaussian process returned an unsupported posterior type."
                )
            return GPyTorchPosterior(posterior.mvn.add_jitter(self.config.jitter))

    def predict(
        self, X: Any, *, observation_noise: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior mean and standard deviation in observation units.

        Parameters
        ----------
        X:
            Points whose trailing dimension is exactly ``D_model``.
        observation_noise:
            Include the likelihood's observation noise when true.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Posterior mean and non-negative standard deviation.
        """

        posterior = self.posterior(X, observation_noise=observation_noise)
        mean = posterior.mean.squeeze(-1)
        variance = posterior.variance.squeeze(-1).clamp_min(0.0)
        return mean, variance.sqrt()

    def sample_posterior(
        self,
        X: Any,
        num_samples: int = 1,
        *,
        seed: int = 1,
        observation_noise: bool = False,
    ) -> torch.Tensor:
        """Draw reproducible Sobol-QMC posterior samples.

        Parameters
        ----------
        X:
            Prediction points in explicit model coordinates.
        num_samples:
            Positive number of joint posterior draws.
        seed:
            Non-negative Sobol sampler seed.
        observation_noise:
            Sample noisy observations rather than latent values when true.

        Returns
        -------
        torch.Tensor
            Samples with leading dimension ``num_samples``.
        """

        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
            raise InvalidModelDataError("num_samples must be a positive integer.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise InvalidModelDataError("seed must be a non-negative integer.")
        with (
            gpytorch.settings.cholesky_jitter(
                float_value=self.config.jitter, double_value=self.config.jitter
            ),
            gpytorch.settings.cholesky_max_tries(self.config.max_cholesky_retries),
        ):
            posterior = self.posterior(X, observation_noise=observation_noise)
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]), seed=seed)
            return sampler(posterior).squeeze(-1)

    def state_dict(self) -> dict[str, Any]:
        """Serialize training data, hyperparameters, and fit diagnostics."""

        return {
            "state_version": self.STATE_VERSION,
            "input_dimension": self.input_dimension,
            "dtype": str(self.dtype),
            "X_train": None if self.X_train is None else self.X_train.detach().clone(),
            "y_train": None if self.y_train is None else self.y_train.detach().clone(),
            "observation_noise": (
                None if self.observation_noise is None else self.observation_noise.detach().clone()
            ),
            "model_state": (None if self.model is None else copy.deepcopy(self.model.state_dict())),
            "is_fit": self.is_fit,
            "last_fit_diagnostics": copy.deepcopy(self.last_fit_diagnostics),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a checkpoint after validating model identity and dimensions.

        Parameters
        ----------
        state:
            Complete mapping produced by :meth:`state_dict`.

        Raises
        ------
        ModelStateError
            If version, dimension, training data, or hyperparameters are invalid.
        """

        if not isinstance(state, dict):
            raise ModelStateError("GP state must be a dictionary.")
        if state.get("state_version") != self.STATE_VERSION:
            raise ModelStateError("Unsupported GP state version.")
        if state.get("input_dimension") != self.input_dimension:
            raise ModelStateError("Checkpoint input_dimension does not match this GaussianProcess.")
        X = state.get("X_train")
        y = state.get("y_train")
        if (X is None) != (y is None):
            raise ModelStateError("Checkpoint training data are incomplete.")
        if X is None:
            self.reset()
            return
        try:
            self.set_data(X, y, state.get("observation_noise"))
            model_state = state.get("model_state")
            if model_state is None:
                raise ModelStateError("Checkpoint is missing model_state.")
            assert self.model is not None and self.likelihood is not None
            self.model.load_state_dict(model_state)
            self.is_fit = bool(state.get("is_fit", False))
            self.last_fit_diagnostics = copy.deepcopy(state.get("last_fit_diagnostics", {}))
            self.model.eval() if self.is_fit else self.model.train()
            self.likelihood.eval() if self.is_fit else self.likelihood.train()
        except ModelStateError:
            raise
        except Exception as exc:
            raise ModelStateError(f"Invalid Gaussian-process checkpoint: {exc}") from exc

    def reset(self) -> None:
        """Discard all observations, fitted state, and diagnostics."""

        self.X_train = None
        self.y_train = None
        self.observation_noise = None
        self.model = None
        self.likelihood = None
        self.mll = None
        self.is_fit = False
        self.last_fit_diagnostics = {}


__all__ = [
    "GaussianProcess",
    "GaussianProcessConfig",
    "GaussianProcessError",
    "InvalidGaussianProcessConfigError",
    "InvalidModelDataError",
    "ModelFitError",
    "ModelNotFitError",
    "ModelNotInitializedError",
    "ModelStateError",
]
