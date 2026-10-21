"""Probability of Good for Gaussian-process good-action identification.

The acquisition is the posterior probability that the latent reward exceeds a
fixed aspiration threshold. The first release follows the paper's maximization
convention and keeps all model coordinates in the full environment action space.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Literal

import numpy as np
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from basat.algorithms.algorithm import (
    Algorithm,
    AlgorithmConfigurationError,
    AlgorithmError,
    resolve_batch_configuration,
)
from basat.algorithms.helper import SourceRecord
from basat.models.Acquisitions import (
    DiscreteAcquisitionCandidates,
    optimize_acquisition_continuous,
)
from basat.models.GaussianProcess import GaussianProcess, GaussianProcessConfig
from basat.models.GoodActionGaussianProcess import GoodActionGaussianProcess, GoodActionGPConfig
from basat.models.Kernels import KernelConfig
from basat.models.utils import Bounds
from basat.warnings import BasatNumericalWarning, translate_backend_warnings

from ._good_action_released import ReleasedGoodActionModel
from .Exception.exception_PG import PGCheckpointError, PGException, PGPosteriorError


@dataclass(frozen=True, slots=True)
class GoodActionConfig:
    """Configuration shared by PG, EG, GS, and Satisficing TS.

    Parameters
    ----------
    threshold:
        Latent reward level defining a good action, ``f(x) >= threshold``.
    candidate_domain:
        Optional sub-bounds or explicit finite action tensor. ``None`` uses the
        algorithm's environment-space bounds.
    initial_points:
        Number of initialization rounds. ``None`` uses ``max(2 * D_env, 4)``.
    initial_points_sampling_method:
        ``"paper"``, ``"sobol"``, or ``"uniform"`` initial-design sampling.
    hyperparameter_refit_cycle:
        Number of newly completed evaluations between full GP hyperparameter
        optimizations. Between those optimizations the latest learned
        hyperparameters are reused with the enlarged training table.
    acquisition_restarts:
        Number of starts used by the differentiable continuous PG/EG
        acquisition optimizer. GS and STS use their source-defined finite
        Monte Carlo candidate representation instead.
    source_acquisition_protocol:
        ``"native"`` preserves Basat's default search. ``"cai_2021"`` uses
        the author code's independent restart design for PG/EG, random exact
        Monte Carlo score ties for GS, and a fresh 1000-point conditional
        Thompson fallback for continuous STS. PG/EG still honor the explicit
        ``candidate_samples`` and ``acquisition_restarts`` budgets; the source
        experiment sets these to 1000 and 10. This does not reproduce the
        released prior-covariance bug or its changing-random-function local
        refinement for GS/STS, and never accesses the benchmark.
        ``"cai_2021_released"`` explicitly reproduces the released PG/EG/GS/STS
        numerical/search protocol, including its discarded fitted parameters,
        prior-covariance samples, exact-row deduplication, float32 Sobol
        candidates/prior covariance with float64 observations, shared legacy
        NumPy random stream, and resampled GS/STS local objectives. It is not
        the paper's conditional-GP rule. This mode requires CPU float64,
        maximization, noiseless scalar feedback, three uniform initial points,
        no candidate-domain/reference overrides, ten acquisition restarts,
        a three-query discarded-fit cadence, and the explicit released
        ``source_gp`` flags. Candidate budgets must be 1000 for PG/EG or
        ``100*D`` for GS/STS; GS requires 100 Monte Carlo draws. Unsupported
        combinations raise an error rather than silently replacing settings.
    reference_point:
        STS reference action; the domain center is used when omitted.
    lookahead_samples, maximum_value_samples:
        Legacy GS Monte Carlo budget names; their product is the number of
        posterior draws used to estimate each candidate's good-action probability.
    candidate_samples:
        Size of the reproducible continuous search representation.
    fit_maxiter:
        Maximum optimizer iterations for each exact-GP fit.
    kernel:
        Full-dimensional GP kernel configuration.
    gp_config:
        Optional complete surrogate settings. Overrides ``kernel`` and allows
        fixed observation noise, zero mean, and source-specific kernel
        constraints without changing the acquisition rule.
    source_gp:
        Optional Basat-owned h/v RBF model configuration. Uses normalized
        coordinates, population standardization and the declared fitting
        restart policy. Mutually exclusive with ``gp_config``; no notebook
        subclass or external GP implementation is needed.
    seed:
        Algorithm and candidate-generation seed.
    """

    threshold: float
    candidate_domain: Bounds | torch.Tensor | np.ndarray[Any, Any] | None = None
    initial_points: int | None = None
    initial_points_sampling_method: Literal["paper", "sobol", "uniform"] = "paper"
    hyperparameter_refit_cycle: int = 3
    acquisition_restarts: int = 10
    source_acquisition_protocol: Literal["native", "cai_2021", "cai_2021_released"] = "native"
    reference_point: torch.Tensor | np.ndarray[Any, Any] | None = None
    lookahead_samples: int = 64
    maximum_value_samples: int = 32
    candidate_samples: int = 128
    fit_maxiter: int = 50
    posterior_std_floor: float = 1e-12
    kernel: KernelConfig = field(default_factory=KernelConfig)
    gp_config: GaussianProcessConfig | None = None
    source_gp: GoodActionGPConfig | None = None
    batch_size: int | None = 1
    q_batch: int | None = None
    seed: int = 1

    def __post_init__(self) -> None:
        """Canonicalize NumPy configuration arrays to immutable tensor copies."""

        if isinstance(self.candidate_domain, np.ndarray):
            object.__setattr__(
                self,
                "candidate_domain",
                torch.as_tensor(self.candidate_domain).detach().clone(),
            )
        if isinstance(self.reference_point, np.ndarray):
            object.__setattr__(
                self,
                "reference_point",
                torch.as_tensor(self.reference_point).detach().clone(),
            )


@dataclass(slots=True)
class GoodActionState:
    """Mutable full-coordinate state shared by the good-action methods.

    Attributes
    ----------
    X_train, y_train, observation_noise_variance:
        Physical ``(N, D_env)`` history, responses, and known variances.
    best_posterior_mean_action, best_posterior_mean_value:
        Current latent posterior-mean recommendation and value.
    good_action_candidates:
        Candidate-domain rows currently satisfying the aspiration rule.
    last_score, last_probability_of_good:
        Most recent method acquisition and posterior-goodness diagnostics.
    last_fit_evaluations, last_fit_reused_hyperparameters:
        GP refit-cycle state, retained with learned parameters on checkpoints.
        Later scheduled fits use the currently configured numerical budget.
    last_continuous_optimizer_restarts:
        Number of independently requested acquisition restarts.
    """

    X_train: torch.Tensor
    y_train: torch.Tensor
    observation_noise_variance: torch.Tensor | None
    best_posterior_mean_action: torch.Tensor | None
    best_posterior_mean_value: float | None
    good_action_candidates: list[torch.Tensor]
    last_score: float | None
    last_probability_of_good: float | None
    last_fit_evaluations: int
    last_fit_reused_hyperparameters: bool
    last_continuous_optimizer_restarts: int | None


class _ThresholdAcquisition(AcquisitionFunction):
    """Differentiable one-point PG/EG acquisition over continuous actions."""

    def __init__(
        self,
        *,
        model: Any,
        kind: Literal["pg", "eg"],
        threshold: float,
        std_floor: float,
    ) -> None:
        """Initialize a differentiable threshold objective around ``model``."""

        super().__init__(model=model)
        self.kind = kind
        self.threshold = threshold
        self.std_floor = std_floor

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Return one scalar threshold-acquisition value per t-batch.

        Parameters
        ----------
        X:
            Candidate tensor with trailing ``q=1`` and action dimensions.

        Returns
        -------
        torch.Tensor
            PG or EG threshold scores over the leading t-batch dimensions.
        """

        posterior: Any = self.model.posterior(X)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.squeeze(-1).clamp_min(0.0).sqrt()
        safe_std = std.clamp_min(self.std_floor)
        difference = mean - self.threshold
        if self.kind == "pg":
            # Phi is monotone, so the standardized score avoids saturated CDF
            # gradients without changing the PG maximizer.
            value = difference / safe_std
        else:
            z = difference / safe_std
            normal = torch.distributions.Normal(
                torch.zeros((), dtype=mean.dtype, device=mean.device),
                torch.ones((), dtype=mean.dtype, device=mean.device),
            )
            value = difference * normal.cdf(z) + safe_std * torch.exp(normal.log_prob(z))
            value = torch.where(
                std <= self.std_floor,
                difference.clamp_min(0.0),
                value,
            )
        return value.squeeze(-1)


class PG(Algorithm):
    """Select the action with the largest posterior probability of goodness.

    Examples
    --------
    >>> import basat
    >>> bounds = basat.Bounds(lower=[-1.0, -1.0], upper=[1.0, 1.0])
    >>> algorithm = basat.PG(
    ...     bounds=bounds,
    ...     config=basat.GoodActionConfig(threshold=0.5, initial_points=5, seed=7),
    ... )
    """

    ACQUISITION_POLICY = "posterior probability of exceeding the threshold"
    CHECKPOINT_ADDED_CONFIG_DEFAULTS = MappingProxyType(
        {
            **Algorithm.CHECKPOINT_ADDED_CONFIG_DEFAULTS,
            "gp_config": None,
            "source_acquisition_protocol": "native",
            "source_gp": None,
        }
    )

    EXCEPTION_CLASS = PGException
    native_batching = False
    POSTERIOR_ERROR_TYPE: ClassVar[type[AlgorithmError]] = PGPosteriorError
    CHECKPOINT_ERROR_TYPE: ClassVar[type[AlgorithmError]] = PGCheckpointError
    native_environments = frozenset({"stochastic"})
    SOURCE = SourceRecord(
        paper_title="Lenient Regret and Good-Action Identification in Gaussian Process Bandits",
        paper_year=2021,
        paper_url="https://proceedings.mlr.press/v139/cai21c.html",
        repository_url="https://github.com/caitree/GoodAction",
        repository_status="author",
        notes="Full-dimensional clean-room PG with explicit finite-data adaptation.",
    )

    def __init__(
        self,
        *,
        bounds: Bounds,
        config: GoodActionConfig,
        maximize: bool = True,
        dtype: torch.dtype | str = torch.float32,
        device: torch.device | str = "cpu",
        plot: bool = False,
        verbose: bool = False,
        verbose_setting: bool = False,
        verbose_iter_by_iter: bool = False,
    ) -> None:
        """Initialize PG with a full-coordinate exact GP and immutable config."""

        self.config = config
        self._validate_config(config)
        super().__init__(
            bounds=bounds,
            batch_size=config.batch_size,
            q_batch=config.q_batch,
            maximize=maximize,
            dtype=dtype,
            device=device,
            seed=config.seed,
            initial_points_sampling_method=config.initial_points_sampling_method,
            plot=plot,
            verbose=verbose,
            verbose_setting=verbose_setting,
            verbose_iter_by_iter=verbose_iter_by_iter,
        )
        self.model_input_dimension = self.action_dimension
        self.gp = GaussianProcess(
            config.gp_config
            or GaussianProcessConfig(
                kernel=config.kernel,
                mean="constant",
                noise_mode="learned",
                fit_method="mle",
                dtype=self.dtype,
                device=self.device,
            ),
            input_dimension=self.action_dimension,
        )
        if config.source_acquisition_protocol == "cai_2021_released":
            self._validate_released_configuration()
            assert config.source_gp is not None
            self.gp = ReleasedGoodActionModel(config.source_gp, bounds=bounds, seed=config.seed)
            self.invalidate_source_guarantees("basat.released_nonconditional_gp")
            warnings.warn(
                "Basat released-code compatibility explicitly retains discarded fitted "
                "parameters, prior-covariance draws, mixed precision and stochastic local "
                "refinement. This is not the paper's fitted conditional-GP protocol.",
                BasatNumericalWarning, stacklevel=2,
            )
        elif config.source_gp is not None:
            self.gp = GoodActionGaussianProcess(
                config.source_gp, bounds=bounds, seed=config.seed,
                dtype=str(self.dtype).removeprefix("torch."), device=str(self.device),
            )
            if (not config.source_gp.apply_fitted_parameters
                    or config.source_gp.sampling_covariance != "posterior"):
                self.invalidate_source_guarantees("basat.released_nonconditional_gp")
        self.state = self._empty_state()
        self._last_candidates: torch.Tensor | None = None
        self._last_candidate_indices: torch.Tensor | None = None
        if config.candidate_domain is None:
            self.invalidate_source_guarantees("basat.default_continuous_candidate_domain")

    @staticmethod
    def _validate_config(config: GoodActionConfig) -> None:
        """Validate shared sequential good-action configuration fields."""

        if not isinstance(config, GoodActionConfig):
            raise AlgorithmConfigurationError("config must be GoodActionConfig.")
        if not math.isfinite(float(config.threshold)):
            raise AlgorithmConfigurationError("threshold must be finite.")
        if config.source_acquisition_protocol not in {"native", "cai_2021", "cai_2021_released"}:
            raise AlgorithmConfigurationError(
                "source_acquisition_protocol must be 'native', 'cai_2021', or 'cai_2021_released'."
            )
        if config.gp_config is not None and not isinstance(config.gp_config, GaussianProcessConfig):
            raise AlgorithmConfigurationError("gp_config must be GaussianProcessConfig or None.")
        if config.source_gp is not None:
            if not isinstance(config.source_gp, GoodActionGPConfig):
                raise AlgorithmConfigurationError("source_gp must be GoodActionGPConfig or None.")
            if config.gp_config is not None:
                raise AlgorithmConfigurationError("Choose source_gp or gp_config, not both.")
        _, q_batch = resolve_batch_configuration(
            batch_size=config.batch_size, q_batch=config.q_batch
        )
        if q_batch is not None and q_batch > 1:
            raise AlgorithmConfigurationError(
                "Good-action algorithms do not define a joint q-acquisition; use batch_size "
                "for repeated independent good-action queries."
            )
        integer_fields = {
            "hyperparameter_refit_cycle": config.hyperparameter_refit_cycle,
            "acquisition_restarts": config.acquisition_restarts,
            "lookahead_samples": config.lookahead_samples,
            "maximum_value_samples": config.maximum_value_samples,
            "candidate_samples": config.candidate_samples,
            "fit_maxiter": config.fit_maxiter,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise AlgorithmConfigurationError(f"{name} must be a positive integer.")
        if config.initial_points is not None and (
            isinstance(config.initial_points, bool)
            or not isinstance(config.initial_points, int)
            or config.initial_points < 1
        ):
            raise AlgorithmConfigurationError("initial_points must be positive or None.")
        if not math.isfinite(float(config.posterior_std_floor)) or config.posterior_std_floor <= 0:
            raise AlgorithmConfigurationError("posterior_std_floor must be positive and finite.")

    def _empty_state(self) -> GoodActionState:
        """Create empty mutable state in full environment coordinates."""

        return GoodActionState(
            X_train=torch.empty((0, self.action_dimension), dtype=self.dtype, device=self.device),
            y_train=torch.empty((0,), dtype=self.dtype, device=self.device),
            observation_noise_variance=None,
            best_posterior_mean_action=None,
            best_posterior_mean_value=None,
            good_action_candidates=[],
            last_score=None,
            last_probability_of_good=None,
            last_fit_evaluations=-1,
            last_fit_reused_hyperparameters=False,
            last_continuous_optimizer_restarts=None,
        )

    def _validate_released_configuration(self) -> None:
        """Reject unverified hybrid protocols; no user setting is overridden."""
        name = type(self).__name__
        if name not in {"PG", "EG", "GS", "Satisficing_TS"}:
            raise AlgorithmConfigurationError("Released good-action mode supports PG, EG, GS and STS only.")
        expected_gp = GoodActionGPConfig(apply_fitted_parameters=False, sampling_covariance="prior")
        count = 1000 if name in {"PG", "EG"} else 100 * self.action_dimension
        c = self.config
        if (self.dtype != torch.float64 or self.device.type != "cpu" or not self.maximize
                or c.batch_size != 1 or c.q_batch not in {None, 1}
                or c.initial_points != 3 or c.initial_points_sampling_method != "uniform"
                or c.candidate_domain is not None or c.reference_point is not None
                or c.acquisition_restarts != 10 or c.hyperparameter_refit_cycle != 3
                or c.candidate_samples != count or c.source_gp != expected_gp
                or c.posterior_std_floor != 1e-8 or not 0 <= c.seed < 2**32
                or (name == "GS" and c.lookahead_samples * c.maximum_value_samples != 100)):
            raise AlgorithmConfigurationError(
                "cai_2021_released requires CPU float64 maximization; batch_size=1; "
                "initial_points=3 with uniform sampling; no candidate/reference override; "
                "acquisition_restarts=10; hyperparameter_refit_cycle=3; "
                f"candidate_samples={count}; posterior_std_floor=1e-8; a uint32 seed; "
                "the default GoodActionGPConfig numeric parameters with explicit "
                "apply_fitted_parameters=False and sampling_covariance='prior'; "
                "and 100 total MC samples for GS."
            )

    def _validate_candidate_domain(self) -> None:
        """Validate an optional sub-box or explicit finite candidate table."""

        domain = self.config.candidate_domain
        if domain is None:
            return
        lower, upper = self._bounds_on_algorithm_device()
        if isinstance(domain, Bounds):
            if domain.dimension != self.action_dimension:
                raise AlgorithmConfigurationError("candidate_domain bounds dimension mismatch.")
            candidate_lower = domain.lower.to(dtype=self.dtype, device=self.device)
            candidate_upper = domain.upper.to(dtype=self.dtype, device=self.device)
            if not torch.all(candidate_lower >= lower) or not torch.all(candidate_upper <= upper):
                raise AlgorithmConfigurationError("candidate_domain bounds exceed action bounds.")
            return
        candidates = torch.as_tensor(domain, dtype=self.dtype, device=self.device)
        if candidates.ndim != 2 or candidates.shape[1] != self.action_dimension:
            raise AlgorithmConfigurationError("candidate_domain tensor must have shape (M, D_env).")
        if candidates.shape[0] == 0 or not torch.isfinite(candidates).all():
            raise AlgorithmConfigurationError(
                "candidate_domain tensor must be nonempty and finite."
            )
        if not torch.all((candidates >= lower) & (candidates <= upper)):
            raise AlgorithmConfigurationError("candidate_domain contains an out-of-bounds action.")

    def _initialize_from_environment_impl(self, data: Any) -> None:
        """Load learner-visible history without consulting hidden source data."""

        self._validate_candidate_domain()
        requested = torch.as_tensor(data.actions, dtype=self.dtype, device=self.device)
        evaluated_value = getattr(data, "evaluated_actions", None)
        evaluated = (
            requested
            if evaluated_value is None
            else torch.as_tensor(evaluated_value, dtype=self.dtype, device=self.device)
        )
        observations = torch.as_tensor(
            data.observations, dtype=self.dtype, device=self.device
        ).reshape(-1)
        variance_value = getattr(data, "observation_noise_variance", None)
        variance: torch.Tensor | None = None
        if variance_value is not None:
            candidate_variance = torch.as_tensor(
                variance_value, dtype=self.dtype, device=self.device
            ).reshape(-1)
            if torch.isfinite(candidate_variance).all():
                variance = candidate_variance
        self.state.X_train = evaluated.detach().clone()
        self.state.y_train = observations.detach().clone()
        self.state.observation_noise_variance = variance
        if observations.numel():
            self.gp.set_data(self.state.X_train, self.state.y_train, variance)

    def _prepare_discrete_candidates_impl(
        self,
        *,
        candidate_actions: torch.Tensor,
        candidate_dataset_indices: torch.Tensor,
        round_info: Mapping[str, Any],
    ) -> DiscreteAcquisitionCandidates:
        """Keep complete finite rows unchanged in full GP coordinates."""

        del round_info
        return DiscreteAcquisitionCandidates(
            environment_actions=candidate_actions.detach().clone(),
            model_inputs=candidate_actions.detach().clone(),
            dataset_indices=candidate_dataset_indices.detach().clone(),
        )

    @staticmethod
    def _standardized_goodness(
        mean: torch.Tensor,
        std: torch.Tensor,
        threshold: float,
        *,
        std_floor: float = 1e-12,
    ) -> torch.Tensor:
        """Return ``(mean - threshold) / max(std, std_floor)``."""

        return (mean - threshold) / std.clamp_min(std_floor)

    @classmethod
    def _probability_of_good(
        cls,
        mean: torch.Tensor,
        std: torch.Tensor,
        threshold: float,
        *,
        std_floor: float = 1e-12,
    ) -> torch.Tensor:
        """Return the stable Gaussian posterior probability of goodness."""

        z = cls._standardized_goodness(mean, std, threshold, std_floor=std_floor)
        normal = torch.distributions.Normal(
            torch.zeros((), dtype=mean.dtype, device=mean.device),
            torch.ones((), dtype=mean.dtype, device=mean.device),
        )
        probabilities = normal.cdf(z)
        deterministic = std <= std_floor
        return torch.where(deterministic, (mean >= threshold).to(mean.dtype), probabilities)

    def _fit_model(self) -> None:
        """Fit on schedule and otherwise use the last optimized parameters."""

        if self.state.y_train.numel() == 0:
            return
        if isinstance(self.gp, ReleasedGoodActionModel):
            self.gp.condition_only()
            self.state.last_fit_evaluations = int(self.state.y_train.numel())
            self.state.last_fit_reused_hyperparameters = True
            return
        if self.gp.X_train is None:
            self.gp.set_data(
                self.state.X_train,
                self.state.y_train,
                self.state.observation_noise_variance,
            )
        row_count = int(self.state.y_train.numel())
        rows_since_refit = row_count - self.state.last_fit_evaluations
        refit_due = (
            self.state.last_fit_evaluations < 0
            or rows_since_refit >= self.config.hyperparameter_refit_cycle
        )
        if self.gp.is_fit and not refit_due:
            return
        try:
            self.gp.fit(maxiter=self.config.fit_maxiter)
        except Exception as error:
            raise self.POSTERIOR_ERROR_TYPE(
                f"{type(self).__name__} Gaussian-process fitting failed: {error}",
                algorithm=self,
            ) from error
        self.state.last_fit_evaluations = row_count
        self.state.last_fit_reused_hyperparameters = False

    def _capture_hyperparameters(self) -> dict[str, torch.Tensor]:
        """Copy learned scalar/kernel parameters before rebuilding on new data."""

        if not self.gp.is_fit or self.gp.model is None:
            return {}
        return {
            name: parameter.detach().clone() for name, parameter in self.gp.model.named_parameters()
        }

    def _restore_hyperparameters(self, parameters: Mapping[str, torch.Tensor]) -> bool:
        """Restore compatible learned parameters into the rebuilt exact GP."""

        if not parameters or self.gp.model is None or self.gp.likelihood is None:
            return False
        current = dict(self.gp.model.named_parameters())
        if set(parameters) - set(current):
            return False
        if any(current[name].shape != value.shape for name, value in parameters.items()):
            return False
        with torch.no_grad():
            for name, value in parameters.items():
                current[name].copy_(
                    value.to(dtype=current[name].dtype, device=current[name].device)
                )
        self.gp.model.eval()
        self.gp.likelihood.eval()
        self.gp.is_fit = True
        self.gp.last_fit_diagnostics = {
            "method": "reuse",
            "success": True,
            "training_rows": int(self.state.y_train.numel()),
            "hyperparameters_fit_at_rows": self.state.last_fit_evaluations,
        }
        return True

    def _initial_design_size(self) -> int:
        """Return the explicit or dimension-scaled Sobol initialization size."""

        return self.config.initial_points or max(2 * self.action_dimension, 4)

    def _candidate_set(
        self, round_info: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return finite rows, an explicit domain, or a reproducible Sobol design."""

        if "candidate_actions" in round_info:
            table = self.prepare_discrete_candidates(
                candidate_actions=round_info["candidate_actions"],
                candidate_dataset_indices=round_info["candidate_dataset_indices"],
                round_info=round_info,
            )
            return table.environment_actions, table.dataset_indices
        domain = self.config.candidate_domain
        if domain is not None and not isinstance(domain, Bounds):
            candidates = torch.as_tensor(domain, dtype=self.dtype, device=self.device)
            return candidates.detach().clone(), None
        search_domain = self.bounds if domain is None else domain
        candidates = self._sample_design(
            search_domain,
            self.config.candidate_samples,
            phase=(
                "initial"
                if self.state.y_train.numel() < self._initial_design_size()
                else "acquisition"
            ),
            seed=self.config.seed + self.iteration,
        )
        return candidates, None

    def _score_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
        """Compute the standardized posterior probability-of-good ordering."""

        mean, std = self.gp.predict(candidates)
        return self._standardized_goodness(
            mean,
            std,
            self.config.threshold,
            std_floor=self.config.posterior_std_floor,
        )

    def _continuous_acquisition_kind(self) -> Literal["pg", "eg"]:
        """Return the differentiable threshold objective used by this method."""

        return "pg"

    def _supports_continuous_optimization(self) -> bool:
        """Whether this method has a differentiable continuous acquisition."""

        return True

    def _optimize_continuous_candidate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Run deterministic multistart optimization over the configured box."""

        domain = self.config.candidate_domain
        if domain is None:
            search_domain = self.bounds
        elif isinstance(domain, Bounds):
            search_domain = domain
        else:  # pragma: no cover - guarded by the caller and config validation
            raise self.POSTERIOR_ERROR_TYPE(
                "Continuous optimization requires a box candidate domain.",
                algorithm=self,
            )
        search_bounds = search_domain.to(dtype=self.dtype, device=self.device)
        acquisition = _ThresholdAcquisition(
            model=self.gp.botorch_model,
            kind=self._continuous_acquisition_kind(),
            threshold=self.config.threshold,
            std_floor=self.config.posterior_std_floor,
        )
        if self.config.source_acquisition_protocol == "cai_2021":
            return self._optimize_source_candidate(search_bounds, acquisition)
        devices: list[int] = []
        if self.device.type == "cuda":
            devices = [self.device.index or torch.cuda.current_device()]
        # BoTorch's initializer consumes the process RNG. Isolating and seeding
        # it by round makes pending-ask and post-tell replay independent of
        # unrelated application-level random draws.
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.config.seed + self.iteration)
                candidate, value = optimize_acquisition_continuous(
                    acquisition=acquisition,
                    bounds=search_bounds.as_tensor(),
                    q=1,
                    num_restarts=self.config.acquisition_restarts,
                    raw_samples=self.config.candidate_samples,
                    sampling_method=self.acquisition_sampling_method,
                    sampling_generator=self.generator,
                    sampling_seed=self.config.seed + self.iteration,
                )
        except Exception as error:
            raise self.POSTERIOR_ERROR_TYPE(
                f"{type(self).__name__} continuous acquisition optimization failed: {error}",
                algorithm=self,
            ) from error
        self.state.last_continuous_optimizer_restarts = self.config.acquisition_restarts
        return candidate, value

    def _optimize_source_candidate(
        self, bounds: Bounds, acquisition: _ThresholdAcquisition
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retain the best raw point and refine independent unit-box starts.

        The released PG/EG search takes 1000 Sobol candidates followed by 10
        new Sobol starts, not 10 high-scoring points from the raw pool. The
        explicit config budgets remain authoritative. SciPy's finite-
        difference L-BFGS-B runs in unit coordinates as in the source; every
        objective evaluation is a deterministic GP acquisition, not a reward
        function call. Native user sampling overrides are still respected.
        """
        from scipy.optimize import minimize

        count = self.config.candidate_samples
        design = self._sample_design(
            bounds,
            count + self.config.acquisition_restarts,
            phase="acquisition",
            seed=self.config.seed + self.iteration,
        )
        with torch.no_grad(), translate_backend_warnings(operation="source acquisition screening"):
            scores = self._score_candidates(design[:count]).reshape(-1)
        if not torch.isfinite(scores).all():
            raise self.POSTERIOR_ERROR_TYPE(
                "Source acquisition scores are non-finite.", algorithm=self
            )
        tied = torch.nonzero(scores == scores.max(), as_tuple=False).flatten()
        draw = int(torch.randint(len(tied), (1,), generator=self.generator, device=self.device))
        best = design[int(tied[draw])].detach().clone()
        best_value = float(scores.max())
        lower, width = bounds.lower, bounds.upper - bounds.lower
        starts = ((design[count:] - lower) / width).detach().cpu().numpy()
        # EG in the source is evaluated in standardized outcome units.
        # Positive scaling preserves the maximizer, but not SciPy's stopping
        # tolerances, so match those units when this GP standardizes outcomes.
        objective_scale = 1.0
        targets = self.gp.y_train
        if acquisition.kind == "eg" and self.gp.config.standardize_outcomes and targets is not None:
            objective_scale = float(targets.std(correction=0).clamp_min(1e-12))

        def objective(unit: np.ndarray) -> float:
            point = lower + width * torch.as_tensor(unit, dtype=self.dtype, device=self.device)
            with (
                torch.no_grad(),
                translate_backend_warnings(operation="source acquisition refinement"),
            ):
                return (
                    -float(self._score_candidates(point.reshape(1, -1)).reshape(-1)[0])
                    / objective_scale
                )

        for start in starts:
            result = minimize(
                objective,
                start,
                bounds=[(0.0, 1.0)] * self.action_dimension,
                method="L-BFGS-B",
            )
            if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
                continue
            value = -float(result.fun) * objective_scale
            if value >= best_value:
                best = lower + width * torch.as_tensor(
                    result.x, dtype=self.dtype, device=self.device
                )
                best_value = value
        self.state.last_continuous_optimizer_restarts = self.config.acquisition_restarts
        return best.reshape(1, -1), torch.tensor(best_value, dtype=self.dtype, device=self.device)

    def _select_candidate(
        self,
        candidates: torch.Tensor,
        scores: torch.Tensor,
        dataset_indices: torch.Tensor | None,
    ) -> int:
        """Resolve tolerant score ties by the smallest stable dataset index."""

        maximum = torch.max(scores)
        tolerance = 1e-12 + 1e-9 * abs(float(maximum.item()))
        tied = torch.nonzero(scores >= maximum - tolerance, as_tuple=False).flatten()
        if dataset_indices is None:
            return int(tied[0].item())
        tied_indices = dataset_indices[tied]
        return int(tied[torch.argmin(tied_indices)].item())

    def _ask_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Select one initialization or posterior-goodness action."""

        if isinstance(self.gp, ReleasedGoodActionModel):
            return self._ask_released(round_info=round_info)

        candidates, dataset_indices = self._candidate_set(round_info)
        self._last_candidates = candidates.detach().clone()
        self._last_candidate_indices = (
            None if dataset_indices is None else dataset_indices.detach().clone()
        )
        if self.state.y_train.numel() < self._initial_design_size():
            position = int(self.state.y_train.numel()) % candidates.shape[0]
            score = None
            probability = None
        else:
            self._fit_model()
            is_continuous = "candidate_actions" not in round_info and (
                self.config.candidate_domain is None
                or isinstance(self.config.candidate_domain, Bounds)
            )
            if is_continuous and self._supports_continuous_optimization():
                selected_continuous, optimized_score = self._optimize_continuous_candidate()
                # Retain the raw representation for recommendation diagnostics,
                # while placing the optimizer result at a stable first position.
                candidates = torch.cat((selected_continuous, candidates), dim=0)
                self._last_candidates = candidates.detach().clone()
                position = 0
                score = float(optimized_score.reshape(-1)[0].item())
            else:
                scores = self._score_candidates(candidates)
                position = self._select_candidate(candidates, scores, dataset_indices)
                score = float(scores[position].item())
            mean, std = self.gp.predict(candidates[position : position + 1])
            probability = float(
                self._probability_of_good(
                    mean,
                    std,
                    self.config.threshold,
                    std_floor=self.config.posterior_std_floor,
                )[0].item()
            )
            self._last_acquisition_value = score
        selected = candidates[position]
        self.state.last_score = score
        self.state.last_probability_of_good = probability
        self._record_last_ask(
            actions=selected,
            model_inputs=selected,
            round_info=round_info,
            algorithm_metadata={
                "candidate_position": position,
                "dataset_index": (
                    None if dataset_indices is None else int(dataset_indices[position].item())
                ),
                "score": score,
                "probability_of_good": probability,
            },
        )
        return selected

    def _ask_released(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Dispatch observation-only released arithmetic without extra RNG draws."""
        assert isinstance(self.gp, ReleasedGoodActionModel)
        if ("candidate_actions" in round_info or self.acquisition_sampling_method != "paper"
                or self.initial_points_sampling_method != "uniform"):
            raise self.POSTERIOR_ERROR_TYPE(
                "Released good-action mode requires its continuous source Sobol acquisition "
                "and uniform initialization; finite tables/sampling overrides are unsupported.",
                algorithm=self,
            )
        method = "sts" if type(self).__name__ == "Satisficing_TS" else type(self).__name__.lower()
        selected = torch.from_numpy(self.gp.propose(method, self.config.threshold, 3))
        report = dict(self.gp.last_decision)
        self._last_candidates = selected.reshape(1, -1).clone()
        self._last_candidate_indices = None
        self.state.last_score = report.get("score")
        self.state.last_probability_of_good = None
        self._last_acquisition_value = self.state.last_score
        self.state.last_continuous_optimizer_restarts = report["restarts"] or None
        self._record_last_ask(actions=selected, model_inputs=selected, round_info=round_info,
                              algorithm_metadata={"released_source_protocol": True, **report,
                                                  "evaluation_dtype": report["source_physical_action_dtype"]})
        return selected

    @property
    def _supports_shared_trial_streams(self) -> bool:
        """Whether an experiment may hand off the released proposal streams."""
        return isinstance(self.gp, ReleasedGoodActionModel)

    def _export_random_streams(self) -> dict[str, Any]:
        """Copy released proposal streams at a completed experiment boundary.

        This private orchestration hook neither samples nor includes model or
        objective state. Ordinary independent algorithm runs do not use it.
        """
        if not self._supports_shared_trial_streams:
            raise NotImplementedError("Shared trial streams require cai_2021_released mode.")
        assert isinstance(self.gp, ReleasedGoodActionModel)
        if self.last_ask is not None or self.gp.pending_unit is not None:
            raise ValueError("Cannot export trial streams while a proposal awaits feedback.")
        name, keys, position, has_gaussian, cached_gaussian = self.gp.rng.get_state()
        return {
            "numpy": {"kind": name, "keys": keys.tolist(), "position": int(position),  # type: ignore[attr-defined]  # NumPy stub mis-types the MT19937 legacy tuple.
                      "has_gauss": int(has_gaussian), "cached_gaussian": float(cached_gaussian)},
            "torch": self.gp.torch_state.detach().cpu().clone(),
        }

    def _import_random_streams(self, state: Mapping[str, Any]) -> None:
        """Validate and atomically hand off released streams without a draw."""
        if not self._supports_shared_trial_streams:
            raise NotImplementedError("Shared trial streams require cai_2021_released mode.")
        assert isinstance(self.gp, ReleasedGoodActionModel)
        if self.last_ask is not None or self.gp.pending_unit is not None:
            raise ValueError("Cannot replace trial streams while a proposal awaits feedback.")
        try:
            if not isinstance(state, Mapping) or set(state) != {"numpy", "torch"}:
                raise ValueError("Expected exactly numpy and torch random streams.")
            numpy_state = state["numpy"]
            if not isinstance(numpy_state, Mapping):
                raise ValueError("Expected an MT19937 state mapping.")
            keys_value = numpy_state["keys"]
            if isinstance(keys_value, torch.Tensor):
                keys_value = keys_value.detach().cpu().numpy()
            keys = np.asarray(keys_value)
            position = numpy_state["position"]
            has_gauss = numpy_state["has_gauss"]
            cached = numpy_state["cached_gaussian"]
            if (numpy_state["kind"] != "MT19937" or keys.shape != (624,)
                    or not np.issubdtype(keys.dtype, np.integer)
                    or np.any((keys < 0) | (keys > 2**32 - 1))
                    or isinstance(position, bool) or not isinstance(position, int)
                    or not 0 <= position <= 624
                    or isinstance(has_gauss, bool) or not isinstance(has_gauss, int)
                    or has_gauss not in (0, 1)
                    or isinstance(cached, bool) or not isinstance(cached, (int, float))
                    or not np.isfinite(cached)):
                raise ValueError("Invalid MT19937 state.")
            restored_numpy = np.random.RandomState(0)
            restored_numpy.set_state(("MT19937", keys.astype(np.uint32), position,
                                      has_gauss, float(cached)))
            torch_value = torch.as_tensor(state["torch"], device="cpu")
            if (torch_value.ndim != 1 or torch_value.is_floating_point()
                    or torch_value.dtype == torch.bool
                    or bool(((torch_value < 0) | (torch_value > 255)).any())):
                raise ValueError("Torch RNG state must be a one-dimensional byte sequence.")
            restored_torch = torch.Generator(device="cpu")
            restored_torch.set_state(torch_value.to(dtype=torch.uint8).contiguous())
        except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as error:
            raise ValueError(f"Invalid shared trial random streams: {error}") from error
        self.gp.rng = restored_numpy
        self.gp.torch_state = restored_torch.get_state().clone()

    def _tell_impl(
        self,
        *,
        action: torch.Tensor,
        feedback: Mapping[str, Any],
        round_info: Mapping[str, Any],
    ) -> None:
        """Append evaluated full-coordinate feedback and retain GP parameters."""

        del action, round_info
        if isinstance(self.gp, ReleasedGoodActionModel):
            X = torch.cat((self.state.X_train, feedback["evaluated_action"]), dim=0)
            y = torch.cat((self.state.y_train, feedback["observation"]), dim=0)
            # Validate the exact source proposal/noiseless contract before
            # mutating either the algorithm or the numerical model history.
            self.gp.set_data(X, y, feedback.get("observation_noise_variance"))
            self.state.X_train, self.state.y_train = X, y
            self.state.observation_noise_variance = torch.zeros_like(y)
            self.state.last_fit_reused_hyperparameters = True
            return
        previous_hyperparameters = self._capture_hyperparameters()
        evaluated = feedback["evaluated_action"]
        observations = feedback["observation"]
        variance_value = feedback.get("observation_noise_variance")
        variance = None
        if variance_value is not None:
            candidate_variance = torch.as_tensor(
                variance_value, dtype=self.dtype, device=self.device
            ).reshape(-1)
            if torch.isfinite(candidate_variance).all():
                variance = candidate_variance
        self.state.X_train = torch.cat((self.state.X_train, evaluated), dim=0)
        self.state.y_train = torch.cat((self.state.y_train, observations), dim=0)
        if self.state.observation_noise_variance is not None and variance is not None:
            self.state.observation_noise_variance = torch.cat(
                (self.state.observation_noise_variance, variance), dim=0
            )
        elif self.state.y_train.numel() == observations.numel() and variance is not None:
            self.state.observation_noise_variance = variance.detach().clone()
        else:
            self.state.observation_noise_variance = None
        self.gp.set_data(
            self.state.X_train,
            self.state.y_train,
            self.state.observation_noise_variance,
        )
        self.state.last_fit_reused_hyperparameters = self._restore_hyperparameters(
            previous_hyperparameters
        )

    def _update_recommendation(self) -> None:
        """Recommend the maximum latent posterior-mean candidate."""

        if self.state.y_train.numel() == 0:
            return
        if isinstance(self.gp, ReleasedGoodActionModel) and self.state.y_train.numel() < 3:
            position = int(torch.argmax(self.state.y_train))
            self.state.best_posterior_mean_action = self.state.X_train[position].clone()
            self.state.best_posterior_mean_value = float(self.state.y_train[position])
            return
        self._fit_model()
        if self._last_candidates is not None:
            candidates = self._last_candidates
        elif self.config.candidate_domain is not None and not isinstance(
            self.config.candidate_domain, Bounds
        ):
            candidates = torch.as_tensor(
                self.config.candidate_domain, dtype=self.dtype, device=self.device
            )
        else:
            candidates = self.state.X_train
        mean, std = self.gp.predict(candidates)
        position = int(torch.argmax(mean).item())
        self.state.best_posterior_mean_action = candidates[position].detach().clone()
        self.state.best_posterior_mean_value = float(mean[position].item())
        probabilities = self._probability_of_good(
            mean,
            std,
            self.config.threshold,
            std_floor=self.config.posterior_std_floor,
        )
        good = candidates[probabilities >= 0.5]
        self.state.good_action_candidates = [row.detach().clone() for row in good]

    def _recommendation_impl(self) -> tuple[torch.Tensor | None, float | None]:
        """Return the current latent posterior-mean recommendation."""

        self._update_recommendation()
        return self.state.best_posterior_mean_action, self.state.best_posterior_mean_value

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Expose threshold, fit cadence, acquisition, and goodness diagnostics."""

        return {
            "threshold": self.config.threshold,
            "source_acquisition_protocol": self.config.source_acquisition_protocol,
            "training_rows": int(self.state.y_train.numel()),
            "last_score": self.state.last_score,
            "last_probability_of_good": self.state.last_probability_of_good,
            "good_candidate_count": len(self.state.good_action_candidates),
            "hyperparameter_refit_cycle": self.config.hyperparameter_refit_cycle,
            "hyperparameters_fit_at_rows": self.state.last_fit_evaluations,
            "reused_hyperparameters_after_last_update": (
                self.state.last_fit_reused_hyperparameters
            ),
            "continuous_optimizer_restarts": (self.state.last_continuous_optimizer_restarts),
            "gp_fit": dict(self.gp.last_fit_diagnostics),
            **({"released_source_decision": dict(self.gp.last_decision)}
               if isinstance(self.gp, ReleasedGoodActionModel) else {}),
        }

    def _result_metadata_impl(self) -> Mapping[str, Any]:
        """Describe the aspiration threshold and recommendation semantics."""

        return {
            "threshold": self.config.threshold,
            "posterior_good_candidate_count": len(self.state.good_action_candidates),
            "recommendation_kind": "maximum_latent_posterior_mean",
            **({"released_source_protocol": "cai_2021_released",
                "conditional_gp_guarantees": False,
                "source_revision": "ff0ffe18a72e5eef824254ab630929708dd80455"}
               if isinstance(self.gp, ReleasedGoodActionModel) else {}),
        }

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize observations, candidates, recommendations, and fit cadence."""

        return {
            "config": self._config_state(),
            "gp": self.gp.state_dict(),
            "X_train": self.state.X_train.detach().clone(),
            "y_train": self.state.y_train.detach().clone(),
            "observation_noise_variance": (
                None
                if self.state.observation_noise_variance is None
                else self.state.observation_noise_variance.detach().clone()
            ),
            "best_posterior_mean_action": (
                None
                if self.state.best_posterior_mean_action is None
                else self.state.best_posterior_mean_action.detach().clone()
            ),
            "best_posterior_mean_value": self.state.best_posterior_mean_value,
            "good_action_candidates": [
                value.detach().clone() for value in self.state.good_action_candidates
            ],
            "last_score": self.state.last_score,
            "last_probability_of_good": self.state.last_probability_of_good,
            "last_fit_evaluations": self.state.last_fit_evaluations,
            "last_fit_reused_hyperparameters": (self.state.last_fit_reused_hyperparameters),
            "last_continuous_optimizer_restarts": (self.state.last_continuous_optimizer_restarts),
            "last_candidates": (
                None if self._last_candidates is None else self._last_candidates.detach().clone()
            ),
            "last_candidate_indices": (
                None
                if self._last_candidate_indices is None
                else self._last_candidate_indices.detach().clone()
            ),
        }

    def _config_state(self) -> dict[str, Any]:
        """Return a tensor-safe immutable configuration representation."""

        state = asdict(self.config)
        domain = self.config.candidate_domain
        if isinstance(domain, Bounds):
            state["candidate_domain"] = {
                "kind": "bounds",
                "lower": domain.lower.detach().cpu().clone(),
                "upper": domain.upper.detach().cpu().clone(),
            }
        elif isinstance(domain, torch.Tensor):
            state["candidate_domain"] = domain.detach().cpu().clone()
        if self.config.reference_point is not None:
            state["reference_point"] = (
                torch.as_tensor(self.config.reference_point).detach().cpu().clone()
            )
        state["kernel"] = repr(self.config.kernel)
        return state

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Validate checkpoint fields and coordinate-compatible shapes."""

        del strict
        required = {
            "config",
            "X_train",
            "y_train",
            "observation_noise_variance",
            "best_posterior_mean_action",
            "best_posterior_mean_value",
            "good_action_candidates",
            "last_score",
            "last_probability_of_good",
            "last_fit_evaluations",
            "last_fit_reused_hyperparameters",
            "last_continuous_optimizer_restarts",
            "last_candidates",
            "last_candidate_indices",
        }
        missing = required - set(state)
        if missing:
            raise self.CHECKPOINT_ERROR_TYPE(
                f"{type(self).__name__} state is missing {sorted(missing)}.",
                algorithm=self,
            )
        X = torch.as_tensor(state["X_train"], dtype=self.dtype, device=self.device)
        y = torch.as_tensor(state["y_train"], dtype=self.dtype, device=self.device)
        if X.ndim != 2 or X.shape[1] != self.action_dimension or y.shape != (X.shape[0],):
            raise self.CHECKPOINT_ERROR_TYPE(
                f"{type(self).__name__} training state has incompatible shapes.",
                algorithm=self,
            )
        if isinstance(self.gp, ReleasedGoodActionModel):
            if "gp" not in state:
                raise self.CHECKPOINT_ERROR_TYPE("Released checkpoint requires its complete GP/RNG state.",
                                                 algorithm=self)
            assert self.config.source_gp is not None
            checked = ReleasedGoodActionModel(self.config.source_gp, bounds=self.bounds, seed=self.config.seed)
            try:
                checked.load_state_dict(dict(state["gp"]))
                same = (checked.X_train is None and len(X) == 0) or (
                    checked.X_train is not None and checked.y_train is not None
                    and torch.equal(checked.X_train, X) and torch.equal(checked.y_train, y))
                if not same:
                    raise ValueError("Released GP and algorithm observation tables disagree.")
            except Exception as error:
                raise self.CHECKPOINT_ERROR_TYPE(f"Invalid released checkpoint: {error}",
                                                 algorithm=self) from error
        last_fit = state["last_fit_evaluations"]
        if (
            isinstance(last_fit, bool)
            or not isinstance(last_fit, int)
            or last_fit < -1
            or last_fit > X.shape[0]
        ):
            raise self.CHECKPOINT_ERROR_TYPE(
                "last_fit_evaluations is inconsistent with the training table.",
                algorithm=self,
            )
        if not isinstance(state["last_fit_reused_hyperparameters"], bool):
            raise self.CHECKPOINT_ERROR_TYPE(
                "last_fit_reused_hyperparameters must be Boolean.",
                algorithm=self,
            )
        restarts = state["last_continuous_optimizer_restarts"]
        if restarts is not None and (
            isinstance(restarts, bool) or not isinstance(restarts, int) or restarts < 1
        ):
            raise self.CHECKPOINT_ERROR_TYPE(
                "last_continuous_optimizer_restarts must be positive or None.",
                algorithm=self,
            )

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Restore method data and the GP parameters governing its refit cadence."""

        del strict
        self.state.X_train = (
            torch.as_tensor(state["X_train"], dtype=self.dtype, device=self.device).detach().clone()
        )
        self.state.y_train = (
            torch.as_tensor(state["y_train"], dtype=self.dtype, device=self.device).detach().clone()
        )
        variance = state["observation_noise_variance"]
        self.state.observation_noise_variance = (
            None
            if variance is None
            else torch.as_tensor(variance, dtype=self.dtype, device=self.device).detach().clone()
        )
        action = state["best_posterior_mean_action"]
        self.state.best_posterior_mean_action = (
            None
            if action is None
            else torch.as_tensor(action, dtype=self.dtype, device=self.device).detach().clone()
        )
        value = state["best_posterior_mean_value"]
        self.state.best_posterior_mean_value = None if value is None else float(value)
        self.state.good_action_candidates = [
            torch.as_tensor(row, dtype=self.dtype, device=self.device).detach().clone()
            for row in state["good_action_candidates"]
        ]
        self.state.last_score = None if state["last_score"] is None else float(state["last_score"])
        self.state.last_probability_of_good = (
            None
            if state["last_probability_of_good"] is None
            else float(state["last_probability_of_good"])
        )
        self.state.last_fit_evaluations = state["last_fit_evaluations"]
        self.state.last_fit_reused_hyperparameters = state["last_fit_reused_hyperparameters"]
        optimizer_restarts = state["last_continuous_optimizer_restarts"]
        self.state.last_continuous_optimizer_restarts = (
            None if optimizer_restarts is None else int(optimizer_restarts)
        )
        candidates = state["last_candidates"]
        self._last_candidates = (
            None
            if candidates is None
            else torch.as_tensor(candidates, dtype=self.dtype, device=self.device).detach().clone()
        )
        indices = state["last_candidate_indices"]
        self._last_candidate_indices = (
            None
            if indices is None
            else torch.as_tensor(indices, dtype=torch.long, device=self.device).detach().clone()
        )
        if "gp" in state:
            self.gp.load_state_dict(dict(state["gp"]))
        else:
            # Legacy checkpoints omitted parameters. A fresh fit is the only
            # possible continuation for those files; new saves retain the
            # original fit schedule and numerical warm start exactly.
            self.gp.reset()
            self.state.last_fit_evaluations = -1
            self.state.last_fit_reused_hyperparameters = False

    def _reset_impl(self) -> None:
        """Clear learned state while preserving immutable extension reporting."""

        self.gp.reset()
        self.state = self._empty_state()
        self._last_candidates = None
        self._last_candidate_indices = None
        if self.config.candidate_domain is None:
            self.invalidate_source_guarantees("basat.default_continuous_candidate_domain")


__all__ = ["PG", "GoodActionConfig", "GoodActionState"]
