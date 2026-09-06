"""Probability of Good for Gaussian-process good-action identification.

The acquisition is the posterior probability that the latent reward exceeds a
fixed aspiration threshold. The first release follows the paper's maximization
convention and keeps all model coordinates in the full environment action space.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from bayopy.algorithms.algorithm import (
    Algorithm,
    AlgorithmConfigurationError,
    AlgorithmError,
    resolve_batch_configuration,
)
from bayopy.algorithms.helper import SourceRecord
from bayopy.models.Acquisitions import (
    DiscreteAcquisitionCandidates,
    optimize_acquisition_continuous,
)
from bayopy.models.GaussianProcess import GaussianProcess, GaussianProcessConfig
from bayopy.models.Kernels import KernelConfig
from bayopy.models.utils import Bounds

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
        Number of deterministic Sobol initialization rounds. ``None`` uses
        ``max(2 * D_env, 4)``.
    hyperparameter_refit_cycle:
        Number of newly completed evaluations between full GP hyperparameter
        optimizations. Between those optimizations the latest learned
        hyperparameters are reused with the enlarged training table.
    acquisition_restarts:
        Number of starts used by the differentiable continuous PG/EG
        acquisition optimizer. GS and STS use their source-defined finite
        Monte Carlo candidate representation instead.
    reference_point:
        STS reference action; the domain center is used when omitted.
    lookahead_samples, maximum_value_samples:
        Monte Carlo sizes for GS fantasy and posterior-maximum calculations.
    candidate_samples:
        Size of the reproducible continuous search representation.
    fit_maxiter:
        Maximum optimizer iterations for each exact-GP fit.
    kernel:
        Full-dimensional GP kernel configuration.
    seed:
        Algorithm and candidate-generation seed.
    """

    threshold: float
    candidate_domain: Bounds | torch.Tensor | np.ndarray[Any, Any] | None = None
    initial_points: int | None = None
    hyperparameter_refit_cycle: int = 3
    acquisition_restarts: int = 10
    reference_point: torch.Tensor | np.ndarray[Any, Any] | None = None
    lookahead_samples: int = 64
    maximum_value_samples: int = 32
    candidate_samples: int = 128
    fit_maxiter: int = 50
    posterior_std_floor: float = 1e-12
    kernel: KernelConfig = field(default_factory=KernelConfig)
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
        In-process GP refit-cycle state. Loading a checkpoint resets this cycle
        so the restored observations are fitted with the current budget.
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
    >>> import bayopy
    >>> bounds = bayopy.Bounds(lower=[-1.0, -1.0], upper=[1.0, 1.0])
    >>> algorithm = bayopy.PG(
    ...     bounds=bounds,
    ...     config=bayopy.GoodActionConfig(threshold=0.5, initial_points=5, seed=7),
    ... )
    """

    ACQUISITION_POLICY = "posterior probability of exceeding the threshold"

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
            plot=plot,
            verbose=verbose,
            verbose_setting=verbose_setting,
            verbose_iter_by_iter=verbose_iter_by_iter,
        )
        self.model_input_dimension = self.action_dimension
        self.gp = GaussianProcess(
            GaussianProcessConfig(
                kernel=config.kernel,
                mean="constant",
                noise_mode="learned",
                fit_method="mle",
                dtype=self.dtype,
                device=self.device,
            ),
            input_dimension=self.action_dimension,
        )
        self.state = self._empty_state()
        self._last_candidates: torch.Tensor | None = None
        self._last_candidate_indices: torch.Tensor | None = None
        if config.candidate_domain is None:
            self.invalidate_source_guarantees("bayopy.default_continuous_candidate_domain")

    @staticmethod
    def _validate_config(config: GoodActionConfig) -> None:
        """Validate shared sequential good-action configuration fields."""

        if not isinstance(config, GoodActionConfig):
            raise AlgorithmConfigurationError("config must be GoodActionConfig.")
        if not math.isfinite(float(config.threshold)):
            raise AlgorithmConfigurationError("threshold must be finite.")
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

    def _tell_impl(
        self,
        *,
        action: torch.Tensor,
        feedback: Mapping[str, Any],
        round_info: Mapping[str, Any],
    ) -> None:
        """Append evaluated full-coordinate feedback and retain GP parameters."""

        del action, round_info
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
        }

    def _result_metadata_impl(self) -> Mapping[str, Any]:
        """Describe the aspiration threshold and recommendation semantics."""

        return {
            "threshold": self.config.threshold,
            "posterior_good_candidate_count": len(self.state.good_action_candidates),
            "recommendation_kind": "maximum_latent_posterior_mean",
        }

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize observations, candidates, recommendations, and fit cadence."""

        return {
            "config": self._config_state(),
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
        """Restore method data and reset the GP for a fresh configured fit."""

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
        # A resumed run refits from restored observations with the current
        # fit_maxiter budget rather than restoring surrogate parameters.
        self.state.last_fit_evaluations = -1
        self.state.last_fit_reused_hyperparameters = False
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
        self.gp.reset()

    def _reset_impl(self) -> None:
        """Clear learned state while preserving immutable extension reporting."""

        self.gp.reset()
        self.state = self._empty_state()
        self._last_candidates = None
        self._last_candidate_indices = None
        if self.config.candidate_domain is None:
            self.invalidate_source_guarantees("bayopy.default_continuous_candidate_domain")


__all__ = ["PG", "GoodActionConfig", "GoodActionState"]
