"""StableOPT for robust Gaussian-process optimization under perturbations.

StableOPT distinguishes three physical coordinates: the robust center, the
paper-defined pessimistic query point inside the center's perturbation set, and
the point actually evaluated by an external adversarial environment.  Only the
last coordinate is added to the GP training table.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import gpytorch
import torch
from gpytorch.utils.warnings import NumericalWarning

from bayopy.models.Acquisitions import DiscreteAcquisitionCandidates
from bayopy.models.GaussianProcess import GaussianProcess, GaussianProcessConfig
from bayopy.models.Kernels import KernelConfig
from bayopy.models.utils import Bounds

from ..algorithm import Algorithm, AlgorithmConfigurationError, resolve_batch_configuration
from ..helper import SourceRecord
from .Exception.exception_StableOPT import (
    StableOPTCheckpointError,
    StableOPTException,
    StableOPTInnerSolverError,
    StableOPTOuterSolverError,
    StableOPTPerturbationSetError,
    StableOPTWarning,
)


@dataclass(frozen=True, slots=True)
class DistanceSpec:
    """Perturbation distance specification.

    Parameters
    ----------
    norm:
        Norm used to measure perturbations in physical ``D_env`` action
        coordinates.
    """

    norm: Literal["l2", "linf"] = "l2"


@dataclass(frozen=True, slots=True)
class L2Distance(DistanceSpec):
    """Euclidean perturbation distance.

    Attributes
    ----------
    norm:
        Fixed ``"l2"`` discriminator used by the robust-set solver.
    """

    norm: Literal["l2"] = field(default="l2", init=False)


@dataclass(frozen=True, slots=True)
class LinfDistance(DistanceSpec):
    """Maximum-coordinate perturbation distance.

    Attributes
    ----------
    norm:
        Fixed ``"linf"`` discriminator used by the robust-set solver.
    """

    norm: Literal["linf"] = field(default="linf", init=False)


@dataclass(frozen=True, slots=True)
class MultiStartSolver:
    """Deterministic Sobol multistart search with residual recomputation.

    Parameters
    ----------
    method:
        Solver discriminator. Version 1 accepts only ``"multistart"`` and
        independently certifies the returned point against the robust set.
    """

    method: Literal["multistart"] = "multistart"


RobustInnerSolverSpec = MultiStartSolver
RobustOuterSolverSpec = MultiStartSolver


# Feasibility is certified independently of the optimizer's termination
# tolerance.  This stricter threshold is part of BayOPy's solver contract.
_FEASIBILITY_TOLERANCE = 1e-8

# Querying a GPyTorch posterior with ``(B, Q, D)`` inputs evaluates ``B``
# independent Q-point batches.  This avoids both the Python overhead of one
# posterior construction per robust center and the quadratic test/test
# covariance that would result from flattening all centers into one giant Q.
# Keeping this internal (rather than configurable) makes it a pure execution
# detail: it cannot alter StableOPT's candidate set or paper settings.
_RAW_CONFIDENCE_CENTER_BATCH = 32

# Deterministic retry offset and the explicit provenance marker used if both
# exact-MLE optimizations fail but the previous fitted hyperparameters still
# define a finite posterior on the expanded training table.
_GP_RETRY_SEED_OFFSET = 104_729
_GP_RETAINED_HYPERPARAMETERS_REASON = "stableopt.gp_mle_retained_hyperparameters"


def theoretical_beta(t: int, *, delta: float = 0.1) -> float:
    """Return a standard increasing GP-UCB confidence schedule.

    Parameters
    ----------
    t:
        One-based optimization round; non-positive values are clamped to one.
    delta:
        Positive confidence-failure probability.

    Returns
    -------
    float
        Non-negative exploration coefficient for round ``t``.
    """

    time_index = max(int(t), 1)
    return float(2.0 * math.log((time_index**2) * math.pi**2 / (6.0 * delta) + 1.0))


@dataclass(frozen=True, slots=True)
class StableOPTConfig:
    """Immutable StableOPT configuration in physical action coordinates.

    Parameters
    ----------
    perturbation_radius:
        Non-negative radius of the adversarial neighborhood around a robust
        center, measured by ``distance`` in ``D_env`` coordinates.
    distance:
        Norm specification for the perturbation set intersected with the
        environment bounds.
    beta:
        Non-negative scalar or round-indexed callable that supplies the GP
        confidence multiplier.
    inner_solver:
        Solver for pessimistic/optimistic optimization inside a center's
        perturbation set.
    outer_solver:
        Solver for selecting the robust center over the full action box.
    inner_raw_samples:
        Sobol candidates used to initialize each inner robust solve.
    outer_raw_samples:
        Sobol candidates used to initialize each outer center solve.
    inner_restarts:
        Best inner candidates refined by the continuous optimizer.
    outer_restarts:
        Best outer candidates refined by the continuous optimizer.
    solver_tolerance:
        Numerical optimizer termination tolerance; feasibility is recomputed
        separately at the stricter library tolerance.
    initial_points:
        Sequential observations collected before model-based robust search.
    fit_maxiter:
        Maximum GP hyperparameter-optimization iterations per refit.
    kernel:
        Kernel construction settings for the GP over physical actions.
    batch_size:
        Requested proposal count. StableOPT version 1 requires ``1``.
    seed:
        Seed for algorithm-owned Sobol draws and replayable random state.
    """

    perturbation_radius: float
    distance: DistanceSpec = field(default_factory=L2Distance)
    beta: Callable[[int], float] | float = theoretical_beta
    inner_solver: RobustInnerSolverSpec = field(default_factory=MultiStartSolver)
    outer_solver: RobustOuterSolverSpec = field(default_factory=MultiStartSolver)
    inner_raw_samples: int = 256
    outer_raw_samples: int = 1024
    inner_restarts: int = 10
    outer_restarts: int = 20
    solver_tolerance: float = 1e-6
    initial_points: int = 1
    fit_maxiter: int = 100
    kernel: KernelConfig = field(default_factory=KernelConfig)
    batch_size: int | None = 1
    q_batch: int | None = None
    seed: int = 1


@dataclass(slots=True)
class StableOPTState:
    """Mutable state that influences future robust proposals.

    Attributes
    ----------
    X_query_train:
        Evaluated GP input coordinates with shape ``(N, D_env)``. Under an
        external attack these can differ from the requested query points.
    y_train:
        Learner-visible observations with shape ``(N,)``.
    observation_noise_variance:
        Known observation-noise variances with shape ``(N,)``; zeros denote
        feedback without a supplied variance.
    robust_centers:
        Proposed robust centers, each with shape ``(D_env,)``.
    query_perturbations:
        Paper-defined internal perturbations from robust center to requested
        query point, each with shape ``(D_env,)``.
    query_points:
        Actions requested from the environment, each with shape ``(D_env,)``.
    evaluated_points:
        Physical coordinates returned as ``evaluated_action`` and used by the
        GP, each with shape ``(D_env,)``.
    external_perturbations:
        Publicly reported external perturbations, or ``None`` when feedback
        does not expose them. Hidden attack state is never copied here.
    robust_ucb_values:
        Robust upper-confidence values recorded for proposed centers.
    robust_lcb_values:
        Pre-update robust lower-confidence values for proposed centers, as in
        the paper's Eq. (17). Initial-design entries have no confidence value.
    """

    X_query_train: torch.Tensor
    y_train: torch.Tensor
    observation_noise_variance: torch.Tensor
    robust_centers: list[torch.Tensor]
    query_perturbations: list[torch.Tensor]
    query_points: list[torch.Tensor]
    evaluated_points: list[torch.Tensor]
    external_perturbations: list[torch.Tensor | None]
    robust_ucb_values: list[float]
    robust_lcb_values: list[float]


class StableOPT(Algorithm):
    """Nested robust-UCB optimizer with paper-faithful robust recommendations.

    Examples
    --------
    >>> import bayopy
    >>> bounds = bayopy.Bounds(lower=[-1.0, -1.0], upper=[1.0, 1.0])
    >>> algorithm = bayopy.StableOPT(
    ...     bounds=bounds,
    ...     config=bayopy.StableOPTConfig(
    ...         perturbation_radius=0.1, beta=2.0, initial_points=3, seed=7
    ...     ),
    ... )
    """

    ACQUISITION_POLICY = "nested robust GP-UCB"

    EXCEPTION_CLASS = StableOPTException
    native_batching = False
    native_environments = frozenset({"stochastic", "adversarial"})
    SOURCE = SourceRecord(
        paper_title="Adversarially Robust Optimization with Gaussian Processes",
        paper_year=2018,
        paper_url="https://arxiv.org/abs/1810.10775",
        repository_url="https://github.com/Bilkent-CYBORG/AdveRS",
        repository_status="related",
        notes="The paper governs StableOPT; the repository contains a related robust baseline.",
    )

    def __init__(
        self,
        *,
        bounds: Bounds,
        config: StableOPTConfig,
        maximize: bool = True,
        dtype: torch.dtype | str = torch.float32,
        device: torch.device | str = "cpu",
        plot: bool = False,
        verbose: bool = False,
        verbose_setting: bool = False,
        verbose_iter_by_iter: bool = False,
    ) -> None:
        """Initialize a sequential StableOPT optimizer.

        Parameters
        ----------
        bounds:
            Box constraints for physical environment actions, with shape
            ``(D_env,)`` at each endpoint.
        config:
            Immutable robust-search, GP, solver, and random-seed settings.
        maximize:
            Optimization direction. StableOPT v1 accepts only ``True``.
        dtype, device:
            Tensor precision and execution device used by all model state.
        plot, verbose, verbose_setting, verbose_iter_by_iter:
            Presentation flags inherited from :class:`Algorithm`; they do not
            affect the decision rule.

        Raises
        ------
        AlgorithmConfigurationError
            If minimization is requested or common configuration is invalid.
        StableOPTPerturbationSetError
            If the robust neighborhood geometry is invalid.
        StableOPTInnerSolverError, StableOPTOuterSolverError
            If a configured nested solver is unsupported.
        """

        self.config = config
        self._validate_config(config)
        if not maximize:
            raise AlgorithmConfigurationError("StableOPT version 1 supports maximization only.")
        super().__init__(
            bounds=bounds,
            batch_size=config.batch_size,
            q_batch=config.q_batch,
            maximize=True,
            dtype=dtype,
            device=device,
            seed=config.seed,
            plot=plot,
            verbose=verbose,
            verbose_setting=verbose_setting,
            verbose_iter_by_iter=verbose_iter_by_iter,
        )
        self.gp = GaussianProcess(
            GaussianProcessConfig(
                kernel=config.kernel,
                dtype=self.dtype,
                device=self.device,
                fit_method="mle",
            ),
            input_dimension=self.action_dimension,
        )
        self.state = self._empty_state()
        self._posterior_marginal_cache: dict[str, Any] | None = None
        self._recommendation_cache: tuple[torch.Tensor | None, float | None] | None = None
        self._inner_unit_design_cache: tuple[int, int, torch.Tensor] | None = None
        self._last_outer_pruning_certificate: dict[str, float] = {}
        self._last_recommendation_pruning_certificate: dict[str, float] = {}

    @staticmethod
    def _validate_config(config: StableOPTConfig) -> None:
        """Validate immutable sequential, geometry, and solver settings."""

        if not isinstance(config, StableOPTConfig):
            raise AlgorithmConfigurationError("config must be StableOPTConfig.")
        _, q_batch = resolve_batch_configuration(
            batch_size=config.batch_size, q_batch=config.q_batch
        )
        if q_batch is not None and q_batch > 1:
            raise AlgorithmConfigurationError(
                "StableOPT does not define a joint q-acquisition; use batch_size for "
                "repeated independent robust queries."
            )
        if not math.isfinite(config.perturbation_radius) or config.perturbation_radius < 0:
            raise StableOPTPerturbationSetError(
                "perturbation_radius must be finite and non-negative."
            )
        if config.distance.norm not in {"l2", "linf"}:
            raise StableOPTPerturbationSetError("distance norm must be l2 or linf.")
        if (
            not isinstance(config.inner_solver, MultiStartSolver)
            or config.inner_solver.method != "multistart"
        ):
            raise StableOPTInnerSolverError(
                "inner_solver must select the deterministic multistart solver."
            )
        if (
            not isinstance(config.outer_solver, MultiStartSolver)
            or config.outer_solver.method != "multistart"
        ):
            raise StableOPTOuterSolverError(
                "outer_solver must select the deterministic multistart solver."
            )
        for name in (
            "inner_raw_samples",
            "outer_raw_samples",
            "inner_restarts",
            "outer_restarts",
            "initial_points",
            "fit_maxiter",
        ):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise AlgorithmConfigurationError(f"{name} must be a positive integer.")
        if not math.isfinite(config.solver_tolerance) or config.solver_tolerance <= 0:
            raise AlgorithmConfigurationError("solver_tolerance must be finite and positive.")

    def _empty_state(self) -> StableOPTState:
        """Create empty learner-visible state in physical action coordinates."""

        return StableOPTState(
            X_query_train=torch.empty(
                (0, self.action_dimension), dtype=self.dtype, device=self.device
            ),
            y_train=torch.empty(0, dtype=self.dtype, device=self.device),
            observation_noise_variance=torch.empty(0, dtype=self.dtype, device=self.device),
            robust_centers=[],
            query_perturbations=[],
            query_points=[],
            evaluated_points=[],
            external_perturbations=[],
            robust_ucb_values=[],
            robust_lcb_values=[],
        )

    def _initialize_from_environment_impl(self, data: Any) -> None:
        """Initialize the GP from public historical evaluated coordinates.

        Requested actions may differ from externally evaluated actions in an
        adversarial environment. Only ``evaluated_actions`` and their visible
        observations enter the GP; latent attack state is never inspected.
        """

        requested = torch.as_tensor(data.actions, dtype=self.dtype, device=self.device)
        evaluated_raw = getattr(data, "evaluated_actions", None)
        evaluated = (
            requested
            if evaluated_raw is None
            else torch.as_tensor(evaluated_raw, dtype=self.dtype, device=self.device)
        )
        values = torch.as_tensor(data.observations, dtype=self.dtype, device=self.device).reshape(
            -1
        )
        variance_raw = getattr(data, "observation_noise_variance", None)
        variance = (
            torch.zeros_like(values)
            if variance_raw is None
            else torch.as_tensor(variance_raw, dtype=self.dtype, device=self.device).reshape(-1)
        )
        self.state.X_query_train = evaluated.detach().clone()
        self.state.y_train = values.detach().clone()
        self.state.observation_noise_variance = variance.detach().clone()
        if values.numel():
            self.gp.set_data(evaluated, values, variance)
            self._posterior_marginal_cache = None

    def _prepare_discrete_candidates_impl(
        self,
        *,
        candidate_actions: torch.Tensor,
        candidate_dataset_indices: torch.Tensor,
        round_info: Mapping[str, Any],
    ) -> DiscreteAcquisitionCandidates:
        """Preserve finite rows under StableOPT's identity coordinate map."""

        del round_info
        return DiscreteAcquisitionCandidates(
            environment_actions=candidate_actions.detach().clone(),
            model_inputs=candidate_actions.detach().clone(),
            dataset_indices=candidate_dataset_indices.detach().clone(),
        )

    def _beta(self) -> float:
        """Resolve and validate the confidence multiplier for the next round."""

        raw = (
            self.config.beta(self.iteration + 1) if callable(self.config.beta) else self.config.beta
        )
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise StableOPTInnerSolverError("beta must resolve to a finite non-negative scalar.")
        return value

    def _gp_training_state_is_current(self) -> bool:
        """Return whether the fitted GP already represents committed data."""

        known_noise = self.gp.observation_noise
        return bool(
            self.gp.is_fit
            and self.gp.X_train is not None
            and self.gp.y_train is not None
            and known_noise is not None
            and torch.equal(self.gp.X_train, self.state.X_query_train)
            and torch.equal(self.gp.y_train, self.state.y_train)
            and torch.equal(known_noise, self.state.observation_noise_variance)
        )

    def _current_gp_model_state(self) -> dict[str, torch.Tensor] | None:
        """Copy fitted module values for a same-model exact-MLE warm start."""

        if self.gp.model is None or not self.gp.is_fit:
            return None
        return {name: value.detach().clone() for name, value in self.gp.model.state_dict().items()}

    def _restore_compatible_gp_model_state(
        self, model_state: Mapping[str, torch.Tensor] | None
    ) -> None:
        """Restore shape-compatible values after rebuilding the training table."""

        if model_state is None or self.gp.model is None:
            return
        current = self.gp.model.state_dict()
        for name, value in model_state.items():
            if name in current and current[name].shape == value.shape:
                current[name] = value.to(
                    dtype=current[name].dtype,
                    device=current[name].device,
                )
        self.gp.model.load_state_dict(current, strict=False)

    @contextmanager
    def _owned_torch_rng(self, *, seed: int) -> Iterator[None]:
        """Seed GP-fit retries without mutating the caller's ambient RNG."""

        cuda_devices: list[int] = []
        if self.device.type == "cuda":
            cuda_devices.append(
                torch.cuda.current_device() if self.device.index is None else self.device.index
            )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            yield

    def _initialize_response_scaled_gp(self) -> None:
        """Set a deterministic response-scaled start for the unchanged MLE."""

        model = self.gp.model
        if model is None:
            return
        targets = self.state.y_train
        mean = targets.mean()
        variance = (targets - mean).square().mean()
        signal_variance = max(
            float(variance.item()),
            float(self.gp.config.jitter),
            1e-4,
        )
        scale_modules = [
            module
            for module in model.modules()
            if hasattr(module, "raw_outputscale") and hasattr(module, "outputscale")
        ]
        scale_value = signal_variance ** (1.0 / max(len(scale_modules), 1))
        with torch.no_grad():
            mean_module = model.mean_module
            if hasattr(mean_module, "raw_constant"):
                mean_module.raw_constant.copy_(mean.to(mean_module.raw_constant))
            for module in scale_modules:
                module.outputscale = torch.as_tensor(
                    scale_value,
                    dtype=self.dtype,
                    device=self.device,
                )
            for module in model.modules():
                if not hasattr(module, "raw_noise") or not hasattr(module, "noise"):
                    continue
                constraint = getattr(module, "raw_noise_constraint", None)
                lower_bound = 0.0
                if constraint is not None and hasattr(constraint, "lower_bound"):
                    lower_bound = float(
                        torch.as_tensor(constraint.lower_bound).reshape(-1)[0].item()
                    )
                noise_value = max(
                    lower_bound + float(self.gp.config.jitter),
                    1e-4 * signal_variance,
                    10.0 * float(self.gp.config.jitter),
                )
                module.noise = torch.as_tensor(
                    noise_value,
                    dtype=self.dtype,
                    device=self.device,
                )

    def _retain_previous_hyperparameters(
        self,
        model_state: Mapping[str, torch.Tensor],
        fit_errors: list[Exception],
    ) -> None:
        """Certify preceding fitted values on the expanded training table."""

        self.gp.set_data(
            self.state.X_query_train,
            self.state.y_train,
            self.state.observation_noise_variance,
        )
        self._restore_compatible_gp_model_state(model_state)
        if self.gp.model is None or self.gp.likelihood is None:
            raise RuntimeError("The rebuilt GP is unavailable for fallback certification.")
        if any(not torch.isfinite(parameter).all() for parameter in self.gp.model.parameters()):
            raise RuntimeError("The previous fitted GP contains non-finite parameters.")
        self.gp.model.eval()
        self.gp.likelihood.eval()
        self.gp.is_fit = True
        certificate_points = self.state.X_query_train[-min(2, self.state.X_query_train.shape[0]) :]
        mean, std = self.gp.predict(certificate_points)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise RuntimeError("The retained GP produced a non-finite posterior certificate.")
        self.gp.last_fit_diagnostics = {
            "method": "mle",
            "success": False,
            "usable_posterior": True,
            "initialization": "retained_previous_hyperparameters",
            "training_rows": int(self.state.X_query_train.shape[0]),
            "input_dimension": self.action_dimension,
            "fit_errors": tuple(str(error) for error in fit_errors),
        }
        warnings.warn(
            "StableOPT GP MLE exhausted its warm-started and deterministic "
            "response-scaled attempts. The preceding finite fitted "
            "hyperparameters were retained on the expanded training table after "
            "a finite-posterior certificate; source guarantees no longer apply.",
            StableOPTWarning,
            stacklevel=3,
        )
        self.invalidate_source_guarantees(_GP_RETAINED_HYPERPARAMETERS_REASON)

    def _fit_model(self) -> None:
        """Fit through deterministic, transactional exact-MLE attempts."""

        if self.state.y_train.numel() == 0:
            return
        if self._gp_training_state_is_current():
            # ``Experiment.run_one_step`` asks for a public result after every
            # tell.  The following ask sees exactly that already-fitted table;
            # rebuilding and refitting it would be mathematically redundant.
            return
        self._posterior_marginal_cache = None
        previous_gp_state = self.gp.state_dict()
        warm_model_state = self._current_gp_model_state()
        fit_errors: list[Exception] = []
        for attempt in range(2):
            try:
                self.gp.set_data(
                    self.state.X_query_train,
                    self.state.y_train,
                    self.state.observation_noise_variance,
                )
                if attempt == 0:
                    self._restore_compatible_gp_model_state(warm_model_state)
                else:
                    self._initialize_response_scaled_gp()
                with self._owned_torch_rng(
                    seed=self.config.seed + self.iteration + attempt * _GP_RETRY_SEED_OFFSET
                ):
                    self.gp.fit(maxiter=self.config.fit_maxiter)
            except Exception as error:
                fit_errors.append(error)
                continue
            if attempt:
                warnings.warn(
                    "StableOPT GP MLE exhausted its warm-started attempt; the same "
                    "exact MLE succeeded from one deterministic response-scaled "
                    "initialization. The GP, likelihood, nested robust objective, "
                    "and inner/outer solvers are unchanged.",
                    StableOPTWarning,
                    stacklevel=3,
                )
                self.gp.last_fit_diagnostics["initialization"] = (
                    "deterministic_response_scaled_retry"
                )
            else:
                self.gp.last_fit_diagnostics["initialization"] = "warm_start"
            return

        if warm_model_state is not None:
            try:
                self._retain_previous_hyperparameters(warm_model_state, fit_errors)
                return
            except Exception as fallback_error:
                fit_errors.append(fallback_error)
        try:
            self.gp.load_state_dict(previous_gp_state)
        except Exception as rollback_error:
            raise StableOPTInnerSolverError(
                f"StableOPT GP fitting failed and rollback failed: {rollback_error}",
                algorithm=self,
            ) from rollback_error
        details = "; ".join(
            f"attempt {index}: {error}" for index, error in enumerate(fit_errors, start=1)
        )
        raise StableOPTInnerSolverError(
            f"StableOPT GP fitting failed after deterministic attempts: {details}",
            algorithm=self,
        ) from fit_errors[-1]

    def _posterior_marginals(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate exact latent marginals using GPyTorch's prediction caches.

        ``GaussianProcess.predict`` constructs a posterior distribution, which
        is useful when callers need its joint covariance.  StableOPT's SLSQP
        objective needs only independent marginal means and variances and can
        reuse the exact prediction strategy factorization built on the first
        call.  The algebra here is the same exact-GP posterior computation:
        ``K_xt alpha`` for the mean and the inverse-root correction for each
        marginal variance.
        """

        model = self.gp.botorch_model
        cache = self._posterior_marginal_cache
        if cache is None or cache.get("model") is not model:
            # The public prediction path creates and certifies GPyTorch's
            # prediction-strategy caches under BayOPy's configured jitter.
            mean, std = self.gp.predict(points)
            strategy: Any = model.prediction_strategy
            if strategy is None:  # pragma: no cover - defensive GPyTorch boundary
                return mean, std
            self._posterior_marginal_cache = {
                "model": model,
                "train_inputs": strategy.train_inputs[0].detach(),
                "mean_cache": strategy.mean_cache.detach(),
                "covar_cache": strategy.covar_cache.detach(),
            }
            return mean, std

        train_inputs = cache["train_inputs"]
        mean_cache = cache["mean_cache"]
        covar_cache = cache["covar_cache"]
        # Calling module ``forward`` directly avoids construction of lazy
        # joint-covariance wrappers; these are marginal-only computations and
        # the same kernel/mean implementations and fitted parameters are used.
        cross_covariance = model.covar_module.forward(points, train_inputs).to_dense()
        mean = model.mean_module.forward(points) + cross_covariance.matmul(mean_cache)
        prior_variance = model.covar_module.forward(points, points, diag=True)
        projection = cross_covariance.matmul(covar_cache)
        variance = prior_variance - projection.square().sum(dim=-1)
        min_variance = gpytorch.settings.min_variance.value(variance.dtype)
        if variance.lt(min_variance).any():
            warnings.warn(
                "Negative variance values detected. This is likely due to numerical "
                f"instabilities. Rounding negative variances up to {min_variance}.",
                NumericalWarning,
                stacklevel=2,
            )
            variance = variance.clamp_min(min_variance)
        return mean, variance.sqrt()

    def _distance(self, center: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Measure configured perturbation distances in physical coordinates."""

        delta = points - center
        if self.config.distance.norm == "linf":
            return delta.abs().amax(dim=-1)
        return torch.linalg.vector_norm(delta, ord=2, dim=-1)

    def _normalized_lexicographic_key(self, point: torch.Tensor) -> tuple[float, ...]:
        """Return the deterministic continuous-domain tie-breaking key."""

        lower, upper = self._bounds_on_algorithm_device()
        normalized = (point - lower) / (upper - lower)
        return tuple(float(value) for value in normalized.detach().cpu().tolist())

    def _feasibility_residual(
        self,
        point: torch.Tensor,
        *,
        center: torch.Tensor | None = None,
        radius: float | None = None,
    ) -> float:
        """Recompute box and optional perturbation constraint violations."""

        lower, upper = self._bounds_on_algorithm_device()
        lower_violation = float((lower - point).clamp_min(0).max().item())
        upper_violation = float((point - upper).clamp_min(0).max().item())
        residual = max(lower_violation, upper_violation)
        if center is not None and radius is not None:
            distance_violation = (
                float(self._distance(center, point.unsqueeze(0))[0].item()) - radius
            )
            residual = max(residual, distance_violation)
        return max(residual, 0.0)

    @staticmethod
    def _finite_positions(values: torch.Tensor) -> torch.Tensor:
        """Return positions with finite scalar objective values."""

        return torch.nonzero(torch.isfinite(values), as_tuple=False).reshape(-1)

    def _best_position(
        self,
        values: torch.Tensor,
        points: torch.Tensor,
        *,
        maximize: bool,
        dataset_indices: torch.Tensor | None = None,
    ) -> int:
        """Select an exact optimum using the documented stable tie rules."""

        finite = self._finite_positions(values)
        if finite.numel() == 0:
            raise ValueError("No finite objective value is available.")
        target = values[finite].max() if maximize else values[finite].min()
        tolerance = 1e-12 + 1e-9 * max(1.0, abs(float(target.item())))
        tied = (
            finite[values[finite] >= target - tolerance]
            if maximize
            else finite[values[finite] <= target + tolerance]
        )
        if dataset_indices is not None:
            return int(tied[torch.argmin(dataset_indices[tied])].item())
        return min(
            (int(position.item()) for position in tied),
            key=lambda position: self._normalized_lexicographic_key(points[position]),
        )

    def _continuous_multistart_minimize(
        self,
        *,
        solver: MultiStartSolver,
        objective: Callable[[torch.Tensor], float],
        raw_points: torch.Tensor,
        raw_values: torch.Tensor,
        restarts: int,
        center: torch.Tensor | None,
        radius: float | None,
        error_type: type[StableOPTInnerSolverError] | type[StableOPTOuterSolverError],
        label: str,
        objective_gradient: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, float, dict[str, float]]:
        """Run deterministic constrained SLSQP starts and certify each result.

        Raw Sobol points are valid multistart candidates in their own right.
        Refined points are accepted only after independently recomputing their
        objective and every feasibility residual in torch coordinates.
        """

        if not isinstance(solver, MultiStartSolver) or solver.method != "multistart":
            raise error_type(
                f"{label} does not support solver {getattr(solver, 'method', None)!r}.",
                algorithm=self,
            )

        finite = self._finite_positions(raw_values)
        if finite.numel() == 0:
            raise error_type(f"{label} raw search produced no finite value.", algorithm=self)
        order = finite[torch.argsort(raw_values[finite], stable=True)]
        selected = order[: min(restarts, order.numel())]
        candidates: list[tuple[torch.Tensor, float, float, float]] = []

        lower, upper = self._bounds_on_algorithm_device()
        if center is not None and radius is not None:
            lower = torch.maximum(lower, center - radius)
            upper = torch.minimum(upper, center + radius)

        # A multistart method starts from the best discrete search points, and
        # those points remain valid solutions when a local refinement stalls.
        # Retaining them is especially important for nearly interpolating GPs:
        # SLSQP can report a failed line search on an almost-flat confidence
        # surface even though its finite, feasible starting point is certified.
        # Refined results below still have to improve on these starts and pass
        # the same independent feasibility check.
        for position in selected:
            point = raw_points[int(position.item())]
            residual = self._feasibility_residual(point, center=center, radius=radius)
            if residual <= _FEASIBILITY_TOLERANCE:
                candidates.append(
                    (
                        point.detach().clone(),
                        float(raw_values[int(position.item())].item()),
                        residual,
                        0.0,
                    )
                )
        if torch.equal(lower, upper):
            point = lower.detach().clone()
            value = float(objective(point))
            residual = self._feasibility_residual(point, center=center, radius=radius)
            if not math.isfinite(value) or residual > _FEASIBILITY_TOLERANCE:
                raise error_type(
                    f"{label} singleton domain failed independent certification.",
                    algorithm=self,
                )
            return (
                point,
                value,
                {
                    "feasibility_residual": residual,
                    "solver_success": 1.0,
                    "solver_calls": 0.0,
                },
            )
        scipy_bounds = list(
            zip(
                lower.detach().cpu().tolist(),
                upper.detach().cpu().tolist(),
                strict=True,
            )
        )
        constraints: list[dict[str, Any]] = []
        if center is not None and radius is not None and self.config.distance.norm == "l2":
            center_cpu = center.detach().cpu()

            def ball_constraint(array: Any) -> float:
                """Return the nonnegative Euclidean-ball feasibility slack."""

                point = torch.as_tensor(array, dtype=center_cpu.dtype)
                return float(radius - torch.linalg.vector_norm(point - center_cpu).item())

            constraints.append({"type": "ineq", "fun": ball_constraint})
        try:
            from scipy.optimize import minimize
        except ImportError as error:  # pragma: no cover - SciPy is a core dependency
            raise error_type(
                f"{label} requires SciPy's SLSQP optimizer.", algorithm=self
            ) from error

        # SLSQP repeatedly requests the same point, especially when several
        # restarts converge to a shared box/ball boundary.  Cache only the
        # callback evaluations: independently certified result values below
        # still call ``objective`` directly.  Seed the cache with the already
        # evaluated raw design so a restart does not rebuild the same GP
        # posterior merely to score its initial point.
        objective_cache = {
            self._objective_cache_key(raw_points[int(position.item())]): float(
                raw_values[int(position.item())].item()
            )
            for position in finite
        }
        gradient_cache: dict[tuple[float, ...], torch.Tensor] = {}
        certified_objective_cache: dict[tuple[float, ...], float] = {}
        solver_calls = 0
        for position in selected:
            start = raw_points[int(position.item())]

            def scipy_objective(array: Any) -> float:
                """Evaluate the torch objective at one SciPy coordinate vector."""

                point = torch.as_tensor(array, dtype=self.dtype, device=self.device)
                cache_key = self._objective_cache_key(point)
                cached = objective_cache.get(cache_key)
                if cached is not None:
                    return cached
                value = float(objective(point))
                finite_value = value if math.isfinite(value) else float("inf")
                objective_cache[cache_key] = finite_value
                return finite_value

            def scipy_gradient(array: Any) -> Any:
                """Return an exact acquisition gradient for one SciPy point."""

                assert objective_gradient is not None
                point = torch.as_tensor(array, dtype=self.dtype, device=self.device)
                cache_key = self._objective_cache_key(point)
                cached = gradient_cache.get(cache_key)
                if cached is None:
                    cached = objective_gradient(point).detach().clone()
                    if cached.shape != point.shape or not torch.isfinite(cached).all():
                        # SLSQP treats a non-finite derivative like any other
                        # failed restart; use a zero vector so the independently
                        # scored raw candidate remains available.
                        cached = torch.zeros_like(point)
                    gradient_cache[cache_key] = cached
                return cached.detach().cpu().numpy()

            try:
                result = minimize(
                    scipy_objective,
                    start.detach().cpu().numpy(),
                    method="SLSQP",
                    jac=scipy_gradient if objective_gradient is not None else None,
                    bounds=scipy_bounds,
                    constraints=constraints,
                    options={"ftol": self.config.solver_tolerance, "maxiter": 200},
                )
            except (RuntimeError, TypeError, ValueError):
                solver_calls += 1
                continue
            solver_calls += 1
            point = torch.as_tensor(result.x, dtype=self.dtype, device=self.device)
            # Solver convergence is necessary but not sufficient: objective
            # and constraints are independently recomputed before acceptance.
            # Identical results returned by multiple restarts share the first
            # such recomputation; the GP and beta are immutable throughout
            # this multistart solve.
            if not result.success:
                continue
            cache_key = self._objective_cache_key(point)
            cached_value = certified_objective_cache.get(cache_key)
            if cached_value is None:
                certified_value = float(objective(point))
                certified_objective_cache[cache_key] = certified_value
            else:
                certified_value = cached_value
            residual = self._feasibility_residual(point, center=center, radius=radius)
            if math.isfinite(certified_value) and residual <= _FEASIBILITY_TOLERANCE:
                candidates.append(
                    (
                        point.detach().clone(),
                        certified_value,
                        residual,
                        float(bool(result.success)),
                    )
                )

        if not candidates:
            raise error_type(
                f"{label} returned no finite point satisfying the independently "
                "recomputed constraints.",
                algorithm=self,
            )
        best_value = min(item[1] for item in candidates)
        tolerance = 1e-12 + 1e-9 * max(1.0, abs(best_value))
        tied = [item for item in candidates if item[1] <= best_value + tolerance]
        best = min(tied, key=lambda item: self._normalized_lexicographic_key(item[0]))
        return (
            best[0],
            best[1],
            {
                "feasibility_residual": best[2],
                "optimizer_success": best[3],
                "solver_calls": float(solver_calls),
            },
        )

    @staticmethod
    def _objective_cache_key(point: torch.Tensor) -> tuple[float, ...]:
        """Return an exact, device-independent key for one solver coordinate."""

        return tuple(float(value) for value in point.detach().cpu().reshape(-1).tolist())

    def _perturbation_set(
        self,
        center: torch.Tensor,
        *,
        candidates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return feasible explicit inner points with recomputed norm residuals."""

        lower, upper = self._bounds_on_algorithm_device()
        radius = self.config.perturbation_radius
        if candidates is None:
            local = (
                Bounds(
                    torch.maximum(lower, center - radius),
                    torch.minimum(upper, center + radius),
                )
                if radius > 0
                else None
            )
            if local is None:
                points = center.unsqueeze(0)
            else:
                seed = self.seed + 104729 * (self.iteration + 1)
                resolved_method = self._resolved_sampling_method("acquisition")
                if resolved_method in {"paper", "sobol"}:
                    # Every center uses the same seeded Sobol points mapped to
                    # its local box.  Cache those unit coordinates once per
                    # round; affine mapping below is exactly the operation
                    # performed by ``sobol_sample(local, ...)``.
                    cache = self._inner_unit_design_cache
                    if cache is None or cache[:2] != (
                        self.iteration,
                        self.config.inner_raw_samples,
                    ):
                        unit_bounds = Bounds(
                            torch.zeros(
                                self.action_dimension,
                                dtype=self.dtype,
                                device=self.device,
                            ),
                            torch.ones(
                                self.action_dimension,
                                dtype=self.dtype,
                                device=self.device,
                            ),
                        )
                        unit = self._sample_design(
                            unit_bounds,
                            self.config.inner_raw_samples,
                            phase="acquisition",
                            seed=seed,
                        )
                        cache = (
                            self.iteration,
                            self.config.inner_raw_samples,
                            unit.detach().clone(),
                        )
                        self._inner_unit_design_cache = cache
                    unit = cache[2]
                    points = local.lower + unit * local.width
                else:
                    # Uniform overrides intentionally retain their stateful
                    # per-center draws and therefore cannot share a design.
                    points = self._sample_design(
                        local,
                        self.config.inner_raw_samples,
                        phase="acquisition",
                        seed=seed,
                    )
                points = torch.cat((center.unsqueeze(0), points), dim=0)
        else:
            points = candidates
        feasible = self._distance(center, points) <= radius + _FEASIBILITY_TOLERANCE
        feasible &= (
            (points >= lower - _FEASIBILITY_TOLERANCE) & (points <= upper + _FEASIBILITY_TOLERANCE)
        ).all(dim=-1)
        result = points[feasible]
        if result.shape[0] == 0:
            raise StableOPTPerturbationSetError(
                "The perturbation set has no explicitly evaluable point.", algorithm=self
            )
        residual = self._distance(center, result).max().item() - radius
        if residual > _FEASIBILITY_TOLERANCE:
            raise StableOPTPerturbationSetError(
                f"Perturbation residual {residual:.3e} exceeds tolerance.", algorithm=self
            )
        return result

    @staticmethod
    def _raw_minimum_return_upper_bound(values: torch.Tensor) -> float:
        """Bound the value returned by a minimization seeded by ``values``.

        The multistart solver always retains its best raw point.  Its stable
        tie rule may return a point within ``1e-12 + 1e-9 * max(1, |best|)``
        of the numerical minimum, so the same term is included in this bound.
        """

        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return float("inf")
        raw_minimum = float(finite.min().item())
        tolerance = 1e-12 + 1e-9 * max(1.0, abs(raw_minimum))
        return raw_minimum + tolerance

    def _certified_top_refined_positions(
        self,
        *,
        raw_inner_values: list[torch.Tensor],
        refine: Callable[[int], float],
        count: int,
    ) -> tuple[list[int], dict[int, float], dict[str, float]]:
        """Lazily refine exactly enough inner minima to certify the top set.

        An inner refinement can never exceed its raw-minimum upper bound
        because all raw starts remain solver candidates.  Centers are refined
        in descending bound order until every unrefined bound is strictly
        below the current ``count``-th exact value.  The selected starts are
        therefore identical to an eager refinement of every center, including
        stable original-order ties.
        """

        if count < 1:
            raise ValueError("count must be positive.")
        required = min(count, len(raw_inner_values))
        bounds = torch.tensor(
            [self._raw_minimum_return_upper_bound(values) for values in raw_inner_values],
            dtype=torch.float64,
        )
        order = torch.argsort(bounds, descending=True, stable=True).tolist()
        exact: dict[int, float] = {}
        stopped_at = len(order)
        for offset, position in enumerate(order):
            exact[int(position)] = float(refine(int(position)))
            if len(exact) < required:
                continue
            exact_order = sorted(exact, key=lambda item: (-exact[item], item))
            threshold = exact[exact_order[required - 1]]
            remaining = order[offset + 1 :]
            if not remaining or max(float(bounds[item].item()) for item in remaining) < threshold:
                stopped_at = offset + 1
                break
        selected = sorted(exact, key=lambda item: (-exact[item], item))[:required]
        # Present selected centers in original order. The shared multistart
        # routine performs its own stable value sort, reproducing eager order.
        selected.sort()
        remaining_bounds = [float(bounds[item].item()) for item in order[stopped_at:]]
        certificate = {
            "raw_centers": float(len(raw_inner_values)),
            "refined_centers": float(len(exact)),
            "pruned_centers": float(len(raw_inner_values) - len(exact)),
            "largest_pruned_upper_bound": (
                max(remaining_bounds) if remaining_bounds else float("-inf")
            ),
        }
        return selected, exact, certificate

    def _confidence(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior UCB and LCB vectors at physical query points."""

        # StableOPT uses a derivative-free continuous solver; retaining an
        # autograd graph for every posterior query only increases runtime and
        # memory and cannot affect the chosen point.
        with torch.no_grad():
            mean, std = self._posterior_marginals(points)
            width = math.sqrt(self._beta()) * std
        return mean + width, mean - width

    def _confidence_gradient(
        self,
        point: torch.Tensor,
        *,
        upper: bool,
    ) -> torch.Tensor:
        """Return the exact autodiff gradient of one GP confidence bound.

        SLSQP otherwise estimates a six-dimensional acquisition gradient with
        seven separate nested posterior evaluations.  Differentiating the
        same fitted exact-GP marginal changes neither objective nor solver and
        every returned point is still independently rescored and certified.
        """

        if self._posterior_marginal_cache is None:
            # Populate GPyTorch's exact prediction-strategy factors once.
            self._confidence(point.unsqueeze(0))
        cache = self._posterior_marginal_cache
        if cache is None:  # pragma: no cover - defensive GPyTorch boundary
            return torch.zeros_like(point)
        model = self.gp.botorch_model
        variable = point.detach().clone().requires_grad_(True)
        points = variable.unsqueeze(0)
        train_inputs = cache["train_inputs"]
        mean_cache = cache["mean_cache"]
        covar_cache = cache["covar_cache"]
        cross_covariance = model.covar_module.forward(points, train_inputs).to_dense()
        mean = model.mean_module.forward(points) + cross_covariance.matmul(mean_cache)
        prior_variance = model.covar_module.forward(points, points, diag=True)
        projection = cross_covariance.matmul(covar_cache)
        variance = prior_variance - projection.square().sum(dim=-1)
        minimum = gpytorch.settings.min_variance.value(variance.dtype)
        standard_deviation = variance.clamp_min(minimum).sqrt()
        direction = 1.0 if upper else -1.0
        bound = mean + direction * math.sqrt(self._beta()) * standard_deviation
        gradient = torch.autograd.grad(bound.sum(), variable, allow_unused=False)[0]
        return gradient.detach()

    def _batched_raw_inner_bound(
        self,
        centers: torch.Tensor,
        *,
        candidates: torch.Tensor | None,
        upper: bool,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Build and score every raw inner confidence table in GP batches.

        The point tables are still produced by :meth:`_perturbation_set` in
        center order, so Sobol/uniform draws, feasibility filtering, and tie
        order are unchanged.  Variable-size L2 or finite neighborhoods are
        padded only for posterior evaluation; padded scores are discarded.
        """

        tables = [self._perturbation_set(center, candidates=candidates) for center in centers]
        scored: list[tuple[torch.Tensor, torch.Tensor]] = []
        for offset in range(0, len(tables), _RAW_CONFIDENCE_CENTER_BATCH):
            group = tables[offset : offset + _RAW_CONFIDENCE_CENTER_BATCH]
            maximum = max(points.shape[0] for points in group)
            padded = torch.stack(
                [
                    points
                    if points.shape[0] == maximum
                    else torch.cat(
                        (
                            points,
                            points[-1:].expand(maximum - points.shape[0], -1),
                        ),
                        dim=0,
                    )
                    for points in group
                ]
            )
            ucb, lcb = self._confidence(padded)
            bound = ucb if upper else lcb
            if bound.shape != padded.shape[:-1]:
                # A user/test override written for the historical two-
                # dimensional confidence input may not preserve leading batch
                # dimensions.  Keep that extension point correct by falling
                # back to the original per-center calling convention.
                scored.extend(
                    (
                        points,
                        self._confidence(points)[0 if upper else 1].detach().clone(),
                    )
                    for points in group
                )
                continue
            scored.extend(
                (points, bound[position, : points.shape[0]].detach().clone())
                for position, points in enumerate(group)
            )
        return scored

    def _batched_raw_inner_ucb(
        self,
        centers: torch.Tensor,
        *,
        candidates: torch.Tensor | None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Build and score raw UCB tables without changing center order."""

        return self._batched_raw_inner_bound(
            centers,
            candidates=candidates,
            upper=True,
        )

    def _robust_ucb_from_raw_table(
        self,
        center: torch.Tensor,
        points: torch.Tensor,
        ucb: torch.Tensor,
        *,
        candidates: torch.Tensor | None,
        candidate_dataset_indices: torch.Tensor | None,
    ) -> tuple[float, torch.Tensor, dict[str, float]]:
        """Finish one robust-UCB solve from an already scored raw table."""

        if candidates is not None:
            indices = None
            if candidate_dataset_indices is not None:
                # Exact candidate filtering in _perturbation_set preserves row
                # order, so the same feasibility mask aligns hidden indices.
                lower, upper = self._bounds_on_algorithm_device()
                feasible = self._distance(center, candidates) <= (
                    self.config.perturbation_radius + _FEASIBILITY_TOLERANCE
                )
                feasible &= (
                    (candidates >= lower - _FEASIBILITY_TOLERANCE)
                    & (candidates <= upper + _FEASIBILITY_TOLERANCE)
                ).all(dim=-1)
                indices = candidate_dataset_indices[feasible]
            try:
                position = self._best_position(ucb, points, maximize=False, dataset_indices=indices)
            except ValueError as error:
                raise StableOPTInnerSolverError(
                    "The finite inner UCB table contains no finite value.", algorithm=self
                ) from error
            point = points[position].detach().clone()
            return (
                float(ucb[position].item()),
                point,
                {
                    "feasibility_residual": self._feasibility_residual(
                        point,
                        center=center,
                        radius=self.config.perturbation_radius,
                    ),
                    "optimizer_success": 1.0,
                    "solver_calls": 0.0,
                },
            )

        def objective(point: torch.Tensor) -> float:
            """Return scalar UCB for continuous inner minimization."""

            value, _ = self._confidence(point.unsqueeze(0))
            return float(value[0].item())

        point, value, certificate = self._continuous_multistart_minimize(
            solver=self.config.inner_solver,
            objective=objective,
            raw_points=points,
            raw_values=ucb,
            restarts=self.config.inner_restarts,
            center=center,
            radius=self.config.perturbation_radius,
            error_type=StableOPTInnerSolverError,
            label="StableOPT inner UCB minimization",
            objective_gradient=lambda point: self._confidence_gradient(point, upper=True),
        )
        return value, point, certificate

    def _robust_ucb_value(
        self,
        center: torch.Tensor,
        *,
        candidates: torch.Tensor | None = None,
        candidate_dataset_indices: torch.Tensor | None = None,
    ) -> tuple[float, torch.Tensor, dict[str, float]]:
        """Minimize UCB over one center's perturbation neighborhood.

        Finite mode selects an exact public candidate row. Continuous mode
        refines deterministic Sobol starts and returns the independently
        recomputed feasibility certificate with the pessimistic query point.
        """

        points = self._perturbation_set(center, candidates=candidates)
        ucb, _ = self._confidence(points)
        return self._robust_ucb_from_raw_table(
            center,
            points,
            ucb,
            candidates=candidates,
            candidate_dataset_indices=candidate_dataset_indices,
        )

    def _robust_ucb_center_gradient(
        self,
        center: torch.Tensor,
        query: torch.Tensor,
        certificate: Mapping[str, float],
    ) -> torch.Tensor:
        """Differentiate the implemented nested UCB value with respect to center.

        For a successful inner SLSQP refinement this is the standard envelope
        derivative through the active perturbation constraint.  If the stable
        tie rule retained a raw Sobol point, it is instead the exact derivative
        of that selected point's affine local-box map.  Thus outer SLSQP avoids
        finite-differencing six complete inner programs while optimizing the
        same deterministic nested objective.
        """

        query_gradient = self._confidence_gradient(query, upper=True)
        radius = self.config.perturbation_radius
        if radius == 0.0:
            return query_gradient
        optimized = bool(certificate.get("optimizer_success", 0.0))
        lower, upper = self._bounds_on_algorithm_device()
        local_lower = torch.maximum(lower, center - radius)
        local_upper = torch.minimum(upper, center + radius)
        tolerance = max(10.0 * self.config.solver_tolerance, 1e-7)
        if optimized:
            if self.config.distance.norm == "linf":
                active_lower = torch.isclose(
                    query,
                    local_lower,
                    rtol=0.0,
                    atol=tolerance,
                ) & (center - radius > lower + tolerance)
                active_upper = torch.isclose(
                    query,
                    local_upper,
                    rtol=0.0,
                    atol=tolerance,
                ) & (center + radius < upper - tolerance)
                return torch.where(active_lower | active_upper, query_gradient, 0.0)
            delta = query - center
            distance = torch.linalg.vector_norm(delta)
            if float(distance.item()) >= radius - tolerance and float(distance.item()) > 0.0:
                direction = delta / distance
                return torch.dot(query_gradient, direction) * direction
            return torch.zeros_like(center)

        # A retained raw point is either the explicit center or an affine map
        # of the cached unit Sobol coordinate into the clipped local box.
        if torch.equal(query, center):
            return query_gradient
        width = local_upper - local_lower
        unit = torch.where(width > 0, (query - local_lower) / width, 0.0)
        lower_derivative = (center - radius > lower).to(self.dtype)
        upper_derivative = (center + radius < upper).to(self.dtype)
        query_center_derivative = lower_derivative + unit * (upper_derivative - lower_derivative)
        return query_gradient * query_center_derivative

    def _select_robust_center(
        self,
        *,
        candidates: torch.Tensor | None,
        dataset_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, float]:
        """Maximize the neighborhood-minimum UCB over robust centers.

        Both centers and their inner perturbations use physical environment
        coordinates. Finite ties are resolved by dataset index; continuous
        ties use normalized lexicographic order through the shared solver.
        """

        centers = candidates
        if centers is None:
            centers = self._sample_design(
                self.bounds,
                self.config.outer_raw_samples,
                phase="acquisition",
                seed=self.seed + 15485863 * (self.iteration + 1),
            )
        values: list[float] = []
        solver_centers = centers
        solver_restarts = self.config.outer_restarts
        outer_raw_details: dict[int, tuple[float, torch.Tensor, dict[str, float]]] = {}
        selected: list[int] = []
        native_robust_method = (
            "_robust_ucb_value" not in self.__dict__
            and type(self)._robust_ucb_value is StableOPT._robust_ucb_value
        )
        if native_robust_method:
            raw_tables = self._batched_raw_inner_ucb(centers, candidates=candidates)
            if candidates is None:

                def refine(position: int) -> float:
                    """Return one exact nested raw-center objective."""

                    points, ucb = raw_tables[position]
                    details = self._robust_ucb_from_raw_table(
                        centers[position],
                        points,
                        ucb,
                        candidates=None,
                        candidate_dataset_indices=None,
                    )
                    outer_raw_details[position] = details
                    return details[0]

                selected, exact, certificate = self._certified_top_refined_positions(
                    raw_inner_values=[ucb for _, ucb in raw_tables],
                    refine=refine,
                    count=self.config.outer_restarts,
                )
                index = torch.tensor(selected, dtype=torch.long, device=self.device)
                solver_centers = centers[index]
                values = [exact[position] for position in selected]
                solver_restarts = len(selected)
                self._last_outer_pruning_certificate = certificate
            else:
                for center, (points, ucb) in zip(centers, raw_tables, strict=True):
                    value, _, _ = self._robust_ucb_from_raw_table(
                        center,
                        points,
                        ucb,
                        candidates=candidates,
                        candidate_dataset_indices=dataset_indices,
                    )
                    values.append(value)
                self._last_outer_pruning_certificate = {
                    "raw_centers": float(len(values)),
                    "refined_centers": float(len(values)),
                    "pruned_centers": 0.0,
                    "largest_pruned_upper_bound": float("-inf"),
                }
        else:
            # Respect instance monkeypatches and subclass overrides of the
            # documented nested objective boundary.
            for center in centers:
                value, _, _ = self._robust_ucb_value(
                    center,
                    candidates=candidates,
                    candidate_dataset_indices=dataset_indices,
                )
                values.append(value)
            self._last_outer_pruning_certificate = {
                "raw_centers": float(len(values)),
                "refined_centers": float(len(values)),
                "pruned_centers": 0.0,
                "largest_pruned_upper_bound": float("-inf"),
            }
        tensor = torch.tensor(values, dtype=self.dtype, device=self.device)
        if dataset_indices is not None:
            try:
                position = self._best_position(
                    tensor,
                    centers,
                    maximize=True,
                    dataset_indices=dataset_indices,
                )
            except ValueError as error:
                raise StableOPTOuterSolverError(
                    "No finite robust UCB center was found.", algorithm=self
                ) from error
            return centers[position].detach().clone(), float(tensor[position].item())

        robust_cache: dict[tuple[float, ...], tuple[float, torch.Tensor, dict[str, float]]] = {
            self._objective_cache_key(centers[position]): details
            for position, details in outer_raw_details.items()
            if position in selected
        }

        def robust_evaluation(
            point: torch.Tensor,
        ) -> tuple[float, torch.Tensor, dict[str, float]]:
            """Evaluate and share one deterministic nested outer objective."""

            key = self._objective_cache_key(point)
            cached = robust_cache.get(key)
            if cached is None:
                cached = self._robust_ucb_value(point)
                robust_cache[key] = cached
            return cached

        def objective(point: torch.Tensor) -> float:
            """Negate robust UCB so the outer solver can minimize it."""

            return -robust_evaluation(point)[0]

        def objective_gradient(point: torch.Tensor) -> torch.Tensor:
            """Return the nested robust-UCB envelope derivative."""

            _, query, certificate = robust_evaluation(point)
            return -self._robust_ucb_center_gradient(point, query, certificate)

        try:
            center, negative_value, _ = self._continuous_multistart_minimize(
                solver=self.config.outer_solver,
                objective=objective,
                raw_points=solver_centers,
                raw_values=-tensor,
                restarts=solver_restarts,
                center=None,
                radius=None,
                error_type=StableOPTOuterSolverError,
                label="StableOPT outer robust-UCB maximization",
                objective_gradient=objective_gradient,
            )
        except StableOPTInnerSolverError as error:
            raise StableOPTOuterSolverError(
                f"Nested inner solve failed during outer optimization: {error}",
                algorithm=self,
            ) from error
        return center, -negative_value

    def _select_internal_query_point(
        self,
        center: torch.Tensor,
        *,
        candidates: torch.Tensor | None,
        candidate_dataset_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Minimize LCB inside the selected center's perturbation set.

        The returned point is StableOPT's internal physical query. An
        adversarial environment may subsequently evaluate a different point;
        that distinction is preserved by :meth:`_tell_impl`.
        """

        points = self._perturbation_set(center, candidates=candidates)
        _, lcb = self._confidence(points)
        if candidates is not None:
            indices = None
            if candidate_dataset_indices is not None:
                lower, upper = self._bounds_on_algorithm_device()
                feasible = self._distance(center, candidates) <= (
                    self.config.perturbation_radius + _FEASIBILITY_TOLERANCE
                )
                feasible &= (
                    (candidates >= lower - _FEASIBILITY_TOLERANCE)
                    & (candidates <= upper + _FEASIBILITY_TOLERANCE)
                ).all(dim=-1)
                indices = candidate_dataset_indices[feasible]
            try:
                position = self._best_position(lcb, points, maximize=False, dataset_indices=indices)
            except ValueError as error:
                raise StableOPTInnerSolverError(
                    "The finite inner LCB table contains no finite value.", algorithm=self
                ) from error
            return points[position].detach().clone(), float(lcb[position].item())

        def objective(point: torch.Tensor) -> float:
            """Return scalar LCB for continuous inner minimization."""

            _, value = self._confidence(point.unsqueeze(0))
            return float(value[0].item())

        point, value, _ = self._continuous_multistart_minimize(
            solver=self.config.inner_solver,
            objective=objective,
            raw_points=points,
            raw_values=lcb,
            restarts=self.config.inner_restarts,
            center=center,
            radius=self.config.perturbation_radius,
            error_type=StableOPTInnerSolverError,
            label="StableOPT inner LCB minimization",
            objective_gradient=lambda point: self._confidence_gradient(point, upper=False),
        )
        return point, value

    def _initial_action(
        self, round_info: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Choose the deterministic finite or Sobol warm-start action."""

        if "candidate_actions" in round_info:
            table = self.prepare_discrete_candidates(
                candidate_actions=round_info["candidate_actions"],
                candidate_dataset_indices=round_info["candidate_dataset_indices"],
                round_info=round_info,
            )
            position = int(torch.argmin(table.dataset_indices).item())
            return table.environment_actions[position], table.dataset_indices
        point = self._sample_design(
            self.bounds,
            1,
            phase="initial",
            seed=self.seed + self.iteration,
        )[0]
        return point, None

    def _ask_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Return one internal pessimistic query in ``D_env`` coordinates.

        The robust center, internal perturbation, and confidence values are
        snapshotted with the pending request so ``tell`` can update state
        transactionally without recomputing the decision.
        """

        if self.state.y_train.numel() < self.config.initial_points:
            query, _ = self._initial_action(round_info)
            center = query.detach().clone()
            robust_ucb = float("nan")
            robust_lcb = float("nan")
        else:
            self._fit_model()
            candidates = None
            indices = None
            if "candidate_actions" in round_info:
                table = self.prepare_discrete_candidates(
                    candidate_actions=round_info["candidate_actions"],
                    candidate_dataset_indices=round_info["candidate_dataset_indices"],
                    round_info=round_info,
                )
                candidates, indices = table.environment_actions, table.dataset_indices
            center, robust_ucb = self._select_robust_center(
                candidates=candidates, dataset_indices=indices
            )
            query, robust_lcb = self._select_internal_query_point(
                center,
                candidates=candidates,
                candidate_dataset_indices=indices,
            )
        delta = query - center
        self._last_acquisition_value = None if math.isnan(robust_ucb) else robust_ucb
        self._record_last_ask(
            actions=query,
            round_info=round_info,
            model_inputs=query,
            algorithm_metadata={
                "robust_center": center,
                "internal_delta": delta,
                "robust_ucb": robust_ucb,
                "robust_lcb": robust_lcb,
            },
        )
        return query

    def _tell_impl(
        self,
        *,
        action: torch.Tensor,
        feedback: Mapping[str, Any],
        round_info: Mapping[str, Any],
    ) -> None:
        """Commit learner-visible feedback while preserving all coordinates.

        The GP is trained on ``feedback['evaluated_action']`` because that is
        where the observation was obtained. The robust center, StableOPT's
        internal query, and an optional public external perturbation remain
        separate audit fields; no hidden attack state is accessed.
        """

        del round_info
        assert self.last_ask is not None
        metadata = self.last_ask.algorithm_metadata
        evaluated = feedback["evaluated_action"].reshape(1, -1)
        observation = feedback["observation"].reshape(-1)
        variance_raw = feedback.get("observation_noise_variance")
        variance = (
            torch.zeros_like(observation)
            if variance_raw is None
            else torch.as_tensor(variance_raw, dtype=self.dtype, device=self.device).reshape(-1)
        )
        self.state.X_query_train = torch.cat((self.state.X_query_train, evaluated))
        self.state.y_train = torch.cat((self.state.y_train, observation))
        self.state.observation_noise_variance = torch.cat(
            (self.state.observation_noise_variance, variance)
        )
        center = torch.as_tensor(metadata["robust_center"], dtype=self.dtype, device=self.device)
        internal_delta = torch.as_tensor(
            metadata["internal_delta"], dtype=self.dtype, device=self.device
        )
        self.state.robust_centers.append(center.detach().clone())
        self.state.query_perturbations.append(internal_delta.detach().clone())
        self.state.query_points.append(action[0].detach().clone())
        self.state.evaluated_points.append(evaluated[0].detach().clone())
        external = feedback.get("perturbation")
        self.state.external_perturbations.append(
            None
            if external is None
            else torch.as_tensor(external, dtype=self.dtype, device=self.device)
            .reshape(-1)
            .detach()
            .clone()
        )
        self.state.robust_ucb_values.append(float(metadata["robust_ucb"]))
        self.state.robust_lcb_values.append(float(metadata["robust_lcb"]))
        # Keep the preceding fitted model intact until ``_fit_model`` rebuilds
        # the expanded table.  Its learned module state is the deterministic
        # warm start for the next exact-MLE fit; rebuilding here would discard
        # that information before the fitter can capture it.
        self._posterior_marginal_cache = None
        self._recommendation_cache = None

    def _robust_recommendation(self) -> tuple[torch.Tensor | None, float | None]:
        """Select the best recorded pre-update robust LCB (paper Eq. 17).

        Recomputing all historical bounds with the final posterior changes the
        paper's recommendation and makes a read-only result call run expensive
        fits/solvers. The bounds needed here are already recorded by ask/tell.
        """

        if self._recommendation_cache is not None:
            action, value = self._recommendation_cache
            return None if action is None else action.detach().clone(), value
        if not self.state.robust_centers:
            return self._best_observed()
        value_tensor = torch.tensor(
            self.state.robust_lcb_values, dtype=self.dtype, device=self.device
        )
        finite = torch.isfinite(value_tensor)
        if not finite.any():
            return self._best_observed()
        positions = torch.nonzero(finite, as_tuple=False).reshape(-1)
        centers = torch.stack(self.state.robust_centers)[positions]
        value_tensor = value_tensor[positions]
        position = self._best_position(value_tensor, centers, maximize=True)
        value = float(value_tensor[position].item())
        recommendation = (
            centers[position].detach().clone(),
            value if self.maximize else -value,
        )
        self._recommendation_cache = recommendation
        return recommendation[0].detach().clone(), recommendation[1]

    def _recommendation_impl(self) -> tuple[torch.Tensor | None, float | None]:
        """Expose the robust-center recommendation to the shared result API."""

        return self._robust_recommendation()

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Summarize robust confidence and coordinate-accounting state."""

        return {
            "robust_centers": len(self.state.robust_centers),
            "last_robust_ucb": self.state.robust_ucb_values[-1]
            if self.state.robust_ucb_values
            else None,
            "last_robust_lcb": self.state.robust_lcb_values[-1]
            if self.state.robust_lcb_values
            else None,
            "internal_external_coordinates_preserved": True,
            "outer_raw_refinement_certificate": dict(self._last_outer_pruning_certificate),
        }

    def _result_metadata_impl(self) -> Mapping[str, Any]:
        """Return detached center, internal-query, and evaluated-point traces."""

        has_confidence_bound = bool(self.state.robust_centers) and any(
            math.isfinite(value) for value in self.state.robust_lcb_values
        )
        fallback_kind = "observed_reward" if self.observed_values.numel() else "unavailable"
        fallback_rule = "best_observed" if self.observed_values.numel() else "unavailable"
        return {
            "recommendation_value_kind": (
                "historical_robust_confidence_bound" if has_confidence_bound else fallback_kind
            ),
            "recommendation_rule": "paper_eq_17" if has_confidence_bound else fallback_rule,
            "robust_centers": [item.detach().clone() for item in self.state.robust_centers],
            "query_points": [item.detach().clone() for item in self.state.query_points],
            "evaluated_points": [item.detach().clone() for item in self.state.evaluated_points],
        }

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize mutable proposal, observation, and audit progress."""

        return {
            "perturbation_radius": self.config.perturbation_radius,
            "distance_norm": self.config.distance.norm,
            "state": {
                "X_query_train": self.state.X_query_train,
                "y_train": self.state.y_train,
                "observation_noise_variance": self.state.observation_noise_variance,
                "robust_centers": self.state.robust_centers,
                "query_perturbations": self.state.query_perturbations,
                "query_points": self.state.query_points,
                "evaluated_points": self.state.evaluated_points,
                "external_perturbations": self.state.external_perturbations,
                "robust_ucb_values": self.state.robust_ucb_values,
                "robust_lcb_values": self.state.robust_lcb_values,
            },
        }

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Validate checkpoint geometry and required StableOPT state sections."""

        del strict
        if (
            state.get("perturbation_radius") != self.config.perturbation_radius
            or state.get("distance_norm") != self.config.distance.norm
        ):
            raise StableOPTCheckpointError(
                "Stored perturbation geometry is incompatible.", algorithm=self
            )
        if not isinstance(state.get("state"), Mapping):
            raise StableOPTCheckpointError("StableOPT state is incomplete.", algorithm=self)

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Restore validated traces and reset the GP for a fresh fit."""

        del strict
        raw = state["state"]
        try:
            self.state = StableOPTState(
                X_query_train=torch.as_tensor(
                    raw["X_query_train"], dtype=self.dtype, device=self.device
                ),
                y_train=torch.as_tensor(raw["y_train"], dtype=self.dtype, device=self.device),
                observation_noise_variance=torch.as_tensor(
                    raw["observation_noise_variance"], dtype=self.dtype, device=self.device
                ),
                robust_centers=[
                    torch.as_tensor(x, dtype=self.dtype, device=self.device)
                    for x in raw["robust_centers"]
                ],
                query_perturbations=[
                    torch.as_tensor(x, dtype=self.dtype, device=self.device)
                    for x in raw["query_perturbations"]
                ],
                query_points=[
                    torch.as_tensor(x, dtype=self.dtype, device=self.device)
                    for x in raw["query_points"]
                ],
                evaluated_points=[
                    torch.as_tensor(x, dtype=self.dtype, device=self.device)
                    for x in raw["evaluated_points"]
                ],
                external_perturbations=[
                    None if x is None else torch.as_tensor(x, dtype=self.dtype, device=self.device)
                    for x in raw["external_perturbations"]
                ],
                robust_ucb_values=[float(x) for x in raw["robust_ucb_values"]],
                robust_lcb_values=[float(x) for x in raw["robust_lcb_values"]],
            )
            self.gp.reset()
            self._posterior_marginal_cache = None
            self._recommendation_cache = None
            self._inner_unit_design_cache = None
            self._last_outer_pruning_certificate = {}
        except Exception as error:
            raise StableOPTCheckpointError(
                f"Invalid StableOPT checkpoint: {error}", algorithm=self
            ) from error

    def _reset_impl(self) -> None:
        """Clear all mutable StableOPT and GP state."""

        self.state = self._empty_state()
        self.gp.reset()
        self._posterior_marginal_cache = None
        self._recommendation_cache = None
        self._inner_unit_design_cache = None
        self._last_outer_pruning_certificate = {}


__all__ = [
    "DistanceSpec",
    "L2Distance",
    "LinfDistance",
    "MultiStartSolver",
    "RobustInnerSolverSpec",
    "RobustOuterSolverSpec",
    "StableOPT",
    "StableOPTConfig",
    "StableOPTState",
    "theoretical_beta",
]
