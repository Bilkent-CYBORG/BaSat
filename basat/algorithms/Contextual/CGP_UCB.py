"""Contextual Gaussian-process UCB with action-context kernel composition."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from numbers import Real
from types import MappingProxyType
from typing import Any, Literal

import gpytorch
import numpy as np
import torch

from basat.models.Acquisitions import DiscreteAcquisitionCandidates
from basat.models.FixedKernelGaussianProcess import FixedKernelGaussianProcess
from basat.models.GaussianProcess import GaussianProcess, GaussianProcessConfig
from basat.models.Kernels import KernelConfig, build_kernel
from basat.models.utils import Bounds, to_numpy

from ..algorithm import (
    Algorithm,
    AlgorithmConfigurationError,
    AlgorithmDataError,
    resolve_batch_configuration,
)
from ..helper import SourceRecord
from .Exception.exception_CGP_UCB import (
    CGP_UCBAcquisitionError,
    CGP_UCBCheckpointError,
    CGP_UCBContextError,
    CGP_UCBDataCompatibilityError,
    CGP_UCBException,
    CGP_UCBKernelError,
)

_SOLVER_FEASIBILITY_TOLERANCE = 1e-8
_ABSOLUTE_TIE_TOLERANCE = 1e-12
_RELATIVE_TIE_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class AcquisitionOptimizerSpec:
    """Continuous action-only UCB optimizer settings.

    Parameters
    ----------
    raw_samples:
        Sobol actions used to seed fixed-context acquisition optimization.
    restarts:
        Highest-scoring starts refined by the continuous optimizer.
    maxiter:
        Maximum iterations allowed for each acquisition restart.
    fit_maxiter:
        Maximum GP hyperparameter-optimization iterations per refit. Set to
        zero to condition on observations with fixed covariance parameters.
    solver_tolerance:
        Optimizer termination tolerance; action-bound feasibility is
        independently recomputed after each solve.
    """

    raw_samples: int = 512
    restarts: int = 10
    maxiter: int = 200
    fit_maxiter: int = 100
    solver_tolerance: float = 1e-6


@dataclass(frozen=True, slots=True)
class CGPUCBConfig:
    """Immutable contextual GP-UCB configuration.

    Exactly one of ``action_bounds`` and ``action_set`` defines the function-
    backed action domain.  Data-backed rounds always use the environment's
    currently exposed candidates.

    Parameters
    ----------
    action_bounds:
        Continuous physical action box. When provided, it must exactly match
        the algorithm's constructor bounds in ``D_action`` coordinates.
    action_set:
        Alternative finite action table with shape ``(M, D_action)``. Its
        coordinates are visible, but it contains no response values.
    action_kernel:
        Kernel specification operating on the action slice of joint GP inputs.
    context_kernel:
        Kernel specification operating on the learner-visible context slice.
    kernel_combination:
        Product or additive composition of the action and context kernels.
    beta:
        ``"finite-theory"``, a non-negative scalar, or a round-indexed
        callable supplying the UCB confidence multiplier.
    delta:
        Confidence-failure probability used by the finite-theory schedule.
    finite_domain_cardinality:
        Known finite joint action-context cardinality required when the
        theoretical schedule cannot infer it from a finite action table.
    acquisition_optimizer:
        Settings for continuous optimization with context held fixed.
    initial_points:
        Sequential observations collected before GP-guided acquisition.
    gp_config:
        Optional noise, mean, and fitting settings. The action-context kernel
        is still constructed from ``action_kernel`` and ``context_kernel``.
        With ``fit_maxiter=0``, supply a calibrated fixed-noise, zero-mean GP
        and center observations using calibration data outside the algorithm.
    gp_backend:
        ``"standard"`` uses the ordinary fitted GP. ``"fixed_incremental"``
        performs exact fixed-parameter conditioning on CPU/float64 for finite
        action tables; requires zero mean, fixed noise, no standardization,
        and ``fit_maxiter=0``. All observed history is retained.
    batch_size:
        Requested proposal count. Contextual version 1 requires ``1``.
    seed:
        Seed for algorithm-owned candidate generation and deterministic replay.
    """

    action_bounds: Bounds | None = None
    action_set: torch.Tensor | np.ndarray | None = None
    action_kernel: KernelConfig | gpytorch.kernels.Kernel = field(default_factory=KernelConfig)
    context_kernel: KernelConfig | gpytorch.kernels.Kernel = field(default_factory=KernelConfig)
    kernel_combination: Literal["product", "additive"] = "product"
    beta: Callable[[int], float] | float | Literal["finite-theory"] = "finite-theory"
    delta: float = 0.1
    finite_domain_cardinality: int | None = None
    acquisition_optimizer: AcquisitionOptimizerSpec = field(
        default_factory=AcquisitionOptimizerSpec
    )
    initial_points: int = 1
    batch_size: int | None = 1
    q_batch: int | None = None
    seed: int = 1
    gp_config: GaussianProcessConfig | None = None
    gp_backend: Literal["standard", "fixed_incremental"] = "standard"


@dataclass(slots=True)
class CGPUCBState:
    """Mutable joint-surrogate and context history.

    Attributes
    ----------
    actions_train:
        Evaluated physical actions with shape ``(N, D_action)``.
    contexts_train:
        Realized learner-visible contexts with shape ``(N, D_context)``.
    joint_inputs_train:
        Concatenated GP inputs with shape
        ``(N, D_action + D_context)``.
    y_train:
        Learner-visible observations with shape ``(N,)``.
    observation_noise_variance:
        Known observation-noise variances with shape ``(N,)``; zeros denote
        feedback without a supplied variance.
    current_context:
        Context frozen for the outstanding ``ask()``, with shape
        ``(D_context,)``, or ``None`` when no action is pending.
    context_history:
        Contexts consumed by completed rounds. Context is public before action;
        no hidden environment state is retained.
    """

    actions_train: torch.Tensor
    contexts_train: torch.Tensor
    joint_inputs_train: torch.Tensor
    y_train: torch.Tensor
    observation_noise_variance: torch.Tensor
    current_context: torch.Tensor | None
    context_history: list[torch.Tensor]


class CGP_UCB(Algorithm):
    """Select actions by UCB while holding the revealed context fixed.

    ``bounds`` and ``action_bounds`` both describe the action coordinates. The
    revealed context is appended internally only when the GP model is fitted.

    Examples
    --------
    >>> import basat
    >>> action_bounds = basat.Bounds(lower=[-1.0], upper=[1.0])
    >>> algorithm = basat.CGP_UCB(
    ...     bounds=action_bounds,
    ...     config=basat.CGPUCBConfig(
    ...         action_bounds=action_bounds, beta=2.0, initial_points=3, seed=7
    ...     ),
    ... )
    """

    ACQUISITION_POLICY = "contextual GP-UCB"
    CHECKPOINT_ADDED_CONFIG_DEFAULTS = MappingProxyType(
        {**Algorithm.CHECKPOINT_ADDED_CONFIG_DEFAULTS, "gp_config": None, "gp_backend": "standard"}
    )

    EXCEPTION_CLASS = CGP_UCBException
    native_batching = False
    native_environments = frozenset({"contextual"})
    required_round_info = frozenset({"context"})
    required_feedback = frozenset({"observation", "evaluated_action", "context"})
    required_capabilities = frozenset({"contextual", "revealed_context"})
    SOURCE = SourceRecord(
        paper_title="Contextual Gaussian Process Bandit Optimization",
        paper_year=2011,
        paper_url="https://proceedings.neurips.cc/paper/2011/hash/f3f1b7fc5a8779a9e618e1f23a7b7860-Abstract.html",
        repository_url="https://github.com/ardaegeunlu/Contextual-Gaussian-Process-Bandit-Optimization",
        repository_status="community",
        notes="The Krause-Ong paper governs semantics; the repository is a regression reference.",
    )

    def __init__(
        self,
        *,
        bounds: Bounds,
        config: CGPUCBConfig,
        maximize: bool = True,
        dtype: torch.dtype | str = torch.float32,
        device: torch.device | str = "cpu",
        plot: bool = False,
        verbose: bool = False,
        verbose_setting: bool = False,
        verbose_iter_by_iter: bool = False,
    ) -> None:
        """Initialize contextual GP-UCB in environment action coordinates.

        Parameters
        ----------
        bounds:
            Physical action bounds with dimension ``D_action``.
        config:
            Immutable kernel, confidence-schedule, action-domain, optimizer,
            batching, and seed configuration.
        maximize:
            Whether UCB is formed for reward maximization.  Minimization uses
            the shared quality-sign convention.
        dtype, device:
            Tensor precision and execution device for every model-owned value.
        plot, verbose, verbose_setting, verbose_iter_by_iter:
            Presentation controls inherited from :class:`Algorithm`; they do
            not affect proposals.

        Notes
        -----
        The GP dimension remains unknown until a context is observed.  It is
        then frozen to ``D_action + D_context`` transactionally.
        """

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
        self._validate_action_domain()
        if config.beta != "finite-theory":
            self.invalidate_source_guarantees("basat.user_defined_cgp_ucb_beta")
        self.context_dimension: int | None = None
        self.model_input_dimension = None
        self.gp: GaussianProcess | None = None
        self.state = CGPUCBState(
            actions_train=torch.empty(
                (0, self.action_dimension), dtype=self.dtype, device=self.device
            ),
            contexts_train=torch.empty((0, 0), dtype=self.dtype, device=self.device),
            joint_inputs_train=torch.empty((0, 0), dtype=self.dtype, device=self.device),
            y_train=torch.empty(0, dtype=self.dtype, device=self.device),
            observation_noise_variance=torch.empty(0, dtype=self.dtype, device=self.device),
            current_context=None,
            context_history=[],
        )

    @staticmethod
    def _validate_config(config: CGPUCBConfig) -> None:
        """Validate immutable sequential, kernel, beta, and optimizer controls."""

        if not isinstance(config, CGPUCBConfig):
            raise AlgorithmConfigurationError("config must be CGPUCBConfig.")
        _, q_batch = resolve_batch_configuration(
            batch_size=config.batch_size, q_batch=config.q_batch
        )
        if q_batch is not None and q_batch > 1:
            raise AlgorithmConfigurationError(
                "CGP-UCB does not define a joint q-acquisition; use batch_size for repeated "
                "independent actions under the revealed context."
            )
        if (config.action_bounds is None) == (config.action_set is None):
            raise AlgorithmConfigurationError(
                "Exactly one of action_bounds and action_set must be supplied."
            )
        if config.kernel_combination not in {"product", "additive"}:
            raise CGP_UCBKernelError("kernel_combination must be product or additive.")
        if (
            isinstance(config.delta, bool)
            or not isinstance(config.delta, Real)
            or not math.isfinite(config.delta)
            or not 0 < config.delta < 1
        ):
            raise AlgorithmConfigurationError("delta must lie strictly between zero and one.")
        if isinstance(config.beta, str):
            if config.beta != "finite-theory":
                raise AlgorithmConfigurationError(
                    "beta string values must be exactly 'finite-theory'."
                )
        elif not callable(config.beta) and (
            isinstance(config.beta, bool)
            or not isinstance(config.beta, Real)
            or not math.isfinite(config.beta)
            or config.beta < 0
        ):
            raise AlgorithmConfigurationError(
                "beta must be 'finite-theory', callable, or a finite non-negative scalar."
            )
        if config.finite_domain_cardinality is not None and (
            isinstance(config.finite_domain_cardinality, bool)
            or not isinstance(config.finite_domain_cardinality, int)
            or config.finite_domain_cardinality < 1
        ):
            raise AlgorithmConfigurationError(
                "finite_domain_cardinality must be a positive integer or None."
            )
        optimizer = config.acquisition_optimizer
        if not isinstance(optimizer, AcquisitionOptimizerSpec):
            raise AlgorithmConfigurationError(
                "acquisition_optimizer must be AcquisitionOptimizerSpec."
            )
        for name in ("raw_samples", "restarts", "maxiter"):
            value = getattr(optimizer, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise AlgorithmConfigurationError(f"{name} must be a positive integer.")
        if (
            isinstance(optimizer.fit_maxiter, bool)
            or not isinstance(optimizer.fit_maxiter, int)
            or optimizer.fit_maxiter < 0
        ):
            raise AlgorithmConfigurationError("fit_maxiter must be a non-negative integer.")
        if config.gp_config is not None and not isinstance(config.gp_config, GaussianProcessConfig):
            raise AlgorithmConfigurationError("gp_config must be GaussianProcessConfig or None.")
        if config.gp_backend not in {"standard", "fixed_incremental"}:
            raise AlgorithmConfigurationError(
                "gp_backend must be 'standard' or 'fixed_incremental'."
            )
        if config.gp_backend == "fixed_incremental" and (
            config.acquisition_optimizer.fit_maxiter != 0 or config.action_bounds is not None
        ):
            raise AlgorithmConfigurationError(
                "fixed_incremental requires finite actions and fit_maxiter=0."
            )
        if (
            isinstance(config.initial_points, bool)
            or not isinstance(config.initial_points, int)
            or config.initial_points < 1
        ):
            raise AlgorithmConfigurationError("initial_points must be a positive integer.")
        if (
            isinstance(optimizer.solver_tolerance, bool)
            or not isinstance(optimizer.solver_tolerance, Real)
            or not math.isfinite(optimizer.solver_tolerance)
            or optimizer.solver_tolerance <= 0
        ):
            raise AlgorithmConfigurationError("solver_tolerance must be finite and positive.")

    def _validate_action_domain(self) -> None:
        """Require one finite action domain aligned exactly with constructor bounds."""

        lower, upper = self._bounds_on_algorithm_device()
        if self.config.action_bounds is not None:
            if not isinstance(self.config.action_bounds, Bounds):
                raise AlgorithmConfigurationError("action_bounds must be a Bounds instance.")
            configured = self.config.action_bounds.to(dtype=self.dtype, device=self.device)
            if not torch.equal(configured.lower, lower) or not torch.equal(configured.upper, upper):
                raise AlgorithmConfigurationError(
                    "config.action_bounds must exactly match constructor bounds."
                )
            return
        try:
            actions = torch.as_tensor(self.config.action_set, dtype=self.dtype, device=self.device)
        except (TypeError, ValueError, RuntimeError) as error:
            raise AlgorithmConfigurationError("action_set must be numeric.") from error
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != self.action_dimension:
            raise AlgorithmConfigurationError("action_set must have shape (M, D_env) with M > 0.")
        if not torch.isfinite(actions).all() or not ((actions >= lower) & (actions <= upper)).all():
            raise AlgorithmConfigurationError("action_set must be finite and inside bounds.")

    def prepare_round_info(self, round_info: Mapping[str, Any] | None) -> dict[str, Any]:
        """Use a revealed context, falling back to an internal zero vector."""
        info = super().prepare_round_info(round_info)
        if info.get("context") is None:
            context = self._default_context()
            if self.context_dimension is None and info.get("context_support") is not None:
                support = torch.as_tensor(info["context_support"])
                if support.ndim == 2:
                    context = torch.zeros(support.shape[1], dtype=self.dtype, device=self.device)
            self._warn_environment_default("context", str(context.tolist()))
            info["context"] = context
            info["context_is_default"] = True
        return info

    def prepare_feedback(self, feedback: Mapping[str, Any], *, action: Any) -> dict[str, Any]:
        result = super().prepare_feedback(feedback, action=action)
        if result.get("context") is None:
            current = self.last_ask.round_info_snapshot.get("context") if self.last_ask else None
            result["context"] = (
                self._default_context()
                if current is None
                else torch.as_tensor(current, dtype=self.dtype, device=self.device).detach().clone()
            )
            self._warn_environment_default("feedback.context", str(result["context"].tolist()))
        return result

    def _default_context(self) -> torch.Tensor:
        """Internal neutral context used only when the environment omits it."""
        dimension = getattr(self, "context_dimension", None)
        if not dimension:
            kernel = self.config.context_kernel
            dimension = getattr(kernel, "ard_num_dims", None) or 1
        return torch.zeros(int(dimension), dtype=self.dtype, device=self.device)

    def _ensure_context_dimension(self, context: Any) -> torch.Tensor:
        """Validate one context and freeze the joint GP coordinate dimension.

        Parameters
        ----------
        context:
            Numeric learner-visible context, flattened to ``(D_context,)``.

        Returns
        -------
        torch.Tensor
            Detached context on the algorithm device.

        Raises
        ------
        CGP_UCBContextError
            If the context is empty, non-finite, non-numeric, or changes the
            dimension frozen by history or an earlier round.
        """

        try:
            tensor = torch.as_tensor(context, dtype=self.dtype, device=self.device).reshape(-1)
        except (TypeError, ValueError, RuntimeError) as error:
            raise CGP_UCBContextError("context must be numeric.", algorithm=self) from error
        if tensor.numel() == 0 or not torch.isfinite(tensor).all():
            raise CGP_UCBContextError("context must be a non-empty finite vector.", algorithm=self)
        dimension = int(tensor.numel())
        if self.context_dimension is None:
            self.context_dimension = dimension
            self.model_input_dimension = self.action_dimension + dimension
            try:
                self._build_gp()
            except Exception:
                # Context freezing is transactional: a malformed kernel must
                # not leave a partially dimensioned algorithm after a failed
                # ask or initialization attempt.
                self.context_dimension = None
                self.model_input_dimension = None
                self.gp = None
                raise
            self.state.contexts_train = torch.empty(
                (0, dimension), dtype=self.dtype, device=self.device
            )
            self.state.joint_inputs_train = torch.empty(
                (0, self.model_input_dimension), dtype=self.dtype, device=self.device
            )
        elif dimension != self.context_dimension:
            raise CGP_UCBContextError(
                f"Context dimension changed from {self.context_dimension} to {dimension}.",
                algorithm=self,
            )
        return tensor.detach().clone()

    @staticmethod
    def _activate_custom_kernel(
        kernel: gpytorch.kernels.Kernel, active_dims: tuple[int, ...]
    ) -> gpytorch.kernels.Kernel:
        """Copy a custom component kernel onto specified joint coordinates."""

        result = copy.deepcopy(kernel)
        result.active_dims = torch.tensor(active_dims, dtype=torch.long)
        return result

    def _component_kernel(
        self,
        specification: KernelConfig | gpytorch.kernels.Kernel,
        active_dims: tuple[int, ...],
    ) -> gpytorch.kernels.Kernel:
        """Build one action or context kernel on explicit active dimensions.

        A configured ARD count must equal the component dimension; it is never
        silently rewritten to the full action-context dimension.
        """

        assert self.model_input_dimension is not None
        if isinstance(specification, KernelConfig):
            if specification.ard_num_dims not in {None, len(active_dims)}:
                raise CGP_UCBKernelError(
                    "A component kernel's ard_num_dims must match its own "
                    "action or context coordinate dimension.",
                    algorithm=self,
                )
            adjusted = replace(
                specification,
                active_dims=active_dims,
            )
            return build_kernel(adjusted, input_dimension=self.model_input_dimension)
        return self._activate_custom_kernel(specification, active_dims)

    def _build_joint_kernel(self) -> gpytorch.kernels.Kernel:
        """Compose action and context kernels by product or additive semantics."""

        if self.context_dimension is None:
            raise CGP_UCBContextError("Context dimension is not frozen.", algorithm=self)
        action_dims = tuple(range(self.action_dimension))
        context_dims = tuple(
            range(self.action_dimension, self.action_dimension + self.context_dimension)
        )
        action_kernel = self._component_kernel(self.config.action_kernel, action_dims)
        context_kernel = self._component_kernel(self.config.context_kernel, context_dims)
        if self.config.kernel_combination == "product":
            return gpytorch.kernels.ProductKernel(action_kernel, context_kernel)
        return gpytorch.kernels.AdditiveKernel(action_kernel, context_kernel)

    def _build_gp(self) -> None:
        """Construct the joint-coordinate GP after context dimension freezing."""

        assert self.model_input_dimension is not None
        model_class = (
            FixedKernelGaussianProcess
            if self.config.gp_backend == "fixed_incremental"
            else GaussianProcess
        )
        self.gp = model_class(
            replace(
                self.config.gp_config or GaussianProcessConfig(),
                kernel=self._build_joint_kernel(),
                dtype=self.dtype,
                device=self.device,
            ),
            input_dimension=self.model_input_dimension,
        )

    def _joint_input(self, actions: Any, context: Any) -> torch.Tensor:
        """Append one fixed context to every action row.

        ``actions`` accepts ``(D_action,)`` or ``(N, D_action)`` and the result
        has shape ``(N, D_action + D_context)``.  The returned tensor is the GP
        coordinate representation; actions sent to the environment remain in
        the original ``D_action`` coordinates.
        """

        context_tensor = self._ensure_context_dimension(context)
        action_tensor = torch.as_tensor(actions, dtype=self.dtype, device=self.device)
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        if action_tensor.ndim != 2 or action_tensor.shape[1] != self.action_dimension:
            raise AlgorithmDataError("actions must have shape (N, D_env).")
        contexts = context_tensor.unsqueeze(0).expand(action_tensor.shape[0], -1)
        return torch.cat((action_tensor, contexts), dim=-1)

    def _initialize_from_environment_impl(self, data: Any) -> None:
        """Load learner-visible action-context history into the joint GP.

        Historical evaluated actions, realized contexts, observations, and
        optional known noise variances must be row aligned.  No hidden dataset
        response or unrevealed context is read by this method.
        """

        actions = torch.as_tensor(data.actions, dtype=self.dtype, device=self.device)
        y = torch.as_tensor(data.observations, dtype=self.dtype, device=self.device).reshape(-1)
        if y.numel() == 0:
            return
        if data.contexts is None:
            default = self._default_context()
            self._warn_environment_default("previous_data.contexts", str(default.tolist()))
            contexts = default.unsqueeze(0).expand(actions.shape[0], -1).clone()
        else:
            contexts = torch.as_tensor(data.contexts, dtype=self.dtype, device=self.device)
        if contexts.ndim == 1:
            contexts = contexts.unsqueeze(-1)
        self._ensure_context_dimension(contexts[0])
        assert self.context_dimension is not None and self.gp is not None
        if contexts.shape != (actions.shape[0], self.context_dimension):
            raise CGP_UCBDataCompatibilityError(
                "Historical context rows do not match the frozen dimension.", algorithm=self
            )
        evaluated_raw = getattr(data, "evaluated_actions", None)
        evaluated = (
            actions
            if evaluated_raw is None
            else torch.as_tensor(evaluated_raw, dtype=self.dtype, device=self.device)
        )
        joint = torch.cat((evaluated, contexts), dim=-1)
        variance_raw = getattr(data, "observation_noise_variance", None)
        configured_noise = (
            self.config.gp_config.noise_variance if self.config.gp_config is not None else None
        )
        variance = (
            torch.full_like(y, configured_noise or 0.0)
            if variance_raw is None
            else torch.as_tensor(variance_raw, dtype=self.dtype, device=self.device).reshape(-1)
        )
        self.state.actions_train = evaluated.detach().clone()
        self.state.contexts_train = contexts.detach().clone()
        self.state.joint_inputs_train = joint.detach().clone()
        self.state.y_train = y.detach().clone()
        self.state.observation_noise_variance = variance.detach().clone()
        self.gp.set_data(joint, y, variance)

    def _prepare_discrete_candidates_impl(
        self,
        *,
        candidate_actions: torch.Tensor,
        candidate_dataset_indices: torch.Tensor,
        round_info: Mapping[str, Any],
    ) -> DiscreteAcquisitionCandidates:
        """Attach the revealed context to exact data-backed action candidates.

        The environment actions and stable dataset indices are preserved
        unchanged; only their GP model representation is augmented.
        """

        joint = self._joint_input(candidate_actions, round_info["context"])
        return DiscreteAcquisitionCandidates(
            candidate_actions.detach().clone(),
            joint,
            candidate_dataset_indices.detach().clone(),
        )

    def _beta(self) -> float:
        """Resolve the current finite-theory or user confidence multiplier."""

        beta = self.config.beta
        if beta == "finite-theory":
            cardinality = self.config.finite_domain_cardinality
            if cardinality is None:
                raise CGP_UCBContextError(
                    "finite-theory beta requires known finite_domain_cardinality.",
                    algorithm=self,
                )
            t = self.iteration + 1
            value = 2.0 * math.log(cardinality * (t**2) * math.pi**2 / (6.0 * self.config.delta))
        else:
            try:
                value = float(beta(self.iteration + 1) if callable(beta) else beta)
            except (TypeError, ValueError, OverflowError) as error:
                raise CGP_UCBContextError(
                    "beta must resolve to a numeric scalar.", algorithm=self
                ) from error
        if not math.isfinite(value) or value < 0:
            raise CGP_UCBContextError("beta must resolve to a non-negative scalar.", algorithm=self)
        return value

    def _fit_model(self) -> None:
        """Fit the joint GP using all observed action-context rows and variances."""

        if self.state.y_train.numel() == 0:
            return
        assert self.gp is not None
        self.gp.set_data(
            self.state.joint_inputs_train,
            self.state.y_train,
            self.state.observation_noise_variance,
        )
        try:
            if self.config.acquisition_optimizer.fit_maxiter == 0:
                self.gp.condition_only()
            else:
                self.gp.fit(maxiter=self.config.acquisition_optimizer.fit_maxiter)
        except Exception as error:
            raise CGP_UCBKernelError(
                f"Contextual GP fitting failed: {error}", algorithm=self
            ) from error

    def _ucb(self, joint_inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate posterior UCB on ``(N, D_action + D_context)`` inputs."""

        assert self.gp is not None
        mean, std = self.gp.predict(joint_inputs)
        quality_mean = mean if self.maximize else -mean
        return quality_mean + math.sqrt(self._beta()) * std

    def _score_available_actions(
        self, actions: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate fixed-context UCB for physical actions of shape ``(N, D_action)``."""

        return self._ucb(self._joint_input(actions, context))

    @staticmethod
    def _stable_position(
        scores: torch.Tensor, actions: torch.Tensor, indices: torch.Tensor | None
    ) -> int:
        """Resolve UCB ties by source index or lexicographic physical action."""

        maximum = scores.max()
        tolerance = _ABSOLUTE_TIE_TOLERANCE + _RELATIVE_TIE_TOLERANCE * max(
            1.0, abs(float(maximum.item()))
        )
        tied = torch.nonzero(scores >= maximum - tolerance, as_tuple=False).reshape(-1)
        if indices is not None:
            return int(tied[torch.argmin(indices[tied])].item())
        rows = to_numpy(actions[tied])
        order = np.lexsort(tuple(rows[:, column] for column in reversed(range(rows.shape[1]))))
        return int(tied[int(order[0])].item())

    def _continuous_candidates(self) -> torch.Tensor:
        """Generate the deterministic raw design for continuous action search."""

        assert self.config.action_bounds is not None
        return self._sample_design(
            self.config.action_bounds,
            self.config.acquisition_optimizer.raw_samples,
            phase=(
                "initial"
                if self.state.y_train.numel() < self.config.initial_points
                else "acquisition"
            ),
            seed=self.seed + 67867967 * (self.iteration + 1),
        )

    def _refine_continuous_action(
        self,
        raw_actions: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Refine fixed-context UCB starts and certify the returned action.

        Parameters
        ----------
        raw_actions:
            Sobol starts in physical coordinates with shape ``(N, D_env)``.
        context:
            Learner-visible context held fixed throughout every objective call.

        Returns
        -------
        action, value:
            Lexicographically stable maximizer and its independently recomputed
            UCB value.

        Raises
        ------
        CGP_UCBAcquisitionError
            If no deterministic multistart solve returns a finite feasible point.
        """

        try:
            from scipy.optimize import minimize
        except ImportError as error:  # pragma: no cover - SciPy is a core dependency
            raise CGP_UCBAcquisitionError(
                "Continuous CGP-UCB optimization requires scipy.", algorithm=self
            ) from error

        raw_scores = self._score_available_actions(raw_actions, context)
        order = torch.argsort(raw_scores, descending=True, stable=True)
        restart_count = min(
            self.config.acquisition_optimizer.restarts,
            raw_actions.shape[0],
        )
        lower, upper = self._bounds_on_algorithm_device()
        scipy_bounds = list(zip(to_numpy(lower), to_numpy(upper), strict=True))
        accepted_actions: list[torch.Tensor] = []
        accepted_scores: list[float] = []

        def objective(candidate: np.ndarray) -> float:
            """Return negative fixed-context UCB at one SciPy action vector."""

            point = torch.as_tensor(candidate, dtype=self.dtype, device=self.device)
            value = self._score_available_actions(point.unsqueeze(0), context)[0]
            return -float(value.item())

        for raw_position in order[:restart_count]:
            start = to_numpy(raw_actions[int(raw_position.item())].double())
            try:
                result = minimize(
                    objective,
                    start,
                    method="L-BFGS-B",
                    bounds=scipy_bounds,
                    options={
                        "ftol": self.config.acquisition_optimizer.solver_tolerance,
                        "gtol": self.config.acquisition_optimizer.solver_tolerance,
                        "eps": max(
                            self.config.acquisition_optimizer.solver_tolerance,
                            1e-4 if self.dtype == torch.float32 else 1e-8,
                        ),
                        "maxiter": self.config.acquisition_optimizer.maxiter,
                    },
                )
            except (RuntimeError, ValueError, FloatingPointError):
                continue
            try:
                result_x = np.asarray(result.x, dtype=float)
                result_fun = float(result.fun)
            except (TypeError, ValueError):
                continue
            if (
                not result.success
                or result_x.shape != (self.action_dimension,)
                or not np.isfinite(result_x).all()
                or not math.isfinite(result_fun)
            ):
                continue
            point = torch.as_tensor(result_x, dtype=self.dtype, device=self.device)
            residual = torch.maximum(lower - point, point - upper).clamp_min(0.0).max()
            if float(residual.item()) > _SOLVER_FEASIBILITY_TOLERANCE:
                continue
            score = float(self._score_available_actions(point.unsqueeze(0), context)[0].item())
            if math.isfinite(score):
                accepted_actions.append(point.detach().clone())
                accepted_scores.append(score)

        if not accepted_actions:
            raise CGP_UCBAcquisitionError(
                "Every continuous fixed-context UCB restart failed certification.",
                algorithm=self,
            )
        actions = torch.stack(accepted_actions)
        scores = torch.tensor(accepted_scores, dtype=self.dtype, device=self.device)
        position = self._stable_position(scores, actions, None)
        return actions[position], accepted_scores[position]

    def _ask_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Select one action while holding the learner-visible context fixed.

        Data-backed rounds score only the exact candidates supplied by the
        environment.  Function-backed continuous rounds use certified
        multistart refinement after the configured initial-design phase.
        """

        context = self._ensure_context_dimension(round_info["context"])
        indices = None
        if "candidate_actions" in round_info:
            table = self.prepare_discrete_candidates(
                candidate_actions=round_info["candidate_actions"],
                candidate_dataset_indices=round_info["candidate_dataset_indices"],
                round_info=round_info,
            )
            actions, indices = table.environment_actions, table.dataset_indices
        elif self.config.action_set is not None:
            actions = torch.as_tensor(self.config.action_set, dtype=self.dtype, device=self.device)
        else:
            actions = self._continuous_candidates()
        if self.state.y_train.numel() < self.config.initial_points:
            scores = torch.zeros(actions.shape[0], dtype=self.dtype, device=self.device)
            position = self._stable_position(scores, actions, indices)
            action = actions[position]
            acquisition_value = math.nan
        else:
            self._fit_model()
            if self.config.action_bounds is not None and indices is None:
                action, acquisition_value = self._refine_continuous_action(actions, context)
            else:
                scores = self._score_available_actions(actions, context)
                position = self._stable_position(scores, actions, indices)
                action = actions[position]
                acquisition_value = float(scores[position].item())
        action = action.detach().clone()
        joint = self._joint_input(action, context)[0]
        self.state.current_context = context.detach().clone()
        self._last_acquisition_value = None if math.isnan(acquisition_value) else acquisition_value
        self._record_last_ask(
            actions=action,
            round_info=round_info,
            model_inputs=joint,
            algorithm_metadata={"context": context, "ucb": acquisition_value},
        )
        return action

    def _tell_impl(
        self,
        *,
        action: torch.Tensor,
        feedback: Mapping[str, Any],
        round_info: Mapping[str, Any],
    ) -> None:
        """Append evaluated action, matching context, observation, and variance.

        The context in feedback and round information must equal the context
        frozen at :meth:`ask`; a perturbed evaluated action is stored exactly
        as returned by the environment.
        """

        assert self.last_ask is not None and self.gp is not None
        context = self._ensure_context_dimension(round_info["context"])
        feedback_context = self._ensure_context_dimension(feedback["context"])
        pending_context = torch.as_tensor(
            self.last_ask.algorithm_metadata["context"], dtype=self.dtype, device=self.device
        )
        if not torch.equal(context, pending_context) or (
            not round_info.get("context_is_default", False)
            and not torch.equal(feedback_context, pending_context)
        ):
            raise CGP_UCBContextError(
                "tell context differs from the context held fixed during ask.", algorithm=self
            )
        if round_info.get("context_is_default", False):
            # The environment reveals the actual realization only after acting.
            # Preserve its observation coordinates rather than fabricating a context.
            context = feedback_context
        evaluated = feedback["evaluated_action"].reshape(1, -1)
        observation = feedback["observation"].reshape(-1)
        joint = torch.cat((evaluated, context.reshape(1, -1)), dim=-1)
        raw_variance = feedback.get("observation_noise_variance")
        configured_noise = (
            self.config.gp_config.noise_variance if self.config.gp_config is not None else None
        )
        variance = (
            torch.full_like(observation, configured_noise or 0.0)
            if raw_variance is None
            else torch.as_tensor(raw_variance, dtype=self.dtype, device=self.device).reshape(-1)
        )
        self.state.actions_train = torch.cat((self.state.actions_train, evaluated))
        self.state.contexts_train = torch.cat((self.state.contexts_train, context.reshape(1, -1)))
        self.state.joint_inputs_train = torch.cat((self.state.joint_inputs_train, joint))
        self.state.y_train = torch.cat((self.state.y_train, observation))
        self.state.observation_noise_variance = torch.cat(
            (self.state.observation_noise_variance, variance)
        )
        self.state.context_history.append(context.detach().clone())
        self.state.current_context = None
        self.gp.set_data(
            self.state.joint_inputs_train,
            self.state.y_train,
            self.state.observation_noise_variance,
        )

    def _recommendation_impl(self) -> tuple[torch.Tensor | None, float | None]:
        """Recommend the largest posterior-mean action at the latest context."""

        if not self.state.context_history or self.state.y_train.numel() == 0:
            return self._best_observed()
        context = self.state.context_history[-1]
        self._fit_model()
        if self.config.action_set is not None:
            actions = torch.as_tensor(self.config.action_set, dtype=self.dtype, device=self.device)
        else:
            actions = self._continuous_candidates()
        assert self.gp is not None
        mean, _ = self.gp.predict(self._joint_input(actions, context))
        quality = mean if self.maximize else -mean
        position = self._stable_position(quality, actions, None)
        return actions[position].detach().clone(), float(mean[position].item())

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Return contextual dimension, kernel, beta, and completed-round diagnostics."""

        return {
            "context_dimension": self.context_dimension,
            "kernel_combination": self.config.kernel_combination,
            "context_rounds": len(self.state.context_history),
            "beta_schedule": "finite-theory" if self.config.beta == "finite-theory" else "user",
        }

    def _result_metadata_impl(self) -> Mapping[str, Any]:
        """Return detached realized context history for the public result."""

        return {"context_history": [item.detach().clone() for item in self.state.context_history]}

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize contextual observations, progress, and kernel mode."""

        state = {
            "context_dimension": self.context_dimension,
            "kernel_combination": self.config.kernel_combination,
            "state": {name: getattr(self.state, name) for name in self.state.__dataclass_fields__},
        }
        # Ordinarily the observed history is sufficient to rebuild a fixed GP.
        # If emergency jitter was needed, also preserve its exact row locations
        # rather than replacing them with a different global regularizer.
        if isinstance(self.gp, FixedKernelGaussianProcess) and bool(
            (self.gp.jitter_adjustments > 0).any()
        ):
            state["fixed_gp_state"] = self.gp.state_dict()
        return state

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Validate contextual checkpoint identity before any state is restored."""

        del strict
        if state.get("kernel_combination") != self.config.kernel_combination:
            raise CGP_UCBCheckpointError(
                "Stored kernel combination is incompatible.", algorithm=self
            )
        if not isinstance(state.get("state"), Mapping):
            raise CGP_UCBCheckpointError("CGP-UCB state is incomplete.", algorithm=self)

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Restore tensors and histories, then reset the GP for a fresh fit."""

        del strict
        raw = state["state"]
        try:
            dimension = state.get("context_dimension")
            if dimension is not None:
                self._ensure_context_dimension(
                    torch.zeros(dimension, dtype=self.dtype, device=self.device)
                )

            def tensor(value: Any) -> torch.Tensor:
                """Convert one checkpoint value to the algorithm tensor policy."""

                return torch.as_tensor(value, dtype=self.dtype, device=self.device)

            self.state = CGPUCBState(
                actions_train=tensor(raw["actions_train"]),
                contexts_train=tensor(raw["contexts_train"]),
                joint_inputs_train=tensor(raw["joint_inputs_train"]),
                y_train=tensor(raw["y_train"]),
                observation_noise_variance=tensor(raw["observation_noise_variance"]),
                current_context=None
                if raw["current_context"] is None
                else tensor(raw["current_context"]),
                context_history=[tensor(x) for x in raw["context_history"]],
            )
            if self.gp is not None:
                self.gp.reset()
                fixed_state = state.get("fixed_gp_state")
                if fixed_state is not None:
                    if not isinstance(self.gp, FixedKernelGaussianProcess) or not isinstance(
                        fixed_state, dict
                    ):
                        raise ValueError("Unexpected fixed GP numerical state.")
                    self.gp.load_state_dict(fixed_state)
                    if (
                        self.gp.X_train is None
                        or self.gp.y_train is None
                        or self.gp.observation_noise is None
                        or not torch.equal(self.gp.X_train, self.state.joint_inputs_train)
                        or not torch.equal(self.gp.y_train, self.state.y_train)
                        or not torch.equal(
                            self.gp.observation_noise, self.state.observation_noise_variance
                        )
                    ):
                        raise ValueError("Fixed GP numerical state differs from observed history.")
        except Exception as error:
            raise CGP_UCBCheckpointError(
                f"Invalid CGP-UCB checkpoint: {error}", algorithm=self
            ) from error

    def _reset_impl(self) -> None:
        """Clear the frozen context dimension, training history, and GP cache."""

        if self.gp is not None:
            self.gp.reset()
        self.context_dimension = None
        self.model_input_dimension = None
        self.gp = None
        self.state = CGPUCBState(
            actions_train=torch.empty(
                (0, self.action_dimension), dtype=self.dtype, device=self.device
            ),
            contexts_train=torch.empty((0, 0), dtype=self.dtype, device=self.device),
            joint_inputs_train=torch.empty((0, 0), dtype=self.dtype, device=self.device),
            y_train=torch.empty(0, dtype=self.dtype, device=self.device),
            observation_noise_variance=torch.empty(0, dtype=self.dtype, device=self.device),
            current_context=None,
            context_history=[],
        )


__all__ = [
    "CGP_UCB",
    "AcquisitionOptimizerSpec",
    "CGPUCBConfig",
    "CGPUCBState",
]
