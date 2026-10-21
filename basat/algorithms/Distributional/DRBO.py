"""Distributionally robust Bayesian optimization over an MMD ambiguity set."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, ClassVar, Literal

import gpytorch
import numpy as np
import torch

from basat.models.GaussianProcess import GaussianProcessConfig
from basat.models.Kernels import KernelConfig
from basat.models.utils import Bounds, to_numpy

from ..algorithm import AlgorithmConfigurationError, resolve_batch_configuration
from ..helper import SourceRecord
from .Exception.exception_DRBO import (
    DRBOActionOptimizerError,
    DRBOCheckpointError,
    DRBOContextSupportError,
    DRBODataCompatibilityError,
    DRBOException,
    DRBOInnerSolverError,
    DRBOMarginError,
    DRBOMMDMatrixError,
    DRBOReferenceDistributionError,
)
from .Exception.exception_RoBOS import RoBOSContextSupportError
from .RoBOS import _FiniteContextGPAlgorithm, theoretical_beta


@dataclass(frozen=True, slots=True)
class ActionOptimizerSpec:
    """Outer action candidate and GP fit settings.

    Parameters
    ----------
    raw_samples:
        Sobol actions used to initialize continuous outer optimization.
    restarts:
        Highest-valued starts refined by the outer optimizer.
    maxiter:
        Maximum iterations allowed for each outer action restart.
    fit_maxiter:
        Maximum joint-GP hyperparameter-optimization iterations per refit.
        Zero conditions at fixed configured hyperparameters.
    """

    raw_samples: int = 256
    restarts: int = 10
    maxiter: int = 200
    fit_maxiter: int = 100


@dataclass(frozen=True, slots=True)
class DRBOConfig:
    """Immutable configuration for finite-context DRBO.

    Parameters
    ----------
    context_kernel:
        Kernel for context coordinates and the ``(K, K)`` MMD Gram matrix.
    action_kernel:
        Kernel operating on the ``D_action`` slice of joint GP inputs.
    beta:
        Non-negative scalar or round-indexed callable supplying the GP
        confidence multiplier.
    ambiguity_metric:
        Distribution distance used by the inner ambiguity set; version 1 fixes
        this discriminator to ``"mmd"``.
    inner_solver:
        CVXPY, SLSQP, or enumeration backend for minimizing expected UCB over
        the finite MMD probability ball. ``"source_cvxpy"`` uses fresh, cold
        CLARABEL programs without solver fallback, and NumPy score arithmetic
        matching released finite experiments. Non-full balls are numerically
        solved even when a minimum
        vertex is feasible. The explicit PSD eigen-factor and independent
        residual checks remain enabled; historical dependency bugs are not
        reproduced. Zero-radius and full-simplex limits remain exact.
    action_optimizer:
        Continuous outer action-search and joint-GP fitting settings.
    distribution_grid:
        Optional simplex candidate table with shape ``(G, K)`` used only by
        enumeration. It never represents the hidden true sampling distribution.
    enumeration_resolution:
        Integer-composition resolution used when no enumeration grid is given.
    initial_points:
        Sequential observations collected before robust model-based selection.
    gp_config:
        Optional complete joint surrogate settings. Its kernel takes precedence
        for GP predictions; ``context_kernel`` continues to define MMD.
    gp_backend:
        ``"standard"`` or explicit ``"fixed_gpy"`` full-conditioning
        numerical arithmetic. The latter requires a zero-mean CPU float64
        fixed-noise GP config and zero GP fitting steps.
    gp_training_regularization:
        Fixed training-only diagonal for ``"fixed_gpy"``. This is distinct
        from observation noise and conditional jitter; additions are warned.
    feasibility_tolerance:
        Tolerance for independently recomputed simplex, MMD-radius, and
        action-bound residuals.
    optimizer_tolerance:
        Numerical termination tolerance passed to the selected solvers.
    finite_action_tie_breaking:
        ``"stable"`` groups numerically close scores using the native tolerance.
        ``"first"`` uses the first exact maximum in candidate-table order,
        matching released finite-grid experimental loops using ``argmax``.
    batch_size:
        Requested proposal count. DRBO version 1 requires ``1``.
    seed:
        Seed for algorithm-owned candidate generation and deterministic replay.
    """

    context_kernel: KernelConfig | gpytorch.kernels.Kernel = field(default_factory=KernelConfig)
    action_kernel: KernelConfig | gpytorch.kernels.Kernel = field(default_factory=KernelConfig)
    beta: Callable[[int], float] | float = theoretical_beta
    ambiguity_metric: Literal["mmd"] = "mmd"
    inner_solver: Literal["cvxpy", "source_cvxpy", "slsqp", "enumeration"] = "cvxpy"
    action_optimizer: ActionOptimizerSpec = field(default_factory=ActionOptimizerSpec)
    distribution_grid: torch.Tensor | np.ndarray | None = None
    enumeration_resolution: int = 20
    initial_points: int = 1
    gp_config: GaussianProcessConfig | None = None
    gp_backend: Literal["standard", "fixed_gpy"] = "standard"
    gp_training_regularization: float = 0.0
    feasibility_tolerance: float = 1e-8
    optimizer_tolerance: float = 1e-6
    finite_action_tie_breaking: Literal["stable", "first"] = "stable"
    batch_size: int | None = 1
    q_batch: int | None = None
    seed: int = 1

    @property
    def action_raw_samples(self) -> int:
        """Number of continuous outer action candidates."""

        return self.action_optimizer.raw_samples

    @property
    def fit_maxiter(self) -> int:
        """Maximum GP hyperparameter optimizer iterations."""

        return self.action_optimizer.fit_maxiter


@dataclass(slots=True)
class DRBOState:
    """Mutable joint-GP and ambiguity-program history.

    Attributes
    ----------
    actions_train:
        Evaluated physical actions with shape ``(N, D_action)``.
    contexts_train:
        Realized support contexts with shape ``(N, D_context)``.
    joint_inputs_train:
        Concatenated GP coordinates with shape
        ``(N, D_action + D_context)``.
    y_train:
        Learner-visible observations with shape ``(N,)``.
    observation_noise_variance:
        Known observation-noise variances with shape ``(N,)``; zeros denote
        feedback without a supplied variance.
    reference_history:
        Learner-visible ambiguity centers, each a simplex vector of shape
        ``(K,)``; hidden true sampling distributions are never stored.
    margin_history:
        Public non-negative MMD radii paired with the reference history.
    selected_actions:
        Nominal actions selected by DRBO, each with shape ``(D_action,)``.
    robust_values:
        Worst-case expected UCB values assigned to selected actions.
    worst_distributions:
        Certified inner-solver simplex witnesses, each with shape ``(K,)``.
    solver_residuals:
        Independently recomputed simplex, nonnegativity, MMD-radius, and outer
        bound residuals for each completed robust solve.
    realized_contexts:
        Learner-visible contexts returned after evaluation, each with shape
        ``(D_context,)``.
    """

    actions_train: torch.Tensor
    contexts_train: torch.Tensor
    joint_inputs_train: torch.Tensor
    y_train: torch.Tensor
    observation_noise_variance: torch.Tensor
    reference_history: list[torch.Tensor]
    margin_history: list[float]
    selected_actions: list[torch.Tensor]
    robust_values: list[float]
    worst_distributions: list[torch.Tensor]
    solver_residuals: list[dict[str, float]]
    realized_contexts: list[torch.Tensor]


class DRBO(_FiniteContextGPAlgorithm):
    """Maximize worst-case optimistic value over an MMD probability ball.

    Examples
    --------
    >>> import basat
    >>> bounds = basat.Bounds(lower=[-1.0, 0.0], upper=[1.0, 1.0])
    >>> algorithm = basat.DRBO(
    ...     bounds=bounds,
    ...     config=basat.DRBOConfig(
    ...         inner_solver="enumeration", initial_points=3, seed=7
    ...     ),
    ... )
    """

    ACQUISITION_POLICY = "worst-case optimistic MMD value"
    CHECKPOINT_ADDED_CONFIG_DEFAULTS: ClassVar[Mapping[str, Any]] = {
        **_FiniteContextGPAlgorithm.CHECKPOINT_ADDED_CONFIG_DEFAULTS,
        "gp_config": None,
        "gp_backend": "standard",
        "gp_training_regularization": 0.0,
        "finite_action_tie_breaking": "stable",
    }

    EXCEPTION_CLASS = DRBOException
    native_batching = False
    native_environments = frozenset({"distributional"})
    required_round_info = frozenset({"reference_distribution", "margin"})
    required_feedback = frozenset({"observation", "evaluated_action", "context"})
    required_capabilities = frozenset({"distributional", "finite_context_support"})
    require_tangent_pd = False
    SOURCE = SourceRecord(
        paper_title="Distributionally Robust Bayesian Optimization",
        paper_year=2020,
        paper_url="https://proceedings.mlr.press/v108/kirschner20a.html",
        repository_url="https://github.com/Bilkent-CYBORG/RoBOS",
        repository_status="related",
        notes="The paper governs DRBO; RoBOS experiments are a related regression reference.",
    )

    def __init__(
        self,
        *,
        bounds: Bounds,
        config: DRBOConfig,
        maximize: bool = True,
        dtype: torch.dtype | str = torch.float32,
        device: torch.device | str = "cpu",
        plot: bool = False,
        verbose: bool = False,
        verbose_setting: bool = False,
        verbose_iter_by_iter: bool = False,
    ) -> None:
        """Initialize a finite-context distributionally robust optimizer.

        Parameters
        ----------
        bounds:
            Box constraints for the ``D_action`` learner-controlled coordinates.
        config:
            Immutable MMD-program, GP, and optimizer settings. Context support
            belongs to the environment and is bound before optimization.
        maximize:
            Optimization direction.  Version 1 implements maximization only.
        dtype, device:
            Tensor dtype and device used for all persistent model state.
        plot, verbose, verbose_setting, verbose_iter_by_iter:
            Shared presentation and logging controls.

        Raises
        ------
        AlgorithmConfigurationError
            If the immutable configuration or optimization direction is invalid.
        DRBOContextSupportError
            If the finite context support cannot define the joint GP model.
        """

        self.config = config
        self._validate_config(config)
        if not maximize:
            raise AlgorithmConfigurationError("DRBO version 1 supports maximization only.")
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
        try:
            self._initialize_finite_context_model(
                contexts=None,
                action_kernel=config.action_kernel,
                context_kernel=config.context_kernel,
                joint_kernel_combination="product",
                joint_kernel=None,
            )
        except Exception as error:
            raise DRBOContextSupportError(str(error), algorithm=self) from error
        self.state = self._empty_state()

    @staticmethod
    def _validate_config(config: DRBOConfig) -> None:
        """Validate sequential DRBO, MMD, solver, and optimizer invariants."""

        if not isinstance(config, DRBOConfig):
            raise AlgorithmConfigurationError("config must be DRBOConfig.")
        if config.gp_config is not None and not isinstance(config.gp_config, GaussianProcessConfig):
            raise AlgorithmConfigurationError("gp_config must be GaussianProcessConfig or None.")
        _FiniteContextGPAlgorithm._validate_gp_backend(config)
        _, q_batch = resolve_batch_configuration(
            batch_size=config.batch_size, q_batch=config.q_batch
        )
        if q_batch is not None and q_batch > 1:
            raise AlgorithmConfigurationError(
                "DRBO does not define a joint q-acquisition; use batch_size for repeated "
                "independent decisions under the same ambiguity set."
            )
        if config.ambiguity_metric != "mmd":
            raise AlgorithmConfigurationError(
                "DRBO version 1 supports ambiguity_metric='mmd' only."
            )
        if config.inner_solver not in {"cvxpy", "source_cvxpy", "slsqp", "enumeration"}:
            raise AlgorithmConfigurationError("Unknown inner_solver.")
        if config.distribution_grid is not None and config.inner_solver != "enumeration":
            raise AlgorithmConfigurationError(
                "distribution_grid is accepted only for inner_solver='enumeration'."
            )
        if config.finite_action_tie_breaking not in {"stable", "first"}:
            raise AlgorithmConfigurationError(
                "finite_action_tie_breaking must be 'stable' or 'first'."
            )
        if isinstance(config.beta, str) or (
            not callable(config.beta)
            and (
                isinstance(config.beta, bool)
                or not isinstance(config.beta, Real)
                or not math.isfinite(config.beta)
                or config.beta < 0
            )
        ):
            raise AlgorithmConfigurationError(
                "beta must be callable or a finite non-negative scalar."
            )
        if not isinstance(config.action_optimizer, ActionOptimizerSpec):
            raise AlgorithmConfigurationError("action_optimizer must be ActionOptimizerSpec.")
        for name, value in (
            ("raw_samples", config.action_optimizer.raw_samples),
            ("restarts", config.action_optimizer.restarts),
            ("maxiter", config.action_optimizer.maxiter),
            ("fit_maxiter", config.action_optimizer.fit_maxiter),
            ("enumeration_resolution", config.enumeration_resolution),
            ("initial_points", config.initial_points),
        ):
            minimum = 0 if name == "fit_maxiter" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise AlgorithmConfigurationError(f"{name} must be an integer >= {minimum}.")
        if any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            for value in (config.feasibility_tolerance, config.optimizer_tolerance)
        ):
            raise AlgorithmConfigurationError("Solver tolerances must be finite.")
        if config.feasibility_tolerance <= 0 or config.optimizer_tolerance <= 0:
            raise AlgorithmConfigurationError("Solver tolerances must be positive.")

    def _empty_state(self) -> DRBOState:
        """Create empty history with explicit action, context, and joint coordinates."""

        assert self.model_input_dimension is not None
        return DRBOState(
            actions_train=torch.empty(
                (0, self.action_dimension), dtype=self.dtype, device=self.device
            ),
            contexts_train=torch.empty(
                (0, self.context_dimension), dtype=self.dtype, device=self.device
            ),
            joint_inputs_train=torch.empty(
                (0, self.model_input_dimension), dtype=self.dtype, device=self.device
            ),
            y_train=torch.empty(0, dtype=self.dtype, device=self.device),
            observation_noise_variance=torch.empty(0, dtype=self.dtype, device=self.device),
            reference_history=[],
            margin_history=[],
            selected_actions=[],
            robust_values=[],
            worst_distributions=[],
            solver_residuals=[],
            realized_contexts=[],
        )

    def bind_environment_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Bind public support and preserve DRBO-specific error reporting."""
        try:
            super().bind_environment_metadata(metadata)
        except RoBOSContextSupportError as error:
            raise DRBOContextSupportError(str(error), algorithm=self) from error

    def _validate_context_support(self) -> None:
        """Validate finite contexts and translate failures to the DRBO error type."""

        try:
            super()._validate_context_support()
        except Exception as error:
            raise DRBOContextSupportError(str(error), algorithm=self) from error

    def _context_gram_matrix(self) -> torch.Tensor:
        """Return the ``(K, K)`` positive-semidefinite MMD Gram matrix.

        Unlike RoBOS, DRBO permits a singular matrix: the ambiguity set then
        uses the induced seminorm.  Numerical failures are exposed as a typed
        MMD-matrix error rather than silently regularizing the geometry.
        """

        try:
            return super()._context_gram_matrix()
        except Exception as error:
            raise DRBOMMDMatrixError(str(error), algorithm=self) from error

    def _match_contexts(self, contexts: torch.Tensor) -> None:
        """Match realized ``(N, D_context)`` rows to the configured support."""

        try:
            super()._match_contexts(contexts)
        except Exception as error:
            raise DRBODataCompatibilityError(str(error), algorithm=self) from error

    def _initialize_from_environment_impl(self, data: Any) -> None:
        """Restore learner-visible history through the DRBO error boundary.

        The shared finite-context loader is implemented by the RoBOS base
        class, but a DRBO caller must never receive a sibling algorithm's
        exception type.  This boundary preserves the shared confidentiality
        and joint-coordinate logic while translating every incompatible
        historical-data failure to :class:`DRBODataCompatibilityError`.

        Parameters
        ----------
        data:
            Learner-visible historical actions, observations, evaluated
            actions, realized contexts, and optional known noise variances.

        Raises
        ------
        DRBODataCompatibilityError
            If non-empty history lacks realized contexts or cannot be loaded
            into the configured finite action-context model.
        """

        try:
            super()._initialize_from_environment_impl(data)
        except DRBODataCompatibilityError:
            raise
        except Exception as error:
            raise DRBODataCompatibilityError(str(error), algorithm=self) from error

    def _validate_reference(self, value: Any) -> torch.Tensor:
        """Return a validated learner-visible reference vector of shape ``(K,)``."""

        try:
            return super()._validate_reference(value)
        except Exception as error:
            raise DRBOReferenceDistributionError(str(error), algorithm=self) from error

    def _beta(self) -> float:
        """Resolve DRBO's confidence coefficient through its own error boundary.

        Returns
        -------
        float
            Finite non-negative coefficient for the current one-based round.

        Raises
        ------
        DRBOInnerSolverError
            If a configured callable or scalar produces an invalid value.
        """

        try:
            return super()._beta()
        except DRBOInnerSolverError:
            raise
        except Exception as error:
            raise DRBOInnerSolverError(str(error), algorithm=self) from error

    def _simplex_grid(self) -> torch.Tensor:
        """Build an enumeration grid through the DRBO solver error boundary.

        Returns
        -------
        torch.Tensor
            Validated probability rows shaped ``(G, K)``.

        Raises
        ------
        DRBOInnerSolverError
            If a supplied grid is nonnumeric or violates simplex constraints,
            or if automatic enumeration exceeds its safe row cap.
        """

        try:
            return super()._simplex_grid()
        except DRBOInnerSolverError:
            raise
        except Exception as error:
            raise DRBOInnerSolverError(str(error), algorithm=self) from error

    def _validate_reference_and_margin(
        self, reference: Any, margin: Any
    ) -> tuple[torch.Tensor, float]:
        """Validate the public ambiguity-set center and non-negative MMD radius.

        The reference is a probability vector over the configured ``K``
        contexts.  The margin is a finite scalar radius; neither value reveals
        the environment's hidden true sampling distribution.
        """

        distribution = self._validate_reference(reference)
        try:
            radius_tensor = torch.as_tensor(margin, dtype=self.dtype, device=self.device).squeeze()
        except (TypeError, ValueError, RuntimeError) as error:
            raise DRBOMarginError("margin must be numeric.", algorithm=self) from error
        if radius_tensor.ndim != 0 or not torch.isfinite(radius_tensor) or radius_tensor < 0:
            raise DRBOMarginError("margin must be a finite non-negative scalar.", algorithm=self)
        return distribution, float(radius_tensor.item())

    def _fit_model(self) -> None:
        """Fit the joint action-context GP and map fitting failures to DRBO."""

        try:
            super()._fit_model()
        except Exception as error:
            raise DRBOInnerSolverError(str(error), algorithm=self) from error

    @staticmethod
    def _mmd_squared(
        distribution: torch.Tensor, reference: torch.Tensor, gram: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate squared MMD for ``(..., K)`` distributions and a ``(K, K)`` Gram matrix."""

        difference = distribution - reference
        return torch.einsum("...i,ij,...j->...", difference, gram, difference).clamp_min(0.0)

    def _residuals(
        self,
        distribution: torch.Tensor,
        reference: torch.Tensor,
        margin: float,
        gram: torch.Tensor,
    ) -> dict[str, float]:
        """Independently recompute simplex and MMD-ball feasibility residuals.

        The returned residuals measure unit-sum error, negative probability
        mass, and excess squared MMD radius.  They are computed from the solver
        output in Basat's tensor representation, not trusted solver metadata.
        """

        squared = float(self._mmd_squared(distribution, reference, gram).item())
        return {
            "simplex_sum": abs(float(distribution.sum().item()) - 1.0),
            "nonnegativity": max(0.0, -float(distribution.min().item())),
            "mmd_radius": max(0.0, squared - margin**2),
            "mmd_squared": squared,
        }

    def _check_solution(
        self,
        distribution: torch.Tensor,
        reference: torch.Tensor,
        margin: float,
        gram: torch.Tensor,
    ) -> dict[str, float]:
        """Certify a worst distribution against the configured feasibility tolerance.

        Raises
        ------
        DRBOInnerSolverError
            If any independently recomputed simplex or MMD residual exceeds
            ``feasibility_tolerance``.
        """

        residuals = self._residuals(distribution, reference, margin, gram)
        tolerance = self.config.feasibility_tolerance
        if any(
            residuals[name] > tolerance for name in ("simplex_sum", "nonnegativity", "mmd_radius")
        ):
            raise DRBOInnerSolverError(
                f"Worst-distribution residual check failed: {residuals}.", algorithm=self
            )
        return residuals

    def _solve_cvxpy(
        self, ucb: torch.Tensor, reference: torch.Tensor, margin: float, gram: torch.Tensor
    ) -> torch.Tensor:
        """Solve the finite MMD-constrained linear program through CVXPY.

        ``ucb`` and ``reference`` have shape ``(K,)`` and ``gram`` has shape
        ``(K, K)``.  The returned probability vector is independently certified
        by :meth:`_worst_case_expected_ucb`; solver status alone is insufficient.

        Raises
        ------
        DRBOInnerSolverError
            If the optional robust dependency is unavailable or no CVXPY solver
            returns an admissible solution status.
        """

        try:
            import cvxpy as cp
        except (
            ImportError
        ) as error:  # pragma: no cover - optional dependency availability is host-specific
            raise DRBOInnerSolverError(
                "CVXPY solver requires the 'robust' extra: pip install basat[robust].",
                algorithm=self,
            ) from error
        except Exception as error:
            raise DRBOInnerSolverError(
                f"The CVXPY backend could not be initialized: {error}", algorithm=self
            ) from error
        objective = to_numpy(ucb.double())
        center = to_numpy(reference.double())
        matrix = to_numpy(gram.double())
        key = (center.tobytes(), matrix.tobytes(), margin)
        source_protocol = self.config.inner_solver == "source_cvxpy"
        cached = None if source_protocol else getattr(self, "_cvxpy_problem_cache", None)
        if cached is None or cached[0] != key:
            variable = cp.Variable(center.size)
            coefficients = cp.Parameter(center.size)
            constraints = (
                [cp.sum(variable) == 1, variable >= 0]
                if source_protocol
                else [variable >= 0, cp.sum(variable) == 1]
            )
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            factor = np.sqrt(np.clip(eigenvalues, 0.0, None))[:, None] * eigenvectors.T
            constraints.append(cp.norm(factor @ (variable - center), 2) <= margin)
            problem = cp.Problem(cp.Minimize(coefficients @ variable), constraints)
            self._cvxpy_problem_cache = (key, variable, coefficients, problem)
        else:
            _, variable, coefficients, problem = cached
        coefficients.value = objective
        solver_accuracy = min(
            self.config.optimizer_tolerance,
            self.config.feasibility_tolerance / 10.0,
        )
        try:
            problem.solve(
                solver="CLARABEL",
                tol_gap_abs=solver_accuracy,
                tol_feas=solver_accuracy,
                warm_start=not source_protocol,
            )
        except Exception as clarabel_error:
            if source_protocol:
                # A reproduction must not silently switch numerical solvers.
                # The native mode below retains its ordinary SCS fallback.
                raise DRBOInnerSolverError(
                    "Source CVXPY protocol requires a cold CLARABEL solve; "
                    f"CLARABEL reported {clarabel_error}.",
                    algorithm=self,
                ) from clarabel_error
            # SCS remains an explicitly CVXPY-managed solver, not a change in
            # the selected mathematical formulation.
            try:
                problem.solve(
                    solver="SCS",
                    eps=solver_accuracy,
                    max_iters=100_000,
                    warm_start=not source_protocol,
                )
            except Exception as scs_error:
                raise DRBOInnerSolverError(
                    "Both CVXPY solvers failed: "
                    f"CLARABEL reported {clarabel_error}; SCS reported {scs_error}.",
                    algorithm=self,
                ) from scs_error
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or variable.value is None:
            raise DRBOInnerSolverError(f"CVXPY returned status {problem.status!r}.", algorithm=self)
        return torch.as_tensor(variable.value, dtype=self.dtype, device=self.device)

    def _solve_slsqp(
        self, ucb: torch.Tensor, reference: torch.Tensor, margin: float, gram: torch.Tensor
    ) -> torch.Tensor:
        """Solve the MMD ambiguity program by deterministic certified multistart.

        Starts comprise the reference center, simplex barycenter, and vertices.
        Every successful SLSQP result is rechecked for simplex and MMD residuals;
        raw starts are never retained as an unchecked optimizer fallback.

        Raises
        ------
        DRBOInnerSolverError
            If no restart returns a finite solution within
            ``feasibility_tolerance``.
        """

        try:
            from scipy.optimize import minimize
        except ImportError as error:  # pragma: no cover - SciPy is a declared core dependency
            raise DRBOInnerSolverError("SLSQP requires scipy.", algorithm=self) from error
        objective = to_numpy(ucb.double())
        center = to_numpy(reference.double())
        matrix = to_numpy(gram.double())
        starts = [center, np.full_like(center, 1.0 / center.size)]
        starts.extend(np.eye(center.size))
        best_distribution: torch.Tensor | None = None
        best_value = math.inf
        for start in starts:
            difference = start - center
            if (
                float(difference @ matrix @ difference)
                > margin**2 + self.config.feasibility_tolerance
            ):
                continue
            try:
                result = minimize(
                    lambda w: float(w @ objective),
                    start,
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * center.size,
                    constraints=[
                        {"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},
                        {
                            "type": "ineq",
                            "fun": lambda w: float(
                                margin**2 - (w - center) @ matrix @ (w - center)
                            ),
                        },
                    ],
                    options={
                        "ftol": min(
                            self.config.optimizer_tolerance,
                            self.config.feasibility_tolerance / 10.0,
                        ),
                        "maxiter": 1000,
                    },
                )
            except (FloatingPointError, RuntimeError, ValueError):
                continue
            try:
                result_x = np.asarray(result.x, dtype=float)
                result_fun = float(result.fun)
            except (TypeError, ValueError):
                continue
            if (
                not result.success
                or result_x.shape != center.shape
                or not math.isfinite(result_fun)
                or not np.isfinite(result_x).all()
            ):
                continue
            candidate = torch.as_tensor(result_x, dtype=self.dtype, device=self.device)
            residuals = self._residuals(candidate, reference, margin, gram)
            if any(
                residuals[name] > self.config.feasibility_tolerance
                for name in ("simplex_sum", "nonnegativity", "mmd_radius")
            ):
                continue
            candidate_value = float((candidate @ ucb).item())
            if not math.isfinite(candidate_value):
                continue
            if candidate_value < best_value:
                best_distribution = candidate.detach().clone()
                best_value = candidate_value
            elif candidate_value == best_value and best_distribution is not None:
                if tuple(to_numpy(candidate)) < tuple(to_numpy(best_distribution)):
                    best_distribution = candidate.detach().clone()
        if best_distribution is None:
            raise DRBOInnerSolverError("No SLSQP ambiguity solution converged.", algorithm=self)
        return best_distribution

    def _solve_enumeration(
        self, ucb: torch.Tensor, reference: torch.Tensor, margin: float, gram: torch.Tensor
    ) -> torch.Tensor:
        """Minimize expected UCB over the feasible configured simplex grid.

        The exact reference center is added when a coarse grid contains no
        feasible row, because that center is mathematically feasible for every
        non-negative margin.
        """

        try:
            grid = self._simplex_grid()
        except Exception as error:
            raise DRBOInnerSolverError(str(error), algorithm=self) from error
        feasible = (
            self._mmd_squared(grid, reference, gram)
            <= margin**2 + self.config.feasibility_tolerance
        )
        if not feasible.any():
            # The exact center is always feasible even when it is absent from a
            # coarse user grid.
            grid = torch.cat((reference.unsqueeze(0), grid), dim=0)
            feasible = (
                self._mmd_squared(grid, reference, gram)
                <= margin**2 + self.config.feasibility_tolerance
            )
        candidates = grid[feasible]
        return candidates[int(torch.argmin(candidates @ ucb).item())]

    def _worst_case_expected_ucb(
        self,
        ucb: torch.Tensor,
        reference: torch.Tensor,
        margin: float,
        gram: torch.Tensor | None = None,
    ) -> tuple[float, torch.Tensor, dict[str, float]]:
        """Compute and certify the worst expected UCB in an MMD ambiguity ball.

        Parameters
        ----------
        ucb:
            Optimistic values at all ``K`` context-support points.
        reference:
            Public simplex center with shape ``(K,)``.
        margin:
            Non-negative MMD radius.
        gram:
            Optional precomputed context Gram matrix of shape ``(K, K)``.

        Returns
        -------
        value, distribution, residuals:
            Worst expected UCB, its probability vector, and independently
            recomputed simplex/MMD feasibility certificate.

        Raises
        ------
        DRBOInnerSolverError
            If the selected solver fails certification or the objective is not
            finite.
        """

        matrix = self._context_gram_matrix() if gram is None else gram
        source_protocol = self.config.inner_solver == "source_cvxpy"
        vertex = torch.zeros_like(reference)
        vertex[int(torch.argmin(ucb))] = 1.0
        if margin == 0:
            worst = reference.detach().clone()
        elif (
            self.config.inner_solver != "enumeration"
            and float(self._mmd_squared(vertex, reference, matrix)) <= margin**2
            and (
                not source_protocol
                or bool(
                    (
                        self._mmd_squared(
                            torch.eye(len(reference), dtype=self.dtype, device=self.device),
                            reference,
                            matrix,
                        )
                        <= margin**2
                    ).all()
                )
            )
        ):
            # A feasible minimizer over the entire simplex is also a global
            # minimizer over this ambiguity ball. This is WRBO exactly when
            # the radius covers the simplex, with no numerical solve needed.
            worst = vertex
        elif self.config.inner_solver in {"cvxpy", "source_cvxpy"}:
            worst = self._solve_cvxpy(ucb, reference, margin, matrix)
        elif self.config.inner_solver == "slsqp":
            worst = self._solve_slsqp(ucb, reference, margin, matrix)
        else:
            worst = self._solve_enumeration(ucb, reference, margin, matrix)
        residuals = self._check_solution(worst, reference, margin, matrix)
        value = (
            float(np.dot(to_numpy(worst).reshape(1, -1), to_numpy(ucb).reshape(-1, 1)).item())
            if source_protocol
            else float((worst @ ucb).item())
        )
        if not math.isfinite(value):
            raise DRBOInnerSolverError("Worst-case expected UCB is non-finite.", algorithm=self)
        return value, worst.detach().clone(), residuals

    def _select_robust_action(
        self,
        actions: torch.Tensor,
        reference: torch.Tensor,
        margin: float,
        indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, float, torch.Tensor, dict[str, float]]:
        """Select the best exact action candidate after certified inner solves.

        ``actions`` has shape ``(N, D_action)``.  Finite-data ``indices`` retain
        dataset ordering for exact, stable tie resolution; no continuous outer
        refinement is applied to those candidates.
        """

        gram = self._context_gram_matrix()
        values: list[float] = []
        worst: list[torch.Tensor] = []
        residuals: list[dict[str, float]] = []
        batched_upper = (
            self._ucb_matrix(actions)
            if self.config.gp_backend == "fixed_gpy" or self.config.inner_solver == "source_cvxpy"
            else None
        )
        for index, action in enumerate(actions):
            value, distribution, certificate = self._worst_case_expected_ucb(
                self._ucb_vector(action) if batched_upper is None else batched_upper[:, index],
                reference,
                margin,
                gram,
            )
            values.append(value)
            worst.append(distribution)
            residuals.append(certificate)
        if margin == 0 and self.config.inner_solver == "source_cvxpy":
            assert batched_upper is not None
            values = (
                np.dot(to_numpy(reference).reshape(1, -1), to_numpy(batched_upper))
                .reshape(-1)
                .tolist()
            )
        scores = torch.tensor(values, dtype=self.dtype, device=self.device)
        position = (
            int(torch.argmax(scores))
            if indices is not None and self.config.finite_action_tie_breaking == "first"
            else self._stable_position(scores, scores, actions, indices, minimize=False)
        )
        certificate = dict(residuals[position])
        certificate["outer_bounds"] = self._action_bound_residual(actions[position])
        return actions[position], values[position], worst[position], certificate

    def _action_bound_residual(self, action: torch.Tensor) -> float:
        """Return the independently recomputed box-feasibility violation."""

        lower, upper = self._bounds_on_algorithm_device()
        violation = torch.maximum(lower - action, action - upper).clamp_min(0.0)
        return float(violation.max().item())

    def _refine_continuous_action(
        self,
        raw_actions: torch.Tensor,
        reference: torch.Tensor,
        margin: float,
    ) -> tuple[torch.Tensor, float, torch.Tensor, dict[str, float]]:
        """Refine robust-UCB starts using a certified black-box outer solve.

        The inner MMD program is rerun after each successful local solve. A
        solver endpoint must improve on its start beyond the configured
        numerical tolerance; otherwise that start is independently recertified.
        This prevents derivative-free line searches from moving to a box corner
        on a flat acquisition. If every restart fails, no unchecked screening
        point is returned as a fallback.
        """

        try:
            from scipy.optimize import minimize
        except ImportError as error:  # pragma: no cover - SciPy is a core dependency
            raise DRBOActionOptimizerError(
                "Continuous DRBO action refinement requires scipy.", algorithm=self
            ) from error

        gram = self._context_gram_matrix()
        raw_values: list[float] = []
        for action in raw_actions:
            value, _, _ = self._worst_case_expected_ucb(
                self._ucb_vector(action), reference, margin, gram
            )
            raw_values.append(value)

        lower, upper = self._bounds_on_algorithm_device()
        normalized = (raw_actions - lower) / (upper - lower)
        order = sorted(
            range(raw_actions.shape[0]),
            key=lambda position: (
                -raw_values[position],
                tuple(float(x) for x in normalized[position].tolist()),
            ),
        )
        scipy_bounds = list(zip(to_numpy(lower), to_numpy(upper), strict=True))
        accepted_actions: list[torch.Tensor] = []
        accepted_values: list[float] = []
        accepted_worst: list[torch.Tensor] = []
        accepted_residuals: list[dict[str, float]] = []

        def objective(candidate: np.ndarray) -> float:
            """Return negative certified robust UCB for one SciPy action row."""

            point = torch.as_tensor(candidate, dtype=self.dtype, device=self.device)
            try:
                value, _, _ = self._worst_case_expected_ucb(
                    self._ucb_vector(point), reference, margin, gram
                )
            except DRBOInnerSolverError:
                return 1e12
            return -value if math.isfinite(value) else 1e12

        restart_count = min(self.config.action_optimizer.restarts, len(order))
        for position in order[:restart_count]:
            start = to_numpy(raw_actions[position].double())
            try:
                result = minimize(
                    objective,
                    start,
                    method="Powell",
                    bounds=scipy_bounds,
                    options={
                        "xtol": self.config.optimizer_tolerance,
                        "ftol": self.config.optimizer_tolerance,
                        "maxiter": self.config.action_optimizer.maxiter,
                    },
                )
            except (FloatingPointError, RuntimeError, ValueError):
                continue
            try:
                result_x = np.asarray(result.x, dtype=float)
                result_fun = float(result.fun)
            except (TypeError, ValueError):
                continue
            if (
                not result.success
                or result_x.shape != (self.action_dimension,)
                or not math.isfinite(result_fun)
                or not np.isfinite(result_x).all()
            ):
                continue
            action = torch.as_tensor(result_x, dtype=self.dtype, device=self.device)
            bound_residual = self._action_bound_residual(action)
            if bound_residual > self.config.feasibility_tolerance:
                continue
            try:
                value, worst, certificate = self._worst_case_expected_ucb(
                    self._ucb_vector(action), reference, margin, gram
                )
            except DRBOInnerSolverError:
                continue
            if not math.isfinite(value):
                continue
            improvement_tolerance = self.config.optimizer_tolerance * (
                1.0 + abs(raw_values[position])
            )
            if value <= raw_values[position] + improvement_tolerance:
                # Powell can report success after drifting across a plateau or
                # returning a slightly worse point. A successful local solve
                # cannot justify discarding its superior certified start.
                action = raw_actions[position].detach().clone()
                bound_residual = self._action_bound_residual(action)
                if bound_residual > self.config.feasibility_tolerance:
                    continue
                try:
                    value, worst, certificate = self._worst_case_expected_ucb(
                        self._ucb_vector(action), reference, margin, gram
                    )
                except DRBOInnerSolverError:
                    continue
                if not math.isfinite(value):
                    continue
            certificate = dict(certificate)
            certificate["outer_bounds"] = bound_residual
            accepted_actions.append(action.detach().clone())
            accepted_values.append(value)
            accepted_worst.append(worst)
            accepted_residuals.append(certificate)

        if not accepted_actions:
            raise DRBOActionOptimizerError(
                "Every continuous DRBO action-optimizer restart failed certification.",
                algorithm=self,
            )
        actions = torch.stack(accepted_actions)
        values = torch.tensor(accepted_values, dtype=self.dtype, device=self.device)
        selected = self._stable_position(values, values, actions, None, minimize=False)
        return (
            actions[selected],
            accepted_values[selected],
            accepted_worst[selected],
            accepted_residuals[selected],
        )

    def _ask_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Choose one action from public reference and margin information.

        Function-backed domains use certified continuous outer refinement after
        initialization.  Finite-data domains preserve their exact available
        rows.  The environment's true sampling distribution is neither required
        nor inspected.
        """

        self.bind_environment_metadata(round_info)
        reference, margin = self._validate_reference_and_margin(
            round_info["reference_distribution"], round_info["margin"]
        )
        actions, indices = self._candidate_actions(round_info)
        residuals: dict[str, float]
        if self.state.y_train.numel() < self.config.initial_points:
            position = 0 if indices is None else int(torch.argmin(indices).item())
            action, robust_value, worst, residuals = actions[position], math.nan, reference, {}
        else:
            self._fit_model()
            if indices is None:
                action, robust_value, worst, residuals = self._refine_continuous_action(
                    actions, reference, margin
                )
            else:
                action, robust_value, worst, residuals = self._select_robust_action(
                    actions, reference, margin, indices
                )
        self._last_acquisition_value = None if math.isnan(robust_value) else robust_value
        self._record_last_ask(
            actions=action,
            round_info=round_info,
            model_inputs=self._joint_input(action, reference @ self.contexts)[0],
            algorithm_metadata={
                "reference": reference,
                "margin": margin,
                "robust_value": robust_value,
                "worst_distribution": worst,
                "solver_residuals": residuals,
            },
        )
        return action

    def _tell_impl(
        self,
        *,
        action: torch.Tensor,
        feedback: Mapping[str, Any],
        round_info: Mapping[str, Any],
    ) -> None:
        """Consume learner-visible observation, evaluated action, and context.

        The realized context is aligned to the configured support and appended
        to the joint GP history.  Hidden true sampling probabilities remain an
        environment audit concern and are never stored in algorithm state.
        """

        del round_info
        assert self.last_ask is not None
        context, _, _ = self._append_joint_feedback(feedback)
        metadata = self.last_ask.algorithm_metadata
        self.state.reference_history.append(
            torch.as_tensor(metadata["reference"], dtype=self.dtype, device=self.device)
        )
        self.state.margin_history.append(float(metadata["margin"]))
        self.state.selected_actions.append(action[0].detach().clone())
        self.state.robust_values.append(float(metadata["robust_value"]))
        self.state.worst_distributions.append(
            torch.as_tensor(metadata["worst_distribution"], dtype=self.dtype, device=self.device)
        )
        self.state.solver_residuals.append(dict(metadata["solver_residuals"]))
        self.state.realized_contexts.append(context.detach().clone())

    def _recommendation_impl(self) -> tuple[torch.Tensor | None, float | None]:
        """Return the action with the largest finite recorded robust value."""

        if not self.state.selected_actions:
            return self._best_observed()
        scores = torch.tensor(self.state.robust_values, dtype=self.dtype, device=self.device)
        finite = torch.isfinite(scores)
        if finite.any():
            indices = torch.nonzero(finite, as_tuple=False).reshape(-1)
            position = int(indices[torch.argmax(scores[indices])].item())
        else:
            return self._best_observed()
        value = float(scores[position].item())
        return self.state.selected_actions[position].detach().clone(), (
            value if self.maximize else -value
        )

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Expose solver choice and the latest public/certified round summary."""

        return {
            "context_count": int(self.contexts.shape[0]),
            "inner_solver": self.config.inner_solver,
            "last_margin": self.state.margin_history[-1] if self.state.margin_history else None,
            "last_robust_value": self.state.robust_values[-1] if self.state.robust_values else None,
            "last_solver_residuals": self.state.solver_residuals[-1]
            if self.state.solver_residuals
            else None,
        }

    def _result_metadata_impl(self) -> Mapping[str, Any]:
        """Return detached public ambiguity history and worst distributions."""

        return {
            "reference_history": [x.detach().clone() for x in self.state.reference_history],
            "margin_history": tuple(self.state.margin_history),
            "worst_distributions": [x.detach().clone() for x in self.state.worst_distributions],
        }

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize finite support and mutable DRBO progress."""

        return {
            "contexts": self.contexts,
            "context_support_is_dynamic": getattr(self, "_dynamic_context_support", False),
            "gp": self.gp.state_dict(),
            "ambiguity_metric": self.config.ambiguity_metric,
            "state": {name: getattr(self.state, name) for name in self.state.__dataclass_fields__},
        }

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Validate checkpoint support, ambiguity metric, and required mappings."""

        del strict
        if not self.contexts.numel() or state.get("context_support_is_dynamic", False):
            self.bind_environment_metadata(
                {
                    "context_support": state.get("contexts"),
                    "context_support_is_dynamic": state.get("context_support_is_dynamic", False),
                }
            )
        if (
            not self._support_matches(state.get("contexts"))
            or state.get("ambiguity_metric") != "mmd"
        ):
            raise DRBOCheckpointError("Stored DRBO configuration is incompatible.", algorithm=self)
        if not isinstance(state.get("state"), Mapping):
            raise DRBOCheckpointError("DRBO state is incomplete.", algorithm=self)

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Restore DRBO progress and reset the GP for a fresh fit."""

        del strict
        raw = state["state"]
        try:

            def tensor(value: Any) -> torch.Tensor:
                """Convert one checkpoint value to algorithm tensor storage."""

                return torch.as_tensor(value, dtype=self.dtype, device=self.device)

            self.state = DRBOState(
                actions_train=tensor(raw["actions_train"]),
                contexts_train=tensor(raw["contexts_train"]),
                joint_inputs_train=tensor(raw["joint_inputs_train"]),
                y_train=tensor(raw["y_train"]),
                observation_noise_variance=tensor(raw["observation_noise_variance"]),
                reference_history=[tensor(x) for x in raw["reference_history"]],
                margin_history=[float(x) for x in raw["margin_history"]],
                selected_actions=[tensor(x) for x in raw["selected_actions"]],
                robust_values=[float(x) for x in raw["robust_values"]],
                worst_distributions=[tensor(x) for x in raw["worst_distributions"]],
                solver_residuals=[dict(x) for x in raw["solver_residuals"]],
                realized_contexts=[tensor(x) for x in raw["realized_contexts"]],
            )
            if isinstance(state.get("gp"), dict):
                self.gp.load_state_dict(dict(state["gp"]))
            else:
                self.gp.reset()
        except Exception as error:
            raise DRBOCheckpointError(
                f"Invalid DRBO checkpoint: {error}", algorithm=self
            ) from error

    def _reset_impl(self) -> None:
        """Clear mutable DRBO history and reset the joint GP wrapper."""

        self.state = self._empty_state()
        self.gp.reset()


__all__ = ["DRBO", "ActionOptimizerSpec", "DRBOConfig", "DRBOState"]
