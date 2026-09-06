"""Adaptive Linear Embedding Bayesian Optimization (ALEBO).

ALEBO uses the exact polytope ``{y: -1 <= B^dagger y <= 1}``; it never
replaces that domain by a box and never clips a substantively infeasible map.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import gpytorch
import torch
import torch.nn.functional as F

from bayopy.algorithms.algorithm import (
    Algorithm,
    AlgorithmConfigurationError,
    resolve_batch_configuration,
)
from bayopy.algorithms.helper import SourceRecord, to_quality
from bayopy.models.Acquisitions import (
    AcquisitionConfig,
    DiscreteAcquisitionCandidates,
    build_acquisition,
    optimize_acquisition_discrete,
)
from bayopy.models.GaussianProcess import GaussianProcess, GaussianProcessConfig
from bayopy.models.utils import Bounds, from_symmetric_box, to_symmetric_box

from .Exception.exception_ALEBO import (
    ALEBOAcquisitionOptimizationError,
    ALEBOCheckpointError,
    ALEBODataCompatibilityError,
    ALEBOException,
    ALEBOHistoricalDataError,
    ALEBOMetricError,
    ALEBOPolytopeError,
)


class ALEBOMahalanobisKernel(gpytorch.kernels.Kernel):
    """Squared-exponential kernel with learned positive-definite metric."""

    _ROUNDOFF_MARGIN = 8.0

    def __init__(self, dimension: int, min_diagonal: float = 1e-6) -> None:
        """Initialize an unconstrained Cholesky parameter for ``D_model``."""

        super().__init__()
        self.raw_lower = torch.nn.Parameter(torch.zeros(dimension, dimension))
        self.min_diagonal = min_diagonal

    def _lower_factor(self, raw_lower: torch.Tensor | None = None) -> torch.Tensor:
        """Return ALEBO's constrained lower-triangular metric factor."""

        raw = self.raw_lower if raw_lower is None else raw_lower
        lower = torch.tril(raw)
        diagonal = F.softplus(torch.diagonal(lower)) + self.min_diagonal
        lower = lower - torch.diag_embed(torch.diagonal(lower)) + torch.diag_embed(diagonal)
        return lower

    def _roundoff_jitter(self, lower: torch.Tensor) -> torch.Tensor:
        """Return a backward-error-scale diagonal stabilizer for ``L L^T``.

        A nonsingular triangular ``L`` makes ``L L^T`` positive definite
        analytically.  Forming the Gram matrix can nevertheless lose its
        smallest eigenvalue when the learned metric is very ill-conditioned.
        The margin below is proportional to the floating-point backward error
        of that matrix product; detaching it keeps this numerical certificate
        out of ALEBO's metric-parameter optimization objective.
        """

        gram_diagonal = lower.square().sum(dim=-1)
        scale = gram_diagonal.amax().clamp_min(1.0).detach()
        dimension = lower.shape[-1]
        return scale * (self._ROUNDOFF_MARGIN * dimension * torch.finfo(lower.dtype).eps)

    def _metric_from_factor(self, lower: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a numerically certifiable metric and its explicit jitter."""

        jitter = self._roundoff_jitter(lower)
        identity = torch.eye(lower.shape[-1], dtype=lower.dtype, device=lower.device)
        gram = lower @ lower.T
        return gram + jitter * identity, jitter

    def metric_matrix(self) -> torch.Tensor:
        """Return the roundoff-stabilized ``Gamma = L L^T + delta I``."""

        metric, _ = self._metric_from_factor(self._lower_factor())
        return metric

    def forward(
        self, y1: torch.Tensor, y2: torch.Tensor, *, diag: bool = False, **params: Any
    ) -> torch.Tensor:
        """Evaluate Mahalanobis squared-exponential covariance.

        Parameters
        ----------
        y1, y2:
            Embedded-coordinate tensors with final dimension ``D_model``.
        diag:
            Return only the covariance diagonal when ``True``.

        Returns
        -------
        torch.Tensor
            Pairwise covariance or its diagonal with batch dimensions preserved.
        """

        del params
        difference = y1.unsqueeze(-2) - y2.unsqueeze(-3)
        lower = self._lower_factor()
        jitter = self._roundoff_jitter(lower)
        transformed = difference @ lower
        squared = transformed.square().sum(dim=-1)
        squared = squared + jitter * difference.square().sum(dim=-1)
        covariance = torch.exp(-0.5 * squared)
        return torch.diagonal(covariance, dim1=-2, dim2=-1) if diag else covariance


@dataclass(frozen=True, slots=True)
class ALEBOConfig:
    """Configuration of the ALEBO polytope, metric, and solver.

    Parameters
    ----------
    embedding_dimension:
        Explicit reduced ``D_model``.
    initial_points, acquisition:
        Initial-design size and EI/UCB acquisition settings.
    metric_inference, metric_posterior_samples:
        Laplace or qualified MAP Mahalanobis-metric approximation controls.
    feasibility_tolerance, acquisition_solver:
        Polytope certificate tolerance and SciPy solver.
    raw_feasible_samples, num_restarts:
        Deterministic feasible-design and multistart budgets.
    batch_size, seed:
        Sequential request size and algorithm-owned seed.
    """

    embedding_dimension: int
    initial_points: int | None = None
    acquisition: AcquisitionConfig = field(default_factory=lambda: AcquisitionConfig(name="EI"))
    metric_inference: Literal["laplace", "map"] = "laplace"
    metric_posterior_samples: int = 25
    feasibility_tolerance: float = 1e-9
    acquisition_solver: Literal["slsqp", "trust-constr"] = "slsqp"
    raw_feasible_samples: int = 2048
    num_restarts: int = 20
    batch_size: int | None = 1
    q_batch: int | None = None
    seed: int = 1


@dataclass(slots=True)
class ALEBOState:
    """Mutable ALEBO geometry, training data, and metric approximation.

    Attributes
    ----------
    B, B_pinv:
        Immutable embedding matrix and its exact Moore-Penrose inverse.
    A_ineq, b_ineq:
        Linear inequalities defining the feasible ``D_model`` polytope.
    Y_train, X_train_symmetric, y_train:
        Aligned reduced, ambient symmetric-box, and response history.
    metric_mode, metric_map_parameters, metric_posterior_samples:
        Active approximation and in-memory fitted Mahalanobis parameters. The
        fitted parameters are rebuilt from observations after checkpoint load.
    last_ask_Y, last_ask_X:
        Pending reduced and physical proposal coordinates.
    """

    B: torch.Tensor
    B_pinv: torch.Tensor
    A_ineq: torch.Tensor
    b_ineq: torch.Tensor
    Y_train: torch.Tensor
    X_train_symmetric: torch.Tensor
    y_train: torch.Tensor
    metric_mode: str
    metric_map_parameters: dict[str, torch.Tensor]
    metric_posterior_samples: list[dict[str, torch.Tensor]]
    last_ask_Y: torch.Tensor | None
    last_ask_X: torch.Tensor | None


class ALEBO(Algorithm):
    """Optimize a Mahalanobis GP over ALEBO's exact linear polytope.

    Examples
    --------
    >>> import bayopy
    >>> bounds = bayopy.Bounds(lower=[-1.0] * 6, upper=[1.0] * 6)
    >>> algorithm = bayopy.ALEBO(
    ...     bounds=bounds,
    ...     config=bayopy.ALEBOConfig(embedding_dimension=2, seed=7),
    ... )
    """

    ACQUISITION_POLICY = "configurable EI or UCB over the ALEBO polytope"

    EXCEPTION_CLASS = ALEBOException
    native_batching = False
    native_environments = frozenset({"stochastic"})
    SOURCE = SourceRecord(
        paper_title="Re-Examining Linear Embeddings for High-Dimensional Bayesian Optimization",
        paper_year=2020,
        paper_url="https://proceedings.neurips.cc/paper/2020/hash/10fb6cfa4c990d2bad5ddef4f70e8ba2-Abstract.html",
        repository_url="https://github.com/facebookresearch/alebo",
        repository_status="official",
        notes="Clean-room exact-polytope implementation; MAP is explicitly qualified.",
    )

    def __init__(
        self,
        *,
        bounds: Bounds,
        config: ALEBOConfig,
        maximize: bool = True,
        dtype: torch.dtype | str = torch.float64,
        device: torch.device | str = "cpu",
        plot: bool = False,
        verbose: bool = False,
        verbose_setting: bool = False,
        verbose_iter_by_iter: bool = False,
    ) -> None:
        """Initialize ALEBO, its exact polytope, and Mahalanobis GP.

        Parameters
        ----------
        bounds:
            Physical action bounds in ``D_env`` coordinates.
        config:
            Embedding, metric-inference, feasibility, and solver settings.
        maximize:
            Optimize larger observations when ``True``.
        dtype, device:
            Tensor precision and execution device.
        plot, verbose, verbose_setting, verbose_iter_by_iter:
            Presentation flags inherited from :class:`Algorithm`.

        Raises
        ------
        AlgorithmConfigurationError
            If dimensions, inference mode, or solver constants are invalid.
        ALEBOPolytopeError
            If a full-row-rank projection cannot be sampled.
        """

        self.config = config
        self._validate_config(config, bounds.dimension)
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
        B = self._sample_projection(config.embedding_dimension, self.action_dimension)
        B_pinv, A_ineq, b_ineq = self._make_polytope(B)
        self.state = ALEBOState(
            B=B,
            B_pinv=B_pinv,
            A_ineq=A_ineq,
            b_ineq=b_ineq,
            Y_train=torch.empty(
                (0, config.embedding_dimension), dtype=self.dtype, device=self.device
            ),
            X_train_symmetric=torch.empty(
                (0, self.action_dimension), dtype=self.dtype, device=self.device
            ),
            y_train=torch.empty((0,), dtype=self.dtype, device=self.device),
            metric_mode=config.metric_inference,
            metric_map_parameters={},
            metric_posterior_samples=[],
            last_ask_Y=None,
            last_ask_X=None,
        )
        self.model_input_dimension = config.embedding_dimension
        self.metric_kernel = ALEBOMahalanobisKernel(config.embedding_dimension).to(
            dtype=self.dtype, device=self.device
        )
        self.gp = GaussianProcess(
            GaussianProcessConfig(
                kernel=gpytorch.kernels.ScaleKernel(self.metric_kernel),
                mean="constant",
                noise_mode="learned",
                fit_method="map" if config.metric_inference == "laplace" else "mle",
                dtype=self.dtype,
                device=self.device,
            ),
            input_dimension=config.embedding_dimension,
        )
        self._last_finite_valid_mask: torch.Tensor | None = None
        self._last_rejected_finite_rows = 0
        if config.metric_inference == "map":
            self.invalidate_source_guarantees("alebo.map_metric_approximation")

    @staticmethod
    def _validate_config(config: ALEBOConfig, D: int) -> None:
        """Validate immutable ALEBO settings against ambient dimension ``D``."""

        if not isinstance(config, ALEBOConfig):
            raise AlgorithmConfigurationError("config must be ALEBOConfig.")
        _, q_batch = resolve_batch_configuration(
            batch_size=config.batch_size, q_batch=config.q_batch
        )
        if q_batch is not None and q_batch > 1:
            raise AlgorithmConfigurationError(
                "ALEBO does not define a joint q-acquisition; use batch_size for repeated "
                "independent embedded-space queries."
            )
        if (
            isinstance(config.embedding_dimension, bool)
            or not isinstance(config.embedding_dimension, int)
            or not 1 <= config.embedding_dimension <= D
        ):
            raise AlgorithmConfigurationError("ALEBO requires 1 <= embedding_dimension <= D.")
        if config.initial_points is not None and (
            isinstance(config.initial_points, bool)
            or not isinstance(config.initial_points, int)
            or config.initial_points < 1
        ):
            raise AlgorithmConfigurationError("initial_points must be positive or None.")
        if not isinstance(config.acquisition, AcquisitionConfig):
            raise AlgorithmConfigurationError("acquisition must be AcquisitionConfig.")
        if config.metric_inference not in {"laplace", "map"}:
            raise AlgorithmConfigurationError("metric_inference must be 'laplace' or 'map'.")
        for name in ("metric_posterior_samples", "raw_feasible_samples", "num_restarts"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise AlgorithmConfigurationError(f"{name} must be a positive integer.")
        if not math.isfinite(config.feasibility_tolerance) or config.feasibility_tolerance <= 0:
            raise AlgorithmConfigurationError("feasibility_tolerance must be positive and finite.")
        if config.acquisition_solver not in {"slsqp", "trust-constr"}:
            raise AlgorithmConfigurationError("Unknown ALEBO acquisition_solver.")

    def _sample_projection(self, d: int, D: int) -> torch.Tensor:
        """Sample unit-sphere columns until B has full row rank."""

        for _ in range(100):
            matrix = torch.randn(
                (d, D), dtype=self.dtype, device=self.device, generator=self.generator
            )
            norms = torch.linalg.vector_norm(matrix, dim=0)
            if bool((norms <= torch.finfo(self.dtype).tiny).any()):
                continue
            B = matrix / norms
            if int(torch.linalg.matrix_rank(B).item()) == d:
                return B
        raise ALEBOPolytopeError("Unable to sample a full-row-rank ALEBO matrix.", algorithm=self)

    def _make_polytope(self, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct B-pseudoinverse and exact ambient box inequalities."""

        B_pinv = torch.linalg.pinv(B, rcond=1e-12)
        A_ineq = torch.cat((B_pinv, -B_pinv), dim=0)
        b_ineq = torch.ones(2 * self.action_dimension, dtype=self.dtype, device=self.device)
        return B_pinv, A_ineq, b_ineq

    def _is_feasible_y(self, Y: torch.Tensor) -> torch.Tensor:
        """Return a row mask for exact-polytope feasibility in ``D_model``."""

        residual = Y @ self.state.A_ineq.T - self.state.b_ineq
        return residual.max(dim=-1).values <= self.config.feasibility_tolerance

    def _map_y_to_action(self, Y: torch.Tensor) -> torch.Tensor:
        """Map feasible y through B-dagger, rejecting substantive violations."""

        if not bool(self._is_feasible_y(Y).all()):
            raise ALEBOPolytopeError("ALEBO candidate left the exact polytope.", algorithm=self)
        X = Y @ self.state.B_pinv.T
        excess = float((X.abs() - 1.0).clamp_min(0).max().item())
        if excess > 10 * self.config.feasibility_tolerance:
            raise ALEBOPolytopeError("ALEBO map violated the ambient box.", algorithm=self)
        return from_symmetric_box(
            X.clamp(-1.0, 1.0), self.bounds.to(dtype=self.dtype, device=self.device)
        )

    def _recover_y_if_exact(
        self, actions: torch.Tensor, *, historical: bool = False
    ) -> torch.Tensor:
        """Use ALEBO's defined y=Bx map and verify B-dagger y equals x."""

        X = to_symmetric_box(actions, self.bounds.to(dtype=self.dtype, device=self.device))
        Y = X @ self.state.B.T
        reconstructed = Y @ self.state.B_pinv.T
        valid = torch.isclose(reconstructed, X, rtol=0.0, atol=1e-8).all(dim=-1)
        valid &= self._is_feasible_y(Y)
        if not bool(valid.all()):
            error = ALEBOHistoricalDataError if historical else ALEBODataCompatibilityError
            raise error("Actions do not lie in ALEBO's exact embedded image.", algorithm=self)
        return Y.detach().clone()

    def _initialize_from_environment_impl(self, data: Any) -> None:
        """Load history only when every action lies in ALEBO's embedded image."""

        actions = torch.as_tensor(data.actions, dtype=self.dtype, device=self.device)
        if actions.shape[0] == 0:
            return
        self.state.Y_train = self._recover_y_if_exact(actions, historical=True)
        self.state.X_train_symmetric = to_symmetric_box(
            actions, self.bounds.to(dtype=self.dtype, device=self.device)
        )
        self.state.y_train = (
            torch.as_tensor(data.observations, dtype=self.dtype, device=self.device)
            .detach()
            .clone()
        )
        self._set_gp_data()

    def _prepare_discrete_candidates_impl(
        self,
        *,
        candidate_actions: torch.Tensor,
        candidate_dataset_indices: torch.Tensor,
        round_info: Mapping[str, Any],
    ) -> DiscreteAcquisitionCandidates:
        """Mark representable finite rows and return their ALEBO coordinates.

        The returned table preserves all source rows so the caller can apply
        the stored validity mask without losing stable dataset indices.
        """

        del round_info
        X = to_symmetric_box(
            candidate_actions, self.bounds.to(dtype=self.dtype, device=self.device)
        )
        Y = X @ self.state.B.T
        reconstructed = Y @ self.state.B_pinv.T
        valid = torch.isclose(reconstructed, X, rtol=0.0, atol=1e-8).all(
            dim=-1
        ) & self._is_feasible_y(Y)
        if not bool(valid.any()):
            raise ALEBODataCompatibilityError(
                "No finite candidate lies in ALEBO's exact embedded image.",
                algorithm=self,
            )
        self._last_finite_valid_mask = valid.detach().clone()
        self._last_rejected_finite_rows = int((~valid).sum().item())
        return DiscreteAcquisitionCandidates(
            environment_actions=candidate_actions.detach().clone(),
            model_inputs=Y,
            dataset_indices=candidate_dataset_indices.detach().clone(),
        )

    def _coordinate_bounds(self) -> Bounds:
        """Compute exact finite coordinate extrema with linear programs."""

        try:
            import numpy as np
            from scipy.optimize import linprog
        except ImportError as error:  # scipy is a core dependency
            raise ALEBOPolytopeError(
                "SciPy is required for ALEBO polytope bounds.", algorithm=self
            ) from error
        A = self.state.A_ineq.detach().cpu().numpy()
        b = self.state.b_ineq.detach().cpu().numpy()
        lower: list[float] = []
        upper: list[float] = []
        for coordinate in range(self.config.embedding_dimension):
            objective = np.zeros(self.config.embedding_dimension)
            objective[coordinate] = 1.0
            minimum = linprog(objective, A_ub=A, b_ub=b, bounds=(None, None), method="highs")
            maximum = linprog(-objective, A_ub=A, b_ub=b, bounds=(None, None), method="highs")
            if not minimum.success or not maximum.success:
                raise ALEBOPolytopeError("ALEBO coordinate linear program failed.", algorithm=self)
            lower.append(float(minimum.fun))
            upper.append(float(-maximum.fun))
        return Bounds(
            torch.tensor(lower, dtype=self.dtype, device=self.device),
            torch.tensor(upper, dtype=self.dtype, device=self.device),
        )

    def _sample_feasible_Y(
        self,
        count: int,
        *,
        phase: Literal["initial", "acquisition"] = "initial",
    ) -> torch.Tensor:
        """Sample the polytope by rejection, then hit-and-run if necessary."""

        coordinate_bounds = self._coordinate_bounds()
        raw = self._sample_design(
            coordinate_bounds,
            max(count, self.config.raw_feasible_samples),
            phase=phase,
            seed=self.config.seed + self.iteration,
        )
        feasible = raw[self._is_feasible_y(raw)]
        if feasible.shape[0] >= count:
            return feasible[:count]
        current = torch.zeros(self.config.embedding_dimension, dtype=self.dtype, device=self.device)
        samples: list[torch.Tensor] = [row for row in feasible]
        burn = 10 * self.config.embedding_dimension
        steps = 0
        while len(samples) < count and steps < burn + count * 50:
            direction = torch.randn(
                self.config.embedding_dimension,
                dtype=self.dtype,
                device=self.device,
                generator=self.generator,
            )
            direction /= direction.norm().clamp_min(torch.finfo(self.dtype).eps)
            denominator = self.state.A_ineq @ direction
            slack = self.state.b_ineq - self.state.A_ineq @ current
            positive = denominator > 1e-14
            negative = denominator < -1e-14
            high = (
                torch.min(slack[positive] / denominator[positive])
                if bool(positive.any())
                else torch.tensor(float("inf"), device=self.device)
            )
            low = (
                torch.max(slack[negative] / denominator[negative])
                if bool(negative.any())
                else torch.tensor(float("-inf"), device=self.device)
            )
            if torch.isfinite(low) and torch.isfinite(high) and low <= high:
                fraction = torch.rand(
                    (), dtype=self.dtype, device=self.device, generator=self.generator
                )
                current = current + (low + fraction * (high - low)) * direction
                if steps >= burn and self._is_feasible_y(current.unsqueeze(0))[0]:
                    samples.append(current.detach().clone())
            steps += 1
        if len(samples) < count:
            raise ALEBOPolytopeError(
                "ALEBO feasible sampler could not produce enough points.", algorithm=self
            )
        return torch.stack(samples[:count])

    def _set_gp_data(self) -> None:
        """Synchronize non-empty ALEBO training tensors with the GP wrapper."""

        if self.state.y_train.numel():
            self.gp.set_data(
                self.state.Y_train,
                to_quality(self.state.y_train, maximize=self.maximize),
            )

    def _fit_model(self) -> None:
        """Fit the GP and record MAP or local-Laplace metric state.

        Raises
        ------
        ALEBOMetricError
            If fitting fails or the resulting metric is not positive definite.
        """

        self._set_gp_data()
        try:
            self.gp.fit(maxiter=75)
        except Exception as error:
            raise ALEBOMetricError(
                f"ALEBO metric fitting failed: {error}", algorithm=self
            ) from error
        base = self.gp.model.covar_module.base_kernel  # type: ignore[union-attr]
        lower = base._lower_factor()
        metric, roundoff_jitter = base._metric_from_factor(lower)
        metric = metric.detach().clone()
        self.state.metric_map_parameters = {
            "raw_lower": base.raw_lower.detach().clone(),
            "Gamma": metric,
            "roundoff_jitter": roundoff_jitter.detach().clone(),
        }
        _, certificate = torch.linalg.cholesky_ex(metric)
        if not bool(torch.isfinite(metric).all()) or int(certificate.item()) != 0:
            raise ALEBOMetricError(
                "ALEBO metric failed its finite Cholesky positive-definiteness certificate.",
                algorithm=self,
            )
        if self.config.metric_inference == "laplace":
            # A stabilized local Gaussian approximation in unconstrained metric
            # parameters.  The fitted MAP remains the acquisition center.
            raw = base.raw_lower.detach()
            self.state.metric_posterior_samples = []
            for _ in range(self.config.metric_posterior_samples):
                perturbation = 1e-3 * torch.randn(
                    raw.shape,
                    dtype=self.dtype,
                    device=self.device,
                    generator=self.generator,
                )
                sample_raw = raw + torch.tril(perturbation)
                lower = base._lower_factor(sample_raw)
                sample_metric, sample_jitter = base._metric_from_factor(lower)
                self.state.metric_posterior_samples.append(
                    {
                        "raw_lower": sample_raw,
                        "Gamma": sample_metric,
                        "roundoff_jitter": sample_jitter,
                    }
                )

    def _optimize_constrained_acquisition(self, acquisition: Any) -> tuple[torch.Tensor, float]:
        """Run configured SciPy local solves and independently check residuals."""

        try:
            import numpy as np
            from scipy.optimize import LinearConstraint, minimize
        except ImportError as error:
            raise ALEBOAcquisitionOptimizationError(
                "SciPy is required by ALEBO.", algorithm=self
            ) from error
        starts = self._sample_feasible_Y(self.config.num_restarts, phase="acquisition")
        A = self.state.A_ineq.detach().cpu().numpy()
        b = self.state.b_ineq.detach().cpu().numpy()
        constraint = LinearConstraint(A, -np.inf, b)

        def objective(values: Any) -> float:
            """Return the negated acquisition value for SciPy minimization."""

            Y = torch.as_tensor(values, dtype=self.dtype, device=self.device).reshape(1, 1, -1)
            with torch.no_grad():
                return -float(acquisition(Y).item())

        solutions: list[tuple[torch.Tensor, float]] = []
        method = "SLSQP" if self.config.acquisition_solver == "slsqp" else "trust-constr"
        for start in starts:
            result = minimize(
                objective,
                start.detach().cpu().numpy(),
                method=method,
                constraints=[constraint],
                options={"maxiter": 200, "ftol": 1e-6} if method == "SLSQP" else {"maxiter": 200},
            )
            candidate = torch.as_tensor(result.x, dtype=self.dtype, device=self.device).reshape(
                1, -1
            )
            residual = float((candidate @ self.state.A_ineq.T - self.state.b_ineq).max().item())
            if result.success and residual <= self.config.feasibility_tolerance:
                solutions.append((candidate, -objective(result.x)))
        if not solutions:
            raise ALEBOAcquisitionOptimizationError(
                "Every ALEBO constrained acquisition solve failed feasibility checks.",
                algorithm=self,
            )
        solutions.sort(key=lambda item: item[1], reverse=True)
        return solutions[0]

    def _initial_design_size(self) -> int:
        """Return the configured or dimension-derived initial-design size."""

        return self.config.initial_points or max(2 * self.config.embedding_dimension, 4)

    def _ask_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Select the next action over the exact polytope or finite image rows."""

        table: DiscreteAcquisitionCandidates | None = None
        if "candidate_actions" in round_info:
            table = self.prepare_discrete_candidates(
                candidate_actions=round_info["candidate_actions"],
                candidate_dataset_indices=round_info["candidate_dataset_indices"],
                round_info=round_info,
            )
            if self._last_finite_valid_mask is None:
                raise ALEBODataCompatibilityError(
                    "Finite candidate validation did not produce an ALEBO membership mask.",
                    algorithm=self,
                )
            valid = self._last_finite_valid_mask
            table = DiscreteAcquisitionCandidates(
                environment_actions=table.environment_actions[valid],
                model_inputs=table.model_inputs[valid],
                dataset_indices=table.dataset_indices[valid],
            )
        if self.state.y_train.numel() < self._initial_design_size():
            if table is None:
                Y = self._sample_feasible_Y(1, phase="initial")
                action = self._map_y_to_action(Y)
            else:
                position = int(torch.argmin(table.dataset_indices).item())
                Y = table.model_inputs[position : position + 1]
                action = table.environment_actions[position : position + 1]
            value = None
        else:
            self._fit_model()
            quality = self.state.y_train if self.maximize else -self.state.y_train
            acquisition = build_acquisition(
                model=self.gp,
                config=self.config.acquisition,
                q=1,
                best_f=(
                    quality.max() if self.config.acquisition.name.lower().startswith("e") else None
                ),
            )
            if table is None:
                Y, value = self._optimize_constrained_acquisition(acquisition)
                action = self._map_y_to_action(Y)
            else:
                result = optimize_acquisition_discrete(
                    acquisition=acquisition, candidates=table, q=1
                )
                Y = result.model_inputs
                action = result.environment_actions
                value = float(result.acquisition_values.item())
            self._last_acquisition_value = value
        X = to_symmetric_box(action, self.bounds.to(dtype=self.dtype, device=self.device))
        self.state.last_ask_Y = Y.detach().clone()
        self.state.last_ask_X = X.detach().clone()
        residual = float((Y @ self.state.A_ineq.T - self.state.b_ineq).max().item())
        self._record_last_ask(
            actions=action,
            model_inputs=Y,
            round_info=round_info,
            algorithm_metadata={"constraint_residual": residual, "acquisition_value": value},
        )
        return action

    def _tell_impl(
        self, *, action: torch.Tensor, feedback: Mapping[str, Any], round_info: Mapping[str, Any]
    ) -> None:
        """Append feedback using the exact embedded coordinates retained by ``ask``."""

        del round_info
        if self.last_ask is None or self.last_ask.model_inputs is None:
            raise ALEBODataCompatibilityError("ALEBO tell is missing ask state.", algorithm=self)
        pending_Y = self.last_ask.model_inputs
        pending_X = to_symmetric_box(action, self.bounds.to(dtype=self.dtype, device=self.device))
        self.state.Y_train = torch.cat((self.state.Y_train, pending_Y), dim=0)
        self.state.X_train_symmetric = torch.cat((self.state.X_train_symmetric, pending_X), dim=0)
        self.state.y_train = torch.cat((self.state.y_train, feedback["observation"]), dim=0)
        self.state.last_ask_Y = None
        self.state.last_ask_X = None
        self._set_gp_data()

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Return projection, metric, finite-row, and training diagnostics."""

        return {
            "embedding_dimension": self.config.embedding_dimension,
            "training_rows": int(self.state.y_train.numel()),
            "projection_rank": int(torch.linalg.matrix_rank(self.state.B).item()),
            "metric_mode": self.state.metric_mode,
            "metric_samples": len(self.state.metric_posterior_samples),
            "rejected_finite_rows": self._last_rejected_finite_rows,
        }

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize ALEBO geometry, observations, and search progress."""

        return {
            "config": repr(asdict(self.config)),
            "B": self.state.B,
            "B_pinv": self.state.B_pinv,
            "A_ineq": self.state.A_ineq,
            "b_ineq": self.state.b_ineq,
            "Y_train": self.state.Y_train,
            "X_train_symmetric": self.state.X_train_symmetric,
            "y_train": self.state.y_train,
            "metric_mode": self.state.metric_mode,
            "last_ask_Y": self.state.last_ask_Y,
            "last_ask_X": self.state.last_ask_X,
        }

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Validate required ALEBO checkpoint fields and projection shape."""

        required = {
            "config",
            "B",
            "B_pinv",
            "A_ineq",
            "b_ineq",
            "Y_train",
            "X_train_symmetric",
            "y_train",
            "metric_mode",
            "last_ask_Y",
            "last_ask_X",
        }
        if missing := required - set(state):
            raise ALEBOCheckpointError(f"ALEBO state is missing {sorted(missing)}.", algorithm=self)
        if torch.as_tensor(state["B"]).shape != self.state.B.shape:
            raise ALEBOCheckpointError("ALEBO B shape is incompatible.", algorithm=self)
        del strict

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Restore ALEBO geometry and data, then reset fitted model state."""

        del strict
        for name in ("B", "B_pinv", "A_ineq", "b_ineq", "Y_train", "X_train_symmetric", "y_train"):
            setattr(
                self.state,
                name,
                torch.as_tensor(state[name], dtype=self.dtype, device=self.device).detach().clone(),
            )
        self.state.metric_mode = str(state["metric_mode"])
        # Metric parameters are fitted surrogate state, not experiment
        # progress. The next model-based ask reconstructs them from Y_train and
        # y_train with the current fitting configuration.
        self.state.metric_map_parameters = {}
        self.state.metric_posterior_samples = []
        self.state.last_ask_Y = self._optional_tensor(state["last_ask_Y"])
        self.state.last_ask_X = self._optional_tensor(state["last_ask_X"])
        self.gp.reset()

    def _optional_tensor(self, value: Any) -> torch.Tensor | None:
        """Convert an optional checkpoint value to an owned algorithm tensor."""

        return (
            None
            if value is None
            else torch.as_tensor(value, dtype=self.dtype, device=self.device).detach().clone()
        )

    def _reset_impl(self) -> None:
        """Clear observations and redraw ALEBO's seeded projection/polytope."""

        self.gp.reset()
        self._last_finite_valid_mask = None
        self._last_rejected_finite_rows = 0
        B = self._sample_projection(self.config.embedding_dimension, self.action_dimension)
        B_pinv, A_ineq, b_ineq = self._make_polytope(B)
        self.state = ALEBOState(
            B=B,
            B_pinv=B_pinv,
            A_ineq=A_ineq,
            b_ineq=b_ineq,
            Y_train=torch.empty(
                (0, self.config.embedding_dimension), dtype=self.dtype, device=self.device
            ),
            X_train_symmetric=torch.empty(
                (0, self.action_dimension), dtype=self.dtype, device=self.device
            ),
            y_train=torch.empty((0,), dtype=self.dtype, device=self.device),
            metric_mode=self.config.metric_inference,
            metric_map_parameters={},
            metric_posterior_samples=[],
            last_ask_Y=None,
            last_ask_X=None,
        )


__all__ = ["ALEBO", "ALEBOConfig", "ALEBOMahalanobisKernel", "ALEBOState"]
