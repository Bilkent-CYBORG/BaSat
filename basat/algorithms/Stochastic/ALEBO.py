"""Adaptive Linear Embedding Bayesian Optimization (ALEBO).

Continuous ALEBO uses the exact polytope ``{y: -1 <= B^dagger y <= 1}``; it never
replaces that domain by a box and never clips a substantively infeasible map.
Finite-row selection uses a separately warned projection extension by default.
"""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import gpytorch
import torch
import torch.nn.functional as F
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from linear_operator.utils.errors import NanError, NotPSDError  # type: ignore[import-untyped]

from basat.algorithms.algorithm import (
    Algorithm,
    AlgorithmConfigurationError,
    resolve_batch_configuration,
)
from basat.algorithms.helper import LastAskRecord, SourceRecord, to_quality
from basat.models.Acquisitions import (
    AcquisitionConfig,
    DiscreteAcquisitionCandidates,
    build_acquisition,
    optimize_acquisition_discrete,
)
from basat.models.GaussianProcess import GaussianProcess, GaussianProcessConfig
from basat.models.utils import Bounds, from_symmetric_box, to_symmetric_box
from basat.warnings import translate_backend_warnings

from .Exception.exception_ALEBO import (
    ALEBOAcquisitionOptimizationError,
    ALEBOCheckpointError,
    ALEBODataCompatibilityError,
    ALEBOException,
    ALEBOHistoricalDataError,
    ALEBOMetricError,
    ALEBOPolytopeError,
    ALEBOWarning,
)

_AX_OPTIMIZATION_POLICY = "ax-fresh-map-ordinary-ei-v2"


class ALEBOMahalanobisKernel(gpytorch.kernels.Kernel):
    """Squared-exponential kernel with learned positive-definite metric."""

    _ROUNDOFF_MARGIN = 8.0

    def __init__(
        self, dimension: int, min_diagonal: float = 1e-6, *, source_coordinates: bool = False
    ) -> None:
        """Initialize an unconstrained Cholesky parameter for ``D_model``."""

        super().__init__()
        self.raw_lower = torch.nn.Parameter(torch.zeros(dimension, dimension))
        self.min_diagonal = min_diagonal
        self.source_coordinates = source_coordinates

    def _lower_factor(self, raw_lower: torch.Tensor | None = None) -> torch.Tensor:
        """Return ALEBO's constrained lower-triangular metric factor."""

        raw = self.raw_lower if raw_lower is None else raw_lower
        lower = torch.tril(raw)
        if self.source_coordinates:
            return lower
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


class _ALEBOMomentMatchedModel(Model):
    """Moment-match the metric-posterior Gaussian mixture, including covariance."""

    train_X: torch.Tensor
    factors: torch.Tensor
    jitter: torch.Tensor
    scale: torch.Tensor
    constant: torch.Tensor
    noise: torch.Tensor
    outcome_mean: torch.Tensor
    outcome_std: torch.Tensor
    cholesky: torch.Tensor
    alpha: torch.Tensor

    def __init__(self, gp: GaussianProcess, metrics: list[dict[str, torch.Tensor]]) -> None:
        super().__init__()
        model: Any = gp.model
        assert model is not None
        base = model.covar_module.base_kernel
        self.register_buffer("train_X", model.train_inputs[0].detach().clone())
        self.register_buffer(
            "factors",
            torch.stack([base._lower_factor(row["raw_lower"]).detach() for row in metrics]),
        )
        self.register_buffer(
            "jitter", torch.stack([row["roundoff_jitter"].detach() for row in metrics])
        )
        self.register_buffer("scale", model.covar_module.outputscale.detach().clone())
        self.register_buffer("constant", model.mean_module.constant.detach().clone())
        self.register_buffer("noise", model.likelihood.noise.detach().clone())
        transform = getattr(model, "outcome_transform", None)
        self.register_buffer(
            "outcome_mean",
            self.train_X.new_tensor(0.0)
            if transform is None
            else transform.means.detach().reshape(()),
        )
        self.register_buffer(
            "outcome_std",
            self.train_X.new_tensor(1.0)
            if transform is None
            else transform.stdvs.detach().reshape(()),
        )
        with torch.no_grad():
            covariance = self._covariance(self.train_X, self.train_X)
            covariance = covariance + torch.diag_embed(
                self.noise.expand(covariance.shape[0], self.train_X.shape[0])
            )
            factor = torch.linalg.cholesky(covariance)
            centered = model.train_targets.detach() - self.constant
            self.register_buffer("cholesky", factor)
            self.register_buffer(
                "alpha",
                torch.cholesky_solve(
                    centered.expand(covariance.shape[0], -1).unsqueeze(-1), factor
                ),
            )

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    def _covariance(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        Xs = X.unsqueeze(-3) @ self.factors
        Zs = Z.unsqueeze(-3) @ self.factors
        delta = Xs.unsqueeze(-2) - Zs.unsqueeze(-3)
        physical = X.unsqueeze(-2) - Z.unsqueeze(-3)
        return self.scale * torch.exp(
            -0.5
            * (
                delta.square().sum(-1)
                + self.jitter[:, None, None] * physical.square().sum(-1).unsqueeze(-3)
            )
        )

    def posterior(
        self,
        X: torch.Tensor,
        output_indices: Any = None,
        observation_noise: Any = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        del kwargs
        if output_indices is not None:
            raise ValueError("ALEBO supports one output.")
        cross = self._covariance(X, self.train_X)
        means = self.constant + (cross @ self.alpha).squeeze(-1)
        solved = torch.linalg.solve_triangular(self.cholesky, cross.transpose(-1, -2), upper=False)
        covariances = self._covariance(X, X) - solved.transpose(-1, -2) @ solved
        mean = means.mean(dim=-2)
        centered = means - mean.unsqueeze(-2)
        covariance = (
            covariances.mean(dim=-3) + (centered.transpose(-1, -2) @ centered) / means.shape[-2]
        )
        if observation_noise:
            covariance = covariance + self.noise.mean() * torch.eye(
                X.shape[-2], dtype=X.dtype, device=X.device
            )
        mean = mean * self.outcome_std + self.outcome_mean
        covariance = covariance * self.outcome_std.square()
        posterior = GPyTorchPosterior(gpytorch.distributions.MultivariateNormal(mean, covariance))
        return posterior if posterior_transform is None else posterior_transform(posterior)


@dataclass(frozen=True, slots=True)
class ALEBOConfig:
    """Configuration of the ALEBO polytope, metric, and solver.

    Parameters
    ----------
    finite_candidate_mode:
        "project" scores actual finite rows in compressed coordinates (a warned
        extension); "exact" requires original embedding geometry. Continuous
        proposal generation is unchanged.
    embedding_dimension:
        Explicit reduced ``D_model``.
    initial_points, acquisition:
        Initial-design size and EI/UCB acquisition settings.
    initial_points_sampling_method:
        ``"paper"``, ``"sobol"``, or ``"uniform"`` initial-design sampling.
    source_protocol:
        ``"basat"`` retains the native metric implementation. Historical
        ``"ax_0_1_17"`` and ``"ax_0_2_2"`` use equivalent custom ALEBO
        protocols: unrestricted triangular U, TNC fitting, finite-difference
        Laplace curvature and ambient projection/rejection initializers.
        In these modes ``noise_variance`` is in standardized reward units;
        set it to 1e-7 for the released deterministic benchmark. Every update
        uses fresh MAP restarts and ordinary standardized EI, not a warm start
        or LogEI. Checkpoints retain observations and random-stream progress.
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
    source_protocol: Literal["basat", "ax_0_1_17", "ax_0_2_2"] = "basat"
    finite_candidate_mode: Literal["project", "exact"] = field(default="project", kw_only=True)
    initial_points: int | None = None
    initial_points_sampling_method: Literal["paper", "sobol", "uniform"] = "paper"
    acquisition: AcquisitionConfig = field(default_factory=lambda: AcquisitionConfig(name="EI"))
    metric_inference: Literal["laplace", "map"] = "laplace"
    metric_posterior_samples: int = 25
    feasibility_tolerance: float = 1e-9
    acquisition_solver: Literal["slsqp", "trust-constr"] = "slsqp"
    raw_feasible_samples: int = 2048
    num_restarts: int = 20
    fit_restarts: int = 10
    noise_variance: float = 1e-6
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
    >>> import basat
    >>> bounds = basat.Bounds(lower=[-1.0] * 6, upper=[1.0] * 6)
    >>> algorithm = basat.ALEBO(
    ...     bounds=bounds,
    ...     config=basat.ALEBOConfig(embedding_dimension=2, seed=7),
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
            initial_points_sampling_method=config.initial_points_sampling_method,
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
        self.metric_kernel = ALEBOMahalanobisKernel(
            config.embedding_dimension, source_coordinates=config.source_protocol.startswith("ax_")
        ).to(dtype=self.dtype, device=self.device)
        self.gp = GaussianProcess(
            GaussianProcessConfig(
                kernel=gpytorch.kernels.ScaleKernel(self.metric_kernel),
                mean="constant",
                noise_mode="fixed",
                noise_variance=config.noise_variance,
                fit_method="map" if config.metric_inference == "laplace" else "mle",
                dtype=self.dtype,
                device=self.device,
            ),
            input_dimension=config.embedding_dimension,
        )
        self._last_finite_valid_mask: torch.Tensor | None = None
        self._metric_fit_state: dict[str, torch.Tensor] | None = None
        self._last_rejected_finite_rows = 0
        if config.metric_inference == "map":
            self.invalidate_source_guarantees("alebo.map_metric_approximation")

    @staticmethod
    def _validate_config(config: ALEBOConfig, D: int) -> None:
        """Validate immutable ALEBO settings against ambient dimension ``D``."""

        if not isinstance(config, ALEBOConfig):
            raise AlgorithmConfigurationError("config must be ALEBOConfig.")
        if config.source_protocol not in {"basat", "ax_0_1_17", "ax_0_2_2"}:
            raise AlgorithmConfigurationError("Unknown ALEBO source_protocol.")
        if config.finite_candidate_mode not in {"project", "exact"}:
            raise AlgorithmConfigurationError("finite_candidate_mode must be project or exact.")
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
        for name in (
            "metric_posterior_samples",
            "raw_feasible_samples",
            "num_restarts",
            "fit_restarts",
        ):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise AlgorithmConfigurationError(f"{name} must be a positive integer.")
        if not math.isfinite(config.feasibility_tolerance) or config.feasibility_tolerance <= 0:
            raise AlgorithmConfigurationError("feasibility_tolerance must be positive and finite.")
        if not math.isfinite(config.noise_variance) or config.noise_variance < 0:
            raise AlgorithmConfigurationError("noise_variance must be finite and nonnegative.")
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

    def _perturb_continuous_batch_record(self, record: LastAskRecord, offset: int) -> LastAskRecord:
        """Separate repeated optima by a feasible convex move in latent space."""

        if record.model_inputs is None:
            raise ALEBODataCompatibilityError("ALEBO batch proposal has no latent coordinates.")
        Y = record.model_inputs.detach().clone()
        coordinate = offset % self.config.embedding_dimension
        target = torch.zeros_like(Y)
        radius = self.state.B_pinv[:, coordinate].abs().max().reciprocal()
        direction = -1.0 if Y[0, coordinate] > 0 else 1.0
        target[0, coordinate] = direction * radius * 0.5
        fraction = min(0.5, 1e-4 * (1 + offset // self.config.embedding_dimension))
        Y = (1.0 - fraction) * Y + fraction * target
        actions = self._map_y_to_action(Y)
        metadata = dict(record.algorithm_metadata)
        metadata["constraint_residual"] = float(
            (Y @ self.state.A_ineq.T - self.state.b_ineq).max().item()
        )
        return LastAskRecord(
            actions=actions,
            model_inputs=Y,
            round_info_snapshot=record.round_info_snapshot,
            algorithm_metadata=metadata,
        )

    def _recover_y_if_exact(
        self, actions: torch.Tensor, *, historical: bool = False
    ) -> torch.Tensor:
        """Use y=Bx; require reconstruction only for the strict finite policy."""

        X = to_symmetric_box(actions, self.bounds.to(dtype=self.dtype, device=self.device))
        Y = X @ self.state.B.T
        if self.config.finite_candidate_mode == "project":
            return Y.detach().clone()
        reconstructed = Y @ self.state.B_pinv.T
        valid = torch.isclose(reconstructed, X, rtol=0.0, atol=1e-8).all(dim=-1)
        valid &= self._is_feasible_y(Y)
        if not bool(valid.all()):
            error = ALEBOHistoricalDataError if historical else ALEBODataCompatibilityError
            raise error("Actions do not lie in ALEBO's exact embedded image.", algorithm=self)
        return Y.detach().clone()

    def _initialize_from_environment_impl(self, data: Any) -> None:
        """Encode visible history, with optional strict embedded-image checks."""

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
        if self.config.finite_candidate_mode == "project":
            valid = torch.ones_like(valid)
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

        if self.config.source_protocol.startswith("ax_"):
            # Ax ALEBOInitializer: ambient uniform points, projection, then
            # rejection with a shared box half-width beginning at16.
            sample_count = 10000 if phase == "initial" else 10 * count
            unit = torch.rand(
                (sample_count, self.action_dimension),
                dtype=self.dtype,
                device=self.device,
                generator=self.generator,
            )
            width = 16.0
            while True:
                ambient = width * (2 * unit - 1)
                latent = ambient @ self.state.B.T
                mapped = latent @ self.state.B_pinv.T
                feasible = latent[((mapped >= -1) & (mapped <= 1)).all(dim=-1)]
                if feasible.shape[0] >= count:
                    return feasible[:count]
                width /= 2
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
            quality = to_quality(self.state.y_train, maximize=self.maximize)
            source_noise = None
            if self.config.source_protocol.startswith("ax_"):
                variance = quality.var() if quality.numel() > 1 else quality.new_tensor(1.0)
                if not torch.isfinite(variance) or variance < 1e-16:
                    variance = quality.new_tensor(1.0)
                # The common GP wrapper standardizes its raw-unit Yvar. Ax
                # floors AFTER StandardizeY, so undo that transform here.
                source_noise = self.config.noise_variance * variance
            self.gp.set_data(
                self.state.Y_train,
                quality,
                noise_variance=source_noise,
            )

    def _fit_source_gp(self) -> None:
        """Fit exactly mean, raw outputscale and unrestricted triangular U via TNC."""
        import numpy as np
        from scipy.optimize import minimize

        model, mll = self.gp.model, self.gp.mll
        assert model is not None and mll is not None
        base = model.covar_module.base_kernel
        indices = torch.tril_indices(
            self.config.embedding_dimension, self.config.embedding_dimension
        )
        parameters = (
            model.mean_module.raw_constant,
            model.covar_module.raw_outputscale,
            base.raw_lower,
        )
        initial = (
            torch.cat(
                (
                    parameters[0].reshape(-1),
                    parameters[1].reshape(-1),
                    parameters[2][indices[0], indices[1]],
                )
            )
            .detach()
            .cpu()
            .double()
            .numpy()
        )

        def objective(vector: Any) -> tuple[float, Any]:
            values = torch.as_tensor(vector, dtype=self.dtype, device=self.device)
            with torch.no_grad():
                parameters[0].copy_(values[0].reshape_as(parameters[0]))
                parameters[1].copy_(values[1].reshape_as(parameters[1]))
                parameters[2].zero_()
                parameters[2][indices[0], indices[1]] = values[2:]
            loss = -mll(model(model.train_inputs[0]), model.train_targets)
            gradients = torch.autograd.grad(loss, parameters)
            gradient = torch.cat(
                (
                    gradients[0].reshape(-1),
                    gradients[1].reshape(-1),
                    gradients[2][indices[0], indices[1]],
                )
            )
            return float(loss.detach()), gradient.detach().cpu().double().numpy().astype(np.float64)

        model.train()
        mll.train()
        with translate_backend_warnings(operation="Historical Ax ALEBO TNC hyperparameter fit"):
            fitted = minimize(objective, initial, jac=True, method="TNC")
            objective(fitted.x)
        model.eval()
        mll.eval()
        self.gp.is_fit = True

    def _fit_model(self) -> None:
        """Fit the GP and record MAP or local-Laplace metric state.

        Raises
        ------
        ALEBOMetricError
            If fitting fails or the resulting metric is not positive definite.
        """

        # Both pinned Ax versions set refit_on_update=True. Their update()
        # passes state_dicts=None, selecting all fresh MAP restarts every time;
        # the one-restart warm-start branch belongs to the non-default policy.
        self._set_gp_data()
        assert self.gp.model is not None
        fresh_state = copy.deepcopy(self.gp.model.state_dict())
        try:
            best_loss = math.inf
            best_state = None
            for restart in range(self.config.fit_restarts):
                del restart
                model = self.gp.model
                assert model is not None
                model.load_state_dict(fresh_state)
                dimension = self.action_dimension
                rotation = torch.linalg.qr(
                    torch.randn(
                        dimension,
                        dimension,
                        dtype=self.dtype,
                        device=self.device,
                        generator=self.generator,
                    )
                ).Q
                embedding = rotation[: self.config.embedding_dimension] @ self.state.B_pinv
                factor = torch.linalg.cholesky(embedding.T @ embedding)
                raw = factor.clone()
                if not self.config.source_protocol.startswith("ax_"):
                    diagonal = (factor.diag() - 1e-6).clamp_min(1e-12)
                    raw.diagonal().copy_(torch.log(torch.expm1(diagonal)))
                with torch.no_grad():
                    model.covar_module.base_kernel.raw_lower.copy_(raw)
                if self.config.source_protocol.startswith("ax_"):
                    self._fit_source_gp()
                else:
                    self.gp.fit(maxiter=75)
                fitted_model = self.gp.model
                assert fitted_model is not None and self.gp.mll is not None
                fitted_model.train()
                objective = float(
                    (
                        -self.gp.mll(
                            fitted_model(fitted_model.train_inputs[0]), fitted_model.train_targets
                        )
                    ).detach()
                )
                fitted_model.eval()
                if objective < best_loss:
                    best_loss = objective
                    best_state = copy.deepcopy(fitted_model.state_dict())
            assert self.gp.model is not None and best_state is not None
            self.gp.model.load_state_dict(best_state)
            self._metric_fit_state = None
        except Exception as error:
            raise ALEBOMetricError(
                f"ALEBO metric fitting failed: {error}", algorithm=self
            ) from error
        base = self.gp.model.covar_module.base_kernel
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
            # Diagonal curvature, not an arbitrary perturbation scale, defines
            # the Laplace posterior. Include the MAP as the first sample as Ax
            # does, then use all draws in moment-matched acquisition prediction.
            model = self.gp.model
            mll = self.gp.mll
            assert model is not None and mll is not None
            model.train()
            mll.train()
            raw = base.raw_lower.detach().clone()
            loss = -mll(model(model.train_inputs[0]), model.train_targets)
            gradient = torch.autograd.grad(loss, base.raw_lower, create_graph=True)[0]
            variance = torch.zeros_like(raw)
            for row in range(raw.shape[0]):
                for column in range(row + 1):
                    if self.config.source_protocol.startswith("ax_"):
                        epsilon = 1e-4 + 1e-3 * raw[row, column].abs()
                        with torch.no_grad():
                            base.raw_lower.copy_(raw)
                            base.raw_lower[row, column] += epsilon
                        shifted_loss = -mll(model(model.train_inputs[0]), model.train_targets)
                        shifted_gradient = torch.autograd.grad(shifted_loss, base.raw_lower)[0]
                        curvature = (
                            shifted_gradient[row, column] - gradient[row, column].detach()
                        ) / epsilon
                        # NumPy's released multivariate_normal uses singular
                        # values if its numerical covariance is indefinite.
                        variance[row, column] = (
                            (curvature.detach() + 1e-3).abs().clamp_min(1e-12).reciprocal()
                        )
                    else:
                        curvature = torch.autograd.grad(
                            gradient[row, column], base.raw_lower, retain_graph=True
                        )[0][row, column]
                        variance[row, column] = (
                            (curvature.detach() + 1e-3).clamp_min(1e-3).reciprocal()
                        )
            with torch.no_grad():
                base.raw_lower.copy_(raw)
            model.eval()
            mll.eval()
            self.state.metric_posterior_samples = []
            for sample_index in range(self.config.metric_posterior_samples):
                perturbation = variance.sqrt() * torch.randn(
                    raw.shape,
                    dtype=self.dtype,
                    device=self.device,
                    generator=self.generator,
                )
                if sample_index == 0:
                    perturbation.zero_()
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
            self._mixture_model = _ALEBOMomentMatchedModel(
                self.gp, self.state.metric_posterior_samples
            )

    def _optimize_constrained_acquisition(self, acquisition: Any) -> tuple[torch.Tensor, float]:
        """Screen feasible starts, refine with derivatives, and retain certified points."""
        try:
            import numpy as np
            from scipy.optimize import LinearConstraint, minimize
        except ImportError as error:
            raise ALEBOAcquisitionOptimizationError(
                "SciPy is required by ALEBO.", algorithm=self
            ) from error
        pool = self._sample_feasible_Y(
            max(self.config.raw_feasible_samples, self.config.num_restarts), phase="acquisition"
        )
        with torch.no_grad():
            pool_values = acquisition(pool.unsqueeze(-2)).reshape(-1)
        finite = (
            torch.isfinite(pool_values)
            & torch.isfinite(pool).all(dim=-1)
            & self._is_feasible_y(pool)
        )
        if not bool(finite.any()):
            raise ALEBOAcquisitionOptimizationError(
                "ALEBO acquisition has no finite value on its feasible screened points.",
                algorithm=self,
            )
        indices = torch.nonzero(finite, as_tuple=False).reshape(-1)
        order = torch.argsort(pool_values[indices], descending=True, stable=True)
        selected = indices[order[: self.config.num_restarts]]
        if self.config.source_protocol.startswith("ax_"):
            from botorch.optim.initializers import initialize_q_batch_nonneg

            with (
                torch.random.fork_rng(devices=[]),
                translate_backend_warnings(
                    operation="Historical Ax ALEBO nonnegative-EI starting points"
                ),
            ):
                torch.manual_seed(self.config.seed + self.iteration)
                starts, _ = initialize_q_batch_nonneg(
                    X=pool[indices].unsqueeze(1),
                    acq_vals=pool_values[indices],
                    n=self.config.num_restarts,
                )
            starts = starts.squeeze(1)
        else:
            starts = pool[selected]
        best = pool[selected[0] : selected[0] + 1].detach().clone()
        source_protocol = self.config.source_protocol.startswith("ax_")
        # The source selects the best optimized endpoint, not the best of
        # endpoints and the original raw candidate pool.
        best_value = -math.inf if source_protocol else float(pool_values[selected[0]])
        A = self.state.A_ineq.detach().cpu().numpy()
        b = self.state.b_ineq.detach().cpu().numpy()
        constraint = LinearConstraint(A, -np.inf, b)

        def objective(values: Any) -> float:
            point = torch.as_tensor(values, dtype=self.dtype, device=self.device).reshape(1, 1, -1)
            with torch.no_grad():
                value = -float(acquisition(point).item())
            return value

        class UnavailableGradientError(Exception):
            """This acquisition provides no derivative with respect to its input."""

        numerical_errors = (FloatingPointError, torch.linalg.LinAlgError, NanError, NotPSDError)

        def objective_with_gradient(values: Any) -> tuple[float, Any]:
            with torch.enable_grad():
                point = (
                    torch.as_tensor(values, dtype=self.dtype, device=self.device)
                    .reshape(1, 1, -1)
                    .detach()
                    .requires_grad_(True)
                )
                loss = -acquisition(point).reshape(())
                if not loss.requires_grad:
                    raise UnavailableGradientError(
                        "Acquisition does not provide an analytic gradient."
                    )
                raw_gradient = torch.autograd.grad(loss, point, allow_unused=True)[0]
                if raw_gradient is None:
                    raise UnavailableGradientError(
                        "Acquisition is not differentiably connected to its input."
                    )
                gradient = raw_gradient.reshape(-1)
            if not bool(torch.isfinite(loss) & torch.isfinite(gradient).all()):
                raise FloatingPointError("Non-finite ALEBO acquisition derivative.")
            return float(loss.detach()), gradient.detach().cpu().double().numpy()

        method = "SLSQP" if self.config.acquisition_solver == "slsqp" else "trust-constr"
        accepted = 0
        failures: list[dict[str, Any]] = []
        for start in starts:
            x0 = start.detach().cpu().double().numpy()
            try:
                objective_with_gradient(x0)
                has_gradient = True
            except (*numerical_errors, UnavailableGradientError) as error:
                has_gradient = False
                failures.append(
                    {
                        "success": False,
                        "stage": "analytic gradient; using finite differences",
                        "message": str(error),
                        "residual": None,
                    }
                )
            try:
                result = minimize(
                    objective_with_gradient if has_gradient else objective,
                    x0,
                    jac=True if has_gradient else None,
                    method=method,
                    constraints=[constraint],
                    options=(
                        {}
                        if source_protocol and method == "SLSQP"
                        else {"maxiter": 200, "ftol": 1e-9}
                        if method == "SLSQP"
                        else {"maxiter": 200}
                    ),
                )
            except (*numerical_errors, UnavailableGradientError) as error:
                failures.append({"success": False, "message": str(error), "residual": None})
                continue
            candidate = torch.as_tensor(result.x, dtype=self.dtype, device=self.device).reshape(
                1, -1
            )
            residual = float((candidate @ self.state.A_ineq.T - self.state.b_ineq).max())
            if (
                not bool(torch.isfinite(candidate).all())
                or not math.isfinite(residual)
                or residual > self.config.feasibility_tolerance
            ):
                failures.append(
                    {
                        "success": bool(result.success),
                        "stage": "finite-point and polytope certification",
                        "message": str(result.message),
                        "residual": residual,
                    }
                )
                continue
            try:
                value = -objective(result.x)
            except (*numerical_errors, UnavailableGradientError) as error:
                failures.append(
                    {
                        "success": False,
                        "stage": "endpoint acquisition reevaluation",
                        "message": str(error),
                        "residual": residual,
                    }
                )
                continue
            if not math.isfinite(value):
                failures.append(
                    {
                        "success": False,
                        "stage": "endpoint acquisition reevaluation",
                        "message": "Non-finite acquisition value.",
                        "residual": residual,
                    }
                )
                continue
            accepted += 1
            if value > best_value:
                best, best_value = candidate.detach().clone(), value
            if not result.success:
                failures.append(
                    {"success": False, "message": str(result.message), "residual": residual}
                )
        if accepted == 0 and source_protocol:
            best_value = float(pool_values[selected[0]])
            failures.append(
                {
                    "success": False,
                    "stage": "all source optimizer endpoints failed certification",
                    "message": "Using the best finite feasible screened point as an explicit fallback.",
                    "residual": None,
                }
            )
        if failures:
            warnings.warn(
                f"ALEBO constrained refinement diagnostics: {failures}. "
                f"Returning the best independently feasible finite acquisition point; {accepted} local endpoints were feasible. "
                "A screened feasible point is retained if refinement cannot improve it.",
                ALEBOWarning,
                stacklevel=2,
            )
            self.invalidate_source_guarantees("alebo.constrained_acquisition_refinement_warning")
        return best, best_value

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
            if self.config.source_protocol.startswith(
                "ax_"
            ) and self.config.acquisition.name.lower() in {"ei", "expectedimprovement"}:
                from botorch.acquisition.analytic import ExpectedImprovement
                from botorch.acquisition.objective import ScalarizedPosteriorTransform

                model = self.gp.model
                assert model is not None
                transform: Any = model.outcome_transform
                mean = transform.means.reshape(())
                std = transform.stdvs.reshape(())
                # Historical Ax applies StandardizeY before constructing its
                # ordinary analytic EI. Modern generic LogEI is not a drop-in
                # replacement for its nonnegative-EI restart initializer.
                with translate_backend_warnings(operation="Historical Ax ALEBO analytic EI"):
                    acquisition = ExpectedImprovement(
                        model=(
                            self._mixture_model
                            if self.config.metric_inference == "laplace"
                            else model
                        ),
                        best_f=(quality.max() + self.config.acquisition.xi - mean) / std,
                        posterior_transform=ScalarizedPosteriorTransform(
                            weights=std.reciprocal().reshape(1), offset=float(-mean / std)
                        ),
                        maximize=True,
                    )
            else:
                acquisition = build_acquisition(
                    model=self._mixture_model
                    if self.config.metric_inference == "laplace"
                    else self.gp,
                    config=self.config.acquisition,
                    q=1,
                    best_f=(
                        quality.max()
                        if self.config.acquisition.name.lower().startswith("e")
                        else None
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
            "source_optimization_policy": (
                _AX_OPTIMIZATION_POLICY if self.config.source_protocol.startswith("ax_") else None
            ),
        }

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize ALEBO geometry, observations, and search progress."""

        return {
            "config": repr(asdict(self.config)),
            "source_optimization_policy": (
                _AX_OPTIMIZATION_POLICY if self.config.source_protocol.startswith("ax_") else None
            ),
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
        if (
            self.config.source_protocol.startswith("ax_")
            and state.get("source_optimization_policy") != _AX_OPTIMIZATION_POLICY
        ):
            raise ALEBOCheckpointError(
                "ALEBO historical-source optimization policy differs: this version uses fresh "
                "MAP restarts and ordinary standardized EI on every update. Preserve the old "
                "warm-start/LogEI results and start a new output directory.",
                algorithm=self,
            )
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
        # The GP is rebuilt from the learner's observations. Historical Ax
        # uses fresh random-restart fitting, so no optimizer warm-start or
        # trained GP is restored. The base RNG state preserves fresh draws.
        self.state.metric_map_parameters = {}
        self.state.metric_posterior_samples = []
        self._metric_fit_state = None
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
        self._metric_fit_state = None
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
