"""Adaptive Linear Embedding Bayesian Optimization (ALEBO).

Continuous ALEBO uses the exact polytope ``{y: -1 <= B^dagger y <= 1}``; it never
replaces that domain by a box. The explicit Ax 0.1.17 source protocol retains
finite optimized endpoints and clips their ambient map, as its release does;
the native and Ax 0.2.2 policies retain their strict feasibility certificates.
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
import numpy as np
import torch
import torch.nn.functional as F
from botorch.acquisition.analytic import ExpectedImprovement
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

from ._ax_source_random import export_streams, validate_streams
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
_AX01_OPTIMIZATION_POLICY = "ax-0.1.17-owned-streams-exact-metric-ei-ambient-clip-v4"


class _Ax01ExpectedImprovement(ExpectedImprovement):
    """Ordinary historical EI with its floor in transformed outcome units."""

    def _mean_and_sigma(
        self, X: torch.Tensor, compute_sigma: bool = True, min_var: float = 1e-9
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del min_var
        return super()._mean_and_sigma(X, compute_sigma=compute_sigma, min_var=1e-9)


class _Ax01StandardizedModel(Model):
    """Expose the historical GP's standardized posterior without a round trip."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.source_model = model

    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(
        self, X: torch.Tensor, output_indices: Any = None, observation_noise: Any = False,
        posterior_transform: Any = None, **kwargs: Any,
    ) -> GPyTorchPosterior:
        if output_indices is not None or observation_noise is not False:
            raise ValueError("Historical ALEBO EI requires a single latent output.")
        if isinstance(self.source_model, _ALEBOMomentMatchedModel):
            posterior = self.source_model.posterior(X, standardized_outputs=True, **kwargs)
        else:
            # ExactGP.__call__ returns standardized coordinates; SingleTaskGP's
            # posterior() would undo them with its outcome transform.
            posterior = GPyTorchPosterior(self.source_model(X))
        return posterior if posterior_transform is None else posterior_transform(posterior)


class ALEBOMahalanobisKernel(gpytorch.kernels.Kernel):
    """Squared-exponential kernel with learned positive-definite metric."""

    _ROUNDOFF_MARGIN = 8.0

    def __init__(
        self, dimension: int, min_diagonal: float = 1e-6, *, source_coordinates: bool = False,
        exact_source_metric: bool = False,
    ) -> None:
        """Initialize an unconstrained Cholesky parameter for ``D_model``."""

        super().__init__()
        self.raw_lower = torch.nn.Parameter(torch.zeros(dimension, dimension))
        self.min_diagonal = min_diagonal
        self.source_coordinates = source_coordinates
        self.exact_source_metric = exact_source_metric

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

        if self.exact_source_metric:
            return lower.new_zeros(())
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
        if self.exact_source_metric:
            lower = self._lower_factor()
            squared = self.covar_dist(y1 @ lower, y2 @ lower, diag=diag, square_dist=True)
            return squared.div(-2).exp()
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
        self.exact_source_metric = base.exact_source_metric
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
        if self.exact_source_metric:
            from gpytorch.kernels.kernel import sq_dist

            return self.scale * sq_dist(Xs, Zs, x1_eq_x2=torch.equal(Xs, Zs)).div(-2).exp()
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
        standardized_outputs: bool = False,
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
        if self.exact_source_metric:
            covariance = (
                covariances.mean(dim=-3)
                + (means.transpose(-1, -2) @ means) / means.shape[-2]
                - mean.unsqueeze(-1) @ mean.unsqueeze(-2)
            )
        if observation_noise:
            covariance = covariance + self.noise.mean() * torch.eye(
                X.shape[-2], dtype=X.dtype, device=X.device
            )
        if not standardized_outputs:
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
        ``"ax_0_1_17"`` and ``"ax_0_2_2"`` use custom ALEBO
        protocols: unrestricted triangular U, TNC fitting, finite-difference
        Laplace curvature and ambient projection/rejection initializers.
        In these modes ``noise_variance`` is in standardized reward units;
        set it to 1e-7 for the released deterministic benchmark. Every update
        uses fresh MAP restarts and ordinary standardized EI, not a warm start
        or LogEI. Checkpoints retain observations and random-stream progress.
        Ax 0.1.17 additionally preserves the released upper-triangle parameter
        order, NumPy metric samples and unregularized metric. Its unrecorded
        initializer entropy is replaced by a separately checkpointed seed stream;
        this makes new trials reproducible, not identical to unknown author seeds.
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
        if config.source_protocol == "ax_0_1_17" and self.device.type != "cpu":
            raise AlgorithmConfigurationError("Historical Ax streams require device='cpu'.")
        self._source_rng = np.random.RandomState(config.seed)
        self._initializer_seed_rng = np.random.RandomState(config.seed)
        self._initializer_seeds: list[dict[str, Any]] = []
        self._source_evaluation_draws = 0
        self._source_readback_draws = 0
        self._source_last_sampled_X: torch.Tensor | None = None
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
            config.embedding_dimension, source_coordinates=config.source_protocol.startswith("ax_"),
            exact_source_metric=config.source_protocol == "ax_0_1_17",
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
            norms = (
                torch.sqrt((matrix ** 2).sum(dim=0))
                if self.config.source_protocol == "ax_0_1_17"
                else torch.linalg.vector_norm(matrix, dim=0)
            )
            if bool((norms <= torch.finfo(self.dtype).tiny).any()):
                continue
            B = matrix / norms
            if int(torch.linalg.matrix_rank(B).item()) == d:
                return B
        raise ALEBOPolytopeError("Unable to sample a full-row-rank ALEBO matrix.", algorithm=self)

    def _make_polytope(self, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct B-pseudoinverse and exact ambient box inequalities."""

        B_pinv = (
            torch.pinverse(B)
            if self.config.source_protocol == "ax_0_1_17"
            else torch.linalg.pinv(B, rcond=1e-12)
        )
        A_ineq = torch.cat((B_pinv, -B_pinv), dim=0)
        b_ineq = torch.ones(2 * self.action_dimension, dtype=self.dtype, device=self.device)
        return B_pinv, A_ineq, b_ineq

    def _is_feasible_y(self, Y: torch.Tensor) -> torch.Tensor:
        """Return a row mask for exact-polytope feasibility in ``D_model``."""

        residual = Y @ self.state.A_ineq.T - self.state.b_ineq
        return residual.max(dim=-1).values <= self.config.feasibility_tolerance

    def _map_y_to_action(self, Y: torch.Tensor) -> torch.Tensor:
        """Map through B-dagger with the selected protocol's box policy."""

        if self.config.source_protocol == "ax_0_1_17":
            # Ax 0.1.17 ALEBO.gen selects the best SLSQP endpoint before
            # projecting up, then clamps Xopt. A post-hoc latent certificate
            # would discard source candidates and change the selected action.
            X = (self.state.B_pinv @ Y.T).T
            if not bool(torch.isfinite(Y).all() & torch.isfinite(X).all()):
                raise ALEBOPolytopeError("ALEBO source map is non-finite.", algorithm=self)
            excess = float((X.abs() - 1.0).clamp_min(0).max().item())
            if excess > 0:
                warnings.warn(
                    "Ax 0.1.17 ALEBO source projection: clipping the selected "
                    f"ambient action to [-1, 1]; maximum box excess={excess:.17g}. "
                    "This follows the released source endpoint policy; feedback "
                    "is projected from the evaluated, clipped action.",
                    ALEBOWarning,
                    stacklevel=2,
                )
            return from_symmetric_box(
                X.clamp(-1.0, 1.0), self.bounds.to(dtype=self.dtype, device=self.device)
            )
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

        if self.config.source_protocol == "ax_0_1_17":
            # Ax reconstructs an entropy-seeded NumPy UniformGenerator for
            # each initial gen(), and another for each acquisition pool. A
            # separately owned recorded seed stream replaces that unavailable
            # entropy without consuming either historical proposal stream.
            seed = int(self._initializer_seed_rng.randint(0, 2**32, dtype=np.uint32))
            self._initializer_seeds.append({"phase": phase, "seed": seed})
            rng = np.random.RandomState(seed)
            sample_count = 10000 if phase == "initial" else 10 * count
            source_unit = rng.uniform(size=(sample_count, self.action_dimension))
            B = self.state.B.detach().cpu().numpy()
            projection = np.linalg.pinv(B) @ B
            width = 16.0
            while True:
                source_ambient = (2 * width * source_unit - width) @ projection.T
                source_feasible = source_ambient[((source_ambient >= -1) & (source_ambient <= 1)).all(axis=1)]
                if len(source_feasible) >= count:
                    self._source_last_sampled_X = torch.as_tensor(
                        source_feasible[:count].copy(), dtype=self.dtype, device=self.device
                    )
                    return self._source_last_sampled_X @ self.state.B.T
                width /= 2
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
            quality = (
                self.state.y_train
                if self.config.source_protocol == "ax_0_1_17"
                else to_quality(self.state.y_train, maximize=self.maximize)
            )
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
            if self.config.source_protocol == "ax_0_1_17":
                from ._historical_ax01 import standardize_ax01_outcomes

                standardized, mean, std = standardize_ax01_outcomes(quality)
                source_model: Any = self.gp.model
                assert source_model is not None
                # Ax 0.1.17 StandardizeY uses NumPy population SD (ddof=0),
                # unlike modern BoTorch's sample SD. Preserve raw learner data
                # while matching targets, transform and fixed noise exactly.
                source_model.set_train_data(targets=standardized, strict=False)
                with torch.no_grad():
                    transform = source_model.outcome_transform
                    transform.means.fill_(mean)
                    transform.stdvs.fill_(std)
                    transform._stdvs_sq.fill_(std * std)
                    source_model.likelihood.noise_covar.noise.fill_(self.config.noise_variance)
                    assert self.gp.observation_noise is not None
                    self.gp.observation_noise.fill_(self.config.noise_variance * std * std)

    def _fit_source_gp(self) -> None:
        """Fit exactly mean, raw outputscale and unrestricted triangular U via TNC."""
        import numpy as np
        from scipy.optimize import minimize

        model, mll = self.gp.model, self.gp.mll
        assert model is not None and mll is not None
        base = model.covar_module.base_kernel
        indices = self._metric_parameter_indices()
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
        if not fitted.success:
            warnings.warn(
                f"ALEBO TNC hyperparameter fit returned status {fitted.status}: {fitted.message}. "
                "Retaining its finite endpoint, as in the released source.",
                ALEBOWarning, stacklevel=2,
            )
        model.eval()
        mll.eval()
        self.gp.is_fit = True

    def _metric_parameter_indices(self) -> torch.Tensor:
        """Map source upper-triangle row order into transposed lower storage."""
        d = self.config.embedding_dimension
        if self.config.source_protocol == "ax_0_1_17":
            return torch.triu_indices(d, d, device=self.device).flip(0)
        return torch.tril_indices(d, d, device=self.device)

    def _draw_initial_metric(self) -> torch.Tensor:
        """One source ALEBOKernel construction, including its owned Torch draw."""
        dimension = self.action_dimension
        rotation = torch.linalg.qr(torch.randn(
            dimension, dimension, dtype=self.dtype, device=self.device, generator=self.generator
        )).Q
        embedding = rotation[: self.config.embedding_dimension] @ self.state.B_pinv
        return torch.linalg.cholesky(embedding.T @ embedding)

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
                factor = self._draw_initial_metric()
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
            if self.config.source_protocol == "ax_0_1_17":
                # get_map_model constructs one more kernel before loading the
                # selected MAP state. Its random initialization is overwritten.
                self._draw_initial_metric()
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
        if not bool(torch.isfinite(metric).all()) or (
            self.config.source_protocol != "ax_0_1_17" and int(certificate.item()) != 0
        ):
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
            source_curvature: list[float] = []
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
                        ) / (
                            (raw[row, column] + epsilon) - raw[row, column]
                            if self.config.source_protocol == "ax_0_1_17" else epsilon
                        )
                        if self.config.source_protocol == "ax_0_1_17":
                            source_curvature.append(float(curvature.detach()))
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
            source_samples = None
            if self.config.source_protocol == "ax_0_1_17":
                # The release samples the triangular vector with NumPy's SVD
                # multivariate-normal implementation, not a dense Torch matrix
                # or componentwise absolute/clipped Hessian variances.
                indices = self._metric_parameter_indices()
                curvatures = torch.zeros_like(raw)
                curvatures[torch.tril_indices(raw.shape[0], raw.shape[0]).unbind()] = raw.new_tensor(source_curvature)
                source_diagonal = curvatures[indices[0], indices[1]].cpu().numpy() + 1e-3
                covariance = np.linalg.inv(np.diag(source_diagonal))
                mean = raw[indices[0], indices[1]].cpu().numpy()
                # NumPy reports this warning at its caller, so the generic
                # dependency-path translator cannot identify it reliably.
                # Relabel locally without repairing the covariance or changing
                # the released SVD sampler, its draws, or its random stream.
                with warnings.catch_warnings(record=True) as sampling_warnings:
                    warnings.simplefilter("always", RuntimeWarning)
                    samples = self._source_rng.multivariate_normal(
                        mean, covariance, size=self.config.metric_posterior_samples - 1
                    )
                for warning in sampling_warnings:
                    warnings.warn(
                        "Ax 0.1.17 ALEBO Laplace metric sampling: "
                        f"{warning.message}",
                        ALEBOWarning,
                        stacklevel=2,
                    )
                source_samples = np.vstack((mean, samples))
            for sample_index in range(self.config.metric_posterior_samples):
                if source_samples is not None:
                    sample_raw = torch.zeros_like(raw)
                    sample_raw[indices[0], indices[1]] = torch.as_tensor(
                        source_samples[sample_index], dtype=self.dtype, device=self.device
                    )
                else:
                    perturbation = variance.sqrt() * torch.randn(
                        raw.shape, dtype=self.dtype, device=self.device, generator=self.generator,
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
            if self.config.source_protocol == "ax_0_1_17":
                # get_batch_model constructs another subsequently overwritten
                # kernel, consuming one additional D-by-D Torch normal draw.
                self._draw_initial_metric()
            self._mixture_model = _ALEBOMomentMatchedModel(
                self.gp, self.state.metric_posterior_samples
            )

    def _optimize_constrained_acquisition(self, acquisition: Any) -> tuple[torch.Tensor, float]:
        """Refine feasible starts and apply the selected endpoint policy."""
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
                if self.config.source_protocol == "ax_0_1_17":
                    torch.set_rng_state(self.generator.get_state())
                else:
                    torch.manual_seed(self.config.seed + self.iteration)
                try:
                    if self.config.source_protocol == "ax_0_1_17":
                        from ._historical_ax01 import select_nonnegative_starts

                        starts = select_nonnegative_starts(
                            pool[indices].unsqueeze(1), pool_values[indices], self.config.num_restarts
                        )
                    else:
                        starts, _ = initialize_q_batch_nonneg(
                            X=pool[indices].unsqueeze(1),
                            acq_vals=pool_values[indices],
                            n=self.config.num_restarts,
                        )
                finally:
                    if self.config.source_protocol == "ax_0_1_17":
                        self.generator.set_state(torch.get_rng_state())
            starts = starts.squeeze(1)
        else:
            starts = pool[selected]
        best = pool[selected[0] : selected[0] + 1].detach().clone()
        source_protocol = self.config.source_protocol.startswith("ax_")
        source_clip = self.config.source_protocol == "ax_0_1_17"
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
                or (not source_clip and residual > self.config.feasibility_tolerance)
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
            if source_clip:
                raise ALEBOAcquisitionOptimizationError(
                    "Ax 0.1.17 ALEBO has no finite optimized acquisition endpoint. "
                    f"No screened-point substitution was made. Diagnostics: {failures}",
                    algorithm=self,
                )
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
            endpoint_policy = (
                "Returning the best finite source endpoint, with released ambient "
                f"clipping on projection; {accepted} local endpoints were finite. "
                "Non-finite endpoints were not substituted with screened points."
                if source_clip else
                "Returning the best independently feasible finite acquisition point; "
                f"{accepted} local endpoints were feasible. A screened feasible point "
                "is retained if refinement cannot improve it."
            )
            warnings.warn(
                f"ALEBO constrained refinement diagnostics: {failures}. "
                f"{endpoint_policy}",
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
                if self.config.source_protocol == "ax_0_1_17":
                    assert self._source_last_sampled_X is not None
                    action = from_symmetric_box(
                        self._source_last_sampled_X,
                        self.bounds.to(dtype=self.dtype, device=self.device),
                    )
                else:
                    action = self._map_y_to_action(Y)
            else:
                position = int(torch.argmin(table.dataset_indices).item())
                Y = table.model_inputs[position : position + 1]
                action = table.environment_actions[position : position + 1]
            value = None
        else:
            self._fit_model()
            quality = self.state.y_train if self.maximize else -self.state.y_train
            source_maximize = True
            if self.config.source_protocol == "ax_0_1_17":
                quality = self.state.y_train
                source_maximize = self.maximize
            if self.config.source_protocol.startswith(
                "ax_"
            ) and self.config.acquisition.name.lower() in {"ei", "expectedimprovement"}:
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
                    acquisition_class = (
                        _Ax01ExpectedImprovement
                        if self.config.source_protocol == "ax_0_1_17" else ExpectedImprovement
                    )
                    incumbent = (
                        quality.max() + self.config.acquisition.xi
                        if source_maximize else quality.min() - self.config.acquisition.xi
                    )
                    acquisition_model = (
                        self._mixture_model if self.config.metric_inference == "laplace" else model
                    )
                    if self.config.source_protocol == "ax_0_1_17":
                        best_f = (model.train_targets.max() if source_maximize
                                  else model.train_targets.min())
                        best_f = best_f + ((1 if source_maximize else -1) * self.config.acquisition.xi / std)
                        acquisition = acquisition_class(
                            model=_Ax01StandardizedModel(acquisition_model),
                            best_f=best_f, maximize=source_maximize,
                        )
                    else:
                        acquisition = acquisition_class(
                            model=acquisition_model,
                            best_f=(incumbent - mean) / std,
                            posterior_transform=ScalarizedPosteriorTransform(
                                weights=std.reciprocal().reshape(1), offset=float(-mean / std)
                            ),
                            maximize=source_maximize,
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
        if self.config.source_protocol == "ax_0_1_17":
            pending_Y = (self.state.B @ pending_X.T).T
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
            "source_optimization_policy": self._source_optimization_policy,
            "source_evaluation_draws": self._source_evaluation_draws,
            "source_readback_draws": self._source_readback_draws,
            "initializer_seed_records": copy.deepcopy(self._initializer_seeds),
        }

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize ALEBO geometry, observations, and search progress."""

        return {
            "config": repr(asdict(self.config)),
            "source_optimization_policy": self._source_optimization_policy,
            "source_random_streams": export_streams(
                self._source_rng, self.generator, self._initializer_seed_rng
            ),
            "source_evaluation_draws": self._source_evaluation_draws,
            "source_readback_draws": self._source_readback_draws,
            "initializer_seed_records": copy.deepcopy(self._initializer_seeds),
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
            and state.get("source_optimization_policy") != self._source_optimization_policy
        ):
            raise ALEBOCheckpointError(
                "ALEBO historical-source numerical/random-stream policy differs. Preserve "
                "the old results and start a new output directory.",
                algorithm=self,
            )
        if self.config.source_protocol == "ax_0_1_17":
            try:
                _, _, initializer = validate_streams(state["source_random_streams"])
                if initializer is None:
                    raise ValueError("Missing initializer seed stream.")
                for name in ("source_evaluation_draws", "source_readback_draws"):
                    value = state[name]
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise ValueError(f"Invalid {name}.")
                records = state["initializer_seed_records"]
                if not isinstance(records, list) or any(
                    not isinstance(row, Mapping)
                    or set(row) != {"phase", "seed"}
                    or row["phase"] not in {"initial", "acquisition"}
                    or isinstance(row["seed"], bool)
                    or not isinstance(row["seed"], int)
                    or not 0 <= row["seed"] < 2**32
                    for row in records
                ):
                    raise ValueError("Invalid initializer seed records.")
            except (KeyError, ValueError, TypeError, RuntimeError) as error:
                raise ALEBOCheckpointError(f"Invalid Ax source state: {error}", algorithm=self) from error
        del strict

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Restore ALEBO geometry and data, then reset fitted model state."""

        del strict
        if self.config.source_protocol == "ax_0_1_17":
            self._import_random_streams(state["source_random_streams"])
            self._source_evaluation_draws = state["source_evaluation_draws"]
            self._source_readback_draws = state["source_readback_draws"]
            self._initializer_seeds = copy.deepcopy(state["initializer_seed_records"])
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
        self._source_rng.seed(self.seed)
        self._initializer_seed_rng.seed(self.seed)
        self._initializer_seeds = []
        self._source_evaluation_draws = 0
        self._source_readback_draws = 0
        self._source_last_sampled_X = None
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

    @property
    def _supports_ax017_evaluation(self) -> bool:
        return self.config.source_protocol == "ax_0_1_17"

    @property
    def _supports_shared_trial_streams(self) -> bool:
        return self.config.source_protocol == "ax_0_1_17"

    @property
    def _source_optimization_policy(self) -> str | None:
        if self.config.source_protocol == "ax_0_1_17":
            return _AX01_OPTIMIZATION_POLICY
        return _AX_OPTIMIZATION_POLICY if self.config.source_protocol == "ax_0_2_2" else None

    def _export_random_streams(self) -> dict[str, Any]:
        """Copy Ax streams without modifying progress, including pending asks."""
        if self.config.source_protocol != "ax_0_1_17":
            raise NotImplementedError("Ax stream exchange requires source_protocol='ax_0_1_17'.")
        return export_streams(self._source_rng, self.generator, self._initializer_seed_rng)

    def _import_random_streams(self, state: Mapping[str, Any]) -> None:
        """Validate streams atomically; an omitted initializer stream is preserved."""
        if self.config.source_protocol != "ax_0_1_17":
            raise NotImplementedError("Ax stream exchange requires source_protocol='ax_0_1_17'.")
        numpy_rng, generator, initializer = validate_streams(state)
        self._source_rng, self.generator = numpy_rng, generator
        if initializer is not None:
            self._initializer_seed_rng = initializer

    def _consume_source_evaluation_draws(self, count: int) -> None:
        """Advance only the source's discarded zero-noise evaluation draws."""
        if self.config.source_protocol != "ax_0_1_17":
            raise NotImplementedError("Ax evaluation draws require source_protocol='ax_0_1_17'.")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Source evaluation-draw count must be a nonnegative integer.")
        self._source_rng.randn(count)
        self._source_evaluation_draws += count

    def _consume_source_readback_draws(self, count: int) -> None:
        """Advance Ax's cached-data metric draws without new evaluations.

        Experiment supplies the full environment history count before ``ask``.
        For an interleaved strategy this is not the selected child's local count.
        """
        if self.config.source_protocol != "ax_0_1_17":
            raise NotImplementedError("Ax readback draws require source_protocol='ax_0_1_17'.")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Source readback-draw count must be a nonnegative integer.")
        self._source_rng.randn(count)
        self._source_readback_draws += count

    def _reset_shared_random_streams(self, state: Mapping[str, Any]) -> None:
        """Reconstruct one projection from the supplied continuing Torch stream."""
        if self.evaluations or self.last_ask is not None:
            raise ValueError("Projection reset is allowed only on a fresh source learner.")
        validate_streams(state)
        self.reset()
        self._import_random_streams(state)
        B = self._sample_projection(self.config.embedding_dimension, self.action_dimension)
        self.state.B = B
        self.state.B_pinv, self.state.A_ineq, self.state.b_ineq = self._make_polytope(B)


__all__ = ["ALEBO", "ALEBOConfig", "ALEBOMahalanobisKernel", "ALEBOState"]
