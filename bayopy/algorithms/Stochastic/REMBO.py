"""Random EMbedding Bayesian Optimization (REMBO).

The search lives in ``y in Y subset R^d``.  Actions are generated only through
the paper map ``x = p_X(Ay)`` in the symmetric ambient box and then transformed
back to physical environment bounds. Finite rows use a warned projected-coordinate
selection extension by default; exact stored-coordinate mode remains available.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import gpytorch
import torch

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
    optimize_acquisition_continuous,
    optimize_acquisition_discrete,
)
from bayopy.models.GaussianProcess import GaussianProcess, GaussianProcessConfig
from bayopy.models.utils import Bounds, from_symmetric_box, to_symmetric_box

from .Exception.exception_REMBO import (
    REMBOAcquisitionError,
    REMBOCheckpointError,
    REMBODataCompatibilityError,
    REMBOException,
    REMBOHistoricalDataError,
    REMBOOptimizerDependencyError,
)


class _HammingKernel(gpytorch.kernels.Kernel):
    """Paper categorical covariance on already discretized ambient rows."""

    has_lengthscale = True

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, *, diag: bool = False, **params: Any
    ) -> torch.Tensor:
        """Evaluate Hamming-distance covariance between ambient row batches.

        Parameters
        ----------
        x1, x2:
            Ambient tensors whose final dimension is ``D_env``.
        diag:
            Return only the covariance diagonal when ``True``.

        Returns
        -------
        torch.Tensor
            Pairwise covariance, or its diagonal, with leading batch dimensions preserved.
        """

        del params
        distance = (x1.unsqueeze(-2) != x2.unsqueeze(-3)).to(x1.dtype).sum(dim=-1)
        covariance = torch.exp(-0.5 * self.lengthscale.squeeze().square() * distance.square())
        return torch.diagonal(covariance, dim1=-2, dim2=-1) if diag else covariance


class _MappedAcquisition(torch.nn.Module):
    """Evaluate a high-dimensional kernel acquisition over REMBO y-coordinates."""

    def __init__(self, acquisition: Any, algorithm: REMBO) -> None:
        """Bind an ambient-coordinate acquisition to a REMBO embedding map."""

        super().__init__()
        self.acquisition = acquisition
        self.algorithm = algorithm

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """Evaluate the wrapped acquisition after mapping into ``D_env``.

        Parameters
        ----------
        Y:
            Embedded-coordinate candidates with trailing dimension
            ``D_model``.

        Returns
        -------
        torch.Tensor
            Acquisition values returned for the projected ambient actions.
        """

        return self.acquisition(self.algorithm._project_to_X(Y))

    def set_X_pending(self, X_pending: torch.Tensor | None = None) -> None:
        """Forward pending embedded points after applying the REMBO projection.

        Parameters
        ----------
        X_pending:
            Pending candidates in ``D_model`` coordinates, or ``None`` to
            clear pending state.
        """

        setter = getattr(self.acquisition, "set_X_pending", None)
        if callable(setter):
            setter(None if X_pending is None else self.algorithm._project_to_X(X_pending))


@dataclass(frozen=True, slots=True)
class REMBOConfig:
    """Complete REMBO paper and controlled-extension configuration.

    Parameters
    ----------
    embedding_dimension, Y_lower, Y_upper:
        Reduced dimension and optional explicit search-box endpoints.
    kernel_type, custom_kernel:
        Paper kernel coordinate system or qualified custom kernel.
    acquisition, initial_points, batch_size:
        EI/UCB rule, optional initial-design count, and sequential or
        documented BayOPy q-extension size.
    initial_points_sampling_method:
        ``"paper"``, ``"sobol"``, or ``"uniform"`` initial-design sampling.
    lengthscale_lower, lengthscale_upper, initial_lengthscale:
        Algorithm-3 lengthscale interval and initial value.
    t_sigma, refit_interval, exploitation_count_limit, upper_bound_multiplier:
        Source exploitation/refit and interval-shrink constants.
    seed:
        Embedding, optimizer, and acquisition seed.
    """

    embedding_dimension: int
    finite_candidate_mode: Literal["project", "exact"] = field(default="project", kw_only=True)
    Y_lower: Any | None = None
    Y_upper: Any | None = None
    kernel_type: Literal["low_dimensional", "high_dimensional", "categorical"] = "low_dimensional"
    custom_kernel: gpytorch.kernels.Kernel | None = None
    acquisition: Literal["EI", "UCB"] = "EI"
    initial_points: int | None = None
    initial_points_sampling_method: Literal["paper", "sobol", "uniform"] = "paper"
    batch_size: int | None = 1
    q_batch: int | None = None
    lengthscale_lower: float = 0.01
    lengthscale_upper: float = 50.0
    initial_lengthscale: float | None = None
    t_sigma: float = 0.002
    refit_interval: int = 20
    exploitation_count_limit: int = 5
    upper_bound_multiplier: float = 0.9
    seed: int = 1


@dataclass(slots=True)
class REMBOState:
    """Mutable REMBO embedding, data, and Algorithm 3 state.

    Attributes
    ----------
    A:
        Immutable dense map from ``D_model`` into ``D_env``.
    Y_train, X_train, function_values:
        Aligned embedded, physical, and learner-visible response history.
    lengthscale, lengthscale_upper, C, iteration, last_refit_iteration:
        Complete source Algorithm-3 adaptation state.
    last_ask_y, last_ask_x:
        Pending reduced and physical proposal coordinates.
    """

    A: torch.Tensor
    Y_train: torch.Tensor
    X_train: torch.Tensor
    function_values: torch.Tensor
    lengthscale: float
    lengthscale_upper: float
    C: int
    iteration: int
    last_refit_iteration: int | None
    last_ask_y: torch.Tensor | None
    last_ask_x: torch.Tensor | None


class REMBO(Algorithm):
    """Optimize a bounded objective through one immutable dense embedding.

    Examples
    --------
    >>> import bayopy
    >>> bounds = bayopy.Bounds(lower=[-1.0] * 6, upper=[1.0] * 6)
    >>> algorithm = bayopy.REMBO(
    ...     bounds=bounds,
    ...     config=bayopy.REMBOConfig(embedding_dimension=2, acquisition="EI", seed=7),
    ... )
    """

    ACQUISITION_POLICY = "configurable EI or UCB in the random embedding"
    CHECKPOINT_ADDED_CONFIG_DEFAULTS: ClassVar[Mapping[str, Any]] = {
        "initial_points": None,
        "initial_points_sampling_method": "paper",
    }

    EXCEPTION_CLASS = REMBOException
    native_environments = frozenset({"stochastic"})
    native_batching = True
    SOURCE = SourceRecord(
        paper_title="Bayesian Optimization in a Billion Dimensions via Random Embeddings",
        paper_year=2016,
        paper_url="https://doi.org/10.1613/jair.4806",
        repository_url="https://github.com/ziyuw/rembo",
        repository_status="author",
        notes="Clean-room PyTorch implementation; q>1 and custom kernels are explicit extensions.",
    )

    def __init__(
        self,
        *,
        bounds: Bounds,
        config: REMBOConfig,
        maximize: bool = True,
        dtype: torch.dtype | str = torch.float32,
        device: torch.device | str = "cpu",
        plot: bool = False,
        verbose: bool = False,
        verbose_setting: bool = False,
        verbose_iter_by_iter: bool = False,
    ) -> None:
        """Initialize REMBO and draw its immutable seeded embedding.

        Parameters
        ----------
        bounds:
            Physical action bounds in ``D_env`` coordinates.
        config:
            Embedding, kernel, acquisition, and Algorithm 3 settings.
        maximize:
            Optimize larger observations when ``True``.
        dtype, device:
            Tensor precision and execution device.
        plot, verbose, verbose_setting, verbose_iter_by_iter:
            Presentation flags inherited from :class:`Algorithm`.

        Raises
        ------
        AlgorithmConfigurationError
            If dimensions, regions, kernel settings, or update constants are invalid.
        """

        self.config = config
        self._validate_config(config, ambient_dimension=bounds.dimension)
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
        self.Y_lower, self.Y_upper = self._make_Y(
            config.embedding_dimension, config.Y_lower, config.Y_upper
        )
        A = self._sample_embedding(self.action_dimension, config.embedding_dimension)
        lengthscale = (
            math.sqrt(config.lengthscale_lower * config.lengthscale_upper)
            if config.initial_lengthscale is None
            else float(config.initial_lengthscale)
        )
        self.state = REMBOState(
            A=A,
            Y_train=torch.empty(
                (0, config.embedding_dimension), dtype=self.dtype, device=self.device
            ),
            X_train=torch.empty((0, self.action_dimension), dtype=self.dtype, device=self.device),
            function_values=torch.empty((0,), dtype=self.dtype, device=self.device),
            lengthscale=lengthscale,
            lengthscale_upper=config.lengthscale_upper,
            C=0,
            iteration=0,
            last_refit_iteration=None,
            last_ask_y=None,
            last_ask_x=None,
        )
        self.model_input_dimension = (
            config.embedding_dimension
            if config.kernel_type == "low_dimensional"
            else self.action_dimension
        )
        self.gp = self._build_gp()
        if self.request_size > 1 or self.acquisition_sampling_method != "paper":
            reason = (
                "bayopy.joint_q_acquisition"
                if self.uses_joint_batch
                else "bayopy.independent_acquisition_batch"
            )
            self.invalidate_source_guarantees(reason)
        if config.custom_kernel is not None:
            self.invalidate_source_guarantees("bayopy.custom_rembo_kernel")

    @staticmethod
    def _validate_config(config: REMBOConfig, *, ambient_dimension: int) -> None:
        """Validate immutable REMBO configuration against ``D_env``."""

        if not isinstance(config, REMBOConfig):
            raise AlgorithmConfigurationError("config must be REMBOConfig.")
        if config.finite_candidate_mode not in {"project", "exact"}:
            raise AlgorithmConfigurationError("finite_candidate_mode must be project or exact.")
        resolve_batch_configuration(batch_size=config.batch_size, q_batch=config.q_batch)
        if (
            isinstance(config.embedding_dimension, bool)
            or not isinstance(config.embedding_dimension, int)
            or not 1 <= config.embedding_dimension <= ambient_dimension
        ):
            raise AlgorithmConfigurationError("REMBO requires 1 <= embedding_dimension <= D.")
        if (config.Y_lower is None) != (config.Y_upper is None):
            raise AlgorithmConfigurationError("Y_lower and Y_upper must be supplied together.")
        if config.kernel_type not in {"low_dimensional", "high_dimensional", "categorical"}:
            raise AlgorithmConfigurationError("Unknown REMBO kernel_type.")
        if config.custom_kernel is not None and config.kernel_type == "categorical":
            raise AlgorithmConfigurationError(
                "custom_kernel cannot replace the categorical kernel."
            )
        if config.acquisition not in {"EI", "UCB"}:
            raise AlgorithmConfigurationError("REMBO acquisition must be EI or UCB.")
        if config.initial_points is not None and (
            isinstance(config.initial_points, bool)
            or not isinstance(config.initial_points, int)
            or config.initial_points < 1
        ):
            raise AlgorithmConfigurationError("initial_points must be positive or None.")
        for name in ("refit_interval", "exploitation_count_limit"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise AlgorithmConfigurationError(f"{name} must be a positive integer.")
        numeric = {
            "lengthscale_lower": config.lengthscale_lower,
            "lengthscale_upper": config.lengthscale_upper,
            "t_sigma": config.t_sigma,
            "upper_bound_multiplier": config.upper_bound_multiplier,
        }
        if any(not math.isfinite(float(value)) or value <= 0 for value in numeric.values()):
            raise AlgorithmConfigurationError(
                "REMBO scale/rule parameters must be positive and finite."
            )
        if config.lengthscale_upper <= config.lengthscale_lower:
            raise AlgorithmConfigurationError("lengthscale_upper must exceed lengthscale_lower.")
        if config.initial_lengthscale is not None and not (
            config.lengthscale_lower <= config.initial_lengthscale <= config.lengthscale_upper
        ):
            raise AlgorithmConfigurationError("initial_lengthscale must lie in [L, U].")

    def _sample_embedding(self, D: int, d: int) -> torch.Tensor:
        """Draw the immutable ``D x d`` standard-normal matrix A."""

        return torch.randn((D, d), dtype=self.dtype, device=self.device, generator=self.generator)

    def _make_Y(
        self, d: int, lower: Any | None, upper: Any | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct and validate the bounded low-dimensional paper region."""

        if lower is None:
            radius = math.sqrt(d)
            low = torch.full((d,), -radius, dtype=self.dtype, device=self.device)
            high = torch.full((d,), radius, dtype=self.dtype, device=self.device)
        else:
            low = torch.as_tensor(lower, dtype=self.dtype, device=self.device)
            high = torch.as_tensor(upper, dtype=self.dtype, device=self.device)
        if (
            low.shape != (d,)
            or high.shape != (d,)
            or not torch.isfinite(low).all()
            or not torch.isfinite(high).all()
        ):
            raise AlgorithmConfigurationError(f"Y bounds must be finite vectors of shape ({d},).")
        if not torch.all(low < high):
            raise AlgorithmConfigurationError("Every Y lower bound must be smaller than Y upper.")
        return low.detach().clone(), high.detach().clone()

    def _to_unit_box(self, actions: torch.Tensor) -> torch.Tensor:
        """Map physical actions to the symmetric REMBO ambient box."""

        return to_symmetric_box(actions, self.bounds.to(dtype=self.dtype, device=self.device))

    def _from_unit_box(self, X: torch.Tensor) -> torch.Tensor:
        """Map symmetric REMBO points to physical environment bounds."""

        return from_symmetric_box(X, self.bounds.to(dtype=self.dtype, device=self.device))

    def _project_to_X(self, Y: torch.Tensor) -> torch.Tensor:
        """Compute ``p_X(Ay)`` without changing the embedded coordinate."""

        return (Y @ self.state.A.T).clamp(-1.0, 1.0)

    def _map_y_to_action(self, Y: torch.Tensor) -> torch.Tensor:
        """Map one or more y rows into physical actions."""

        return self._from_unit_box(self._project_to_X(Y))

    def _kernel_coordinates(self, Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Return the paper-selected coordinates consumed by the GP kernel."""

        return Y if self.config.kernel_type == "low_dimensional" else X

    def _build_gp(self) -> GaussianProcess:
        """Build the configured GP in embedded or projected ambient coordinates.

        Returns
        -------
        GaussianProcess
            An unfitted GP whose input dimension matches the selected kernel coordinates.

        Notes
        -----
        The warm-start lengthscale is projected into the strict interior of a
        shrunken Algorithm 3 interval before constructing the GPyTorch kernel.
        """

        if self.config.custom_kernel is not None:
            kernel: Any = self.config.custom_kernel
        elif self.config.kernel_type == "categorical":
            kernel = gpytorch.kernels.ScaleKernel(_HammingKernel())
        else:
            lower = self.config.lengthscale_lower
            upper = max(
                self.state.lengthscale_upper,
                lower * (1.0 + 1e-6),
            )
            # GPyTorch's interval transform is defined on the open interval.
            # A paper update can move the upper constraint below the previous
            # fitted lengthscale, so project the numerical warm start into the
            # new interval before constructing the replacement model.
            if not lower < self.state.lengthscale < upper:
                self.state.lengthscale = 0.5 * (lower + upper)
            base_kernel = gpytorch.kernels.RBFKernel(
                lengthscale_constraint=gpytorch.constraints.Interval(
                    lower,
                    upper,
                )
            )
            base_kernel.initialize(lengthscale=self.state.lengthscale)
            kernel = gpytorch.kernels.ScaleKernel(base_kernel)
        return GaussianProcess(
            GaussianProcessConfig(
                kernel=kernel,
                mean="constant",
                noise_mode="learned",
                fit_method="mle",
                dtype=self.dtype,
                device=self.device,
            ),
            input_dimension=(
                self.config.embedding_dimension
                if self.config.kernel_type == "low_dimensional"
                else self.action_dimension
            ),
        )

    def _finite_coordinates(self, actions: torch.Tensor, metadata: Mapping[str, Any], *, historical: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress measured rows without replacing actions or imputing rewards."""
        if self.config.finite_candidate_mode == "exact":
            Y = self._embedded_metadata(metadata, actions.shape[0])
            return Y, self._verify_representation(actions, Y, historical=historical)
        X = to_symmetric_box(actions, self.bounds.to(dtype=self.dtype, device=self.device))
        return X @ torch.linalg.pinv(self.state.A).T, X

    def _embedded_metadata(self, source: Mapping[str, Any], rows: int) -> torch.Tensor:
        """Read exact ``(rows, D_model)`` REMBO coordinates from public metadata.

        Raises
        ------
        REMBODataCompatibilityError
            If coordinates are absent or have an incompatible shape.
        """

        nested = source.get("initial_metadata")
        if isinstance(nested, Mapping):
            source = {**nested, **source}
        for name in ("rembo_Y", "embedded_coordinates", "model_inputs"):
            if name in source:
                Y = torch.as_tensor(source[name], dtype=self.dtype, device=self.device)
                if Y.shape != (rows, self.config.embedding_dimension):
                    raise REMBODataCompatibilityError(
                        f"{name} must have shape ({rows}, {self.config.embedding_dimension}).",
                        algorithm=self,
                    )
                return Y
        raise REMBODataCompatibilityError(
            "Finite/historical REMBO actions require exact stored y-coordinates; no pseudoinverse is used.",
            algorithm=self,
        )

    def _verify_representation(
        self, actions: torch.Tensor, Y: torch.Tensor, *, historical: bool
    ) -> torch.Tensor:
        """Verify that stored ``Y`` exactly reproduces physical actions.

        Returns
        -------
        torch.Tensor
            Projected symmetric-box coordinates with shape ``(N, D_env)``.

        Raises
        ------
        REMBOHistoricalDataError
            If historical rows are not represented exactly.
        REMBODataCompatibilityError
            If current finite candidates are not represented exactly.
        """

        expected = self._map_y_to_action(Y)
        if not torch.isclose(expected, actions, rtol=0.0, atol=1e-8).all():
            error = REMBOHistoricalDataError if historical else REMBODataCompatibilityError
            raise error(
                "Stored REMBO y-coordinates do not reproduce the ambient actions.", algorithm=self
            )
        return self._project_to_X(Y)

    def _initialize_from_environment_impl(self, data: Any) -> None:
        """Encode learner-visible history according to the finite-data policy."""

        rows = int(data.observations.shape[0])
        if rows == 0:
            return
        metadata = data.metadata if isinstance(data.metadata, Mapping) else {}
        actions = torch.as_tensor(data.actions, dtype=self.dtype, device=self.device)
        Y, X = self._finite_coordinates(actions, metadata, historical=True)
        self.state.Y_train = Y.detach().clone()
        self.state.X_train = X.detach().clone()
        self.state.function_values = (
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
        """Construct the one-to-one finite table in configured kernel coordinates."""

        Y, X = self._finite_coordinates(candidate_actions, round_info)
        return DiscreteAcquisitionCandidates(
            environment_actions=candidate_actions.detach().clone(),
            model_inputs=self._kernel_coordinates(Y, X).detach().clone(),
            dataset_indices=candidate_dataset_indices.detach().clone(),
        )

    def _set_gp_data(self) -> None:
        """Synchronize non-empty REMBO training tensors with the GP wrapper."""

        if self.state.function_values.numel():
            self.gp.set_data(
                self._kernel_coordinates(self.state.Y_train, self.state.X_train),
                to_quality(self.state.function_values, maximize=self.maximize),
            )

    def _fit_embedded_gp(self) -> None:
        """Fit or warm-start the GP according to the Algorithm 3 refit cycle.

        Raises
        ------
        REMBOAcquisitionError
            If GP construction, warm-start restoration, or fitting fails.
        """

        previous_state = (
            None
            if self.gp.model is None or not self.gp.is_fit
            else {
                name: value.detach().clone() for name, value in self.gp.model.state_dict().items()
            }
        )
        self._set_gp_data()
        if previous_state is not None and self.gp.model is not None:
            current = self.gp.model.state_dict()
            for name, value in previous_state.items():
                if name in current and current[name].shape == value.shape:
                    current[name] = value.to(
                        dtype=current[name].dtype,
                        device=current[name].device,
                    )
            self.gp.model.load_state_dict(current, strict=False)
        should_refit = (
            self.state.last_refit_iteration is None
            or self.state.iteration - self.state.last_refit_iteration >= self.config.refit_interval
        )
        try:
            if should_refit:
                self.gp.fit(maxiter=50)
                self.state.last_refit_iteration = self.state.iteration
                if (
                    self.config.custom_kernel is None
                    and self.config.kernel_type != "categorical"
                    and self.gp.model is not None
                ):
                    covariance = self.gp.model.covar_module
                    base_kernel = getattr(covariance, "base_kernel", covariance)
                    self.state.lengthscale = float(base_kernel.lengthscale.detach().mean().item())
            elif self.gp.model is not None and self.gp.likelihood is not None:
                self.gp.model.eval()
                self.gp.likelihood.eval()
                self.gp.is_fit = True
        except Exception as error:
            raise REMBOAcquisitionError(
                f"REMBO GP fitting failed: {error}", algorithm=self
            ) from error

    def _build_acquisition(self) -> Any:
        """Build EI or UCB in the kernel coordinate system for the current batch."""

        quality = self.state.function_values if self.maximize else -self.state.function_values
        acquisition = build_acquisition(
            model=self.gp,
            config=AcquisitionConfig(
                name=self.config.acquisition,
                beta=2.0,
                seed=self.config.seed + self.iteration,
            ),
            batch_size=self.batch_size,
            q_batch=self.q_batch,
            best_f=quality.max() if self.config.acquisition == "EI" else None,
        )
        return (
            acquisition
            if self.config.kernel_type == "low_dimensional"
            else _MappedAcquisition(acquisition, self)
        )

    def _maximize_paper_acquisition(self, acquisition: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Maximize an acquisition over the bounded ``D_model`` search region.

        Sequential paper mode compares SciPy DIRECT and CMA-ES. Documented
        batch mode delegates to the joint q-acquisition optimizer.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Selected embedded points with shape ``(q, D_model)`` and score.

        Raises
        ------
        REMBOOptimizerDependencyError
            If a sequential paper optimizer dependency is unavailable.
        """

        bounds = torch.stack((self.Y_lower, self.Y_upper))
        if self.request_size > 1:
            points, acquisition_values = optimize_acquisition_continuous(
                acquisition=acquisition,
                bounds=bounds,
                batch_size=self.batch_size,
                q_batch=self.q_batch,
                num_restarts=10,
                raw_samples=256,
                sampling_method=self.acquisition_sampling_method,
                sampling_generator=self.generator,
                sampling_seed=self.config.seed + self.iteration,
            )
            return points, acquisition_values.reshape(-1).sum()
        try:
            import cma
            from scipy.optimize import direct
        except ImportError as error:
            raise REMBOOptimizerDependencyError(
                "Sequential paper-mode REMBO requires scipy and the 'optimizers' cma extra.",
                algorithm=self,
            ) from error
        except Exception as error:
            raise REMBOAcquisitionError(
                f"The sequential REMBO optimizer backend could not be initialized: {error}",
                algorithm=self,
            ) from error

        def objective(values: Any) -> float:
            """Return the negated acquisition value for external minimizers."""

            Y = torch.as_tensor(values, dtype=self.dtype, device=self.device).reshape(1, 1, -1)
            with torch.no_grad():
                return -float(acquisition(Y).reshape(-1)[0].item())

        scipy_bounds = list(
            zip(self.Y_lower.cpu().tolist(), self.Y_upper.cpu().tolist(), strict=True)
        )
        try:
            direct_result = direct(objective, scipy_bounds, maxfun=256, locally_biased=False)
        except Exception as error:
            raise REMBOAcquisitionError(
                f"DIRECT acquisition optimization failed: {error}", algorithm=self
            ) from error
        midpoint = ((self.Y_lower + self.Y_upper) / 2).cpu().tolist()
        sigma = float((self.Y_upper - self.Y_lower).min().item() / 4)
        try:
            cma_result = cma.fmin(
                objective,
                midpoint,
                max(sigma, 1e-6),
                options={
                    "bounds": [self.Y_lower.cpu().tolist(), self.Y_upper.cpu().tolist()],
                    "maxfevals": 256,
                    "verbose": -9,
                    "seed": self.config.seed + self.iteration,
                },
            )
            cma_candidate = cma_result[0]
        except Exception as error:
            raise REMBOAcquisitionError(
                f"CMA-ES acquisition optimization failed: {error}", algorithm=self
            ) from error
        candidates = [direct_result.x, cma_candidate]
        values = [-objective(candidate) for candidate in candidates]
        position = 0 if values[0] >= values[1] else 1
        Y = torch.as_tensor(candidates[position], dtype=self.dtype, device=self.device).reshape(
            1, -1
        )
        return Y, torch.tensor(values[position], dtype=self.dtype, device=self.device)

    def _initial_design_size(self) -> int:
        """Return the paper initial-design size in embedded coordinates."""

        return self.config.initial_points or max(2 * self.config.embedding_dimension, 4)

    def _ask_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Select a continuous or finite REMBO action batch."""

        finite = "candidate_actions" in round_info
        table: DiscreteAcquisitionCandidates | None = None
        finite_Y: torch.Tensor | None = None
        if finite:
            table = self.prepare_discrete_candidates(
                candidate_actions=round_info["candidate_actions"],
                candidate_dataset_indices=round_info["candidate_dataset_indices"],
                round_info=round_info,
            )
            finite_Y, _ = self._finite_coordinates(table.environment_actions, round_info)
        if self.state.function_values.numel() < self._initial_design_size():
            if table is not None:
                order = torch.argsort(table.dataset_indices)[: self.request_size]
                actions = table.environment_actions[order]
                assert finite_Y is not None
                Y = finite_Y[order]
                X = to_symmetric_box(actions, self.bounds.to(dtype=self.dtype, device=self.device))
            else:
                region = Bounds(self.Y_lower, self.Y_upper)
                design = self._sample_design(
                    region,
                    self.request_size,
                    phase="initial",
                    seed=self.config.seed + self.iteration,
                )
                Y = design
                X = self._project_to_X(Y)
                actions = self._from_unit_box(X)
            value = None
        else:
            self._fit_embedded_gp()
            acquisition = self._build_acquisition()
            if table is not None:
                # Finite candidate model inputs are already expressed in the
                # selected kernel coordinates.  The mapping wrapper is needed
                # only while a continuous optimizer proposes embedded points.
                if isinstance(acquisition, _MappedAcquisition):
                    acquisition = acquisition.acquisition
                result = optimize_acquisition_discrete(
                    acquisition=acquisition,
                    candidates=table,
                    batch_size=self.batch_size,
                    q_batch=self.q_batch,
                )
                actions = result.environment_actions
                assert finite_Y is not None
                Y = finite_Y[result.selected_rows]
                X = to_symmetric_box(actions, self.bounds.to(dtype=self.dtype, device=self.device))
                value = float(result.acquisition_values.sum().item())
            else:
                Y, acquisition_value = self._maximize_paper_acquisition(acquisition)
                X = self._project_to_X(Y)
                actions = self._from_unit_box(X)
                value = float(acquisition_value.reshape(-1).sum().item())
            self._last_acquisition_value = value
            _, posterior_std = self.gp.predict(self._kernel_coordinates(Y, X))
            if bool((posterior_std < self.config.t_sigma).all()):
                self.state.C += 1
            else:
                self.state.C = 0
        self.state.last_ask_y = Y.detach().clone()
        self.state.last_ask_x = X.detach().clone()
        self._record_last_ask(
            actions=actions,
            model_inputs=self._kernel_coordinates(Y, X),
            round_info=round_info,
            algorithm_metadata={"Y": Y, "X_symmetric": X, "acquisition_value": value},
        )
        return actions

    def _tell_impl(
        self,
        *,
        action: torch.Tensor,
        feedback: Mapping[str, Any],
        round_info: Mapping[str, Any],
    ) -> None:
        """Append evaluated data and apply the Algorithm 3 lengthscale update."""

        del action, round_info
        if self.state.last_ask_y is None or self.state.last_ask_x is None:
            raise REMBODataCompatibilityError(
                "REMBO tell is missing embedded ask state.", algorithm=self
            )
        self.state.Y_train = torch.cat((self.state.Y_train, self.state.last_ask_y), dim=0)
        self.state.X_train = torch.cat((self.state.X_train, self.state.last_ask_x), dim=0)
        self.state.function_values = torch.cat(
            (self.state.function_values, feedback["observation"]), dim=0
        )
        self.state.iteration += 1
        if self.config.exploitation_count_limit <= self.state.C:
            self.state.lengthscale_upper = max(
                self.config.upper_bound_multiplier * self.state.lengthscale,
                self.config.lengthscale_lower * (1.0 + 1e-6),
            )
            self.state.C = 0
            self.state.last_refit_iteration = None
            self.gp = self._build_gp()
        self.state.last_ask_y = None
        self.state.last_ask_x = None

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Return detached embedding, lengthscale, and training diagnostics."""

        return {
            "embedding_dimension": self.config.embedding_dimension,
            "training_rows": int(self.state.function_values.numel()),
            "lengthscale": self.state.lengthscale,
            "lengthscale_upper": self.state.lengthscale_upper,
            "exploitation_counter": self.state.C,
            "embedding_matrix": self.state.A,
        }

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize REMBO observations and search progress, excluding GP state."""

        return {
            "config": self._config_repr(),
            "A": self.state.A,
            "Y_train": self.state.Y_train,
            "X_train": self.state.X_train,
            "function_values": self.state.function_values,
            "lengthscale": self.state.lengthscale,
            "lengthscale_upper": self.state.lengthscale_upper,
            "C": self.state.C,
            "method_iteration": self.state.iteration,
            "last_refit_iteration": self.state.last_refit_iteration,
            "last_ask_y": self.state.last_ask_y,
            "last_ask_x": self.state.last_ask_x,
        }

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Validate required REMBO checkpoint fields and embedding dimensions."""

        required = {
            "config",
            "A",
            "Y_train",
            "X_train",
            "function_values",
            "lengthscale",
            "lengthscale_upper",
            "C",
            "method_iteration",
            "last_refit_iteration",
            "last_ask_y",
            "last_ask_x",
        }
        if missing := required - set(state):
            raise REMBOCheckpointError(f"REMBO state is missing {sorted(missing)}.", algorithm=self)
        A = torch.as_tensor(state["A"])
        if A.shape != (self.action_dimension, self.config.embedding_dimension):
            raise REMBOCheckpointError("REMBO state has an incompatible A matrix.", algorithm=self)
        del strict

    def _config_repr(self) -> str:
        """Return a stable fingerprint without deep-copying custom kernels."""

        values = {
            field: getattr(self.config, field)
            for field in self.config.__dataclass_fields__
            if field != "custom_kernel"
        }
        values["custom_kernel"] = repr(self.config.custom_kernel)
        return repr(values)

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Restore REMBO progress and construct a fresh unfitted GP."""

        del strict
        for name in ("A", "Y_train", "X_train", "function_values"):
            setattr(
                self.state,
                name,
                torch.as_tensor(state[name], dtype=self.dtype, device=self.device).detach().clone(),
            )
        self.state.lengthscale = float(state["lengthscale"])
        self.state.lengthscale_upper = float(state["lengthscale_upper"])
        self.state.C = int(state["C"])
        self.state.iteration = int(state["method_iteration"])
        # Force a fresh fit from restored observations on the next acquisition.
        self.state.last_refit_iteration = None
        self.state.last_ask_y = self._optional_tensor(state["last_ask_y"])
        self.state.last_ask_x = self._optional_tensor(state["last_ask_x"])
        self.gp = self._build_gp()

    def _optional_tensor(self, value: Any) -> torch.Tensor | None:
        """Convert an optional checkpoint value to an owned algorithm tensor."""

        return (
            None
            if value is None
            else torch.as_tensor(value, dtype=self.dtype, device=self.device).detach().clone()
        )

    def _reset_impl(self) -> None:
        """Clear observations and redraw the seeded REMBO embedding."""

        self.gp.reset()
        self.state = REMBOState(
            A=self._sample_embedding(self.action_dimension, self.config.embedding_dimension),
            Y_train=torch.empty(
                (0, self.config.embedding_dimension), dtype=self.dtype, device=self.device
            ),
            X_train=torch.empty((0, self.action_dimension), dtype=self.dtype, device=self.device),
            function_values=torch.empty((0,), dtype=self.dtype, device=self.device),
            lengthscale=(
                math.sqrt(self.config.lengthscale_lower * self.config.lengthscale_upper)
                if self.config.initial_lengthscale is None
                else float(self.config.initial_lengthscale)
            ),
            lengthscale_upper=self.config.lengthscale_upper,
            C=0,
            iteration=0,
            last_refit_iteration=None,
            last_ask_y=None,
            last_ask_x=None,
        )


__all__ = ["REMBO", "REMBOConfig", "REMBOState"]
