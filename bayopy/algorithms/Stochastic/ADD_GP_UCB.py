"""Additive Gaussian-process UCB with one shared covariance model."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import gpytorch
import torch

from bayopy.algorithms.algorithm import (
    Algorithm,
    AlgorithmConfigurationError,
    resolve_batch_configuration,
)
from bayopy.algorithms.helper import SourceRecord, to_quality
from bayopy.models.Acquisitions import DiscreteAcquisitionCandidates
from bayopy.models.GaussianProcess import GaussianProcess, GaussianProcessConfig
from bayopy.models.utils import Bounds, normalize, unnormalize

from .Exception.exception_ADD_GP_UCB import (
    ADD_GP_UCBAcquisitionError,
    ADD_GP_UCBCheckpointError,
    ADD_GP_UCBDecompositionError,
    ADD_GP_UCBException,
    ADD_GP_UCBGroupError,
    ADD_GP_UCBKernelError,
)


@dataclass(frozen=True, slots=True)
class AddGPUCBConfig:
    """Configuration of known or learned additive coordinate groups.

    Parameters
    ----------
    groups:
        Optional exact partition of ``D_env`` coordinate indices.
    maximum_group_size, learn_decomposition, decomposition_candidates, refresh_cycle:
        Candidate-partition limits and refresh policy when groups are learned.
    beta:
        Paper schedule, scalar, or callable confidence multiplier.
    initial_points:
        Optional initial-design size; ``None`` selects the documented default.
    component_optimizer:
        Optional callable optimizer at the additive-group boundary.
    batch_size:
        Synchronous size; version 1 requires one.
    seed:
        Algorithm-owned random seed.
    """

    groups: tuple[tuple[int, ...], ...] | None = None
    maximum_group_size: int = 5
    learn_decomposition: bool = False
    decomposition_candidates: int | None = None
    refresh_cycle: int = 25
    beta: Callable[[int, int], float] | float | Literal["paper"] = "paper"
    initial_points: int | None = None
    component_optimizer: Any | None = None
    batch_size: int | None = 1
    q_batch: int | None = None
    seed: int = 1


@dataclass(slots=True)
class AddGPUCBState:
    """Mutable additive partition, shared data, and decomposition diagnostics.

    Attributes
    ----------
    groups:
        Current exact coordinate partition.
    X_train, y_train:
        Shared ``(N, D_env)`` GP inputs and ``(N,)`` responses.
    component_kernels, additive_gp:
        Fitted component kernels and the single shared additive GP.
    last_decomposition_refresh, candidate_decomposition_scores:
        Refresh generation and most recent candidate scores.
    last_ask_action:
        Pending complete ``D_env`` action, if any.
    """

    groups: tuple[tuple[int, ...], ...]
    X_train: torch.Tensor
    y_train: torch.Tensor
    component_kernels: tuple[gpytorch.kernels.ScaleKernel, ...]
    additive_gp: GaussianProcess
    last_decomposition_refresh: int
    candidate_decomposition_scores: list[float]
    last_ask_action: torch.Tensor | None


class ADD_GP_UCB(Algorithm):
    """Optimize group UCBs derived from a single additive GP covariance.

    Examples
    --------
    >>> import bayopy
    >>> bounds = bayopy.Bounds(lower=[-1.0] * 4, upper=[1.0] * 4)
    >>> algorithm = bayopy.ADD_GP_UCB(
    ...     bounds=bounds,
    ...     config=bayopy.AddGPUCBConfig(groups=((0, 1), (2, 3)), seed=7),
    ... )
    """

    ACQUISITION_POLICY = "additive GP-UCB"

    EXCEPTION_CLASS = ADD_GP_UCBException
    native_batching = False
    native_environments = frozenset({"stochastic"})
    SOURCE = SourceRecord(
        paper_title="High Dimensional Bayesian Optimisation and Bandits via Additive Models",
        paper_year=2015,
        paper_url="https://proceedings.mlr.press/v37/kandasamy15.html",
        repository_url="https://github.com/kirthevasank/add-gp-bandits",
        repository_status="author",
        notes="One shared additive covariance; finite rows are scored without synthesis.",
    )

    def __init__(
        self,
        *,
        bounds: Bounds,
        config: AddGPUCBConfig,
        maximize: bool = True,
        dtype: torch.dtype | str = torch.float64,
        device: torch.device | str = "cpu",
        plot: bool = False,
        verbose: bool = False,
        verbose_setting: bool = False,
        verbose_iter_by_iter: bool = False,
    ) -> None:
        """Initialize a shared additive GP and complete coordinate partition.

        Parameters
        ----------
        bounds:
            Physical action bounds in ``D_env`` coordinates.
        config:
            Group partition, decomposition-learning, beta, and optimizer settings.
        maximize:
            Optimize larger observations when ``True``.
        dtype, device:
            Tensor precision and execution device.
        plot, verbose, verbose_setting, verbose_iter_by_iter:
            Presentation flags inherited from :class:`Algorithm`.

        Raises
        ------
        AlgorithmConfigurationError
            If sequential, beta, or decomposition settings are invalid.
        ADD_GP_UCBGroupError
            If explicit groups do not partition ``D_env`` exactly once.
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
        groups = self._resolve_initial_groups()
        component_kernels, gp = self._build_additive_kernel(groups)
        self.state = AddGPUCBState(
            groups=groups,
            X_train=torch.empty((0, self.action_dimension), dtype=self.dtype, device=self.device),
            y_train=torch.empty((0,), dtype=self.dtype, device=self.device),
            component_kernels=component_kernels,
            additive_gp=gp,
            last_decomposition_refresh=0,
            candidate_decomposition_scores=[],
            last_ask_action=None,
        )
        self.model_input_dimension = self.action_dimension
        self._last_component_values: tuple[float, ...] = ()

    @staticmethod
    def _validate_config(config: AddGPUCBConfig, D: int) -> None:
        """Validate immutable additive-GP settings against dimension ``D``."""

        if not isinstance(config, AddGPUCBConfig):
            raise AlgorithmConfigurationError("config must be AddGPUCBConfig.")
        _, q_batch = resolve_batch_configuration(
            batch_size=config.batch_size, q_batch=config.q_batch
        )
        if q_batch is not None and q_batch > 1:
            raise AlgorithmConfigurationError(
                "ADD-GP-UCB does not define a joint q-acquisition; use batch_size for "
                "repeated independent additive-UCB queries."
            )
        for name in ("maximum_group_size", "refresh_cycle"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise AlgorithmConfigurationError(f"{name} must be positive.")
        if config.decomposition_candidates is not None and (
            isinstance(config.decomposition_candidates, bool)
            or not isinstance(config.decomposition_candidates, int)
            or config.decomposition_candidates < 1
        ):
            raise AlgorithmConfigurationError("decomposition_candidates must be positive or None.")
        if config.initial_points is not None and (
            isinstance(config.initial_points, bool)
            or not isinstance(config.initial_points, int)
            or config.initial_points < 1
        ):
            raise AlgorithmConfigurationError("initial_points must be positive or None.")
        if config.groups is not None:
            ADD_GP_UCB._validate_groups_static(config.groups, D)
        if isinstance(config.beta, str) and config.beta != "paper":
            raise AlgorithmConfigurationError("beta string must be 'paper'.")
        if (
            not isinstance(config.beta, str)
            and not callable(config.beta)
            and (not math.isfinite(float(config.beta)) or float(config.beta) < 0)
        ):
            raise AlgorithmConfigurationError("beta must be finite and non-negative.")

    @staticmethod
    def _validate_groups_static(groups: tuple[tuple[int, ...], ...], D: int) -> None:
        """Require non-empty integer groups that partition ``range(D)`` exactly."""

        if not groups or any(not group for group in groups):
            raise ADD_GP_UCBGroupError("Additive groups must be non-empty.")
        flattened = [coordinate for group in groups for coordinate in group]
        if any(isinstance(index, bool) or not isinstance(index, int) for index in flattened):
            raise ADD_GP_UCBGroupError("Group indices must be integers.")
        if sorted(flattened) != list(range(D)):
            raise ADD_GP_UCBGroupError("Groups must partition every coordinate exactly once.")

    def _validate_groups(self, groups: tuple[tuple[int, ...], ...]) -> None:
        """Validate a complete disjoint coordinate partition."""

        self._validate_groups_static(groups, self.action_dimension)

    def _random_partition(self) -> tuple[tuple[int, ...], ...]:
        """Draw a seeded partition with groups no larger than the configured cap."""

        permutation = torch.randperm(
            self.action_dimension, generator=self.generator, device=self.device
        ).tolist()
        size = self.config.maximum_group_size
        return tuple(
            tuple(sorted(permutation[start : start + size]))
            for start in range(0, self.action_dimension, size)
        )

    def _resolve_initial_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return validated explicit groups or one seeded random partition."""

        groups = self.config.groups
        if groups is None:
            groups = self._random_partition()
        self._validate_groups(groups)
        return tuple(tuple(group) for group in groups)

    def _build_additive_kernel(
        self, groups: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[gpytorch.kernels.ScaleKernel, ...], GaussianProcess]:
        """Build component kernels and their single shared additive GP."""

        components = tuple(
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=len(group), active_dims=torch.tensor(group, dtype=torch.long)
                )
            ).to(dtype=self.dtype, device=self.device)
            for group in groups
        )
        additive = gpytorch.kernels.AdditiveKernel(*components)
        gp = GaussianProcess(
            GaussianProcessConfig(
                kernel=additive,
                mean="zero",
                noise_mode="learned",
                fit_method="mle",
                dtype=self.dtype,
                device=self.device,
            ),
            input_dimension=self.action_dimension,
        )
        return components, gp

    def _initialize_from_environment_impl(self, data: Any) -> None:
        """Normalize evaluated historical actions and load visible observations."""

        evaluated = data.actions if data.evaluated_actions is None else data.evaluated_actions
        actions = torch.as_tensor(evaluated, dtype=self.dtype, device=self.device)
        if actions.shape[0] == 0:
            return
        self.state.X_train = (
            normalize(actions, self.bounds.to(dtype=self.dtype, device=self.device))
            .detach()
            .clone()
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
        """Normalize finite environment actions while preserving row identity."""

        del round_info
        return DiscreteAcquisitionCandidates(
            environment_actions=candidate_actions.detach().clone(),
            model_inputs=normalize(
                candidate_actions, self.bounds.to(dtype=self.dtype, device=self.device)
            ),
            dataset_indices=candidate_dataset_indices.detach().clone(),
        )

    def _set_gp_data(self) -> None:
        """Synchronize non-empty normalized history with the shared additive GP."""

        if self.state.y_train.numel():
            self.state.additive_gp.set_data(
                self.state.X_train,
                to_quality(self.state.y_train, maximize=self.maximize),
            )

    def _initialize_fit_scales(self) -> None:
        """Seed MLE at scales compatible with the observed response magnitude.

        The additive model uses a zero mean, so unit output scales can be a very
        poor numerical starting point when otherwise valid observations have a
        large offset.  This initialization changes neither the additive kernel
        nor the marginal-likelihood objective: every component still receives
        full MLE fitting.  The total initial prior variance is split equally
        across the additive components.

        Raises
        ------
        ADD_GP_UCBKernelError
            If a model is unexpectedly absent or the empirical scale cannot be
            represented as a finite value.
        """

        gp = self.state.additive_gp
        model = gp.model
        if model is None:
            raise ADD_GP_UCBKernelError(
                "Shared additive GP was not constructed from the observations.",
                algorithm=self,
            )
        quality = to_quality(self.state.y_train, maximize=self.maximize)
        signal_scale = quality.square().mean().clamp_min(1e-3)
        if not torch.isfinite(signal_scale):
            raise ADD_GP_UCBKernelError(
                "Shared additive GP observation scale is not finite.", algorithm=self
            )
        component_scale = signal_scale / len(self.state.groups)
        for kernel in model.covar_module.kernels:
            kernel.outputscale = component_scale

        likelihood = gp.likelihood
        likelihood_noise = getattr(likelihood, "noise", None)
        if likelihood is not None and likelihood_noise is not None:
            # GaussianLikelihood's default lower bound is strictly above 1e-4.
            likelihood.noise = (0.01 * signal_scale).clamp_min(1e-3)

    def _fit_shared_gp(self) -> None:
        """Fit the single additive covariance model and refresh component handles.

        Raises
        ------
        ADD_GP_UCBKernelError
            If shared-GP fitting or component recovery fails.
        """

        self._set_gp_data()
        try:
            self._initialize_fit_scales()
            self.state.additive_gp.fit(maxiter=75)
            model = self.state.additive_gp.model
            if model is not None:
                self.state.component_kernels = tuple(model.covar_module.kernels)
        except Exception as error:
            raise ADD_GP_UCBKernelError(
                f"Shared additive GP fitting failed: {error}", algorithm=self
            ) from error

    def _component_covariance(
        self, group_id: int, U: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate one fitted component on group-coordinate matrices."""

        kernel = self.state.component_kernels[group_id]
        base = kernel.base_kernel
        lengthscale = base.lengthscale.detach().reshape(1, 1, -1)
        difference = (U.unsqueeze(-2) - V.unsqueeze(-3)) / lengthscale
        return kernel.outputscale.detach() * torch.exp(-0.5 * difference.square().sum(dim=-1))

    def component_posterior(
        self, group_id: int, U: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a component posterior using the full covariance inverse.

        Parameters
        ----------
        group_id:
            Index of the additive coordinate group.
        U:
            ``(N, |group|)`` candidate coordinates in normalized model units.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Component posterior mean and standard deviation, each shaped
            ``(N,)``.

        Raises
        ------
        ADD_GP_UCBGroupError
            If ``group_id`` does not identify a configured group.
        ADD_GP_UCBKernelError
            If no observations or fitted component kernel are available.
        """

        if not 0 <= group_id < len(self.state.groups):
            raise ADD_GP_UCBGroupError(
                "group_id is outside the additive partition.", algorithm=self
            )
        if self.state.y_train.numel() == 0:
            raise ADD_GP_UCBKernelError(
                "Component posterior requires observations.", algorithm=self
            )
        group = self.state.groups[group_id]
        U = torch.as_tensor(U, dtype=self.dtype, device=self.device)
        if U.ndim == 1:
            U = U.unsqueeze(0)
        if U.ndim != 2 or U.shape[1] != len(group):
            raise ADD_GP_UCBGroupError(
                f"Component query must have shape (N, {len(group)}).", algorithm=self
            )
        X = self.state.X_train
        full_covariance = torch.zeros(
            (X.shape[0], X.shape[0]), dtype=self.dtype, device=self.device
        )
        for index, coordinates in enumerate(self.state.groups):
            full_covariance += self._component_covariance(
                index, X[:, coordinates], X[:, coordinates]
            )
        noise = torch.tensor(1e-4, dtype=self.dtype, device=self.device)
        model = self.state.additive_gp.model
        if model is not None:
            likelihood = self.state.additive_gp.likelihood
            candidate_noise = getattr(likelihood, "noise", None)
            if candidate_noise is not None:
                noise = torch.as_tensor(
                    candidate_noise, dtype=self.dtype, device=self.device
                ).mean()
        delta = full_covariance + (noise + 1e-8) * torch.eye(
            X.shape[0], dtype=self.dtype, device=self.device
        )
        cholesky = torch.linalg.cholesky(delta)
        quality = to_quality(self.state.y_train, maximize=self.maximize)
        alpha = torch.cholesky_solve(quality[:, None], cholesky)
        cross = self._component_covariance(group_id, U, X[:, group])
        mean = (cross @ alpha).squeeze(-1)
        solved = torch.cholesky_solve(cross.T, cholesky)
        prior = self._component_covariance(group_id, U, U)
        covariance = 0.5 * ((prior - cross @ solved) + (prior - cross @ solved).T)
        variance = torch.diagonal(covariance).clamp_min(0)
        return mean, variance

    def _beta(self) -> float:
        """Resolve and validate the current non-negative exploration coefficient."""

        beta = self.config.beta
        if callable(beta):
            value = beta(self.iteration + 1, self.action_dimension)
        elif beta == "paper":
            maximum_group = max(len(group) for group in self.state.groups)
            value = 2.0 * math.log(
                max(2.0, (self.iteration + 2) ** (maximum_group + 2) * math.pi**2 / 0.6)
            )
        else:
            value = float(beta)
        if not math.isfinite(float(value)) or value < 0:
            raise ADD_GP_UCBAcquisitionError("beta returned an invalid value.", algorithm=self)
        return float(value)

    def _group_ucb(self, group_id: int, U: torch.Tensor) -> torch.Tensor:
        """Evaluate one component UCB on normalized group coordinates ``U``."""

        mean, variance = self.component_posterior(group_id, U)
        return mean + math.sqrt(self._beta()) * variance.sqrt()

    def _sum_group_ucb(self, X: torch.Tensor) -> torch.Tensor:
        """Sum component UCB values for full normalized action rows."""

        values = torch.zeros(X.shape[0], dtype=self.dtype, device=self.device)
        for group_id, group in enumerate(self.state.groups):
            values += self._group_ucb(group_id, X[:, group])
        return values

    def _optimize_component(
        self, group_id: int, group: tuple[int, ...]
    ) -> tuple[torch.Tensor, float]:
        """Optimize one component UCB with the configured exact boundary."""

        optimizer = self.config.component_optimizer
        if optimizer is not None:
            try:
                raw = optimizer(
                    group_id=group_id,
                    bounds=Bounds(
                        torch.zeros(len(group), dtype=self.dtype, device=self.device),
                        torch.ones(len(group), dtype=self.dtype, device=self.device),
                    ),
                    acquisition=lambda query: self._group_ucb(
                        group_id,
                        torch.as_tensor(query, dtype=self.dtype, device=self.device).reshape(
                            -1, len(group)
                        ),
                    ),
                    generator=self.generator,
                )
            except Exception as error:
                raise ADD_GP_UCBAcquisitionError(
                    f"The component optimizer failed for group {group_id}: {error}",
                    algorithm=self,
                ) from error
            point = torch.as_tensor(raw, dtype=self.dtype, device=self.device).reshape(-1)
            if point.shape != (len(group),) or not torch.isfinite(point).all():
                raise ADD_GP_UCBAcquisitionError(
                    "component_optimizer must return one finite point in group coordinates.",
                    algorithm=self,
                )
            if not torch.all((point >= 0) & (point <= 1)):
                raise ADD_GP_UCBAcquisitionError(
                    "component_optimizer returned a point outside [0, 1].",
                    algorithm=self,
                )
            value = float(self._group_ucb(group_id, point.reshape(1, -1)).item())
            return point, value

        try:
            from scipy.optimize import direct
        except ImportError as error:
            raise ADD_GP_UCBAcquisitionError(
                "SciPy DIRECT is required by Add-GP-UCB.", algorithm=self
            ) from error

        def objective(values_array: Any) -> float:
            """Return one component's negated UCB for SciPy DIRECT."""

            query = torch.as_tensor(values_array, dtype=self.dtype, device=self.device).reshape(
                1, -1
            )
            return -float(self._group_ucb(group_id, query).item())

        result = direct(objective, [(0.0, 1.0)] * len(group), maxfun=256)
        point = torch.as_tensor(result.x, dtype=self.dtype, device=self.device)
        return point, -float(result.fun)

    def _refresh_decomposition_if_required(self) -> None:
        """Score seeded partition candidates at the configured refresh cycle.

        Candidate partitions share a fixed-hyperparameter marginal-likelihood
        comparison. The winning partition receives a fresh additive GP.

        Raises
        ------
        ADD_GP_UCBDecompositionError
            If every candidate covariance is numerically invalid.
        """

        if not self.config.learn_decomposition or self.state.y_train.numel() < 2:
            return
        if self.iteration - self.state.last_decomposition_refresh < self.config.refresh_cycle:
            return
        count = self.config.decomposition_candidates or self.action_dimension
        candidates = [self.state.groups] + [self._random_partition() for _ in range(count)]
        scores: list[float] = []
        for groups in candidates:
            # Fixed-hyperparameter additive marginal likelihood provides a
            # deterministic decomposition comparison before the winning model
            # receives full hyperparameter fitting.
            covariance = torch.zeros(
                (self.state.X_train.shape[0], self.state.X_train.shape[0]),
                dtype=self.dtype,
                device=self.device,
            )
            for group in groups:
                difference = self.state.X_train[:, None, group] - self.state.X_train[None, :, group]
                covariance += torch.exp(-0.5 * difference.square().sum(dim=-1))
            covariance += 1e-4 * torch.eye(
                covariance.shape[0], dtype=self.dtype, device=self.device
            )
            try:
                cholesky = torch.linalg.cholesky(covariance)
                quality = to_quality(self.state.y_train, maximize=self.maximize)
                alpha = torch.cholesky_solve(quality[:, None], cholesky)
                score = -0.5 * float((quality[:, None].T @ alpha).item())
                score -= float(torch.log(torch.diagonal(cholesky)).sum().item())
            except RuntimeError:
                score = float("-inf")
            scores.append(score)
        if not any(math.isfinite(score) for score in scores):
            raise ADD_GP_UCBDecompositionError(
                "Every candidate additive decomposition failed.", algorithm=self
            )
        selected = max(range(len(scores)), key=scores.__getitem__)
        groups = candidates[selected]
        components, gp = self._build_additive_kernel(groups)
        self.state.groups = groups
        self.state.component_kernels = components
        self.state.additive_gp = gp
        self.state.candidate_decomposition_scores = scores
        self.state.last_decomposition_refresh = self.iteration
        self._set_gp_data()

    def _initial_design_size(self) -> int:
        """Return the configured or ambient-dimension initial-design size."""

        return self.config.initial_points or max(2 * self.action_dimension, 4)

    def _ask_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Select an initial, finite, or componentwise continuous action."""

        finite = "candidate_actions" in round_info
        table = None
        if finite:
            table = self.prepare_discrete_candidates(
                candidate_actions=round_info["candidate_actions"],
                candidate_dataset_indices=round_info["candidate_dataset_indices"],
                round_info=round_info,
            )
        if self.state.y_train.numel() < self._initial_design_size():
            if table is None:
                unit = self._sample_design(
                    Bounds(
                        torch.zeros(self.action_dimension, dtype=self.dtype, device=self.device),
                        torch.ones(self.action_dimension, dtype=self.dtype, device=self.device),
                    ),
                    1,
                    phase="initial",
                    seed=self.config.seed + self.iteration,
                )
                action = unnormalize(unit, self.bounds.to(dtype=self.dtype, device=self.device))
            else:
                position = int(torch.argmin(table.dataset_indices).item())
                action = table.environment_actions[position : position + 1]
                unit = table.model_inputs[position : position + 1]
            component_values: tuple[float, ...] = ()
            value = None
        else:
            self._refresh_decomposition_if_required()
            self._fit_shared_gp()
            if table is not None:
                scores = self._sum_group_ucb(table.model_inputs)
                maximum = scores.max()
                tied = torch.nonzero(scores == maximum, as_tuple=False).flatten()
                position = int(tied[torch.argmin(table.dataset_indices[tied])].item())
                unit = table.model_inputs[position : position + 1]
                action = table.environment_actions[position : position + 1]
                value = float(scores[position].item())
                component_values = tuple(
                    float(self._group_ucb(index, unit[:, group]).item())
                    for index, group in enumerate(self.state.groups)
                )
            else:
                if self.acquisition_sampling_method != "paper":
                    sampled = self._sample_design(
                        Bounds(
                            torch.zeros(
                                self.action_dimension, dtype=self.dtype, device=self.device
                            ),
                            torch.ones(self.action_dimension, dtype=self.dtype, device=self.device),
                        ),
                        256,
                        phase="acquisition",
                        seed=self.config.seed + self.iteration,
                    )
                    sampled_scores = self._sum_group_ucb(sampled)
                    selected = torch.argsort(sampled_scores, descending=True, stable=True)[
                        : self.request_size
                    ]
                    unit = sampled[selected]
                    action = unnormalize(unit, self.bounds.to(dtype=self.dtype, device=self.device))
                    component_values = tuple(
                        float(self._group_ucb(index, unit[:, group]).sum().item())
                        for index, group in enumerate(self.state.groups)
                    )
                    value = float(sampled_scores[selected].sum().item())
                else:
                    unit = torch.empty(
                        (1, self.action_dimension), dtype=self.dtype, device=self.device
                    )
                    values: list[float] = []
                    for group_id, group in enumerate(self.state.groups):
                        group_point, group_value = self._optimize_component(group_id, group)
                        unit[0, list(group)] = group_point
                        values.append(group_value)
                    action = unnormalize(unit, self.bounds.to(dtype=self.dtype, device=self.device))
                    component_values = tuple(values)
                    value = sum(values)
            self._last_acquisition_value = value
        self._last_component_values = component_values
        self.state.last_ask_action = action.detach().clone()
        self._record_last_ask(
            actions=action,
            model_inputs=unit,
            round_info=round_info,
            algorithm_metadata={"groups": self.state.groups, "component_ucb": component_values},
        )
        return action

    def _tell_impl(
        self, *, action: torch.Tensor, feedback: Mapping[str, Any], round_info: Mapping[str, Any]
    ) -> None:
        """Append evaluated normalized actions and observations to shared history."""

        del action, round_info
        evaluated_unit = normalize(
            feedback["evaluated_action"], self.bounds.to(dtype=self.dtype, device=self.device)
        )
        self.state.X_train = torch.cat((self.state.X_train, evaluated_unit), dim=0)
        self.state.y_train = torch.cat((self.state.y_train, feedback["observation"]), dim=0)
        self.state.last_ask_action = None
        self._set_gp_data()

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Return group, decomposition, component-UCB, and training diagnostics."""

        return {
            "groups": self.state.groups,
            "group_sizes": tuple(len(group) for group in self.state.groups),
            "training_rows": int(self.state.y_train.numel()),
            "last_component_ucb": self._last_component_values,
            "decomposition_scores": tuple(self.state.candidate_decomposition_scores),
            "last_decomposition_refresh": self.state.last_decomposition_refresh,
        }

    def _config_repr(self) -> str:
        """Return a stable representation including callable configuration fields."""

        values = asdict(self.config)
        values["beta"] = repr(self.config.beta)
        values["component_optimizer"] = repr(self.config.component_optimizer)
        return repr(values)

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize partition, normalized history, and diagnostic progress."""

        return {
            "config": self._config_repr(),
            "groups": self.state.groups,
            "X_train": self.state.X_train,
            "y_train": self.state.y_train,
            "last_decomposition_refresh": self.state.last_decomposition_refresh,
            "candidate_decomposition_scores": tuple(self.state.candidate_decomposition_scores),
            "last_ask_action": self.state.last_ask_action,
            "last_component_values": self._last_component_values,
        }

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Validate additive checkpoint fields, partition, and strict config."""

        required = {
            "config",
            "groups",
            "X_train",
            "y_train",
            "last_decomposition_refresh",
            "candidate_decomposition_scores",
            "last_ask_action",
            "last_component_values",
        }
        if missing := required - set(state):
            raise ADD_GP_UCBCheckpointError(
                f"Add-GP-UCB state is missing {sorted(missing)}.", algorithm=self
            )
        self._validate_groups(tuple(tuple(group) for group in state["groups"]))
        del strict

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Restore partition and history with a fresh unfitted shared GP."""

        del strict
        groups = tuple(tuple(group) for group in state["groups"])
        components, gp = self._build_additive_kernel(groups)
        self.state.groups = groups
        self.state.component_kernels = components
        self.state.additive_gp = gp
        self.state.X_train = (
            torch.as_tensor(state["X_train"], dtype=self.dtype, device=self.device).detach().clone()
        )
        self.state.y_train = (
            torch.as_tensor(state["y_train"], dtype=self.dtype, device=self.device).detach().clone()
        )
        self.state.last_decomposition_refresh = int(state["last_decomposition_refresh"])
        self.state.candidate_decomposition_scores = [
            float(value) for value in state["candidate_decomposition_scores"]
        ]
        value = state["last_ask_action"]
        self.state.last_ask_action = (
            None
            if value is None
            else torch.as_tensor(value, dtype=self.dtype, device=self.device).detach().clone()
        )
        self._last_component_values = tuple(
            float(value) for value in state["last_component_values"]
        )

    def _reset_impl(self) -> None:
        """Clear observations and rebuild the seeded initial additive model."""

        groups = self._resolve_initial_groups()
        components, gp = self._build_additive_kernel(groups)
        self.state = AddGPUCBState(
            groups=groups,
            X_train=torch.empty((0, self.action_dimension), dtype=self.dtype, device=self.device),
            y_train=torch.empty((0,), dtype=self.dtype, device=self.device),
            component_kernels=components,
            additive_gp=gp,
            last_decomposition_refresh=0,
            candidate_decomposition_scores=[],
            last_ask_action=None,
        )
        self._last_component_values = ()


__all__ = ["ADD_GP_UCB", "AddGPUCBConfig", "AddGPUCBState"]
