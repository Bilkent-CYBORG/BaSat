"""TuRBO-1 local Bayesian optimization with adaptive trust regions."""

from __future__ import annotations

import math
from collections.abc import Mapping
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
from bayopy.models.Acquisitions import (
    AcquisitionConfig,
    DiscreteAcquisitionCandidates,
    build_acquisition,
    optimize_acquisition_discrete,
)
from bayopy.models.GaussianProcess import GaussianProcess, GaussianProcessConfig
from bayopy.models.Kernels import KernelConfig
from bayopy.models.utils import Bounds, normalize, unnormalize

from .Exception.exception_TuRBO import (
    TuRBOCandidateGenerationError,
    TuRBOCheckpointError,
    TuRBOException,
    TuRBOFiniteCandidateError,
    TuRBOTrustRegionError,
)


@dataclass(frozen=True, slots=True)
class TuRBOConfig:
    """TuRBO-1 constants matching the official noise-free implementation.

    Parameters
    ----------
    batch_size, initial_points:
        Synchronous proposal size and initial-design budget.
    length_init, length_min, length_max:
        Normalized trust-region side-length bounds.
    success_tolerance, failure_tolerance:
        Consecutive-event thresholds for expansion and contraction.
    candidate_count, perturbation_cap:
        Local acquisition-candidate budget and sparse-mask scaling constant.
    acquisition:
        ``"TS"`` for the paper's posterior Thompson sampling, or an explicit
        :class:`~bayopy.AcquisitionConfig` selecting EI or UCB. In every case,
        the acquisition is optimized only over TuRBO's local candidate cloud.
    success_relative_tolerance, noisy_center:
        Success comparison tolerance and noisy-incumbent policy.
    max_cholesky_size, gp_training_steps:
        Local GP linear-algebra and optimization budgets. The shared Gaussian
        process applies its positive default jitter to posterior covariances.
    seed:
        Algorithm-owned candidate and posterior-sampling seed.
    """

    batch_size: int | None = 1
    q_batch: int | None = None
    initial_points: int | None = None
    length_init: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    success_tolerance: int = 3
    failure_tolerance: int | None = None
    candidate_count: int | None = None
    perturbation_cap: float = 20.0
    acquisition: Literal["TS"] | AcquisitionConfig = "TS"
    success_relative_tolerance: float = 1e-3
    noisy_center: Literal["posterior_mean", "best_observation"] = "posterior_mean"
    max_cholesky_size: int = 2000
    gp_training_steps: int = 50
    seed: int = 1


@dataclass(slots=True)
class TrustRegionState:
    """Mutable TuRBO-1 trust-region state.

    Attributes
    ----------
    length, success_count, failure_count, restart_count:
        Current side length, event counters, and completed restarts.
    center_unit, lengthscale_weights:
        Unit-cube center and normalized ARD side-length weights.
    best_value_before_batch:
        Pre-batch quality used for one atomic success decision.
    needs_restart:
        Whether contraction crossed ``length_min``.
    """

    length: float
    success_count: int
    failure_count: int
    restart_count: int
    center_unit: torch.Tensor | None
    lengthscale_weights: torch.Tensor | None
    best_value_before_batch: float | None
    needs_restart: bool


class TuRBO(Algorithm):
    """Run one local GP region and restart after trust-region collapse.

    TuRBO fits a local Matérn Gaussian process. The paper setting selects
    candidates with posterior Thompson sampling; users may instead supply an
    explicit EI or UCB :class:`~bayopy.AcquisitionConfig`.

    Examples
    --------
    >>> import bayopy
    >>> bounds = bayopy.Bounds(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    >>> algorithm = bayopy.TuRBO(
    ...     bounds=bounds,
    ...     config=bayopy.TuRBOConfig(
    ...         initial_points=10,
    ...         candidate_count=200,
    ...         acquisition=bayopy.AcquisitionConfig(
    ...             name="EI", beta=2.0, xi=0.0, mc_samples=512, seed=42
    ...         ),
    ...         batch_size=1,
    ...         seed=42,
    ...     ),
    ...     dtype="float64",
    ...     device="cpu",
    ... )
    """

    ACQUISITION_POLICY = "configurable trust-region TS, EI, or UCB"

    EXCEPTION_CLASS = TuRBOException
    native_environments = frozenset({"stochastic"})
    native_batching = True
    SOURCE = SourceRecord(
        paper_title="Scalable Global Optimization via Local Bayesian Optimization",
        paper_year=2019,
        paper_url="https://proceedings.neurips.cc/paper/2019/hash/6c990b7aca7bc7058f5e98ea909e924b-Abstract.html",
        repository_url="https://github.com/uber-research/TuRBO",
        repository_status="official",
        notes="TuRBO-1 candidate mask and batch Thompson selection; finite rows are an extension.",
    )

    def __init__(
        self,
        *,
        bounds: Bounds,
        config: TuRBOConfig | None = None,
        maximize: bool = True,
        dtype: torch.dtype | str = torch.float64,
        device: torch.device | str = "cpu",
        plot: bool = False,
        verbose: bool = False,
        verbose_setting: bool = False,
        verbose_iter_by_iter: bool = False,
    ) -> None:
        """Initialize TuRBO-1 with an empty local trust region.

        Parameters
        ----------
        bounds:
            Physical action bounds in ``D_env`` coordinates.
        config:
            Optional trust-region, GP, candidate, and batch settings.
        maximize:
            Optimize larger observations when ``True``.
        dtype, device:
            Tensor precision and execution device.
        plot, verbose, verbose_setting, verbose_iter_by_iter:
            Presentation flags inherited from :class:`Algorithm`.

        Raises
        ------
        AlgorithmConfigurationError
            If batch, trust-region, or GP constants are invalid.
        """

        resolved_config = TuRBOConfig() if config is None else config
        self.config = resolved_config
        self._validate_config(resolved_config, bounds.dimension)
        super().__init__(
            bounds=bounds,
            batch_size=resolved_config.batch_size,
            q_batch=resolved_config.q_batch,
            maximize=maximize,
            dtype=dtype,
            device=device,
            seed=resolved_config.seed,
            plot=plot,
            verbose=verbose,
            verbose_setting=verbose_setting,
            verbose_iter_by_iter=verbose_iter_by_iter,
        )
        if isinstance(resolved_config.acquisition, AcquisitionConfig):
            self.invalidate_source_guarantees("bayopy.turbo_acquisition_override")
        self.state = self._new_trust_region_state()
        self.X_train_unit = torch.empty(
            (0, self.action_dimension), dtype=self.dtype, device=self.device
        )
        self.y_train = torch.empty((0,), dtype=self.dtype, device=self.device)
        self.local_start = 0
        self.local_gp = self._new_gp()
        self.model_input_dimension = self.action_dimension
        self._last_candidate_mask_sparsity: float | None = None
        self._finite_exhaustion_events = 0

    @staticmethod
    def _validate_config(config: TuRBOConfig, D: int) -> None:
        """Validate immutable TuRBO settings against ambient dimension ``D``."""

        if not isinstance(config, TuRBOConfig):
            raise AlgorithmConfigurationError("config must be TuRBOConfig.")
        if config.acquisition != "TS" and not isinstance(config.acquisition, AcquisitionConfig):
            raise AlgorithmConfigurationError(
                "acquisition must be 'TS' or an AcquisitionConfig for EI/UCB."
            )
        if isinstance(config.acquisition, AcquisitionConfig):
            acquisition_name = config.acquisition.name.lower().replace("-", "").replace("_", "")
            if acquisition_name not in {
                "ei",
                "expectedimprovement",
                "ucb",
                "upperconfidencebound",
            }:
                raise AlgorithmConfigurationError(
                    "TuRBO AcquisitionConfig.name must select EI or UCB."
                )
        batch_size, q_batch = resolve_batch_configuration(
            batch_size=config.batch_size, q_batch=config.q_batch
        )
        if q_batch is not None and q_batch > 1 and config.acquisition == "TS":
            raise AlgorithmConfigurationError(
                "TuRBO uses Thompson sampling rather than EI/UCB and does not support "
                "q_batch>1; use batch_size for independent Thompson points."
            )
        request_size = q_batch if q_batch is not None else batch_size
        assert request_size is not None
        for name in ("success_tolerance", "max_cholesky_size", "gp_training_steps"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise AlgorithmConfigurationError(f"{name} must be positive.")
        for name in ("initial_points", "failure_tolerance", "candidate_count"):
            value = getattr(config, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise AlgorithmConfigurationError(f"{name} must be positive or None.")
        scales = (
            config.length_init,
            config.length_min,
            config.length_max,
            config.perturbation_cap,
            config.success_relative_tolerance,
        )
        if any(not math.isfinite(float(value)) or value <= 0 for value in scales):
            raise AlgorithmConfigurationError("TuRBO scale constants must be positive and finite.")
        if not config.length_min < config.length_init <= config.length_max:
            raise AlgorithmConfigurationError("Require length_min < length_init <= length_max.")
        if config.noisy_center not in {"posterior_mean", "best_observation"}:
            raise AlgorithmConfigurationError("Unknown noisy_center policy.")
        if request_size > D * 1000000:
            raise AlgorithmConfigurationError("The selected batch size is unreasonably large.")

    def _new_trust_region_state(self) -> TrustRegionState:
        """Return a fresh trust-region state at configured initial length."""

        return TrustRegionState(
            length=self.config.length_init,
            success_count=0,
            failure_count=0,
            restart_count=0,
            center_unit=None,
            lengthscale_weights=None,
            best_value_before_batch=None,
            needs_restart=False,
        )

    def _new_gp(self) -> GaussianProcess:
        """Build an unfitted ARD Matérn-5/2 GP in normalized action space."""

        return GaussianProcess(
            GaussianProcessConfig(
                kernel=KernelConfig(name="matern52", ard_num_dims=self.action_dimension),
                mean="constant",
                noise_mode="learned",
                fit_method="mle",
                dtype=self.dtype,
                device=self.device,
            ),
            input_dimension=self.action_dimension,
        )

    def _resolved_initial_points(self) -> int:
        """Return the explicit or dimension-derived local initial-design size."""

        return self.config.initial_points or max(2 * self.action_dimension, self.request_size)

    def _resolved_failure_tolerance(self) -> int:
        """Return the explicit or paper-derived failure counter threshold."""

        return self.config.failure_tolerance or math.ceil(
            max(4.0 / self.request_size, self.action_dimension / self.request_size)
        )

    def _resolved_candidate_count(self) -> int:
        """Return the explicit or dimension-capped Thompson candidate count."""

        return self.config.candidate_count or min(100 * self.action_dimension, 5000)

    def _initialize_from_environment_impl(self, data: Any) -> None:
        """Normalize evaluated historical actions while preserving global history."""

        evaluated = data.actions if data.evaluated_actions is None else data.evaluated_actions
        actions = torch.as_tensor(evaluated, dtype=self.dtype, device=self.device)
        if actions.shape[0] == 0:
            return
        self.X_train_unit = (
            normalize(actions, self.bounds.to(dtype=self.dtype, device=self.device))
            .detach()
            .clone()
        )
        self.y_train = (
            torch.as_tensor(data.observations, dtype=self.dtype, device=self.device)
            .detach()
            .clone()
        )

    def _prepare_discrete_candidates_impl(
        self,
        *,
        candidate_actions: torch.Tensor,
        candidate_dataset_indices: torch.Tensor,
        round_info: Mapping[str, Any],
    ) -> DiscreteAcquisitionCandidates:
        """Normalize finite rows while preserving environment actions and indices."""

        del round_info
        return DiscreteAcquisitionCandidates(
            environment_actions=candidate_actions.detach().clone(),
            model_inputs=normalize(
                candidate_actions, self.bounds.to(dtype=self.dtype, device=self.device)
            ),
            dataset_indices=candidate_dataset_indices.detach().clone(),
        )

    def _restart(self) -> None:
        """Start a new local run while preserving global observations."""

        self.state.length = self.config.length_init
        self.state.success_count = 0
        self.state.failure_count = 0
        self.state.restart_count += 1
        self.state.center_unit = None
        self.state.lengthscale_weights = None
        self.state.best_value_before_batch = None
        self.state.needs_restart = False
        self.local_start = self.X_train_unit.shape[0]
        self.local_gp = self._new_gp()

    def _adjust_length(self, old_best: float, new_values: torch.Tensor) -> bool:
        """Apply the official relative success/failure update."""

        quality = to_quality(new_values, maximize=self.maximize)
        new_best = float(quality.max().item())
        improvement = new_best - old_best
        success = improvement > self.config.success_relative_tolerance * max(abs(old_best), 1e-12)
        if success:
            self.state.success_count += 1
            self.state.failure_count = 0
        else:
            self.state.success_count = 0
            self.state.failure_count += 1
        if self.state.success_count >= self.config.success_tolerance:
            self.state.length = min(2.0 * self.state.length, self.config.length_max)
            self.state.success_count = 0
        elif self.state.failure_count >= self._resolved_failure_tolerance():
            self.state.length *= 0.5
            self.state.failure_count = 0
        self.state.needs_restart = self.state.length < self.config.length_min
        return success

    def _local_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized actions and observations since the latest restart."""

        return self.X_train_unit[self.local_start :], self.y_train[self.local_start :]

    def _fit_local_gp(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Standardize local quality values and fit the trust-region GP.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Local normalized actions and unstandardized maximization-quality values.

        Raises
        ------
        TuRBOTrustRegionError
            If no local observations exist or GP fitting fails.
        """

        X, raw_y = self._local_data()
        if X.shape[0] == 0:
            raise TuRBOTrustRegionError("No local observations are available.", algorithm=self)
        quality = to_quality(raw_y, maximize=self.maximize)
        median = quality.median()
        scale = quality.std(unbiased=False).clamp_min(1e-8)
        standardized = (quality - median) / scale
        try:
            self.local_gp.set_data(X, standardized)
            with gpytorch.settings.max_cholesky_size(self.config.max_cholesky_size):
                self.local_gp.fit(maxiter=self.config.gp_training_steps)
        except Exception as error:
            raise TuRBOTrustRegionError(
                f"Local GP fitting failed: {error}", algorithm=self
            ) from error
        return X, quality

    def _select_center(self, X: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        """Choose the normalized center by posterior mean or observed quality."""

        if self.config.noisy_center == "posterior_mean":
            mean, _ = self.local_gp.predict(X)
            position = int(torch.argmax(mean).item())
        else:
            position = int(torch.argmax(quality).item())
        return X[position].detach().clone()

    def _lengthscale_weights(self) -> torch.Tensor:
        """Return ARD weights normalized to unit geometric mean."""

        model = self.local_gp.model
        if model is None:
            return torch.ones(self.action_dimension, dtype=self.dtype, device=self.device)
        kernel = model.covar_module
        base = getattr(kernel, "base_kernel", kernel)
        lengthscale = base.lengthscale.detach().reshape(-1)
        if lengthscale.numel() == 1:
            lengthscale = lengthscale.repeat(self.action_dimension)
        weights = lengthscale / lengthscale.mean()
        weights = weights / weights.prod().pow(1.0 / self.action_dimension)
        return weights

    def _make_trust_region(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct clipped normalized trust-region bounds around the center.

        Raises
        ------
        TuRBOTrustRegionError
            If no local center has been selected.
        """

        if self.state.center_unit is None:
            raise TuRBOTrustRegionError("Trust-region center is undefined.", algorithm=self)
        weights = self._lengthscale_weights()
        self.state.lengthscale_weights = weights.detach().clone()
        lower = (self.state.center_unit - weights * self.state.length / 2).clamp(0.0, 1.0)
        upper = (self.state.center_unit + weights * self.state.length / 2).clamp(0.0, 1.0)
        return lower, upper

    def _generate_paper_candidates(self, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        """Generate the sparse-perturbation TuRBO candidate cloud.

        Parameters
        ----------
        lower, upper:
            Normalized trust-region bounds with shape ``(D_env,)``.

        Returns
        -------
        torch.Tensor
            Candidate rows with shape ``(N_candidates, D_env)``; every row
            perturbs at least one center coordinate.
        """

        count = self._resolved_candidate_count()
        sobol = self._sample_design(
            Bounds(lower, upper),
            count,
            phase="acquisition",
            seed=self.config.seed + self.iteration + self.state.restart_count * 100003,
        )
        probability = min(self.config.perturbation_cap / self.action_dimension, 1.0)
        mask = (
            torch.rand(
                (count, self.action_dimension),
                dtype=self.dtype,
                device=self.device,
                generator=self.generator,
            )
            <= probability
        )
        empty = ~mask.any(dim=1)
        if bool(empty.any()):
            forced = torch.randint(
                self.action_dimension,
                (int(empty.sum().item()),),
                device=self.device,
                generator=self.generator,
            )
            rows = torch.nonzero(empty, as_tuple=False).flatten()
            mask[rows, forced] = True
        assert self.state.center_unit is not None
        candidates = self.state.center_unit.expand(count, -1).clone()
        candidates[mask] = sobol[mask]
        self._last_candidate_mask_sparsity = float(mask.to(self.dtype).mean().item())
        return candidates

    def _filter_finite_candidates(
        self, table: DiscreteAcquisitionCandidates, lower: torch.Tensor, upper: torch.Tensor
    ) -> DiscreteAcquisitionCandidates:
        """Restrict finite rows to the trust region or trigger restart policy.

        Raises
        ------
        TuRBOFiniteCandidateError
            If fewer than ``batch_size`` finite rows remain after a restart.
        """

        inside = ((table.model_inputs >= lower) & (table.model_inputs <= upper)).all(dim=-1)
        if int(inside.sum().item()) < self.request_size:
            self._finite_exhaustion_events += 1
            self._restart()
            if table.model_inputs.shape[0] < self.request_size:
                raise TuRBOFiniteCandidateError(
                    "Not enough finite rows remain for the configured batch.", algorithm=self
                )
            return table
        return DiscreteAcquisitionCandidates(
            environment_actions=table.environment_actions[inside],
            model_inputs=table.model_inputs[inside],
            dataset_indices=table.dataset_indices[inside],
        )

    def _thompson_select(
        self, candidates: torch.Tensor, dataset_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Select distinct rows from independent posterior Thompson samples.

        Stable ties use the smallest candidate position in continuous mode or
        the smallest dataset index in finite mode.

        Returns
        -------
        torch.Tensor
            Selected candidate positions with shape ``(batch_size,)``.

        Raises
        ------
        TuRBOCandidateGenerationError
            If posterior sampling fails.
        """

        try:
            samples = self.local_gp.sample_posterior(
                candidates,
                num_samples=self.request_size,
                seed=self.config.seed + self.iteration,
            ).reshape(self.request_size, candidates.shape[0])
        except Exception as error:
            raise TuRBOCandidateGenerationError(
                f"Posterior Thompson sampling failed: {error}", algorithm=self
            ) from error
        available = torch.ones(candidates.shape[0], dtype=torch.bool, device=self.device)
        selected: list[int] = []
        for batch_index in range(self.request_size):
            scores = samples[batch_index].clone()
            scores[~available] = -torch.inf
            maximum = scores.max()
            tied = torch.nonzero(scores == maximum, as_tuple=False).flatten()
            if dataset_indices is None:
                position = int(tied[0].item())
            else:
                position = int(tied[torch.argmin(dataset_indices[tied])].item())
            selected.append(position)
            available[position] = False
        return torch.tensor(selected, dtype=torch.long, device=self.device)

    def _acquisition_select(
        self,
        candidates: DiscreteAcquisitionCandidates,
    ) -> torch.Tensor:
        """Select candidate rows with the configured local acquisition policy.

        TuRBO's paper mode uses one posterior Thompson draw per requested point.
        An explicit :class:`AcquisitionConfig` instead builds EI/UCB from the
        fitted local GP and optimizes it over the exact same trust-region rows.
        The benchmark is never available to this method.
        """

        if self.config.acquisition == "TS":
            return self._thompson_select(
                candidates.model_inputs,
                candidates.dataset_indices,
            )
        assert isinstance(self.config.acquisition, AcquisitionConfig)
        if self.local_gp.y_train is None or self.local_gp.y_train.numel() == 0:
            raise TuRBOCandidateGenerationError(
                "A fitted local GP is required before acquisition optimization.",
                algorithm=self,
            )
        try:
            acquisition = build_acquisition(
                model=self.local_gp,
                config=self.config.acquisition,
                batch_size=self.batch_size,
                q_batch=self.q_batch,
                best_f=self.local_gp.y_train.max(),
            )
            result = optimize_acquisition_discrete(
                acquisition=acquisition,
                candidates=candidates,
                batch_size=self.batch_size,
                q_batch=self.q_batch,
            )
        except Exception as error:
            raise TuRBOCandidateGenerationError(
                f"Local acquisition optimization failed: {error}", algorithm=self
            ) from error
        values = result.acquisition_values.reshape(-1)
        if values.numel() > 0:
            self._last_acquisition_value = float(values.max().item())
        return result.selected_rows

    def _ask_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Select an initial-design batch or optimize the local acquisition."""

        if self.state.needs_restart:
            self._restart()
        table = None
        if "candidate_actions" in round_info:
            table = self.prepare_discrete_candidates(
                candidate_actions=round_info["candidate_actions"],
                candidate_dataset_indices=round_info["candidate_dataset_indices"],
                round_info=round_info,
            )
        local_count = self.X_train_unit.shape[0] - self.local_start
        if local_count < self._resolved_initial_points():
            if table is None:
                unit = self._sample_design(
                    Bounds(
                        torch.zeros(self.action_dimension, dtype=self.dtype, device=self.device),
                        torch.ones(self.action_dimension, dtype=self.dtype, device=self.device),
                    ),
                    self.request_size,
                    phase="initial",
                    seed=self.config.seed + self.iteration + self.state.restart_count * 100003,
                )
                actions = unnormalize(unit, self.bounds.to(dtype=self.dtype, device=self.device))
            else:
                order = torch.argsort(table.dataset_indices)[: self.request_size]
                unit = table.model_inputs[order]
                actions = table.environment_actions[order]
            self.state.best_value_before_batch = None
        else:
            X, quality = self._fit_local_gp()
            self.state.center_unit = self._select_center(X, quality)
            lower, upper = self._make_trust_region()
            if table is None:
                candidates = self._generate_paper_candidates(lower, upper)
                selection_table = DiscreteAcquisitionCandidates(
                    environment_actions=candidates,
                    model_inputs=candidates,
                    dataset_indices=torch.arange(
                        candidates.shape[0], dtype=torch.long, device=self.device
                    ),
                )
                positions = self._acquisition_select(selection_table)
                unit = candidates[positions]
                actions = unnormalize(unit, self.bounds.to(dtype=self.dtype, device=self.device))
            else:
                filtered = self._filter_finite_candidates(table, lower, upper)
                positions = self._acquisition_select(filtered)
                unit = filtered.model_inputs[positions]
                actions = filtered.environment_actions[positions]
            self.state.best_value_before_batch = float(quality.max().item())
        self._record_last_ask(
            actions=actions,
            model_inputs=unit,
            round_info=round_info,
            algorithm_metadata={
                "trust_region_length": self.state.length,
                "center_unit": self.state.center_unit,
                "restart_count": self.state.restart_count,
            },
        )
        return actions

    def _tell_impl(
        self, *, action: torch.Tensor, feedback: Mapping[str, Any], round_info: Mapping[str, Any]
    ) -> None:
        """Append evaluated feedback and update trust-region success counters."""

        del action, round_info
        old_best = self.state.best_value_before_batch
        evaluated_unit = normalize(
            feedback["evaluated_action"], self.bounds.to(dtype=self.dtype, device=self.device)
        )
        self.X_train_unit = torch.cat((self.X_train_unit, evaluated_unit), dim=0)
        self.y_train = torch.cat((self.y_train, feedback["observation"]), dim=0)
        if old_best is not None:
            self._adjust_length(old_best, feedback["observation"])
        self.state.best_value_before_batch = None

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Return local-region geometry, counters, and candidate diagnostics."""

        volume = None
        if self.state.center_unit is not None and self.state.lengthscale_weights is not None:
            lower, upper = self._make_trust_region()
            volume = float((upper - lower).prod().item())
        return {
            "training_rows": int(self.y_train.numel()),
            "local_training_rows": int(self.y_train[self.local_start :].numel()),
            "trust_region_length": self.state.length,
            "success_count": self.state.success_count,
            "failure_count": self.state.failure_count,
            "restart_count": self.state.restart_count,
            "needs_restart": self.state.needs_restart,
            "center_unit": self.state.center_unit,
            "lengthscale_weights": self.state.lengthscale_weights,
            "local_box_volume": volume,
            "candidate_mask_sparsity": self._last_candidate_mask_sparsity,
            "finite_exhaustion_events": self._finite_exhaustion_events,
            "acquisition": (
                self.config.acquisition
                if isinstance(self.config.acquisition, str)
                else self.config.acquisition.name.upper()
            ),
        }

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize observations and trust-region progress, excluding GP state."""

        return {
            "config": repr(asdict(self.config)),
            "trust_region": {
                "length": self.state.length,
                "success_count": self.state.success_count,
                "failure_count": self.state.failure_count,
                "restart_count": self.state.restart_count,
                "center_unit": self.state.center_unit,
                "lengthscale_weights": self.state.lengthscale_weights,
                "best_value_before_batch": self.state.best_value_before_batch,
                "needs_restart": self.state.needs_restart,
            },
            "X_train_unit": self.X_train_unit,
            "y_train": self.y_train,
            "local_start": self.local_start,
            "last_candidate_mask_sparsity": self._last_candidate_mask_sparsity,
            "finite_exhaustion_events": self._finite_exhaustion_events,
        }

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Validate TuRBO progress fields and restored observation shapes."""

        required = {
            "config",
            "trust_region",
            "X_train_unit",
            "y_train",
            "local_start",
            "last_candidate_mask_sparsity",
            "finite_exhaustion_events",
        }
        if missing := required - set(state):
            raise TuRBOCheckpointError(f"TuRBO state is missing {sorted(missing)}.", algorithm=self)
        X = torch.as_tensor(state["X_train_unit"])
        y = torch.as_tensor(state["y_train"])
        if X.ndim != 2 or X.shape[1] != self.action_dimension or y.shape != (X.shape[0],):
            raise TuRBOCheckpointError("TuRBO training shapes are invalid.", algorithm=self)
        del strict

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Restore progress and observations while constructing a fresh local GP."""

        del strict
        tr = state["trust_region"]
        self.state = TrustRegionState(
            length=float(tr["length"]),
            success_count=int(tr["success_count"]),
            failure_count=int(tr["failure_count"]),
            restart_count=int(tr["restart_count"]),
            center_unit=self._optional_tensor(tr["center_unit"]),
            lengthscale_weights=self._optional_tensor(tr["lengthscale_weights"]),
            best_value_before_batch=(
                None
                if tr["best_value_before_batch"] is None
                else float(tr["best_value_before_batch"])
            ),
            needs_restart=bool(tr["needs_restart"]),
        )
        self.X_train_unit = (
            torch.as_tensor(state["X_train_unit"], dtype=self.dtype, device=self.device)
            .detach()
            .clone()
        )
        self.y_train = (
            torch.as_tensor(state["y_train"], dtype=self.dtype, device=self.device).detach().clone()
        )
        self.local_start = int(state["local_start"])
        self._last_candidate_mask_sparsity = state["last_candidate_mask_sparsity"]
        self._finite_exhaustion_events = int(state["finite_exhaustion_events"])
        # Surrogate state is intentionally not restored. The next acquisition
        # refits a fresh GP from X_train_unit/y_train using the current
        # gp_training_steps value.
        self.local_gp = self._new_gp()

    def _optional_tensor(self, value: Any) -> torch.Tensor | None:
        """Convert an optional checkpoint value to an owned algorithm tensor."""

        return (
            None
            if value is None
            else torch.as_tensor(value, dtype=self.dtype, device=self.device).detach().clone()
        )

    def _reset_impl(self) -> None:
        """Clear all observations and restore the initial trust-region state."""

        self.state = self._new_trust_region_state()
        self.X_train_unit = torch.empty(
            (0, self.action_dimension), dtype=self.dtype, device=self.device
        )
        self.y_train = torch.empty((0,), dtype=self.dtype, device=self.device)
        self.local_start = 0
        self.local_gp = self._new_gp()
        self._last_candidate_mask_sparsity = None
        self._finite_exhaustion_events = 0


__all__ = ["TrustRegionState", "TuRBO", "TuRBOConfig"]
