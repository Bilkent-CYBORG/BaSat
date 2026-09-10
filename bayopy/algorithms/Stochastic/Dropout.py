"""High-dimensional Bayesian optimization using per-round coordinate dropout."""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Literal

import torch

from bayopy.algorithms.algorithm import (
    Algorithm,
    AlgorithmConfigurationError,
    resolve_batch_configuration,
)
from bayopy.algorithms.helper import SourceRecord, to_quality
from bayopy.models.Acquisitions import (
    AcquisitionConfig,
    AcquisitionError,
    DiscreteAcquisitionCandidates,
    build_acquisition,
    optimize_acquisition_continuous,
    optimize_acquisition_discrete,
)
from bayopy.models.GaussianProcess import (
    GaussianProcess,
    GaussianProcessConfig,
    ModelFitError,
)
from bayopy.models.Kernels import KernelConfig
from bayopy.models.utils import Bounds, normalize, unnormalize

from .Exception.exception_Dropout import (
    DropoutCheckpointError,
    DropoutDataCompatibilityError,
    DropoutException,
    DropoutFillStrategyError,
    DropoutWarning,
)

_CONTINUOUS_OPTIMIZER_FAILURE_PREFIX = "Continuous acquisition optimization failed:"
_NUMERICAL_FALLBACK_REASON = "dropout.acquisition_sobol_fallback"
_NUMERICAL_FALLBACK_RAW_SAMPLES = 256
_GP_RETRY_SEED_OFFSET = 104_729
_ACQUISITION_SEED_OFFSET = 1_000_003
_PROCESS_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+(?=>)")


def theoretical_beta(iteration: int) -> float:
    """Return a standard logarithmic UCB exploration schedule.

    Parameters
    ----------
    iteration:
        Zero-based completed algorithm iteration.

    Returns
    -------
    float
        Positive exploration coefficient for the next proposal.
    """

    return 2.0 * math.log(max(2.0, (iteration + 1) ** 2 * math.pi**2 / 0.6))


@dataclass(frozen=True, slots=True)
class DropoutConfig:
    """Configuration of active-coordinate sampling and inactive filling.

    Parameters
    ----------
    active_dimension:
        Number of coordinates modeled in each round.
    fill_strategy, random_fill_probability:
        Paper inactive-coordinate policy and mix probability.
    beta:
        Scalar or callable UCB confidence coefficient.
    acquisition_optimizer:
        Optional active-space optimizer callback.
    initial_points:
        Number of sampled actions before fitting the active-space GP.
    initial_points_sampling_method:
        ``"paper"``, ``"sobol"``, or ``"uniform"`` initial-design sampling.
    batch_size, seed:
        Sequential request size and algorithm-owned seed.
    """

    active_dimension: int
    fill_strategy: Literal["random", "copy", "mix"] = "mix"
    random_fill_probability: float = 0.1
    beta: Callable[[int], float] | float = theoretical_beta
    acquisition_optimizer: Any | None = None
    initial_points: int = 1
    initial_points_sampling_method: Literal["paper", "sobol", "uniform"] = "paper"
    batch_size: int | None = 1
    q_batch: int | None = None
    seed: int = 1


@dataclass(slots=True)
class DropoutState:
    """Full history plus the current round-specific projected model.

    Attributes
    ----------
    X_train_unit, y_train:
        Complete normalized ``(N, D_env)`` history and responses.
    active_indices, active_gp:
        Current ``D_model`` coordinate subset and its projected GP.
    incumbent_index:
        Stable full-history index used by copy/mix filling.
    last_ask_active_point, last_ask_full_point:
        Pending reduced and complete action coordinates.
    active_history:
        Checkpointed sequence of sampled coordinate subsets.
    """

    X_train_unit: torch.Tensor
    y_train: torch.Tensor
    active_indices: torch.Tensor | None
    active_gp: GaussianProcess | None
    incumbent_index: int | None
    last_ask_active_point: torch.Tensor | None
    last_ask_full_point: torch.Tensor | None
    active_history: list[torch.Tensor]


class Dropout(Algorithm):
    """Rebuild a GP on a new uniformly sampled active subset every round.

    Examples
    --------
    >>> import bayopy
    >>> bounds = bayopy.Bounds(lower=[-1.0] * 6, upper=[1.0] * 6)
    >>> algorithm = bayopy.Dropout(
    ...     bounds=bounds,
    ...     config=bayopy.DropoutConfig(active_dimension=2, seed=7),
    ... )
    """

    ACQUISITION_POLICY = "UCB in the sampled active coordinates"
    CHECKPOINT_ADDED_CONFIG_DEFAULTS: ClassVar[Mapping[str, Any]] = {
        "initial_points": 1,
        "initial_points_sampling_method": "paper",
    }

    EXCEPTION_CLASS = DropoutException
    native_batching = False
    native_environments = frozenset({"stochastic"})
    SOURCE = SourceRecord(
        paper_title="High Dimensional Bayesian Optimization Using Dropout",
        paper_year=2017,
        paper_url="https://www.ijcai.org/proceedings/2017/0291.pdf",
        repository_url="https://github.com/zhandawei/Bayesian_Optimization_Algorithms",
        repository_status="related",
        notes="Paper-faithful active/fill rules; finite candidate scoring is a BayOPy extension.",
    )

    def __init__(
        self,
        *,
        bounds: Bounds,
        config: DropoutConfig,
        maximize: bool = True,
        dtype: torch.dtype | str = torch.float64,
        device: torch.device | str = "cpu",
        plot: bool = False,
        verbose: bool = False,
        verbose_setting: bool = False,
        verbose_iter_by_iter: bool = False,
    ) -> None:
        """Initialize Dropout BO with full history and no active subset.

        Parameters
        ----------
        bounds:
            Physical action bounds in ``D_env`` coordinates.
        config:
            Active dimension, fill strategy, beta, optimizer, and seed.
        maximize:
            Optimize larger observations when ``True``.
        dtype, device:
            Tensor precision and execution device.
        plot, verbose, verbose_setting, verbose_iter_by_iter:
            Presentation flags inherited from :class:`Algorithm`.

        Raises
        ------
        AlgorithmConfigurationError
            If dimensions, fill policy, beta, or batch semantics are invalid.
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
        self.state = self._empty_state()
        self.model_input_dimension = config.active_dimension
        self._last_fill_mode: str | None = None
        self._last_finite_projection_duplicates = 0
        self._last_acquisition_optimizer: str | None = None
        self._acquisition_fallback_count = 0
        self._last_gp_fit_mode: str | None = None
        self._gp_fit_retry_count = 0

    @staticmethod
    def _validate_config(config: DropoutConfig, D: int) -> None:
        """Validate immutable Dropout settings against ambient dimension ``D``."""

        if not isinstance(config, DropoutConfig):
            raise AlgorithmConfigurationError("config must be DropoutConfig.")
        if (
            isinstance(config.active_dimension, bool)
            or not isinstance(config.active_dimension, int)
            or not 1 <= config.active_dimension <= D
        ):
            raise AlgorithmConfigurationError("Dropout requires 1 <= active_dimension <= D.")
        if config.fill_strategy not in {"random", "copy", "mix"}:
            raise AlgorithmConfigurationError("Unknown Dropout fill_strategy.")
        resolve_batch_configuration(batch_size=config.batch_size, q_batch=config.q_batch)
        if (
            isinstance(config.initial_points, bool)
            or not isinstance(config.initial_points, int)
            or config.initial_points < 1
        ):
            raise AlgorithmConfigurationError("initial_points must be a positive integer.")
        probability = config.random_fill_probability
        if not math.isfinite(float(probability)) or not 0.0 <= probability <= 1.0:
            raise AlgorithmConfigurationError("random_fill_probability must lie in [0, 1].")
        if not callable(config.beta) and (
            not math.isfinite(float(config.beta)) or float(config.beta) < 0
        ):
            raise AlgorithmConfigurationError("beta must be finite and non-negative.")

    def _empty_state(self) -> DropoutState:
        """Return empty full-dimensional history and round-specific state."""

        return DropoutState(
            X_train_unit=torch.empty(
                (0, self.action_dimension), dtype=self.dtype, device=self.device
            ),
            y_train=torch.empty((0,), dtype=self.dtype, device=self.device),
            active_indices=None,
            active_gp=None,
            incumbent_index=None,
            last_ask_active_point=None,
            last_ask_full_point=None,
            active_history=[],
        )

    def _initialize_from_environment_impl(self, data: Any) -> None:
        """Normalize evaluated historical actions and establish the incumbent."""

        evaluated = data.actions if data.evaluated_actions is None else data.evaluated_actions
        actions = torch.as_tensor(evaluated, dtype=self.dtype, device=self.device)
        if actions.shape[0] == 0:
            return
        self.state.X_train_unit = (
            normalize(actions, self.bounds.to(dtype=self.dtype, device=self.device))
            .detach()
            .clone()
        )
        self.state.y_train = (
            torch.as_tensor(data.observations, dtype=self.dtype, device=self.device)
            .detach()
            .clone()
        )
        self._update_incumbent()

    def _prepare_discrete_candidates_impl(
        self,
        *,
        candidate_actions: torch.Tensor,
        candidate_dataset_indices: torch.Tensor,
        round_info: Mapping[str, Any],
    ) -> DiscreteAcquisitionCandidates:
        """Project finite rows onto the current active coordinate subset.

        Raises
        ------
        DropoutDataCompatibilityError
            If the active subset has not yet been sampled for this round.
        """

        del round_info
        if self.state.active_indices is None:
            raise DropoutDataCompatibilityError(
                "Active indices must be sampled before finite candidate projection.", algorithm=self
            )
        unit = normalize(candidate_actions, self.bounds.to(dtype=self.dtype, device=self.device))
        projected = unit[:, self.state.active_indices]
        self._last_finite_projection_duplicates = int(
            projected.shape[0] - torch.unique(projected, dim=0).shape[0]
        )
        return DiscreteAcquisitionCandidates(
            environment_actions=candidate_actions.detach().clone(),
            model_inputs=projected.detach().clone(),
            dataset_indices=candidate_dataset_indices.detach().clone(),
        )

    def _sample_active_indices(self) -> torch.Tensor:
        """Draw a fresh uniform subset without replacement."""

        return (
            torch.randperm(self.action_dimension, generator=self.generator, device=self.device)[
                : self.config.active_dimension
            ]
            .sort()
            .values
        )

    def _project_history(self) -> torch.Tensor:
        """Project full normalized history into the current active subspace."""

        if self.state.active_indices is None:
            raise DropoutDataCompatibilityError("Active indices are undefined.", algorithm=self)
        return self.state.X_train_unit[:, self.state.active_indices]

    def _new_active_gp(self) -> GaussianProcess:
        """Build an unfit GP for the current active-coordinate projection."""

        return GaussianProcess(
            GaussianProcessConfig(
                kernel=KernelConfig(name="matern52", ard_num_dims=self.config.active_dimension),
                mean="constant",
                noise_mode="learned",
                fit_method="mle",
                dtype=self.dtype,
                device=self.device,
            ),
            input_dimension=self.config.active_dimension,
        )

    @contextmanager
    def _owned_torch_rng(self, *, seed: int) -> Iterator[None]:
        """Temporarily seed every torch operation without changing ambient RNG state."""

        cuda_devices: list[int] = []
        if self.device.type == "cuda":
            cuda_devices.append(
                torch.cuda.current_device() if self.device.index is None else self.device.index
            )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            yield

    def _fit_active_gp_once(self, gp: GaussianProcess, *, seed: int) -> None:
        """Fit one MLE restart without mutating PyTorch's ambient RNG state."""

        with self._owned_torch_rng(seed=seed):
            gp.fit(maxiter=50)

    def _fit_active_gp(self) -> GaussianProcess:
        """Build and fit a reproducible round-specific active-space GP.

        BoTorch may randomize MLE restart initialization.  The randomness is
        isolated and seeded from Dropout's owned seed so a checkpoint resumes
        identically in a new process.  One fresh deterministic restart retains
        the same GP and MLE objective when the numerical solver exhausts its
        first set of starts.
        """

        if self.state.y_train.numel() == 0:
            raise DropoutDataCompatibilityError("Dropout GP requires observations.", algorithm=self)
        projected = self._project_history()
        quality = to_quality(self.state.y_train, maximize=self.maximize)
        fit_errors: list[ModelFitError] = []
        for attempt in range(2):
            gp = self._new_active_gp()
            try:
                gp.set_data(projected, quality)
                self._fit_active_gp_once(
                    gp,
                    seed=(self.config.seed + self.iteration + (attempt * _GP_RETRY_SEED_OFFSET)),
                )
            except ModelFitError as error:
                fit_errors.append(error)
                continue
            except Exception as error:
                raise DropoutDataCompatibilityError(
                    f"Round-specific Dropout GP fitting failed: {error}", algorithm=self
                ) from error
            if attempt:
                warnings.warn(
                    "Dropout's active-space GP MLE failed on its first deterministic "
                    "initialization; the same MLE fit succeeded after one fresh "
                    "deterministic restart. The GP model, objective, acquisition, and "
                    "inactive-coordinate fill rules are unchanged.",
                    DropoutWarning,
                    stacklevel=3,
                )
                self._gp_fit_retry_count += 1
                self._last_gp_fit_mode = "mle_deterministic_retry"
            else:
                self._last_gp_fit_mode = "mle_deterministic"
            self.state.active_gp = gp
            return gp
        details = "; ".join(str(error) for error in fit_errors)
        raise DropoutDataCompatibilityError(
            "Round-specific Dropout GP fitting failed after two deterministic "
            f"MLE attempts: {details}",
            algorithm=self,
        ) from fit_errors[-1]

    def _beta(self) -> float:
        """Resolve and validate the current non-negative UCB coefficient."""

        value = (
            self.config.beta(self.iteration + 1) if callable(self.config.beta) else self.config.beta
        )
        if not math.isfinite(float(value)) or float(value) < 0:
            raise DropoutFillStrategyError(
                "Dropout beta returned an invalid value.", algorithm=self
            )
        return float(value)

    def _choose_fill_mode(self) -> Literal["random", "copy"]:
        """Resolve the configured deterministic or Bernoulli inactive fill mode."""

        if self.config.fill_strategy == "random":
            return "random"
        if self.config.fill_strategy == "copy":
            return "copy"
        random_draw = torch.rand((), dtype=self.dtype, device=self.device, generator=self.generator)
        return "random" if random_draw < self.config.random_fill_probability else "copy"

    def _fill_inactive_coordinates(
        self, active_point: torch.Tensor
    ) -> tuple[torch.Tensor, Literal["random", "copy"]]:
        """Fill inactive unit coordinates with the configured paper strategy."""

        if self.state.active_indices is None:
            raise DropoutDataCompatibilityError("Active indices are undefined.", algorithm=self)
        mode = self._choose_fill_mode()
        if mode == "random" or self.state.incumbent_index is None:
            full = torch.rand(
                (active_point.shape[0], self.action_dimension),
                dtype=self.dtype,
                device=self.device,
                generator=self.generator,
            )
            if mode == "copy" and self.state.incumbent_index is None:
                mode = "random"
        else:
            full = (
                self.state.X_train_unit[self.state.incumbent_index : self.state.incumbent_index + 1]
                .expand(active_point.shape[0], -1)
                .clone()
            )
        full[:, self.state.active_indices] = active_point
        return full, mode

    def _update_incumbent(self) -> None:
        """Update the best learner-visible full-history row for copy filling."""

        if self.state.y_train.numel() == 0:
            self.state.incumbent_index = None
            return
        quality = to_quality(self.state.y_train, maximize=self.maximize)
        self.state.incumbent_index = int(torch.argmax(quality).item())

    def _select_finite_row(self, table: DiscreteAcquisitionCandidates, scores: torch.Tensor) -> int:
        """Resolve acquisition ties using fill preference, then stable row index."""

        maximum = scores.max()
        tied = torch.nonzero(scores == maximum, as_tuple=False).flatten()
        if tied.numel() == 1:
            return int(tied.item())
        assert self.state.active_indices is not None
        target_active = table.model_inputs[tied[0] : tied[0] + 1]
        target_full, mode = self._fill_inactive_coordinates(target_active)
        self._last_fill_mode = mode
        unit_rows = normalize(
            table.environment_actions[tied], self.bounds.to(dtype=self.dtype, device=self.device)
        )
        inactive_mask = torch.ones(self.action_dimension, dtype=torch.bool, device=self.device)
        inactive_mask[self.state.active_indices] = False
        distance = (
            (unit_rows[:, inactive_mask] - target_full[:, inactive_mask]).square().sum(dim=-1)
        )
        best_distance = distance.min()
        closest = tied[torch.nonzero(distance == best_distance, as_tuple=False).flatten()]
        return int(closest[torch.argmin(table.dataset_indices[closest])].item())

    def _optimize_active_acquisition(self, acquisition: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Optimize active-space UCB, with an explicit numerical-only fallback.

        The primary call is the paper-profile local optimizer.  A failure from
        that numerical solve may occur when UCB has a flat or singular gradient
        at a training location.  In that one case, select the exact maximum UCB
        score *within* a deterministic Sobol raw design.  This preserves the
        acquisition and inactive-coordinate fill rules, while reporting that
        continuous maximization was approximated.
        """

        bounds = torch.stack(
            (
                torch.zeros(self.config.active_dimension, dtype=self.dtype, device=self.device),
                torch.ones(self.config.active_dimension, dtype=self.dtype, device=self.device),
            )
        )
        arguments: dict[str, Any] = {
            "acquisition": acquisition,
            "bounds": bounds,
            "batch_size": self.batch_size,
            "q_batch": self.q_batch,
            "num_restarts": 10,
            "raw_samples": _NUMERICAL_FALLBACK_RAW_SAMPLES,
            "sampling_method": self.acquisition_sampling_method,
            "sampling_generator": self.generator,
            "sampling_seed": self.config.seed + self.iteration,
        }
        try:
            with self._owned_torch_rng(
                seed=self.config.seed + self.iteration + _ACQUISITION_SEED_OFFSET
            ):
                point, score = optimize_acquisition_continuous(**arguments)
        except AcquisitionError as error:
            is_local_numerical_failure = self.acquisition_sampling_method == "paper" and str(
                error
            ).startswith(_CONTINUOUS_OPTIMIZER_FAILURE_PREFIX)
            if not is_local_numerical_failure:
                raise
            fallback_arguments = dict(arguments)
            fallback_arguments["sampling_method"] = "sobol"
            try:
                with self._owned_torch_rng(
                    seed=self.config.seed + self.iteration + _ACQUISITION_SEED_OFFSET
                ):
                    point, score = optimize_acquisition_continuous(**fallback_arguments)
            except AcquisitionError as fallback_error:
                raise AcquisitionError(
                    f"{error} The deterministic Sobol fallback also failed: {fallback_error}"
                ) from fallback_error
            warnings.warn(
                "Dropout's paper UCB local optimizer encountered a numerical failure; "
                "BayOPy selected the exact best finite UCB score from a deterministic "
                f"{_NUMERICAL_FALLBACK_RAW_SAMPLES}-set Sobol design. The UCB and "
                "inactive-coordinate fill rules are unchanged, but continuous acquisition "
                "maximization is approximated.",
                DropoutWarning,
                stacklevel=3,
            )
            self.invalidate_source_guarantees(_NUMERICAL_FALLBACK_REASON)
            self._acquisition_fallback_count += 1
            self._last_acquisition_optimizer = "sobol_numerical_fallback"
            return point, score
        self._last_acquisition_optimizer = self.acquisition_sampling_method
        return point, score

    def _ask_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Sample active coordinates and select a continuous or finite action."""

        self.state.active_indices = self._sample_active_indices()
        self.state.active_history.append(self.state.active_indices.detach().clone())
        mode: Literal["random", "copy", "finite"]
        table = None
        if "candidate_actions" in round_info:
            table = self.prepare_discrete_candidates(
                candidate_actions=round_info["candidate_actions"],
                candidate_dataset_indices=round_info["candidate_dataset_indices"],
                round_info=round_info,
            )
        if self.state.y_train.numel() < self.config.initial_points:
            if table is None:
                active_point = self._sample_design(
                    Bounds(
                        torch.zeros(
                            self.config.active_dimension,
                            dtype=self.dtype,
                            device=self.device,
                        ),
                        torch.ones(
                            self.config.active_dimension,
                            dtype=self.dtype,
                            device=self.device,
                        ),
                    ),
                    self.request_size,
                    phase="initial",
                    seed=self.config.seed + self.iteration,
                    paper_method="uniform",
                )
                full_unit, mode = self._fill_inactive_coordinates(active_point)
                action = unnormalize(
                    full_unit, self.bounds.to(dtype=self.dtype, device=self.device)
                )
                self._last_acquisition_optimizer = "initial_design"
            else:
                if table.dataset_indices.numel() < self.request_size:
                    raise DropoutDataCompatibilityError(
                        "The finite table has fewer rows than the requested Dropout batch.",
                        algorithm=self,
                    )
                positions = torch.argsort(table.dataset_indices, stable=True)[: self.request_size]
                active_point = table.model_inputs[positions]
                action = table.environment_actions[positions]
                full_unit = normalize(action, self.bounds.to(dtype=self.dtype, device=self.device))
                mode = "finite"
                self._last_acquisition_optimizer = "finite_initial_row"
            value = None
        else:
            gp = self._fit_active_gp()
            acquisition = build_acquisition(
                model=gp,
                config=AcquisitionConfig(
                    name="UCB", beta=self._beta(), seed=self.config.seed + self.iteration
                ),
                batch_size=self.batch_size,
                q_batch=self.q_batch,
            )
            if table is None:
                active_point, score = self._optimize_active_acquisition(acquisition)
                full_unit, mode = self._fill_inactive_coordinates(active_point)
                action = unnormalize(
                    full_unit, self.bounds.to(dtype=self.dtype, device=self.device)
                )
                value = float(score.item())
            else:
                if self.request_size == 1:
                    with torch.no_grad():
                        scores = acquisition(table.model_inputs.unsqueeze(-2)).reshape(-1)
                    position = self._select_finite_row(table, scores)
                    active_point = table.model_inputs[position : position + 1]
                    action = table.environment_actions[position : position + 1]
                    value = float(scores[position].item())
                else:
                    result = optimize_acquisition_discrete(
                        acquisition=acquisition,
                        candidates=table,
                        batch_size=self.batch_size,
                        q_batch=self.q_batch,
                    )
                    active_point = result.model_inputs
                    action = result.environment_actions
                    value = float(result.acquisition_values.sum().item())
                full_unit = normalize(action, self.bounds.to(dtype=self.dtype, device=self.device))
                mode = "finite"
                self._last_acquisition_optimizer = "finite_candidate_scoring"
            self._last_acquisition_value = value
        self._last_fill_mode = mode
        self.state.last_ask_active_point = active_point.detach().clone()
        self.state.last_ask_full_point = full_unit.detach().clone()
        self._record_last_ask(
            actions=action,
            model_inputs=active_point,
            round_info=round_info,
            algorithm_metadata={
                "active_indices": self.state.active_indices,
                "fill_mode": mode,
                "acquisition_value": value,
                "acquisition_optimizer": self._last_acquisition_optimizer,
                "gp_fit_mode": self._last_gp_fit_mode,
            },
        )
        return action

    def _tell_impl(
        self, *, action: torch.Tensor, feedback: Mapping[str, Any], round_info: Mapping[str, Any]
    ) -> None:
        """Append evaluated full actions and discard the round-specific GP."""

        del action, round_info
        evaluated = normalize(
            feedback["evaluated_action"], self.bounds.to(dtype=self.dtype, device=self.device)
        )
        self.state.X_train_unit = torch.cat((self.state.X_train_unit, evaluated), dim=0)
        self.state.y_train = torch.cat((self.state.y_train, feedback["observation"]), dim=0)
        self._update_incumbent()
        self.state.last_ask_active_point = None
        self.state.last_ask_full_point = None
        # The next active subset is different, so retaining this projected GP
        # would associate observations with the wrong model coordinates.
        self.state.active_gp = None

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Return active-subset, fill, duplicate, incumbent, and history diagnostics."""

        return {
            "active_dimension": self.config.active_dimension,
            "active_indices": self.state.active_indices,
            "active_history": tuple(self.state.active_history),
            "incumbent_index": self.state.incumbent_index,
            "last_fill_mode": self._last_fill_mode,
            "finite_projection_duplicates": self._last_finite_projection_duplicates,
            "last_acquisition_optimizer": self._last_acquisition_optimizer,
            "acquisition_fallback_count": self._acquisition_fallback_count,
            "last_gp_fit_mode": self._last_gp_fit_mode,
            "gp_fit_retry_count": self._gp_fit_retry_count,
            "training_rows": int(self.state.y_train.numel()),
        }

    def _config_repr(self) -> str:
        """Return a process-stable representation of all configuration fields."""

        return repr(self._fingerprint_value(self.config))

    def _legacy_config_repr(self) -> str:
        """Return the address-bearing representation used by old checkpoints."""

        values = asdict(self.config)
        values["beta"] = repr(self.config.beta)
        values["acquisition_optimizer"] = repr(self.config.acquisition_optimizer)
        return repr(values)

    def _matches_legacy_config_repr(self, value: Any) -> bool:
        """Recognize an otherwise-identical legacy repr across Python processes."""

        if not isinstance(value, str):
            return False
        stored = _PROCESS_ADDRESS.sub("", value)
        current = _PROCESS_ADDRESS.sub("", self._legacy_config_repr())
        return stored == current

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize full observation history, active-round state, and diagnostics."""

        return {
            "config": self._config_repr(),
            "X_train_unit": self.state.X_train_unit,
            "y_train": self.state.y_train,
            "active_indices": self.state.active_indices,
            "incumbent_index": self.state.incumbent_index,
            "last_ask_active_point": self.state.last_ask_active_point,
            "last_ask_full_point": self.state.last_ask_full_point,
            "active_history": tuple(self.state.active_history),
            "last_fill_mode": self._last_fill_mode,
            "finite_projection_duplicates": self._last_finite_projection_duplicates,
            "last_acquisition_optimizer": self._last_acquisition_optimizer,
            "acquisition_fallback_count": self._acquisition_fallback_count,
            "last_gp_fit_mode": self._last_gp_fit_mode,
            "gp_fit_retry_count": self._gp_fit_retry_count,
        }

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Validate Dropout checkpoint fields, history shapes, and strict config."""

        required = {
            "config",
            "X_train_unit",
            "y_train",
            "active_indices",
            "incumbent_index",
            "last_ask_active_point",
            "last_ask_full_point",
            "active_history",
            "last_fill_mode",
            "finite_projection_duplicates",
        }
        if missing := required - set(state):
            raise DropoutCheckpointError(
                f"Dropout state is missing {sorted(missing)}.", algorithm=self
            )
        X = torch.as_tensor(state["X_train_unit"])
        y = torch.as_tensor(state["y_train"])
        if X.ndim != 2 or X.shape[1] != self.action_dimension or y.shape != (X.shape[0],):
            raise DropoutCheckpointError("Dropout training shapes are invalid.", algorithm=self)
        del strict

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Restore full history and coordinates without a fitted active GP."""

        del strict
        self.state.X_train_unit = (
            torch.as_tensor(state["X_train_unit"], dtype=self.dtype, device=self.device)
            .detach()
            .clone()
        )
        self.state.y_train = (
            torch.as_tensor(state["y_train"], dtype=self.dtype, device=self.device).detach().clone()
        )
        self.state.active_indices = self._optional_tensor(state["active_indices"], dtype=torch.long)
        incumbent = state["incumbent_index"]
        self.state.incumbent_index = None if incumbent is None else int(incumbent)
        self.state.last_ask_active_point = self._optional_tensor(state["last_ask_active_point"])
        self.state.last_ask_full_point = self._optional_tensor(state["last_ask_full_point"])
        self.state.active_history = [
            torch.as_tensor(value, dtype=torch.long, device=self.device).detach().clone()
            for value in state["active_history"]
        ]
        self._last_fill_mode = state["last_fill_mode"]
        self._last_finite_projection_duplicates = int(state["finite_projection_duplicates"])
        self._last_acquisition_optimizer = state.get("last_acquisition_optimizer")
        self._acquisition_fallback_count = int(state.get("acquisition_fallback_count", 0))
        self._last_gp_fit_mode = state.get("last_gp_fit_mode")
        self._gp_fit_retry_count = int(state.get("gp_fit_retry_count", 0))
        self.state.active_gp = None

    def _optional_tensor(
        self, value: Any, *, dtype: torch.dtype | None = None
    ) -> torch.Tensor | None:
        """Convert an optional checkpoint value to an owned tensor and dtype."""

        return (
            None
            if value is None
            else torch.as_tensor(
                value, dtype=self.dtype if dtype is None else dtype, device=self.device
            )
            .detach()
            .clone()
        )

    def _reset_impl(self) -> None:
        """Clear full history and all round-specific active-subspace state."""

        self.state = self._empty_state()
        self._last_fill_mode = None
        self._last_finite_projection_duplicates = 0
        self._last_acquisition_optimizer = None
        self._acquisition_fallback_count = 0
        self._last_gp_fit_mode = None
        self._gp_fit_retry_count = 0


__all__ = ["Dropout", "DropoutConfig", "DropoutState", "theoretical_beta"]
