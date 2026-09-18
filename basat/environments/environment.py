"""Shared environment data structures and evaluation protocol.

The environment layer deliberately separates *learner-visible* information from
experiment-only audit information.  A finite :class:`EnvironmentDataset` is an
evaluation source, not training data: its response column is never returned by
``before_action`` or ``get_history_data`` until the corresponding row has been
selected and evaluated.

All actions in this module use environment coordinates with shape ``(D_env,)``.
Surrogate-model coordinate transformations belong to concrete algorithms.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import torch

from basat.models.utils import (
    Bounds,
    as_tensor,
    resolve_device,
    resolve_dtype,
    validate_bounds,
)

MATCH_ATOL = 1e-8
"""Absolute tolerance used for finite-table coordinate matching."""


class EnvironmentError(Exception):
    """Base class for environment-layer failures."""


class EnvironmentConfigurationError(EnvironmentError):
    """Raised when an environment configuration is internally inconsistent."""


class EnvironmentDataError(EnvironmentError):
    """Raised when finite or historical environment data are invalid."""


class EnvironmentLookupError(EnvironmentError):
    """Raised when an exact finite-table response cannot be located."""


class EnvironmentProtocolError(EnvironmentError):
    """Raised when environment methods are called out of protocol order."""


class InvalidActionError(EnvironmentError):
    """Raised when an action has an invalid shape, value, or bound."""


class NonFiniteObservationError(EnvironmentError):
    """Raised when an objective or provider produces a non-finite response."""


class NoAvailableActionError(EnvironmentError):
    """Raised when no unrevealed finite candidate remains."""


@runtime_checkable
class NoiseProvider(Protocol):
    """Protocol for reproducible custom observation-noise providers."""

    def __call__(
        self,
        *,
        action: torch.Tensor | None,
        context: torch.Tensor | None,
        generator: torch.Generator,
    ) -> Any:
        """Return one realized scalar observation-noise value."""


@dataclass(frozen=True, slots=True)
class EnvironmentDataset:
    """Hidden finite evaluation table.

    Parameters
    ----------
    actions:
        Candidate actions with shape ``(N, D_env)``.
    observations:
        Stored responses with shape ``(N,)`` or ``(N, 1)``.
    contexts:
        Optional contexts with shape ``(N, D_context)``.  Contextual and
        distributional finite environments require this field.
    observation_noise_variance:
        Optional known variance, either a scalar or one value per stored row.
        It describes the stored response and does not request another draw.
    metadata:
        Experiment metadata.  It is retained by the environment but never
        included in pre-action learner information.
    """

    actions: Any
    observations: Any
    contexts: Any | None = None
    observation_noise_variance: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EnvironmentData:
    """Already learner-visible historical observations.

    ``actions`` are the requested actions.  ``evaluated_actions`` may differ in
    adversarial environments and defaults to ``actions`` when omitted.  A
    partially known variance vector may contain ``NaN`` at unknown positions
    when data are assembled from several history records.

    Parameters
    ----------
    actions:
        Requested learner-visible actions shaped ``(N, D_env)``.
    observations:
        Revealed finite responses shaped ``(N,)`` or ``(N, 1)``.
    evaluated_actions:
        Optional actually evaluated ``(N, D_env)`` actions; adversarial rows
        may differ from ``actions``.
    contexts:
        Optional realized public contexts shaped ``(N, D_context)``.
    observation_noise_variance:
        Optional known scalar or row-aligned variances.
    metadata:
        Learner-safe detached history metadata.
    """

    actions: Any
    observations: Any
    evaluated_actions: Any | None = None
    contexts: Any | None = None
    observation_noise_variance: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def number_of_observations(self) -> int:
        """Return the number of learner-visible observations."""

        return int(self.observations.shape[0])


@dataclass(frozen=True, slots=True)
class EnvironmentRecord:
    """One successfully revealed environment evaluation.

    ``round_info`` and ``feedback_extra`` contain only learner-visible values.
    ``audit`` is intentionally excluded from :meth:`Environment.get_history_data`
    and is available to experiment infrastructure only through
    :meth:`Environment.last_record` or checkpoint state.

    Parameters
    ----------
    round_index:
        Zero-based environment round shared by all members of a q-batch.
    action, evaluated_action:
        Requested and actually evaluated ``D_env`` coordinates.
    observation:
        Scalar learner-visible response.
    round_info, feedback_extra:
        Learner-visible mappings before and after evaluation.
    context:
        Optional realized public context.
    observation_noise_variance:
        Optional known variance of the revealed response.
    dataset_index:
        Optional hidden-table audit row, never exposed before action selection.
    metadata:
        Public experiment metadata.
    audit:
        Experiment-only latent values excluded from learner history.
    """

    round_index: int
    action: torch.Tensor
    evaluated_action: torch.Tensor
    observation: torch.Tensor
    round_info: Mapping[str, Any] = field(default_factory=dict)
    feedback_extra: Mapping[str, Any] = field(default_factory=dict)
    context: torch.Tensor | None = None
    observation_noise_variance: torch.Tensor | None = None
    dataset_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    audit: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(slots=True)
class EnvironmentHistory:
    """Previous learner data and records revealed during this run.

    Parameters
    ----------
    initial_data:
        Optional immutable learner-visible history supplied at construction.
    records:
        Ordered records committed atomically by successful environment steps.
    """

    initial_data: EnvironmentData | None = None
    records: list[EnvironmentRecord] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AvailableActions:
    """Learner-visible finite actions and stable representative row indices.

    Parameters
    ----------
    actions:
        Complete available coordinates shaped ``(M, D_env)``.
    dataset_indices:
        Stable representative source rows shaped ``(M,)``; these indices do
        not reveal observations.
    """

    actions: torch.Tensor
    dataset_indices: torch.Tensor


@dataclass(frozen=True, slots=True)
class _Evaluation:
    """Internal, not-yet-persisted result of evaluating one action."""

    action: torch.Tensor
    evaluated_action: torch.Tensor
    observation: torch.Tensor
    feedback_extra: Mapping[str, Any] = field(default_factory=dict)
    context: torch.Tensor | None = None
    observation_noise_variance: torch.Tensor | None = None
    dataset_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    audit: Mapping[str, Any] = field(default_factory=dict)


def _clone_tree(value: Any) -> Any:
    """Clone tensors and containers before crossing a visibility boundary."""

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    try:
        return copy.deepcopy(value)
    except (TypeError, ValueError):
        return value


def _scalar_tensor(
    value: Any,
    *,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
    error_type: type[EnvironmentError] = EnvironmentDataError,
) -> torch.Tensor:
    """Convert a scalar-like value and reject non-finite results."""

    try:
        result = as_tensor(value, dtype=dtype, device=device, copy=True).squeeze()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise error_type(f"{name} must be convertible to a scalar tensor.") from exc
    if result.ndim != 0:
        raise error_type(f"{name} must contain exactly one scalar value.")
    if not bool(torch.isfinite(result)):
        raise error_type(f"{name} must be finite.")
    return result


class Environment(ABC):
    """Abstract evaluation source shared by all environment families.

    Exactly one of ``function`` and ``dataset`` must be supplied.  The base
    class owns validation, reproducible observation noise, finite availability,
    learner-visible history, batching, and state restoration.  Subclasses only
    define the family-specific evaluation performed by :meth:`_evaluate_one`.

    Parameters
    ----------
    bounds:
        Legal action bounds in ``D_env`` dimensions.
    function, dataset:
        Mutually exclusive callable or hidden finite evaluation source.
    previous_data:
        Optional observations that were visible before this run.
    dtype, device:
        Tensor dtype and device used by the environment.
    noise_std:
        Standard deviation of default zero-mean Gaussian observation noise.
    noise_fn:
        Optional custom realized-noise provider.  It receives the environment
        generator when its signature accepts that keyword.
    add_noise_to_data:
        Add configured synthetic noise to stored finite responses.  Defaults to
        ``False`` to prevent accidental double-noising.
    allow_repeated_actions:
        Keep previously selected finite coordinates available.
    data_row_selection:
        Select the first or a seeded random hidden row among duplicate matches.
    seed:
        Seed for the environment-owned random generator.
    """

    environment_type: str = "base"
    package_version: str = "0.1.0"
    provided_round_info: frozenset[str] = frozenset()
    provided_feedback: frozenset[str] = frozenset({"observation", "evaluated_action"})
    capabilities: frozenset[str] = frozenset()

    def __init__(
        self,
        *,
        bounds: Bounds,
        function: Callable[..., Any] | None = None,
        dataset: EnvironmentDataset | None = None,
        previous_data: EnvironmentData | None = None,
        dtype: torch.dtype | str = torch.float32,
        device: torch.device | str = "cpu",
        noise_std: float = 0.0,
        noise_fn: NoiseProvider | Callable[..., Any] | None = None,
        add_noise_to_data: bool = False,
        allow_repeated_actions: bool = False,
        data_row_selection: Literal["first", "random"] = "first",
        seed: int = 1,
    ) -> None:
        """Initialize a validated function-backed or finite-data environment."""

        if (function is None) == (dataset is None):
            message = (
                "Supply function or dataset."
                if function is None
                else "Supply function or dataset, not both."
            )
            raise EnvironmentConfigurationError(message)
        if function is not None and not callable(function):
            raise EnvironmentConfigurationError("function must be callable.")

        self.dtype = resolve_dtype(dtype)
        self.device = resolve_device(device)
        try:
            self.bounds = Bounds(
                lower=as_tensor(
                    bounds.lower,
                    dtype=self.dtype,
                    device=self.device,
                    copy=True,
                ),
                upper=as_tensor(
                    bounds.upper,
                    dtype=self.dtype,
                    device=self.device,
                    copy=True,
                ),
            )
            validate_bounds(self.bounds)
        except AttributeError as exc:
            raise EnvironmentConfigurationError("bounds must be a valid Bounds instance.") from exc

        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise EnvironmentConfigurationError("seed must be a non-negative integer.")
        try:
            noise_std_value = float(noise_std)
        except (TypeError, ValueError) as exc:
            raise EnvironmentConfigurationError(
                "noise_std must be a finite non-negative number."
            ) from exc
        if not math.isfinite(noise_std_value) or noise_std_value < 0.0:
            raise EnvironmentConfigurationError("noise_std must be a finite non-negative number.")
        if noise_fn is not None and not callable(noise_fn):
            raise EnvironmentConfigurationError("noise_fn must be callable.")
        if data_row_selection not in {"first", "random"}:
            raise EnvironmentConfigurationError("data_row_selection must be 'first' or 'random'.")

        self.function = function
        self.source_mode: Literal["function", "data"] = (
            "function" if function is not None else "data"
        )
        self.noise_std = noise_std_value
        self.noise_fn = noise_fn
        self.add_noise_to_data = bool(add_noise_to_data)
        self.allow_repeated_actions = bool(allow_repeated_actions)
        self.data_row_selection = data_row_selection
        self.seed = seed
        self.generator = self._new_generator(seed)
        self.round = 0
        self._pending_round_info: dict[str, Any] | None = None
        self._frozen_source_fingerprint: str | None = None

        self.dataset = None if dataset is None else self._validate_dataset(dataset)
        initial_data = (
            None if previous_data is None else self._validate_environment_data(previous_data)
        )
        self.history = EnvironmentHistory(initial_data=initial_data)

    @property
    def action_dimension(self) -> int:
        """Return ``D_env``, the dimension accepted by :meth:`step`."""

        return int(self.bounds.dimension)

    @property
    def records(self) -> tuple[EnvironmentRecord, ...]:
        """Return an immutable view of newly revealed evaluation records."""

        return tuple(self.history.records)

    def _new_generator(self, seed: int) -> torch.Generator:
        """Create a generator compatible with the configured tensor device."""

        try:
            generator = torch.Generator(device=self.device)
        except (RuntimeError, TypeError):
            # Some accelerators do not expose a device-local Generator.  The
            # CPU generator remains deterministic and is used for scalar/index
            # draws before results are moved to the target device.
            generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return generator

    def _validate_dataset(self, value: EnvironmentDataset) -> EnvironmentDataset:
        """Validate and copy a hidden finite evaluation table."""

        if not isinstance(value, EnvironmentDataset):
            raise EnvironmentDataError("dataset must be an EnvironmentDataset.")
        actions = self._coerce_action_matrix(value.actions, name="dataset.actions")
        if actions.shape[0] == 0:
            raise EnvironmentDataError("dataset must contain at least one row.")
        observations = self._coerce_observation_vector(
            value.observations,
            expected_rows=actions.shape[0],
            name="dataset.observations",
        )
        contexts = self._coerce_context_matrix(
            value.contexts,
            expected_rows=actions.shape[0],
            name="dataset.contexts",
        )
        variances = self._coerce_variance_vector(
            value.observation_noise_variance,
            expected_rows=actions.shape[0],
            name="dataset.observation_noise_variance",
            allow_nan=False,
        )
        if not isinstance(value.metadata, Mapping):
            raise EnvironmentDataError("dataset.metadata must be a mapping.")

        # Tolerance equality must be transitive.  Otherwise a query could match
        # two overlapping duplicate groups and row selection would be ambiguous.
        self._validate_tolerance_equivalence(actions, "dataset action")
        if contexts is not None:
            self._validate_tolerance_equivalence(contexts, "dataset context")
            self._validate_tolerance_equivalence(
                torch.cat((actions, contexts), dim=-1),
                "dataset action-context",
            )
        return EnvironmentDataset(
            actions=actions,
            observations=observations,
            contexts=contexts,
            observation_noise_variance=variances,
            metadata=_clone_tree(value.metadata),
        )

    def _validate_environment_data(self, value: EnvironmentData) -> EnvironmentData:
        """Validate and copy learner-visible historical observations."""

        if not isinstance(value, EnvironmentData):
            raise EnvironmentDataError("previous_data must be an EnvironmentData instance.")
        actions = self._coerce_action_matrix(
            value.actions,
            name="previous_data.actions",
            allow_empty=True,
        )
        observations = self._coerce_observation_vector(
            value.observations,
            expected_rows=actions.shape[0],
            name="previous_data.observations",
        )
        evaluated_actions = (
            actions.clone()
            if value.evaluated_actions is None
            else self._coerce_action_matrix(
                value.evaluated_actions,
                name="previous_data.evaluated_actions",
                allow_empty=True,
            )
        )
        if evaluated_actions.shape != actions.shape:
            raise EnvironmentDataError("previous_data.evaluated_actions must match actions.shape.")
        contexts = self._coerce_context_matrix(
            value.contexts,
            expected_rows=actions.shape[0],
            name="previous_data.contexts",
        )
        variances = self._coerce_variance_vector(
            value.observation_noise_variance,
            expected_rows=actions.shape[0],
            name="previous_data.observation_noise_variance",
            allow_nan=True,
        )
        if not isinstance(value.metadata, Mapping):
            raise EnvironmentDataError("previous_data.metadata must be a mapping.")
        return EnvironmentData(
            actions=actions,
            observations=observations,
            evaluated_actions=evaluated_actions,
            contexts=contexts,
            observation_noise_variance=variances,
            metadata=_clone_tree(value.metadata),
        )

    def _coerce_action_matrix(
        self,
        value: Any,
        *,
        name: str,
        allow_empty: bool = False,
    ) -> torch.Tensor:
        """Convert and validate a two-dimensional action matrix."""

        try:
            actions = as_tensor(
                value,
                dtype=self.dtype,
                device=self.device,
                copy=True,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise EnvironmentDataError(f"{name} must be numeric.") from exc
        if actions.numel() == 0 and allow_empty:
            actions = actions.reshape(0, self.action_dimension)
        if actions.ndim != 2 or actions.shape[1] != self.action_dimension:
            raise EnvironmentDataError(f"{name} must have shape (N, {self.action_dimension}).")
        if not bool(torch.isfinite(actions).all()):
            raise EnvironmentDataError(f"{name} contains non-finite values.")
        if bool((actions < self.bounds.lower).any()) or bool((actions > self.bounds.upper).any()):
            raise EnvironmentDataError(f"{name} contains an out-of-bounds action.")
        return actions

    def _coerce_observation_vector(
        self,
        value: Any,
        *,
        expected_rows: int,
        name: str,
    ) -> torch.Tensor:
        """Convert observations to a finite vector with ``expected_rows``."""

        try:
            observations = as_tensor(
                value,
                dtype=self.dtype,
                device=self.device,
                copy=True,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise EnvironmentDataError(f"{name} must be numeric.") from exc
        if observations.ndim == 2 and observations.shape[1] == 1:
            observations = observations[:, 0]
        if observations.ndim != 1 or observations.shape[0] != expected_rows:
            raise EnvironmentDataError(f"{name} must have shape ({expected_rows},).")
        if not bool(torch.isfinite(observations).all()):
            raise EnvironmentDataError(f"{name} contains non-finite values.")
        return observations

    def _coerce_context_matrix(
        self,
        value: Any | None,
        *,
        expected_rows: int,
        name: str,
    ) -> torch.Tensor | None:
        """Convert optional contexts to shape ``(N, D_context)``."""

        if value is None:
            return None
        try:
            contexts = as_tensor(
                value,
                dtype=self.dtype,
                device=self.device,
                copy=True,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise EnvironmentDataError(f"{name} must be numeric.") from exc
        if contexts.ndim == 1:
            if contexts.shape[0] != expected_rows:
                raise EnvironmentDataError(f"{name} must have one context per observation.")
            contexts = contexts.unsqueeze(-1)
        if contexts.ndim != 2 or contexts.shape[0] != expected_rows:
            raise EnvironmentDataError(f"{name} must have shape ({expected_rows}, D_context).")
        if contexts.shape[1] == 0:
            raise EnvironmentDataError(f"{name} cannot have zero columns.")
        if not bool(torch.isfinite(contexts).all()):
            raise EnvironmentDataError(f"{name} contains non-finite values.")
        return contexts

    def _coerce_variance_vector(
        self,
        value: Any | None,
        *,
        expected_rows: int,
        name: str,
        allow_nan: bool,
    ) -> torch.Tensor | None:
        """Convert optional non-negative observation variances."""

        if value is None:
            return None
        try:
            variance = as_tensor(
                value,
                dtype=self.dtype,
                device=self.device,
                copy=True,
            ).squeeze()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise EnvironmentDataError(f"{name} must be numeric.") from exc
        if variance.ndim == 0:
            variance = variance.expand(expected_rows).clone()
        if variance.ndim != 1 or variance.shape[0] != expected_rows:
            raise EnvironmentDataError(f"{name} must be scalar or shape ({expected_rows},).")
        allowed = torch.isfinite(variance)
        if allow_nan:
            allowed = allowed | torch.isnan(variance)
        if not bool(allowed.all()) or bool((variance[~torch.isnan(variance)] < 0).any()):
            raise EnvironmentDataError(
                f"{name} must contain non-negative finite values"
                + (" or NaN for unknown values." if allow_nan else ".")
            )
        return variance

    @staticmethod
    def _row_match(
        matrix: torch.Tensor,
        row: torch.Tensor,
        *,
        atol: float = MATCH_ATOL,
    ) -> torch.Tensor:
        """Return exact-relative, tolerance-absolute row matches."""

        return torch.isclose(matrix, row, rtol=0.0, atol=atol).all(dim=-1)

    def _validate_tolerance_equivalence(
        self,
        coordinates: torch.Tensor,
        label: str,
    ) -> None:
        """Reject non-transitive tolerance overlap among finite coordinates."""

        visited = torch.zeros(coordinates.shape[0], dtype=torch.bool, device=coordinates.device)
        for index in range(coordinates.shape[0]):
            if bool(visited[index]):
                continue
            group = self._row_match(coordinates, coordinates[index])
            members = torch.nonzero(group, as_tuple=False).flatten()
            for member in members.tolist():
                if not bool(torch.equal(group, self._row_match(coordinates, coordinates[member]))):
                    raise EnvironmentDataError(
                        f"Ambiguous {label} rows overlap at atol={MATCH_ATOL}."
                    )
            visited |= group

    def validate_action(self, action: Any) -> torch.Tensor:
        """Validate one action against ``D_env`` and the legal bounds.

        Parameters
        ----------
        action:
            Numeric one-dimensional action-like value.

        Returns
        -------
        torch.Tensor
            A detached copy with shape ``(D_env,)``.
        """

        try:
            result = as_tensor(
                action,
                dtype=self.dtype,
                device=self.device,
                copy=True,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise InvalidActionError("action must be numeric.") from exc
        if result.ndim != 1 or result.shape[0] != self.action_dimension:
            raise InvalidActionError(f"action must have shape ({self.action_dimension},).")
        if not bool(torch.isfinite(result).all()):
            raise InvalidActionError("action contains non-finite values.")
        if bool((result < self.bounds.lower).any()) or bool((result > self.bounds.upper).any()):
            raise InvalidActionError("action is outside the environment bounds.")
        return result

    def _coerce_action_batch(self, action: Any) -> tuple[torch.Tensor, bool]:
        """Return a validated action batch and whether input was explicitly batched."""

        try:
            tensor = as_tensor(
                action,
                dtype=self.dtype,
                device=self.device,
                copy=True,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise InvalidActionError("action must be numeric.") from exc
        was_batched = tensor.ndim == 2
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != self.action_dimension:
            raise InvalidActionError(
                "action must have shape "
                f"({self.action_dimension},) or (q, {self.action_dimension})."
            )
        if tensor.shape[0] == 0:
            raise InvalidActionError("an action batch cannot be empty.")
        validated = torch.stack([self.validate_action(row) for row in tensor])
        if not self.allow_repeated_actions:
            self._reject_duplicate_batch_actions(validated)
        return validated, was_batched

    def _reject_duplicate_batch_actions(self, actions: torch.Tensor) -> None:
        """Reject a duplicate coordinate twice in one non-repeating batch."""

        for index in range(actions.shape[0]):
            matches = self._row_match(actions[index + 1 :], actions[index])
            if bool(matches.any()):
                raise InvalidActionError("A non-repeating batch cannot contain duplicate actions.")

    def evaluate_function(self, *args: torch.Tensor) -> torch.Tensor:
        """Evaluate the callable source and validate one scalar response."""

        if self.source_mode != "function" or self.function is None:
            raise EnvironmentProtocolError(
                "evaluate_function() is only valid in function-backed mode."
            )
        result = self.function(*[_clone_tree(arg) for arg in args])
        return _scalar_tensor(
            result,
            dtype=self.dtype,
            device=self.device,
            name="function response",
            error_type=NonFiniteObservationError,
        )

    def sample_noise(
        self,
        *,
        action: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draw one realized observation-noise value reproducibly.

        Parameters
        ----------
        action:
            Optional requested action supplied to a custom noise provider.
        context:
            Optional realized context supplied to a custom noise provider.

        Returns
        -------
        torch.Tensor
            Finite scalar realized noise on the environment device.
        """

        if self.noise_fn is not None:
            value = self._invoke_provider(
                self.noise_fn,
                primary_name=None,
                primary_value=None,
                action=_clone_tree(action),
                context=_clone_tree(context),
                generator=self.generator,
            )
            return _scalar_tensor(
                value,
                dtype=self.dtype,
                device=self.device,
                name="noise_fn response",
                error_type=NonFiniteObservationError,
            )
        if self.noise_std == 0.0:
            return torch.zeros((), dtype=self.dtype, device=self.device)
        generator_device = getattr(self.generator, "device", torch.device("cpu"))
        draw = torch.randn((), dtype=self.dtype, device=generator_device, generator=self.generator)
        return draw.to(device=self.device) * self.noise_std

    def make_data_observation(
        self,
        stored_observation: Any,
        *,
        action: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a stored response and any explicitly requested added noise.

        Parameters
        ----------
        stored_observation:
            Scalar response selected from the hidden finite table.
        action, context:
            Coordinates passed only to a configured stochastic noise provider.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Learner observation and realized added-noise scalar.
        """

        observation = _scalar_tensor(
            stored_observation,
            dtype=self.dtype,
            device=self.device,
            name="stored observation",
            error_type=NonFiniteObservationError,
        )
        added_noise = torch.zeros((), dtype=self.dtype, device=self.device)
        if self.add_noise_to_data:
            added_noise = self.sample_noise(action=action, context=context)
            observation = observation + added_noise
        if not bool(torch.isfinite(observation)):
            raise NonFiniteObservationError("Noised stored observation is non-finite.")
        return observation, added_noise

    def _candidate_dataset_indices(
        self,
        *,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return hidden row indices compatible with a context restriction."""

        if self.dataset is None:
            raise EnvironmentProtocolError("No finite dataset is configured.")
        indices = torch.arange(self.dataset.actions.shape[0], device=self.device, dtype=torch.long)
        if context is None:
            return indices
        if self.dataset.contexts is None:
            raise EnvironmentLookupError(
                "A context was requested but the finite dataset has no contexts."
            )
        context_value = self._validate_context(context)
        return indices[self._row_match(self.dataset.contexts, context_value)]

    def find_dataset_indices(
        self,
        *,
        action: torch.Tensor,
        context: torch.Tensor | None = None,
        atol: float = MATCH_ATOL,
    ) -> torch.Tensor:
        """Find all hidden rows matching an exact action or action-context pair.

        Parameters
        ----------
        action:
            Full ``D_env`` requested action.
        context:
            Optional context required by contextual finite tables.
        atol:
            Finite non-negative absolute coordinate tolerance; relative
            tolerance is always zero.

        Returns
        -------
        torch.Tensor
            One-dimensional matching hidden-row indices.
        """

        if self.source_mode != "data" or self.dataset is None:
            raise EnvironmentProtocolError(
                "find_dataset_indices() is only valid in data-backed mode."
            )
        if not math.isfinite(float(atol)) or float(atol) < 0:
            raise EnvironmentConfigurationError("atol must be finite and non-negative.")
        action_value = self.validate_action(action)
        match = self._row_match(self.dataset.actions, action_value, atol=float(atol))
        if context is not None:
            if self.dataset.contexts is None:
                return torch.empty(0, dtype=torch.long, device=self.device)
            context_value = self._validate_context(context)
            match &= self._row_match(self.dataset.contexts, context_value, atol=float(atol))
        return torch.nonzero(match, as_tuple=False).flatten().to(dtype=torch.long)

    def lookup_dataset_observation(
        self,
        *,
        action: Any,
        context: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """Reveal one response from rows matching an action-context coordinate.

        Duplicate responses remain hidden until this method selects one.  The
        returned row follows ``data_row_selection`` and the environment RNG.

        Parameters
        ----------
        action:
            Exact requested action coordinate.
        context:
            Optional exact context coordinate.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None, int]
            Stored observation, optional known variance, and selected row index.
        """

        matches = self.find_dataset_indices(action=action, context=context)
        if matches.numel() == 0:
            suffix = " and context" if context is not None else ""
            raise EnvironmentLookupError(
                f"No finite dataset row matches the requested action{suffix}."
            )
        if self.data_row_selection == "first":
            row = int(matches.min().item())
        else:
            generator_device = getattr(self.generator, "device", torch.device("cpu"))
            offset = int(
                torch.randint(
                    matches.numel(),
                    (),
                    generator=self.generator,
                    device=generator_device,
                ).item()
            )
            row = int(matches[offset].item())
        assert self.dataset is not None
        variance = (
            None
            if self.dataset.observation_noise_variance is None
            else self.dataset.observation_noise_variance[row].clone()
        )
        return self.dataset.observations[row].clone(), variance, row

    def _collapse_action_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Collapse duplicate actions to their smallest stable row index."""

        assert self.dataset is not None
        ordered = indices.to(device=self.device, dtype=torch.long).sort().values
        candidates = self.dataset.actions[ordered]
        remaining = torch.ones(ordered.numel(), dtype=torch.bool, device=self.device)
        representatives: list[int] = []
        for offset in range(ordered.numel()):
            if not bool(remaining[offset]):
                continue
            representatives.append(int(ordered[offset]))
            # Mark matches in one tensor operation rather than one Python call
            # per pair. Greedy first-representative semantics are unchanged,
            # including absolute tolerance; no quadratic match matrix is kept.
            remaining &= ~self._row_match(candidates, candidates[offset])
        return torch.tensor(representatives, dtype=torch.long, device=self.device)

    def _remove_observed_actions(
        self,
        *,
        actions: torch.Tensor,
        indices: torch.Tensor,
        observed: EnvironmentData,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Remove previously revealed finite coordinate groups."""

        if observed.actions.shape[0] == 0:
            return actions, indices
        observed_actions = observed.actions
        if context is not None and observed.contexts is not None:
            context_value = self._validate_context(context)
            observed_actions = observed_actions[self._row_match(observed.contexts, context_value)]
        keep = torch.ones(actions.shape[0], dtype=torch.bool, device=self.device)
        for observed_action in observed_actions:
            keep &= ~self._row_match(actions, observed_action)
        return actions[keep], indices[keep]

    def available_actions(
        self,
        *,
        context: Any | None = None,
    ) -> AvailableActions:
        """Return unique available finite actions without response values.

        Parameters
        ----------
        context:
            Optional revealed context used to restrict compatible rows.

        Returns
        -------
        AvailableActions
            Stable unique coordinates and representative hidden-row indices.
        """

        if self.source_mode != "data" or self.dataset is None:
            raise EnvironmentProtocolError(
                "available_actions() is only valid for a data-backed environment."
            )
        context_value = None if context is None else self._validate_context(context)
        compatible = self._candidate_dataset_indices(context=context_value)
        representatives = self._collapse_action_indices(compatible)
        actions = self.dataset.actions[representatives]
        if not self.allow_repeated_actions:
            actions, representatives = self._remove_observed_actions(
                actions=actions,
                indices=representatives,
                observed=self.get_history_data(),
                context=context_value,
            )
        if actions.shape[0] == 0:
            raise NoAvailableActionError("The finite candidate set is empty.")
        return AvailableActions(
            actions=actions.detach().clone(),
            dataset_indices=representatives.detach().clone(),
        )

    def _assert_action_available(
        self,
        action: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
    ) -> None:
        """Ensure a requested finite action was learner-visible this round."""

        available = self.available_actions(context=context)
        if not bool(self._row_match(available.actions, action).any()):
            raise EnvironmentLookupError(
                "The requested action is not in the current finite candidate set."
            )

    def before_action(self) -> dict[str, Any]:
        """Return learner-visible information available before selection."""

        self._source_fingerprint()
        if self._pending_round_info is not None:
            return _clone_tree(self._pending_round_info)
        if self.source_mode == "function":
            info: dict[str, Any] = {}
        else:
            available = self.available_actions()
            info = {
                "candidate_actions": available.actions,
                "candidate_dataset_indices": available.dataset_indices,
            }
        self._pending_round_info = _clone_tree(info)
        return _clone_tree(info)

    def step(self, action: Any) -> dict[str, Any]:
        """Evaluate one action or one synchronous batch.

        A batch increments the round counter once and appends one record per
        member with the same ``round_index``.  Results are prepared fully before
        records are committed, so a failed member never leaves a partial batch
        in learner-visible history.

        Parameters
        ----------
        action:
            Shape ``(D_env,)`` or synchronous ``(q, D_env)`` batch.

        Returns
        -------
        dict[str, Any]
            Learner-visible feedback, scalar-shaped for one unbatched action and
            leading-``q`` vectorized for an explicit batch.
        """

        self._source_fingerprint()
        actions, was_batched = self._coerce_action_batch(action)
        round_info = {} if self._pending_round_info is None else self._pending_round_info
        generator_state = self.generator.get_state().clone()
        try:
            evaluations = [self._evaluate_one(row) for row in actions]
        except Exception:
            # Built-in stochastic choices are transactional.  External provider
            # side effects cannot be undone and must be managed by that provider.
            self.generator.set_state(generator_state)
            raise

        for evaluation in evaluations:
            self.append_record(
                action=evaluation.action,
                evaluated_action=evaluation.evaluated_action,
                observation=evaluation.observation,
                round_info=round_info,
                feedback_extra=evaluation.feedback_extra,
                context=evaluation.context,
                observation_noise_variance=evaluation.observation_noise_variance,
                dataset_index=evaluation.dataset_index,
                metadata=evaluation.metadata,
                audit=evaluation.audit,
            )
        self.round += 1
        self._pending_round_info = None
        self._after_step()
        return self._collate_feedback(evaluations, was_batched=was_batched)

    @abstractmethod
    def _evaluate_one(self, action: torch.Tensor) -> _Evaluation:
        """Evaluate one validated action without mutating public history."""

    def _after_step(self) -> None:
        """Clear subclass round-local state after a committed batch."""

        return None

    def append_record(
        self,
        *,
        action: Any,
        evaluated_action: Any,
        observation: Any,
        round_info: Mapping[str, Any],
        feedback_extra: Mapping[str, Any] | None = None,
        context: Any | None = None,
        observation_noise_variance: Any | None = None,
        dataset_index: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        audit: Mapping[str, Any] | None = None,
    ) -> None:
        """Append exactly one validated revealed record for the current round.

        Parameters
        ----------
        action, evaluated_action:
            Requested and actually evaluated ``D_env`` coordinates.
        observation:
            Finite scalar learner-visible response.
        round_info, feedback_extra:
            Learner-visible public mappings.
        context:
            Optional realized context.
        observation_noise_variance:
            Optional known non-negative variance.
        dataset_index:
            Optional valid hidden-row audit index.
        metadata:
            Public experiment metadata.
        audit:
            Experiment-only values never returned to the learner.
        """

        requested = self.validate_action(action)
        evaluated = self.validate_action(evaluated_action)
        observed = _scalar_tensor(
            observation,
            dtype=self.dtype,
            device=self.device,
            name="observation",
            error_type=NonFiniteObservationError,
        )
        context_value = None if context is None else self._validate_context(context)
        variance = None
        if observation_noise_variance is not None:
            variance = _scalar_tensor(
                observation_noise_variance,
                dtype=self.dtype,
                device=self.device,
                name="observation_noise_variance",
            )
            if bool(variance < 0):
                raise EnvironmentDataError("observation_noise_variance cannot be negative.")
        if dataset_index is not None and (
            isinstance(dataset_index, bool)
            or not isinstance(dataset_index, int)
            or self.dataset is None
            or not 0 <= dataset_index < self.dataset.actions.shape[0]
        ):
            raise EnvironmentDataError("dataset_index is outside the configured dataset.")
        feedback_mapping = {} if feedback_extra is None else feedback_extra
        metadata_mapping = {} if metadata is None else metadata
        audit_mapping = {} if audit is None else audit
        for name, value in {
            "round_info": round_info,
            "feedback_extra": feedback_mapping,
            "metadata": metadata_mapping,
            "audit": audit_mapping,
        }.items():
            if not isinstance(value, Mapping):
                raise EnvironmentDataError(f"{name} must be a mapping.")
        record = EnvironmentRecord(
            round_index=self.round,
            action=requested,
            evaluated_action=evaluated,
            observation=observed,
            round_info=_clone_tree(round_info),
            feedback_extra=_clone_tree(feedback_mapping),
            context=context_value,
            observation_noise_variance=variance,
            dataset_index=dataset_index,
            metadata=_clone_tree(metadata_mapping),
            audit=_clone_tree(audit_mapping),
        )
        self.history.records.append(record)

    def _collate_feedback(
        self,
        evaluations: list[_Evaluation],
        *,
        was_batched: bool,
    ) -> dict[str, Any]:
        """Build learner feedback without copying any audit-only values."""

        feedback: dict[str, Any] = {
            "observation": torch.stack([item.observation for item in evaluations]),
            "evaluated_action": torch.stack([item.evaluated_action for item in evaluations]),
        }
        if all(item.context is not None for item in evaluations):
            feedback["context"] = torch.stack(
                [item.context for item in evaluations if item.context is not None]
            )
        if any(item.observation_noise_variance is not None for item in evaluations):
            feedback["observation_noise_variance"] = torch.stack(
                [
                    item.observation_noise_variance
                    if item.observation_noise_variance is not None
                    else torch.full((), float("nan"), dtype=self.dtype, device=self.device)
                    for item in evaluations
                ]
            )
        common_keys = (
            set.intersection(*(set(item.feedback_extra) for item in evaluations))
            if evaluations
            else set()
        )
        for key in sorted(common_keys):
            values = [item.feedback_extra[key] for item in evaluations]
            feedback[key] = self._stack_public_values(values)
        if not was_batched:
            for key, value in list(feedback.items()):
                if isinstance(value, torch.Tensor) and value.shape[0] == 1:
                    feedback[key] = value[0].clone()
        return _clone_tree(feedback)

    def _stack_public_values(self, values: list[Any]) -> Any:
        """Stack homogeneous tensor extras and otherwise return cloned values."""

        tensors: list[torch.Tensor] = []
        for value in values:
            try:
                tensors.append(
                    as_tensor(
                        value,
                        dtype=self.dtype,
                        device=self.device,
                        copy=True,
                    )
                )
            except (TypeError, ValueError, RuntimeError):
                return _clone_tree(values)
        try:
            return torch.stack(tensors)
        except RuntimeError:
            return _clone_tree(values)

    def get_history_data(self) -> EnvironmentData:
        """Return only previous and newly revealed learner-visible data."""

        action_parts: list[torch.Tensor] = []
        observation_parts: list[torch.Tensor] = []
        evaluated_parts: list[torch.Tensor] = []
        context_parts: list[torch.Tensor | None] = []
        variance_parts: list[torch.Tensor | None] = []
        initial = self.history.initial_data
        if initial is not None and initial.observations.shape[0] > 0:
            action_parts.extend(initial.actions.unbind(0))
            observation_parts.extend(initial.observations.unbind(0))
            assert initial.evaluated_actions is not None
            evaluated_parts.extend(initial.evaluated_actions.unbind(0))
            context_parts.extend(
                [None] * initial.actions.shape[0]
                if initial.contexts is None
                else list(initial.contexts.unbind(0))
            )
            variance_parts.extend(
                [None] * initial.actions.shape[0]
                if initial.observation_noise_variance is None
                else [
                    None if bool(torch.isnan(value)) else value
                    for value in initial.observation_noise_variance.unbind(0)
                ]
            )
        for record in self.history.records:
            action_parts.append(record.action)
            observation_parts.append(record.observation)
            evaluated_parts.append(record.evaluated_action)
            context_parts.append(record.context)
            variance_parts.append(record.observation_noise_variance)

        if action_parts:
            actions = torch.stack(action_parts)
            observations = torch.stack(observation_parts)
            evaluated_actions = torch.stack(evaluated_parts)
        else:
            actions = torch.empty((0, self.action_dimension), dtype=self.dtype, device=self.device)
            observations = torch.empty((0,), dtype=self.dtype, device=self.device)
            evaluated_actions = actions.clone()

        contexts: torch.Tensor | None = None
        if context_parts and any(value is not None for value in context_parts):
            if not all(value is not None for value in context_parts):
                raise EnvironmentDataError(
                    "History mixes contextual and context-free observations."
                )
            contexts = torch.stack([value for value in context_parts if value is not None])

        variances: torch.Tensor | None = None
        if variance_parts and any(value is not None for value in variance_parts):
            variances = torch.stack(
                [
                    value
                    if value is not None
                    else torch.full((), float("nan"), dtype=self.dtype, device=self.device)
                    for value in variance_parts
                ]
            )
        return EnvironmentData(
            actions=actions.detach().clone(),
            observations=observations.detach().clone(),
            evaluated_actions=evaluated_actions.detach().clone(),
            contexts=None if contexts is None else contexts.detach().clone(),
            observation_noise_variance=(None if variances is None else variances.detach().clone()),
            metadata={
                "initial_metadata": ({} if initial is None else _clone_tree(initial.metadata)),
                "record_metadata": [
                    _clone_tree(record.metadata) for record in self.history.records
                ],
                "round_indices": torch.tensor(
                    [record.round_index for record in self.history.records],
                    dtype=torch.long,
                    device=self.device,
                ),
                "dataset_indices": tuple(record.dataset_index for record in self.history.records),
            },
        )

    def last_record(self) -> dict[str, Any]:
        """Return experiment-only audit information for the latest evaluation."""

        if not self.history.records:
            return {}
        return _clone_tree(self.history.records[-1].audit)

    def state_dict(self) -> dict[str, Any]:
        """Return complete reproducible environment state.

        The configured callable or hidden response table is not serialized.
        Instead, source metadata and a dataset fingerprint are checked on load.
        """

        return {
            "schema_version": 1,
            "package_version": self.package_version,
            "class_identity": f"{type(self).__module__}.{type(self).__qualname__}",
            "environment_type": self.environment_type,
            "source_mode": self.source_mode,
            "round": self.round,
            "seed": self.seed,
            "generator_state": self.generator.get_state().clone(),
            "dtype": str(self.dtype),
            "device_type": self.device.type,
            "bounds_lower": self.bounds.lower.detach().cpu().clone(),
            "bounds_upper": self.bounds.upper.detach().cpu().clone(),
            "noise_std": self.noise_std,
            "has_noise_fn": self.noise_fn is not None,
            "add_noise_to_data": self.add_noise_to_data,
            "allow_repeated_actions": self.allow_repeated_actions,
            "data_row_selection": self.data_row_selection,
            "dataset_fingerprint": self._dataset_fingerprint(),
            "source_fingerprint": self._source_fingerprint(),
            "history": self._history_state_dict(),
            "pending_round_info": _clone_tree(self._pending_round_info),
            "extra_state": self._extra_state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any], *, strict: bool = True) -> None:
        """Validate identity/configuration and restore mutable environment state.

        ``strict=False`` is retained for forward-compatible optional fields; it
        does not permit a different source, class, geometry, or noise policy.

        Parameters
        ----------
        state:
            Complete mapping produced by :meth:`state_dict`.
        strict:
            Forward-compatibility flag; source identity and immutable
            configuration are always enforced.
        """

        del strict
        if not isinstance(state, Mapping):
            raise EnvironmentConfigurationError("environment state must be a mapping.")
        expected = {
            "schema_version": 1,
            "package_version": self.package_version,
            "class_identity": f"{type(self).__module__}.{type(self).__qualname__}",
            "environment_type": self.environment_type,
            "source_mode": self.source_mode,
            "seed": self.seed,
            "noise_std": self.noise_std,
            "has_noise_fn": self.noise_fn is not None,
            "add_noise_to_data": self.add_noise_to_data,
            "allow_repeated_actions": self.allow_repeated_actions,
            "data_row_selection": self.data_row_selection,
            "dataset_fingerprint": self._dataset_fingerprint(),
            "source_fingerprint": self._source_fingerprint(),
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise EnvironmentConfigurationError(
                    f"Checkpoint {key!r} does not match this environment configuration."
                )
        lower = as_tensor(
            state.get("bounds_lower"),
            dtype=self.dtype,
            device=self.device,
            copy=True,
        )
        upper = as_tensor(
            state.get("bounds_upper"),
            dtype=self.dtype,
            device=self.device,
            copy=True,
        )
        if not torch.equal(lower, self.bounds.lower) or not torch.equal(upper, self.bounds.upper):
            raise EnvironmentConfigurationError(
                "Checkpoint bounds do not match this environment configuration."
            )
        round_value = state.get("round")
        if isinstance(round_value, bool) or not isinstance(round_value, int) or round_value < 0:
            raise EnvironmentDataError("Checkpoint round must be non-negative.")
        history = self._history_from_state(state.get("history"))
        self._validate_restored_rounds(history.records, round_value)
        generator_state = state.get("generator_state")
        if not isinstance(generator_state, torch.Tensor):
            raise EnvironmentDataError("Checkpoint generator_state is invalid.")

        # Commit only after every validation succeeds.
        self.generator.set_state(generator_state.detach().cpu())
        self.round = round_value
        self.history = history
        pending = state.get("pending_round_info")
        if pending is not None and not isinstance(pending, Mapping):
            raise EnvironmentDataError("pending_round_info must be a mapping or None.")
        self._pending_round_info = None if pending is None else _clone_tree(pending)
        self._load_extra_state_dict(state.get("extra_state", {}))

    def reset(self, seed: int | None = None) -> None:
        """Discard run records while preserving configured previous data.

        Parameters
        ----------
        seed:
            Optional non-negative replacement seed for future provider/noise draws.
        """

        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise EnvironmentConfigurationError("reset seed must be a non-negative integer.")
            self.seed = seed
        self.generator = self._new_generator(self.seed)
        self.round = 0
        self.history.records.clear()
        self._pending_round_info = None
        self._reset_extra_state()

    def _history_state_dict(self) -> dict[str, Any]:
        """Serialize learner history and experiment-only record audits."""

        return {
            "initial_data": self._data_to_state(self.history.initial_data),
            "records": [self._record_to_state(record) for record in self.history.records],
        }

    def _history_from_state(self, value: Any) -> EnvironmentHistory:
        """Validate and reconstruct history from checkpoint state."""

        if not isinstance(value, Mapping):
            raise EnvironmentDataError("Checkpoint history must be a mapping.")
        initial_raw = value.get("initial_data")
        initial = None
        if initial_raw is not None:
            initial = self._validate_environment_data(self._data_from_state(initial_raw))
        # Checkpoints may not replace the immutable previous-data configuration.
        configured = self.history.initial_data
        if not self._environment_data_equal(initial, configured):
            raise EnvironmentConfigurationError(
                "Checkpoint previous data do not match the environment configuration."
            )
        records_raw = value.get("records")
        if not isinstance(records_raw, list):
            raise EnvironmentDataError("Checkpoint records must be a list.")
        records = [self._record_from_state(item) for item in records_raw]
        return EnvironmentHistory(initial_data=configured, records=records)

    def _validate_restored_rounds(
        self,
        records: list[EnvironmentRecord],
        round_value: int,
    ) -> None:
        """Require ordered, contiguous completed round identifiers."""

        round_indices = [record.round_index for record in records]
        if round_indices != sorted(round_indices):
            raise EnvironmentDataError("Checkpoint records are not round ordered.")
        unique = sorted(set(round_indices))
        if unique != list(range(round_value)):
            raise EnvironmentDataError(
                "Checkpoint records do not cover exactly the completed rounds."
            )

    def _record_to_state(self, record: EnvironmentRecord) -> dict[str, Any]:
        """Serialize one evaluation record."""

        return {
            "round_index": record.round_index,
            "action": record.action.detach().cpu().clone(),
            "evaluated_action": record.evaluated_action.detach().cpu().clone(),
            "observation": record.observation.detach().cpu().clone(),
            "round_info": _clone_tree(record.round_info),
            "feedback_extra": _clone_tree(record.feedback_extra),
            "context": (None if record.context is None else record.context.detach().cpu().clone()),
            "observation_noise_variance": (
                None
                if record.observation_noise_variance is None
                else record.observation_noise_variance.detach().cpu().clone()
            ),
            "dataset_index": record.dataset_index,
            "metadata": _clone_tree(record.metadata),
            "audit": _clone_tree(record.audit),
        }

    def _record_from_state(self, value: Any) -> EnvironmentRecord:
        """Validate and reconstruct one checkpoint record."""

        if not isinstance(value, Mapping):
            raise EnvironmentDataError("Each checkpoint record must be a mapping.")
        round_index = value.get("round_index")
        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
            raise EnvironmentDataError("Record round_index must be non-negative.")
        action = self.validate_action(value.get("action"))
        evaluated = self.validate_action(value.get("evaluated_action"))
        observation = _scalar_tensor(
            value.get("observation"),
            dtype=self.dtype,
            device=self.device,
            name="record observation",
            error_type=EnvironmentDataError,
        )
        context = (
            None if value.get("context") is None else self._validate_context(value.get("context"))
        )
        variance = None
        if value.get("observation_noise_variance") is not None:
            variance = _scalar_tensor(
                value.get("observation_noise_variance"),
                dtype=self.dtype,
                device=self.device,
                name="record observation_noise_variance",
            )
            if bool(variance < 0):
                raise EnvironmentDataError("Record variance cannot be negative.")
        dataset_index = value.get("dataset_index")
        if dataset_index is not None and (
            isinstance(dataset_index, bool)
            or not isinstance(dataset_index, int)
            or self.dataset is None
            or not 0 <= dataset_index < self.dataset.actions.shape[0]
        ):
            raise EnvironmentDataError("Record dataset_index is invalid.")
        mappings: dict[str, Mapping[str, Any]] = {}
        for name in ("round_info", "feedback_extra", "metadata", "audit"):
            item = value.get(name, {})
            if not isinstance(item, Mapping):
                raise EnvironmentDataError(f"Record {name} must be a mapping.")
            mappings[name] = item
        return EnvironmentRecord(
            round_index=round_index,
            action=action,
            evaluated_action=evaluated,
            observation=observation,
            round_info=_clone_tree(mappings["round_info"]),
            feedback_extra=_clone_tree(mappings["feedback_extra"]),
            context=context,
            observation_noise_variance=variance,
            dataset_index=dataset_index,
            metadata=_clone_tree(mappings["metadata"]),
            audit=_clone_tree(mappings["audit"]),
        )

    def _data_to_state(self, data: EnvironmentData | None) -> dict[str, Any] | None:
        """Serialize optional learner-visible data."""

        if data is None:
            return None
        if data.evaluated_actions is None:  # pragma: no cover - normalized at ingress
            raise EnvironmentDataError("Normalized history requires evaluated_actions.")
        return {
            "actions": data.actions.detach().cpu().clone(),
            "observations": data.observations.detach().cpu().clone(),
            "evaluated_actions": data.evaluated_actions.detach().cpu().clone(),
            "contexts": None if data.contexts is None else data.contexts.detach().cpu().clone(),
            "observation_noise_variance": (
                None
                if data.observation_noise_variance is None
                else data.observation_noise_variance.detach().cpu().clone()
            ),
            "metadata": _clone_tree(data.metadata),
        }

    @staticmethod
    def _data_from_state(value: Any) -> EnvironmentData:
        """Construct raw data from serialized fields before validation."""

        if not isinstance(value, Mapping):
            raise EnvironmentDataError("Checkpoint initial_data must be a mapping.")
        return EnvironmentData(
            actions=value.get("actions"),
            observations=value.get("observations"),
            evaluated_actions=value.get("evaluated_actions"),
            contexts=value.get("contexts"),
            observation_noise_variance=value.get("observation_noise_variance"),
            metadata=value.get("metadata", {}),
        )

    @staticmethod
    def _environment_data_equal(
        left: EnvironmentData | None,
        right: EnvironmentData | None,
    ) -> bool:
        """Compare immutable previous data used by checkpoint validation."""

        if left is None or right is None:
            return left is right
        tensor_names = (
            "actions",
            "observations",
            "evaluated_actions",
            "contexts",
            "observation_noise_variance",
        )
        for name in tensor_names:
            left_value = getattr(left, name)
            right_value = getattr(right, name)
            if left_value is None or right_value is None:
                if left_value is not right_value:
                    return False
            elif not Environment._tensor_equal_with_nan(left_value, right_value):
                return False
        return Environment._tree_equal(left.metadata, right.metadata)

    @staticmethod
    def _tensor_equal_with_nan(left: torch.Tensor, right: torch.Tensor) -> bool:
        """Compare tensors exactly while treating paired NaNs as equal."""

        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        equal = left == right
        if left.is_floating_point() or left.is_complex():
            equal |= torch.isnan(left) & torch.isnan(right)
        return bool(equal.all())

    @staticmethod
    def _tree_equal(left: Any, right: Any) -> bool:
        """Compare metadata trees without ambiguous tensor truth values."""

        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            return Environment._tensor_equal_with_nan(left, right)
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            return set(left) == set(right) and all(
                Environment._tree_equal(left[key], right[key]) for key in left
            )
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return len(left) == len(right) and all(
                Environment._tree_equal(a, b) for a, b in zip(left, right, strict=True)
            )
        try:
            comparison = left == right
            return bool(comparison)
        except (TypeError, ValueError, RuntimeError):
            return False

    def _dataset_fingerprint(self) -> str | None:
        """Hash finite source identity without serializing hidden response values."""

        if self.dataset is None:
            return None
        digest = hashlib.sha256()
        for tensor in (
            self.dataset.actions,
            self.dataset.observations,
            self.dataset.contexts,
            self.dataset.observation_noise_variance,
        ):
            if tensor is None:
                digest.update(b"none")
                continue
            cpu = tensor.detach().cpu().contiguous()
            digest.update(str(cpu.dtype).encode())
            digest.update(str(tuple(cpu.shape)).encode())
            digest.update(cpu.numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _update_fingerprint_digest(
        digest: Any,
        value: Any,
        *,
        seen: set[int] | None = None,
    ) -> None:
        """Add a deterministic provider/configuration value to ``digest``.

        The encoder includes callable defaults and closure contents in addition
        to source text.  This distinguishes, for example, two objective
        factories that share bytecode but capture different constants, without
        serializing the callable itself into a checkpoint.
        """

        active = set() if seen is None else seen
        identity = id(value)
        if identity in active:
            digest.update(b"cycle")
            return

        if value is None or isinstance(value, (str, int, float, bool)):
            digest.update(f"{type(value).__qualname__}:{value!r}".encode())
            return
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
            return
        if (
            type(value).__module__.startswith("numpy")
            and hasattr(value, "dtype")
            and hasattr(value, "shape")
            and hasattr(value, "tobytes")
        ):
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.tobytes())
            return
        if isinstance(value, Mapping):
            active.add(identity)
            try:
                for key in sorted(value, key=lambda item: repr(item)):
                    Environment._update_fingerprint_digest(digest, key, seen=active)
                    Environment._update_fingerprint_digest(digest, value[key], seen=active)
            finally:
                active.remove(identity)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            active.add(identity)
            try:
                items = list(value)
                if isinstance(value, (set, frozenset)):
                    items.sort(key=repr)
                for item in items:
                    Environment._update_fingerprint_digest(digest, item, seen=active)
            finally:
                active.remove(identity)
            return
        if callable(value):
            active.add(identity)
            try:
                callable_identity = (
                    getattr(value, "__module__", None) or type(value).__module__,
                    getattr(value, "__qualname__", None) or type(value).__qualname__,
                )
                digest.update(".".join(callable_identity).encode())
                try:
                    digest.update(inspect.getsource(value).encode())
                except (OSError, TypeError):
                    code = getattr(value, "__code__", None)
                    if code is not None:
                        digest.update(code.co_code)
                        Environment._update_fingerprint_digest(digest, code.co_consts, seen=active)
                Environment._update_fingerprint_digest(
                    digest, getattr(value, "__defaults__", None), seen=active
                )
                Environment._update_fingerprint_digest(
                    digest, getattr(value, "__kwdefaults__", None), seen=active
                )
                closure = getattr(value, "__closure__", None)
                if closure:
                    for cell in closure:
                        try:
                            contents = cell.cell_contents
                        except ValueError:
                            digest.update(b"empty_closure_cell")
                        else:
                            Environment._update_fingerprint_digest(digest, contents, seen=active)
                attributes = getattr(value, "__dict__", None)
                if isinstance(attributes, Mapping):
                    Environment._update_fingerprint_digest(digest, attributes, seen=active)
            finally:
                active.remove(identity)
            return
        digest.update(f"{type(value).__module__}.{type(value).__qualname__}".encode())

    def _compute_source_fingerprint(self) -> str:
        """Hash the configured evaluation/provider identities.

        Checkpoints intentionally omit callables and hidden source rows.  Their
        stable identities and source text (when available) are hashed instead,
        so a resume cannot silently attach learner state to a different
        objective, attack, context process, or distribution provider.
        """

        digest = hashlib.sha256()
        digest.update(f"{type(self).__module__}.{type(self).__qualname__}".encode())
        digest.update(self.source_mode.encode())
        provider_names = (
            "function",
            "noise_fn",
            "attack",
            "context_provider",
            "reference_distribution_provider",
            "sampling_distribution_provider",
            "margin_provider",
        )
        for name in provider_names:
            if not hasattr(self, name):
                continue
            provider = getattr(self, name)
            digest.update(name.encode())
            self._update_fingerprint_digest(digest, provider)
        support = getattr(self, "context_support", None)
        if isinstance(support, torch.Tensor):
            cpu = support.detach().cpu().contiguous()
            digest.update(str(cpu.dtype).encode())
            digest.update(str(tuple(cpu.shape)).encode())
            digest.update(cpu.numpy().tobytes())
        return digest.hexdigest()

    def _freeze_source_fingerprint(self) -> None:
        """Capture immutable provider identity before any provider is invoked."""

        self._frozen_source_fingerprint = self._compute_source_fingerprint()

    def _source_fingerprint(self) -> str:
        """Return the construction-time source/configuration fingerprint."""

        if self._frozen_source_fingerprint is None:
            self._freeze_source_fingerprint()
        assert self._frozen_source_fingerprint is not None
        return self._frozen_source_fingerprint

    def _validate_context(self, context: Any) -> torch.Tensor:
        """Convert a single finite context vector."""

        try:
            result = as_tensor(
                context,
                dtype=self.dtype,
                device=self.device,
                copy=True,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise EnvironmentDataError("context must be numeric.") from exc
        if result.ndim == 0:
            result = result.unsqueeze(0)
        if result.ndim != 1 or result.numel() == 0:
            raise EnvironmentDataError("context must be a non-empty vector.")
        if not bool(torch.isfinite(result).all()):
            raise EnvironmentDataError("context contains non-finite values.")
        expected_dimension = self._context_dimension()
        if expected_dimension is not None and result.shape[0] != expected_dimension:
            raise EnvironmentDataError(f"context must have shape ({expected_dimension},).")
        return result

    def _context_dimension(self) -> int | None:
        """Infer a fixed context dimension from configured finite/history data."""

        if self.dataset is not None and self.dataset.contexts is not None:
            return int(self.dataset.contexts.shape[1])
        initial = self.history.initial_data if hasattr(self, "history") else None
        if initial is not None and initial.contexts is not None:
            return int(initial.contexts.shape[1])
        return None

    @staticmethod
    def _invoke_provider(
        provider: Callable[..., Any],
        *,
        primary_name: str | None,
        primary_value: Any | None,
        **available: Any,
    ) -> Any:
        """Call a provider with only the reproducibility inputs it declares.

        Providers can use the documented keyword names or a single positional
        primary value (for example ``lambda x: ...`` for an attack).  Signature
        inspection avoids catching and misclassifying exceptions raised inside
        provider code.
        """

        try:
            signature = inspect.signature(provider)
        except (TypeError, ValueError):
            if primary_name is None:
                return provider(**available)
            return provider(primary_value, **available)
        parameters = signature.parameters
        accepts_extra = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        kwargs = {
            name: value
            for name, value in available.items()
            if accepts_extra
            or (
                name in parameters
                and parameters[name].kind
                in {
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                }
            )
        }
        args: list[Any] = []
        if primary_name is not None:
            parameter = parameters.get(primary_name)
            if accepts_extra or (
                parameter is not None
                and parameter.kind
                in {
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                }
            ):
                kwargs[primary_name] = primary_value
            else:
                args.append(primary_value)
        return provider(*args, **kwargs)

    def _extra_state_dict(self) -> dict[str, Any]:
        """Return subclass round-local state for checkpoints."""

        return {}

    def _load_extra_state_dict(self, state: Any) -> None:
        """Restore subclass state after common validation."""

        if not isinstance(state, Mapping):
            raise EnvironmentDataError("extra_state must be a mapping.")

    def _reset_extra_state(self) -> None:
        """Reset subclass round-local state."""

        return None


__all__ = [
    "MATCH_ATOL",
    "AvailableActions",
    "Environment",
    "EnvironmentConfigurationError",
    "EnvironmentData",
    "EnvironmentDataError",
    "EnvironmentDataset",
    "EnvironmentError",
    "EnvironmentHistory",
    "EnvironmentLookupError",
    "EnvironmentProtocolError",
    "EnvironmentRecord",
    "InvalidActionError",
    "NoAvailableActionError",
    "NoiseProvider",
    "NonFiniteObservationError",
]
