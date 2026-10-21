"""Adversarial action-perturbation environment."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import torch

from basat.models.utils import Bounds
from basat.workflow import RandomStream

from .environment import (
    Environment,
    AvailableActions,
    EnvironmentConfigurationError,
    EnvironmentData,
    EnvironmentDataError,
    EnvironmentDataset,
    InvalidActionError,
    EnvironmentLookupError,
    EnvironmentProtocolError,
    NoAvailableActionError,
    NoiseProvider,
    _Evaluation,
    _clone_tree,
)

_SOURCE_NORMAL_POLICY = "advers-numpy126-scalar-polar-v1"


@runtime_checkable
class AttackProvider(Protocol):
    """Protocol for reproducible adversarial perturbation providers."""

    def __call__(
        self,
        *,
        action: torch.Tensor,
        round_index: int,
        generator: torch.Generator,
    ) -> Any:
        """Return a perturbation vector with shape ``(D_env,)``."""


@dataclass(frozen=True)
class LCBAttack:
    """Choose the first finite-domain LCB minimum within a Euclidean radius.

    Parameters
    ----------
    radius : float
        Nonnegative physical-coordinate perturbation radius. Boundary points
        are included. Equal lower bounds are resolved in candidate row order.

    Notes
    -----
    The immutable configuration holds no learner or objective reference.
    ``Experiment`` supplies the active learner's ``confidence_bounds`` method
    at evaluation time; that method must return ``(upper, lower)`` vectors.
    Standalone callers can supply it to ``AdversarialEnvironment.step``.

    Examples
    --------
    >>> import basat
    >>> attack = basat.LCBAttack(radius=0.5)
    >>> attack.radius
    0.5
    """

    radius: float

    def __post_init__(self) -> None:
        if isinstance(self.radius, bool):
            raise EnvironmentConfigurationError("LCBAttack radius must be finite and nonnegative.")
        try:
            radius = float(self.radius)
        except (TypeError, ValueError) as exc:
            raise EnvironmentConfigurationError(
                "LCBAttack radius must be finite and nonnegative."
            ) from exc
        if not math.isfinite(radius) or radius < 0.0:
            raise EnvironmentConfigurationError("LCBAttack radius must be finite and nonnegative.")
        object.__setattr__(self, "radius", radius)

    def __call__(
        self,
        *,
        action: torch.Tensor,
        candidate_actions: torch.Tensor,
        confidence_bounds: Callable[[torch.Tensor], tuple[Any, Any]],
    ) -> torch.Tensor:
        """Return the displacement to the lowest-LCB admissible candidate."""
        if not callable(confidence_bounds):
            raise EnvironmentProtocolError("LCBAttack requires a runtime confidence_bounds callback.")
        # Match the released attack's distance arithmetic and row ordering.
        # This numerical helper does not inspect objective values.
        from sklearn.metrics.pairwise import euclidean_distances

        distances = euclidean_distances(
            candidate_actions.detach().cpu().numpy(),
            action.detach().cpu().numpy().reshape(1, -1),
        ).reshape(-1)
        within = torch.as_tensor(distances <= self.radius, device=candidate_actions.device)
        indices = torch.nonzero(within, as_tuple=False).flatten()
        if indices.numel() == 0:
            raise InvalidActionError("LCBAttack has no candidate inside its radius.")
        result = confidence_bounds(candidate_actions.detach().clone())
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise EnvironmentProtocolError("confidence_bounds must return (upper, lower).")
        lower = torch.as_tensor(result[1], dtype=candidate_actions.dtype,
                                device=candidate_actions.device).reshape(-1)
        if lower.numel() != candidate_actions.shape[0] or not bool(torch.isfinite(lower).all()):
            raise EnvironmentProtocolError("confidence_bounds must return one finite LCB per candidate.")
        selected = indices[int(torch.argmin(lower[indices]))]
        calculation_device = "cpu" if action.device.type == "mps" else action.device
        return (candidate_actions[selected].to(dtype=torch.float64, device=calculation_device)
                - action.to(dtype=torch.float64, device=calculation_device))


@dataclass(frozen=True)
class RandomAttack:
    """Uniformly choose an admissible finite-domain row, without rewards.

    ``radius`` is physical. ``squared_distance`` preserves the released
    AdveRS squared-Euclidean arithmetic instead of rewriting its comparison.
    A supplied source stream owns the choice; ordinary calls use the
    environment's Torch generator. ``source_epsilon_floor`` is an explicit
    opt-in floor in distance-metric units for the released nonpositive-budget
    branch. Neither objective values nor a learner reference are accepted.
    """

    radius: float
    squared_distance: bool = False
    source_epsilon_floor: float = 0.0

    def __post_init__(self) -> None:
        for name in ("radius", "source_epsilon_floor"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise EnvironmentConfigurationError(f"RandomAttack {name} must be finite and nonnegative.")
            try:
                value = float(value)
            except (ValueError, TypeError) as error:
                raise EnvironmentConfigurationError(f"RandomAttack {name} must be finite and nonnegative.") from error
            if not math.isfinite(value) or value < 0:
                raise EnvironmentConfigurationError(f"RandomAttack {name} must be finite and nonnegative.")
            object.__setattr__(self, name, value)
        if not isinstance(self.squared_distance, bool):
            raise EnvironmentConfigurationError("RandomAttack squared_distance must be Boolean.")

    def __call__(self, *, action: torch.Tensor, candidate_actions: torch.Tensor,
                 source_random_stream: RandomStream | None = None,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        from sklearn.metrics.pairwise import euclidean_distances
        import numpy as np

        threshold = self.radius*self.radius if self.squared_distance else self.radius
        if threshold <= 0:
            threshold = self.source_epsilon_floor
        distances = euclidean_distances(candidate_actions.detach().cpu().numpy(),
            action.detach().cpu().numpy().reshape(1, -1), squared=self.squared_distance).reshape(-1)
        indices = np.where(distances <= threshold)[0]
        if not len(indices):
            raise InvalidActionError("RandomAttack has no candidate inside its radius.")
        if source_random_stream is None:
            device = "cpu" if generator is None else generator.device
            offset = int(torch.randint(len(indices), (), generator=generator, device=device))
            selected = int(indices[offset])
        else:
            if not isinstance(source_random_stream, RandomStream):
                raise EnvironmentConfigurationError("RandomAttack source_random_stream must be a RandomStream.")
            selected = int(source_random_stream.choice(indices))
        calculation_device = "cpu" if action.device.type == "mps" else action.device
        return (candidate_actions[selected].to(dtype=torch.float64, device=calculation_device)
                - action.to(dtype=torch.float64, device=calculation_device))


@dataclass(frozen=True)
class WorstCaseAttack:
    """Attack the first true-payoff minimum within a physical Euclidean radius.

    This is a finite-domain, noiseless-objective attacker, not a lower
    confidence bound attacker. Equal payoffs retain candidate row order. The
    environment privately prepares and checkpoints the ordered payoff table;
    neither that table nor a live objective reference is stored in this
    immutable configuration or exposed in learner round information.

    When an MT19937 source stream is explicitly supplied during evaluation,
    observation noise uses the scalar legacy polar-normal operation order
    validated against NumPy 1.26.4. Ordinary environment noise is unchanged.
    """

    radius: float
    implementation_version: int = field(default=2, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.radius, bool):
            raise EnvironmentConfigurationError("WorstCaseAttack radius must be finite and nonnegative.")
        try:
            radius = float(self.radius)
        except (TypeError, ValueError) as exc:
            raise EnvironmentConfigurationError(
                "WorstCaseAttack radius must be finite and nonnegative."
            ) from exc
        if not math.isfinite(radius) or radius < 0:
            raise EnvironmentConfigurationError("WorstCaseAttack radius must be finite and nonnegative.")
        object.__setattr__(self, "radius", radius)

    def __call__(
        self, *, action: torch.Tensor, candidate_actions: torch.Tensor,
        true_values: Any,
    ) -> torch.Tensor:
        """Return a displacement using explicitly supplied evaluation-only truth.

        The distance arithmetic, inclusive boundary, row ordering and first
        ``argmin`` follow ``adversarial_models.worst_case_attack`` in AdveRS
        revision 71a3fe89c7d051b7646d048fb6fc314e74a33095.
        """
        from sklearn.metrics.pairwise import euclidean_distances

        values = torch.as_tensor(true_values, dtype=candidate_actions.dtype,
                                 device=candidate_actions.device).reshape(-1)
        if values.numel() != len(candidate_actions) or not bool(torch.isfinite(values).all()):
            raise EnvironmentProtocolError("WorstCaseAttack requires one finite true payoff per candidate.")
        distances = euclidean_distances(
            candidate_actions.detach().cpu().numpy(),
            action.detach().cpu().numpy().reshape(1, -1),
        ).reshape(-1)
        indices = torch.nonzero(torch.as_tensor(
            distances <= self.radius, device=candidate_actions.device
        ), as_tuple=False).flatten()
        if indices.numel() == 0:
            raise InvalidActionError("WorstCaseAttack has no candidate inside its radius.")
        selected = indices[int(torch.argmin(values[indices]))]
        calculation_device = "cpu" if action.device.type == "mps" else action.device
        return (candidate_actions[selected].to(dtype=torch.float64, device=calculation_device)
                - action.to(dtype=torch.float64, device=calculation_device))


def _source_mt19937_normal(stream: RandomStream) -> float:
    """Legacy scalar polar normal with explicit floating-point operation order.

    NumPy's legacy algorithm uses the same MT19937 words in newer runtimes,
    but compiler contraction can change normal values/cached state by a final
    bit. Scalar Python operations reproduce the NumPy 1.26.4 source-reference
    runtime, including the cached Gaussian, without changing RandomStream's
    ordinary implementation or advancing additional uniform draws.
    """
    state = stream.get_state()
    if state.get("bit_generator") != "MT19937":
        raise EnvironmentProtocolError("The released normal sampler requires MT19937.")
    if state["has_gaussian"]:
        value = state["cached_gaussian"]
        state["has_gaussian"] = 0
        state["cached_gaussian"] = 0.0
        stream.set_state(state)
        return float(value)
    while True:
        first = 2.0 * stream.uniform() - 1.0
        second = 2.0 * stream.uniform() - 1.0
        squared_radius = first * first + second * second
        if squared_radius < 1.0 and squared_radius != 0.0:
            break
    factor = math.sqrt(-2.0 * math.log(squared_radius) / squared_radius)
    state = stream.get_state()
    state["has_gaussian"] = 1
    state["cached_gaussian"] = factor * first
    stream.set_state(state)
    return factor * second


class AdversarialEnvironment(Environment):
    """Apply an attack before evaluating the requested action.

    The requested action remains in ``action`` while ``evaluated_action`` is
    ``action + perturbation``.  No clipping or interpolation is performed.  In
    data-backed mode, both the nominal action and the attacked coordinate must
    satisfy their respective exact finite-table contracts.

    Parameters
    ----------
    bounds : Bounds
        Physical-coordinate bounds for both requested and attacked actions.
    attack : callable, LCBAttack, RandomAttack or WorstCaseAttack
        Perturbation provider. ``LCBAttack`` obtains model confidence values
        from the active learner at each ``Experiment`` evaluation.
        ``WorstCaseAttack`` uses the private finite noiseless payoff table.
    function : callable, optional
        Scalar objective. Online observations evaluate only the attacked
        coordinate. ``WorstCaseAttack`` additionally prepares a private
        noiseless finite payoff cache, counted separately from observations.
    candidate_actions : array-like, optional
        Ordered finite domain for a callable objective, shape ``(N, D)``.
        Only coordinates and stable row indices are exposed to the learner.
        Requested and attacked actions must match stored coordinates under
        Basat's finite-domain absolute tolerance; no interpolation occurs.
    dataset : EnvironmentDataset, optional
        Alternative hidden finite response table; mutually exclusive with
        ``function`` and ``candidate_actions``.
    previous_data : EnvironmentData, optional
        Already observed actions and rewards supplied by the caller.
    dtype, device : optional
        Storage precision and device; default float32 on CPU.
    noise_std : float, optional
        Standard deviation of Gaussian observation noise, default zero.
    noise_fn : callable, optional
        Custom noise provider replacing the Gaussian model.
    add_noise_to_data : bool, optional
        Whether to add synthetic noise to stored dataset observations.
    allow_repeated_actions : bool, optional
        Whether nominal finite actions remain available after evaluation.
    data_row_selection : {"first", "random"}, optional
        Selection rule for duplicate rows in a hidden finite response table.
    seed : int, optional
        Seed for the ordinary environment-owned random generator.

    Examples
    --------
    >>> import torch
    >>> import basat
    >>> bounds = basat.Bounds(lower=[-1.0], upper=[1.0])
    >>> def attack(*, action, round_index, generator):
    ...     return torch.zeros_like(action)
    >>> environment = basat.AdversarialEnvironment(
    ...     bounds=bounds, attack=attack,
    ...     function=lambda x: -(x.square().sum()),
    ...     dtype="float64", device="cpu", noise_std=0.0, seed=42,
    ... )
    """

    environment_type = "adversarial"
    provided_feedback = frozenset({"observation", "evaluated_action", "perturbation"})
    capabilities = frozenset({"adversarial", "perturbed_actions"})

    def __init__(
        self,
        *,
        bounds: Bounds,
        attack: AttackProvider | Callable[..., Any],
        function: Callable[..., Any] | None = None,
        candidate_actions: Any | None = None,
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
        """Initialize the source and its required reproducible attack provider."""

        if not callable(attack):
            raise EnvironmentConfigurationError("attack must be callable.")
        self.attack = attack
        self._candidate_actions: torch.Tensor | None = None
        self._runtime_confidence_bounds: Callable[..., Any] | None = None
        self._runtime_source_random_stream: RandomStream | None = None
        self._runtime_source_normal_policy = "native"
        self._worst_case_values: list[float] = []
        self._attack_oracle_evaluations = 0
        super().__init__(
            bounds=bounds,
            function=function,
            dataset=dataset,
            previous_data=previous_data,
            dtype=dtype,
            device=device,
            noise_std=noise_std,
            noise_fn=noise_fn,
            add_noise_to_data=add_noise_to_data,
            allow_repeated_actions=allow_repeated_actions,
            data_row_selection=data_row_selection,
            seed=seed,
        )
        if self.dataset is not None and self.dataset.contexts is not None:
            raise EnvironmentDataError("An adversarial finite dataset must not contain contexts.")
        initial = self.history.initial_data
        if initial is not None and initial.contexts is not None:
            raise EnvironmentDataError("Adversarial previous_data must not contain contexts.")
        if candidate_actions is not None:
            if self.source_mode != "function":
                raise EnvironmentConfigurationError(
                    "candidate_actions is only supported with a function; datasets already define candidates."
                )
            candidates = self._coerce_action_matrix(candidate_actions, name="candidate_actions")
            if candidates.shape[0] == 0:
                raise EnvironmentDataError("candidate_actions cannot be empty.")
            for index in range(candidates.shape[0]):
                if bool(self._row_match(candidates[index + 1:], candidates[index]).any()):
                    raise EnvironmentDataError("candidate_actions must contain distinct coordinates.")
            self._candidate_actions = candidates
            if initial is not None:
                for action in (*initial.actions, *initial.evaluated_actions):
                    self.validate_action(action)
        if isinstance(self.attack, (LCBAttack, RandomAttack, WorstCaseAttack)) and self._candidate_actions is None and self.dataset is None:
            raise EnvironmentConfigurationError(
                f"{type(self.attack).__name__} requires candidate_actions or a finite dataset."
            )
        if isinstance(self.attack, WorstCaseAttack) and self.dataset is not None:
            for index, action in enumerate(self.dataset.actions):
                if bool(self._row_match(self.dataset.actions[index + 1:], action).any()):
                    raise EnvironmentConfigurationError(
                        "WorstCaseAttack requires distinct finite dataset coordinates."
                    )
        self._freeze_source_fingerprint()

    @property
    def candidate_actions(self) -> torch.Tensor | None:
        """Return finite function-domain coordinates without objective values."""
        return None if self._candidate_actions is None else self._candidate_actions.detach().clone()

    def _attack_payoff_checksum(self, values: list[float]) -> str:
        digest = hashlib.sha256(b"basat-worst-case-cache-v2")
        self._update_fingerprint_digest(digest, self._source_fingerprint())
        self._update_fingerprint_digest(digest, self._dataset_fingerprint())
        self._update_fingerprint_digest(digest, values)
        return digest.hexdigest()

    @property
    def attack_oracle_metadata(self) -> dict[str, Any]:
        """Return cache identity/accounting without revealing unobserved rewards.

        Preparation calls are separate from initialization and online queries.
        Reset keeps the immutable cache but resets this run's preparation count;
        callers that prewarm a shared environment should retain its metadata.
        ``cache_total_unique_function_preparation_evaluations`` is the retained
        cumulative cache provenance, including earlier partial preparation.
        It does not count cache reuse as new calls and is not reset with a run.
        """
        candidates = self._candidate_actions if self._candidate_actions is not None else (
            None if self.dataset is None else self.dataset.actions
        )
        return {
            "source_fingerprint": self._source_fingerprint(),
            "dataset_fingerprint": self._dataset_fingerprint(),
            "payoff_checksum": self._attack_payoff_checksum(self._worst_case_values),
            "preparation_evaluations": self._attack_oracle_evaluations,
            "cache_total_unique_function_preparation_evaluations": (
                len(self._worst_case_values) if self.source_mode == "function" else 0
            ),
            "cached_points": len(self._worst_case_values),
            "total_points": 0 if candidates is None else len(candidates),
        }

    def prepare_attack_cache(self, *, cache_directory: str | Path | None = None) -> dict[str, Any]:
        """Privately tabulate the finite noiseless objective for WorstCaseAttack.

        Evaluation order is candidate order, without RNG consumption. A cache
        restored from a checkpoint is reused, including any saved partial
        preparation. Function evaluations here are preparation calls, never GP
        training rows. Data-backed tables are read without objective calls.
        ``cache_directory`` persists an atomic fingerprint/checksum-validated
        artifact so reconstructing a completed notebook performs no new oracle
        calls. Use separate directories for different environment identities;
        an incompatible existing artifact is rejected rather than overwritten.
        """
        if not isinstance(self.attack, WorstCaseAttack):
            raise EnvironmentProtocolError("prepare_attack_cache requires WorstCaseAttack.")
        store = None
        artifact_name = "worst_case_payoffs.json"
        evaluations_before = self._attack_oracle_evaluations
        cache_hit = False
        disk_points = 0
        if cache_directory is not None:
            from basat.workflow import ArtifactStore
            store = ArtifactStore(str(cache_directory))
            if store.exists(artifact_name):
                existing = store.read_json(artifact_name)
                values, _ = self._validated_worst_case_cache(existing)
                disk_points = len(values)
                overlap = min(disk_points, len(self._worst_case_values))
                if values[:overlap] != self._worst_case_values[:overlap]:
                    raise EnvironmentDataError(
                        "Worst-case disk and in-memory payoff caches conflict."
                    )
                # A checkpoint may have a longer prefix than an older disk
                # artifact, or vice versa. Preserve the longest validated
                # prefix without repeating calls or losing run accounting.
                if disk_points > len(self._worst_case_values):
                    self._worst_case_values = list(values)
                cache_hit = True
        candidates = self._candidate_actions if self._candidate_actions is not None else self.dataset.actions
        loaded_points = len(self._worst_case_values)
        try:
            while len(self._worst_case_values) < len(candidates):
                index = len(self._worst_case_values)
                if self.source_mode == "function":
                    value = float(self.evaluate_function(candidates[index]))
                    self._attack_oracle_evaluations += 1
                else:
                    value = float(self.dataset.observations[index])
                self._worst_case_values.append(value)
        except BaseException:
            # Keep completed oracle calls recoverable even when preparation is
            # interrupted before any learner observation or experiment save.
            if store is not None:
                store.write_json(artifact_name, self._extra_state_dict())
            raise
        if store is not None and (not cache_hit or len(self._worst_case_values) > disk_points):
            store.write_json(artifact_name, self._extra_state_dict())
        return {
            **self.attack_oracle_metadata,
            "prepared_this_call": len(self._worst_case_values) > loaded_points,
            "objective_evaluations_this_call": self._attack_oracle_evaluations - evaluations_before,
            "disk_cache_hit": cache_hit,
        }

    def _extra_state_dict(self) -> dict[str, Any]:
        if not isinstance(self.attack, WorstCaseAttack):
            return super()._extra_state_dict()
        return {"worst_case_cache": {
            "version": 2, **self.attack_oracle_metadata,
            "values": list(self._worst_case_values),
        }}

    def _load_extra_state_dict(self, state: Any) -> None:
        super()._load_extra_state_dict(state)
        if not isinstance(self.attack, WorstCaseAttack):
            if "worst_case_cache" in state:
                raise EnvironmentDataError("Unexpected worst-case attack cache.")
            return
        values, count = self._validated_worst_case_cache(state)
        self._worst_case_values = list(values)
        self._attack_oracle_evaluations = count

    def _validated_worst_case_cache(self, state: Any) -> tuple[list[float], int]:
        """Validate a cache artifact completely without changing live state."""
        if not isinstance(state, Mapping):
            raise EnvironmentDataError("Worst-case cache state must be a mapping.")
        cache = state.get("worst_case_cache")
        candidates = self._candidate_actions if self._candidate_actions is not None else self.dataset.actions
        if not isinstance(cache, Mapping):
            raise EnvironmentDataError("Missing worst-case attack cache state.")
        values = cache.get("values")
        count = cache.get("preparation_evaluations")
        if (
            cache.get("version") != 2
            or cache.get("source_fingerprint") != self._source_fingerprint()
            or cache.get("dataset_fingerprint") != self._dataset_fingerprint()
            or not isinstance(values, list)
            or len(values) > len(candidates)
            or any(isinstance(v, bool) or not isinstance(v, (float, int))
                   or not math.isfinite(v) for v in values)
            or cache.get("cached_points") != len(values)
            or cache.get("total_points") != len(candidates)
            or cache.get("cache_total_unique_function_preparation_evaluations") != (
                len(values) if self.source_mode == "function" else 0
            )
            or cache.get("payoff_checksum") != self._attack_payoff_checksum(values)
            or isinstance(count, bool) or not isinstance(count, int)
            or not 0 <= count <= len(values)
            or (self.source_mode == "data" and count != 0)
        ):
            raise EnvironmentDataError("Invalid worst-case attack cache state or checksum.")
        return list(values), count

    def _reset_extra_state(self) -> None:
        # Cached objective truth is immutable; only per-run accounting resets.
        self._attack_oracle_evaluations = 0

    def _compute_source_fingerprint(self) -> str:
        fingerprint = super()._compute_source_fingerprint()
        if self._candidate_actions is None:
            return fingerprint
        digest = hashlib.sha256(fingerprint.encode())
        self._update_fingerprint_digest(digest, self._candidate_actions)
        return digest.hexdigest()

    def validate_action(self, action: Any) -> torch.Tensor:
        result = super().validate_action(action)
        if self._candidate_actions is not None:
            indices = torch.nonzero(self._row_match(self._candidate_actions, result),
                                    as_tuple=False).flatten()
            if indices.numel() != 1:
                raise EnvironmentLookupError("The action is not an exact finite-domain candidate.")
            return self._candidate_actions[int(indices[0])].detach().clone()
        return result

    def available_actions(self, *, context: Any | None = None) -> AvailableActions:
        """Return finite candidates and their stable indices, never rewards."""
        if self._candidate_actions is None:
            return super().available_actions(context=context)
        if context is not None:
            raise EnvironmentProtocolError("An adversarial finite domain does not accept contexts.")
        actions = self._candidate_actions
        indices = torch.arange(len(actions), device=self.device, dtype=torch.int64)
        if not self.allow_repeated_actions:
            actions, indices = self._remove_observed_actions(
                actions=actions, indices=indices, observed=self.get_history_data()
            )
        if len(actions) == 0:
            raise NoAvailableActionError("The finite candidate set is empty.")
        return AvailableActions(actions=actions.detach().clone(), dataset_indices=indices.clone())

    def before_action(self) -> dict[str, Any]:
        if self._candidate_actions is None:
            return super().before_action()
        self._source_fingerprint()
        if self._pending_round_info is None:
            available = self.available_actions()
            self._pending_round_info = {
                "candidate_actions": available.actions,
                "candidate_dataset_indices": available.dataset_indices,
            }
        return _clone_tree(self._pending_round_info)

    def step(
        self, action: Any, *, confidence_bounds: Callable[..., Any] | None = None,
        source_random_stream: RandomStream | None = None,
        source_normal_policy: Literal["native", "advers-numpy126-scalar-polar-v1"] = "native",
    ) -> dict[str, Any]:
        """Evaluate with optional temporary learner confidence and source RNG.

        The runtime callback and stream are not retained or fingerprinted.
        The caller owns stream checkpointing. A failed evaluation restores the
        supplied stream exactly, including its cached Gaussian draw.
        ``source_normal_policy`` explicitly opts into the NumPy 1.26 scalar
        normal arithmetic and feedback RNG audit. The camera-ready evaluation
        protocol enables it for LCB attacks and checkpoints that policy; ordinary
        and legacy LCB calls retain their native noise arithmetic.
        """
        if source_random_stream is not None and not isinstance(source_random_stream, RandomStream):
            raise EnvironmentConfigurationError("source_random_stream must be a RandomStream.")
        if source_random_stream is not None and self.noise_fn is not None:
            raise EnvironmentConfigurationError("source_random_stream requires the default Gaussian noise model.")
        if source_normal_policy not in ("native", _SOURCE_NORMAL_POLICY):
            raise EnvironmentConfigurationError("Unknown source_normal_policy.")
        if source_normal_policy == _SOURCE_NORMAL_POLICY and (
            source_random_stream is None
            or source_random_stream.get_state().get("bit_generator") != "MT19937"
        ):
            raise EnvironmentConfigurationError("The source normal policy requires an MT19937 source_random_stream.")
        state = None if source_random_stream is None else source_random_stream.get_state()
        previous = (self._runtime_confidence_bounds, self._runtime_source_random_stream,
                    self._runtime_source_normal_policy)
        self._runtime_confidence_bounds = confidence_bounds
        self._runtime_source_random_stream = source_random_stream
        self._runtime_source_normal_policy = source_normal_policy
        try:
            return super().step(action)
        except BaseException as error:
            if source_random_stream is not None and (
                isinstance(error, Exception) or source_normal_policy == _SOURCE_NORMAL_POLICY
            ):
                source_random_stream.set_state(state)
            raise
        finally:
            (self._runtime_confidence_bounds, self._runtime_source_random_stream,
             self._runtime_source_normal_policy) = previous

    def sample_noise(self, *, action: torch.Tensor | None = None,
                     context: torch.Tensor | None = None) -> torch.Tensor:
        if self._runtime_source_random_stream is None:
            return super().sample_noise(action=action, context=context)
        stream = self._runtime_source_random_stream
        if (isinstance(self.attack, WorstCaseAttack)
                or self._runtime_source_normal_policy == _SOURCE_NORMAL_POLICY
        ) and stream.get_state().get("bit_generator") == "MT19937":
            draw = 0.0 + self.noise_std * _source_mt19937_normal(stream)
        else:
            draw = stream.normal(scale=self.noise_std)
        return torch.as_tensor(draw,
                               dtype=self.dtype, device=self.device)

    def _attack_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve and validate one perturbation and evaluated action."""

        if isinstance(self.attack, LCBAttack):
            candidates = (self._candidate_actions if self._candidate_actions is not None
                          else self.dataset.actions)
            raw = self.attack(action=action, candidate_actions=candidates,
                              confidence_bounds=self._runtime_confidence_bounds)
        elif isinstance(self.attack, RandomAttack):
            candidates = (self._candidate_actions if self._candidate_actions is not None
                          else self.dataset.actions)
            raw = self.attack(action=action, candidate_actions=candidates,
                              source_random_stream=self._runtime_source_random_stream,
                              generator=self.generator)
        elif isinstance(self.attack, WorstCaseAttack):
            self.prepare_attack_cache()
            candidates = (self._candidate_actions if self._candidate_actions is not None
                          else self.dataset.actions)
            raw = self.attack(action=action, candidate_actions=candidates,
                              true_values=self._worst_case_values)
        else:
            raw = self._invoke_provider(
                self.attack,
                primary_name="action",
                primary_value=action.detach().clone(),
                round_index=self.round,
                generator=self.generator,
            )
        try:
            calculation_device = "cpu" if self.device.type == "mps" else self.device
            precise_perturbation = (
                torch.as_tensor(raw, dtype=torch.float64, device=calculation_device)
                .detach()
                .clone()
            )
            perturbation = precise_perturbation.to(dtype=self.dtype, device=self.device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise InvalidActionError("attack must return a numeric vector.") from exc
        if perturbation.shape != action.shape:
            raise InvalidActionError(f"attack must return shape ({self.action_dimension},).")
        if not bool(torch.isfinite(perturbation).all()):
            raise InvalidActionError("attack returned a non-finite perturbation.")
        # A finite attack may return a precise difference to a stored point.
        # Casting that difference before adding can lose the destination by
        # more than the exact-table tolerance in float32. Round once, after
        # adding, without relaxing the lookup or bounds contract.
        evaluated_action = self.validate_action(
            action.to(device=calculation_device, dtype=torch.float64) + precise_perturbation
        )
        return perturbation, evaluated_action

    def _evaluate_one(self, action: torch.Tensor) -> _Evaluation:
        """Prepare an attacked evaluation without committing history."""

        if self.source_mode == "data" or self._candidate_actions is not None:
            self._assert_action_available(action)
        perturbation, evaluated_action = self._attack_action(action)

        if self.source_mode == "function":
            latent = self.evaluate_function(evaluated_action)
            realized_noise = self.sample_noise(action=evaluated_action)
            observation = latent + realized_noise
            variance = (
                torch.as_tensor(self.noise_std**2, dtype=self.dtype, device=self.device)
                if self.noise_fn is None
                else None
            )
            row = None
            audit: dict[str, Any] = {
                "latent_observation": latent,
                "realized_noise": realized_noise,
                "attack_perturbation": perturbation,
            }
            if self._candidate_actions is not None:
                audit["candidate_index"] = int(torch.nonzero(
                    self._row_match(self._candidate_actions, action), as_tuple=False
                )[0, 0])
                audit["evaluated_candidate_index"] = int(torch.nonzero(
                    self._row_match(self._candidate_actions, evaluated_action), as_tuple=False
                )[0, 0])
        else:
            stored, variance, row = self.lookup_dataset_observation(action=evaluated_action)
            observation, added_noise = self.make_data_observation(stored, action=evaluated_action)
            if self.add_noise_to_data:
                if self.noise_fn is None:
                    added_variance = torch.as_tensor(
                        self.noise_std**2, dtype=self.dtype, device=self.device
                    )
                    variance = added_variance if variance is None else variance + added_variance
                else:
                    variance = None
            audit = {
                "stored_observation": stored,
                "realized_noise": added_noise,
                "attack_perturbation": perturbation,
            }
        if (isinstance(self.attack, WorstCaseAttack)
                or self._runtime_source_normal_policy == _SOURCE_NORMAL_POLICY
        ) and self._runtime_source_random_stream is not None:
            source_state = self._runtime_source_random_stream.get_state()
            audit["source_rng_after_feedback_sha256"] = hashlib.sha256(json.dumps(
                source_state, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest()
            if source_state.get("bit_generator") == "MT19937":
                audit["source_normal_protocol"] = "advers-numpy126-scalar-polar-v1"
                audit["source_rng_after_feedback_transition"] = {
                    "keys_sha256": hashlib.sha256(json.dumps(
                        source_state["keys"], separators=(",", ":"),
                    ).encode()).hexdigest(),
                    "position": source_state["position"],
                    "has_gaussian": source_state["has_gaussian"],
                    "cached_gaussian": source_state["cached_gaussian"],
                }
        return _Evaluation(
            action=action,
            evaluated_action=evaluated_action,
            observation=observation,
            feedback_extra={"perturbation": perturbation},
            observation_noise_variance=variance,
            dataset_index=row,
            metadata={"source_mode": self.source_mode},
            audit=audit,
        )


__all__ = ["AdversarialEnvironment", "AttackProvider", "LCBAttack", "RandomAttack", "WorstCaseAttack"]
