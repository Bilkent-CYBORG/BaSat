"""Independent sequential algorithms sharing one round-robin query budget."""
from __future__ import annotations

import copy
import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any, ClassVar, Literal, cast

import numpy as np
import torch

from basat.environments.environment import EnvironmentData
from basat.trials import restore_rng, rng_state

from .algorithm import (
    Algorithm,
    AlgorithmConfigurationError,
    AlgorithmDataError,
    AlgorithmExceptionController,
    AlgorithmProtocolError,
    AlgorithmStateError,
)
from .helper import ExceptionDiagnosticReport, SourceRecord


class _InterleavedException(AlgorithmExceptionController):
    def diagnose_initialization(self) -> ExceptionDiagnosticReport:
        report = super().diagnose_initialization()
        if getattr(self.environment, "source_mode", None) != "function":
            self.add_issue("error", "interleaved.function_only",
                           "InterleavedAlgorithm requires a callable stochastic environment, not a finite dataset.")
        return ExceptionDiagnosticReport(
            algorithm_name=report.algorithm_name, environment_type=report.environment_type,
            issues=tuple(self.issues), guarantees_applicable=False,
        )


class InterleavedAlgorithm(Algorithm):
    """Run independent learners one action at a time in insertion order.

    Parameters
    ----------
    algorithms : mapping of str to Algorithm
        Fresh, distinct algorithm instances. Each must propose one sequential
        action, use the same bounds, precision, device and optimization
        direction, and support a context-free stochastic environment. Mapping
        insertion order is the immutable round-robin schedule. The wrapper
        owns these instances; do not run or modify them independently.
    seed : int, default 1
        Wrapper seed. Child streams start from each child's explicit seed.
        Trial offsets shift every child seed as well as the wrapper seed.
        Choose separated child seed ranges when trials should not reuse the
        same seed across embeddings (for example 2021, 3021, 4021, 5021).
    rng_policy : {"independent", "shared"}, default "independent"
        ``shared`` gives compatible source protocols one continuing proposal
        stream while keeping their observations separate. All projections are
        constructed sequentially before the first initial-design query. The
        separately recorded initializer seed stream also continues across
        learners; this reconstructs, rather than recovers, entropy-seeded runs.
    plot, verbose, verbose_setting, verbose_iter_by_iter : bool
        Standard Algorithm reporting settings, applied to the combined run.

    Notes
    -----
    A wrapper evaluation is one query by one child, not a query by every
    child. Four children therefore receive 125 observations each under a
    strict 500-query Experiment budget. Children never receive each other's
    observations. Initialization requires empty history; shared warm starts,
    finite datasets and contextual/adversarial feedback are unsupported.

    Python, NumPy and Torch process RNG streams are isolated per child in
    addition to each child's private state. Checkpoints include every child,
    the pending owner and scheduler. Exact numerical continuation also
    requires each child to implement an exact state_dict/load_state_dict pair.
    A wrapper source iteration is one completed ask/tell exchange; it is not
    a complete sweep or a nested child's multi-request source program.

    Examples
    --------
    >>> import basat
    >>> bounds = basat.Bounds(lower=[-5.] * 10, upper=[5.] * 10)
    >>> learners = {f"embedding-{i}": basat.REMBO(
    ...     bounds=bounds,
    ...     config=basat.REMBOConfig(embedding_dimension=2, seed=2021+1000*i),
    ...     dtype="float64") for i in range(4)}
    >>> method = basat.InterleavedAlgorithm(learners, seed=2021)
    >>> # Pass method to a normal Experiment with a user-defined environment.
    """

    EXCEPTION_CLASS = _InterleavedException
    native_environments = frozenset({"stochastic"})
    cross_environment_policy: ClassVar[Literal["error", "warning"]] = "error"
    native_batching = True
    ACQUISITION_POLICY = "round-robin delegation to independent child acquisition policies"
    SOURCE = SourceRecord(
        paper_title="Independent round-robin composition", paper_year=2026,
        paper_url=None, repository_url=None, repository_status="local",
        notes="Generic scheduling composition; child algorithms retain their individual source records.",
    )
    _POLICY_VERSION = 1

    def __init__(
        self, algorithms: Mapping[str, Algorithm], *, seed: int = 1,
        rng_policy: Literal["independent", "shared"] = "independent",
        plot: bool = False, verbose: bool = False,
        verbose_setting: bool = False, verbose_iter_by_iter: bool = False,
    ) -> None:
        if not isinstance(algorithms, Mapping) or not algorithms:
            raise AlgorithmConfigurationError("algorithms must be a nonempty ordered mapping.")
        if rng_policy not in {"independent", "shared"}:
            raise AlgorithmConfigurationError("rng_policy must be 'independent' or 'shared'.")
        children = dict(algorithms)
        if any(not isinstance(label, str) or not label.strip() for label in children):
            raise AlgorithmConfigurationError("Each child label must be a nonempty string.")
        if any(not isinstance(child, Algorithm) for child in children.values()):
            raise AlgorithmConfigurationError("Each child must be an Algorithm instance.")
        if len({id(child) for child in children.values()}) != len(children):
            raise AlgorithmConfigurationError("Children must be distinct instances, not shared aliases.")
        first = next(iter(children.values()))
        for child in children.values():
            if child._initialized or child.evaluations or child.last_ask is not None:
                raise AlgorithmConfigurationError("Children must be fresh and uninitialized.")
            if child.batch_size != 1 or child.q_batch is not None:
                raise AlgorithmConfigurationError("Interleaving requires batch_size=1 and q_batch=None.")
            if (child.action_dimension != first.action_dimension or child.dtype != first.dtype
                    or child.device != first.device or child.maximize != first.maximize
                    or not torch.equal(child.bounds.lower, first.bounds.lower)
                    or not torch.equal(child.bounds.upper, first.bounds.upper)):
                raise AlgorithmConfigurationError("Children must share bounds, dtype, device and maximize.")
            if ((child.native_environments and "stochastic" not in child.native_environments)
                    or child.required_round_info or child.required_capabilities
                    or not child.required_feedback.issubset({"observation", "evaluated_action",
                                                            "observation_noise_variance"})):
                raise AlgorithmConfigurationError("Only context-free stochastic children are supported.")
        super().__init__(bounds=first.bounds, batch_size=1, q_batch=None,
                         maximize=first.maximize, dtype=first.dtype, device=first.device,
                         seed=seed, plot=plot, verbose=verbose, verbose_setting=verbose_setting,
                         verbose_iter_by_iter=verbose_iter_by_iter)
        self._children = children
        self._order = tuple(children)
        self._cursor = 0
        self._pending_owner: str | None = None
        self._child_rng: dict[str, dict[str, Any] | None] = dict.fromkeys(children)
        self.rng_policy = rng_policy
        self._shared_source_rng: dict[str, Any] | None = None
        if rng_policy == "shared":
            required = ("_reset_shared_random_streams", "_export_random_streams",
                        "_import_random_streams", "_consume_source_evaluation_draws")
            if any(not getattr(child, "_supports_shared_trial_streams", False)
                   or not all(callable(getattr(child, key, None)) for key in required)
                   for child in children.values()):
                raise AlgorithmConfigurationError(
                    "Shared interleaving requires a source protocol with shared-stream hooks.")
            from .Stochastic._tree_numerics import random_state_dict
            state = copy.deepcopy(cast(Any, first)._export_random_streams())
            if any(set(cast(Any, child)._export_random_streams()) != set(state)
                   for child in children.values()):
                raise AlgorithmConfigurationError("Shared learners must use compatible random-stream schemas.")
            state["numpy"] = random_state_dict(np.random.RandomState(seed))
            state["torch"] = torch.Generator(device="cpu").manual_seed(seed).get_state()
            if "initializer_seed_stream" in state:
                state["initializer_seed_stream"] = random_state_dict(np.random.RandomState(seed))
            previous = rng_state()
            try:
                for child in children.values():
                    cast(Any, child)._reset_shared_random_streams(state)
                    state = copy.deepcopy(cast(Any, child)._export_random_streams())
            finally:
                restore_rng(previous)
            self._shared_source_rng = state
        self.invalidate_source_guarantees("basat.independent_round_robin_composition")

    @property
    def _supports_ax017_evaluation(self) -> bool:
        return all(getattr(child, "_supports_ax017_evaluation", False)
                   for child in self._children.values())

    @property
    def algorithms(self) -> Mapping[str, Algorithm]:
        """Read-only child mapping in the immutable scheduling order."""
        return MappingProxyType(self._children)

    @property
    def next_algorithm(self) -> str:
        """Owner of the next query, or the current unconsumed query."""
        return self._order[self._cursor]

    @property
    def observation_owners(self) -> tuple[str, ...]:
        """Child label for every chronological observed reward."""
        return tuple(self._order[i % len(self._order)] for i in range(self.evaluations))

    @contextmanager
    def _child_stream(self, label: str) -> Iterator[None]:
        previous = rng_state()
        try:
            saved = self._child_rng[self._order[0] if self.rng_policy == "shared" else label]
            if saved is None:
                seed = self.seed if self.rng_policy == "shared" else self._children[label].seed
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
            else:
                restore_rng(saved)
            if self.rng_policy == "shared":
                cast(Any, self._children[label])._import_random_streams(copy.deepcopy(self._shared_source_rng))
            yield
        finally:
            self._child_rng[label] = rng_state()
            if self.rng_policy == "shared":
                self._shared_source_rng = copy.deepcopy(cast(Any, self._children[label])._export_random_streams())
                self._child_rng = {key: copy.deepcopy(self._child_rng[label]) for key in self._order}
            restore_rng(previous)

    def bind_environment_metadata(self, metadata: Mapping[str, Any]) -> None:
        if metadata:
            raise AlgorithmDataError("InterleavedAlgorithm does not support contextual or distributional metadata.")

    def configure_sampling_methods(self, *, initial: Literal["paper", "sobol", "uniform"] = "paper",
                                   acquisition: Literal["paper", "sobol", "uniform"] = "paper") -> None:
        super().configure_sampling_methods(initial=initial, acquisition=acquisition)
        for child in self._children.values():
            child.configure_sampling_methods(initial=initial, acquisition=acquisition)

    def _initialize_from_environment_impl(self, data: EnvironmentData) -> None:
        if data.number_of_observations or data.contexts is not None:
            raise AlgorithmDataError(
                "InterleavedAlgorithm requires empty, context-free history; observations cannot be broadcast to children. "
                "Resume the wrapper checkpoint instead of passing aggregate previous_data.")
        snapshots = {label: child.state_dict() for label, child in self._children.items()}
        streams = copy.deepcopy(self._child_rng)
        source_stream = copy.deepcopy(self._shared_source_rng)
        try:
            for label, child in self._children.items():
                with self._child_stream(label):
                    child.initialize_from_environment(copy.deepcopy(data))
        except BaseException:
            for label, child in self._children.items():
                with self._child_stream(label):
                    child.load_state_dict(snapshots[label])
            self._child_rng = streams
            self._shared_source_rng = source_stream
            raise

    def _ask_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        if round_info:
            raise AlgorithmDataError("InterleavedAlgorithm supports only continuous, context-free stochastic queries.")
        label = self.next_algorithm
        child = self._children[label]
        saved, stream = child.state_dict(), copy.deepcopy(self._child_rng[label])
        all_streams = copy.deepcopy(self._child_rng)
        source_stream = copy.deepcopy(self._shared_source_rng)
        try:
            with self._child_stream(label):
                action = child.ask(round_info=round_info)
            assert child.last_ask is not None
            metadata = dict(child.last_ask.algorithm_metadata)
            metadata["interleaved_owner"] = label
            self._record_last_ask(actions=action, round_info=round_info,
                                  model_inputs=child.last_ask.model_inputs, algorithm_metadata=metadata)
            self._pending_owner = label
            return action
        except BaseException:
            with self._child_stream(label):
                child.load_state_dict(saved)
            self._child_rng[label] = stream
            if self.rng_policy == "shared":
                self._child_rng = all_streams
            self._shared_source_rng = source_stream
            self._pending_owner = None
            self.last_ask = None
            raise

    def _tell_impl(self, *, action: torch.Tensor, feedback: Mapping[str, Any],
                   round_info: Mapping[str, Any]) -> None:
        if self._pending_owner != self.next_algorithm:
            raise AlgorithmProtocolError("Interleaved feedback has no matching pending owner.")
        if round_info or any(key in feedback for key in ("context", "adversarial_action", "perturbation")):
            raise AlgorithmDataError("Contextual/adversarial feedback cannot be routed by InterleavedAlgorithm.")
        label = self._pending_owner
        child = self._children[label]
        saved, stream = child.state_dict(), copy.deepcopy(self._child_rng[label])
        all_streams = copy.deepcopy(self._child_rng)
        source_stream = copy.deepcopy(self._shared_source_rng)
        try:
            with self._child_stream(label):
                child.tell(action=action, feedback=feedback, round_info=round_info)
        except BaseException:
            with self._child_stream(label):
                child.load_state_dict(saved)
            self._child_rng[label] = stream
            if self.rng_policy == "shared":
                self._child_rng = all_streams
            self._shared_source_rng = source_stream
            raise
        self._pending_owner = None
        self._cursor = (self._cursor + 1) % len(self._order)

    def _configuration_snapshot(self) -> dict[str, Any]:
        return {"base": self._fingerprint_value(self._base_configuration()),
                "algorithm_config": {"schedule": "round_robin", "version": self._POLICY_VERSION,
                    **({"rng_policy": "shared", "shared_policy_version": 1}
                       if self.rng_policy == "shared" else {}),
                    "children": [{"label": label, "class": f"{type(child).__module__}.{type(child).__qualname__}",
                                  "configuration": child._configuration_snapshot()}
                                 for label, child in self._children.items()]}}

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        return {"schedule": "round_robin", "next_algorithm": self.next_algorithm,
                "rng_policy": self.rng_policy,
                "pending_owner": self._pending_owner,
                "child_evaluations": {label: child.evaluations for label, child in self._children.items()},
                "child_source_guarantees": {label: child.source_guarantees_applicable
                                            for label, child in self._children.items()}}

    def _state_dict_impl(self) -> Mapping[str, Any]:
        return {"version": self._POLICY_VERSION, "order": self._order,
                "cursor": self._cursor, "pending_owner": self._pending_owner,
                "children": {label: child.state_dict() for label, child in self._children.items()},
                **({"shared_source_rng": copy.deepcopy(self._shared_source_rng)}
                   if self.rng_policy == "shared" else {}),
                "child_rng": copy.deepcopy(self._child_rng)}

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        del strict
        self._require_state_keys(state, {"version", "order", "cursor", "pending_owner", "children", "child_rng"})
        if self.rng_policy == "shared" and not isinstance(state.get("shared_source_rng"), Mapping):
            raise AlgorithmStateError("Shared interleaving requires its saved source RNG state.")
        if self.rng_policy == "shared":
            from .Stochastic._ax_source_random import validate_streams
            try:
                expected = cast(Any, next(iter(self._children.values())))._export_random_streams()
                if set(state["shared_source_rng"]) != set(expected):
                    raise ValueError("Shared source RNG schema does not match its learners.")
                validate_streams(state["shared_source_rng"])
            except (ValueError, TypeError, KeyError, RuntimeError) as error:
                raise AlgorithmStateError("Invalid shared source RNG state.") from error
        if (state["version"] != self._POLICY_VERSION or tuple(state["order"]) != self._order
                or not isinstance(state["children"], Mapping) or tuple(state["children"]) != self._order
                or not isinstance(state["child_rng"], Mapping) or tuple(state["child_rng"]) != self._order
                or isinstance(state["cursor"], bool) or not isinstance(state["cursor"], int)
                or not 0 <= state["cursor"] < len(self._order)
                or state["pending_owner"] not in (None, self._order[state["cursor"]])):
            raise AlgorithmStateError("Invalid interleaved policy, child order, cursor or pending owner.")

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        self._cursor = state["cursor"]
        self._pending_owner = state["pending_owner"]
        self._child_rng = copy.deepcopy(dict(state["child_rng"]))
        self._shared_source_rng = copy.deepcopy(state.get("shared_source_rng"))
        for label, child in self._children.items():
            child.load_state_dict(state["children"][label], strict=strict)
            stream = self._child_rng[label]
            if stream is not None:
                previous = rng_state()
                try:
                    restore_rng(stream)
                except Exception as error:
                    raise AlgorithmStateError(f"Invalid child RNG state for {label}.") from error
                finally:
                    restore_rng(previous)
        self._validate_routing_state()

    def _validate_routing_state(self) -> None:
        if (self.iteration != self.evaluations or self._cursor != self.evaluations % len(self._order)
                or self.observed_values.numel() != self.evaluations
                or ((self.evaluations or self.last_ask is not None) and not self._initialized)
                or bool(self.last_ask is not None) != bool(self._pending_owner is not None)):
            raise AlgorithmStateError("Interleaved progress and pending state disagree.")
        for index, (label, child) in enumerate(self._children.items()):
            values = self.observed_values[index::len(self._order)]
            actions = self.observed_actions[index::len(self._order)]
            if (child.evaluations != values.numel() or child.iteration != child.evaluations
                    or child._initialized != self._initialized
                    or not torch.equal(child.observed_values, values)
                    or not torch.equal(child.observed_actions, actions)
                    or bool(child.last_ask is not None) != (label == self._pending_owner)
                    or (self._initialized and self._child_rng[label] is None)):
                raise AlgorithmStateError(f"Child {label} history/initialization/pending state disagrees with ownership.")
            if label == self._pending_owner:
                assert self.last_ask is not None and child.last_ask is not None
                if (not torch.equal(child.last_ask.actions, self.last_ask.actions)
                        or self.last_ask.algorithm_metadata.get("interleaved_owner") != label
                        or child.last_ask.round_info_snapshot or self.last_ask.round_info_snapshot):
                    raise AlgorithmStateError("Pending action or owner metadata disagree.")

    def load_state_dict(self, state: Mapping[str, Any], *, strict: bool = True) -> None:
        """Validate all nested states on a fresh staging wrapper before adoption."""
        previous = rng_state()
        try:
            staged = self.clone_for_trial(0)
            Algorithm.load_state_dict(staged, state, strict=strict)
        finally:
            restore_rng(previous)
        self.__dict__.update(staged.__dict__)

    def clone_for_trial(self, seed_offset: int) -> InterleavedAlgorithm:
        """Reconstruct all children with shifted seeds, never copy fitted models."""
        if (isinstance(seed_offset, bool) or not isinstance(seed_offset, int)
                or self.seed + seed_offset < 0
                or any(child.seed + seed_offset < 0 for child in self._children.values())):
            raise AlgorithmConfigurationError("seed_offset must preserve non-negative wrapper and child seeds.")
        previous = rng_state()
        try:
            children = {}
            for label, child in self._children.items():
                seed = child.seed + seed_offset
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
                children[label] = child.clone_for_trial(seed_offset)
            result = type(self)(children, seed=self.seed + seed_offset, plot=self.plot,
                                rng_policy=self.rng_policy,
                                verbose=self.verbose, verbose_setting=self.verbose_setting,
                                verbose_iter_by_iter=self.verbose_iter_by_iter)
            # Child clone_for_trial already preserves its explicit sampler.
            result.initial_points_sampling_method = self.initial_points_sampling_method
            result.acquisition_sampling_method = self.acquisition_sampling_method
            return result
        finally:
            restore_rng(previous)

    def reset(self, seed: int | None = None) -> None:
        """Atomically replace every child with a fresh, consistently reseeded instance."""
        selected = self.seed if seed is None else seed
        if isinstance(selected, bool) or not isinstance(selected, int) or selected < 0:
            raise AlgorithmConfigurationError("reset seed must be a non-negative integer.")
        staged = self.clone_for_trial(selected - self.seed)
        self.__dict__.update(staged.__dict__)

    def _consume_source_evaluation_draws(self, count: int) -> None:
        """Advance the pending owner's source NumPy stream, not its model."""
        if self._pending_owner is None:
            raise AlgorithmProtocolError("Source evaluation draws require a pending proposal.")
        with self._child_stream(self._pending_owner):
            cast(Any, self._children[self._pending_owner])._consume_source_evaluation_draws(count)

    def _consume_source_readback_draws(self, count: int) -> None:
        """Route the total wrapper history's cached-data fetch to the next owner."""
        hook = getattr(self._children[self.next_algorithm], "_consume_source_readback_draws", None)
        if not callable(hook):
            return
        with self._child_stream(self.next_algorithm):
            hook(count)


__all__ = ["InterleavedAlgorithm"]
