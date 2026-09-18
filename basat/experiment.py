"""Crash-safe orchestration for Basat ask-tell experiments.

The experiment coordinates public information flow. It deliberately contains no
embedding, projection, contextual kernel, or other algorithm-specific transform.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
import time
import traceback
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Literal

import torch

from .algorithms.algorithm import Algorithm
from .algorithms.helper import AlgorithmResult
from .environments.adapters import adapt_environment_for_algorithm
from .environments.environment import Environment, NoAvailableActionError
from .stopping.stopping_rule import (
    CompositeStoppingRule,
    MaximumEvaluationsStoppingRule,
    StoppingDecision,
    StoppingRule,
)
from .trials import (
    ACTIVE,
    ExperimentTrialsResult,
    aggregate_document,
    rng_state,
    run_trials,
    trial_rng,
)
from .warnings import BasatExperimentWarning, translate_backend_warnings

SamplingMethod = Literal["paper", "sobol", "uniform"]
PerAlgorithmSamplingMethod = SamplingMethod | Mapping[str, SamplingMethod | None] | None


class ExperimentConfigurationError(ValueError):
    """Raised when an experiment cannot be configured safely."""


class ExperimentCheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, corrupt, or incompatible."""


class ExperimentProtocolError(RuntimeError):
    """Raised when an environment or algorithm violates the ask-tell contract."""


@dataclass(frozen=True, slots=True)
class ExperimentFailure:
    """Recorded failure of one independent multi-algorithm child run."""

    label: str
    phase: Literal["initialization", "run"]
    exception_type: str
    message: str
    completed_rounds: int
    traceback: str | None = None


@dataclass(frozen=True, slots=True)
class ExperimentComparisonResult:
    """Results and optional shared figure from independent experiments.

    Parameters
    ----------
    results:
        Read-only mapping from the user-supplied experiment label to that
        experiment's :class:`AlgorithmResult`.
    figure:
        Combined Matplotlib figure, or ``None`` when plotting was disabled.
    axes:
        Observed-reward and incumbent-reward axes, or ``None`` when plotting
        was disabled.
    failures:
        Read-only mapping of algorithms that failed independently. Successful
        algorithms remain available in ``results`` and continue running.

    Examples
    --------
    Access one independently produced result by its comparison label:

    >>> comparison.results["TuRBO"].evaluations  # doctest: +SKIP
    50
    >>> comparison.figure.savefig("comparison.png")  # doctest: +SKIP
    """

    results: Mapping[str, AlgorithmResult]
    figure: Any | None = None
    axes: tuple[Any, Any] | None = None
    failures: Mapping[str, ExperimentFailure] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze a private copy of the result mapping."""

        if not isinstance(self.results, Mapping):
            raise ExperimentConfigurationError("results must be a mapping.")
        copied = dict(self.results)
        if any(not isinstance(name, str) or not name.strip() for name in copied):
            raise ExperimentConfigurationError(
                "Every comparison result label must be a non-empty string."
            )
        if any(not isinstance(result, AlgorithmResult) for result in copied.values()):
            raise ExperimentConfigurationError(
                "Every comparison result value must be an AlgorithmResult."
            )
        if not isinstance(self.failures, Mapping):
            raise ExperimentConfigurationError("failures must be a mapping.")
        copied_failures = dict(self.failures)
        if any(not isinstance(item, ExperimentFailure) for item in copied_failures.values()):
            raise ExperimentConfigurationError(
                "Every comparison failure value must be an ExperimentFailure."
            )
        if set(copied) & set(copied_failures):
            raise ExperimentConfigurationError(
                "An algorithm label cannot be both successful and failed."
            )
        if not copied and not copied_failures:
            raise ExperimentConfigurationError(
                "A comparison must contain at least one result or failure."
            )
        if (self.figure is None) != (self.axes is None):
            raise ExperimentConfigurationError(
                "figure and axes must either both be supplied or both be None."
            )
        object.__setattr__(self, "results", MappingProxyType(copied))
        object.__setattr__(self, "failures", MappingProxyType(copied_failures))

    def __getitem__(self, name: str) -> AlgorithmResult:
        """Return the individual result stored under ``name``."""

        return self.results[name]


class Experiment:
    """Coordinate an environment, algorithm, stopping rule, and persistence.

    Parameters
    ----------
    environment:
        An :class:`Environment` instance that exclusively owns evaluation and
        learner-visible history. Built-in choices are
        :class:`StochasticEnvironment`, :class:`ContextualEnvironment`,
        :class:`DistributionalEnvironment`, and :class:`AdversarialEnvironment`.
        User environments may subclass :class:`Environment`.
    algorithm:
        One :class:`Algorithm`, or a named mapping/sequence of algorithms. In
        multi-algorithm mode every algorithm is run in an independent child
        experiment. A single environment acts as a template and is deep-copied
        for each child; alternatively supply a matching environment mapping.
    stopping_rule:
        Optional :class:`StoppingRule`. Built-in limits are
        :class:`MaximumRoundsStoppingRule`,
        :class:`MaximumEvaluationsStoppingRule`, and
        :class:`MaximumTimeStoppingRule`; combine limits with :func:`any_of` or
        :func:`all_of`. If omitted, :meth:`run` requires ``max_rounds`` or ``max_evaluations``.
    output_directory:
        Directory for transactional checkpoints and diagnostics. In
        multi-algorithm mode it is a parent containing ``multi-experiment.json``
        and one ``algorithms/<number>-<name>/`` checkpoint directory per
        algorithm. If omitted, the run remains in memory.
    resume:
        Resume the generation selected by ``latest.json`` when that manifest
        exists. If the output directory has no manifest yet, initialize a new
        run. An existing corrupt or incompatible checkpoint still raises an
        error instead of silently starting over. ``checkpoint-latest.pt`` is
        retained as a convenience alias, not as the transactional authority.
        Checkpoints restore observations and search progress, not fitted GP
        parameters. A resumed algorithm refits its surrogate from its restored
        observations using its current ``gp_training_steps``, ``gp_maxiter``,
        or ``fit_maxiter`` value.
    save_every:
        Number of completed environment rounds between archived checkpoints.
        With the default transactional policy, latest progress is also saved
        around every ``tell``. With periodic policy this is the save cadence.
    checkpoint_policy:
        ``"transactional"`` (default) persists environment feedback before
        ``tell`` and learner progress afterward. ``"periodic"`` saves only
        every ``save_every`` rounds and at a successful run-call return or
        stopping decision. A crash can lose up to ``save_every`` recent rounds;
        those objective evaluations may be repeated on resume. Use periodic
        saving only when that tradeoff is acceptable, such as replayable
        offline datasets. Neither policy changes GP training or observations.
    constant_context:
        Reserved compatibility keyword; only ``None`` is accepted. Supply a
        context through the environment. Algorithm defaults are internal.
    initial_points_sampling_method, acquisition_sampling_method:
        Optional continuous-design policies. A single value applies to every
        algorithm; multi-algorithm experiments may provide a mapping keyed by
        algorithm label. ``None`` preserves each algorithm's current choice.
        Explicit values are ``paper``, ``sobol``, or ``uniform``.
    initial_sampling_method:
        Backward-compatible alias for ``initial_points_sampling_method``.

    Notes
    -----
    A newly revealed environment record is persisted before it is passed to the
    algorithm. If the process fails in that interval, resume replays the record
    without evaluating the source again. Problem-defining settings remain
    checkpoint-compatible; surrogate fitting budgets may change on resume.

    Examples
    --------
    >>> import basat
    >>> def objective(x):
    ...     return -((x - 0.25).square().sum())
    >>> bounds = basat.Bounds(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    >>> environment = basat.StochasticEnvironment(
    ...     bounds=bounds, function=objective, noise_std=0.0,
    ...     dtype="float64", device="cpu", seed=42,
    ... )
    >>> algorithm = basat.TuRBO(
    ...     bounds=bounds,
    ...     config=basat.TuRBOConfig(
    ...         initial_points=10, acquisition="TS", batch_size=1, seed=42
    ...     ),
    ...     dtype="float64", device="cpu",
    ... )
    >>> experiment = basat.Experiment(
    ...     environment=environment, algorithm=algorithm,
    ...     stopping_rule=basat.MaximumEvaluationsStoppingRule(150),
    ...     output_directory="./results/turbo", resume=True, save_every=10,
    ... )
    >>> result = experiment.run(max_rounds=150)

    A named mapping runs independently and uses each algorithm's own ``plot``
    flag for one combined figure:

    >>> experiment = basat.Experiment(  # doctest: +SKIP
    ...     environment=environment,
    ...     algorithm={"TuRBO": turbo, "REMBO": rembo, "HeSBO": hesbo},
    ...     output_directory="./results/comparison",
    ...     resume=True,
    ... )
    >>> comparison = experiment.run(max_rounds=50)  # doctest: +SKIP
    """

    CHECKPOINT_SCHEMA_VERSION = 1
    PACKAGE_VERSION = "0.1.0"

    def __init__(
        self,
        *,
        environment: Environment | Mapping[str, Environment],
        algorithm: Algorithm | Mapping[str, Algorithm] | Sequence[Algorithm],
        stopping_rule: StoppingRule | None = None,
        output_directory: str | os.PathLike[str] | None = None,
        resume: bool = False,
        save_every: int = 1,
        checkpoint_policy: Literal["transactional", "periodic"] = "transactional",
        constant_context: Any | None = None,
        initial_points_sampling_method: PerAlgorithmSamplingMethod = None,
        acquisition_sampling_method: PerAlgorithmSamplingMethod = None,
        initial_sampling_method: PerAlgorithmSamplingMethod = None,
    ) -> None:
        """Initialize the coordinator from the components documented above."""

        if checkpoint_policy not in {"transactional", "periodic"}:
            raise ExperimentConfigurationError(
                "checkpoint_policy must be 'transactional' or 'periodic'."
            )
        self.checkpoint_policy = checkpoint_policy
        if constant_context is not None:
            raise ExperimentConfigurationError(
                "constant_context no longer configures algorithm defaults. Supply context "
                "through the environment (or an explicit environment adapter)."
            )
        if (
            output_directory is not None
            and not resume
            and (Path(output_directory) / "trials.json").exists()
        ):
            raise ExperimentConfigurationError(
                "Existing trial results require resume=True or a new output_directory."
            )
        self.trial_experiments: list[Experiment] = []
        self._trial_managed = ACTIVE.get()
        self._trial_rng_state: dict[str, Any] | None = None
        self._trial_constant_context = constant_context
        self.warnings: dict[str, list[str]] = {}
        self.failures: dict[str, ExperimentFailure] = {}
        self._run_stop_reason: str | None = None
        self._last_requested_limits: dict[str, Any] = {}
        parameters: dict[str, Any] = dict(
            environment=environment,
            algorithm=algorithm,
            stopping_rule=stopping_rule,
            output_directory=output_directory,
            resume=resume,
            save_every=save_every,
            constant_context=constant_context,
            initial_points_sampling_method=initial_points_sampling_method,
            acquisition_sampling_method=acquisition_sampling_method,
            initial_sampling_method=initial_sampling_method,
        )
        if isinstance(algorithm, Algorithm):
            with self._capture_warnings(type(algorithm).__name__):
                self._initialize(**parameters)
        else:
            self._initialize(**parameters)
        self._restore_saved_warnings()

    def _restore_saved_warnings(self) -> None:
        """Carry previous diagnostics through checkpoint continuation."""

        if not self.resume or self.output_directory is None:
            return
        path = self.output_directory / (
            "results.json" if self.is_multi_algorithm else "result.json"
        )
        if not path.is_file():
            return
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # The checkpoint manifest remains authoritative; this optional
            # readable export must never invalidate a recoverable checkpoint.
            return
        if not isinstance(document, dict):
            return
        rows = document.get("algorithms", {})
        # Multi-algorithm parents save a compact per-child result document.
        if "label" in document and "warnings" in document and not self.is_multi_algorithm:
            rows = {type(self.algorithm).__name__: document}
        if not isinstance(rows, dict):
            return
        for label in self.algorithms:
            row = rows.get(label, {})
            if not isinstance(row, dict) or not isinstance(row.get("warnings", []), list):
                continue
            stored = self.warnings.setdefault(label, [])
            for message in row.get("warnings", []):
                if isinstance(message, str) and message not in stored:
                    stored.append(message)

    @contextmanager
    def _capture_warnings(self, label: str) -> Iterator[None]:
        """Retain distinct warning messages and display each new message once."""

        captured: tuple[warnings.WarningMessage, ...] = ()
        try:
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                try:
                    yield
                finally:
                    captured = tuple(records)
        finally:
            stored = self.warnings.setdefault(label, [])
            for record in captured:
                message = f"{record.category.__name__}: {record.message}"
                if message not in stored:
                    stored.append(message)
                    warnings.warn(str(record.message), record.category, stacklevel=3)

    def _initialize(
        self,
        *,
        environment: Environment | Mapping[str, Environment],
        algorithm: Algorithm | Mapping[str, Algorithm] | Sequence[Algorithm],
        stopping_rule: StoppingRule | None = None,
        output_directory: str | os.PathLike[str] | None = None,
        resume: bool = False,
        save_every: int = 1,
        constant_context: Any | None = None,
        initial_points_sampling_method: PerAlgorithmSamplingMethod = None,
        acquisition_sampling_method: PerAlgorithmSamplingMethod = None,
        initial_sampling_method: PerAlgorithmSamplingMethod = None,
    ) -> None:
        """Initialize the coordinator from the components documented above."""

        if initial_points_sampling_method is not None and initial_sampling_method is not None:
            raise ExperimentConfigurationError(
                "Specify initial_points_sampling_method, not both it and the legacy "
                "initial_sampling_method alias."
            )
        selected_initial_sampling = (
            initial_points_sampling_method
            if initial_points_sampling_method is not None
            else initial_sampling_method
        )
        algorithms = self._normalize_algorithm_collection(algorithm)
        self.is_multi_algorithm = algorithms is not None
        self.experiments: Mapping[str, Experiment] = MappingProxyType({})
        self.algorithms: Mapping[str, Algorithm] = MappingProxyType({})
        self._comparison_elapsed_seconds = 0.0
        if algorithms is not None:
            self._initialize_multiple(
                environment=environment,
                algorithms=algorithms,
                stopping_rule=stopping_rule,
                output_directory=output_directory,
                resume=resume,
                save_every=save_every,
                constant_context=constant_context,
                initial_sampling_method=selected_initial_sampling,
                acquisition_sampling_method=acquisition_sampling_method,
            )
            return

        if not isinstance(environment, Environment):
            raise ExperimentConfigurationError(
                "environment must be an Environment instance, for example "
                "StochasticEnvironment or ContextualEnvironment."
            )
        if not isinstance(algorithm, Algorithm):  # narrowed by collection normalization
            raise ExperimentConfigurationError(
                "algorithm must be an Algorithm instance, for example TuRBO or CGP_UCB."
            )
        if stopping_rule is not None and not isinstance(stopping_rule, StoppingRule):
            raise ExperimentConfigurationError(
                "stopping_rule must be a StoppingRule instance or None."
            )
        if isinstance(save_every, bool) or not isinstance(save_every, int) or save_every < 1:
            raise ExperimentConfigurationError("save_every must be a positive integer.")
        if getattr(algorithm, "EXCEPTION_CLASS", None) is None:
            raise ExperimentConfigurationError(
                f"{type(algorithm).__name__} must declare EXCEPTION_CLASS."
            )

        self.source_environment = environment
        self.source_environments = MappingProxyType({type(algorithm).__name__: environment})
        self.environment = adapt_environment_for_algorithm(
            environment,
            algorithm,
            constant_context=constant_context,
        )
        self.algorithm = algorithm
        public_support = getattr(self.environment, "context_support", None)
        self.algorithm.bind_environment_metadata(
            {}
            if public_support is None
            else {
                "context_support": public_support,
                "context_support_is_dynamic": getattr(
                    self.environment, "context_support_is_dynamic", False
                ),
            }
        )
        self.algorithms = MappingProxyType({type(algorithm).__name__: algorithm})
        if isinstance(selected_initial_sampling, Mapping) or isinstance(
            acquisition_sampling_method, Mapping
        ):
            raise ExperimentConfigurationError(
                "Sampling-method mappings are valid only for multi-algorithm experiments."
            )
        if selected_initial_sampling is not None or acquisition_sampling_method is not None:
            self.algorithm.configure_sampling_methods(
                initial=(
                    self.algorithm.initial_sampling_method
                    if selected_initial_sampling is None
                    else selected_initial_sampling
                ),
                acquisition=(
                    self.algorithm.acquisition_sampling_method
                    if acquisition_sampling_method is None
                    else acquisition_sampling_method
                ),
            )
        self.stopping_rule = stopping_rule
        self.output_directory = None if output_directory is None else Path(output_directory)
        self.resume = bool(resume)
        self.save_every = int(save_every)
        self.algorithm_consumed_environment_records = 0
        self.completed_rounds = int(getattr(self.environment, "round", 0))
        self.stopping_decision: Any | None = None
        self._started_at = time.monotonic()
        self._last_checkpoint_phase: str | None = None

        self._prepare_storage()
        self.algorithm_exception = self._initialize_algorithm_exception_handler()
        self.initialization_exception_diagnostics = (
            self.algorithm_exception.validate_initialization()
        )
        for issue in self.initialization_exception_diagnostics.issues:
            if bool(issue.details.get("affects_source_guarantees", False)):
                self.algorithm.invalidate_source_guarantees(issue.code)

        checkpoint_exists = bool(
            self.output_directory is not None and (self.output_directory / "latest.json").is_file()
        )
        if self.resume and checkpoint_exists:
            self.load_checkpoint()
            with trial_rng(self):
                self._replay_unconsumed_environment_records()
        else:
            self._initialize_algorithm_from_environment()

    @staticmethod
    def _normalize_algorithm_collection(
        algorithm: Algorithm | Mapping[str, Algorithm] | Sequence[Algorithm],
    ) -> dict[str, Algorithm] | None:
        """Return a validated named algorithm collection, or ``None`` for one algorithm."""

        if isinstance(algorithm, Algorithm):
            return None
        if isinstance(algorithm, Mapping):
            if not algorithm:
                raise ExperimentConfigurationError("algorithm mapping cannot be empty.")
            algorithms = dict(algorithm)
            if any(not isinstance(label, str) or not label.strip() for label in algorithms):
                raise ExperimentConfigurationError(
                    "Every algorithm mapping label must be a non-empty string."
                )
        elif isinstance(algorithm, Sequence) and not isinstance(algorithm, (str, bytes)):
            if not algorithm:
                raise ExperimentConfigurationError("algorithm sequence cannot be empty.")
            algorithms = {}
            counts: dict[str, int] = {}
            for item in algorithm:
                if not isinstance(item, Algorithm):
                    raise ExperimentConfigurationError(
                        "Every item in an algorithm sequence must be an Algorithm instance."
                    )
                base = type(item).__name__
                counts[base] = counts.get(base, 0) + 1
                label = base if counts[base] == 1 else f"{base} {counts[base]}"
                algorithms[label] = item
        else:
            raise ExperimentConfigurationError(
                "algorithm must be an Algorithm or a non-empty mapping/sequence of Algorithms."
            )
        if any(not isinstance(item, Algorithm) for item in algorithms.values()):
            raise ExperimentConfigurationError(
                "Every algorithm collection value must be an Algorithm instance."
            )
        identities = [id(item) for item in algorithms.values()]
        if len(set(identities)) != len(identities):
            raise ExperimentConfigurationError(
                "Each multi-algorithm entry must use a different Algorithm instance."
            )
        return algorithms

    @staticmethod
    def _checkpoint_component_name(index: int, label: str, algorithm: Algorithm) -> str:
        """Return a stable readable child-directory name."""

        normalized = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
        if not normalized:
            normalized = re.sub(r"[^a-z0-9]+", "-", type(algorithm).__name__.lower()).strip("-")
        return f"{index:02d}-{normalized or 'algorithm'}"

    @staticmethod
    def _clone_environment_template(environment: Environment, *, label: str) -> Environment:
        """Clone an environment template so one algorithm cannot affect another."""

        try:
            cloned = copy.deepcopy(environment)
        except Exception as error:
            raise ExperimentConfigurationError(
                f"The environment template could not be copied for {label!r}. Supply an "
                "environment mapping with a separate instance for every algorithm."
            ) from error
        if not isinstance(cloned, Environment):
            raise ExperimentConfigurationError(
                f"Copying the environment template for {label!r} did not produce an Environment."
            )
        return cloned

    @staticmethod
    def _clone_stopping_rule(
        stopping_rule: StoppingRule | None, *, label: str
    ) -> StoppingRule | None:
        """Clone mutable stopping state for one independent child experiment."""

        if stopping_rule is None:
            return None
        try:
            cloned = copy.deepcopy(stopping_rule)
        except Exception as error:
            raise ExperimentConfigurationError(
                f"The stopping rule could not be copied for {label!r}."
            ) from error
        if not isinstance(cloned, StoppingRule):
            raise ExperimentConfigurationError(
                f"Copying the stopping rule for {label!r} did not produce a StoppingRule."
            )
        return cloned

    @staticmethod
    def _report_child_failure(
        *, label: str, phase: Literal["initialization", "run"], error: Exception
    ) -> None:
        """Print one child traceback and announce that sibling runs will continue."""

        warnings.warn(
            f"Basat skipped {label!r} during {phase}: {type(error).__name__}: {error}. "
            "Independent algorithms will continue; the full traceback is available in "
            f"experiment.failures[{label!r}].traceback.",
            BasatExperimentWarning,
            stacklevel=3,
        )

    @staticmethod
    def _failure_record(
        *,
        label: str,
        phase: Literal["initialization", "run"],
        error: Exception,
        completed_rounds: int,
    ) -> ExperimentFailure:
        """Create an immutable public failure summary."""

        return ExperimentFailure(
            label=label,
            phase=phase,
            exception_type=f"{type(error).__module__}.{type(error).__qualname__}",
            message=str(error),
            completed_rounds=completed_rounds,
            traceback="".join(traceback.format_exception(type(error), error, error.__traceback__)),
        )

    @staticmethod
    def _sampling_methods_by_label(
        *,
        labels: Sequence[str],
        value: PerAlgorithmSamplingMethod,
        parameter: str,
    ) -> dict[str, SamplingMethod | None]:
        """Resolve one validated optional sampling method per algorithm label."""

        if isinstance(value, Mapping):
            missing = [label for label in labels if label not in value]
            extra = [label for label in value if label not in labels]
            if missing or extra:
                raise ExperimentConfigurationError(
                    f"A {parameter} mapping must contain exactly the algorithm labels; "
                    f"missing={missing}, extra={extra}."
                )
            resolved = dict(value)
        else:
            resolved = dict.fromkeys(labels, value)
        allowed = {None, "paper", "sobol", "uniform"}
        invalid = {label: method for label, method in resolved.items() if method not in allowed}
        if invalid:
            raise ExperimentConfigurationError(
                f"{parameter} values must be 'paper', 'sobol', 'uniform', or None; "
                f"invalid={invalid}."
            )
        return resolved

    def _initialize_multiple(
        self,
        *,
        environment: Environment | Mapping[str, Environment],
        algorithms: Mapping[str, Algorithm],
        stopping_rule: StoppingRule | None,
        output_directory: str | os.PathLike[str] | None,
        resume: bool,
        save_every: int,
        constant_context: Any | None,
        initial_sampling_method: PerAlgorithmSamplingMethod,
        acquisition_sampling_method: PerAlgorithmSamplingMethod,
    ) -> None:
        """Build isolated child experiments behind one public coordinator."""

        if stopping_rule is not None and not isinstance(stopping_rule, StoppingRule):
            raise ExperimentConfigurationError(
                "stopping_rule must be a StoppingRule instance or None."
            )
        if isinstance(save_every, bool) or not isinstance(save_every, int) or save_every < 1:
            raise ExperimentConfigurationError("save_every must be a positive integer.")
        labels = tuple(algorithms)
        initial_methods = self._sampling_methods_by_label(
            labels=labels,
            value=initial_sampling_method,
            parameter="initial_points_sampling_method",
        )
        acquisition_methods = self._sampling_methods_by_label(
            labels=labels,
            value=acquisition_sampling_method,
            parameter="acquisition_sampling_method",
        )
        if isinstance(environment, Mapping):
            missing = [label for label in labels if label not in environment]
            extra = [label for label in environment if label not in algorithms]
            if missing or extra:
                raise ExperimentConfigurationError(
                    "An environment mapping must contain exactly the algorithm labels; "
                    f"missing={missing}, extra={extra}."
                )
            environments = dict(environment)
            if any(not isinstance(item, Environment) for item in environments.values()):
                raise ExperimentConfigurationError(
                    "Every environment mapping value must be an Environment instance."
                )
            identities = [id(item) for item in environments.values()]
            if len(set(identities)) != len(identities):
                raise ExperimentConfigurationError(
                    "Every environment mapping entry must use a different Environment instance."
                )
            source_environment: Environment | Mapping[str, Environment] = MappingProxyType(
                environments
            )
        elif isinstance(environment, Environment):
            environments = {
                label: self._clone_environment_template(environment, label=label)
                for label in labels
            }
            source_environment = environment
        else:
            raise ExperimentConfigurationError(
                "environment must be one Environment template or a mapping matching the "
                "algorithm labels."
            )

        root = None if output_directory is None else Path(output_directory)
        if resume and root is None:
            raise ExperimentConfigurationError("resume=True requires output_directory.")
        child_directories: dict[str, Path | None] = {}
        layout: list[dict[str, str]] = []
        for index, (label, item) in enumerate(algorithms.items(), start=1):
            component = self._checkpoint_component_name(index, label, item)
            relative = Path("algorithms") / component
            child_directories[label] = None if root is None else root / relative
            layout.append(
                {
                    "label": label,
                    "algorithm_class": f"{type(item).__module__}.{type(item).__qualname__}",
                    "checkpoint_directory": relative.as_posix(),
                }
            )

        manifest = {
            "schema_version": self.CHECKPOINT_SCHEMA_VERSION,
            "package": "basat",
            "package_version": self.PACKAGE_VERSION,
            "mode": "multi_algorithm",
            "algorithms": layout,
        }
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
            manifest_path = root / "multi-experiment.json"
            if resume and manifest_path.is_file():
                try:
                    stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise ExperimentCheckpointError(
                        f"Multi-experiment manifest is unreadable: {manifest_path}"
                    ) from error
                if stored_manifest != manifest:
                    raise ExperimentCheckpointError(
                        "Multi-experiment labels, order, or algorithm classes do not match "
                        f"{manifest_path}."
                    )
            else:
                self._atomic_json_save(manifest, manifest_path)

        children: dict[str, Experiment] = {}
        initialization_failures: dict[str, ExperimentFailure] = {}
        for label, item in algorithms.items():
            try:
                with (
                    self._capture_warnings(label),
                    translate_backend_warnings(operation=f"{label} experiment initialization"),
                ):
                    children[label] = Experiment(
                        environment=environments[label],
                        algorithm=item,
                        stopping_rule=self._clone_stopping_rule(stopping_rule, label=label),
                        output_directory=child_directories[label],
                        resume=resume,
                        save_every=save_every,
                        checkpoint_policy=self.checkpoint_policy,
                        constant_context=constant_context,
                        initial_points_sampling_method=initial_methods[label],
                        acquisition_sampling_method=acquisition_methods[label],
                    )
                    child = children[label]
                    # Inactive children must not spend their time budget while
                    # preceding algorithms are running or being initialized.
                    child._comparison_elapsed_seconds = time.monotonic() - child._started_at
            except Exception as error:
                child_directory = child_directories[label]
                if (
                    resume
                    and child_directory is not None
                    and (child_directory / "latest.json").exists()
                ):
                    # A saved run must not be silently replaced by an empty
                    # failed child when its checkpoint cannot be restored.
                    raise
                self._report_child_failure(label=label, phase="initialization", error=error)
                initialization_failures[label] = self._failure_record(
                    label=label,
                    phase="initialization",
                    error=error,
                    completed_rounds=0,
                )

        self.source_environment = (
            source_environment
            if isinstance(source_environment, Environment)
            else next(iter(environments.values()))
        )
        self.source_environments = MappingProxyType(environments)
        self.environment = MappingProxyType(environments)
        self.algorithms = MappingProxyType(dict(algorithms))
        # Preserve the constructor field for public inspection while single-run
        # internals continue to receive an Algorithm object only.
        self.algorithm = self.algorithms  # type: ignore[assignment]
        self.experiments = MappingProxyType(children)
        self.initialization_failures = MappingProxyType(initialization_failures)
        self.failures = dict(initialization_failures)
        self._checkpoint_directories = MappingProxyType(child_directories)
        self.stopping_rule = stopping_rule
        self.output_directory = root
        self.resume = bool(resume)
        self.save_every = save_every
        self.completed_rounds = min(
            (child.completed_rounds for child in children.values()),
            default=0,
        )
        self.stopping_decision = None
        self._started_at = time.monotonic()
        self._last_checkpoint_phase = None

    @property
    def completed_rounds_by_algorithm(self) -> Mapping[str, int]:
        """Return round counts, or each algorithm's common prefix across trials."""

        if hasattr(self, "_trial_count"):
            return MappingProxyType(
                {
                    label: (
                        min(
                            child.completed_rounds_by_algorithm[label]
                            for child in self.trial_experiments
                        )
                        if len(self.trial_experiments) == self._trial_count
                        else 0
                    )
                    for label in self.algorithms
                }
            )
        if not self.is_multi_algorithm:
            return MappingProxyType({type(self.algorithm).__name__: self.completed_rounds})
        return MappingProxyType(
            {
                label: (
                    self.experiments[label].completed_rounds
                    if label in self.experiments
                    else self.initialization_failures[label].completed_rounds
                )
                for label in self.algorithms
            }
        )

    @property
    def checkpoint_directories(self) -> Mapping[str, Path | None]:
        """Return the checkpoint directory assigned to each algorithm."""

        if not self.is_multi_algorithm:
            return MappingProxyType({type(self.algorithm).__name__: self.output_directory})
        return MappingProxyType(dict(self._checkpoint_directories))

    def _result_document(self) -> dict[str, Any]:
        """Build a portable audit document without serializing live models."""

        if hasattr(self, "_trial_count"):
            return aggregate_document(self)
        rows: dict[str, Any] = {}
        for label, algorithm in self.algorithms.items():
            child = self.experiments.get(label) if self.is_multi_algorithm else self
            failure = self.failures.get(label)
            result = None
            if child is not None:
                try:
                    result = algorithm.result()
                except Exception:
                    # A failed learner can still have useful durable environment
                    # history. Do not let a broken result mask its original error.
                    if failure is None:
                        raise
            rows[label] = {
                "algorithm_class": f"{type(algorithm).__module__}.{type(algorithm).__qualname__}",
                "configuration": self._jsonable(getattr(algorithm, "config", None)),
                "maximize": algorithm.maximize,
                "bounds": {
                    "lower": self._jsonable(algorithm.bounds.lower),
                    "upper": self._jsonable(algorithm.bounds.upper),
                },
                "execution_settings": {
                    "dtype": str(algorithm.dtype),
                    "device": str(algorithm.device),
                    "initial_points_sampling_method": algorithm.initial_points_sampling_method,
                    "acquisition_sampling_method": algorithm.acquisition_sampling_method,
                },
                "environment_type": getattr(
                    self.source_environments[label], "environment_type", None
                ),
                "stopping_rule": (
                    None
                    if child is None or child.stopping_rule is None
                    else self._jsonable(child.stopping_rule.state_dict())
                ),
                "result": self._jsonable(result),
                "observations": self._jsonable(algorithm.observed_values),
                "initial_observation_count": (
                    0
                    if child is None or child.environment.history.initial_data is None
                    else len(child.environment.history.initial_data.actions)
                ),
                "completed_rounds": 0 if child is None else child.completed_rounds,
                "environment_evaluations": (
                    0 if child is None else len(child.environment.history.records)
                ),
                "stopping_reason": (
                    None
                    if child is None
                    else (
                        child.stopping_decision.reason
                        if child.stopping_decision is not None and child.stopping_decision.stop
                        else child._run_stop_reason
                    )
                ),
                "warnings": list(self.warnings.get(label, [])),
                "failure": self._jsonable(failure),
                "output_directory": (
                    None
                    if self.checkpoint_directories[label] is None
                    else str(self.checkpoint_directories[label])
                ),
            }
        return {
            "schema_version": self.CHECKPOINT_SCHEMA_VERSION,
            "package": "basat",
            "package_version": self.PACKAGE_VERSION,
            "mode": "multi_algorithm" if self.is_multi_algorithm else "single_algorithm",
            "limits": self._jsonable(self._last_requested_limits),
            "persistence": {
                "checkpoint_policy": self.checkpoint_policy,
                "save_every": self.save_every,
                "periodic_unsaved_round_bound": (
                    self.save_every if self.checkpoint_policy == "periodic" else 0
                ),
            },
            "algorithm_order": list(self.algorithms),
            "algorithms": rows,
        }

    def save(self, path: str | os.PathLike[str] | None = None) -> Path:
        """Save readable results, settings, warnings and errors as JSON.

        With an output directory, run() writes this automatically. Multiple
        algorithms also get individual result.json files in their own checkpoint
        directories. Checkpoints remain available for exact continuation.
        """

        if path is None:
            if self.output_directory is None:
                raise ExperimentConfigurationError("save() needs path or output_directory.")
            path = self.output_directory / (
                "results.json" if self.is_multi_algorithm else "result.json"
            )
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        document = self._result_document()
        self._atomic_json_save(document, target)
        if self.is_multi_algorithm and not hasattr(self, "_trial_count"):
            for label, directory in self.checkpoint_directories.items():
                if directory is not None:
                    directory.mkdir(parents=True, exist_ok=True)
                    self._atomic_json_save(
                        {"label": label, **document["algorithms"][label]}, directory / "result.json"
                    )
        return target

    def summary(self) -> str:
        """Return one concise result line for every requested algorithm."""

        if hasattr(self, "_trial_count"):
            return "\n".join(
                [f"Basat: {self._trial_count} trials"]
                + [
                    f"{label}: evaluations per trial={row['trial_evaluations']}; "
                    f"failures={sum(f is not None for f in row['trial_failures'])}; "
                    f"mean final incumbent={row['result']['best_observed_value']}"
                    for label, row in self._result_document()["algorithms"].items()
                ]
            )
        lines = [f"Basat experiment: {len(self.algorithms)} algorithm(s)"]
        for label, row in self._result_document()["algorithms"].items():
            if row["failure"] is not None:
                lines.append(
                    f"{label}: FAILED during {row['failure']['phase']}; "
                    f"evaluations={row['environment_evaluations']}"
                )
            else:
                result = row["result"]
                lines.append(
                    f"{label}: evaluations={result['evaluations']}, "
                    f"best={result['best_observed_value']}, "
                    f"stop={row['stopping_reason'] or result['status']}"
                )
        return "\n".join(lines)

    def diagnostics(self, *, include_tracebacks: bool = False) -> str:
        """Return all distinct warnings and failures, in algorithm order."""

        if hasattr(self, "_trial_count"):
            grouped: dict[str, list[int]] = {}
            for i, child in enumerate(self.trial_experiments, start=1):
                for label in self.algorithms:
                    for message in child.warnings.get(label, []):
                        grouped.setdefault(f"{label}: {message}", []).append(i)
                    failure = child.failures.get(label)
                    if failure is not None:
                        text = f"{label}: ERROR ({failure.phase}): {failure.message}"
                        if include_tracebacks and failure.traceback:
                            text += "\n" + failure.traceback
                        grouped.setdefault(text, []).append(i)
            return (
                "\n".join(f"Trials {indices}: {message}" for message, indices in grouped.items())
                or "No warnings or errors."
            )
        lines = []
        for label in self.algorithms:
            lines.append(f"{label}:")
            messages = self.warnings.get(label, [])
            lines.extend(f"  {message}" for message in messages)
            failure = self.failures.get(label)
            if failure is not None:
                lines.append(
                    f"  ERROR ({failure.phase}): {failure.exception_type}: {failure.message}"
                )
                if include_tracebacks and failure.traceback:
                    lines.append(failure.traceback)
            elif not messages:
                lines.append("  No warnings or errors.")
        return "\n".join(lines)

    def plot(self, **settings: Any) -> Any:
        """Plot this experiment through basat.plot, with one or several traces."""

        from .plotting import plot

        return plot(self, **settings)

    @staticmethod
    def _rule_evaluation_ceiling(rule: StoppingRule | None) -> int | None:
        """Find a hard evaluation bound guaranteed by built-in rule composition."""

        if isinstance(rule, MaximumEvaluationsStoppingRule):
            return rule.maximum_evaluations
        if isinstance(rule, CompositeStoppingRule):
            bounds = [Experiment._rule_evaluation_ceiling(child) for child in rule.rules]
            finite = [bound for bound in bounds if bound is not None]
            if rule.mode == "any":
                return min(finite) if finite else None
            # all_of is only bounded in evaluations if every child is bounded.
            return max(finite) if len(finite) == len(bounds) else None
        return None

    def _initialize_algorithm_exception_handler(self) -> Any:
        """Construct the concrete algorithm's initialization controller."""

        exception_class = self.algorithm.EXCEPTION_CLASS
        if exception_class is None:  # guarded during construction; retained for type safety
            raise ExperimentConfigurationError(
                f"{type(self.algorithm).__name__} must declare EXCEPTION_CLASS."
            )
        return exception_class(environment=self.environment, algorithm=self.algorithm)

    def _prepare_storage(self) -> None:
        """Create the run directory without reading history or evaluating data."""

        if self.output_directory is None:
            if self.resume:
                raise ExperimentConfigurationError("resume=True requires output_directory.")
            return
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def _initialize_algorithm_from_environment(self) -> None:
        """Pass only learner-visible history to a fresh algorithm."""

        data = self.environment.get_history_data()
        self.algorithm.initialize_from_environment(data)
        self.algorithm_consumed_environment_records = len(self.environment.history.records)

    @staticmethod
    def _validate_round_info(round_info: Any, required: frozenset[str]) -> dict[str, Any]:
        """Return a copied round-information mapping after required-key checks."""

        if not isinstance(round_info, Mapping):
            raise ExperimentProtocolError("before_action() must return a mapping.")
        missing = required.difference(round_info)
        if missing:
            raise ExperimentProtocolError(f"Round information is missing {sorted(missing)}.")
        return dict(round_info)

    @staticmethod
    def _validate_feedback(feedback: Any, required: frozenset[str]) -> dict[str, Any]:
        """Return copied feedback after validating its public required fields."""

        if not isinstance(feedback, Mapping):
            raise ExperimentProtocolError("step() must return a feedback mapping.")
        missing = required.difference(feedback)
        if missing:
            raise ExperimentProtocolError(f"Feedback is missing {sorted(missing)}.")
        return dict(feedback)

    @staticmethod
    def _jsonable(value: Any) -> Any:
        """Convert diagnostics metadata to a stable JSON-compatible structure."""

        if hasattr(value, "to_dict"):
            return Experiment._jsonable(value.to_dict())
        if is_dataclass(value) and not isinstance(value, type):
            return Experiment._jsonable(asdict(value))
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, Mapping):
            return {str(k): Experiment._jsonable(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [Experiment._jsonable(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    def _checkpoint_payload(self, *, phase: str) -> dict[str, Any]:
        """Assemble the complete versioned state written at a crash boundary."""

        stopping_state = None if self.stopping_rule is None else self.stopping_rule.state_dict()
        return {
            "schema_version": self.CHECKPOINT_SCHEMA_VERSION,
            "package": "basat",
            "package_version": self.PACKAGE_VERSION,
            "experiment_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "algorithm_class": (
                f"{type(self.algorithm).__module__}.{type(self.algorithm).__qualname__}"
            ),
            "environment_class": (
                f"{type(self.environment).__module__}.{type(self.environment).__qualname__}"
            ),
            "trial_rng_state": rng_state() if self._trial_managed else None,
            "algorithm_state": self.algorithm.state_dict(),
            "environment_state": self.environment.state_dict(),
            "algorithm_consumed_environment_records": (self.algorithm_consumed_environment_records),
            "stopping_state": stopping_state,
            "initialization_exception_diagnostics": self._jsonable(
                self.initialization_exception_diagnostics
            ),
            "compatibility": {
                "compatibility_status": self.algorithm.compatibility_status,
                "adapter_name": self.algorithm.adapter_name,
                "source_guarantees_applicable": (self.algorithm.source_guarantees_applicable),
                "warning_codes": list(self.algorithm._last_warning_codes),
            },
            "experiment_state": {
                "completed_rounds": self.completed_rounds,
                "save_every": self.save_every,
                "checkpoint_policy": self.checkpoint_policy,
                "phase": phase,
                "elapsed_seconds": time.monotonic() - self._started_at,
                "stopping_decision": self._jsonable(self.stopping_decision),
            },
        }

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        """Flush a directory entry after an atomic checkpoint replacement.

        POSIX durability requires both the file contents and the containing
        directory entry to reach stable storage.  Windows does not expose a
        portable directory ``fsync`` operation, so the already flushed file and
        atomic replacement remain the strongest available guarantee there.
        """

        if os.name == "nt":  # pragma: no cover - exercised on Windows CI
            return
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _atomic_torch_save(payload: Mapping[str, Any], target: Path) -> str:
        """Atomically save a checkpoint and return its SHA-256 digest."""

        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as stream:
                temporary_name = stream.name
                torch.save(dict(payload), stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary = Path(temporary_name)
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            os.replace(temporary, target)
            Experiment._fsync_directory(target.parent)
            return digest
        finally:
            if temporary_name is not None:
                temporary = Path(temporary_name)
                if temporary.exists():
                    temporary.unlink()

    @staticmethod
    def _atomic_json_save(payload: Mapping[str, Any], target: Path) -> None:
        """Write a JSON manifest using flush, atomic replacement, and directory sync."""

        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as stream:
                temporary_name = stream.name
                json.dump(dict(payload), stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, target)
            Experiment._fsync_directory(target.parent)
        finally:
            if temporary_name is not None:
                temporary = Path(temporary_name)
                if temporary.exists():
                    temporary.unlink()

    def save_checkpoint(self, *, phase: str = "post_tell", archive: bool = False) -> Path | None:
        """Persist a versioned checkpoint and checksum manifest atomically.

        Parameters
        ----------
        phase:
            Nonempty label for the current crash boundary.
        archive:
            Also retain a round-numbered immutable checkpoint when true.

        Returns
        -------
        pathlib.Path | None
            Convenience latest-checkpoint path, or ``None`` for in-memory runs.
        """

        if self.is_multi_algorithm:
            raise ExperimentConfigurationError(
                "Multi-algorithm checkpoints are saved by each child experiment during run(); "
                "inspect experiment.experiments[label] for an individual checkpoint."
            )
        if not isinstance(phase, str) or not phase.strip():
            raise ExperimentConfigurationError("checkpoint phase must be a non-empty string.")
        if not isinstance(archive, bool):
            raise ExperimentConfigurationError("checkpoint archive must be Boolean.")
        if self.output_directory is None:
            return None
        payload = self._checkpoint_payload(phase=phase)
        manifest_path = self.output_directory / "latest.json"
        previous_slot: str | None = None
        if manifest_path.is_file():
            try:
                previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(previous_manifest, Mapping):
                    previous_slot = previous_manifest.get("checkpoint")
            except (OSError, json.JSONDecodeError):
                previous_slot = None
        slot_name = (
            "checkpoint-slot-b.pt"
            if previous_slot == "checkpoint-slot-a.pt"
            else "checkpoint-slot-a.pt"
        )
        authoritative = self.output_directory / slot_name
        digest = self._atomic_torch_save(payload, authoritative)

        # Preserve a conventional latest filename for inspection. Recovery uses
        # the alternating slot selected by the checksumed atomic manifest, so a
        # crash between these independent replacements always leaves one valid
        # manifest-generation pair.
        latest = self.output_directory / "checkpoint-latest.pt"
        self._atomic_torch_save(payload, latest)
        manifest = {
            "schema_version": self.CHECKPOINT_SCHEMA_VERSION,
            "checkpoint": authoritative.name,
            "sha256": digest,
            "phase": phase,
            "completed_rounds": self.completed_rounds,
            "consumed_records": self.algorithm_consumed_environment_records,
            "compatibility_status": self.algorithm.compatibility_status,
            "adapter_name": self.algorithm.adapter_name,
            "source_guarantees_applicable": self.algorithm.source_guarantees_applicable,
            "warning_codes": list(self.algorithm._last_warning_codes),
        }
        self._atomic_json_save(manifest, manifest_path)
        if archive:
            archive_path = self.output_directory / f"checkpoint-{self.completed_rounds:08d}.pt"
            self._atomic_torch_save(payload, archive_path)
        self._last_checkpoint_phase = phase
        return latest

    def load_checkpoint(self) -> None:
        """Validate and restore the latest transactional checkpoint."""

        if self.is_multi_algorithm:
            raise ExperimentConfigurationError(
                "Multi-algorithm checkpoints are loaded independently during construction."
            )
        if self.output_directory is None:
            raise ExperimentCheckpointError("No output directory is configured.")
        manifest_path = self.output_directory / "latest.json"
        if not manifest_path.is_file():
            raise ExperimentCheckpointError(f"Checkpoint manifest not found: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, Mapping):
                raise TypeError("manifest must be a mapping")
            checkpoint_name = manifest["checkpoint"]
            if checkpoint_name not in {"checkpoint-slot-a.pt", "checkpoint-slot-b.pt"}:
                raise ValueError("manifest checkpoint target is invalid")
            checkpoint_path = self.output_directory / checkpoint_name
            raw = checkpoint_path.read_bytes()
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ExperimentCheckpointError("Checkpoint manifest is unreadable.") from exc
        actual_digest = hashlib.sha256(raw).hexdigest()
        if actual_digest != manifest.get("sha256"):
            raise ExperimentCheckpointError("Checkpoint checksum does not match the manifest.")
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as exc:  # pragma: no cover - backend supplies many error types
            raise ExperimentCheckpointError("Checkpoint payload is unreadable.") from exc
        if not isinstance(payload, Mapping):
            raise ExperimentCheckpointError("Checkpoint payload must be a mapping.")
        if payload.get("schema_version") != self.CHECKPOINT_SCHEMA_VERSION:
            raise ExperimentCheckpointError("Unsupported checkpoint schema version.")
        if (
            payload.get("package") != "basat"
            or payload.get("package_version") != self.PACKAGE_VERSION
        ):
            raise ExperimentCheckpointError("Checkpoint belongs to an incompatible package.")
        expected_identities = {
            "experiment_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "algorithm_class": (
                f"{type(self.algorithm).__module__}.{type(self.algorithm).__qualname__}"
            ),
            "environment_class": (
                f"{type(self.environment).__module__}.{type(self.environment).__qualname__}"
            ),
        }
        for key, identity in expected_identities.items():
            if payload.get(key) != identity:
                raise ExperimentCheckpointError(
                    f"Checkpoint {key} does not match the configured experiment."
                )

        required = {
            "algorithm_state",
            "environment_state",
            "algorithm_consumed_environment_records",
            "stopping_state",
            "experiment_state",
        }
        if missing := required.difference(payload):
            raise ExperimentCheckpointError(
                f"Checkpoint payload is missing required fields: {sorted(missing)}."
            )

        checkpoint_stopping_state = payload["stopping_state"]
        if (self.stopping_rule is None) != (checkpoint_stopping_state is None):
            raise ExperimentCheckpointError(
                "Checkpoint stopping-rule presence does not match configuration."
            )

        cursor = payload["algorithm_consumed_environment_records"]
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ExperimentCheckpointError(
                "Checkpoint consumed-record cursor must be a non-negative integer."
            )
        state = payload["experiment_state"]
        if not isinstance(state, Mapping):
            raise ExperimentCheckpointError("Checkpoint experiment_state must be a mapping.")
        environment_state = payload["environment_state"]
        if not isinstance(environment_state, Mapping):
            raise ExperimentCheckpointError("Checkpoint environment_state must be a mapping.")
        completed_rounds = state.get("completed_rounds")
        if (
            isinstance(completed_rounds, bool)
            or not isinstance(completed_rounds, int)
            or completed_rounds < 0
            or completed_rounds != environment_state.get("round")
        ):
            raise ExperimentCheckpointError(
                "Checkpoint completed_rounds is invalid or disagrees with environment state."
            )
        saved_frequency = state.get("save_every")
        if (
            isinstance(saved_frequency, bool)
            or not isinstance(saved_frequency, int)
            or saved_frequency != self.save_every
        ):
            raise ExperimentCheckpointError(
                "Checkpoint save_every does not match experiment configuration."
            )
        elapsed_seconds = state.get("elapsed_seconds")
        if (
            isinstance(elapsed_seconds, bool)
            or not isinstance(elapsed_seconds, (int, float))
            or not math.isfinite(float(elapsed_seconds))
            or elapsed_seconds < 0
        ):
            raise ExperimentCheckpointError(
                "Checkpoint elapsed_seconds must be finite and non-negative."
            )
        phase = state.get("phase")
        if phase is not None and (not isinstance(phase, str) or not phase.strip()):
            raise ExperimentCheckpointError("Checkpoint phase must be a non-empty string.")

        decision = state.get("stopping_decision")
        restored_decision: Any | None
        if decision is None:
            restored_decision = None
        elif isinstance(decision, Mapping):
            from .stopping.stopping_rule import StoppingDecision

            stop_value = decision.get("stop")
            if not isinstance(stop_value, bool):
                raise ExperimentCheckpointError(
                    "Checkpoint stopping_decision.stop must be Boolean."
                )
            try:
                restored_decision = StoppingDecision(
                    stop=stop_value,
                    reason=decision.get("reason"),
                    details=decision.get("details", {}),
                )
            except (TypeError, ValueError) as exc:
                raise ExperimentCheckpointError("Checkpoint stopping_decision is invalid.") from exc
        else:
            raise ExperimentCheckpointError(
                "Checkpoint stopping_decision must be a mapping or None."
            )

        self._trial_rng_state = payload.get("trial_rng_state")
        self.environment.load_state_dict(environment_state, strict=True)
        self.algorithm.load_state_dict(payload["algorithm_state"], strict=True)
        if self.stopping_rule is not None:
            self.stopping_rule.load_state_dict(checkpoint_stopping_state, strict=True)

        self.algorithm_consumed_environment_records = cursor
        self.completed_rounds = completed_rounds
        self._started_at = time.monotonic() - float(elapsed_seconds)
        self._last_checkpoint_phase = phase
        self.stopping_decision = restored_decision

    @staticmethod
    def _records_to_exchange(
        records: Sequence[Any],
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        """Reconstruct one public ask-tell exchange from persisted records."""

        if not records:
            raise ExperimentCheckpointError("Cannot replay an empty record group.")
        actions = torch.stack([record.action for record in records])
        evaluated = torch.stack([record.evaluated_action for record in records])
        observations = torch.stack(
            [torch.as_tensor(record.observation).reshape(()) for record in records]
        )
        feedback: dict[str, Any] = {
            "observation": observations,
            "evaluated_action": evaluated,
        }
        contexts = [getattr(record, "context", None) for record in records]
        if all(context is not None for context in contexts):
            feedback["context"] = torch.stack([torch.as_tensor(context) for context in contexts])
        variances = [getattr(record, "observation_noise_variance", None) for record in records]
        if all(variance is not None for variance in variances):
            feedback["observation_noise_variance"] = torch.stack(
                [torch.as_tensor(variance).reshape(()) for variance in variances]
            )
        extras = [getattr(record, "feedback_extra", {}) or {} for record in records]
        public_keys = set.intersection(*(set(extra) for extra in extras)) if extras else set()
        for key in public_keys:
            values = [extra[key] for extra in extras]
            try:
                feedback[key] = torch.stack([torch.as_tensor(value) for value in values])
            except (TypeError, RuntimeError, ValueError):
                feedback[key] = values
        round_info = dict(records[0].round_info)
        if len(records) == 1:
            actions = actions[0]
            for key, value in tuple(feedback.items()):
                if isinstance(value, torch.Tensor) and value.shape[:1] == (1,):
                    feedback[key] = value[0]
        return actions, feedback, round_info

    def _replay_unconsumed_environment_records(self) -> None:
        """Replay durable records into the algorithm without reevaluation."""

        records = self.environment.history.records
        cursor = self.algorithm_consumed_environment_records
        if cursor < 0 or cursor > len(records):
            raise ExperimentCheckpointError("Consumed-record cursor is outside history.")
        if (
            0 < cursor < len(records)
            and records[cursor - 1].round_index == records[cursor].round_index
        ):
            raise ExperimentCheckpointError(
                "Consumed-record cursor splits an atomic environment batch."
            )
        while cursor < len(records):
            round_index = records[cursor].round_index
            end = cursor + 1
            while end < len(records) and records[end].round_index == round_index:
                end += 1
            action, feedback, round_info = self._records_to_exchange(records[cursor:end])
            pending = self.algorithm.last_ask
            if pending is not None:
                recorded_rows = action.unsqueeze(0) if action.ndim == 1 else action
                expected_rows = pending.actions.to(
                    dtype=self.environment.dtype, device=self.environment.device
                )
                if not torch.equal(recorded_rows, expected_rows):
                    raise ExperimentCheckpointError(
                        "Persisted environment actions do not match the pending ask "
                        "at the environment's precision."
                    )
                # The environment may store a lower-precision representation
                # than the learner. Normal execution tells the original ask,
                # so replay must recover that same proposal from durable state
                # after validating its persisted environment representation.
                # Keep Algorithm.tell's exact pending-action check unchanged.
                action = pending.actions.detach().clone()
            self.algorithm.tell(action=action, feedback=feedback, round_info=round_info)
            self._check_stopping(records[cursor:end], feedback)
            cursor = end
            self.algorithm_consumed_environment_records = cursor
        if self.output_directory is not None and self._last_checkpoint_phase == "pre_tell":
            self.save_checkpoint(phase="post_replay", archive=False)

    def _stopping_context(self, records: Sequence[Any], feedback: Mapping[str, Any]) -> Any:
        """Build a learner-visible stopping snapshot for the latest exchange."""

        from .stopping.stopping_rule import StoppingContext

        return StoppingContext(
            completed_rounds=self.completed_rounds,
            evaluations=int(getattr(self.algorithm, "evaluations", 0)),
            elapsed_seconds=time.monotonic() - self._started_at,
            latest_records=tuple(records),
            latest_feedback=dict(feedback),
            history=self.environment.get_history_data(),
            algorithm_diagnostics=self.algorithm.diagnostics(),
            algorithm_result=self.algorithm.result(),
        )

    def _check_stopping(self, records: Sequence[Any], feedback: Mapping[str, Any]) -> Any | None:
        """Evaluate and retain the configured stopping decision, if any."""

        if self.stopping_rule is None:
            return None
        self.stopping_decision = self.stopping_rule.should_stop(
            self._stopping_context(records, feedback)
        )
        return self.stopping_decision

    def _should_checkpoint(self) -> bool:
        """Return whether the current round is due for an archived checkpoint."""

        return self.completed_rounds % self.save_every == 0

    def _report_settings(self) -> None:
        """Print the user-selected algorithm and experiment settings once per run call."""

        algorithm = self.algorithm
        print(f"Algorithm: {type(algorithm).__name__}")
        print(f"Acquisition policy: {algorithm.ACQUISITION_POLICY}")
        print(f"Bounds: lower={algorithm.bounds.lower}, upper={algorithm.bounds.upper}")
        print(f"Configuration: {getattr(algorithm, 'config', None)!r}")
        print(
            "Execution: "
            f"maximize={algorithm.maximize}, dtype={algorithm.dtype}, device={algorithm.device}, "
            f"initial_points_sampling={algorithm.initial_points_sampling_method!r}, "
            f"acquisition_sampling={algorithm.acquisition_sampling_method!r}"
        )
        print(
            "Persistence: "
            f"output_directory={self.output_directory}, resume={self.resume}, "
            f"save_every={self.save_every}"
        )

    def _report_iteration(self, result: AlgorithmResult, feedback: Mapping[str, Any]) -> None:
        """Print one compact line after a completed ask-evaluate-tell round."""

        observation = self._jsonable(feedback.get("observation"))
        acquisition = result.diagnostics.last_acquisition_value
        print(
            f"[{type(self.algorithm).__name__}] round={self.completed_rounds} "
            f"evaluations={result.evaluations} observation={observation} "
            f"best={result.best_observed_value} acquisition={acquisition}"
        )

    def _report_final(self, result: AlgorithmResult) -> None:
        """Print the final recommendation after one run invocation."""

        print(
            f"[{type(self.algorithm).__name__}] finished run call: "
            f"rounds={self.completed_rounds}, evaluations={result.evaluations}, "
            f"recommended_value={result.recommended_value}, "
            f"best_observed_value={result.best_observed_value}"
        )

    def run_one_step(self) -> AlgorithmResult:
        """Execute one complete environment round and return current result."""

        if hasattr(self, "_trial_count"):
            raise ExperimentConfigurationError(
                "run_one_step() cannot advance a repeated experiment; "
                "continue with run(trials=the_saved_trial_count)."
            )
        if self.is_multi_algorithm:
            raise ExperimentConfigurationError(
                "run_one_step() is available only for a single-algorithm Experiment; "
                "call run() to execute a multi-algorithm experiment independently."
            )
        if self.stopping_decision is not None and self.stopping_decision.stop:
            return self.algorithm.result()
        # Check limits before evaluating the objective, including a zero budget
        # and wall-time elapsed between separate run() calls.
        decision = self._check_stopping((), {})
        if decision is not None and decision.stop:
            self.save_checkpoint(phase="stopped", archive=False)
            return self.algorithm.result()
        evaluation_ceiling = self._rule_evaluation_ceiling(self.stopping_rule)
        if (
            evaluation_ceiling is not None
            and self.algorithm.evaluations + self.algorithm.request_size > evaluation_ceiling
        ):
            self.stopping_decision = StoppingDecision(
                stop=True,
                reason="insufficient_evaluations_for_batch",
                details={
                    "limit": evaluation_ceiling,
                    "evaluations": self.algorithm.evaluations,
                    "batch_size": self.algorithm.request_size,
                },
            )
            self.save_checkpoint(phase="stopped", archive=False)
            return self.algorithm.result()
        try:
            round_info = self._validate_round_info(
                self.algorithm.prepare_round_info(self.environment.before_action()),
                self.algorithm.required_round_info,
            )
        except NoAvailableActionError:
            self.stopping_decision = StoppingDecision(
                stop=True, reason="finite_candidates_exhausted"
            )
            self.save_checkpoint(phase="stopped", archive=False)
            return self.algorithm.result()
        candidates = round_info.get("candidate_actions")
        if (
            isinstance(candidates, torch.Tensor)
            and candidates.shape[0] < self.algorithm.request_size
        ):
            self.stopping_decision = StoppingDecision(
                stop=True,
                reason="insufficient_candidates_for_batch",
                details={
                    "available": candidates.shape[0],
                    "batch_size": self.algorithm.request_size,
                },
            )
            self.save_checkpoint(phase="stopped", archive=False)
            return self.algorithm.result()
        action = self.algorithm.ask(round_info=round_info)
        record_start = len(self.environment.history.records)
        feedback = self._validate_feedback(
            self.algorithm.prepare_feedback(self.environment.step(action), action=action),
            self.algorithm.required_feedback,
        )
        new_records = self.environment.history.records[record_start:]
        if not new_records:
            raise ExperimentProtocolError("step() returned feedback without appending history.")

        # Transactional mode persists feedback before learner state. Periodic
        # mode is explicit opt-in for replayable offline objectives: up to
        # save_every rounds since the last save can be lost after a crash.
        self.completed_rounds = int(self.environment.round)
        if self.checkpoint_policy == "transactional":
            self.save_checkpoint(phase="pre_tell", archive=False)
        self.algorithm.tell(action=action, feedback=feedback, round_info=round_info)
        self.algorithm_consumed_environment_records = len(self.environment.history.records)
        self._check_stopping(new_records, feedback)
        if self.checkpoint_policy == "transactional" or self._should_checkpoint():
            self.save_checkpoint(phase="post_tell", archive=self._should_checkpoint())
        elif self.stopping_decision is not None and self.stopping_decision.stop:
            self.save_checkpoint(phase="stopped", archive=False)
        result = self.algorithm.result()
        if self.algorithm.verbose_iter_by_iter:
            self._report_iteration(result, feedback)
        return result

    def run(
        self,
        *,
        trials: int = 1,
        max_rounds: int | Mapping[str, int | None] | None = None,
        max_evaluations: int | Mapping[str, int | None] | None = None,
        budget_policy: Literal["strict", "finish_iteration"] = "strict",
        plot: bool | None = None,
        show: bool = True,
        title: str = "Basat algorithm comparison",
        progress: bool = False,
    ) -> AlgorithmResult | ExperimentComparisonResult | ExperimentTrialsResult:
        """Run one algorithm or every configured algorithm independently.

        trials is a positive integer (default 1). With trials > 1, each algorithm
        is reconstructed with seed base_seed + trial_index, as are explicitly
        seeded nested settings. The environment seed advances independently from
        its own base seed. Every trial has its own history, stopping rule and
        evaluation ceiling. Results include raw outcomes and pointwise means,
        sample standard deviations and contributing counts. A best-so-far curve
        is computed within each trial before averaging.

        Repeated runs use trials/trial-NNN checkpoint directories. Resume with
        the same base settings and trials count; run-call ceilings are totals per
        trial, not additional evaluations. trials=1 keeps the existing return
        type; trials>1 returns ExperimentTrialsResult. Its results[label] contains
        one outcome per trial (None for a failed algorithm trial).

        max_evaluations is a total objective-evaluation ceiling per algorithm,
        including observations restored from checkpoints. max_rounds is a total
        ask/evaluate/tell-round ceiling. Either can be a label-to-limit mapping
        for multiple algorithms. The first limit or stopping rule reached wins.
        A batch that does not fit the remaining budget is not evaluated; its
        batch size and acquisition semantics are preserved. A later run call may
        extend a run-call ceiling. Configured stopping rules remain stopped.

        budget_policy="strict" preserves this hard evaluation ceiling.
        "finish_iteration" treats it as a nominal source budget: a started
        batch or nested outer iteration completes before the evaluation limit
        is checked. Actual evaluations can exceed the nominal budget and are
        always retained in results. Initialization counts toward the budget.
        Explicit max_rounds and stopping rules remain hard limits. Resuming an
        interrupted source iteration finishes it; a completed boundary at or
        beyond the target performs no more objective evaluations.

        plot=True creates a single figure containing all configured algorithms,
        including a labeled legend entry for failed algorithms. show=False
        creates it without displaying it. progress=True prints each algorithm's
        start and completion. Warnings, failures and machine-readable results
        are retained on the experiment and saved under output_directory.
        """

        if not isinstance(budget_policy, str) or budget_policy not in {"strict", "finish_iteration"}:
            raise ExperimentConfigurationError(
                "budget_policy must be 'strict' or 'finish_iteration'."
            )
        if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
            raise ExperimentConfigurationError("trials must be a positive integer.")
        if trials == 1 and (
            hasattr(self, "_trial_count")
            or (
                self.output_directory is not None
                and (self.output_directory / "trials.json").exists()
            )
        ):
            raise ExperimentConfigurationError(
                "This is a repeated experiment; supply its saved trials count."
            )
        self._last_requested_limits = {
            "max_rounds": max_rounds, "max_evaluations": max_evaluations,
            "budget_policy": budget_policy,
        }
        self._run_stop_reason = None
        parameters: dict[str, Any] = dict(
            max_rounds=max_rounds,
            max_evaluations=max_evaluations,
            budget_policy=budget_policy,
            plot=plot,
            show=show,
            title=title,
            progress=progress,
        )
        if trials > 1:
            return run_trials(self, trials, parameters)
        try:
            if self.is_multi_algorithm:
                return self._run_impl(**parameters)
            label = type(self.algorithm).__name__
            with self._capture_warnings(label), trial_rng(self):
                result = self._run_impl(**parameters)
            self.failures.pop(label, None)
            return result
        except Exception as error:
            if not self.is_multi_algorithm:
                label = type(self.algorithm).__name__
                self.failures[label] = self._failure_record(
                    label=label,
                    phase="run",
                    error=error,
                    completed_rounds=self.completed_rounds,
                )
            raise
        finally:
            if self.output_directory is not None:
                self.save()

    def _run_impl(
        self,
        *,
        max_rounds: int | Mapping[str, int | None] | None = None,
        max_evaluations: int | Mapping[str, int | None] | None = None,
        budget_policy: Literal["strict", "finish_iteration"] = "strict",
        plot: bool | None = None,
        show: bool = True,
        title: str = "Basat algorithm comparison",
        progress: bool = False,
    ) -> AlgorithmResult | ExperimentComparisonResult:
        """Run until a stopping rule fires or ``max_rounds`` are completed.

        Parameters
        ----------
        max_rounds:
            Optional non-negative total round ceiling. If a checkpoint has
            fewer completed rounds, Basat runs only the difference; if it has
            already reached or exceeded the ceiling, no new evaluation occurs.
            A multi-algorithm experiment also accepts a mapping from labels to
            separate total ceilings.
        plot:
            Override the algorithm's ``plot`` presentation flag for this call.
            ``None`` preserves each algorithm's configured flag. In
            multi-algorithm mode only enabled algorithms appear in the one
            combined figure. ``True`` forces every trace on and ``False``
            disables plotting for this call.
        show:
            Display the progress figure when plotting is enabled. Set to
            ``False`` to create the figure without displaying it.
        title:
            Combined figure title in multi-algorithm mode.

        Returns
        -------
        AlgorithmResult or ExperimentComparisonResult
            One result for a single algorithm, or all independent results and
            the optional combined figure for multiple algorithms.

        Raises
        ------
        ExperimentConfigurationError
            If neither a finite round bound nor a stopping rule is configured.
        """

        if not isinstance(progress, bool):
            raise ExperimentConfigurationError("progress must be bool.")
        if plot is not None and not isinstance(plot, bool):
            raise ExperimentConfigurationError("plot must be bool or None.")
        if not isinstance(show, bool):
            raise ExperimentConfigurationError("show must be bool.")
        if not isinstance(title, str) or not title.strip():
            raise ExperimentConfigurationError("title must be a non-empty string.")
        if self.is_multi_algorithm:
            ceilings = self._comparison_round_ceilings(tuple(self.algorithms), max_rounds)
            evaluation_ceilings = self._comparison_round_ceilings(
                tuple(self.algorithms), max_evaluations, parameter="max_evaluations"
            )
            typed_results: dict[str, AlgorithmResult] = {}
            failures = dict(self.initialization_failures)
            for label, child in self.experiments.items():
                child._started_at = time.monotonic() - child._comparison_elapsed_seconds
                if progress:
                    print(f"[{label}] starting", flush=True)
                try:
                    with (
                        self._capture_warnings(label),
                        translate_backend_warnings(operation=f"{label} experiment run"),
                    ):
                        result = child.run(
                            max_rounds=ceilings[label],
                            max_evaluations=evaluation_ceilings[label],
                            budget_policy=budget_policy,
                            plot=False,
                            show=False,
                        )
                    if not isinstance(result, AlgorithmResult):  # defensive: child is single mode
                        raise ExperimentProtocolError(
                            "A child experiment unexpectedly returned a comparison result."
                        )
                    typed_results[label] = result
                except Exception as error:
                    self._report_child_failure(label=label, phase="run", error=error)
                    failures[label] = self._failure_record(
                        label=label,
                        phase="run",
                        error=error,
                        completed_rounds=child.completed_rounds,
                    )
                finally:
                    child._comparison_elapsed_seconds = time.monotonic() - child._started_at
                    self.failures = dict(failures)
                    if progress:
                        outcome = (
                            "FAILED"
                            if label in failures
                            else (
                                f"evaluations={child.algorithm.evaluations}, "
                                f"best={child.algorithm.result().best_observed_value}"
                            )
                        )
                        print(f"[{label}] {outcome}", flush=True)
                    if self.output_directory is not None:
                        self.save()
            self.completed_rounds = min(
                (child.completed_rounds for child in self.experiments.values()),
                default=0,
            )
            self.failures = dict(failures)
            plotted = {
                label: algorithm
                for label, algorithm in self.algorithms.items()
                if plot is True or (plot is None and algorithm.plot)
            }
            if not plotted or not any(
                algorithm.observed_values.numel() > 0 or label in failures
                for label, algorithm in plotted.items()
            ):
                return ExperimentComparisonResult(results=typed_results, failures=failures)
            from .plotting import plot as plot_experiment

            visual = plot_experiment(
                SimpleNamespace(
                    algorithms=plotted,
                    failures={label: failures[label] for label in plotted if label in failures},
                ),
                show=show,
                title=title,
            )
            return ExperimentComparisonResult(
                results=typed_results,
                figure=visual.figure,
                axes=tuple(visual.axes),
                failures=failures,
            )

        if isinstance(max_rounds, Mapping):
            raise ExperimentConfigurationError(
                "A max_rounds mapping is valid only for a multi-algorithm Experiment."
            )
        if isinstance(max_evaluations, Mapping):
            raise ExperimentConfigurationError(
                "A max_evaluations mapping is valid only for a multi-algorithm Experiment."
            )
        if max_rounds is None and max_evaluations is None and self.stopping_rule is None:
            raise ExperimentConfigurationError(
                "run() requires max_rounds, max_evaluations or a stopping_rule "
                "to prevent an unbounded run."
            )
        if max_evaluations is not None and (
            isinstance(max_evaluations, bool)
            or not isinstance(max_evaluations, int)
            or max_evaluations < 0
        ):
            raise ExperimentConfigurationError("max_evaluations must be a non-negative integer.")
        if max_rounds is not None and (
            isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 0
        ):
            raise ExperimentConfigurationError("max_rounds must be a non-negative integer.")
        target = max_rounds
        if self.algorithm.verbose_setting:
            self._report_settings()
        if self.algorithm.verbose:
            requested = (
                "stopping rule"
                if target is None
                else f"total round ceiling={target}; remaining={max(0, target - self.completed_rounds)}"
            )
            print(
                f"[{type(self.algorithm).__name__}] starting from round "
                f"{self.completed_rounds}; requested={requested}."
            )
        while target is None or self.completed_rounds < target:
            if self.stopping_decision is not None and self.stopping_decision.stop:
                break
            evaluations = self.algorithm.evaluations
            reached_budget = max_evaluations is not None and (
                (
                    budget_policy == "strict"
                    and evaluations + self.algorithm.request_size > max_evaluations
                )
                or (
                    budget_policy == "finish_iteration"
                    and evaluations >= max_evaluations
                    and (evaluations == 0 or self.algorithm.source_iteration_complete)
                )
            )
            if reached_budget:
                assert max_evaluations is not None
                self._run_stop_reason = (
                    "maximum_evaluations"
                    if self.algorithm.evaluations >= max_evaluations
                    else "insufficient_evaluations_for_batch"
                )
                self.save_checkpoint(phase="stopped", archive=False)
                break
            self.run_one_step()
        if target is not None and self.completed_rounds >= target and self._run_stop_reason is None:
            self._run_stop_reason = "maximum_rounds"
        if self.checkpoint_policy == "periodic":
            # Flush the non-cadence tail on every successful run-call return.
            self.save_checkpoint(phase="run_end", archive=False)
        result = self.algorithm.result()
        if self.algorithm.verbose:
            self._report_final(result)
        should_plot = self.algorithm.plot if plot is None else plot
        if should_plot and result.evaluations > 0:
            self.algorithm.plot_progress(show=show)
        return result

    @staticmethod
    def _validate_comparison_experiments(
        experiments: Mapping[str, Experiment],
    ) -> dict[str, Experiment]:
        """Validate and copy an independent named experiment collection."""

        if not isinstance(experiments, Mapping) or not experiments:
            raise ExperimentConfigurationError(
                "experiments must be a non-empty mapping of labels to Experiment objects."
            )
        copied = dict(experiments)
        invalid_labels = [
            label for label in copied if not isinstance(label, str) or not label.strip()
        ]
        if invalid_labels:
            raise ExperimentConfigurationError("Every experiment label must be a non-empty string.")
        invalid = [
            label for label, experiment in copied.items() if not isinstance(experiment, Experiment)
        ]
        if invalid:
            raise ExperimentConfigurationError(
                f"Comparison entries must be Experiment objects; invalid labels: {invalid}."
            )
        nested = [label for label, experiment in copied.items() if experiment.is_multi_algorithm]
        if nested:
            raise ExperimentConfigurationError(
                f"Comparison entries must be single-algorithm experiments: {nested}."
            )

        experiment_ids = [id(experiment) for experiment in copied.values()]
        algorithm_ids = [id(experiment.algorithm) for experiment in copied.values()]
        environment_ids = [id(experiment.source_environment) for experiment in copied.values()]
        if len(set(experiment_ids)) != len(experiment_ids):
            raise ExperimentConfigurationError(
                "Each comparison label must refer to a different Experiment object."
            )
        if len(set(algorithm_ids)) != len(algorithm_ids):
            raise ExperimentConfigurationError(
                "Each comparison experiment must own a different Algorithm instance."
            )
        if len(set(environment_ids)) != len(environment_ids):
            raise ExperimentConfigurationError(
                "Each comparison experiment must own a different Environment instance "
                "so histories and random states remain independent."
            )

        output_paths = [
            experiment.output_directory.resolve()
            for experiment in copied.values()
            if experiment.output_directory is not None
        ]
        if len(set(output_paths)) != len(output_paths):
            raise ExperimentConfigurationError(
                "Comparison experiments must not share an output_directory."
            )
        return copied

    @staticmethod
    def _comparison_round_ceilings(
        labels: Sequence[str],
        max_rounds: int | Mapping[str, int | None] | None,
        *,
        parameter: str = "max_rounds",
    ) -> dict[str, int | None]:
        """Resolve one total round ceiling for every comparison label."""

        if isinstance(max_rounds, Mapping):
            missing = [label for label in labels if label not in max_rounds]
            extra = [label for label in max_rounds if label not in labels]
            if missing or extra:
                raise ExperimentConfigurationError(
                    f"A {parameter} mapping must contain exactly the experiment labels; "
                    f"missing={missing}, extra={extra}."
                )
            raw_ceilings = dict(max_rounds)
        else:
            raw_ceilings = dict.fromkeys(labels, max_rounds)

        ceilings: dict[str, int | None] = {}
        for label, ceiling in raw_ceilings.items():
            if ceiling is None:
                ceilings[label] = None
                continue
            if isinstance(ceiling, bool) or not isinstance(ceiling, int) or ceiling < 0:
                raise ExperimentConfigurationError(
                    f"{parameter} for {label!r} must be a non-negative integer or None."
                )
            ceilings[label] = ceiling
        return ceilings

    @classmethod
    def plot_comparison(
        cls,
        experiments: Mapping[str, Experiment],
        *,
        show: bool = True,
        title: str = "Basat algorithm comparison",
    ) -> tuple[Any, tuple[Any, Any]]:
        """Plot independent algorithms together using distinct styles.

        Parameters
        ----------
        experiments:
            Non-empty mapping from legend labels to already-run, independent
            :class:`Experiment` objects.
        show:
            Display the combined figure when true. The figure is always
            returned and may be saved by the caller.
        title:
            Figure title.

        Returns
        -------
        tuple[Any, tuple[Any, Any]]
            Matplotlib figure and the observed/incumbent axes.

        Raises
        ------
        ExperimentConfigurationError
            If experiments are shared, unlabeled, or have no observations.
        MissingOptionalDependencyError
            If Matplotlib is unavailable. Install ``basat[viz]``.
        """

        copied = cls._validate_comparison_experiments(experiments)
        if not isinstance(show, bool):
            raise ExperimentConfigurationError("show must be bool.")
        if not isinstance(title, str) or not title.strip():
            raise ExperimentConfigurationError("title must be a non-empty string.")
        empty = [
            label
            for label, experiment in copied.items()
            if experiment.algorithm.observed_values.numel() == 0
        ]
        if empty:
            raise ExperimentConfigurationError(
                f"Cannot plot experiments without observations: {empty}."
            )
        try:
            import matplotlib.pyplot as plt
        except ImportError as error:  # pragma: no cover - optional dependency boundary
            from .models.utils import MissingOptionalDependencyError

            raise MissingOptionalDependencyError(
                "Plotting requires Matplotlib; install basat[viz]."
            ) from error

        count = len(copied)
        legend_columns = min(4, count)
        legend_rows = math.ceil(count / legend_columns)
        figure, axes_array = plt.subplots(1, 2, figsize=(13.5, 5.0 + 0.25 * (legend_rows - 1)))
        observed_axis, incumbent_axis = axes_array
        color_map = plt.get_cmap("turbo")
        markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "p", "*")
        lines = []
        for index, (label, experiment) in enumerate(copied.items()):
            values = experiment.algorithm.observed_values.detach().cpu()
            evaluations = torch.arange(1, values.numel() + 1)
            incumbent = (
                torch.cummax(values, dim=0).values
                if experiment.algorithm.maximize
                else torch.cummin(values, dim=0).values
            )
            color = color_map(0.5 if count == 1 else index / (count - 1))
            marker = markers[index % len(markers)]
            mark_every = max(1, values.numel() // 20)
            (line,) = observed_axis.plot(
                evaluations,
                values,
                color=color,
                marker=marker,
                markevery=mark_every,
                linewidth=1.8,
                markersize=4.5,
                label=label,
            )
            incumbent_axis.plot(
                evaluations,
                incumbent,
                color=color,
                marker=marker,
                markevery=mark_every,
                linewidth=1.8,
                markersize=4.5,
                label=label,
            )
            lines.append(line)

        observed_axis.set(
            title="Observed reward by evaluation",
            xlabel="Evaluation",
            ylabel="Observed reward",
        )
        incumbent_axis.set(
            title="Incumbent reward",
            xlabel="Evaluation",
            ylabel="Incumbent reward",
        )
        for axis in (observed_axis, incumbent_axis):
            axis.grid(alpha=0.25)
        figure.suptitle(title)
        legend = figure.legend(
            handles=lines,
            labels=list(copied),
            loc="lower center",
            ncol=legend_columns,
        )
        figure.canvas.draw()
        legend_top = legend.get_window_extent().transformed(figure.transFigure.inverted()).y1
        figure.tight_layout(rect=(0.0, legend_top + 0.025, 1.0, 0.94))
        if show and str(plt.get_backend()).lower() != "agg":
            plt.show()
        elif not show:
            # Detach from pyplot so Jupyter's end-of-cell hook cannot auto-display it.
            # The returned figure remains usable for explicit display or saving.
            plt.close(figure)
        return figure, (observed_axis, incumbent_axis)


__all__ = [
    "Experiment",
    "ExperimentCheckpointError",
    "ExperimentComparisonResult",
    "ExperimentConfigurationError",
    "ExperimentFailure",
    "ExperimentProtocolError",
]
