"""Crash-safe orchestration for BayOPy ask-tell experiments.

The experiment coordinates public information flow. It deliberately contains no
embedding, projection, contextual kernel, or other algorithm-specific transform.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from .algorithms.algorithm import Algorithm
from .algorithms.helper import AlgorithmResult
from .environments.adapters import adapt_environment_for_algorithm
from .environments.environment import Environment
from .stopping.stopping_rule import StoppingRule


class ExperimentConfigurationError(ValueError):
    """Raised when an experiment cannot be configured safely."""


class ExperimentCheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, corrupt, or incompatible."""


class ExperimentProtocolError(RuntimeError):
    """Raised when an environment or algorithm violates the ask-tell contract."""


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
        An :class:`Algorithm` instance implementing BayOPy's ask-tell protocol,
        such as :class:`TuRBO`, :class:`CGP_UCB`, or :class:`DRBO`. It proposes
        actions but never receives the environment's benchmark function.
    stopping_rule:
        Optional :class:`StoppingRule`. Built-in limits are
        :class:`MaximumRoundsStoppingRule`,
        :class:`MaximumEvaluationsStoppingRule`, and
        :class:`MaximumTimeStoppingRule`; combine limits with :func:`any_of` or
        :func:`all_of`. If omitted, :meth:`run` requires ``max_rounds``.
    output_directory:
        Directory for transactional checkpoints and diagnostics. If omitted, the
        run remains in memory.
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
        Number of completed environment rounds between archived checkpoints. The
        latest transactional checkpoint is still updated around every ``tell``.
    constant_context:
        Optional fixed learner context used by constant-context adapters. For
        DRBO/RoBOS it must match one configured support row; otherwise the
        first configured row is used as the point-mass location.
    initial_sampling_method, acquisition_sampling_method:
        Optional shared continuous-design policies. ``None`` preserves the
        algorithm's current choice; explicit values are ``paper``, ``sobol``,
        or ``uniform``.

    Notes
    -----
    A newly revealed environment record is persisted before it is passed to the
    algorithm. If the process fails in that interval, resume replays the record
    without evaluating the source again. Problem-defining settings remain
    checkpoint-compatible; surrogate fitting budgets may change on resume.

    Examples
    --------
    >>> import bayopy
    >>> def objective(x):
    ...     return -((x - 0.25).square().sum())
    >>> bounds = bayopy.Bounds(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    >>> environment = bayopy.StochasticEnvironment(
    ...     bounds=bounds, function=objective, noise_std=0.0,
    ...     dtype="float64", device="cpu", seed=42,
    ... )
    >>> algorithm = bayopy.TuRBO(
    ...     bounds=bounds,
    ...     config=bayopy.TuRBOConfig(
    ...         initial_points=10, acquisition="TS", batch_size=1, seed=42
    ...     ),
    ...     dtype="float64", device="cpu",
    ... )
    >>> experiment = bayopy.Experiment(
    ...     environment=environment, algorithm=algorithm,
    ...     stopping_rule=bayopy.MaximumEvaluationsStoppingRule(150),
    ...     output_directory="./results/turbo", resume=True, save_every=10,
    ... )
    >>> result = experiment.run(max_rounds=150)
    """

    CHECKPOINT_SCHEMA_VERSION = 1
    PACKAGE_VERSION = "0.1.0"

    def __init__(
        self,
        *,
        environment: Environment,
        algorithm: Algorithm,
        stopping_rule: StoppingRule | None = None,
        output_directory: str | os.PathLike[str] | None = None,
        resume: bool = False,
        save_every: int = 1,
        constant_context: Any | None = None,
        initial_sampling_method: Literal["paper", "sobol", "uniform"] | None = None,
        acquisition_sampling_method: Literal["paper", "sobol", "uniform"] | None = None,
    ) -> None:
        """Initialize the coordinator from the components documented above."""

        if not isinstance(environment, Environment):
            raise ExperimentConfigurationError(
                "environment must be an Environment instance, for example "
                "StochasticEnvironment or ContextualEnvironment."
            )
        if not isinstance(algorithm, Algorithm):
            raise ExperimentConfigurationError(
                "algorithm must be an Algorithm instance, for example TuRBO or CGP_UCB."
            )
        if stopping_rule is not None and not isinstance(stopping_rule, StoppingRule):
            raise ExperimentConfigurationError(
                "stopping_rule must be a StoppingRule instance or None."
            )
        if int(save_every) < 1:
            raise ExperimentConfigurationError("save_every must be at least one.")
        if getattr(algorithm, "EXCEPTION_CLASS", None) is None:
            raise ExperimentConfigurationError(
                f"{type(algorithm).__name__} must declare EXCEPTION_CLASS."
            )

        self.source_environment = environment
        self.environment = adapt_environment_for_algorithm(
            environment,
            algorithm,
            constant_context=constant_context,
        )
        self.algorithm = algorithm
        if initial_sampling_method is not None or acquisition_sampling_method is not None:
            self.algorithm.configure_sampling_methods(
                initial=(
                    self.algorithm.initial_sampling_method
                    if initial_sampling_method is None
                    else initial_sampling_method
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
            self._replay_unconsumed_environment_records()
        else:
            self._initialize_algorithm_from_environment()

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
            "package": "bayopy",
            "package_version": self.PACKAGE_VERSION,
            "experiment_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "algorithm_class": (
                f"{type(self.algorithm).__module__}.{type(self.algorithm).__qualname__}"
            ),
            "environment_class": (
                f"{type(self.environment).__module__}.{type(self.environment).__qualname__}"
            ),
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
            payload.get("package") != "bayopy"
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
            f"initial_sampling={algorithm.initial_sampling_method!r}, "
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

        if self.stopping_decision is not None and self.stopping_decision.stop:
            return self.algorithm.result()

        round_info = self._validate_round_info(
            self.environment.before_action(), self.algorithm.required_round_info
        )
        action = self.algorithm.ask(round_info=round_info)
        record_start = len(self.environment.history.records)
        feedback = self._validate_feedback(
            self.environment.step(action), self.algorithm.required_feedback
        )
        new_records = self.environment.history.records[record_start:]
        if not new_records:
            raise ExperimentProtocolError("step() returned feedback without appending history.")

        # This is the crash boundary: durable environment data precedes learner state.
        self.completed_rounds = int(self.environment.round)
        self.save_checkpoint(phase="pre_tell", archive=False)
        self.algorithm.tell(action=action, feedback=feedback, round_info=round_info)
        self.algorithm_consumed_environment_records = len(self.environment.history.records)
        self._check_stopping(new_records, feedback)
        self.save_checkpoint(phase="post_tell", archive=self._should_checkpoint())
        result = self.algorithm.result()
        if self.algorithm.verbose_iter_by_iter:
            self._report_iteration(result, feedback)
        return result

    def run(self, *, max_rounds: int | None = None) -> AlgorithmResult:
        """Run until a stopping rule fires or ``max_rounds`` are completed.

        Parameters
        ----------
        max_rounds:
            Optional non-negative number of additional environment rounds.

        Returns
        -------
        AlgorithmResult
            Final public recommendation and diagnostics.

        Raises
        ------
        ExperimentConfigurationError
            If neither a finite round bound nor a stopping rule is configured.
        """

        if max_rounds is None and self.stopping_rule is None:
            raise ExperimentConfigurationError(
                "run() requires max_rounds or a stopping_rule to prevent an unbounded run."
            )
        if max_rounds is not None and int(max_rounds) < 0:
            raise ExperimentConfigurationError("max_rounds cannot be negative.")
        target = None if max_rounds is None else self.completed_rounds + int(max_rounds)
        if self.algorithm.verbose_setting:
            self._report_settings()
        if self.algorithm.verbose:
            requested = "stopping rule" if max_rounds is None else f"{int(max_rounds)} rounds"
            print(
                f"[{type(self.algorithm).__name__}] starting from round "
                f"{self.completed_rounds}; requested={requested}."
            )
        while target is None or self.completed_rounds < target:
            if self.stopping_decision is not None and self.stopping_decision.stop:
                break
            self.run_one_step()
        result = self.algorithm.result()
        if self.algorithm.verbose:
            self._report_final(result)
        if self.algorithm.plot:
            self.algorithm.plot_progress(show=True)
        return result


__all__ = [
    "Experiment",
    "ExperimentCheckpointError",
    "ExperimentConfigurationError",
    "ExperimentProtocolError",
]
