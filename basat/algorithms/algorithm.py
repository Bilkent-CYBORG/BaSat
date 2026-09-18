"""Abstract, synchronous ask-tell contract shared by Basat algorithms.

The base class owns interaction protocol, generic environment-history
validation, finite-candidate boundary checks, counters, diagnostics, and the
portable part of checkpoint state.  Concrete algorithms implement only the
source-specific decisions in protected hooks.  In particular, this module
never projects an environment action into a lower-dimensional model space;
that transformation belongs to the concrete method.

The benchmark is deliberately absent from this layer. Environments own reward
functions and hidden datasets; algorithms receive only learner-visible round
information and feedback. A concrete algorithm must declare the acquisition or
decision policy used by its :meth:`Algorithm.ask` implementation.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import torch

from basat.models.utils import (
    Bounds,
    make_generator,
    resolve_device,
    resolve_dtype,
    sobol_sample,
    uniform_sample,
)

from .helper import (
    AlgorithmDiagnostics,
    AlgorithmResult,
    CompatibilityStatus,
    ExceptionDiagnosticIssue,
    ExceptionDiagnosticReport,
    LastAskRecord,
    SourceRecord,
    _snapshot_value,
    to_quality,
)

if TYPE_CHECKING:
    from basat.environments.environment import EnvironmentData
    from basat.models.Acquisitions import DiscreteAcquisitionCandidates


class AlgorithmError(Exception):
    """Base class for generic algorithm contract failures.

    Algorithm-specific exception modules subclass this error and preserve the
    associated environment, algorithm, and diagnostic report as required by
    the initialization contract.
    """

    def __init__(
        self,
        message: str,
        *,
        environment: Any | None = None,
        algorithm: Algorithm | None = None,
        diagnostics: ExceptionDiagnosticReport | None = None,
    ) -> None:
        """Initialize an error with optional runtime objects and diagnostics."""

        super().__init__(message)
        self.environment = environment
        self.algorithm = algorithm
        self.diagnostics = diagnostics


class AlgorithmConfigurationError(AlgorithmError, ValueError):
    """Raised when an algorithm cannot be constructed from its configuration."""


class AlgorithmDataError(AlgorithmError, ValueError):
    """Raised when learner-visible data violate the common data contract."""


class AlgorithmProtocolError(AlgorithmError, RuntimeError):
    """Raised for an invalid synchronous ask-tell transition."""


class AlgorithmStateError(AlgorithmError, ValueError):
    """Raised when versioned state is malformed or incompatible."""


class AlgorithmCompatibilityError(AlgorithmError):
    """Raised when an algorithm and environment cannot be paired safely."""


class AlgorithmCompatibilityWarning(UserWarning):
    """Warning category for supported but qualified configurations."""


class CrossEnvironmentWarning(AlgorithmCompatibilityWarning):
    """Warn that an executable non-native pairing has no source guarantee."""


class AdapterAppliedWarning(AlgorithmCompatibilityWarning):
    """Warn that an explicit neutral environment adapter was applied."""


class AssumptionViolationWarning(AlgorithmCompatibilityWarning):
    """Warn that a source-paper assumption is not satisfied by the run."""


def resolve_batch_configuration(
    *, batch_size: int | None = 1, q_batch: int | None = None
) -> tuple[int | None, int | None]:
    """Validate and canonicalize independent versus joint batch settings.

    ``batch_size`` means independently scored points and ``q_batch`` means one
    jointly scored q-acquisition.  ``q_batch`` may replace the historical
    default ``batch_size=1`` so that ``Config(q_batch=k)`` is ergonomic.

    Parameters
    ----------
    batch_size:
        Number of independently scored points, or ``None`` when joint mode is
        explicitly selected.
    q_batch:
        Number of jointly scored points, or ``None`` in independent mode.

    Returns
    -------
    tuple[int | None, int | None]
        Canonical ``(batch_size, q_batch)`` with exactly one non-``None`` item.

    Raises
    ------
    AlgorithmConfigurationError
        If values are invalid or conflicting.
    """

    if q_batch is not None:
        if batch_size not in (None, 1):
            raise AlgorithmConfigurationError(
                "batch_size and q_batch are mutually exclusive; choose exactly one."
            )
        if isinstance(q_batch, bool) or not isinstance(q_batch, int):
            raise AlgorithmConfigurationError("q_batch must be an integer, not bool.")
        if q_batch < 1:
            raise AlgorithmConfigurationError("q_batch must be at least one.")
        return None, q_batch
    if batch_size is None:
        raise AlgorithmConfigurationError("Choose exactly one of batch_size or q_batch.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise AlgorithmConfigurationError("batch_size must be an integer, not bool.")
    if batch_size < 1:
        raise AlgorithmConfigurationError("batch_size must be at least one.")
    return batch_size, None


class AlgorithmExceptionController:
    """Reusable, confidentiality-safe initialization diagnostic controller.

    Concrete ``exception_<Algorithm>.py`` modules subclass this controller and
    set :attr:`ERROR_TYPE` and :attr:`WARNING_TYPE` to their named local types.
    The generic checks inspect only declarations and learner-visible history;
    they never access ``environment.dataset`` or another hidden response table.
    """

    ERROR_TYPE: ClassVar[type[AlgorithmError]] = AlgorithmCompatibilityError
    WARNING_TYPE: ClassVar[type[Warning]] = AlgorithmCompatibilityWarning

    def __init__(self, *, environment: Any, algorithm: Algorithm) -> None:
        """Bind the controller to one environment-algorithm pairing."""

        self.environment = environment
        self.algorithm = algorithm
        self.issues: list[ExceptionDiagnosticIssue] = []
        self.report: ExceptionDiagnosticReport | None = None

    @property
    def environment_type(self) -> str:
        """Return the environment's declared family without inspecting data."""

        declared = getattr(self.environment, "environment_type", None)
        if isinstance(declared, str) and declared:
            return declared
        return type(self.environment).__name__

    def add_issue(
        self,
        severity: Literal["info", "warning", "error"],
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        affects_source_guarantees: bool = False,
    ) -> None:
        """Append one normalized diagnostic issue.

        ``affects_source_guarantees`` is stored in structured details so the
        resulting report explains exactly why paper-level guarantees no longer
        apply.

        Parameters
        ----------
        severity:
            ``"info"``, ``"warning"``, or ``"error"``.
        code:
            Stable machine-readable issue identifier.
        message:
            Actionable user-facing explanation.
        details:
            Optional structured diagnostic context.
        affects_source_guarantees:
            Whether this issue disables governing-source guarantees.
        """

        if code == "data_backed_environment" and any(issue.code == code for issue in self.issues):
            # The shared source-mode check owns this universal extension; a
            # concrete controller may independently discover the same fact.
            return

        issue_details = dict(details or {})
        if affects_source_guarantees:
            issue_details["affects_source_guarantees"] = True
        self.issues.append(
            ExceptionDiagnosticIssue(
                severity=severity,
                code=code,
                message=message,
                details=issue_details,
            )
        )

    @staticmethod
    def _declaration(owner: Any, name: str) -> frozenset[str]:
        """Read a public capability declaration as a normalized set."""

        value: Any = getattr(owner, name, frozenset())
        if value is None:
            return frozenset()
        try:
            result = frozenset(value)
        except TypeError:
            return frozenset()
        return frozenset(item for item in result if isinstance(item, str))

    def _diagnose_environment_family(self) -> None:
        """Check the declared native/cross-environment policy."""

        native = self.algorithm.native_environments
        adapter_name = getattr(self.environment, "adapter_name", None)
        if not native or (self.environment_type in native and not adapter_name):
            return
        if self.environment_type in native:
            message = (
                f"{type(self.algorithm).__name__} uses runtime defaults for missing "
                f"information in the {self.environment_type!r} environment."
            )
        else:
            message = (
                f"{type(self.algorithm).__name__} is outside its native environment "
                f"type {sorted(native)} in the {self.environment_type!r} environment."
            )
        if isinstance(adapter_name, str) and adapter_name:
            self.add_issue(
                "warning",
                "environment.adapter_applied",
                f"{message} Applied the explicit {adapter_name!r} adapter.",
                details={
                    "native_environments": sorted(native),
                    "adapter_name": adapter_name,
                },
                affects_source_guarantees=True,
            )
            assumption = getattr(self.environment, "adapter_assumption_message", None)
            if isinstance(assumption, str) and assumption:
                self.add_issue(
                    "warning",
                    "environment.adapter_assumption",
                    assumption,
                    details={"adapter_name": adapter_name},
                    affects_source_guarantees=True,
                )
        else:
            self.add_issue(
                "warning",
                "environment.cross_family_extension",
                message + " No required inputs are missing; no internal defaults were needed. "
                "Additional environment information is ignored.",
                details={"native_environments": sorted(native)},
                affects_source_guarantees=True,
            )

    def _diagnose_source_mode(self) -> None:
        """Mark a supported finite-table adaptation as a Basat extension."""

        source_mode = getattr(self.environment, "source_mode", None)
        if (
            source_mode == "data"
            and getattr(getattr(self.algorithm, "config", None), "finite_candidate_mode", None)
            == "project"
        ):
            self.add_issue(
                "warning",
                "data.projected_embedding",
                f"{type(self.algorithm).__name__} compresses available finite action rows into embedding coordinates "
                "and scores those rows directly. It returns the actual recorded action; rewards are not interpolated. "
                "Context, distribution and perturbation information are not used by this stochastic policy. "
                "This finite-data extension does not inherit the original embedding guarantees. "
                "Set finite_candidate_mode='exact' to require the original embedding geometry.",
                details={"finite_candidate_mode": "project"},
                affects_source_guarantees=True,
            )
        if source_mode == "data" and self.environment_type in self.algorithm.native_environments:
            self.add_issue(
                "warning",
                "data_backed_environment",
                (
                    f"{type(self.algorithm).__name__} finite-table selection is a "
                    "Basat ask-tell extension to the governing source."
                ),
                details={"source_mode": "data"},
                affects_source_guarantees=True,
            )

    def _diagnose_batch_mode(self) -> None:
        """Report the generic repeated-native extension used by sequential papers."""

        if (
            self.algorithm.batch_size is not None
            and self.algorithm.batch_size > 1
            and self.algorithm.native_batching is False
        ):
            self.add_issue(
                "warning",
                "batch.repeated_native_extension",
                (
                    f"{type(self.algorithm).__name__} forms batch_size>1 by repeating its "
                    "native one-point selection without intermediate feedback."
                ),
                details={"batch_size": self.algorithm.batch_size},
                affects_source_guarantees=True,
            )

    def _diagnose_declared_contract(self) -> None:
        """Compare required learner-visible fields with environment declarations."""

        comparisons = (
            (
                "round_info",
                self.algorithm.required_round_info,
                self._declaration(self.environment, "provided_round_info"),
            ),
            (
                "feedback",
                self.algorithm.required_feedback,
                self._declaration(self.environment, "provided_feedback"),
            ),
            (
                "capabilities",
                self.algorithm.required_capabilities,
                self._declaration(self.environment, "capabilities"),
            ),
        )
        for label, required, provided in comparisons:
            missing = sorted(required - provided)
            if missing:
                self.add_issue(
                    "error",
                    f"environment.missing_{label}",
                    f"Environment does not declare required {label}: {missing}.",
                    details={"required": sorted(required), "provided": sorted(provided)},
                )

    def _learner_visible_history(self) -> EnvironmentData | None:
        """Retrieve only the environment's public historical observations."""

        getter = getattr(self.environment, "get_history_data", None)
        if callable(getter):
            return getter()
        history = getattr(self.environment, "history", None)
        return None if history is None else getattr(history, "initial_data", None)

    def _diagnose_history(self) -> EnvironmentData | None:
        """Run the base data validator without initializing algorithm state."""

        try:
            data = self._learner_visible_history()
        except Exception as error:  # pragma: no cover - defensive provider boundary
            self.add_issue(
                "error",
                "history.unavailable",
                f"Learner-visible history could not be retrieved: {error}",
            )
            return None
        if data is None:
            return None
        try:
            self.algorithm._validate_historical_data(data)
        except (AlgorithmDataError, TypeError, ValueError) as error:
            self.add_issue("error", "history.invalid", str(error))
        if (
            getattr(self.environment, "source_mode", None) != "data"
            and getattr(getattr(self.algorithm, "config", None), "finite_candidate_mode", None)
            == "project"
            and data.observations.numel()
        ):
            self.add_issue(
                "warning",
                "history.projected_embedding",
                f"{type(self.algorithm).__name__} compresses historical action rows into reduced coordinates. "
                "These may differ from their original generating coordinates; projection can merge rows. "
                "Set finite_candidate_mode='exact' to require original embedding geometry.",
                affects_source_guarantees=True,
            )
        return data

    def diagnose_initialization(self) -> ExceptionDiagnosticReport:
        """Build, but do not enforce, an initialization compatibility report."""

        self.issues = []
        self._diagnose_environment_family()
        self._diagnose_source_mode()
        self._diagnose_batch_mode()
        self._diagnose_declared_contract()
        data = self._diagnose_history()

        for issue in self.algorithm._configuration_diagnostic_issues(self.environment):
            if not isinstance(issue, ExceptionDiagnosticIssue):
                raise TypeError("_configuration_diagnostic_issues must return diagnostic issues.")
            self.issues.append(issue)
        for issue in self.algorithm._environment_diagnostic_issues(
            self.environment, learner_visible_data=data
        ):
            if not isinstance(issue, ExceptionDiagnosticIssue):
                raise TypeError("_environment_diagnostic_issues must return diagnostic issues.")
            self.issues.append(issue)

        guarantee_issues = [
            issue
            for issue in self.issues
            if bool(issue.details.get("affects_source_guarantees", False))
        ]
        guarantees = self.algorithm.source_guarantees_applicable and not guarantee_issues
        self.report = ExceptionDiagnosticReport(
            algorithm_name=type(self.algorithm).__name__,
            environment_type=self.environment_type,
            issues=tuple(self.issues),
            guarantees_applicable=guarantees,
        )
        return self.report

    def validate_initialization(self) -> ExceptionDiagnosticReport:
        """Enforce diagnostics, emit warnings, and retain run metadata."""

        report = self.diagnose_initialization()
        self.algorithm.initialization_diagnostics = report

        self._apply_source_guarantee_issues(report)

        for issue in report.issues:
            if issue.severity == "warning":
                warnings.warn(issue.message, self.WARNING_TYPE, stacklevel=2)
        if report.has_errors:
            messages = "; ".join(
                issue.message for issue in report.issues if issue.severity == "error"
            )
            raise self.ERROR_TYPE(
                messages,
                environment=self.environment,
                algorithm=self.algorithm,
                diagnostics=report,
            )
        return report

    def _apply_source_guarantee_issues(self, report: ExceptionDiagnosticReport) -> None:
        """Persist compatibility metadata and source-guarantee warning codes.

        Warning emission belongs to :meth:`validate_initialization`.  Keeping
        this helper side-effect free prevents a concrete algorithm controller
        from emitting both a shared compatibility warning and its own warning
        for the same diagnostic issue.
        """

        self.algorithm.compatibility_status = report.compatibility_status
        self.algorithm.adapter_name = report.adapter_name
        warning_codes: list[str] = []
        for issue in report.issues:
            if issue.severity != "warning":
                continue
            if issue.code == "environment.adapter_applied":
                warning_codes.append("ADAPTER_APPLIED")
            elif issue.code == "environment.cross_family_extension":
                warning_codes.append("CROSS_ENVIRONMENT")
            elif bool(issue.details.get("affects_source_guarantees", False)):
                warning_codes.append("ASSUMPTION_VIOLATION")
            else:
                warning_codes.append(issue.code)
        self.algorithm._last_warning_codes = tuple(dict.fromkeys(warning_codes))

        guarantee_reasons = [
            issue.code
            for issue in report.issues
            if bool(issue.details.get("affects_source_guarantees", False))
        ]
        if guarantee_reasons:
            self.algorithm.source_guarantees_applicable = False
            for reason in guarantee_reasons:
                self.algorithm._add_source_guarantee_reason(reason)


class Algorithm(ABC):
    """Base class for all synchronous Basat algorithms.

    Concrete methods implement :meth:`_ask_impl` and :meth:`_tell_impl`.
    Public :meth:`ask` and :meth:`tell` remain final by convention: they enforce
    the shared protocol and normalize internal batches before invoking those
    hooks.

    Parameters
    ----------
    bounds:
        Finite action bounds in the environment coordinate system.
    batch_size:
        Number of independently scored actions returned by each ``ask`` call.
    q_batch:
        Number of actions selected by one joint q-acquisition.  It is mutually
        exclusive with ``batch_size``.
    maximize:
        If ``False``, observations are sign-converted only at the shared quality
        boundary; source-specific equations remain unchanged.
    dtype, device:
        Tensor precision and execution device.
    seed:
        Non-negative seed for the algorithm-owned random generator.
    plot, verbose, verbose_setting, verbose_iter_by_iter:
        User-interface flags.  They never change mathematical state.

    Notes
    -----
    The constructor never accepts a benchmark function. :meth:`ask` proposes
    actions by the concrete algorithm's declared :attr:`ACQUISITION_POLICY`;
    :meth:`tell` receives rewards after an environment evaluates those actions.
    """

    STATE_SCHEMA_VERSION: ClassVar[int] = 1
    PACKAGE_VERSION: ClassVar[str] = "0.1.0"
    EXCEPTION_CLASS: ClassVar[type[AlgorithmExceptionController] | None] = None
    native_environments: ClassVar[frozenset[str]] = frozenset()
    cross_environment_policy: ClassVar[Literal["error", "warning"]] = "warning"
    required_round_info: ClassVar[frozenset[str]] = frozenset()
    required_feedback: ClassVar[frozenset[str]] = frozenset({"observation", "evaluated_action"})
    required_capabilities: ClassVar[frozenset[str]] = frozenset()
    native_batching: ClassVar[bool | None] = None
    ACQUISITION_POLICY: ClassVar[str | None] = None
    RESUME_MUTABLE_CONFIG_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"fit_maxiter", "gp_maxiter", "gp_training_steps"}
    )
    CHECKPOINT_ADDED_CONFIG_DEFAULTS: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {"initial_points_sampling_method": "paper"}
    )
    SOURCE: ClassVar[SourceRecord] = SourceRecord(
        paper_title="Basat abstract algorithm contract",
        paper_year=2026,
        paper_url=None,
        repository_url=None,
        repository_status="local",
        notes="Base class only; concrete algorithms replace this record.",
    )

    def __init__(
        self,
        *,
        bounds: Bounds,
        batch_size: int | None = 1,
        q_batch: int | None = None,
        maximize: bool = True,
        dtype: torch.dtype | str = torch.float32,
        device: torch.device | str = "cpu",
        seed: int = 1,
        initial_points_sampling_method: Literal["paper", "sobol", "uniform"] = "paper",
        plot: bool = False,
        verbose: bool = False,
        verbose_setting: bool = False,
        verbose_iter_by_iter: bool = False,
    ) -> None:
        """Initialize shared protocol, coordinate, RNG, and reporting state."""

        policy = self.ACQUISITION_POLICY
        if type(self) is not Algorithm and (not isinstance(policy, str) or not policy.strip()):
            raise AlgorithmConfigurationError(
                f"{type(self).__name__} must declare a non-empty ACQUISITION_POLICY."
            )

        self.bounds = bounds
        self.action_dimension = self._bounds_dimension(bounds)
        self.batch_size, self.q_batch = resolve_batch_configuration(
            batch_size=batch_size,
            q_batch=q_batch,
        )
        self.maximize = bool(maximize)
        self.dtype = resolve_dtype(dtype)
        self.device = resolve_device(device)
        if isinstance(seed, bool):
            raise AlgorithmConfigurationError("seed must be a non-negative integer, not bool.")
        try:
            self.seed = int(seed)
        except (TypeError, ValueError) as error:
            raise AlgorithmConfigurationError("seed must be a non-negative integer.") from error
        if self.seed < 0:
            raise AlgorithmConfigurationError("seed must be non-negative.")
        self.generator = make_generator(seed=self.seed, device=self.device)
        if initial_points_sampling_method not in {"paper", "sobol", "uniform"}:
            raise AlgorithmConfigurationError(
                "initial_points_sampling_method must be 'paper', 'sobol', or 'uniform'."
            )
        self.initial_points_sampling_method = initial_points_sampling_method
        self.acquisition_sampling_method: Literal["paper", "sobol", "uniform"] = "paper"

        self.plot = bool(plot)
        self.verbose = bool(verbose)
        self.verbose_setting = bool(verbose_setting)
        self.verbose_iter_by_iter = bool(verbose_iter_by_iter)

        self.iteration = 0
        self.evaluations = 0
        self.last_ask: LastAskRecord | None = None
        self.source_guarantees_applicable = True
        self.source_guarantee_reasons: list[str] = []
        self.compatibility_status: CompatibilityStatus = "native"
        self.adapter_name: str | None = None
        self.initialization_diagnostics: ExceptionDiagnosticReport | None = None
        self.model_input_dimension: int | None = self.action_dimension

        self._initialized = False
        self._status = "created"
        self._last_acquisition_value: float | None = None
        self._last_warning_codes: tuple[str, ...] = ()
        self._observed_actions = torch.empty(
            (0, self.action_dimension), dtype=self.dtype, device=self.device
        )
        self._observed_values = torch.empty((0,), dtype=self.dtype, device=self.device)
        self._active_request_size: int | None = None
        if self.batch_size is not None and self.batch_size > 1 and self.native_batching is False:
            self.invalidate_source_guarantees("basat.repeated_native_batch")
        if self.initial_points_sampling_method != "paper":
            self.invalidate_source_guarantees("basat.sampling_method_override")

    @property
    def source_iteration_complete(self) -> bool:
        """Whether an outer iteration has finished receiving its feedback.

        ``Experiment.run(budget_policy="finish_iteration")`` uses this boundary
        to finish a started batch or nested optimizer program. Algorithms with
        multi-request source iterations override it using checkpointed state.
        Reading the property must not sample, fit a model, or change state.
        Ordinary algorithms finish an iteration after each ask/tell exchange.
        """

        return self.last_ask is None

    @property
    def request_size(self) -> int:
        """Number of actions in each synchronous request."""

        if self._active_request_size is not None:
            return self._active_request_size
        selected = self.q_batch if self.q_batch is not None else self.batch_size
        assert selected is not None
        return selected

    @property
    def initial_sampling_method(self) -> Literal["paper", "sobol", "uniform"]:
        """Backward-compatible alias for ``initial_points_sampling_method``."""

        return self.initial_points_sampling_method

    @initial_sampling_method.setter
    def initial_sampling_method(self, value: Literal["paper", "sobol", "uniform"]) -> None:
        self.initial_points_sampling_method = value

    @property
    def uses_joint_batch(self) -> bool:
        """Whether the configured action batch is scored jointly."""

        return self.q_batch is not None

    @property
    def batch_mode_name(self) -> str:
        """Public configuration field controlling the current request."""

        return "q_batch" if self.uses_joint_batch else "batch_size"

    @staticmethod
    def _bounds_dimension(bounds: Bounds) -> int:
        """Read and validate the dimension exposed by ``Bounds``."""

        dimension = getattr(bounds, "dimension", None)
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise AlgorithmConfigurationError(
                "bounds must expose a positive integer 'dimension' property."
            )
        lower = getattr(bounds, "lower", None)
        upper = getattr(bounds, "upper", None)
        if not isinstance(lower, torch.Tensor) or not isinstance(upper, torch.Tensor):
            raise AlgorithmConfigurationError("bounds lower and upper values must be tensors.")
        if lower.ndim != 1 or upper.ndim != 1 or lower.shape != upper.shape:
            raise AlgorithmConfigurationError(
                "bounds must contain aligned one-dimensional vectors."
            )
        if lower.numel() != dimension:
            raise AlgorithmConfigurationError(
                "bounds dimension does not match its coordinate vectors."
            )
        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
            raise AlgorithmConfigurationError("bounds must contain only finite values.")
        if not torch.all(lower < upper):
            raise AlgorithmConfigurationError(
                "every lower bound must be less than its upper bound."
            )
        return dimension

    def _bounds_on_algorithm_device(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action bounds in the algorithm's dtype and device."""

        lower = self.bounds.lower.to(dtype=self.dtype, device=self.device)
        upper = self.bounds.upper.to(dtype=self.dtype, device=self.device)
        return lower, upper

    def _add_source_guarantee_reason(self, reason: str) -> None:
        """Record one stable reason why source guarantees do not apply."""

        if reason and reason not in self.source_guarantee_reasons:
            self.source_guarantee_reasons.append(reason)

    def invalidate_source_guarantees(self, reason: str) -> None:
        """Mark a documented Basat extension or deviation from paper mode.

        Parameters
        ----------
        reason:
            Nonempty stable reason code retained in diagnostics/checkpoints.
        """

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A non-empty source-guarantee reason is required.")
        self.source_guarantees_applicable = False
        self._add_source_guarantee_reason(reason)

    def configure_sampling_methods(
        self,
        *,
        initial: Literal["paper", "sobol", "uniform"] = "paper",
        acquisition: Literal["paper", "sobol", "uniform"] = "paper",
    ) -> None:
        """Select independent initial-design and acquisition-candidate samplers.

        ``paper`` preserves each concrete algorithm's existing sampler. The
        explicit ``sobol`` and ``uniform`` choices apply whenever the method
        constructs a continuous candidate design; finite environment rows are
        always scored without synthesizing replacement coordinates.

        Parameters
        ----------
        initial:
            Initial-design policy: ``paper``, ``sobol``, or ``uniform``.
        acquisition:
            Acquisition-candidate policy: ``paper``, ``sobol``, or ``uniform``.
        """

        allowed = {"paper", "sobol", "uniform"}
        if initial not in allowed or acquisition not in allowed:
            raise AlgorithmConfigurationError(
                "Sampling methods must be 'paper', 'sobol', or 'uniform'."
            )
        if self._initialized or self.last_ask is not None or self.evaluations:
            raise AlgorithmProtocolError(
                "Sampling methods must be configured before algorithm initialization."
            )
        self.initial_points_sampling_method = initial
        self.acquisition_sampling_method = acquisition
        if initial != "paper" or acquisition != "paper":
            self.invalidate_source_guarantees("basat.sampling_method_override")

    def _sample_design(
        self,
        bounds: Bounds,
        count: int,
        *,
        phase: Literal["initial", "acquisition"],
        seed: int,
        paper_method: Literal["sobol", "uniform"] = "sobol",
    ) -> torch.Tensor:
        """Draw a finite, distinct continuous design under the selected policy."""

        configured = (
            self.initial_sampling_method if phase == "initial" else self.acquisition_sampling_method
        )
        method = paper_method if configured == "paper" else configured
        target = bounds.to(dtype=self.dtype, device=self.device)
        if method == "sobol":
            design = sobol_sample(target, count, seed=seed)
        else:
            pieces: list[torch.Tensor] = []
            remaining = count
            attempts = 0
            while remaining and attempts < 8:
                pieces.append(
                    uniform_sample(
                        target,
                        max(remaining * 2, remaining),
                        generator=self.generator,
                    )
                )
                draws = torch.cat(pieces, dim=0)
                # torch.unique sorts rows even with sorted=False. Truncating
                # that ordering biases a uniform design toward small first
                # coordinates. Keep the first occurrence in sampling order.
                _, inverse = torch.unique(draws, dim=0, return_inverse=True)
                positions = torch.arange(draws.shape[0], device=draws.device)
                first = torch.full_like(positions, draws.shape[0])
                first.scatter_reduce_(0, inverse, positions, reduce="amin")
                selected = first[first < draws.shape[0]].sort().values
                combined = draws[selected]
                pieces = [combined]
                remaining = max(0, count - combined.shape[0])
                attempts += 1
            design = pieces[0][:count]
            if design.shape[0] != count:
                raise AlgorithmConfigurationError(
                    "The selected sampler could not produce enough distinct points."
                )
        if not bool(torch.isfinite(design).all()) or design.shape != (
            count,
            target.dimension,
        ):
            raise AlgorithmConfigurationError(
                "The selected sampler returned an invalid candidate design."
            )
        return design

    def _resolved_sampling_method(
        self,
        phase: Literal["initial", "acquisition"],
        *,
        paper_method: Literal["sobol", "uniform"] = "sobol",
    ) -> Literal["paper", "sobol", "uniform"]:
        """Return ``paper`` or the explicit sampler selected for one phase."""

        configured = (
            self.initial_sampling_method if phase == "initial" else self.acquisition_sampling_method
        )
        if configured == "paper":
            return "paper"
        return configured

    def _configuration_diagnostic_issues(
        self, environment: Any
    ) -> Iterable[ExceptionDiagnosticIssue]:
        """Return concrete configuration findings without mutating state."""

        del environment
        if self.initial_sampling_method != "paper" or self.acquisition_sampling_method != "paper":
            return (
                ExceptionDiagnosticIssue(
                    severity="info",
                    code="basat.sampling_method_override",
                    message=(
                        "Basat configured sampling methods: initial_points_sampling_method="
                        f"{self.initial_sampling_method!r}, acquisition="
                        f"{self.acquisition_sampling_method!r}."
                    ),
                    details={
                        "initial_points_sampling_method": self.initial_points_sampling_method,
                        "acquisition_sampling_method": self.acquisition_sampling_method,
                        "affects_source_guarantees": True,
                    },
                ),
            )
        return ()

    def _environment_diagnostic_issues(
        self,
        environment: Any,
        *,
        learner_visible_data: EnvironmentData | None,
    ) -> Iterable[ExceptionDiagnosticIssue]:
        """Return concrete environment/data findings without hidden access."""

        del environment, learner_visible_data
        return ()

    def _validate_historical_data(self, data: EnvironmentData) -> None:
        """Validate generic row counts, dimensions, bounds, and finiteness.

        The method deliberately performs no embedding, active-variable
        selection, inverse mapping, or other paper-specific conversion.
        """

        if data is None:
            raise AlgorithmDataError("EnvironmentData must not be None.")
        if not hasattr(data, "actions") or not hasattr(data, "observations"):
            raise AlgorithmDataError("EnvironmentData requires actions and observations.")

        try:
            actions = torch.as_tensor(data.actions, dtype=self.dtype, device=self.device)
            observations = torch.as_tensor(data.observations, dtype=self.dtype, device=self.device)
        except (TypeError, ValueError, RuntimeError) as error:
            raise AlgorithmDataError(
                "Historical actions and observations must be numeric."
            ) from error

        if actions.ndim != 2:
            raise AlgorithmDataError("Historical actions must have shape (N, D_env).")
        if actions.shape[1] != self.action_dimension:
            raise AlgorithmDataError(
                "Historical action dimension does not match the environment action bounds."
            )
        if observations.ndim != 1:
            raise AlgorithmDataError("Historical observations must have shape (N,).")
        if observations.shape[0] != actions.shape[0]:
            raise AlgorithmDataError("Historical action and observation row counts do not match.")
        if not torch.isfinite(actions).all() or not torch.isfinite(observations).all():
            raise AlgorithmDataError("Historical actions and observations must be finite.")

        lower, upper = self._bounds_on_algorithm_device()
        if actions.numel() and not torch.all((actions >= lower) & (actions <= upper)):
            raise AlgorithmDataError("Historical actions lie outside the configured bounds.")

        row_count = actions.shape[0]
        self._validate_optional_history_tensor(
            data, "evaluated_actions", row_count=row_count, action_shaped=True
        )
        self._validate_optional_history_tensor(data, "contexts", row_count=row_count)
        self._validate_optional_history_tensor(
            data, "observation_noise_variance", row_count=row_count, vector_shaped=True
        )

        metadata = getattr(data, "metadata", None)
        if metadata is not None:
            if isinstance(metadata, Mapping):
                # Environment-level metadata are allowed in addition to the
                # row-aligned list form described by the algorithm contract.
                pass
            elif not isinstance(metadata, list) or len(metadata) != row_count:
                raise AlgorithmDataError(
                    "Historical metadata must be a mapping or a row-aligned list."
                )
            elif not all(isinstance(item, Mapping) for item in metadata):
                raise AlgorithmDataError("Every historical metadata entry must be a mapping.")

    def _validate_optional_history_tensor(
        self,
        data: EnvironmentData,
        name: str,
        *,
        row_count: int,
        action_shaped: bool = False,
        vector_shaped: bool = False,
    ) -> None:
        """Validate one optional learner-visible history field."""

        value = getattr(data, name, None)
        if value is None:
            return
        try:
            tensor = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        except (TypeError, ValueError, RuntimeError) as error:
            raise AlgorithmDataError(f"Historical {name} must be numeric.") from error
        if action_shaped:
            if tensor.ndim != 2 or tensor.shape != (row_count, self.action_dimension):
                raise AlgorithmDataError(f"Historical {name} must have shape (N, D_env).")
        elif vector_shaped:
            if tensor.ndim != 1 or tensor.shape[0] != row_count:
                raise AlgorithmDataError(f"Historical {name} must have shape (N,).")
            if torch.any(tensor < 0):
                raise AlgorithmDataError(f"Historical {name} must be non-negative.")
        elif tensor.ndim < 1 or tensor.shape[0] != row_count:
            raise AlgorithmDataError(f"Historical {name} must have N leading rows.")
        if not torch.isfinite(tensor).all():
            raise AlgorithmDataError(f"Historical {name} must contain only finite values.")

    def bind_environment_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Receive declared public domain metadata before loading visible history.

        Most policies require no extra metadata. Distributional policies use
        this hook to initialize their GP from the environment's finite support.
        No objective, hidden rewards, or true sampling distribution is supplied.
        """
        del metadata

    def initialize_from_environment(self, data: EnvironmentData) -> None:
        """Initialize from observations that are already visible to the learner.

        Concrete transformations run before base state is committed.  A failed
        initialization therefore does not mark the algorithm as initialized.

        Parameters
        ----------
        data:
            Learner-visible history in full ``D_env`` coordinates.

        Raises
        ------
        AlgorithmProtocolError
            If initialization already occurred or feedback is pending.
        AlgorithmDataError
            If historical shapes, values, or metadata are invalid.
        """

        if self._initialized:
            raise AlgorithmProtocolError("The algorithm has already been initialized.")
        if self.last_ask is not None:
            raise AlgorithmProtocolError("Cannot initialize while feedback is pending.")
        self._validate_historical_data(data)
        self._initialize_from_environment_impl(data)

        actions = torch.as_tensor(data.actions, dtype=self.dtype, device=self.device)
        observations = torch.as_tensor(
            data.observations, dtype=self.dtype, device=self.device
        ).reshape(-1)
        self._observed_actions = actions.detach().clone()
        self._observed_values = observations.detach().clone()
        self._initialized = True
        self._status = "initialized"

    def _initialize_from_environment_impl(self, data: EnvironmentData) -> None:
        """Load source-specific coordinates from validated historical data."""

        del data

    @staticmethod
    def _normalize_round_info(round_info: Mapping[str, Any] | None) -> dict[str, Any]:
        """Return an ownership-safe round-information dictionary."""

        if round_info is None:
            return {}
        if not isinstance(round_info, Mapping):
            raise AlgorithmProtocolError("round_info must be a mapping or None.")
        return _snapshot_value(round_info)

    def _warn_environment_default(self, field: str, value: str) -> None:
        """Report each missing-input assumption once and retain it in checkpoints."""
        reason = f"environment.default.{field}"
        if reason not in self.source_guarantee_reasons:
            self.invalidate_source_guarantees(reason)
            warning_type = getattr(self.EXCEPTION_CLASS, "WARNING_TYPE", UserWarning)
            warnings.warn(
                f"{type(self).__name__}: environment omitted {field}; internal default {value} is used. "
                "Environment-provided values take precedence. Native source guarantees do not apply.",
                warning_type,
                stacklevel=3,
            )

    def prepare_round_info(self, round_info: Mapping[str, Any] | None) -> dict[str, Any]:
        """Copy public information and fill algorithm-specific missing inputs."""
        return self._normalize_round_info(round_info)

    def prepare_feedback(self, feedback: Mapping[str, Any], *, action: Any) -> dict[str, Any]:
        """Copy revealed feedback and supply missing physical-coordinate metadata."""
        result = dict(_snapshot_value(feedback))
        if result.get("evaluated_action") is None:
            self._warn_environment_default("evaluated_action", "the selected nominal action")
            result["evaluated_action"] = torch.as_tensor(
                action, dtype=self.dtype, device=self.device
            ).clone()
        return result

    def _validate_round_info(self, round_info: Mapping[str, Any]) -> None:
        """Validate required fields and the common finite-candidate table."""

        missing = sorted(self.required_round_info - set(round_info))
        if missing:
            raise AlgorithmProtocolError(f"round_info is missing required keys: {missing}.")

        has_actions = "candidate_actions" in round_info
        has_indices = "candidate_dataset_indices" in round_info
        if has_actions != has_indices:
            raise AlgorithmProtocolError(
                "candidate_actions and candidate_dataset_indices must be supplied together."
            )
        if not has_actions:
            return
        try:
            actions = torch.as_tensor(
                round_info["candidate_actions"], dtype=self.dtype, device=self.device
            )
            indices = torch.as_tensor(
                round_info["candidate_dataset_indices"],
                dtype=torch.long,
                device=self.device,
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise AlgorithmDataError("Finite candidate data must be numeric.") from error
        if actions.ndim != 2 or actions.shape[1] != self.action_dimension:
            raise AlgorithmDataError("candidate_actions must have shape (M, D_env).")
        if indices.ndim != 1 or indices.shape[0] != actions.shape[0]:
            raise AlgorithmDataError(
                "candidate_dataset_indices must have one entry per candidate action."
            )
        if actions.shape[0] == 0:
            raise AlgorithmDataError("No finite candidate action is available.")
        if self.request_size > actions.shape[0]:
            raise AlgorithmDataError(f"{self.batch_mode_name} exceeds the finite candidate count.")
        if not torch.isfinite(actions).all():
            raise AlgorithmDataError("candidate_actions must contain only finite values.")
        if torch.any(indices < 0):
            raise AlgorithmDataError("candidate_dataset_indices must be non-negative.")
        if torch.unique(indices).numel() != indices.numel():
            raise AlgorithmDataError("candidate_dataset_indices must be unique.")
        lower, upper = self._bounds_on_algorithm_device()
        if not torch.all((actions >= lower) & (actions <= upper)):
            raise AlgorithmDataError("candidate_actions lie outside the configured bounds.")

    def prepare_discrete_candidates(
        self,
        *,
        candidate_actions: torch.Tensor,
        candidate_dataset_indices: torch.Tensor,
        round_info: Mapping[str, Any],
    ) -> DiscreteAcquisitionCandidates:
        """Build the method-specific finite model-coordinate candidate table.

        The base class validates inputs and the returned one-to-one row mapping,
        but deliberately provides no identity projection or pseudoinverse.

        Parameters
        ----------
        candidate_actions:
            Complete available rows with shape ``(M, D_env)``.
        candidate_dataset_indices:
            Stable hidden-table representatives with shape ``(M,)``.
        round_info:
            Current learner-visible pre-action information.

        Returns
        -------
        DiscreteAcquisitionCandidates
            Exact row mapping into the concrete method's ``D_model`` coordinates.
        """

        combined = dict(round_info)
        combined["candidate_actions"] = candidate_actions
        combined["candidate_dataset_indices"] = candidate_dataset_indices
        self._validate_round_info(combined)
        result = self._prepare_discrete_candidates_impl(
            candidate_actions=torch.as_tensor(
                candidate_actions, dtype=self.dtype, device=self.device
            ),
            candidate_dataset_indices=torch.as_tensor(
                candidate_dataset_indices, dtype=torch.long, device=self.device
            ),
            round_info=combined,
        )

        from basat.models.Acquisitions import validate_discrete_candidates

        validate_discrete_candidates(result)
        expected_actions = torch.as_tensor(
            candidate_actions,
            dtype=result.environment_actions.dtype,
            device=result.environment_actions.device,
        )
        expected_indices = torch.as_tensor(
            candidate_dataset_indices,
            dtype=result.dataset_indices.dtype,
            device=result.dataset_indices.device,
        )
        if not torch.equal(result.environment_actions, expected_actions):
            raise AlgorithmDataError(
                "Discrete candidate transformation must preserve environment action rows."
            )
        if not torch.equal(result.dataset_indices, expected_indices):
            raise AlgorithmDataError(
                "Discrete candidate transformation must preserve dataset row indices."
            )
        return result

    def _prepare_discrete_candidates_impl(
        self,
        *,
        candidate_actions: torch.Tensor,
        candidate_dataset_indices: torch.Tensor,
        round_info: Mapping[str, Any],
    ) -> DiscreteAcquisitionCandidates:
        """Transform finite rows into source-defined model coordinates."""

        del candidate_actions, candidate_dataset_indices, round_info
        raise NotImplementedError(
            f"{type(self).__name__} must define an exact finite-data representation."
        )

    def _normalize_actions(self, actions: Any, *, field_name: str = "actions") -> torch.Tensor:
        """Convert a public action or batch to internal shape ``(q, D_env)``."""

        try:
            tensor = torch.as_tensor(actions, dtype=self.dtype, device=self.device)
        except (TypeError, ValueError, RuntimeError) as error:
            raise AlgorithmProtocolError(f"{field_name} must be numeric.") from error
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != self.action_dimension:
            raise AlgorithmProtocolError(f"{field_name} must have shape (D_env,) or (q, D_env).")
        if tensor.shape[0] != self.request_size:
            raise AlgorithmProtocolError(
                f"{field_name} contains {tensor.shape[0]} rows; expected {self.request_size}."
            )
        if not torch.isfinite(tensor).all():
            raise AlgorithmProtocolError(f"{field_name} must contain only finite values.")
        return tensor

    def _validate_proposed_actions(
        self, actions: torch.Tensor, round_info: Mapping[str, Any]
    ) -> None:
        """Enforce action bounds and finite-source membership."""

        lower, upper = self._bounds_on_algorithm_device()
        if not torch.all((actions >= lower) & (actions <= upper)):
            raise AlgorithmProtocolError("The proposed action lies outside the configured bounds.")
        if "candidate_actions" not in round_info:
            return
        candidates = torch.as_tensor(
            round_info["candidate_actions"], dtype=self.dtype, device=self.device
        )
        matches = torch.isclose(
            actions.unsqueeze(1), candidates.unsqueeze(0), rtol=0.0, atol=1e-8
        ).all(dim=-1)
        if not torch.all(matches.any(dim=1)):
            raise AlgorithmProtocolError(
                "A data-backed action must match an available candidate row."
            )

    def _record_last_ask(
        self,
        *,
        actions: torch.Tensor,
        round_info: Mapping[str, Any],
        model_inputs: torch.Tensor | None = None,
        algorithm_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record method coordinates and metadata for the current request.

        Concrete ``_ask_impl`` hooks call this helper when their ``_tell_impl``
        needs embedded coordinates, an active variable set, a robust center, or
        other method-owned decision metadata.  If they do not call it, the
        public wrapper records actions alone.
        """

        if self.last_ask is not None:
            raise AlgorithmProtocolError("An ask result is already awaiting feedback.")
        action_rows = self._normalize_actions(actions)
        normalized_model_inputs: torch.Tensor | None = None
        if model_inputs is not None:
            try:
                normalized_model_inputs = torch.as_tensor(
                    model_inputs, dtype=self.dtype, device=self.device
                )
            except (TypeError, ValueError, RuntimeError) as error:
                raise AlgorithmProtocolError("model_inputs must be numeric.") from error
            if normalized_model_inputs.ndim == 1:
                normalized_model_inputs = normalized_model_inputs.unsqueeze(0)
        self.last_ask = LastAskRecord(
            actions=action_rows,
            model_inputs=normalized_model_inputs,
            round_info_snapshot=dict(round_info),
            algorithm_metadata=dict(algorithm_metadata or {}),
        )

    def _last_ask_from_dict(self, value: Mapping[str, Any]) -> LastAskRecord:
        """Reconstruct one internal microbatch request from its safe snapshot."""

        return LastAskRecord(
            actions=torch.as_tensor(value["actions"], dtype=self.dtype, device=self.device),
            model_inputs=(
                None
                if value["model_inputs"] is None
                else torch.as_tensor(value["model_inputs"], dtype=self.dtype, device=self.device)
            ),
            round_info_snapshot=dict(value["round_info_snapshot"]),
            algorithm_metadata=dict(value["algorithm_metadata"]),
        )

    def _replace_unavailable_finite_action(
        self,
        proposed: torch.Tensor,
        record: LastAskRecord,
        round_info: Mapping[str, Any],
    ) -> tuple[torch.Tensor, LastAskRecord]:
        """Choose a nearby remaining row if native queued state repeats a finite action."""

        candidates = torch.as_tensor(
            round_info["candidate_actions"], dtype=self.dtype, device=self.device
        )
        matches = torch.isclose(
            proposed.unsqueeze(1), candidates.unsqueeze(0), rtol=0.0, atol=1e-8
        ).all(dim=-1)
        if bool(matches.any(dim=1).all()):
            return proposed, record
        if candidates.shape[0] == 0:
            raise AlgorithmProtocolError("No distinct finite candidate remains for this batch.")

        distances = torch.linalg.vector_norm(candidates - proposed[0], dim=1)
        position = int(torch.argmin(distances).item())
        replacement = candidates[position : position + 1]
        metadata = dict(record.algorithm_metadata)
        indices = torch.as_tensor(
            round_info["candidate_dataset_indices"], dtype=torch.long, device=self.device
        )
        if "dataset_indices" in metadata:
            metadata["dataset_indices"] = indices[position : position + 1]

        # Transform the complete public table so row-aligned embedding
        # metadata remains available. A raw slice cannot reconstruct unit-cube,
        # signed-hash, or other algorithm-specific model coordinates.
        projected = self._prepare_discrete_candidates_impl(
            candidate_actions=candidates,
            candidate_dataset_indices=indices,
            round_info=round_info,
        )
        model_inputs = projected.model_inputs[position : position + 1]
        if "acquisition_value" in metadata:
            metadata["acquisition_value"] = None
        metadata["batch_candidate_replaced"] = True
        return replacement, LastAskRecord(
            actions=replacement,
            model_inputs=model_inputs,
            round_info_snapshot=dict(round_info),
            algorithm_metadata=metadata,
        )

    def _microbatch_round_info(
        self, round_info: Mapping[str, Any], selected_actions: list[torch.Tensor]
    ) -> dict[str, Any]:
        """Remove already selected finite rows before another native selection."""

        info = _snapshot_value(round_info)
        if not selected_actions or "candidate_actions" not in info:
            return info
        candidates = torch.as_tensor(
            info["candidate_actions"], dtype=self.dtype, device=self.device
        )
        selected = torch.stack(selected_actions)
        duplicate = (
            torch.isclose(candidates.unsqueeze(1), selected.unsqueeze(0), rtol=0.0, atol=1e-8)
            .all(dim=-1)
            .any(dim=1)
        )
        keep = ~duplicate
        original_rows = candidates.shape[0]
        info["candidate_actions"] = candidates[keep]
        indices = torch.as_tensor(
            info["candidate_dataset_indices"], dtype=torch.long, device=self.device
        )
        info["candidate_dataset_indices"] = indices[keep]
        for key in ("rembo_Y", "hesbo_Y", "embedded_coordinates", "model_inputs"):
            value = info.get(key)
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == original_rows
            ):
                info[key] = value[keep]
        return info

    def _make_continuous_action_distinct(
        self,
        action: torch.Tensor,
        selected_actions: list[torch.Tensor],
        offset: int,
        *,
        record: LastAskRecord,
    ) -> torch.Tensor:
        """Resolve duplicate proposals while preserving their model coordinates."""

        if not selected_actions:
            return action
        previous = torch.stack(selected_actions)
        duplicate = torch.isclose(previous, action, rtol=0.0, atol=1e-8).all(dim=-1).any()
        if not bool(duplicate):
            return action
        for attempt in range(max(32, 2 * self.action_dimension)):
            replacement = self._perturb_continuous_batch_record(record, offset + attempt)
            result = replacement.actions[0]
            if not torch.isclose(previous, result, rtol=0.0, atol=1e-8).all(dim=-1).any():
                record.actions = replacement.actions
                record.model_inputs = replacement.model_inputs
                record.algorithm_metadata = replacement.algorithm_metadata
                return result
        raise AlgorithmProtocolError(
            "Unable to construct a distinct continuous batch action in the algorithm's domain."
        )

    def _perturb_continuous_batch_record(self, record: LastAskRecord, offset: int) -> LastAskRecord:
        """Displace an ambient-space proposal and recompute its model projection.

        Embedding algorithms override this hook to move in their feasible
        latent domain. An ambient displacement alone can leave that image and
        associate feedback with incorrect GP training coordinates.
        """

        lower, upper = self._bounds_on_algorithm_device()
        action = record.actions[0].detach().clone()
        coordinate = offset % self.action_dimension
        span = upper[coordinate] - lower[coordinate]
        step = span * (1e-5 * (1 + offset // self.action_dimension))
        direction = (
            -1.0 if action[coordinate] > (lower[coordinate] + upper[coordinate]) / 2 else 1.0
        )
        action[coordinate] = torch.clamp(
            action[coordinate] + direction * step,
            min=lower[coordinate],
            max=upper[coordinate],
        )
        actions = action.unsqueeze(0)
        projection = self._prepare_discrete_candidates_impl(
            candidate_actions=actions,
            candidate_dataset_indices=torch.zeros(1, dtype=torch.long, device=self.device),
            round_info=record.round_info_snapshot,
        )
        return LastAskRecord(
            actions=actions,
            model_inputs=projection.model_inputs,
            round_info_snapshot=record.round_info_snapshot,
            algorithm_metadata=record.algorithm_metadata,
        )

    def _ask_repeated_native_batch(self, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Form an independent batch by repeating the native one-point rule."""

        request_size = self.request_size
        original_batch_size = self.batch_size
        original_iteration = self.iteration
        selected_actions: list[torch.Tensor] = []
        records: list[LastAskRecord] = []
        try:
            self._active_request_size = 1
            self.batch_size = 1
            for offset in range(request_size):
                self.iteration = original_iteration + offset
                info = self._microbatch_round_info(round_info, selected_actions)
                proposed = self._normalize_actions(self._ask_impl(round_info=info))
                record = self.last_ask
                if record is None:
                    record = LastAskRecord(
                        actions=proposed,
                        model_inputs=proposed,
                        round_info_snapshot=info,
                        algorithm_metadata={},
                    )
                if "candidate_actions" in info:
                    proposed, record = self._replace_unavailable_finite_action(
                        proposed, record, info
                    )
                action = proposed[0]
                if "candidate_actions" not in info:
                    action = self._make_continuous_action_distinct(
                        action, selected_actions, offset, record=record
                    )
                proposed = action.unsqueeze(0)
                self._validate_proposed_actions(proposed, info)
                records.append(
                    LastAskRecord(
                        actions=proposed,
                        model_inputs=record.model_inputs,
                        round_info_snapshot=info,
                        algorithm_metadata=record.algorithm_metadata,
                    )
                )
                selected_actions.append(action.detach().clone())
                self.last_ask = None
        finally:
            self.iteration = original_iteration
            self.batch_size = original_batch_size
            self._active_request_size = None
        actions = torch.stack(selected_actions)
        model_inputs: torch.Tensor | None = None
        if records and all(record.model_inputs is not None for record in records):
            widths = {
                int(record.model_inputs.shape[1])
                for record in records
                if record.model_inputs is not None
            }
            if len(widths) == 1:
                model_inputs = torch.cat(
                    [record.model_inputs for record in records if record.model_inputs is not None],
                    dim=0,
                )
        self.last_ask = LastAskRecord(
            actions=actions,
            model_inputs=model_inputs,
            round_info_snapshot=dict(round_info),
            algorithm_metadata={
                "batch_strategy": "repeated_native_without_intermediate_feedback",
                "microbatch_records": tuple(record.to_dict() for record in records),
            },
        )
        return actions

    def ask(self, *, round_info: Mapping[str, Any] | None = None) -> torch.Tensor:
        """Optimize the concrete decision policy and propose the next action(s).

        This method cannot access or evaluate the benchmark. The owning
        environment evaluates the returned action and passes its learner-visible
        reward back through :meth:`tell`.

        Parameters
        ----------
        round_info:
            Learner-visible information returned by the environment before acting.

        Returns
        -------
        torch.Tensor
            Shape ``(D_env,)`` when the selected request size is one and
            ``(request_size, D_env)`` otherwise.

        Raises
        ------
        AlgorithmProtocolError
            If the algorithm is uninitialized or a previous request is still
            awaiting feedback.
        """

        if not self._initialized:
            raise AlgorithmProtocolError("initialize_from_environment must be called before ask.")
        if self.last_ask is not None:
            raise AlgorithmProtocolError("Cannot call ask while feedback is pending.")
        info = self.prepare_round_info(round_info)
        self._validate_round_info(info)

        try:
            proposed = (
                self._ask_repeated_native_batch(info)
                if self.batch_size is not None
                and self.batch_size > 1
                and self.native_batching is False
                else self._ask_impl(round_info=info)
            )
            action_rows = self._normalize_actions(proposed)
            self._validate_proposed_actions(action_rows, info)
            if self.last_ask is None:
                self._record_last_ask(actions=action_rows, round_info=info)
            elif not torch.equal(self.last_ask.actions, action_rows):
                raise AlgorithmProtocolError(
                    "_ask_impl returned actions different from its LastAskRecord."
                )
        except Exception:
            # A failed ask must not leave a half-created synchronous request.
            self.last_ask = None
            raise

        self._status = "awaiting_feedback"
        result = action_rows.detach().clone()
        return result[0] if self.request_size == 1 else result

    def _feedback_row(
        self, feedback: Mapping[str, Any], index: int, *, batch_rows: int
    ) -> dict[str, Any]:
        """Slice normalized batch feedback for one native sequential update."""

        result: dict[str, Any] = {}
        for key, value in feedback.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_rows:
                if key in {"observation", "evaluated_action", "observation_noise_variance"}:
                    result[key] = value[index : index + 1]
                else:
                    result[key] = value[index]
            elif isinstance(value, list) and len(value) == batch_rows:
                result[key] = value[index]
            else:
                result[key] = value
        return result

    def _tell_repeated_native_batch(
        self,
        *,
        action: torch.Tensor,
        feedback: Mapping[str, Any],
        round_info: Mapping[str, Any],
        records: tuple[Mapping[str, Any], ...],
    ) -> None:
        """Replay a completed independent batch through the native tell rule."""

        pending = self.last_ask
        original_batch_size = self.batch_size
        try:
            self._active_request_size = 1
            self.batch_size = 1
            for index, value in enumerate(records):
                record = self._last_ask_from_dict(value)
                self.last_ask = record
                self._tell_impl(
                    action=action[index : index + 1],
                    feedback=self._feedback_row(feedback, index, batch_rows=len(records)),
                    round_info=record.round_info_snapshot,
                )
        except Exception:
            self.last_ask = pending
            raise
        finally:
            self.batch_size = original_batch_size
            self._active_request_size = None
        self.last_ask = pending

    @abstractmethod
    def _ask_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Implement the concrete source-defined decision rule."""

    @staticmethod
    def _mapping_values_equal(left: Any, right: Any) -> bool:
        """Compare persisted round-information values without coercing tensors."""

        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            try:
                left_tensor = torch.as_tensor(left)
                right_tensor = torch.as_tensor(right)
            except (TypeError, ValueError, RuntimeError):
                return False
            return left_tensor.shape == right_tensor.shape and torch.equal(
                left_tensor.cpu(), right_tensor.cpu()
            )
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            return set(left) == set(right) and all(
                Algorithm._mapping_values_equal(left[key], right[key]) for key in left
            )
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return len(left) == len(right) and all(
                Algorithm._mapping_values_equal(a, b) for a, b in zip(left, right, strict=True)
            )
        try:
            return bool(left == right)
        except (TypeError, ValueError, RuntimeError):
            return False

    def _validate_tell_round_info(self, round_info: Mapping[str, Any]) -> None:
        """Require supplied round information to match the pending snapshot."""

        assert self.last_ask is not None
        snapshot = self.last_ask.round_info_snapshot
        for key in self.required_round_info | {
            "candidate_actions",
            "candidate_dataset_indices",
        }:
            if key not in snapshot and key not in round_info:
                continue
            if key not in snapshot or key not in round_info:
                raise AlgorithmProtocolError(
                    f"tell round_info does not match the pending value for {key!r}."
                )
            if not self._mapping_values_equal(snapshot[key], round_info[key]):
                raise AlgorithmProtocolError(
                    f"tell round_info does not match the pending value for {key!r}."
                )

    def _normalize_feedback(
        self, feedback: Mapping[str, Any]
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Validate required feedback and normalize common batch-shaped fields."""

        if not isinstance(feedback, Mapping):
            raise AlgorithmProtocolError("feedback must be a mapping.")
        missing = sorted(self.required_feedback - set(feedback))
        if missing:
            raise AlgorithmProtocolError(f"feedback is missing required keys: {missing}.")

        try:
            observations = torch.as_tensor(
                feedback["observation"], dtype=self.dtype, device=self.device
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise AlgorithmDataError("feedback observation must be numeric.") from error
        if observations.ndim == 0:
            observations = observations.unsqueeze(0)
        if observations.ndim != 1 or observations.shape[0] != self.request_size:
            raise AlgorithmDataError("feedback observation must have one value per action.")
        if not torch.isfinite(observations).all():
            raise AlgorithmDataError("feedback observation must contain only finite values.")

        evaluated = self._normalize_actions(
            feedback["evaluated_action"], field_name="feedback evaluated_action"
        )
        normalized = _snapshot_value(feedback)
        normalized["observation"] = observations.detach().clone()
        normalized["evaluated_action"] = evaluated.detach().clone()
        return normalized, observations

    def tell(
        self,
        *,
        action: torch.Tensor,
        feedback: Mapping[str, Any],
        round_info: Mapping[str, Any] | None = None,
    ) -> None:
        """Consume feedback for the single outstanding request.

        The concrete update runs before counters and pending state are committed.
        If it raises, the request remains pending so a caller may diagnose or
        retry without fabricating a new evaluation.

        Parameters
        ----------
        action:
            Exact action or q-batch returned by the outstanding :meth:`ask`.
        feedback:
            Learner-visible environment feedback containing all required fields.
        round_info:
            Optional repeated pre-action mapping; required values must match the
            pending snapshot exactly.

        Raises
        ------
        AlgorithmProtocolError
            If no request is pending or action/round information does not match.
        AlgorithmDataError
            If feedback shapes or values are invalid.
        """

        if self.last_ask is None:
            raise AlgorithmProtocolError("tell requires a preceding unconsumed ask.")
        action_rows = self._normalize_actions(action, field_name="action")
        if not torch.equal(action_rows, self.last_ask.actions):
            raise AlgorithmProtocolError(
                "tell action does not exactly match the pending ask result."
            )

        info = (
            _snapshot_value(self.last_ask.round_info_snapshot)
            if round_info is None
            else self.prepare_round_info(round_info)
        )
        self._validate_round_info(info)
        self._validate_tell_round_info(info)
        normalized_feedback, observations = self._normalize_feedback(
            self.prepare_feedback(feedback, action=action)
        )

        microbatch_values = self.last_ask.algorithm_metadata.get("microbatch_records")
        if isinstance(microbatch_values, tuple) and microbatch_values:
            self._tell_repeated_native_batch(
                action=action_rows.detach().clone(),
                feedback=normalized_feedback,
                round_info=info,
                records=microbatch_values,
            )
        else:
            self._tell_impl(
                action=action_rows.detach().clone(),
                feedback=normalized_feedback,
                round_info=info,
            )

        self._observed_actions = torch.cat(
            (self._observed_actions, action_rows.detach().clone()), dim=0
        )
        self._observed_values = torch.cat(
            (self._observed_values, observations.detach().clone()), dim=0
        )
        self.iteration += 1
        self.evaluations += self.request_size
        self.last_ask = None
        self._status = "running"

    @abstractmethod
    def _tell_impl(
        self,
        *,
        action: torch.Tensor,
        feedback: Mapping[str, Any],
        round_info: Mapping[str, Any],
    ) -> None:
        """Update source-specific state from normalized batch feedback."""

    def _best_observed(self) -> tuple[torch.Tensor | None, float | None]:
        """Return the best learner-visible query under the selected direction."""

        if self._observed_values.numel() == 0:
            return None, None
        quality = to_quality(self._observed_values, maximize=self.maximize)
        position = int(torch.argmax(quality).item())
        return (
            self._observed_actions[position].detach().clone(),
            float(self._observed_values[position].item()),
        )

    @property
    def observed_actions(self) -> torch.Tensor:
        """Return a detached copy of all learner-visible evaluated actions."""

        return self._observed_actions.detach().clone()

    @property
    def observed_values(self) -> torch.Tensor:
        """Return a detached copy of all learner-visible observed rewards."""

        return self._observed_values.detach().clone()

    def plot_progress(self, *, show: bool = True) -> tuple[Any, Any]:
        """Plot observed rewards and the incumbent reward.

        Parameters
        ----------
        show:
            Call :func:`matplotlib.pyplot.show` before returning when true.

        Returns
        -------
        tuple[Any, Any]
            The Matplotlib figure and its two axes.

        Raises
        ------
        ValueError
            If no observations have been received yet.
        MissingOptionalDependencyError
            If Matplotlib is unavailable. Install ``basat[viz]``.
        """

        if self._observed_values.numel() == 0:
            raise ValueError("Cannot plot progress before any observation is available.")
        try:
            import matplotlib.pyplot as plt
        except ImportError as error:  # pragma: no cover - optional dependency boundary
            from basat.models.utils import MissingOptionalDependencyError

            raise MissingOptionalDependencyError(
                "Plotting requires Matplotlib; install basat[viz]."
            ) from error
        from basat.models.utils import plot_best_so_far, plot_iteration_values

        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        name = type(self).__name__
        plot_iteration_values(self._observed_values, ax=axes[0], label=name)
        axes[0].set_title("Observed reward by evaluation")
        plot_best_so_far(
            self._observed_values,
            maximize=self.maximize,
            ax=axes[1],
            label=name,
        )
        axes[1].set_title("Incumbent reward")
        for axis in axes:
            axis.set_xlim(1, max(2, int(self._observed_values.numel())))
        figure.suptitle(f"{name} optimization progress")
        figure.tight_layout()
        if show and str(plt.get_backend()).lower() != "agg":
            plt.show()
        elif not show:
            # Detach from pyplot so Jupyter's end-of-cell hook cannot auto-display it.
            # The returned figure remains usable for explicit display or saving.
            plt.close(figure)
        return figure, axes

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Return concrete diagnostic values for the common report."""

        return {}

    def diagnostics(self) -> AlgorithmDiagnostics:
        """Return a detached snapshot of generic and method-specific state."""

        best_action, best_value = self._best_observed()
        specific = dict(self._diagnostics_impl())
        specific.setdefault("initialized", self._initialized)
        specific.setdefault("pending_feedback", self.last_ask is not None)
        specific.setdefault("source_guarantee_reasons", tuple(self.source_guarantee_reasons))
        specific.setdefault("initial_points_sampling_method", self.initial_points_sampling_method)
        specific.setdefault("acquisition_sampling_method", self.acquisition_sampling_method)
        specific.setdefault("acquisition_policy", self.ACQUISITION_POLICY)
        return AlgorithmDiagnostics(
            iterations=self.iteration,
            evaluations=self.evaluations,
            best_value=best_value,
            best_action=best_action,
            model_input_dimension=self.model_input_dimension,
            last_acquisition_value=self._last_acquisition_value,
            last_warning_codes=self._last_warning_codes,
            source_guarantees_applicable=self.source_guarantees_applicable,
            algorithm_specific=specific,
            compatibility_status=self.compatibility_status,
            adapter_name=self.adapter_name,
        )

    def _recommendation_impl(self) -> tuple[torch.Tensor | None, float | None]:
        """Return the method-specific recommendation; default to observed best."""

        return self._best_observed()

    def _result_metadata_impl(self) -> Mapping[str, Any]:
        """Return concrete, non-diagnostic result metadata."""

        return {}

    def result(self) -> AlgorithmResult:
        """Return the current recommendation and complete source metadata."""

        recommended_action, recommended_value = self._recommendation_impl()
        best_action, best_value = self._best_observed()
        return AlgorithmResult(
            recommended_action=recommended_action,
            recommended_value=recommended_value,
            best_observed_action=best_action,
            best_observed_value=best_value,
            evaluations=self.evaluations,
            status=self._status,
            diagnostics=self.diagnostics(),
            source=self.SOURCE,
            metadata=dict(self._result_metadata_impl()),
        )

    def _base_configuration(self) -> dict[str, Any]:
        """Return immutable common configuration used for strict state loading."""

        lower, upper = self._bounds_on_algorithm_device()
        return {
            "bounds_lower": lower.detach().cpu().clone(),
            "bounds_upper": upper.detach().cpu().clone(),
            "batch_size": self.batch_size,
            "q_batch": self.q_batch,
            "maximize": self.maximize,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "seed": self.seed,
            "initial_sampling_method": self.initial_sampling_method,
            "acquisition_sampling_method": self.acquisition_sampling_method,
        }

    @classmethod
    def _fingerprint_value(cls, value: Any) -> Any:
        """Convert immutable configuration values to stable JSON data.

        Tensor contents, nested dataclasses, bounds, devices, dtypes, mappings,
        and provider identities are normalized without relying on process-local
        object addresses.  This lets strict checkpoint loading detect changes
        to an algorithm-specific config, not only to the shared base options.
        """

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            return {
                "kind": "tensor",
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "values": tensor.tolist(),
            }
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            return {
                "kind": "numpy.ndarray",
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "values": array.tolist(),
            }
        if isinstance(value, np.generic):
            return cls._fingerprint_value(value.item())
        if isinstance(value, Bounds):
            return {
                "kind": "bounds",
                "lower": cls._fingerprint_value(value.lower),
                "upper": cls._fingerprint_value(value.upper),
            }
        if isinstance(value, (torch.dtype, torch.device)):
            return str(value)
        if isinstance(value, torch.nn.Module):
            # Custom kernels are immutable configuration inputs even though
            # PyTorch represents their parameters as module state.  Including
            # that state prevents two differently configured instances of the
            # same kernel class from sharing a checkpoint fingerprint.
            public_scalars = {
                name: cls._fingerprint_value(item)
                for name, item in vars(value).items()
                if not name.startswith("_")
                and isinstance(item, (str, int, float, bool, type(None)))
            }
            return {
                "kind": "torch.module",
                "class": f"{type(value).__module__}.{type(value).__qualname__}",
                "state": cls._fingerprint_value(value.state_dict()),
                "public_scalars": public_scalars,
            }
        if is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: cls._fingerprint_value(getattr(value, field.name))
                for field in fields(value)
            }
        if isinstance(value, Mapping):
            return {
                str(key): cls._fingerprint_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [cls._fingerprint_value(item) for item in value]
        if isinstance(value, (set, frozenset)):
            normalized = [cls._fingerprint_value(item) for item in value]
            return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
        if callable(value):
            result: dict[str, Any] = {
                "kind": "callable",
                "module": getattr(value, "__module__", type(value).__module__),
                "qualname": getattr(value, "__qualname__", type(value).__qualname__),
            }
            code = getattr(value, "__code__", None)
            if code is not None:
                result["bytecode_sha256"] = hashlib.sha256(code.co_code).hexdigest()
                result["constants"] = cls._fingerprint_value(code.co_consts)
                result["defaults"] = cls._fingerprint_value(getattr(value, "__defaults__", None))
                result["keyword_defaults"] = cls._fingerprint_value(
                    getattr(value, "__kwdefaults__", None)
                )
                closure = getattr(value, "__closure__", None)
                if closure:
                    cells: list[Any] = []
                    for cell in closure:
                        try:
                            cells.append(cls._fingerprint_value(cell.cell_contents))
                        except ValueError:
                            cells.append({"kind": "empty_closure_cell"})
                    result["closure"] = cells
            elif inspect.ismethod(value):
                result["owner"] = (
                    f"{type(value.__self__).__module__}.{type(value.__self__).__qualname__}"
                )
            return result
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return {
            "kind": "object",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
        }

    def _configuration_fingerprint(self) -> str:
        """Hash immutable common configuration in a platform-stable form."""

        payload = self._configuration_snapshot()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _configuration_snapshot(self) -> dict[str, Any]:
        """Return the normalized configuration used by checkpoint identity."""

        return {
            "base": self._fingerprint_value(self._base_configuration()),
            "algorithm_config": self._without_resume_mutable_fields(
                self._fingerprint_value(getattr(self, "config", None))
            ),
        }

    @classmethod
    def _without_resume_mutable_fields(cls, value: Any) -> Any:
        """Remove GP-fitting budgets that may intentionally change after resume."""

        if isinstance(value, Mapping):
            return {
                key: cls._without_resume_mutable_fields(item)
                for key, item in value.items()
                if key not in cls.RESUME_MUTABLE_CONFIG_FIELDS
            }
        if isinstance(value, list):
            return [cls._without_resume_mutable_fields(item) for item in value]
        return value

    @classmethod
    def _configuration_differences(
        cls,
        stored: Any,
        current: Any,
        *,
        path: str = "configuration",
    ) -> list[str]:
        """Return leaf paths whose normalized checkpoint values differ."""

        if isinstance(stored, Mapping) and isinstance(current, Mapping):
            differences: list[str] = []
            for key in sorted(set(stored) | set(current), key=str):
                child_path = f"{path}.{key}"
                if key not in stored:
                    differences.append(
                        f"{child_path}: missing in checkpoint; current={current[key]!r}"
                    )
                elif key not in current:
                    differences.append(f"{child_path}: checkpoint={stored[key]!r}; now removed")
                else:
                    differences.extend(
                        cls._configuration_differences(stored[key], current[key], path=child_path)
                    )
            return differences
        if isinstance(stored, list) and isinstance(current, list):
            if len(stored) != len(current):
                return [f"{path}: checkpoint length={len(stored)}; current length={len(current)}"]
            differences = []
            for index, (old, new) in enumerate(zip(stored, current, strict=True)):
                differences.extend(
                    cls._configuration_differences(old, new, path=f"{path}[{index}]")
                )
            return differences
        return [] if stored == current else [f"{path}: checkpoint={stored!r}; current={current!r}"]

    def _stored_configuration_snapshot(self, state: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Read a new snapshot or reconstruct one from a legacy simple config repr."""

        snapshot = state.get("configuration_snapshot")
        if isinstance(snapshot, Mapping):
            # Snapshots written before fitting budgets became resume-mutable
            # still contain those keys. Filter both old and new snapshots so a
            # training-budget change never masquerades as a problem mismatch.
            filtered = self._without_resume_mutable_fields(snapshot)
            return self._with_checkpoint_added_defaults(filtered)
        implementation = state.get("algorithm_state")
        if not isinstance(implementation, Mapping):
            return None
        serialized_config = implementation.get("config")
        if isinstance(serialized_config, Mapping):
            parsed_config: Any = serialized_config
        elif isinstance(serialized_config, str):
            try:
                parsed_config = ast.literal_eval(serialized_config)
            except (SyntaxError, ValueError):
                return None
        else:
            return None
        reconstructed = {
            "base": self._fingerprint_value(state.get("base_configuration")),
            "algorithm_config": self._without_resume_mutable_fields(
                self._fingerprint_value(parsed_config)
            ),
        }
        return self._with_checkpoint_added_defaults(reconstructed)

    @classmethod
    def _with_checkpoint_added_defaults(cls, snapshot: Any) -> Mapping[str, Any] | None:
        """Fill newly introduced default config fields in older checkpoints."""

        if not isinstance(snapshot, Mapping):
            return None
        normalized = dict(snapshot)
        algorithm_config = normalized.get("algorithm_config")
        if isinstance(algorithm_config, Mapping):
            expanded = dict(algorithm_config)
            for key, default in cls.CHECKPOINT_ADDED_CONFIG_DEFAULTS.items():
                expanded.setdefault(key, cls._fingerprint_value(default))
            normalized["algorithm_config"] = expanded
        return normalized

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Return all mutable source-specific state affecting future actions."""

        return {}

    def state_dict(self) -> dict[str, Any]:
        """Return versioned observations and search progress for continuation.

        Fitted surrogate parameters are intentionally excluded by concrete
        algorithms. After loading, the surrogate is rebuilt from the restored
        algorithm observations with the current fitting budget.
        """

        snapshot = self._configuration_snapshot()
        encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "package_version": self.PACKAGE_VERSION,
            "class_identity": f"{type(self).__module__}.{type(self).__qualname__}",
            "configuration_fingerprint": hashlib.sha256(encoded).hexdigest(),
            "configuration_snapshot": snapshot,
            "base_configuration": self._base_configuration(),
            "iteration": self.iteration,
            "evaluations": self.evaluations,
            "initialized": self._initialized,
            "status": self._status,
            "model_input_dimension": self.model_input_dimension,
            "source_guarantees_applicable": self.source_guarantees_applicable,
            "source_guarantee_reasons": tuple(self.source_guarantee_reasons),
            "compatibility_status": self.compatibility_status,
            "adapter_name": self.adapter_name,
            "last_warning_codes": tuple(self._last_warning_codes),
            "last_acquisition_value": self._last_acquisition_value,
            "generator_state": self.generator.get_state().detach().clone(),
            "last_ask": None if self.last_ask is None else self.last_ask.to_dict(),
            "observed_actions": self._observed_actions.detach().clone(),
            "observed_values": self._observed_values.detach().clone(),
            "algorithm_state": _snapshot_value(self._state_dict_impl()),
        }

    @staticmethod
    def _require_state_keys(state: Mapping[str, Any], keys: set[str]) -> None:
        """Raise a stable error for missing versioned-state fields."""

        missing = sorted(keys - set(state))
        if missing:
            raise AlgorithmStateError(f"Algorithm state is missing keys: {missing}.")

    def _validate_base_configuration(self, stored: Mapping[str, Any], *, strict: bool) -> None:
        """Validate all source-critical common settings.

        ``strict=False`` may tolerate unknown optional checkpoint fields, but it
        never permits an identity, dimension, or immutable-configuration
        mismatch.
        """

        del strict

        self._require_state_keys(
            stored,
            {
                "bounds_lower",
                "bounds_upper",
                "batch_size",
                "q_batch",
                "maximize",
                "dtype",
                "device",
                "seed",
            },
        )
        current = self._base_configuration()
        for key in ("bounds_lower", "bounds_upper"):
            candidate = torch.as_tensor(stored[key], dtype=current[key].dtype)
            if candidate.shape != current[key].shape or not torch.equal(candidate, current[key]):
                raise AlgorithmStateError(f"Stored {key} is incompatible with this algorithm.")
        for key in ("batch_size", "q_batch", "maximize", "dtype", "device", "seed"):
            if stored[key] != current[key]:
                raise AlgorithmStateError(f"Stored {key} is incompatible with this algorithm.")

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Validate source-specific state before any base state is committed."""

        del state, strict

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Commit previously validated source-specific mutable state."""

        del state, strict

    def load_state_dict(self, state: Mapping[str, Any], *, strict: bool = True) -> None:
        """Restore versioned state after validating identity, shapes, and config.

        Parameters
        ----------
        state:
            Complete mapping produced by :meth:`state_dict`.
        strict:
            Reserved for forward-compatible optional fields; identity,
            dimensions, and immutable configuration always remain strict.

        Raises
        ------
        AlgorithmStateError
            If required fields, identities, shapes, or fingerprints disagree.
        """

        if not isinstance(state, Mapping):
            raise AlgorithmStateError("state must be a mapping.")
        required = {
            "schema_version",
            "package_version",
            "class_identity",
            "configuration_fingerprint",
            "base_configuration",
            "iteration",
            "evaluations",
            "initialized",
            "status",
            "model_input_dimension",
            "source_guarantees_applicable",
            "source_guarantee_reasons",
            "last_warning_codes",
            "last_acquisition_value",
            "generator_state",
            "last_ask",
            "observed_actions",
            "observed_values",
            "algorithm_state",
        }
        self._require_state_keys(state, required)
        if state["schema_version"] != self.STATE_SCHEMA_VERSION:
            raise AlgorithmStateError(
                f"Unsupported algorithm state schema {state['schema_version']!r}; "
                f"expected {self.STATE_SCHEMA_VERSION}."
            )
        if state["package_version"] != self.PACKAGE_VERSION:
            raise AlgorithmStateError(
                "State was produced by an incompatible Basat package version."
            )
        identity = f"{type(self).__module__}.{type(self).__qualname__}"
        if state["class_identity"] != identity:
            raise AlgorithmStateError("State belongs to a different algorithm class.")
        if not isinstance(state["base_configuration"], Mapping):
            raise AlgorithmStateError("base_configuration must be a mapping.")
        self._validate_base_configuration(state["base_configuration"], strict=strict)
        if state["configuration_fingerprint"] != self._configuration_fingerprint():
            stored_snapshot = self._stored_configuration_snapshot(state)
            if stored_snapshot is None:
                raise AlgorithmStateError(
                    "State configuration fingerprint does not match; the legacy checkpoint "
                    "does not contain enough information to identify individual fields."
                )
            current_snapshot = self._with_checkpoint_added_defaults(self._configuration_snapshot())
            differences = self._configuration_differences(stored_snapshot, current_snapshot)
            if differences:
                detail = "; ".join(differences[:8])
                if len(differences) > 8:
                    detail += f"; and {len(differences) - 8} more difference(s)"
                raise AlgorithmStateError(
                    f"State configuration does not match. Differences: {detail}"
                )

        iteration = state["iteration"]
        evaluations = state["evaluations"]
        if (
            isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or iteration < 0
            or isinstance(evaluations, bool)
            or not isinstance(evaluations, int)
            or evaluations < 0
        ):
            raise AlgorithmStateError("Stored iteration/evaluation counters are invalid.")
        model_dimension = state["model_input_dimension"]
        if model_dimension is not None and (
            isinstance(model_dimension, bool)
            or not isinstance(model_dimension, int)
            or model_dimension < 1
        ):
            raise AlgorithmStateError("Stored model_input_dimension is invalid.")
        compatibility_status = state.get("compatibility_status", "native")
        if compatibility_status not in {
            "native",
            "adapted",
            "cross_environment",
            "incompatible",
        }:
            raise AlgorithmStateError("Stored compatibility_status is invalid.")
        adapter_name = state.get("adapter_name")
        if adapter_name is not None and (not isinstance(adapter_name, str) or not adapter_name):
            raise AlgorithmStateError("Stored adapter_name is invalid.")

        observed_actions = torch.as_tensor(
            state["observed_actions"], dtype=self.dtype, device=self.device
        )
        observed_values = torch.as_tensor(
            state["observed_values"], dtype=self.dtype, device=self.device
        )
        if observed_actions.ndim != 2 or observed_actions.shape[1] != self.action_dimension:
            raise AlgorithmStateError("Stored observed_actions has an invalid shape.")
        if observed_values.ndim != 1 or observed_values.shape[0] != observed_actions.shape[0]:
            raise AlgorithmStateError("Stored observed_values has an invalid shape.")
        if not torch.isfinite(observed_actions).all() or not torch.isfinite(observed_values).all():
            raise AlgorithmStateError("Stored observations must be finite.")

        generator_state = torch.as_tensor(state["generator_state"], dtype=torch.uint8).cpu()
        temporary_generator = make_generator(seed=0, device=self.device)
        try:
            temporary_generator.set_state(generator_state)
        except RuntimeError as error:
            raise AlgorithmStateError("Stored random-generator state is invalid.") from error

        pending_value = state["last_ask"]
        pending: LastAskRecord | None = None
        if pending_value is not None:
            if not isinstance(pending_value, Mapping):
                raise AlgorithmStateError("Stored last_ask must be a mapping or None.")
            self._require_state_keys(
                pending_value,
                {"actions", "model_inputs", "round_info_snapshot", "algorithm_metadata"},
            )
            try:
                pending = LastAskRecord(
                    actions=torch.as_tensor(
                        pending_value["actions"], dtype=self.dtype, device=self.device
                    ),
                    model_inputs=(
                        None
                        if pending_value["model_inputs"] is None
                        else torch.as_tensor(
                            pending_value["model_inputs"],
                            dtype=self.dtype,
                            device=self.device,
                        )
                    ),
                    round_info_snapshot=dict(pending_value["round_info_snapshot"]),
                    algorithm_metadata=dict(pending_value["algorithm_metadata"]),
                )
            except (TypeError, ValueError, RuntimeError) as error:
                raise AlgorithmStateError("Stored last_ask is invalid.") from error
            if pending.batch_size != self.request_size:
                raise AlgorithmStateError("Stored last_ask batch size is incompatible.")

        algorithm_state = state["algorithm_state"]
        if not isinstance(algorithm_state, Mapping):
            raise AlgorithmStateError("algorithm_state must be a mapping.")
        self._validate_state_dict_impl(algorithm_state, strict=strict)

        self.iteration = iteration
        self.evaluations = evaluations
        self._initialized = bool(state["initialized"])
        self._status = str(state["status"])
        self.model_input_dimension = model_dimension
        self.source_guarantees_applicable = bool(state["source_guarantees_applicable"])
        self.source_guarantee_reasons = list(state["source_guarantee_reasons"])
        self.compatibility_status = compatibility_status
        self.adapter_name = adapter_name
        self._last_warning_codes = tuple(state["last_warning_codes"])
        acquisition_value = state["last_acquisition_value"]
        self._last_acquisition_value = (
            None if acquisition_value is None else float(acquisition_value)
        )
        self.generator.set_state(generator_state)
        self.last_ask = pending
        self._observed_actions = observed_actions.detach().clone()
        self._observed_values = observed_values.detach().clone()
        self._load_state_dict_impl(algorithm_state, strict=strict)

    def _reset_impl(self) -> None:
        """Clear all source-specific mutable state."""

        return None

    def clone_for_trial(self, seed_offset: int) -> Algorithm:
        """Build a fresh method with all explicit configuration seeds offset.

        Custom algorithms may override this method when constructor arguments
        are not available as instance attributes.
        """
        from basat.trials import fresh_algorithm

        return fresh_algorithm(self, seed_offset)

    def reset(self, seed: int | None = None) -> None:
        """Return the algorithm to a fresh, uninitialized state.

        Parameters
        ----------
        seed:
            Optional replacement seed.  Omitting it reuses the configured seed.
        """

        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise AlgorithmConfigurationError("reset seed must be a non-negative integer.")
            self.seed = seed
        # A reset clears observations and mutable method state, but it does not
        # change the immutable algorithm configuration or the environment mode
        # for which the instance was diagnosed.  Consequently, documented
        # deviations from source behavior remain true after a reset.
        guarantee_reasons = tuple(self.source_guarantee_reasons)
        guarantees_applicable = self.source_guarantees_applicable
        compatibility_status = self.compatibility_status
        adapter_name = self.adapter_name
        self.generator = make_generator(seed=self.seed, device=self.device)
        self.iteration = 0
        self.evaluations = 0
        self.last_ask = None
        self.source_guarantees_applicable = guarantees_applicable
        self.source_guarantee_reasons = list(guarantee_reasons)
        self.compatibility_status = compatibility_status
        self.adapter_name = adapter_name
        self.initialization_diagnostics = None
        self._initialized = False
        self._status = "created"
        self._last_acquisition_value = None
        self._last_warning_codes = ()
        self._observed_actions = torch.empty(
            (0, self.action_dimension), dtype=self.dtype, device=self.device
        )
        self._observed_values = torch.empty((0,), dtype=self.dtype, device=self.device)
        self._reset_impl()


__all__ = [
    "AdapterAppliedWarning",
    "Algorithm",
    "AlgorithmCompatibilityError",
    "AlgorithmCompatibilityWarning",
    "AlgorithmConfigurationError",
    "AlgorithmDataError",
    "AlgorithmError",
    "AlgorithmExceptionController",
    "AlgorithmProtocolError",
    "AlgorithmStateError",
    "AssumptionViolationWarning",
    "CrossEnvironmentWarning",
    "resolve_batch_configuration",
]
