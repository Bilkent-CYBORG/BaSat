"""Shared records and direction conversions for BayOPy algorithms.

This module intentionally contains no method-specific optimization logic.  It
defines the small, stable records exchanged by algorithms, exception
controllers, experiments, and checkpoint/reporting code.  Tensor-valued
records are detached snapshots so callers cannot accidentally mutate an
algorithm through a diagnostic object.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import torch

RepositoryStatus: TypeAlias = Literal[
    "official",
    "author",
    "community",
    "related",
    "none",
    "local",
]
IssueSeverity: TypeAlias = Literal["info", "warning", "error"]
CompatibilityStatus: TypeAlias = Literal["native", "adapted", "cross_environment", "incompatible"]

_REPOSITORY_STATUSES = frozenset({"official", "author", "community", "related", "none", "local"})
_ISSUE_SEVERITIES = frozenset({"info", "warning", "error"})


def _snapshot_value(value: Any) -> Any:
    """Return a detached, recursively copied reporting/checkpoint value.

    The operation deliberately preserves mapping keys and container kinds that
    are part of the public records.  Arbitrary opaque objects are retained by
    reference; algorithm state serializers are responsible for converting such
    objects to stable values before using this helper.
    """

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _snapshot_value(item) for key, item in value.items()}
    if isinstance(value, Mapping):
        return {key: _snapshot_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot_value(item) for item in value)
    if isinstance(value, list):
        return [_snapshot_value(item) for item in value]
    if isinstance(value, set):
        return {_snapshot_value(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(_snapshot_value(item) for item in value)
    return value


def _require_nonempty_text(value: str, *, field_name: str) -> None:
    """Validate a required human-readable string field."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """Traceability information for a concrete algorithm implementation.

    Parameters
    ----------
    paper_title:
        Full title of the governing publication or local manuscript.
    paper_year:
        Publication year.  It must be a positive integer.
    paper_url:
        Stable paper URL when one exists.
    repository_url:
        Repository used for source-level comparison, if any.
    repository_status:
        Relationship of that repository to the governing source.
    notes:
        Concise qualification of translations or BayOPy extensions.
    """

    paper_title: str
    paper_year: int
    paper_url: str | None
    repository_url: str | None
    repository_status: RepositoryStatus
    notes: str = ""

    def __post_init__(self) -> None:
        """Reject incomplete or internally inconsistent source metadata."""

        _require_nonempty_text(self.paper_title, field_name="paper_title")
        if isinstance(self.paper_year, bool) or not isinstance(self.paper_year, int):
            raise TypeError("paper_year must be an integer.")
        if self.paper_year <= 0:
            raise ValueError("paper_year must be positive.")
        if self.paper_url is not None:
            _require_nonempty_text(self.paper_url, field_name="paper_url")
        if self.repository_url is not None:
            _require_nonempty_text(self.repository_url, field_name="repository_url")
        if self.repository_status not in _REPOSITORY_STATUSES:
            allowed = ", ".join(sorted(_REPOSITORY_STATUSES))
            raise ValueError(f"repository_status must be one of: {allowed}.")
        if self.repository_status == "none" and self.repository_url is not None:
            raise ValueError("repository_url must be None when repository_status is 'none'.")
        if not isinstance(self.notes, str):
            raise TypeError("notes must be a string.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the source record."""

        return {
            "paper_title": self.paper_title,
            "paper_year": self.paper_year,
            "paper_url": self.paper_url,
            "repository_url": self.repository_url,
            "repository_status": self.repository_status,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class ExceptionDiagnosticIssue:
    """One structured finding produced by an exception controller.

    Parameters
    ----------
    severity:
        ``"info"``, ``"warning"``, or ``"error"``.
    code:
        Stable machine-readable diagnostic identifier.
    message:
        Human-readable explanation and remediation when applicable.
    details:
        Detached structured evidence; hidden environment values are forbidden.
    """

    severity: IssueSeverity
    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate issue identity and detach mutable diagnostic details."""

        if self.severity not in _ISSUE_SEVERITIES:
            allowed = ", ".join(sorted(_ISSUE_SEVERITIES))
            raise ValueError(f"severity must be one of: {allowed}.")
        _require_nonempty_text(self.code, field_name="code")
        _require_nonempty_text(self.message, field_name="message")
        if not isinstance(self.details, Mapping):
            raise TypeError("details must be a mapping.")
        object.__setattr__(self, "details", _snapshot_value(self.details))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached dictionary suitable for run metadata."""

        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "details": _snapshot_value(self.details),
        }


@dataclass(frozen=True, slots=True)
class ExceptionDiagnosticReport:
    """Immutable initialization report for an algorithm/environment pair.

    Parameters
    ----------
    algorithm_name, environment_type:
        Concrete public identities used to contextualize every finding.
    issues:
        Ordered immutable diagnostics emitted by the controller.
    guarantees_applicable:
        Whether the selected configuration remains within source guarantees.
    """

    algorithm_name: str
    environment_type: str
    issues: tuple[ExceptionDiagnosticIssue, ...]
    guarantees_applicable: bool
    compatibility_status: CompatibilityStatus = "native"
    adapter_name: str | None = None

    def __post_init__(self) -> None:
        """Normalize issue iterables and validate report identifiers."""

        _require_nonempty_text(self.algorithm_name, field_name="algorithm_name")
        _require_nonempty_text(self.environment_type, field_name="environment_type")
        issues = tuple(self.issues)
        if not all(isinstance(issue, ExceptionDiagnosticIssue) for issue in issues):
            raise TypeError("issues must contain only ExceptionDiagnosticIssue objects.")
        if any(issue.severity == "error" for issue in issues):
            issues = tuple(
                ExceptionDiagnosticIssue(
                    severity="info",
                    code=issue.code,
                    message=issue.message,
                    details=issue.details,
                )
                if issue.code
                in {"environment.adapter_applied", "environment.cross_family_extension"}
                else issue
                for issue in issues
            )
        object.__setattr__(self, "issues", issues)
        if not isinstance(self.guarantees_applicable, bool):
            raise TypeError("guarantees_applicable must be a bool.")
        status: CompatibilityStatus = self.compatibility_status
        if any(issue.severity == "error" for issue in issues):
            status = "incompatible"
        elif any(issue.code == "environment.adapter_applied" for issue in issues):
            status = "adapted"
        elif any(issue.code == "environment.cross_family_extension" for issue in issues):
            status = "cross_environment"
        if status not in {
            "native",
            "adapted",
            "cross_environment",
            "incompatible",
        }:
            raise ValueError("compatibility_status is invalid.")
        object.__setattr__(self, "compatibility_status", status)

        adapter_name = self.adapter_name
        if adapter_name is None:
            for issue in issues:
                candidate = issue.details.get("adapter_name")
                if isinstance(candidate, str) and candidate:
                    adapter_name = candidate
                    break
        if adapter_name is not None and (not isinstance(adapter_name, str) or not adapter_name):
            raise ValueError("adapter_name must be a non-empty string or None.")
        object.__setattr__(self, "adapter_name", adapter_name)

    @property
    def has_errors(self) -> bool:
        """Whether at least one diagnostic issue prevents initialization."""

        return any(issue.severity == "error" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Whether at least one diagnostic issue qualifies the run."""

        return any(issue.severity == "warning" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached dictionary suitable for persistence or display."""

        return {
            "algorithm_name": self.algorithm_name,
            "environment_type": self.environment_type,
            "issues": [issue.to_dict() for issue in self.issues],
            "guarantees_applicable": self.guarantees_applicable,
            "compatibility_status": self.compatibility_status,
            "adapter_name": self.adapter_name,
            "has_errors": self.has_errors,
            "has_warnings": self.has_warnings,
        }


@dataclass(slots=True)
class LastAskRecord:
    """Internal snapshot of the single outstanding synchronous request.

    ``actions`` always has shape ``(q, D_env)`` even though the public
    :meth:`Algorithm.ask` result is one-dimensional when ``q == 1``.
    ``model_inputs`` is optional and, when present, has shape
    ``(q, D_model)``.

    Parameters
    ----------
    actions:
        Outstanding physical actions shaped ``(q, D_env)``.
    model_inputs:
        Optional aligned algorithm coordinates shaped ``(q, D_model)``.
    round_info_snapshot:
        Detached learner-visible pre-action information used for matching.
    algorithm_metadata:
        Method-specific proposal state required by ``tell`` and checkpoints.
    """

    actions: torch.Tensor
    model_inputs: torch.Tensor | None
    round_info_snapshot: dict[str, Any]
    algorithm_metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate aligned batch rows and take ownership-safe snapshots."""

        if not isinstance(self.actions, torch.Tensor):
            raise TypeError("actions must be a torch.Tensor.")
        if self.actions.ndim != 2:
            raise ValueError("actions must have shape (q, D_env).")
        if self.actions.shape[0] < 1 or self.actions.shape[1] < 1:
            raise ValueError("actions must contain at least one row and one coordinate.")
        if not torch.isfinite(self.actions).all():
            raise ValueError("actions must contain only finite values.")

        if self.model_inputs is not None:
            if not isinstance(self.model_inputs, torch.Tensor):
                raise TypeError("model_inputs must be a torch.Tensor or None.")
            if self.model_inputs.ndim != 2:
                raise ValueError("model_inputs must have shape (q, D_model).")
            if self.model_inputs.shape[0] != self.actions.shape[0]:
                raise ValueError("actions and model_inputs must have the same row count.")
            if self.model_inputs.shape[1] < 1:
                raise ValueError("model_inputs must contain at least one coordinate.")
            if not torch.isfinite(self.model_inputs).all():
                raise ValueError("model_inputs must contain only finite values.")

        if not isinstance(self.round_info_snapshot, dict):
            raise TypeError("round_info_snapshot must be a dictionary.")
        if not isinstance(self.algorithm_metadata, dict):
            raise TypeError("algorithm_metadata must be a dictionary.")

        self.actions = self.actions.detach().clone()
        if self.model_inputs is not None:
            self.model_inputs = self.model_inputs.detach().clone()
        self.round_info_snapshot = _snapshot_value(self.round_info_snapshot)
        self.algorithm_metadata = _snapshot_value(self.algorithm_metadata)

    @property
    def batch_size(self) -> int:
        """Number of actions awaiting feedback."""

        return int(self.actions.shape[0])

    def to_dict(self) -> dict[str, Any]:
        """Serialize the pending request without exposing mutable state."""

        return {
            "actions": self.actions.detach().clone(),
            "model_inputs": (
                None if self.model_inputs is None else self.model_inputs.detach().clone()
            ),
            "round_info_snapshot": _snapshot_value(self.round_info_snapshot),
            "algorithm_metadata": _snapshot_value(self.algorithm_metadata),
        }


@dataclass(frozen=True, slots=True)
class AlgorithmDiagnostics:
    """Common diagnostic snapshot returned by every algorithm.

    Parameters
    ----------
    iterations, evaluations:
        Completed ask-tell rounds and individual evaluations.
    best_value, best_action:
        Best learner-visible observation and its ``D_env`` action, if known.
    model_input_dimension:
        Explicit ``D_model`` used by the surrogate, or ``None`` before known.
    last_acquisition_value:
        Most recent method score when the algorithm defines one.
    last_warning_codes:
        Stable diagnostic codes qualifying the current run.
    source_guarantees_applicable:
        Whether all governing-source assumptions still hold.
    algorithm_specific:
        Detached method-specific diagnostics and numerical certificates.
    """

    iterations: int
    evaluations: int
    best_value: float | None
    best_action: torch.Tensor | None
    model_input_dimension: int | None
    last_acquisition_value: float | None
    last_warning_codes: tuple[str, ...]
    source_guarantees_applicable: bool
    algorithm_specific: Mapping[str, Any]
    compatibility_status: CompatibilityStatus = "native"
    adapter_name: str | None = None

    def __post_init__(self) -> None:
        """Validate counters and detach tensor/mapping fields."""

        if self.iterations < 0 or self.evaluations < 0:
            raise ValueError("iterations and evaluations must be non-negative.")
        if self.model_input_dimension is not None and self.model_input_dimension < 1:
            raise ValueError("model_input_dimension must be positive or None.")
        if self.best_action is not None:
            if not isinstance(self.best_action, torch.Tensor) or self.best_action.ndim != 1:
                raise ValueError("best_action must be a one-dimensional tensor or None.")
            object.__setattr__(self, "best_action", self.best_action.detach().clone())
        warning_codes = tuple(self.last_warning_codes)
        if not all(isinstance(code, str) and code for code in warning_codes):
            raise ValueError("last_warning_codes must contain non-empty strings.")
        object.__setattr__(self, "last_warning_codes", warning_codes)
        if not isinstance(self.algorithm_specific, Mapping):
            raise TypeError("algorithm_specific must be a mapping.")
        object.__setattr__(self, "algorithm_specific", _snapshot_value(self.algorithm_specific))
        if self.compatibility_status not in {
            "native",
            "adapted",
            "cross_environment",
            "incompatible",
        }:
            raise ValueError("compatibility_status is invalid.")
        if self.adapter_name is not None and (
            not isinstance(self.adapter_name, str) or not self.adapter_name
        ):
            raise ValueError("adapter_name must be a non-empty string or None.")

    def to_dict(self) -> dict[str, Any]:
        """Return an ownership-safe dictionary representation."""

        return {
            "iterations": self.iterations,
            "evaluations": self.evaluations,
            "best_value": self.best_value,
            "best_action": (
                None if self.best_action is None else self.best_action.detach().clone()
            ),
            "model_input_dimension": self.model_input_dimension,
            "last_acquisition_value": self.last_acquisition_value,
            "last_warning_codes": list(self.last_warning_codes),
            "source_guarantees_applicable": self.source_guarantees_applicable,
            "compatibility_status": self.compatibility_status,
            "adapter_name": self.adapter_name,
            "algorithm_specific": _snapshot_value(self.algorithm_specific),
        }


@dataclass(frozen=True, slots=True)
class AlgorithmResult:
    """Final or intermediate algorithm recommendation and audit metadata.

    Parameters
    ----------
    recommended_action, recommended_value:
        Method-defined recommendation in ``D_env`` and its latent/value score.
    best_observed_action, best_observed_value:
        Best learner-visible evaluated pair, kept separate from recommendation.
    evaluations:
        Number of individual feedback rows consumed.
    status:
        Stable human-readable run state.
    diagnostics:
        Common and method-specific diagnostic snapshot.
    source:
        Governing publication and clean-room regression traceability.
    metadata:
        Detached result details, solver residuals, and extension reasons.
    """

    recommended_action: torch.Tensor | None
    recommended_value: float | None
    best_observed_action: torch.Tensor | None
    best_observed_value: float | None
    evaluations: int
    status: str
    diagnostics: AlgorithmDiagnostics
    source: SourceRecord
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate and detach all mutable result fields."""

        if self.evaluations < 0:
            raise ValueError("evaluations must be non-negative.")
        _require_nonempty_text(self.status, field_name="status")
        if not isinstance(self.diagnostics, AlgorithmDiagnostics):
            raise TypeError("diagnostics must be an AlgorithmDiagnostics object.")
        if not isinstance(self.source, SourceRecord):
            raise TypeError("source must be a SourceRecord object.")
        for field_name in ("recommended_action", "best_observed_action"):
            value = getattr(self, field_name)
            if value is not None:
                if not isinstance(value, torch.Tensor) or value.ndim != 1:
                    raise ValueError(f"{field_name} must be a one-dimensional tensor or None.")
                object.__setattr__(self, field_name, value.detach().clone())
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        object.__setattr__(self, "metadata", _snapshot_value(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return an ownership-safe dictionary representation."""

        return {
            "recommended_action": (
                None
                if self.recommended_action is None
                else self.recommended_action.detach().clone()
            ),
            "recommended_value": self.recommended_value,
            "best_observed_action": (
                None
                if self.best_observed_action is None
                else self.best_observed_action.detach().clone()
            ),
            "best_observed_value": self.best_observed_value,
            "evaluations": self.evaluations,
            "status": self.status,
            "diagnostics": self.diagnostics.to_dict(),
            "source": self.source.to_dict(),
            "metadata": _snapshot_value(self.metadata),
        }


def to_quality(observations: torch.Tensor, *, maximize: bool) -> torch.Tensor:
    """Convert observations to BayOPy's higher-is-better convention.

    The conversion is an API-layer sign change only.  It does not rewrite a
    source paper's equations or alter the input tensor in place.

    Parameters
    ----------
    observations:
        Observation tensor of any shape.
    maximize:
        Keep values unchanged when true; negate them for minimization.

    Returns
    -------
    torch.Tensor
        Higher-is-better values with the original shape.
    """

    if not isinstance(observations, torch.Tensor):
        raise TypeError("observations must be a torch.Tensor.")
    return observations if maximize else -observations


def from_quality(quality: torch.Tensor, *, maximize: bool) -> torch.Tensor:
    """Convert higher-is-better quality values back to observations.

    Parameters
    ----------
    quality:
        Internal higher-is-better tensor of any shape.
    maximize:
        Keep values unchanged when true; negate them for minimization.

    Returns
    -------
    torch.Tensor
        Values in the user's observation direction and original shape.
    """

    if not isinstance(quality, torch.Tensor):
        raise TypeError("quality must be a torch.Tensor.")
    return quality if maximize else -quality


__all__ = [
    "AlgorithmDiagnostics",
    "AlgorithmResult",
    "ExceptionDiagnosticIssue",
    "ExceptionDiagnosticReport",
    "IssueSeverity",
    "LastAskRecord",
    "RepositoryStatus",
    "SourceRecord",
    "from_quality",
    "to_quality",
]
