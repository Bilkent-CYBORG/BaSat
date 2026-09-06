"""Initialization diagnostics and runtime errors for Probability of Good."""

from __future__ import annotations

import math
import warnings
from typing import ClassVar

from ...algorithm import AlgorithmError, AlgorithmExceptionController
from ...helper import ExceptionDiagnosticReport


class PGError(AlgorithmError):
    """Base error for Probability of Good."""


class PGWarning(UserWarning):
    """Warning category for qualified PG configurations."""


class PGThresholdError(PGError):
    """Raised for an invalid satisfaction threshold or direction."""


class PGPosteriorError(PGError):
    """Raised when the posterior cannot define a PG score."""


class PGDataCompatibilityError(PGError):
    """Raised when learner-visible data cannot be scored by PG."""


class PGCheckpointError(PGError):
    """Raised for incompatible PG checkpoint state."""


ERROR_BY_CODE = {
    "threshold": PGThresholdError,
    "direction": PGThresholdError,
    "posterior": PGPosteriorError,
    "environment_data": PGDataCompatibilityError,
    "previous_data": PGDataCompatibilityError,
    "data_backed_environment": PGDataCompatibilityError,
    "checkpoint": PGCheckpointError,
}


class PGException(AlgorithmExceptionController):
    """Validate PG without inspecting hidden environment responses."""

    ERROR_TYPE: ClassVar[type[AlgorithmError]] = PGError
    WARNING_TYPE: ClassVar[type[Warning]] = PGWarning

    def _check_threshold(self) -> None:
        """Validate the finite maximization aspiration-threshold convention."""

        threshold = getattr(getattr(self.algorithm, "config", None), "threshold", None)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            self.add_issue("error", "threshold", "PG threshold must be a finite scalar.")
        elif not math.isfinite(float(threshold)):
            self.add_issue("error", "threshold", "PG threshold must be finite.")
        if not self.algorithm.maximize:
            self.add_issue(
                "error",
                "direction",
                "PG version 1 implements the paper's maximization threshold convention only.",
            )

    def _check_posterior(self) -> None:
        """The base controller reports repeated-native batch extensions."""

    def _check_environment_data(self) -> None:
        """Reserve method-specific finite-data checks for public coordinates."""

    def _check_previous_data(self) -> None:
        """Generic learner-visible history validation is inherited."""

    def _check_data_backed_environment(self) -> None:
        """PG accepts full-dimensional finite candidates directly."""

    def diagnose_initialization(self) -> ExceptionDiagnosticReport:
        """Collect common and PG-specific issues without raising or leaking data."""

        report = super().diagnose_initialization()
        self.issues = list(report.issues)
        self._check_threshold()
        self._check_posterior()
        self._check_environment_data()
        self._check_previous_data()
        self._check_data_backed_environment()
        self.report = ExceptionDiagnosticReport(
            algorithm_name=type(self.algorithm).__name__,
            environment_type=self.environment_type,
            issues=tuple(self.issues),
            guarantees_applicable=(
                report.guarantees_applicable
                and not any(
                    issue.severity == "error"
                    or bool(issue.details.get("affects_source_guarantees", False))
                    for issue in self.issues
                )
            ),
        )
        return self.report

    def validate_initialization(self) -> ExceptionDiagnosticReport:
        """Emit mapped warnings or raise the first mapped PG initialization error."""

        report = self.diagnose_initialization()
        self.algorithm.initialization_diagnostics = report
        self._apply_source_guarantee_issues(report)
        for issue in report.issues:
            if issue.severity == "warning":
                warnings.warn(issue.message, PGWarning, stacklevel=2)
            elif issue.severity == "error":
                error_type = ERROR_BY_CODE.get(issue.code, PGError)
                raise error_type(
                    issue.message,
                    environment=self.environment,
                    algorithm=self.algorithm,
                    diagnostics=report,
                )
        return report


__all__ = [
    "ERROR_BY_CODE",
    "PGCheckpointError",
    "PGDataCompatibilityError",
    "PGError",
    "PGException",
    "PGPosteriorError",
    "PGThresholdError",
    "PGWarning",
]
