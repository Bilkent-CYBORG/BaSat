"""Initialization diagnostics and runtime errors for Expected over Good."""

from __future__ import annotations

import warnings

from ...helper import ExceptionDiagnosticReport
from .exception_PG import PGError, PGException


class EGError(PGError):
    """Base error for Expected Improvement over Good."""


class EGWarning(UserWarning):
    """Warning category for qualified EG configurations."""


class EGThresholdError(EGError):
    """Raised for an invalid satisfaction threshold or direction."""


class EGPosteriorError(EGError):
    """Raised when the posterior cannot define an EG score."""


class EGDataCompatibilityError(EGError):
    """Raised when learner-visible data cannot be scored by EG."""


class EGCheckpointError(EGError):
    """Raised for incompatible EG checkpoint state."""


ERROR_BY_CODE = {
    "threshold": EGThresholdError,
    "direction": EGThresholdError,
    "posterior": EGPosteriorError,
    "environment_data": EGDataCompatibilityError,
    "previous_data": EGDataCompatibilityError,
    "data_backed_environment": EGDataCompatibilityError,
    "checkpoint": EGCheckpointError,
}


class EGException(PGException):
    """Validate EG using the common good-action contract."""

    ERROR_TYPE = EGError
    WARNING_TYPE = EGWarning

    def validate_initialization(self) -> ExceptionDiagnosticReport:
        """Map shared good-action diagnostics to EG-specific warning/error types."""

        report = self.diagnose_initialization()
        self.algorithm.initialization_diagnostics = report
        self._apply_source_guarantee_issues(report)
        for issue in report.issues:
            if issue.severity == "warning":
                warnings.warn(issue.message, EGWarning, stacklevel=2)
            elif issue.severity == "error":
                raise ERROR_BY_CODE.get(issue.code, EGError)(
                    issue.message,
                    environment=self.environment,
                    algorithm=self.algorithm,
                    diagnostics=report,
                )
        return report


__all__ = [
    "ERROR_BY_CODE",
    "EGCheckpointError",
    "EGDataCompatibilityError",
    "EGError",
    "EGException",
    "EGPosteriorError",
    "EGThresholdError",
    "EGWarning",
]
