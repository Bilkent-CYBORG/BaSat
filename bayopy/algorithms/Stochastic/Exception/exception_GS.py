"""Initialization diagnostics and runtime errors for Good-Action Search."""

from __future__ import annotations

import warnings

from ...helper import ExceptionDiagnosticReport
from .exception_PG import PGError, PGException


class GSError(PGError):
    """Base error for Good-Action Search."""


class GSWarning(UserWarning):
    """Warning category for qualified GS configurations."""


class GSThresholdError(GSError):
    """Raised for an invalid satisfaction threshold or direction."""


class GSFantasyError(GSError):
    """Raised when fantasy conditioning fails."""


class GSMaximumSamplingError(GSError):
    """Raised when posterior-maximum sampling fails."""


class GSDataCompatibilityError(GSError):
    """Raised when learner-visible data cannot be scored by GS."""


class GSCheckpointError(GSError):
    """Raised for incompatible GS checkpoint state."""


ERROR_BY_CODE = {
    "threshold": GSThresholdError,
    "direction": GSThresholdError,
    "posterior": GSFantasyError,
    "fantasy": GSFantasyError,
    "maximum_sampling": GSMaximumSamplingError,
    "environment_data": GSDataCompatibilityError,
    "previous_data": GSDataCompatibilityError,
    "data_backed_environment": GSDataCompatibilityError,
    "checkpoint": GSCheckpointError,
}


class GSException(PGException):
    """Validate GS Monte Carlo and fantasy configuration."""

    ERROR_TYPE = GSError
    WARNING_TYPE = GSWarning

    def _check_posterior(self) -> None:
        """Validate sequential operation and positive fantasy sample budgets."""

        super()._check_posterior()
        config = getattr(self.algorithm, "config", None)
        if getattr(config, "lookahead_samples", 0) < 1:
            self.add_issue("error", "fantasy", "lookahead_samples must be positive.")
        if getattr(config, "maximum_value_samples", 0) < 1:
            self.add_issue("error", "maximum_sampling", "maximum_value_samples must be positive.")

    def validate_initialization(self) -> ExceptionDiagnosticReport:
        """Map shared and fantasy diagnostics to GS-specific error types."""

        report = self.diagnose_initialization()
        self.algorithm.initialization_diagnostics = report
        self._apply_source_guarantee_issues(report)
        for issue in report.issues:
            if issue.severity == "warning":
                warnings.warn(issue.message, GSWarning, stacklevel=2)
            elif issue.severity == "error":
                raise ERROR_BY_CODE.get(issue.code, GSError)(
                    issue.message,
                    environment=self.environment,
                    algorithm=self.algorithm,
                    diagnostics=report,
                )
        return report


__all__ = [
    "ERROR_BY_CODE",
    "GSCheckpointError",
    "GSDataCompatibilityError",
    "GSError",
    "GSException",
    "GSFantasyError",
    "GSMaximumSamplingError",
    "GSThresholdError",
    "GSWarning",
]
