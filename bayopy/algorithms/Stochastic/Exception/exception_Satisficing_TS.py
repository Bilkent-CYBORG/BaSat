"""Diagnostics and runtime errors for Satisficing Thompson Sampling."""

from __future__ import annotations

import warnings

from ...helper import ExceptionDiagnosticReport
from .exception_PG import PGError, PGException


class Satisficing_TSError(PGError):
    """Base error for Satisficing Thompson Sampling."""


class Satisficing_TSWarning(UserWarning):
    """Warning category for qualified STS configurations."""


class Satisficing_TSThresholdError(Satisficing_TSError):
    """Raised for an invalid satisfaction threshold or direction."""


class Satisficing_TSReferenceError(Satisficing_TSError):
    """Raised for an invalid reference action."""


class Satisficing_TSPosteriorSampleError(Satisficing_TSError):
    """Raised when a joint posterior function sample cannot be drawn."""


class Satisficing_TSDataCompatibilityError(Satisficing_TSError):
    """Raised when learner-visible data cannot be scored by STS."""


class Satisficing_TSCheckpointError(Satisficing_TSError):
    """Raised for incompatible STS checkpoint state."""


ERROR_BY_CODE = {
    "threshold": Satisficing_TSThresholdError,
    "direction": Satisficing_TSThresholdError,
    "posterior": Satisficing_TSPosteriorSampleError,
    "reference": Satisficing_TSReferenceError,
    "posterior_sample": Satisficing_TSPosteriorSampleError,
    "environment_data": Satisficing_TSDataCompatibilityError,
    "previous_data": Satisficing_TSDataCompatibilityError,
    "data_backed_environment": Satisficing_TSDataCompatibilityError,
    "checkpoint": Satisficing_TSCheckpointError,
}


class Satisficing_TSException(PGException):
    """Validate threshold, reference action, and joint sampling contract."""

    ERROR_TYPE = Satisficing_TSError
    WARNING_TYPE = Satisficing_TSWarning

    def _check_posterior(self) -> None:
        """Validate sequential sampling and the optional reference action."""

        super()._check_posterior()
        reference = getattr(getattr(self.algorithm, "config", None), "reference_point", None)
        if reference is not None:
            try:
                normalized = self.algorithm._normalize_actions(
                    reference, field_name="reference_point"
                )
            except Exception as error:
                self.add_issue("error", "reference", str(error))
            else:
                lower, upper = self.algorithm._bounds_on_algorithm_device()
                if not ((normalized >= lower) & (normalized <= upper)).all():
                    self.add_issue("error", "reference", "reference_point lies outside bounds.")

    def validate_initialization(self) -> ExceptionDiagnosticReport:
        """Map threshold/reference diagnostics to STS-specific error types."""

        report = self.diagnose_initialization()
        self.algorithm.initialization_diagnostics = report
        self._apply_source_guarantee_issues(report)
        for issue in report.issues:
            if issue.severity == "warning":
                warnings.warn(issue.message, Satisficing_TSWarning, stacklevel=2)
            elif issue.severity == "error":
                raise ERROR_BY_CODE.get(issue.code, Satisficing_TSError)(
                    issue.message,
                    environment=self.environment,
                    algorithm=self.algorithm,
                    diagnostics=report,
                )
        return report


__all__ = [
    "ERROR_BY_CODE",
    "Satisficing_TSCheckpointError",
    "Satisficing_TSDataCompatibilityError",
    "Satisficing_TSError",
    "Satisficing_TSException",
    "Satisficing_TSPosteriorSampleError",
    "Satisficing_TSReferenceError",
    "Satisficing_TSThresholdError",
    "Satisficing_TSWarning",
]
