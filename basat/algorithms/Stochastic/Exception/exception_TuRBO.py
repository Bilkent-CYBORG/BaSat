"""Diagnostics and failures specific to TuRBO-1."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

from ...algorithm import AlgorithmError, AlgorithmExceptionController
from ...helper import ExceptionDiagnosticReport

if TYPE_CHECKING:
    from ..TuRBO import TuRBO


class TuRBOError(AlgorithmError):
    """Base error for TuRBO."""


class TuRBOWarning(UserWarning):
    """Warning for supported TuRBO extensions."""


class TuRBOTrustRegionError(TuRBOError):
    """Raised for invalid trust-region constants or state."""


class TuRBORestartError(TuRBOError):
    """Raised when a required trust-region restart fails."""


class TuRBOCandidateGenerationError(TuRBOError):
    """Raised when the paper candidate cloud cannot be generated."""


class TuRBODataCompatibilityError(TuRBOError):
    """Raised for incompatible learner-visible data."""


class TuRBOFiniteCandidateError(TuRBOError):
    """Raised when no finite candidate can satisfy restart policy."""


class TuRBOCheckpointError(TuRBOError):
    """Raised for incompatible TuRBO state."""


ERROR_BY_CODE = {
    "trust_region": TuRBOTrustRegionError,
    "restart": TuRBORestartError,
    "candidate_generation": TuRBOCandidateGenerationError,
    "environment_data": TuRBODataCompatibilityError,
    "previous_data": TuRBODataCompatibilityError,
    "data_backed_environment": TuRBODataCompatibilityError,
    "finite_candidate": TuRBOFiniteCandidateError,
    "checkpoint": TuRBOCheckpointError,
}


class TuRBOException(AlgorithmExceptionController):
    """Validate TuRBO trust-region constants and finite extension policy."""

    ERROR_TYPE = TuRBOError
    WARNING_TYPE = TuRBOWarning

    def diagnose_initialization(self) -> ExceptionDiagnosticReport:
        """Report trust-region constants and finite-environment compatibility."""

        report = super().diagnose_initialization()
        self.issues = list(report.issues)
        config = cast("TuRBO", self.algorithm).config
        if not config.length_min < config.length_init <= config.length_max:
            self.add_issue(
                "error", "trust_region", "TuRBO requires length_min < length_init <= length_max."
            )
        if getattr(self.environment, "source_mode", None) == "data":
            self.add_issue(
                "warning",
                "data_backed_environment",
                "Finite-candidate TuRBO filtering is a Basat extension.",
                affects_source_guarantees=True,
            )
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
        """Enforce TuRBO diagnostics and apply source-guarantee warnings."""

        report = self.diagnose_initialization()
        self.algorithm.initialization_diagnostics = report
        self._apply_source_guarantee_issues(report)
        for issue in report.issues:
            if issue.severity == "warning":
                warnings.warn(issue.message, TuRBOWarning, stacklevel=2)
                self.algorithm.invalidate_source_guarantees(issue.code)
            elif issue.severity == "error":
                raise ERROR_BY_CODE.get(issue.code, TuRBOError)(
                    issue.message,
                    environment=self.environment,
                    algorithm=self.algorithm,
                    diagnostics=report,
                )
        return report


__all__ = [name for name in globals() if name.startswith("TuRBO") or name == "ERROR_BY_CODE"]
