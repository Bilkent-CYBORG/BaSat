"""Diagnostics and failures specific to Dropout Bayesian optimization."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

from ...algorithm import AlgorithmError, AlgorithmExceptionController
from ...helper import ExceptionDiagnosticReport

if TYPE_CHECKING:
    from ..Dropout import Dropout


class DropoutError(AlgorithmError):
    """Base error for Dropout BO."""


class DropoutWarning(UserWarning):
    """Warning for a qualified Dropout configuration."""


class DropoutDimensionError(DropoutError):
    """Raised when the active dimension is invalid."""


class DropoutFillStrategyError(DropoutError):
    """Raised for an invalid inactive-coordinate fill rule."""


class DropoutDataCompatibilityError(DropoutError):
    """Raised when finite candidates cannot be projected safely."""


class DropoutHistoricalDataError(DropoutError):
    """Raised for malformed Dropout history."""


class DropoutCheckpointError(DropoutError):
    """Raised for incompatible Dropout state."""


ERROR_BY_CODE = {
    "dimension": DropoutDimensionError,
    "fill_strategy": DropoutFillStrategyError,
    "environment_data": DropoutDataCompatibilityError,
    "previous_data": DropoutHistoricalDataError,
    "data_backed_environment": DropoutDataCompatibilityError,
    "checkpoint": DropoutCheckpointError,
}


class DropoutException(AlgorithmExceptionController):
    """Validate the active dimension and inactive fill strategy."""

    ERROR_TYPE = DropoutError
    WARNING_TYPE = DropoutWarning

    def diagnose_initialization(self) -> ExceptionDiagnosticReport:
        """Report active-dimension, fill-policy, and environment compatibility."""

        report = super().diagnose_initialization()
        self.issues = list(report.issues)
        algorithm = cast("Dropout", self.algorithm)
        config = algorithm.config
        if not 1 <= config.active_dimension <= algorithm.action_dimension:
            self.add_issue("error", "dimension", "Dropout requires 1 <= active_dimension <= D.")
        if config.fill_strategy not in {"random", "copy", "mix"}:
            self.add_issue("error", "fill_strategy", "Unknown Dropout fill strategy.")
        if getattr(self.environment, "source_mode", None) == "data":
            self.add_issue(
                "warning",
                "data_backed_environment",
                "Finite-row Dropout selection is a BayOPy extension.",
                affects_source_guarantees=True,
            )
        if algorithm.q_batch is not None and algorithm.q_batch > 1:
            self.add_issue(
                "warning",
                "acquisition",
                "Joint q-batch Dropout uses a BayOPy qUCB extension in the active subspace.",
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
        """Enforce Dropout diagnostics and apply source-guarantee warnings."""

        report = self.diagnose_initialization()
        self.algorithm.initialization_diagnostics = report
        self._apply_source_guarantee_issues(report)
        for issue in report.issues:
            if issue.severity == "warning":
                warnings.warn(issue.message, DropoutWarning, stacklevel=2)
                self.algorithm.invalidate_source_guarantees(issue.code)
            elif issue.severity == "error":
                raise ERROR_BY_CODE.get(issue.code, DropoutError)(
                    issue.message,
                    environment=self.environment,
                    algorithm=self.algorithm,
                    diagnostics=report,
                )
        return report


__all__ = [name for name in globals() if name.startswith("Dropout") or name == "ERROR_BY_CODE"]
