"""Diagnostics and failures specific to Add-GP-UCB."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

from ...algorithm import AlgorithmError, AlgorithmExceptionController
from ...helper import ExceptionDiagnosticReport

if TYPE_CHECKING:
    from ..ADD_GP_UCB import ADD_GP_UCB


class ADD_GP_UCBError(AlgorithmError):
    """Base error for Add-GP-UCB."""


class ADD_GP_UCBWarning(UserWarning):
    """Warning for qualified Add-GP-UCB settings."""


class ADD_GP_UCBGroupError(ADD_GP_UCBError):
    """Raised for an invalid additive partition."""


class ADD_GP_UCBDecompositionError(ADD_GP_UCBError):
    """Raised when learned decomposition search fails."""


class ADD_GP_UCBKernelError(ADD_GP_UCBError):
    """Raised when shared additive covariance cannot be built."""


class ADD_GP_UCBAcquisitionError(ADD_GP_UCBError):
    """Raised when component UCB evaluation fails."""


class ADD_GP_UCBDataCompatibilityError(ADD_GP_UCBError):
    """Raised for incompatible learner-visible data."""


class ADD_GP_UCBCheckpointError(ADD_GP_UCBError):
    """Raised for incompatible Add-GP-UCB state."""


ERROR_BY_CODE = {
    "groups": ADD_GP_UCBGroupError,
    "decomposition": ADD_GP_UCBDecompositionError,
    "kernel": ADD_GP_UCBKernelError,
    "acquisition": ADD_GP_UCBAcquisitionError,
    "environment_data": ADD_GP_UCBDataCompatibilityError,
    "previous_data": ADD_GP_UCBDataCompatibilityError,
    "data_backed_environment": ADD_GP_UCBDataCompatibilityError,
    "checkpoint": ADD_GP_UCBCheckpointError,
}


class ADD_GP_UCBException(AlgorithmExceptionController):
    """Validate the additive partition and sequential source contract."""

    ERROR_TYPE = ADD_GP_UCBError
    WARNING_TYPE = ADD_GP_UCBWarning

    def diagnose_initialization(self) -> ExceptionDiagnosticReport:
        """Report additive partition, decomposition, and environment compatibility."""

        report = super().diagnose_initialization()
        self.issues = list(report.issues)
        algorithm = cast("ADD_GP_UCB", self.algorithm)
        try:
            algorithm._validate_groups(algorithm.state.groups)
        except Exception as error:
            self.add_issue("error", "groups", str(error))
        if algorithm.config.learn_decomposition:
            self.add_issue(
                "info",
                "decomposition",
                "The additive decomposition will be refreshed from learner-visible observations.",
            )
        self.report = ExceptionDiagnosticReport(
            algorithm_name=type(self.algorithm).__name__,
            environment_type=self.environment_type,
            issues=tuple(self.issues),
            guarantees_applicable=(
                report.guarantees_applicable
                and not any(
                    bool(issue.details.get("affects_source_guarantees", False))
                    for issue in self.issues
                )
            ),
        )
        return self.report

    def validate_initialization(self) -> ExceptionDiagnosticReport:
        """Enforce Add-GP-UCB diagnostics and apply source-guarantee warnings."""

        report = self.diagnose_initialization()
        self.algorithm.initialization_diagnostics = report
        self._apply_source_guarantee_issues(report)
        for issue in report.issues:
            if issue.severity == "warning":
                warnings.warn(issue.message, ADD_GP_UCBWarning, stacklevel=2)
            elif issue.severity == "error":
                raise ERROR_BY_CODE.get(issue.code, ADD_GP_UCBError)(
                    issue.message,
                    environment=self.environment,
                    algorithm=self.algorithm,
                    diagnostics=report,
                )
        return report


__all__ = [name for name in globals() if name.startswith("ADD_GP_UCB") or name == "ERROR_BY_CODE"]
