"""Typed diagnostics and runtime failures for contextual GP-UCB."""

from __future__ import annotations

import warnings
from typing import Any, cast

from ...algorithm import AlgorithmError, AlgorithmExceptionController
from ...helper import ExceptionDiagnosticReport


class CGP_UCBError(AlgorithmError):
    """Base error for contextual GP-UCB."""


class CGP_UCBWarning(UserWarning):
    """Warning emitted for qualified contextual GP-UCB configurations."""


class CGP_UCBContextError(CGP_UCBError):
    """Raised when context timing, dimension, or support is invalid."""


class CGP_UCBKernelError(CGP_UCBError):
    """Raised for a malformed action-context kernel composition."""


class CGP_UCBDataCompatibilityError(CGP_UCBError):
    """Raised when contextual finite/history data are not representable."""


class CGP_UCBCheckpointError(CGP_UCBError):
    """Raised for malformed contextual GP-UCB checkpoint state."""


class CGP_UCBAcquisitionError(CGP_UCBError):
    """Raised when fixed-context action optimization cannot be certified."""


ERROR_BY_CODE = {
    "context": CGP_UCBContextError,
    "kernel": CGP_UCBKernelError,
    "environment_data": CGP_UCBDataCompatibilityError,
    "previous_data": CGP_UCBDataCompatibilityError,
    "data_backed_environment": CGP_UCBDataCompatibilityError,
    "checkpoint": CGP_UCBCheckpointError,
    "acquisition": CGP_UCBAcquisitionError,
}


class CGP_UCBException(AlgorithmExceptionController):
    """Validate the contextual contract using learner-visible declarations."""

    ERROR_TYPE = CGP_UCBError
    WARNING_TYPE = CGP_UCBWarning

    def _check_environment_contract(self) -> None:
        """The base controller checks required context and feedback fields."""

    def _check_native_environment(self) -> None:
        """Native contextual-family validation is inherited."""

    def _check_context(self) -> None:
        """Require finite-domain cardinality when theorem beta is requested."""

        config = cast(Any, self.algorithm).config
        if config.beta == "finite-theory" and config.finite_domain_cardinality is None:
            self.add_issue(
                "error",
                "context",
                "finite-theory beta requires finite_domain_cardinality.",
            )

    def _check_kernel(self) -> None:
        """Record an error for an unsupported joint-kernel composition."""

        if cast(Any, self.algorithm).config.kernel_combination not in {
            "product",
            "additive",
        }:
            self.add_issue("error", "kernel", "kernel_combination must be product or additive.")

    def _check_environment_data(self) -> None:
        """Historical context alignment is validated before initialization."""

    def _check_previous_data(self) -> None:
        """Generic row checks are inherited; the algorithm checks context shape."""

    def _check_data_backed_environment(self) -> None:
        """Contextual environments expose exact current-context candidates."""

    def diagnose_initialization(self) -> ExceptionDiagnosticReport:
        """Build a non-mutating report from learner-visible declarations only.

        Hidden contextual dataset observations are deliberately absent from
        these checks.  Existing BayOPy-extension guarantee markers are retained
        when the report's ``guarantees_applicable`` flag is computed.
        """

        report = super().diagnose_initialization()
        self.issues = list(report.issues)
        self._check_environment_contract()
        self._check_native_environment()
        self._check_context()
        self._check_kernel()
        self._check_environment_data()
        self._check_previous_data()
        self._check_data_backed_environment()
        self.report = ExceptionDiagnosticReport(
            algorithm_name=type(self.algorithm).__name__,
            environment_type=self.environment_type,
            issues=tuple(self.issues),
            guarantees_applicable=(
                self.algorithm.source_guarantees_applicable
                and not any(
                    bool(issue.details.get("affects_source_guarantees", False))
                    for issue in self.issues
                )
            ),
        )
        return self.report

    def validate_initialization(self) -> ExceptionDiagnosticReport:
        """Persist diagnostics, apply guarantee issues, warn, and raise typed errors."""

        report = self.diagnose_initialization()
        self.algorithm.initialization_diagnostics = report
        self._apply_source_guarantee_issues(report)
        for issue in report.issues:
            if issue.severity == "warning":
                warnings.warn(issue.message, CGP_UCBWarning, stacklevel=2)
            elif issue.severity == "error":
                raise ERROR_BY_CODE.get(issue.code, CGP_UCBError)(
                    issue.message,
                    environment=self.environment,
                    algorithm=self.algorithm,
                    diagnostics=report,
                )
        return report


__all__ = [name for name in globals() if name.startswith("CGP_UCB") or name == "ERROR_BY_CODE"]
