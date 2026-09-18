"""Diagnostics and failures specific to Random EMbedding Bayesian Optimization."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

from ...algorithm import AlgorithmError, AlgorithmExceptionController
from ...helper import ExceptionDiagnosticReport

if TYPE_CHECKING:
    from ..REMBO import REMBO


class REMBOError(AlgorithmError):
    """Base error for REMBO."""


class REMBOWarning(UserWarning):
    """Warning for a supported REMBO extension."""


class REMBOEmbeddingDimensionError(REMBOError):
    """Raised when the embedded dimension is invalid."""


class REMBORegionError(REMBOError):
    """Raised when the low-dimensional search region is invalid."""


class REMBOKernelError(REMBOError):
    """Raised when kernel coordinates violate the REMBO contract."""


class REMBOAcquisitionError(REMBOError):
    """Raised when acquisition construction or optimization fails."""


class REMBOOptimizerDependencyError(REMBOAcquisitionError, ImportError):
    """Raised when paper-mode DIRECT/CMA-ES dependencies are unavailable."""


class REMBODataCompatibilityError(REMBOError):
    """Raised when finite candidates lack exact REMBO coordinates."""


class REMBOHistoricalDataError(REMBOError):
    """Raised when historical actions lack exact REMBO coordinates."""


class REMBOCheckpointError(REMBOError):
    """Raised for incompatible REMBO checkpoint state."""


ERROR_BY_CODE = {
    "embedding_dimension": REMBOEmbeddingDimensionError,
    "embedded_region": REMBORegionError,
    "kernel": REMBOKernelError,
    "acquisition": REMBOAcquisitionError,
    "optimizer_dependency": REMBOOptimizerDependencyError,
    "environment_data": REMBODataCompatibilityError,
    "previous_data": REMBOHistoricalDataError,
    "data_backed_environment": REMBODataCompatibilityError,
    "checkpoint": REMBOCheckpointError,
}


class REMBOException(AlgorithmExceptionController):
    """Validate source-critical REMBO declarations without reading responses."""

    ERROR_TYPE = REMBOError
    WARNING_TYPE = REMBOWarning

    def diagnose_initialization(self) -> ExceptionDiagnosticReport:
        """Report REMBO dimension, kernel, batch, and environment compatibility."""

        report = super().diagnose_initialization()
        self.issues = list(report.issues)
        algorithm = cast("REMBO", self.algorithm)
        config = algorithm.config
        if not 1 <= config.embedding_dimension <= algorithm.action_dimension:
            self.add_issue("error", "embedding_dimension", "REMBO requires 1 <= d <= D.")
        if config.custom_kernel is not None:
            self.add_issue(
                "warning",
                "kernel",
                "A custom REMBO kernel is a Basat extension.",
                affects_source_guarantees=True,
            )
        if self.algorithm.request_size > 1:
            self.add_issue(
                "warning",
                "acquisition",
                "Batch REMBO uses a documented q-acquisition extension.",
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
        """Enforce REMBO diagnostics and apply source-guarantee warnings."""

        report = self.diagnose_initialization()
        self.algorithm.initialization_diagnostics = report
        self._apply_source_guarantee_issues(report)
        for issue in report.issues:
            if issue.severity == "warning":
                warnings.warn(issue.message, REMBOWarning, stacklevel=2)
                self.algorithm.invalidate_source_guarantees(issue.code)
            elif issue.severity == "error":
                raise ERROR_BY_CODE.get(issue.code, REMBOError)(
                    issue.message,
                    environment=self.environment,
                    algorithm=self.algorithm,
                    diagnostics=report,
                )
        return report


__all__ = [name for name in globals() if name.startswith("REMBO") or name == "ERROR_BY_CODE"]
