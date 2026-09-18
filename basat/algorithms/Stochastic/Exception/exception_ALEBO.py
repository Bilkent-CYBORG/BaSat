"""Diagnostics and failures specific to ALEBO."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import torch

from ...algorithm import AlgorithmError, AlgorithmExceptionController
from ...helper import ExceptionDiagnosticReport

if TYPE_CHECKING:
    from ..ALEBO import ALEBO


class ALEBOError(AlgorithmError):
    """Base error for ALEBO."""


class ALEBOWarning(UserWarning):
    """Warning for qualified ALEBO approximations."""


class ALEBOEmbeddingDimensionError(ALEBOError):
    """Raised for an invalid ALEBO embedding dimension."""


class ALEBOPolytopeError(ALEBOError):
    """Raised for an infeasible or numerically invalid ALEBO polytope."""


class ALEBOMetricError(ALEBOError):
    """Raised when the Mahalanobis metric is not positive definite."""


class ALEBOAcquisitionOptimizationError(ALEBOError):
    """Raised when constrained acquisition optimization fails."""


class ALEBODataCompatibilityError(ALEBOError):
    """Raised when finite rows lie outside the exact embedded image."""


class ALEBOHistoricalDataError(ALEBOError):
    """Raised when history lies outside the exact embedded image."""


class ALEBOCheckpointError(ALEBOError):
    """Raised for incompatible ALEBO checkpoint state."""


ERROR_BY_CODE = {
    "embedding_dimension": ALEBOEmbeddingDimensionError,
    "polytope": ALEBOPolytopeError,
    "metric": ALEBOMetricError,
    "acquisition": ALEBOAcquisitionOptimizationError,
    "environment_data": ALEBODataCompatibilityError,
    "previous_data": ALEBOHistoricalDataError,
    "data_backed_environment": ALEBODataCompatibilityError,
    "checkpoint": ALEBOCheckpointError,
}


class ALEBOException(AlgorithmExceptionController):
    """Validate ALEBO embedding, polytope, and metric state."""

    ERROR_TYPE = ALEBOError
    WARNING_TYPE = ALEBOWarning

    def diagnose_initialization(self) -> ExceptionDiagnosticReport:
        """Report ALEBO dimension, metric, polytope, and environment compatibility."""

        report = super().diagnose_initialization()
        self.issues = list(report.issues)
        algorithm = cast("ALEBO", self.algorithm)
        config = algorithm.config
        if not 1 <= config.embedding_dimension <= algorithm.action_dimension:
            self.add_issue("error", "embedding_dimension", "ALEBO requires 1 <= d_e <= D.")
        if config.metric_inference == "map":
            self.add_issue(
                "warning",
                "metric",
                "MAP-only ALEBO metric inference is an explicit fast approximation.",
                affects_source_guarantees=True,
            )
        if torch.linalg.matrix_rank(algorithm.state.B) != config.embedding_dimension:
            self.add_issue("error", "polytope", "ALEBO projection matrix is rank deficient.")
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
        """Enforce ALEBO diagnostics and apply source-guarantee warnings."""

        report = self.diagnose_initialization()
        self.algorithm.initialization_diagnostics = report
        self._apply_source_guarantee_issues(report)
        for issue in report.issues:
            if issue.severity == "warning":
                warnings.warn(issue.message, ALEBOWarning, stacklevel=2)
                self.algorithm.invalidate_source_guarantees(issue.code)
            elif issue.severity == "error":
                raise ERROR_BY_CODE.get(issue.code, ALEBOError)(
                    issue.message,
                    environment=self.environment,
                    algorithm=self.algorithm,
                    diagnostics=report,
                )
        return report


__all__ = [name for name in globals() if name.startswith("ALEBO") or name == "ERROR_BY_CODE"]
