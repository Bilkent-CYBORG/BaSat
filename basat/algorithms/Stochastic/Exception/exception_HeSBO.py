"""Diagnostics and failures specific to HeSBO."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

from ...algorithm import AlgorithmError, AlgorithmExceptionController
from ...helper import ExceptionDiagnosticReport

if TYPE_CHECKING:
    from ..HeSBO import HeSBO


class HeSBOError(AlgorithmError):
    """Base error for HeSBO."""


class HeSBOWarning(UserWarning):
    """Warning for qualified HeSBO configurations."""


class HeSBOEmbeddingDimensionError(HeSBOError):
    """Raised for an invalid embedding dimension."""


class HeSBOHashError(HeSBOError):
    """Raised for malformed hash/sign state."""


class HeSBOEmptyBucketError(HeSBOError):
    """Raised when an inverse depends on an unidentifiable empty bucket."""


class HeSBODataCompatibilityError(HeSBOError):
    """Raised when a finite action is not exactly hash representable."""


class HeSBOHistoricalDataError(HeSBOError):
    """Raised when historical actions are not exactly representable."""


class HeSBOCheckpointError(HeSBOError):
    """Raised for incompatible HeSBO checkpoint state."""


ERROR_BY_CODE = {
    "embedding_dimension": HeSBOEmbeddingDimensionError,
    "hash": HeSBOHashError,
    "empty_bucket": HeSBOEmptyBucketError,
    "environment_data": HeSBODataCompatibilityError,
    "previous_data": HeSBOHistoricalDataError,
    "data_backed_environment": HeSBODataCompatibilityError,
    "checkpoint": HeSBOCheckpointError,
}


class HeSBOException(AlgorithmExceptionController):
    """Validate HeSBO dimension, immutable hash, and empty buckets."""

    ERROR_TYPE = HeSBOError
    WARNING_TYPE = HeSBOWarning

    def diagnose_initialization(self) -> ExceptionDiagnosticReport:
        """Report HeSBO dimension, empty-bucket, and environment compatibility."""

        report = super().diagnose_initialization()
        self.issues = list(report.issues)
        algorithm = cast("HeSBO", self.algorithm)
        if not 1 <= algorithm.config.embedding_dimension <= algorithm.action_dimension:
            self.add_issue("error", "embedding_dimension", "HeSBO requires 1 <= d <= D.")
        if algorithm.empty_buckets:
            self.add_issue(
                "warning",
                "empty_bucket",
                "The sampled HeSBO map has empty buckets; finite inverse coordinates require metadata.",
            )
        if algorithm.q_batch is not None and algorithm.q_batch > 1:
            self.add_issue(
                "warning",
                "acquisition",
                "Joint q-batch HeSBO uses a Basat q-acquisition extension.",
                affects_source_guarantees=True,
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
        """Enforce HeSBO diagnostics and apply source-guarantee warnings."""

        report = self.diagnose_initialization()
        self.algorithm.initialization_diagnostics = report
        self._apply_source_guarantee_issues(report)
        for issue in report.issues:
            if issue.severity == "warning":
                warnings.warn(issue.message, HeSBOWarning, stacklevel=2)
            elif issue.severity == "error":
                raise ERROR_BY_CODE.get(issue.code, HeSBOError)(
                    issue.message,
                    environment=self.environment,
                    algorithm=self.algorithm,
                    diagnostics=report,
                )
        return report


__all__ = [name for name in globals() if name.startswith("HeSBO") or name == "ERROR_BY_CODE"]
