"""Typed diagnostics and runtime failures for RoBOS."""

from __future__ import annotations

import importlib.util
import math
import warnings
from typing import Any, cast

from ...algorithm import AlgorithmError, AlgorithmExceptionController
from ...helper import ExceptionDiagnosticReport


class RoBOSError(AlgorithmError):
    """Base error for RoBOS."""


class RoBOSWarning(UserWarning):
    """Warning emitted for qualified RoBOS configurations."""


class RoBOSContextSupportError(RoBOSError):
    """Raised when finite context support is malformed or misaligned."""


class RoBOSReferenceDistributionError(RoBOSError):
    """Raised for a reference vector outside the aligned simplex."""


class RoBOSAspirationError(RoBOSError):
    """Raised for an invalid aspiration level."""


class RoBOSMMDMatrixError(RoBOSError):
    """Raised when the context Gram matrix is degenerate on the simplex."""


class RoBOSFragilitySolverError(RoBOSError):
    """Raised when no feasible, residual-checked fragility solution exists."""


class RoBOSDataCompatibilityError(RoBOSError):
    """Raised for incomplete or misaligned finite/history data."""


class RoBOSCheckpointError(RoBOSError):
    """Raised for malformed RoBOS checkpoint state."""


ERROR_BY_CODE = {
    "context_support": RoBOSContextSupportError,
    "reference_distribution": RoBOSReferenceDistributionError,
    "aspiration": RoBOSAspirationError,
    "mmd_matrix": RoBOSMMDMatrixError,
    "fragility_solver": RoBOSFragilitySolverError,
    "environment_data": RoBOSDataCompatibilityError,
    "previous_data": RoBOSDataCompatibilityError,
    "data_backed_environment": RoBOSDataCompatibilityError,
    "checkpoint": RoBOSCheckpointError,
}


class RoBOSException(AlgorithmExceptionController):
    """Validate RoBOS without inspecting a true sampling distribution."""

    ERROR_TYPE = RoBOSError
    WARNING_TYPE = RoBOSWarning

    def _check_environment_contract(self) -> None:
        """Shared declaration checks enforce reference/context fields."""

    def _check_native_environment(self) -> None:
        """Native distributional-family validation is inherited."""

    def _check_context_support(self) -> None:
        """Record malformed or non-finite configured context support."""

        try:
            cast(Any, self.algorithm)._validate_context_support()
        except Exception as error:
            self.add_issue("error", "context_support", str(error))

    def _check_reference_distribution(self) -> None:
        """Round-dependent references are checked at ask time."""

    def _check_aspiration(self) -> None:
        """Require a finite fixed aspiration level for fragility ratios."""

        if not math.isfinite(float(cast(Any, self.algorithm).config.aspiration)):
            self.add_issue("error", "aspiration", "aspiration must be finite.")

    def _check_mmd_matrix(self) -> None:
        """Require context-kernel definiteness on the simplex tangent space."""

        try:
            cast(Any, self.algorithm)._context_gram_matrix()
        except Exception as error:
            self.add_issue("error", "mmd_matrix", str(error))

    def _check_fragility_solver(self) -> None:
        """Validate the simplex solver choice and SOCP dependency capability."""

        config = cast(Any, self.algorithm).config
        solver = config.simplex_solver
        if solver not in {"socp", "slsqp", "enumeration", "source_cobyla"}:
            self.add_issue("error", "fragility_solver", f"Unknown solver {solver!r}.")
        elif solver == "socp" and importlib.util.find_spec("cvxpy") is None:
            self.add_issue(
                "error",
                "fragility_solver",
                "The RoBOS SOCP solver requires the 'robust' extra (cvxpy).",
            )
        elif solver == "source_cobyla":
            self.add_issue(
                "warning",
                "fragility_solver.source_cobyla",
                "RoBOS source_cobyla reproduces the author's approximate one-start "
                "regularized ratio solver, not a certified optimum. SciPy before 1.16 "
                "uses Powell COBYLA; newer SciPy uses PRIMA and may select different actions.",
                affects_source_guarantees=True,
            )
        elif solver == "enumeration" and config.distribution_grid is None:
            self.add_issue(
                "warning",
                "fragility_solver.generated_grid_enumeration",
                (
                    "RoBOS simplex_solver='enumeration' is using an automatically "
                    "generated finite simplex grid at "
                    f"enumeration_resolution={config.enumeration_resolution}. This coarse "
                    "grid approximates the governing source's continuous fragility program."
                ),
                details={
                    "simplex_solver": solver,
                    "grid_source": "generated",
                    "enumeration_resolution": config.enumeration_resolution,
                },
                affects_source_guarantees=True,
            )
        elif solver == "enumeration":
            self.add_issue(
                "warning",
                "fragility_solver.explicit_grid_enumeration",
                (
                    "RoBOS simplex_solver='enumeration' is restricted to the explicit "
                    "distribution_grid supplied by the user. The table is preserved verbatim, "
                    "but it is a finite approximation of the governing source's continuous "
                    "fragility program."
                ),
                details={
                    "simplex_solver": solver,
                    "grid_source": "user_supplied",
                },
                affects_source_guarantees=True,
            )

    def _check_environment_data(self) -> None:
        """Public historical context rows are validated by the algorithm."""

    def _check_previous_data(self) -> None:
        """Generic history validation is inherited."""

    def _check_data_backed_environment(self) -> None:
        """Require exact alignment between finite environment and RoBOS supports."""

        support = getattr(self.environment, "context_support", None)
        if support is not None and not cast(Any, self.algorithm)._support_matches(support):
            self.add_issue(
                "error",
                "data_backed_environment",
                "Environment and RoBOS context supports are not aligned.",
            )

    def diagnose_initialization(self) -> ExceptionDiagnosticReport:
        """Build a non-mutating report without reading hidden sampling state.

        Checks use only declared capabilities, public finite context support,
        kernel geometry, and configured solver availability.  The environment's
        true sampling distribution remains confidential audit information.
        """

        report = super().diagnose_initialization()
        self.issues = list(report.issues)
        self._check_context_support()
        self._check_reference_distribution()
        self._check_aspiration()
        self._check_mmd_matrix()
        self._check_fragility_solver()
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
                warnings.warn(issue.message, RoBOSWarning, stacklevel=2)
            elif issue.severity == "error":
                raise ERROR_BY_CODE.get(issue.code, RoBOSError)(
                    issue.message,
                    environment=self.environment,
                    algorithm=self.algorithm,
                    diagnostics=report,
                )
        return report


__all__ = [name for name in globals() if name.startswith("RoBOS") or name == "ERROR_BY_CODE"]
