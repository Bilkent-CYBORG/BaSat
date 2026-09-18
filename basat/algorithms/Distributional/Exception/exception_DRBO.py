"""Typed diagnostics and runtime failures for distributionally robust BO."""

from __future__ import annotations

import importlib.util
import warnings
from typing import Any, cast

from ...algorithm import AlgorithmError, AlgorithmExceptionController
from ...helper import ExceptionDiagnosticReport


class DRBOError(AlgorithmError):
    """Base error for DRBO."""


class DRBOWarning(UserWarning):
    """Warning emitted for qualified DRBO configurations."""


class DRBOContextSupportError(DRBOError):
    """Raised when finite context support is malformed or misaligned."""


class DRBOReferenceDistributionError(DRBOError):
    """Raised for a reference vector outside the simplex."""


class DRBOMarginError(DRBOError):
    """Raised for an invalid ambiguity radius."""


class DRBOMMDMatrixError(DRBOError):
    """Raised when the MMD matrix is not positive semidefinite."""


class DRBOInnerSolverError(DRBOError):
    """Raised when the ambiguity-set program fails strict residual checks."""


class DRBOActionOptimizerError(DRBOError):
    """Raised when continuous outer-action refinement cannot be certified."""


class DRBODataCompatibilityError(DRBOError):
    """Raised for incomplete or misaligned finite/history data."""


class DRBOCheckpointError(DRBOError):
    """Raised for malformed DRBO checkpoint state."""


ERROR_BY_CODE = {
    "context_support": DRBOContextSupportError,
    "reference_distribution": DRBOReferenceDistributionError,
    "margin": DRBOMarginError,
    "mmd_matrix": DRBOMMDMatrixError,
    "inner_solver": DRBOInnerSolverError,
    "action_optimizer": DRBOActionOptimizerError,
    "environment_data": DRBODataCompatibilityError,
    "previous_data": DRBODataCompatibilityError,
    "data_backed_environment": DRBODataCompatibilityError,
    "checkpoint": DRBOCheckpointError,
}


class DRBOException(AlgorithmExceptionController):
    """Validate DRBO using public context support and dependency capability."""

    ERROR_TYPE = DRBOError
    WARNING_TYPE = DRBOWarning

    def _check_environment_contract(self) -> None:
        """The base controller checks reference, margin, and realized context."""

    def _check_native_environment(self) -> None:
        """Native distributional-family validation is inherited."""

    def _check_context_support(self) -> None:
        """Record malformed or non-finite configured context support."""

        try:
            cast(Any, self.algorithm)._validate_context_support()
        except Exception as error:
            self.add_issue("error", "context_support", str(error))

    def _check_reference_distribution(self) -> None:
        """Round-dependent reference vectors are checked at ask time."""

    def _check_margin(self) -> None:
        """Round-dependent ambiguity margins are checked at ask time."""

    def _check_mmd_matrix(self) -> None:
        """Record failure to construct a positive-semidefinite MMD Gram matrix."""

        try:
            cast(Any, self.algorithm)._context_gram_matrix()
        except Exception as error:
            self.add_issue("error", "mmd_matrix", str(error))

    def _check_inner_solver(self) -> None:
        """Validate the selected ambiguity solver and its optional dependency."""

        config = cast(Any, self.algorithm).config
        solver = config.inner_solver
        if solver not in {"cvxpy", "source_cvxpy", "slsqp", "enumeration"}:
            self.add_issue("error", "inner_solver", f"Unknown DRBO solver {solver!r}.")
        elif solver in {"cvxpy", "source_cvxpy"} and importlib.util.find_spec("cvxpy") is None:
            self.add_issue(
                "error",
                "inner_solver",
                "The DRBO CVXPY solver requires the 'robust' extra: pip install basat[robust].",
            )
        elif solver == "enumeration" and config.distribution_grid is None:
            self.add_issue(
                "warning",
                "inner_solver.generated_grid_enumeration",
                (
                    "DRBO inner_solver='enumeration' is using an automatically generated "
                    "finite simplex grid at "
                    f"enumeration_resolution={config.enumeration_resolution}. This coarse "
                    "grid approximates the governing source's continuous ambiguity-set "
                    "optimization."
                ),
                details={
                    "inner_solver": solver,
                    "grid_source": "generated",
                    "enumeration_resolution": config.enumeration_resolution,
                },
                affects_source_guarantees=True,
            )
        elif solver == "enumeration":
            self.add_issue(
                "warning",
                "inner_solver.explicit_grid_enumeration",
                (
                    "DRBO inner_solver='enumeration' is restricted to the explicit "
                    "distribution_grid supplied by the user. The table is preserved verbatim, "
                    "but it is a finite approximation of the governing source's continuous "
                    "ambiguity-set optimization."
                ),
                details={
                    "inner_solver": solver,
                    "grid_source": "user_supplied",
                },
                affects_source_guarantees=True,
            )

    def _check_environment_data(self) -> None:
        """Public historical context rows are validated by the algorithm."""

    def _check_previous_data(self) -> None:
        """Generic history validation is inherited."""

    def _check_data_backed_environment(self) -> None:
        """Require exact alignment between finite environment and DRBO supports."""

        support = getattr(self.environment, "context_support", None)
        if support is not None and not cast(Any, self.algorithm)._support_matches(support):
            self.add_issue(
                "error",
                "data_backed_environment",
                "Environment and DRBO context supports are not aligned.",
            )

    def diagnose_initialization(self) -> ExceptionDiagnosticReport:
        """Build a non-mutating DRBO report from learner-visible declarations.

        The diagnostic path checks the public reference/context contract,
        configured MMD geometry, solver capability, and support alignment.  It
        deliberately does not inspect a distributional environment's hidden
        true sampling distribution.
        """

        report = super().diagnose_initialization()
        self.issues = list(report.issues)
        self._check_context_support()
        self._check_reference_distribution()
        self._check_margin()
        self._check_mmd_matrix()
        self._check_inner_solver()
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
                warnings.warn(issue.message, DRBOWarning, stacklevel=2)
            elif issue.severity == "error":
                raise ERROR_BY_CODE.get(issue.code, DRBOError)(
                    issue.message,
                    environment=self.environment,
                    algorithm=self.algorithm,
                    diagnostics=report,
                )
        return report


__all__ = [name for name in globals() if name.startswith("DRBO") or name == "ERROR_BY_CODE"]
