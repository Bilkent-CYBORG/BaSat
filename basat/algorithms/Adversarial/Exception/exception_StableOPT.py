"""Typed diagnostics and runtime failures for :mod:`StableOPT`."""

from __future__ import annotations

import math
import warnings
from typing import Any, cast

from ...algorithm import AlgorithmError, AlgorithmExceptionController
from ...helper import ExceptionDiagnosticReport


class StableOPTError(AlgorithmError):
    """Base error for StableOPT."""


class StableOPTWarning(UserWarning):
    """Warning emitted for qualified StableOPT configurations."""


class StableOPTPerturbationSetError(StableOPTError):
    """Raised when the robust perturbation set is invalid."""


class StableOPTInnerSolverError(StableOPTError):
    """Raised when an inner robust optimization has no valid certificate."""


class StableOPTOuterSolverError(StableOPTError):
    """Raised when robust-center optimization fails."""


class StableOPTDataCompatibilityError(StableOPTError):
    """Raised when finite or historical coordinates are not representable."""


class StableOPTCheckpointError(StableOPTError):
    """Raised for malformed StableOPT checkpoint state."""


ERROR_BY_CODE = {
    "perturbation_set": StableOPTPerturbationSetError,
    "inner_solver": StableOPTInnerSolverError,
    "outer_solver": StableOPTOuterSolverError,
    "environment_data": StableOPTDataCompatibilityError,
    "previous_data": StableOPTDataCompatibilityError,
    "data_backed_environment": StableOPTDataCompatibilityError,
    "checkpoint": StableOPTCheckpointError,
}


class StableOPTException(AlgorithmExceptionController):
    """Diagnose StableOPT using public geometry and environment declarations."""

    ERROR_TYPE = StableOPTError
    WARNING_TYPE = StableOPTWarning

    def _check_environment_contract(self) -> None:
        """The inherited declaration checker owns common feedback fields."""

    def _check_native_environment(self) -> None:
        """Qualify the extra external attack used by adversarial environments."""

        if self.environment_type == "adversarial":
            self.add_issue(
                "warning",
                "environment.external_attack_extension",
                "Applying an external attack after StableOPT's internal robust query is a Basat extension.",
                affects_source_guarantees=True,
            )

    def _check_perturbation_set(self) -> None:
        """Record public configuration errors in perturbation geometry."""

        config = cast(Any, self.algorithm).config
        if not math.isfinite(float(config.perturbation_radius)) or config.perturbation_radius < 0:
            self.add_issue(
                "error", "perturbation_set", "perturbation_radius must be finite and non-negative."
            )
        if getattr(config.distance, "norm", None) not in {"l2", "linf"}:
            self.add_issue(
                "error", "perturbation_set", "StableOPT distance norm must be 'l2' or 'linf'."
            )

    def _check_inner_solver(self) -> None:
        """Require the supported deterministic inner multistart solver."""

        config = cast(Any, self.algorithm).config
        if getattr(config.inner_solver, "method", None) != "multistart":
            self.add_issue("error", "inner_solver", "StableOPT requires a multistart inner solver.")

    def _check_outer_solver(self) -> None:
        """Require the supported deterministic outer multistart solver."""

        config = cast(Any, self.algorithm).config
        if getattr(config.outer_solver, "method", None) != "multistart":
            self.add_issue("error", "outer_solver", "StableOPT requires a multistart outer solver.")

    def _check_environment_data(self) -> None:
        """Full-dimensional historical coordinates need no inverse transform."""

    def _check_previous_data(self) -> None:
        """Generic history validation is inherited."""

    def _check_data_backed_environment(self) -> None:
        """Finite compatibility is checked against each public candidate table."""

    def diagnose_initialization(self) -> ExceptionDiagnosticReport:
        """Build a confidentiality-safe StableOPT compatibility report.

        Returns
        -------
        ExceptionDiagnosticReport
            Immutable findings derived only from configuration, declarations,
            and learner-visible history.
        """

        report = super().diagnose_initialization()
        self.issues = list(report.issues)
        self._check_environment_contract()
        self._check_native_environment()
        self._check_perturbation_set()
        self._check_inner_solver()
        self._check_outer_solver()
        self._check_environment_data()
        self._check_previous_data()
        self._check_data_backed_environment()
        self.report = ExceptionDiagnosticReport(
            algorithm_name=type(self.algorithm).__name__,
            environment_type=self.environment_type,
            issues=tuple(self.issues),
            guarantees_applicable=not any(
                bool(issue.details.get("affects_source_guarantees", False)) for issue in self.issues
            ),
        )
        return self.report

    def validate_initialization(self) -> ExceptionDiagnosticReport:
        """Persist diagnostics, emit warnings, and raise the mapped typed error.

        Returns
        -------
        ExceptionDiagnosticReport
            The validated report when no error-severity issue is present.

        Raises
        ------
        StableOPTError
            The algorithm-specific error mapped from the first error finding.
        """

        report = self.diagnose_initialization()
        self.algorithm.initialization_diagnostics = report
        self._apply_source_guarantee_issues(report)
        for issue in report.issues:
            if bool(issue.details.get("affects_source_guarantees", False)):
                self.algorithm.invalidate_source_guarantees(issue.code)
            if issue.severity == "warning":
                warnings.warn(issue.message, StableOPTWarning, stacklevel=2)
            elif issue.severity == "error":
                raise ERROR_BY_CODE.get(issue.code, StableOPTError)(
                    issue.message,
                    environment=self.environment,
                    algorithm=self.algorithm,
                    diagnostics=report,
                )
        return report


__all__ = [name for name in globals() if name.startswith("StableOPT") or name == "ERROR_BY_CODE"]
