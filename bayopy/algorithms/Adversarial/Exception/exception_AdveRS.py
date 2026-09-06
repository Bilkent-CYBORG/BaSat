"""Typed diagnostics and runtime failures for the AdveRS family."""

from __future__ import annotations

import math
import warnings
from typing import Any, cast

from ...algorithm import AlgorithmError, AlgorithmExceptionController
from ...helper import ExceptionDiagnosticReport


class AdveRSError(AlgorithmError):
    """Base error for AdveRS variants."""


class AdveRSWarning(UserWarning):
    """Warning emitted for qualified AdveRS configurations."""


class AdveRSVariantError(AdveRSError):
    """Raised for an unknown or internally inconsistent variant."""


class AdveRSAspirationError(AdveRSError):
    """Raised for an invalid aspiration level."""


class AdveRSPerturbationError(AdveRSError):
    """Raised for invalid perturbation geometry."""


class AdveRSInnerSolverError(AdveRSError):
    """Raised when fragility or radius optimization fails."""


class AdveRSDataCompatibilityError(AdveRSError):
    """Raised when a finite neighborhood cannot represent the method."""


class AdveRSCheckpointError(AdveRSError):
    """Raised for malformed AdveRS checkpoint state."""


ERROR_BY_CODE = {
    "variant": AdveRSVariantError,
    "aspiration": AdveRSAspirationError,
    "perturbation": AdveRSPerturbationError,
    "inner_solver": AdveRSInnerSolverError,
    "environment_data": AdveRSDataCompatibilityError,
    "previous_data": AdveRSDataCompatibilityError,
    "data_backed_environment": AdveRSDataCompatibilityError,
    "checkpoint": AdveRSCheckpointError,
}


class AdveRSException(AlgorithmExceptionController):
    """Validate AdveRS configuration without reading latent attacks."""

    ERROR_TYPE = AdveRSError
    WARNING_TYPE = AdveRSWarning

    def _check_environment_contract(self) -> None:
        """Common feedback declarations are checked by the base controller."""

    def _check_native_environment(self) -> None:
        """Native adversarial-family validation is inherited."""

    def _check_variant(self) -> None:
        """Record inconsistent RS variant, exponent, and decision settings."""

        config = cast(Any, self.algorithm).config
        if config.variant not in {"RS1", "RS2", "RSG"}:
            self.add_issue("error", "variant", "variant must be RS1, RS2, or RSG.")
        if config.decision_model == "thompson" and config.variant != "RSG":
            self.add_issue("error", "variant", "Thompson decisions are supported only for RSG.")
        if config.variant in {"RS1", "RS2"} and config.p != 1.0:
            self.add_issue("error", "variant", "RS1 and RS2 require p=1.")

    def _check_aspiration(self) -> None:
        """Require a finite public aspiration threshold."""

        config = cast(Any, self.algorithm).config
        if not math.isfinite(float(config.aspiration)):
            self.add_issue("error", "aspiration", "aspiration must be finite.")

    def _check_perturbation(self) -> None:
        """Record invalid public perturbation-domain geometry."""

        domain = cast(Any, self.algorithm).config.perturbation_domain
        if domain.radius is not None and (
            not math.isfinite(float(domain.radius)) or domain.radius < 0
        ):
            self.add_issue("error", "perturbation", "Perturbation radius must be non-negative.")
        if domain.norm not in {"l2", "linf"}:
            self.add_issue("error", "perturbation", "Perturbation norm must be l2 or linf.")

    def _check_inner_solver(self) -> None:
        """Require the supported deterministic candidate-search solver."""

        config = cast(Any, self.algorithm).config
        if config.candidate_solver.method != "multistart":
            self.add_issue("error", "inner_solver", "AdveRS requires multistart candidate search.")

    def _check_environment_data(self) -> None:
        """Data are full-dimensional and checked by the shared base."""

    def _check_previous_data(self) -> None:
        """Generic history validation is inherited."""

    def _check_data_backed_environment(self) -> None:
        """Each public candidate table defines the finite perturbation graph."""

    def diagnose_initialization(self) -> ExceptionDiagnosticReport:
        """Build a confidentiality-safe AdveRS compatibility report.

        Returns
        -------
        ExceptionDiagnosticReport
            Immutable findings derived only from configuration, declarations,
            and learner-visible history; latent attacks are never inspected.
        """

        report = super().diagnose_initialization()
        self.issues = list(report.issues)
        self._check_environment_contract()
        self._check_native_environment()
        self._check_variant()
        self._check_aspiration()
        self._check_perturbation()
        self._check_inner_solver()
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
        AdveRSError
            The algorithm-specific error mapped from the first error finding.
        """

        report = self.diagnose_initialization()
        self.algorithm.initialization_diagnostics = report
        self._apply_source_guarantee_issues(report)
        for issue in report.issues:
            if issue.severity == "warning":
                warnings.warn(issue.message, AdveRSWarning, stacklevel=2)
            elif issue.severity == "error":
                raise ERROR_BY_CODE.get(issue.code, AdveRSError)(
                    issue.message,
                    environment=self.environment,
                    algorithm=self.algorithm,
                    diagnostics=report,
                )
        return report


__all__ = [name for name in globals() if name.startswith("AdveRS") or name == "ERROR_BY_CODE"]
