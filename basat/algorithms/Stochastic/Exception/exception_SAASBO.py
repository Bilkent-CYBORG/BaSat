"""Diagnostics and named failures for fully Bayesian SAASBO."""

from __future__ import annotations

import importlib.util
import math
import warnings
from typing import Any, cast

from ...algorithm import AlgorithmError, AlgorithmExceptionController
from ...helper import ExceptionDiagnosticReport


class SAASBOError(AlgorithmError):
    """Base error for SAASBO."""


class SAASBOWarning(UserWarning):
    """Warning category for qualified SAASBO configurations."""


class SAASBOPriorError(SAASBOError):
    """Raised when the hierarchical SAAS prior is invalid."""


class SAASBOMCMCError(SAASBOError):
    """Raised when NUTS cannot be configured or executed."""


class SAASBODivergenceError(SAASBOMCMCError):
    """Raised when posterior diagnostics invalidate retained samples."""


class SAASBODataCompatibilityError(SAASBOError):
    """Raised when learner-visible full-dimensional rows are invalid."""


class SAASBOCheckpointError(SAASBOError):
    """Raised for incompatible SAASBO checkpoint state."""


ERROR_BY_CODE = {
    "prior": SAASBOPriorError,
    "mcmc": SAASBOMCMCError,
    "divergence": SAASBODivergenceError,
    "environment_data": SAASBODataCompatibilityError,
    "previous_data": SAASBODataCompatibilityError,
    "data_backed_environment": SAASBODataCompatibilityError,
    "checkpoint": SAASBOCheckpointError,
}


class SAASBOException(AlgorithmExceptionController):
    """Validate the SAAS prior, NUTS settings, and optional capability."""

    ERROR_TYPE = SAASBOError
    WARNING_TYPE = SAASBOWarning

    @property
    def _config(self) -> Any:
        """Return the concrete algorithm configuration for typed diagnostics."""

        return cast(Any, self.algorithm).config

    def _check_prior(self) -> None:
        """Validate the SAAS shrinkage scale and flag non-paper presets."""

        scale = self._config.shrinkage_scale
        if not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or scale <= 0:
            self.add_issue("error", "prior", "shrinkage_scale must be finite and positive.")
        if scale != 0.1:
            self.add_issue(
                "warning",
                "prior",
                "A shrinkage scale other than 0.1 changes the paper preset.",
                affects_source_guarantees=True,
            )

    def _check_mcmc(self) -> None:
        """Validate NUTS settings and availability of the optional Pyro extra."""

        config = self._config
        if config.warmup_steps < 1:
            self.add_issue("error", "mcmc", "warmup_steps must be positive.")
        if config.posterior_samples < 1 or config.thinning < 1:
            self.add_issue("error", "mcmc", "posterior_samples and thinning must be positive.")
        elif config.posterior_samples // config.thinning < 1:
            self.add_issue("error", "mcmc", "MCMC thinning would retain no samples.")
        if config.num_chains < 1 or config.max_tree_depth < 1:
            self.add_issue("error", "mcmc", "num_chains and max_tree_depth must be positive.")
        if config.min_effective_sample_size is None:
            self.add_issue(
                "warning",
                "mcmc",
                "Effective-sample-size rejection is disabled; this setting is intended "
                "only for non-inferential smoke runs.",
                affects_source_guarantees=True,
            )
        if importlib.util.find_spec("pyro") is None:
            self.add_issue(
                "error",
                "mcmc",
                "SAASBO requires Pyro; install the 'basat[mcmc]' capability extra.",
            )
        if self.algorithm.request_size > 1:
            self.add_issue(
                "warning",
                "mcmc",
                "Batch SAAS acquisition is a Basat extension to the sequential source preset.",
                affects_source_guarantees=True,
            )

    def diagnose_initialization(self) -> ExceptionDiagnosticReport:
        """Build a public-data-only SAAS prior and sampler report.

        Returns
        -------
        ExceptionDiagnosticReport
            Prior, NUTS, batching, and optional-capability findings.
        """

        base = super().diagnose_initialization()
        self.issues = list(base.issues)
        self._check_prior()
        self._check_mcmc()
        guarantees = base.guarantees_applicable and not any(
            bool(issue.details.get("affects_source_guarantees", False)) for issue in self.issues
        )
        self.report = ExceptionDiagnosticReport(
            type(self.algorithm).__name__, self.environment_type, tuple(self.issues), guarantees
        )
        return self.report

    def validate_initialization(self) -> ExceptionDiagnosticReport:
        """Persist guarantees, emit warnings, and raise mapped SAASBO errors."""

        report = self.diagnose_initialization()
        self.algorithm.initialization_diagnostics = report
        self._apply_source_guarantee_issues(report)
        self.algorithm.source_guarantees_applicable = report.guarantees_applicable
        for issue in report.issues:
            if issue.severity == "warning":
                warnings.warn(issue.message, SAASBOWarning, stacklevel=2)
                if issue.details.get("affects_source_guarantees", False):
                    self.algorithm._add_source_guarantee_reason(issue.code)
            elif issue.severity == "error":
                raise ERROR_BY_CODE.get(issue.code, SAASBOError)(
                    issue.message,
                    environment=self.environment,
                    algorithm=self.algorithm,
                    diagnostics=report,
                )
        return report


__all__ = [
    "ERROR_BY_CODE",
    "SAASBOCheckpointError",
    "SAASBODataCompatibilityError",
    "SAASBODivergenceError",
    "SAASBOError",
    "SAASBOException",
    "SAASBOMCMCError",
    "SAASBOPriorError",
    "SAASBOWarning",
]
