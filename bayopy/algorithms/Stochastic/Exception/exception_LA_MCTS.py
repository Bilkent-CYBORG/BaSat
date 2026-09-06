"""Diagnostics and named failures for Learning-Augmented MCTS."""

from __future__ import annotations

import importlib.util
import math
import warnings
from typing import Any, cast

from ...algorithm import AlgorithmError, AlgorithmExceptionController
from ...helper import ExceptionDiagnosticReport


class LA_MCTSError(AlgorithmError):
    """Base error for LA-MCTS."""


class LA_MCTSWarning(UserWarning):
    """Warning category for qualified LA-MCTS configurations."""


class LA_MCTSTreeError(LA_MCTSError):
    """Raised for invalid region-tree structure."""


class LA_MCTSClassifierError(LA_MCTSError):
    """Raised when clustering or classification cannot be performed."""


class LA_MCTSSplitError(LA_MCTSError):
    """Raised for an invalid region split."""


class LA_MCTSRegionError(LA_MCTSError):
    """Raised when a selected classifier region is infeasible."""


class LA_MCTSLocalOptimizerError(LA_MCTSError):
    """Raised when the configured local sampler is invalid or unavailable."""


class LA_MCTSDataCompatibilityError(LA_MCTSError):
    """Raised when no available finite row belongs to a feasible leaf."""


class LA_MCTSCheckpointError(LA_MCTSError):
    """Raised for incompatible LA-MCTS checkpoint state."""


ERROR_BY_CODE = {
    "tree": LA_MCTSTreeError,
    "classifier": LA_MCTSClassifierError,
    "split": LA_MCTSSplitError,
    "region": LA_MCTSRegionError,
    "local_optimizer": LA_MCTSLocalOptimizerError,
    "environment_data": LA_MCTSDataCompatibilityError,
    "previous_data": LA_MCTSDataCompatibilityError,
    "data_backed_environment": LA_MCTSDataCompatibilityError,
    "checkpoint": LA_MCTSCheckpointError,
}


class LA_MCTSException(AlgorithmExceptionController):
    """Validate tree, classifier, region, and local-sampler settings."""

    ERROR_TYPE = LA_MCTSError
    WARNING_TYPE = LA_MCTSWarning

    @property
    def _config(self) -> Any:
        """Return the concrete algorithm configuration for typed diagnostics."""

        return cast(Any, self.algorithm).config

    def _check_tree(self) -> None:
        """Validate initial design, split sizes, and maximum tree depth."""

        config = self._config
        if config.initial_points < 1 or config.split_threshold < 2:
            self.add_issue(
                "error", "tree", "initial_points must be positive and split_threshold at least two."
            )
        if config.min_child_size < 1 or config.max_depth < 1:
            self.add_issue("error", "tree", "min_child_size and max_depth must be positive.")
        if config.split_threshold < 2 * config.min_child_size:
            self.add_issue(
                "error", "split", "split_threshold must allow two minimum-size children."
            )

    def _check_classifier(self) -> None:
        """Require the RBF-SVM mode and optional scikit-learn tree extra."""

        if importlib.util.find_spec("sklearn") is None:
            self.add_issue(
                "error",
                "classifier",
                "LA-MCTS requires scikit-learn; install the 'bayopy[tree]' capability extra.",
            )
        if self._config.classifier != "svm_rbf":
            self.add_issue("error", "classifier", "classifier must be 'svm_rbf'.")

    def _check_region(self) -> None:
        """Validate rejection-sampling capacity and region exploration."""

        config = self._config
        if config.candidate_pool_size < 2 or config.max_resampling_attempts < 1:
            self.add_issue(
                "error",
                "region",
                "candidate_pool_size must exceed one and max_resampling_attempts be positive.",
            )
        if not math.isfinite(config.exploration_constant) or config.exploration_constant < 0:
            self.add_issue(
                "error", "region", "exploration_constant must be finite and non-negative."
            )

    def _check_local_optimizer(self) -> None:
        """Validate local search and its conditional CMA-ES capability boundary."""

        sampler = self._config.local_sampler
        if sampler not in {"turbo", "bo", "cmaes", "random"}:
            self.add_issue(
                "error",
                "local_optimizer",
                "local_sampler must be 'turbo', 'bo', 'cmaes', or 'random'.",
            )
        if sampler == "cmaes" and importlib.util.find_spec("cma") is None:
            self.add_issue(
                "error",
                "local_optimizer",
                "CMA-ES sampling requires the 'bayopy[optimizers]' capability extra.",
            )
        if self._config.gp_maxiter < 1:
            self.add_issue("error", "local_optimizer", "gp_maxiter must be positive.")

    def _check_preset(self) -> None:
        """Mark settings that depart from the explicit paper-general preset."""

        config = self._config
        defaults = (30, 1.0, 20, "turbo")
        selected = (
            config.initial_points,
            config.exploration_constant,
            config.split_threshold,
            config.local_sampler,
        )
        if selected != defaults:
            self.add_issue(
                "warning",
                "tree",
                "Selected LA-MCTS values differ from the paper-general preset.",
                details={"selected": selected, "paper_general": defaults},
                affects_source_guarantees=True,
            )

    def diagnose_initialization(self) -> ExceptionDiagnosticReport:
        """Build a public-data-only LA-MCTS compatibility report.

        Returns
        -------
        ExceptionDiagnosticReport
            Tree, classifier, region, local optimizer, dependency, and preset
            findings without reading hidden finite responses.
        """

        base = super().diagnose_initialization()
        self.issues = list(base.issues)
        self._check_tree()
        self._check_classifier()
        self._check_region()
        self._check_local_optimizer()
        self._check_preset()
        guarantees = base.guarantees_applicable and not any(
            bool(issue.details.get("affects_source_guarantees", False)) for issue in self.issues
        )
        self.report = ExceptionDiagnosticReport(
            type(self.algorithm).__name__, self.environment_type, tuple(self.issues), guarantees
        )
        return self.report

    def validate_initialization(self) -> ExceptionDiagnosticReport:
        """Persist guarantees, emit warnings, and raise mapped LA-MCTS errors."""

        report = self.diagnose_initialization()
        self.algorithm.initialization_diagnostics = report
        self._apply_source_guarantee_issues(report)
        self.algorithm.source_guarantees_applicable = report.guarantees_applicable
        for issue in report.issues:
            if issue.severity == "warning":
                warnings.warn(issue.message, LA_MCTSWarning, stacklevel=2)
                if issue.details.get("affects_source_guarantees", False):
                    self.algorithm._add_source_guarantee_reason(issue.code)
            elif issue.severity == "error":
                raise ERROR_BY_CODE.get(issue.code, LA_MCTSError)(
                    issue.message,
                    environment=self.environment,
                    algorithm=self.algorithm,
                    diagnostics=report,
                )
        return report


__all__ = [
    "ERROR_BY_CODE",
    "LA_MCTSCheckpointError",
    "LA_MCTSClassifierError",
    "LA_MCTSDataCompatibilityError",
    "LA_MCTSError",
    "LA_MCTSException",
    "LA_MCTSLocalOptimizerError",
    "LA_MCTSRegionError",
    "LA_MCTSSplitError",
    "LA_MCTSTreeError",
    "LA_MCTSWarning",
]
