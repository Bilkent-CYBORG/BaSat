"""Diagnostics and named failures for Monte Carlo Tree Search VS."""

from __future__ import annotations

import math
import warnings
from typing import Any, cast

from ...algorithm import AlgorithmError, AlgorithmExceptionController
from ...helper import ExceptionDiagnosticReport


class MCTS_VSError(AlgorithmError):
    """Base error for MCTS-VS."""


class MCTS_VSWarning(UserWarning):
    """Warning category for qualified MCTS-VS configurations."""


class MCTS_VSTreeError(MCTS_VSError):
    """Raised when the variable tree is structurally invalid."""


class MCTS_VSScoreError(MCTS_VSError):
    """Raised when variable scores cannot be computed safely."""


class MCTS_VSSplitError(MCTS_VSError):
    """Raised when split parameters violate tree invariants."""


class MCTS_VSInnerOptimizerError(MCTS_VSError):
    """Raised when the configured nested optimizer is unavailable or invalid."""


class MCTS_VSDataCompatibilityError(MCTS_VSError):
    """Raised when learner-visible rows lack a valid variable representation."""


class MCTS_VSCheckpointError(MCTS_VSError):
    """Raised for incompatible method-specific checkpoint state."""


ERROR_BY_CODE = {
    "tree": MCTS_VSTreeError,
    "score": MCTS_VSScoreError,
    "split": MCTS_VSSplitError,
    "inner_optimizer": MCTS_VSInnerOptimizerError,
    "environment_data": MCTS_VSDataCompatibilityError,
    "previous_data": MCTS_VSDataCompatibilityError,
    "data_backed_environment": MCTS_VSDataCompatibilityError,
    "checkpoint": MCTS_VSCheckpointError,
}


class MCTS_VSException(AlgorithmExceptionController):
    """Validate MCTS-VS configuration without reading hidden responses."""

    ERROR_TYPE = MCTS_VSError
    WARNING_TYPE = MCTS_VSWarning

    @property
    def _config(self) -> Any:
        """Return the concrete algorithm configuration for typed diagnostics."""

        return cast(Any, self.algorithm).config

    def _check_tree(self) -> None:
        """Validate rollout counts, archive limits, and the mutable root node."""

        config = self._config
        if config.Nv < 1 or config.Ns < 1:
            self.add_issue("error", "tree", "Nv and Ns must both be positive.")
        if config.archive_size < 1 or config.bad_count_limit < 1:
            self.add_issue("error", "tree", "archive_size and bad_count_limit must be positive.")
        state = getattr(self.algorithm, "state", None)
        if state is not None and state.nodes:
            root = state.nodes.get(state.root_id)
            if root is None or root.parent_id is not None:
                self.add_issue("error", "tree", "The variable tree has no valid root node.")

    def _check_score(self) -> None:
        """Require a finite nonnegative tree exploration constant."""

        value = self._config.exploration_constant
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            self.add_issue(
                "error", "score", "exploration_constant must be finite and non-negative."
            )

    def _check_split(self) -> None:
        """Require a split threshold capable of producing variable children."""

        if self._config.split_threshold < 2:
            self.add_issue("error", "split", "split_threshold must be at least two.")

    def _check_inner_optimizer(self) -> None:
        """Validate BO/TuRBO pool, fit, and sequential transport settings."""

        config = self._config
        if config.inner_optimizer not in {"bo", "turbo"}:
            self.add_issue("error", "inner_optimizer", "inner_optimizer must be 'bo' or 'turbo'.")
        if config.candidate_pool_size < 2:
            self.add_issue(
                "error",
                "inner_optimizer",
                "candidate_pool_size must contain at least two points.",
            )
        if config.gp_maxiter < 1:
            self.add_issue("error", "inner_optimizer", "gp_maxiter must be positive.")

    def _check_source_defaults(self) -> None:
        """Mark any departure from the official regression defaults."""

        config = self._config
        source_defaults = (2, 3, 20, 5, 3, 0.1)
        selected = (
            config.Nv,
            config.Ns,
            config.archive_size,
            config.bad_count_limit,
            config.split_threshold,
            config.exploration_constant,
        )
        if selected != source_defaults:
            self.add_issue(
                "warning",
                "tree",
                "Non-default MCTS-VS tree parameters change source-regression mode.",
                details={"selected": selected, "source_defaults": source_defaults},
                affects_source_guarantees=True,
            )

    def diagnose_initialization(self) -> ExceptionDiagnosticReport:
        """Build a public-data-only MCTS-VS compatibility report.

        Returns
        -------
        ExceptionDiagnosticReport
            Tree, score, optimizer, and source-preset findings without access to
            hidden finite responses.
        """

        base = super().diagnose_initialization()
        self.issues = list(base.issues)
        self._check_tree()
        self._check_score()
        self._check_split()
        self._check_inner_optimizer()
        self._check_source_defaults()
        guarantees = base.guarantees_applicable and not any(
            bool(issue.details.get("affects_source_guarantees", False)) for issue in self.issues
        )
        self.report = ExceptionDiagnosticReport(
            type(self.algorithm).__name__, self.environment_type, tuple(self.issues), guarantees
        )
        return self.report

    def validate_initialization(self) -> ExceptionDiagnosticReport:
        """Persist guarantees, emit warnings, and raise mapped MCTS-VS errors."""

        report = self.diagnose_initialization()
        self.algorithm.initialization_diagnostics = report
        self._apply_source_guarantee_issues(report)
        self.algorithm.source_guarantees_applicable = report.guarantees_applicable
        for issue in report.issues:
            if issue.severity == "warning":
                warnings.warn(issue.message, MCTS_VSWarning, stacklevel=2)
                if issue.details.get("affects_source_guarantees", False):
                    self.algorithm._add_source_guarantee_reason(issue.code)
            elif issue.severity == "error":
                raise ERROR_BY_CODE.get(issue.code, MCTS_VSError)(
                    issue.message,
                    environment=self.environment,
                    algorithm=self.algorithm,
                    diagnostics=report,
                )
        return report


__all__ = [
    "ERROR_BY_CODE",
    "MCTS_VSCheckpointError",
    "MCTS_VSDataCompatibilityError",
    "MCTS_VSError",
    "MCTS_VSException",
    "MCTS_VSInnerOptimizerError",
    "MCTS_VSScoreError",
    "MCTS_VSSplitError",
    "MCTS_VSTreeError",
    "MCTS_VSWarning",
]
