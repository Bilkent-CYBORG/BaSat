"""Diagnostics and named failures for adaptive grouped variable selection."""

from __future__ import annotations

import math
import warnings
from typing import Any, cast

from ...algorithm import AlgorithmError, AlgorithmExceptionController
from ...helper import ExceptionDiagnosticReport


class AG_VSError(AlgorithmError):
    """Base error for AG-VS."""


class AG_VSWarning(UserWarning):
    """Warning category for qualified AG-VS configurations."""


class AGVSEffectiveDimensionError(AG_VSError):
    """Raised for an invalid manuscript effective dimension."""


class AGVSAnchorError(AG_VSError):
    """Raised when a pre-batch best-point completion is invalid."""


class AGVSNodeError(AG_VSError):
    """Raised for invalid node-local state."""


class AGVSSplitError(AG_VSError):
    """Raised when a coordinate split violates disjointness."""


class AGVSMergeError(AG_VSError):
    """Raised when a sibling merge violates the manuscript rule."""


class AGVSDataCompatibilityError(AG_VSError):
    """Raised when finite candidates cannot represent a node acquisition."""


class AGVSCheckpointError(AG_VSError):
    """Raised for incompatible AG-VS checkpoint state."""


ERROR_BY_CODE = {
    "effective_dimension": AGVSEffectiveDimensionError,
    "anchor": AGVSAnchorError,
    "node": AGVSNodeError,
    "split_merge": AGVSSplitError,
    "merge": AGVSMergeError,
    "environment_data": AGVSDataCompatibilityError,
    "previous_data": AGVSDataCompatibilityError,
    "data_backed_environment": AGVSDataCompatibilityError,
    "checkpoint": AGVSCheckpointError,
}


class AG_VSException(AlgorithmExceptionController):
    """Validate local-manuscript dimensions, anchors, nodes, and transitions."""

    ERROR_TYPE = AG_VSError
    WARNING_TYPE = AG_VSWarning

    @property
    def _config(self) -> Any:
        """Return the concrete algorithm configuration for typed diagnostics."""

        return cast(Any, self.algorithm).config

    def _check_effective_dimension(self) -> None:
        """Require the manuscript effective dimension to lie within ``D_env``."""

        value = self._config.effective_dimension
        if value < 1 or value > self.algorithm.action_dimension:
            self.add_issue(
                "error",
                "effective_dimension",
                "effective_dimension must lie between one and D_env.",
            )

    def _check_anchor(self) -> None:
        """Validate every present node anchor as a finite full-space action."""

        state = getattr(self.algorithm, "state", None)
        if state is None:
            return
        for node in state.nodes.values():
            if node.anchor is not None and (
                node.anchor.ndim != 1
                or node.anchor.numel() != self.algorithm.action_dimension
                or not node.anchor.isfinite().all()
            ):
                self.add_issue("error", "anchor", f"Node {node.node_id} has an invalid anchor.")

    def _check_node(self) -> None:
        """Validate visit, initial-design, optimization, and restart settings."""

        config = self._config
        if config.sequence_length < 1 or config.initial_points < 1:
            self.add_issue("error", "node", "sequence_length and initial_points must be positive.")
        if (
            config.gp_maxiter < 1
            or config.acquisition_restarts < 1
            or config.acquisition_raw_samples < 1
        ):
            self.add_issue("error", "node", "GP and acquisition budgets must be positive.")
        if config.stagnation_patience is not None and config.stagnation_patience < 1:
            self.add_issue("error", "node", "stagnation_patience must be None or positive.")

    def _check_split_merge(self) -> None:
        """Require enough groups for a meaningful disjoint split."""

        if self._config.split_groups < 2:
            self.add_issue("error", "split_merge", "split_groups must be at least two.")

    def _check_stopping(self) -> None:
        """Validate paired posterior-stopping settings and sequential semantics."""

        config = self._config
        enabled = config.stopping_threshold is not None or config.stopping_delta is not None
        if enabled and (
            config.stopping_threshold is None
            or config.stopping_delta is None
            or not math.isfinite(config.stopping_threshold)
            or not 0 < config.stopping_delta < 1
        ):
            self.add_issue(
                "error",
                "node",
                "stopping_threshold and a stopping_delta in (0, 1) must be supplied together.",
            )

    def diagnose_initialization(self) -> ExceptionDiagnosticReport:
        """Build a public-data-only AG-VS tree and completion-point report.

        Returns
        -------
        ExceptionDiagnosticReport
            Dimension, node, split/merge, and stopping findings.
        """

        base = super().diagnose_initialization()
        self.issues = list(base.issues)
        self._check_effective_dimension()
        self._check_anchor()
        self._check_node()
        self._check_split_merge()
        self._check_stopping()
        self.report = ExceptionDiagnosticReport(
            type(self.algorithm).__name__,
            self.environment_type,
            tuple(self.issues),
            base.guarantees_applicable,
        )
        return self.report

    def validate_initialization(self) -> ExceptionDiagnosticReport:
        """Persist guarantees, emit warnings, and raise mapped AG-VS errors."""

        report = self.diagnose_initialization()
        self.algorithm.initialization_diagnostics = report
        self._apply_source_guarantee_issues(report)
        self.algorithm.source_guarantees_applicable = report.guarantees_applicable
        for issue in report.issues:
            if issue.severity == "warning":
                warnings.warn(issue.message, AG_VSWarning, stacklevel=2)
            elif issue.severity == "error":
                raise ERROR_BY_CODE.get(issue.code, AG_VSError)(
                    issue.message,
                    environment=self.environment,
                    algorithm=self.algorithm,
                    diagnostics=report,
                )
        return report


__all__ = [
    "ERROR_BY_CODE",
    "AGVSAnchorError",
    "AGVSCheckpointError",
    "AGVSDataCompatibilityError",
    "AGVSEffectiveDimensionError",
    "AGVSMergeError",
    "AGVSNodeError",
    "AGVSSplitError",
    "AG_VSError",
    "AG_VSException",
    "AG_VSWarning",
]
