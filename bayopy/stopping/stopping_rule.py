"""Composable stopping-rule contracts for experiment execution."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal


class StoppingRuleError(ValueError):
    """Raised when a stopping rule or its checkpoint state is invalid."""


@dataclass(frozen=True, slots=True, init=False)
class StoppingContext:
    """Learner-coordinate-free information supplied to stopping rules.

    Parameters
    ----------
    round:
        Number of completed environment rounds.
    evaluations:
        Number of completed individual objective evaluations; a q-batch adds q.
    elapsed_time:
        Non-negative wall-clock seconds since experiment execution began.
    history:
        Optional learner-visible environment history.  Rules must not transform
        its action coordinates into model coordinates.
    algorithm_diagnostics, algorithm_result:
        Optional public algorithm reports.
    metrics:
        Additional named scalar or structured experiment metrics.
    """

    completed_rounds: int = 0
    evaluations: int = 0
    elapsed_seconds: float = 0.0
    latest_records: tuple[Any, ...] = ()
    latest_feedback: Mapping[str, Any] = field(default_factory=dict)
    history: Any | None = None
    algorithm_diagnostics: Any | None = None
    algorithm_result: Any | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        completed_rounds: int | None = None,
        round: int | None = None,
        evaluations: int = 0,
        elapsed_seconds: float | None = None,
        elapsed_time: float | None = None,
        latest_records: Sequence[Any] = (),
        latest_feedback: Mapping[str, Any] | None = None,
        history: Any | None = None,
        algorithm_diagnostics: Any | None = None,
        algorithm_result: Any | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize using descriptive names or the compact public aliases."""

        if completed_rounds is not None and round is not None and completed_rounds != round:
            raise StoppingRuleError("completed_rounds and round cannot specify different values.")
        resolved_rounds = (
            0
            if completed_rounds is None and round is None
            else (completed_rounds if completed_rounds is not None else round)
        )
        if (
            elapsed_seconds is not None
            and elapsed_time is not None
            and elapsed_seconds != elapsed_time
        ):
            raise StoppingRuleError(
                "elapsed_seconds and elapsed_time cannot specify different values."
            )
        resolved_elapsed = (
            0.0
            if elapsed_seconds is None and elapsed_time is None
            else (elapsed_seconds if elapsed_seconds is not None else elapsed_time)
        )
        object.__setattr__(self, "completed_rounds", resolved_rounds)
        object.__setattr__(self, "evaluations", evaluations)
        object.__setattr__(self, "elapsed_seconds", resolved_elapsed)
        object.__setattr__(self, "latest_records", tuple(latest_records))
        object.__setattr__(
            self, "latest_feedback", {} if latest_feedback is None else latest_feedback
        )
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "algorithm_diagnostics", algorithm_diagnostics)
        object.__setattr__(self, "algorithm_result", algorithm_result)
        object.__setattr__(self, "metrics", {} if metrics is None else metrics)
        self._validate()

    def _validate(self) -> None:
        """Validate counters, elapsed time, and public mapping fields."""

        for name, value in (
            ("completed_rounds", self.completed_rounds),
            ("evaluations", self.evaluations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise StoppingRuleError(f"{name} must be a non-negative integer.")
        if not math.isfinite(float(self.elapsed_seconds)) or self.elapsed_seconds < 0:
            raise StoppingRuleError("elapsed_seconds must be finite and non-negative.")
        if not isinstance(self.latest_feedback, Mapping):
            raise StoppingRuleError("latest_feedback must be a mapping.")
        if not isinstance(self.metrics, Mapping):
            raise StoppingRuleError("metrics must be a mapping.")

    @property
    def round(self) -> int:
        """Compact alias for the number of completed rounds."""

        return self.completed_rounds

    @property
    def completed_evaluations(self) -> int:
        """Alias for the number of completed individual evaluations."""

        return self.evaluations

    @property
    def elapsed_time(self) -> float:
        """Compact alias for elapsed wall-clock seconds."""

        return self.elapsed_seconds


@dataclass(frozen=True, slots=True)
class StoppingDecision:
    """Structured result returned by every stopping rule.

    Parameters
    ----------
    stop:
        Whether experiment execution should terminate.
    reason:
        Stable non-empty reason when ``stop`` is true.
    details:
        Learner-safe rule metrics and child decisions.
    """

    stop: bool
    reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject malformed decisions before they reach experiment control flow."""

        if not isinstance(self.stop, bool):
            raise StoppingRuleError("StoppingDecision.stop must be bool.")
        if self.stop and (self.reason is None or not self.reason.strip()):
            raise StoppingRuleError("A stopping decision requires a non-empty reason.")
        if not isinstance(self.details, Mapping):
            raise StoppingRuleError("StoppingDecision.details must be a mapping.")

    def __bool__(self) -> bool:
        """Allow natural ``if decision`` checks without losing its reason."""

        return self.stop


class StoppingRule(ABC):
    """Abstract interface for serializable stopping policies."""

    @abstractmethod
    def evaluate(self, context: StoppingContext) -> StoppingDecision:
        """Return whether execution should stop for ``context``.

        Parameters
        ----------
        context:
            Public experiment state at a completed round boundary.

        Returns
        -------
        StoppingDecision
            Structured stop flag, reason, and rule-specific metrics.
        """

    def should_stop(self, context: StoppingContext) -> StoppingDecision:
        """Evaluate the rule through the validated experiment-facing API.

        Parameters
        ----------
        context:
            Public experiment state at a completed round boundary.

        Returns
        -------
        StoppingDecision
            The rule's structured decision.

        Raises
        ------
        StoppingRuleError
            If ``context`` is not a :class:`StoppingContext`.
        """

        if not isinstance(context, StoppingContext):
            raise StoppingRuleError("context must be a StoppingContext.")
        return self.evaluate(context)

    def __call__(self, context: StoppingContext) -> StoppingDecision:
        """Delegate to :meth:`should_stop`."""

        return self.should_stop(context)

    def reset(self) -> None:
        """Reset mutable rule state; built-in threshold rules are stateless."""

        return None

    def state_dict(self) -> dict[str, Any]:
        """Return versioned rule state suitable for checkpointing."""

        return {
            "schema_version": 1,
            "rule_type": f"{type(self).__module__}.{type(self).__qualname__}",
            "configuration": self._configuration_state(),
            "mutable_state": self._mutable_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any], *, strict: bool = True) -> None:
        """Restore state after mandatory identity/configuration checks.

        ``strict=False`` is reserved for forward-compatible optional fields; it
        never permits a different rule type or threshold configuration.

        Parameters
        ----------
        state:
            Versioned mapping returned by :meth:`state_dict`.
        strict:
            Whether subclass loaders reject unknown optional mutable fields.

        Raises
        ------
        StoppingRuleError
            If the schema, rule identity, configuration, or mutable state is
            incompatible with this rule instance.
        """

        if not isinstance(state, Mapping):
            raise StoppingRuleError("stopping-rule state must be a mapping.")
        if state.get("schema_version") != 1:
            raise StoppingRuleError("Unsupported stopping-rule state schema.")
        expected_type = f"{type(self).__module__}.{type(self).__qualname__}"
        if state.get("rule_type") != expected_type:
            raise StoppingRuleError("Checkpoint stopping-rule type does not match.")
        if state.get("configuration") != self._configuration_state():
            raise StoppingRuleError("Checkpoint stopping-rule configuration does not match.")
        self._load_mutable_state(state.get("mutable_state", {}), strict=strict)

    def _configuration_state(self) -> Mapping[str, Any]:
        """Return immutable fields used to reject incompatible checkpoints."""

        return {}

    def _mutable_state(self) -> Mapping[str, Any]:
        """Return state that may change while the experiment is running."""

        return {}

    def _load_mutable_state(self, state: Any, *, strict: bool) -> None:
        """Validate and restore subclass-specific mutable checkpoint state."""

        if not isinstance(state, Mapping):
            raise StoppingRuleError("mutable stopping-rule state must be a mapping.")


def _validate_limit(value: int, name: str) -> int:
    """Validate a non-negative integral stopping threshold."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StoppingRuleError(f"{name} must be a non-negative integer.")
    return value


class MaximumRoundsStoppingRule(StoppingRule):
    """Stop once the configured number of rounds has completed.

    Examples
    --------
    >>> import bayopy
    >>> rule = bayopy.MaximumRoundsStoppingRule(maximum_rounds=150)
    """

    def __init__(self, maximum_rounds: int) -> None:
        """Initialize the rule with a non-negative completed-round limit."""

        self.maximum_rounds = _validate_limit(maximum_rounds, "maximum_rounds")

    def evaluate(self, context: StoppingContext) -> StoppingDecision:
        """Compare completed rounds with the configured limit.

        Parameters
        ----------
        context:
            Public experiment counters and learner-visible state.

        Returns
        -------
        StoppingDecision
            A decision whose details include the current count and limit.
        """

        stop = context.round >= self.maximum_rounds
        return StoppingDecision(
            stop=stop,
            reason="maximum_rounds" if stop else None,
            details={"current": context.round, "limit": self.maximum_rounds},
        )

    def _configuration_state(self) -> Mapping[str, Any]:
        """Serialize the immutable round limit."""

        return {"maximum_rounds": self.maximum_rounds}


class MaximumEvaluationsStoppingRule(StoppingRule):
    """Stop once the configured number of evaluations has completed.

    Examples
    --------
    >>> import bayopy
    >>> rule = bayopy.MaximumEvaluationsStoppingRule(maximum_evaluations=500)
    """

    def __init__(self, maximum_evaluations: int) -> None:
        """Initialize the rule with a non-negative evaluation limit."""

        self.maximum_evaluations = _validate_limit(maximum_evaluations, "maximum_evaluations")

    def evaluate(self, context: StoppingContext) -> StoppingDecision:
        """Compare completed evaluations with the configured limit.

        Parameters
        ----------
        context:
            Public experiment counters and learner-visible state.

        Returns
        -------
        StoppingDecision
            A decision whose details include the current count and limit.
        """

        stop = context.evaluations >= self.maximum_evaluations
        return StoppingDecision(
            stop=stop,
            reason="maximum_evaluations" if stop else None,
            details={
                "current": context.evaluations,
                "limit": self.maximum_evaluations,
            },
        )

    def _configuration_state(self) -> Mapping[str, Any]:
        """Serialize the immutable evaluation limit."""

        return {"maximum_evaluations": self.maximum_evaluations}


class MaximumTimeStoppingRule(StoppingRule):
    """Stop once elapsed wall-clock time reaches a finite threshold.

    Examples
    --------
    >>> import bayopy
    >>> rule = bayopy.MaximumTimeStoppingRule(maximum_seconds=3600.0)
    """

    def __init__(self, maximum_seconds: float) -> None:
        """Initialize the rule with a finite non-negative wall-time limit."""

        try:
            value = float(maximum_seconds)
        except (TypeError, ValueError) as exc:
            raise StoppingRuleError(
                "maximum_seconds must be a finite non-negative number."
            ) from exc
        if not math.isfinite(value) or value < 0:
            raise StoppingRuleError("maximum_seconds must be a finite non-negative number.")
        self.maximum_seconds = value

    def evaluate(self, context: StoppingContext) -> StoppingDecision:
        """Compare elapsed wall time with the configured limit.

        Parameters
        ----------
        context:
            Public experiment counters and learner-visible state.

        Returns
        -------
        StoppingDecision
            A decision whose details include elapsed seconds and the limit.
        """

        stop = context.elapsed_time >= self.maximum_seconds
        return StoppingDecision(
            stop=stop,
            reason="maximum_time" if stop else None,
            details={"current": context.elapsed_time, "limit": self.maximum_seconds},
        )

    def _configuration_state(self) -> Mapping[str, Any]:
        """Serialize the immutable wall-time limit."""

        return {"maximum_seconds": self.maximum_seconds}


class CompositeStoppingRule(StoppingRule):
    """Combine child rules with deterministic ``any`` or ``all`` semantics."""

    def __init__(
        self,
        rules: Sequence[StoppingRule],
        *,
        mode: Literal["any", "all"],
    ) -> None:
        """Initialize and validate an ordered non-empty child-rule collection."""

        if mode not in {"any", "all"}:
            raise StoppingRuleError("Composite mode must be 'any' or 'all'.")
        if not isinstance(rules, Sequence) or len(rules) == 0:
            raise StoppingRuleError("A composite requires at least one child rule.")
        if not all(isinstance(rule, StoppingRule) for rule in rules):
            raise StoppingRuleError("Every composite child must be a StoppingRule.")
        self.rules = tuple(rules)
        self.mode = mode

    def evaluate(self, context: StoppingContext) -> StoppingDecision:
        """Evaluate every child and combine decisions deterministically.

        Parameters
        ----------
        context:
            Public experiment state supplied unchanged to every child rule.

        Returns
        -------
        StoppingDecision
            The ``any`` or ``all`` aggregate, including every child decision.
        """

        decisions = tuple(rule.should_stop(context) for rule in self.rules)
        stop = (
            any(decision.stop for decision in decisions)
            if self.mode == "any"
            else all(decision.stop for decision in decisions)
        )
        reasons = tuple(
            decision.reason
            for decision in decisions
            if decision.stop and decision.reason is not None
        )
        return StoppingDecision(
            stop=stop,
            reason=f"{self.mode}_of:" + ",".join(reasons) if stop else None,
            details={"child_decisions": decisions},
        )

    def reset(self) -> None:
        """Reset each child rule while preserving composite configuration."""

        for rule in self.rules:
            rule.reset()

    def _configuration_state(self) -> Mapping[str, Any]:
        """Serialize the composition mode and child configurations."""

        return {
            "mode": self.mode,
            "children": [rule.state_dict()["configuration"] for rule in self.rules],
            "child_types": [rule.state_dict()["rule_type"] for rule in self.rules],
        }

    def _mutable_state(self) -> Mapping[str, Any]:
        """Serialize complete child states for nested checkpoint recovery."""

        return {"children": [rule.state_dict() for rule in self.rules]}

    def _load_mutable_state(self, state: Any, *, strict: bool) -> None:
        """Restore each child after validating arity and state shape."""

        super()._load_mutable_state(state, strict=strict)
        children = state.get("children")
        if not isinstance(children, list) or len(children) != len(self.rules):
            raise StoppingRuleError("Composite checkpoint child state is invalid.")
        for rule, child_state in zip(self.rules, children, strict=True):
            rule.load_state_dict(child_state, strict=strict)


class AnyStoppingRule(CompositeStoppingRule):
    """Stop when at least one child rule stops."""

    def __init__(self, rules: Sequence[StoppingRule]) -> None:
        """Construct an ``any`` composite from ``rules``."""

        super().__init__(rules, mode="any")


class AllStoppingRule(CompositeStoppingRule):
    """Stop only when every child rule stops."""

    def __init__(self, rules: Sequence[StoppingRule]) -> None:
        """Construct an ``all`` composite from ``rules``."""

        super().__init__(rules, mode="all")


def any_of(*rules: StoppingRule) -> AnyStoppingRule:
    """Construct an :class:`AnyStoppingRule` from positional children.

    Examples
    --------
    >>> import bayopy
    >>> rule = bayopy.any_of(
    ...     bayopy.MaximumRoundsStoppingRule(150),
    ...     bayopy.MaximumTimeStoppingRule(3600.0),
    ... )
    """

    return AnyStoppingRule(rules)


def all_of(*rules: StoppingRule) -> AllStoppingRule:
    """Construct an :class:`AllStoppingRule` from positional children.

    Examples
    --------
    >>> import bayopy
    >>> rule = bayopy.all_of(
    ...     bayopy.MaximumRoundsStoppingRule(150),
    ...     bayopy.MaximumEvaluationsStoppingRule(150),
    ... )
    """

    return AllStoppingRule(rules)


# Compact aliases keep experiment configuration readable while the descriptive
# class names remain the canonical public API.
MaxRounds = MaximumRoundsStoppingRule
MaxEvaluations = MaximumEvaluationsStoppingRule
MaxTime = MaximumTimeStoppingRule


__all__ = [
    "AllStoppingRule",
    "AnyStoppingRule",
    "CompositeStoppingRule",
    "MaxEvaluations",
    "MaxRounds",
    "MaxTime",
    "MaximumEvaluationsStoppingRule",
    "MaximumRoundsStoppingRule",
    "MaximumTimeStoppingRule",
    "StoppingContext",
    "StoppingDecision",
    "StoppingRule",
    "StoppingRuleError",
    "all_of",
    "any_of",
]
