"""Public stopping-rule interfaces and built-in limits."""

from .stopping_rule import (
    AllStoppingRule,
    AnyStoppingRule,
    CompositeStoppingRule,
    MaxEvaluations,
    MaximumEvaluationsStoppingRule,
    MaximumRoundsStoppingRule,
    MaximumTimeStoppingRule,
    MaxRounds,
    MaxTime,
    StoppingContext,
    StoppingDecision,
    StoppingRule,
    StoppingRuleError,
    all_of,
    any_of,
)

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
