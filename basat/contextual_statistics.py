"""Evaluation-only selection errors for chronological finite context tables.

The full response table is consulted here to compute an evaluation reference,
never by a learner or an acquisition function. Repeated context coordinates
are deliberately not used as grouping keys: different days are different
observational rows even when they have the same time-of-day context.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any

import torch

from .environments.SequentialContextualTable import SequentialContextualTableEnvironment


@dataclass(frozen=True)
class ContextualSelectionStatistics:
    """Best-of-query-set errors, without smoothing or calibration observations.

    All series contain only completed chronological rows. ``pending_queries``
    reports a final incomplete row, which is validated but is not assigned an
    error. ``elapsed_hours`` uses right interval endpoints; it does not stretch
    the elapsed time to match a reference figure. Selected readings and
    references are in the original, uncentered response units.
    """

    selection: str
    maximize: bool
    interval_minutes: float
    queries_per_context: int
    context_rows: tuple[int, ...]
    elapsed_hours: tuple[float, ...]
    selected_indices: tuple[tuple[int, ...], ...]
    selected_readings: tuple[tuple[float, ...], ...]
    reference_values: tuple[float, ...]
    selected_values: tuple[float, ...]
    instantaneous_errors: tuple[float, ...]
    cumulative_mean_errors: tuple[float, ...]
    completed_contexts: int
    pending_queries: int
    total_contexts: int
    observed_queries: int
    expected_queries: int

    @property
    def complete(self) -> bool:
        """Whether all planned contexts have their full query set."""
        return self.observed_queries == self.expected_queries

    @property
    def completion_fraction(self) -> float:
        """Fraction of planned queries present, including a partial last row."""
        return self.observed_queries / self.expected_queries

    def as_dict(self) -> dict[str, Any]:
        """Return owned JSON-safe data for :class:`basat.ArtifactStore`."""
        def lists(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: lists(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return [lists(item) for item in value]
            return value

        result = lists(asdict(self))
        result.update(
            complete=self.complete,
            completion_fraction=self.completion_fraction,
            aggregation="arithmetic cumulative mean over completed chronological contexts",
            reference="best uncentered response across all actions in the same table row",
        )
        return result

    def to_dict(self) -> dict[str, Any]:
        """Alias for :meth:`as_dict`, consistent with other Basat statistics."""
        return self.as_dict()


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def _matches(value: Any, expected: torch.Tensor, *, name: str) -> None:
    try:
        actual = torch.as_tensor(value).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must match the recorded response table.") from error
    expected = expected.detach().cpu()
    if (actual.shape != expected.shape or actual.is_complex() or actual.dtype == torch.bool
            or not bool(torch.isfinite(actual).all())
            or not torch.equal(actual.to(dtype=expected.dtype), expected)):
        raise ValueError(f"{name} does not match the recorded response table.")


def contextual_selection_statistics(
    environment: SequentialContextualTableEnvironment,
    *,
    selection: str = "best",
    maximize: bool = True,
    interval_minutes: float = 5,
) -> ContextualSelectionStatistics:
    """Evaluate a chronological finite-context selection experiment.

    Parameters
    ----------
    environment : SequentialContextualTableEnvironment
        Environment owning the hidden response table and committed records.
        This function is evaluation-only and neither calls the objective nor
        supplies additional responses to the learner.
    selection : {"best"}
        Compare the best selected response with the best response among all
        actions in that row. For maximum-temperature selection this is the
        minimum regret among the queried sensors, not the coldest reading.
    maximize : bool
        Maximize responses when true; when false compare selected and global
        minima. Errors have the same response units and are nonnegative.
    interval_minutes : positive finite float
        Physical interval between consecutive context rows. Times are right
        endpoints, beginning at one interval, with no artificial zero point.

    Notes
    -----
    Every record must be a chronological, distinct-within-row table query and
    must agree with the table, context, centered feedback and audit reading.
    Malformed histories are rejected rather than silently sorted, deduplicated
    or assigned zero error. An unfinished final row is reported separately.
    Calling this function does not mutate environment state or random streams.
    """
    if not isinstance(environment, SequentialContextualTableEnvironment):
        raise TypeError("environment must be a SequentialContextualTableEnvironment.")
    if selection != "best":
        raise ValueError("selection must be 'best'.")
    if not isinstance(maximize, bool):
        raise ValueError("maximize must be Boolean.")
    if (isinstance(interval_minutes, bool) or not isinstance(interval_minutes, Real)
            or not math.isfinite(interval_minutes) or interval_minutes <= 0):
        raise ValueError("interval_minutes must be finite and positive.")

    query_count = environment.queries_per_context
    total_contexts = len(environment._contexts)
    records = environment.records
    observed_queries = len(records)
    if (_integer(environment.round, name="environment.round") != observed_queries
            or observed_queries > environment.maximum_rounds):
        raise ValueError("Environment round and chronological record count are inconsistent.")

    indices: list[tuple[int, ...]] = []
    readings: list[tuple[float, ...]] = []
    references: list[float] = []
    selected: list[float] = []
    errors: list[float] = []
    current_indices: list[int] = []
    current_readings: list[float] = []
    # Access private table values only inside this evaluation helper. They are
    # not copied into algorithm state or the environment's public feedback.
    table = environment._observations
    for query, record in enumerate(records):
        row = query // query_count
        try:
            context_row = _integer(record.audit["context_row"], name="context_row")
            index = _integer(record.audit["action_index"], name="action_index")
            stored = record.audit["stored_observation"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Query {query} lacks valid contextual selection audit fields.") from error
        if (_integer(record.round_index, name="record.round_index") != query
                or context_row != row):
            raise ValueError(f"Query {query} has a missing or out-of-order chronological context row.")
        if not 0 <= index < len(environment._actions):
            raise ValueError(f"Query {query} has an out-of-range action index.")
        if index in current_indices:
            raise ValueError(f"Context row {row} contains a duplicate selected action.")

        expected_reading = table[row, index]
        _matches(record.action, environment._actions[index], name="Requested action")
        _matches(record.evaluated_action, environment._actions[index], name="Evaluated action")
        _matches(record.context, environment._contexts[row], name="Context")
        _matches(stored, expected_reading, name="Uncentered audit reading")
        _matches(record.observation, expected_reading - environment.observation_offset,
                 name="Centered feedback")
        if "context" not in record.round_info:
            raise ValueError(f"Query {query} lacks its revealed context.")
        _matches(record.round_info["context"], environment._contexts[row], name="Revealed context")
        current_indices.append(index)
        current_readings.append(float(expected_reading))

        if len(current_indices) == query_count:
            best = max(current_readings) if maximize else min(current_readings)
            reference = float(table[row].max() if maximize else table[row].min())
            error = reference - best if maximize else best - reference
            # No clamping or replacement: matching table values guarantee the
            # selected subset cannot outperform its all-action reference.
            indices.append(tuple(current_indices))
            readings.append(tuple(current_readings))
            references.append(reference)
            selected.append(best)
            errors.append(error)
            current_indices = []
            current_readings = []

    completed = len(errors)
    cumulative = []
    running_total = 0.0
    for number, error in enumerate(errors, start=1):
        running_total += error
        cumulative.append(running_total / number)
    return ContextualSelectionStatistics(
        selection=selection, maximize=maximize, interval_minutes=float(interval_minutes),
        queries_per_context=query_count, context_rows=tuple(range(completed)),
        elapsed_hours=tuple((row + 1) * float(interval_minutes) / 60 for row in range(completed)),
        selected_indices=tuple(indices), selected_readings=tuple(readings),
        reference_values=tuple(references), selected_values=tuple(selected),
        instantaneous_errors=tuple(errors), cumulative_mean_errors=tuple(cumulative),
        completed_contexts=completed, pending_queries=len(current_indices),
        total_contexts=total_contexts, observed_queries=observed_queries,
        expected_queries=environment.maximum_rounds,
    )
