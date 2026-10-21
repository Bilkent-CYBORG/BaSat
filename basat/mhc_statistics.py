"""Evaluation-only regrets for a sequential finite contextual experiment.

Despite the module name, no MHC-specific objective, context count, or response
scale is built into the calculations. The finite response table is consulted
only here, after observations have been committed, and never passed to a
learner. A known conditioning noise variance does not change recorded rewards.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral
from collections.abc import Mapping
from typing import Any

import torch

from .environments.Contextual import ContextualEnvironment


@dataclass(frozen=True)
class ContextualRegretStatistics:
    """Unsmoothed maximum-reward regret with explicit partial-run coverage.

    Global query indices and within-context query indices are one-based. The
    maximum-simple-regret curve includes a local query number only once every
    declared context has that many observations. It is never the maximum over
    just the tasks seen so far. Initial/warm-start observations are rejected by
    default. With an explicit exclusion policy they condition the learner, but
    do not enter query counts, incumbents, or either regret curve.
    """

    context_order: tuple[tuple[float, ...], ...]
    queries_per_context: int
    query_indices: tuple[int, ...]
    context_indices: tuple[int, ...]
    local_query_indices: tuple[int, ...]
    selected_dataset_indices: tuple[int, ...]
    selected_actions: tuple[tuple[float, ...], ...]
    selected_contexts: tuple[tuple[float, ...], ...]
    selected_rewards: tuple[float, ...]
    reference_values: tuple[float, ...]
    instantaneous_regret: tuple[float, ...]
    cumulative_regret: tuple[float, ...]
    average_regret: tuple[float, ...]
    per_context_query_counts: tuple[int, ...]
    per_context_incumbent_values: tuple[tuple[float, ...], ...]
    per_context_simple_regret: tuple[tuple[float, ...], ...]
    aligned_local_queries: tuple[int, ...]
    maximum_simple_regret: tuple[float, ...]
    completed_contexts: int
    pending_queries: int
    total_contexts: int
    observed_queries: int
    expected_queries: int
    initial_excluded_count: int = 0

    @property
    def complete(self) -> bool:
        return self.observed_queries == self.expected_queries

    @property
    def completion_fraction(self) -> float:
        return self.observed_queries / self.expected_queries

    @property
    def running_average_regret(self) -> tuple[float, ...]:
        """Alias emphasizing that this is a global, not per-context, average."""
        return self.average_regret

    def as_dict(self) -> dict[str, Any]:
        """Return owned, finite JSON-safe arrays plus metric definitions."""
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
            maximize=True,
            instantaneous_definition="context optimum minus observed noiseless reward",
            average_definition="cumulative instantaneous regret divided by global query index",
            maximum_definition="maximum across all declared contexts of optimum minus local incumbent",
            reference="complete finite dataset, evaluation-only; not learner feedback",
            initial_history_included=False,
        )
        return result

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def _matches(value: Any, expected: torch.Tensor, *, name: str) -> None:
    try:
        actual = torch.as_tensor(value).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} does not match the recorded dataset.") from error
    expected = expected.detach().cpu()
    if (actual.shape != expected.shape or actual.is_complex() or actual.dtype == torch.bool
            or not bool(torch.isfinite(actual).all())
            or not torch.equal(actual.to(dtype=expected.dtype), expected)):
        raise ValueError(f"{name} does not match the recorded dataset.")


def contextual_regret_statistics(
    environment: ContextualEnvironment,
    *,
    context_order: Any,
    queries_per_context: int = 50,
    exclude_initial: bool = False,
) -> ContextualRegretStatistics:
    """Evaluate noiseless, dataset-backed sequential context blocks.

    ``context_order`` is a nonempty, distinct ``(number_of_contexts,
    context_dimension)`` array. Exactly ``queries_per_context`` consecutive
    scalar queries are planned for each row in that order. Incomplete histories
    are supported, but missing task curves are not invented. Every committed
    observation must match its finite dataset row, action, and context. The
    optional recorded ``stored_observation`` audit value is checked as well.

    Nonempty ``previous_data`` is rejected unless ``exclude_initial=True``.
    With this explicit opt-in, warm-start data remains available to the learner
    but is excluded from all metric histories and query counts, including each
    task's incumbent. Reference optima still use only the searchable finite
    dataset, not the warm-start rewards. ``initial_excluded_count`` records the
    number of omitted warm-start observations. An online repeat of a warm-start
    action remains an ordinary counted query.

    The returned instantaneous regret is ``f_star[context] - reward``; its
    cumulative sum and global running average retain all observed queries.
    Per-task simple regret is instead ``f_star - running_max(rewards)``.
    ``maximum_simple_regret[n-1]`` is the maximum of those simple regrets over
    *all* declared tasks at local query ``n``. It is not cumulative regret and
    does not average the task gaps. No objective, acquisition, or random draw
    is executed by this helper.
    """
    if not isinstance(environment, ContextualEnvironment):
        raise TypeError("environment must be a ContextualEnvironment.")
    if not isinstance(exclude_initial, bool):
        raise TypeError("exclude_initial must be a boolean.")
    dataset = environment.dataset
    if dataset is None or dataset.contexts is None:
        raise ValueError("contextual regret requires a finite contextual dataset.")
    per_context = _integer(queries_per_context, name="queries_per_context")
    if per_context <= 0:
        raise ValueError("queries_per_context must be positive.")
    try:
        order = torch.as_tensor(context_order).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("context_order must be a finite two-dimensional numeric array.") from error
    if (order.ndim != 2 or not len(order) or order.shape[1] != dataset.contexts.shape[1]
            or order.is_complex() or order.dtype == torch.bool
            or not bool(torch.isfinite(order).all())):
        raise ValueError("context_order must be a finite two-dimensional numeric array.")
    # Reconstruct from the original input, not the default-float32 validation
    # tensor: decimal-valued contexts must not silently lose float64 precision.
    order = torch.as_tensor(context_order, dtype=dataset.contexts.dtype).detach().cpu()
    if len(torch.unique(order, dim=0)) != len(order):
        raise ValueError("context_order must contain distinct contexts.")
    initial = environment.history.initial_data
    initial_count = 0 if initial is None else initial.number_of_observations
    if initial_count and not exclude_initial:
        raise ValueError("contextual regret requires all observations in the declared query history, not warm-start data.")

    table_contexts = dataset.contexts.detach().cpu()
    table_rewards = dataset.observations.detach().cpu().reshape(-1)
    references: list[float] = []
    for context in order:
        matching = torch.all(table_contexts == context, dim=1)
        if not bool(matching.any()):
            raise ValueError("Every context_order row must occur in the finite dataset.")
        references.append(float(table_rewards[matching].max()))
    records = environment.records
    observed = len(records)
    expected = len(order) * per_context
    if _integer(environment.round, name="environment.round") != observed or observed > expected:
        raise ValueError("Environment round and sequential record count are inconsistent with the query budget.")

    contexts: list[int] = []
    local_indices: list[int] = []
    dataset_indices: list[int] = []
    actions: list[tuple[float, ...]] = []
    selected_contexts: list[tuple[float, ...]] = []
    rewards: list[float] = []
    instantaneous: list[float] = []
    cumulative: list[float] = []
    average: list[float] = []
    incumbents: list[list[float]] = [[] for _ in order]
    simple: list[list[float]] = [[] for _ in order]
    running_total = 0.0
    for query, record in enumerate(records):
        context_index, local = divmod(query, per_context)
        if _integer(record.round_index, name="record.round_index") != query:
            raise ValueError("History contains missing or out-of-order scalar queries.")
        row = _integer(record.dataset_index, name="record.dataset_index")
        if not 0 <= row < len(table_rewards):
            raise ValueError("History contains an out-of-range dataset_index.")
        _matches(table_contexts[row], order[context_index], name="Sequential context")
        _matches(record.context, table_contexts[row], name="Recorded context")
        if "context" not in record.round_info:
            raise ValueError("History lacks a revealed pre-query context.")
        _matches(record.round_info["context"], table_contexts[row], name="Revealed context")
        _matches(record.action, dataset.actions[row], name="Requested action")
        _matches(record.evaluated_action, dataset.actions[row], name="Evaluated action")
        _matches(record.observation, table_rewards[row], name="Noiseless observation")
        if "stored_observation" in record.audit:
            _matches(record.audit["stored_observation"], table_rewards[row], name="Stored audit reward")

        reward = float(record.observation)
        regret = references[context_index] - reward
        running_total += regret
        best = max(reward, incumbents[context_index][-1]) if local else reward
        incumbents[context_index].append(best)
        simple[context_index].append(references[context_index] - best)
        contexts.append(context_index)
        local_indices.append(local + 1)
        dataset_indices.append(row)
        actions.append(tuple(float(v) for v in record.evaluated_action))
        selected_contexts.append(tuple(float(v) for v in record.context))
        rewards.append(reward)
        instantaneous.append(regret)
        cumulative.append(running_total)
        average.append(running_total / (query + 1))

    counts = tuple(len(values) for values in incumbents)
    common = min(counts)
    return ContextualRegretStatistics(
        context_order=tuple(tuple(float(v) for v in row) for row in order),
        queries_per_context=per_context, query_indices=tuple(range(1, observed + 1)),
        context_indices=tuple(contexts), local_query_indices=tuple(local_indices),
        selected_dataset_indices=tuple(dataset_indices), selected_actions=tuple(actions),
        selected_contexts=tuple(selected_contexts), selected_rewards=tuple(rewards),
        reference_values=tuple(references), instantaneous_regret=tuple(instantaneous),
        cumulative_regret=tuple(cumulative), average_regret=tuple(average),
        per_context_query_counts=counts,
        per_context_incumbent_values=tuple(tuple(values) for values in incumbents),
        per_context_simple_regret=tuple(tuple(values) for values in simple),
        aligned_local_queries=tuple(range(1, common + 1)),
        maximum_simple_regret=tuple(max(values[n] for values in simple) for n in range(common)),
        completed_contexts=sum(count == per_context for count in counts),
        pending_queries=observed % per_context, total_contexts=len(order),
        observed_queries=observed, expected_queries=expected,
        initial_excluded_count=initial_count,
    )


def compare_contextual_regret_statistics(
    statistics: ContextualRegretStatistics,
    reference: Mapping[str, Any],
    *,
    metric: str = "maximum_simple_regret",
) -> dict[str, Any]:
    """Compare explicitly aligned queries with a digitized contextual curve.

    ``reference['curves'][metric]`` must contain equally sized ``query_indices``
    and ``values`` arrays, plus ``extraction_absolute_uncertainty`` (a scalar or
    pointwise array). Query indices must be strictly increasing positive
    integers. This function performs no interpolation, smoothing, seed
    selection, or scientific pass/fail classification. Extraction uncertainty
    describes the graphical reference, not uncertainty in the experiment.
    """
    if not isinstance(statistics, ContextualRegretStatistics):
        raise TypeError("statistics must be ContextualRegretStatistics.")
    if metric not in ("maximum_simple_regret", "average_regret"):
        raise ValueError("metric must be 'maximum_simple_regret' or 'average_regret'.")
    if not isinstance(reference, Mapping):
        raise TypeError("reference must be a mapping containing digitized curves.")
    try:
        curve = reference["curves"][metric]
        reference_indices = tuple(_integer(value, name="reference query index")
                                  for value in curve["query_indices"])
        raw_values = curve["values"]
        values = tuple(float(value) for value in raw_values)
        raw_uncertainty = curve["extraction_absolute_uncertainty"]
        if isinstance(raw_uncertainty, (tuple, list)):
            uncertainty = tuple(float(value) for value in raw_uncertainty)
        else:
            uncertainty = (float(raw_uncertainty),) * len(values)
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("Reference curve lacks valid indices, values, or extraction uncertainty.") from error
    if (not reference_indices or len(reference_indices) != len(values)
            or len(uncertainty) != len(values)
            or any(index < 1 for index in reference_indices)
            or any(left >= right for left, right in zip(reference_indices, reference_indices[1:]))
            or any(not math.isfinite(value) for value in values)
            or any(not math.isfinite(value) or value < 0 for value in uncertainty)):
        raise ValueError("Reference curve indices/values/uncertainty are invalid or inconsistent.")
    if metric == "maximum_simple_regret":
        measured = dict(zip(statistics.aligned_local_queries, statistics.maximum_simple_regret))
        planned_endpoint = statistics.queries_per_context
    else:
        measured = dict(zip(statistics.query_indices, statistics.average_regret))
        planned_endpoint = statistics.expected_queries
    if reference_indices[-1] > planned_endpoint:
        raise ValueError("Reference curve exceeds the declared experiment query budget.")
    points = []
    for index, value, error in zip(reference_indices, values, uncertainty):
        if index not in measured:
            continue
        gap = measured[index] - value
        points.append({
            "query_index": index, "measured": measured[index], "reference": value,
            "signed_difference": gap, "absolute_difference": abs(gap),
            "extraction_absolute_uncertainty": error,
            "absolute_difference_lower_bound": max(0.0, abs(gap) - error),
            "absolute_difference_upper_bound": abs(gap) + error,
        })
    endpoint = points[-1] if points and points[-1]["query_index"] == reference_indices[-1] else None
    return {
        "schema": "basat-contextual-regret-comparison-v1", "metric": metric,
        "experiment_complete": statistics.complete,
        "observed_queries": statistics.observed_queries,
        "expected_queries": statistics.expected_queries,
        "reference_point_count": len(reference_indices), "compared_point_count": len(points),
        "all_reference_points_available": len(points) == len(reference_indices),
        "pointwise": points, "reference_endpoint_query": reference_indices[-1],
        "endpoint": None if endpoint is None else dict(endpoint),
        "sampled_mean_absolute_difference": (
            sum(point["absolute_difference"] for point in points) / len(points) if points else None),
        "sampled_maximum_absolute_difference": (
            max(point["absolute_difference"] for point in points) if points else None),
        "reference_schema": reference.get("schema"),
        "reference_source": dict(reference.get("source", {})),
        "interpretation": "Descriptive comparison to a graphical extraction; no numerical acceptance threshold is inferred.",
    }
