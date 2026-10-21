"""First-hit success curves for grouped, independently repeated experiments."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from .warnings import BasatExperimentWarning
from .workflow import to_list


@dataclass(frozen=True)
class SuccessRun:
    """One planned run, including unresolved or failed runs.

    ``raw_first_hit`` is the one-based first satisfying observation, or None.
    ``first_hit`` charges the whole initialization batch before declaring
    success. A None hit is censored only when ``resolved`` is true; otherwise
    the outcome is unknown, not a failed search.
    """

    algorithm: str
    group_index: int
    repetition: int
    status: str
    evaluations: int
    initial_observation_count: int
    raw_first_hit: int | None
    first_hit: int | None
    resolved: bool
    failure: Any = None


@dataclass(frozen=True)
class SuccessCurve:
    """Population statistics across group fractions, not across individual runs."""

    group_success_fractions: tuple[tuple[float, ...], ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    scientific_endpoint: float
    displayed_endpoint: float
    exact_budget_successes: int


@dataclass(frozen=True)
class SuccessComparison:
    """Descriptive differences from a reference, never a reproduction verdict.

    Extraction tolerance describes the precision of a digitized reference;
    it is not a statistical confidence interval or a hypothesis test.
    """

    methods: dict[str, dict[str, Any]]
    queries: tuple[int, ...]
    complete: bool
    extraction_tolerance: float
    error_scale: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe comparison, retaining the partial-data qualification."""
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class SuccessStatistics:
    """Absorbing success curves with explicit planned-run coverage.

    ``methods`` contains curves calculated on ``included_groups``. Every
    planned run, including missing/failed runs, remains in ``records``.
    Scientific endpoints always count hits at the budget; the opt-in source
    convention changes only the displayed curves and displayed endpoint.
    """

    methods: dict[str, SuccessCurve]
    queries: tuple[int, ...]
    records: tuple[SuccessRun, ...]
    group_count: int
    repetitions: int
    included_groups: tuple[int, ...]
    excluded_groups: tuple[int, ...]
    max_evaluations: int
    threshold: float
    maximize: bool
    source_exclude_exact_budget: bool

    @property
    def complete(self) -> bool:
        """Whether every planned run has a valid completed outcome."""
        return not self.excluded_groups

    @property
    def planned_runs(self) -> int:
        """Number of requested runs, including absent records."""
        return self.group_count * self.repetitions * len(self.methods)

    @property
    def resolved_runs(self) -> int:
        """Number of completed, scientifically resolved run records."""
        return sum(record.resolved for record in self.records)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe curves, raw first hits, and full coverage metadata."""
        result = _json_safe(asdict(self))
        result.update(complete=self.complete, planned_runs=self.planned_runs,
                      resolved_runs=self.resolved_runs,
                      uncertainty="population SD across group success fractions (ddof=0)")
        return result

    def compare(
        self, reference: Mapping[str, Mapping[str, Any]], *,
        extraction_tolerance: float = 0.02, error_scale: float = 0.5,
    ) -> SuccessComparison:
        """Compare against reference means and optional error-bar half widths.

        Each label maps to ``{'mean': [...], 'error': [...]}``; ``error`` is
        optional. Optional ``queries`` must equal this object's query grid.
        Return signed point differences, maximum absolute difference, RMSE,
        and optional error-width differences. The tolerance flag is only a
        descriptive digitization check, including when the data are partial.
        """
        if (not math.isfinite(extraction_tolerance) or extraction_tolerance < 0
                or not math.isfinite(error_scale) or error_scale < 0):
            raise ValueError("Comparison tolerance and error_scale must be finite and nonnegative.")
        if set(reference) != set(self.methods):
            raise ValueError("Reference labels must exactly match the success curves.")
        output = {}
        for label, curve in self.methods.items():
            entry = reference[label]
            if "queries" in entry and tuple(to_list(entry["queries"])) != self.queries:
                raise ValueError(f"Reference query grid differs for {label!r}.")
            mean = _vector(entry["mean"], len(self.queries), "reference mean")
            if np.any((mean < 0) | (mean > 1)):
                raise ValueError("Reference success fractions must lie in [0, 1].")
            difference = np.asarray(curve.mean) - mean
            item = {"differences": difference.tolist(),
                    "max_abs_difference": float(np.abs(difference).max()),
                    "rmse": float(np.sqrt(np.mean(difference**2))),
                    "within_extraction_tolerance": bool(np.all(np.abs(difference) <= extraction_tolerance))}
            if entry.get("error") is not None:
                error = _vector(entry["error"], len(self.queries), "reference error")
                if np.any(error < 0):
                    raise ValueError("Reference error half widths must be nonnegative.")
                widths = np.asarray(curve.std) * error_scale
                error_difference = widths - error
                item["error_widths"] = {"actual": widths.tolist(), "reference": error.tolist(),
                    "differences": error_difference.tolist(),
                    "max_abs_difference": float(np.abs(error_difference).max()),
                    "rmse": float(np.sqrt(np.mean(error_difference**2)))}
            output[label] = item
        return SuccessComparison(output, self.queries, self.complete,
                                 extraction_tolerance, error_scale)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _positive_integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    return value


def _vector(value: Any, count: int | None, name: str) -> np.ndarray:
    array = np.asarray(to_list(value), dtype=float)
    if array.ndim != 1 or (count is not None and len(array) != count) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one-dimensional vector of the expected length.")
    return array


def success_statistics(
    data: Any, *, threshold: float, max_evaluations: int,
    initial_points: int | None = None, group_count: int | None = None,
    repetitions: int | None = None, algorithms: Sequence[str] | None = None,
    query_grid: Any = None, maximize: bool = True,
    source_exclude_exact_budget: bool = False,
    incomplete: Literal["error", "complete_groups"] = "error",
) -> SuccessStatistics:
    """Calculate first-hit success curves from grouped experiment records.

    Parameters
    ----------
    data : grouped experiment result or sequence of mappings
        Run records contain ``algorithm``, zero-based ``group_index`` and
        ``repetition``, full ``observations`` (including initialization),
        ``initial_observation_count`` and ``status``. A completed run either
        reaches the threshold or consumes the full query budget. Plain
        record sequences require explicit group_count, repetitions and labels.
    threshold : float
        Reward threshold, inclusive. ``maximize=False`` reverses the test.
    max_evaluations : int
        Total observations allowed per run, including initialization.
    query_grid : array-like, optional
        Increasing integer query counts within [0, max_evaluations]; by
        default all counts are returned. Success is absorbing after first hit.
    incomplete : {'error', 'complete_groups'}
        The default refuses partial/failed cohorts. Explicit complete_groups
        uses only groups complete across every method, emits a warning and
        preserves all excluded records in the result. No missing run is
        silently converted to a failed search or removed from a denominator.
    source_exclude_exact_budget : bool
        Opt-in display convention excluding hits whose charged cost equals
        the budget. Raw first hits and scientific endpoints remain unchanged.

    Examples
    --------
    >>> records = [dict(algorithm='A', group_index=0, repetition=0,
    ...     observations=[0., 1., 0.], initial_observation_count=3,
    ...     status='completed')]
    >>> result = success_statistics(records, threshold=.5, max_evaluations=5,
    ...     group_count=1, repetitions=1, algorithms=['A'])
    >>> result.records[0].raw_first_hit, result.records[0].first_hit
    (2, 3)
    """
    budget = _positive_integer(max_evaluations, "max_evaluations")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    if not isinstance(maximize, bool) or not isinstance(source_exclude_exact_budget, bool):
        raise ValueError("maximize and source_exclude_exact_budget must be Boolean.")
    if incomplete not in {"error", "complete_groups"}:
        raise ValueError("incomplete must be 'error' or 'complete_groups'.")
    if initial_points is not None:
        _positive_integer(initial_points, "initial_points", minimum=0)
        if initial_points > budget:
            raise ValueError("initial_points exceeds the total budget.")
    groups = _positive_integer(group_count if group_count is not None else getattr(data, "group_count", None), "group_count")
    repeats = _positive_integer(repetitions if repetitions is not None else getattr(data, "repetitions", None), "repetitions")
    label_values = algorithms if algorithms is not None else getattr(data, "algorithm_names", None)
    if (label_values is None or isinstance(label_values, str)
            or not all(isinstance(label, str) and label for label in label_values)):
        raise ValueError("algorithms must contain the planned nonempty method labels.")
    labels = tuple(label_values)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("algorithms must be nonempty and contain no duplicate labels.")
    grid = np.arange(budget+1) if query_grid is None else _vector(query_grid, None, "query_grid")
    if (not len(grid) or np.any(grid != np.floor(grid)) or grid[0] < 0
            or grid[-1] > budget or np.any(np.diff(grid) <= 0)):
        raise ValueError("query_grid must contain increasing integers within the query budget.")
    indexed = {}
    for row in getattr(data, "records", data):
        if not isinstance(row, Mapping):
            raise ValueError("Each success record must be a mapping.")
        group = _positive_integer(row.get("group_index"), "group_index", minimum=0)
        repeat = _positive_integer(row.get("repetition"), "repetition", minimum=0)
        label = row.get("algorithm")
        if not isinstance(label, str) or group >= groups or repeat >= repeats or label not in labels:
            raise ValueError("Run record lies outside the declared experiment plan.")
        key = (label, group, repeat)
        if key in indexed:
            raise ValueError("Duplicate method/group/repetition record.")
        observations = _vector(row.get("observations", []), None, "observations")
        count = len(observations)
        if count > budget:
            raise ValueError("Observed query count exceeds max_evaluations.")
        if "evaluations" in row and _positive_integer(row["evaluations"], "evaluations", minimum=0) != count:
            raise ValueError("Recorded evaluation count differs from observations.")
        initial = _positive_integer(row.get("initial_observation_count", initial_points or 0), "initial_observation_count", minimum=0)
        if initial > budget or (initial_points is not None and initial != initial_points):
            raise ValueError("Initialization count differs from the declared protocol.")
        status = row.get("status")
        if status not in {"completed", "failed", "running", "pending", "incomplete", "stopped"}:
            raise ValueError("Run status must explicitly identify completed, failed, or incomplete work.")
        hit_indices = np.flatnonzero(observations >= threshold if maximize else observations <= threshold)
        raw = int(hit_indices[0])+1 if len(hit_indices) else None
        hit = max(initial, raw) if raw is not None else None
        resolved = (status == "completed" and not row.get("failure") and count >= initial
                    and (hit is not None or count == budget))
        indexed[key] = SuccessRun(label, group, repeat, status, count, initial,
                                  raw, hit, resolved, _json_safe(row.get("failure")))
    records = tuple(indexed.get((label, group, repeat),
        SuccessRun(label, group, repeat, "missing", 0, initial_points or 0, None, None, False))
        for label in labels for group in range(groups) for repeat in range(repeats))
    excluded = tuple(group for group in range(groups) if any(
        not record.resolved for record in records if record.group_index == group))
    included = tuple(group for group in range(groups) if group not in excluded)
    if excluded and incomplete == "error":
        raise ValueError(f"Success cohort has unresolved or failed runs in groups {excluded}; "
                         "use incomplete='complete_groups' explicitly for a partial summary.")
    if not included:
        raise ValueError("No completely covered group is available for success statistics.")
    if excluded:
        warnings.warn(f"Basat partial success statistics use {len(included)}/{groups} complete "
                      f"groups; excluded groups {excluded}. All planned records remain in coverage.",
                      BasatExperimentWarning, stacklevel=2)
    curves = {}
    for label in labels:
        selected = [record for record in records if record.algorithm == label and record.group_index in included]
        costs = np.asarray([budget+1 if record.first_hit is None else record.first_hit
                            for record in selected]).reshape(len(included), repeats)
        displayed = costs.copy()
        if source_exclude_exact_budget:
            displayed[displayed == budget] = budget+1
        fractions = (displayed[:, :, None] <= grid[None, None, :]).mean(axis=1)
        curves[label] = SuccessCurve(
            tuple(tuple(float(value) for value in row) for row in fractions),
            tuple(float(value) for value in fractions.mean(axis=0)),
            tuple(float(value) for value in fractions.std(axis=0, ddof=0)),
            float((costs <= budget).mean()), float((displayed <= budget).mean()),
            int((costs == budget).sum()))
    return SuccessStatistics(curves, tuple(int(value) for value in grid), records,
        groups, repeats, included, excluded, budget, float(threshold), maximize,
        source_exclude_exact_budget)


__all__ = ["SuccessComparison", "SuccessCurve", "SuccessRun", "SuccessStatistics", "success_statistics"]
