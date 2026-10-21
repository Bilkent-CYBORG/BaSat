"""Compare saved scientific statistics with uncertain image-derived references."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from .contextual_statistics import ContextualSelectionStatistics
from .regret import RSRegretStatistics


def _finite(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real number.") from error
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative." if nonnegative else "."))
    return result


def compare_reference_statistics(
    statistics: Mapping[str, RSRegretStatistics | ContextualSelectionStatistics],
    reference: Mapping[str, Any],
    *,
    planned_runs: int | None = None,
) -> dict[str, Any]:
    """Compare measured curves at explicitly digitized reference coordinates.

    Parameters
    ----------
    statistics : mapping of str to RSRegretStatistics or ContextualSelectionStatistics
        Basat population summaries, normally from ``rs_regret_statistics``.
        The ``error`` field is the actual plotted uncertainty half-width.
        Contextual selection statistics compare their cumulative mean errors,
        with one chronological run and no estimated sampling uncertainty.
    reference : mapping
        An image-derived reference with ``kind='image_derived_reference'``,
        ``is_released_numerical_data=False``, and a ``methods`` mapping. Each
        method contains ``points`` with zero-based integer ``source_index``,
        ``mean`` and nonnegative ``mean_extraction_uncertainty``. Optional
        ``sd_over_2`` and ``sd_over_2_extraction_uncertainty`` compare bars.
        A null mean marks an occluded or otherwise unavailable reference;
        no interpolation, extrapolation or substituted curve is invented.
    planned_runs : int, optional
        Declared per-method trial count. A smaller observed count is reported
        as partial; uncertainty never makes an incomplete campaign complete.

    Returns
    -------
    dict
        Detached, JSON-safe pointwise and endpoint differences, observed run
        coverage, extraction intervals and complete reference provenance.
        This is a descriptive comparison, not a hypothesis test or an
        automatic scientific-reproduction acceptance decision.

    Examples
    --------
    >>> from basat.regret import rs_regret_statistics
    >>> measured = rs_regret_statistics([[1., 2.], [2., 3.]], error_scale=.5)
    >>> reference = {"kind": "image_derived_reference",
    ...     "is_released_numerical_data": False,
    ...     "methods": {"A": {"points": [{"source_index": 1, "mean": 2.4,
    ...         "mean_extraction_uncertainty": .1}]}}}
    >>> report = compare_reference_statistics({"A": measured}, reference, planned_runs=2)
    >>> report["methods"]["A"]["endpoint"]["measured_mean"]
    2.5

    Notes
    -----
    Pixel extraction uncertainty and statistical run-to-run uncertainty are
    different quantities. They are retained separately and never combined
    to manufacture an acceptance threshold. Source indices refer to the saved
    curve itself; callers must not shift them to one-based query numbers.
    """

    if not isinstance(statistics, Mapping) or not statistics:
        raise ValueError("statistics must contain at least one measured method.")
    if planned_runs is not None and (isinstance(planned_runs, bool)
                                    or not isinstance(planned_runs, int) or planned_runs < 1):
        raise ValueError("planned_runs must be a positive integer or None.")
    if (not isinstance(reference, Mapping) or reference.get("kind") != "image_derived_reference"
            or reference.get("is_released_numerical_data") is not False
            or not isinstance(reference.get("methods"), Mapping)):
        raise ValueError("reference must explicitly identify image-derived, non-original numerical data.")
    methods = {}
    for label, summary in statistics.items():
        contextual_coverage = None
        if isinstance(summary, ContextualSelectionStatistics):
            from types import SimpleNamespace

            import numpy as np
            if planned_runs not in (None, 1):
                raise ValueError("Contextual statistics describe one chronological run.")
            count = summary.completed_contexts
            contextual_coverage = {
                "complete": summary.complete, "completed_contexts": count,
                "total_contexts": summary.total_contexts,
                "observed_queries": summary.observed_queries,
                "expected_queries": summary.expected_queries,
                "pending_queries": summary.pending_queries,
            }
            summary = SimpleNamespace(
                mean=summary.cumulative_mean_errors, error=np.zeros(count),
                run_count=1, round_count=count, error_scale=0.0,
            )
        elif not isinstance(summary, RSRegretStatistics):
            raise TypeError(f"{label}: expected RSRegretStatistics from Basat.")
        if summary.run_count < 1 or summary.round_count < 1:
            raise ValueError(f"{label}: measured statistics must be nonempty.")
        if planned_runs is not None and summary.run_count > planned_runs:
            raise ValueError(f"{label}: observed runs exceed the declared planned_runs.")
        if label not in reference["methods"]:
            raise ValueError(f"No image reference was supplied for measured method {label!r}.")
        points = reference["methods"][label].get("points")
        if not isinstance(points, (list, tuple)) or not points:
            raise ValueError(f"{label}: reference points must be a nonempty sequence.")
        previous_index = -1
        comparisons = []
        for point in points:
            if not isinstance(point, Mapping):
                raise ValueError(f"{label}: every reference point must be a mapping.")
            index = point.get("source_index")
            if (isinstance(index, bool) or not isinstance(index, int) or index <= previous_index
                    or not 0 <= index < summary.round_count):
                raise ValueError(f"{label}: source indices must be increasing and within the saved curve.")
            previous_index = index
            measured_mean = _finite(summary.mean[index], f"{label} measured mean")
            measured_error = _finite(summary.error[index], f"{label} measured error", nonnegative=True)
            row = {"source_index": index, "measured_mean": measured_mean,
                   "measured_error_half_width": measured_error, "reference_mean": None,
                   "reference_mean_extraction_uncertainty": None,
                   "measured_minus_reference": None, "absolute_mean_gap": None,
                   "minimum_gap_after_extraction_uncertainty": None,
                   "within_extraction_interval": None}
            mean = point.get("mean")
            if mean is None:
                row["status"] = "reference_occluded_or_unavailable"
            else:
                mean = _finite(mean, f"{label} reference mean")
                uncertainty = _finite(point.get("mean_extraction_uncertainty"),
                                      f"{label} mean extraction uncertainty", nonnegative=True)
                difference = measured_mean - mean
                row.update(status="compared", reference_mean=mean,
                           reference_mean_extraction_uncertainty=uncertainty,
                           measured_minus_reference=difference, absolute_mean_gap=abs(difference),
                           minimum_gap_after_extraction_uncertainty=max(abs(difference) - uncertainty, 0.0),
                           within_extraction_interval=abs(difference) <= uncertainty)
            reference_error = point.get("sd_over_2")
            if reference_error is not None:
                if not math.isclose(summary.error_scale, .5, rel_tol=0, abs_tol=1e-15):
                    raise ValueError(f"{label}: reference SD/2 bars require measured error_scale=0.5.")
                reference_error = _finite(reference_error, f"{label} reference SD/2", nonnegative=True)
                uncertainty = _finite(point.get("sd_over_2_extraction_uncertainty"),
                                      f"{label} SD/2 extraction uncertainty", nonnegative=True)
                row.update(reference_sd_over_2=reference_error,
                           reference_sd_over_2_extraction_uncertainty=uncertainty,
                           error_half_width_difference=measured_error - reference_error)
            comparisons.append(row)
        available = [point for point in comparisons if point["status"] == "compared"]
        endpoint = next((point for point in comparisons if point["source_index"] == summary.round_count - 1), None)
        complete_runs = (int(contextual_coverage["complete"]) if contextual_coverage is not None
                         else summary.run_count)
        if contextual_coverage is not None and not contextual_coverage["complete"]:
            endpoint = None
        methods[label] = {
            "completed_runs": complete_runs, "planned_runs": planned_runs,
            "coverage": "unplanned" if planned_runs is None else "complete" if complete_runs == planned_runs else "partial",
            "round_count": summary.round_count, "compared_points": len(available),
            "unavailable_reference_points": len(comparisons) - len(available), "points": comparisons,
            "endpoint": copy.deepcopy(endpoint),
            "mean_absolute_gap_at_available_points": (sum(point["absolute_mean_gap"] for point in available) / len(available)) if available else None,
            "maximum_absolute_gap_at_available_points": max((point["absolute_mean_gap"] for point in available), default=None),
        }
        if contextual_coverage is not None:
            methods[label]["contextual_coverage"] = contextual_coverage
    return {"schema_version": 1, "kind": "descriptive_image_reference_comparison",
            "status": "complete" if planned_runs is not None and all(row["coverage"] == "complete" for row in methods.values()) else "partial_or_unplanned",
            "planned_runs": planned_runs, "methods": methods,
            "reference": copy.deepcopy(dict(reference)),
            "interpretation": (
                "Differences are descriptive. Image extraction uncertainty is retained separately from population variability. "
                "No missing reference point is estimated, and no scientific acceptance threshold is implied."
            )}


__all__ = ["compare_reference_statistics"]
