"""Read-only comparisons of independent trial measurements and published arrays."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .trials import ExperimentTrialsResult
from .workflow import to_list


def compare_trials(
    result: ExperimentTrialsResult | Mapping[str, Any], reference: Mapping[str, Any], *,
    expected_trials: int, expected_evaluations: int,
    reference_labels: Mapping[str, str | None] | None = None,
    reference_kind: str = "incumbent", maximize: bool | None = None,
    endpoint_tolerance: float = 0.1, allow_partial: bool = False,
) -> dict[str, Any]:
    """Compare per-run incumbent curves without rerunning any objective or model.

    Parameters
    ----------
    result : ExperimentTrialsResult or mapping
        Ordinary Basat trials with retained raw result-document snapshots, or
        method label to raw trial-by-evaluation observations. The mapping form
        requires explicit ``maximize`` and assumes no uncharged warm starts.
        Different completed-trial counts are permitted only for partial reports.
    reference : mapping
        Method label to a numeric trials-by-evaluations matrix. A plain JSON
        ``{"__type": "ndarray", "value": matrix}`` envelope is also accepted;
        no external class deserializer is used.
    expected_trials, expected_evaluations : int
        Explicit planned coverage. Missing, failed or short measured runs are
        never silently dropped or padded. Reference matrices must have exactly
        this shape. ``allow_partial=True`` permits fewer complete measured trials
        for pilot reporting, but cannot establish endpoint agreement.
    reference_labels : mapping, optional
        Basat label to reference label; None or an absent reference marks a
        method as an addition without published comparison. Unknown Basat keys
        are errors. No absent baseline is synthesized.
    reference_kind : {"incumbent", "observations"}
        Published incumbents are validated as monotone. Raw observations are
        accumulated independently within each reference trial.
    maximize : bool, optional
        If supplied, must agree with every measured method's recorded direction.
        Otherwise each method's recorded direction is used.
    endpoint_tolerance : float, default 0.1
        Absolute tolerance for the difference in final mean incumbent. This is
        a declared numerical criterion, not a statistical equivalence test.
        The report also retains all mean-curve differences and distributions;
        passing an endpoint alone does not establish reproduction.

    Returns
    -------
    dict
        JSON-safe data with means, population SD, per-evaluation differences,
        final per-trial distributions and separately labeled Monte Carlo standard
        errors. No uncertainty-based enlargement of the tolerance is applied.

    Examples
    --------
    >>> report = basat.compare_trials(result, published_arrays,
    ...     expected_trials=50, expected_evaluations=50,
    ...     maximize=False, endpoint_tolerance=0.1)  # doctest: +SKIP
    """
    if not isinstance(result, (ExperimentTrialsResult, Mapping)):
        raise TypeError("compare_trials requires a trial result or raw observation mapping.")
    if not isinstance(reference, Mapping):
        raise TypeError("reference must map method labels to numeric trial arrays.")
    for name, value in (("expected_trials", expected_trials), ("expected_evaluations", expected_evaluations)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if (isinstance(endpoint_tolerance, bool) or not np.isfinite(endpoint_tolerance)
            or endpoint_tolerance < 0):
        raise ValueError("endpoint_tolerance must be finite and nonnegative.")
    if not isinstance(allow_partial, bool) or (maximize is not None and not isinstance(maximize, bool)):
        raise TypeError("allow_partial and an explicit maximize must be boolean.")
    if reference_kind not in {"incumbent", "observations"}:
        raise ValueError("reference_kind must be 'incumbent' or 'observations'.")
    if isinstance(result, ExperimentTrialsResult):
        if not result.records:
            raise ValueError("The result must retain raw trial records.")
        labels = list(result.results)
        by_label = {label: [row["algorithms"][label] for row in result.records] for label in labels}
        directions = {label: result.statistics[label].maximize for label in labels}
        planned = result.planned_trials
    else:
        if maximize is None:
            raise ValueError("Raw observation mappings require explicit maximize.")
        labels = list(result)
        by_label = {}
        for label in labels:
            matrix = np.asarray(to_list(result[label]), dtype=float)
            if matrix.ndim != 2 or not all(matrix.shape):
                raise ValueError(f"{label}: measured arrays must be nonempty trial-by-evaluation matrices.")
            by_label[label] = [{"observations": values.tolist(), "maximize": maximize,
                "initial_observation_count": 0, "environment_evaluations": matrix.shape[1],
                "failure": None} for values in matrix]
        directions = dict.fromkeys(labels, maximize)
        planned = expected_trials
    if not labels or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("At least one nonempty measured method label is required.")
    mapping = {} if reference_labels is None else dict(reference_labels)
    if set(mapping) - set(labels):
        raise ValueError("reference_labels contains an unknown measured method.")
    if any(value is not None and not isinstance(value, str) for value in mapping.values()):
        raise TypeError("Reference label values must be strings or None.")
    coverage = {label: len(by_label[label]) for label in labels}
    if (planned not in (None, expected_trials) or any(count > expected_trials for count in coverage.values())
            or (not allow_partial and any(count != expected_trials for count in coverage.values()))):
        raise ValueError("Measured trial coverage does not match the declared plan.")
    complete = all(count == expected_trials for count in coverage.values())
    methods: dict[str, Any] = {}
    for label in labels:
        rows = []
        actual_trials = coverage[label]
        direction = directions[label]
        if maximize is not None and direction != maximize:
            raise ValueError(f"{label}: recorded optimization direction does not match maximize.")
        accumulate = np.maximum.accumulate if direction else np.minimum.accumulate
        for record in by_label[label]:
            if record.get("failure") is not None or record.get("maximize") != direction:
                raise ValueError(f"{label}: failed or inconsistently directed trials cannot be compared.")
            values = np.asarray(record["observations"], dtype=float)
            offset = record.get("initial_observation_count", 0)
            if (isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
                    or values.ndim != 1 or not np.isfinite(values).all()
                    or len(values) - offset != expected_evaluations
                    or record.get("environment_evaluations") != expected_evaluations):
                raise ValueError(f"{label}: each measured trial must contain the exact evaluation budget.")
            rows.append(accumulate(values)[offset:])
        measured = np.asarray(rows)
        mean, sd = measured.mean(axis=0), measured.std(axis=0, ddof=0)
        entry = {"measured_trials": actual_trials, "mean": mean.tolist(), "population_sd": sd.tolist(),
                 "final_values": measured[:, -1].tolist(), "maximize": direction}
        ref_label = mapping.get(label, label)
        if ref_label is None or ref_label not in reference:
            entry.update(status="no_published_reference", reference_label=ref_label,
                         endpoint_within_tolerance=None)
        else:
            values = reference[ref_label]
            if isinstance(values, Mapping):
                if values.get("__type") != "ndarray" or "value" not in values:
                    raise ValueError(f"{ref_label}: unsupported numeric JSON envelope.")
                values = values["value"]
            published = np.asarray(to_list(values), dtype=float)
            if (published.shape != (expected_trials, expected_evaluations)
                    or not np.isfinite(published).all()):
                raise ValueError(f"{ref_label}: reference must have finite, exact planned trial-by-evaluation shape.")
            if reference_kind == "observations":
                published = accumulate(published, axis=1)
            elif ((np.diff(published, axis=1) < 0).any() if direction
                  else (np.diff(published, axis=1) > 0).any()):
                raise ValueError(f"{ref_label}: published incumbent rows must be monotone.")
            reference_mean, reference_sd = published.mean(axis=0), published.std(axis=0, ddof=0)
            difference = mean - reference_mean
            # SE uses sample variance, separately from population SD used in
            # the descriptive curves. Trial pairings/seeds are not presumed.
            se = (np.sqrt(measured.var(axis=0, ddof=1) / actual_trials
                          + published.var(axis=0, ddof=1) / expected_trials)
                  if min(actual_trials, expected_trials) > 1 else None)
            method_complete = actual_trials == expected_trials
            entry.update(status="complete_comparison" if method_complete else "partial_comparison",
                reference_label=ref_label, reference_trials=expected_trials,
                reference_mean=reference_mean.tolist(), reference_population_sd=reference_sd.tolist(),
                reference_final_values=published[:, -1].tolist(), mean_difference=difference.tolist(),
                absolute_mean_difference=np.abs(difference).tolist(),
                difference_standard_error=None if se is None else se.tolist(),
                endpoint_mean_difference=float(difference[-1]),
                endpoint_absolute_difference=float(abs(difference[-1])),
                endpoint_within_tolerance=bool(abs(difference[-1]) <= endpoint_tolerance) if method_complete else None)
        methods[label] = entry
    compared = [row for row in methods.values() if row["status"] != "no_published_reference"]
    return {"schema_version": 1, "metric": "mean per-run incumbent objective",
            "status": "complete" if complete else "partial", "planned_trials": expected_trials,
            "measured_trials": min(coverage.values()), "measured_trials_by_method": coverage,
            "evaluations": expected_evaluations,
            "endpoint_tolerance": float(endpoint_tolerance), "methods": methods,
            "all_compared_endpoints_within_tolerance": (
                all(row["endpoint_within_tolerance"] for row in compared) if complete and compared else None),
            "qualifications": ["Endpoint tolerance is not a statistical equivalence or reproduction claim.",
                "Reference and measured seeds/trials are not assumed paired.",
                "Missing-reference methods have no paper-agreement verdict.",
                "Pilot uncertainty never widens the declared endpoint tolerance."]}
