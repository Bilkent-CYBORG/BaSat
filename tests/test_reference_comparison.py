"""Reference pixels are uncertain observations, not exact scientific truth."""

import copy
import json

import pytest

from basat.reference_comparison import compare_reference_statistics
from basat.regret import rs_regret_statistics


def reference():
    return {"kind": "image_derived_reference", "is_released_numerical_data": False,
            "reference_image_sha256": "test-provenance", "methods": {"A": {"points": [
                {"source_index": 0, "mean": None, "mean_extraction_uncertainty": None},
                {"source_index": 1, "mean": 2.25, "mean_extraction_uncertainty": .125,
                 "sd_over_2": .2, "sd_over_2_extraction_uncertainty": .05},
            ]}}}


def test_gap_uncertainty_sd_and_provenance_are_separate():
    summary = rs_regret_statistics([[1.,2.],[2.,3.]], error_scale=.5)
    source = reference()
    result = compare_reference_statistics({"A": summary}, source, planned_runs=2)
    row = result["methods"]["A"]
    assert result["status"] == "complete"
    assert row["endpoint"]["measured_mean"] == 2.5
    assert row["endpoint"]["absolute_mean_gap"] == .25
    assert row["endpoint"]["minimum_gap_after_extraction_uncertainty"] == .125
    assert row["endpoint"]["measured_error_half_width"] == .25
    assert row["endpoint"]["within_extraction_interval"] is False
    assert row["unavailable_reference_points"] == 1
    assert row["points"][0]["reference_mean"] is None
    json.dumps(result, allow_nan=False)
    source["methods"]["A"]["points"][1]["mean"] = 500
    assert result["reference"]["methods"]["A"]["points"][1]["mean"] == 2.25


def test_partial_runs_and_occluded_endpoint_are_not_filled():
    source = reference()
    source["methods"]["A"]["points"][-1] = {"source_index": 1, "mean": None}
    summary = rs_regret_statistics([[1.,2.]], error_scale=.5)
    result = compare_reference_statistics({"A": summary}, source, planned_runs=50)
    assert result["status"] == "partial_or_unplanned"
    assert result["methods"]["A"]["endpoint"]["status"] == "reference_occluded_or_unavailable"
    assert result["methods"]["A"]["maximum_absolute_gap_at_available_points"] is None


@pytest.mark.parametrize("field,value", [("source_index", -1), ("source_index", 2),
                                        ("mean", float("nan")), ("mean_extraction_uncertainty", -.1),
                                        ("mean_extraction_uncertainty", None)])
def test_invalid_reference_cannot_produce_a_numeric_comparison(field, value):
    source = reference()
    source["methods"]["A"]["points"][-1][field] = value
    with pytest.raises(ValueError):
        compare_reference_statistics({"A": rs_regret_statistics([[1.,2.]], error_scale=.5)}, source)


def test_wrong_error_convention_rejected_and_inputs_unchanged():
    source = reference()
    before = copy.deepcopy(source)
    with pytest.raises(ValueError, match="error_scale=0.5"):
        compare_reference_statistics({"A": rs_regret_statistics([[1.,2.]])}, source)
    assert source == before


def test_excess_run_count_and_fake_original_arrays_rejected():
    summary = rs_regret_statistics([[1.,2.],[1.,3.]], error_scale=.5)
    with pytest.raises(ValueError, match="exceed"):
        compare_reference_statistics({"A": summary}, reference(), planned_runs=1)
    source = reference()
    source["is_released_numerical_data"] = True
    with pytest.raises(ValueError, match="image-derived"):
        compare_reference_statistics({"A": summary}, source, planned_runs=2)
