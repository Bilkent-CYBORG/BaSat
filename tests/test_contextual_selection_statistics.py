"""Chronological best-of-five evaluation, not daily-context pooling."""

from dataclasses import replace
import json

import pytest
import torch

import basat


def make_environment(*, offset=20.0, contexts=None):
    return basat.SequentialContextualTableEnvironment(
        actions=[[float(i)] for i in range(6)],
        # All three rows deliberately have the same visible context.
        contexts=contexts if contexts is not None else [[0.0], [0.0], [0.0]],
        observations=[
            [20.0, 23.0, 22.0, 21.0, 24.0, 30.0],
            [25.0, 28.0, 27.0, 24.0, 21.0, 18.0],
            [30.0, 24.0, 28.0, 22.0, 19.0, 35.0],
        ],
        queries_per_context=5,
        observation_offset=offset,
        noise_variance=0.1,
        dtype="float64",
    )


def reveal(environment, choices):
    for index in choices:
        environment.before_action()
        environment.step(environment._actions[index])


def test_repeated_contexts_stay_chronological_and_readings_are_uncentered():
    environment = make_environment()
    reveal(environment, [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    summary = basat.contextual_selection_statistics(environment)
    assert isinstance(summary, basat.ContextualSelectionStatistics)
    assert summary.context_rows == (0, 1, 2)
    assert summary.elapsed_hours == pytest.approx((5 / 60, 10 / 60, 15 / 60))
    assert summary.selected_indices == ((0, 1, 2, 3, 4),) * 3
    assert summary.selected_readings == (
        (20.0, 23.0, 22.0, 21.0, 24.0),
        (25.0, 28.0, 27.0, 24.0, 21.0),
        (30.0, 24.0, 28.0, 22.0, 19.0),
    )
    assert summary.reference_values == (30.0, 28.0, 35.0)
    assert summary.selected_values == (24.0, 28.0, 30.0)
    assert summary.instantaneous_errors == (6.0, 0.0, 5.0)
    assert summary.cumulative_mean_errors == pytest.approx((6.0, 3.0, 11.0 / 3))
    assert summary.completed_contexts == 3
    assert summary.pending_queries == 0
    assert summary.total_contexts == 3
    assert summary.observed_queries == summary.expected_queries == 15
    assert summary.complete
    assert summary.completion_fraction == 1.0
    exported = summary.as_dict()
    assert json.loads(json.dumps(exported, allow_nan=False)) == exported
    assert summary.to_dict() == exported
    exported["selected_indices"][0][0] = 99
    assert summary.selected_indices[0][0] == 0


def test_minimize_uses_global_and_selected_minima_and_no_offset_dependence():
    env = make_environment(offset=20.0)
    plain = make_environment(offset=0.0)
    choices = [0, 1, 2, 3, 4] * 3
    reveal(env, choices)
    reveal(plain, choices)
    summary = basat.contextual_selection_statistics(env, maximize=False, interval_minutes=10)
    assert summary.reference_values == (20.0, 18.0, 19.0)
    assert summary.selected_values == (20.0, 21.0, 19.0)
    assert summary.instantaneous_errors == (0.0, 3.0, 0.0)
    assert summary.cumulative_mean_errors == (0.0, 1.5, 1.0)
    assert summary.elapsed_hours == pytest.approx((1 / 6, 1 / 3, 1 / 2))
    assert (basat.contextual_selection_statistics(env).as_dict()
            == basat.contextual_selection_statistics(plain).as_dict())


@pytest.mark.parametrize("count", [0, 1, 4, 5, 6, 9, 10, 14, 15])
def test_partial_only_emits_completed_groups_with_explicit_coverage(count):
    env = make_environment()
    reveal(env, ([0, 1, 2, 3, 4] * 3)[:count])
    summary = basat.contextual_selection_statistics(env)
    assert summary.completed_contexts == count // 5
    assert summary.pending_queries == count % 5
    assert summary.observed_queries == count
    assert summary.completion_fraction == count / 15
    assert summary.complete is (count == 15)
    assert len(summary.selected_indices) == count // 5
    assert len(summary.instantaneous_errors) == count // 5
    assert len(summary.elapsed_hours) == count // 5


def test_statistics_do_not_query_environment_or_modify_learner_visible_history(monkeypatch):
    env = make_environment()
    reveal(env, [0, 1, 2, 3, 4, 0])
    before = env.get_history_data()
    rng = env.generator.get_state().clone()
    records = env.records

    def forbidden(*args, **kwargs):
        raise AssertionError("Statistics must not evaluate an objective or advance the environment.")

    monkeypatch.setattr(env, "step", forbidden)
    monkeypatch.setattr(env, "before_action", forbidden)
    monkeypatch.setattr(env, "_read_table", forbidden)
    monkeypatch.setattr(env, "function", forbidden)
    basat.contextual_selection_statistics(env)
    after = env.get_history_data()
    assert torch.equal(before.actions, after.actions)
    assert torch.equal(before.observations, after.observations)
    assert env.records == records
    assert torch.equal(env.generator.get_state(), rng)
    assert before.number_of_observations == 6
    assert set(before.metadata) == set(after.metadata)
    assert "stored_observation" not in after.metadata
    assert "observations" not in after.metadata


@pytest.mark.parametrize("field,value,match", [
    ("action", torch.tensor([5.0], dtype=torch.float64), "Requested action"),
    ("evaluated_action", torch.tensor([5.0], dtype=torch.float64), "Evaluated action"),
    ("context", torch.tensor([1.0], dtype=torch.float64), "Context"),
    ("observation", torch.tensor(20.0, dtype=torch.float64), "Centered feedback"),
    ("round_index", 1, "out-of-order"),
])
def test_malformed_record_rejected_even_in_partial_group(field, value, match):
    env = make_environment()
    reveal(env, [0])
    env.history.records[0] = replace(env.history.records[0], **{field: value})
    with pytest.raises(ValueError, match=match):
        basat.contextual_selection_statistics(env)


@pytest.mark.parametrize("audit,match", [
    ({}, "audit fields"),
    ({"context_row": 1, "action_index": 0, "stored_observation": torch.tensor(20.)}, "out-of-order"),
    ({"context_row": 0.0, "action_index": 0, "stored_observation": torch.tensor(20.)}, "integer"),
    ({"context_row": 0, "action_index": 6, "stored_observation": torch.tensor(20.)}, "out-of-range"),
    ({"context_row": 0, "action_index": True, "stored_observation": torch.tensor(20.)}, "integer"),
    ({"context_row": 0, "action_index": 0, "stored_observation": torch.tensor(float("nan"))}, "audit reading"),
    ({"context_row": 0, "action_index": 0, "stored_observation": torch.tensor(21.)}, "audit reading"),
])
def test_malformed_audit_rejected(audit, match):
    env = make_environment()
    reveal(env, [0])
    env.history.records[0] = replace(env.history.records[0], audit=audit)
    with pytest.raises(ValueError, match=match):
        basat.contextual_selection_statistics(env)


def test_duplicate_action_rejected_instead_of_deduplicated():
    env = make_environment()
    reveal(env, [0, 1])
    first = env.history.records[0]
    env.history.records[1] = replace(first, round_index=1)
    with pytest.raises(ValueError, match="duplicate selected"):
        basat.contextual_selection_statistics(env)


def test_missing_or_reordered_records_rejected():
    env = make_environment()
    reveal(env, [0, 1, 2, 3, 4, 0])
    env.history.records.pop(1)
    with pytest.raises(ValueError, match="record count"):
        basat.contextual_selection_statistics(env)
    env.round -= 1
    with pytest.raises(ValueError, match="out-of-order"):
        basat.contextual_selection_statistics(env)


@pytest.mark.parametrize("round_info,match", [({}, "revealed context"),
    ({"context": torch.tensor([1.0])}, "Revealed context")])
def test_wrong_prequery_context_rejected(round_info, match):
    env = make_environment()
    reveal(env, [0])
    env.history.records[0] = replace(env.history.records[0], round_info=round_info)
    with pytest.raises(ValueError, match=match):
        basat.contextual_selection_statistics(env)


@pytest.mark.parametrize("kwargs", [
    {"selection": "mean"}, {"maximize": 1}, {"interval_minutes": 0},
    {"interval_minutes": -1}, {"interval_minutes": float("nan")},
    {"interval_minutes": float("inf")}, {"interval_minutes": True},
])
def test_invalid_public_arguments_rejected(kwargs):
    with pytest.raises(ValueError):
        basat.contextual_selection_statistics(make_environment(), **kwargs)


def test_wrong_environment_type_rejected():
    with pytest.raises(TypeError, match="SequentialContextualTableEnvironment"):
        basat.contextual_selection_statistics(object())


def sensor_reference(indices):
    return {
        "kind": "image_derived_reference", "is_released_numerical_data": False,
        "reference_image_sha256": "fixture-image-provenance",
        "methods": {"CGP-UCB": {"points": [
            {"source_index": index, "mean": 2.0, "mean_extraction_uncertainty": 0.1}
            for index in indices
        ]}},
    }


def test_reference_comparison_accepts_contextual_cumulative_error_and_no_error_bars():
    env = make_environment()
    reveal(env, [0, 1, 2, 3, 4] * 3)
    summary = basat.contextual_selection_statistics(env)
    comparison = basat.compare_reference_statistics(
        {"CGP-UCB": summary}, sensor_reference([0, 1, 2]), planned_runs=1,
    )
    method = comparison["methods"]["CGP-UCB"]
    assert comparison["status"] == "complete"
    assert method["completed_runs"] == 1
    assert method["coverage"] == "complete"
    assert method["endpoint"]["measured_mean"] == pytest.approx(11 / 3)
    assert method["endpoint"]["absolute_mean_gap"] == pytest.approx(5 / 3)
    assert method["endpoint"]["measured_error_half_width"] == 0.0
    assert [point["measured_mean"] for point in method["points"]] == pytest.approx([6.0, 3.0, 11 / 3])
    assert method["contextual_coverage"] == {
        "complete": True, "completed_contexts": 3, "total_contexts": 3,
        "observed_queries": 15, "expected_queries": 15, "pending_queries": 0,
    }
    json.dumps(comparison, allow_nan=False)


@pytest.mark.parametrize("count", [5, 6, 9, 10, 11, 14])
def test_reference_comparison_partial_context_not_counted_as_curve_endpoint(count):
    env = make_environment()
    reveal(env, ([0, 1, 2, 3, 4] * 3)[:count])
    summary = basat.contextual_selection_statistics(env)
    comparison = basat.compare_reference_statistics(
        {"CGP-UCB": summary}, sensor_reference(range(count // 5)), planned_runs=1,
    )
    method = comparison["methods"]["CGP-UCB"]
    assert comparison["status"] == "partial_or_unplanned"
    assert method["completed_runs"] == 0
    assert method["coverage"] == "partial"
    assert method["endpoint"] is None
    assert method["round_count"] == count // 5
    assert method["contextual_coverage"]["pending_queries"] == count % 5
    assert [point["measured_mean"] for point in method["points"]] == [6., 3.][:count // 5]
    with pytest.raises(ValueError, match="within the saved curve"):
        basat.compare_reference_statistics(
            {"CGP-UCB": summary}, sensor_reference([count // 5]), planned_runs=1,
        )


def test_contextual_comparison_rejects_invented_trials_or_sd_bars():
    env = make_environment()
    reveal(env, [0, 1, 2, 3, 4] * 3)
    summary = basat.contextual_selection_statistics(env)
    reference = sensor_reference([2])
    with pytest.raises(ValueError, match="one chronological run"):
        basat.compare_reference_statistics({"CGP-UCB": summary}, reference, planned_runs=50)
    reference["methods"]["CGP-UCB"]["points"][0].update(
        sd_over_2=0.1, sd_over_2_extraction_uncertainty=0.01,
    )
    with pytest.raises(ValueError, match="SD/2 bars require"):
        basat.compare_reference_statistics({"CGP-UCB": summary}, reference, planned_runs=1)
