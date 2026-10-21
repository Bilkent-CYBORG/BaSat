"""Independent finite-table checks for the MHC contextual evaluation metrics."""

from dataclasses import replace
import json

import pytest
import torch

from basat import Bounds, ContextualEnvironment, EnvironmentData, EnvironmentDataset
from basat.mhc_statistics import (
    ContextualRegretStatistics, contextual_regret_statistics,
    compare_contextual_regret_statistics,
)


ORDER = [[0.0], [1.0], [2.0]]


def make_environment(*, previous_data=None, allow_repeated_actions=False):
    table = [[2.0, 5.0, 8.0], [10.0, 9.0, 4.0], [-4.0, -1.0, -2.0]]
    return ContextualEnvironment(
        bounds=Bounds([0.0], [2.0]),
        dataset=EnvironmentDataset(
            actions=[[float(a)] for _ in ORDER for a in range(3)],
            contexts=[context for context in ORDER for _ in range(3)],
            observations=[v for values in table for v in values],
            # This is conditioning uncertainty, not synthetic added noise.
            observation_noise_variance=0.75,
        ),
        context_provider=lambda *, round_index, generator: ORDER[round_index // 2],
        previous_data=previous_data,
        allow_repeated_actions=allow_repeated_actions,
        dtype="float64", seed=2011,
    )


def reveal(env, count=6):
    for action in [0, 1, 2, 1, 2, 0][:count]:
        env.before_action()
        env.step([float(action)])


def summary(env):
    return contextual_regret_statistics(env, context_order=ORDER, queries_per_context=2)


def test_exact_instantaneous_average_and_maximum_simple_regret_are_distinct():
    env = make_environment()
    reveal(env)
    stats = summary(env)
    assert isinstance(stats, ContextualRegretStatistics)
    assert stats.query_indices == (1, 2, 3, 4, 5, 6)
    assert stats.context_indices == (0, 0, 1, 1, 2, 2)
    assert stats.local_query_indices == (1, 2, 1, 2, 1, 2)
    assert stats.selected_dataset_indices == (0, 1, 5, 4, 8, 6)
    assert stats.selected_rewards == (2.0, 5.0, 4.0, 9.0, -2.0, -4.0)
    assert stats.reference_values == (8.0, 10.0, -1.0)
    assert stats.instantaneous_regret == (6.0, 3.0, 6.0, 1.0, 1.0, 3.0)
    assert stats.cumulative_regret == (6.0, 9.0, 15.0, 16.0, 17.0, 20.0)
    assert stats.average_regret == pytest.approx((6, 4.5, 5, 4, 3.4, 20 / 6))
    assert stats.running_average_regret == stats.average_regret
    assert stats.per_context_incumbent_values == ((2.0, 5.0), (4.0, 9.0), (-2.0, -2.0))
    assert stats.per_context_simple_regret == ((6.0, 3.0), (6.0, 1.0), (1.0, 1.0))
    assert stats.aligned_local_queries == (1, 2)
    assert stats.maximum_simple_regret == (6.0, 3.0)
    assert stats.complete and stats.completion_fraction == 1.0
    assert stats.completed_contexts == 3 and stats.pending_queries == 0
    assert stats.initial_excluded_count == 0
    saved = stats.as_dict()
    assert json.loads(json.dumps(saved, allow_nan=False)) == saved
    saved["selected_rewards"][0] = 999
    assert stats.selected_rewards[0] == 2.0
    assert stats.to_dict() == stats.as_dict()


def test_warm_start_requires_explicit_metric_exclusion():
    env = make_environment(previous_data=EnvironmentData(
        actions=[[2.0]], contexts=[[0.0]], observations=[8.0]))
    reveal(env, 1)
    with pytest.raises(ValueError, match="warm-start"):
        summary(env)
    with pytest.raises(ValueError, match="warm-start"):
        contextual_regret_statistics(
            env, context_order=ORDER, queries_per_context=2, exclude_initial=False)


def test_excluded_initial_values_never_enter_optima_incumbents_or_query_counts():
    # Warm-start responses deliberately exceed all searchable rewards. Their
    # values must not leak into evaluation references or per-task incumbents.
    initial = EnvironmentData(
        actions=[[2.0], [0.0], [1.0]], contexts=ORDER,
        observations=[100.0, 200.0, 300.0])
    env = make_environment(previous_data=initial)
    reveal(env)
    stats = contextual_regret_statistics(
        env, context_order=ORDER, queries_per_context=2, exclude_initial=True)
    control = make_environment()
    reveal(control)
    expected = replace(summary(control), initial_excluded_count=3)
    assert stats == expected
    assert stats.initial_excluded_count == 3
    assert stats.observed_queries == 6 and stats.expected_queries == 6
    assert stats.per_context_query_counts == (2, 2, 2)
    assert stats.per_context_incumbent_values == ((2.0, 5.0), (4.0, 9.0), (-2.0, -2.0))
    assert stats.as_dict()["initial_excluded_count"] == 3
    assert stats.as_dict()["initial_history_included"] is False
    assert env.history.initial_data.number_of_observations == 3
    assert len(env.records) == 6


def test_initial_exclusion_preserves_partial_coverage_and_counts_online_repeats():
    initial = EnvironmentData(actions=[[0.0]], contexts=[[0.0]], observations=[2.0])
    env = make_environment(previous_data=initial, allow_repeated_actions=True)
    empty = contextual_regret_statistics(
        env, context_order=ORDER, queries_per_context=2, exclude_initial=True)
    assert empty.initial_excluded_count == 1
    assert empty.observed_queries == 0 and empty.completion_fraction == 0
    assert empty.maximum_simple_regret == ()
    # The first online query repeats the initial action, and still counts.
    reveal(env, 1)
    partial = contextual_regret_statistics(
        env, context_order=ORDER, queries_per_context=2, exclude_initial=True)
    assert partial.query_indices == (1,)
    assert partial.per_context_query_counts == (1, 0, 0)
    assert partial.average_regret == (6.0,)
    assert partial.maximum_simple_regret == ()
    assert partial.initial_excluded_count == 1


@pytest.mark.parametrize("exclude_initial", [None, 0, 1, "true"])
def test_initial_exclusion_requires_explicit_boolean(exclude_initial):
    with pytest.raises(TypeError, match="exclude_initial must be a boolean"):
        contextual_regret_statistics(
            make_environment(), context_order=ORDER, exclude_initial=exclude_initial)


def test_explicit_exclusion_without_initial_data_keeps_existing_metric_values():
    env = make_environment()
    reveal(env)
    assert contextual_regret_statistics(
        env, context_order=ORDER, queries_per_context=2, exclude_initial=True) == summary(env)


@pytest.mark.parametrize("count", range(7))
def test_partial_coverage_does_not_invent_unseen_task_incumbents(count):
    env = make_environment()
    reveal(env, count)
    stats = summary(env)
    assert stats.observed_queries == count
    assert stats.expected_queries == 6
    assert stats.completion_fraction == count / 6
    assert stats.completed_contexts == count // 2
    assert stats.pending_queries == count % 2
    assert stats.per_context_query_counts == tuple(min(2, max(0, count - 2 * i)) for i in range(3))
    assert len(stats.maximum_simple_regret) == max(0, count - 4)
    assert len(stats.average_regret) == count


def test_statistics_have_no_objective_calls_or_learner_information_side_effects(monkeypatch):
    env = make_environment()
    reveal(env, 5)
    before = env.get_history_data()
    rng = env.generator.get_state().clone()
    records = env.records

    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation statistics cannot query or update the environment.")

    monkeypatch.setattr(env, "step", forbidden)
    monkeypatch.setattr(env, "before_action", forbidden)
    monkeypatch.setattr(env, "lookup_dataset_observation", forbidden)
    summary(env)
    after = env.get_history_data()
    assert torch.equal(before.observations, after.observations)
    assert torch.equal(before.actions, after.actions)
    assert torch.equal(env.generator.get_state(), rng)
    assert env.records == records
    assert after.metadata.keys() == before.metadata.keys()
    for key, value in before.metadata.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, after.metadata[key])
        else:
            assert value == after.metadata[key]


@pytest.mark.parametrize("field,value,match", [
    ("dataset_index", None, "integer"),
    ("dataset_index", -1, "out-of-range"),
    ("dataset_index", 3, "Sequential context"),
    ("round_index", 1, "out-of-order"),
    ("context", torch.tensor([1.0]), "Recorded context"),
    ("action", torch.tensor([1.0]), "Requested action"),
    ("evaluated_action", torch.tensor([1.0]), "Evaluated action"),
    ("observation", torch.tensor(2.1), "Noiseless observation"),
    ("audit", {"stored_observation": torch.tensor(2.1)}, "Stored audit"),
    ("round_info", {}, "revealed pre-query context"),
])
def test_inconsistent_histories_rejected_not_silently_repaired(field, value, match):
    env = make_environment()
    reveal(env, 1)
    env.history.records[0] = replace(env.history.records[0], **{field: value})
    with pytest.raises(ValueError, match=match):
        summary(env)


def test_optional_audit_omission_still_validates_actual_response():
    env = make_environment()
    reveal(env, 1)
    env.history.records[0] = replace(env.history.records[0], audit={})
    assert summary(env).selected_rewards == (2.0,)


def test_decimal_context_order_preserves_environment_float64_precision():
    env = ContextualEnvironment(
        bounds=Bounds([0.0], [1.0]), dtype="float64",
        context_provider=lambda *, round_index, generator: [0.1],
        dataset=EnvironmentDataset(actions=[[0.0], [1.0]], contexts=[[0.1], [0.1]],
                                   observations=[0.2, 0.4]),
    )
    env.before_action()
    env.step([0.0])
    stats = contextual_regret_statistics(env, context_order=[[0.1]], queries_per_context=1)
    assert stats.context_order == ((0.1,),)
    assert stats.maximum_simple_regret == pytest.approx((0.2,))


@pytest.mark.parametrize("order", [[], [0, 1, 2], [[0], [0]], [[9]], [[float('nan')]], [[True]], [[0, 1]]])
def test_bad_context_order(order):
    with pytest.raises(ValueError):
        contextual_regret_statistics(make_environment(), context_order=order)


@pytest.mark.parametrize("count", [0, -1, True, 2.0])
def test_bad_query_count(count):
    with pytest.raises(ValueError):
        contextual_regret_statistics(make_environment(), context_order=ORDER, queries_per_context=count)


def test_exact_seven_by_fifty_budget_and_alignment():
    actions = [[float(a)] for _ in range(7) for a in range(50)]
    contexts = [[float(c)] for c in range(7) for _ in range(50)]
    env = ContextualEnvironment(
        bounds=Bounds([0.0], [49.0]), dtype="float64",
        context_provider=lambda *, round_index, generator: [float(round_index // 50)],
        dataset=EnvironmentDataset(actions=actions, contexts=contexts,
                                   observations=[float(a + c) for c in range(7) for a in range(50)]),
    )
    for query in range(350):
        env.before_action()
        env.step([float(query % 50)])
    stats = contextual_regret_statistics(env, context_order=[[c] for c in range(7)])
    assert stats.complete and stats.observed_queries == 350
    assert stats.per_context_query_counts == (50,) * 7
    assert stats.maximum_simple_regret == tuple(float(49 - a) for a in range(50))
    assert stats.average_regret[-1] == 24.5


def test_reference_comparison_exact_alignment_and_uncertainty():
    env = make_environment()
    reveal(env)
    reference = {"curves": {"maximum_simple_regret": {
        "query_indices": [1, 2], "values": [5.5, 3.05],
        "extraction_absolute_uncertainty": [0.1, 0.2],
    }}}
    report = compare_contextual_regret_statistics(summary(env), reference)
    assert report["all_reference_points_available"]
    assert report["endpoint"]["query_index"] == 2
    assert report["endpoint"]["signed_difference"] == pytest.approx(-0.05)
    assert report["endpoint"]["absolute_difference_lower_bound"] == 0
    assert report["endpoint"]["absolute_difference_upper_bound"] == pytest.approx(0.25)
    assert report["sampled_mean_absolute_difference"] == pytest.approx(0.275)
    assert "passed" not in report
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_partial_comparison_does_not_report_missing_endpoint_or_interpolate():
    env = make_environment()
    reveal(env, 3)
    reference = {"curves": {"average_regret": {
        "query_indices": [2, 4, 6], "values": [4, 4, 3],
        "extraction_absolute_uncertainty": 0.1,
    }}}
    report = compare_contextual_regret_statistics(summary(env), reference, metric="average_regret")
    assert report["compared_point_count"] == 1
    assert report["pointwise"][0]["query_index"] == 2
    assert report["endpoint"] is None
    assert not report["experiment_complete"]
    assert not report["all_reference_points_available"]


@pytest.mark.parametrize("changes", [
    {"query_indices": [0, 2]}, {"query_indices": [2, 1]}, {"query_indices": [1, 1]},
    {"query_indices": [1, 3]}, {"values": [3]}, {"values": [float('nan'), 3]},
    {"extraction_absolute_uncertainty": -1}, {"extraction_absolute_uncertainty": [0.1]},
])
def test_bad_reference_rejected(changes):
    curve = {"query_indices": [1, 2], "values": [6, 3], "extraction_absolute_uncertainty": 0.1}
    curve.update(changes)
    with pytest.raises(ValueError):
        compare_contextual_regret_statistics(summary(make_environment()),
                                             {"curves": {"maximum_simple_regret": curve}})
