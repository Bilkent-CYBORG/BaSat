"""Behavioral regressions for the supplied AG-VS reference protocol."""

from copy import deepcopy

import pytest
import torch

import basat
from basat.algorithms.Stochastic.Exception.exception_AG_VS import AGVSCheckpointError, AGVSNodeError


def algorithm(**overrides):
    config = dict(effective_dimension=2, split_groups=3, initial_points=6,
                  batch_size=3, sequence_length=2, seed=2021,
                  stagnation_enabled=False)
    config.update(overrides)
    result = basat.AG_VS(bounds=basat.Bounds([0.] * 6, [1.] * 6),
                         config=basat.AG_VSConfig(**config))
    result.initialize_from_environment(basat.EnvironmentData(
        actions=torch.empty((0, 6), dtype=torch.float64),
        observations=torch.empty(0, dtype=torch.float64)))
    return result


def reward(action):
    return -(action[..., :2] - .25).square().sum(-1)


def step(subject):
    action = subject.ask()
    subject.tell(action=action, feedback={"evaluated_action": action, "observation": reward(action)})
    return action


def test_source_defaults_and_explicit_split_validation():
    subject = basat.AG_VS(bounds=basat.Bounds([0.] * 10, [1.] * 10),
                          config=basat.AG_VSConfig(effective_dimension=2, split_groups=3))
    assert subject.config.initial_points == 11
    assert subject.config.stagnation_patience == 9
    assert subject.config.batch_size == 3
    assert subject.config.sequence_length == 2
    assert subject.config.noise_variance == 1e-4
    assert subject.config.acquisition.name == "EI"
    with pytest.raises(AGVSNodeError, match="split_groups"):
        basat.AG_VS(bounds=subject.bounds, config=basat.AG_VSConfig(effective_dimension=2))
    with pytest.raises(AGVSNodeError, match="effective_dimension < split_groups"):
        basat.AG_VS(bounds=subject.bounds,
                    config=basat.AG_VSConfig(effective_dimension=2, split_groups=2))


def test_public_factory_resolves_string_ei_to_source_offset():
    subject = basat.make_algorithm(
        "AG_VS", bounds=basat.Bounds([0.] * 6, [1.] * 6),
        effective_dimension=2, split_groups=3, acquisition="EI", seed=2021,
    )
    assert subject.config.acquisition.name == "EI"
    assert subject.config.acquisition.xi == 1e-4


@pytest.mark.parametrize("configuration", [
    {"stopping_threshold": 0., "stopping_delta": .05},
    {"gp_fit_method": "map"},
    {"acquisition": basat.AcquisitionConfig(name="UCB")},
])
def test_incompatible_legacy_semantics_are_rejected(configuration):
    with pytest.raises(AGVSNodeError):
        algorithm(**configuration)


def test_observation_score_and_anchor_ignore_gp_predictions():
    subject = algorithm()
    for _ in range(2):
        step(subject)
    root = subject.state.nodes[0]
    root.score_history_rows = [3, 4, 5]
    root.previous_benchmark_history = [-.5, -.5, -.5]
    expected = (subject.state.global_observations[3:6] + .5).abs().mean()
    root.model = None
    assert subject._leaf_score(root) == pytest.approx(float(expected))
    subject.state.global_posterior_means[:] = torch.arange(6, dtype=torch.float64) * 1e5
    subject._set_anchor_from_current_best(root)
    best = subject.state.global_observations.argmax()
    torch.testing.assert_close(root.anchor, subject.state.global_actions[best])
    signed = algorithm(abs_mode=False)
    signed.state.global_actions = subject.state.global_actions.clone()
    signed.state.global_observations = subject.state.global_observations.clone()
    node = signed.state.nodes[0]
    node.score_history_rows, node.previous_benchmark_history = [3, 4, 5], [-.5] * 3
    assert signed._leaf_score(node) == pytest.approx(float((signed.state.global_observations[3:6] + .5).mean()))


def test_initial_lhs_has_exact_size_and_pending_remainder_resumes():
    subject = algorithm(initial_points=7)
    step(subject)
    step(subject)
    assert subject.request_size == 1
    pending = subject.ask()
    restored = algorithm(initial_points=7)
    restored.load_state_dict(subject.state_dict())
    assert restored.request_size == 1
    assert restored.last_ask is not None
    torch.testing.assert_close(restored.last_ask.actions[0], pending)
    feedback = {"evaluated_action": pending, "observation": reward(pending)}
    subject.tell(action=pending, feedback=feedback)
    restored.tell(action=pending, feedback=feedback)
    assert subject.evaluations == restored.evaluations == 7
    assert subject.request_size == restored.request_size == 3
    design = subject.state.global_actions
    for dimension in range(6):
        assert sorted(torch.floor(design[:, dimension] * 7).long().tolist()) == list(range(7))
    torch.testing.assert_close(subject.ask(), restored.ask(), rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["bo", "turbo"])
def test_lazy_fit_and_checkpoint_resume_preserve_future_batches(backend, monkeypatch):
    basat.set_seed(2021, threads=1)
    subject = algorithm(inner_optimizer=backend)
    for _ in range(5):
        step(subject)
    subject.ask()
    snapshot = subject.state_dict()
    restored = algorithm(inner_optimizer=backend)
    restored.load_state_dict(snapshot)
    assert restored.last_ask is not None
    action = subject.last_ask.actions
    feedback = {"evaluated_action": action, "observation": reward(action)}
    before_model = subject.state.nodes[subject.state.current_leaf_id].model
    subject.tell(action=action, feedback=feedback)
    restored.tell(action=action, feedback=feedback)
    # tell completes the visit and may merge/split, but it never fits a GP.
    assert before_model is not None
    for _ in range(5):
        expected, actual = subject.ask(), restored.ask()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        feedback = {"evaluated_action": expected, "observation": reward(expected)}
        subject.tell(action=expected, feedback=feedback)
        restored.tell(action=actual, feedback=feedback)
    torch.testing.assert_close(subject.observed_actions, restored.observed_actions, rtol=0, atol=0)
    torch.testing.assert_close(subject.result().recommended_action, restored.result().recommended_action, rtol=0, atol=0)
    assert subject.result().recommended_value == restored.result().recommended_value


def test_no_trailing_gp_fit_after_acquisition_tell(monkeypatch):
    subject = algorithm()
    for _ in range(2):
        step(subject)
    calls = []
    original = subject._new_node_gp

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(subject, "_new_node_gp", spy)
    step(subject)
    assert calls == []  # initial fitted root is still current before this acquisition
    step(subject)
    assert len(calls) == 1
    assert len(subject.state.active_leaf_ids) == 3
    assert all(subject.state.nodes[idx].model is None for idx in subject.state.active_leaf_ids)


def test_nested_pending_first_visit_outranks_scored_excluded_branch():
    subject = algorithm()
    root = subject.state.nodes[0]
    root.score, root.cohort_evaluated = 1., True
    child_ids = subject._split(root)
    high, branch, other = [subject.state.nodes[idx] for idx in child_ids]
    high.score, high.cohort_evaluated = 100., True
    other.score, other.cohort_evaluated = 50., True
    branch.score, branch.cohort_evaluated = 0., True
    branch.score_mass = 1
    pending_ids = subject._split(branch)
    assert subject._traverse()[0].node_id == pending_ids[0]
    for idx in pending_ids:
        subject.state.nodes[idx].cohort_evaluated = True
        subject.state.nodes[idx].score = 0.
        subject.state.nodes[idx].score_mass = 1
    subject._propagate(branch.node_id)
    assert subject._traverse()[0] is high


def test_restart_uses_active_child_union_and_tolerant_pair_deduplication():
    subject = algorithm()
    root = subject.state.nodes[0]
    subject.state.global_actions = torch.tensor([[0.] * 6, [.1] * 6, [.1 + 5e-11] * 6, [.2] * 6], dtype=torch.float64)
    subject.state.global_observations = torch.tensor([0., 1., 1. + 5e-11, 2.], dtype=torch.float64)
    root.training_rows = [0, 1, 2, 3]
    children = subject._split(root)
    for idx in children:
        subject.state.nodes[idx].training_rows = [1, 2, 3]
    subject._reinitialize()
    replacement = subject.state.nodes[0]
    assert replacement.training_rows == [1, 3]
    assert replacement.model is None
    assert replacement.score_history_rows == []
    assert subject.state.next_node_id == 1
    assert len(subject.state.global_actions) == 4


def test_target_retained_node_syncs_full_history_without_inherited_scores():
    subject = algorithm()
    for _ in range(2):
        step(subject)
    root = subject.state.nodes[0]
    ids = subject._split(root)
    children = [subject.state.nodes[idx] for idx in ids]
    # Singleton retained groups make the merged node reach the target size.
    children[0].active_indices = torch.tensor([0])
    children[1].active_indices = torch.tensor([1])
    children[2].active_indices = torch.tensor([2, 3, 4, 5])
    for order, child in enumerate(children):
        child.cohort_evaluated, child.score = True, 3. - order
        child.training_rows = [order]
    merged = subject.state.nodes[subject._merge(root)]
    assert merged.training_rows == list(range(6))
    assert merged.score_history_rows == []
    subject.state.global_actions = torch.cat((subject.state.global_actions, torch.zeros((1, 6), dtype=torch.float64)))
    subject.state.global_observations = torch.cat((subject.state.global_observations, torch.tensor([1.], dtype=torch.float64)))
    assert subject._start_visit() is merged
    assert merged.training_rows == list(range(7))
    assert merged.score_history_rows == []


def test_legacy_checkpoint_is_rejected_before_mutation():
    subject = algorithm()
    step(subject)
    snapshot = subject.state_dict()
    snapshot["algorithm_state"].pop("source_behavior_version")
    restored = algorithm()
    with pytest.raises(AGVSCheckpointError, match="predates"):
        restored.load_state_dict(snapshot)
    assert restored.evaluations == 0
    assert len(restored.state.global_actions) == 0


def test_pending_checkpoint_contains_advancing_source_streams():
    subject = algorithm()
    for _ in range(2):
        step(subject)
    before = deepcopy(subject.numpy_generator.bit_generator.state)
    subject.ask()
    stored = subject.state_dict()["algorithm_state"]
    assert stored["numpy_generator_state"] != before
    assert stored["source_behavior_version"] == subject.SOURCE_BEHAVIOR_VERSION
    assert stored["nodes"][0]["model"] is not None
    assert stored["current_leaf_id"] == 0


def test_turbo_fit_failure_raises_instead_of_generating_random_proposals(monkeypatch):
    subject = algorithm(inner_optimizer="turbo")
    for _ in range(2):
        step(subject)
    root = subject.state.nodes[0]
    root.model = None

    def fail(*_args, **_kwargs):
        raise RuntimeError("deliberate numerical failure")

    monkeypatch.setattr(subject, "_new_node_gp", fail)
    with pytest.raises(AGVSNodeError, match="deliberate numerical failure"):
        subject.ask()
    assert subject.last_ask is None


def test_cpu_checkpoint_before_optional_cuda_stream_field_still_resumes():
    subject = algorithm(inner_optimizer="turbo")
    for _ in range(3):
        step(subject)
    snapshot = subject.state_dict()
    snapshot["algorithm_state"].pop("source_cuda_generator_states")
    restored = algorithm(inner_optimizer="turbo")
    restored.load_state_dict(snapshot)
    torch.testing.assert_close(subject.ask(), restored.ask(), rtol=0, atol=0)


@pytest.mark.parametrize("version", [None, 1, 2])
def test_old_turbo_numerical_policy_checkpoint_requires_fresh_trajectory(version):
    subject = algorithm(inner_optimizer="turbo")
    step(subject)
    snapshot = subject.state_dict()
    if version is None:
        snapshot["algorithm_state"].pop("source_turbo_numerics_version")
    else:
        snapshot["algorithm_state"]["source_turbo_numerics_version"] = version
    restored = algorithm(inner_optimizer="turbo")
    with pytest.raises(AGVSCheckpointError, match="start a fresh trajectory"):
        restored.load_state_dict(snapshot)
    assert restored.evaluations == 0


def test_bo_checkpoint_does_not_require_a_turbo_numerics_marker():
    subject = algorithm()
    for _ in range(3):
        step(subject)
    snapshot = subject.state_dict()
    snapshot["algorithm_state"].pop("source_turbo_numerics_version")
    restored = algorithm()
    restored.load_state_dict(snapshot)
    torch.testing.assert_close(subject.ask(), restored.ask(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("min_cuda", [0, 1024])
def test_cuda_proposals_own_device_stream_and_resume(min_cuda):
    def create():
        subject = basat.AG_VS(
            bounds=basat.Bounds([0.] * 6, [1.] * 6), device="cuda",
            config=basat.AG_VSConfig(effective_dimension=2, split_groups=3,
                                      initial_points=6, inner_optimizer="turbo",
                                      turbo_min_cuda=min_cuda, stagnation_enabled=False,
                                      gp_maxiter=5, seed=2021))
        subject.initialize_from_environment(basat.EnvironmentData(
            actions=torch.empty((0, 6), dtype=torch.float64, device="cuda"),
            observations=torch.empty(0, dtype=torch.float64, device="cuda")))
        return subject

    subject = create()
    for _ in range(2):
        step(subject)
    global_state = torch.cuda.get_rng_state().clone()
    step(subject)
    torch.testing.assert_close(torch.cuda.get_rng_state(), global_state, rtol=0, atol=0)
    assert bool(subject._source_cuda_generators) is (min_cuda == 0)
    restored = create()
    restored.load_state_dict(subject.state_dict())
    torch.testing.assert_close(subject.ask(), restored.ask(), rtol=1e-6, atol=1e-6)

