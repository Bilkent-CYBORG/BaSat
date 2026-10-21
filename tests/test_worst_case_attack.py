"""Finite true-objective attacks, cache isolation and checkpoint validation."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch
from sklearn.metrics.pairwise import euclidean_distances

import basat
from basat.environments.environment import (
    EnvironmentConfigurationError, EnvironmentDataError, EnvironmentProtocolError,
)


def quadratic(x):
    return x.square().sum()


def make(radius=1.0):
    return basat.AdversarialEnvironment(
        bounds=basat.Bounds(lower=[-1.0], upper=[1.0]),
        candidate_actions=[[-1.0], [0.0], [1.0]], function=quadratic,
        attack=basat.WorstCaseAttack(radius), dtype="float64", noise_std=0.,
        allow_repeated_actions=True,
    )


@pytest.mark.parametrize("radius", [True, -1, float("nan"), float("inf"), "invalid"])
def test_invalid_radius(radius):
    with pytest.raises(EnvironmentConfigurationError, match="finite and nonnegative"):
        basat.WorstCaseAttack(radius)


def test_finite_domain_required():
    with pytest.raises(EnvironmentConfigurationError, match="finite"):
        basat.AdversarialEnvironment(
            bounds=basat.Bounds(lower=[-1.], upper=[1.]), function=quadratic,
            attack=basat.WorstCaseAttack(.5),
        )


def test_inclusive_boundary_and_first_minimum():
    attack = basat.WorstCaseAttack(.5)
    candidates = torch.tensor([[0.], [-.5], [.5], [.5000000001]], dtype=torch.float64)
    delta = attack(action=candidates[0], candidate_actions=candidates,
                   true_values=[10., -2., -2., -10.])
    assert delta.item() == -.5


@pytest.mark.parametrize("values", [[0.], [0., float("nan")]])
def test_invalid_true_payoffs(values):
    with pytest.raises(EnvironmentProtocolError, match="finite true payoff"):
        basat.WorstCaseAttack(1.)(
            action=torch.tensor([0.]), candidate_actions=torch.tensor([[0.], [1.]]),
            true_values=values,
        )


@pytest.mark.parametrize("radius,squared,budget", [(.5, False, .5), (.5, True, .25),
                                                   (.5**.5, True, .5)])
def test_every_grid_center_matches_released_mask_and_first_argmin(radius, squared, budget):
    """The source uses squared distances for RS2, p=2 and RO attacks.

    Independently calculate its inclusive mask/first minimum at all 1024
    centers, including the p=2 source's unusual sqrt(.5) physical radius.
    """
    benchmark = basat.AdveRSSynthetic(formula="released", resolution=32)
    grid = benchmark.grid()
    coordinates = grid.numpy()
    truth = torch.stack([benchmark(point) for point in grid]).numpy().reshape(-1)
    attack = basat.WorstCaseAttack(radius)
    for center, action in enumerate(grid):
        distances = euclidean_distances(coordinates, coordinates[center:center + 1], squared=squared)
        indices = np.where(distances <= budget)[0]
        expected = int(indices[np.argmin(truth[indices])])
        delta = attack(action=action, candidate_actions=grid, true_values=truth)
        torch.testing.assert_close(action + delta, grid[expected], rtol=0, atol=1e-14)


def test_cache_is_private_rng_free_and_not_learner_history(monkeypatch):
    env = make()
    calls = []
    evaluate = env.evaluate_function
    monkeypatch.setattr(env, "evaluate_function", lambda action: (calls.append(action.clone()), evaluate(action))[1])
    rng = env.generator.get_state().clone()
    metadata = env.prepare_attack_cache()
    assert metadata["objective_evaluations_this_call"] == 3
    assert metadata["cached_points"] == 3
    assert metadata["prepared_this_call"]
    assert "values" not in metadata
    assert env.history.records == []
    assert env.history.initial_data is None
    assert set(env.before_action()) == {"candidate_actions", "candidate_dataset_indices"}
    assert torch.equal(env.generator.get_state(), rng)
    assert env.prepare_attack_cache()["objective_evaluations_this_call"] == 0
    feedback = env.step(torch.tensor([1.], dtype=torch.float64))
    assert len(calls) == 4  # three oracle calls and one observed objective query
    assert feedback["observation"].item() == 0.
    assert "true_values" not in feedback and "worst_case_cache" not in feedback
    assert len(env.history.records) == 1


def test_feedback_rng_digest_is_experiment_only():
    env = make()
    stream = basat.RandomStream(123, engine="mt19937")
    feedback = env.step(torch.tensor([1.], dtype=torch.float64), source_random_stream=stream)
    digest = hashlib.sha256(json.dumps(
        stream.get_state(), sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    assert env.history.records[-1].audit["source_rng_after_feedback_sha256"] == digest
    assert "source_rng_after_feedback_sha256" not in feedback
    state = stream.get_state()
    assert env.history.records[-1].audit["source_rng_after_feedback_transition"] == {
        "keys_sha256": hashlib.sha256(json.dumps(
            state["keys"], separators=(",", ":"),
        ).encode()).hexdigest(),
        "position": state["position"], "has_gaussian": state["has_gaussian"],
        "cached_gaussian": state["cached_gaussian"],
    }
    assert "source_rng_after_feedback_transition" not in feedback


def test_atomic_disk_cache_reuse_and_reset(tmp_path, monkeypatch):
    first = make()
    saved = first.prepare_attack_cache(cache_directory=tmp_path)
    assert saved["objective_evaluations_this_call"] == 3
    second = make()
    monkeypatch.setattr(second, "evaluate_function", lambda *args: pytest.fail("extra oracle call"))
    restored = second.prepare_attack_cache(cache_directory=tmp_path)
    assert restored["disk_cache_hit"]
    assert not restored["prepared_this_call"]
    assert restored["objective_evaluations_this_call"] == 0
    assert restored["payoff_checksum"] == saved["payoff_checksum"]
    first.reset(seed=42)
    assert first.attack_oracle_metadata["preparation_evaluations"] == 0
    assert first.prepare_attack_cache()["objective_evaluations_this_call"] == 0


def test_disk_cache_configuration_and_checksum_rejection(tmp_path):
    make().prepare_attack_cache(cache_directory=tmp_path)
    with pytest.raises(EnvironmentDataError, match="checksum"):
        make(.5).prepare_attack_cache(cache_directory=tmp_path)
    path = tmp_path / "worst_case_payoffs.json"
    value = json.loads(path.read_text())
    value["worst_case_cache"]["values"][0] += 1
    # ArtifactStore is the same atomic public helper used by the library.
    basat.ArtifactStore(str(tmp_path)).write_json(path.name, value)
    with pytest.raises(EnvironmentDataError, match="checksum"):
        make().prepare_attack_cache(cache_directory=tmp_path)


def test_checkpoint_cache_restoration_and_tampering(monkeypatch):
    env = make()
    env.prepare_attack_cache()
    state = env.state_dict()
    restored = make()
    restored.load_state_dict(state)
    monkeypatch.setattr(restored, "evaluate_function", lambda *args: pytest.fail("extra oracle call"))
    assert restored.prepare_attack_cache()["objective_evaluations_this_call"] == 0
    assert restored.attack_oracle_metadata == env.attack_oracle_metadata
    tampered = copy.deepcopy(state)
    tampered["extra_state"]["worst_case_cache"]["values"][0] += 1.
    with pytest.raises(EnvironmentDataError, match="checksum"):
        make().load_state_dict(tampered)
    with pytest.raises(EnvironmentConfigurationError, match="source_fingerprint"):
        make(.5).load_state_dict(state)


def test_partial_cache_checkpoint_resumes_only_missing_values(monkeypatch):
    env = make()
    evaluate = env.evaluate_function
    calls = []

    def interrupted(action):
        if len(calls) == 1:
            raise RuntimeError("interrupted preparation")
        calls.append(action.clone())
        return evaluate(action)

    monkeypatch.setattr(env, "evaluate_function", interrupted)
    with pytest.raises(RuntimeError, match="interrupted preparation"):
        env.prepare_attack_cache()
    state = env.state_dict()
    restored = make()
    restored.load_state_dict(state)
    info = restored.prepare_attack_cache()
    assert info["objective_evaluations_this_call"] == 2
    assert info["preparation_evaluations"] == 3


def test_interrupted_disk_preparation_resumes_missing_values(tmp_path, monkeypatch):
    env = make()
    evaluate = env.evaluate_function
    calls = []

    def interrupted(action):
        if calls:
            raise KeyboardInterrupt("interrupted preparation")
        calls.append(action.clone())
        return evaluate(action)

    monkeypatch.setattr(env, "evaluate_function", interrupted)
    with pytest.raises(KeyboardInterrupt):
        env.prepare_attack_cache(cache_directory=tmp_path)
    restored = make()
    info = restored.prepare_attack_cache(cache_directory=tmp_path)
    assert info["objective_evaluations_this_call"] == 2
    assert info["prepared_this_call"]
    assert info["preparation_evaluations"] == 2  # Actual calls in this environment.
    assert info["cache_total_unique_function_preparation_evaluations"] == 3
    artifact = basat.ArtifactStore(str(tmp_path)).read_json("worst_case_payoffs.json")
    assert artifact["worst_case_cache"]["cache_total_unique_function_preparation_evaluations"] == 3
    complete = make().prepare_attack_cache(cache_directory=tmp_path)
    assert complete["objective_evaluations_this_call"] == 0
    assert complete["cached_points"] == 3
    assert complete["cache_total_unique_function_preparation_evaluations"] == 3


def test_longer_memory_cache_preserves_suffix_over_partial_disk(tmp_path, monkeypatch):
    env = make()
    env.prepare_attack_cache()
    partial = make()
    partial._worst_case_values = [1.]
    partial._attack_oracle_evaluations = 1
    basat.ArtifactStore(str(tmp_path)).write_json("worst_case_payoffs.json", partial._extra_state_dict())
    monkeypatch.setattr(env, "evaluate_function", lambda *args: pytest.fail("repeated cached objective"))
    info = env.prepare_attack_cache(cache_directory=tmp_path)
    assert info["objective_evaluations_this_call"] == 0
    assert info["preparation_evaluations"] == 3
    assert info["cache_total_unique_function_preparation_evaluations"] == 3
    fresh = make().prepare_attack_cache(cache_directory=tmp_path)
    assert fresh["objective_evaluations_this_call"] == 0
    assert fresh["cached_points"] == 3


def test_conflicting_valid_cache_prefix_rejected_without_mutation(tmp_path):
    memory = make()
    memory.prepare_attack_cache()
    artifact_env = make()
    artifact_env._worst_case_values = [9.]
    artifact_env._attack_oracle_evaluations = 1
    store = basat.ArtifactStore(str(tmp_path))
    store.write_json("worst_case_payoffs.json", artifact_env._extra_state_dict())
    before = memory.attack_oracle_metadata
    disk_before = store.read_json("worst_case_payoffs.json")
    with pytest.raises(EnvironmentDataError, match="conflict"):
        memory.prepare_attack_cache(cache_directory=tmp_path)
    assert memory.attack_oracle_metadata == before
    assert store.read_json("worst_case_payoffs.json") == disk_before


def test_data_cache_fingerprint_includes_hidden_rewards(tmp_path):
    def environment(rewards):
        return basat.AdversarialEnvironment(
            bounds=basat.Bounds(lower=[-1.], upper=[1.]),
            dataset=basat.EnvironmentDataset(actions=[[-1.], [0.], [1.]], observations=rewards),
            attack=basat.WorstCaseAttack(1.), dtype="float64", allow_repeated_actions=True,
        )
    original = environment([1., 0., 1.])
    changed = environment([0., 1., 2.])
    info = original.prepare_attack_cache(cache_directory=tmp_path)
    assert info["cache_total_unique_function_preparation_evaluations"] == 0
    assert changed._source_fingerprint() == original._source_fingerprint()
    assert changed._dataset_fingerprint() != original._dataset_fingerprint()
    with pytest.raises(EnvironmentDataError, match="checksum"):
        changed.prepare_attack_cache(cache_directory=tmp_path)
    assert changed.attack_oracle_metadata["cached_points"] == 0
    assert info["payoff_checksum"] != changed._attack_payoff_checksum(original._worst_case_values)


def test_v1_cache_rejected(tmp_path):
    env = make()
    env.prepare_attack_cache()
    state = env._extra_state_dict()
    state["worst_case_cache"]["version"] = 1
    basat.ArtifactStore(str(tmp_path)).write_json("worst_case_payoffs.json", state)
    with pytest.raises(EnvironmentDataError, match="checksum"):
        make().prepare_attack_cache(cache_directory=tmp_path)


def test_data_backed_cache_requires_no_objective_calls():
    env = basat.AdversarialEnvironment(
        bounds=basat.Bounds(lower=[-1.], upper=[1.]),
        dataset=basat.EnvironmentDataset(actions=[[-1.], [0.], [1.]], observations=[1., 0., 1.]),
        attack=basat.WorstCaseAttack(1.), dtype="float64", allow_repeated_actions=True,
    )
    assert env.prepare_attack_cache()["objective_evaluations_this_call"] == 0
    assert env.step(torch.tensor([1.], dtype=torch.float64))["observation"].item() == 0.
