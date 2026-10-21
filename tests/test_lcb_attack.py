"""Finite LCB attack, runtime model binding and transactional source randomness."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from basat.environments.Adversarial import AdversarialEnvironment, LCBAttack
from basat.environments.environment import (
    EnvironmentConfigurationError,
    EnvironmentDataError,
    EnvironmentLookupError,
    EnvironmentProtocolError,
    NonFiniteObservationError,
)
from basat.models.utils import Bounds
from basat.workflow import RandomStream


POINTS = torch.tensor([[0., 0.], [.25, 0.], [.5, 0.], [1., 0.]], dtype=torch.float64)


def reward(x):
    return x[0].square()


def confidence(points):
    lower = torch.tensor([1., -2., -2., -9.], dtype=points.dtype)
    return lower + 10., lower


def make_environment(**kwargs):
    defaults = dict(
        bounds=Bounds(lower=[0., 0.], upper=[1., 1.]),
        function=reward,
        candidate_actions=POINTS,
        attack=LCBAttack(.5),
        dtype="float64",
        allow_repeated_actions=True,
        noise_std=1.,
        seed=19,
    )
    defaults.update(kwargs)
    return AdversarialEnvironment(**defaults)


def test_first_lcb_tie_matches_source_distance_and_never_uses_reward():
    environment = make_environment(noise_std=0.)
    round_info = environment.before_action()
    assert set(round_info) == {"candidate_actions", "candidate_dataset_indices"}
    assert torch.equal(round_info["candidate_actions"], POINTS)
    assert round_info["candidate_dataset_indices"].tolist() == [0, 1, 2, 3]

    result = environment.step(POINTS[0], confidence_bounds=confidence)

    assert torch.equal(result["evaluated_action"], POINTS[1])
    assert result["observation"].item() == .0625
    assert environment.history.records[0].audit["candidate_index"] == 0
    assert environment.history.records[0].audit["evaluated_candidate_index"] == 1
    assert environment.history.records[0].audit["latent_observation"].item() == .0625


def test_radius_boundary_inclusion_and_just_outside_exclusion():
    def bounds(points):
        lower = -torch.arange(len(points), dtype=points.dtype)
        return lower + 10, lower

    environment = make_environment(noise_std=0.)
    assert torch.equal(environment.step(POINTS[0], confidence_bounds=bounds)["evaluated_action"],
                       POINTS[2])
    environment = make_environment(noise_std=0., attack=LCBAttack(np.nextafter(.5, 0.)))
    assert torch.equal(environment.step(POINTS[0], confidence_bounds=bounds)["evaluated_action"],
                       POINTS[1])


def test_candidates_keep_row_order_and_do_not_expose_objective_values():
    calls = []

    def tracked_reward(x):
        calls.append(x)
        return reward(x)

    environment = make_environment(function=tracked_reward, allow_repeated_actions=False)
    assert calls == []
    info = environment.before_action()
    info["candidate_actions"][0, 0] = .9
    assert torch.equal(environment.before_action()["candidate_actions"], POINTS)
    environment.step(POINTS[0], confidence_bounds=confidence)
    assert len(calls) == 1
    assert environment.before_action()["candidate_dataset_indices"].tolist() == [1, 2, 3]
    with pytest.raises(EnvironmentLookupError):
        environment.step(POINTS[0], confidence_bounds=confidence)
    with pytest.raises(EnvironmentLookupError):
        environment.step([.123, 0.], confidence_bounds=confidence)


def test_attack_configuration_immutable_and_callback_must_be_supplied_each_step():
    attack = LCBAttack(.5)
    with pytest.raises(FrozenInstanceError):
        attack.radius = 1.
    environment = make_environment(attack=attack)
    with pytest.raises(EnvironmentProtocolError, match="runtime confidence_bounds"):
        environment.step(POINTS[0])
    environment.step(POINTS[0], confidence_bounds=confidence)
    with pytest.raises(EnvironmentProtocolError, match="runtime confidence_bounds"):
        environment.step(POINTS[0])
    assert environment.round == 1
    assert environment._runtime_confidence_bounds is None


@pytest.mark.parametrize("radius", [-1., float("inf"), float("nan"), True, "bad"])
def test_invalid_attack_radius_rejected(radius):
    with pytest.raises(EnvironmentConfigurationError):
        LCBAttack(radius)


def test_duplicate_candidates_rejected_and_checkpoint_detects_geometry_or_radius_change():
    with pytest.raises(EnvironmentDataError, match="distinct"):
        make_environment(candidate_actions=torch.cat([POINTS, POINTS[:1]]))
    environment = make_environment()
    environment.step(POINTS[0], confidence_bounds=confidence)
    state = environment.state_dict()
    for incompatible in (
        make_environment(candidate_actions=POINTS.flip(0)),
        make_environment(attack=LCBAttack(.25)),
    ):
        with pytest.raises(EnvironmentConfigurationError, match="source_fingerprint"):
            incompatible.load_state_dict(state)
    restored = make_environment()
    restored.load_state_dict(state)
    assert restored.round == 1
    assert restored.history.records[0].audit["evaluated_candidate_index"] == 1


def test_shared_stream_matches_numpy_noise_and_resumes_with_external_state():
    stream = RandomStream(2021)
    expected = np.random.RandomState(2021)
    stream.choice(17, size=3, replace=False)
    expected.choice(17, size=3, replace=False)
    environment = make_environment()
    first = environment.step(POINTS[0], confidence_bounds=confidence, source_random_stream=stream)
    assert first["observation"].item() == .0625 + expected.normal()
    saved_environment, saved_stream = environment.state_dict(), stream.get_state()
    restored = make_environment()
    restored.load_state_dict(saved_environment)
    resumed_stream = RandomStream(0)
    resumed_stream.set_state(saved_stream)
    assert restored._runtime_source_random_stream is None
    for env, rng in ((environment, stream), (restored, resumed_stream)):
        env.step(POINTS[0], confidence_bounds=confidence, source_random_stream=rng)
    assert stream.get_state() == resumed_stream.get_state()
    assert torch.equal(environment.history.records[-1].observation,
                       restored.history.records[-1].observation)
    assert environment._source_fingerprint() == restored._source_fingerprint()


def test_failed_batch_restores_cached_gaussian_stream_and_history():
    calls = 0

    def fail_second(x):
        nonlocal calls
        calls += 1
        return reward(x) if calls == 1 else float("nan")

    environment = make_environment(function=fail_second)
    stream = RandomStream(2021)
    stream.normal()  # Include the cached member of NumPy's normal pair.
    state = stream.get_state()
    with pytest.raises(NonFiniteObservationError):
        environment.step(POINTS[:2], confidence_bounds=confidence, source_random_stream=stream)
    assert state == stream.get_state()
    assert environment.round == 0
    assert environment.history.records == []
    assert environment._runtime_confidence_bounds is None
    assert environment._runtime_source_random_stream is None


def test_unconfigured_attack_and_noise_keep_existing_behavior():
    def no_attack(action):
        return torch.zeros_like(action)

    environment = make_environment(attack=no_attack, candidate_actions=None, noise_std=0.)
    assert environment.before_action() == {}
    assert environment.step([.123, 0.])["observation"].item() == .123 ** 2
