"""Released random-prefix geometry, owned streams and feedback transactions."""
import copy

import numpy as np
import pytest
from sklearn.metrics.pairwise import euclidean_distances
import torch

import basat
from basat.environments.Adversarial import _source_mt19937_normal
from basat.environments.environment import EnvironmentConfigurationError


@pytest.mark.parametrize("squared", [False, True])
@pytest.mark.parametrize("radius", [.25, .5, 2.])
def test_exhaustive_grid_choices_and_mt_match_literal_released_geometry(squared, radius):
    grid = basat.AdveRSSynthetic().grid()
    actual, expected = basat.RandomStream(0), basat.RandomStream(0)
    attack = basat.RandomAttack(radius, squared_distance=squared)
    values = grid.numpy()
    for action in grid:
        distances = euclidean_distances(values, action.numpy().reshape(1, -1), squared=squared).reshape(-1)
        admissible = np.where(distances <= (radius**2 if squared else radius))[0]
        selected = int(expected.choice(admissible))
        displacement = attack(action=action, candidate_actions=grid, source_random_stream=actual)
        assert torch.equal(displacement, grid[selected] - action)
        # Finite environments resolve this difference back to the exact stored
        # row; subtraction/addition itself need not round-trip bit-for-bit.
        assert torch.allclose(action + displacement, grid[selected], rtol=0., atol=5e-16)
        assert actual.get_state() == expected.get_state()


def test_uniform_attack_has_no_payoff_access_and_noise_follows_its_choice():
    calls = []
    env = basat.AdversarialEnvironment(bounds=basat.Bounds(lower=[-1.], upper=[1.]),
        candidate_actions=[[-1.], [0.], [1.]], function=lambda x: (calls.append(x.clone()), 7.)[1],
        attack=basat.RandomAttack(1.), noise_std=.2, allow_repeated_actions=True, dtype="float64")
    actual, expected = basat.RandomStream(17), basat.RandomStream(17)
    for _ in range(20):
        selected = expected.choice(3)
        noisy = 7. + .2 * _source_mt19937_normal(expected)
        feedback = env.step([0.], source_random_stream=actual,
            source_normal_policy="advers-numpy126-scalar-polar-v1")
        assert torch.equal(calls[-1], env.candidate_actions[selected])
        assert float(feedback["observation"]) == noisy
        assert actual.get_state() == expected.get_state()
        assert "latent_observation" not in feedback and "source_rng_after_feedback_sha256" not in feedback
    assert len(calls) == 20
    assert env._attack_oracle_evaluations == 0 and env._worst_case_values == []


def test_failed_feedback_restores_choice_and_gaussian_state(monkeypatch):
    env = basat.AdversarialEnvironment(bounds=basat.Bounds(lower=[-1.], upper=[1.]),
        candidate_actions=[[-1.], [0.], [1.]], function=lambda _: 7.,
        attack=basat.RandomAttack(1.), noise_std=.2, allow_repeated_actions=True, dtype="float64")
    stream = basat.RandomStream(17)
    stream.normal()
    before = copy.deepcopy(stream.get_state())
    original = env.sample_noise
    def fail(**kwargs):
        original(**kwargs)
        raise RuntimeError("after random choice and noise")
    monkeypatch.setattr(env, "sample_noise", fail)
    with pytest.raises(RuntimeError, match="after random"):
        env.step([0.], source_random_stream=stream, source_normal_policy="advers-numpy126-scalar-polar-v1")
    assert stream.get_state() == before and env.history.records == [] and env.round == 0
    assert env._runtime_source_random_stream is None


def test_default_torch_stream_and_explicit_zero_budget_floor():
    grid = basat.array([[0.], [.0001], [.1]], dtype="float64")
    generator, expected = torch.Generator().manual_seed(12), torch.Generator().manual_seed(12)
    attack = basat.RandomAttack(.2)
    for _ in range(10):
        index = int(torch.randint(3, (), generator=expected))
        assert torch.equal(attack(action=grid[0], candidate_actions=grid, generator=generator), grid[index])
    assert torch.equal(generator.get_state(), expected.get_state())
    actual, reference = basat.RandomStream(3), basat.RandomStream(3)
    attack = basat.RandomAttack(0., squared_distance=True, source_epsilon_floor=1e-6)
    for _ in range(10):
        index = reference.choice([0, 1])
        assert torch.equal(attack(action=grid[0], candidate_actions=grid, source_random_stream=actual), grid[index])
    assert actual.get_state() == reference.get_state()
    for radius in (-1., float("inf"), float("nan"), True):
        with pytest.raises(EnvironmentConfigurationError):
            basat.RandomAttack(radius)
