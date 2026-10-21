"""Finite source benchmark isolation and signed expected-payoff regret."""

from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest
import torch

from basat import workflow
from basat._robos_regret import RoBOSSourceRegret
from basat.benchmarks import RoBOSSynthetic
from basat.environments import DistributionalEnvironment
from basat.regret import RSRegret


@pytest.fixture
def benchmark(monkeypatch, tmp_path):
    """Exercise benchmark geometry without a network or ignored local asset."""
    table = torch.arange(400, dtype=torch.float64).reshape(50, 8) / 400
    calls = []

    def load_array(path, **kwargs):
        calls.append((path, kwargs))
        return table.clone()

    monkeypatch.setattr(workflow, "load_array", load_array)
    objective = RoBOSSynthetic(cache_directory=tmp_path / "cache")
    return objective, table, calls


def test_benchmark_source_geometry_probabilities_and_mmd(benchmark):
    objective, table, calls = benchmark
    assert objective.actions.shape == (8, 1)
    assert objective.context_support.shape == (50, 1)
    assert torch.equal(objective.actions[:, 0], torch.from_numpy(np.linspace(-3.5, 3.5, 8)))
    assert torch.equal(objective.context_support[:, 0], torch.from_numpy(np.linspace(-12, 12, 50)))
    assert objective.epsilon == pytest.approx(0.4343303639429951, abs=1e-15)
    assert float(objective.reference_distribution.sum()) == pytest.approx(1.0)
    assert float(objective.sampling_distribution.sum()) == pytest.approx(1.0)
    assert objective.maximize
    assert objective.bounds.lower.tolist() == [-3.5]
    assert objective.bounds.upper.tolist() == [3.5]
    dataset = objective.dataset
    assert dataset.observation_noise_variance is None
    assert torch.equal(dataset.observations, table.flatten())
    assert torch.equal(dataset.actions[:8], objective.actions)
    assert torch.equal(dataset.actions[8:16], objective.actions)
    assert torch.equal(dataset.contexts[:8], objective.context_support[0].repeat(8, 1))
    assert torch.equal(objective(dataset.actions, dataset.contexts), table.flatten())
    assert torch.equal(
        objective(objective.actions[None, :, :], objective.context_support[:, None, :]), table,
    )
    assert objective([objective.actions[3, 0].item()], [objective.context_support[31, 0].item()]) == table[31, 3]
    assert calls[0][1] == {
        "url": RoBOSSynthetic.SOURCE_URL, "sha256": RoBOSSynthetic.SOURCE_SHA256,
        "dtype": "float64", "device": "cpu",
    }
    assert objective.metadata["source_revision"] == "96bf39400a0b2044e3c21e08b4fbeb8886584ee3"
    assert objective.metadata["source_sha256"] == "737ab7d505b23466ce7017dd7878a28a9bf7894615931e2dd950f951e70d54e5"


def test_benchmark_properties_are_owned_copies_and_off_grid_rejected(benchmark):
    objective, table, _ = benchmark
    for name in ("actions", "context_support", "payoff_table", "reference_distribution",
                 "sampling_distribution", "context_gram"):
        copy = getattr(objective, name)
        copy.zero_()
        assert bool(getattr(objective, name).abs().sum() > 0)
    objective.dataset.observations.zero_()
    assert torch.equal(objective.payoff_table, table)
    with pytest.raises(ValueError, match="exact stored"):
        objective([-3.5 + 1e-12], [-12.])
    with pytest.raises(ValueError, match="exact stored"):
        objective([-3.5], [-11.9])
    with pytest.raises(ValueError, match="trailing dimension"):
        objective(-3.5, [-12.])
    with pytest.raises(TypeError, match="real-valued"):
        objective([True], [-12.])
    with pytest.raises(TypeError, match="real-valued"):
        objective([-3.5 + 1j], [-12.])


def test_dataset_environment_does_not_reveal_table_or_sampling_probabilities(benchmark):
    objective, _, _ = benchmark
    environment = DistributionalEnvironment(
        bounds=objective.bounds, dataset=objective.dataset,
        context_support=objective.context_support,
        reference_distribution=objective.reference_distribution,
        sampling_distribution=objective.sampling_distribution,
        margin=objective.epsilon, add_noise_to_data=True, noise_std=.02,
        allow_repeated_actions=True, dtype="float64", seed=7,
    )
    info = environment.before_action()
    assert set(info) == {"context_support", "reference_distribution", "margin",
                         "candidate_actions", "candidate_dataset_indices"}
    assert torch.equal(info["candidate_actions"], objective.actions)
    assert environment.get_history_data().number_of_observations == 0
    feedback = environment.step(objective.actions[0])
    assert set(feedback) == {"observation", "evaluated_action", "context", "observation_noise_variance"}
    assert float(feedback["observation_noise_variance"]) == pytest.approx(.0004)
    assert "true_sampling_distribution" in environment.history.records[0].audit
    assert environment.get_history_data().number_of_observations == 1


def test_local_asset_loading_verifies_checksum_without_downloading(monkeypatch, tmp_path):
    buffer = io.BytesIO()
    np.save(buffer, np.zeros((50, 8), dtype=np.float64))
    content = buffer.getvalue()
    path = tmp_path / "custom.npy"
    path.write_bytes(content)

    def forbidden(*args, **kwargs):
        raise AssertionError("Explicit local_asset must never download.")

    monkeypatch.setattr(workflow.urllib.request, "urlopen", forbidden)
    with pytest.raises(ValueError, match="checksum"):
        RoBOSSynthetic(local_asset=path)
    # A deterministic test asset exercises the complete verified-load path.
    monkeypatch.setattr(RoBOSSynthetic, "SOURCE_SHA256", hashlib.sha256(content).hexdigest())
    objective = RoBOSSynthetic(local_asset=path)
    assert objective.payoff_table.shape == (50, 8)
    assert not bool(objective.payoff_table.any())
    assert path.read_bytes() == content
    with pytest.raises(FileNotFoundError, match="existing NPY"):
        RoBOSSynthetic(local_asset=tmp_path / "missing.npy")


@pytest.mark.parametrize("shape", [(8, 50), (400,)])
def test_benchmark_rejects_malformed_table(monkeypatch, tmp_path, shape):
    monkeypatch.setattr(workflow, "load_array", lambda *args, **kwargs: torch.zeros(shape))
    with pytest.raises(ValueError, match="50-context by 8-action"):
        RoBOSSynthetic(cache_directory=tmp_path)


def test_signed_oracle_is_not_clipped_and_existing_metric_stays_strict():
    metric = RoBOSSourceRegret(aspiration=.6, epsilon=.5)
    expected = [.1, .7, .3]
    kappa = [-2., 0., .2]
    np.testing.assert_allclose(metric.instantaneous(expected, source_kappa=kappa), [1.5, 0., .2])
    np.testing.assert_allclose(metric.evaluate(expected, source_kappa=kappa), [1.5, 1.5, 1.7])
    np.testing.assert_allclose(
        metric.evaluate(expected, source_kappa=kappa, cumulative=False), [1.5, 0., .2],
    )
    with pytest.raises(ValueError, match="nonnegative"):
        RSRegret(aspiration=.6, epsilon=.5, optimal_fragility=-2.)


def test_torch_backend_and_population_statistics():
    metric = RoBOSSourceRegret(aspiration=1., epsilon=.5)
    rewards = torch.tensor([[0., 0.], [0., 0.]], dtype=torch.float32)
    kappas = torch.tensor([[-2., 0.], [0., 2.]], dtype=torch.float64)
    actual = metric.evaluate(rewards, source_kappa=kappas)
    assert actual.dtype == rewards.dtype and actual.device == rewards.device
    torch.testing.assert_close(actual, torch.tensor([[2., 3.], [1., 1.]]))
    summary = metric.statistics(rewards, source_kappa=kappas, error_scale=.5)
    assert summary.mean == (1.5, 2.)
    assert summary.std == (.5, 1.)
    assert summary.error == (.25, .5)
    assert metric.evaluate(torch.zeros(2, dtype=torch.int64), source_kappa=0.).dtype == torch.float64
    torch.testing.assert_close(
        metric.evaluate(rewards, source_kappa=[0., 2.]), torch.tensor([[1., 1.], [1., 1.]]),
    )


@pytest.mark.parametrize("kappa", [[], [0.], [[1.], [2.]], [float("nan"), 0.], [True, False], [1j, 0.]])
def test_metric_rejects_invalid_or_misaligned_oracle_values(kappa):
    metric = RoBOSSourceRegret(aspiration=.6, epsilon=.5)
    with pytest.raises((TypeError, ValueError)):
        metric.evaluate([[0., 0.], [0., 0.]], source_kappa=kappa)


@pytest.mark.parametrize("rewards", [[], 0., [float("inf")], [True], [1j]])
def test_metric_rejects_invalid_expected_rewards(rewards):
    with pytest.raises((TypeError, ValueError)):
        RoBOSSourceRegret(aspiration=.6, epsilon=.5).evaluate(rewards, source_kappa=0.)


@pytest.mark.parametrize("kwargs", [
    {"aspiration": True, "epsilon": .5}, {"aspiration": [.6], "epsilon": .5},
    {"aspiration": .6, "epsilon": -1.}, {"aspiration": .6, "epsilon": float("inf")},
    {"aspiration": .6, "epsilon": .5, "cumulative": 1},
])
def test_metric_configuration_is_explicit(kwargs):
    with pytest.raises((TypeError, ValueError)):
        RoBOSSourceRegret(**kwargs)


def test_metric_rejects_overflow_and_invalid_cumulative_override():
    metric = RoBOSSourceRegret(aspiration=1., epsilon=1.)
    with pytest.raises(ValueError, match="overflows"):
        metric.evaluate([-1e308, -1e308], source_kappa=0.)
    with pytest.raises(ValueError, match="overflows"):
        metric.evaluate([-1e308], source_kappa=-1e308)
    with pytest.raises(TypeError, match="cumulative"):
        metric.evaluate([0.], source_kappa=0., cumulative=1)
