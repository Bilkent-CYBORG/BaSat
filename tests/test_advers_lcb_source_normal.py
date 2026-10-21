"""Explicit LCB source arithmetic and audit stay isolated from default noise."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch

import basat
from basat._advers_protocol import restore
from basat.environments.environment import EnvironmentConfigurationError
from basat.experiment import ExperimentCheckpointError


POLICY = "advers-numpy126-scalar-polar-v1"


def confidence(points):
    lower = torch.zeros(len(points), dtype=points.dtype)
    return lower + 1., lower


def environment(function=lambda _: 0.):
    return basat.AdversarialEnvironment(
        bounds=basat.Bounds(lower=[-1.], upper=[1.]), candidate_actions=[[0.]],
        function=function, attack=basat.LCBAttack(.5), noise_std=1., dtype="float64",
        allow_repeated_actions=True,
    )


def digest_rng(stream):
    return hashlib.sha256(json.dumps(stream.get_state(), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def test_opted_in_lcb_matches_independent_numpy126_noise_and_audits_every_draw():
    # Independently generated with the official locked NumPy1.26.4 runtime:
    # seed derive_seed(2026,1,0,0), ten-point design, then 100 scalar normals.
    stream = basat.RandomStream(2351327623)
    stream.choice(1024, size=10, replace=False)
    calls = []
    env = environment(lambda action: (calls.append(action.clone()), 0.)[1])
    values = []
    for _ in range(100):
        feedback = env.step([0.], confidence_bounds=confidence, source_random_stream=stream,
                            source_normal_policy=POLICY)
        values.append(float(feedback["observation"]))
        audit = env.history.records[-1].audit
        assert audit["source_rng_after_feedback_sha256"] == digest_rng(stream)
        assert audit["source_normal_protocol"] == POLICY
        assert "source_rng_after_feedback_sha256" not in feedback
        state = stream.get_state()
        assert audit["source_rng_after_feedback_transition"]["cached_gaussian"] == state["cached_gaussian"]
        assert env._runtime_source_normal_policy == "native"
    assert hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest() == (
        "20910686c28ca8fdf8c9d782b32453cf7fb30a1cb9abf5b346841552bc205d4c")
    assert len(calls) == len(env.history.records) == 100
    assert env._worst_case_values == []
    assert env._attack_oracle_evaluations == 0
    assert env._runtime_confidence_bounds is None
    assert env._runtime_source_random_stream is None


def test_default_lcb_retains_current_randomstream_normal_and_no_source_audit():
    stream = basat.RandomStream(2351327623)
    expected = np.random.RandomState(2351327623)
    stream.choice(1024, size=10, replace=False)
    expected.choice(1024, size=10, replace=False)
    env = environment()
    for _ in range(100):
        feedback = env.step([0.], confidence_bounds=confidence, source_random_stream=stream)
        assert float(feedback["observation"]) == expected.normal()
        assert "source_rng_after_feedback_sha256" not in env.history.records[-1].audit
        assert "source_normal_protocol" not in env.history.records[-1].audit
    assert env._runtime_source_normal_policy == "native"


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_failed_opted_in_evaluation_restores_cached_stream_and_runtime_policy(monkeypatch, failure):
    env = environment()
    stream = basat.RandomStream(123)
    stream.normal()  # Include a cached Gaussian in the exact rollback snapshot.
    before = copy.deepcopy(stream.get_state())
    original_noise = env.sample_noise

    def interrupted(**kwargs):
        original_noise(**kwargs)
        raise failure("injected after source draw")

    monkeypatch.setattr(env, "sample_noise", interrupted)
    with pytest.raises(failure, match="injected after source draw"):
        env.step([0.], confidence_bounds=confidence, source_random_stream=stream,
                 source_normal_policy=POLICY)
    assert stream.get_state() == before
    assert env.round == 0 and env.history.records == []
    assert env._runtime_source_normal_policy == "native"
    assert env._runtime_source_random_stream is None
    assert env._runtime_confidence_bounds is None


@pytest.mark.parametrize("policy,stream", [("unknown", None), (POLICY, None),
                                         (POLICY, basat.RandomStream(123, engine="pcg64"))])
def test_source_policy_requires_explicit_compatible_stream(policy, stream):
    env = environment(lambda _: pytest.fail("objective evaluated before policy validation"))
    with pytest.raises(EnvironmentConfigurationError, match="source_normal_policy|MT19937"):
        env.step([0.], confidence_bounds=confidence, source_random_stream=stream,
                 source_normal_policy=policy)
    assert env.history.records == []


@pytest.mark.parametrize("changed", [None, "native", "unrecognized"])
def test_camera_ready_lcb_rejects_missing_or_changed_checkpoint_noise_policy(changed):
    from test_advers_camera_ready_lcb_execution import make

    original = make()
    original.run(max_rounds=1, plot=False, show=False)
    saved = original._checkpoint_payload(phase="post_tell")
    bootstrap = copy.deepcopy(saved["experiment_state"]["advers_bootstrap"])
    assert bootstrap["version"] == 2 and bootstrap["source_normal_policy"] == POLICY
    if changed is None:
        del bootstrap["source_normal_policy"]
    else:
        bootstrap["source_normal_policy"] = changed
    with pytest.raises(ExperimentCheckpointError, match="initialization state"):
        restore(make(), bootstrap, saved["environment_state"], saved["algorithm_state"])
