"""Camera-ready source equations and versioned transactional execution.

Source revision 6de58579174a71a247bb92f7f920f015f3331ff9, synthetic.py.
The small acquisition fixtures were independently evaluated by that revision's
unchanged adv_RS1.py; complete GPy trajectory fixtures are recorded separately.
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import basat
from basat._advers_protocol import validate
from basat.algorithms.algorithm import AlgorithmConfigurationError
from basat.algorithms.Adversarial.Exception.exception_AdveRS import AdveRSCheckpointError
from basat.algorithms.Adversarial.Exception.exception_StableOPT import StableOPTCheckpointError
from basat.experiment import ExperimentConfigurationError


PROTOCOL = "advers_2026_camera_ready"
torch.set_num_threads(1)


def make(method=0, repetition=0, directory=None, resume=False, *, protocol=PROTOCOL, beta=1.):
    seed = basat.derive_seed(2026, 2, method, repetition)
    benchmark = basat.AdveRSSynthetic()
    kernel = basat.make_kernel(
        basat.KernelConfig(name="polynomial", kwargs={"power": 6}), input_dimension=2,
        parameters={"offset": 2.84 / 1.11, "outputscale": 1.32 * 1.11**6}, dtype="float64",
    )
    gp = basat.GaussianProcessConfig(
        kernel=kernel, mean="zero", noise_mode="fixed", noise_variance=1.,
        standardize_outcomes=False, dtype="float64", jitter=1e-10,
    )
    common = dict(bounds=benchmark.bounds, dtype="float64", verbose=False, plot=False)
    config = dict(beta=beta, initial_points=10, fit_maxiter=0, gp_config=gp,
                  gp_backend="polynomial_features", source_protocol=protocol, seed=seed)
    if method < 3:
        algorithm = basat.AdveRS(**common, config=basat.AdveRSConfig(
            **config, variant=("RS1", "RS2", "RSG")[method], p=2. if method == 2 else 1.,
            aspiration=-10., finite_tie_break="first"))
    else:
        algorithm = basat.StableOPT(**common, config=basat.StableOPTConfig(
            **config, perturbation_radius=(.5, .25, 2.)[method - 3],
            query_policy="center", finite_tie_break="random"))
    environment = basat.AdversarialEnvironment(
        bounds=benchmark.bounds, function=benchmark, candidate_actions=benchmark.grid(),
        attack=basat.WorstCaseAttack(.5), noise_std=1., dtype="float64", seed=seed,
        allow_repeated_actions=True,
    )
    return basat.Experiment(
        environment=environment, algorithm=algorithm, evaluation_protocol="advers_2025",
        output_directory=directory, resume=resume, save_every=100,
    )


def trajectory(experiment):
    return [(row.audit["candidate_index"], row.audit["evaluated_candidate_index"],
             float(row.observation), row.audit["source_rng_after_feedback_sha256"])
            for row in experiment.environment.history.records]


@pytest.mark.parametrize("method", range(6))
def test_independent_camera_ready_gpy_trajectory_all_100_rounds(method):
    fixture = json.loads((Path(__file__).parent / "fixtures/advers_2026_camera_ready_worst.json").read_text())
    assert fixture["source_revision"] == "6de58579174a71a247bb92f7f920f015f3331ff9"
    expected = fixture["records"][method]
    experiment = make(method)
    metric = basat.RSRegret(aspiration=-10., optimal_fragility=20.129238666994205, epsilon=.5)
    result = experiment.run(max_rounds=100, regret=metric, plot=False, show=False)
    actual = trajectory(experiment)
    assert experiment._advers_bootstrap["indices"] == expected["initial_indices"]
    np.testing.assert_allclose(experiment._advers_bootstrap["observations"],
                               expected["initial_observations"], rtol=0, atol=1e-9)
    assert [row[0] for row in actual] == expected["proposed"]
    assert [row[1] for row in actual] == expected["evaluated"]
    np.testing.assert_allclose([row[2] for row in actual], expected["observations"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(result.metadata["regret"]["values"],
                               expected["cumulative_rs_regret"], rtol=0, atol=1e-8)
    assert [row[3] for row in actual] == expected["rng_after_feedback_sha256"]
    final_rng_digest = hashlib.sha256(json.dumps(
        experiment.algorithm.get_source_random_state(), sort_keys=True,
        separators=(",", ":"),
    ).encode()).hexdigest()
    assert final_rng_digest == expected["final_rng_sha256"]
    assert result.metadata["query_counts"] == {
        "initialization_queries": 10, "online_queries": 100, "total_objective_queries": 110}


@pytest.mark.parametrize("method", [0, 3])
@pytest.mark.parametrize("count", [10, 11])
def test_camera_ready_confidence_uses_std_and_squared_beta_setting(monkeypatch, method, count):
    """The actual source multiplier is sqrt(config.beta), with no variance switch."""
    experiment = make(method, beta=9.)
    algorithm = experiment.algorithm
    algorithm.state.y_train = torch.zeros(count, dtype=torch.float64)
    monkeypatch.setattr(algorithm, "_fit_model", lambda: None)
    if method == 0:
        monkeypatch.setattr(algorithm.gp, "predict", lambda points: (
            torch.full((len(points), 1), 2., dtype=torch.float64),
            torch.full((len(points), 1), 3., dtype=torch.float64)))
    else:
        monkeypatch.setattr(algorithm, "_posterior_marginals", lambda points: (
            torch.full((len(points),), 2., dtype=torch.float64),
            torch.full((len(points),), 3., dtype=torch.float64)))
    upper, lower = algorithm.confidence_bounds(experiment.environment.candidate_actions[:2])
    assert upper.tolist() == [11., 11.]
    assert lower.tolist() == [-7., -7.]


def test_camera_ready_rsg_euclidean_distance_changes_released_decision(monkeypatch):
    """Independent source fixture: Euclidean selects x=3; squared selects x=2."""
    points = torch.tensor([[0., 0.], [2., 0.], [3., 0.], [4., 0.]], dtype=torch.float64)
    values = torch.tensor([-26., -10., -10., -11.], dtype=torch.float64)
    for protocol, expected_index, expected_score in (
        (PROTOCOL, 2, 4. / 3.), ("advers_2025", 1, 1.),
    ):
        algorithm = make(2, protocol=protocol).algorithm
        monkeypatch.setattr(algorithm, "_optimistic_function", lambda _: values)
        point, score, _, _ = algorithm._source_finite_selection(points)
        assert torch.equal(point, points[expected_index])
        assert score == expected_score


@pytest.mark.parametrize("method", [0, 2])
def test_camera_ready_all_safe_fallback_matches_source_choice_and_rng(monkeypatch, method):
    algorithm = make(method).algorithm
    algorithm.set_source_random_state(basat.RandomStream(123).get_state())
    points = torch.tensor([[0., 0.], [1., 0.], [2., 0.], [3., 0.]], dtype=torch.float64)
    monkeypatch.setattr(algorithm, "_optimistic_function", lambda _: torch.zeros(4, dtype=torch.float64))
    point, score, _, _ = algorithm._source_finite_selection(points)
    # Unchanged adv_RS1.py at the pinned revision returns (2, -inf), seed123.
    assert torch.equal(point, points[2])
    assert score == -float("inf")
    expected = basat.RandomStream(123)
    expected.choice(4)
    assert algorithm.get_source_random_state() == expected.get_state()


@pytest.mark.parametrize("method", [0, 4])
@pytest.mark.parametrize("phase", ["initial_design", "initial_observation", "initialized",
                                  "pending_proposal", "pre_tell", "post_tell"])
def test_camera_ready_interrupted_resume_is_exact(tmp_path, monkeypatch, method, phase):
    reference = make(method)
    reference.run(max_rounds=4, plot=False, show=False)
    interrupted = make(method, directory=tmp_path)
    save = interrupted.save_checkpoint

    def crash(**kwargs):
        value = save(**kwargs)
        if kwargs.get("phase") == phase:
            raise RuntimeError("injected camera-ready interruption")
        return value

    monkeypatch.setattr(interrupted, "save_checkpoint", crash)
    with pytest.raises(RuntimeError, match="injected camera-ready"):
        interrupted.run(max_rounds=4, plot=False, show=False)
    resumed = make(method, directory=tmp_path, resume=True)
    resumed.run(max_rounds=4, plot=False, show=False)
    assert trajectory(resumed) == trajectory(reference)
    assert resumed.algorithm.get_source_random_state() == reference.algorithm.get_source_random_state()
    assert resumed._advers_bootstrap["version"] == 2
    assert resumed._advers_bootstrap["source_protocol"] == PROTOCOL
    assert resumed.environment.history.initial_data.metadata == {
        "generated_initialization": PROTOCOL, "protocol_version": 2}
    assert resumed.query_counts == {
        "initialization_queries": 10, "online_queries": 4, "total_objective_queries": 14}
    monkeypatch.setattr(resumed.environment, "evaluate_function", lambda *args: pytest.fail("extra objective call"))
    resumed.run(max_rounds=4, plot=False, show=False)
    assert trajectory(resumed) == trajectory(reference)


@pytest.mark.parametrize("method", [0, 4])
def test_source_version_and_rng_checkpoint_are_distinct(method):
    algorithm = make(method).algorithm
    state = algorithm._state_dict_impl()
    assert state["source_protocol_version"] == 2
    algorithm._validate_state_dict_impl(state, strict=True)
    for changed in (
        {"source_protocol": "advers_2025", "source_protocol_version": 1},
        {"source_protocol_version": 1},
        {"source_rng": None},
    ):
        incompatible = copy.deepcopy(state)
        incompatible.update(changed)
        with pytest.raises((ValueError, AdveRSCheckpointError, StableOPTCheckpointError)):
            algorithm._validate_state_dict_impl(incompatible, strict=True)
    restored = make(method).algorithm
    restored._load_state_dict_impl(state, strict=True)
    assert restored.get_source_random_state() == algorithm.get_source_random_state()


def test_initialization_protocol_validation_is_explicit():
    validate(make(), version=2)
    validate(make(protocol="advers_2025"), version=1)
    with pytest.raises(ExperimentConfigurationError, match="version is incompatible"):
        validate(make(), version=1)


def test_source_protocol_rejects_continuous_geometry():
    with pytest.raises(AlgorithmConfigurationError, match="full-domain Euclidean"):
        basat.AdveRS(
            bounds=basat.Bounds(lower=[0.], upper=[1.]),
            config=basat.AdveRSConfig(
                variant="RSG", aspiration=-10., source_protocol=PROTOCOL,
                perturbation_domain=basat.BallPerturbations(.5)))
