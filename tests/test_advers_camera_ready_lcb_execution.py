"""Latest official LCB trajectories through the ordinary transactional API.

The fixture comes from unchanged GPy synthetic.run at revision 6de58579, using
the retained LCB reconstruction seeds and all 100 online rounds for six methods.
It is independent of the worst-case fixture and never runs GPy inside pytest.
"""
import hashlib
import json
from pathlib import Path

import basat
import numpy as np
import pytest
import torch


PROTOCOL = "advers_2026_camera_ready"
REVISION = "6de58579174a71a247bb92f7f920f015f3331ff9"
FIXTURE = Path(__file__).parent / "fixtures/advers_2026_camera_ready_lcb.json"
torch.set_num_threads(1)


def make(method=0, repetition=0, directory=None, resume=False, gp_backend="polynomial_features"):
    seed = basat.derive_seed(2026, 1, method, repetition)
    benchmark = basat.AdveRSSynthetic()
    kernel = basat.make_kernel(
        basat.KernelConfig(name="polynomial", kwargs={"power": 6}), input_dimension=2,
        parameters={"offset": 2.84 / 1.11, "outputscale": 1.32 * 1.11**6}, dtype="float64",
    )
    if gp_backend == "fixed_gpy":
        kernel = basat.FixedPolynomialKernel(
            input_dimension=2, variance=1.32, scale=1.11, bias=2.84, order=6,
        )
    gp = basat.GaussianProcessConfig(
        kernel=kernel, mean="zero", noise_mode="fixed", noise_variance=1.,
        standardize_outcomes=False, dtype="float64", jitter=1e-10,
    )
    common = dict(bounds=benchmark.bounds, dtype="float64", verbose=False, plot=False)
    config = dict(beta=1., initial_points=10, fit_maxiter=0, gp_config=gp,
                  gp_backend=gp_backend, source_protocol=PROTOCOL, seed=seed)
    if gp_backend == "fixed_gpy":
        config["gp_training_regularization"] = 1e-8
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
        attack=basat.LCBAttack(.5), noise_std=1., dtype="float64", seed=seed,
        allow_repeated_actions=True,
    )
    return basat.Experiment(
        environment=environment, algorithm=algorithm, evaluation_protocol="advers_2025",
        output_directory=directory, resume=resume, save_every=100,
    )


def metric():
    return basat.RSRegret(aspiration=-10., optimal_fragility=20.129238666994205, epsilon=.5)


def rng_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def trajectory(experiment):
    return [(row.audit["candidate_index"], row.audit["evaluated_candidate_index"],
             float(row.observation), float(row.audit["latent_observation"]),
             row.audit["source_rng_after_feedback_sha256"])
            for row in experiment.environment.history.records]


@pytest.mark.parametrize("method", range(6))
def test_independent_camera_ready_lcb_gpy_trajectory_all_100_rounds(method):
    fixture = json.loads(FIXTURE.read_text())
    assert fixture["source_revision"] == REVISION
    assert fixture["scientific_controls"]["attack_mode"] == 1
    assert fixture["seed_rule"] == "numpy.SeedSequence([2026,1,method_index,repetition]).generate_state(1)[0]"
    assert fixture["runtime"]["uv_lock_exact"]
    expected = fixture["records"][method]
    assert expected["seed"] == basat.derive_seed(2026, 1, method, 0)
    experiment = make(method)
    assert isinstance(experiment.environment.attack, basat.LCBAttack)
    assert experiment.environment.attack.radius == .5
    result = experiment.run(max_rounds=100, regret=metric(), plot=False, show=False)
    actual = trajectory(experiment)
    assert len(actual) == 100
    assert experiment._advers_bootstrap["source_normal_policy"] == "advers-numpy126-scalar-polar-v1"
    assert experiment.environment.history.initial_data.metadata["source_normal_policy"] == "advers-numpy126-scalar-polar-v1"
    assert experiment._advers_bootstrap["indices"] == expected["initial_indices"]
    np.testing.assert_array_equal(experiment.environment.history.initial_data.actions,
                                  expected["initial_actions"])
    np.testing.assert_allclose(experiment._advers_bootstrap["observations"],
                               expected["initial_observations"], rtol=0, atol=1e-9)
    assert [row[0] for row in actual] == expected["proposed"]
    assert [row[1] for row in actual] == expected["evaluated"]
    np.testing.assert_allclose([row[2] for row in actual], expected["observations"], rtol=0, atol=1e-9)
    np.testing.assert_allclose([row[3] for row in actual], expected["latent_observations"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(result.metadata["regret"]["values"],
                               expected["cumulative_rs_regret"], rtol=0, atol=1e-8)
    assert [row[4] for row in actual] == expected["rng_after_feedback_sha256"]
    assert rng_digest(experiment.algorithm.get_source_random_state()) == expected["final_rng_sha256"]
    assert result.metadata["query_counts"] == {
        "initialization_queries": 10, "online_queries": 100, "total_objective_queries": 110}
    assert experiment.environment._runtime_confidence_bounds is None
    assert experiment.environment._runtime_source_random_stream is None
    assert all(row.audit["source_normal_protocol"] == "advers-numpy126-scalar-polar-v1"
               for row in experiment.environment.history.records)


@pytest.mark.parametrize("method", [0, 4])
@pytest.mark.parametrize("phase", ["initial_design", "initial_observation", "initialized",
                                  "pending_proposal", "pre_tell", "post_tell"])
def test_camera_ready_lcb_interruption_resume_and_completed_noop(tmp_path, monkeypatch, method, phase):
    reference = make(method)
    reference_result = reference.run(max_rounds=4, regret=metric(), plot=False, show=False)
    interrupted = make(method, directory=tmp_path)
    save = interrupted.save_checkpoint

    def crash(**kwargs):
        value = save(**kwargs)
        if kwargs.get("phase") == phase:
            raise RuntimeError("injected latest LCB interruption")
        return value

    monkeypatch.setattr(interrupted, "save_checkpoint", crash)
    with pytest.raises(RuntimeError, match="injected latest LCB"):
        interrupted.run(max_rounds=4, regret=metric(), plot=False, show=False)
    resumed = make(method, directory=tmp_path, resume=True)
    resumed_result = resumed.run(max_rounds=4, regret=metric(), plot=False, show=False)
    assert trajectory(resumed) == trajectory(reference)
    assert resumed.algorithm.get_source_random_state() == reference.algorithm.get_source_random_state()
    assert resumed_result.metadata["regret"]["values"] == reference_result.metadata["regret"]["values"]
    assert resumed._advers_bootstrap["version"] == 2
    assert resumed._advers_bootstrap["source_protocol"] == PROTOCOL
    assert resumed._advers_bootstrap["source_normal_policy"] == "advers-numpy126-scalar-polar-v1"
    assert resumed.environment.history.initial_data.metadata["source_normal_policy"] == "advers-numpy126-scalar-polar-v1"
    assert resumed.environment._runtime_confidence_bounds is None
    assert resumed.environment._runtime_source_random_stream is None
    assert resumed.query_counts == {
        "initialization_queries": 10, "online_queries": 4, "total_objective_queries": 14}
    monkeypatch.setattr(resumed.environment, "evaluate_function", lambda *args: pytest.fail("extra objective call"))
    monkeypatch.setattr(resumed.algorithm, "ask", lambda *args, **kwargs: pytest.fail("extra proposal"))
    resumed.run(max_rounds=4, regret=metric(), plot=False, show=False)
    assert trajectory(resumed) == trajectory(reference)


def test_lcb_source_fixture_is_independent_of_worst_case_fixture():
    fixture = json.loads(FIXTURE.read_text())
    assert len(fixture["records"]) == 6
    assert fixture["scientific_controls"]["physical_attack_radius"] == .5
    assert fixture["scientific_controls"]["rsg_p2_metric"] == "euclidean"
    for method, row in enumerate(fixture["records"]):
        assert row["seed"] == basat.derive_seed(2026, 1, method, 0)
        assert len(row["proposed"]) == len(row["evaluated"]) == len(row["attack_trace"]) == 100
        assert [attack["proposed"] for attack in row["attack_trace"]] == row["proposed"]
        assert [attack["evaluated"] for attack in row["attack_trace"]] == row["evaluated"]
        assert all(attack["physical_displacement"] <= .5 + 1e-12 for attack in row["attack_trace"])
