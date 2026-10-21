"""Worst-case source-policy execution, durable recovery and clean accounting."""
import numpy as np
import pytest
import torch

import basat
from basat.environments.environment import EnvironmentConfigurationError

torch.set_num_threads(1)

# Independently executed unchanged pinned AdveRS GPy loops (attack_mode=2),
# source revision 71a3fe89c7d051b7646d048fb6fc314e74a33095. These are the first
# five online observations of complete 100-round runs, not hand-picked runs.
FIXTURES = [
    ([417,932,606,89,416,914,987,18,798,158], [1018,997,597,383,0],
     [1015,994,693,287,0], [-32.561530498238994,-40.26131730153719,-33.64455172004636,-36.66364870267335,-46.24416039459717]),
    ([1000,108,134,469,272,167,137,115,704,677], [1023,0,328,328,328],
     [1023,0,424,424,424], [-46.2399533134142,-46.86128925171553,-7.503765965394547,-7.691955835315013,-9.243793604872856]),
    ([835,354,554,498,167,419,467,98,858,432], [0,1023,31,1003,640],
     [0,1023,31,998,672], [-48.29814182636905,-43.16986838689671,-66.57514369047122,-29.44960904537919,-57.42278628203628]),
    ([101,872,514,160,870,927,880,790,337,709], [31,511,1018,223,992],
     [31,607,1015,127,992], [-67.9821912591095,-43.700708369952395,-30.96388645413569,-39.50869245461649,-64.59315276616451]),
    ([471,580,55,233,612,426,792,69,486,306], [1023,1009,992,255,1000],
     [1023,1012,992,159,997], [-44.643968160873676,-31.890135148112865,-63.56756519561515,-37.07711533141128,-30.888572099316672]),
    ([978,1,474,691,45,521,844,262,102,60], [896,959,481,300,270],
     [992,1023,576,335,305], [-63.872941295218226,-45.21989273542972,-52.31618061764111,-9.931687012027513,-13.29665872706137]),
]


def make(method=0, repetition=0, directory=None, resume=False, radius=None):
    seed = basat.derive_seed(2026, 2, method, repetition)
    benchmark = basat.AdveRSSynthetic()
    kernel = basat.make_kernel(
        basat.KernelConfig(name="polynomial", kwargs={"power": 6}), input_dimension=2,
        parameters={"offset": 2.84/1.11, "outputscale": 1.32*1.11**6}, dtype="float64",
    )
    gp = basat.GaussianProcessConfig(
        kernel=kernel, mean="zero", noise_mode="fixed", noise_variance=1.,
        standardize_outcomes=False, dtype="float64", jitter=1e-10,
    )
    common = dict(bounds=benchmark.bounds, dtype="float64", verbose=False, plot=False)
    config = dict(beta=4., initial_points=10, fit_maxiter=0, gp_config=gp,
                  gp_backend="polynomial_features", source_protocol="advers_2025", seed=seed)
    if method < 3:
        algorithm = basat.AdveRS(**common, config=basat.AdveRSConfig(
            **config, variant=("RS1", "RS2", "RSG")[method], p=2. if method == 2 else 1.,
            aspiration=-10., finite_tie_break="first"))
    else:
        algorithm = basat.StableOPT(**common, config=basat.StableOPTConfig(
            **config, perturbation_radius=(.5, .25, 2.)[method-3],
            query_policy="center", finite_tie_break="random"))
    environment = basat.AdversarialEnvironment(
        bounds=benchmark.bounds, function=benchmark, candidate_actions=benchmark.grid(),
        attack=basat.WorstCaseAttack((.5**.5 if method == 2 else .5) if radius is None else radius),
        noise_std=1., dtype="float64", seed=seed, allow_repeated_actions=True,
    )
    return basat.Experiment(environment=environment, algorithm=algorithm,
                            evaluation_protocol="advers_2025", output_directory=directory,
                            resume=resume, save_every=100)


def trajectory(experiment):
    return [(r.audit["candidate_index"], r.audit["evaluated_candidate_index"],
             float(r.observation)) for r in experiment.environment.history.records]


@pytest.mark.parametrize("method", range(6))
def test_pinned_worst_case_source_trajectories(method):
    initial, proposed, attacked, observations = FIXTURES[method]
    experiment = make(method)
    result = experiment.run(max_rounds=5, plot=False, show=False)
    actual = trajectory(experiment)
    assert experiment._advers_bootstrap["indices"] == initial
    assert [r[0] for r in actual] == proposed
    assert [r[1] for r in actual] == attacked
    np.testing.assert_allclose([r[2] for r in actual], observations, rtol=0, atol=1e-9)
    assert result.metadata["query_counts"] == {
        "initialization_queries": 10, "online_queries": 5, "total_objective_queries": 15}
    assert experiment.environment.attack_oracle_metadata["preparation_evaluations"] == 1024
    assert experiment.algorithm.evaluations == 5  # Online feedback only, never oracle rows.
    assert len(experiment.algorithm.state.y_train) == 15
    state_before_prediction = experiment.algorithm.get_source_random_state()
    # StableOPT conditions lazily at the next proposal/inspection, not tell.
    experiment.algorithm.confidence_bounds(experiment.environment.candidate_actions)
    assert len(experiment.algorithm.gp.X_train) == 15
    assert experiment.algorithm.get_source_random_state() == state_before_prediction


@pytest.mark.parametrize("method", [0, 4])
@pytest.mark.parametrize("phase", ["initial_design", "initial_observation", "initialized",
                                  "pending_proposal", "pre_tell", "post_tell"])
def test_interrupt_and_exact_resume(tmp_path, monkeypatch, method, phase):
    reference = make(method)
    reference.run(max_rounds=4, plot=False, show=False)
    interrupted = make(method, directory=tmp_path)
    original_save = interrupted.save_checkpoint

    def crash(**kwargs):
        value = original_save(**kwargs)
        if kwargs.get("phase") == phase:
            raise RuntimeError("injected durable interruption")
        return value

    monkeypatch.setattr(interrupted, "save_checkpoint", crash)
    with pytest.raises(RuntimeError, match="injected durable"):
        interrupted.run(max_rounds=4, plot=False, show=False)
    recovered = make(method, directory=tmp_path, resume=True)
    recovered.run(max_rounds=4, plot=False, show=False)
    assert trajectory(recovered) == trajectory(reference)
    assert recovered.algorithm.get_source_random_state() == reference.algorithm.get_source_random_state()
    assert recovered.environment.attack_oracle_metadata == reference.environment.attack_oracle_metadata
    before = recovered.query_counts
    monkeypatch.setattr(recovered.environment, "evaluate_function", lambda *args: pytest.fail("extra objective call"))
    recovered.run(max_rounds=4, plot=False, show=False)
    assert recovered.query_counts == before


def test_completed_reconstruction_and_changed_attack_checkpoint(tmp_path, monkeypatch):
    original = make(directory=tmp_path)
    original.run(max_rounds=2, plot=False, show=False)
    recovered = make(directory=tmp_path, resume=True)
    monkeypatch.setattr(recovered.environment, "evaluate_function", lambda *args: pytest.fail("extra objective call"))
    recovered.run(max_rounds=2, plot=False, show=False)
    assert trajectory(recovered) == trajectory(original)
    with pytest.raises(EnvironmentConfigurationError, match="source_fingerprint"):
        make(directory=tmp_path, resume=True, radius=.25)


def test_exact_110_queries_and_latent_common_rs1_regret():
    experiment = make()
    metric = basat.RSRegret(aspiration=-10., optimal_fragility=20.129238666994205, epsilon=.5)
    result = experiment.run(max_rounds=100, regret=metric, plot=False, show=False)
    assert experiment.query_counts == {
        "initialization_queries": 10, "online_queries": 100, "total_objective_queries": 110}
    assert len(experiment.environment.history.initial_data.observations) == 10
    assert len(experiment.environment.history.records) == 100
    latent = [float(r.audit["latent_observation"]) for r in experiment.environment.history.records]
    np.testing.assert_allclose(result.metadata["regret"]["values"], metric.evaluate(latent), rtol=0, atol=1e-8)
    assert result.metadata["regret"]["values"][-1] == pytest.approx(85.74711078918716, abs=1e-8)


def test_generated_initialization_budget_and_private_oracle(tmp_path):
    experiment = make(directory=tmp_path)
    experiment.run(max_evaluations=4, plot=False, show=False)
    assert experiment.query_counts == {
        "initialization_queries": 4, "online_queries": 0, "total_objective_queries": 4}
    assert experiment.environment.attack_oracle_metadata["preparation_evaluations"] == 0
    resumed = make(directory=tmp_path, resume=True)
    resumed.run(max_evaluations=12, plot=False, show=False)
    assert resumed.query_counts == {
        "initialization_queries": 10, "online_queries": 2, "total_objective_queries": 12}
    assert resumed.algorithm.evaluations == 2
    public = resumed.environment.before_action()
    assert "candidate_actions" in public
    assert not any(name in public for name in ["values", "payoffs", "truth", "worst_case_cache"])
    assert not any(name in resumed.environment.attack_oracle_metadata for name in ["values", "payoffs", "truth"])


def test_method_state_is_independent():
    first, second = make(0), make(4)
    first.run(max_rounds=2, plot=False, show=False)
    first_state = first.algorithm.get_source_random_state()
    before = trajectory(first)
    second.run(max_rounds=2, plot=False, show=False)
    assert trajectory(first) == before
    assert first.algorithm.get_source_random_state() == first_state
    assert first.environment.history is not second.environment.history
    assert first._advers_bootstrap["indices"] != second._advers_bootstrap["indices"]
