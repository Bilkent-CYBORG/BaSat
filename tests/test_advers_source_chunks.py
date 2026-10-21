"""Native CRC32-task streams continue across fresh, transactional learners."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import zlib

import numpy as np
import pytest
import torch

import basat
from basat._advers_source_protocol import digest
from basat.environments.Adversarial import _source_mt19937_normal
from basat.experiment import ExperimentCheckpointError, ExperimentConfigurationError

torch.set_num_threads(1)


def constant_objective(action):
    return 2.


def make(directory, *, resume=False, chunk=3, prefix="toy", seed=0, roles=None):
    roles = roles or {"RS1": "RS1", "RSG": "RSG"}
    bounds = basat.Bounds(lower=[0., 0.], upper=[1., 1.])
    grid = basat.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype="float64")
    algorithms, environments = {}, {}
    for label, role in roles.items():
        kernel = basat.FixedPolynomialKernel(input_dimension=2, scale=.5, bias=1., variance=1., order=2)
        gp = basat.GaussianProcessConfig(kernel=kernel, mean="zero", noise_mode="fixed", noise_variance=.01,
            standardize_outcomes=False, dtype="float64", device="cpu")
        algorithms[label] = basat.AdveRS(bounds=bounds, dtype="float64", device="cpu", verbose=False, plot=False,
            config=basat.AdveRSConfig(variant=role, aspiration=-100., p=2. if role == "RSG" else 1.,
                beta=0., initial_points=2, fit_maxiter=0, gp_config=gp, gp_backend="fixed_gpy",
                gp_training_regularization=1e-8, source_protocol="advers_2026_camera_ready", seed=123))
        environments[label] = basat.AdversarialEnvironment(bounds=bounds, function=constant_objective,
            candidate_actions=grid, attack=basat.LCBAttack(0.), noise_std=.1, dtype="float64", seed=123,
            allow_repeated_actions=True)
    return basat.Experiment(environment=environments, algorithm=algorithms, output_directory=directory,
        resume=resume, evaluation_protocol=basat.AdveRSSourceProtocol(method_roles=roles,
            seed=seed, chunk_size=chunk, task_prefix=prefix))


def run(experiment, *, ceiling=None, trials=4, rounds=3):
    return experiment.run(trials=trials, max_trials=ceiling, max_rounds=rounds,
        regret=basat.RSRegret(aspiration=-100., optimal_fragility=0., epsilon=.5),
        fail_fast=True, plot=False, show=False)


def load(path):
    return json.loads(path.read_text())


def native_artifacts(directory):
    return {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*")
            if path.is_file() and ("source-tasks" in path.parts or "trials" in path.parts)}


def test_crc32_exact_released_tag_and_seed_without_seedsequence():
    protocol = basat.AdveRSSourceProtocol({"label": "RS1"})
    assert protocol.task_tag("RS1", 0, 10) == "synthetic_RS1_atk1_runs_000_010"
    assert protocol.task_seed("RS1", 0, 10) == 1961
    assert protocol.task_seed("RS1", 10, 20) == zlib.crc32(b"synthetic_RS1_atk1_runs_010_020") % 100000


def test_native_task_stream_is_own_previous_child_end_not_reseeded(tmp_path):
    experiment = make(tmp_path)
    result = run(experiment)
    assert result.planned_trials == 4 and len(result.records) == 4
    assert set(result.results) == {"RS1", "RSG"} and all(len(runs) == 4 for runs in result.regret_runs.values())
    protocol = experiment._advers_source_protocol
    for label, role in protocol.method_roles.items():
        for start, stop in ((0, 3), (3, 4)):
            tag = protocol.task_tag(role, start, stop)
            expected = basat.RandomStream(protocol.task_seed(role, start, stop))
            for repetition in range(start, stop):
                directory = tmp_path / f"trials/trial-{repetition+1:03d}/algorithms" / ("01-rs1" if label == "RS1" else "02-rsg")
                entry = load(directory / "source-entry.json")
                assert entry["rng_state"] == expected.get_state()
                latest = load(directory / "latest.json")
                checkpoint = torch.load(directory / latest["checkpoint"], map_location="cpu", weights_only=False)
                initial = expected.choice(4, size=2, replace=False).tolist()
                assert checkpoint["experiment_state"]["advers_bootstrap"]["indices"] == initial
                records = checkpoint["environment_state"]["history"]["records"]
                for record in records:
                    proposed = expected.choice(4)
                    noise = .1 * _source_mt19937_normal(expected)
                    assert record["audit"]["candidate_index"] == proposed
                    assert float(record["observation"]) == 2.+noise
                assert checkpoint["algorithm_state"]["algorithm_state"]["source_rng"] == expected.get_state()
            task = load(tmp_path / "source-tasks" / tag / "state.json")
            assert task["status"] == "completed" and task["rng_state"] == expected.get_state()
    experiment.save(tmp_path / "native-results.json")
    assert load(tmp_path / "native-results.json")["completed_trials"] == 4


def test_pilot_ceiling_preserves_planned_task_stop_seed_and_resumes_each_method(tmp_path):
    result = run(make(tmp_path), ceiling=1)
    assert len(result.records) == 1
    for role in ("RS1", "RSG"):
        tag = f"toy_{role}_atk1_runs_000_003"
        state = load(tmp_path / "source-tasks" / tag / "state.json")
        assert state["next_repetition"] == 1 and state["stop"] == 3
        assert not (tmp_path / "source-tasks" / f"toy_{role}_atk1_runs_000_001").exists()
    outcome = run(make(tmp_path, resume=True))
    assert len(outcome.records) == 4


@pytest.mark.parametrize("phase", ["initial_design", "initial_observation", "initialized", "pending_proposal", "pre_tell", "post_tell"])
def test_interrupt_resume_uses_ordinary_durable_child_without_extra_queries(tmp_path, monkeypatch, phase):
    baseline = run(make(tmp_path / "baseline"), trials=2)
    original = basat.Experiment.save_checkpoint
    fired = []
    def crash(self, **kwargs):
        value = original(self, **kwargs)
        if kwargs.get("phase") == phase and self._advers_source_entry is not None and not fired:
            fired.append(True)
            raise RuntimeError("source child interrupted after durable boundary")
        return value
    monkeypatch.setattr(basat.Experiment, "save_checkpoint", crash)
    with pytest.raises(RuntimeError, match="interrupted"):
        run(make(tmp_path / "interrupted"), trials=2)
    monkeypatch.setattr(basat.Experiment, "save_checkpoint", original)
    restored = run(make(tmp_path / "interrupted", resume=True), trials=2)
    for label in baseline.results:
        for expected, actual in zip(baseline.records, restored.records):
            assert actual["algorithms"][label]["observations"] == expected["algorithms"][label]["observations"]
            assert actual["algorithms"][label]["query_counts"] == expected["algorithms"][label]["query_counts"]
    before = native_artifacts(tmp_path / "interrupted")
    monkeypatch.setattr(basat.AdversarialEnvironment, "evaluate_function", lambda *args, **kwargs: pytest.fail("extra objective query"))
    run(make(tmp_path / "interrupted", resume=True), trials=2)
    assert native_artifacts(tmp_path / "interrupted") == before


def test_completed_child_before_parent_cursor_commit_is_recovered_without_reexecution(tmp_path, monkeypatch):
    original = basat.Experiment._atomic_json_save
    fired = []
    def crash(document, path):
        if Path(path).name == "state.json" and document.get("next_repetition") == 1 and not fired:
            fired.append(True)
            raise RuntimeError("parent cursor crash")
        return original(document, path)
    monkeypatch.setattr(basat.Experiment, "_atomic_json_save", staticmethod(crash))
    with pytest.raises(RuntimeError, match="parent cursor"):
        run(make(tmp_path), trials=2)
    monkeypatch.setattr(basat.Experiment, "_atomic_json_save", staticmethod(original))
    outcome = run(make(tmp_path, resume=True), trials=2)
    assert len(outcome.records) == 2


@pytest.mark.parametrize("change", ["seed", "prefix", "chunk", "trials"])
def test_source_plan_identity_rejects_changed_schedule(tmp_path, change):
    run(make(tmp_path), ceiling=1)
    changes = {"seed": {"seed": 1}, "prefix": {"prefix": "other"}, "chunk": {"chunk": 2}, "trials": {}}[change]
    with pytest.raises(ExperimentCheckpointError, match="configuration/runtime changed"):
        run(make(tmp_path, resume=True, **changes), trials=5 if change == "trials" else 4)


def test_corrupt_task_chain_and_entry_are_rejected_before_queries(tmp_path, monkeypatch):
    run(make(tmp_path), ceiling=1)
    state_path = tmp_path / "source-tasks/toy_RS1_atk1_runs_000_003/state.json"
    state = load(state_path)
    state["rng_state"]["position"] = (state["rng_state"]["position"]+1) % 624
    basat.Experiment._atomic_json_save(state, state_path)
    monkeypatch.setattr(basat.AdversarialEnvironment, "evaluate_function", lambda *args, **kwargs: pytest.fail("query before chain validation"))
    with pytest.raises(ExperimentCheckpointError, match="not its native child chain"):
        run(make(tmp_path, resume=True))


def test_completed_entry_and_regret_identity_cannot_change_on_resume(tmp_path, monkeypatch):
    run(make(tmp_path), ceiling=1)
    monkeypatch.setattr(basat.AdversarialEnvironment, "evaluate_function", lambda *args, **kwargs: pytest.fail("query before identity validation"))
    with pytest.raises(ExperimentCheckpointError, match="configuration/runtime changed"):
        make(tmp_path, resume=True).run(trials=4, max_rounds=3, fail_fast=True,
            regret=basat.RSRegret(aspiration=-99., optimal_fragility=0., epsilon=.5), plot=False, show=False)
    entry_path = tmp_path / "trials/trial-001/algorithms/01-rs1/source-entry.json"
    entry = load(entry_path)
    entry["repetition"] = 19
    basat.Experiment._atomic_json_save(entry, entry_path)
    with pytest.raises(ExperimentCheckpointError, match="checksum or entry identity"):
        run(make(tmp_path, resume=True))


def test_ordinary_independent_trial_api_is_not_repurposed(tmp_path):
    experiment = make(tmp_path)
    with pytest.raises(ExperimentConfigurationError, match="own their shared"):
        experiment.run(trials=4, trial_seeds=[1,2,3,4], max_rounds=3, fail_fast=True)
    with pytest.raises(ExperimentConfigurationError, match="fail_fast=True"):
        experiment.run(trials=4, max_rounds=3)


@pytest.mark.skipif(np.__version__ != "1.26.4", reason="Exact released numerical stack is explicitly opt-in.")
def test_first_two_released_rs1_runs_match_fresh_native_protocol_not_trace_training(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "results/reproducibility/advers2026_camera_ready_lcb_released_chunks_v1/source_validation/method_0"
    if not (source / "run_001.json").is_file():
        pytest.skip("Independent released chunk records are not present.")
    benchmark = basat.AdveRSSynthetic()
    kernel = basat.FixedPolynomialKernel(input_dimension=2, scale=1.11, bias=2.84, variance=1.32, order=6)
    gp = basat.GaussianProcessConfig(kernel=kernel, mean="zero", noise_mode="fixed", noise_variance=1.,
        standardize_outcomes=False, dtype="float64", jitter=1e-10)
    algorithm = basat.AdveRS(bounds=benchmark.bounds, dtype="float64", plot=False, verbose=False,
        config=basat.AdveRSConfig(variant="RS1", aspiration=-10., beta=1., initial_points=10,
            fit_maxiter=0, gp_config=gp, gp_backend="fixed_gpy", gp_training_regularization=1e-8,
            source_protocol="advers_2026_camera_ready", finite_tie_break="first"))
    environment = basat.AdversarialEnvironment(bounds=benchmark.bounds, function=benchmark,
        candidate_actions=benchmark.grid(), attack=basat.LCBAttack(.5), noise_std=1.,
        allow_repeated_actions=True, dtype="float64")
    experiment = basat.Experiment(environment={"RS1": environment}, algorithm={"RS1": algorithm},
        output_directory=tmp_path, evaluation_protocol=basat.AdveRSSourceProtocol({"RS1": "RS1"}))
    outcome = experiment.run(trials=100, max_trials=2, max_rounds=100,
        regret=basat.RSRegret(aspiration=-10., optimal_fragility=20.129238666994205, epsilon=.5),
        fail_fast=True, plot=False, show=False)
    assert len(outcome.records) == 2
    for repetition in range(2):
        expected = load(source / f"run_{repetition:03d}.json")  # Read only AFTER native decisions.
        directory = tmp_path / f"trials/trial-{repetition+1:03d}/algorithms/01-rs1"
        latest = load(directory / "latest.json")
        checkpoint = torch.load(directory / latest["checkpoint"], map_location="cpu", weights_only=False)
        records = checkpoint["environment_state"]["history"]["records"]
        assert checkpoint["experiment_state"]["advers_bootstrap"]["indices"] == expected["initial_indices"]
        assert [row["audit"]["candidate_index"] for row in records] == expected["proposed"]
        assert [row["audit"]["evaluated_candidate_index"] for row in records] == expected["evaluated"]
        np.testing.assert_allclose([row["observation"] for row in records], expected["observations"], rtol=0, atol=1e-9)
        assert [row["audit"]["source_rng_after_feedback_sha256"] for row in records] == expected["rng_after_feedback_sha256"]
        assert checkpoint["algorithm_state"]["algorithm_state"]["source_rng"] == expected["rng_state"]
