"""Serial native state handoffs, durable prefix/LCB separation and no-op guards."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics.pairwise import euclidean_distances
import torch

import basat
import basat._advers_serial_protocol as serial
from basat.environments.Adversarial import _source_mt19937_normal
from basat.experiment import ExperimentCheckpointError, ExperimentConfigurationError
from basat.models.FixedKernelGaussianProcess import FixedKernelGaussianProcess

torch.set_num_threads(1)


def objective(action):
    return 2.


def make(directory, *, resume=False, seed=0, radius=1.01):
    bounds = basat.Bounds(lower=[0., 0.], upper=[1., 1.])
    grid = basat.array([[0.,0.], [0.,1.], [1.,0.], [1.,1.]], dtype="float64")
    algorithms, environments = {}, {}
    for role in serial.ROLES:
        kernel = basat.FixedPolynomialKernel(input_dimension=2, scale=.5, bias=1., variance=1., order=2)
        gp = basat.GaussianProcessConfig(kernel=kernel, mean="zero", noise_mode="fixed", noise_variance=.01,
            standardize_outcomes=False, dtype="float64", device="cpu")
        shared = dict(beta=0.,initial_points=2,fit_maxiter=0,gp_config=gp,gp_backend="fixed_gpy",
            gp_training_regularization=1e-8,source_protocol="advers_2026_camera_ready",seed=123)
        options = dict(bounds=bounds,dtype="float64",device="cpu",plot=False,verbose=False)
        if role.startswith("RO_"):
            config = basat.StableOPTConfig(perturbation_radius=.5,query_policy="center",finite_tie_break="first",**shared)
            algorithms[role] = basat.StableOPT(config=config,**options)
        else:
            config = basat.AdveRSConfig(variant=role,aspiration=-100.,p=2. if role=="RSG" else 1.,**shared)
            algorithms[role] = basat.AdveRS(config=config,**options)
        environments[role] = basat.AdversarialEnvironment(bounds=bounds,function=objective,
            candidate_actions=grid,attack=basat.LCBAttack(radius),noise_std=.1,allow_repeated_actions=True,
            dtype="float64",device="cpu",seed=123)
    return basat.Experiment(environment=environments,algorithm=algorithms,output_directory=directory,
        resume=resume,evaluation_protocol=serial.AdveRSSerialSourceProtocol({role:role for role in serial.ROLES},seed=seed))


def run(experiment, *, ceiling=None, trials=2, rounds=2):
    return experiment.run(trials=trials,max_source_runs=ceiling,max_rounds=rounds,
        regret=basat.RSRegret(aspiration=-100.,optimal_fragility=0.,epsilon=.5),
        fail_fast=True,plot=False,show=False)


def load(path):
    return json.loads(path.read_text())


def artifacts(root):
    return {str(path.relative_to(root)):path.read_bytes() for path in root.rglob("*") if path.is_file()
            and ("trials" in path.parts or path.name in ("source-serial-protocol.json","source-serial-state.json"))}


def checkpoint(directory):
    latest=load(directory/"latest.json")
    return torch.load(directory/latest["checkpoint"],map_location="cpu",weights_only=False)


def test_public_protocol_requires_complete_source_order_and_uint32_seed():
    roles={role:role for role in serial.ROLES}
    assert serial.AdveRSSerialSourceProtocol(roles).seed==0
    assert basat.AdveRSSerialSourceProtocol is serial.AdveRSSerialSourceProtocol
    for bad in ({"RS1":"RS1"},dict(reversed(list(roles.items())))):
        with pytest.raises(ValueError,match="six unique labels"):
            serial.AdveRSSerialSourceProtocol(bad)
    for seed in (-1,2**32,True):
        with pytest.raises(ValueError,match="seed"):
            serial.AdveRSSerialSourceProtocol(roles,seed=seed)


def test_flat_ceiling_cannot_skip_random_prefix_repetitions(tmp_path):
    empty=run(make(tmp_path),ceiling=0)
    assert empty.source_summary["completed_source_runs"]==0 and empty.records==()
    assert not (tmp_path/"prefix/trials").exists()
    outcome=run(make(tmp_path,resume=True),ceiling=1)
    assert outcome.records==() and outcome.regret_runs=={}
    state=load(tmp_path/"source-serial-state.json")
    assert state["next_source_run"]==1
    assert (state["phase"],state["method_index"],state["repetition"])==("random_prefix",0,1)
    assert not (tmp_path/"trials").exists()
    assert len(list((tmp_path/"prefix/trials").glob("trial-*/algorithms/*/source-completed.json")))==1
    prefix=run(make(tmp_path,resume=True),ceiling=12)
    assert prefix.source_summary["completed_prefix_runs"]==12
    assert prefix.source_summary["completed_lcb_runs"]==0 and prefix.records==()
    state=load(tmp_path/"source-serial-state.json")
    assert (state["phase"],state["method_index"],state["repetition"])==("lcb",0,0)
    outcome=run(make(tmp_path,resume=True))
    assert len(outcome.records)==2
    assert all(len(rows)==2 for rows in outcome.regret_runs.values())
    assert outcome.source_summary["completed_source_runs"]==24
    assert outcome.source_summary["completed_prefix_runs"]==outcome.source_summary["completed_lcb_runs"]==12
    assert all(row["source_schedule"]=="advers_serial_random_prefix_then_lcb" and row["plotted_phase"]=="lcb"
               for row in outcome.records)
    assert outcome.source_summary["query_counts_by_phase"]=={
        phase:dict(initialization_queries=24,online_queries=24,total_objective_queries=48) for phase in serial.PHASES}


def test_whole_native_mt_chain_across_every_method_phase_and_repetition(tmp_path,monkeypatch):
    # Released RO deliberately samples exact center ties. Observe the native
    # tie sets without changing selection, rather than assuming zero RO draws.
    original=basat.StableOPT._best_position
    native_ties=[]
    def observe(self,values,points,*,maximize,dataset_indices=None):
        if dataset_indices is not None:
            finite=self._finite_positions(values)
            target=values[finite].max() if maximize else values[finite].min()
            native_ties.append((self.get_source_random_state(),finite[values[finite]==target].tolist()))
        return original(self,values,points,maximize=maximize,dataset_indices=dataset_indices)
    monkeypatch.setattr(basat.StableOPT,"_best_position",observe)
    experiment=make(tmp_path)
    run(experiment)
    expected=basat.RandomStream(0)
    grid=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    for spec in serial._plan(experiment._advers_serial_source_protocol,2):
        for repetition in range(2):
            directory=serial._child_root(experiment,spec,repetition)
            entry=load(directory/"source-entry.json")
            assert entry["task_seed"]==0 and entry["rng_state"]==expected.get_state()
            saved=checkpoint(directory)
            assert saved["experiment_state"]["advers_bootstrap"]["indices"]==expected.choice(4,size=2,replace=False).tolist()
            for record in saved["environment_state"]["history"]["records"]:
                proposed=record["audit"]["candidate_index"]
                if not spec["official_algorithm"].startswith("RO_"):
                    assert proposed==expected.choice(4)
                else:
                    before,tied=native_ties.pop(0)
                    assert before==expected.get_state()
                    assert proposed==expected.choice(tied)
                if spec["phase"]=="random_prefix":
                    squared=spec["official_algorithm"] not in ("RS1","RSG")
                    distances=euclidean_distances(grid,grid[proposed].reshape(1,-1),squared=squared).reshape(-1)
                    within=np.flatnonzero(distances <= (1.01**2 if squared else 1.01))
                    assert record["audit"]["evaluated_candidate_index"]==expected.choice(within)
                assert float(record["observation"])==2.+.1*_source_mt19937_normal(expected)
            assert saved["algorithm_state"]["algorithm_state"]["source_rng"]==expected.get_state()
    state=load(tmp_path/"source-serial-state.json")
    assert state["rng_state"]==expected.get_state() and state["status"]=="completed"
    assert state["phase"]=="completed" and state["next_source_run"]==24
    assert native_ties==[]
    assert all(isinstance(environment.attack,basat.LCBAttack) and not environment.history.records
               for environment in experiment.source_environments.values())


@pytest.mark.parametrize("phase",["initial_design","initial_observation","initialized","pending_proposal","pre_tell","post_tell"])
def test_interrupt_resumes_ordinary_child_without_duplicate_queries(tmp_path,monkeypatch,phase):
    baseline=run(make(tmp_path/"baseline"),trials=1)
    original=basat.Experiment.save_checkpoint
    fired=[]
    def interrupt(self,**kwargs):
        result=original(self,**kwargs)
        if kwargs.get("phase")==phase and self._advers_source_entry is not None and not fired:
            fired.append(True)
            raise RuntimeError("serial durable child interruption")
        return result
    monkeypatch.setattr(basat.Experiment,"save_checkpoint",interrupt)
    with pytest.raises(RuntimeError,match="interruption"):
        run(make(tmp_path/"interrupted"),trials=1)
    monkeypatch.setattr(basat.Experiment,"save_checkpoint",original)
    restored=run(make(tmp_path/"interrupted",resume=True),trials=1)
    assert restored.observations==baseline.observations
    expected=load(tmp_path/"baseline/source-serial-state.json")
    actual=load(tmp_path/"interrupted/source-serial-state.json")
    assert actual["rng_state"]==expected["rng_state"]
    for root in ("prefix/trials","trials"):
        for path in (tmp_path/"baseline"/root).glob("trial-*/algorithms/*/result.json"):
            other=tmp_path/"interrupted"/path.relative_to(tmp_path/"baseline")
            assert load(path)["observations"]==load(other)["observations"]
            assert load(path)["query_counts"]==load(other)["query_counts"]


def test_child_committed_before_cursor_crash_is_not_reexecuted(tmp_path,monkeypatch):
    original=basat.Experiment._atomic_json_save
    fired=[]
    def interrupted(document,path):
        if Path(path).name=="source-serial-state.json" and document.get("next_source_run")==1 and not fired:
            fired.append(True)
            raise RuntimeError("serial parent cursor interruption")
        return original(document,path)
    monkeypatch.setattr(basat.Experiment,"_atomic_json_save",staticmethod(interrupted))
    with pytest.raises(RuntimeError,match="cursor interruption"):run(make(tmp_path),ceiling=1)
    assert load(tmp_path/"source-serial-state.json")["next_source_run"]==0
    assert load(tmp_path/"source-serial-summary.json")["completed_source_runs"]==0
    first=tmp_path/"prefix/trials/trial-001/algorithms/01-rs1"
    before={p.name:p.read_bytes() for p in first.iterdir() if p.is_file()}
    monkeypatch.setattr(basat.Experiment,"_atomic_json_save",staticmethod(original))
    monkeypatch.setattr(basat.AdversarialEnvironment,"evaluate_function",lambda *args,**kwargs:pytest.fail("reexecuted committed child"))
    outcome=run(make(tmp_path,resume=True),ceiling=1)
    assert outcome.source_summary["completed_source_runs"]==1
    assert {p.name:p.read_bytes() for p in first.iterdir() if p.is_file()}==before


def test_report_failure_after_parent_commit_does_not_relabel_scientific_child(tmp_path,monkeypatch):
    original=basat.Experiment._atomic_json_save
    def failed_summary(document,path):
        if Path(path).name=="source-serial-summary.json" and document.get("completed_source_runs")==1:
            raise OSError("report-only summary write interrupted")
        return original(document,path)
    monkeypatch.setattr(basat.Experiment,"_atomic_json_save",staticmethod(failed_summary))
    with pytest.raises(OSError,match="report-only summary"):run(make(tmp_path),ceiling=1)
    state=load(tmp_path/"source-serial-state.json")
    assert state["next_source_run"]==1 and state["status"]=="running" and state["active_failure"] is None
    before=artifacts(tmp_path)
    monkeypatch.setattr(basat.Experiment,"_atomic_json_save",staticmethod(original))
    monkeypatch.setattr(basat.AdversarialEnvironment,"evaluate_function",lambda *args,**kwargs:pytest.fail("report failure repeated scientific query"))
    outcome=run(make(tmp_path,resume=True),ceiling=1)
    assert outcome.source_summary["completed_source_runs"]==1
    assert artifacts(tmp_path)==before


def test_completed_noop_never_constructs_child_or_calls_science(tmp_path,monkeypatch):
    run(make(tmp_path),trials=1)
    before=artifacts(tmp_path)
    for owner,names in ((basat.AdversarialEnvironment,("evaluate_function",)),
                        (basat.AdveRS,("ask","tell")),(basat.StableOPT,("ask","tell")),
                        (FixedKernelGaussianProcess,("fit","predict","predict_marginals")),
                        (basat.FixedPolynomialKernel,("forward",))):
        for name in names:
            monkeypatch.setattr(owner,name,lambda *args,**kwargs:pytest.fail("completed serial scientific call"))
    outcome=run(make(tmp_path,resume=True),trials=1)
    assert outcome.source_summary["completed_source_runs"]==12
    assert artifacts(tmp_path)==before


def test_low_disk_blocks_new_child_persists_cursor_but_allows_completed_readonly_return(tmp_path,monkeypatch):
    run(make(tmp_path),ceiling=1)
    first=tmp_path/"prefix/trials/trial-001/algorithms/01-rs1"
    before={path.name:path.read_bytes() for path in first.iterdir() if path.is_file()}
    monkeypatch.setattr(serial.shutil,"disk_usage",lambda path:SimpleNamespace(free=serial.MINIMUM_FREE_DISK_BYTES-1))
    monkeypatch.setattr(basat.AdversarialEnvironment,"evaluate_function",lambda *args,**kwargs:pytest.fail("low-disk objective query"))
    with pytest.raises(ExperimentCheckpointError,match="at least 2 GiB"):
        run(make(tmp_path,resume=True),ceiling=2)
    state=load(tmp_path/"source-serial-state.json")
    assert state["next_source_run"]==1 and state["status"]=="failed"
    assert state["active_failure"]["source_run"]==1
    assert not (tmp_path/"prefix/trials/trial-002/algorithms/01-rs1/latest.json").exists()
    assert {path.name:path.read_bytes() for path in first.iterdir() if path.is_file()}==before
    assert run(make(tmp_path,resume=True),ceiling=1).source_summary["completed_source_runs"]==1


@pytest.mark.parametrize("change",["seed","trials","radius","metric","implementation"])
def test_changed_scientific_identity_rejected_before_queries(tmp_path,monkeypatch,change):
    run(make(tmp_path),ceiling=1)
    monkeypatch.setattr(basat.AdversarialEnvironment,"evaluate_function",lambda *args,**kwargs:pytest.fail("identity mismatch queried"))
    if change=="implementation":
        original=serial._library_source_identity
        def changed(root):
            value=original(root)
            value["files"]["models/FixedKernelGaussianProcess.py"]="0"*64
            value["sha256"]="0"*64
            return value
        monkeypatch.setattr(serial,"_library_source_identity",changed)
    kwargs={"seed":1} if change=="seed" else {"radius":1.1} if change=="radius" else {}
    experiment=make(tmp_path,resume=True,**kwargs)
    with pytest.raises(ExperimentCheckpointError,match="configuration/runtime/implementation changed"):
        if change=="metric":
            experiment.run(trials=2,max_rounds=2,fail_fast=True,
                regret=basat.RSRegret(aspiration=-99.,optimal_fragility=0.,epsilon=.5),plot=False,show=False)
        else: run(experiment,trials=3 if change=="trials" else 2)


def test_corrupt_flat_cursor_rng_chain_rejected_before_queries(tmp_path,monkeypatch):
    run(make(tmp_path),ceiling=1)
    path=tmp_path/"source-serial-state.json"
    state=load(path)
    state["rng_state"]["position"]=(state["rng_state"]["position"]+1)%624
    basat.Experiment._atomic_json_save(state,path)
    monkeypatch.setattr(basat.AdversarialEnvironment,"evaluate_function",lambda *args,**kwargs:pytest.fail("corrupt chain queried"))
    with pytest.raises(ExperimentCheckpointError,match="not its own native child chain"):
        run(make(tmp_path,resume=True))


def test_full_implementation_identity_covers_gp_kernel_and_benchmark():
    identity=serial._library_source_identity(Path(serial.__file__).parent)
    assert {"models/FixedKernelGaussianProcess.py","models/FixedPolynomialKernel.py","benchmarks.py",
            "_advers_serial_protocol.py"} <= set(identity["files"])
    assert identity["sha256"]==basat.runtime_info(include_source_hash=True)["basat_source_sha256"]


def test_no_independent_pilot_skip_or_ordinary_max_source_runs(tmp_path):
    experiment=make(tmp_path)
    with pytest.raises(ExperimentConfigurationError):
        experiment.run(trials=2,max_trials=1,max_rounds=2,fail_fast=True)
    for invalid in (-1,25,True):
        with pytest.raises(ExperimentConfigurationError):run(make(tmp_path),ceiling=invalid)
    with pytest.raises(ExperimentConfigurationError):
        experiment.run(trials=2,max_rounds=2,trial_seeds=[0,1],fail_fast=True)
