"""Opt-in released AdveRS parallel-task random-stream orchestration.

Only native learners/evaluations run here. No official observations, decisions,
posteriors or random plans are inputs. The namespace affects task seeds only;
the coordinator knows no objective formula, domain resolution or kernel.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Mapping
import zlib

import torch

from .workflow import RandomStream

VERSION = 1
ROLES = ("RS1", "RS2", "RSG", "RO_true", "RO_small", "RO_large")


@dataclass(frozen=True)
class AdveRSSourceProtocol:
    """Released ``helpers/run_parallel.py`` LCB-subset stream schedule.

    ``method_roles`` maps display labels to the corresponding official names
    (RS1, RS2, RSG, RO_true, RO_small, RO_large), in execution order. Each task
    seed is ``seed + crc32(task_tag.encode()) % 100000``, with
    ``task_tag = f'{task_prefix}_{role}_atk1_runs_{start:03d}_{stop:03d}'``.
    One MT19937 stream continues through fresh learners within a chunk; chunks
    reset independently. Every repetition draws its own clean initial design.

    The default schedule matches the LCB tasks of
    ``helpers/run_parallel.py --exps synthetic --seed 0 --chunk 10
    --num-iter 100 --workers 1``. The other attack tasks are independent and
    need not execute for this subset. It is NOT the serial synthetic.py seed-0
    schedule. ``task_prefix`` is an explicit RNG namespace, not an objective.
    Ordinary trial seeds and historical protocol strings are unchanged.
    """

    method_roles: Mapping[str, str]
    seed: int = 0
    chunk_size: int = 10
    task_prefix: str = "synthetic"

    def __post_init__(self):
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, int) or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")
        if not isinstance(self.task_prefix, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", self.task_prefix):
            raise ValueError("task_prefix must be a nonempty safe RNG namespace.")
        roles = dict(self.method_roles)
        if (not roles or any(not isinstance(label, str) or not label for label in roles)
                or any(role not in ROLES for role in roles.values()) or len(set(roles.values())) != len(roles)):
            raise ValueError("method_roles must map unique labels to unique supported official names.")
        object.__setattr__(self, "method_roles", roles)

    def task_tag(self, role: str, start: int, stop: int) -> str:
        return f"{self.task_prefix}_{role}_atk1_runs_{start:03d}_{stop:03d}"

    def task_seed(self, role: str, start: int, stop: int) -> int:
        value = self.seed + zlib.crc32(self.task_tag(role, start, stop).encode()) % 100000
        if value >= 2**32:
            raise ValueError("The released task seed exceeds the MT19937 uint32 range.")
        return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def _require(condition, message):
    from .experiment import ExperimentCheckpointError
    if not condition:
        raise ExperimentCheckpointError(message)


def _entry(identity, tag, seed, repetition, state):
    body = dict(version=VERSION, protocol_sha256=identity, task_tag=tag, task_seed=seed,
                repetition=repetition, rng_state=copy.deepcopy(state))
    return {**body, "sha256": digest(body)}


def _validate(experiment, protocol, rounds, evaluations, fail_fast):
    from .experiment import ExperimentConfigurationError
    from .algorithms.Adversarial.AdveRS import AdveRS
    from .algorithms.Adversarial.StableOPT import StableOPT
    from .environments.Adversarial import AdversarialEnvironment, LCBAttack
    def invalid(message):
        raise ExperimentConfigurationError(message)
    if experiment.output_directory is None or experiment.stopping_rule is not None or not fail_fast:
        invalid("AdveRSSourceProtocol requires output_directory, no stopping_rule and fail_fast=True.")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        invalid("AdveRSSourceProtocol requires a positive integer max_rounds.")
    if tuple(experiment.algorithms) != tuple(protocol.method_roles):
        invalid("method_roles must match the experiment's labels and insertion order exactly.")
    for label, role in protocol.method_roles.items():
        algorithm, environment = experiment.algorithms[label], experiment.source_environments[label]
        is_method = (isinstance(algorithm, AdveRS) and role == algorithm.config.variant
                     or isinstance(algorithm, StableOPT) and role.startswith("RO_"))
        if not is_method or algorithm.config.source_protocol != "advers_2026_camera_ready":
            invalid("Source roles require the corresponding camera-ready AdveRS/StableOPT learner.")
        if (not isinstance(environment, AdversarialEnvironment) or not isinstance(environment.attack, LCBAttack)
                or environment.source_mode != "function" or environment.candidate_actions is None
                or environment.noise_fn is not None or not environment.allow_repeated_actions
                or environment.history.records or environment.history.initial_data is not None or algorithm.evaluations
                or environment.dtype != torch.float64 or environment.device.type != "cpu"):
            invalid("Source chunks require fresh CPU float64 finite callable LCB environments without custom noise.")
        if evaluations not in (None, rounds + algorithm.config.initial_points):
            invalid("max_evaluations must be omitted or include exactly initialization + max_rounds.")


def _manifest(experiment, protocol, trials, rounds, metrics):
    root = Path(__file__).parent
    files = ("_advers_source_protocol.py", "_advers_protocol.py", "experiment.py", "workflow.py",
             "algorithms/Adversarial/AdveRS.py", "algorithms/Adversarial/StableOPT.py", "environments/Adversarial.py")
    return experiment._jsonable(dict(version=VERSION, protocol=asdict(protocol), trials=trials,
        online_rounds=rounds, order="method_chunk_repetition", attack_mode=1,
        implementation={name: file_digest(root / name) for name in files},
        runtime={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "gpytorch", "botorch", "scikit-learn")},
        regret={label: asdict(metric) for label, metric in metrics.items()},
        fit_maxiter={label: algorithm.config.fit_maxiter for label, algorithm in experiment.algorithms.items()},
        algorithms={label: algorithm._configuration_snapshot() for label, algorithm in experiment.algorithms.items()},
        environments={label: experiment._jsonable(environment.state_dict()) for label, environment in experiment.source_environments.items()},
        save_every=experiment.save_every, checkpoint_policy=experiment.checkpoint_policy))


def _completed(child_root, entry):
    """Read a completed native bundle without reconstructing a learner."""
    marker = child_root / "source-completed.json"
    if not marker.is_file():
        return None
    done = load(marker)
    latest_path, result_path = child_root / "latest.json", child_root / "result.json"
    latest = load(latest_path)
    filename = latest.get("checkpoint")
    _require(isinstance(filename, str) and Path(filename).name == filename, "Unsafe source child checkpoint path.")
    checkpoint, bundle = child_root / filename, child_root / "source-completed.pt"
    entry_path = child_root / "source-entry.json"
    _require(all(path.is_file() and not path.is_symlink() for path in
                 (marker, latest_path, result_path, checkpoint, bundle, entry_path))
        and load(entry_path) == entry and done.get("entry_sha256") == entry["sha256"]
        and done.get("latest_sha256") == file_digest(latest_path)
        and done.get("result_sha256") == file_digest(result_path)
        and done.get("checkpoint_sha256") == latest.get("sha256") == file_digest(checkpoint)
        and done.get("bundle_sha256") == file_digest(bundle), "Completed source child checksum or entry identity differs.")
    payload = torch.load(bundle, map_location="cpu", weights_only=False)
    _require(payload.get("entry") == entry and digest(payload["end_rng"]) == done.get("end_rng_sha256"),
             "Completed source child random-stream chain differs.")
    return payload


def _publish(experiment, completed, trials, rounds):
    """Expose only full method groups through ordinary trial-result APIs."""
    from .experiment import ExperimentComparisonResult
    from .trials import ExperimentTrialsResult, statistics
    labels = tuple(experiment.algorithms)
    balanced = min((len(completed[label]) for label in labels), default=0)
    documents, results = [], []
    for repetition in range(balanced):
        documents.append(dict(schema_version=1, package="basat", mode="multi_algorithm",
            source_schedule="advers_parallel_chunks", repetition=repetition,
            algorithms={label: completed[label][repetition]["document"] for label in labels}))
        results.append(ExperimentComparisonResult(results={label: completed[label][repetition]["result"] for label in labels}))
    experiment._trial_count = trials
    experiment._trial_documents, experiment._trial_results = documents, results
    experiment.trial_experiments = [SimpleNamespace(output_directory=experiment.output_directory / "trials" / f"trial-{i+1:03d}",
        completed_rounds=rounds, failures={},
        elapsed_seconds=sum(float(completed[label][i]["document"].get("elapsed_seconds") or 0.) for label in labels))
        for i in range(balanced)]
    experiment.completed_rounds = rounds if balanced == trials else 0
    outcome = ExperimentTrialsResult(tuple(results), {label: statistics(documents, label) for label in labels},
        {label: tuple(completed[label][i]["result"] for i in range(balanced)) for label in labels}, {},
        records=tuple(copy.deepcopy(documents)), planned_trials=trials)
    experiment.trial_result = outcome
    experiment._atomic_json_save(dict(version=VERSION, planned_trials=trials, completed_trial_groups=balanced,
        completed_runs=sum(len(rows) for rows in completed.values()),
        completed_runs_by_method={label: len(rows) for label, rows in completed.items()}), experiment.output_directory / "source-summary.json")
    experiment.save()
    return outcome


def run_source_trials(experiment, *, trials, max_trials, max_rounds, max_evaluations,
                      metrics, plot, show, title, progress, fail_fast):
    from .experiment import Experiment, ExperimentConfigurationError
    protocol = experiment._advers_source_protocol
    _validate(experiment, protocol, max_rounds, max_evaluations, fail_fast)
    if not isinstance(progress, bool) or not isinstance(show, bool) or (plot is not None and not isinstance(plot, bool)):
        raise ExperimentConfigurationError("Source presentation flags must be Boolean (plot may be None).")
    root, labels = experiment.output_directory, tuple(experiment.algorithms)
    _require(not (root / "trials.json").exists() and not (root / "grouped_trials.json").exists(),
             "Source chunks require a separate directory from ordinary/grouped trials.")
    manifest = _manifest(experiment, protocol, trials, max_rounds, metrics)
    identity = digest(manifest)
    path = root / "source-protocol.json"
    if path.exists():
        _require(load(path) == manifest, "AdveRS source task configuration/runtime changed; use a fresh directory.")
        if not experiment.resume:
            raise ExperimentConfigurationError("Existing source chunks require resume=True.")
    else:
        # Validate every planned task seed before writing or querying anything.
        for role in protocol.method_roles.values():
            for start in range(0, trials, protocol.chunk_size):
                protocol.task_seed(role, start, min(start + protocol.chunk_size, trials))
        Experiment._atomic_json_save(manifest, path)
    completed = {label: {} for label in labels}
    experiment._last_requested_limits = dict(max_rounds=max_rounds, max_evaluations=max_evaluations,
        evaluation_protocol=asdict(protocol))
    limit = trials if max_trials is None else max_trials
    for method, (label, role) in enumerate(protocol.method_roles.items(), 1):
        for start in range(0, trials, protocol.chunk_size):
            stop = min(start + protocol.chunk_size, trials)  # Never use pilot ceiling in task tags.
            tag, seed = protocol.task_tag(role, start, stop), protocol.task_seed(role, start, stop)
            state_path = root / "source-tasks" / tag / "state.json"
            initial_rng = RandomStream(seed).get_state()
            state = load(state_path) if state_path.exists() else dict(version=VERSION, protocol_sha256=identity,
                task_tag=tag, task_seed=seed, start=start, stop=stop, next_repetition=start,
                rng_state=initial_rng, records=[], status="running", active_failure=None)
            _require(state.get("protocol_sha256") == identity and state.get("task_tag") == tag
                and state.get("task_seed") == seed and state.get("start") == start and state.get("stop") == stop,
                "Source task cursor identity differs.")
            cursor = state.get("next_repetition")
            _require(type(cursor) is int and start <= cursor <= stop and len(state.get("records", [])) == cursor-start,
                     "Invalid source task continuation cursor.")
            expected_rng = initial_rng
            for repetition in range(start, cursor):
                entry = _entry(identity, tag, seed, repetition, expected_rng)
                component = Experiment._checkpoint_component_name(method, label, experiment.algorithms[label])
                child_root = root / "trials" / f"trial-{repetition+1:03d}" / "algorithms" / component
                payload = _completed(child_root, entry)
                _require(payload is not None and state["records"][repetition-start] == dict(repetition=repetition,
                    entry_sha256=entry["sha256"], end_rng_sha256=digest(payload["end_rng"])), "Committed source task record differs.")
                expected_rng = payload["end_rng"]
                completed[label][repetition] = payload
            _require(state.get("rng_state") == expected_rng, "Source task MT19937 state is not its native child chain.")
            if start >= limit or cursor >= min(stop, limit):
                continue
            if not state_path.exists():
                state_path.parent.mkdir(parents=True, exist_ok=True)
                Experiment._atomic_json_save(state, state_path)
            while cursor < min(stop, limit):
                entry = _entry(identity, tag, seed, cursor, state["rng_state"])
                component = Experiment._checkpoint_component_name(method, label, experiment.algorithms[label])
                child_root = root / "trials" / f"trial-{cursor+1:03d}" / "algorithms" / component
                child_root.mkdir(parents=True, exist_ok=True)
                entry_path = child_root / "source-entry.json"
                if entry_path.exists():
                    _require(load(entry_path) == entry, "Source child entry differs from its native predecessor.")
                else:
                    Experiment._atomic_json_save(entry, entry_path)
                child = None
                try:
                    payload = _completed(child_root, entry)
                    if payload is None:
                        algorithm = experiment.algorithms[label].clone_for_trial(seed-experiment.algorithms[label].seed)
                        environment = copy.deepcopy(experiment.source_environments[label])
                        environment.reset(seed=seed)
                        child = Experiment(environment=environment, algorithm=algorithm,
                            output_directory=child_root, resume=(child_root / "latest.json").exists(),
                            stopping_rule=None, save_every=experiment.save_every, checkpoint_policy="transactional",
                            evaluation_protocol="advers_2025", _advers_source_entry=entry)
                        if progress:
                            print(f"{tag} | {label} | repetition {cursor+1}/{trials}", flush=True)
                        result = child.run(max_rounds=max_rounds, max_evaluations=max_evaluations,
                            regret=metrics.get(label), plot=False, show=False, fail_fast=True)
                        _require(child.completed_rounds == max_rounds and child.algorithm_consumed_environment_records == max_rounds,
                                 "Source child returned before its committed online budget.")
                        document = child._result_document()["algorithms"][type(child.algorithm).__name__]
                        Experiment._atomic_json_save({"label": label, **document}, child_root / "result.json")
                        latest = load(child_root / "latest.json")
                        end_rng = child.algorithm.get_source_random_state()
                        _require(digest(end_rng) == child.environment.history.records[-1].audit["source_rng_after_feedback_sha256"],
                                 "Native source stream advanced beyond final durable feedback.")
                        payload = dict(entry=entry, end_rng=end_rng, document=document, result=result)
                        bundle_hash = Experiment._atomic_torch_save(payload, child_root / "source-completed.pt")
                        Experiment._atomic_json_save(dict(entry_sha256=entry["sha256"], end_rng_sha256=digest(end_rng),
                            latest_sha256=file_digest(child_root / "latest.json"), result_sha256=file_digest(child_root / "result.json"),
                            checkpoint_sha256=latest["sha256"], bundle_sha256=bundle_hash), child_root / "source-completed.json")
                    completed[label][cursor] = payload
                    advanced = copy.deepcopy(state)
                    advanced["records"].append(dict(repetition=cursor, entry_sha256=entry["sha256"], end_rng_sha256=digest(payload["end_rng"])))
                    advanced.update(next_repetition=cursor+1, rng_state=payload["end_rng"], active_failure=None,
                                    status="completed" if cursor+1 == stop else "running")
                    Experiment._atomic_json_save(advanced, state_path)
                    state, cursor = advanced, cursor+1
                    _publish(experiment, completed, trials, max_rounds)
                except BaseException as error:
                    state["active_failure"] = dict(repetition=cursor, type=type(error).__name__, message=str(error))
                    state["status"] = "failed"
                    Experiment._atomic_json_save(state, state_path)
                    if child is not None and not (child_root / "source-completed.json").is_file():
                        document = child._result_document()["algorithms"][type(child.algorithm).__name__]
                        Experiment._atomic_json_save({"label": label, **document}, child_root / "result.json")
                    _publish(experiment, completed, trials, max_rounds)
                    raise
    outcome = _publish(experiment, completed, trials, max_rounds)
    if plot is True or (plot is None and any(a.plot for a in experiment.algorithms.values())):
        from .plotting import plot as draw
        figure = draw(experiment, title=title, show=show)
        outcome.figure, outcome.axes = figure.figure, figure.axes
    return outcome
