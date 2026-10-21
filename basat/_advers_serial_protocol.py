"""Opt-in native serial AdveRS stream: random prefix, then six LCB blocks.

The coordinator knows method roles and execution order, never a benchmark
formula, official trajectory, posterior, reward table, or reference RNG state.
One owned MT19937 stream is seeded once; every handoff comes exclusively from
the preceding completed native learner's durable terminal state.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import shutil
from types import SimpleNamespace
from typing import Mapping

import torch

from ._advers_source_protocol import (
    ROLES, _completed, _entry, _require, _validate, digest, file_digest, load,
)
from .workflow import RandomStream

VERSION = 1
PHASES = ("random_prefix", "lcb")
MINIMUM_FREE_DISK_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class AdveRSSerialSourceProtocol:
    """Run the released serial random-attack prefix before the LCB methods.

    ``method_roles`` maps six user labels, in order, to ``RS1``, ``RS2``,
    ``RSG``, ``RO_true``, ``RO_small`` and ``RO_large``. ``seed`` initializes
    MT19937 once for the complete schedule. Fresh learners first execute all
    random-attack repetitions in method order, then all LCB repetitions in
    that same order; no block or repetition is reseeded or skipped.

    Use with fresh finite function-backed LCB environments. The coordinator
    creates separate private random-attack environment copies for the prefix.
    Ordinary result/statistics APIs contain only the LCB runs; prefix records
    and their query accounting remain explicitly separate on disk.

    ``Experiment.run(trials=100, max_rounds=100, fail_fast=True)`` plans 600
    random-prefix and 600 LCB runs. ``max_source_runs`` is an optional cumulative
    source-run ceiling, not a per-method pilot ceiling: a ceiling of 1 executes
    only the first random-prefix repetition, preserving the full plan. The
    same output directory can then resume without a ceiling.

    This protocol does not change ordinary independent trial seeds or the
    separate ``AdveRSSourceProtocol`` parallel-chunk schedule. It reproduces
    the ordered attack0/attack1 portion of the released serial script, not its
    subsequent attack2 blocks. Algorithms, objective and metric stay explicit
    user inputs; no governing paper arrays or official states are accepted.
    """

    method_roles: Mapping[str, str]
    seed: int = 0

    def __post_init__(self):
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        roles = dict(self.method_roles)
        if (tuple(roles.values()) != ROLES or len(roles) != 6
                or any(not isinstance(label, str) or not label for label in roles)):
            raise ValueError("method_roles must map six unique labels to RS1, RS2, RSG, RO_true, RO_small, RO_large in that order.")
        object.__setattr__(self, "method_roles", roles)


def _library_source_identity(root: Path):
    """Conservatively bind every Basat Python implementation, not only config.

    The digest encoding matches ``runtime_info(include_source_hash=True)``.
    Explicit file entries also detect additions/removals and expose coverage.
    Only the new serial protocol uses this identity; old manifest formats are
    not rewritten or relaxed.
    """
    root = Path(root)
    files, content_digest = {}, hashlib.sha256()
    paths = sorted(root.rglob("*.py"))
    _require(bool(paths), "Basat implementation source is unavailable.")
    for path in paths:
        _require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
                 "Serial implementation identity refuses symlinked/out-of-package Python sources.")
        content = path.read_bytes()
        name = path.relative_to(root).as_posix()
        files[name] = hashlib.sha256(content).hexdigest()
        content_digest.update(name.encode() + b"\0")
        content_digest.update(len(content).to_bytes(8, "big"))
        content_digest.update(content)
    return dict(scope="complete-basat-python-source-v1", python_files=len(files),
                files=files, sha256=content_digest.hexdigest())


def _plan(protocol, trials):
    return [dict(block=phase_index * 6 + method, phase=phase, attack_mode=phase_index,
                 method_index=method, label=label, official_algorithm=role, repetitions=trials,
                 task_tag=f"serial_atk{phase_index}_{role}")
            for phase_index, phase in enumerate(PHASES)
            for method, (label, role) in enumerate(protocol.method_roles.items())]


def _position(plan, trials, cursor):
    if cursor == len(plan) * trials:
        return dict(phase="completed", method_index=None, repetition=None, block=None)
    block, repetition = divmod(cursor, trials)
    spec = plan[block]
    return dict(phase=spec["phase"], method_index=spec["method_index"], repetition=repetition, block=block)


def _require_disk_space(root):
    """A minimum safety floor, not an estimate of total campaign storage."""
    _require(shutil.disk_usage(root).free >= MINIMUM_FREE_DISK_BYTES,
             "Serial source execution requires at least 2 GiB free disk before a scientific child; free space and resume the preserved cursor.")


def _prefix_environments(experiment, protocol):
    from .environments.Adversarial import RandomAttack
    environments = {}
    for label, role in protocol.method_roles.items():
        original = experiment.source_environments[label]
        _require(original.attack.radius > 0, "Serial source protocol requires a positive physical attack radius.")
        environment = copy.deepcopy(original)
        environment.attack = RandomAttack(radius=original.attack.radius,
                                           squared_distance=role not in ("RS1", "RSG"))
        environment._freeze_source_fingerprint()
        environments[label] = environment
    return environments


def _manifest(experiment, protocol, trials, rounds, metrics, prefix_environments):
    return experiment._jsonable(dict(
        version=VERSION, protocol="advers_serial_random_prefix_then_lcb",
        method_roles=dict(protocol.method_roles), seed=protocol.seed, trials_per_method=trials,
        online_rounds=rounds, planned_random_prefix_runs=6*trials, planned_lcb_runs=6*trials,
        minimum_free_disk_bytes=MINIMUM_FREE_DISK_BYTES,
        plan=_plan(protocol, trials), rng_policy="single_owned_mt19937_native_terminal_handoffs",
        implementation=_library_source_identity(Path(__file__).parent),
        runtime={"python": platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "torch", "gpytorch", "botorch", "scikit-learn")}},
        algorithms={label: algorithm._configuration_snapshot() for label, algorithm in experiment.algorithms.items()},
        fit_maxiter={label: algorithm.config.fit_maxiter for label, algorithm in experiment.algorithms.items()},
        lcb_environments={label: experiment._jsonable(environment.state_dict())
                          for label, environment in experiment.source_environments.items()},
        prefix_environments={label: experiment._jsonable(environment.state_dict())
                             for label, environment in prefix_environments.items()},
        regret={label: experiment._jsonable(metric) for label, metric in metrics.items()},
        save_every=experiment.save_every, checkpoint_policy=experiment.checkpoint_policy))


def _child_root(experiment, spec, repetition):
    from .experiment import Experiment
    component = Experiment._checkpoint_component_name(spec["method_index"]+1, spec["label"],
                                                      experiment.algorithms[spec["label"]])
    root = experiment.output_directory
    if spec["phase"] == "random_prefix":
        root = root / "prefix"
    return root / "trials" / f"trial-{repetition+1:03d}" / "algorithms" / component


def _record(spec, repetition, source_run, entry, payload, child_root, root):
    return dict(source_run=source_run, phase=spec["phase"], method_index=spec["method_index"],
                repetition=repetition, child_directory=str(child_root.relative_to(root)),
                entry_sha256=entry["sha256"], end_rng_sha256=digest(payload["end_rng"]))


def _summary(experiment, state, completed, trials, rounds):
    counts = {phase: {label: len(rows) for label, rows in completed[phase].items()} for phase in PHASES}
    prefix, lcb = (sum(counts[phase].values()) for phase in PHASES)
    initial = {phase: sum(len(rows)*experiment.algorithms[label].config.initial_points
                          for label, rows in completed[phase].items()) for phase in PHASES}
    return dict(version=VERSION, source_schedule="advers_serial_random_prefix_then_lcb",
        planned_trials_per_lcb_method=trials, planned_source_runs=12*trials,
        planned_prefix_runs=6*trials, planned_lcb_runs=6*trials,
        completed_source_runs=prefix+lcb, completed_prefix_runs=prefix, completed_lcb_runs=lcb,
        completed_runs_by_phase_and_method=counts,
        completed_lcb_trial_groups=min(counts["lcb"].values(), default=0),
        query_counts_by_phase={phase: dict(initialization_queries=initial[phase],
             online_queries=sum(counts[phase].values())*rounds,
             total_objective_queries=initial[phase]+sum(counts[phase].values())*rounds) for phase in PHASES},
        cursor={name: state[name] for name in ("next_source_run", "phase", "method_index", "repetition", "block", "status", "active_failure")},
        prefix_is_not_plotted=True)


def _publish(experiment, completed, trials, rounds, summary):
    """Ordinary compatible output containing ONLY balanced completed LCB runs."""
    from .experiment import ExperimentComparisonResult
    from .trials import ExperimentTrialsResult, statistics
    lcb, labels = completed["lcb"], tuple(experiment.algorithms)
    balanced = min((len(lcb[label]) for label in labels), default=0)
    documents, results = [], []
    for repetition in range(balanced):
        documents.append(dict(schema_version=1, package="basat", mode="multi_algorithm",
            source_schedule="advers_serial_random_prefix_then_lcb", plotted_phase="lcb", repetition=repetition,
            algorithms={label: lcb[label][repetition]["document"] for label in labels}))
        results.append(ExperimentComparisonResult(results={label: lcb[label][repetition]["result"] for label in labels}))
    experiment._trial_count = trials
    experiment._trial_documents, experiment._trial_results = documents, results
    experiment.trial_experiments = [SimpleNamespace(
        output_directory=experiment.output_directory / "trials" / f"trial-{i+1:03d}",
        completed_rounds=rounds, failures={},
        elapsed_seconds=sum(float(lcb[label][i]["document"].get("elapsed_seconds") or 0.) for label in labels))
        for i in range(balanced)]
    experiment.completed_rounds = rounds if balanced == trials else 0
    outcome = ExperimentTrialsResult(tuple(results), {label: statistics(documents, label) for label in labels},
        {label: tuple(lcb[label][i]["result"] for i in range(balanced)) for label in labels}, {},
        records=tuple(copy.deepcopy(documents)), planned_trials=trials)
    outcome.source_summary = copy.deepcopy(summary)
    experiment.trial_result = outcome
    experiment._atomic_json_save(summary, experiment.output_directory / "source-serial-summary.json")
    experiment.save()
    return outcome


def run_source_trials(experiment, *, trials, max_source_runs, max_rounds, max_evaluations,
                      metrics, plot, show, title, progress, fail_fast):
    from .experiment import Experiment, ExperimentConfigurationError
    protocol = experiment._advers_serial_source_protocol
    _validate(experiment, protocol, max_rounds, max_evaluations, fail_fast)
    if (isinstance(trials, bool) or not isinstance(trials, int) or trials < 1
            or isinstance(max_source_runs, bool)
            or max_source_runs is not None and (not isinstance(max_source_runs, int) or not 0 <= max_source_runs <= 12*trials)):
        raise ExperimentConfigurationError("Serial max_source_runs must be a cumulative integer from0 to12*trials, or None.")
    if not isinstance(progress, bool) or not isinstance(show, bool) or (plot is not None and not isinstance(plot, bool)):
        raise ExperimentConfigurationError("Serial presentation flags must be Boolean (plot may be None).")
    root = experiment.output_directory
    _require(not any((root / name).exists() for name in ("trials.json", "grouped_trials.json", "source-protocol.json")),
             "Serial source protocol requires a separate directory from ordinary/grouped/chunk trials.")
    prefix = _prefix_environments(experiment, protocol)
    manifest = _manifest(experiment, protocol, trials, max_rounds, metrics, prefix)
    identity, plan = digest(manifest), manifest["plan"]
    manifest_path = root / "source-serial-protocol.json"
    if manifest_path.exists():
        _require(load(manifest_path) == manifest, "Serial source configuration/runtime/implementation changed; use a fresh directory.")
        if not experiment.resume:
            raise ExperimentConfigurationError("Existing serial source runs require resume=True.")
    else:
        Experiment._atomic_json_save(manifest, manifest_path)
    initial_rng = RandomStream(protocol.seed, engine="mt19937").get_state()
    state_path = root / "source-serial-state.json"
    state = load(state_path) if state_path.exists() else dict(
        version=VERSION, protocol_sha256=identity, next_source_run=0, rng_state=initial_rng,
        records=[], active_failure=None, status="running", **_position(plan, trials, 0))
    cursor = state.get("next_source_run")
    _require(state.get("version") == VERSION and state.get("protocol_sha256") == identity
             and type(cursor) is int and 0 <= cursor <= 12*trials
             and len(state.get("records", [])) == cursor
             and all(state.get(name) == value for name, value in _position(plan, trials, cursor).items()),
             "Serial source cursor identity/phase/method/repetition differs.")
    completed = {phase: {label: {} for label in protocol.method_roles} for phase in PHASES}
    expected_rng = initial_rng
    for source_run in range(cursor):
        block, repetition = divmod(source_run, trials)
        spec = plan[block]
        entry = _entry(identity, spec["task_tag"], protocol.seed, repetition, expected_rng)
        child_root = _child_root(experiment, spec, repetition)
        payload = _completed(child_root, entry)
        _require(payload is not None and state["records"][source_run] ==
                 _record(spec, repetition, source_run, entry, payload, child_root, root),
                 "Serial committed native child or predecessor chain differs.")
        completed[spec["phase"]][spec["label"]][repetition] = payload
        expected_rng = payload["end_rng"]
    _require(state.get("rng_state") == expected_rng, "Serial MT19937 state is not its own native child chain.")
    if not state_path.exists():
        Experiment._atomic_json_save(state, state_path)
    experiment._last_requested_limits = dict(max_rounds=max_rounds, max_evaluations=max_evaluations,
        source_schedule="advers_serial_random_prefix_then_lcb", max_source_runs=max_source_runs)
    ceiling = 12*trials if max_source_runs is None else max_source_runs
    while cursor < ceiling:
        block, repetition = divmod(cursor, trials)
        spec = plan[block]
        entry = _entry(identity, spec["task_tag"], protocol.seed, repetition, state["rng_state"])
        child_root = _child_root(experiment, spec, repetition)
        child_root.mkdir(parents=True, exist_ok=True)
        entry_path = child_root / "source-entry.json"
        if entry_path.exists():
            _require(load(entry_path) == entry, "Serial child entry differs from its native predecessor.")
        else:
            Experiment._atomic_json_save(entry, entry_path)
        child = None
        try:
            payload = _completed(child_root, entry)
            if payload is None:
                _require_disk_space(root)
                label = spec["label"]
                algorithm = experiment.algorithms[label].clone_for_trial(protocol.seed - experiment.algorithms[label].seed)
                environment = copy.deepcopy(prefix[label] if spec["phase"] == "random_prefix" else experiment.source_environments[label])
                environment.reset(seed=protocol.seed)
                child = Experiment(environment=environment, algorithm=algorithm, output_directory=child_root,
                    resume=(child_root / "latest.json").exists(), stopping_rule=None, save_every=experiment.save_every,
                    checkpoint_policy="transactional", evaluation_protocol="advers_2025", _advers_source_entry=entry)
                if progress:
                    print(f"serial {spec['phase']} | {label} | repetition {repetition+1}/{trials} | source run {cursor+1}/{12*trials}", flush=True)
                result = child.run(max_rounds=max_rounds, max_evaluations=max_evaluations,
                    regret=metrics.get(label), plot=False, show=False, fail_fast=True)
                _require(child.completed_rounds == max_rounds and child.algorithm_consumed_environment_records == max_rounds,
                         "Serial source child returned before its committed online budget.")
                document = child._result_document()["algorithms"][type(child.algorithm).__name__]
                Experiment._atomic_json_save({"label": label, **document}, child_root / "result.json")
                latest = load(child_root / "latest.json")
                end_rng = child.algorithm.get_source_random_state()
                _require(digest(end_rng) == child.environment.history.records[-1].audit["source_rng_after_feedback_sha256"],
                         "Serial native stream advanced beyond final durable feedback.")
                payload = dict(entry=entry, end_rng=end_rng, document=document, result=result)
                bundle_hash = Experiment._atomic_torch_save(payload, child_root / "source-completed.pt")
                Experiment._atomic_json_save(dict(entry_sha256=entry["sha256"], end_rng_sha256=digest(end_rng),
                    latest_sha256=file_digest(child_root / "latest.json"), result_sha256=file_digest(child_root / "result.json"),
                    checkpoint_sha256=latest["sha256"], bundle_sha256=bundle_hash), child_root / "source-completed.json")
            completed[spec["phase"]][spec["label"]][repetition] = payload
            advanced = copy.deepcopy(state)
            advanced["records"].append(_record(spec, repetition, cursor, entry, payload, child_root, root))
            advanced.update(next_source_run=cursor+1, rng_state=payload["end_rng"], active_failure=None,
                            status="completed" if cursor+1 == 12*trials else "running",
                            **_position(plan, trials, cursor+1))
            Experiment._atomic_json_save(advanced, state_path)
            state, cursor = advanced, cursor+1
        except BaseException as error:
            # A completed child may exist before its parent cursor commits.
            # Keep summaries aligned to the authoritative committed cursor;
            # the checksumed child bundle remains available for no-query
            # recovery on the next invocation.
            completed[spec["phase"]][spec["label"]].pop(repetition, None)
            state["active_failure"] = dict(source_run=cursor, phase=spec["phase"], label=spec["label"], repetition=repetition,
                                           type=type(error).__name__, message=str(error))
            state["status"] = "failed"
            Experiment._atomic_json_save(state, state_path)
            if child is not None and not (child_root / "source-completed.json").is_file():
                document = child._result_document()["algorithms"][type(child.algorithm).__name__]
                Experiment._atomic_json_save({"label": spec["label"], **document}, child_root / "result.json")
            _publish(experiment, completed, trials, max_rounds, _summary(experiment, state, completed, trials, max_rounds))
            raise
        # A failed report-only summary write must not relabel the preceding
        # durably committed scientific child as a failed/uncommitted run.
        Experiment._atomic_json_save(_summary(experiment, state, completed, trials, max_rounds), root / "source-serial-summary.json")
    outcome = _publish(experiment, completed, trials, max_rounds, _summary(experiment, state, completed, trials, max_rounds))
    if plot is True or (plot is None and any(algorithm.plot for algorithm in experiment.algorithms.values())):
        from .plotting import plot as draw
        figure = draw(experiment, title=title, show=show)
        outcome.figure, outcome.axes = figure.figure, figure.axes
    return outcome
