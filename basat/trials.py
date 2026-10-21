"""Independent seeded repetitions and evaluation-aligned statistics."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import random
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from numbers import Integral
from typing import Any

import numpy as np
import torch

ACTIVE: ContextVar[bool] = ContextVar("basat_trial", default=False)


@dataclass(frozen=True)
class GroupedTrialConfig:
    """Sequential repetitions sharing an initial design and random streams.

    Parameters
    ----------
    groups, repetitions : int
        Number of independently seeded groups and repetitions per method.
    initial_points : int
        Shared uniform design size. The released good-action protocol requires 3.
    seed : int
        Group ``g`` starts from ``seed + g``. Within a group streams continue
        in algorithm insertion order, then repetition order, without reseeding.
    order, rng_policy : str
        Currently only ``'group_algorithm_repeat'`` and ``'shared'`` are supported.

    Notes
    -----
    This explicitly opted-in protocol currently supports released good-action
    methods with private NumPy/Torch stream handoff, a noiseless callable
    environment, and transactional checkpoints. It does not change ordinary
    ``Experiment.run(trials=...)`` semantics.
    """

    groups: int = 25
    repetitions: int = 10
    initial_points: int = 3
    seed: int = 2021
    order: str = "group_algorithm_repeat"
    rng_policy: str = "shared"

    def __post_init__(self) -> None:
        for name in ("groups", "repetitions", "initial_points"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if self.seed + self.groups - 1 >= 2**32:
            raise ValueError("Group seeds must fit the NumPy MT19937 seed range.")
        if self.initial_points != 3:
            raise ValueError("The shared released protocol requires initial_points=3.")
        if self.order != "group_algorithm_repeat" or self.rng_policy != "shared":
            raise ValueError("Only group_algorithm_repeat order with shared streams is supported.")


@dataclass(frozen=True)
class GroupedExperimentResult:
    """Raw grouped outcomes, retaining the planned denominator and censoring.

    ``records`` include initialization, evaluation dtypes, stopping reasons and
    failures. Unstarted repetitions are absent, never counted as successes or
    silently removed from the plan. ``groups`` retain the shared designs.
    """

    records: tuple[dict[str, Any], ...]
    groups: tuple[dict[str, Any], ...]
    group_count: int
    repetitions: int
    algorithm_names: tuple[str, ...]
    protocol: dict[str, Any]

    @property
    def evaluations(self) -> dict[str, tuple[int, ...]]:
        return {
            name: tuple(row["evaluations"] for row in self.records if row["algorithm"] == name)
            for name in self.algorithm_names
        }


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def trial_rng(experiment: Any) -> Iterator[None]:
    """Isolate process RNGs, including their checkpoint continuation state."""
    if not experiment._trial_managed:
        yield
        return
    previous = rng_state()
    try:
        if experiment._trial_rng_state is None:
            seed = experiment.algorithm.seed
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
        else:
            restore_rng(experiment._trial_rng_state)
        yield
    finally:
        experiment._trial_rng_state = rng_state()
        restore_rng(previous)


def shifted(value: Any, offset: int) -> Any:
    """Copy nested configurations, incrementing explicitly specified seeds."""
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                f.name: (
                    getattr(value, f.name) + offset
                    if f.name == "seed" and isinstance(getattr(value, f.name), int)
                    else shifted(getattr(value, f.name), offset)
                )
                for f in fields(value)
                if f.init
            },
        )
    if isinstance(value, dict):
        return {
            k: (v + offset if k == "seed" and isinstance(v, int) else shifted(v, offset))
            for k, v in value.items()
        }
    if isinstance(value, tuple):
        return tuple(shifted(v, offset) for v in value)
    if isinstance(value, list):
        return [shifted(v, offset) for v in value]
    return copy.deepcopy(value)


def fresh_algorithm(template: Any, offset: int) -> Any:
    """Reconstruct a method, so seed-dependent embeddings and solvers are fresh."""
    from .experiment import ExperimentConfigurationError

    arguments = {}
    for name, parameter in inspect.signature(type(template)).parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if not hasattr(template, name):
            raise ExperimentConfigurationError(
                f"{type(template).__name__} must expose constructor setting {name!r} "
                "or override clone_for_trial(seed_offset)."
            )
        value = getattr(template, name)
        arguments[name] = value + offset if name == "seed" else shifted(value, offset)
    result = type(template)(**arguments)
    result.configure_sampling_methods(
        initial=template.initial_sampling_method, acquisition=template.acquisition_sampling_method
    )
    return result


@dataclass(frozen=True)
class TrialStatistics:
    """Pointwise sample statistics; absent observations are never extrapolated."""

    evaluations: list[int]
    count: list[int]
    mean: list[float]
    std: list[float | None]
    incumbent_mean: list[float]
    incumbent_std: list[float | None]
    maximize: bool


def statistics(rows: list[dict[str, Any]], label: str) -> TrialStatistics:
    records = [row["algorithms"][label] for row in rows if label in row["algorithms"]]
    maximize = records[0]["maximize"] if records else True
    all_values = [np.asarray(row["observations"], dtype=float) for row in records]
    offsets = [row.get("initial_observation_count", 0) for row in records]
    values = [v[offset:] for v, offset in zip(all_values, offsets, strict=True)]
    n = max((len(v) for v in values), default=0)
    counts, means, stds, best_means, best_stds = [], [], [], [], []
    incumbents = [
        (np.maximum.accumulate(v) if maximize else np.minimum.accumulate(v))[offset:]
        for v, offset in zip(all_values, offsets, strict=True)
    ]
    for i in range(n):
        observed = [v[i] for v in values if len(v) > i]
        best = [v[i] for v in incumbents if len(v) > i]
        counts.append(len(observed))
        means.append(float(np.mean(observed)))
        best_means.append(float(np.mean(best)))
        stds.append(float(np.std(observed, ddof=1)) if len(observed) > 1 else None)
        best_stds.append(float(np.std(best, ddof=1)) if len(best) > 1 else None)
    return TrialStatistics(
        list(range(1, n + 1)), counts, means, stds, best_means, best_stds, maximize
    )


@dataclass
class ExperimentTrialsResult:
    """Raw trial results and aggregate curves. No averaged action is invented.

    ``records`` are detached result-document snapshots in trial order, including
    settings, observations, warm-start counts, diagnostics and failures. They
    contain no live GP/checkpoint objects. Missing and failed trials are never
    silently represented as completed observations.
    """

    trials: tuple[Any, ...]
    statistics: Mapping[str, TrialStatistics]
    results: Mapping[str, tuple[Any, ...]]
    failures: Mapping[str, Any]
    figure: Any = None
    axes: Any = None
    records: tuple[dict[str, Any], ...] = ()
    planned_trials: int | None = None

    @property
    def observations(self) -> dict[str, tuple[tuple[float, ...], ...]]:
        """Raw observations by method/trial, including explicitly supplied warm starts."""
        return {label: tuple(tuple(row["algorithms"][label]["observations"])
                             for row in self.records) for label in self.results}

    @property
    def diagnostics(self) -> dict[str, tuple[dict[str, Any], ...]]:
        """Detached model diagnostics, warnings, failures and timing by method/trial."""
        return {label: tuple(copy.deepcopy({
            "diagnostics": (row["algorithms"][label].get("result") or {}).get("diagnostics"),
            **{key: row["algorithms"][label].get(key) for key in (
                "warnings", "failure", "stopping_reason", "elapsed_seconds",
                "environment_evaluations", "initial_observation_count")},
        }) for row in self.records) for label in self.results}

    @property
    def regret_runs(self) -> dict[str, tuple[tuple[float, ...], ...]]:
        """Return completed post-hoc regret curves, transformed within each trial.

        Labels without an explicitly requested regret metric are omitted.
        Failed trials are never padded or represented by invented values.
        """

        collected: dict[str, tuple[tuple[float, ...], ...]] = {}
        for label, runs in self.results.items():
            curves = []
            for result in runs:
                if result is None:
                    continue
                report = result.metadata.get("regret")
                if report is None:
                    continue
                curves.append(tuple(float(value) for value in report["values"]))
            if curves:
                collected[label] = tuple(curves)
        return collected

    @property
    def regret_statistics(self) -> dict[str, Any]:
        """Return population summaries of equal-length completed regret runs."""

        from .regret import rs_regret_statistics

        summaries = {}
        for label, runs in self.regret_runs.items():
            lengths = {len(run) for run in runs}
            if len(lengths) != 1:
                raise ValueError(
                    f"{label}: regret runs have unequal lengths; select a common explicit "
                    "evaluation prefix before aggregation."
                )
            if lengths != {0}:
                summaries[label] = rs_regret_statistics(runs)
        return summaries

    def to_dict(self) -> dict[str, Any]:
        """Return portable copies of raw records and already-computed statistics.

        Save with ``basat.ArtifactStore(...).write_json(..., result.to_dict())``.
        Reading this snapshot cannot resume optimizers or evaluate objectives.
        """
        return {"schema_version": 1, "package": "basat", "mode": "trials",
                "planned_trials": self.planned_trials, "completed_trials": len(self.records),
                "statistics": {label: asdict(stat) for label, stat in self.statistics.items()},
                "records": copy.deepcopy(list(self.records))}

    @property
    def evaluations(self) -> Any:
        counts = {
            label: tuple(None if r is None else r.evaluations for r in runs)
            for label, runs in self.results.items()
        }
        return next(iter(counts.values())) if len(counts) == 1 else counts


def aggregate_document(experiment: Any) -> dict[str, Any]:
    rows = experiment._trial_documents
    labels = list(experiment.algorithms)
    algorithms = {}
    for label in labels:
        stat = statistics(rows, label)
        records = [row["algorithms"][label] for row in rows]
        failures = [row["failure"] for row in records]
        algorithms[label] = {
            "maximize": stat.maximize,
            "observations": stat.mean,
            "statistics": experiment._jsonable(stat),
            "result": {
                "evaluations": len(stat.mean),
                "best_observed_value": stat.incumbent_mean[-1] if stat.mean else None,
                "status": "trial_aggregate",
            },
            "environment_evaluations": max(
                (r["environment_evaluations"] for r in records), default=0
            ),
            "trial_evaluations": [r["environment_evaluations"] for r in records],
            "trial_failures": failures,
            "failure": next((f for f in failures if f), None) if all(failures) else None,
            "warnings": sorted({w for r in records for w in r["warnings"]}),
            "elapsed_seconds": sum(float(r.get("elapsed_seconds") or 0.0) for r in records),
            "trial_elapsed_seconds": [r.get("elapsed_seconds") for r in records],
            "stopping_reason": "see individual trial records",
            "trials": records,
        }
    return {
        "schema_version": 1,
        "package": "basat",
        "mode": "trials",
        "trial_count": experiment._trial_count,
        "completed_trials": len(rows),
        "elapsed_seconds": experiment.elapsed_seconds,
        "limits": experiment._last_requested_limits,
        "algorithm_order": labels,
        "uncertainty": "mean +/- 1.96 sample standard deviations; not a confidence interval",
        "algorithms": algorithms,
        "trial_directories": [
            str(child.output_directory) for child in experiment.trial_experiments
        ],
    }


def _manifest(experiment: Any, count: int) -> dict[str, Any]:
    """Validate the entire repeated experiment before modifying saved results.

    Use the same immutable identities as ordinary checkpoints, so fitting
    budgets remain changeable on resume while the objective and acquisition do
    not silently change between trials.
    """
    from .algorithms.algorithm import Algorithm

    def environment_identity(state: Mapping[str, Any]) -> dict[str, Any]:
        identity = {
            key: state[key]
            for key in (
                "class_identity",
                "environment_type",
                "source_mode",
                "seed",
                "dtype",
                "device_type",
                "bounds_lower",
                "bounds_upper",
                "noise_std",
                "has_noise_fn",
                "add_noise_to_data",
                "allow_repeated_actions",
                "data_row_selection",
                "dataset_fingerprint",
                "source_fingerprint",
                "adapter_name",
                "constant_context",
                "context_support",
            )
            if key in state
        }
        if "source_state" in state:
            identity["source_state"] = environment_identity(state["source_state"])
        if "history" in state:
            identity["initial_data"] = state["history"].get("initial_data")
        return identity

    environments = {
        label: Algorithm._fingerprint_value(environment_identity(environment.state_dict()))
        for label, environment in experiment._trial_environments.items()
    }
    rule = experiment._trial_rule
    return {
        "schema_version": 2,
        "trials": count,
        "multi_algorithm": experiment.is_multi_algorithm,
        "algorithms": {
            label: {
                "class": f"{type(a).__module__}.{type(a).__qualname__}",
                "base_seed": a.seed,
                "configuration": a._configuration_snapshot(),
            }
            for label, a in experiment._trial_templates.items()
        },
        "environment_seeds": {label: e.seed for label, e in experiment._trial_environments.items()},
        "environments": environments,
        "stopping_rule": None if rule is None else Algorithm._fingerprint_value(rule.state_dict()),
        "save_every": experiment.save_every,
        "constant_context": Algorithm._fingerprint_value(experiment._trial_constant_context),
        **({"evaluation_protocol": experiment.evaluation_protocol}
           if experiment.evaluation_protocol != "standard" else {}),
        **({"trial_seeds": experiment._trial_seed_schedule}
           if experiment._trial_seed_schedule is not None else {}),
    }


def _validate_manifest(saved: Any, current: dict[str, Any]) -> None:
    from .experiment import ExperimentCheckpointError, ExperimentConfigurationError

    if not isinstance(saved, dict) or saved.get("schema_version") not in {1, 2}:
        raise ExperimentCheckpointError("Trial manifest schema is invalid or unsupported.")
    if saved["schema_version"] == 1:
        if current.get("trial_seeds") is not None:
            raise ExperimentConfigurationError(
                "Legacy trial manifests cannot resume with an explicit trial_seeds schedule."
            )
        if current.get("evaluation_protocol", "standard") != "standard":
            raise ExperimentConfigurationError("Legacy trial manifests require standard evaluation protocol.")
        # Legacy trial manifests contain only these identities. Their individual
        # checkpoints still enforce full configuration during construction.
        current = {
            "schema_version": 1,
            "trials": current["trials"],
            "algorithms": {
                k: {field: v[field] for field in ("class", "base_seed")}
                for k, v in current["algorithms"].items()
            },
            "environment_seeds": current["environment_seeds"],
        }
    if saved != current:
        raise ExperimentConfigurationError(
            "Saved trial settings differ: count, labels, classes, base seeds, "
            "trial seeds, algorithm configuration, environment or stopping rule must match."
        )


def _trial_seeds(
    value: Sequence[int] | Mapping[str, Sequence[int]] | None,
    labels: Sequence[str],
    count: int,
) -> dict[str, list[int]] | None:
    """Validate an explicit schedule before constructing or resuming trials."""
    from .experiment import ExperimentConfigurationError

    if value is None:
        return None
    if isinstance(value, Mapping):
        if set(value) != set(labels):
            raise ExperimentConfigurationError(
                "trial_seeds must contain exactly the experiment's algorithm labels."
            )
        schedules = value
    else:
        schedules = {label: value for label in labels}
    normalized = {}
    for label, seeds in schedules.items():
        if (isinstance(seeds, (str, bytes))
                or not isinstance(seeds, (Sequence, np.ndarray))
                or (isinstance(seeds, np.ndarray) and seeds.ndim != 1)):
            raise ExperimentConfigurationError(
                f"trial_seeds for {label!r} must be a sequence of {count} integers."
            )
        if len(seeds) != count:
            raise ExperimentConfigurationError(
                f"trial_seeds for {label!r} must contain exactly {count} seeds."
            )
        if any(isinstance(seed, bool) or not isinstance(seed, Integral)
               or not 0 <= int(seed) < 2**32 for seed in seeds):
            raise ExperimentConfigurationError(
                f"trial_seeds for {label!r} must be integers in [0, 2**32); "
                "booleans are not valid seeds."
            )
        normalized[label] = [int(seed) for seed in seeds]
    return normalized


def run_trials(
    experiment: Any, count: int, parameters: dict[str, Any], *, max_trials: int | None = None,
    trial_seeds: Sequence[int] | Mapping[str, Sequence[int]] | None = None,
) -> ExperimentTrialsResult:
    """Run independent trials, optionally using exact per-method seed schedules.

    Explicit seeds initialize both the algorithm and its environment. Omitting
    them retains the existing base-seed-plus-trial-index convention. Schedules
    are checkpoint identities and cannot change when continuing saved trials.
    """
    from .experiment import Experiment, ExperimentCheckpointError, ExperimentConfigurationError

    if parameters["plot"] is not None and not isinstance(parameters["plot"], bool):
        raise ExperimentConfigurationError("plot must be bool or None.")
    if not isinstance(parameters["show"], bool):
        raise ExperimentConfigurationError("show must be bool.")
    schedule = _trial_seeds(trial_seeds, tuple(experiment.algorithms), count)
    root = experiment.output_directory
    manifest_path = None if root is None else root / "trials.json"
    if not hasattr(experiment, "_trial_count"):

        def has_new_observations(label: str, algorithm: Any) -> bool:
            history = experiment.source_environments[label].history
            return bool(history.records) or algorithm.evaluations > 0

        if any(has_new_observations(k, a) for k, a in experiment.algorithms.items()):
            raise ExperimentConfigurationError(
                "Start trials from a fresh Experiment; this one has observations."
            )
        experiment._trial_templates = dict(experiment.algorithms)
        experiment._trial_environments = copy.deepcopy(dict(experiment.source_environments))
        experiment._trial_rule = copy.deepcopy(experiment.stopping_rule)
        experiment._trial_count = count
        experiment._trial_seed_schedule = schedule
        experiment.trial_experiments = []
        experiment._trial_documents = []
        experiment._trial_results = []
    if count != experiment._trial_count:
        raise ExperimentConfigurationError(
            "The number of trials cannot change during continuation."
        )
    if schedule != experiment._trial_seed_schedule:
        raise ExperimentConfigurationError(
            "The trial_seeds schedule cannot change during continuation."
        )
    manifest = _manifest(experiment, count)
    if manifest_path is not None:
        if manifest_path.exists():
            try:
                saved = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ExperimentCheckpointError("Trial manifest is unreadable.") from error
            _validate_manifest(saved, manifest)
            if not experiment.resume and not experiment.trial_experiments:
                raise ExperimentConfigurationError(
                    "Trial results already exist; use resume=True or a new directory."
                )
        else:
            experiment._atomic_json_save(manifest, manifest_path)
    results = experiment._trial_results
    limit = count if max_trials is None else max_trials
    token = ACTIVE.set(True)
    try:
        for index in range(limit):
            with experiment._capture_warnings("trial runs"):
                if parameters["progress"]:
                    seed_description = (f"seed offset +{index}" if schedule is None else
                                        ", ".join(f"{label} seed {seeds[index]}"
                                                  for label, seeds in schedule.items()))
                    print(f"Trial {index + 1}/{count} ({seed_description})", flush=True)
                if index < len(experiment.trial_experiments):
                    child = experiment.trial_experiments[index]
                else:
                    algorithms = {
                        k: a.clone_for_trial(index if schedule is None else schedule[k][index] - a.seed)
                        for k, a in experiment._trial_templates.items()
                    }
                    environments = copy.deepcopy(experiment._trial_environments)
                    for label, env in environments.items():
                        env.reset(seed=env.seed + index if schedule is None else schedule[label][index])
                    multi = experiment.is_multi_algorithm
                    child = Experiment(
                        environment=environments if multi else next(iter(environments.values())),
                        algorithm=algorithms if multi else next(iter(algorithms.values())),
                        stopping_rule=copy.deepcopy(experiment._trial_rule),
                        output_directory=None
                        if root is None
                        else root / "trials" / f"trial-{index + 1:03d}",
                        resume=experiment.resume,
                        save_every=experiment.save_every,
                        checkpoint_policy=experiment.checkpoint_policy,
                        constant_context=experiment._trial_constant_context,
                        evaluation_protocol=experiment.evaluation_protocol,
                    )
                    experiment.trial_experiments.append(child)
                    if not child.is_multi_algorithm:
                        if child._reported_elapsed_seconds is None:
                            child._reported_elapsed_seconds = time.monotonic() - child._started_at
                        child._elapsed_reporting_managed = True
                child._elapsed_report_started_at = time.monotonic()
                result = None
                try:
                    result = child.run(**{**parameters, "plot": False, "show": False})
                except (ExperimentConfigurationError, ExperimentCheckpointError):
                    raise
                except Exception as error:
                    if child.is_multi_algorithm:
                        raise  # Individual method failures are handled by the child.
                    label = next(iter(child.algorithms))
                    child._report_child_failure(
                        label=f"{label} / trial {index + 1}", phase="run", error=error,
                        continuing=not parameters["fail_fast"],
                    )
                    if parameters["fail_fast"]:
                        raise
                finally:
                    if not child.is_multi_algorithm:
                        child._reported_elapsed_seconds = child.elapsed_seconds
                    child._elapsed_report_started_at = None
                    if index < len(results):
                        results[index] = result
                    else:
                        results.append(result)
                    document = child._result_document()
                    if index < len(experiment._trial_documents):
                        experiment._trial_documents[index] = document
                    else:
                        experiment._trial_documents.append(document)
                    experiment.completed_rounds = (
                        min(c.completed_rounds for c in experiment.trial_experiments)
                        if len(experiment.trial_experiments) == count
                        else 0
                    )
                    experiment.failures = {
                        f"{label} / trial {i + 1}": failure
                        for i, item in enumerate(experiment.trial_experiments)
                        for label, failure in item.failures.items()
                    }
                    if root is not None:
                        experiment.save()
    finally:
        ACTIVE.reset(token)
    stat = {k: statistics(experiment._trial_documents, k) for k in experiment.algorithms}
    raw = {
        k: tuple((r if not experiment.is_multi_algorithm or r is None else r.results.get(k))
                 for r in results)
        for k in experiment.algorithms
    }
    failures = {
        f"{label} / trial {i + 1}": failure
        for i, child in enumerate(experiment.trial_experiments)
        for label, failure in child.failures.items()
    }
    combined = ExperimentTrialsResult(tuple(results), stat, raw, failures,
        records=tuple(copy.deepcopy(experiment._trial_documents)), planned_trials=count)
    experiment.trial_result = combined
    experiment.failures = dict(failures)
    if parameters["plot"] is True or (
        parameters["plot"] is None and any(a.plot for a in experiment.algorithms.values())
    ):
        from .plotting import plot

        figure = plot(experiment, title=parameters["title"], show=parameters["show"])
        combined.figure, combined.axes = figure.figure, figure.axes
    if root is not None:
        experiment.save()
    return combined


def _group_design(environment: Any, config: GroupedTrialConfig, index: int) -> tuple[Any, Any]:
    """Draw only the group's shared initial design, retaining its consumed stream."""
    generator = np.random.RandomState(config.seed + index)
    lower = environment.bounds.lower.detach().cpu().numpy()
    upper = environment.bounds.upper.detach().cpu().numpy()
    actions = generator.uniform(lower, upper, size=(config.initial_points, len(lower)))
    state: Any = generator.get_state(legacy=True)
    streams = {
        "numpy": {
            "kind": state[0], "keys": state[1].tolist(), "position": state[2],
            "has_gauss": state[3], "cached_gaussian": state[4],
        },
        "torch": torch.Generator().manual_seed(config.seed + index).get_state().tolist(),
    }
    return actions.tolist(), streams


def _grouped_record(child: Any, *, group: int, seed: int, label: str, repetition: int,
                    initial_points: int, failure: Any = None) -> dict[str, Any]:
    records = child.environment.history.records
    reason = (
        child.stopping_decision.reason
        if child.stopping_decision is not None and child.stopping_decision.stop
        else child._run_stop_reason
    )
    checkpoint_manifest = {}
    try:
        checkpoint_manifest = json.loads((child.output_directory / "latest.json").read_text())
    except (OSError, ValueError):
        if failure is None:
            raise
    return {
        "group_index": group, "group_seed": seed, "algorithm": label,
        "repetition": repetition, "status": "failed" if failure else "completed",
        "observations": [float(r.observation) for r in records],
        "actions": [r.action.detach().cpu().tolist() for r in records],
        "evaluation_dtypes": [
            r.metadata.get("evaluation_dtype", str(child.environment.dtype).removeprefix("torch."))
            for r in records
        ],
        "initial_observation_count": initial_points, "evaluations": len(records),
        "new_evaluations": max(0, len(records) - initial_points),
        "stopping_reason": reason, "failure": failure,
        "output_directory": str(child.output_directory),
        "checkpoint": checkpoint_manifest.get("checkpoint"),
        "checkpoint_sha256": checkpoint_manifest.get("sha256"),
        "elapsed_seconds": time.monotonic() - child._started_at,
        "warnings": copy.deepcopy(child.warnings),
    }


def run_grouped_trials(
    experiment: Any,
    config: GroupedTrialConfig,
    *,
    max_evaluations: int,
    max_groups: int | None = None,
    progress: bool = False,
) -> GroupedExperimentResult:
    """Run ordered shared-stream groups with an atomic continuation cursor.

    The evaluation ceiling includes the shared initial observations. Initial
    *actions* are shared; each repetition evaluates them independently. Group
    ``g`` has seed ``config.seed + g``. A completed group is read-only on resume.
    ``max_groups`` is a cumulative runtime ceiling, not part of the scientific
    protocol: use 1 for a pilot, then omit it to continue the unchanged plan.

    Every child has a transactional Experiment checkpoint. The parent commits
    the next cursor and exported private streams together only after that child
    completes. Recovery resumes a pending child before handing its final streams
    to the next repetition. A failure is recorded and raised, never skipped.
    """
    from .algorithms.algorithm import Algorithm
    from .experiment import Experiment, ExperimentCheckpointError, ExperimentConfigurationError

    def invalid(message: str) -> None:
        raise ExperimentConfigurationError(message)

    if not isinstance(config, GroupedTrialConfig):
        invalid("grouped_trials must be a GroupedTrialConfig.")
    if experiment.evaluation_protocol != "standard":
        invalid("Grouped shared-stream trials require evaluation_protocol='standard'.")
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int):
        invalid("Grouped max_evaluations must be an integer including initialization.")
    if max_evaluations < config.initial_points:
        invalid("Grouped max_evaluations cannot be smaller than initial_points.")
    if max_groups is not None and (
        isinstance(max_groups, bool) or not isinstance(max_groups, int)
        or not 0 <= max_groups <= config.groups
    ):
        invalid("max_groups must be between zero and the planned number of groups.")
    if not isinstance(progress, bool):
        invalid("progress must be Boolean.")
    root = experiment.output_directory
    if root is None:
        invalid("Grouped trials require output_directory for durable shared-stream continuation.")
    assert root is not None
    if experiment.checkpoint_policy != "transactional":
        invalid("Shared-stream grouped trials require checkpoint_policy='transactional'.")
    if hasattr(experiment, "_trial_count") or (root / "trials.json").exists():
        invalid("Ordinary and grouped trials require separate output directories.")
    algorithms = dict(experiment.algorithms)
    environments = dict(experiment.source_environments)
    labels = list(algorithms)
    if experiment.failures or set(environments) != set(algorithms):
        invalid("Every grouped algorithm must be compatible with its environment.")
    first_environment = environments[labels[0]]
    for label, algorithm in algorithms.items():
        environment = environments[label]
        if not getattr(algorithm, "_supports_shared_trial_streams", False):
            invalid(f"{label} does not support the explicit released shared-stream protocol.")
        if algorithm.request_size != 1:
            invalid("Grouped shared-stream methods must request one observation at a time.")
        if environment.history.records or algorithm.evaluations or environment.history.initial_data is not None:
            invalid("Start grouped trials from fresh templates without previous observations.")
        if (environment.environment_type != "stochastic" or environment.source_mode != "function"
                or environment.noise_std != 0 or environment.noise_fn is not None):
            invalid("Grouped shared initialization currently requires a noiseless stochastic callable.")
        if environment.dtype != torch.float64 or environment.device.type != "cpu":
            invalid("The released shared-stream protocol requires CPU float64 storage.")
        if not torch.equal(environment.bounds.lower, first_environment.bounds.lower) or not torch.equal(
            environment.bounds.upper, first_environment.bounds.upper
        ):
            invalid("Grouped methods must share the same initial-design bounds.")

    protocol = {
        "schema_version": 1, "grouped_trials": asdict(config),
        "algorithm_order": labels, "max_evaluations": max_evaluations,
        "initialization": "numpy_mt19937_uniform_shared_actions_independent_evaluation",
        "budget_includes_initialization": True,
        "algorithms": {label: algorithm._configuration_snapshot() for label, algorithm in algorithms.items()},
        "environments": {label: Algorithm._fingerprint_value(env.state_dict()) for label, env in environments.items()},
        "stopping_rule": None if experiment.stopping_rule is None else Algorithm._fingerprint_value(experiment.stopping_rule.state_dict()),
        "save_every": experiment.save_every, "checkpoint_policy": experiment.checkpoint_policy,
    }
    protocol = Experiment._jsonable(protocol)
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    manifest_path = root / "grouped_trials.json"
    if manifest_path.exists():
        try:
            saved = json.loads(manifest_path.read_text())
        except (OSError, ValueError) as error:
            raise ExperimentCheckpointError("Grouped trial manifest is unreadable.") from error
        if saved != protocol:
            invalid("Saved grouped protocol differs; preserve it and choose a new output_directory.")
        if not experiment.resume and not hasattr(experiment, "grouped_result"):
            invalid("Existing grouped results require resume=True or a new output_directory.")
    else:
        Experiment._atomic_json_save(protocol, manifest_path)

    group_documents = []
    limit = config.groups if max_groups is None else max_groups
    for index in range(config.groups):
        directory = root / "groups" / f"group-{index:03d}-seed-{config.seed + index}"
        state_path = directory / "state.json"
        state = None
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
            except (OSError, ValueError) as error:
                raise ExperimentCheckpointError(f"Unreadable group state: {state_path}") from error
            if (not isinstance(state, dict) or state.get("protocol_sha256") != identity
                    or state.get("group_index") != index):
                raise ExperimentCheckpointError(f"Invalid group state identity: {state_path}")
            cursor = state.get("next_run_index")
            if (isinstance(cursor, bool) or not isinstance(cursor, int)
                    or not 0 <= cursor <= len(labels) * config.repetitions
                    or len(state.get("records", [])) != cursor):
                raise ExperimentCheckpointError(f"Invalid group continuation cursor: {state_path}")
            for slot, record in enumerate(state["records"]):
                method, repetition = divmod(slot, config.repetitions)
                expected_directory = directory / f"method-{method:02d}" / f"repeat-{repetition:03d}"
                if (record.get("algorithm") != labels[method]
                        or record.get("repetition") != repetition
                        or record.get("group_index") != index
                        or record.get("status") != "completed"
                        or record.get("checkpoint") not in {"checkpoint-slot-a.pt", "checkpoint-slot-b.pt"}):
                    raise ExperimentCheckpointError("Completed grouped record order or identity differs.")
                checkpoint = expected_directory / record["checkpoint"]
                if not checkpoint.is_file() or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != record.get("checkpoint_sha256"):
                    raise ExperimentCheckpointError("Completed grouped checkpoint checksum differs.")
        if index >= limit:
            if state is not None:
                group_documents.append(state)
            continue
        if state is None:
            directory.mkdir(parents=True, exist_ok=True)
            actions, streams = _group_design(first_environment, config, index)
            state = {
                "schema_version": 1, "protocol_sha256": identity, "group_index": index,
                "group_seed": config.seed + index, "initial_actions": actions,
                "random_streams": streams, "next_run_index": 0, "records": [],
                "status": "running", "active_record": None,
            }
            Experiment._atomic_json_save(state, state_path)
        while state["next_run_index"] < len(labels) * config.repetitions:
            slot = state["next_run_index"]
            method_index, repetition = divmod(slot, config.repetitions)
            label = labels[method_index]
            child_root = directory / f"method-{method_index:02d}" / f"repeat-{repetition:03d}"
            child_root.mkdir(parents=True, exist_ok=True)
            if progress:
                print(f"Group {index + 1}/{config.groups} / {label} / repetition {repetition + 1}/{config.repetitions}", flush=True)
            child = None
            try:
                environment = copy.deepcopy(environments[label])
                environment.reset(seed=config.seed + index)
                algorithm = algorithms[label].clone_for_trial(config.seed + index - algorithms[label].seed)
                checkpoint_exists = (child_root / "latest.json").exists()
                if not checkpoint_exists:
                    # Initial observations are separately durable before the
                    # learner is initialized; do not evaluate committed rows twice.
                    initial_path = child_root / "initialization.pt"
                    if initial_path.exists():
                        payload = torch.load(initial_path, map_location="cpu", weights_only=True)
                        if payload.get("protocol_sha256") != identity:
                            raise ExperimentCheckpointError("Initial-design checkpoint identity differs.")
                        environment.load_state_dict(payload["environment_state"])
                    count = len(environment.history.records)
                    if count > config.initial_points:
                        raise ExperimentCheckpointError("Initial-design checkpoint contains excess observations.")
                    for row, expected in zip(environment.history.records, state["initial_actions"], strict=False):
                        if row.action.detach().cpu().tolist() != expected:
                            raise ExperimentCheckpointError("Initial-design checkpoint action differs.")
                    for action in state["initial_actions"][count:]:
                        environment.step(action)
                        Experiment._atomic_torch_save({
                            "protocol_sha256": identity, "environment_state": environment.state_dict(),
                        }, initial_path)
                child = Experiment(
                    environment=environment, algorithm=algorithm,
                    stopping_rule=copy.deepcopy(experiment.stopping_rule),
                    output_directory=child_root, resume=checkpoint_exists,
                    save_every=experiment.save_every, checkpoint_policy="transactional",
                    evaluation_protocol=experiment.evaluation_protocol,
                )
                stream_algorithm: Any = child.algorithm
                if not checkpoint_exists:
                    stream_algorithm._import_random_streams(state["random_streams"])
                    child.save_checkpoint(phase="group_initialization")
                child.run(max_evaluations=max_evaluations - config.initial_points, plot=False, show=False)
                record = _grouped_record(child, group=index, seed=config.seed + index,
                                         label=label, repetition=repetition, initial_points=config.initial_points)
                if record["evaluations"] < max_evaluations and not (
                    child.stopping_decision is not None and child.stopping_decision.stop
                ):
                    raise ExperimentCheckpointError("Grouped run returned before its budget or stopping rule.")
                advanced = copy.deepcopy(state)
                advanced["random_streams"] = Experiment._jsonable(stream_algorithm._export_random_streams())
                advanced["records"].append(record)
                advanced["next_run_index"] = slot + 1
                advanced["active_record"] = None
                advanced["status"] = "completed" if slot + 1 == len(labels) * config.repetitions else "running"
                Experiment._atomic_json_save(advanced, state_path)
                state = advanced
            except BaseException as error:
                failure = {"type": type(error).__name__, "message": str(error)}
                state["active_record"] = (
                    _grouped_record(child, group=index, seed=config.seed + index,
                                    label=label, repetition=repetition,
                                    initial_points=config.initial_points, failure=failure)
                    if child is not None else {
                        "group_index": index, "group_seed": config.seed + index,
                        "algorithm": label, "repetition": repetition, "status": "failed",
                        "observations": [], "actions": [], "evaluation_dtypes": [],
                        "evaluations": 0, "new_evaluations": 0,
                        "initial_observation_count": config.initial_points,
                        "stopping_reason": None, "failure": failure,
                        "output_directory": str(child_root),
                    }
                )
                state["status"] = "failed"
                Experiment._atomic_json_save(state, state_path)
                raise
        group_documents.append(state)
    records = tuple(row for group in group_documents for row in (
        group["records"] + ([] if group.get("active_record") is None else [group["active_record"]])
    ))
    result = GroupedExperimentResult(records, tuple(group_documents), config.groups,
                                     config.repetitions, tuple(labels), protocol)
    experiment.grouped_result = result
    Experiment._atomic_json_save(Experiment._jsonable(result), root / "grouped_results.json")
    return result
