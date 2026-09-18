"""Independent seeded repetitions and evaluation-aligned statistics."""

from __future__ import annotations

import copy
import inspect
import json
import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

import numpy as np
import torch

ACTIVE: ContextVar[bool] = ContextVar("basat_trial", default=False)


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
    """Raw trial results and aggregate curves. No averaged action is invented."""

    trials: tuple[Any, ...]
    statistics: Mapping[str, TrialStatistics]
    results: Mapping[str, tuple[Any, ...]]
    failures: Mapping[str, Any]
    figure: Any = None
    axes: Any = None

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
            "stopping_reason": "see individual trial records",
            "trials": records,
        }
    return {
        "schema_version": 1,
        "package": "basat",
        "mode": "trials",
        "trial_count": experiment._trial_count,
        "completed_trials": len(rows),
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
    }


def _validate_manifest(saved: Any, current: dict[str, Any]) -> None:
    from .experiment import ExperimentCheckpointError, ExperimentConfigurationError

    if not isinstance(saved, dict) or saved.get("schema_version") not in {1, 2}:
        raise ExperimentCheckpointError("Trial manifest schema is invalid or unsupported.")
    if saved["schema_version"] == 1:
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
            "algorithm configuration, environment or stopping rule must match."
        )


def run_trials(experiment: Any, count: int, parameters: dict[str, Any]) -> ExperimentTrialsResult:
    from .experiment import Experiment, ExperimentCheckpointError, ExperimentConfigurationError

    if parameters["plot"] is not None and not isinstance(parameters["plot"], bool):
        raise ExperimentConfigurationError("plot must be bool or None.")
    if not isinstance(parameters["show"], bool):
        raise ExperimentConfigurationError("show must be bool.")
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
        experiment.trial_experiments = []
        experiment._trial_documents = []
    if count != experiment._trial_count:
        raise ExperimentConfigurationError(
            "The number of trials cannot change during continuation."
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
    results = []
    token = ACTIVE.set(True)
    try:
        for index in range(count):
            with experiment._capture_warnings("trial runs"):
                if parameters["progress"]:
                    print(f"Trial {index + 1}/{count} (seed offset +{index})", flush=True)
                if index < len(experiment.trial_experiments):
                    child = experiment.trial_experiments[index]
                else:
                    algorithms = {
                        k: a.clone_for_trial(index) for k, a in experiment._trial_templates.items()
                    }
                    environments = copy.deepcopy(experiment._trial_environments)
                    for env in environments.values():
                        env.reset(seed=env.seed + index)
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
                    )
                    experiment.trial_experiments.append(child)
                try:
                    result = child.run(**{**parameters, "plot": False, "show": False})
                    results.append(result)
                except (ExperimentConfigurationError, ExperimentCheckpointError):
                    raise
                except Exception as error:
                    if child.is_multi_algorithm:
                        raise  # Individual method failures are handled by the child.
                    label = next(iter(child.algorithms))
                    child._report_child_failure(
                        label=f"{label} / trial {index + 1}", phase="run", error=error
                    )
                    results.append(None)
                finally:
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
                    if root is not None:
                        experiment.save()
    finally:
        ACTIVE.reset(token)
    stat = {k: statistics(experiment._trial_documents, k) for k in experiment.algorithms}
    raw = {
        k: tuple((r if not experiment.is_multi_algorithm else r.results.get(k)) for r in results)
        for k in experiment.algorithms
    }
    failures = {
        f"{label} / trial {i + 1}": failure
        for i, child in enumerate(experiment.trial_experiments)
        for label, failure in child.failures.items()
    }
    combined = ExperimentTrialsResult(tuple(results), stat, raw, failures)
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
