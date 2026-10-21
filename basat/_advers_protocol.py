"""Transactional initialization for the opt-in released AdveRS evaluation policy.

The protocol owns observation scheduling and random-stream handoff only. It
never selects optimizer actions or supplies hidden objective values to a learner.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

import torch

from .environments.Adversarial import (
    AdversarialEnvironment, LCBAttack, RandomAttack, WorstCaseAttack, _SOURCE_NORMAL_POLICY,
)
from .environments.environment import EnvironmentData
from .workflow import RandomStream

VERSION = 1
SOURCE_PROTOCOL_VERSIONS = {"advers_2025": VERSION, "advers_2026_camera_ready": 2}


def _entry(experiment: Any) -> dict[str, Any] | None:
    """Validate a protocol-owned, native-predecessor stream before any draw."""
    from .experiment import ExperimentCheckpointError
    value = getattr(experiment, "_advers_source_entry", None)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ExperimentCheckpointError("Invalid shared AdveRS source entry.")
    body = {key: item for key, item in value.items() if key != "sha256"}
    expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if (set(value) != {"version", "protocol_sha256", "task_tag", "task_seed", "repetition", "rng_state", "sha256"}
            or value.get("sha256") != expected or value.get("version") != 1
            or value.get("task_seed") != experiment.algorithm.seed
            or experiment.algorithm.config.source_protocol != "advers_2026_camera_ready"
            or not isinstance(experiment.environment.attack, (LCBAttack, RandomAttack))):
        raise ExperimentCheckpointError("Shared AdveRS source entry identity differs.")
    stream = RandomStream(experiment.algorithm.seed)
    stream.set_state(value["rng_state"])
    return copy.deepcopy(value)


def source_normal_policy(experiment: Any) -> str | None:
    """Return camera-ready scalar-normal arithmetic for LCB/random attacks."""
    if (experiment.algorithm.config.source_protocol == "advers_2026_camera_ready"
            and isinstance(experiment.environment.attack, (LCBAttack, RandomAttack))):
        return _SOURCE_NORMAL_POLICY
    return None


def step_options(experiment: Any) -> dict[str, str]:
    """Keep source noise arithmetic explicit at the environment evaluation boundary."""
    policy = source_normal_policy(experiment)
    return {} if policy is None else {"source_normal_policy": policy}


def validate(experiment: Any, *, version: int | None = None) -> None:
    from .experiment import ExperimentConfigurationError

    env, algorithm = experiment.environment, experiment.algorithm
    if (
        not isinstance(env, AdversarialEnvironment)
        or not isinstance(env.attack, (LCBAttack, RandomAttack, WorstCaseAttack))
        or env.source_mode != "function"
        or env.candidate_actions is None
        or env.noise_fn is not None
        or env.history.initial_data is not None
        or not env.allow_repeated_actions
        or getattr(algorithm.config, "source_protocol", None) not in SOURCE_PROTOCOL_VERSIONS
    ):
        raise ExperimentConfigurationError(
            "advers_2025 evaluation requires a function-backed finite "
            "AdversarialEnvironment with LCBAttack, RandomAttack or WorstCaseAttack, repeated actions enabled, "
            "no previous_data/custom noise, and an advers_2025 or "
            "advers_2026_camera_ready source algorithm."
        )
    if version is not None and (
        isinstance(version, bool)
        or version != SOURCE_PROTOCOL_VERSIONS[algorithm.config.source_protocol]
    ):
        raise ExperimentConfigurationError("The requested AdveRS initialization version is incompatible.")
    _entry(experiment)
    count = algorithm.config.initial_points
    if not 1 <= count <= len(env.candidate_actions) or algorithm.batch_size != 1:
        raise ExperimentConfigurationError(
            "advers_2025 requires batch_size=1 and a feasible initial_points count."
        )


def query_counts(experiment: Any) -> dict[str, int]:
    bootstrap = experiment._advers_bootstrap
    initial = 0 if bootstrap is None else len(bootstrap["observations"])
    online = len(experiment.environment.history.records)
    return {
        "initialization_queries": initial,
        "online_queries": online,
        "total_objective_queries": initial + online,
    }


def _initial_data(experiment: Any, bootstrap: dict[str, Any]) -> EnvironmentData:
    env = experiment.environment
    points = env.candidate_actions[bootstrap["indices"]]
    return env._validate_environment_data(EnvironmentData(
        actions=points,
        observations=torch.tensor(bootstrap["observations"], dtype=env.dtype, device=env.device),
        observation_noise_variance=torch.full(
            (len(points),), env.noise_std**2, dtype=env.dtype, device=env.device
        ),
        metadata={"generated_initialization": experiment.algorithm.config.source_protocol,
                  "protocol_version": SOURCE_PROTOCOL_VERSIONS[experiment.algorithm.config.source_protocol],
                  **step_options(experiment)},
    ))


def initialize(experiment: Any, ceiling: int | None = None) -> bool:
    """Commit each clean initial query once, then reveal the complete design."""
    env, algorithm = experiment.environment, experiment.algorithm
    bootstrap = experiment._advers_bootstrap
    if bootstrap is None:
        stream = RandomStream(algorithm.seed, engine="mt19937")
        entry = _entry(experiment)
        if entry is not None:
            stream.set_state(entry["rng_state"])
        indices = stream.choice(
            len(env.candidate_actions), size=algorithm.config.initial_points, replace=False
        ).tolist()
        bootstrap = {
            "version": SOURCE_PROTOCOL_VERSIONS[algorithm.config.source_protocol],
            "seed": algorithm.seed, "indices": indices,
            "observations": [], "initialized": False,
        }
        if algorithm.config.source_protocol == "advers_2026_camera_ready":
            bootstrap["source_protocol"] = algorithm.config.source_protocol
        bootstrap.update(step_options(experiment))
        if entry is not None:
            bootstrap["source_entry"] = entry
        experiment._advers_bootstrap = bootstrap
        algorithm.set_source_random_state(stream.get_state())
        experiment.save_checkpoint(phase="initial_design", archive=False)
    if bootstrap["initialized"]:
        return True
    while len(bootstrap["observations"]) < len(bootstrap["indices"]):
        if ceiling is not None and query_counts(experiment)["total_objective_queries"] >= ceiling:
            return False
        index = bootstrap["indices"][len(bootstrap["observations"])]
        value = float(env.evaluate_function(env.candidate_actions[index]))
        bootstrap["observations"].append(value)
        experiment.save_checkpoint(phase="initial_observation", archive=False)
    env.history.initial_data = _initial_data(experiment, bootstrap)
    experiment._initialize_algorithm_from_environment()
    bootstrap["initialized"] = True
    experiment.save_checkpoint(phase="initialized", archive=False)
    return True


def restore(experiment: Any, raw: Any, environment_state: dict, algorithm_state: dict) -> None:
    """Validate generated history separately from immutable user previous_data."""
    from .experiment import ExperimentCheckpointError

    if raw is None:
        if algorithm_state.get("initialized") or environment_state["history"].get("initial_data"):
            raise ExperimentCheckpointError("Missing advers_2025 initialization state.")
        return
    bootstrap = copy.deepcopy(raw)
    stream = RandomStream(experiment.algorithm.seed, engine="mt19937")
    entry = _entry(experiment)
    if entry is not None:
        stream.set_state(entry["rng_state"])
    expected = stream.choice(
        len(experiment.environment.candidate_actions),
        size=experiment.algorithm.config.initial_points, replace=False,
    ).tolist()
    observations = bootstrap.get("observations") if isinstance(bootstrap, dict) else None
    if (
        not isinstance(bootstrap, dict)
        or bootstrap.get("version") != SOURCE_PROTOCOL_VERSIONS[experiment.algorithm.config.source_protocol]
        or (experiment.algorithm.config.source_protocol == "advers_2026_camera_ready"
            and bootstrap.get("source_protocol") != "advers_2026_camera_ready")
        or bootstrap.get("source_normal_policy") != source_normal_policy(experiment)
        or bootstrap.get("source_entry") != entry
        or bootstrap.get("seed") != experiment.algorithm.seed
        or bootstrap.get("indices") != expected
        or not isinstance(observations, list)
        or len(observations) > len(expected)
        or any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in observations)
        or not isinstance(bootstrap.get("initialized"), bool)
        or bootstrap["initialized"] != algorithm_state.get("initialized")
        or (bootstrap["initialized"] and len(observations) != len(expected))
    ):
        raise ExperimentCheckpointError("Invalid advers_2025 initialization state.")
    if bootstrap["initialized"]:
        experiment.environment.history.initial_data = _initial_data(experiment, bootstrap)
    elif environment_state["history"].get("initial_data") is not None:
        raise ExperimentCheckpointError("Incomplete initialization has unexpected history.")
    experiment._advers_bootstrap = bootstrap
