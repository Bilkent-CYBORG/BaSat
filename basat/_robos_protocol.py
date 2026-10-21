"""Transactional replay of the released RoBOS synthetic experiment.

The coordinator controls random draws and evaluation only. Acquisition decisions
remain in native DRBO/RoBOS; complete payoffs are never passed to those learners.
Source: Bilkent-CYBORG/RoBOS, revision 96bf394 (MIT).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

VERSION = 1
SOURCE = "96bf39400a0b2044e3c21e08b4fbeb8886584ee3"
ROLES = ("drbo_3epsilon", "drbo_epsilon", "drbo_epsilon_over_3", "robos")


@dataclass(frozen=True)
class RoBOSSourceProtocol:
    """Released shared-stream trial policy, separate from algorithm mathematics.

    ``seed`` initializes MT19937 once for the entire campaign. The three
    initialization rewards are evaluated once per trial and shared. Excluded
    WRBO/SO noise draws remain reserved in ``rng_schedule='released'``. Method
    roles map display labels to ``drbo_3epsilon``, ``drbo_epsilon``,
    ``drbo_epsilon_over_3``, and ``robos`` in that order.

    Examples
    --------
    >>> policy = RoBOSSourceProtocol(method_roles={
    ...     'DRBO 3e': 'drbo_3epsilon', 'DRBO e': 'drbo_epsilon',
    ...     'DRBO e/3': 'drbo_epsilon_over_3', 'RoBOS': 'robos'})
    >>> policy.online_rounds
    201
    """
    seed: int = 123
    initial_points: int = 3
    online_rounds: int = 201
    method_roles: Mapping[str, str] | None = None
    rng_schedule: str = "released"

    def __post_init__(self):
        for name in ("seed", "initial_points", "online_rounds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be a valid nonnegative seed or positive count.")
        if self.seed >= 2**32 or self.rng_schedule != "released":
            raise ValueError("Use a uint32 seed and rng_schedule='released'.")
        if self.method_roles is None or tuple(self.method_roles.values()) != ROLES:
            raise ValueError(f"method_roles must contain the four ordered roles {ROLES}.")
        object.__setattr__(self, "method_roles", dict(self.method_roles))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _rng_state(rng):
    kind, keys, position, has_gauss, cached = rng.get_state()
    return dict(kind=kind, keys=keys.tolist(), position=position, has_gauss=has_gauss, cached=cached)


def _restore_rng(state):
    rng = np.random.RandomState()
    rng.set_state((state["kind"], np.asarray(state["keys"], dtype=np.uint32),
                   state["position"], state["has_gauss"], state["cached"]))
    return rng


def make_plan(protocol, rng, probability, *, action_count, trial_index):
    """Draw a whole trial in released order, without evaluating any objective."""
    contexts, rounds = len(probability), protocol.online_rounds
    start = _rng_state(rng)
    initial = rng.randint(0, contexts * action_count, protocol.initial_points)
    discarded_initial_noise = rng.randn(protocol.initial_points)
    uniforms = rng.rand(rounds)
    context_ids = [int(np.sum(np.cumsum(probability) <= u)) for u in uniforms]
    # Fourth row is the excluded WRBO noise stream, not an additional run.
    baseline_noise = rng.randn(4, rounds) * .02
    proposals, robos_noise = [], []
    for _ in range(rounds):
        starts = [[], []]
        for _action in range(action_count):
            for who in (0, 1):
                starts[who].append(np.diff(np.r_[0., np.sort(rng.rand(contexts - 1)), 1.]).tolist())
        proposals.append(dict(learner=starts[0], oracle=starts[1]))
        robos_noise.append(float(rng.randn(1, 1).item() * .02))
    discarded_so_noise = rng.randn(rounds) * .02
    body = dict(version=VERSION, trial_index=trial_index, start_rng=start, end_rng=_rng_state(rng),
        initial_indices=initial.tolist(), discarded_initial_normals=discarded_initial_noise.tolist(),
        context_uniforms=uniforms.tolist(), context_indices=context_ids,
        baseline_noise=baseline_noise.tolist(), robos_noise=robos_noise, starts=proposals,
        discarded_so_noise=discarded_so_noise.tolist())
    return {**body, "sha256": digest(body)}


def _check_plan(plan):
    from .experiment import ExperimentCheckpointError
    if not isinstance(plan, dict) or plan.get("sha256") != digest({k:v for k,v in plan.items() if k != "sha256"}):
        raise ExperimentCheckpointError("RoBOS random plan checksum mismatch.")


def _runtime():
    return {name: importlib.metadata.version(name) for name in
            ("numpy", "scipy", "cvxpy", "clarabel", "torch", "gpytorch", "botorch")}


def _require_runtime():
    from .experiment import ExperimentConfigurationError
    expected = {"numpy": "2.4.6", "scipy": "1.15.3", "cvxpy": "1.9.2", "clarabel": "0.11.1"}
    actual = _runtime()
    if any(actual[k] != v for k,v in expected.items()):
        raise ExperimentConfigurationError(
            f"RoBOS released execution requires the dedicated reproduction kernel {expected}; got {actual}. "
            "The main environment need not be changed; completed results remain readable.")
    return actual


def prepare_proposal(child):
    from .experiment import ExperimentProtocolError
    execution = child._robos_execution
    if execution is None:
        raise ExperimentProtocolError("robos_2023 must be run through a RoBOSSourceProtocol coordinator.")
    if execution.get("progress") and child.completed_rounds % 25 == 0:
        print(f"  {execution['label']}: {child.completed_rounds}/{len(execution['plan']['context_indices'])} online", flush=True)
    if child.algorithm.last_ask is None and execution["role"] == "robos":
        child.algorithm.set_source_solver_starts(execution["plan"]["starts"][child.completed_rounds]["learner"])


def evaluate_proposal(child, action):
    execution, t = child._robos_execution, child.completed_rounds
    plan, role = execution["plan"], execution["role"]
    noise = plan["robos_noise"][t] if role == "robos" else plan["baseline_noise"][ROLES.index(role)][t]
    selected = int(np.argmin(np.abs(execution["actions"] - float(action.reshape(-1)[0]))))
    feedback = child.environment._step_source_scheduled(action,
        context_index=plan["context_indices"][t], noise=noise,
        expected_observation=float(execution["expected"][selected]), plan_digest=plan["sha256"])
    child.environment.history.records[-1].audit["source_diagnostics"] = copy.deepcopy(
        child.algorithm.diagnostics().algorithm_specific)
    return feedback


def attach_regret(child, result, *, label, metric):
    from .experiment import ExperimentProtocolError
    execution = child._robos_execution
    if execution is None or "source_kappa" not in execution:
        raise ExperimentProtocolError("Source RS regret requires the completed, common numerical oracle.")
    rewards = [float(r.audit["expected_observation"]) for r in child.environment.history.records]
    kappa = execution["source_kappa"][:len(rewards)]
    values = metric.evaluate(rewards, source_kappa=kappa)
    report = dict(kind="robos_2023_source_rs", definition=asdict(metric),
        reward_source="expected_observation", source_kappa=kappa,
        global_optimum_certified=False, values=values.tolist(),
        instantaneous=metric.instantaneous(rewards, source_kappa=kappa).tolist())
    child.regret_curves[label] = torch.as_tensor(values, dtype=torch.float64)
    child._regret_reports[label] = report
    return replace(result, metadata={**dict(result.metadata), "regret": report})


def _validate(experiment, protocol, metrics):
    from .experiment import ExperimentConfigurationError
    from .environments.Distributional import DistributionalEnvironment
    from ._robos_regret import RoBOSSourceRegret
    from .algorithms.Distributional.DRBO import DRBO
    from .algorithms.Distributional.RoBOS import RoBOS
    if (not experiment.is_multi_algorithm or tuple(experiment.algorithms) != tuple(protocol.method_roles)
            or experiment.output_directory is None or experiment.stopping_rule is not None):
        raise ExperimentConfigurationError("RoBOS source protocol needs the four ordered methods, an output directory, and no stopping rule.")
    if not metrics or any(not isinstance(m, RoBOSSourceRegret) for m in metrics.values()):
        raise ExperimentConfigurationError("Supply a common RoBOSSourceRegret for source trials.")
    definitions = [asdict(m) for m in metrics.values()]
    if any(m != definitions[0] for m in definitions):
        raise ExperimentConfigurationError("All four methods must share the same source RS metric.")
    base = None
    for label, role in protocol.method_roles.items():
        env, algorithm = experiment.source_environments[label], experiment.algorithms[label]
        if (not isinstance(env, DistributionalEnvironment) or env.source_mode != "data"
                or env.noise_std != .02 or not env.add_noise_to_data or env.noise_fn is not None
                or not env.allow_repeated_actions or env.data_row_selection != "first"
                or env.history.initial_data is not None or env.history.records
                or env.dtype != torch.float64 or env.device.type != "cpu"
                or callable(env.reference_distribution_provider) or callable(env.sampling_distribution_provider)):
            raise ExperimentConfigurationError("Source trials require fresh CPU float64 finite DistributionalEnvironments with fixed distributions and noise SD .02.")
        expected_cls = RoBOS if role == "robos" else DRBO
        if (not isinstance(algorithm, expected_cls) or algorithm.config.gp_backend != "fixed_gpy"
                or algorithm.config.gp_training_regularization != 1e-8 or algorithm.config.beta != 4):
            raise ExperimentConfigurationError("Source methods require fixed_gpy, training regularization 1e-8, beta=4.")
        if role == "robos" and (algorithm.config.simplex_solver != "source_cobyla"
                                or algorithm.config.aspiration != definitions[0]["aspiration"]):
            raise ExperimentConfigurationError("RoBOS requires matching aspiration and source_cobyla.")
        if role != "robos" and (algorithm.config.inner_solver != "source_cvxpy"
                                or algorithm.config.finite_action_tie_breaking != "first"):
            raise ExperimentConfigurationError("DRBO requires cold source_cvxpy and first exact ties.")
        if algorithm.config.fit_maxiter != 0:
            raise ExperimentConfigurationError("Released source GP hyperparameters are fixed, not fitted.")
        stored_variance = env.dataset.observation_noise_variance
        if stored_variance is not None and bool(torch.any(stored_variance != 0)):
            raise ExperimentConfigurationError("Source stored table must not already include observation noise variance.")
        factor = {"drbo_3epsilon":3., "drbo_epsilon":1., "drbo_epsilon_over_3":1/3, "robos":1.}[role]
        if callable(env.margin_provider) or not math.isclose(float(env.margin_provider),
                factor * definitions[0]["epsilon"], rel_tol=0., abs_tol=1e-14):
            raise ExperimentConfigurationError("Source DRBO margin must match its declared radius role.")
        identity = (env._dataset_fingerprint(), torch.as_tensor(env.reference_distribution_provider).tolist(),
                    torch.as_tensor(env.sampling_distribution_provider).tolist())
        if base is not None and identity != base:
            raise ExperimentConfigurationError("The four methods must share objective and context distributions.")
        base = identity


def _implementation_identity():
    root = Path(__file__).parent
    paths = ["_robos_protocol.py", "_robos_regret.py", "benchmarks.py",
             "algorithms/Distributional/DRBO.py", "algorithms/Distributional/RoBOS.py",
             "environments/Distributional.py"]
    return {name:hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}


def run_source_trials(experiment, *, trials, max_trials, max_rounds, max_evaluations,
                      metrics, plot, show, title, progress):
    """Execute native child experiments using immutable, durable trial plans."""
    from .experiment import Experiment, ExperimentCheckpointError, ExperimentConfigurationError, ExperimentComparisonResult
    from .environments.environment import EnvironmentData
    from .trials import ExperimentTrialsResult, statistics
    protocol = experiment._robos_protocol
    _validate(experiment, protocol, metrics)
    if max_rounds != protocol.online_rounds or max_evaluations not in (None, protocol.online_rounds + protocol.initial_points):
        raise ExperimentConfigurationError("Source run budget must match online_rounds (initialization is additional).")
    root, labels = experiment.output_directory, tuple(experiment.algorithms)
    metric = next(iter(metrics.values()))
    templates = experiment.algorithms
    env_templates = experiment.source_environments
    first = next(iter(env_templates.values()))
    actions = first.available_actions().actions.detach().cpu().numpy().reshape(-1)
    contexts = first.context_support.detach().cpu().numpy().reshape(-1)
    # Read the private table for metric audit, not through the objective-query
    # hook. Completed campaigns can therefore be opened with evaluation disabled.
    if (not np.array_equal(first.dataset.actions.cpu().numpy().reshape(-1), np.tile(actions, len(contexts)))
            or not np.array_equal(first.dataset.contexts.cpu().numpy().reshape(-1), np.repeat(contexts, len(actions)))):
        raise ExperimentConfigurationError("RoBOS source table must be a scalar context-major Cartesian grid.")
    truth = first.dataset.observations.detach().cpu().numpy().reshape(len(contexts), len(actions))
    reference = np.asarray(first.reference_distribution_provider, dtype=np.float64).reshape(-1)
    probability = np.asarray(first.sampling_distribution_provider, dtype=np.float64).reshape(-1)
    # These are source PMFs, already normalized in source NumPy arithmetic.
    expected = np.dot(probability.reshape(1, -1), truth).reshape(-1)
    manifest = dict(version=VERSION, source_revision=SOURCE, implementation=_implementation_identity(),
        protocol=asdict(protocol), trials=trials,
        metric=asdict(metric), dataset_fingerprint=first._dataset_fingerprint(),
        reference=reference.tolist(), sampling_distribution=probability.tolist(),
        algorithm_fingerprints={k:v.state_dict()["configuration_fingerprint"] for k,v in templates.items()},
        environment_fingerprints={k:v._source_fingerprint() for k,v in env_templates.items()},
        save_every=experiment.save_every, stored_noise_variance=Experiment._jsonable(first.dataset.observation_noise_variance))
    manifest = Experiment._jsonable(manifest)
    path = root / "source_protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ExperimentCheckpointError("RoBOS source scientific configuration changed; use a fresh output directory.")
        if not experiment.resume:
            raise ExperimentConfigurationError("Existing source trials require resume=True.")
    else:
        Experiment._atomic_json_save(manifest, path)
    experiment._trial_count = trials
    experiment._trial_documents, experiment._trial_results, experiment.trial_experiments = [], [], []
    experiment._last_requested_limits = dict(max_rounds=max_rounds, max_evaluations=max_evaluations,
                                             evaluation_protocol=manifest["protocol"])
    rng = np.random.RandomState(protocol.seed)
    limit = trials if max_trials is None else max_trials
    for trial in range(limit):
        trial_root = root / "trials" / f"trial-{trial+1:03d}"
        trial_root.mkdir(parents=True, exist_ok=True)
        plan_path = trial_root / "random-plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text()); _check_plan(plan)
            if plan["start_rng"] != _rng_state(rng) or plan["trial_index"] != trial:
                raise ExperimentCheckpointError("RoBOS trial random-stream chain is inconsistent.")
            rng = _restore_rng(plan["end_rng"])
        else:
            _require_runtime()
            plan = make_plan(protocol, rng, probability, action_count=len(actions), trial_index=trial)
            Experiment._atomic_json_save(plan, plan_path)
        completed_path = trial_root / "completed.json"
        if completed_path.exists():
            done = json.loads(completed_path.read_text())
            bundle_path = trial_root / "completed.pt"
            if (done["plan_sha256"] != plan["sha256"] or not bundle_path.is_file()
                    or hashlib.sha256(bundle_path.read_bytes()).hexdigest() != done["bundle_sha256"]):
                raise ExperimentCheckpointError("RoBOS completed-trial integrity check failed.")
            bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
            experiment._trial_documents.append(bundle["document"])
            experiment._trial_results.append(ExperimentComparisonResult(results=bundle["results"]))
            experiment.trial_experiments.append(SimpleNamespace(output_directory=trial_root,
                elapsed_seconds=bundle["document"]["elapsed_seconds"], completed_rounds=max_rounds))
            if progress:
                print(f"RoBOS source trial {trial+1}/{trials}: restored completed results (no new evaluations).", flush=True)
            continue
        runtime = _require_runtime()
        runtime_path = root / "numerical-runtime.json"
        if runtime_path.exists() and json.loads(runtime_path.read_text()) != runtime:
            raise ExperimentCheckpointError("Source numerical runtime differs from the campaign lock.")
        if not runtime_path.exists():
            Experiment._atomic_json_save(runtime, runtime_path)
        init_path = trial_root / "initialization.json"
        initial = json.loads(init_path.read_text()) if init_path.exists() else dict(plan_sha256=plan["sha256"], observations=[])
        if initial["plan_sha256"] != plan["sha256"]:
            raise ExperimentCheckpointError("Shared initialization does not match its random plan.")
        while len(initial["observations"]) < protocol.initial_points:
            j = plan["initial_indices"][len(initial["observations"])]
            initial["observations"].append(float(truth.reshape(-1)[j]))
            Experiment._atomic_json_save(initial, init_path)
        indices = np.asarray(plan["initial_indices"])
        if initial["observations"] != truth.reshape(-1)[indices].tolist():
            raise ExperimentCheckpointError("Shared initialization rewards do not match the source dataset.")
        envs = copy.deepcopy(dict(env_templates))
        for env in envs.values():
            env._source_exact_probabilities = True
            env.history.initial_data = env._validate_environment_data(EnvironmentData(
                actions=torch.as_tensor(actions[indices % len(actions), None]),
                contexts=torch.as_tensor(contexts[indices // len(actions), None]),
                observations=torch.tensor(initial["observations"], dtype=torch.float64),
                observation_noise_variance=torch.full((protocol.initial_points,), .0004, dtype=torch.float64),
                metadata={"generated_initialization": "robos_2023", "plan_sha256": plan["sha256"]}))
        child = Experiment(environment=envs, algorithm={k:v.clone_for_trial(0) for k,v in templates.items()},
            output_directory=trial_root, resume=experiment.resume, save_every=experiment.save_every,
            evaluation_protocol="robos_2023")
        if child.initialization_failures:
            raise ExperimentCheckpointError(f"Source child initialization failed: {child.initialization_failures}")
        results = {}
        for label, role in protocol.method_roles.items():
            item = child.experiments[label]
            if item._robos_plan_digest not in (None, plan["sha256"]):
                raise ExperimentCheckpointError("Child checkpoint random-plan identity mismatch.")
            item._robos_plan_digest = plan["sha256"]
            item._robos_execution = dict(role=role, plan=plan, actions=actions, expected=expected,
                                        progress=progress, label=f"trial {trial+1} | {label}")
            if progress:
                print(f"RoBOS source trial {trial+1}/{trials} | {label}: {item.completed_rounds}/{max_rounds}", flush=True)
            item._elapsed_report_started_at = time.monotonic()
            try:
                results[label] = item.run(max_rounds=max_rounds, plot=False, show=False, progress=progress, fail_fast=True)
            finally:
                item._reported_elapsed_seconds = item.elapsed_seconds
                item._elapsed_report_started_at = None
            child.warnings[label] = list(item.warnings.get(type(item.algorithm).__name__, []))
        oracle_path = trial_root / "source-oracle.json"
        oracle = json.loads(oracle_path.read_text()) if oracle_path.exists() else dict(plan_sha256=plan["sha256"], values=[], diagnostics=[])
        if oracle["plan_sha256"] != plan["sha256"]:
            raise ExperimentCheckpointError("Source oracle random-plan identity mismatch.")
        # Separate metric-only object: no true payoff is given to a learner.
        oracle_algorithm = templates[labels[-1]].clone_for_trial(0)
        oracle_algorithm.bind_environment_metadata({"context_support": torch.as_tensor(contexts[:, None])})
        gram = oracle_algorithm.context_gram_matrix()
        ref = torch.as_tensor(reference)
        for t in range(len(oracle["values"]), max_rounds):
            oracle_algorithm.set_source_solver_starts(plan["starts"][t]["oracle"])
            scores, diagnostics = [], []
            with child._capture_warnings("source metric oracle"):
                for a in range(len(actions)):
                    scores.append(oracle_algorithm.evaluate_fragility(truth[:,a], ref, gram)[0])
                    diagnostics.append(copy.deepcopy(oracle_algorithm.diagnostics().algorithm_specific["source_solver"]))
            kappa = float(min(scores))
            if not math.isfinite(kappa):
                raise RuntimeError("Source metric oracle returned a nonfinite value; campaign halted.")
            oracle["values"].append(kappa); oracle["diagnostics"].append(diagnostics)
            Experiment._atomic_json_save(oracle, oracle_path)
        for label in labels:
            item = child.experiments[label]
            item._robos_execution["source_kappa"] = oracle["values"]
            results[label] = attach_regret(item, results[label], label=label, metric=metric)
            # _result_document obtains algorithm.result and attaches saved report.
            item._regret_reports[type(item.algorithm).__name__] = item._regret_reports[label]
            child._regret_reports[label] = item._regret_reports[label]
        child.completed_rounds = max_rounds
        document = child._result_document()
        for label in labels:
            document["algorithms"][label]["result"] = Experiment._jsonable(results[label])
            document["algorithms"][label]["raw_observations"] = Experiment._jsonable(
                child.experiments[label].environment.history.records)
        document["source_accounting"] = dict(shared_initialization_queries=protocol.initial_points,
            initial_training_rows=protocol.initial_points * len(labels), online_queries=max_rounds * len(labels),
            plan_sha256=plan["sha256"], metric_table_entries=int(truth.size))
        Experiment._atomic_json_save(document, trial_root / "results.json")
        sha = Experiment._atomic_torch_save(dict(document=document, results=results), trial_root / "completed.pt")
        Experiment._atomic_json_save(dict(plan_sha256=plan["sha256"], bundle_sha256=sha), completed_path)
        experiment._trial_documents.append(document)
        experiment._trial_results.append(ExperimentComparisonResult(results=results))
        experiment.trial_experiments.append(child)
        _save_summary(experiment, protocol)
        if progress:
            print(f"RoBOS source trial {trial+1}/{trials}: all four methods and common RS oracle committed.", flush=True)
    results = {label:tuple(r.results[label] for r in experiment._trial_results) for label in labels}
    combined = ExperimentTrialsResult(tuple(experiment._trial_results),
        {label:statistics(experiment._trial_documents, label) for label in labels}, results, {},
        records=tuple(copy.deepcopy(experiment._trial_documents)), planned_trials=trials)
    experiment.trial_result = combined
    experiment.completed_rounds = max_rounds if len(combined.records) == trials else 0
    _save_summary(experiment, protocol)
    if plot is True or (plot is None and any(a.plot for a in templates.values())):
        from .plotting import plot as plot_trials
        figure = plot_trials(experiment, title=title, show=show)
        combined.figure, combined.axes = figure.figure, figure.axes
    return combined


def _save_summary(experiment, protocol):
    count = len(experiment._trial_documents)
    initial, online = count * protocol.initial_points, count * 4 * protocol.online_rounds
    experiment._atomic_json_save(dict(planned_trials=experiment._trial_count, completed_trials=count,
        completed_runs=4*count, online_queries=online, shared_initialization_queries=initial,
        initial_training_rows=4*initial, total_objective_queries=initial+online,
        elapsed_seconds=experiment.elapsed_seconds, protocol_version=VERSION),
        experiment.output_directory / "source_summary.json")
    experiment.save()
