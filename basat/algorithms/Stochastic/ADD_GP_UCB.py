"""Additive Gaussian-process UCB with one shared covariance model."""

from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import warnings
import weakref
from collections import OrderedDict
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Any, Literal

import gpytorch
import numpy as np
import torch

from basat.algorithms.algorithm import (
    Algorithm,
    AlgorithmConfigurationError,
    resolve_batch_configuration,
)
from basat.algorithms.helper import AlgorithmResult, SourceRecord, to_quality
from basat.models.Acquisitions import DiscreteAcquisitionCandidates
from basat.models.GaussianProcess import GaussianProcess, GaussianProcessConfig
from basat.models.utils import Bounds, normalize, unnormalize

from ._ax_source_random import validate_streams
from ._tree_numerics import random_state_dict, restore_random_state
from .Exception.exception_ADD_GP_UCB import (
    ADD_GP_UCBAcquisitionError,
    ADD_GP_UCBCheckpointError,
    ADD_GP_UCBDecompositionError,
    ADD_GP_UCBException,
    ADD_GP_UCBGroupError,
    ADD_GP_UCBKernelError,
    ADD_GP_UCBWarning,
)


def _install_dragonfly_kernel_cache() -> dict[str, Any]:
    """Memoize unchanged source components without changing arithmetic or RNG.

    Called only inside the isolated, pinned source process. Source hyperparameter
    samplers frequently change one coordinate while recomputing every additive
    component. Exact input/parameter byte keys avoid this redundant work; original
    left-to-right summation and uncached matrix ownership are preserved. Payload
    accounting deliberately charges shared data bytes to *every* component entry,
    so the stated memory limit is conservative (bounded Python metadata is extra).
    """
    import numpy as np
    from dragonfly.gp.kernel import AdditiveKernel, MaternKernel  # type: ignore[import-untyped]

    original = AdditiveKernel._child_evaluate
    original_scalar = MaternKernel._eval_kernel_values_unnormalised
    values: OrderedDict[Any, tuple[Any, int]] = OrderedDict()
    datasets: OrderedDict[Any, int] = OrderedDict()
    constants: OrderedDict[Any, Any] = OrderedDict()
    audited_error_policy = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}
    limits = {
        "component_entries": 1024,
        "component_payload_bytes": 32 * 1024 * 1024,
        "dataset_entries": 16,
        "dataset_payload_bytes": 4 * 1024 * 1024,
        "scalar_entries": 64,
        "matrix_elements": 2500,
    }
    statistics: dict[str, Any] = {
        "implementation": "exact-additive-component-lru-v1",
        "limits": limits,
        "numpy_error_policy": audited_error_policy,
        "hits": 0,
        "misses": 0,
        "evictions": 0,
        "bypassed": 0,
        "component_payload_bytes": 0,
        "dataset_payload_bytes": 0,
    }

    def fingerprint(value: Any) -> tuple[Any, ...]:
        array = np.asarray(value)
        return array.dtype.str, array.shape, array.tobytes()

    def scalar(kernel: Any, distance: Any) -> Any:
        if isinstance(distance, np.ndarray) or distance != 0 or np.geterr() != audited_error_policy:
            return original_scalar(kernel, distance)
        key = (kernel.p, fingerprint(kernel.hyperparams["nu"]), type(distance))
        if key in constants:
            constants.move_to_end(key)
            return constants[key]
        result = original_scalar(kernel, distance)
        if np.isfinite(result):
            constants[key] = result
            if len(constants) > limits["scalar_entries"]:
                constants.popitem(last=False)
        return result

    def cached(additive: Any, x1: Any, x2: Any) -> Any:
        if (
            len(x1) * len(x2) > limits["matrix_elements"]
            or np.geterr() != audited_error_policy
            or any(type(kernel) is not MaternKernel for kernel in additive.kernel_list)
        ):
            statistics["bypassed"] += 1
            return original(additive, x1, x2)
        # Keep the source's array conversion and floating-point summation order.
        x1, x2 = np.array(x1), np.array(x2)
        data_bytes = x1.nbytes + x2.nbytes
        if data_bytes > limits["dataset_payload_bytes"]:
            statistics["bypassed"] += 1
            return original(additive, x1, x2)
        candidate = (fingerprint(x1), fingerprint(x2))
        if candidate in datasets:
            datasets.move_to_end(candidate)
            # Reuse the canonical key's bytes instead of retaining new copies.
            data = next(reversed(datasets))
        else:
            data = candidate
            datasets[data] = data_bytes
            statistics["dataset_payload_bytes"] += data_bytes
            while (
                len(datasets) > limits["dataset_entries"]
                or statistics["dataset_payload_bytes"] > limits["dataset_payload_bytes"]
            ):
                _, evicted_bytes = datasets.popitem(last=False)
                statistics["dataset_payload_bytes"] -= evicted_bytes
        result = np.zeros((len(x1), len(x2)))
        for kernel, group in zip(additive.kernel_list, additive.groupings, strict=True):
            parameters = (
                fingerprint(kernel.norm_constant),
                fingerprint(kernel.hyperparams["nu"]),
                fingerprint(kernel.hyperparams["scale"]),
                fingerprint(kernel.hyperparams["dim_bandwidths"]),
            )
            key = (data, tuple(group), kernel.dim, kernel.p, *parameters)
            if key in values:
                statistics["hits"] += 1
                values.move_to_end(key)
                component, _ = values[key]
            else:
                component = kernel(x1[:, group], x2[:, group])
                statistics["misses"] += 1
                entry_bytes = (
                    data_bytes
                    + component.nbytes
                    + sum(len(parameter[-1]) for parameter in parameters)
                )
                # Do not cache invalid computations: repeated numerical warnings
                # must remain visible through the source-warning translator.
                if (
                    np.isfinite(component).all()
                    and entry_bytes <= limits["component_payload_bytes"]
                ):
                    values[key] = (component.copy(), entry_bytes)
                    statistics["component_payload_bytes"] += entry_bytes
                    while (
                        len(values) > limits["component_entries"]
                        or statistics["component_payload_bytes"] > limits["component_payload_bytes"]
                    ):
                        _, (_, evicted_bytes) = values.popitem(last=False)
                        statistics["component_payload_bytes"] -= evicted_bytes
                        statistics["evictions"] += 1
            # Cached component matrices never escape or become mutation targets.
            result += component
        return additive.hyperparams["scale"] * result

    AdditiveKernel._child_evaluate = cached
    MaternKernel._eval_kernel_values_unnormalised = scalar
    return statistics


def _dragonfly_source_worker() -> None:
    """Run pinned author code across an action/reward-only JSON boundary.

    This private entry point is intentionally a process, not a thread: the
    historical optimizer owns NumPy's global RNG. Compatibility aliases remain
    confined to that process and do not change the source algorithm.
    """
    import random
    import traceback
    from argparse import Namespace
    from importlib.metadata import version

    import numpy as np

    protocol = sys.stdout
    sys.stdout = sys.stderr

    def send(value: Mapping[str, Any]) -> None:
        protocol.write(json.dumps(value, allow_nan=False) + "\n")
        protocol.flush()

    def source_warning(
        message: Any, category: Any, filename: str, lineno: int, file: Any = None, line: Any = None
    ) -> None:
        del file, line
        send(
            {
                "kind": "warning",
                "message": str(message),
                "category": category.__name__,
                "location": f"{filename}:{lineno}",
            }
        )

    try:
        request = json.loads(sys.stdin.readline())
        if version("dragonfly-opt") != "0.1.4":
            raise RuntimeError("The source backend requires dragonfly-opt==0.1.4.")
        np.asscalar = lambda value: np.asarray(value).item()  # type: ignore[attr-defined]
        np.math = math  # type: ignore[attr-defined]
        for name, value in (("int", int), ("float", float), ("object", object), ("bool", bool)):
            if name not in np.__dict__:
                setattr(np, name, value)
        warnings.showwarning = source_warning
        # NumPy 2 no longer permits writing a length-one array into a scalar
        # ndarray slot. Restore the three historical coercions in the source
        # sampler, without changing the sampler, prior or random draws.
        import inspect
        import textwrap

        from dragonfly import maximise_function, minimise_function  # type: ignore[import-untyped]
        from dragonfly.gp import gp_core  # type: ignore[import-untyped]
        from dragonfly.utils import doo  # type: ignore[import-untyped]
        from dragonfly.utils.oper_utils import direct_ft_wrap  # type: ignore[import-untyped]

        sampling = inspect.getsource(gp_core.GPFitter._sample_cts_dscr_hps_for_post_sampling)
        old_assignment = "self.hps[self.curr_hp] = x\n"
        if sampling.count(old_assignment) != 2:
            raise RuntimeError(
                "Unexpected Dragonfly 0.1.4 source; scalar compatibility audit failed."
            )
        scope = dict(vars(gp_core))
        old_initial = "self.hps[i] = self.hp_priors[i].draw_samples('random', 1)"
        if sampling.count(old_initial) != 1:
            raise RuntimeError("Unexpected Dragonfly source hyperparameter initializer.")
        sampling = sampling.replace(
            old_initial,
            "self.hps[i] = np.asarray(self.hp_priors[i].draw_samples('random', 1)).item()",
        )
        old_seed = "int(np.squeeze(self.hp_sampler_dscr(_model, seed, 1), axis=1))"
        if sampling.count(old_seed) != 1:
            raise RuntimeError("Unexpected Dragonfly source grouping sampler.")
        sampling = sampling.replace(
            old_seed, "int(np.asarray(self.hp_sampler_dscr(_model, seed, 1)).item())"
        )
        exec(
            textwrap.dedent(
                sampling.replace(old_assignment, "self.hps[self.curr_hp] = np.asarray(x).item()\n")
            ),
            scope,
        )
        gp_core.GPFitter._sample_cts_dscr_hps_for_post_sampling = scope[
            "_sample_cts_dscr_hps_for_post_sampling"
        ]

        # PDOO asks scalar objectives that may return a length-one array.
        def scalar_objective(wrapper: Any, point: Any) -> float:
            if wrapper.vectorised:
                point = np.asarray(point).reshape((1, wrapper.domain_dim))
            return float(np.asarray(wrapper.func(point)).item())

        doo.DOOFunction.eval_single_point = scalar_objective
        cache_statistics = _install_dragonfly_kernel_cache()

        np.random.seed(request["seed"])
        random.seed(request["seed"])
        history = request["history"]
        callbacks = 0
        replay_started = time.monotonic()

        def objective(point: Any) -> float:
            nonlocal callbacks
            point = np.asarray(point, dtype=float)
            index = callbacks
            callbacks += 1
            if index < len(history):
                expected = history[index]
                if not np.allclose(point, expected["action"], rtol=1e-11, atol=1e-11):
                    raise RuntimeError(f"Source history replay diverged at callback {index + 1}.")
                if callbacks == len(history):
                    send(
                        {
                            "kind": "replay_complete",
                            "replayed_callbacks": callbacks,
                            "replay_seconds": time.monotonic() - replay_started,
                            "kernel_cache": cache_statistics,
                        }
                    )
                return float(expected["observation"])
            send(
                {
                    "kind": "proposal",
                    "action": point.tolist(),
                    "callback": callbacks,
                    "kernel_cache": cache_statistics,
                }
            )
            response = json.loads(sys.stdin.readline())
            if response.get("kind") != "observation":
                raise RuntimeError("Source worker requires one observation for its proposal.")
            return float(response["value"])

        send(
            {
                "kind": "ready",
                "version": "0.1.4",
                "native_direct": direct_ft_wrap is not None,
                "kernel_cache": cache_statistics,
            }
        )
        optimize = maximise_function if request["maximize"] else minimise_function
        result = optimize(
            objective,
            domain=request["bounds"],
            max_capital=request["max_capital"],
            options=Namespace(acq="add_ucb"),
            reporter="silent",
        )
        send(
            {
                "kind": "complete",
                "callbacks": callbacks,
                "history_rows": len(result[-1].query_vals),
                "kernel_cache": cache_statistics,
            }
        )
    except BaseException as error:
        send({"kind": "error", "message": str(error), "traceback": traceback.format_exc()})


def _close_source_process(process: subprocess.Popen[str], log: Any) -> None:
    """Reap the child even after interruption, reset, or garbage collection."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
    if process.stdin is not None:
        process.stdin.close()
    if process.stdout is not None:
        process.stdout.close()
    log.close()


def _read_source_events(stream: Any, events: queue.Queue[Any]) -> None:
    """Read bounded protocol messages without blocking the caller's timeout."""
    try:
        for line in stream:
            events.put(json.loads(line))
    except Exception as error:
        events.put({"kind": "error", "message": str(error)})
    finally:
        events.put({"kind": "eof"})


@dataclass(frozen=True, slots=True)
class AddGPUCBConfig:
    """Configuration of known or learned additive coordinate groups.

    Parameters
    ----------
    groups:
        Optional exact partition of ``D_env`` coordinate indices.
    maximum_group_size, learn_decomposition, decomposition_candidates, refresh_cycle:
        Candidate-partition limits and refresh policy when groups are learned.
    beta:
        Paper schedule, scalar, or callable confidence multiplier.
    initial_points:
        Optional initial-design size; ``None`` selects the documented default.
    initial_points_sampling_method:
        ``"paper"``, ``"sobol"``, or ``"uniform"`` initial-design sampling.
    component_optimizer:
        Optional callable optimizer at the additive-group boundary.
    batch_size:
        Synchronous size; version 1 requires one.
    seed:
        Algorithm-owned random seed.
    implementation:
        ``"basat"`` uses the configurable native implementation.
        ``"basat-icml-2015"`` uses the paper-first 2015 protocol implemented
        natively in physical coordinates, with shared isotropic SE kernels,
        periodic fitting and componentwise original DIRECT. Exact known groups
        may omit irrelevant coordinates; those coordinates use bounds midpoints.
        ``"basat-alebo-2020"`` uses Basat's internal, source-equivalent
        implementation of the ALEBO paper's additive comparator: local NumPy
        RNG, native GP fitting, slice sampling and component-UCB optimization.
        It requires no Dragonfly package, subprocess or external algorithm.
        ``"dragonfly-0.1.4"``
        is an opt-in, pinned author-code backend used by the ALEBO paper's
        Add-GP-UCB comparator. It uses Dragonfly's own initialization, kernel
        fitting, additive decompositions and component acquisition optimization.
        Other native tuning fields must remain at their defaults in this mode.
    source_max_capital:
        Required historical evaluation capital for either fixed comparator
        implementation. This is not Basat's run-round limit: the published
        value 49 produces 50 observed evaluations, including initialization.
    source_timeout_seconds:
        Maximum wait for a historical subprocess proposal, including replay.
        It does not limit the in-process native comparator's GP computation.
        ``close()`` releases the isolated source process without losing history.
    acquisition_evaluations : int
        Nominal total DIRECT evaluations per proposal in the 2015 protocol.
        Each component receives ``ceil(total / number_of_groups)``. A callback
        cap stops DIRECT at that limit and retains its best evaluated point;
        actual calls and budget stops are logged.
    common_mean, noise_variance, kernel_scale_bounds:
        Raw-reward GP calibration for the 2015 protocol. Noise is a fixed
        diagonal variance, not objective observation noise. ``None`` derives
        noise/scale bounds from the initial observations, never extra queries.
    bandwidth_bounds, initial_bandwidth, initial_bandwidth_steps:
        Physical-coordinate bandwidth bounds and the initial fixed-bandwidth
        phase, counted after the initial design (25 proposals by default).
    hyperparameter_iterations : int
        DIRECT iterations per likelihood search, not objective evaluations.
    """

    groups: tuple[tuple[int, ...], ...] | None = None
    maximum_group_size: int = 5
    learn_decomposition: bool = False
    decomposition_candidates: int | None = None
    refresh_cycle: int = 25
    beta: Callable[[int, int], float] | float | Literal["paper"] = "paper"
    initial_points: int | None = None
    initial_points_sampling_method: Literal["paper", "sobol", "uniform"] = "paper"
    component_optimizer: Any | None = None
    batch_size: int | None = 1
    q_batch: int | None = None
    seed: int = 1
    implementation: Literal["basat", "basat-icml-2015", "basat-alebo-2020", "dragonfly-0.1.4"] = "basat"
    source_max_capital: int | None = None
    source_timeout_seconds: float = 3600.0
    acquisition_evaluations: int = 900
    common_mean: float = 0.0
    noise_variance: float | None = None
    kernel_scale_bounds: tuple[float, float] | None = None
    bandwidth_bounds: tuple[float, float] = (1e-5, 5.0)
    initial_bandwidth: float = 1e-5
    initial_bandwidth_steps: int = 25
    hyperparameter_iterations: int = 20


@dataclass(slots=True)
class AddGPUCBState:
    """Mutable additive partition, shared data, and decomposition diagnostics.

    Attributes
    ----------
    groups:
        Current exact coordinate partition.
    X_train, y_train:
        Shared ``(N, D_env)`` GP inputs and ``(N,)`` responses.
    component_kernels, additive_gp:
        Fitted component kernels and the single shared additive GP.
    last_decomposition_refresh, candidate_decomposition_scores:
        Refresh generation and most recent candidate scores.
    last_ask_action:
        Pending complete ``D_env`` action, if any.
    """

    groups: tuple[tuple[int, ...], ...]
    X_train: torch.Tensor
    y_train: torch.Tensor
    component_kernels: tuple[gpytorch.kernels.ScaleKernel, ...]
    additive_gp: GaussianProcess
    last_decomposition_refresh: int
    candidate_decomposition_scores: list[float]
    last_ask_action: torch.Tensor | None


class ADD_GP_UCB(Algorithm):
    """Optimize group UCBs derived from a single additive GP covariance.

    Examples
    --------
    >>> import basat
    >>> bounds = basat.Bounds(lower=[-1.0] * 4, upper=[1.0] * 4)
    >>> algorithm = basat.ADD_GP_UCB(
    ...     bounds=bounds,
    ...     config=basat.AddGPUCBConfig(groups=((0, 1), (2, 3)), seed=7),
    ... )
    """

    ACQUISITION_POLICY = "additive GP-UCB"
    CHECKPOINT_ADDED_CONFIG_DEFAULTS = MappingProxyType(
        {
            **Algorithm.CHECKPOINT_ADDED_CONFIG_DEFAULTS,
            "implementation": "basat",
            "source_max_capital": None,
            "source_timeout_seconds": 3600.0,
            "acquisition_evaluations": 900,
            "common_mean": 0.0,
            "noise_variance": None,
            "kernel_scale_bounds": None,
            "bandwidth_bounds": (1e-5, 5.0),
            "initial_bandwidth": 1e-5,
            "initial_bandwidth_steps": 25,
            "hyperparameter_iterations": 20,
        }
    )

    EXCEPTION_CLASS = ADD_GP_UCBException
    native_batching = False
    native_environments = frozenset({"stochastic"})
    SOURCE = SourceRecord(
        paper_title="High Dimensional Bayesian Optimisation and Bandits via Additive Models",
        paper_year=2015,
        paper_url="https://proceedings.mlr.press/v37/kandasamy15.html",
        repository_url="https://github.com/kirthevasank/add-gp-bandits",
        repository_status="author",
        notes="One shared additive covariance; finite rows are scored without synthesis.",
    )

    def __init__(
        self,
        *,
        bounds: Bounds,
        config: AddGPUCBConfig,
        maximize: bool = True,
        dtype: torch.dtype | str = torch.float64,
        device: torch.device | str = "cpu",
        plot: bool = False,
        verbose: bool = False,
        verbose_setting: bool = False,
        verbose_iter_by_iter: bool = False,
    ) -> None:
        """Initialize a shared additive GP and complete coordinate partition.

        Parameters
        ----------
        bounds:
            Physical action bounds in ``D_env`` coordinates.
        config:
            Group partition, decomposition-learning, beta, and optimizer settings.
        maximize:
            Optimize larger observations when ``True``.
        dtype, device:
            Tensor precision and execution device.
        plot, verbose, verbose_setting, verbose_iter_by_iter:
            Presentation flags inherited from :class:`Algorithm`.

        Raises
        ------
        AlgorithmConfigurationError
            If sequential, beta, or decomposition settings are invalid.
        ADD_GP_UCBGroupError
            If explicit groups do not partition ``D_env`` exactly once.
        """

        self.config = config
        self._validate_config(config, bounds.dimension)
        super().__init__(
            bounds=bounds,
            batch_size=config.batch_size,
            q_batch=config.q_batch,
            maximize=maximize,
            dtype=dtype,
            device=device,
            seed=config.seed,
            initial_points_sampling_method=config.initial_points_sampling_method,
            plot=plot,
            verbose=verbose,
            verbose_setting=verbose_setting,
            verbose_iter_by_iter=verbose_iter_by_iter,
        )
        groups = self._resolve_initial_groups()
        component_kernels, gp = self._build_additive_kernel(groups)
        self.state = AddGPUCBState(
            groups=groups,
            X_train=torch.empty((0, self.action_dimension), dtype=self.dtype, device=self.device),
            y_train=torch.empty((0,), dtype=self.dtype, device=self.device),
            component_kernels=component_kernels,
            additive_gp=gp,
            last_decomposition_refresh=0,
            candidate_decomposition_scores=[],
            last_ask_action=None,
        )
        self.model_input_dimension = self.action_dimension
        self._last_component_values: tuple[float, ...] = ()
        self._source_process: subprocess.Popen[str] | None = None
        self._source_events: queue.Queue[Any] = queue.Queue()
        self._source_finalizer: Any = None
        self._source_pending = False
        self._source_provenance: dict[str, Any] = {}
        self._native_additive_policy: Any = None
        self._native_random_journal: list[dict[str, Any]] = []
        self._native_numpy_snapshot: dict[str, Any] | None = random_state_dict(np.random.RandomState(config.seed))
        self._native_fit_rows_snapshot: tuple[int, ...] = ()
        self._additive_2015_policy: Any = None
        self._posterior_factor_cache: Any = None
        if config.implementation != "basat" and self.dtype != torch.float64:
            raise AlgorithmConfigurationError(
                "Source-derived additive protocols require dtype='float64' to preserve proposals."
            )
        if config.implementation == "basat-icml-2015":
            if self.device.type != "cpu":
                raise AlgorithmConfigurationError("The 2015 numerical protocol requires device='cpu'.")
            self.state.groups = self._paper_2015().groups

    @staticmethod
    def _validate_config(config: AddGPUCBConfig, D: int) -> None:
        """Validate immutable additive-GP settings against dimension ``D``."""

        if not isinstance(config, AddGPUCBConfig):
            raise AlgorithmConfigurationError("config must be AddGPUCBConfig.")
        if config.implementation not in {"basat", "basat-icml-2015", "basat-alebo-2020", "dragonfly-0.1.4"}:
            raise AlgorithmConfigurationError("Unknown Add-GP-UCB implementation.")
        if not math.isfinite(config.source_timeout_seconds) or config.source_timeout_seconds <= 0:
            raise AlgorithmConfigurationError("source_timeout_seconds must be positive and finite.")
        if config.implementation in {"basat-alebo-2020", "dragonfly-0.1.4"}:
            if (
                isinstance(config.source_max_capital, bool)
                or not isinstance(config.source_max_capital, int)
                or config.source_max_capital < 1
            ):
                raise AlgorithmConfigurationError(
                    "Source mode requires positive source_max_capital."
                )
            defaults = AddGPUCBConfig()
            native_fields = (
                "groups",
                "maximum_group_size",
                "learn_decomposition",
                "decomposition_candidates",
                "refresh_cycle",
                "beta",
                "initial_points",
                "initial_points_sampling_method",
                "component_optimizer",
                "batch_size",
                "q_batch",
            )
            for name in native_fields:
                if getattr(config, name) != getattr(defaults, name):
                    raise AlgorithmConfigurationError(
                        f"{name} is a native Basat setting; source mode requires its default."
                    )
        _, q_batch = resolve_batch_configuration(
            batch_size=config.batch_size, q_batch=config.q_batch
        )
        if q_batch is not None and q_batch > 1:
            raise AlgorithmConfigurationError(
                "ADD-GP-UCB does not define a joint q-acquisition; use batch_size for "
                "repeated independent additive-UCB queries."
            )
        for name in ("maximum_group_size", "refresh_cycle"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise AlgorithmConfigurationError(f"{name} must be positive.")
        if config.decomposition_candidates is not None and (
            isinstance(config.decomposition_candidates, bool)
            or not isinstance(config.decomposition_candidates, int)
            or config.decomposition_candidates < 1
        ):
            raise AlgorithmConfigurationError("decomposition_candidates must be positive or None.")
        if config.initial_points is not None and (
            isinstance(config.initial_points, bool)
            or not isinstance(config.initial_points, int)
            or config.initial_points < 1
        ):
            raise AlgorithmConfigurationError("initial_points must be positive or None.")
        if config.groups is not None:
            ADD_GP_UCB._validate_groups_static(
                config.groups, D, allow_subset=config.implementation == "basat-icml-2015"
            )
        if config.implementation == "basat-icml-2015":
            if config.beta != "paper" or config.component_optimizer is not None:
                raise AlgorithmConfigurationError(
                    "The 2015 protocol requires its dimension-dependent UCB and original DIRECT."
                )
            if config.initial_points_sampling_method not in {"paper", "uniform"}:
                raise AlgorithmConfigurationError("The 2015 initial design must be uniform.")
            if config.batch_size not in (None, 1) or config.q_batch is not None:
                raise AlgorithmConfigurationError("The 2015 protocol is sequential (batch_size=1).")
        else:
            defaults = AddGPUCBConfig()
            for name in ("acquisition_evaluations", "common_mean", "noise_variance",
                         "kernel_scale_bounds", "bandwidth_bounds", "initial_bandwidth",
                         "initial_bandwidth_steps", "hyperparameter_iterations"):
                if getattr(config, name) != getattr(defaults, name):
                    raise AlgorithmConfigurationError(
                        f"{name} requires implementation='basat-icml-2015'."
                    )
        if isinstance(config.beta, str) and config.beta != "paper":
            raise AlgorithmConfigurationError("beta string must be 'paper'.")
        if (
            not isinstance(config.beta, str)
            and not callable(config.beta)
            and (not math.isfinite(float(config.beta)) or float(config.beta) < 0)
        ):
            raise AlgorithmConfigurationError("beta must be finite and non-negative.")

    @staticmethod
    def _validate_groups_static(
        groups: tuple[tuple[int, ...], ...], D: int, *, allow_subset: bool = False
    ) -> None:
        """Require non-empty integer groups that partition ``range(D)`` exactly."""

        if not groups or any(not group for group in groups):
            raise ADD_GP_UCBGroupError("Additive groups must be non-empty.")
        flattened = [coordinate for group in groups for coordinate in group]
        if any(isinstance(index, bool) or not isinstance(index, int) for index in flattened):
            raise ADD_GP_UCBGroupError("Group indices must be integers.")
        if allow_subset:
            if len(set(flattened)) != len(flattened) or any(i < 0 or i >= D for i in flattened):
                raise ADD_GP_UCBGroupError("Groups must be disjoint valid coordinate subsets.")
        elif sorted(flattened) != list(range(D)):
            raise ADD_GP_UCBGroupError("Groups must partition every coordinate exactly once.")

    def _validate_groups(self, groups: tuple[tuple[int, ...], ...]) -> None:
        """Validate a complete disjoint coordinate partition."""

        self._validate_groups_static(
            groups, self.action_dimension,
            allow_subset=self.config.implementation == "basat-icml-2015",
        )

    def _random_partition(self) -> tuple[tuple[int, ...], ...]:
        """Draw a seeded partition with groups no larger than the configured cap."""

        permutation = torch.randperm(
            self.action_dimension, generator=self.generator, device=self.device
        ).tolist()
        size = self.config.maximum_group_size
        return tuple(
            tuple(sorted(permutation[start : start + size]))
            for start in range(0, self.action_dimension, size)
        )

    def _resolve_initial_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return validated explicit groups or one seeded random partition."""

        groups = self.config.groups
        if groups is None:
            if self.config.implementation == "basat-icml-2015":
                size = self.config.maximum_group_size
                groups = tuple(tuple(range(i, min(i + size, self.action_dimension)))
                               for i in range(0, self.action_dimension, size))
            else:
                groups = self._random_partition()
        self._validate_groups(groups)
        return tuple(tuple(group) for group in groups)

    def _build_additive_kernel(
        self, groups: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[gpytorch.kernels.ScaleKernel, ...], GaussianProcess]:
        """Build component kernels and their single shared additive GP."""

        components = tuple(
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=len(group), active_dims=torch.tensor(group, dtype=torch.long)
                )
            ).to(dtype=self.dtype, device=self.device)
            for group in groups
        )
        additive = gpytorch.kernels.AdditiveKernel(*components)
        gp = GaussianProcess(
            GaussianProcessConfig(
                kernel=additive,
                mean="zero",
                noise_mode="learned",
                fit_method="mle",
                dtype=self.dtype,
                device=self.device,
            ),
            input_dimension=self.action_dimension,
        )
        return components, gp

    def _initialize_from_environment_impl(self, data: Any) -> None:
        """Normalize evaluated historical actions and load visible observations."""

        evaluated = data.actions if data.evaluated_actions is None else data.evaluated_actions
        actions = torch.as_tensor(evaluated, dtype=self.dtype, device=self.device)
        if actions.shape[0] == 0:
            return
        if self.config.implementation != "basat":
            raise AlgorithmConfigurationError(
                "Source mode accepts only source-generated history through load_state_dict, "
                "not arbitrary previous_data."
            )
        self.state.X_train = (
            normalize(actions, self.bounds.to(dtype=self.dtype, device=self.device))
            .detach()
            .clone()
        )
        self.state.y_train = (
            torch.as_tensor(data.observations, dtype=self.dtype, device=self.device)
            .detach()
            .clone()
        )
        self._set_gp_data()

    def _prepare_discrete_candidates_impl(
        self,
        *,
        candidate_actions: torch.Tensor,
        candidate_dataset_indices: torch.Tensor,
        round_info: Mapping[str, Any],
    ) -> DiscreteAcquisitionCandidates:
        """Normalize finite environment actions while preserving row identity."""

        del round_info
        return DiscreteAcquisitionCandidates(
            environment_actions=candidate_actions.detach().clone(),
            model_inputs=normalize(
                candidate_actions, self.bounds.to(dtype=self.dtype, device=self.device)
            ),
            dataset_indices=candidate_dataset_indices.detach().clone(),
        )

    def _set_gp_data(self) -> None:
        """Synchronize non-empty normalized history with the shared additive GP."""

        self._posterior_factor_cache = None
        if self.state.y_train.numel():
            self.state.additive_gp.set_data(
                self.state.X_train,
                to_quality(self.state.y_train, maximize=self.maximize),
            )
            model = self.state.additive_gp.model
            if model is not None:
                self.state.component_kernels = tuple(model.covar_module.kernels)

    def _initialize_fit_scales(self) -> None:
        """Seed MLE at scales compatible with the observed response magnitude.

        Use the transformed training targets used by the shared GP, rather than
        raw targets with an unrelated offset/scale. Every component still
        receives full MLE fitting; the initial total variance is split equally.

        Raises
        ------
        ADD_GP_UCBKernelError
            If a model is unexpectedly absent or the empirical scale cannot be
            represented as a finite value.
        """

        gp = self.state.additive_gp
        model = gp.model
        if model is None:
            raise ADD_GP_UCBKernelError(
                "Shared additive GP was not constructed from the observations.",
                algorithm=self,
            )
        quality = model.train_targets.detach()
        signal_scale = quality.square().mean().clamp_min(1e-3)
        if not torch.isfinite(signal_scale):
            raise ADD_GP_UCBKernelError(
                "Shared additive GP observation scale is not finite.", algorithm=self
            )
        component_scale = signal_scale / len(self.state.groups)
        for kernel in model.covar_module.kernels:
            kernel.outputscale = component_scale

        likelihood = gp.likelihood
        likelihood_noise = getattr(likelihood, "noise", None)
        if likelihood is not None and likelihood_noise is not None:
            # GaussianLikelihood's default lower bound is strictly above 1e-4.
            likelihood.noise = (0.01 * signal_scale).clamp_min(1e-3)

    def _fit_shared_gp(self) -> None:
        """Fit the single additive covariance model and refresh component handles.

        Raises
        ------
        ADD_GP_UCBKernelError
            If shared-GP fitting or component recovery fails.
        """

        self._set_gp_data()
        try:
            self._initialize_fit_scales()
            self.state.additive_gp.fit(maxiter=75)
            model = self.state.additive_gp.model
            if model is not None:
                self.state.component_kernels = tuple(model.covar_module.kernels)
        except Exception as error:
            raise ADD_GP_UCBKernelError(
                f"Shared additive GP fitting failed: {error}", algorithm=self
            ) from error

    def _component_covariance(
        self, group_id: int, U: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate one fitted component on group-coordinate matrices."""

        kernel = self.state.component_kernels[group_id]
        base = kernel.base_kernel
        lengthscale = base.lengthscale.detach().reshape(1, 1, -1)
        difference = (U.unsqueeze(-2) - V.unsqueeze(-3)) / lengthscale
        return kernel.outputscale.detach() * torch.exp(-0.5 * difference.square().sum(dim=-1))

    def component_posterior(
        self, group_id: int, U: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a component posterior using the full covariance inverse.

        Parameters
        ----------
        group_id:
            Index of the additive coordinate group.
        U:
            ``(N, |group|)`` candidate coordinates in normalized model units,
            or physical coordinates for ``basat-icml-2015``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Component posterior mean and variance, each shaped ``(N,)`` in
            reward-quality units. The general native standardized model shares
            its constant outcome offset equally between component means.

        Raises
        ------
        ADD_GP_UCBGroupError
            If ``group_id`` does not identify a configured group.
        ADD_GP_UCBKernelError
            If no observations or fitted component kernel are available.
        """

        if self.config.implementation == "basat-icml-2015":
            points = torch.as_tensor(U, dtype=self.dtype, device=self.device)
            mean, variance = self._paper_2015().component_posterior(
                group_id, points.detach().cpu().numpy()
            )
            return (torch.as_tensor(mean, dtype=self.dtype, device=self.device),
                    torch.as_tensor(variance, dtype=self.dtype, device=self.device))
        if self.config.implementation != "basat":
            raise ADD_GP_UCBKernelError(
                "Comparator component models are managed by their own fitted policy; "
                "the configurable-native component_posterior method does not apply.",
                algorithm=self,
            )
        if not 0 <= group_id < len(self.state.groups):
            raise ADD_GP_UCBGroupError(
                "group_id is outside the additive partition.", algorithm=self
            )
        if self.state.y_train.numel() == 0:
            raise ADD_GP_UCBKernelError(
                "Component posterior requires observations.", algorithm=self
            )
        group = self.state.groups[group_id]
        U = torch.as_tensor(U, dtype=self.dtype, device=self.device)
        if U.ndim == 1:
            U = U.unsqueeze(0)
        if U.ndim != 2 or U.shape[1] != len(group):
            raise ADD_GP_UCBGroupError(
                f"Component query must have shape (N, {len(group)}).", algorithm=self
            )
        X = self.state.X_train
        model = self.state.additive_gp.model
        likelihood = self.state.additive_gp.likelihood
        if model is None or likelihood is None:
            raise ADD_GP_UCBKernelError("Component posterior requires a shared GP.", algorithm=self)
        noise = torch.as_tensor(
            likelihood.noise, dtype=self.dtype, device=self.device
        ).detach().reshape(-1)
        transform = getattr(model, "outcome_transform", None)
        scale = (torch.ones((), dtype=self.dtype, device=self.device) if transform is None
                 else transform.stdvs.detach().reshape(()))
        offset = (torch.zeros((), dtype=self.dtype, device=self.device) if transform is None
                  else transform.means.detach().reshape(()))
        kernel_values = tuple(
            (tuple(k.base_kernel.lengthscale.detach().reshape(-1).tolist()),
             float(k.outputscale.detach())) for k in self.state.component_kernels
        )
        key = (id(model), X.data_ptr(), X._version, self.state.y_train.data_ptr(),
               self.state.y_train._version, tuple(X.shape), self.state.groups,
               kernel_values, tuple(noise.tolist()), float(scale), float(offset))
        cache = self._posterior_factor_cache
        if cache is None or cache[0] != key:
            full_covariance = torch.zeros(
                (X.shape[0], X.shape[0]), dtype=self.dtype, device=self.device
            )
            for index, coordinates in enumerate(self.state.groups):
                full_covariance += self._component_covariance(
                    index, X[:, coordinates], X[:, coordinates]
                )
            delta = full_covariance + torch.diag(noise.expand(X.shape[0]))
            from linear_operator.utils.cholesky import (  # type: ignore[import-untyped]
                psd_safe_cholesky,
            )

            from basat.warnings import translate_backend_warnings
            with translate_backend_warnings(operation="ADD_GP_UCB shared posterior factorization"):
                cholesky = psd_safe_cholesky(delta)
            quality = to_quality(self.state.y_train, maximize=self.maximize)
            centered = (quality - offset) / scale
            alpha = torch.cholesky_solve(centered[:, None], cholesky)
            self._posterior_factor_cache = (key, cholesky, alpha)
        else:
            _, cholesky, alpha = cache
        cross = self._component_covariance(group_id, U, X[:, group])
        mean = (cross @ alpha).squeeze(-1) * scale + offset / len(self.state.groups)
        solved = torch.cholesky_solve(cross.T, cholesky)
        prior_variance = self.state.component_kernels[group_id].outputscale.detach()
        variance = (prior_variance - (cross * solved.T).sum(-1)).clamp_min(0) * scale.square()
        return mean, variance

    def _beta(self) -> float:
        """Resolve and validate the current non-negative exploration coefficient."""

        beta = self.config.beta
        if callable(beta):
            value = beta(self.iteration + 1, self.action_dimension)
        elif beta == "paper":
            maximum_group = max(len(group) for group in self.state.groups)
            value = 2.0 * math.log(
                max(2.0, (self.iteration + 2) ** (maximum_group + 2) * math.pi**2 / 0.6)
            )
        else:
            value = float(beta)
        if not math.isfinite(float(value)) or value < 0:
            raise ADD_GP_UCBAcquisitionError("beta returned an invalid value.", algorithm=self)
        return float(value)

    def _group_ucb(self, group_id: int, U: torch.Tensor) -> torch.Tensor:
        """Evaluate one component UCB on normalized group coordinates ``U``."""

        mean, variance = self.component_posterior(group_id, U)
        return mean + math.sqrt(self._beta()) * variance.sqrt()

    def _sum_group_ucb(self, X: torch.Tensor) -> torch.Tensor:
        """Sum component UCB values for full normalized action rows."""

        values = torch.zeros(X.shape[0], dtype=self.dtype, device=self.device)
        for group_id, group in enumerate(self.state.groups):
            values += self._group_ucb(group_id, X[:, group])
        return values

    def _optimize_component(
        self, group_id: int, group: tuple[int, ...]
    ) -> tuple[torch.Tensor, float]:
        """Optimize one component UCB with the configured exact boundary."""

        optimizer = self.config.component_optimizer
        if optimizer is not None:
            try:
                raw = optimizer(
                    group_id=group_id,
                    bounds=Bounds(
                        torch.zeros(len(group), dtype=self.dtype, device=self.device),
                        torch.ones(len(group), dtype=self.dtype, device=self.device),
                    ),
                    acquisition=lambda query: self._group_ucb(
                        group_id,
                        torch.as_tensor(query, dtype=self.dtype, device=self.device).reshape(
                            -1, len(group)
                        ),
                    ),
                    generator=self.generator,
                )
            except Exception as error:
                raise ADD_GP_UCBAcquisitionError(
                    f"The component optimizer failed for group {group_id}: {error}",
                    algorithm=self,
                ) from error
            point = torch.as_tensor(raw, dtype=self.dtype, device=self.device).reshape(-1)
            if point.shape != (len(group),) or not torch.isfinite(point).all():
                raise ADD_GP_UCBAcquisitionError(
                    "component_optimizer must return one finite point in group coordinates.",
                    algorithm=self,
                )
            if not torch.all((point >= 0) & (point <= 1)):
                raise ADD_GP_UCBAcquisitionError(
                    "component_optimizer returned a point outside [0, 1].",
                    algorithm=self,
                )
            value = float(self._group_ucb(group_id, point.reshape(1, -1)).item())
            return point, value

        try:
            from scipy.optimize import direct
        except ImportError as error:
            raise ADD_GP_UCBAcquisitionError(
                "SciPy DIRECT is required by Add-GP-UCB.", algorithm=self
            ) from error

        def objective(values_array: Any) -> float:
            """Return one component's negated UCB for SciPy DIRECT."""

            query = torch.as_tensor(values_array, dtype=self.dtype, device=self.device).reshape(
                1, -1
            )
            return -float(self._group_ucb(group_id, query).item())

        result = direct(objective, [(0.0, 1.0)] * len(group), maxfun=256)
        point = torch.as_tensor(result.x, dtype=self.dtype, device=self.device)
        return point, -float(result.fun)

    def _refresh_decomposition_if_required(self) -> None:
        """Score seeded partition candidates at the configured refresh cycle.

        Candidate partitions share a fixed-hyperparameter marginal-likelihood
        comparison. The winning partition receives a fresh additive GP.

        Raises
        ------
        ADD_GP_UCBDecompositionError
            If every candidate covariance is numerically invalid.
        """

        if not self.config.learn_decomposition or self.state.y_train.numel() < 2:
            return
        if self.iteration - self.state.last_decomposition_refresh < self.config.refresh_cycle:
            return
        count = self.config.decomposition_candidates or self.action_dimension
        candidates = [self.state.groups] + [self._random_partition() for _ in range(count)]
        scores: list[float] = []
        for groups in candidates:
            # Fixed-hyperparameter additive marginal likelihood provides a
            # deterministic decomposition comparison before the winning model
            # receives full hyperparameter fitting.
            covariance = torch.zeros(
                (self.state.X_train.shape[0], self.state.X_train.shape[0]),
                dtype=self.dtype,
                device=self.device,
            )
            for group in groups:
                difference = self.state.X_train[:, None, group] - self.state.X_train[None, :, group]
                covariance += torch.exp(-0.5 * difference.square().sum(dim=-1))
            covariance += 1e-4 * torch.eye(
                covariance.shape[0], dtype=self.dtype, device=self.device
            )
            try:
                cholesky = torch.linalg.cholesky(covariance)
                quality = to_quality(self.state.y_train, maximize=self.maximize)
                alpha = torch.cholesky_solve(quality[:, None], cholesky)
                score = -0.5 * float((quality[:, None].T @ alpha).item())
                score -= float(torch.log(torch.diagonal(cholesky)).sum().item())
            except RuntimeError:
                score = float("-inf")
            scores.append(score)
        if not any(math.isfinite(score) for score in scores):
            raise ADD_GP_UCBDecompositionError(
                "Every candidate additive decomposition failed.", algorithm=self
            )
        selected = max(range(len(scores)), key=scores.__getitem__)
        groups = candidates[selected]
        components, gp = self._build_additive_kernel(groups)
        self.state.groups = groups
        self.state.component_kernels = components
        self.state.additive_gp = gp
        self.state.candidate_decomposition_scores = scores
        self.state.last_decomposition_refresh = self.iteration
        self._set_gp_data()

    def _initial_design_size(self) -> int:
        """Return the configured or ambient-dimension initial-design size."""

        return self.config.initial_points or (
            10 if self.config.implementation == "basat-icml-2015"
            else max(2 * self.action_dimension, 4)
        )

    def _new_paper_2015(self) -> Any:
        """Construct a benchmark-independent, independently checkpointed policy."""
        from ._additive_2015 import Additive2015Policy
        config = self.config
        return Additive2015Policy(
            lower=self.bounds.lower.detach().cpu().tolist(),
            upper=self.bounds.upper.detach().cpu().tolist(),
            groups=config.groups, maximum_group_size=config.maximum_group_size,
            learn_decomposition=config.learn_decomposition,
            initial_points=self._initial_design_size(), seed=self.seed,
            refresh_cycle=config.refresh_cycle,
            decomposition_candidates=config.decomposition_candidates or 5,
            acquisition_evaluations=config.acquisition_evaluations,
            common_mean=config.common_mean if self.maximize else -config.common_mean,
            noise_variance=config.noise_variance,
            kernel_scale_bounds=config.kernel_scale_bounds,
            bandwidth_bounds=config.bandwidth_bounds,
            initial_bandwidth=config.initial_bandwidth,
            initial_bandwidth_steps=config.initial_bandwidth_steps,
            hyperparameter_iterations=config.hyperparameter_iterations,
        )

    def _paper_2015(self) -> Any:
        """Return the native 2015 policy without replaying or refitting history."""
        if self._additive_2015_policy is None:
            self._additive_2015_policy = self._new_paper_2015()
        return self._additive_2015_policy

    def _ask_paper_2015(self, round_info: Mapping[str, Any]) -> torch.Tensor:
        if "candidate_actions" in round_info:
            raise AlgorithmConfigurationError("The 2015 protocol requires continuous bounds.")
        if (self.initial_points_sampling_method not in {"paper", "uniform"}
                or self.acquisition_sampling_method != "paper"):
            raise AlgorithmConfigurationError("The 2015 protocol uses uniform initialization and DIRECT.")
        policy = self._paper_2015()
        action = torch.as_tensor(policy.propose(), dtype=self.dtype, device=self.device).reshape(1, -1)
        self.state.groups = tuple(tuple(g) for g in policy.groups)
        self.state.last_ask_action = action.detach().clone()
        self._record_last_ask(
            actions=action, model_inputs=action, round_info=round_info,
            algorithm_metadata={"implementation": "basat-icml-2015", "groups": self.state.groups},
        )
        return action

    def close(self) -> None:
        """Release the optional source process; saved action/reward history remains.

        A subsequent ``ask`` deterministically replays the history without
        evaluating the benchmark. Use this after a source-backed run, especially
        when stopping earlier than ``source_max_capital``.
        """
        if self._source_finalizer is not None and self._source_finalizer.alive:
            self._source_finalizer()
        self._source_process = None
        self._source_pending = False
        self._capture_native_progress()
        self._native_additive_policy = None

    def _native_additive(self) -> Any:
        """Rebuild Basat's policy from its own observations after a resume.

        Checkpoints store progress, not a surrogate dataset or fitted GP.
        Replay validates every proposal without querying an environment.
        """
        if self._native_additive_policy is not None:
            return self._native_additive_policy
        from ._additive_policy import AdditivePolicy

        policy = AdditivePolicy(
            torch.stack((self.bounds.lower, self.bounds.upper), -1).cpu().tolist(),
            seed=self.seed,
            max_capital=self.config.source_max_capital,
        )
        physical = unnormalize(
            self.state.X_train, self.bounds.to(dtype=self.dtype, device=self.device)
        )
        quality = to_quality(self.state.y_train, maximize=self.maximize)
        for index, (expected, observation) in enumerate(zip(physical, quality, strict=True)):
            self._apply_native_random_journal(policy, index, "before_ask")
            proposed = torch.as_tensor(policy.propose(), dtype=self.dtype, device=self.device)
            if not torch.allclose(expected, proposed, rtol=1e-11, atol=1e-11):
                raise ADD_GP_UCBCheckpointError(
                    "Native additive history replay diverged; the saved progress was not changed.",
                    algorithm=self,
                )
            self._apply_native_random_journal(policy, index, "before_tell")
            policy.observe(float(observation.item()))
        self._apply_native_random_journal(policy, len(quality), "before_ask")
        if self.state.last_ask_action is not None:
            proposed = torch.as_tensor(policy.propose(), dtype=self.dtype, device=self.device)
            if not torch.allclose(
                proposed.reshape_as(self.state.last_ask_action), self.state.last_ask_action,
                rtol=1e-11, atol=1e-11,
            ):
                raise ADD_GP_UCBCheckpointError("Native additive pending replay diverged.", algorithm=self)
            self._apply_native_random_journal(policy, len(quality), "before_tell")
        self._native_additive_policy = policy
        return policy

    def _apply_native_random_journal(self, policy: Any, row: int, phase: str) -> None:
        """Replay only recorded stream operations, never benchmark evaluations."""
        for event in self._native_random_journal:
            if event["row"] != row or event["phase"] != phase:
                continue
            if event["kind"] == "import":
                restore_random_state(policy.rng, event["state"])
            else:
                policy.rng.randn(event["count"])

    def _capture_native_progress(self) -> None:
        """Snapshot lightweight source diagnostics without constructing a GP."""
        if self._native_additive_policy is not None:
            self._native_numpy_snapshot = random_state_dict(self._native_additive_policy.rng)
            self._native_fit_rows_snapshot = tuple(self._native_additive_policy.build_rows)

    @property
    def _supports_ax017_evaluation(self) -> bool:
        return self.config.implementation == "basat-alebo-2020"

    def _export_random_streams(self) -> dict[str, Any]:
        """Export current source streams without lazy history replay or fitting."""
        if self.config.implementation != "basat-alebo-2020":
            raise NotImplementedError("Ax stream exchange requires the native ALEBO comparator.")
        self._capture_native_progress()
        if self._native_numpy_snapshot is None:
            raise ValueError("Legacy additive checkpoint has no RNG snapshot; resume a proposal before exporting streams.")
        return {"numpy": deepcopy(self._native_numpy_snapshot), "torch": self.generator.get_state().clone()}

    def _import_random_streams(self, state: Mapping[str, Any]) -> None:
        """Record an atomic stream handoff for deterministic partial-run replay."""
        if self.config.implementation != "basat-alebo-2020":
            raise NotImplementedError("Ax stream exchange requires the native ALEBO comparator.")
        numpy_rng, generator, initializer = validate_streams(state)
        if initializer is not None:
            raise ValueError("The additive comparator has no entropy-backed Ax initializer.")
        snapshot = random_state_dict(numpy_rng)
        self._native_random_journal.append({
            "row": len(self.state.y_train),
            "phase": "before_tell" if self.state.last_ask_action is not None else "before_ask",
            "kind": "import", "state": snapshot,
        })
        self._native_numpy_snapshot = deepcopy(snapshot)
        self.generator = generator
        if self._native_additive_policy is not None:
            self._native_additive_policy.rng = numpy_rng

    def _consume_source_evaluation_draws(self, count: int) -> None:
        """Record Ax's discarded normal draws before source feedback/refitting."""
        if self.config.implementation != "basat-alebo-2020":
            raise NotImplementedError("Ax evaluation draws require the native ALEBO comparator.")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Source evaluation-draw count must be a nonnegative integer.")
        if self.state.last_ask_action is None:
            raise ValueError("Source evaluation draws require a pending proposal.")
        self._capture_native_progress()
        if self._native_numpy_snapshot is None:
            self._native_additive()
            self._capture_native_progress()
        assert self._native_numpy_snapshot is not None
        rng = np.random.RandomState(0)
        restore_random_state(rng, self._native_numpy_snapshot)
        rng.randn(count)
        self._native_numpy_snapshot = random_state_dict(rng)
        if self._native_additive_policy is not None:
            self._native_additive_policy.rng = rng
        self._native_random_journal.append({
            "row": len(self.state.y_train), "phase": "before_tell", "kind": "draw", "count": count,
        })

    def _reset_shared_random_streams(self, state: Mapping[str, Any]) -> None:
        validate_streams(state)
        self.reset()
        self._import_random_streams(state)

    def _ask_native_additive(self, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Ask Basat's internal component-UCB policy for a physical action."""
        if "candidate_actions" in round_info:
            raise AlgorithmConfigurationError(
                "The ALEBO additive comparator requires continuous bounds."
            )
        if (
            self.initial_points_sampling_method != "paper"
            or self.acquisition_sampling_method != "paper"
        ):
            raise AlgorithmConfigurationError(
                "The ALEBO additive comparator requires its paper sampling policy."
            )
        action = torch.as_tensor(
            self._native_additive().propose(), dtype=self.dtype, device=self.device
        ).reshape(1, -1)
        self.state.last_ask_action = action.detach().clone()
        self._capture_native_progress()
        self._record_last_ask(
            actions=action,
            model_inputs=normalize(action, self.bounds.to(dtype=self.dtype, device=self.device)),
            round_info=round_info,
            algorithm_metadata={"implementation": "basat-alebo-2020"},
        )
        return action

    def _start_source(self) -> None:
        """Start an isolated, version-pinned author optimizer and replay history."""
        from importlib.metadata import PackageNotFoundError, version

        try:
            installed = version("dragonfly-opt")
        except PackageNotFoundError as error:
            raise AlgorithmConfigurationError(
                "Install dragonfly-opt==0.1.4 to use this optional source backend."
            ) from error
        if installed != "0.1.4":
            raise AlgorithmConfigurationError("Source backend requires dragonfly-opt==0.1.4.")
        self.close()
        environment = os.environ.copy()
        # Seed isolation is also numerical-thread isolation; the released
        # comparator sets these two variables to 1 before importing Dragonfly.
        environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        log = tempfile.TemporaryFile(mode="w+t")  # noqa: SIM115 -- lifetime owned by finalizer
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                "from basat.algorithms.Stochastic.ADD_GP_UCB import _dragonfly_source_worker; "
                "_dragonfly_source_worker()",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
            env=environment,
        )
        self._source_process = process
        self._source_finalizer = weakref.finalize(self, _close_source_process, process, log)
        self._source_events = queue.Queue()
        threading.Thread(
            target=_read_source_events, args=(process.stdout, self._source_events), daemon=True
        ).start()
        physical = unnormalize(
            self.state.X_train, self.bounds.to(dtype=self.dtype, device=self.device)
        )
        self._send_source(
            {
                "seed": self.config.seed,
                "maximize": self.maximize,
                "bounds": torch.stack((self.bounds.lower, self.bounds.upper), -1).cpu().tolist(),
                "max_capital": self.config.source_max_capital,
                "history": [
                    {"action": x, "observation": y}
                    for x, y in zip(
                        physical.cpu().tolist(), self.state.y_train.cpu().tolist(), strict=True
                    )
                ],
            }
        )

    def _send_source(self, message: Mapping[str, Any]) -> None:
        """Send one explicit history or observation message to the source process."""
        process = self._source_process
        if process is None or process.stdin is None:
            raise ADD_GP_UCBAcquisitionError("Source process is not available.", algorithm=self)
        try:
            process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            self.close()
            raise ADD_GP_UCBAcquisitionError(
                "Source process closed unexpectedly.", algorithm=self
            ) from error

    def _next_source_action(self) -> torch.Tensor:
        """Receive a proposal or explicit source error under a bounded timeout."""
        if self._source_process is None:
            self._start_source()
        deadline = time.monotonic() + self.config.source_timeout_seconds
        while True:
            try:
                event = self._source_events.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty as error:
                self.close()
                raise ADD_GP_UCBAcquisitionError(
                    "Dragonfly source proposal/replay exceeded source_timeout_seconds; "
                    "completed observations remain resumable.",
                    algorithm=self,
                ) from error
            kind = event.get("kind")
            if kind == "warning":
                warnings.warn(
                    f"Basat ADD_GP_UCB source backend ({event['category']} at "
                    f"{event['location']}): {event['message']}",
                    ADD_GP_UCBWarning,
                    stacklevel=3,
                )
            elif kind in {"ready", "replay_complete"}:
                self._source_provenance.update(event)
            elif kind == "proposal":
                self._source_provenance.update(
                    {key: value for key, value in event.items() if key != "action"}
                )
                self._source_pending = True
                return torch.tensor(event["action"], dtype=self.dtype, device=self.device).reshape(
                    1, -1
                )
            else:
                self._source_provenance.update(event)
                self.close()
                if kind == "complete":
                    raise ADD_GP_UCBAcquisitionError(
                        f"Dragonfly source budget completed after {event['callbacks']} callbacks.",
                        algorithm=self,
                    )
                raise ADD_GP_UCBAcquisitionError(
                    f"Dragonfly source backend failed: {event.get('message', 'unexpected EOF')}",
                    algorithm=self,
                )

    def _ask_source(self, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Select a continuous source proposal with no benchmark access."""
        if "candidate_actions" in round_info:
            raise AlgorithmConfigurationError("Dragonfly source mode requires continuous bounds.")
        if (
            self.initial_points_sampling_method != "paper"
            or self.acquisition_sampling_method != "paper"
        ):
            raise AlgorithmConfigurationError(
                "Source mode requires its own 'paper' sampling policies."
            )
        action = self._next_source_action()
        self.state.last_ask_action = action.detach().clone()
        self._record_last_ask(
            actions=action,
            model_inputs=normalize(action, self.bounds.to(dtype=self.dtype, device=self.device)),
            round_info=round_info,
            algorithm_metadata={"implementation": "dragonfly-0.1.4"},
        )
        return action

    def _ask_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Select an initial, finite, or componentwise continuous action."""

        if self.config.implementation == "dragonfly-0.1.4":
            return self._ask_source(round_info)
        if self.config.implementation == "basat-alebo-2020":
            return self._ask_native_additive(round_info)
        if self.config.implementation == "basat-icml-2015":
            return self._ask_paper_2015(round_info)

        finite = "candidate_actions" in round_info
        table = None
        if finite:
            table = self.prepare_discrete_candidates(
                candidate_actions=round_info["candidate_actions"],
                candidate_dataset_indices=round_info["candidate_dataset_indices"],
                round_info=round_info,
            )
        if self.state.y_train.numel() < self._initial_design_size():
            if table is None:
                unit = self._sample_design(
                    Bounds(
                        torch.zeros(self.action_dimension, dtype=self.dtype, device=self.device),
                        torch.ones(self.action_dimension, dtype=self.dtype, device=self.device),
                    ),
                    1,
                    phase="initial",
                    seed=self.config.seed + self.iteration,
                )
                action = unnormalize(unit, self.bounds.to(dtype=self.dtype, device=self.device))
            else:
                position = int(torch.argmin(table.dataset_indices).item())
                action = table.environment_actions[position : position + 1]
                unit = table.model_inputs[position : position + 1]
            component_values: tuple[float, ...] = ()
            value = None
        else:
            self._refresh_decomposition_if_required()
            self._fit_shared_gp()
            if table is not None:
                scores = self._sum_group_ucb(table.model_inputs)
                maximum = scores.max()
                tied = torch.nonzero(scores == maximum, as_tuple=False).flatten()
                position = int(tied[torch.argmin(table.dataset_indices[tied])].item())
                unit = table.model_inputs[position : position + 1]
                action = table.environment_actions[position : position + 1]
                value = float(scores[position].item())
                component_values = tuple(
                    float(self._group_ucb(index, unit[:, group]).item())
                    for index, group in enumerate(self.state.groups)
                )
            else:
                if self.acquisition_sampling_method != "paper":
                    sampled = self._sample_design(
                        Bounds(
                            torch.zeros(
                                self.action_dimension, dtype=self.dtype, device=self.device
                            ),
                            torch.ones(self.action_dimension, dtype=self.dtype, device=self.device),
                        ),
                        256,
                        phase="acquisition",
                        seed=self.config.seed + self.iteration,
                    )
                    sampled_scores = self._sum_group_ucb(sampled)
                    selected = torch.argsort(sampled_scores, descending=True, stable=True)[
                        : self.request_size
                    ]
                    unit = sampled[selected]
                    action = unnormalize(unit, self.bounds.to(dtype=self.dtype, device=self.device))
                    component_values = tuple(
                        float(self._group_ucb(index, unit[:, group]).sum().item())
                        for index, group in enumerate(self.state.groups)
                    )
                    value = float(sampled_scores[selected].sum().item())
                else:
                    unit = torch.empty(
                        (1, self.action_dimension), dtype=self.dtype, device=self.device
                    )
                    values: list[float] = []
                    for group_id, group in enumerate(self.state.groups):
                        group_point, group_value = self._optimize_component(group_id, group)
                        unit[0, list(group)] = group_point
                        values.append(group_value)
                    action = unnormalize(unit, self.bounds.to(dtype=self.dtype, device=self.device))
                    component_values = tuple(values)
                    value = sum(values)
            self._last_acquisition_value = value
        self._last_component_values = component_values
        self.state.last_ask_action = action.detach().clone()
        self._record_last_ask(
            actions=action,
            model_inputs=unit,
            round_info=round_info,
            algorithm_metadata={"groups": self.state.groups, "component_ucb": component_values},
        )
        return action

    def _tell_impl(
        self, *, action: torch.Tensor, feedback: Mapping[str, Any], round_info: Mapping[str, Any]
    ) -> None:
        """Append evaluated normalized actions and observations to shared history."""

        del round_info
        source = self.config.implementation == "dragonfly-0.1.4"
        native_comparator = self.config.implementation == "basat-alebo-2020"
        paper_2015 = self.config.implementation == "basat-icml-2015"
        if paper_2015:
            if not torch.equal(action, feedback["evaluated_action"]):
                raise ADD_GP_UCBCheckpointError(
                    "The evaluated 2015 proposal was modified; use a float64 environment "
                    "without an action-transforming adapter.", algorithm=self)
            self._paper_2015().observe(
                action.detach().cpu().numpy().reshape(-1),
                float(to_quality(feedback["observation"], maximize=self.maximize).item()),
            )
        if native_comparator:
            policy = self._native_additive()
            replayed = torch.as_tensor(
                policy.propose(), dtype=self.dtype, device=self.device
            ).reshape(1, -1)
            if not torch.allclose(
                action, feedback["evaluated_action"], rtol=1e-11, atol=1e-11
            ) or not torch.allclose(action, replayed, rtol=1e-11, atol=1e-11):
                raise ADD_GP_UCBCheckpointError(
                    "Native additive feedback must match its pending proposal.", algorithm=self
                )
            quality = float(to_quality(feedback["observation"], maximize=self.maximize).item())
            policy.observe(quality)
            self._capture_native_progress()
        if source:
            if not torch.allclose(action, feedback["evaluated_action"], rtol=1e-11, atol=1e-11):
                raise AlgorithmConfigurationError(
                    "Source feedback must evaluate the proposed action unchanged."
                )
            # A checkpoint may have been taken between ask and tell. Replay to
            # the pending action without issuing another environment evaluation.
            if not self._source_pending:
                replayed = self._next_source_action()
                if not torch.allclose(action, replayed, rtol=1e-11, atol=1e-11):
                    self.close()
                    raise ADD_GP_UCBCheckpointError(
                        "Pending source action replay diverged.", algorithm=self
                    )
            self._send_source(
                {"kind": "observation", "value": float(feedback["observation"].item())}
            )
            self._source_pending = False
        evaluated_unit = (feedback["evaluated_action"] if paper_2015 else normalize(
            feedback["evaluated_action"], self.bounds.to(dtype=self.dtype, device=self.device)
        ))
        self.state.X_train = torch.cat((self.state.X_train, evaluated_unit), dim=0)
        self.state.y_train = torch.cat((self.state.y_train, feedback["observation"]), dim=0)
        self.state.last_ask_action = None
        if not source and not native_comparator and not paper_2015:
            self._set_gp_data()

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Return group, decomposition, component-UCB, and training diagnostics."""

        if self.config.implementation == "basat-icml-2015":
            return {"implementation": "basat-icml-2015", "algorithm_owner": "basat",
                    "training_rows": int(self.state.y_train.numel()),
                    "external_algorithm_runner": False, **self._paper_2015().diagnostics()}
        if self.config.implementation == "basat-alebo-2020":
            policy = self._native_additive_policy
            return {
                "implementation": "basat-alebo-2020",
                "algorithm_owner": "basat",
                "external_algorithm_runner": False,
                "training_rows": int(self.state.y_train.numel()),
                "source_max_capital": self.config.source_max_capital,
                "native_fit_rows": self._native_fit_rows_snapshot if policy is None else tuple(policy.build_rows),
                "source_evaluation_draws": sum(
                    event["count"] for event in self._native_random_journal if event["kind"] == "draw"
                ),
                "source_readback_draws": 0,
                "native_policy_revision": "basat-native-alebo-additive-v1",
            }
        if self.config.implementation == "dragonfly-0.1.4":
            return {
                "implementation": self.config.implementation,
                "source_backend": dict(self._source_provenance),
                "source_max_capital": self.config.source_max_capital,
                "training_rows": int(self.state.y_train.numel()),
            }

        return {
            "implementation": self.config.implementation,
            "source_backend": dict(self._source_provenance),
            "groups": self.state.groups,
            "group_sizes": tuple(len(group) for group in self.state.groups),
            "training_rows": int(self.state.y_train.numel()),
            "last_component_ucb": self._last_component_values,
            "decomposition_scores": tuple(self.state.candidate_decomposition_scores),
            "last_decomposition_refresh": self.state.last_decomposition_refresh,
        }

    def result(self) -> AlgorithmResult:
        """Return ordinary results with the selected implementation's source record.

        The class-level native ``SOURCE`` remains immutable. Source-backed
        instances qualify only their returned result, never another instance's
        source metadata.
        """
        result = super().result()
        if self.config.implementation == "basat-icml-2015":
            return replace(result, source=replace(
                self.SOURCE,
                repository_url=("https://github.com/kirthevasank/add-gp-bandits/tree/"
                                "d248c26b9c1c6d74bfa6aed23b8f8c41b086ac14"),
                notes=("Independent Basat-native paper-first 2015 additive GP implementation; "
                       "25 fixed-bandwidth BO proposals, followed by periodic fitting. "
                       "Released-code discrepancies are explicit, not silently copied."),
            ))
        if self.config.implementation == "basat-alebo-2020":
            return replace(
                result,
                source=SourceRecord(
                    paper_title=self.SOURCE.paper_title,
                    paper_year=2015,
                    paper_url=self.SOURCE.paper_url,
                    repository_url="https://github.com/dragonfly/dragonfly",
                    repository_status="author",
                    notes="Basat-native source-derived numerical implementation of the ALEBO 2020 additive comparator; no external algorithm runner.",
                ),
            )
        if self.config.implementation == "dragonfly-0.1.4":
            return replace(
                result,
                source=SourceRecord(
                    paper_title=self.SOURCE.paper_title,
                    paper_year=2015,
                    paper_url=self.SOURCE.paper_url,
                    repository_url="https://github.com/dragonfly/dragonfly",
                    repository_status="author",
                    notes="Pinned dragonfly-opt 0.1.4 action/reward backend; ALEBO comparator protocol.",
                ),
            )
        return result

    def _config_repr(self) -> str:
        """Return a stable representation including callable configuration fields."""

        values = asdict(self.config)
        values["beta"] = repr(self.config.beta)
        values["component_optimizer"] = repr(self.config.component_optimizer)
        return repr(values)

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize partition, normalized history, and diagnostic progress."""

        self._capture_native_progress()
        return {
            "config": self._config_repr(),
            "groups": self.state.groups,
            "X_train": self.state.X_train,
            "y_train": self.state.y_train,
            "last_decomposition_refresh": self.state.last_decomposition_refresh,
            "candidate_decomposition_scores": tuple(self.state.candidate_decomposition_scores),
            "last_ask_action": self.state.last_ask_action,
            "last_component_values": self._last_component_values,
            "source_backend": dict(self._source_provenance),
            "native_random_journal": deepcopy(self._native_random_journal),
            "native_numpy_snapshot": deepcopy(self._native_numpy_snapshot),
            "native_fit_rows_snapshot": self._native_fit_rows_snapshot,
            "native_policy_revision": (
                "basat-native-alebo-additive-v1"
                if self.config.implementation == "basat-alebo-2020"
                else None
            ),
            "paper_2015_policy": (
                self._paper_2015().state_dict()
                if self.config.implementation == "basat-icml-2015" else None
            ),
        }

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Validate additive checkpoint fields, partition, and strict config."""

        required = {
            "config",
            "groups",
            "X_train",
            "y_train",
            "last_decomposition_refresh",
            "candidate_decomposition_scores",
            "last_ask_action",
            "last_component_values",
        }
        if missing := required - set(state):
            raise ADD_GP_UCBCheckpointError(
                f"Add-GP-UCB state is missing {sorted(missing)}.", algorithm=self
            )
        if (
            self.config.implementation == "basat-alebo-2020"
            and state.get("native_policy_revision") != "basat-native-alebo-additive-v1"
        ):
            raise ADD_GP_UCBCheckpointError(
                "Native additive policy revision does not match the checkpoint.", algorithm=self
            )
        if self.config.implementation == "basat-alebo-2020":
            try:
                rows = len(torch.as_tensor(state["y_train"]))
                journal = state.get("native_random_journal", [])
                if not isinstance(journal, list):
                    raise ValueError("Random journal must be a list.")
                previous = (-1, -1)
                for event in journal:
                    if not isinstance(event, Mapping):
                        raise ValueError("Invalid random-journal event.")
                    row, phase, kind = event.get("row"), event.get("phase"), event.get("kind")
                    if (isinstance(row, bool) or not isinstance(row, int) or not 0 <= row <= rows
                            or phase not in {"before_ask", "before_tell"}
                            or kind not in {"import", "draw"}):
                        raise ValueError("Invalid random-journal position or operation.")
                    position = (row, int(phase == "before_tell"))
                    if position < previous:
                        raise ValueError("Random-journal events are out of order.")
                    previous = position
                    if phase == "before_tell" and row == rows and state["last_ask_action"] is None:
                        raise ValueError("Pending random-journal event has no proposal.")
                    if kind == "import":
                        restore_random_state(np.random.RandomState(0), event["state"])
                    else:
                        count = event["count"]
                        if (isinstance(count, bool) or not isinstance(count, int) or count < 0
                                or phase != "before_tell"):
                            raise ValueError("Invalid discarded-normal event.")
                snapshot = state.get("native_numpy_snapshot")
                if snapshot is not None:
                    restore_random_state(np.random.RandomState(0), snapshot)
                if journal and snapshot is None:
                    raise ValueError("Random-journal checkpoint has no RNG snapshot.")
                fit_rows = state.get("native_fit_rows_snapshot", ())
                if any(isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= rows for n in fit_rows):
                    raise ValueError("Invalid source fitting-row diagnostics.")
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                raise ADD_GP_UCBCheckpointError(f"Invalid source RNG journal: {error}", algorithm=self) from error
        self._validate_groups(tuple(tuple(group) for group in state["groups"]))
        if self.config.implementation == "basat-icml-2015":
            saved = state.get("paper_2015_policy")
            if not isinstance(saved, Mapping):
                raise ADD_GP_UCBCheckpointError("Missing native 2015 policy state.", algorithm=self)
            try:
                candidate = self._new_paper_2015()
                candidate.load_state_dict(saved)
            except (ValueError, TypeError, KeyError) as error:
                raise ADD_GP_UCBCheckpointError(
                    f"Invalid native 2015 policy state: {error}", algorithm=self
                ) from error
            if tuple(tuple(g) for g in candidate.groups) != tuple(tuple(g) for g in state["groups"]):
                raise ADD_GP_UCBCheckpointError("Stored 2015 groups disagree.", algorithm=self)
            stored_x = torch.as_tensor(state["X_train"], dtype=self.dtype, device=self.device)
            stored_y = to_quality(torch.as_tensor(state["y_train"], dtype=self.dtype,
                                                  device=self.device), maximize=self.maximize)
            if (not torch.equal(stored_x, torch.as_tensor(candidate.X, dtype=self.dtype,
                                                          device=self.device))
                    or not torch.equal(stored_y, torch.as_tensor(candidate.y, dtype=self.dtype,
                                                               device=self.device))):
                raise ADD_GP_UCBCheckpointError("Stored 2015 policy and learner histories disagree.",
                                                algorithm=self)
            pending = state["last_ask_action"]
            if (pending is None) != (candidate.pending is None):
                raise ADD_GP_UCBCheckpointError("Stored 2015 pending proposals disagree.", algorithm=self)
            if pending is not None and not torch.equal(
                torch.as_tensor(pending, dtype=self.dtype, device=self.device).reshape(-1),
                torch.as_tensor(candidate.pending, dtype=self.dtype, device=self.device),
            ):
                raise ADD_GP_UCBCheckpointError("Stored 2015 pending proposals disagree.", algorithm=self)
        del strict

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Restore partition and history with a fresh unfitted shared GP."""

        del strict
        self.close()
        self._posterior_factor_cache = None
        self._additive_2015_policy = None
        groups = tuple(tuple(group) for group in state["groups"])
        components, gp = self._build_additive_kernel(groups)
        self.state.groups = groups
        self.state.component_kernels = components
        self.state.additive_gp = gp
        self.state.X_train = (
            torch.as_tensor(state["X_train"], dtype=self.dtype, device=self.device).detach().clone()
        )
        self.state.y_train = (
            torch.as_tensor(state["y_train"], dtype=self.dtype, device=self.device).detach().clone()
        )
        self.state.last_decomposition_refresh = int(state["last_decomposition_refresh"])
        self.state.candidate_decomposition_scores = [
            float(value) for value in state["candidate_decomposition_scores"]
        ]
        value = state["last_ask_action"]
        self.state.last_ask_action = (
            None
            if value is None
            else torch.as_tensor(value, dtype=self.dtype, device=self.device).detach().clone()
        )
        self._last_component_values = tuple(
            float(value) for value in state["last_component_values"]
        )
        self._source_provenance = dict(state.get("source_backend", {}))
        self._native_random_journal = deepcopy(state.get("native_random_journal", []))
        self._native_numpy_snapshot = deepcopy(state.get("native_numpy_snapshot"))
        self._native_fit_rows_snapshot = tuple(state.get("native_fit_rows_snapshot", ()))
        if self.config.implementation == "basat-icml-2015":
            self._paper_2015().load_state_dict(state["paper_2015_policy"])

    def _reset_impl(self) -> None:
        """Clear observations and rebuild the seeded initial additive model."""

        self.close()
        self._posterior_factor_cache = None
        self._additive_2015_policy = None
        self._source_provenance = {}
        self._native_random_journal = []
        self._native_numpy_snapshot = random_state_dict(np.random.RandomState(self.seed))
        self._native_fit_rows_snapshot = ()
        groups = self._resolve_initial_groups()
        components, gp = self._build_additive_kernel(groups)
        self.state = AddGPUCBState(
            groups=groups,
            X_train=torch.empty((0, self.action_dimension), dtype=self.dtype, device=self.device),
            y_train=torch.empty((0,), dtype=self.dtype, device=self.device),
            component_kernels=components,
            additive_gp=gp,
            last_decomposition_refresh=0,
            candidate_decomposition_scores=[],
            last_ask_action=None,
        )
        self._last_component_values = ()
        if self.config.implementation == "basat-icml-2015":
            self.state.groups = self._paper_2015().groups


__all__ = ["ADD_GP_UCB", "AddGPUCBConfig", "AddGPUCBState"]
