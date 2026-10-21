"""AG-VS following the AISTATS 2027 anonymous reference implementation.

Observed responses govern scores, inactive-coordinate completion and restart
patience. Fitted posterior records are kept separately for recommendation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Literal

import numpy as np
import torch

from basat.models.Acquisitions import AcquisitionConfig, DiscreteAcquisitionCandidates
from basat.models.utils import Bounds

from ..algorithm import Algorithm, resolve_batch_configuration
from ..helper import SourceRecord, to_quality
from ._agvs_source import SourceBOGP, classical_ei, latin_hypercube, propose_bo
from ._agvs_turbo import (
    SourceTurboGP,
    TurboProposalState,
    propose_turbo,
    update_turbo,
)
from .Exception.exception_AG_VS import (
    AG_VSException,
    AGVSAnchorError,
    AGVSCheckpointError,
    AGVSDataCompatibilityError,
    AGVSNodeError,
)
from .TuRBO import TuRBOConfig


@dataclass(frozen=True, slots=True)
class AG_VSConfig:
    """Reference AG-VS settings, with Basat's independent/joint batch controls.

    Supply ``effective_dimension < split_groups <= D``. An omitted
    ``initial_points`` resolves to ``max(2 * split_groups, D + 1)``; an omitted
    ``stagnation_patience`` resolves to ``3 * split_groups``. Set
    ``stagnation_enabled=False`` to disable restarts. ``paper`` initialization
    is one complete Latin hypercube served through successive asks.

    The reference BO uses EI with xi=1e-4 on 1024 scrambled Sobol candidates.
    ``gp_maxiter=None`` preserves sklearn's optimizer budget; for TuRBO it
    resolves to 50 training steps. Legacy continuous-acquisition budget fields
    remain accepted for configuration compatibility but do not affect this
    finite-candidate reference acquisition. Posterior stopping is disabled in
    the supplied source, so active stopping configurations are rejected.
    """

    effective_dimension: int
    split_groups: int | None = None
    sequence_length: int = 2
    initial_points: int | None = None
    initial_points_sampling_method: Literal["paper", "sobol", "uniform"] = "paper"
    acquisition: AcquisitionConfig = field(
        default_factory=lambda: AcquisitionConfig(name="EI", xi=1e-4, mc_samples=256)
    )
    inner_optimizer: Literal["bo", "turbo"] = "bo"
    inner_turbo_config: TuRBOConfig | None = None
    noise_variance: float = 1e-4
    learn_noise: bool = False
    abs_mode: bool = True
    stagnation_patience: int | None = None
    stagnation_enabled: bool = True
    merge_enabled: bool = True
    turbo_max_evals: int = 50
    turbo_min_cuda: int = 1024
    stopping_threshold: float | None = None
    stopping_delta: float | None = None
    gp_fit_method: Literal["mle", "map"] = "mle"
    gp_maxiter: int | None = None
    acquisition_restarts: int = 10
    acquisition_raw_samples: int = 1024
    batch_size: int | None = 3
    q_batch: int | None = None
    seed: int = 1


@dataclass(slots=True)
class AGVSNode:
    """Coordinate group with separate training and historical score indices."""

    node_id: int
    depth: int
    parent_id: int | None
    child_ids: list[int]
    active_indices: torch.Tensor
    anchor: torch.Tensor | None
    training_rows: list[int]
    local_X: torch.Tensor
    local_y: torch.Tensor
    model: Any
    score_history_rows: list[int]
    posterior_mean_history: list[float]
    posterior_std_history: list[float]
    previous_benchmark_history: list[float]
    score: float
    score_mass: int
    pre_split_score: float | None
    pre_split_mass: int
    excluded: bool
    merge_disabled: bool
    cohort_evaluated: bool
    newly_retained: bool
    archived: bool
    merge_completed: bool
    retained_merged: bool = False
    fitted_rows: tuple[int, ...] | None = None
    proposal_optimizer: Any = None
    warm_model_state: Any = None


@dataclass(slots=True)
class AGVSState:
    """Tree, immutable observations and the frozen in-progress visit."""

    nodes: dict[int, AGVSNode]
    active_leaf_ids: list[int]
    root_id: int
    global_actions: torch.Tensor
    global_observations: torch.Tensor
    global_posterior_means: torch.Tensor
    global_posterior_stds: torch.Tensor
    benchmark_actions: torch.Tensor
    benchmark_means: torch.Tensor
    current_leaf_id: int | None
    current_path: list[int]
    remaining_sequence_steps: int
    visit_start_best: float | None
    visit_evaluations: int
    next_node_id: int
    batch_count: int
    bad_visits: int
    tree_epoch: int
    restart_count: int
    stopped: bool = False
    stopping_probability: float | None = None
    stopping_action: torch.Tensor | None = None
    stopping_mean: float | None = None


class AG_VS(Algorithm):
    """Adaptive group variable selection using the released source logic."""

    EXCEPTION_CLASS = AG_VSException
    native_batching = True
    native_environments = frozenset({"stochastic"})
    ACQUISITION_POLICY = "reference EI candidate ranking or isotropic TuRBO predictive TS"
    SOURCE_BEHAVIOR_VERSION = "aistats2027-observed-v1"
    SOURCE_TURBO_NUMERICS_VERSION = 3
    SOURCE = SourceRecord(
        paper_title="Adaptive Group Variable Selection for High-Dimensional Bayesian Optimization",
        paper_year=2027,
        paper_url=None,
        repository_url=None,
        repository_status="local",
        notes="Aligned with AISTATS2027_Anonymous_Code.zip, including observed-response scores.",
    )

    def __init__(
        self, *, bounds: Bounds, config: AG_VSConfig, maximize: bool = True,
        dtype: torch.dtype | str = torch.float64, device: torch.device | str = "cpu",
        plot: bool = False, verbose: bool = False, verbose_setting: bool = False,
        verbose_iter_by_iter: bool = False,
    ) -> None:
        self._validate_config(bounds, config)
        if config.inner_optimizer == "turbo" and torch.device(device).type not in {"cpu", "cuda"}:
            raise AGVSNodeError("The source AG-VS TuRBO backend supports CPU or CUDA devices.")
        self.config = replace(
            config,
            initial_points=(config.initial_points if config.initial_points is not None
                            else max(2 * config.split_groups, bounds.dimension + 1)),
            stagnation_patience=(config.stagnation_patience if config.stagnation_patience is not None
                                 else 3 * config.split_groups),
        )
        super().__init__(
            bounds=bounds, batch_size=config.batch_size, q_batch=config.q_batch,
            maximize=maximize, dtype=dtype, device=device, seed=config.seed,
            initial_points_sampling_method=config.initial_points_sampling_method,
            plot=plot, verbose=verbose, verbose_setting=verbose_setting,
            verbose_iter_by_iter=verbose_iter_by_iter,
        )
        self._reset_impl()

    @staticmethod
    def _validate_config(bounds: Bounds, config: AG_VSConfig) -> None:
        try:
            resolve_batch_configuration(batch_size=config.batch_size, q_batch=config.q_batch)
        except ValueError as error:
            raise AGVSNodeError(str(error)) from error
        for name, value, minimum in (
            ("effective_dimension", config.effective_dimension, 1),
            ("split_groups", config.split_groups, 2),
            ("sequence_length", config.sequence_length, 1),
            ("seed", config.seed, 0),
            ("turbo_max_evals", config.turbo_max_evals, 2),
            ("turbo_min_cuda", config.turbo_min_cuda, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise AGVSNodeError(f"{name} must be an explicitly supplied integer >= {minimum}.")
        if not config.effective_dimension < config.split_groups <= bounds.dimension:
            raise AGVSNodeError("Require effective_dimension < split_groups <= D_env.")
        if config.effective_dimension >= bounds.dimension:
            raise AGVSNodeError("effective_dimension must be smaller than D_env.")
        for name in ("initial_points", "stagnation_patience", "gp_maxiter"):
            value = getattr(config, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise AGVSNodeError(f"{name} must be None or a positive integer.")
        if config.inner_optimizer not in {"bo", "turbo"}:
            raise AGVSNodeError("inner_optimizer must be 'bo' or 'turbo'.")
        if config.inner_turbo_config is not None:
            raise AGVSNodeError(
                "inner_turbo_config uses incompatible standalone TuRBO semantics; "
                "use AG-VS noise_variance, learn_noise, gp_maxiter and turbo_max_evals."
            )
        if config.acquisition.name.upper() != "EI" or config.acquisition.xi != 1e-4:
            raise AGVSNodeError("Source AG-VS requires EI with xi=1e-4.")
        if config.gp_fit_method != "mle":
            raise AGVSNodeError("Source AG-VS requires marginal-likelihood GP fitting.")
        if config.stopping_threshold is not None or config.stopping_delta is not None:
            raise AGVSNodeError("Posterior stopping is disabled in the supplied AG-VS source.")
        if not math.isfinite(config.noise_variance) or config.noise_variance < 0:
            raise AGVSNodeError("noise_variance must be finite and non-negative.")
        for name in ("abs_mode", "learn_noise", "stagnation_enabled", "merge_enabled"):
            if not isinstance(getattr(config, name), bool):
                raise AGVSNodeError(f"{name} must be a boolean.")
        batch = config.batch_size if config.batch_size is not None else config.q_batch
        if config.turbo_max_evals <= batch:
            raise AGVSNodeError("turbo_max_evals must exceed the configured point batch size.")

    @property
    def request_size(self) -> int:
        """Serve an exact initialization remainder and keep pending shapes stable."""
        pending = getattr(self, "last_ask", None)
        if pending is not None:
            return pending.batch_size
        loading = getattr(self, "_loading_request_size", None)
        if loading is not None:
            return loading
        requested = super().request_size
        state = getattr(self, "state", None)
        if state is not None and state.global_actions.shape[0] < self.config.initial_points:
            requested = min(requested, self.config.initial_points - state.global_actions.shape[0])
        return requested

    @property
    def source_iteration_complete(self) -> bool:
        count = self.state.global_actions.shape[0]
        return (self.last_ask is None and self.state.current_leaf_id is None
                and (count == 0 or count >= self.config.initial_points))

    def _empty_node(
        self, *, node_id: int, depth: int, parent_id: int | None,
        active_indices: torch.Tensor, anchor: torch.Tensor | None = None,
        training_rows: list[int] | None = None, excluded: bool = False,
        merge_disabled: bool = False, newly_retained: bool = False,
    ) -> AGVSNode:
        active = torch.as_tensor(active_indices, dtype=torch.long, device=self.device).reshape(-1)
        if (not active.numel() or active.unique().numel() != active.numel()
                or bool((active < 0).any()) or bool((active >= self.action_dimension).any())):
            raise AGVSNodeError("Node coordinates must be unique and in range.")
        return AGVSNode(
            node_id=node_id, depth=depth, parent_id=parent_id, child_ids=[],
            active_indices=active.clone(), anchor=None if anchor is None else anchor.clone(),
            training_rows=list(training_rows or []),
            local_X=torch.empty((0, active.numel()), dtype=self.dtype, device=self.device),
            local_y=torch.empty(0, dtype=self.dtype, device=self.device), model=None,
            score_history_rows=[], posterior_mean_history=[], posterior_std_history=[],
            previous_benchmark_history=[], score=float("inf"), score_mass=0,
            pre_split_score=None, pre_split_mass=0, excluded=excluded,
            merge_disabled=merge_disabled, cohort_evaluated=False,
            newly_retained=newly_retained, archived=False, merge_completed=False,
        )

    def _fresh_state(self) -> AGVSState:
        root = self._empty_node(
            node_id=0, depth=0, parent_id=None,
            active_indices=torch.arange(self.action_dimension, device=self.device),
            newly_retained=True, merge_disabled=not self.config.merge_enabled,
        )
        empty = torch.empty(0, dtype=self.dtype, device=self.device)
        actions = torch.empty((0, self.action_dimension), dtype=self.dtype, device=self.device)
        return AGVSState(
            nodes={0: root}, active_leaf_ids=[0], root_id=0,
            global_actions=actions.clone(), global_observations=empty.clone(),
            global_posterior_means=empty.clone(), global_posterior_stds=empty.clone(),
            benchmark_actions=actions.clone(), benchmark_means=empty.clone(),
            current_leaf_id=None, current_path=[], remaining_sequence_steps=0,
            visit_start_best=None, visit_evaluations=0, next_node_id=1,
            batch_count=0, bad_visits=0, tree_epoch=0, restart_count=0,
        )

    def _initialize_from_environment_impl(self, data: Any) -> None:
        source = getattr(data, "evaluated_actions", None)
        source = data.actions if source is None else source
        self.state.global_actions = torch.as_tensor(source, dtype=self.dtype, device=self.device).clone()
        self.state.global_observations = torch.as_tensor(
            data.observations, dtype=self.dtype, device=self.device
        ).reshape(-1).clone()
        count = self.state.global_actions.shape[0]
        self.state.global_posterior_means = torch.full((count,), float("nan"), dtype=self.dtype, device=self.device)
        self.state.global_posterior_stds = torch.full_like(self.state.global_posterior_means, float("nan"))
        root = self.state.nodes[0]
        root.training_rows = list(range(count))
        if count >= self.config.initial_points:
            self._fit_node_gp(root)

    def _bounds_numpy(self, node: AGVSNode | None = None) -> np.ndarray:
        lower, upper = self._bounds_on_algorithm_device()
        if node is not None:
            lower, upper = lower[node.active_indices], upper[node.active_indices]
        return torch.stack((lower, upper), dim=1).detach().cpu().numpy()

    def _new_node_gp(self, dimension: int, node: AGVSNode | None = None) -> Any:
        options = dict(noise_variance=self.config.noise_variance, learn_noise=self.config.learn_noise,
                       maxiter=self.config.gp_maxiter, device=self.device, dtype=self.dtype)
        if self.config.inner_optimizer == "bo":
            return SourceBOGP(**options)
        if node is None:
            raise AGVSNodeError("TuRBO GP construction requires node bounds.")
        return SourceTurboGP(bounds=self._bounds_numpy(node), min_cuda=self.config.turbo_min_cuda, **options)

    def _restore_model(self, state: Mapping[str, Any] | None) -> Any:
        if state is None:
            return None
        cls = SourceTurboGP if self.config.inner_optimizer == "turbo" else SourceBOGP
        return cls.from_state_dict(state, device=self.device, dtype=self.dtype)

    def _fit_node_gp(self, node: AGVSNode) -> None:
        signature = tuple(node.training_rows)
        if node.model is not None and node.fitted_rows == signature:
            return
        rows = torch.tensor(node.training_rows, dtype=torch.long, device=self.device)
        node.local_X = self.state.global_actions[rows][:, node.active_indices].clone()
        node.local_y = to_quality(self.state.global_observations[rows], maximize=self.maximize).clone()
        node.model = None
        node.fitted_rows = None
        if not len(rows):
            return
        try:
            model = self._new_node_gp(int(node.active_indices.numel()), node)
            warm = self._restore_model(node.warm_model_state)
            model.fit(node.local_X, node.local_y, warm_model=warm)
        except Exception as error:
            self._last_gp_fit_error = str(error)
            return
        node.model = model
        node.fitted_rows = signature
        node.warm_model_state = None
        self._last_gp_fit_error = None
        if node.active_indices.numel() == self.action_dimension:
            self._last_full_dim_gp_state = deepcopy(model.state_dict())
        self._update_posterior_records(node)

    def _predict_node(self, node: AGVSNode, full_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if node.model is None:
            raise AGVSNodeError("The node has no fitted GP.")
        means, stds = node.model.predict(full_actions[:, node.active_indices])
        return (torch.as_tensor(means, dtype=self.dtype, device=self.device).reshape(-1),
                torch.as_tensor(stds, dtype=self.dtype, device=self.device).reshape(-1))

    def _posterior_key(self, action: torch.Tensor) -> tuple[int, ...]:
        bounds = self._bounds_numpy()
        normalized = (action.detach().cpu().numpy() - bounds[:, 0]) / np.maximum(bounds[:, 1] - bounds[:, 0], 1.0)
        return tuple(np.rint(normalized / 1e-10).astype(np.int64).tolist())

    def _update_posterior_records(self, node: AGVSNode) -> None:
        rows = node.training_rows
        means, stds = self._predict_node(node, self.state.global_actions[rows])
        for row, mean, std in zip(rows, means.tolist(), stds.tolist(), strict=True):
            if not math.isfinite(mean):
                continue
            key = self._posterior_key(self.state.global_actions[row])
            previous = self._posterior_records.get(key)
            if previous is None or mean > previous["mean"]:
                self._posterior_records[key] = {"row": row, "mean": mean, "std": std}
            old = self.state.global_posterior_means[row]
            if not bool(torch.isfinite(old)) or mean > float(old):
                self.state.global_posterior_means[row] = mean
                self.state.global_posterior_stds[row] = std

    def _current_best_benchmark(self) -> tuple[torch.Tensor | None, float | None]:
        quality = to_quality(self.state.global_observations, maximize=self.maximize)
        finite = torch.isfinite(quality)
        if not bool(finite.any()):
            return None, None
        values = torch.where(finite, quality, torch.full_like(quality, float("-inf")))
        index = int(values.argmax())
        return self.state.global_actions[index].clone(), float(values[index])

    def _append_current_benchmark(self) -> None:
        action, mean = self._current_best_benchmark()
        if action is not None:
            self.state.benchmark_actions = torch.cat((self.state.benchmark_actions, action[None]), dim=0)
            self.state.benchmark_means = torch.cat((self.state.benchmark_means, torch.tensor([mean], dtype=self.dtype, device=self.device)))

    def _set_anchor_from_current_best(self, node: AGVSNode) -> float | None:
        action, value = self._current_best_benchmark()
        if action is None:
            bounds = self._bounds_numpy()
            action = torch.as_tensor(self.numpy_generator.uniform(bounds[:, 0], bounds[:, 1]), dtype=self.dtype, device=self.device)
        node.anchor = action.clone()
        return value

    def _inactive_indices(self, node: AGVSNode) -> torch.Tensor:
        mask = torch.ones(self.action_dimension, dtype=torch.bool, device=self.device)
        mask[node.active_indices] = False
        return torch.arange(self.action_dimension, device=self.device)[mask]

    def _fill_active_coordinates(self, node: AGVSNode, active_points: torch.Tensor) -> torch.Tensor:
        points = torch.as_tensor(active_points, dtype=self.dtype, device=self.device)
        vector = points.ndim == 1
        points = points.reshape(-1, node.active_indices.numel())
        if node.anchor is None:
            raise AGVSAnchorError("No pre-batch observed-incumbent anchor is available.")
        full = node.anchor[None].expand(len(points), -1).clone()
        full[:, node.active_indices] = points
        return full[0] if vector else full

    def _path_to_root(self, node_id: int) -> list[int]:
        path = []
        while node_id is not None:
            path.append(node_id)
            node_id = self.state.nodes[node_id].parent_id
        return path[::-1]

    @staticmethod
    def _score_order_key(node: AGVSNode) -> tuple[float, int]:
        return (-node.score if not math.isnan(node.score) else float("inf"), node.node_id)

    def _traverse(self) -> tuple[AGVSNode, list[int]]:
        pending: dict[int, bool] = {}

        def refresh(node_id: int) -> bool:
            node = self.state.nodes[node_id]
            children = [idx for idx in node.child_ids if not self.state.nodes[idx].archived]
            child_pending = [refresh(idx) for idx in children]
            pending[node_id] = any(child_pending) if children else not node.cohort_evaluated
            return pending[node_id]

        refresh(self.state.root_id)
        node = self.state.nodes[self.state.root_id]
        path = [node.node_id]
        while node.child_ids:
            children = [self.state.nodes[idx] for idx in node.child_ids if not self.state.nodes[idx].archived]
            first = [child for child in children if pending[child.node_id]]
            node = min(first or children, key=self._score_order_key)
            path.append(node.node_id)
        return node, path

    def _leaf_score(self, node: AGVSNode) -> float:
        rows = node.score_history_rows
        if not rows:
            return float("inf")
        values = to_quality(self.state.global_observations[rows], maximize=self.maximize)
        references = torch.tensor(node.previous_benchmark_history, dtype=self.dtype, device=self.device)
        difference = values - references
        if self.config.abs_mode:
            difference = difference.abs()
        valid = torch.isfinite(difference)
        return float(difference[valid].mean()) if bool(valid.any()) else float("inf")

    def _refresh_leaf_score(self, node: AGVSNode) -> None:
        node.score = self._leaf_score(node)

    def _update_parent_score(self, parent: AGVSNode) -> None:
        pairs = [(parent.pre_split_score, parent.pre_split_mass)]
        pairs.extend((self.state.nodes[idx].score, self.state.nodes[idx].score_mass)
                     for idx in parent.child_ids if not self.state.nodes[idx].archived)
        valid = [(value, mass) for value, mass in pairs
                 if mass > 0 and value is not None and math.isfinite(value)]
        total = sum(mass for _, mass in valid)
        parent.score = sum(value * mass for value, mass in valid) / total if total else float("inf")

    def _propagate(self, node_id: int | None) -> None:
        while node_id is not None:
            node = self.state.nodes[node_id]
            if node.child_ids:
                self._update_parent_score(node)
            node_id = node.parent_id

    def _update_scores_bottom_up(self, path: list[int]) -> None:
        self._refresh_leaf_score(self.state.nodes[path[-1]])
        for node_id in path[-2::-1]:
            self._update_parent_score(self.state.nodes[node_id])

    def _split(self, node: AGVSNode) -> list[int]:
        dimension = node.active_indices.numel()
        if dimension < 2 or node.child_ids:
            return []
        ordered = self.numpy_generator.permutation(node.active_indices.detach().cpu().numpy()).tolist()
        target = self.config.split_groups
        if dimension <= target:
            groups = [[value] for value in ordered]
        else:
            base, _remainder = divmod(dimension, target)
            groups = [ordered[index * base:(index + 1) * base] for index in range(target)]
            for index, value in enumerate(ordered[base * target:]):
                groups[target - 1 - index].append(value)
        node.pre_split_score, node.pre_split_mass = node.score, node.score_mass
        children = []
        for group in groups:
            node_id = self.state.next_node_id
            self.state.next_node_id += 1
            child = self._empty_node(
                node_id=node_id, depth=node.depth + 1, parent_id=node.node_id,
                active_indices=torch.tensor(group, device=self.device),
                training_rows=node.training_rows, excluded=node.excluded,
                merge_disabled=node.merge_disabled,
            )
            self.state.nodes[node_id] = child
            children.append(node_id)
        node.child_ids = children
        node.merge_completed = False
        node.proposal_optimizer = None
        self.state.active_leaf_ids = [idx for idx in self.state.active_leaf_ids if idx != node.node_id] + children
        self._propagate(node.node_id)
        return children

    def _merge(self, parent: AGVSNode) -> int | None:
        if parent.merge_completed:
            return None
        siblings = [self.state.nodes[idx] for idx in parent.child_ids if not self.state.nodes[idx].archived]
        if not siblings or not all(node.cohort_evaluated for node in siblings):
            return None
        parent.merge_completed = True
        if parent.merge_disabled:
            return None
        if any(not math.isfinite(node.score) for node in siblings):
            parent.merge_completed = False
            return None
        retained = sorted(siblings, key=self._score_order_key)[:self.config.effective_dimension]
        retained_ids = {node.node_id for node in retained}
        active = sorted({int(value) for node in retained for value in node.active_indices.tolist()})
        training = (list(range(len(self.state.global_actions))) if len(active) == self.config.effective_dimension
                    else sorted({row for node in retained for row in node.training_rows}))
        node_id = self.state.next_node_id
        self.state.next_node_id += 1
        merged = self._empty_node(
            node_id=node_id, depth=parent.depth + 1, parent_id=parent.node_id,
            active_indices=torch.tensor(active, device=self.device), training_rows=training,
            newly_retained=True, merge_disabled=len(active) <= self.config.effective_dimension,
        )
        merged.retained_merged = True
        self.state.nodes[node_id] = merged
        outside = []
        for sibling in siblings:
            if sibling.node_id in retained_ids:
                sibling.archived = True
                sibling.proposal_optimizer = None
            else:
                sibling.excluded = sibling.merge_disabled = True
                outside.append(sibling.node_id)
        parent.child_ids = [*outside, node_id]
        self.state.active_leaf_ids = [idx for idx in self.state.active_leaf_ids if idx not in retained_ids] + [node_id]
        self._propagate(parent.node_id)
        return node_id

    def _unique_pair_rows(self, rows: list[int] | None = None) -> list[int]:
        result = []
        for row in sorted(set(range(len(self.state.global_actions)) if rows is None else rows)):
            if not any(
                bool((self.state.global_actions[row] - self.state.global_actions[earlier]).abs().le(1e-10).all())
                and abs(float(self.state.global_observations[row] - self.state.global_observations[earlier])) <= 1e-10
                for earlier in result
            ):
                result.append(row)
        return result

    def _reinitialize(self) -> int:
        rows = sorted({row for idx in self.state.active_leaf_ids
                       if self.state.nodes[idx].parent_id is not None
                       for row in self.state.nodes[idx].training_rows})
        if not rows:
            return self.state.root_id
        root = self._empty_node(
            node_id=0, depth=0, parent_id=None,
            active_indices=torch.arange(self.action_dimension, device=self.device),
            training_rows=self._unique_pair_rows(rows), newly_retained=True,
            merge_disabled=not self.config.merge_enabled,
        )
        root.warm_model_state = deepcopy(self._last_full_dim_gp_state)
        self.state.nodes = {0: root}
        self.state.root_id = 0
        self.state.active_leaf_ids = [0]
        self.state.next_node_id = 1
        self.state.bad_visits = 0
        self.state.tree_epoch += 1
        self.state.restart_count += 1
        return 0

    def _complete_visit(self, node: AGVSNode) -> None:
        node.cohort_evaluated = True
        self._update_scores_bottom_up(self.state.current_path)
        merged = None
        if node.parent_id is not None:
            merged = self._merge(self.state.nodes[node.parent_id])
        _, best_after = self._current_best_benchmark()
        if self.config.stagnation_enabled:
            improved = (best_after is not None and self.state.visit_start_best is not None
                        and best_after > self.state.visit_start_best)
            self.state.bad_visits = 0 if improved else self.state.bad_visits + 1
            children_exist = any(self.state.nodes[idx].parent_id is not None for idx in self.state.active_leaf_ids)
            if self.state.bad_visits >= self.config.stagnation_patience and children_exist:
                self._reinitialize()
                return
        else:
            self.state.bad_visits = 0
        if merged is not None:
            return
        if node.newly_retained:
            node.newly_retained = False
            if node.active_indices.numel() > self.config.effective_dimension:
                self._split(node)
        elif (node.excluded or not self.config.merge_enabled) and node.active_indices.numel() > 1:
            if node.parent_id is None or self.state.nodes[node.parent_id].merge_completed:
                self._split(node)

    def _initial_design(self, round_info: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
        if "candidate_actions" in round_info:
            self.invalidate_source_guarantees("agvs.finite_candidate_extension")
            actions = torch.as_tensor(round_info["candidate_actions"], dtype=self.dtype, device=self.device)
            indices = torch.as_tensor(round_info["candidate_dataset_indices"], dtype=torch.long, device=self.device)
            order = torch.argsort(indices, stable=True)[:self.request_size]
            return actions[order], indices[order]
        if self._initial_design_points is None:
            count = self.config.initial_points - len(self.state.global_actions)
            bounds = self._bounds_numpy()
            if self.initial_points_sampling_method == "paper":
                design = latin_hypercube(bounds, count, self.numpy_generator)
            elif self.initial_points_sampling_method == "uniform":
                design = self.numpy_generator.uniform(bounds[:, 0], bounds[:, 1], size=(count, self.action_dimension))
            else:
                seed = int(self.numpy_generator.integers(0, 500000))
                unit = torch.quasirandom.SobolEngine(self.action_dimension, scramble=True, seed=seed).draw(count).double().numpy()
                design = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
            self._initial_design_points = torch.as_tensor(design, dtype=self.dtype, device=self.device)
            self._initial_design_cursor = 0
        start = self._initial_design_cursor
        return self._initial_design_points[start:start + self.request_size].clone(), None

    def _prepare_discrete_candidates_impl(
        self, *, candidate_actions: torch.Tensor, candidate_dataset_indices: torch.Tensor,
        round_info: Mapping[str, Any],
    ) -> DiscreteAcquisitionCandidates:
        del round_info
        node = self.state.nodes[self.state.current_leaf_id or self.state.root_id]
        return DiscreteAcquisitionCandidates(candidate_actions, candidate_actions[:, node.active_indices], candidate_dataset_indices)

    def _select_active_points(self, node: AGVSNode, round_info: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float | None]:
        self._set_anchor_from_current_best(node)
        self._fit_node_gp(node)
        bounds = self._bounds_numpy(node)
        if "candidate_actions" in round_info:
            self.invalidate_source_guarantees("agvs.finite_candidate_extension")
            if self.config.inner_optimizer == "turbo" or self.q_batch is not None:
                raise AGVSDataCompatibilityError("Finite AG-VS tables support independent BO EI only.")
            actions = torch.as_tensor(round_info["candidate_actions"], dtype=self.dtype, device=self.device)
            indices = torch.as_tensor(round_info["candidate_dataset_indices"], dtype=torch.long, device=self.device)
            active = actions[:, node.active_indices]
            if node.model is None:
                order = torch.argsort(indices, stable=True)[:self.request_size]
                value = None
            else:
                scores = classical_ei(node.model, active.detach().cpu().numpy())
                order = torch.as_tensor(np.argsort(scores)[-self.request_size:], dtype=torch.long, device=self.device)
                value = float(np.asarray(scores)[order.cpu().numpy()].sum())
            return actions[order], active[order], indices[order], value
        if node.model is None:
            if self.config.inner_optimizer == "turbo" and node.training_rows:
                raise AGVSNodeError(
                    "AG-VS TuRBO could not fit the shared node GP: "
                    + str(self._last_gp_fit_error)
                )
            active, values = self.numpy_generator.uniform(bounds[:, 0], bounds[:, 1], size=(self.request_size, len(bounds))), np.zeros(self.request_size)
        elif self.config.inner_optimizer == "turbo":
            if node.proposal_optimizer is None:
                node.proposal_optimizer = TurboProposalState()
            active, values = propose_turbo(
                node.model, bounds, self.request_size, self._source_turbo_rng,
                self._proposal_torch_generator(node), self._sobol_engines, node.proposal_optimizer,
                max_evals=self.config.turbo_max_evals,
            )
        else:
            active, values = propose_bo(
                node.model, bounds, self.request_size, self.numpy_generator,
                joint=self.q_batch is not None, mc_samples=256, xi=1e-4,
            )
        active_tensor = torch.as_tensor(active, dtype=self.dtype, device=self.device)
        return self._fill_active_coordinates(node, active_tensor), active_tensor, None, float(np.asarray(values).sum())

    def _proposal_torch_generator(self, node: AGVSNode) -> torch.Generator:
        """Match predictive sampling to the source's CPU/CUDA candidate device."""
        candidate_count = min(100 * int(node.active_indices.numel()), 5000)
        if self.device.type == "cpu" or candidate_count < self.config.turbo_min_cuda:
            return self._source_torch_generator
        key = str(self.device)
        if key not in self._source_cuda_generators:
            self._source_cuda_generators[key] = torch.Generator(device=self.device).manual_seed(self.seed)
        return self._source_cuda_generators[key]

    def _start_visit(self) -> AGVSNode:
        node, path = self._traverse()
        if node.retained_merged and not node.excluded and node.active_indices.numel() == self.config.effective_dimension:
            node.training_rows = list(range(len(self.state.global_actions)))
        self.state.current_leaf_id, self.state.current_path = node.node_id, path
        self.state.remaining_sequence_steps = self.config.sequence_length
        _, self.state.visit_start_best = self._current_best_benchmark()
        self.state.visit_evaluations = 0
        return node

    def _ask_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        initial = len(self.state.global_actions) < self.config.initial_points
        if initial:
            node = self.state.nodes[self.state.root_id]
            actions, indices = self._initial_design(round_info)
            active, value, reference = actions, None, None
        else:
            node = self._start_visit() if self.state.current_leaf_id is None else self.state.nodes[self.state.current_leaf_id]
            _, reference = self._current_best_benchmark()
            actions, active, indices, value = self._select_active_points(node, round_info)
        self.model_input_dimension = int(node.active_indices.numel())
        self._last_acquisition_value = value
        metadata = {"node_id": node.node_id, "active_indices": node.active_indices,
                    "dataset_indices": indices, "initial_design": initial,
                    "pre_batch_reference": reference}
        if not initial and self.config.inner_optimizer == "turbo":
            metadata["turbo_old_best"] = float(node.local_y.max()) if node.local_y.numel() else float("-inf")
        self._record_last_ask(actions=actions, model_inputs=active, round_info=round_info, algorithm_metadata=metadata)
        return actions[0] if self.request_size == 1 else actions

    def _tell_impl(self, *, action: torch.Tensor, feedback: Mapping[str, Any], round_info: Mapping[str, Any]) -> None:
        del action, round_info
        assert self.last_ask is not None
        metadata = self.last_ask.algorithm_metadata
        node = self.state.nodes[int(metadata["node_id"])]
        observations = torch.as_tensor(feedback["observation"], dtype=self.dtype, device=self.device).reshape(-1)
        evaluated = torch.as_tensor(feedback["evaluated_action"], dtype=self.dtype, device=self.device)
        start = len(self.state.global_actions)
        rows = list(range(start, start + len(observations)))
        self.state.global_actions = torch.cat((self.state.global_actions, evaluated), dim=0)
        self.state.global_observations = torch.cat((self.state.global_observations, observations))
        empty = torch.full_like(observations, float("nan"))
        self.state.global_posterior_means = torch.cat((self.state.global_posterior_means, empty))
        self.state.global_posterior_stds = torch.cat((self.state.global_posterior_stds, empty.clone()))
        self.state.batch_count += 1
        if metadata["initial_design"]:
            node.training_rows.extend(rows)
            self._initial_design_cursor += len(rows)
            if len(self.state.global_actions) >= self.config.initial_points:
                self._fit_node_gp(node)
            self._append_current_benchmark()
            return
        reference = float(metadata["pre_batch_reference"])
        for node_id in self.state.current_path:
            ancestor = self.state.nodes[node_id]
            ancestor.training_rows.extend(row for row in rows if row not in ancestor.training_rows)
            ancestor.score_history_rows.extend(rows)
            ancestor.previous_benchmark_history.extend([reference] * len(rows))
            ancestor.score_mass += len(rows)
        if self.config.inner_optimizer == "turbo" and node.proposal_optimizer is not None:
            quality = to_quality(observations, maximize=self.maximize).cpu().numpy()
            update_turbo(node.proposal_optimizer, metadata["turbo_old_best"], quality,
                         int(node.active_indices.numel()), self.batch_size or self.q_batch)
        self._append_current_benchmark()
        self.state.visit_evaluations += len(rows)
        self.state.remaining_sequence_steps -= 1
        if self.state.remaining_sequence_steps <= 0:
            self._complete_visit(node)
            self.state.current_leaf_id = None
            self.state.current_path = []
            self.state.remaining_sequence_steps = 0
            self.state.visit_start_best = None
            self.state.visit_evaluations = 0

    def _recommendation_impl(self) -> tuple[torch.Tensor | None, float | None]:
        if self._posterior_records:
            record = max(self._posterior_records.values(), key=lambda value: value["mean"])
            return self.state.global_actions[record["row"]].clone(), record["mean"] if self.maximize else -record["mean"]
        action, _ = self._current_best_benchmark()
        return action, float("nan") if action is not None else None

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        return {
            "source_behavior_version": self.SOURCE_BEHAVIOR_VERSION,
            "source_turbo_numerics_version": (
                self.SOURCE_TURBO_NUMERICS_VERSION if self.config.inner_optimizer == "turbo" else None
            ),
            "node_count": len(self.state.nodes), "root_id": self.state.root_id,
            "active_leaf_ids": tuple(self.state.active_leaf_ids),
            "current_leaf_id": self.state.current_leaf_id,
            "remaining_sequence_steps": self.state.remaining_sequence_steps,
            "tree_epoch": self.state.tree_epoch, "restart_count": self.state.restart_count,
            "bad_visits": self.state.bad_visits, "batch_count": self.state.batch_count,
            "node_scores": {key: node.score for key, node in self.state.nodes.items()},
            "node_score_masses": {key: node.score_mass for key, node in self.state.nodes.items()},
            "node_active_indices": {key: node.active_indices.clone() for key, node in self.state.nodes.items()},
            "last_gp_fit_error": self._last_gp_fit_error,
            "stopped": False, "stopping_probability": None,
        }

    def _node_state(self, node: AGVSNode) -> dict[str, Any]:
        result = {item.name: deepcopy(getattr(node, item.name)) for item in fields(node)
                  if item.name not in {"model", "proposal_optimizer"}}
        result["model"] = None if node.model is None else node.model.state_dict()
        result["proposal_optimizer"] = None if node.proposal_optimizer is None else node.proposal_optimizer.state_dict()
        return result

    def _state_dict_impl(self) -> Mapping[str, Any]:
        result = {item.name: deepcopy(getattr(self.state, item.name)) for item in fields(self.state) if item.name != "nodes"}
        result.update(
            source_behavior_version=self.SOURCE_BEHAVIOR_VERSION, config=asdict(self.config),
            source_turbo_numerics_version=(
                self.SOURCE_TURBO_NUMERICS_VERSION if self.config.inner_optimizer == "turbo" else None
            ),
            nodes={key: self._node_state(node) for key, node in self.state.nodes.items()},
            numpy_generator_state=deepcopy(self.numpy_generator.bit_generator.state),
            source_turbo_rng_state=deepcopy(self._source_turbo_rng.get_state()),
            source_torch_generator_state=self._source_torch_generator.get_state().clone(),
            source_cuda_generator_states={key: generator.get_state().cpu().clone()
                                          for key, generator in self._source_cuda_generators.items()},
            initial_design_points=self._initial_design_points, initial_design_cursor=self._initial_design_cursor,
            posterior_records=deepcopy(self._posterior_records),
            last_full_dim_gp_state=deepcopy(self._last_full_dim_gp_state),
            last_gp_fit_error=self._last_gp_fit_error,
            sobol_engines={dimension: {"seed": engine.seed, "num_generated": engine.num_generated}
                           for dimension, engine in self._sobol_engines.items()},
        )
        return result

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        del strict
        if state.get("source_behavior_version") != self.SOURCE_BEHAVIOR_VERSION:
            raise AGVSCheckpointError("AG-VS checkpoint predates source alignment; start a fresh trajectory.")
        self._validate_turbo_numerics_version(state)
        required = {item.name for item in fields(AGVSState)} | {
            "numpy_generator_state", "source_turbo_rng_state", "source_torch_generator_state", "initial_design_points",
            "initial_design_cursor", "posterior_records", "last_full_dim_gp_state", "sobol_engines",
        }
        missing = required - set(state)
        if missing:
            raise AGVSCheckpointError(f"AG-VS state is missing {sorted(missing)}.")
        actions = torch.as_tensor(state["global_actions"])
        count = len(actions)
        if actions.ndim != 2 or actions.shape[1] != self.action_dimension:
            raise AGVSCheckpointError("Invalid global action shape.")
        for name in ("global_observations", "global_posterior_means", "global_posterior_stds"):
            if torch.as_tensor(state[name]).shape != (count,):
                raise AGVSCheckpointError("Global observation/posterior records are not aligned.")
        nodes = state["nodes"]
        if not isinstance(nodes, Mapping) or state["root_id"] not in nodes:
            raise AGVSCheckpointError("AG-VS tree has no root.")
        for key, node in nodes.items():
            if node["node_id"] != key:
                raise AGVSCheckpointError("Node key and identity disagree.")
            active = torch.as_tensor(node["active_indices"])
            if (active.ndim != 1 or not active.numel() or active.unique().numel() != active.numel()
                    or bool((active < 0).any()) or bool((active >= self.action_dimension).any())):
                raise AGVSCheckpointError("Invalid checkpoint node coordinates.")
            for name in ("training_rows", "score_history_rows"):
                rows = node[name]
                if len(rows) != len(set(rows)) or any(not 0 <= row < count for row in rows):
                    raise AGVSCheckpointError("Invalid checkpoint node row indices.")
            if len(node["score_history_rows"]) != len(node["previous_benchmark_history"]) or node["score_mass"] != len(node["score_history_rows"]):
                raise AGVSCheckpointError("Historical score evidence is not aligned.")
            if any(child not in nodes or nodes[child]["parent_id"] != key for child in node["child_ids"]):
                raise AGVSCheckpointError("Invalid tree adjacency.")
            if node["model"] is not None:
                self._restore_model(node["model"])
        try:
            temporary = np.random.default_rng()
            temporary.bit_generator.state = deepcopy(state["numpy_generator_state"])
            np.random.RandomState().set_state(state["source_turbo_rng_state"])
            torch.Generator(device="cpu").set_state(torch.as_tensor(state["source_torch_generator_state"], dtype=torch.uint8).cpu())
            for key, value in state.get("source_cuda_generator_states", {}).items():
                if self.device.type != "cuda" or key != str(self.device):
                    raise ValueError("A CUDA random stream belongs to a different device.")
                torch.Generator(device=self.device).set_state(torch.as_tensor(value, dtype=torch.uint8).cpu())
        except Exception as error:
            raise AGVSCheckpointError("Invalid source RNG state.") from error

    def load_state_dict(self, state: Mapping[str, Any], *, strict: bool = True) -> None:
        implementation = state.get("algorithm_state", {})
        if implementation.get("source_behavior_version") != self.SOURCE_BEHAVIOR_VERSION:
            raise AGVSCheckpointError("AG-VS checkpoint predates source alignment; start a fresh trajectory.")
        self._validate_turbo_numerics_version(implementation)
        pending = state.get("last_ask")
        self._loading_request_size = len(pending["actions"]) if pending is not None else None
        try:
            super().load_state_dict(state, strict=strict)
        finally:
            self._loading_request_size = None

    def _validate_turbo_numerics_version(self, state: Mapping[str, Any]) -> None:
        if (self.config.inner_optimizer == "turbo"
                and state.get("source_turbo_numerics_version") != self.SOURCE_TURBO_NUMERICS_VERSION):
            raise AGVSCheckpointError(
                "AG-VS TuRBO checkpoint uses an older numerical policy; "
                f"start a fresh trajectory with source numerical version {self.SOURCE_TURBO_NUMERICS_VERSION}."
            )

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        del strict
        values = {item.name: deepcopy(state[item.name]) for item in fields(AGVSState) if item.name != "nodes"}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                values[key] = value.to(dtype=self.dtype, device=self.device).clone()
        nodes = {}
        for key, value in state["nodes"].items():
            stored = deepcopy(value)
            model_state = stored.pop("model")
            proposal_state = stored.pop("proposal_optimizer")
            for name in ("active_indices", "anchor", "local_X", "local_y"):
                if stored[name] is not None:
                    stored[name] = torch.as_tensor(stored[name], dtype=torch.long if name == "active_indices" else self.dtype, device=self.device).clone()
            stored["fitted_rows"] = None if stored["fitted_rows"] is None else tuple(stored["fitted_rows"])
            stored["model"] = self._restore_model(model_state)
            stored["proposal_optimizer"] = None if proposal_state is None else TurboProposalState.from_state_dict(proposal_state)
            nodes[int(key)] = AGVSNode(**stored)
        self.state = AGVSState(nodes=nodes, **values)
        self.numpy_generator.bit_generator.state = deepcopy(state["numpy_generator_state"])
        self._source_turbo_rng.set_state(state["source_turbo_rng_state"])
        self._source_torch_generator.set_state(torch.as_tensor(state["source_torch_generator_state"], dtype=torch.uint8).cpu())
        self._source_cuda_generators = {}
        for key, value in state.get("source_cuda_generator_states", {}).items():
            generator = torch.Generator(device=self.device)
            generator.set_state(torch.as_tensor(value, dtype=torch.uint8).cpu())
            self._source_cuda_generators[key] = generator
        points = state["initial_design_points"]
        self._initial_design_points = None if points is None else torch.as_tensor(points, dtype=self.dtype, device=self.device).clone()
        self._initial_design_cursor = int(state["initial_design_cursor"])
        self._posterior_records = deepcopy(state["posterior_records"])
        self._last_full_dim_gp_state = deepcopy(state["last_full_dim_gp_state"])
        self._last_gp_fit_error = state.get("last_gp_fit_error")
        self._sobol_engines = {}
        for dimension, value in state["sobol_engines"].items():
            engine = torch.quasirandom.SobolEngine(int(dimension), scramble=True, seed=int(value["seed"]))
            if value["num_generated"]:
                engine.fast_forward(int(value["num_generated"]))
            self._sobol_engines[int(dimension)] = engine

    def _reset_impl(self) -> None:
        self.numpy_generator = np.random.default_rng(self.seed)
        self._source_turbo_rng = np.random.RandomState(self.seed)
        self._source_torch_generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self._source_cuda_generators: dict[str, torch.Generator] = {}
        self._sobol_engines: dict[int, Any] = {}
        self._initial_design_points = None
        self._initial_design_cursor = 0
        self._posterior_records: dict[tuple[int, ...], dict[str, Any]] = {}
        self._last_full_dim_gp_state = None
        self._last_gp_fit_error = None
        self.state = self._fresh_state()
        self.model_input_dimension = self.action_dimension


AGVSConfig = AG_VSConfig

__all__ = ["AG_VS", "AGVSConfig", "AGVSNode", "AGVSState", "AG_VSConfig"]
