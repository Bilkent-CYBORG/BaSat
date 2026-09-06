"""Adaptive Group Variable Selection (AG-VS).

This implementation follows the local AG-VS manuscript's index-based tree
state. Node training data and score records are separate global-index sets;
inactive coordinates are filled from the best stored-posterior point before
each acquisition batch. In particular, Eq. (8) is an unscaled mean absolute
posterior gap--it is not a percentage and has no denominator normalization.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch

from bayopy.models.Acquisitions import (
    AcquisitionConfig,
    DiscreteAcquisitionCandidates,
    build_acquisition,
    optimize_acquisition_continuous,
    optimize_acquisition_discrete,
)
from bayopy.models.GaussianProcess import GaussianProcess, GaussianProcessConfig
from bayopy.models.Kernels import KernelConfig
from bayopy.models.utils import Bounds

from ..algorithm import Algorithm, resolve_batch_configuration
from ..helper import ExceptionDiagnosticIssue, SourceRecord, to_quality
from .Exception.exception_AG_VS import (
    AG_VSException,
    AGVSAnchorError,
    AGVSCheckpointError,
    AGVSDataCompatibilityError,
    AGVSMergeError,
    AGVSNodeError,
    AGVSSplitError,
)


@dataclass(frozen=True, slots=True)
class AGVSConfig:
    """Configuration for the AG-VS procedure in the local manuscript.

    ``batch_size`` is the manuscript batch size ``B`` and independently ranks
    points with the configured acquisition. ``q_batch`` is BayOPy's optional
    joint-acquisition extension (for example EI becomes qEI). Exactly one is
    active; the default is sequential ``batch_size=1, q_batch=None``.

    Parameters
    ----------
    effective_dimension:
        Target retained coordinate count ``d_eff``.
    sequence_length, split_groups, initial_points:
        Acquisition batches per visit, maximum split groups, and initial
        full-coordinate design size.
    acquisition:
        EI or UCB acquisition configuration used within the active coordinates.
    stopping_threshold, stopping_delta:
        Optional ``tau`` and ``delta_stop`` parameters for Eq. (9).
    stagnation_patience:
        Optional ``N_bad`` complete-visit count; ``None`` disables reinitialization.
    gp_fit_method, gp_maxiter:
        Node-local isotropic Matérn-5/2 GP fit method and optimizer budget.
    acquisition_restarts, acquisition_raw_samples:
        Continuous acquisition optimizer budgets.
    batch_size, q_batch:
        Mutually exclusive independent and joint batch controls.
    seed:
        Non-negative algorithm-owned random seed.
    """

    effective_dimension: int
    sequence_length: int = 3
    split_groups: int = 2
    initial_points: int = 5
    acquisition: AcquisitionConfig = field(default_factory=lambda: AcquisitionConfig(name="UCB"))
    stopping_threshold: float | None = None
    stopping_delta: float | None = None
    stagnation_patience: int | None = None
    gp_fit_method: Literal["mle", "map"] = "mle"
    gp_maxiter: int = 50
    acquisition_restarts: int = 10
    acquisition_raw_samples: int = 128
    batch_size: int | None = 1
    q_batch: int | None = None
    seed: int = 1


@dataclass(slots=True)
class AGVSNode:
    """One coordinate-subset node in the AG-VS tree.

    ``training_rows`` is :math:`V_N`; ``score_history_rows`` is
    :math:`I_N`. ``posterior_mean_history`` and
    ``previous_benchmark_history`` are immutable records made when score rows
    were acquired. The current GP is queried afresh when Eq. (8) is updated.

    Attributes
    ----------
    node_id, depth, parent_id, child_ids:
        Stable tree identity and adjacency.
    active_indices, anchor:
        Active coordinates and the most recent pre-batch completion point.
    training_rows, local_X, local_y, model:
        Global ``V_N`` indices and their fitted node-local GP representation.
    score_history_rows, posterior_mean_history, posterior_std_history:
        Global ``I_N`` indices and immutable fitted records from their batches.
    previous_benchmark_history:
        Immutable pre-batch references ``M_i^-`` aligned with ``I_N``.
    score, score_mass, pre_split_score, pre_split_mass:
        Current and frozen evidence-weighted score state.
    excluded, merge_disabled, cohort_evaluated, newly_retained:
        Algorithm 6 transition flags.
    archived, merge_completed:
        Replacement and one-merge-only bookkeeping flags.
    """

    node_id: int
    depth: int
    parent_id: int | None
    child_ids: list[int]
    active_indices: torch.Tensor
    anchor: torch.Tensor | None
    training_rows: list[int]
    local_X: torch.Tensor
    local_y: torch.Tensor
    model: GaussianProcess | None
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


@dataclass(slots=True)
class AGVSState:
    """Index-based AG-VS observations and search progress for continuation.

    Attributes
    ----------
    nodes, active_leaf_ids, root_id:
        Current coordinate tree and active frontier.
    global_actions, global_observations:
        Preserved full raw history indexed globally.
    global_posterior_means, global_posterior_stds:
        Immutable fitted posterior records aligned with the raw history.
    benchmark_actions, benchmark_means:
        Best stored-posterior point and value after each processed batch.
    current_leaf_id, current_path, remaining_sequence_steps:
        Frozen traversal and remaining batches in the active visit.
    visit_start_best, visit_evaluations:
        Stagnation reference and evaluation count for the active visit.
    next_node_id, batch_count, bad_visits:
        Stable identity, global batch, and stagnation counters.
    tree_epoch, restart_count:
        Data-preserving reinitialization counters.
    stopped, stopping_probability, stopping_action, stopping_mean:
        Eq. (9) stopping state and returned fitted point.
    """

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
    stopped: bool
    stopping_probability: float | None
    stopping_action: torch.Tensor | None
    stopping_mean: float | None


class AG_VS(Algorithm):
    """Tree-structured adaptive group variable selection from the manuscript.

    Examples
    --------
    >>> import bayopy
    >>> bounds = bayopy.Bounds(lower=[-1.0] * 6, upper=[1.0] * 6)
    >>> algorithm = bayopy.AG_VS(
    ...     bounds=bounds,
    ...     config=bayopy.AGVSConfig(effective_dimension=2, initial_points=5, seed=7),
    ... )
    """

    ACQUISITION_POLICY = "configurable EI or UCB within active groups"

    EXCEPTION_CLASS = AG_VSException
    native_batching = True
    native_environments = frozenset({"stochastic"})
    SOURCE = SourceRecord(
        paper_title="Adaptive Group Variable Selection for High-Dimensional Bayesian Optimization",
        paper_year=2027,
        paper_url=None,
        repository_url=None,
        repository_status="local",
        notes="Implementation governed by the local AG-VS manuscript.",
    )

    def __init__(
        self,
        *,
        bounds: Bounds,
        config: AGVSConfig,
        maximize: bool = True,
        dtype: torch.dtype | str = torch.float64,
        device: torch.device | str = "cpu",
        plot: bool = False,
        verbose: bool = False,
        verbose_setting: bool = False,
        verbose_iter_by_iter: bool = False,
    ) -> None:
        """Initialize a full-coordinate retained root and empty global history."""

        self.config = config
        self._validate_config(bounds, config)
        super().__init__(
            bounds=bounds,
            batch_size=config.batch_size,
            q_batch=config.q_batch,
            maximize=maximize,
            dtype=dtype,
            device=device,
            seed=config.seed,
            plot=plot,
            verbose=verbose,
            verbose_setting=verbose_setting,
            verbose_iter_by_iter=verbose_iter_by_iter,
        )
        if self.q_batch is not None and self.q_batch > 1:
            self.invalidate_source_guarantees("bayopy.joint_q_acquisition")
        self.state = self._fresh_state()

    @staticmethod
    def _validate_config(bounds: Bounds, config: AGVSConfig) -> None:
        """Validate dimensions, batching, GP budgets, stopping, and restarts."""

        try:
            resolve_batch_configuration(batch_size=config.batch_size, q_batch=config.q_batch)
        except ValueError as error:
            raise AGVSNodeError(str(error)) from error
        if (
            isinstance(config.effective_dimension, bool)
            or not isinstance(config.effective_dimension, int)
            or not 1 <= config.effective_dimension <= bounds.dimension
        ):
            raise AGVSNodeError("effective_dimension must lie in [1, D_env].")
        for name, value, minimum in (
            ("sequence_length", config.sequence_length, 1),
            ("split_groups", config.split_groups, 2),
            ("initial_points", config.initial_points, 1),
            ("gp_maxiter", config.gp_maxiter, 1),
            ("acquisition_restarts", config.acquisition_restarts, 1),
            ("acquisition_raw_samples", config.acquisition_raw_samples, 1),
            ("seed", config.seed, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise AGVSNodeError(f"{name} must be an integer >= {minimum}.")
        if config.stagnation_patience is not None and (
            isinstance(config.stagnation_patience, bool)
            or not isinstance(config.stagnation_patience, int)
            or config.stagnation_patience < 1
        ):
            raise AGVSNodeError("stagnation_patience must be None or a positive integer.")
        if config.gp_fit_method not in {"mle", "map"}:
            raise AGVSNodeError("gp_fit_method must be 'mle' or 'map'.")
        stopping_enabled = (
            config.stopping_threshold is not None or config.stopping_delta is not None
        )
        if stopping_enabled and (
            config.stopping_threshold is None
            or config.stopping_delta is None
            or not math.isfinite(config.stopping_threshold)
            or not 0 < config.stopping_delta < 1
        ):
            raise AGVSNodeError(
                "stopping_threshold and stopping_delta in (0, 1) are required together."
            )

    def _configuration_diagnostic_issues(
        self, environment: Any
    ) -> tuple[ExceptionDiagnosticIssue, ...]:
        """Qualify the optional joint-q extension while accepting paper batch size B."""

        del environment
        if self.q_batch is None or self.q_batch <= 1:
            return ()
        return (
            ExceptionDiagnosticIssue(
                severity="warning",
                code="batch.joint_q_extension",
                message=(
                    "AG-VS q_batch uses BayOPy's joint q-acquisition extension; "
                    "the manuscript specifies a generic batch acquisition routine."
                ),
                details={"q_batch": self.q_batch, "affects_source_guarantees": True},
            ),
        )

    def _empty_node(
        self,
        *,
        node_id: int,
        depth: int,
        parent_id: int | None,
        active_indices: torch.Tensor,
        anchor: torch.Tensor | None = None,
        training_rows: list[int] | None = None,
        excluded: bool = False,
        merge_disabled: bool = False,
        newly_retained: bool = False,
    ) -> AGVSNode:
        """Create an unfitted node with manuscript ``+inf`` first-visit priority."""

        active = torch.as_tensor(active_indices, dtype=torch.long, device=self.device).reshape(-1)
        if (
            active.numel() == 0
            or torch.unique(active).numel() != active.numel()
            or (active < 0).any()
            or (active >= self.action_dimension).any()
        ):
            raise AGVSNodeError("A node must contain unique in-range active coordinates.")
        dimension = int(active.numel())
        return AGVSNode(
            node_id=node_id,
            depth=depth,
            parent_id=parent_id,
            child_ids=[],
            active_indices=active.detach().clone(),
            anchor=None if anchor is None else anchor.detach().clone(),
            training_rows=list(training_rows or []),
            local_X=torch.empty((0, dimension), dtype=self.dtype, device=self.device),
            local_y=torch.empty((0,), dtype=self.dtype, device=self.device),
            model=None,
            score_history_rows=[],
            posterior_mean_history=[],
            posterior_std_history=[],
            previous_benchmark_history=[],
            score=float("inf"),
            score_mass=0,
            pre_split_score=None,
            pre_split_mass=0,
            excluded=excluded,
            merge_disabled=merge_disabled,
            cohort_evaluated=False,
            newly_retained=newly_retained,
            archived=False,
            merge_completed=False,
        )

    def _fresh_state(self) -> AGVSState:
        """Create the initially retained full-dimensional root."""

        root_indices = torch.arange(self.action_dimension, dtype=torch.long, device=self.device)
        root = self._empty_node(
            node_id=0,
            depth=0,
            parent_id=None,
            active_indices=root_indices,
            newly_retained=True,
        )
        return AGVSState(
            nodes={0: root},
            active_leaf_ids=[0],
            root_id=0,
            global_actions=torch.empty(
                (0, self.action_dimension), dtype=self.dtype, device=self.device
            ),
            global_observations=torch.empty((0,), dtype=self.dtype, device=self.device),
            global_posterior_means=torch.empty((0,), dtype=self.dtype, device=self.device),
            global_posterior_stds=torch.empty((0,), dtype=self.dtype, device=self.device),
            benchmark_actions=torch.empty(
                (0, self.action_dimension), dtype=self.dtype, device=self.device
            ),
            benchmark_means=torch.empty((0,), dtype=self.dtype, device=self.device),
            current_leaf_id=None,
            current_path=[],
            remaining_sequence_steps=0,
            visit_start_best=None,
            visit_evaluations=0,
            next_node_id=1,
            batch_count=0,
            bad_visits=0,
            tree_epoch=0,
            restart_count=0,
            stopped=False,
            stopping_probability=None,
            stopping_action=None,
            stopping_mean=None,
        )

    def _initialize_from_environment_impl(self, data: Any) -> None:
        """Create the retained root from learner-visible historical observations."""

        evaluated_actions = getattr(data, "evaluated_actions", None)
        action_source = data.actions if evaluated_actions is None else evaluated_actions
        actions = torch.as_tensor(action_source, dtype=self.dtype, device=self.device).clone()
        observations = (
            torch.as_tensor(data.observations, dtype=self.dtype, device=self.device)
            .reshape(-1)
            .clone()
        )
        self.state.global_actions = actions
        self.state.global_observations = observations
        root = self.state.nodes[self.state.root_id]
        root.training_rows = list(range(actions.shape[0]))
        if actions.shape[0] == 0:
            return
        self._fit_node_gp(root)
        means, stds = self._predict_node(root, actions)
        self.state.global_posterior_means = means
        self.state.global_posterior_stds = stds
        self._append_current_benchmark()

    @staticmethod
    def _node_kernel() -> KernelConfig:
        """Return the reference isotropic Matérn-5/2 covariance configuration."""

        return KernelConfig(
            name="matern52",
            ard_num_dims=None,
            active_dims=None,
            use_outputscale=True,
        )

    def _new_node_gp(self, dimension: int) -> GaussianProcess:
        """Construct an unfitted GP in a node's active coordinate system."""

        return GaussianProcess(
            GaussianProcessConfig(
                kernel=self._node_kernel(),
                mean="zero",
                noise_mode="fixed",
                noise_variance=0.01,
                fit_method=self.config.gp_fit_method,
                dtype=self.dtype,
                device=self.device,
            ),
            input_dimension=dimension,
        )

    def _fit_node_gp(self, node: AGVSNode) -> None:
        """Fit ``G_N`` on the indexed rows in ``V_N``, projected to ``A_N``."""

        if not node.training_rows:
            node.local_X = torch.empty(
                (0, node.active_indices.numel()), dtype=self.dtype, device=self.device
            )
            node.local_y = torch.empty((0,), dtype=self.dtype, device=self.device)
            node.model = None
            return
        rows = torch.as_tensor(node.training_rows, dtype=torch.long, device=self.device)
        if (rows < 0).any() or (rows >= self.state.global_actions.shape[0]).any():
            raise AGVSNodeError(f"Node {node.node_id} contains an invalid training index.")
        node.local_X = self.state.global_actions[rows][:, node.active_indices].clone()
        node.local_y = to_quality(
            self.state.global_observations[rows], maximize=self.maximize
        ).clone()
        gp = self._new_node_gp(int(node.active_indices.numel()))
        try:
            gp.set_data(node.local_X, node.local_y)
            if gp.model is not None:
                covariance = getattr(gp.model, "covar_module", None)
                if covariance is not None:
                    base = getattr(covariance, "base_kernel", covariance)
                    if hasattr(base, "lengthscale"):
                        base.initialize(lengthscale=1.0)
                    if hasattr(covariance, "outputscale"):
                        covariance.initialize(outputscale=1.0)
            gp.fit(maxiter=self.config.gp_maxiter)
        except Exception as error:
            raise AGVSNodeError(f"Node {node.node_id} GP fitting failed: {error}") from error
        node.model = gp

    def _predict_node(
        self, node: AGVSNode, full_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict at full actions after projecting them into node coordinates."""

        if node.model is None:
            raise AGVSNodeError(f"Node {node.node_id} has no fitted GP.")
        means, stds = node.model.predict(full_actions[:, node.active_indices])
        return means.reshape(-1).detach().clone(), stds.reshape(-1).detach().clone()

    def _current_best_benchmark(self) -> tuple[torch.Tensor | None, float | None]:
        """Return the point with largest immutable stored posterior mean ``G_l``."""

        means = self.state.global_posterior_means
        if means.numel() and torch.isfinite(means).any():
            eligible = torch.where(
                torch.isfinite(means), means, torch.full_like(means, float("-inf"))
            )
            index = int(torch.argmax(eligible).item())
            return self.state.global_actions[index].clone(), float(means[index].item())
        if self.state.global_observations.numel() == 0:
            return None, None
        quality = to_quality(self.state.global_observations, maximize=self.maximize)
        index = int(torch.argmax(quality).item())
        return self.state.global_actions[index].clone(), float(quality[index].item())

    def _append_current_benchmark(self) -> None:
        """Append the current best stored-posterior point to the batch history."""

        action, mean = self._current_best_benchmark()
        if action is None or mean is None:
            return
        self.state.benchmark_actions = torch.cat(
            (self.state.benchmark_actions, action.unsqueeze(0)), dim=0
        )
        self.state.benchmark_means = torch.cat(
            (
                self.state.benchmark_means,
                torch.tensor([mean], dtype=self.dtype, device=self.device),
            ),
            dim=0,
        )

    def _inactive_indices(self, node: AGVSNode) -> torch.Tensor:
        """Return environment coordinates outside the node's active set."""

        mask = torch.ones(self.action_dimension, dtype=torch.bool, device=self.device)
        mask[node.active_indices] = False
        return torch.arange(self.action_dimension, device=self.device)[mask]

    def _set_anchor_from_current_best(self, node: AGVSNode) -> float | None:
        """Snapshot the best point available before the current acquisition batch."""

        action, mean = self._current_best_benchmark()
        if action is None:
            lower, upper = self._bounds_on_algorithm_device()
            action = (lower + upper) / 2
        node.anchor = action.detach().clone()
        return mean

    def _fill_active_coordinates(self, node: AGVSNode, active_points: torch.Tensor) -> torch.Tensor:
        """Fill inactive coordinates from the node's current pre-batch best point."""

        points = torch.as_tensor(active_points, dtype=self.dtype, device=self.device)
        vector = points.ndim == 1
        if vector:
            points = points.unsqueeze(0)
        if points.ndim != 2 or points.shape[1] != node.active_indices.numel():
            raise AGVSAnchorError("Active points do not match the node coordinate dimension.")
        if node.anchor is None:
            raise AGVSAnchorError(f"Node {node.node_id} has no pre-batch best-point anchor.")
        result = node.anchor.unsqueeze(0).expand(points.shape[0], -1).clone()
        result[:, node.active_indices] = points
        return result[0] if vector else result

    def _path_to_root(self, node_id: int) -> list[int]:
        """Return the root-to-node path."""

        path: list[int] = []
        current: int | None = node_id
        while current is not None:
            path.append(current)
            current = self.state.nodes[current].parent_id
        return list(reversed(path))

    @staticmethod
    def _score_order_key(node: AGVSNode) -> tuple[float, int]:
        """Order larger scores first with the manuscript's local-index tie break."""

        score = node.score if not math.isnan(node.score) else float("-inf")
        return (-score, node.node_id)

    def _traverse(self) -> tuple[AGVSNode, list[int]]:
        """Follow highest-scoring children from the root to an active leaf."""

        node = self.state.nodes[self.state.root_id]
        path = [node.node_id]
        active = set(self.state.active_leaf_ids)
        while node.node_id not in active:
            children = [
                self.state.nodes[index]
                for index in node.child_ids
                if index in self.state.nodes and not self.state.nodes[index].archived
            ]
            if not children:
                raise AGVSNodeError(f"Internal node {node.node_id} has no traversable child.")
            node = sorted(children, key=self._score_order_key)[0]
            path.append(node.node_id)
        if node.archived:
            raise AGVSNodeError("AG-VS traversal reached an archived leaf.")
        return node, path

    def _leaf_score(self, node: AGVSNode) -> float:
        """Compute unnormalized Eq. (8): mean ``|mu_current(x_i) - M_i^-|``."""

        if not node.score_history_rows:
            return float("inf")
        if len(node.score_history_rows) != len(node.previous_benchmark_history):
            raise AGVSNodeError(f"Node {node.node_id} score records are not aligned.")
        if node.model is None:
            raise AGVSNodeError(f"Node {node.node_id} cannot score without a fitted GP.")
        rows = torch.as_tensor(node.score_history_rows, dtype=torch.long, device=self.device)
        current_means, _ = self._predict_node(node, self.state.global_actions[rows])
        previous = torch.tensor(
            node.previous_benchmark_history, dtype=self.dtype, device=self.device
        )
        return float((current_means - previous).abs().mean().item())

    def _refresh_leaf_score(self, node: AGVSNode) -> None:
        """Refresh a leaf's direct score and direct evidence mass."""

        node.score = self._leaf_score(node)
        node.score_mass = len(node.score_history_rows)

    def _update_scores_bottom_up(self, path: list[int]) -> None:
        """Apply Algorithm 2's mass-weighted score propagation."""

        if not path:
            raise AGVSNodeError("A completed visit requires a non-empty root-to-leaf path.")
        leaf = self.state.nodes[path[-1]]
        self._refresh_leaf_score(leaf)
        for node_id in reversed(path[:-1]):
            parent = self.state.nodes[node_id]
            weighted_sum = 0.0
            total_mass = 0
            if parent.pre_split_mass > 0 and parent.pre_split_score is not None:
                weighted_sum += parent.pre_split_mass * parent.pre_split_score
                total_mass += parent.pre_split_mass
            for child_id in parent.child_ids:
                child = self.state.nodes[child_id]
                if child.archived or child.score_mass <= 0:
                    continue
                weighted_sum += child.score_mass * child.score
                total_mass += child.score_mass
            if total_mass > 0:
                parent.score = weighted_sum / total_mass
                parent.score_mass = total_mass
            else:
                parent.score = float("inf")
                parent.score_mass = 0

    def _split(self, node: AGVSNode) -> list[int]:
        """Apply Algorithm 3, including smaller groups before remainder groups."""

        dimension = int(node.active_indices.numel())
        if node.child_ids or dimension <= 1:
            return []
        count = min(self.config.split_groups, dimension)
        if count < 2:
            return []
        order = torch.randperm(dimension, device=self.device, generator=self.generator)
        ordered = node.active_indices[order]
        quotient, remainder = divmod(dimension, count)
        sizes = [quotient + (1 if index >= count - remainder else 0) for index in range(count)]
        groups: list[torch.Tensor] = []
        cursor = 0
        for size in sizes:
            groups.append(ordered[cursor : cursor + size].clone())
            cursor += size
        flattened = torch.cat(groups)
        if cursor != dimension or torch.unique(flattened).numel() != dimension:
            raise AGVSSplitError("AG-VS split groups are not a disjoint partition.")

        node.pre_split_score = node.score
        node.pre_split_mass = node.score_mass
        children: list[int] = []
        for group in groups:
            child_id = self.state.next_node_id
            self.state.next_node_id += 1
            child = self._empty_node(
                node_id=child_id,
                depth=node.depth + 1,
                parent_id=node.node_id,
                active_indices=group,
                training_rows=node.training_rows,
                excluded=node.excluded,
                merge_disabled=node.merge_disabled,
                newly_retained=False,
            )
            self.state.nodes[child_id] = child
            node.child_ids.append(child_id)
            children.append(child_id)
        node.merge_completed = False
        self.state.active_leaf_ids = [
            leaf_id for leaf_id in self.state.active_leaf_ids if leaf_id != node.node_id
        ] + children
        return children

    def _merge(self, parent: AGVSNode) -> int | None:
        """Apply Algorithm 4 after every merge-enabled sibling's first visit."""

        if parent.merge_completed:
            return None
        siblings = [
            self.state.nodes[index]
            for index in parent.child_ids
            if not self.state.nodes[index].archived
        ]
        if len(siblings) < 2 or not all(node.cohort_evaluated for node in siblings):
            return None
        if any(node.merge_disabled for node in siblings):
            return None
        ordered = sorted(siblings, key=self._score_order_key)
        retained = ordered[: min(self.config.effective_dimension, len(ordered))]
        retained_ids = {node.node_id for node in retained}
        active = torch.unique(torch.cat([node.active_indices for node in retained]), sorted=True)
        if active.numel() == 0:
            raise AGVSMergeError("AG-VS merge produced an empty active set.")
        training_rows = sorted({row for node in retained for row in node.training_rows})
        merged_id = self.state.next_node_id
        self.state.next_node_id += 1
        merged = self._empty_node(
            node_id=merged_id,
            depth=parent.depth + 1,
            parent_id=parent.node_id,
            active_indices=active,
            training_rows=training_rows,
            newly_retained=True,
        )
        self.state.nodes[merged_id] = merged

        outside: list[int] = []
        for sibling in siblings:
            if sibling.node_id in retained_ids:
                sibling.archived = True
            else:
                sibling.excluded = True
                sibling.merge_disabled = True
                outside.append(sibling.node_id)
        parent.child_ids = [*outside, merged_id]
        parent.merge_completed = True
        self.state.active_leaf_ids = [
            leaf_id for leaf_id in self.state.active_leaf_ids if leaf_id not in retained_ids
        ]
        if merged_id not in self.state.active_leaf_ids:
            self.state.active_leaf_ids.append(merged_id)
        return merged_id

    def _posterior_stopping_decision(self, mean: float, std: float) -> bool:
        """Evaluate Eq. (9), including its exact zero-variance convention."""

        threshold = self.config.stopping_threshold
        delta = self.config.stopping_delta
        if threshold is None or delta is None:
            return False
        if std <= 0:
            probability = float(mean < threshold)
        else:
            z = torch.tensor((threshold - mean) / std, dtype=self.dtype, device=self.device)
            probability = float(torch.distributions.Normal(0.0, 1.0).cdf(z).item())
        self.state.stopping_probability = probability
        return probability <= delta

    def _prepare_discrete_candidates_impl(
        self,
        *,
        candidate_actions: torch.Tensor,
        candidate_dataset_indices: torch.Tensor,
        round_info: Mapping[str, Any],
    ) -> DiscreteAcquisitionCandidates:
        """Project exact finite rows to the selected node's active coordinates."""

        del round_info
        if self.state.current_leaf_id is None:
            raise AGVSNodeError("No AG-VS leaf is active for candidate projection.")
        node = self.state.nodes[self.state.current_leaf_id]
        return DiscreteAcquisitionCandidates(
            environment_actions=candidate_actions,
            model_inputs=candidate_actions[:, node.active_indices],
            dataset_indices=candidate_dataset_indices,
        )

    def _initial_design(
        self, round_info: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return a deterministic full-coordinate initial-design batch."""

        if "candidate_actions" in round_info:
            candidates = torch.as_tensor(
                round_info["candidate_actions"], dtype=self.dtype, device=self.device
            )
            indices = torch.as_tensor(
                round_info["candidate_dataset_indices"], dtype=torch.long, device=self.device
            )
            chosen: list[int] = []
            reference = self.state.global_actions
            for _ in range(self.request_size):
                available = torch.ones(candidates.shape[0], dtype=torch.bool, device=self.device)
                if chosen:
                    available[torch.tensor(chosen, dtype=torch.long, device=self.device)] = False
                pool = torch.nonzero(available, as_tuple=False).reshape(-1)
                if reference.shape[0] or chosen:
                    prior = reference
                    if chosen:
                        prior = torch.cat((prior, candidates[chosen]), dim=0)
                    distances = torch.cdist(candidates[pool], prior).min(dim=1).values
                    best_distance = distances.max()
                    tied = pool[distances == best_distance]
                else:
                    tied = pool
                tied_indices = indices[tied]
                chosen.append(int(tied[torch.argmin(tied_indices)].item()))
            positions = torch.tensor(chosen, dtype=torch.long, device=self.device)
            return candidates[positions], indices[positions]
        lower, upper = self._bounds_on_algorithm_device()
        return (
            self._sample_design(
                Bounds(lower, upper),
                self.request_size,
                phase="initial",
                seed=self.config.seed + self.iteration,
            ),
            None,
        )

    def _select_active_points(
        self, node: AGVSNode, round_info: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float | None]:
        """Optimize the configured acquisition in ``A_N`` for one batch."""

        self._set_anchor_from_current_best(node)
        if "candidate_actions" in round_info:
            actions = torch.as_tensor(
                round_info["candidate_actions"], dtype=self.dtype, device=self.device
            )
            indices = torch.as_tensor(
                round_info["candidate_dataset_indices"], dtype=torch.long, device=self.device
            )
            table = self.prepare_discrete_candidates(
                candidate_actions=actions,
                candidate_dataset_indices=indices,
                round_info=round_info,
            )
            if node.model is None:
                order = torch.argsort(table.dataset_indices, stable=True)[: self.request_size]
                return (
                    table.environment_actions[order],
                    table.model_inputs[order],
                    table.dataset_indices[order],
                    None,
                )
            acquisition = build_acquisition(
                model=node.model,
                config=self.config.acquisition,
                batch_size=self.batch_size,
                q_batch=self.q_batch,
                best_f=float(node.local_y.max().item()),
            )
            try:
                result = optimize_acquisition_discrete(
                    acquisition=acquisition,
                    candidates=table,
                    batch_size=self.batch_size,
                    q_batch=self.q_batch,
                )
            except Exception as error:
                raise AGVSDataCompatibilityError(
                    f"AG-VS finite acquisition failed: {error}"
                ) from error
            return (
                result.environment_actions,
                result.model_inputs,
                result.dataset_indices,
                float(result.acquisition_values.sum().item()),
            )

        lower, upper = self._bounds_on_algorithm_device()
        active_bounds = torch.stack((lower[node.active_indices], upper[node.active_indices]))
        if node.model is None:
            active = self._sample_design(
                Bounds(active_bounds[0], active_bounds[1]),
                self.request_size,
                phase="acquisition",
                seed=self.config.seed + 982451653 * (self.iteration + 1),
                paper_method="uniform",
            )
            value = None
        else:
            acquisition = build_acquisition(
                model=node.model,
                config=self.config.acquisition,
                batch_size=self.batch_size,
                q_batch=self.q_batch,
                best_f=float(node.local_y.max().item()),
            )
            active, acquisition_value = optimize_acquisition_continuous(
                acquisition=acquisition,
                bounds=active_bounds,
                batch_size=self.batch_size,
                q_batch=self.q_batch,
                num_restarts=self.config.acquisition_restarts,
                raw_samples=self.config.acquisition_raw_samples,
                sampling_method=self.acquisition_sampling_method,
                sampling_generator=self.generator,
                sampling_seed=self.config.seed + self.iteration,
            )
            value = float(acquisition_value.reshape(-1).sum().item())
        full = self._fill_active_coordinates(node, active)
        return full, active, None, value

    def _start_visit(self) -> AGVSNode:
        """Traverse once and freeze the selected leaf for ``t_seq`` batches."""

        leaf, path = self._traverse()
        self.state.current_leaf_id = leaf.node_id
        self.state.current_path = path
        self.state.remaining_sequence_steps = self.config.sequence_length
        _, self.state.visit_start_best = self._current_best_benchmark()
        self.state.visit_evaluations = 0
        self._fit_node_gp(leaf)
        return leaf

    def _ask_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Return a manuscript batch in the current visit's coordinate subset."""

        if self.state.stopped:
            raise AGVSNodeError("AG-VS posterior stopping condition has already fired.")
        if self.state.global_actions.shape[0] < self.config.initial_points:
            actions, indices = self._initial_design(round_info)
            root = self.state.nodes[self.state.root_id]
            self.model_input_dimension = self.action_dimension
            self._record_last_ask(
                actions=actions,
                model_inputs=actions,
                round_info=round_info,
                algorithm_metadata={
                    "node_id": root.node_id,
                    "active_indices": root.active_indices,
                    "dataset_indices": indices,
                    "initial_design": True,
                },
            )
            return actions[0] if self.request_size == 1 else actions

        node = (
            self._start_visit()
            if self.state.current_leaf_id is None
            else self.state.nodes[self.state.current_leaf_id]
        )
        _, pre_batch_reference = self._current_best_benchmark()
        actions, active, indices, value = self._select_active_points(node, round_info)
        self.model_input_dimension = int(node.active_indices.numel())
        self._last_acquisition_value = value
        self._record_last_ask(
            actions=actions,
            model_inputs=active,
            round_info=round_info,
            algorithm_metadata={
                "node_id": node.node_id,
                "active_indices": node.active_indices,
                "dataset_indices": indices,
                "initial_design": False,
                "pre_batch_reference": pre_batch_reference,
                "sequence_step": self.config.sequence_length - self.state.remaining_sequence_steps,
            },
        )
        return actions[0] if self.request_size == 1 else actions

    def _append_global_posterior(self, means: torch.Tensor, stds: torch.Tensor) -> None:
        """Append immutable fitted posterior records for one observed batch."""

        self.state.global_posterior_means = torch.cat(
            (self.state.global_posterior_means, means.reshape(-1)), dim=0
        )
        self.state.global_posterior_stds = torch.cat(
            (self.state.global_posterior_stds, stds.reshape(-1)), dim=0
        )

    def _unique_pair_rows(self) -> list[int]:
        """Keep the earliest index for each exact ``(x, y)`` pair."""

        unique: list[int] = []
        for index in range(self.state.global_actions.shape[0]):
            duplicate = any(
                torch.equal(self.state.global_actions[index], self.state.global_actions[earlier])
                and bool(
                    self.state.global_observations[index] == self.state.global_observations[earlier]
                )
                for earlier in unique
            )
            if not duplicate:
                unique.append(index)
        return unique

    def _reinitialize(self) -> int:
        """Apply Algorithm 5 without discarding raw or posterior history."""

        root_id = self.state.next_node_id
        self.state.next_node_id += 1
        root = self._empty_node(
            node_id=root_id,
            depth=0,
            parent_id=None,
            active_indices=torch.arange(
                self.action_dimension, dtype=torch.long, device=self.device
            ),
            training_rows=self._unique_pair_rows(),
            newly_retained=True,
        )
        self.state.nodes = {root_id: root}
        self.state.root_id = root_id
        self.state.active_leaf_ids = [root_id]
        self.state.current_leaf_id = None
        self.state.current_path = []
        self.state.remaining_sequence_steps = 0
        self.state.visit_start_best = None
        self.state.visit_evaluations = 0
        self.state.bad_visits = 0
        self.state.tree_epoch += 1
        self.state.restart_count += 1
        self._fit_node_gp(root)
        return root_id

    def _complete_visit(self, node: AGVSNode) -> None:
        """Update scores, stagnation, and the split/merge transition."""

        node.cohort_evaluated = True
        self._update_scores_bottom_up(self.state.current_path)
        _, best_after = self._current_best_benchmark()
        improved = (
            best_after is not None
            and self.state.visit_start_best is not None
            and best_after > self.state.visit_start_best
        )
        self.state.bad_visits = 0 if improved else self.state.bad_visits + 1
        patience = self.config.stagnation_patience
        if patience is not None and self.state.bad_visits >= patience:
            self._reinitialize()
            return

        if node.newly_retained:
            node.newly_retained = False
            if node.active_indices.numel() > self.config.effective_dimension:
                self._split(node)
            return
        if node.parent_id is not None:
            parent = self.state.nodes[node.parent_id]
            if self._merge(parent) is not None:
                return
        if node.excluded and node.active_indices.numel() > 1:
            node.merge_disabled = True
            self._split(node)

    def _tell_impl(
        self,
        *,
        action: torch.Tensor,
        feedback: Mapping[str, Any],
        round_info: Mapping[str, Any],
    ) -> None:
        """Append one observed batch and advance only complete visits structurally."""

        del action, round_info
        assert self.last_ask is not None
        metadata = self.last_ask.algorithm_metadata
        node = self.state.nodes[int(metadata["node_id"])]
        observations = torch.as_tensor(
            feedback["observation"], dtype=self.dtype, device=self.device
        ).reshape(-1)
        evaluated = torch.as_tensor(
            feedback["evaluated_action"], dtype=self.dtype, device=self.device
        )
        row_start = self.state.global_actions.shape[0]
        new_rows = list(range(row_start, row_start + evaluated.shape[0]))
        self.state.global_actions = torch.cat((self.state.global_actions, evaluated), dim=0)
        self.state.global_observations = torch.cat(
            (self.state.global_observations, observations), dim=0
        )
        node.training_rows.extend(new_rows)
        self._fit_node_gp(node)
        means, stds = self._predict_node(node, evaluated)
        self._append_global_posterior(means, stds)
        self.state.batch_count += 1

        if bool(metadata.get("initial_design", False)):
            self._append_current_benchmark()
            return

        reference = metadata.get("pre_batch_reference")
        if reference is None or not math.isfinite(float(reference)):
            raise AGVSNodeError("A scored AG-VS batch requires a finite pre-batch reference.")
        node.score_history_rows.extend(new_rows)
        node.posterior_mean_history.extend(float(value) for value in means.tolist())
        node.posterior_std_history.extend(float(value) for value in stds.tolist())
        node.previous_benchmark_history.extend([float(reference)] * len(new_rows))
        self._refresh_leaf_score(node)
        self._append_current_benchmark()
        self.state.visit_evaluations += len(new_rows)

        for position, (mean, std) in enumerate(zip(means.tolist(), stds.tolist(), strict=True)):
            if self._posterior_stopping_decision(float(mean), float(std)):
                self.state.stopped = True
                self.state.stopping_action = evaluated[position].clone()
                self.state.stopping_mean = float(mean)
                break

        # Algorithm 6 returns immediately when Eq. (9) succeeds. In
        # particular, it does not perform a visit-level tree or stagnation
        # transition even when this happened in the visit's final batch.
        if self.state.stopped:
            return
        self.state.remaining_sequence_steps -= 1
        if self.state.remaining_sequence_steps <= 0:
            self._complete_visit(node)
            self.state.current_leaf_id = None
            self.state.current_path = []
            self.state.remaining_sequence_steps = 0
            self.state.visit_start_best = None
            self.state.visit_evaluations = 0

    def _recommendation_impl(self) -> tuple[torch.Tensor | None, float | None]:
        """Return the stopping point or current best stored-posterior point."""

        if self.state.stopping_action is not None and self.state.stopping_mean is not None:
            value = self.state.stopping_mean if self.maximize else -self.state.stopping_mean
            return self.state.stopping_action.clone(), value
        action, mean = self._current_best_benchmark()
        if mean is None:
            return None, None
        return action, mean if self.maximize else -mean

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Report tree, visit, score, batching, restart, and stopping state."""

        return {
            "node_count": len(self.state.nodes),
            "active_leaf_ids": tuple(self.state.active_leaf_ids),
            "root_id": self.state.root_id,
            "tree_epoch": self.state.tree_epoch,
            "restart_count": self.state.restart_count,
            "bad_visits": self.state.bad_visits,
            "current_leaf_id": self.state.current_leaf_id,
            "remaining_sequence_steps": self.state.remaining_sequence_steps,
            "visit_evaluations": self.state.visit_evaluations,
            "batch_mode": self.batch_mode_name,
            "batch_count": self.state.batch_count,
            "stopped": self.state.stopped,
            "stopping_probability": self.state.stopping_probability,
            "effective_dimension": self.config.effective_dimension,
            "score_formula": "mean_abs_current_posterior_minus_pre_batch_best",
            "score_normalized": False,
            "gp_kernel": "ScaleKernel(MaternKernel(nu=2.5, ARD=False))",
            "use_ard": False,
            "split_source": "seeded_random_permutation",
        }

    @staticmethod
    def _node_state(node: AGVSNode) -> dict[str, Any]:
        """Serialize one node's observations and tree progress."""

        return {
            "node_id": node.node_id,
            "depth": node.depth,
            "parent_id": node.parent_id,
            "child_ids": tuple(node.child_ids),
            "active_indices": node.active_indices,
            "anchor": node.anchor,
            "training_rows": tuple(node.training_rows),
            "local_X": node.local_X,
            "local_y": node.local_y,
            "score_history_rows": tuple(node.score_history_rows),
            "posterior_mean_history": tuple(node.posterior_mean_history),
            "posterior_std_history": tuple(node.posterior_std_history),
            "previous_benchmark_history": tuple(node.previous_benchmark_history),
            "score": node.score,
            "score_mass": node.score_mass,
            "pre_split_score": node.pre_split_score,
            "pre_split_mass": node.pre_split_mass,
            "excluded": node.excluded,
            "merge_disabled": node.merge_disabled,
            "cohort_evaluated": node.cohort_evaluated,
            "newly_retained": node.newly_retained,
            "archived": node.archived,
            "merge_completed": node.merge_completed,
        }

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize the tree and every raw, posterior, visit, and restart record."""

        return {
            "config": asdict(self.config),
            "nodes": {key: self._node_state(node) for key, node in self.state.nodes.items()},
            "active_leaf_ids": tuple(self.state.active_leaf_ids),
            "root_id": self.state.root_id,
            "global_actions": self.state.global_actions,
            "global_observations": self.state.global_observations,
            "global_posterior_means": self.state.global_posterior_means,
            "global_posterior_stds": self.state.global_posterior_stds,
            "benchmark_actions": self.state.benchmark_actions,
            "benchmark_means": self.state.benchmark_means,
            "current_leaf_id": self.state.current_leaf_id,
            "current_path": tuple(self.state.current_path),
            "remaining_sequence_steps": self.state.remaining_sequence_steps,
            "visit_start_best": self.state.visit_start_best,
            "visit_evaluations": self.state.visit_evaluations,
            "next_node_id": self.state.next_node_id,
            "batch_count": self.state.batch_count,
            "bad_visits": self.state.bad_visits,
            "tree_epoch": self.state.tree_epoch,
            "restart_count": self.state.restart_count,
            "stopped": self.state.stopped,
            "stopping_probability": self.state.stopping_probability,
            "stopping_action": self.state.stopping_action,
            "stopping_mean": self.state.stopping_mean,
        }

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Validate checkpoint configuration and all global row alignment."""

        required = {
            "config",
            "nodes",
            "active_leaf_ids",
            "root_id",
            "global_actions",
            "global_observations",
            "global_posterior_means",
            "global_posterior_stds",
            "benchmark_actions",
            "benchmark_means",
            "current_leaf_id",
            "current_path",
            "remaining_sequence_steps",
            "visit_start_best",
            "visit_evaluations",
            "next_node_id",
            "batch_count",
            "bad_visits",
            "tree_epoch",
            "restart_count",
            "stopped",
            "stopping_probability",
            "stopping_action",
            "stopping_mean",
        }
        missing = required - set(state)
        if missing:
            raise AGVSCheckpointError(f"AG-VS state is missing: {sorted(missing)}.")
        del strict
        actions = torch.as_tensor(state["global_actions"])
        observations = torch.as_tensor(state["global_observations"])
        means = torch.as_tensor(state["global_posterior_means"])
        stds = torch.as_tensor(state["global_posterior_stds"])
        if actions.ndim != 2 or actions.shape[1] != self.action_dimension:
            raise AGVSCheckpointError("Stored AG-VS actions have an invalid shape.")
        count = actions.shape[0]
        if any(value.ndim != 1 or value.shape[0] != count for value in (observations, means, stds)):
            raise AGVSCheckpointError("Stored AG-VS global records are not row-aligned.")
        benchmark_actions = torch.as_tensor(state["benchmark_actions"])
        benchmark_means = torch.as_tensor(state["benchmark_means"])
        if (
            benchmark_actions.ndim != 2
            or benchmark_actions.shape[1] != self.action_dimension
            or benchmark_means.ndim != 1
            or benchmark_means.shape[0] != benchmark_actions.shape[0]
        ):
            raise AGVSCheckpointError("Stored AG-VS benchmark records are not aligned.")
        nodes = state["nodes"]
        if not isinstance(nodes, Mapping):
            raise AGVSCheckpointError("Stored AG-VS nodes must be a mapping.")
        node_ids = {int(key) for key in nodes}
        if int(state["root_id"]) not in node_ids:
            raise AGVSCheckpointError("Stored AG-VS tree has no valid root.")

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Restore validated nodes, local GPs, global records, and visit state."""

        del strict
        nodes: dict[int, AGVSNode] = {}
        for key, value in state["nodes"].items():
            active = torch.as_tensor(value["active_indices"], dtype=torch.long, device=self.device)
            node = self._empty_node(
                node_id=int(value["node_id"]),
                depth=int(value["depth"]),
                parent_id=value["parent_id"],
                active_indices=active,
                anchor=(
                    None
                    if value["anchor"] is None
                    else torch.as_tensor(value["anchor"], dtype=self.dtype, device=self.device)
                ),
                training_rows=list(value["training_rows"]),
                excluded=bool(value["excluded"]),
                merge_disabled=bool(value["merge_disabled"]),
                newly_retained=bool(value["newly_retained"]),
            )
            node.child_ids = list(value["child_ids"])
            node.local_X = torch.as_tensor(
                value["local_X"], dtype=self.dtype, device=self.device
            ).clone()
            node.local_y = torch.as_tensor(
                value["local_y"], dtype=self.dtype, device=self.device
            ).clone()
            node.score_history_rows = list(value["score_history_rows"])
            node.posterior_mean_history = list(value["posterior_mean_history"])
            node.posterior_std_history = list(value["posterior_std_history"])
            node.previous_benchmark_history = list(value["previous_benchmark_history"])
            node.score = float(value["score"])
            node.score_mass = int(value["score_mass"])
            node.pre_split_score = value["pre_split_score"]
            node.pre_split_mass = int(value["pre_split_mass"])
            node.cohort_evaluated = bool(value["cohort_evaluated"])
            node.archived = bool(value["archived"])
            node.merge_completed = bool(value["merge_completed"])
            nodes[int(key)] = node

        def tensor(name: str) -> torch.Tensor:
            """Restore one floating state tensor on the algorithm device."""

            return torch.as_tensor(state[name], dtype=self.dtype, device=self.device).clone()

        stopping_action = state["stopping_action"]
        self.state = AGVSState(
            nodes=nodes,
            active_leaf_ids=list(state["active_leaf_ids"]),
            root_id=int(state["root_id"]),
            global_actions=tensor("global_actions"),
            global_observations=tensor("global_observations"),
            global_posterior_means=tensor("global_posterior_means"),
            global_posterior_stds=tensor("global_posterior_stds"),
            benchmark_actions=tensor("benchmark_actions"),
            benchmark_means=tensor("benchmark_means"),
            current_leaf_id=state["current_leaf_id"],
            current_path=list(state["current_path"]),
            remaining_sequence_steps=int(state["remaining_sequence_steps"]),
            visit_start_best=state["visit_start_best"],
            visit_evaluations=int(state["visit_evaluations"]),
            next_node_id=int(state["next_node_id"]),
            batch_count=int(state["batch_count"]),
            bad_visits=int(state["bad_visits"]),
            tree_epoch=int(state["tree_epoch"]),
            restart_count=int(state["restart_count"]),
            stopped=bool(state["stopped"]),
            stopping_probability=state["stopping_probability"],
            stopping_action=(
                None
                if stopping_action is None
                else torch.as_tensor(stopping_action, dtype=self.dtype, device=self.device).clone()
            ),
            stopping_mean=state["stopping_mean"],
        )
        if self.state.current_leaf_id is not None and not self.state.stopped:
            self._fit_node_gp(self.state.nodes[self.state.current_leaf_id])

    def _reset_impl(self) -> None:
        """Reset tree and model dimension while retaining immutable configuration."""

        self.state = self._fresh_state()
        self.model_input_dimension = self.action_dimension


__all__ = ["AG_VS", "AGVSConfig", "AGVSNode", "AGVSState"]
