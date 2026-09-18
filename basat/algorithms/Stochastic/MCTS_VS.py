"""Monte Carlo Tree Search based Variable Selection (MCTS-VS).

The tree stores coordinate-index subsets while all environment history remains
in the full ``D_env`` action space.  Inner search projects only learner-visible
rows to the selected subset.  In finite-data mode this module scores complete
available rows and never synthesizes an unevaluable inactive-coordinate fill.
"""

from __future__ import annotations

import math
import platform
import sys
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import Any, ClassVar, Literal

import numpy as np
import torch

from basat.models.Acquisitions import DiscreteAcquisitionCandidates
from basat.models.GaussianProcess import GaussianProcess, GaussianProcessConfig
from basat.models.Kernels import KernelConfig
from basat.models.utils import Bounds

from ..algorithm import Algorithm, resolve_batch_configuration
from ..helper import SourceRecord, to_quality
from ._tree_numerics import (
    expected_improvement,
    latin_hypercube,
    random_state_dict,
    restore_random_state,
    sobol_points,
    validate_random_states,
)
from .Exception.exception_MCTS_VS import (
    MCTS_VSCheckpointError,
    MCTS_VSException,
    MCTS_VSInnerOptimizerError,
    MCTS_VSScoreError,
    MCTS_VSWarning,
)
from .TuRBO import _component_candidates


def _released_bo_scores(
    actions: torch.Tensor,
    quality: torch.Tensor,
    candidates: torch.Tensor,
    *,
    source_arithmetic: bool = False,
) -> torch.Tensor:
    """Released MCTS-VS pointwise EI, including its posterior-mean incumbent."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5),
        alpha=0.1**2 if source_arithmetic else 0.01,
        normalize_y=False,
    )
    X = actions.detach().cpu().numpy()
    with warnings.catch_warnings(record=True) as reported:
        warnings.simplefilter("always")
        gp.fit(X, quality.detach().cpu().numpy())
    for warning in reported:
        warnings.warn(f"Source BO GP fit: {warning.message}", MCTS_VSWarning, stacklevel=3)
    if source_arithmetic:
        scores = expected_improvement(gp, candidates.detach().cpu().numpy(), X, xi=1e-4)
        return torch.as_tensor(scores, dtype=candidates.dtype, device=candidates.device)
    mean, std = gp.predict(candidates.detach().cpu().numpy(), return_std=True)
    mean = torch.as_tensor(mean, dtype=candidates.dtype, device=candidates.device)
    std = torch.as_tensor(std, dtype=candidates.dtype, device=candidates.device)
    improvement = mean - float(gp.predict(X).max()) - 1e-4
    z = improvement / std.clamp_min(torch.finfo(std.dtype).tiny)
    result = improvement * (0.5 * (1 + torch.erf(z / math.sqrt(2))))
    result += std * torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)
    return torch.where(std > 0, result, torch.zeros_like(result))


@dataclass(frozen=True, slots=True)
class MCTSVSConfig:
    """Source and Basat settings for MCTS-VS.

    ``Nv``, ``Ns``, archive/reset/split constants, and the exploration constant
    use the documented source-regression defaults.  Basat transports each
    logical source batch through sequential ``q=1`` ask-tell exchanges.

    Parameters
    ----------
    Nv, Ns:
        Variables selected per rollout and samples evaluated per subset.
    archive_size, initial_points, bad_count_limit, split_threshold:
        Best-k archive, initial sampling count, strict reset counter, and
        leaf-size split rule.
    initial_points_sampling_method:
        ``"paper"``, ``"sobol"``, or ``"uniform"`` initial-design sampling.
    exploration_constant:
        Source UCB tree-exploration coefficient.
    inner_optimizer:
        Sequential BO or TuRBO nested optimization method.
    source_protocol:
        ``"basat"`` preserves insertion order when deduplicating sampled
        variable subsets. ``"mcts_vs_2022"`` follows the released runner's
        Python-set iteration over full-dimensional Boolean mask tuples for
        both BO and TuRBO. Set ordering depends on the Python runtime/platform;
        source checkpoints record and validate that runtime rather than
        promising cross-version trajectory identity.
        This mode also owns continuing NumPy MT19937 and Torch streams,
        preserves insertion-slot best-k fills, and uses the released SciPy EI.
        Initialization completes ``Nv`` distinct raw masks plus complements,
        deduplicated before ``Ns`` ambient LHS points per mask; therefore its
        actual initial count is determined by those masks, not ``initial_points``.
        CPU execution is required; float64 history matches the released runner.
    candidate_pool_size, kernel_lengthscale, acquisition_beta, gp_maxiter:
        Nested GP candidate, kernel, acquisition, and fit controls.
    batch_size, seed:
        Sequential request size and algorithm-owned seed.
    """

    Nv: int = 2
    Ns: int = 3
    archive_size: int = 20
    initial_points: int = 5
    initial_points_sampling_method: Literal["paper", "sobol", "uniform"] = "paper"
    bad_count_limit: int = 5
    split_threshold: int = 3
    exploration_constant: float = 0.1
    inner_optimizer: Literal["bo", "turbo"] = "bo"
    inner_max_evaluations: int = 50
    candidate_pool_size: int = 256
    kernel_lengthscale: float = 0.2
    acquisition_beta: float = 2.0
    gp_maxiter: int = 50
    batch_size: int | None = 1
    q_batch: int | None = None
    seed: int = 1
    source_protocol: Literal["basat", "mcts_vs_2022"] = "basat"


MCTS_VSConfig = MCTSVSConfig


@dataclass(slots=True)
class InformationRecord:
    """Rows generated while optimizing one coordinate subset.

    Attributes
    ----------
    variable_indices:
        Full-coordinate indices active in the logical rollout.
    evaluated_indices:
        Global history rows generated for that subset.
    """

    variable_indices: torch.Tensor
    evaluated_indices: torch.Tensor


@dataclass(slots=True)
class VariableNode:
    """One variable-subset node in the MCTS tree.

    Attributes
    ----------
    node_id, parent_id, child_ids:
        Stable tree identity and adjacency.
    variables:
        Full-coordinate indices represented by this node.
    visits, value, depth:
        Source traversal count, empirical reward, and tree depth.
    """

    node_id: int
    variables: torch.Tensor
    parent_id: int | None
    child_ids: list[int] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    depth: int = 0
    record_indices: list[int] = field(default_factory=list)

    def ucb(self, parent_visits: int, exploration_constant: float) -> float:
        """Return the source UCB score, with infinity for unvisited nodes.

        Parameters
        ----------
        parent_visits:
            Number of visits to this node's parent.
        exploration_constant:
            Non-negative coefficient multiplying the exploration bonus.

        Returns
        -------
        float
            The source-defined exploitation-plus-exploration score.
        """

        if self.visits == 0:
            return math.inf
        return self.value + 2.0 * exploration_constant * math.sqrt(
            2.0 * math.sqrt(max(parent_visits, 1)) / self.visits
        )


@dataclass(slots=True)
class MCTSVSState:
    """Complete mutable state required to continue an MCTS-VS run.

    Attributes
    ----------
    information_records, nodes, root_id, next_node_id:
        Information archive and stable variable-tree topology.
    global_actions, global_observations:
        Complete ``D_env`` learner-visible history.
    variable_scores, top_k_indices:
        Current variable rewards and stable best-k archive rows.
    selected_path, selected_variables, selected_leaf_id:
        Frozen logical-rollout traversal state.
    rollout_subsets, rollout_subset_cursor, rollout_sample_cursor:
        Generated subsets and exact within-rollout progress cursors.
    queued_actions, queued_dataset_indices:
        Complete proposals waiting within the logical rollout.
    logical_iterations, bad_branch_count, tree_resets:
        Source iteration, reset-trigger, and reset counters.
    """

    information_records: list[InformationRecord]
    nodes: dict[int, VariableNode]
    root_id: int
    next_node_id: int
    global_actions: torch.Tensor
    global_observations: torch.Tensor
    variable_scores: torch.Tensor
    top_k_indices: torch.Tensor
    selected_path: list[int]
    selected_variables: torch.Tensor | None
    selected_leaf_id: int | None
    rollout_subsets: list[torch.Tensor]
    rollout_subset_cursor: int
    rollout_sample_cursor: int
    queued_actions: torch.Tensor
    queued_dataset_indices: torch.Tensor | None
    logical_iterations: int
    bad_branch_count: int
    tree_resets: int


class MCTS_VS(Algorithm):
    """MCTS-guided variable selection with nested full-row-safe optimization.

    Examples
    --------
    >>> import basat
    >>> bounds = basat.Bounds(lower=[-1.0] * 6, upper=[1.0] * 6)
    >>> algorithm = basat.MCTS_VS(
    ...     bounds=bounds,
    ...     config=basat.MCTSVSConfig(Nv=2, Ns=3, inner_optimizer="bo", seed=7),
    ... )
    """

    ACQUISITION_POLICY = "MCTS variable score plus configured inner optimizer"
    CHECKPOINT_ADDED_CONFIG_DEFAULTS: ClassVar[Mapping[str, Any]] = {
        "initial_points": 5,
        "initial_points_sampling_method": "paper",
        "source_protocol": "basat",
    }

    EXCEPTION_CLASS = MCTS_VSException
    native_batching = False
    native_environments = frozenset({"stochastic"})
    SOURCE = SourceRecord(
        paper_title=(
            "Monte Carlo Tree Search based Variable Selection for High "
            "Dimensional Bayesian Optimization"
        ),
        paper_year=2022,
        paper_url="https://papers.nips.cc/paper_files/paper/2022/hash/b6a171867138c80de2a35a6125d6757c-Abstract-Conference.html",
        repository_url="https://github.com/lamda-bbo/MCTS-VS",
        repository_status="official",
        notes="Torch translation with a synchronous finite-row Basat interface.",
    )

    def __init__(
        self,
        *,
        bounds: Bounds,
        config: MCTSVSConfig | None = None,
        maximize: bool = True,
        dtype: torch.dtype | str = torch.float32,
        device: torch.device | str = "cpu",
        plot: bool = False,
        verbose: bool = False,
        verbose_setting: bool = False,
        verbose_iter_by_iter: bool = False,
    ) -> None:
        """Initialize MCTS-VS with source-regression defaults when omitted.

        Parameters
        ----------
        bounds:
            Full physical action box in ``D_env`` coordinates.
        config:
            Tree, rollout, inner-GP, and seed settings. ``None`` selects the
            documented ``Nv=2``, ``Ns=3``, archive-size 20, and seed 1 preset.
        maximize:
            Whether larger learner-visible observations are preferable.
        dtype, device:
            Tensor precision and device for history, tree scores, and queues.
        plot, verbose, verbose_setting, verbose_iter_by_iter:
            Presentation flags that do not affect the seeded decision state.
        """

        self.config = MCTSVSConfig() if config is None else config
        self._validate_config(bounds, self.config)
        super().__init__(
            bounds=bounds,
            batch_size=self.config.batch_size,
            q_batch=self.config.q_batch,
            maximize=maximize,
            dtype=dtype,
            device=device,
            seed=self.config.seed,
            initial_points_sampling_method=self.config.initial_points_sampling_method,
            plot=plot,
            verbose=verbose,
            verbose_setting=verbose_setting,
            verbose_iter_by_iter=verbose_iter_by_iter,
        )
        self.state = self._fresh_state()
        if self.config.source_protocol == "mcts_vs_2022" and self.device.type != "cpu":
            raise MCTS_VSInnerOptimizerError(
                "Released source-protocol replay currently requires CPU execution."
            )
        self._turbo_start: int | None = None
        self._source_rng = np.random.RandomState(self.config.seed)
        self._source_torch_rng = torch.Generator().manual_seed(self.config.seed)
        self._source_initial_done = False
        self._source_initial_program = False
        self.model_input_dimension = self.action_dimension

    @property
    def source_iteration_complete(self) -> bool:
        """Whether the released outer program has reached its budget check."""
        return self.last_ask is None and not self.state.rollout_subsets

    @staticmethod
    def _validate_config(bounds: Bounds, config: MCTSVSConfig) -> None:
        """Reject invalid values before any mutable tree state is created."""

        try:
            _, q_batch = resolve_batch_configuration(
                batch_size=config.batch_size, q_batch=config.q_batch
            )
        except ValueError as error:
            raise MCTS_VSInnerOptimizerError(str(error)) from error
        integer_fields = {
            "Nv": config.Nv,
            "Ns": config.Ns,
            "archive_size": config.archive_size,
            "initial_points": config.initial_points,
            "bad_count_limit": config.bad_count_limit,
            "split_threshold": config.split_threshold,
            "candidate_pool_size": config.candidate_pool_size,
            "gp_maxiter": config.gp_maxiter,
            "inner_max_evaluations": config.inner_max_evaluations,
            "seed": config.seed,
        }
        for name, value in integer_fields.items():
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise MCTS_VSInnerOptimizerError(f"{name} must be an integer >= {minimum}.")
        if q_batch is not None and q_batch > 1:
            raise MCTS_VSInnerOptimizerError(
                "MCTS-VS does not define a joint q-acquisition; use batch_size for repeated "
                "independent logical sampling programs."
            )
        if config.candidate_pool_size < config.Ns:
            raise MCTS_VSInnerOptimizerError("candidate_pool_size must be at least Ns.")
        if bounds.dimension < 1:
            raise MCTS_VSInnerOptimizerError("MCTS-VS requires at least one action coordinate.")
        if config.inner_optimizer not in {"bo", "turbo"}:
            raise MCTS_VSInnerOptimizerError("inner_optimizer must be 'bo' or 'turbo'.")
        if config.source_protocol not in {"basat", "mcts_vs_2022"}:
            raise MCTS_VSInnerOptimizerError("Unknown MCTS-VS source_protocol.")
        for name, float_value, allow_zero in (
            ("exploration_constant", config.exploration_constant, True),
            ("kernel_lengthscale", config.kernel_lengthscale, False),
            ("acquisition_beta", config.acquisition_beta, True),
        ):
            if (
                not math.isfinite(float_value)
                or float_value < 0
                or (not allow_zero and float_value == 0)
            ):
                raise MCTS_VSScoreError(f"{name} has an invalid value.")

    def _fresh_state(self) -> MCTSVSState:
        """Create the root tree and empty full-coordinate rollout state.

        The algorithm-owned generator lives in the base class; all queued
        actions, cursors, scores, and tree counters required to continue its
        seeded stream are represented here and included in checkpoints.
        """

        variables = torch.arange(self.action_dimension, dtype=torch.long, device=self.device)
        root = VariableNode(0, variables, None)
        return MCTSVSState(
            information_records=[],
            nodes={0: root},
            root_id=0,
            next_node_id=1,
            global_actions=torch.empty(
                (0, self.action_dimension), dtype=self.dtype, device=self.device
            ),
            global_observations=torch.empty((0,), dtype=self.dtype, device=self.device),
            variable_scores=torch.zeros(
                self.action_dimension, dtype=self.dtype, device=self.device
            ),
            top_k_indices=torch.empty((0,), dtype=torch.long, device=self.device),
            selected_path=[],
            selected_variables=None,
            selected_leaf_id=None,
            rollout_subsets=[],
            rollout_subset_cursor=0,
            rollout_sample_cursor=0,
            queued_actions=torch.empty(
                (0, self.action_dimension), dtype=self.dtype, device=self.device
            ),
            queued_dataset_indices=None,
            logical_iterations=0,
            bad_branch_count=0,
            tree_resets=0,
        )

    def _initialize_from_environment_impl(self, data: Any) -> None:
        """Import learner-visible history in full ``D_env`` coordinates."""

        actions = torch.as_tensor(data.actions, dtype=self.dtype, device=self.device)
        observations = torch.as_tensor(
            data.observations, dtype=self.dtype, device=self.device
        ).reshape(-1)
        self.state.global_actions = actions.detach().clone()
        self.state.global_observations = observations.detach().clone()
        if actions.shape[0]:
            self.state.information_records.append(
                InformationRecord(
                    variable_indices=torch.arange(
                        self.action_dimension, dtype=torch.long, device=self.device
                    ),
                    evaluated_indices=torch.arange(
                        actions.shape[0], dtype=torch.long, device=self.device
                    ),
                )
            )
        self.state.variable_scores = self._compute_variable_scores()
        self.state.top_k_indices = self._build_top_k_archive()
        if actions.shape[0] and self.config.source_protocol == "mcts_vs_2022":
            self._source_initial_done = True

    def _compute_variable_scores(self, record_indices: list[int] | None = None) -> torch.Tensor:
        """Compute the paper's elementwise information-record score ratio."""

        numerator = torch.zeros(self.action_dimension, dtype=self.dtype, device=self.device)
        denominator = torch.zeros_like(numerator)
        quality = to_quality(self.state.global_observations, maximize=self.maximize)
        # The released Node.get_axis_score sums each distinct mask's best
        # reward and divides by its total associated sample counts.
        grouped: dict[tuple[int, ...], list[torch.Tensor]] = {}
        records = (
            self.state.information_records
            if record_indices is None
            else [self.state.information_records[index] for index in record_indices]
        )
        for record in records:
            key = tuple(int(v) for v in record.variable_indices.tolist())
            grouped.setdefault(key, []).append(record.evaluated_indices)
        for key, indices in grouped.items():
            record = InformationRecord(
                torch.tensor(key, dtype=torch.long, device=self.device), torch.cat(indices)
            )
            variables = record.variable_indices.to(device=self.device, dtype=torch.long)
            rows = record.evaluated_indices.to(device=self.device, dtype=torch.long)
            if variables.numel() == 0 or rows.numel() == 0:
                continue
            if torch.any(variables < 0) or torch.any(variables >= self.action_dimension):
                raise MCTS_VSScoreError("An information record contains an invalid variable index.")
            if torch.any(rows < 0) or torch.any(rows >= quality.numel()):
                raise MCTS_VSScoreError("An information record contains an invalid history row.")
            numerator[variables] += quality[rows].max()
            denominator[variables] += rows.numel()
        scores = torch.zeros_like(numerator)
        observed = denominator > 0
        scores[observed] = numerator[observed] / (denominator[observed] + 1e-6)
        if not torch.isfinite(scores).all():
            raise MCTS_VSScoreError("Variable scores became non-finite.")
        if self.config.source_protocol == "mcts_vs_2022":
            source_sum = np.zeros(self.action_dimension)
            source_count = np.zeros(self.action_dimension)
            values = quality.detach().cpu().numpy()
            for key, indices in grouped.items():
                source_rows = torch.cat(indices).cpu().numpy()
                mask = np.zeros(self.action_dimension)
                mask[list(key)] = 1.0
                source_count += mask * len(source_rows)
                source_sum += values[source_rows].max() * mask
            scores = torch.as_tensor(
                source_sum / (source_count + 1e-6), dtype=self.dtype, device=self.device
            )
        return scores

    def _build_top_k_archive(self) -> torch.Tensor:
        """Build a native sorted archive or initialize source insertion slots.

        The source path is called once on imported/initial data. Subsequent
        feedback updates existing slots incrementally, preserving RNG order.
        """

        if self.state.global_observations.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        quality = to_quality(self.state.global_observations, maximize=self.maximize)
        if self.config.source_protocol == "mcts_vs_2022":
            # This is an insertion-slot archive, not a sorted top-k view. Its
            # order determines the subsequent independent-coordinate draws.
            slots: list[int] = []
            for row in range(quality.numel()):
                self._source_archive_insert(slots, row, quality)
            return torch.tensor(slots, dtype=torch.long, device=self.device)
        order = torch.argsort(quality, descending=True, stable=True)
        return order[: min(self.config.archive_size, order.numel())]

    def _source_archive_insert(self, slots: list[int], row: int, quality: torch.Tensor) -> None:
        if len(slots) < self.config.archive_size:
            slots.append(row)
        else:
            stored = quality[slots].detach().cpu().numpy()
            worst = stored.min()
            if float(quality[row]) > worst:
                position = int(self._source_rng.choice(np.flatnonzero(stored == worst)))
                slots[position] = row

    def _fill_from_top_k(
        self, active_point: torch.Tensor, active_indices: torch.Tensor
    ) -> torch.Tensor:
        """Fill each inactive coordinate from an independently sampled archive row."""

        lower, upper = self._bounds_on_algorithm_device()
        action = (lower + upper) / 2
        action = action.clone()
        action[active_indices] = active_point
        inactive_mask = torch.ones(self.action_dimension, dtype=torch.bool, device=self.device)
        inactive_mask[active_indices] = False
        inactive = torch.arange(self.action_dimension, device=self.device)[inactive_mask]
        if inactive.numel() == 0 or self.state.top_k_indices.numel() == 0:
            return action
        archive = self.state.global_actions[self.state.top_k_indices]
        if self.config.source_protocol == "mcts_vs_2022":
            for coordinate in inactive.tolist():
                action[coordinate] = archive[int(self._source_rng.choice(len(archive))), coordinate]
            return action
        source_rows = torch.randint(
            0,
            archive.shape[0],
            (inactive.numel(),),
            generator=self.generator,
            device=self.device,
        )
        action[inactive] = archive[source_rows, inactive]
        return action

    def _traverse(self) -> tuple[VariableNode, list[int]]:
        """Traverse by source UCB, drawing uniformly among exact ties."""

        node = self.state.nodes[self.state.root_id]
        path = [node.node_id]
        while node.child_ids:
            candidates = [self.state.nodes[child_id] for child_id in node.child_ids]
            scores = [
                child.ucb(node.visits, self.config.exploration_constant) for child in candidates
            ]
            best_score = max(scores)
            tied = [
                child
                for child, score in zip(candidates, scores, strict=True)
                if score == best_score
            ]
            choice = (
                int(self._source_rng.choice(len(tied)))
                if self.config.source_protocol == "mcts_vs_2022"
                else int(
                    torch.randint(
                        len(tied), (1,), generator=self.generator, device=self.device
                    ).item()
                )
            )
            node = tied[choice]
            path.append(node.node_id)
        return node, path

    def _rollout_subsets(self, leaf: VariableNode) -> list[torch.Tensor]:
        """Draw a nonempty subset and complement of the leaf variables."""

        variables = leaf.variables
        if self.config.source_protocol == "mcts_vs_2022":
            mask_array = self._source_rng.uniform(0.0, 1.0, len(variables)) < 0.5
            if mask_array.all() or not mask_array.any():
                coordinate = int(self._source_rng.choice(len(variables)))
                mask_array[coordinate] = not mask_array[coordinate]
            mask = torch.as_tensor(mask_array, device=self.device)
            return [item for item in (variables[mask], variables[~mask]) if item.numel()]
        if variables.numel() <= 1:
            return [variables.detach().clone()]
        mask = torch.rand(variables.numel(), generator=self.generator, device=self.device) < 0.5
        if bool(mask.all()) or not bool(mask.any()):
            # Source repairs a degenerate mask by flipping ONE uniformly
            # sampled coordinate; rejection sampling changes the subset law.
            coordinate = int(
                torch.randint(
                    variables.numel(), (1,), generator=self.generator, device=self.device
                ).item()
            )
            mask[coordinate] = ~mask[coordinate]
        subset = torch.sort(variables[mask]).values
        complement = torch.sort(variables[~mask]).values
        return [subset, complement]

    def _start_rollout_program(self) -> None:
        """Freeze one source Algorithm-1 program of ``2 * Nv * Ns`` asks.

        Basat transports the paper's sample batch sequentially because the
        public protocol permits only one outstanding ask.  The ``Ns`` actions
        for each subset are queued before any of their observations are used,
        preserving the source batch's information boundary.
        """

        if self.config.source_protocol == "mcts_vs_2022" and not self._source_initial_done:
            if self.config.Nv > 2**self.action_dimension:
                raise MCTS_VSInnerOptimizerError("Nv exceeds the number of distinct initial masks.")
            masks: list[tuple[bool, ...]] = []
            while len(masks) < self.config.Nv:
                mask = tuple(self._source_rng.uniform(0, 1, self.action_dimension) < 0.5)
                if mask not in masks:
                    masks.append(mask)
            initial_subsets = [
                torch.tensor(
                    [i for i, bit in enumerate(mask) if bit], dtype=torch.long, device=self.device
                )
                for mask in masks + [tuple(not bit for bit in mask) for mask in masks]
            ]
            self.state.rollout_subsets = self._deduplicate_rollout_subsets(initial_subsets)
            self.state.selected_path = [self.state.root_id]
            self.state.selected_leaf_id = self.state.root_id
            self._source_initial_program = True
            return
        self._reset_tree_if_required()
        leaf, path = self._traverse()
        subsets: list[torch.Tensor] = []
        complements: list[torch.Tensor] = []
        for _ in range(self.config.Nv):
            pair = self._rollout_subsets(leaf)
            subsets.append(pair[0].detach().clone())
            complements.extend(item.detach().clone() for item in pair[1:] if item.numel())
        subsets = self._deduplicate_rollout_subsets(subsets + complements)
        if not subsets:
            raise MCTS_VSInnerOptimizerError(
                "The selected leaf did not produce a nonempty variable subset."
            )

        # Source nbad increases for every right-child edge traversed and is not
        # reset merely because a later selection follows a left branch.
        for parent_id, child_id in pairwise(path):
            parent = self.state.nodes[parent_id]
            if parent.child_ids and child_id == parent.child_ids[-1]:
                self.state.bad_branch_count += 1

        self.state.selected_path = path
        self.state.selected_leaf_id = leaf.node_id
        self.state.rollout_subsets = subsets
        self.state.rollout_subset_cursor = 0
        self.state.rollout_sample_cursor = 0
        self.state.queued_actions = torch.empty(
            (0, self.action_dimension), dtype=self.dtype, device=self.device
        )
        self.state.queued_dataset_indices = None

    def _deduplicate_rollout_subsets(self, subsets: list[torch.Tensor]) -> list[torch.Tensor]:
        """Deduplicate masks with the explicitly selected source/native order."""

        if self.config.source_protocol == "mcts_vs_2022":
            # The released utils.feature_dedup builds a set of FULL Boolean
            # masks, not a set of reduced index tuples. Their hashes/order are
            # different, and sequential subspace runs consume updated archives.
            masks: set[tuple[int, ...]] = set()
            for subset in subsets:
                indices = set(subset.tolist())
                masks.add(tuple(int(index in indices) for index in range(self.action_dimension)))
            return [
                torch.tensor(
                    [index for index, selected in enumerate(mask) if selected],
                    dtype=torch.long,
                    device=self.device,
                )
                for mask in masks
            ]
        unique: dict[tuple[int, ...], torch.Tensor] = {}
        for subset in subsets:
            unique.setdefault(tuple(subset.tolist()), subset)
        return list(unique.values())

    @staticmethod
    def _feature_order_runtime() -> dict[str, Any]:
        """Identify the runtime whose native tuple/set order is being reproduced."""

        return {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": sys.platform,
            "machine": platform.machine(),
            "hash_width": sys.hash_info.width,
        }

    def _queue_subset_actions(self, active: torch.Tensor, round_info: Mapping[str, Any]) -> None:
        """Queue up to ``Ns`` proposals without observing within-subset feedback."""

        selected_dataset_indices: torch.Tensor | None = None
        if "candidate_actions" in round_info:
            candidates = torch.as_tensor(
                round_info["candidate_actions"], dtype=self.dtype, device=self.device
            )
            indices = torch.as_tensor(
                round_info["candidate_dataset_indices"], dtype=torch.long, device=self.device
            )
            table = self.prepare_discrete_candidates(
                candidate_actions=candidates,
                candidate_dataset_indices=indices,
                round_info=round_info,
            )
            scores = self._inner_scores(table.model_inputs, active)
            count = min(self.config.Ns, table.environment_actions.shape[0])
            order = torch.argsort(scores, descending=True, stable=True)[:count]
            actions = table.environment_actions[order]
            selected_dataset_indices = table.dataset_indices[order]
        else:
            lower, upper = self._bounds_on_algorithm_device()
            initial_phase = (
                self._source_initial_program
                if self.config.source_protocol == "mcts_vs_2022"
                else self.state.global_actions.shape[0] < self.config.initial_points
            )
            if initial_phase:
                # Source initialization samples every ambient coordinate,
                # independently for each feature mask; no archive fill applies.
                count = self.config.Ns
                if self.initial_points_sampling_method != "paper":
                    self.state.queued_actions = self._sample_design(
                        Bounds(lower, upper),
                        count,
                        phase="initial",
                        seed=self.config.seed + self.iteration,
                        paper_method="uniform",
                    )
                    self.state.queued_dataset_indices = None
                    self.state.rollout_sample_cursor = 0
                    return
                if self.config.source_protocol == "mcts_vs_2022":
                    unit = torch.as_tensor(
                        latin_hypercube(self._source_rng, count, self.action_dimension),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    self.state.queued_actions = lower + unit * (upper - lower)
                    self.state.queued_dataset_indices = None
                    self.state.rollout_sample_cursor = 0
                    return
                unit = torch.empty(
                    count, self.action_dimension, dtype=self.dtype, device=self.device
                )
                for dimension in range(self.action_dimension):
                    unit[:, dimension] = torch.randperm(
                        count, generator=self.generator, device=self.device
                    )
                unit = (
                    unit
                    + torch.rand(
                        unit.shape, generator=self.generator, dtype=self.dtype, device=self.device
                    )
                ) / count
                actions = lower + unit * (upper - lower)
                self.state.queued_actions = actions
                self.state.queued_dataset_indices = None
                self.state.rollout_sample_cursor = 0
                return
            if self.config.inner_optimizer == "turbo":
                if self._turbo_start is None:
                    self._turbo_start = self.state.global_actions.shape[0]
                pool = _component_candidates(
                    bounds=Bounds(lower[active], upper[active]),
                    actions=self.state.global_actions[:, active],
                    observations=self.state.global_observations,
                    start=self._turbo_start,
                    seed=self.config.seed + self._turbo_start,
                    maximize=self.maximize,
                    training_steps=self.config.gp_maxiter,
                    source_rng=self._source_rng
                    if self.config.source_protocol == "mcts_vs_2022"
                    else None,
                    source_torch_rng=self._source_torch_rng
                    if self.config.source_protocol == "mcts_vs_2022"
                    else None,
                )
                if pool.shape[0] == 0:
                    raise MCTS_VSInnerOptimizerError("TuRBO component exhausted its trust region.")
                actions = lower.expand(len(pool), -1).clone()
                actions[:, active] = pool
                if self.config.source_protocol == "basat":
                    actions = torch.stack([self._fill_from_top_k(point, active) for point in pool])
                self.state.queued_actions = actions
                self.state.queued_dataset_indices = None
                self.state.rollout_sample_cursor = 0
                return
            pool = (
                torch.as_tensor(
                    sobol_points(self._source_rng, self.config.candidate_pool_size, len(active)),
                    dtype=self.dtype,
                    device=self.device,
                )
                * (upper - lower)[active]
                + lower[active]
                if self.config.source_protocol == "mcts_vs_2022"
                and self.acquisition_sampling_method == "paper"
                else self._sample_design(
                    Bounds(lower[active], upper[active]),
                    self.config.candidate_pool_size,
                    phase="initial" if initial_phase else "acquisition",
                    seed=self.config.seed + 15485863 * (self.iteration + 1),
                    paper_method="sobol",
                )
            )
            scores = self._inner_scores(pool, active)
            order = (
                torch.as_tensor(
                    np.argsort(scores.detach().cpu().numpy())[-self.config.Ns :], device=self.device
                )
                if self.config.source_protocol == "mcts_vs_2022"
                else torch.argsort(scores, descending=False, stable=True)[-self.config.Ns :]
            )
            actions = lower.expand(len(order), -1).clone()
            actions[:, active] = pool[order]
            if self.config.source_protocol == "basat":
                actions = torch.stack([self._fill_from_top_k(pool[row], active) for row in order])

        if actions.shape[0] == 0:
            raise MCTS_VSInnerOptimizerError(
                "No action can be queued for the selected variable subset."
            )
        self.state.queued_actions = actions.detach().clone()
        self.state.queued_dataset_indices = (
            None if selected_dataset_indices is None else selected_dataset_indices.detach().clone()
        )
        self.state.rollout_sample_cursor = 0
        self._last_acquisition_value = float(scores[order[0]].item())

    def _kernel_posterior(
        self,
        candidate_inputs: torch.Tensor,
        active_indices: torch.Tensor,
        *,
        history_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit the configured nested GP and return its posterior moments."""

        rows = self.state.global_actions.shape[0] if history_size is None else history_size
        if rows < self.config.initial_points:
            zeros = torch.zeros(candidate_inputs.shape[0], dtype=self.dtype, device=self.device)
            return zeros, torch.ones_like(zeros)
        model = self._make_inner_optimizer(active_indices, history_size=history_size)
        try:
            return model.predict(candidate_inputs)
        except Exception as error:
            raise MCTS_VSInnerOptimizerError(
                f"Nested {self.config.inner_optimizer} posterior failed: {error}"
            ) from error

    def _make_inner_optimizer(
        self, active_indices: torch.Tensor, *, history_size: int | None = None
    ) -> GaussianProcess:
        """Construct and fit the rollout's vanilla-BO/TuRBO surrogate.

        TuRBO uses the same exact GP posterior here; its local trust-region
        decision is applied by :meth:`_inner_scores`.  No nested state spans
        ask calls, so reconstructing this model from checkpointed global data is
        exact and avoids duplicating mutable GP caches.
        """

        dimension = int(active_indices.numel())
        model = GaussianProcess(
            GaussianProcessConfig(
                kernel=KernelConfig(name="matern52", ard_num_dims=dimension),
                dtype=self.dtype,
                device=self.device,
            ),
            input_dimension=dimension,
        )
        try:
            model.set_data(
                self.state.global_actions[:history_size, active_indices],
                to_quality(self.state.global_observations[:history_size], maximize=self.maximize),
            )
            if model.model is not None:
                covariance = model.model.covar_module
                base_kernel = getattr(covariance, "base_kernel", covariance)
                base_kernel.initialize(lengthscale=self.config.kernel_lengthscale)
            model.fit(maxiter=self.config.gp_maxiter)
        except Exception as error:
            raise MCTS_VSInnerOptimizerError(
                f"Nested {self.config.inner_optimizer} GP fitting failed: {error}"
            ) from error
        return model

    def _inner_scores(
        self,
        candidate_inputs: torch.Tensor,
        active_indices: torch.Tensor,
        *,
        history_size: int | None = None,
    ) -> torch.Tensor:
        """Score candidate rows in the currently selected model subspace.

        ``candidate_inputs`` has shape ``(M, D_model)`` where ``D_model`` is
        the number of active coordinate indices. The BO score is UCB; the TuRBO
        option additionally penalizes normalized distance from the archive best.
        """

        rows = self.state.global_actions.shape[0] if history_size is None else history_size
        if (
            rows >= self.config.initial_points or self._source_initial_done
        ) and self.config.inner_optimizer == "bo":
            return _released_bo_scores(
                self.state.global_actions[:rows, active_indices],
                to_quality(self.state.global_observations[:rows], maximize=self.maximize),
                candidate_inputs,
                source_arithmetic=self.config.source_protocol == "mcts_vs_2022",
            )
        mean, std = self._kernel_posterior(
            candidate_inputs, active_indices, history_size=history_size
        )
        beta = self.config.acquisition_beta
        archive = self.state.top_k_indices
        if history_size is not None:
            quality = to_quality(
                self.state.global_observations[:history_size], maximize=self.maximize
            )
            archive = torch.argsort(quality, descending=True, stable=True)
        if self.config.inner_optimizer == "turbo" and archive.numel():
            best = self.state.global_actions[archive[0], active_indices]
            lower, upper = self._bounds_on_algorithm_device()
            distance = torch.linalg.vector_norm(
                (candidate_inputs - best) / (upper - lower)[active_indices], dim=-1
            )
            return mean + beta * std - 0.05 * distance
        return mean + beta * std

    def _split_leaf(self, leaf: VariableNode) -> bool:
        """Split variables above/below the leaf-average score when both are nonempty."""

        if leaf.variables.numel() <= self.config.split_threshold or leaf.child_ids:
            return False
        local_scores = self._compute_variable_scores(leaf.record_indices)[leaf.variables]
        threshold = (
            float(np.mean(local_scores.cpu().numpy()))
            if self.config.source_protocol == "mcts_vs_2022"
            else local_scores.mean()
        )
        good = leaf.variables[local_scores >= threshold]
        bad = leaf.variables[local_scores < threshold]
        if good.numel() == 0 or bad.numel() == 0:
            return False
        if max(good.numel(), bad.numel()) < self.config.split_threshold:
            return False
        for variables in (torch.sort(good).values, torch.sort(bad).values):
            node_id = self.state.next_node_id
            self.state.next_node_id += 1
            self.state.nodes[node_id] = VariableNode(
                node_id=node_id,
                variables=variables,
                parent_id=leaf.node_id,
                depth=leaf.depth + 1,
                record_indices=list(leaf.record_indices),
            )
            leaf.child_ids.append(node_id)
        return True

    def _backpropagate(self, path: list[int]) -> None:
        """Update visits and mean variable-score values along a selected path."""

        for node_id in reversed(path):
            node = self.state.nodes[node_id]
            node.visits += 1
            node.value = float(
                self._compute_variable_scores(node.record_indices)[node.variables].max().item()
            )

    def _reset_tree_if_required(self) -> None:
        """Rebuild only tree state after too many consecutive bad branches."""

        if self.state.bad_branch_count < self.config.bad_count_limit:
            return
        variables = torch.arange(self.action_dimension, dtype=torch.long, device=self.device)
        self.state.nodes = {0: VariableNode(0, variables, None)}
        self.state.nodes[0].record_indices = list(range(len(self.state.information_records)))
        self.state.root_id = 0
        self.state.next_node_id = 1
        self.state.selected_path = []
        self.state.selected_variables = None
        self.state.selected_leaf_id = None
        self.state.bad_branch_count = 0
        self.state.tree_resets += 1

    def _prepare_discrete_candidates_impl(
        self,
        *,
        candidate_actions: torch.Tensor,
        candidate_dataset_indices: torch.Tensor,
        round_info: Mapping[str, Any],
    ) -> DiscreteAcquisitionCandidates:
        """Project complete finite rows onto the active variable coordinates.

        Environment actions and stable dataset indices remain one-to-one with
        the public candidate table; only ``model_inputs`` is reduced.
        """

        del round_info
        variables = self.state.selected_variables
        if variables is None:
            variables = self.state.nodes[self.state.root_id].variables
        return DiscreteAcquisitionCandidates(
            environment_actions=candidate_actions,
            model_inputs=candidate_actions[:, variables],
            dataset_indices=candidate_dataset_indices,
        )

    def _ask_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Return the next sequential action from a frozen logical rollout.

        MCTS-VS source batches are transported as ``q=1`` asks. The active
        variables, tree path, logical cursors, and any finite row index are
        snapshotted so feedback cannot alter earlier members of the queued batch.
        """

        if not self.state.rollout_subsets:
            self._start_rollout_program()
        active = self.state.rollout_subsets[self.state.rollout_subset_cursor]
        self.state.selected_variables = active
        initial_ambient = (
            self.config.source_protocol == "mcts_vs_2022" and self._source_initial_program
        )
        self.model_input_dimension = (
            self.action_dimension if initial_ambient else int(active.numel())
        )
        if self.state.queued_actions.shape[0] == 0:
            self._queue_subset_actions(active, round_info)
        else:
            self._refresh_unavailable_queued_actions(active, round_info)
        position = self.state.rollout_sample_cursor
        actions = self.state.queued_actions[position : position + 1]
        if "candidate_actions" not in round_info and (
            self._source_initial_done
            if self.config.source_protocol == "mcts_vs_2022"
            else self.state.global_actions.shape[0] >= self.config.initial_points
        ):
            actions = self._fill_from_top_k(actions[0, active], active).unsqueeze(0)
        selected_dataset_indices = (
            None
            if self.state.queued_dataset_indices is None
            else self.state.queued_dataset_indices[position : position + 1]
        )

        self._record_last_ask(
            actions=actions,
            model_inputs=actions if initial_ambient else actions[:, active],
            round_info=round_info,
            algorithm_metadata={
                "active_indices": active,
                "selected_path": tuple(self.state.selected_path),
                "selected_leaf_id": self.state.selected_leaf_id,
                "rollout_subset_cursor": self.state.rollout_subset_cursor,
                "rollout_sample_cursor": position,
                "dataset_indices": selected_dataset_indices,
            },
        )
        return actions[0]

    def _refresh_unavailable_queued_actions(
        self, active: torch.Tensor, round_info: Mapping[str, Any]
    ) -> None:
        """Refresh unavailable finite rows without consuming within-batch rewards.

        A contextual table may reveal a different available set between two
        sequential members of a source batch. Replace only the still-pending
        proposals using current public rows and the batch-start history prefix.
        The prefix length is recoverable from the checkpointed sample cursor;
        no additional model cache or hidden environmental data is required.
        """

        if "candidate_actions" not in round_info:
            return
        candidates = torch.as_tensor(
            round_info["candidate_actions"], dtype=self.dtype, device=self.device
        )
        indices = torch.as_tensor(
            round_info["candidate_dataset_indices"], dtype=torch.long, device=self.device
        )
        position = self.state.rollout_sample_cursor
        pending = self.state.queued_actions[position]
        matches = torch.isclose(candidates, pending, rtol=0.0, atol=1e-8).all(dim=1)
        if self.state.queued_dataset_indices is not None:
            matches &= indices == self.state.queued_dataset_indices[position]
        if matches.any():
            return
        table = self.prepare_discrete_candidates(
            candidate_actions=candidates,
            candidate_dataset_indices=indices,
            round_info=round_info,
        )
        history_size = self.state.global_actions.shape[0] - position
        scores = self._inner_scores(table.model_inputs, active, history_size=history_size)
        remaining = self.state.queued_actions.shape[0] - position
        count = min(remaining, table.environment_actions.shape[0])
        if count == 0:
            raise MCTS_VSInnerOptimizerError(
                "No currently available action can complete the variable-subset rollout."
            )
        order = torch.argsort(scores, descending=True, stable=True)[:count]
        self.state.queued_actions = (
            torch.cat(
                (self.state.queued_actions[:position], table.environment_actions[order]), dim=0
            )
            .detach()
            .clone()
        )
        previous_indices = self.state.queued_dataset_indices
        if previous_indices is None:
            raise MCTS_VSInnerOptimizerError("A finite rollout lost its queued dataset indices.")
        self.state.queued_dataset_indices = (
            torch.cat((previous_indices[:position], table.dataset_indices[order]), dim=0)
            .detach()
            .clone()
        )
        self._last_acquisition_value = float(scores[order[0]].item())
        self.invalidate_source_guarantees("mcts_vs.dynamic_candidate_batch")
        warnings.warn(
            "MCTS-VS's available finite rows changed within a logical batch. "
            "Unavailable pending rows were replaced by current candidates using "
            "the frozen batch-start observations; source guarantees no longer apply.",
            MCTS_VSWarning,
            stacklevel=3,
        )

    def _tell_impl(
        self,
        *,
        action: torch.Tensor,
        feedback: Mapping[str, Any],
        round_info: Mapping[str, Any],
    ) -> None:
        """Commit one rollout observation and advance tree-program cursors.

        History stays in full environment coordinates while information records
        identify the reduced coordinates used to generate each row. Tree scores
        are updated only after the frozen logical rollout completes.
        """

        del round_info
        assert self.last_ask is not None
        values = torch.as_tensor(
            feedback["observation"], dtype=self.dtype, device=self.device
        ).reshape(-1)
        start = self.state.global_actions.shape[0]
        self.state.global_actions = torch.cat((self.state.global_actions, action), dim=0)
        self.state.global_observations = torch.cat((self.state.global_observations, values), dim=0)
        rows = torch.arange(start, start + action.shape[0], dtype=torch.long, device=self.device)
        active = torch.as_tensor(
            self.last_ask.algorithm_metadata["active_indices"],
            dtype=torch.long,
            device=self.device,
        )
        if (
            self.state.information_records
            and torch.equal(self.state.information_records[-1].variable_indices, active)
            and self.state.rollout_sample_cursor > 0
        ):
            previous = self.state.information_records[-1]
            previous.evaluated_indices = torch.cat((previous.evaluated_indices, rows))
        else:
            self.state.information_records.append(InformationRecord(active.clone(), rows))
            index = len(self.state.information_records) - 1
            for node_id in self.state.selected_path:
                self.state.nodes[node_id].record_indices.append(index)
        if self.config.source_protocol == "mcts_vs_2022":
            if not self._source_initial_program:
                slots = self.state.top_k_indices.tolist()
                quality = to_quality(self.state.global_observations, maximize=self.maximize)
                for row in rows.tolist():
                    self._source_archive_insert(slots, row, quality)
                self.state.top_k_indices = torch.tensor(slots, dtype=torch.long, device=self.device)
        else:
            self.state.top_k_indices = self._build_top_k_archive()

        self.state.rollout_sample_cursor += 1
        if self.state.rollout_sample_cursor >= self.state.queued_actions.shape[0]:
            continuing_turbo = (
                self._turbo_start is not None
                and self.state.global_actions.shape[0] - self._turbo_start
                < self.config.inner_max_evaluations
            )
            if not continuing_turbo:
                if self._source_initial_done or (
                    self.config.source_protocol == "basat"
                    and self.state.global_actions.shape[0] > self.config.initial_points
                ):
                    self._backpropagate(self.state.selected_path)
                self.state.rollout_subset_cursor += 1
                self._turbo_start = None
            self.state.rollout_sample_cursor = 0
            self.state.queued_actions = torch.empty(
                (0, self.action_dimension), dtype=self.dtype, device=self.device
            )
            self.state.queued_dataset_indices = None

        if self.state.rollout_subset_cursor >= len(self.state.rollout_subsets):
            self.state.variable_scores = self._compute_variable_scores()
            if self.state.selected_leaf_id is None:
                raise MCTS_VSScoreError("The logical rollout lost its selected leaf.")
            leaf = self.state.nodes[self.state.selected_leaf_id]
            if self._source_initial_done or (
                self.config.source_protocol == "basat"
                and self.state.global_actions.shape[0] > self.config.initial_points
            ):
                self._split_leaf(leaf)
            if self._source_initial_program:
                self._source_initial_program = False
                self._source_initial_done = True
                self.state.top_k_indices = self._build_top_k_archive()
            else:
                self.state.logical_iterations += 1
            self.state.rollout_subsets = []
            self.state.rollout_subset_cursor = 0
            self.state.selected_path = []
            self.state.selected_variables = None
            self.state.selected_leaf_id = None

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Return tree size, score, reset, archive, and rollout progress state."""

        return {
            "node_count": len(self.state.nodes),
            "information_record_count": len(self.state.information_records),
            "variable_scores": self.state.variable_scores,
            "top_k_indices": self.state.top_k_indices,
            "bad_branch_count": self.state.bad_branch_count,
            "tree_resets": self.state.tree_resets,
            "logical_iterations": self.state.logical_iterations,
            "logical_rollout_progress": (
                self.state.rollout_subset_cursor,
                self.state.rollout_sample_cursor,
            ),
            "inner_optimizer": self.config.inner_optimizer,
            "source_feature_order_policy": (
                "python-set-full-mask-v1" if self.config.source_protocol == "mcts_vs_2022" else None
            ),
            "source_feature_order_runtime": (
                self._feature_order_runtime()
                if self.config.source_protocol == "mcts_vs_2022"
                else None
            ),
        }

    @staticmethod
    def _node_to_dict(node: VariableNode) -> dict[str, Any]:
        """Serialize one variable-subset node without changing tensor ownership."""

        return {
            "node_id": node.node_id,
            "variables": node.variables,
            "parent_id": node.parent_id,
            "child_ids": tuple(node.child_ids),
            "visits": node.visits,
            "value": node.value,
            "depth": node.depth,
            "record_indices": node.record_indices,
        }

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize the complete tree, queue, archive, and logical-program state."""

        return {
            "config": asdict(self.config),
            "source_numerics_version": 2,
            "source_turbo_numerics_version": (
                3
                if self.config.source_protocol == "mcts_vs_2022"
                and self.config.inner_optimizer == "turbo"
                else None
            ),
            "source_numpy_rng": random_state_dict(self._source_rng),
            "source_torch_rng": self._source_torch_rng.get_state(),
            "source_initial_done": self._source_initial_done,
            "source_initial_program": self._source_initial_program,
            "source_feature_order_policy": (
                "python-set-full-mask-v1" if self.config.source_protocol == "mcts_vs_2022" else None
            ),
            "source_feature_order_runtime": (
                self._feature_order_runtime()
                if self.config.source_protocol == "mcts_vs_2022"
                else None
            ),
            "turbo_start": self._turbo_start,
            "information_records": [
                {
                    "variable_indices": item.variable_indices,
                    "evaluated_indices": item.evaluated_indices,
                }
                for item in self.state.information_records
            ],
            "nodes": {key: self._node_to_dict(node) for key, node in self.state.nodes.items()},
            "root_id": self.state.root_id,
            "next_node_id": self.state.next_node_id,
            "global_actions": self.state.global_actions,
            "global_observations": self.state.global_observations,
            "variable_scores": self.state.variable_scores,
            "top_k_indices": self.state.top_k_indices,
            "selected_path": tuple(self.state.selected_path),
            "selected_variables": self.state.selected_variables,
            "selected_leaf_id": self.state.selected_leaf_id,
            "rollout_subsets": tuple(self.state.rollout_subsets),
            "rollout_subset_cursor": self.state.rollout_subset_cursor,
            "rollout_sample_cursor": self.state.rollout_sample_cursor,
            "queued_actions": self.state.queued_actions,
            "queued_dataset_indices": self.state.queued_dataset_indices,
            "logical_iterations": self.state.logical_iterations,
            "bad_branch_count": self.state.bad_branch_count,
            "tree_resets": self.state.tree_resets,
        }

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Validate checkpoint configuration, shapes, and rollout cursor alignment."""

        required = {
            "config",
            "information_records",
            "nodes",
            "root_id",
            "next_node_id",
            "global_actions",
            "global_observations",
            "variable_scores",
            "top_k_indices",
            "selected_path",
            "selected_variables",
            "selected_leaf_id",
            "rollout_subsets",
            "rollout_subset_cursor",
            "rollout_sample_cursor",
            "queued_actions",
            "queued_dataset_indices",
            "logical_iterations",
            "bad_branch_count",
            "tree_resets",
        }
        missing = required - set(state)
        if missing:
            raise MCTS_VSCheckpointError(f"MCTS-VS state is missing: {sorted(missing)}.")
        if self.config.source_protocol == "mcts_vs_2022":
            if state.get("source_numerics_version") != 2:
                raise MCTS_VSCheckpointError(
                    "Source checkpoint predates MT19937/archive/initialization protocol v2; use a fresh output directory."
                )
            if (
                self.config.inner_optimizer == "turbo"
                and state.get("source_turbo_numerics_version") != 3
            ):
                raise MCTS_VSCheckpointError(
                    "Source TuRBO checkpoint does not match explicit Lanczos sampling protocol v3; "
                    "preserve the old trajectory and use a fresh output directory."
                )
            try:
                validate_random_states(state)
                initial_done = state["source_initial_done"]
                initial_program = state["source_initial_program"]
                if (
                    not isinstance(initial_done, bool)
                    or not isinstance(initial_program, bool)
                    or (initial_done and initial_program)
                ):
                    raise ValueError("Invalid initial-program flags.")
                if initial_program and not state["rollout_subsets"]:
                    raise ValueError("Initial-program flag has no queued feature program.")
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                raise MCTS_VSCheckpointError(f"Invalid source state: {error}") from error
            if state.get("source_feature_order_policy") != "python-set-full-mask-v1":
                raise MCTS_VSCheckpointError(
                    "Checkpoint predates the released full-mask set-order policy. Preserve the "
                    "old trajectory separately and start this source protocol in a new output directory."
                )
            if state.get("source_feature_order_runtime") != self._feature_order_runtime():
                raise MCTS_VSCheckpointError(
                    "Source set-order checkpoint runtime/platform differs. Python set ordering "
                    "is not a cross-version replay guarantee; use the recorded runtime or a fresh run."
                )
        del strict
        actions = torch.as_tensor(state["global_actions"])
        values = torch.as_tensor(state["global_observations"])
        if actions.ndim != 2 or actions.shape[1] != self.action_dimension:
            raise MCTS_VSCheckpointError("Stored global_actions has an invalid shape.")
        if values.ndim != 1 or values.shape[0] != actions.shape[0]:
            raise MCTS_VSCheckpointError("Stored global observations are not row-aligned.")
        queued = torch.as_tensor(state["queued_actions"])
        if queued.ndim != 2 or queued.shape[1] != self.action_dimension:
            raise MCTS_VSCheckpointError("Stored queued_actions has an invalid shape.")
        subsets = list(state["rollout_subsets"])
        subset_cursor = int(state["rollout_subset_cursor"])
        sample_cursor = int(state["rollout_sample_cursor"])
        logical_iterations = int(state["logical_iterations"])
        if not 0 <= subset_cursor <= len(subsets):
            raise MCTS_VSCheckpointError("Stored rollout subset cursor is invalid.")
        if not 0 <= sample_cursor <= queued.shape[0]:
            raise MCTS_VSCheckpointError("Stored rollout sample cursor is invalid.")
        if logical_iterations < 0:
            raise MCTS_VSCheckpointError("Stored logical iteration count is invalid.")
        queued_indices = state["queued_dataset_indices"]
        if queued_indices is not None:
            indices = torch.as_tensor(queued_indices)
            if indices.ndim != 1 or indices.shape[0] != queued.shape[0]:
                raise MCTS_VSCheckpointError("Stored queued dataset indices are not row-aligned.")

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Restore validated tree and rollout tensors on the configured device."""

        del strict
        self._turbo_start = state.get("turbo_start")
        if self.config.source_protocol == "mcts_vs_2022":
            restore_random_state(self._source_rng, state["source_numpy_rng"])
            self._source_torch_rng.set_state(
                torch.as_tensor(state["source_torch_rng"], dtype=torch.uint8, device="cpu")
            )
            self._source_initial_done = bool(state["source_initial_done"])
            self._source_initial_program = bool(state["source_initial_program"])
        nodes = {
            int(key): VariableNode(
                node_id=int(value["node_id"]),
                variables=torch.as_tensor(value["variables"], dtype=torch.long, device=self.device),
                parent_id=value["parent_id"],
                child_ids=list(value["child_ids"]),
                visits=int(value["visits"]),
                value=float(value["value"]),
                depth=int(value["depth"]),
                record_indices=list(
                    value.get("record_indices", range(len(state["information_records"])))
                ),
            )
            for key, value in state["nodes"].items()
        }
        records = [
            InformationRecord(
                torch.as_tensor(item["variable_indices"], dtype=torch.long, device=self.device),
                torch.as_tensor(item["evaluated_indices"], dtype=torch.long, device=self.device),
            )
            for item in state["information_records"]
        ]
        selected = state["selected_variables"]
        queued_indices = state["queued_dataset_indices"]
        self.state = MCTSVSState(
            information_records=records,
            nodes=nodes,
            root_id=int(state["root_id"]),
            next_node_id=int(state["next_node_id"]),
            global_actions=torch.as_tensor(
                state["global_actions"], dtype=self.dtype, device=self.device
            ).clone(),
            global_observations=torch.as_tensor(
                state["global_observations"], dtype=self.dtype, device=self.device
            ).clone(),
            variable_scores=torch.as_tensor(
                state["variable_scores"], dtype=self.dtype, device=self.device
            ).clone(),
            top_k_indices=torch.as_tensor(
                state["top_k_indices"], dtype=torch.long, device=self.device
            ).clone(),
            selected_path=list(state["selected_path"]),
            selected_variables=(
                None
                if selected is None
                else torch.as_tensor(selected, dtype=torch.long, device=self.device).clone()
            ),
            selected_leaf_id=(
                None if state["selected_leaf_id"] is None else int(state["selected_leaf_id"])
            ),
            rollout_subsets=[
                torch.as_tensor(item, dtype=torch.long, device=self.device).clone()
                for item in state["rollout_subsets"]
            ],
            rollout_subset_cursor=int(state["rollout_subset_cursor"]),
            rollout_sample_cursor=int(state["rollout_sample_cursor"]),
            queued_actions=torch.as_tensor(
                state["queued_actions"], dtype=self.dtype, device=self.device
            ).clone(),
            queued_dataset_indices=(
                None
                if queued_indices is None
                else torch.as_tensor(queued_indices, dtype=torch.long, device=self.device).clone()
            ),
            logical_iterations=int(state["logical_iterations"]),
            bad_branch_count=int(state["bad_branch_count"]),
            tree_resets=int(state["tree_resets"]),
        )

    def _reset_impl(self) -> None:
        """Reset the tree and logical rollout while retaining immutable config."""

        self.state = self._fresh_state()
        self._turbo_start = None
        self._source_rng.seed(self.config.seed)
        self._source_torch_rng.manual_seed(self.config.seed)
        self._source_initial_done = False
        self._source_initial_program = False
        self.model_input_dimension = self.action_dimension


__all__ = [
    "MCTS_VS",
    "InformationRecord",
    "MCTSVSConfig",
    "MCTSVSState",
    "MCTS_VSConfig",
    "VariableNode",
]
