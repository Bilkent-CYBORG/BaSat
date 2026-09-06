"""Monte Carlo Tree Search based Variable Selection (MCTS-VS).

The tree stores coordinate-index subsets while all environment history remains
in the full ``D_env`` action space.  Inner search projects only learner-visible
rows to the selected subset.  In finite-data mode this module scores complete
available rows and never synthesizes an unevaluable inactive-coordinate fill.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import Any, Literal

import torch

from bayopy.models.Acquisitions import DiscreteAcquisitionCandidates
from bayopy.models.GaussianProcess import GaussianProcess, GaussianProcessConfig
from bayopy.models.Kernels import KernelConfig
from bayopy.models.utils import Bounds

from ..algorithm import Algorithm, resolve_batch_configuration
from ..helper import SourceRecord, to_quality
from .Exception.exception_MCTS_VS import (
    MCTS_VSCheckpointError,
    MCTS_VSException,
    MCTS_VSInnerOptimizerError,
    MCTS_VSScoreError,
)


@dataclass(frozen=True, slots=True)
class MCTSVSConfig:
    """Source and BayOPy settings for MCTS-VS.

    ``Nv``, ``Ns``, archive/reset/split constants, and the exploration constant
    use the documented source-regression defaults.  BayOPy transports each
    logical source batch through sequential ``q=1`` ask-tell exchanges.

    Parameters
    ----------
    Nv, Ns:
        Variables selected per rollout and samples evaluated per subset.
    archive_size, bad_count_limit, split_threshold:
        Best-k archive, strict reset counter, and leaf-size split rule.
    exploration_constant:
        Source UCB tree-exploration coefficient.
    inner_optimizer:
        Sequential BO or TuRBO nested optimization method.
    candidate_pool_size, kernel_lengthscale, acquisition_beta, gp_maxiter:
        Nested GP candidate, kernel, acquisition, and fit controls.
    batch_size, seed:
        Sequential request size and algorithm-owned seed.
    """

    Nv: int = 2
    Ns: int = 3
    archive_size: int = 20
    bad_count_limit: int = 5
    split_threshold: int = 3
    exploration_constant: float = 0.1
    inner_optimizer: Literal["bo", "turbo"] = "bo"
    candidate_pool_size: int = 256
    kernel_lengthscale: float = 0.2
    acquisition_beta: float = 2.0
    gp_maxiter: int = 50
    batch_size: int | None = 1
    q_batch: int | None = None
    seed: int = 1


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
            2.0 * math.log(max(parent_visits, 1)) / self.visits
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
    >>> import bayopy
    >>> bounds = bayopy.Bounds(lower=[-1.0] * 6, upper=[1.0] * 6)
    >>> algorithm = bayopy.MCTS_VS(
    ...     bounds=bounds,
    ...     config=bayopy.MCTSVSConfig(Nv=2, Ns=3, inner_optimizer="bo", seed=7),
    ... )
    """

    ACQUISITION_POLICY = "MCTS variable score plus configured inner optimizer"

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
        notes="Torch translation with a synchronous finite-row BayOPy interface.",
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
            plot=plot,
            verbose=verbose,
            verbose_setting=verbose_setting,
            verbose_iter_by_iter=verbose_iter_by_iter,
        )
        self.state = self._fresh_state()
        self.model_input_dimension = self.action_dimension

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
            "bad_count_limit": config.bad_count_limit,
            "split_threshold": config.split_threshold,
            "candidate_pool_size": config.candidate_pool_size,
            "gp_maxiter": config.gp_maxiter,
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

    def _compute_variable_scores(self) -> torch.Tensor:
        """Compute the paper's elementwise information-record score ratio."""

        numerator = torch.zeros(self.action_dimension, dtype=self.dtype, device=self.device)
        denominator = torch.zeros_like(numerator)
        quality = to_quality(self.state.global_observations, maximize=self.maximize)
        for record in self.state.information_records:
            variables = record.variable_indices.to(device=self.device, dtype=torch.long)
            rows = record.evaluated_indices.to(device=self.device, dtype=torch.long)
            if variables.numel() == 0 or rows.numel() == 0:
                continue
            if torch.any(variables < 0) or torch.any(variables >= self.action_dimension):
                raise MCTS_VSScoreError("An information record contains an invalid variable index.")
            if torch.any(rows < 0) or torch.any(rows >= quality.numel()):
                raise MCTS_VSScoreError("An information record contains an invalid history row.")
            numerator[variables] += quality[rows].sum()
            denominator[variables] += rows.numel()
        scores = torch.zeros_like(numerator)
        observed = denominator > 0
        scores[observed] = numerator[observed] / denominator[observed]
        if not torch.isfinite(scores).all():
            raise MCTS_VSScoreError("Variable scores became non-finite.")
        return scores

    def _build_top_k_archive(self) -> torch.Tensor:
        """Return stable history indices of the current best-k archive."""

        if self.state.global_observations.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        quality = to_quality(self.state.global_observations, maximize=self.maximize)
        order = torch.argsort(quality, descending=True, stable=True)
        return order[: min(self.config.archive_size, order.numel())]

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
        """Traverse by UCB with stable node-id tie breaking."""

        node = self.state.nodes[self.state.root_id]
        path = [node.node_id]
        while node.child_ids:
            candidates = [self.state.nodes[child_id] for child_id in node.child_ids]
            scores = [
                child.ucb(node.visits, self.config.exploration_constant) for child in candidates
            ]
            best_score = max(scores)
            node = min(
                (
                    child
                    for child, score in zip(candidates, scores, strict=True)
                    if score == best_score
                ),
                key=lambda child: child.node_id,
            )
            path.append(node.node_id)
        return node, path

    def _rollout_subsets(self, leaf: VariableNode) -> list[torch.Tensor]:
        """Draw a nonempty subset and complement of the leaf variables."""

        variables = leaf.variables
        if variables.numel() <= 1:
            return [variables.detach().clone()]
        size = int(
            torch.randint(
                1,
                variables.numel(),
                (1,),
                generator=self.generator,
                device=self.device,
            ).item()
        )
        order = torch.randperm(variables.numel(), generator=self.generator, device=self.device)
        subset = torch.sort(variables[order[:size]]).values
        complement = torch.sort(variables[order[size:]]).values
        return [subset, complement]

    def _start_rollout_program(self) -> None:
        """Freeze one source Algorithm-1 program of ``2 * Nv * Ns`` asks.

        BayOPy transports the paper's sample batch sequentially because the
        public protocol permits only one outstanding ask.  The ``Ns`` actions
        for each subset are queued before any of their observations are used,
        preserving the source batch's information boundary.
        """

        self._reset_tree_if_required()
        leaf, path = self._traverse()
        subsets: list[torch.Tensor] = []
        for _ in range(self.config.Nv):
            pair = self._rollout_subsets(leaf)
            subsets.extend(item.detach().clone() for item in pair if item.numel())
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
            pool = self._sample_design(
                Bounds(lower[active], upper[active]),
                self.config.candidate_pool_size,
                phase="acquisition",
                seed=self.config.seed + 15485863 * (self.iteration + 1),
                paper_method="uniform",
            )
            scores = self._inner_scores(pool, active)
            order = torch.argsort(scores, descending=True, stable=True)[: self.config.Ns]
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
        self, candidate_inputs: torch.Tensor, active_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit the configured nested GP and return its posterior moments."""

        if self.state.global_actions.shape[0] < 2:
            zeros = torch.zeros(candidate_inputs.shape[0], dtype=self.dtype, device=self.device)
            return zeros, torch.ones_like(zeros)
        model = self._make_inner_optimizer(active_indices)
        try:
            return model.predict(candidate_inputs)
        except Exception as error:
            raise MCTS_VSInnerOptimizerError(
                f"Nested {self.config.inner_optimizer} posterior failed: {error}"
            ) from error

    def _make_inner_optimizer(self, active_indices: torch.Tensor) -> GaussianProcess:
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
                self.state.global_actions[:, active_indices],
                to_quality(self.state.global_observations, maximize=self.maximize),
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
        self, candidate_inputs: torch.Tensor, active_indices: torch.Tensor
    ) -> torch.Tensor:
        """Score candidate rows in the currently selected model subspace.

        ``candidate_inputs`` has shape ``(M, D_model)`` where ``D_model`` is
        the number of active coordinate indices. The BO score is UCB; the TuRBO
        option additionally penalizes normalized distance from the archive best.
        """

        mean, std = self._kernel_posterior(candidate_inputs, active_indices)
        beta = self.config.acquisition_beta
        if self.config.inner_optimizer == "turbo" and self.state.top_k_indices.numel():
            best = self.state.global_actions[self.state.top_k_indices[0], active_indices]
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
        local_scores = self.state.variable_scores[leaf.variables]
        threshold = local_scores.mean()
        good = leaf.variables[local_scores > threshold]
        bad = leaf.variables[local_scores <= threshold]
        if good.numel() == 0 or bad.numel() == 0:
            return False
        for variables in (torch.sort(good).values, torch.sort(bad).values):
            node_id = self.state.next_node_id
            self.state.next_node_id += 1
            self.state.nodes[node_id] = VariableNode(
                node_id=node_id,
                variables=variables,
                parent_id=leaf.node_id,
                depth=leaf.depth + 1,
            )
            leaf.child_ids.append(node_id)
        return True

    def _backpropagate(self, path: list[int]) -> None:
        """Update visits and mean variable-score values along a selected path."""

        for node_id in reversed(path):
            node = self.state.nodes[node_id]
            node.visits += 1
            node.value = float(self.state.variable_scores[node.variables].mean().item())

    def _reset_tree_if_required(self) -> None:
        """Rebuild only tree state after too many consecutive bad branches."""

        if self.state.bad_branch_count <= self.config.bad_count_limit:
            return
        variables = torch.arange(self.action_dimension, dtype=torch.long, device=self.device)
        self.state.nodes = {0: VariableNode(0, variables, None)}
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
        self.model_input_dimension = int(active.numel())
        if self.state.queued_actions.shape[0] == 0:
            self._queue_subset_actions(active, round_info)
        position = self.state.rollout_sample_cursor
        actions = self.state.queued_actions[position : position + 1]
        selected_dataset_indices = (
            None
            if self.state.queued_dataset_indices is None
            else self.state.queued_dataset_indices[position : position + 1]
        )

        self._record_last_ask(
            actions=actions,
            model_inputs=actions[:, active],
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
        self.state.top_k_indices = self._build_top_k_archive()

        self.state.rollout_sample_cursor += 1
        if self.state.rollout_sample_cursor >= self.state.queued_actions.shape[0]:
            self.state.rollout_subset_cursor += 1
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
            self._split_leaf(leaf)
            self._backpropagate(self.state.selected_path)
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
        }

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize the complete tree, queue, archive, and logical-program state."""

        return {
            "config": asdict(self.config),
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
        nodes = {
            int(key): VariableNode(
                node_id=int(value["node_id"]),
                variables=torch.as_tensor(value["variables"], dtype=torch.long, device=self.device),
                parent_id=value["parent_id"],
                child_ids=list(value["child_ids"]),
                visits=int(value["visits"]),
                value=float(value["value"]),
                depth=int(value["depth"]),
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
        self.model_input_dimension = self.action_dimension


__all__ = [
    "MCTS_VS",
    "InformationRecord",
    "MCTSVSConfig",
    "MCTSVSState",
    "MCTS_VSConfig",
    "VariableNode",
]
