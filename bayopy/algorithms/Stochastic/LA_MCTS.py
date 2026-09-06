"""Learning-Augmented Monte Carlo Tree Search (LA-MCTS).

LA-MCTS learns nonlinear full-dimensional regions by clustering objective
quality and fitting an RBF-SVM separator at eligible tree nodes.  Every child
region is the conjunction of classifier decisions on its root-to-node path.
Finite environments are filtered by that complete path, so proposals always
remain rows that the environment can evaluate.
"""

from __future__ import annotations

import math
import pickle
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch

from bayopy.models.Acquisitions import DiscreteAcquisitionCandidates
from bayopy.models.GaussianProcess import GaussianProcess, GaussianProcessConfig
from bayopy.models.Kernels import KernelConfig
from bayopy.models.utils import Bounds

from ..algorithm import Algorithm, resolve_batch_configuration
from ..helper import SourceRecord, to_quality
from .Exception.exception_LA_MCTS import (
    LA_MCTSCheckpointError,
    LA_MCTSClassifierError,
    LA_MCTSDataCompatibilityError,
    LA_MCTSException,
    LA_MCTSLocalOptimizerError,
    LA_MCTSRegionError,
    LA_MCTSTreeError,
)


@dataclass(frozen=True, slots=True)
class LAMCTSConfig:
    """Tree, classifier, and local-sampler settings for LA-MCTS.

    Parameters
    ----------
    initial_points, exploration_constant:
        Initial design and tree UCT exploration coefficient.
    split_threshold, min_child_size, max_depth:
        Classifier split eligibility and tree-complexity limits.
    classifier, local_sampler:
        Region separator and in-region proposal method.
    candidate_pool_size, max_resampling_attempts:
        Rejection-sampling pool and retry budgets.
    kernel_lengthscale, acquisition_beta, gp_maxiter:
        Local BO kernel, UCB, and GP fit settings.
    batch_size, seed, preset:
        Sequential size, random seed, and explicit named source preset.
    """

    initial_points: int = 30
    exploration_constant: float = 1.0
    split_threshold: int = 20
    min_child_size: int = 5
    max_depth: int = 20
    classifier: Literal["svm_rbf"] = "svm_rbf"
    local_sampler: Literal["turbo", "bo", "cmaes", "random"] = "turbo"
    candidate_pool_size: int = 1024
    max_resampling_attempts: int = 10
    kernel_lengthscale: float = 0.2
    acquisition_beta: float = 2.0
    gp_maxiter: int = 50
    batch_size: int | None = 1
    q_batch: int | None = None
    seed: int = 1
    preset: Literal["paper_general", "repository_generic", "mujoco"] = "paper_general"

    @classmethod
    def from_preset(
        cls, name: Literal["paper_general", "repository_generic", "mujoco"]
    ) -> LAMCTSConfig:
        """Construct an explicit named source or regression preset.

        Parameters
        ----------
        name:
            ``"paper_general"`` for the BayOPy default, or one of the named
            repository/MuJoCo regression configurations.

        Returns
        -------
        LAMCTSConfig
            An immutable configuration with every preset value materialized.

        Raises
        ------
        ValueError
            If ``name`` is not a documented preset.
        """

        if name == "paper_general":
            return cls()
        if name == "repository_generic":
            return cls(
                initial_points=100,
                exploration_constant=0.1,
                split_threshold=10,
                min_child_size=2,
                local_sampler="random",
                candidate_pool_size=20,
                preset=name,
            )
        if name == "mujoco":
            return cls(
                initial_points=30,
                exploration_constant=10.0,
                split_threshold=100,
                min_child_size=10,
                candidate_pool_size=4096,
                preset=name,
            )
        raise ValueError(f"Unknown LA-MCTS preset: {name!r}.")


LA_MCTSConfig = LAMCTSConfig


@dataclass(slots=True)
class RegionClassifier:
    """scikit-learn boundary isolated behind a tensor-safe adapter.

    Attributes
    ----------
    estimator:
        Fitted scikit-learn classifier; persistent algorithm coordinates remain
        Torch tensors outside this adapter.
    """

    estimator: Any

    def predict(self, actions: torch.Tensor) -> torch.Tensor:
        """Predict integer region labels at a third-party NumPy boundary.

        Parameters
        ----------
        actions:
            Full-dimensional ``(N, D_env)`` actions in environment units.

        Returns
        -------
        torch.Tensor
            Integer labels with shape ``(N,)`` on ``actions.device``.
        """

        array = actions.detach().cpu().numpy()
        labels = self.estimator.predict(array)
        return torch.as_tensor(labels, dtype=torch.long, device=actions.device)


@dataclass(slots=True)
class RegionNode:
    """One classifier-defined region and its learner-visible sample rows.

    Attributes
    ----------
    node_id, parent_id, child_ids, depth:
        Stable region-tree identity, adjacency, and depth.
    sample_indices:
        Global learner-visible history rows assigned to this region.
    classifier, good_label, incoming_label:
        Separator and path labels defining exact membership.
    visits, mean_reward:
        UCT traversal statistics.
    unsplittable, exhausted:
        Structural and finite-candidate terminal flags.
    """

    node_id: int
    parent_id: int | None
    child_ids: list[int]
    sample_indices: torch.Tensor
    classifier: RegionClassifier | None
    good_label: int | None
    incoming_label: int | None
    visits: int
    mean_reward: float
    depth: int
    unsplittable: bool = False
    exhausted: bool = False


@dataclass(slots=True)
class LAMCTSState:
    """Complete region tree, history, selection path, and diagnostics.

    Attributes
    ----------
    nodes, root_id, next_node_id:
        Stable learned region tree.
    global_actions, global_observations:
        Complete learner-visible ``D_env`` history.
    selected_leaf_id, selected_path:
        Frozen pending UCT traversal.
    rebuild_count, last_acceptance_rate:
        Classifier rebuild and region-sampling diagnostics.
    finite_exhausted_leaf_ids:
        Leaves proven to contain no available complete finite row.
    """

    nodes: dict[int, RegionNode]
    root_id: int
    next_node_id: int
    global_actions: torch.Tensor
    global_observations: torch.Tensor
    selected_leaf_id: int | None
    selected_path: list[int]
    rebuild_count: int
    last_acceptance_rate: float | None
    finite_exhausted_leaf_ids: set[int]


class LA_MCTS(Algorithm):
    """Classifier-region MCTS with BO, TuRBO, CMA-ES, or random local search.

    Examples
    --------
    >>> import bayopy
    >>> bounds = bayopy.Bounds(lower=[-1.0] * 4, upper=[1.0] * 4)
    >>> algorithm = bayopy.LA_MCTS(
    ...     bounds=bounds,
    ...     config=bayopy.LAMCTSConfig(initial_points=10, local_sampler="turbo", seed=7),
    ... )
    """

    ACQUISITION_POLICY = "MCTS region score plus configured local optimizer"

    EXCEPTION_CLASS = LA_MCTSException
    native_batching = False
    native_environments = frozenset({"stochastic"})
    SOURCE = SourceRecord(
        paper_title=(
            "Learning Search Space Partition for Black-box Optimization using "
            "Monte Carlo Tree Search"
        ),
        paper_year=2020,
        paper_url="https://proceedings.neurips.cc/paper/2020/hash/e2ce14e81dba66dbff9cbc35ecfdb704-Abstract.html",
        repository_url="https://github.com/facebookresearch/LA-MCTS",
        repository_status="official",
        notes="Clean-room Torch/scikit-learn implementation of the paper-general preset.",
    )

    def __init__(
        self,
        *,
        bounds: Bounds,
        config: LAMCTSConfig | None = None,
        maximize: bool = True,
        dtype: torch.dtype | str = torch.float64,
        device: torch.device | str = "cpu",
        plot: bool = False,
        verbose: bool = False,
        verbose_setting: bool = False,
        verbose_iter_by_iter: bool = False,
    ) -> None:
        """Initialize a classifier-region LA-MCTS search.

        Parameters
        ----------
        bounds:
            Full physical action box in ``D_env`` coordinates.
        config:
            Tree, RBF-SVM, local optimizer, preset, and seed settings. ``None``
            selects the paper-general preset; repository-generic and MuJoCo
            behavior must be requested explicitly with ``from_preset``.
        maximize:
            Whether larger learner-visible observations define higher quality.
        dtype, device:
            Tensor precision and device outside third-party NumPy boundaries.
        plot, verbose, verbose_setting, verbose_iter_by_iter:
            Presentation flags that do not alter tree or RNG state.

        Notes
        -----
        scikit-learn is imported only when fitting region classifiers and
        CMA-ES only when that local sampler is selected. Missing capabilities
        raise their actionable ``tree`` or ``optimizers`` extra error.
        """

        self.config = LAMCTSConfig() if config is None else config
        self._validate_config(self.config)
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

    @staticmethod
    def _validate_config(config: LAMCTSConfig) -> None:
        """Validate preset values, tree geometry, and local-optimizer selection."""

        try:
            _, q_batch = resolve_batch_configuration(
                batch_size=config.batch_size, q_batch=config.q_batch
            )
        except ValueError as error:
            raise LA_MCTSLocalOptimizerError(str(error)) from error
        integer_fields = {
            "initial_points": config.initial_points,
            "split_threshold": config.split_threshold,
            "min_child_size": config.min_child_size,
            "max_depth": config.max_depth,
            "candidate_pool_size": config.candidate_pool_size,
            "max_resampling_attempts": config.max_resampling_attempts,
            "gp_maxiter": config.gp_maxiter,
            "seed": config.seed,
        }
        for name, value in integer_fields.items():
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise LA_MCTSTreeError(f"{name} must be an integer >= {minimum}.")
        if q_batch is not None and q_batch > 1:
            raise LA_MCTSLocalOptimizerError(
                "LA-MCTS does not define a joint q-acquisition; use batch_size for repeated "
                "independent leaf-optimizer queries."
            )
        if config.split_threshold < 2 * config.min_child_size:
            raise LA_MCTSTreeError("split_threshold must be at least twice min_child_size.")
        if config.classifier != "svm_rbf":
            raise LA_MCTSClassifierError("classifier must be 'svm_rbf'.")
        if config.local_sampler not in {"turbo", "bo", "cmaes", "random"}:
            raise LA_MCTSLocalOptimizerError("Unsupported local_sampler.")
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
                raise LA_MCTSRegionError(f"{name} has an invalid value.")

    def _fresh_state(self) -> LAMCTSState:
        """Create an empty full-space region tree and replay-critical counters.

        Classifier nodes, selected paths, exhausted finite leaves, and acceptance
        diagnostics are method state; the base checkpoint separately preserves
        the algorithm-owned random generator.
        """

        root = RegionNode(
            node_id=0,
            parent_id=None,
            child_ids=[],
            sample_indices=torch.empty((0,), dtype=torch.long, device=self.device),
            classifier=None,
            good_label=None,
            incoming_label=None,
            visits=0,
            mean_reward=0.0,
            depth=0,
        )
        return LAMCTSState(
            nodes={0: root},
            root_id=0,
            next_node_id=1,
            global_actions=torch.empty(
                (0, self.action_dimension), dtype=self.dtype, device=self.device
            ),
            global_observations=torch.empty((0,), dtype=self.dtype, device=self.device),
            selected_leaf_id=None,
            selected_path=[],
            rebuild_count=0,
            last_acceptance_rate=None,
            finite_exhausted_leaf_ids=set(),
        )

    def _initialize_from_environment_impl(self, data: Any) -> None:
        """Import public full-coordinate history and rebuild classifier regions."""

        self.state.global_actions = torch.as_tensor(
            data.actions, dtype=self.dtype, device=self.device
        ).clone()
        self.state.global_observations = (
            torch.as_tensor(data.observations, dtype=self.dtype, device=self.device)
            .reshape(-1)
            .clone()
        )
        self._build_tree()

    def _quality(self) -> torch.Tensor:
        """Return learner-visible observations in maximization quality convention."""

        return to_quality(self.state.global_observations, maximize=self.maximize)

    def _cluster_quality(self, sample_indices: torch.Tensor) -> torch.Tensor:
        """Cluster the source ``[x, f(x)]`` representation with KMeans."""

        try:
            from sklearn.cluster import KMeans
        except ImportError as error:
            raise LA_MCTSClassifierError(
                "LA-MCTS requires scikit-learn; install 'bayopy[tree]'."
            ) from error
        except Exception as error:
            raise LA_MCTSClassifierError(
                f"The scikit-learn clustering backend could not be initialized: {error}"
            ) from error
        features = torch.cat(
            (
                self.state.global_actions[sample_indices],
                self._quality()[sample_indices].unsqueeze(-1),
            ),
            dim=-1,
        )
        try:
            labels = KMeans(
                n_clusters=2,
                n_init=10,
                random_state=self.config.seed,
            ).fit_predict(features.detach().cpu().numpy())
        except Exception as error:
            raise LA_MCTSClassifierError(f"Quality clustering failed: {error}") from error
        return torch.as_tensor(labels, dtype=torch.long, device=self.device)

    def _fit_region_classifier(
        self, sample_indices: torch.Tensor, labels: torch.Tensor
    ) -> RegionClassifier:
        """Fit the source RBF-SVM boundary at a NumPy-only library boundary."""

        try:
            from sklearn.svm import SVC
        except ImportError as error:
            raise LA_MCTSClassifierError(
                "LA-MCTS requires scikit-learn; install 'bayopy[tree]'."
            ) from error
        except Exception as error:
            raise LA_MCTSClassifierError(
                f"The scikit-learn SVM backend could not be initialized: {error}"
            ) from error
        X = self.state.global_actions[sample_indices].detach().cpu().numpy()
        y = labels.detach().cpu().numpy()
        try:
            estimator = SVC(kernel="rbf", gamma="scale", C=1.0).fit(X, y)
        except Exception as error:
            raise LA_MCTSClassifierError(f"Region SVM fitting failed: {error}") from error
        return RegionClassifier(estimator)

    def _split_node(self, node: RegionNode) -> bool:
        """Split a node only when clustering and SVM preserve two valid children."""

        if (
            node.unsplittable
            or node.child_ids
            or node.depth >= self.config.max_depth
            or node.sample_indices.numel() < self.config.split_threshold
        ):
            return False
        labels = self._cluster_quality(node.sample_indices)
        if torch.unique(labels).numel() != 2:
            node.unsplittable = True
            return False
        classifier = self._fit_region_classifier(node.sample_indices, labels)
        predicted = classifier.predict(self.state.global_actions[node.sample_indices])
        groups = [node.sample_indices[predicted == label] for label in (0, 1)]
        if any(group.numel() < self.config.min_child_size for group in groups):
            node.unsplittable = True
            return False
        quality = self._quality()
        means = [float(quality[group].mean().item()) for group in groups]
        good_label = 0 if means[0] >= means[1] else 1
        node.classifier = classifier
        node.good_label = good_label
        for label, group in enumerate(groups):
            child_id = self.state.next_node_id
            self.state.next_node_id += 1
            self.state.nodes[child_id] = RegionNode(
                node_id=child_id,
                parent_id=node.node_id,
                child_ids=[],
                sample_indices=group.clone(),
                classifier=None,
                good_label=None,
                incoming_label=label,
                visits=int(group.numel()),
                mean_reward=float(quality[group].mean().item()),
                depth=node.depth + 1,
            )
            node.child_ids.append(child_id)
        return True

    def _build_tree(self) -> None:
        """Deterministically rebuild all eligible classifier regions from history."""

        rows = torch.arange(
            self.state.global_actions.shape[0], dtype=torch.long, device=self.device
        )
        quality = self._quality()
        root = RegionNode(
            node_id=0,
            parent_id=None,
            child_ids=[],
            sample_indices=rows,
            classifier=None,
            good_label=None,
            incoming_label=None,
            visits=int(rows.numel()),
            mean_reward=0.0 if rows.numel() == 0 else float(quality.mean().item()),
            depth=0,
        )
        self.state.nodes = {0: root}
        self.state.root_id = 0
        self.state.next_node_id = 1
        queue = [0]
        while queue:
            node = self.state.nodes[queue.pop(0)]
            if self._split_node(node):
                queue.extend(node.child_ids)
        self.state.rebuild_count += 1
        self.state.finite_exhausted_leaf_ids.clear()

    def _contains(self, node_id: int, actions: torch.Tensor) -> torch.Tensor:
        """Evaluate every classifier constraint on a root-to-node path."""

        mask = torch.ones(actions.shape[0], dtype=torch.bool, device=self.device)
        current = self.state.nodes[node_id]
        while current.parent_id is not None:
            parent = self.state.nodes[current.parent_id]
            if parent.classifier is None or current.incoming_label is None:
                raise LA_MCTSRegionError("A region path is missing its classifier decision.")
            mask &= parent.classifier.predict(actions) == current.incoming_label
            current = parent
        return mask

    def _uct_score(self, node: RegionNode, parent_visits: int) -> float:
        """Compute UCT for one region, assigning infinity to unvisited nodes."""

        if node.visits == 0:
            return math.inf
        return node.mean_reward + self.config.exploration_constant * math.sqrt(
            math.log(max(parent_visits, 1)) / node.visits
        )

    def _traverse(self) -> tuple[RegionNode, list[int]]:
        """Choose a non-exhausted leaf by stable UCT traversal."""

        node = self.state.nodes[self.state.root_id]
        path = [node.node_id]
        while node.child_ids:
            children = [
                self.state.nodes[index]
                for index in node.child_ids
                if index not in self.state.finite_exhausted_leaf_ids
            ]
            if not children:
                break
            scores = [self._uct_score(child, node.visits) for child in children]
            best = max(scores)
            node = min(
                (child for child, score in zip(children, scores, strict=True) if score == best),
                key=lambda child: child.node_id,
            )
            path.append(node.node_id)
        return node, path

    def _filter_candidates_to_leaf(self, leaf: RegionNode, actions: torch.Tensor) -> torch.Tensor:
        """Mask full finite rows by every classifier on the leaf path."""

        return self._contains(leaf.node_id, actions)

    def _kernel_scores(self, candidates: torch.Tensor) -> torch.Tensor:
        """Score local candidates using the configured full-dimensional GP."""

        if self.state.global_actions.shape[0] < 2:
            return torch.ones(candidates.shape[0], dtype=self.dtype, device=self.device)
        model = self._make_local_sampler()
        lower, upper = self._bounds_on_algorithm_device()
        scale = upper - lower
        try:
            mean, std = model.predict(candidates)
        except Exception as error:
            raise LA_MCTSLocalOptimizerError(
                f"Local {self.config.local_sampler} posterior failed: {error}"
            ) from error
        scores = mean + self.config.acquisition_beta * std
        if self.config.local_sampler == "turbo":
            leaf_rows = self.state.nodes[self.state.selected_leaf_id].sample_indices  # type: ignore[index]
            if leaf_rows.numel():
                local_quality = self._quality()[leaf_rows]
                center = self.state.global_actions[leaf_rows[int(torch.argmax(local_quality))]]
                normalized_distance = torch.linalg.vector_norm(
                    (candidates - center) / scale, dim=-1
                )
                scores -= 0.05 * normalized_distance
        return scores

    def _make_local_sampler(self) -> GaussianProcess:
        """Fit the BO/TuRBO local GP from complete learner-visible history.

        The selected classifier leaf constrains candidate generation and row
        filtering.  The GP remains full-dimensional, matching the source
        separation between region learning and the nested local optimizer.
        """

        model = GaussianProcess(
            GaussianProcessConfig(
                kernel=KernelConfig(name="matern52", ard_num_dims=self.action_dimension),
                dtype=self.dtype,
                device=self.device,
            ),
            input_dimension=self.action_dimension,
        )
        try:
            model.set_data(self.state.global_actions, self._quality())
            if model.model is not None:
                covariance = model.model.covar_module
                base_kernel = getattr(covariance, "base_kernel", covariance)
                base_kernel.initialize(lengthscale=self.config.kernel_lengthscale)
            model.fit(maxiter=self.config.gp_maxiter)
        except Exception as error:
            raise LA_MCTSLocalOptimizerError(
                f"Local {self.config.local_sampler} GP fitting failed: {error}"
            ) from error
        return model

    def _initial_design(
        self, round_info: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Choose a deterministic finite maximin or continuous Sobol action."""

        if "candidate_actions" in round_info:
            candidates = torch.as_tensor(
                round_info["candidate_actions"], dtype=self.dtype, device=self.device
            )
            indices = torch.as_tensor(
                round_info["candidate_dataset_indices"], dtype=torch.long, device=self.device
            )
            if self.state.global_actions.shape[0]:
                distances = torch.cdist(candidates, self.state.global_actions).min(dim=1).values
                row = int(torch.argmax(distances).item())
            else:
                row = 0
            return candidates[row].unsqueeze(0), indices[row].reshape(1)
        lower, upper = self._bounds_on_algorithm_device()
        return (
            self._sample_design(
                Bounds(lower, upper),
                1,
                phase="initial",
                seed=self.config.seed + self.iteration,
            ),
            None,
        )

    def _sample_continuous_pool(self, leaf: RegionNode) -> torch.Tensor:
        """Generate bounded proposals and retain only complete-path members."""

        lower, upper = self._bounds_on_algorithm_device()
        accepted: list[torch.Tensor] = []
        proposed_count = 0
        if self.config.local_sampler == "cmaes" and self.acquisition_sampling_method == "paper":
            try:
                import cma
            except ImportError as error:
                raise LA_MCTSLocalOptimizerError(
                    "CMA-ES requires the 'bayopy[optimizers]' extra."
                ) from error
            except Exception as error:
                raise LA_MCTSLocalOptimizerError(
                    f"The CMA-ES backend could not be initialized: {error}"
                ) from error
            if leaf.sample_indices.numel():
                quality = self._quality()[leaf.sample_indices]
                center = self.state.global_actions[
                    leaf.sample_indices[int(torch.argmax(quality).item())]
                ]
            else:
                center = (lower + upper) / 2
            options = {
                "bounds": [lower.cpu().tolist(), upper.cpu().tolist()],
                "seed": self.config.seed + self.iteration,
                "verbose": -9,
            }
            try:
                strategy = cma.CMAEvolutionStrategy(
                    center.detach().cpu().tolist(),
                    float((upper - lower).mean().item()) * 0.2,
                    options,
                )
                # CMA-ES returns a Python list of NumPy vectors.  Consolidating
                # it once at this external-library boundary keeps persistent
                # state in Torch without per-element tensor construction.
                raw_array = np.asarray(
                    strategy.ask(self.config.candidate_pool_size), dtype=np.float64
                )
                raw = torch.as_tensor(raw_array, dtype=self.dtype, device=self.device)
            except Exception as error:
                raise LA_MCTSLocalOptimizerError(
                    f"CMA-ES proposal generation failed: {error}"
                ) from error
            proposed_count += raw.shape[0]
            accepted.append(raw[self._contains(leaf.node_id, raw)])
        else:
            for _ in range(self.config.max_resampling_attempts):
                raw = self._sample_design(
                    Bounds(lower, upper),
                    self.config.candidate_pool_size,
                    phase="acquisition",
                    seed=(self.config.seed + self.iteration + 104729 * (proposed_count + 1)),
                    paper_method="uniform",
                )
                proposed_count += raw.shape[0]
                inside = raw[self._contains(leaf.node_id, raw)]
                if inside.numel():
                    accepted.append(inside)
                if sum(item.shape[0] for item in accepted) >= self.config.candidate_pool_size:
                    break
        pool = (
            torch.cat(accepted, dim=0)
            if accepted
            else torch.empty((0, self.action_dimension), dtype=self.dtype, device=self.device)
        )
        self.state.last_acceptance_rate = (
            0.0 if proposed_count == 0 else pool.shape[0] / proposed_count
        )
        if pool.shape[0] == 0:
            # Warm-starting from a known leaf row guarantees classifier membership.
            if leaf.sample_indices.numel():
                pool = self.state.global_actions[leaf.sample_indices].clone()
            else:
                raise LA_MCTSRegionError(f"No feasible proposal was found for leaf {leaf.node_id}.")
        return pool

    def _prepare_discrete_candidates_impl(
        self,
        *,
        candidate_actions: torch.Tensor,
        candidate_dataset_indices: torch.Tensor,
        round_info: Mapping[str, Any],
    ) -> DiscreteAcquisitionCandidates:
        """Preserve complete finite rows in LA-MCTS's full-dimensional model space."""

        del round_info
        return DiscreteAcquisitionCandidates(
            environment_actions=candidate_actions,
            model_inputs=candidate_actions,
            dataset_indices=candidate_dataset_indices,
        )

    def _ask_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Select one full-space action from a learned classifier region.

        Continuous proposals are generated and scored only after satisfying the
        entire root-to-leaf classifier path. Finite mode filters exact public
        rows, marks empty leaves exhausted, and never synthesizes a new row.
        """

        if self.state.global_actions.shape[0] < self.config.initial_points:
            actions, dataset_indices = self._initial_design(round_info)
            leaf_id = self.state.root_id
            path = [self.state.root_id]
            acquisition_value = None
        else:
            if not self.state.nodes:
                self._build_tree()
            candidate_actions = None
            candidate_indices = None
            if "candidate_actions" in round_info:
                candidate_actions = torch.as_tensor(
                    round_info["candidate_actions"], dtype=self.dtype, device=self.device
                )
                candidate_indices = torch.as_tensor(
                    round_info["candidate_dataset_indices"],
                    dtype=torch.long,
                    device=self.device,
                )
            while True:
                leaf, path = self._traverse()
                self.state.selected_leaf_id = leaf.node_id
                self.state.selected_path = path
                if candidate_actions is None:
                    pool = self._sample_continuous_pool(leaf)
                    dataset_indices = None
                    break
                compatible = self._filter_candidates_to_leaf(leaf, candidate_actions)
                if compatible.any():
                    assert candidate_indices is not None
                    table = self.prepare_discrete_candidates(
                        candidate_actions=candidate_actions[compatible],
                        candidate_dataset_indices=candidate_indices[compatible],
                        round_info={
                            **round_info,
                            "candidate_actions": candidate_actions[compatible],
                            "candidate_dataset_indices": candidate_indices[compatible],
                        },
                    )
                    pool = table.environment_actions
                    compatible_indices = table.dataset_indices
                    break
                leaf.exhausted = True
                self.state.finite_exhausted_leaf_ids.add(leaf.node_id)
                leaves = [node for node in self.state.nodes.values() if not node.child_ids]
                if all(node.node_id in self.state.finite_exhausted_leaf_ids for node in leaves):
                    raise LA_MCTSDataCompatibilityError(
                        "No available finite row lies in a selectable LA-MCTS leaf."
                    )
            if self.config.local_sampler == "random":
                selected_row = 0
                scores = torch.zeros(pool.shape[0], dtype=self.dtype, device=self.device)
            else:
                scores = self._kernel_scores(pool)
                selected_row = int(torch.argmax(scores).item())
            actions = pool[selected_row].unsqueeze(0)
            dataset_indices = (
                None if candidate_actions is None else compatible_indices[selected_row].reshape(1)
            )
            leaf_id = leaf.node_id
            acquisition_value = float(scores[selected_row].item())
        self.state.selected_leaf_id = leaf_id
        self.state.selected_path = path
        self._last_acquisition_value = acquisition_value
        self._record_last_ask(
            actions=actions,
            model_inputs=actions,
            round_info=round_info,
            algorithm_metadata={
                "leaf_id": leaf_id,
                "path": tuple(path),
                "dataset_indices": dataset_indices,
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
        """Append one public observation and deterministically rebuild the tree."""

        del round_info
        values = torch.as_tensor(
            feedback["observation"], dtype=self.dtype, device=self.device
        ).reshape(-1)
        self.state.global_actions = torch.cat((self.state.global_actions, action), dim=0)
        self.state.global_observations = torch.cat((self.state.global_observations, values), dim=0)
        self._build_tree()
        self.state.selected_leaf_id = None
        self.state.selected_path = []

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Report tree size, rebuilds, region acceptance, exhaustion, and preset."""

        leaves = [node for node in self.state.nodes.values() if not node.child_ids]
        return {
            "node_count": len(self.state.nodes),
            "leaf_count": len(leaves),
            "rebuild_count": self.state.rebuild_count,
            "last_acceptance_rate": self.state.last_acceptance_rate,
            "finite_exhausted_leaf_ids": tuple(sorted(self.state.finite_exhausted_leaf_ids)),
            "local_sampler": self.config.local_sampler,
            "preset": self.config.preset,
        }

    @staticmethod
    def _node_to_dict(node: RegionNode) -> dict[str, Any]:
        """Serialize a region node and its fitted scikit-learn classifier.

        The estimator is pickled only for BayOPy's trusted transactional
        checkpoint path; tensor computation remains outside this NumPy-backed
        classifier boundary.
        """

        return {
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "child_ids": tuple(node.child_ids),
            "sample_indices": node.sample_indices,
            "classifier": (
                None if node.classifier is None else pickle.dumps(node.classifier.estimator)
            ),
            "good_label": node.good_label,
            "incoming_label": node.incoming_label,
            "visits": node.visits,
            "mean_reward": node.mean_reward,
            "depth": node.depth,
            "unsplittable": node.unsplittable,
            "exhausted": node.exhausted,
        }

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize classifier tree, full history, selection, and exhaustion state."""

        return {
            "config": asdict(self.config),
            "nodes": {key: self._node_to_dict(node) for key, node in self.state.nodes.items()},
            "root_id": self.state.root_id,
            "next_node_id": self.state.next_node_id,
            "global_actions": self.state.global_actions,
            "global_observations": self.state.global_observations,
            "selected_leaf_id": self.state.selected_leaf_id,
            "selected_path": tuple(self.state.selected_path),
            "rebuild_count": self.state.rebuild_count,
            "last_acceptance_rate": self.state.last_acceptance_rate,
            "finite_exhausted_leaf_ids": tuple(sorted(self.state.finite_exhausted_leaf_ids)),
        }

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Validate checkpoint configuration and full-history row alignment."""

        required = {
            "config",
            "nodes",
            "root_id",
            "next_node_id",
            "global_actions",
            "global_observations",
            "selected_leaf_id",
            "selected_path",
            "rebuild_count",
            "last_acceptance_rate",
            "finite_exhausted_leaf_ids",
        }
        missing = required - set(state)
        if missing:
            raise LA_MCTSCheckpointError(f"LA-MCTS state is missing: {sorted(missing)}.")
        del strict
        actions = torch.as_tensor(state["global_actions"])
        observations = torch.as_tensor(state["global_observations"])
        if actions.ndim != 2 or actions.shape[1] != self.action_dimension:
            raise LA_MCTSCheckpointError("Stored LA-MCTS actions have an invalid shape.")
        if observations.ndim != 1 or observations.shape[0] != actions.shape[0]:
            raise LA_MCTSCheckpointError("Stored LA-MCTS observations are not row-aligned.")

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Restore validated nodes and deserialize their classifier estimators."""

        del strict
        nodes: dict[int, RegionNode] = {}
        for key, value in state["nodes"].items():
            classifier_blob = value["classifier"]
            classifier = None
            if classifier_blob is not None:
                try:
                    classifier = RegionClassifier(pickle.loads(classifier_blob))
                except Exception as error:
                    raise LA_MCTSCheckpointError(
                        f"Region classifier {key} cannot be restored: {error}"
                    ) from error
            nodes[int(key)] = RegionNode(
                node_id=int(value["node_id"]),
                parent_id=value["parent_id"],
                child_ids=list(value["child_ids"]),
                sample_indices=torch.as_tensor(
                    value["sample_indices"], dtype=torch.long, device=self.device
                ).clone(),
                classifier=classifier,
                good_label=value["good_label"],
                incoming_label=value["incoming_label"],
                visits=int(value["visits"]),
                mean_reward=float(value["mean_reward"]),
                depth=int(value["depth"]),
                unsplittable=bool(value["unsplittable"]),
                exhausted=bool(value["exhausted"]),
            )
        self.state = LAMCTSState(
            nodes=nodes,
            root_id=int(state["root_id"]),
            next_node_id=int(state["next_node_id"]),
            global_actions=torch.as_tensor(
                state["global_actions"], dtype=self.dtype, device=self.device
            ).clone(),
            global_observations=torch.as_tensor(
                state["global_observations"], dtype=self.dtype, device=self.device
            ).clone(),
            selected_leaf_id=state["selected_leaf_id"],
            selected_path=list(state["selected_path"]),
            rebuild_count=int(state["rebuild_count"]),
            last_acceptance_rate=state["last_acceptance_rate"],
            finite_exhausted_leaf_ids=set(state["finite_exhausted_leaf_ids"]),
        )

    def _reset_impl(self) -> None:
        """Reset all learned regions and method-specific diagnostics."""

        self.state = self._fresh_state()


__all__ = [
    "LA_MCTS",
    "LAMCTSConfig",
    "LAMCTSState",
    "LA_MCTSConfig",
    "RegionClassifier",
    "RegionNode",
]
