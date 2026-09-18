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
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
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
    validate_random_states,
)
from .Exception.exception_LA_MCTS import (
    LA_MCTSCheckpointError,
    LA_MCTSClassifierError,
    LA_MCTSDataCompatibilityError,
    LA_MCTSException,
    LA_MCTSLocalOptimizerError,
    LA_MCTSRegionError,
    LA_MCTSTreeError,
    LA_MCTSWarning,
)
from .TuRBO import _component_candidates


@dataclass(frozen=True, slots=True)
class LAMCTSConfig:
    """Tree, classifier, and local-sampler settings for LA-MCTS.

    Parameters
    ----------
    initial_points, exploration_constant:
        Initial design and tree UCT exploration coefficient.
    initial_points_sampling_method:
        ``"paper"``, ``"sobol"``, or ``"uniform"`` initial-design sampling.
    split_threshold, min_child_size, max_depth:
        Classifier split eligibility and tree-complexity limits.
    classifier, local_sampler:
        Region separator and in-region proposal method.
    candidate_pool_size, max_resampling_attempts:
        Rejection-sampling pool and retry budgets.
    kernel_lengthscale, acquisition_beta, gp_maxiter:
        Local BO kernel, UCB, and GP fit settings.
    batch_size, seed:
        Sequential request size and random seed.
    source_protocol:
        ``"basat"`` retains the native sampler. ``"mcts_vs_2022"`` selects
        the pinned MCTS-VS repository's LA-MCTS runner: float32 clustering
        records, continuing NumPy/SVM randomness, historical KMeans ``n_init=10``,
        good-child-first double-fit tree construction, and either isotropic
        sklearn GP/EI (xi=0.001) over a 10,000-point expanding-box candidate pool,
        or a fresh TuRBO invocation with ten region-initialized points. Only
        initial TuRBO points are classifier constrained. This mode requires CPU;
        use float64 history to match the source. Numerical-library versions can
        still affect fitted boundaries and must be recorded by experiments.
    """

    initial_points: int = 30
    initial_points_sampling_method: Literal["paper", "sobol", "uniform"] = "paper"
    exploration_constant: float = 1.0
    split_threshold: int = 20
    min_child_size: int = 5
    max_depth: int = 20
    classifier: Literal["svm_rbf"] = "svm_rbf"
    classifier_gamma: Literal["auto", "scale"] = "auto"
    local_sampler: Literal["turbo", "bo", "cmaes", "random"] = "turbo"
    inner_max_evaluations: int = 50
    candidate_pool_size: int = 1024
    max_resampling_attempts: int = 10
    kernel_lengthscale: float = 0.2
    acquisition_beta: float = 2.0
    gp_maxiter: int = 50
    batch_size: int | None = 1
    q_batch: int | None = None
    seed: int = 1
    source_protocol: Literal["basat", "mcts_vs_2022"] = "basat"


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
    >>> import basat
    >>> bounds = basat.Bounds(lower=[-1.0] * 4, upper=[1.0] * 4)
    >>> algorithm = basat.LA_MCTS(
    ...     bounds=bounds,
    ...     config=basat.LAMCTSConfig(initial_points=10, local_sampler="turbo", seed=7),
    ... )
    """

    ACQUISITION_POLICY = "MCTS region score plus configured local optimizer"
    CHECKPOINT_ADDED_CONFIG_DEFAULTS: ClassVar[Mapping[str, Any]] = {
        "initial_points_sampling_method": "paper",
        "source_protocol": "basat",
    }

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
        notes="Clean-room Torch/scikit-learn implementation with explicit algorithm settings.",
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
            Tree, RBF-SVM, local optimizer and seed settings. ``None`` uses
            :class:`LAMCTSConfig` defaults. Supply individual configuration
            fields to change the algorithm's settings.
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
            initial_points_sampling_method=self.config.initial_points_sampling_method,
            plot=plot,
            verbose=verbose,
            verbose_setting=verbose_setting,
            verbose_iter_by_iter=verbose_iter_by_iter,
        )
        self.state = self._fresh_state()
        if self.config.source_protocol == "mcts_vs_2022" and self.device.type != "cpu":
            raise LA_MCTSLocalOptimizerError(
                "Released source-protocol replay currently requires CPU execution."
            )
        self._turbo_program: dict[str, Any] | None = None
        self._source_rng = np.random.RandomState(self.config.seed)
        self._source_torch_rng = torch.Generator().manual_seed(self.config.seed)
        self._source_initial_design_cache: torch.Tensor | None = None
        self._source_tree_dirty = True

    @property
    def source_iteration_complete(self) -> bool:
        """Whether a complete local-optimizer invocation has been consumed."""
        return self.last_ask is None and self._turbo_program is None

    @staticmethod
    def _validate_config(config: LAMCTSConfig) -> None:
        """Validate numerical settings, tree geometry and local-optimizer selection."""

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
            "inner_max_evaluations": config.inner_max_evaluations,
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
        if config.source_protocol not in {"basat", "mcts_vs_2022"}:
            raise LA_MCTSLocalOptimizerError("Unknown LA-MCTS source_protocol.")
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
        if self.config.source_protocol == "mcts_vs_2022":
            self._source_tree_dirty = True
        else:
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
                "LA-MCTS requires scikit-learn; install 'basat[tree]'."
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
        source = self.config.source_protocol == "mcts_vs_2022"
        if source:
            features = features.float()
        try:
            cluster = KMeans(
                n_clusters=2,
                n_init=10,
                random_state=self._source_rng if source else self.config.seed,
            ).fit(features.detach().cpu().numpy())
            labels = cluster.predict(features.detach().cpu().numpy()) if source else cluster.labels_
            if source and np.mean(features[:, -1].cpu().numpy()[labels == 1]) > np.mean(
                features[:, -1].cpu().numpy()[labels == 0]
            ):
                labels = 1 - labels
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
                "LA-MCTS requires scikit-learn; install 'basat[tree]'."
            ) from error
        except Exception as error:
            raise LA_MCTSClassifierError(
                f"The scikit-learn SVM backend could not be initialized: {error}"
            ) from error
        X = self.state.global_actions[sample_indices].detach().cpu().numpy()
        if self.config.source_protocol == "mcts_vs_2022":
            X = X.astype(np.float32)
        y = labels.detach().cpu().numpy()
        try:
            estimator = SVC(
                kernel="rbf",
                gamma=self.config.classifier_gamma,
                C=1.0,
                random_state=self._source_rng
                if self.config.source_protocol == "mcts_vs_2022"
                else None,
            ).fit(X, y)
        except Exception as error:
            raise LA_MCTSClassifierError(f"Region SVM fitting failed: {error}") from error
        return RegionClassifier(estimator)

    def _source_prepare_node(self, node: RegionNode) -> None:
        """Source update_bag fits every >2-row node, including unsplit leaves."""
        node.mean_reward = float(
            np.mean(self._quality()[node.sample_indices].float().cpu().numpy())
        )
        if len(node.sample_indices) <= 2:
            node.unsplittable = True
            return
        labels = self._cluster_quality(node.sample_indices)
        if torch.unique(labels).numel() != 2:
            node.unsplittable = True
            return
        node.classifier = self._fit_region_classifier(node.sample_indices, labels)
        predicted = node.classifier.predict(self.state.global_actions[node.sample_indices].float())
        node.unsplittable = torch.unique(predicted).numel() != 2

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
        if self.config.source_protocol != "mcts_vs_2022" and torch.unique(predicted).numel() != 2:
            node.unsplittable = True
            return False
        # Released LA-MCTS stores KMeans-labelled bags; the fitted SVM only
        # supplies the spatial boundary, and need not classify every row exactly.
        groups = [node.sample_indices[labels == label] for label in (0, 1)]
        if any(group.numel() < self.config.min_child_size for group in groups):
            node.unsplittable = True
            return False
        quality = self._quality()
        means = [float(quality[group].mean().item()) for group in groups]
        good_label = (
            0
            if self.config.source_protocol == "mcts_vs_2022"
            else (0 if means[0] >= means[1] else 1)
        )
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
            if self.config.source_protocol == "mcts_vs_2022":
                self._source_prepare_node(self.state.nodes[child_id])
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
        if self.config.source_protocol == "mcts_vs_2022" and len(rows):
            self._source_prepare_node(root)
        queue = [0]
        while queue:
            node = self.state.nodes[queue.pop(0)]
            if self._split_node(node):
                queue.extend(node.child_ids)
        self.state.rebuild_count += 1
        self.state.finite_exhausted_leaf_ids.clear()
        self._source_tree_dirty = False

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
        return node.mean_reward + 2 * self.config.exploration_constant * math.sqrt(
            2 * math.sqrt(max(parent_visits, 1)) / node.visits
        )

    def _traverse(self) -> tuple[RegionNode, list[int]]:
        """Choose a non-exhausted leaf, sampling exact UCT ties uniformly."""

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
            tied = [child for child, score in zip(children, scores, strict=True) if score == best]
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

    def _filter_candidates_to_leaf(self, leaf: RegionNode, actions: torch.Tensor) -> torch.Tensor:
        """Mask full finite rows by every classifier on the leaf path."""

        return self._contains(leaf.node_id, actions)

    def _kernel_scores(self, candidates: torch.Tensor) -> torch.Tensor:
        """Score local candidates using the configured full-dimensional GP."""

        if self.config.source_protocol == "mcts_vs_2022" and self.config.local_sampler == "bo":
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, Matern

            model = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5),
                alpha=0.1**2,
                normalize_y=False,
            )
            with warnings.catch_warnings(record=True) as reported:
                warnings.simplefilter("always")
                model.fit(self.state.global_actions.cpu().numpy(), self._quality().cpu().numpy())
            for item in reported:
                warnings.warn(
                    f"Released LA-MCTS BO GP fit: {item.message}", LA_MCTSWarning, stacklevel=2
                )
            assert self.state.selected_leaf_id is not None
            leaf = self.state.nodes[self.state.selected_leaf_id]
            reference = self.state.global_actions[leaf.sample_indices].float().cpu().numpy()
            source_scores = expected_improvement(
                model, candidates.cpu().numpy(), reference, xi=0.001
            )
            return torch.as_tensor(source_scores, dtype=self.dtype, device=self.device)
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
        if self.initial_points_sampling_method == "paper":
            count = self.config.initial_points
            if self.config.source_protocol == "mcts_vs_2022":
                if self._source_initial_design_cache is None:
                    design = latin_hypercube(self._source_rng, count, self.action_dimension)
                    self._source_initial_design_cache = torch.as_tensor(
                        lower.cpu().numpy() + design * (upper - lower).cpu().numpy(),
                        dtype=self.dtype,
                        device=self.device,
                    )
                position = len(self.state.global_actions)
                return self._source_initial_design_cache[position : position + 1], None
            generator = torch.Generator(device=self.device).manual_seed(self.config.seed)
            unit = torch.empty(count, self.action_dimension, dtype=self.dtype, device=self.device)
            for dimension in range(self.action_dimension):
                unit[:, dimension] = torch.randperm(count, generator=generator, device=self.device)
            unit = (
                unit
                + torch.rand(unit.shape, generator=generator, dtype=self.dtype, device=self.device)
            ) / count
            position = self.state.global_actions.shape[0]
            return lower + unit[position : position + 1] * (upper - lower), None
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
                    "CMA-ES requires the 'basat[optimizers]' extra."
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

        if (
            self.config.source_protocol == "mcts_vs_2022"
            and self._source_tree_dirty
            and len(self.state.global_actions) >= self.config.initial_points
            and self._turbo_program is None
        ):
            self._build_tree()
        if (
            self.config.local_sampler == "turbo"
            and self.state.global_actions.shape[0] >= self.config.initial_points
            and "candidate_actions" not in round_info
        ):
            return self._ask_turbo_impl(round_info=round_info)
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
                    if (
                        self.config.source_protocol == "mcts_vs_2022"
                        and self.config.local_sampler == "bo"
                        and self.acquisition_sampling_method == "paper"
                    ):
                        pool = (
                            self._source_region_pool(leaf, 10000)
                            if len(path) > 1
                            else self._source_uniform(1)
                        )
                    else:
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
            if self.config.local_sampler == "random" or (
                self.config.source_protocol == "mcts_vs_2022"
                and len(path) == 1
                and candidate_actions is None
            ):
                selected_row = 0
                scores = torch.zeros(pool.shape[0], dtype=self.dtype, device=self.device)
            else:
                scores = self._kernel_scores(pool)
                selected_row = (
                    int(np.argsort(scores.cpu().numpy())[-1])
                    if self.config.source_protocol == "mcts_vs_2022"
                    else int(torch.argmax(scores).item())
                )
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

    def _source_leaf_initial_design(self, leaf: RegionNode) -> torch.Tensor:
        """Released expanding Sobol-box initializer for the ten TuRBO points.

        Each center probes 2000 Sobol points, starting with side length1e-4
        and doubling until the box crosses the classifier boundary. The source
        keeps in-region points from that first crossing and samples ten rows
        with replacement. If no box crosses, it falls back to global uniform.
        """
        if self.config.source_protocol == "mcts_vs_2022":
            return self._source_region_pool(leaf, 10)
        lower, upper = self._bounds_on_algorithm_device()
        members = self.state.global_actions[leaf.sample_indices]
        centers = members[self._contains(leaf.node_id, members)]
        accepted = []
        seed = int(torch.randint(1_000_000, (1,), generator=self.generator).item())
        sobol = torch.quasirandom.SobolEngine(self.action_dimension, scramble=True, seed=seed)
        for _ in centers:
            index = int(torch.randint(members.shape[0], (1,), generator=self.generator).item())
            center = members[index]
            unit = sobol.draw(2000).to(dtype=self.dtype, device=self.device)
            length = 1e-4
            while length < float((upper - lower).max()):
                lo = torch.maximum(center - length / 2, lower)
                hi = torch.minimum(center + length / 2, upper)
                proposed = lo + (hi - lo) * unit
                inside = self._contains(leaf.node_id, proposed)
                if not bool(inside.all()):
                    accepted.append(proposed[inside])
                    break
                length *= 2
        pool = torch.cat(accepted) if accepted else members.new_empty((0, self.action_dimension))
        if pool.shape[0] > 10:
            selected = torch.randint(pool.shape[0], (10,), generator=self.generator)
            return pool[selected]
        if pool.shape[0]:
            return pool
        return lower + (upper - lower) * torch.rand(
            (10, self.action_dimension),
            generator=self.generator,
            dtype=self.dtype,
            device=self.device,
        )

    def _source_uniform(self, count: int) -> torch.Tensor:
        lower, upper = self._bounds_on_algorithm_device()
        return torch.as_tensor(
            self._source_rng.uniform(
                lower.cpu().numpy(), upper.cpu().numpy(), size=(count, self.action_dimension)
            ),
            dtype=self.dtype,
            device=self.device,
        )

    def _source_region_pool(self, leaf: RegionNode, count: int) -> torch.Tensor:
        """Released expanding-box Sobol candidate pool; no objective access."""
        lower, upper = (v.cpu().numpy() for v in self._bounds_on_algorithm_device())
        members = self.state.global_actions[leaf.sample_indices].float().cpu().numpy()
        seed = int(self._source_rng.randint(1_000_000))
        sobol = torch.quasirandom.SobolEngine(self.action_dimension, scramble=True, seed=seed)
        member_tensor = torch.as_tensor(members, device=self.device)
        centers = members[self._contains(leaf.node_id, member_tensor).cpu().numpy()]
        accepted = []
        for _ in centers:
            center = members[int(self._source_rng.randint(len(members)))]
            unit = sobol.draw(2000).double().numpy()
            length = 1e-4
            while length < np.max(upper - lower):
                lo, hi = (
                    np.clip(center - length / 2, lower, upper),
                    np.clip(center + length / 2, lower, upper),
                )
                proposed = (hi - lo) * unit + lo
                inside = (
                    self._contains(leaf.node_id, torch.as_tensor(proposed, device=self.device))
                    .cpu()
                    .numpy()
                )
                if not inside.all():
                    accepted.extend(proposed[inside].tolist())
                    break
                length *= 2
        if not accepted:
            return self._source_uniform(count)
        pool = np.asarray(accepted)
        if len(pool) > count:
            pool = pool[self._source_rng.choice(len(pool), count)]
        return torch.as_tensor(pool, dtype=self.dtype, device=self.device)

    def _ask_turbo_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Released LA-MCTS component: ten leaf points, then fresh local TuRBO.

        The source classifier constrains the component's initial points only.
        Subsequent TuRBO proposals use the complete ambient box, a distinction
        from a continuously classifier-constrained local optimizer.
        """
        program = self._turbo_program
        if program is None:
            leaf, path = self._traverse()
            self.state.selected_leaf_id = leaf.node_id
            self.state.selected_path = path
            pool = (
                self._source_leaf_initial_design(leaf)
                if self.acquisition_sampling_method == "paper"
                else self._sample_continuous_pool(leaf)
            )
            if pool.shape[0] < 10 and self.config.source_protocol != "mcts_vs_2022":
                raise LA_MCTSRegionError("The selected leaf cannot supply ten initial points.")
            program = {
                "start": self.state.global_actions.shape[0],
                "queue": pool[:10],
                "cursor": 0,
                "initial_count": min(10, len(pool)),
            }
            self._turbo_program = program
        if program["queue"] is None:
            start = program["start"]
            queue = _component_candidates(
                bounds=self.bounds,
                actions=self.state.global_actions[start:],
                observations=self.state.global_observations[start:],
                start=program.get("initial_count", 10),
                seed=self.config.seed + start,
                maximize=self.maximize,
                training_steps=self.config.gp_maxiter,
                source_rng=self._source_rng
                if self.config.source_protocol == "mcts_vs_2022"
                else None,
                source_torch_rng=self._source_torch_rng
                if self.config.source_protocol == "mcts_vs_2022"
                else None,
            )
            if queue.shape[0] == 0:
                self._turbo_program = None
                self._build_tree()
                return self._ask_turbo_impl(round_info=round_info)
            program["queue"] = queue
            program["cursor"] = 0
        actions = program["queue"][program["cursor"] : program["cursor"] + 1]
        self._record_last_ask(
            actions=actions,
            model_inputs=actions,
            round_info=round_info,
            algorithm_metadata={
                "leaf_id": self.state.selected_leaf_id,
                "path": tuple(self.state.selected_path),
                "dataset_indices": None,
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
        if (
            self.config.source_protocol == "mcts_vs_2022"
            and self.state.selected_path
            and len(self.state.global_actions) > self.config.initial_points
        ):
            for value in to_quality(values, maximize=self.maximize).tolist():
                for node_id in reversed(self.state.selected_path):
                    node = self.state.nodes[node_id]
                    if node.visits:
                        node.mean_reward = (node.mean_reward * node.visits + value) / (
                            node.visits + 1
                        )
                        node.visits += 1
        if self._turbo_program is not None:
            program = self._turbo_program
            program["cursor"] += 1
            if program["cursor"] == program["queue"].shape[0]:
                if (
                    self.state.global_actions.shape[0]
                    - program["start"]
                    + (10 - program.get("initial_count", 10))
                    >= self.config.inner_max_evaluations
                ):
                    self._turbo_program = None
                else:
                    program["queue"] = None
                    program["cursor"] = 0
            if self._turbo_program is not None:
                return
        if self.config.source_protocol == "mcts_vs_2022":
            self._source_tree_dirty = True
        else:
            self._build_tree()
        self.state.selected_leaf_id = None
        self.state.selected_path = []

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Report tree size, rebuilds, region acceptance, exhaustion and sampler."""

        leaves = [node for node in self.state.nodes.values() if not node.child_ids]
        return {
            "node_count": len(self.state.nodes),
            "leaf_count": len(leaves),
            "rebuild_count": self.state.rebuild_count,
            "last_acceptance_rate": self.state.last_acceptance_rate,
            "finite_exhausted_leaf_ids": tuple(sorted(self.state.finite_exhausted_leaf_ids)),
            "local_sampler": self.config.local_sampler,
        }

    @staticmethod
    def _node_to_dict(node: RegionNode) -> dict[str, Any]:
        """Serialize a region node and its fitted scikit-learn classifier.

        The estimator is pickled only for Basat's trusted transactional
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
            "source_numerics_version": 2,
            "source_turbo_numerics_version": (
                3
                if self.config.source_protocol == "mcts_vs_2022"
                and self.config.local_sampler == "turbo"
                else None
            ),
            "source_numpy_rng": random_state_dict(self._source_rng),
            "source_torch_rng": self._source_torch_rng.get_state(),
            "source_initial_design": self._source_initial_design_cache,
            "source_tree_dirty": self._source_tree_dirty,
            "turbo_program": self._turbo_program,
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
        if (
            self.config.source_protocol == "mcts_vs_2022"
            and state.get("source_numerics_version") != 2
        ):
            raise LA_MCTSCheckpointError(
                "Source checkpoint predates LA-MCTS numerical protocol v2; use a fresh output directory."
            )
        if self.config.source_protocol == "mcts_vs_2022":
            if (
                self.config.local_sampler == "turbo"
                and state.get("source_turbo_numerics_version") != 3
            ):
                raise LA_MCTSCheckpointError(
                    "Source TuRBO checkpoint does not match explicit Lanczos sampling protocol v3; "
                    "preserve the old trajectory and use a fresh output directory."
                )
            try:
                validate_random_states(state)
                if not isinstance(state["source_tree_dirty"], bool):
                    raise ValueError("Invalid tree-dirty flag.")
                initial = state["source_initial_design"]
                if initial is not None:
                    design = torch.as_tensor(initial)
                    if (
                        design.shape != (self.config.initial_points, self.action_dimension)
                        or not torch.isfinite(design).all()
                    ):
                        raise ValueError("Invalid cached initial design.")
                program = state["turbo_program"]
                if program is not None:
                    start, cursor = program["start"], program["cursor"]
                    count = program["initial_count"]
                    rows = len(state["global_actions"])
                    if any(
                        isinstance(v, bool) or not isinstance(v, int)
                        for v in (start, cursor, count)
                    ):
                        raise ValueError("Invalid local-program counters.")
                    if not 0 <= start <= rows or not 1 <= count <= 10 or cursor < 0:
                        raise ValueError("Invalid local-program offsets.")
                    queue = program["queue"]
                    if queue is None:
                        if cursor != 0:
                            raise ValueError("An empty local queue has a nonzero cursor.")
                    else:
                        queued = torch.as_tensor(queue)
                        if (
                            queued.ndim != 2
                            or queued.shape[1] != self.action_dimension
                            or not 0 <= cursor < len(queued)
                            or not torch.isfinite(queued).all()
                        ):
                            raise ValueError("Invalid local proposal queue.")
                    if (
                        state["selected_leaf_id"] not in {int(key) for key in state["nodes"]}
                        or not state["selected_path"]
                    ):
                        raise ValueError("The local program has lost its selected leaf/path.")
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                raise LA_MCTSCheckpointError(f"Invalid source state: {error}") from error
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
        self._turbo_program = state.get("turbo_program")
        if self.config.source_protocol == "mcts_vs_2022":
            restore_random_state(self._source_rng, state["source_numpy_rng"])
            self._source_torch_rng.set_state(
                torch.as_tensor(state["source_torch_rng"], dtype=torch.uint8, device="cpu")
            )
            initial = state["source_initial_design"]
            self._source_initial_design_cache = (
                None
                if initial is None
                else torch.as_tensor(initial, dtype=self.dtype, device=self.device).clone()
            )
            self._source_tree_dirty = bool(state["source_tree_dirty"])
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
        self._turbo_program = None
        self._source_rng.seed(self.config.seed)
        self._source_torch_rng.manual_seed(self.config.seed)
        self._source_initial_design_cache = None
        self._source_tree_dirty = True


__all__ = [
    "LA_MCTS",
    "LAMCTSConfig",
    "LAMCTSState",
    "LA_MCTSConfig",
    "RegionClassifier",
    "RegionNode",
]
