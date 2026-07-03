"""Single-file REMBO-UCB optimizer library.

Typical usage:
    ``optimizer = APREMBO(...); result = optimizer.optimize(objective)``

Main constructor arguments:
    * ``original_boundaries``: real high-dimensional search box, shape ``(D, 2)``.
    * ``d_embedding``: REMBO latent dimension.
    * ``max_iter``: total objective-evaluation budget.
    * ``tau``: posterior stopping threshold when threshold tests are requested.
    * ``N``: number of child cells created by each tree split.
    * ``batch_size``: number of points evaluated in one BO iteration.
    * ``n_init``: initial Latin-hypercube design size.
    * ``a`` / ``b``: objective score bounds used in the UCB confidence width.
    * ``R``: known observation-noise standard deviation.
    * ``kernel_name``: fallback GP kernel name.
    * ``auto_select_kernel``: when true, fit supported ARD kernels and keep the
      one with the best marginal likelihood at each GP fit.
    * ``lipschitz_mode``: ``"predict"`` estimates L from the GP posterior mean
      gradient; ``"user"`` keeps the user-provided ``L`` fixed.

Lipschitz handling:
    * With ``lipschitz_mode="user"``, pass ``L=<float>`` to use a fixed
      user-specified Lipschitz constant.
    * With ``lipschitz_mode="predict"``, ``L`` is only a warm-start value before
      the first GP exists; after a selected-node GP is fitted, APREMBO estimates
      L from the fitted GP posterior mean gradient inside that embedded cell.

This module contains the full REMBO-UCB implementation in one importable file:
lengthscale optimization helpers, the Gaussian-process wrapper, tree
initialization/refinement/sampling helpers, and the main ``APREMBO`` class.
Shared kernels, stopping rules, embedding transforms, and sampling utilities
come from the surrounding ``APREMBO`` package.
"""


# =============================================================================
# Lengthscale optimization helpers
# =============================================================================

import math
import random
import warnings

import torch


DEFAULT_REOPTIMIZE_INTERVAL = 20
DEFAULT_POSTERIOR_STD_THRESHOLD = 0.002
DEFAULT_LOW_STD_PATIENCE = 5
DEFAULT_UPPER_SHRINK_FACTOR = 0.5


def _find_lengthscale_kernel(module):
    """
    Recursively find the actual base kernel that exposes a non-None lengthscale.

    Important:
    ScaleKernel and InducedEmbeddingKernel can expose a lengthscale attribute or
    property that is None. Those wrappers are not valid targets. We must keep
    descending until the real RBFKernel / MaternKernel is found.
    """
    if module is None:
        return None

    lengthscale = getattr(module, "lengthscale", None)
    if lengthscale is not None:
        return module

    base_kernel = getattr(module, "base_kernel", None)
    if base_kernel is not None:
        found = _find_lengthscale_kernel(base_kernel)
        if found is not None:
            return found

    for child in module.children():
        found = _find_lengthscale_kernel(child)
        if found is not None:
            return found

    return None


def set_scalar_lengthscale(model, lengthscale):
    """
    Set one scalar lengthscale on the actual base kernel.

    This works for the paper kernel

        k_emb(z,z') = k(p_X(Az), p_X(Az')),

    because it searches recursively through ScaleKernel and InducedEmbeddingKernel
    until it reaches the underlying RBF/Matern base kernel k.
    """
    kernel = _find_lengthscale_kernel(getattr(model, "covar_module", None))
    if kernel is None or getattr(kernel, "lengthscale", None) is None:
        raise AttributeError(
            "Model does not expose a non-None base-kernel lengthscale. "
            "Expected the underlying RBFKernel or MaternKernel inside the induced kernel."
        )

    value = torch.as_tensor(
        float(lengthscale),
        dtype=kernel.lengthscale.dtype,
        device=kernel.lengthscale.device,
    )
    try:
        kernel.lengthscale = value
    except Exception:
        kernel.initialize(lengthscale=value)


def get_scalar_lengthscale(model):
    """Return the mean scalar value of the current base-kernel lengthscale."""
    kernel = _find_lengthscale_kernel(getattr(model, "covar_module", None))
    if kernel is None:
        return None
    return float(kernel.lengthscale.detach().double().mean().cpu().item())


def freeze_lengthscale(model, freeze=True):
    """Freeze or unfreeze the raw lengthscale parameter if it exists."""
    kernel = _find_lengthscale_kernel(getattr(model, "covar_module", None))
    if kernel is not None and hasattr(kernel, "raw_lengthscale"):
        kernel.raw_lengthscale.requires_grad_(not freeze)
        return True
    return False


def _mll_value_for_lengthscale(model, mll, train_x, train_y, lengthscale):
    set_scalar_lengthscale(model, lengthscale)
    model.train()
    model.likelihood.train()

    with torch.no_grad():
        output = model(train_x)
        target = train_y.reshape(-1)
        value = mll(output, target)

    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu().item())

    if not math.isfinite(value):
        return -float("inf")
    return value


def _logspace_candidates(lower, upper, n_grid, n_random, rng):
    log_lower = math.log(lower)
    log_upper = math.log(upper)

    candidates = []
    if n_grid > 1:
        for i in range(int(n_grid)):
            alpha = i / (int(n_grid) - 1)
            candidates.append(math.exp((1.0 - alpha) * log_lower + alpha * log_upper))
    else:
        candidates.append(math.sqrt(lower * upper))

    for _ in range(int(n_random)):
        candidates.append(math.exp(rng.uniform(log_lower, log_upper)))

    return candidates


def _bounded_mll_search(
    model,
    mll,
    train_x,
    train_y,
    bounds,
    n_grid=25,
    n_random=25,
    seed=0,
):
    """
    Bounded scalar lengthscale search by marginal log likelihood.

    This is the safe fallback used when DIRECT is unavailable. It preserves
    the REMBO Algorithm 3 objective: maximize log marginal likelihood over
    a bounded lengthscale interval [L, U].
    """
    lower, upper = float(bounds[0]), float(bounds[1])
    if lower <= 0 or upper <= lower:
        raise ValueError("lengthscale bounds must satisfy 0 < lower < upper.")

    rng = random.Random(seed)
    candidates = _logspace_candidates(lower, upper, n_grid, n_random, rng)

    best_lengthscale = candidates[0]
    best_value = -float("inf")

    for lengthscale in candidates:
        value = _mll_value_for_lengthscale(model, mll, train_x, train_y, lengthscale)
        if value > best_value:
            best_value = value
            best_lengthscale = float(lengthscale)

    set_scalar_lengthscale(model, best_lengthscale)
    return best_lengthscale, best_value


def _try_scipy_direct_search(model, mll, train_x, train_y, bounds, maxfun=80):
    """Use scipy.optimize.direct when available; otherwise return None."""
    try:
        from scipy.optimize import Bounds, direct
    except Exception:
        return None

    lower, upper = float(bounds[0]), float(bounds[1])
    log_lower = math.log(lower)
    log_upper = math.log(upper)

    def objective(log_l_array):
        lengthscale = math.exp(float(log_l_array[0]))
        return -_mll_value_for_lengthscale(model, mll, train_x, train_y, lengthscale)

    try:
        result = direct(
            objective,
            Bounds([log_lower], [log_upper]),
            maxfun=int(maxfun),
            locally_biased=False,
        )
    except Exception:
        return None

    if not getattr(result, "success", False) and not math.isfinite(float(result.fun)):
        return None

    lengthscale = math.exp(float(result.x[0]))
    value = -float(result.fun)
    set_scalar_lengthscale(model, lengthscale)
    return lengthscale, value


def _try_cma_es_search(model, mll, train_x, train_y, bounds, start_lengthscale, maxiter=40, seed=0):
    """
    Refine the DIRECT result with CMA-ES when the optional cma package exists.

    If the external cma package is not installed, this falls back to a small
    dependency-free one-dimensional evolution-strategy refinement in log
    lengthscale space. This keeps the code runnable without a local cma.py file
    or a pip-installed cma package.
    """
    lower, upper = float(bounds[0]), float(bounds[1])
    log_lower = math.log(lower)
    log_upper = math.log(upper)
    start_log = min(max(math.log(float(start_lengthscale)), log_lower), log_upper)

    def objective(log_l_value):
        log_l = min(max(float(log_l_value), log_lower), log_upper)
        lengthscale = math.exp(log_l)
        return -_mll_value_for_lengthscale(model, mll, train_x, train_y, lengthscale)

    try:
        import cma

        es = cma.CMAEvolutionStrategy(
            [start_log],
            max((log_upper - log_lower) / 6.0, 1e-6),
            {
                "bounds": [[log_lower], [log_upper]],
                "maxiter": int(maxiter),
                "seed": int(seed),
                "verbose": -9,
            },
        )
        es.optimize(lambda arr: objective(float(arr[0])))
        best_log_l = float(es.result.xbest[0])
        best_log_l = min(max(best_log_l, log_lower), log_upper)
        lengthscale = math.exp(best_log_l)
        value = -float(es.result.fbest)
        set_scalar_lengthscale(model, lengthscale)
        return lengthscale, value
    except Exception:
        pass

    rng = random.Random(seed)
    best_log_l = start_log
    best_obj = objective(best_log_l)
    sigma = max((log_upper - log_lower) / 6.0, 1e-6)

    for _ in range(int(maxiter)):
        candidates = [
            min(max(rng.gauss(best_log_l, sigma), log_lower), log_upper)
            for _ in range(8)
        ]
        for candidate in candidates:
            obj = objective(candidate)
            if obj < best_obj:
                best_obj = obj
                best_log_l = candidate
        sigma *= 0.85

    lengthscale = math.exp(best_log_l)
    value = -float(best_obj)
    set_scalar_lengthscale(model, lengthscale)
    return lengthscale, value


def select_lengthscale_direct_cma(
    model,
    mll,
    train_x,
    train_y,
    bounds,
    seed=0,
    direct_maxfun=80,
    cma_maxiter=40,
    fallback_grid=25,
    fallback_random=25,
):
    """
    REMBO-style bounded lengthscale optimization.

    REMBO Algorithm 3 learns the lengthscale by solving

        l_t = argmax_{l in [L,U]} log p(y_{1:t} | z_{1:t}, l)

    using DIRECT and CMA-ES. This function follows that structure:

    1. try DIRECT on the bounded interval;
    2. refine with CMA-ES when the optional cma package is available;
    3. otherwise refine with a dependency-free one-dimensional evolution strategy;
    4. fall back to log-space grid + random search when DIRECT is unavailable.

    For the induced kernel

        k_emb(z,z') = k(p_X(Az), p_X(Az')),

    the lengthscale belongs to the underlying base kernel k on the normalized
    original domain [-1,1]^D.
    """
    direct_result = _try_scipy_direct_search(
        model=model,
        mll=mll,
        train_x=train_x,
        train_y=train_y,
        bounds=bounds,
        maxfun=direct_maxfun,
    )

    if direct_result is None:
        direct_result = _bounded_mll_search(
            model=model,
            mll=mll,
            train_x=train_x,
            train_y=train_y,
            bounds=bounds,
            n_grid=fallback_grid,
            n_random=fallback_random,
            seed=seed,
        )

    direct_lengthscale, direct_value = direct_result

    cma_result = _try_cma_es_search(
        model=model,
        mll=mll,
        train_x=train_x,
        train_y=train_y,
        bounds=bounds,
        start_lengthscale=direct_lengthscale,
        maxiter=cma_maxiter,
        seed=seed,
    )

    if cma_result is not None and cma_result[1] > direct_value:
        return cma_result

    set_scalar_lengthscale(model, direct_lengthscale)
    return direct_lengthscale, direct_value


class REMBOAlgorithm3LengthscaleController:
    """
    Stateful implementation of the scheduling logic in REMBO Algorithm 3.

    It controls when lengthscale optimization is run and when the upper bound U
    should be tightened after repeated low posterior-variance queries.
    """

    def __init__(
        self,
        lower_bound=0.01,
        upper_bound=50.0,
        reoptimize_interval=DEFAULT_REOPTIMIZE_INTERVAL,
        posterior_std_threshold=DEFAULT_POSTERIOR_STD_THRESHOLD,
        low_std_patience=DEFAULT_LOW_STD_PATIENCE,
        upper_shrink_factor=DEFAULT_UPPER_SHRINK_FACTOR,
        min_upper_bound=None,
        seed=0,
    ):
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.reoptimize_interval = int(reoptimize_interval)
        self.posterior_std_threshold = float(posterior_std_threshold)
        self.low_std_patience = int(low_std_patience)
        self.upper_shrink_factor = float(upper_shrink_factor)
        self.min_upper_bound = (
            float(min_upper_bound) if min_upper_bound is not None else self.lower_bound * 1.01
        )
        self.seed = int(seed)
        self.low_std_count = 0
        self.last_selected_lengthscale = None
        self.last_mll_value = None

    @property
    def bounds(self):
        return (self.lower_bound, self.upper_bound)

    def observe_posterior_std(self, posterior_std):
        """Update the repeated-low-uncertainty counter."""
        if posterior_std is None:
            return False

        if float(posterior_std) < self.posterior_std_threshold:
            self.low_std_count += 1
        else:
            self.low_std_count = 0

        if self.low_std_count >= self.low_std_patience:
            new_upper = max(self.min_upper_bound, self.upper_bound * self.upper_shrink_factor)
            if new_upper < self.upper_bound:
                self.upper_bound = new_upper
            self.low_std_count = 0
            return True

        return False

    def should_optimize(self, iteration=None, forced=False, upper_bound_was_tightened=False):
        """Decide whether Algorithm 3 should optimize the lengthscale now."""
        if forced or upper_bound_was_tightened:
            return True
        if self.last_selected_lengthscale is None:
            return True
        if iteration is None:
            return True
        return int(iteration) % self.reoptimize_interval == 0

    def optimize(
        self,
        model,
        mll,
        train_x,
        train_y,
        iteration=None,
        posterior_std=None,
        forced=False,
    ):
        """Run the REMBO Algorithm 3 lengthscale update when needed."""
        upper_bound_was_tightened = self.observe_posterior_std(posterior_std)

        if not self.should_optimize(
            iteration=iteration,
            forced=forced,
            upper_bound_was_tightened=upper_bound_was_tightened,
        ):
            if self.last_selected_lengthscale is not None:
                set_scalar_lengthscale(model, self.last_selected_lengthscale)
            return self.last_selected_lengthscale, self.last_mll_value, False

        lengthscale, mll_value = select_lengthscale_direct_cma(
            model=model,
            mll=mll,
            train_x=train_x,
            train_y=train_y,
            bounds=self.bounds,
            seed=self.seed + (0 if iteration is None else int(iteration)),
        )
        self.last_selected_lengthscale = float(lengthscale)
        self.last_mll_value = float(mll_value)
        return self.last_selected_lengthscale, self.last_mll_value, True


# =============================================================================
# Gaussian process model
# =============================================================================

"""Gaussian Process model wrapper used by REMBO and REMBO-UCB.

The model delegates all covariance construction to ``APREMBO.kernel``.  If a
projection matrix is supplied, training data is interpreted as embedded
z-space data and the model is only allowed to use the induced embedded kernel.
"""

import torch
import gpytorch

from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP

try:
    from .kernel import build_rembo_ucb_kernel
except ImportError:
    from kernel import build_rembo_ucb_kernel

class GaussianProcess(SingleTaskGP):
    def __init__(
        self,
        train_x,
        train_y,
        R=0.01,
        train_yvar=None,
        kernel_name="rbf",
        covar_module=None,
        lengthscale_bounds=(0.01, 50.0),
        projection_matrix=None,
        original_bounds=None,
        n_keep_dims=0,
        use_direct_cma_lengthscale=False,
        freeze_direct_cma_lengthscale=True,
        outcome_transform=None,
        input_transform=None,
    ):
        """
        Gaussian Process model for REMBO and REMBO-UCB.

        If projection_matrix is provided, this model uses the paper kernel

            k_emb(z,z') = k(p_X(Az), p_X(Az')).

        The base kernel k is defined on the normalized original domain [-1,1]^D.
        """
        train_X = torch.as_tensor(train_x, dtype=torch.double)
        train_Y = torch.as_tensor(train_y, dtype=torch.double)

        if train_Y.ndim == 1:
            train_Y = train_Y.view(-1, 1)

        self.R = float(R)
        self.kernel_name = str(kernel_name).lower()
        self.kernel_fit_score = -float("inf")
        self.lengthscale_bounds = lengthscale_bounds
        self.use_direct_cma_lengthscale = bool(use_direct_cma_lengthscale)
        self.freeze_direct_cma_lengthscale = bool(freeze_direct_cma_lengthscale)

        if train_yvar is None:
            train_Yvar = torch.full_like(train_Y, self.R ** 2, dtype=torch.double)
        else:
            train_Yvar = torch.as_tensor(train_yvar, dtype=torch.double)
            if train_Yvar.ndim == 1:
                train_Yvar = train_Yvar.view(-1, 1)

        if projection_matrix is not None and covar_module is not None:
            raise ValueError(
                "Embedded REMBO-UCB models can only use package-defined "
                "kernels from APREMBO.kernel through the induced embedding "
                "construction; custom covar_module is not allowed."
            )

        if covar_module is None:
            covar_module = build_rembo_ucb_kernel(
                kernel_name=self.kernel_name,
                input_dim=train_X.shape[-1],
                lengthscale_bounds=lengthscale_bounds,
                projection_matrix=projection_matrix,
                original_bounds=original_bounds,
                n_keep_dims=n_keep_dims,
                embedded=projection_matrix is not None,
            )

        # BoTorch versions differ in how they handle explicitly passed None
        # transforms. Do not pass outcome_transform=None or input_transform=None;
        # let SingleTaskGP use its own defaults instead.
        super_kwargs = {
            "train_X": train_X,
            "train_Y": train_Y,
            "train_Yvar": train_Yvar,
            "covar_module": covar_module,
        }
        if outcome_transform is not None:
            super_kwargs["outcome_transform"] = outcome_transform
        if input_transform is not None:
            super_kwargs["input_transform"] = input_transform

        super().__init__(**super_kwargs)

    def fit(self):
        """
        Fit GP hyperparameters.

        Default:
            BoTorch/GPyTorch optimizes lengthscale, signal variance/outputscale,
            and mean by maximizing the marginal log likelihood.

        If use_direct_cma_lengthscale=True:
            The lengthscale of the underlying base kernel k in
            k(p_X(Az), p_X(Az')) is selected first. Then the remaining
            hyperparameters are optimized by fit_gpytorch_mll.
        """
        self.train()
        self.likelihood.train()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        mll = mll.to(self.train_inputs[0])

        lengthscale_frozen = False

        if self.use_direct_cma_lengthscale:
            if select_lengthscale_direct_cma is None or set_scalar_lengthscale is None:
                raise ImportError(
                    "use_direct_cma_lengthscale=True requires the lengthscale helpers in APREMBO.py."
                )

            selected_lengthscale, _ = select_lengthscale_direct_cma(
                self,
                mll,
                self.train_inputs[0].double(),
                self.train_targets.view(-1, 1).double(),
                self.lengthscale_bounds,
            )
            set_scalar_lengthscale(self, selected_lengthscale)

            if self.freeze_direct_cma_lengthscale:
                if freeze_lengthscale is None:
                    raise ImportError("freeze_lengthscale is missing from APREMBO.py.")
                lengthscale_frozen = freeze_lengthscale(self, freeze=True)

        fit_gpytorch_mll(mll)

        if lengthscale_frozen:
            freeze_lengthscale(self, freeze=False)

        self.eval()
        self.likelihood.eval()
        self.kernel_fit_score = self._compute_kernel_fit_score(mll)
        return self

    def _compute_kernel_fit_score(self, mll):
        """Compute the fitted exact marginal log likelihood for kernel ranking."""
        try:
            self.eval()
            self.likelihood.eval()
            train_x = self.train_inputs[0].double()
            target = self.train_targets.reshape(-1).double()
            with torch.no_grad(), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The input matches the stored training data.*")
                output = self(train_x)
                value = mll(output, target)
            if isinstance(value, torch.Tensor):
                value = float(value.detach().cpu().item())
            if math.isfinite(value):
                return float(value)
        except Exception:
            pass
        return -float("inf")


def build_gp_model(
    train_x,
    train_y,
    R=0.01,
    train_yvar=None,
    kernel_name="rbf",
    covar_module=None,
    lengthscale_bounds=(0.01, 50.0),
    projection_matrix=None,
    original_bounds=None,
    n_keep_dims=0,
    use_direct_cma_lengthscale=False,
    freeze_direct_cma_lengthscale=True,
    fit=True,
):
    """Build and optionally fit a GP model."""
    model = GaussianProcess(
        train_x=train_x,
        train_y=train_y,
        R=R,
        train_yvar=train_yvar,
        kernel_name=kernel_name,
        covar_module=covar_module,
        lengthscale_bounds=lengthscale_bounds,
        projection_matrix=projection_matrix,
        original_bounds=original_bounds,
        n_keep_dims=n_keep_dims,
        use_direct_cma_lengthscale=use_direct_cma_lengthscale,
        freeze_direct_cma_lengthscale=freeze_direct_cma_lengthscale,
    )

    if fit:
        model.fit()

    return model


# =============================================================================
# Tree initialization
# =============================================================================

class APREMBOInitialization:
    """Algorithm 1: initialization and optimistic leaf selection."""
    def __init__(self, optimizer):
        self.optimizer = optimizer

    @staticmethod
    def _root_from_active_leaves(active_leaves):
        node = active_leaves[0]
        while getattr(node, "parent", None) is not None:
            node = node.parent
        return node

    def _refresh_path_bounds(self, active_leaves):
        seen = set()
        for leaf in active_leaves:
            node = leaf
            while node is not None:
                node_id = id(node)
                if node_id not in seen:
                    self.optimizer.ucb_node_bound(node)
                    seen.add(node_id)
                node = getattr(node, "parent", None)

    def select_active_leaf(self, active_leaves):
        if not active_leaves:
            raise ValueError("active_leaves must contain at least one node.")

        self.compute_leaf_bounds(active_leaves)
        self._refresh_path_bounds(active_leaves)

        current = self._root_from_active_leaves(active_leaves)
        while not current.is_leaf():
            for child in current.children:
                self.optimizer.ucb_node_bound(child)
            current = max(current.children, key=lambda node: node.B)
        return current

    @staticmethod
    def initialize_active_tree(root):
        root.B = float("inf")
        return [root], [root]

    def compute_leaf_bounds(self, active_leaves):
        for node in active_leaves:
            self.optimizer.ucb_node_bound(node)


# =============================================================================
# Adaptive refinement
# =============================================================================

class APREMBORefinement:
    """Algorithm 3: probabilistic termination and adaptive refinement."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def should_split(self, node):
        return (
            node.C >= self.optimizer.D_min
            and self.optimizer.confidence_width(node.C)
            <= self.optimizer.L_emb * self.optimizer.v1 * (self.optimizer.rho ** node.h)
        )

    def stopping_probability(self, node, z):
        return self.optimizer.posterior_success_probability(node.probability_model, z)

    def should_stop(self, node, z, t):
        if node.probability_model is None:
            return False, 0.0
        probability = self.stopping_probability(node, z)
        return (
            probability >= 1.0 - self.optimizer.delta
            and self.optimizer.has_enough_iterations_to_stop(t)
        ), probability

    def split_node(self, node, active_leaves, tree, t):
        c_t = self.optimizer.confidence_width(node.C)
        lip_threshold = (
            self.optimizer.L_emb
            * self.optimizer.v1
            * (self.optimizer.rho ** node.h)
        )
        if getattr(self.optimizer, "show", False):
            print(
                f"Splitting node at iteration {t}, depth {node.h}, idx {node.i} | "
                f"C={node.C} | c_t={c_t:.6g} | "
                f"L_emb*v1*rho^h={lip_threshold:.6g} "
                f"(L_emb={self.optimizer.L_emb:.6g}, v1={self.optimizer.v1:.6g}, "
                f"rho^h={(self.optimizer.rho ** node.h):.6g})"
            )
        return self.optimizer.split_ucb_node(
            node,
            self.optimizer.N,
            active_leaves,
            tree,
        )

    def update_b_values(self, active_leaves, t=None):
        self.optimizer.initialization_module.compute_leaf_bounds(active_leaves)

    def terminate_or_refine(self, node, z, t, active_leaves, tree):
        should_stop, probability = self.should_stop(node, z, t)
        if should_stop:
            return "stop", probability

        node.uncertainty = self.optimizer.confidence_width(node.C)
        if self.should_split(node):
            self.split_node(node, active_leaves, tree, t)
        else:
            self.update_b_values(active_leaves, t=t)
        return "continue", probability


# =============================================================================
# Sampling and observation updates
# =============================================================================

import numpy as np


class APREMBOSampling:
    """Algorithm 2: hybrid sampling and local model update."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def select_query_point(self, node, batch_size=1):
        if not getattr(node, "local_dataset", None):
            return self.optimizer.uniform_query_point(node, batch_size=batch_size)
        self.optimizer.fit_local_model_for_node(node)
        return self.optimizer.oracle_query_point(node, batch_size=batch_size)

    def fit_local_model(self, node):
        return self.optimizer.fit_local_model_for_node(node)

    def fit_probability_model(self, node):
        return self.optimizer.fit_probability_model_for_node(node)

    @staticmethod
    def append_observation(node, x, z, y_noisy):
        previous_count = node.C
        node.local_dataset.append((x, z, y_noisy))
        node.C += 1
        if previous_count == 0:
            node.mu_hat = y_noisy
        else:
            node.mu_hat = (previous_count * node.mu_hat + y_noisy) / node.C

    @staticmethod
    def node_to_root_path(node):
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = getattr(current, "parent", None)
        return path

    def append_observation_to_path(self, node, x, z, y_noisy):
        path = self.node_to_root_path(node)
        for path_node in path:
            self.append_observation(path_node, x, z, y_noisy)
            self.optimizer.rebuild_node_tensors(path_node)

        # Recompute every UCB/B value on the selected path after the fresh
        # observation has been included in all path statistics.
        for path_node in path:
            self.optimizer.ucb_node_bound(path_node)

    def sample_observe_and_update(self, node, black_box_function, t, batch_size=1, force_uniform=False):
        self.optimizer.current_algorithm3_iteration = int(t)
        if force_uniform:
            x, z = self.optimizer.uniform_query_point(node, batch_size=batch_size)
        else:
            x, z = self.select_query_point(node, batch_size=batch_size)
        if node.model is not None:
            self.optimizer.update_wang_lengthscale_upper_bound(node.model, z)

        y_raw_values = []
        y_noisy_values = []
        for batch_index in range(z.shape[0]):
            x_one = x[batch_index : batch_index + 1]
            z_one = z[batch_index : batch_index + 1]
            self.optimizer.record_clip_ratio(x_one)
            x_real = self.optimizer._to_real_domain(x_one)
            self.optimizer._debug_evaluation_domains(x_one, x_real)
            y_raw = self.optimizer.as_float(black_box_function(x_real))
            y_noisy = y_raw + self.optimizer.sample_noise()
            recorder = getattr(black_box_function, "__self__", None)
            if recorder is not None and hasattr(recorder, "record_noisy_observation"):
                recorder.record_noisy_observation(x_real, y_raw, y_noisy)
            y_raw_values.append(y_raw)
            y_noisy_values.append(y_noisy)

            self.append_observation_to_path(node, x_one, z_one, y_noisy)

        self.optimizer.rebuild_node_tensors(node)
        self.optimizer.current_algorithm3_iteration = int(t) + int(z.shape[0])
        model = self.fit_local_model(node)
        self.optimizer.update_gp_plot(
            model=model,
            z_history=node.Z_data,
            y_history=node.y_data,
            current_z=z,
            iteration=t,
        )

        return x, z, np.asarray(y_raw_values, dtype=float), np.asarray(y_noisy_values, dtype=float)


# =============================================================================
# Main optimizer
# =============================================================================

"""Main REMBO-UCB optimizer class.

The class coordinates the optimization loop, while reusable library concerns
live in separate modules:

* ``APREMBO.utils`` owns embedded-domain mappings and tree partitioning;
* ``APREMBO.Stopping_rules`` owns posterior and epsilon-delta PAC stopping;
* ``APREMBO.kernel`` owns all valid kernels through the GP wrapper.
"""

import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition import (
    qExpectedImprovement,
    qLogExpectedImprovement,
    qUpperConfidenceBound,
)

try:
    from .utils import (
        TreeNode,
        denormalize_from_minus_one_one,
        estimate_lipschitz_from_gp_mean,
        global_optimization,
        make_random_embedding_matrix,
        project_embedded_to_original,
        sample_embedded_uniform,
        select_best_kernel_by_mll,
        split_bounds_longest_dimension,
        z_in_bounds,
    )
    from .Stopping_rules import (
        posterior_stoppable_thresholds as compute_posterior_stoppable_thresholds,
        posterior_success_probability as compute_posterior_success_probability,
        posterior_summary as compute_posterior_summary,
        risk_tolerance_label as format_risk_tolerance_label,
        should_stop_epsilon_delta_pac,
    )
except ImportError:
    from utils import (
        TreeNode,
        denormalize_from_minus_one_one,
        estimate_lipschitz_from_gp_mean,
        global_optimization,
        make_random_embedding_matrix,
        project_embedded_to_original,
        sample_embedded_uniform,
        select_best_kernel_by_mll,
        split_bounds_longest_dimension,
        z_in_bounds,
    )
    from Stopping_rules import (
        posterior_stoppable_thresholds as compute_posterior_stoppable_thresholds,
        posterior_success_probability as compute_posterior_success_probability,
        posterior_summary as compute_posterior_summary,
        risk_tolerance_label as format_risk_tolerance_label,
        should_stop_epsilon_delta_pac,
    )
try:
    from live_gp_plotter import LiveGPPlotter
except ImportError:
    LiveGPPlotter = None

try:
    from rembo_diagnostics import (
        print_rembo_representability,
        print_rembo_z_star_objective_diagnostic,
    )
except ImportError:
    def print_rembo_representability(*args, **kwargs):
        return None

    def print_rembo_z_star_objective_diagnostic(*args, **kwargs):
        return None


PROGRESS_Y_MIN = -0.5
PROGRESS_Y_MAX = 0.5
PROGRESS_Y_DIVISIONS = 1000
PROGRESS_Y_STEP = (PROGRESS_Y_MAX - PROGRESS_Y_MIN) / PROGRESS_Y_DIVISIONS





Node = TreeNode


class APREMBO:
    def __init__(
        self,
        original_boundaries,
        d_embedding,
        max_iter,
        tau,
        N,
        v1=np.sqrt(5),
        v2=1.0,
        rho=0.5,
        L=None,
        delta=0.1,
        batch_size=1,
        n_init=None,
        show=False,
        split=True,
        acquisition_function=qLogExpectedImprovement,
        embedding_radius=None,
        a=0.0,
        b=1.0,
        R=0.01,
        scaled=True,
        optimum_x=None,
        lengthscale_lower_bound=0.01,
        lengthscale_upper_bound=50.0,
        kernel_name="rbf",
        auto_select_kernel=True,
        kernel_candidates=None,
        lipschitz_mode="predict",
        use_direct_cma_lengthscale=False,
        freeze_direct_cma_lengthscale=True,
        lengthscale_reoptimize_interval=20,
        t_sigma=0.002,
        exploitation_patience=5,
        lengthscale_bounds=None,
        D_min=1,
        effective_dim=None,
        active_indices=None,
        threshold_trace_risk_tolerances=None,
        verbose=False,
    ):
        self.real_original_boundaries = np.asarray(original_boundaries, dtype=float)
        self.normalized_original_boundaries = np.array(
            [[-1.0, 1.0]] * self.real_original_boundaries.shape[0],
            dtype=float,
        )
        self.d_orig = self.real_original_boundaries.shape[0]
        self.d_embedding = int(d_embedding)
        self.original_boundaries = torch.tensor(self.normalized_original_boundaries, dtype=torch.double)
        self.max_iter = int(max_iter)
        self.tau = tau
        self.N = N
        self.delta = delta
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.n_init = self.d_embedding if n_init is None else int(n_init)
        if self.n_init < 0:
            raise ValueError("n_init must be non-negative.")
        self.acquisition_function = acquisition_function
        self.v1 = v1
        self.v2 = v2
        self.rho = rho
        self.L_x = 1.0 if L is None else float(L)
        self.scaled = False
        self.a = float(a)
        self.b = float(b)
        self.R = float(R)
        self.kernel_name = str(kernel_name).lower()
        self.auto_select_kernel = bool(auto_select_kernel)
        if kernel_candidates is None:
            kernel_candidates = ("rbf", "matern12", "matern32", "matern52")
        self.kernel_candidates = tuple(str(name).lower() for name in kernel_candidates)
        self.selected_kernel_name = self.kernel_name
        self.selected_kernel_score = -float("inf")
        self.lipschitz_probe_count = 64
        self.lipschitz_mode = str(lipschitz_mode).lower()
        if self.lipschitz_mode not in {"predict", "user"}:
            raise ValueError("lipschitz_mode must be either 'predict' or 'user'.")
        if self.lipschitz_mode == "user" and L is None:
            raise ValueError("lipschitz_mode='user' requires a numeric L value.")
        self.use_direct_cma_lengthscale = bool(use_direct_cma_lengthscale)
        self.freeze_direct_cma_lengthscale = bool(freeze_direct_cma_lengthscale)
        self.domain_debug = os.environ.get("REMBO_DOMAIN_DEBUG", "").lower() in ("1", "true", "yes", "on")
        self.domain_debug_print_limit = int(os.environ.get("REMBO_DOMAIN_DEBUG_PRINT_LIMIT", "5"))
        self.domain_debug_print_count = 0
        self.optimum_x = optimum_x

        if lengthscale_bounds is not None:
            lengthscale_lower_bound, lengthscale_upper_bound = lengthscale_bounds
        self.lengthscale_lower_bound = float(lengthscale_lower_bound)
        self.lengthscale_upper_bound = float(lengthscale_upper_bound)

        self.t_sigma = float(t_sigma)
        self.exploitation_std_threshold = self.t_sigma
        self.exploitation_patience = int(exploitation_patience)
        self.exploitation_count = 0
        self.lengthscale_optimization_forced = True
        self.lengthscale_reoptimize_interval = int(lengthscale_reoptimize_interval)
        if self.lengthscale_reoptimize_interval <= 0:
            raise ValueError("lengthscale_reoptimize_interval must be positive.")
        self.current_algorithm3_iteration = 0
        self.last_lengthscale_optimization_iter = None
        self.lengthscale_optimization_forced = True
        self.cached_lengthscale = None
        self.D_min = int(D_min)

        if threshold_trace_risk_tolerances is None:
            threshold_trace_risk_tolerances = [self.delta]
        self.threshold_trace_risk_tolerances = [
            float(value) for value in threshold_trace_risk_tolerances
        ]

        self.embedding_radius = (
            np.sqrt(self.d_embedding)
            if embedding_radius is None
            else float(embedding_radius)
        )
        self.split = split

        self.A_np = make_random_embedding_matrix(self.d_orig, self.d_embedding)
        self.A = torch.tensor(self.A_np, dtype=torch.double)

        self.effective_dim = self.d_orig if effective_dim is None else int(effective_dim)
        self.active_indices = None if active_indices is None else np.asarray(active_indices, dtype=int)

        self.representability_info = print_rembo_representability(
            "APREMBO",
            self.A_np,
            self.real_original_boundaries,
            self.optimum_x,
            self.embedding_radius,
            effective_dim=self.effective_dim,
            active_indices=self.active_indices,
        )

        self.A_norm = float(np.linalg.norm(self.A.detach().cpu().numpy(), ord=2))
        self.L_emb = self.L_x * self.A_norm
        self.L = self.L_x
        self.l_emb = self.L_emb

        self.show = show
        self.verbose = bool(verbose)
        self.gp_debug = os.environ.get("GP_DEBUG", "").lower() in ("1", "true", "yes", "on")

        self.initialization_module = APREMBOInitialization(self)
        self.sampling_module = APREMBOSampling(self)
        self.refinement_module = APREMBORefinement(self)

        root_bounds = np.array([[-self.embedding_radius, self.embedding_radius]] * self.d_embedding)
        self.root = Node(0, 1, root_bounds, self.original_boundaries)

        self.history_t = []
        self.history_y = []
        self.history_best = []
        self.threshold_trace = []
        self.best_stoppable_threshold_by_delta = {}
        self.best_score = -float("inf")
        self.best_z = None
        self.best_x = None
        self.best_observed_score = float("nan")
        self.clip_ratios = []
        self.active_clip_ratios = []
        self.space_fig = None
        self.space_ax = None
        self.current_cell_bounds = None
        self.fig = None
        self.ax = None
        self.gp_plotter = None

        if self.show is not False:
            if LiveGPPlotter is None and self.d_embedding == 2:
                raise ImportError("show=True with d_embedding=2 requires live_gp_plotter.py.")
            plt.ion()
            if self.d_embedding == 2:
                self.fig, (self.gp_ax, self.ax) = plt.subplots(1, 2, figsize=(15, 7))
                self.gp_plotter = LiveGPPlotter(
                    title="REMBO-UCB",
                    embedding_bounds=root_bounds,
                    projection_matrix=self.A,
                    optimum_x=self.optimum_x,
                    fig=self.fig,
                    ax=self.gp_ax,
                )
            else:
                self.fig, self.ax = plt.subplots(figsize=(10, 5))

            self.fig.canvas.manager.set_window_title("REMBO-UCB")
            self._try_fullscreen()
            self.ax.set_title("Optimization Progress - REMBO-UCB", fontsize=14)
            self.ax.set_xlabel("Iteration (t)")
            self.ax.set_ylabel("Value")
            self.line_current, = self.ax.plot([], [], "ro", alpha=0.4, markersize=4, label="Current Values")
            self.line_best, = self.ax.plot([], [], "b-", linewidth=2, label="Best Value So Far")
            self.ax.axhline(y=self.tau, color="orange", linestyle="--", linewidth=1.5, label=f"Tau: {self.tau}")
            self._configure_progress_axis()
            self.ax.legend(loc="lower right")
            self.ax.grid(True, linestyle=":", alpha=0.7)
            self.status_text = self.ax.text(
                0.02,
                0.04,
                "",
                transform=self.ax.transAxes,
                ha="left",
                va="bottom",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
            )

    def _try_fullscreen(self):
        manager = getattr(self.fig.canvas, "manager", None)
        try:
            manager.full_screen_toggle()
        except Exception:
            try:
                manager.window.showMaximized()
            except Exception:
                pass

    def _configure_progress_axis(self):
        self.ax.set_ylim(PROGRESS_Y_MIN, PROGRESS_Y_MAX)
        self.ax.set_yticks(np.linspace(PROGRESS_Y_MIN, PROGRESS_Y_MAX, 11))
        self.ax.set_autoscaley_on(False)
        self.progress_y_step = PROGRESS_Y_STEP

    @staticmethod
    def as_float(value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().reshape(-1)[0].item())
        return float(np.asarray(value).reshape(-1)[0])

    def _to_real_domain(self, x_norm):
        return denormalize_from_minus_one_one(x_norm, self.real_original_boundaries)

    def _debug_evaluation_domains(self, x_norm, x_real):
        if not self.domain_debug:
            return
        x_norm = np.asarray(x_norm, dtype=float)
        x_real = np.asarray(x_real, dtype=float)
        lower = self.real_original_boundaries[:, 0]
        upper = self.real_original_boundaries[:, 1]
        assert np.all(x_norm >= -1.000001) and np.all(x_norm <= 1.000001)
        assert np.all(x_real >= lower - 1e-8) and np.all(x_real <= upper + 1e-8)
        if getattr(self, "show", False) and self.domain_debug_print_count < self.domain_debug_print_limit:
            print("x_norm min/max:", float(np.min(x_norm)), float(np.max(x_norm)))
            print("x_real min/max:", float(np.min(x_real)), float(np.max(x_real)))
            print("real bounds min/max:", float(lower.min()), float(upper.max()))
            self.domain_debug_print_count += 1

    def record_clip_ratio(self, x_norm):
        x_norm = np.asarray(x_norm, dtype=float).reshape(-1)
        clip_ratio = np.mean(np.isclose(np.abs(x_norm), 1.0, atol=1e-8))
        if self.active_indices is not None:
            active_x_norm = x_norm[self.active_indices]
        else:
            active_x_norm = x_norm[: self.effective_dim]
        active_clip_ratio = np.mean(np.isclose(np.abs(active_x_norm), 1.0, atol=1e-8))
        self.clip_ratios.append(float(clip_ratio))
        self.active_clip_ratios.append(float(active_clip_ratio))

    def print_clip_ratio_summary(self):
        if not getattr(self, "show", False):
            return
        if not self.clip_ratios:
            return
        print(
            f"[ClipRatio] mean_clip_ratio={np.mean(self.clip_ratios):.4f}, "
            f"mean_active_clip_ratio={np.mean(self.active_clip_ratios):.4f}, "
            f"max_active_clip_ratio={np.max(self.active_clip_ratios):.4f}"
        )

    def print_representability_objective_summary(self, black_box_function):
        if not getattr(self, "show", False):
            return
        print_rembo_z_star_objective_diagnostic(
            "APREMBO",
            black_box_function,
            self.A_np,
            self.real_original_boundaries,
            self.representability_info,
            best_score=self.best_score,
            best_z=self.best_z,
        )

    def _update_plot(self, t, y_curr, probability=None):
        if self.show is False:
            return
        self.history_t.append(t)
        self.history_y.append(y_curr)
        self.history_best.append(max(self.history_y))
        self.line_current.set_data(self.history_t, self.history_y)
        self.line_best.set_data(self.history_t, self.history_best)
        prob_label = "n/a" if probability is None else f"{float(probability):.4f}"
        self.status_text.set_text(f"Current value: {float(y_curr):.4g}\nCurrent prob: {prob_label}")
        self.ax.relim()
        self.ax.autoscale_view(scalex=True, scaley=False)
        self.ax.set_ylim(PROGRESS_Y_MIN, PROGRESS_Y_MAX)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def has_enough_iterations_to_stop(self, iteration):
        return int(iteration) >= self.d_embedding

    def confidence_width(self, count, t=None):
        if count <= 0:
            return float("inf")
        horizon = max(float(self.max_iter), 2.0)
        confidence_scale = math.sqrt(2.0 * math.log(horizon) / count)
        return (self.b - self.a) * confidence_scale + self.R * confidence_scale

    def ucb_node_bound(self, node):
        if node.C <= 0:
            node.uncertainty = float("inf")
            node.B = float("inf")
            return node.B

        node.uncertainty = self.confidence_width(node.C)
        own_bound = node.mu_hat + node.uncertainty
        parent = getattr(node, "parent", None)
        if parent is None:
            node.B = own_bound
        else:
            parent_bound = self.ucb_node_bound(parent)
            node.B = min(
                own_bound,
                parent_bound + self.L_emb * self.v1 * (self.rho ** (node.h - 1)),
            )
        return node.B

    @staticmethod
    def risk_tolerance_label(value):
        return format_risk_tolerance_label(value)

    def posterior_stoppable_thresholds(self, model, z_one):
        return compute_posterior_stoppable_thresholds(
            model,
            z_one,
            self.threshold_trace_risk_tolerances,
        )

    def posterior_summary(self, model, z_one):
        return compute_posterior_summary(model, z_one)

    @staticmethod
    def trace_point(value):
        if value is None:
            return None
        return np.asarray(value, dtype=float).reshape(-1).tolist()

    def record_probability_threshold_trace(
        self,
        iteration,
        model,
        z_one,
        y_raw,
        y_noisy,
        current_node,
        x_one=None,
    ):
        posterior_mean, posterior_std, posterior_variance = self.posterior_summary(model, z_one)
        current_thresholds = self.posterior_stoppable_thresholds(model, z_one)
        best_thresholds = {}
        for label, threshold in current_thresholds.items():
            previous = self.best_stoppable_threshold_by_delta.get(label, -float("inf"))
            best = max(float(previous), float(threshold))
            self.best_stoppable_threshold_by_delta[label] = best
            best_thresholds[label] = best

        self.threshold_trace.append(
            {
                "iteration": int(iteration),
                "posterior_mean": posterior_mean,
                "posterior_std": posterior_std,
                "posterior_variance": posterior_variance,
                "gp_mean": posterior_mean,
                "gp_variance": posterior_variance,
                "true_score": float(y_raw),
                "observed_score": float(y_noisy),
                "best_true_score": float(self.best_score),
                "recommended_true_score": float(self.best_score),
                "recommended_observed_score": float(self.best_observed_score),
                "x": self.trace_point(x_one),
                "z": self.trace_point(z_one),
                "recommended_x": self.trace_point(self.best_x),
                "recommended_z": self.trace_point(self.best_z),
                "node_depth": int(current_node.h),
                "node_index": int(current_node.i),
                "current_stoppable_threshold_by_delta": current_thresholds,
                "best_stoppable_threshold_by_delta": best_thresholds,
            }
        )

    def sample_noise(self):
        return np.random.normal(0.0, self.R)

    def noise_variance(self, count):
        return torch.full((int(count), 1), self.R ** 2, dtype=torch.double)

    def get_best_fitted_model(self, train_X, train_Y, train_Yvar, ard_num_dims):
        optimize_lengthscale_now = self.should_optimize_lengthscale()
        projection_matrix = self.A.detach().cpu().numpy() if isinstance(self.A, torch.Tensor) else self.A

        def fit_kernel(kernel_name):
            return build_gp_model(
                train_x=train_X.double(),
                train_y=train_Y.double(),
                train_yvar=train_Yvar.double() if train_Yvar is not None else None,
                R=self.R,
                kernel_name=kernel_name,
                lengthscale_bounds=self.lengthscale_bounds,
                projection_matrix=projection_matrix,
                original_bounds=self.normalized_original_boundaries,
                n_keep_dims=0,
                use_direct_cma_lengthscale=optimize_lengthscale_now,
                freeze_direct_cma_lengthscale=self.freeze_direct_cma_lengthscale,
                fit=True,
            )

        if self.auto_select_kernel:
            model, kernel_name, kernel_score = select_best_kernel_by_mll(
                self.kernel_candidates,
                fit_kernel,
            )
            self.selected_kernel_name = kernel_name
            self.selected_kernel_score = float(kernel_score)
        else:
            model = fit_kernel(self.kernel_name)
            self.selected_kernel_name = self.kernel_name
            self.selected_kernel_score = float(getattr(model, "kernel_fit_score", -float("inf")))

        if optimize_lengthscale_now:
            self.mark_lengthscale_optimization_complete(model)
        elif self.cached_lengthscale is not None:
            try:
                set_scalar_lengthscale(model, self.cached_lengthscale)
            except Exception:
                pass
        return model

    def update_lipschitz_from_model(self, model, node):
        if self.lipschitz_mode == "user":
            if node is not None:
                node.L_hat = self.L_emb
            return self.L_emb
        if model is None or node is None:
            return self.L_emb
        candidates = None
        if torch.is_tensor(getattr(node, "Z_data", None)) and node.Z_data.numel() > 0:
            candidates = node.Z_data.detach().cpu().numpy()
        L_emb = estimate_lipschitz_from_gp_mean(
            model,
            node.bounds_emb,
            candidate_points=candidates,
            n_random=self.lipschitz_probe_count,
        )
        self.L_emb = float(L_emb)
        self.l_emb = self.L_emb
        self.L = self.L_emb
        self.L_x = self.L_emb / max(float(self.A_norm), 1e-12)
        node.L_hat = self.L_emb
        return self.L_emb

    def fit_global_model_for_node(self, node):
        raise RuntimeError("REMBO-UCB uses local GPs only; fit the selected node with fit_local_model_for_node().")

    def fit_local_model_for_node(self, node):
        if not getattr(node, "local_dataset", None):
            node.model = None
            return None
        if not torch.is_tensor(node.Z_data) or not torch.is_tensor(node.y_data):
            self.rebuild_node_tensors(node)
        noise = self.noise_variance(node.y_data.shape[0])
        node.model = self.get_best_fitted_model(
            node.Z_data,
            node.y_data,
            noise,
            self.d_embedding,
        )
        node.probability_model = node.model
        node.gp_kernel_name = self.selected_kernel_name
        node.gp_kernel_score = self.selected_kernel_score
        self.update_lipschitz_from_model(node.model, node)
        self.ucb_node_bound(node)
        return node.model

    def fit_probability_model_for_node(self, node):
        if not getattr(node, "local_dataset", None):
            node.probability_model = None
            return None
        if not torch.is_tensor(node.Z_data) or not torch.is_tensor(node.y_data):
            self.rebuild_node_tensors(node)
        noise = self.noise_variance(node.y_data.shape[0])
        node.probability_model = self.get_best_fitted_model(
            node.Z_data,
            node.y_data,
            noise,
            self.d_embedding,
        )
        node.gp_kernel_name = self.selected_kernel_name
        node.gp_kernel_score = self.selected_kernel_score
        self.update_lipschitz_from_model(node.probability_model, node)
        self.ucb_node_bound(node)
        return node.probability_model

    @property
    def lengthscale_bounds(self):
        return (self.lengthscale_lower_bound, self.lengthscale_upper_bound)

    def should_optimize_lengthscale(self):
        """
        REMBO Algorithm-3 scheduling rule for lengthscale optimization.

        The bounded marginal-likelihood search is run on the first fit, every
        `lengthscale_reoptimize_interval` iterations, and whenever the upper
        bound U has been tightened after repeated low posterior standard
        deviation events.
        """
        if not self.use_direct_cma_lengthscale:
            return False

        iteration = int(getattr(self, "current_algorithm3_iteration", 0))
        if self.cached_lengthscale is None:
            return True
        if self.lengthscale_optimization_forced:
            return True
        if self.last_lengthscale_optimization_iter == iteration:
            return False
        return iteration % self.lengthscale_reoptimize_interval == 0

    def mark_lengthscale_optimization_complete(self, model):
        """Store the selected scalar lengthscale and update Algorithm-3 state."""
        self.cached_lengthscale = self.current_model_lengthscale(model)
        self.last_lengthscale_optimization_iter = int(
            getattr(self, "current_algorithm3_iteration", 0)
        )
        self.lengthscale_optimization_forced = False

    @staticmethod
    def current_model_lengthscale(model):
        def find_lengthscale_kernel(module):
            if module is None:
                return None

            lengthscale = getattr(module, "lengthscale", None)
            if lengthscale is not None:
                return module

            base_kernel = getattr(module, "base_kernel", None)
            if base_kernel is not None:
                found = find_lengthscale_kernel(base_kernel)
                if found is not None:
                    return found

            for child in module.children():
                found = find_lengthscale_kernel(child)
                if found is not None:
                    return found
            return None

        kernel = find_lengthscale_kernel(getattr(model, "covar_module", None))
        if kernel is None or getattr(kernel, "lengthscale", None) is None:
            return None
        return float(kernel.lengthscale.detach().double().mean().cpu().item())

    def posterior_latent_std(self, model, z):
        with torch.no_grad():
            post = model.posterior(torch.as_tensor(z, dtype=torch.double))
            variance = torch.clamp(post.variance.reshape(-1), min=1e-12)
            return float(torch.sqrt(variance).max().detach().cpu().item())

    def update_wang_lengthscale_upper_bound(self, model, z):
        latent_std = self.posterior_latent_std(model, z)
        if latent_std >= self.exploitation_std_threshold:
            self.exploitation_count = 0
            return

        self.exploitation_count += 1
        if self.exploitation_count < self.exploitation_patience:
            return

        current_lengthscale = self.current_model_lengthscale(model)
        if current_lengthscale is not None and math.isfinite(current_lengthscale):
            new_upper_bound = max(
                0.9 * current_lengthscale,
                self.lengthscale_lower_bound * 1.01,
            )
            if new_upper_bound < self.lengthscale_upper_bound:
                self.lengthscale_upper_bound = new_upper_bound
                self.lengthscale_optimization_forced = True
        self.exploitation_count = 0

    def update_gp_plot(self, model, z_history, y_history, current_z, iteration, cell_bounds=None):
        if self.gp_plotter is None:
            return
        if cell_bounds is None:
            cell_bounds = self.current_cell_bounds
        self.gp_plotter.update(
            model=model,
            z_history=z_history,
            y_history=y_history,
            current_z=current_z,
            iteration=iteration,
            cell_bounds=cell_bounds,
        )

    def cell_plot_record(self, node):
        return {
            "bounds": node.bounds_emb,
            "b_value": getattr(node, "B", None),
            "c_value": getattr(node, "C", None),
            "uncertainty_value": getattr(node, "uncertainty", None),
            "l_rho_h_value": self.L_emb * (self.rho ** node.h),
        }

    def embedded_to_original(self, z):
        """Map embedded z-space points to clipped normalized original x-space."""
        return project_embedded_to_original(
            z,
            self.A_np,
            self.normalized_original_boundaries,
        )

    def _acquisition_name(self):
        return getattr(self.acquisition_function, "__name__", str(self.acquisition_function)).lower()

    def build_acquisition(self, model, best_f):
        """Build the configured BoTorch acquisition function for a local model."""
        acquisition_name = self._acquisition_name()
        if (
            self.acquisition_function == qLogExpectedImprovement
            or "logexpectedimprovement" in acquisition_name
            or "qlogei" in acquisition_name
        ):
            return qLogExpectedImprovement(model=model, best_f=best_f)
        if (
            self.acquisition_function == qExpectedImprovement
            or "expectedimprovement" in acquisition_name
            or "qei" in acquisition_name
        ):
            return qExpectedImprovement(model=model, best_f=best_f)
        if (
            self.acquisition_function == qUpperConfidenceBound
            or "upperconfidencebound" in acquisition_name
            or "ucb" in acquisition_name
        ):
            return qUpperConfidenceBound(model=model, beta=2.0)
        return qLogExpectedImprovement(model=model, best_f=best_f)

    def uniform_query_point(self, node, batch_size=1):
        bounds = np.asarray(node.bounds_emb, dtype=float)
        z_np = sample_embedded_uniform(bounds, batch_size)
        z = torch.as_tensor(z_np, dtype=torch.double)
        x = self.embedded_to_original(z_np)
        return x, z

    def oracle_query_point(self, node, batch_size=1):
        if node.model is None or not torch.is_tensor(node.y_data) or node.y_data.numel() == 0:
            return self.uniform_query_point(node, batch_size=batch_size)

        best_f = float(torch.max(node.y_data.to(torch.double)).detach().cpu().item())
        acquisition = self.build_acquisition(node.model, best_f=best_f)
        z = global_optimization(
            objective_function=acquisition,
            boundaries=np.asarray(node.bounds_emb, dtype=float),
            batch_size=batch_size,
        )
        x = self.embedded_to_original(z.detach().cpu().numpy())
        return x, z

    def rebuild_node_tensors(self, node):
        node.X_data = torch.cat([torch.as_tensor(d[0]).double().view(1, -1) for d in node.local_dataset], dim=0)
        node.Z_data = torch.cat([torch.as_tensor(d[1]).double().view(1, -1) for d in node.local_dataset], dim=0)
        node.y_data = torch.tensor([[d[2]] for d in node.local_dataset], dtype=torch.double)

    def posterior_success_probability(self, model, z, thresholds=None):
        thresholds = [self.tau] if thresholds is None else [float(value) for value in thresholds]
        probabilities = compute_posterior_success_probability(model, z, thresholds)
        if len(thresholds) == 1:
            probabilities = probabilities.reshape(-1)
        if np.asarray(probabilities).ndim == 1 and np.asarray(probabilities).shape[0] == 1:
            return float(probabilities[0])
        return probabilities

    def epsilon_delta_pac_stop(self, model, candidate_points, epsilon, delta, point_t):
        """Evaluate the epsilon-delta PAC stopping rule for candidate points."""
        return should_stop_epsilon_delta_pac(
            model=model,
            candidate_points=candidate_points,
            best_observed=self.best_score,
            epsilon=epsilon,
            delta=delta,
            min_iterations_met=self.has_enough_iterations_to_stop(point_t),
        )

    def _empty_threshold_delta_results(self, thresholds, risk_tolerances):
        return {
            float(threshold): {float(delta): None for delta in risk_tolerances}
            for threshold in thresholds
        }

    def _finalize_threshold_delta_results(self, results, best_value, max_iter, wall_clock_seconds=None):
        for _threshold, delta_results in results.items():
            for delta, entry in list(delta_results.items()):
                if entry is None:
                    delta_results[delta] = {
                        "stopped": False,
                        "success": False,
                        "value": float(best_value),
                        "iterations": int(max_iter),
                        "probability": 0.0,
                        "wall_clock_seconds": wall_clock_seconds,
                    }
        return results

    def split_ucb_node(self, node, N, active_leaves, tree):
        self.exploitation_count = 0
        bounds = node.bounds_emb
        child_bounds_list, longest_dim = split_bounds_longest_dimension(bounds, N)
        children = []

        for j, child_bounds in enumerate(child_bounds_list, start=1):
            child = Node(node.h + 1, (node.i - 1) * N + j, child_bounds, self.original_boundaries)
            child.parent = node
            child.local_dataset = [
                item for item in node.local_dataset
                if z_in_bounds(torch.as_tensor(item[1]).flatten(), child_bounds, dim=longest_dim)
            ]
            if child.local_dataset:
                self.rebuild_node_tensors(child)
                child.C = len(child.local_dataset)
                child.mu_hat = child.y_data.mean().item()
                self.ucb_node_bound(child)
                self.fit_local_model_for_node(child)
            else:
                child.mu_hat = float("inf")
                child.uncertainty = self.confidence_width(0)
                child.B = float("inf")
                child.model = None
                child.probability_model = None
            children.append(child)

        node.children = children
        active_leaves.remove(node)
        active_leaves.extend(children)
        tree.extend(children)
        return children

    def refine_active_node_after_observation(self, node, active_leaves, tree, t):
        node.uncertainty = self.confidence_width(node.C)
        if self.refinement_module.should_split(node):
            self.refinement_module.split_node(node, active_leaves, tree, t)
        else:
            self.refinement_module.update_b_values(active_leaves, t=t)

    def _verbose_start(self):
        if not self.verbose:
            return
        fbest = self.best_score if np.isfinite(self.best_score) else self.b
        print(f"Using dtype = {self.A.dtype}")
        print(f"Using device = {self.A.device}")
        print(f"Starting from fbest = {fbest}")

    def _verbose_iteration(self, iteration, current_value):
        if not self.verbose:
            return
        best_value = self.best_score if np.isfinite(self.best_score) else current_value
        print(
            f"iteration: {int(iteration)} "
            f"Best Value: {float(best_value):.6g} "
            f"Current Value: {float(current_value):.6g}"
        )

    def optimize(
        self,
        black_box_function,
        risk_tolerances=None,
        threshold_values=None,
        pac_epsilon=None,
        pac_delta=None,
    ):
        tree, active_leaves = self.initialization_module.initialize_active_tree(self.root)

        multi_delta_results = None
        pending_deltas = None
        multi_threshold_results = None
        pending_pairs = None
        run_start = time.perf_counter()
        thresholds = [float(self.tau)] if threshold_values is None else [float(value) for value in threshold_values]

        if risk_tolerances is not None and threshold_values is None:
            pending_deltas = sorted({float(delta) for delta in risk_tolerances}, reverse=True)
            multi_delta_results = {}
        elif risk_tolerances is not None:
            risk_tolerances = sorted({float(delta) for delta in risk_tolerances}, reverse=True)
            multi_threshold_results = self._empty_threshold_delta_results(thresholds, risk_tolerances)
            pending_pairs = {
                (float(threshold), float(delta))
                for threshold in thresholds
                for delta in risk_tolerances
            }

        t = 0
        last_observation = None
        self._verbose_start()

        # Initial design: collect n_init LHS samples while still refining the active tree.
        while t < min(self.n_init, self.max_iter):
            current_node = self.initialization_module.select_active_leaf(active_leaves)
            self.current_cell_bounds = [self.cell_plot_record(leaf) for leaf in active_leaves]
            current_batch_size = min(self.batch_size, self.n_init - t, self.max_iter - t)
            x, z, y_raw_batch, y_batch = self.sampling_module.sample_observe_and_update(
                current_node,
                black_box_function,
                t,
                batch_size=current_batch_size,
                force_uniform=True,
            )

            model = current_node.probability_model or current_node.model
            if model is None:
                probabilities = np.zeros(current_batch_size, dtype=float)
            else:
                probabilities = np.asarray(self.posterior_success_probability(model, z), dtype=float).reshape(-1)

            for batch_index, probability in enumerate(probabilities):
                point_t = t + batch_index + 1
                y_raw = float(y_raw_batch[batch_index])
                y = float(y_batch[batch_index])
                last_observation = (y_raw, y)
                if y_raw > self.best_score:
                    self.best_score = y_raw
                    self.best_observed_score = y
                    self.best_x = torch.as_tensor(x[batch_index]).detach().cpu().numpy().reshape(-1)
                    self.best_z = torch.as_tensor(z[batch_index]).detach().cpu().numpy().reshape(-1)
                self._verbose_iteration(point_t, y_raw)
                self.record_probability_threshold_trace(
                    point_t,
                    model,
                    z[batch_index : batch_index + 1],
                    y_raw,
                    y,
                    current_node,
                    x_one=x[batch_index : batch_index + 1],
                )
                self._update_plot(point_t, y, probability)

            t += current_batch_size
            self.refine_active_node_after_observation(current_node, active_leaves, tree, t)

        while t < self.max_iter:
            current_node = self.initialization_module.select_active_leaf(active_leaves)
            self.current_cell_bounds = [self.cell_plot_record(leaf) for leaf in active_leaves]
            current_batch_size = min(self.batch_size, self.max_iter - t)
            x, z, y_raw_batch, y_batch = self.sampling_module.sample_observe_and_update(
                current_node,
                black_box_function,
                t,
                batch_size=current_batch_size,
            )

            model = current_node.probability_model or current_node.model
            if model is None:
                probabilities = np.zeros(current_batch_size, dtype=float)
                threshold_probabilities = np.zeros((len(thresholds), current_batch_size), dtype=float)
            else:
                probabilities = np.asarray(self.posterior_success_probability(model, z), dtype=float).reshape(-1)
                threshold_probabilities = np.asarray(
                    self.posterior_success_probability(model, z, thresholds=thresholds),
                    dtype=float,
                )
                if threshold_probabilities.ndim == 1:
                    threshold_probabilities = threshold_probabilities.reshape(1, -1)

            stop_result = None

            for batch_index, probability in enumerate(probabilities):
                point_t = t + batch_index + 1
                y_raw = float(y_raw_batch[batch_index])
                y = float(y_batch[batch_index])
                last_observation = (y_raw, y)

                if y_raw > self.best_score:
                    self.best_score = y_raw
                    self.best_observed_score = y
                    self.best_x = torch.as_tensor(x[batch_index]).detach().cpu().numpy().reshape(-1)
                    self.best_z = torch.as_tensor(z[batch_index]).detach().cpu().numpy().reshape(-1)
                self._verbose_iteration(point_t, y_raw)

                self.record_probability_threshold_trace(
                    point_t,
                    model,
                    z[batch_index : batch_index + 1],
                    y_raw,
                    y,
                    current_node,
                    x_one=x[batch_index : batch_index + 1],
                )
                self._update_plot(point_t, y, probability)

                if (
                    probability >= 1.0 - self.delta
                    and self.has_enough_iterations_to_stop(point_t)
                    and stop_result is None
                    and multi_delta_results is None
                    and multi_threshold_results is None
                ):
                    stop_result = (
                        True,
                        y_raw > self.tau,
                        y_raw,
                        y,
                        current_node.h,
                        current_node.i,
                        self.tau,
                        point_t,
                    )

                if (
                    pac_epsilon is not None
                    and pac_delta is not None
                    and stop_result is None
                    and multi_delta_results is None
                    and multi_threshold_results is None
                ):
                    pac_stop, pac_info = self.epsilon_delta_pac_stop(
                        model,
                        z[batch_index : batch_index + 1],
                        epsilon=pac_epsilon,
                        delta=pac_delta,
                        point_t=point_t,
                    )
                    if pac_stop:
                        stop_result = (
                            True,
                            bool(self.best_score >= self.tau),
                            y_raw,
                            y,
                            current_node.h,
                            current_node.i,
                            self.tau,
                            point_t,
                            {"stopping_rule": "epsilon_delta_pac", **pac_info},
                        )

                if multi_delta_results is not None and self.has_enough_iterations_to_stop(point_t):
                    for delta_value in list(pending_deltas):
                        if probability >= 1.0 - delta_value:
                            pending_deltas.remove(delta_value)
                            multi_delta_results[delta_value] = {
                                "result": (
                                    True,
                                    y_raw > self.tau,
                                    y_raw,
                                    y,
                                    current_node.h,
                                    current_node.i,
                                    self.tau,
                                    point_t,
                                ),
                                "wall_clock_seconds": time.perf_counter() - run_start,
                            }

                if multi_threshold_results is not None and self.has_enough_iterations_to_stop(point_t):
                    for threshold_index, threshold in enumerate(thresholds):
                        threshold_probability = float(threshold_probabilities[threshold_index, batch_index])
                        matching_deltas = [
                            delta_value
                            for pair_threshold, delta_value in pending_pairs
                            if pair_threshold == threshold
                        ]
                        for delta_value in matching_deltas:
                            if threshold_probability >= 1.0 - delta_value:
                                pending_pairs.remove((threshold, delta_value))
                                multi_threshold_results[threshold][delta_value] = {
                                    "stopped": True,
                                    "success": bool(self.best_score >= threshold),
                                    "value": float(self.best_score),
                                    "iterations": int(point_t),
                                    "probability": threshold_probability,
                                    "wall_clock_seconds": time.perf_counter() - run_start,
                                }

            t += current_batch_size

            if multi_delta_results is not None and not pending_deltas:
                if self.show is not None:
                    plt.ioff()
                plt.close("all")
                self.print_representability_objective_summary(black_box_function)
                self.print_clip_ratio_summary()
                return multi_delta_results

            if multi_threshold_results is not None and not pending_pairs:
                if self.show is not None:
                    plt.ioff()
                plt.close("all")
                self.print_representability_objective_summary(black_box_function)
                self.print_clip_ratio_summary()
                return multi_threshold_results

            if stop_result is not None:
                if self.show is not None:
                    plt.ioff()
                plt.close("all")
                self.print_representability_objective_summary(black_box_function)
                self.print_clip_ratio_summary()
                return stop_result

            self.refine_active_node_after_observation(current_node, active_leaves, tree, t)

        y_raw, y = last_observation if last_observation is not None else (float("nan"), float("nan"))
        self.print_representability_objective_summary(black_box_function)
        self.print_clip_ratio_summary()

        if multi_delta_results is not None:
            for delta_value in pending_deltas:
                multi_delta_results[delta_value] = {
                    "result": (False, False, y_raw, y, 0, 1, self.tau, self.max_iter),
                    "wall_clock_seconds": time.perf_counter() - run_start,
                }
            return multi_delta_results

        if multi_threshold_results is not None:
            best_value = self.best_score if np.isfinite(self.best_score) else y_raw
            return self._finalize_threshold_delta_results(
                multi_threshold_results,
                best_value,
                self.max_iter,
                time.perf_counter() - run_start,
            )

        return False, False, y_raw, y, 0, 1, self.tau, self.max_iter

    def fit(self, black_box_function, **kwargs):
        """Library-style alias for ``optimize``."""
        return self.optimize(black_box_function, **kwargs)
