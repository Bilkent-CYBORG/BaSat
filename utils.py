"""Utility functions for Bayesian-optimization algorithms.

This module contains generic sampling helpers, domain normalization helpers,
embedded-to-original-space mappings, and discrete tree partition utilities.
"""

import numpy as np
import torch
import warnings


SUPPORTED_SAMPLING_METHODS = {
    "latin_hypercube",
    "lhs",
    "uniform",
    "random",
    "sobol",
    "center",
    "grid",
}


def get_rng(rng=None, seed=None):
    """Return a NumPy random generator-like object."""
    if rng is not None:
        return rng
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed)


def set_torch_seed(seed=None):
    """Set torch seed when a seed is provided."""
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def validate_bounds(bounds):
    """Validate and return bounds as a float array with shape (dim, 2)."""
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (dim, 2).")

    lower = bounds[:, 0]
    upper = bounds[:, 1]
    if np.any(lower >= upper):
        raise ValueError("Each lower bound must be smaller than its upper bound.")

    return bounds


def bounds_to_botorch(bounds, dtype=torch.double, device=None):
    """Convert bounds from shape (dim, 2) to BoTorch shape (2, dim)."""
    bounds = validate_bounds(bounds)
    lower = torch.as_tensor(bounds[:, 0], dtype=dtype, device=device)
    upper = torch.as_tensor(bounds[:, 1], dtype=dtype, device=device)
    return torch.stack([lower, upper], dim=0)


def clip_to_bounds(x, bounds):
    """Clip NumPy points to box bounds."""
    bounds = validate_bounds(bounds)
    x = np.asarray(x, dtype=float)
    return np.clip(x, bounds[:, 0], bounds[:, 1])


def scale_unit_samples(unit_samples, bounds):
    """Map samples from [0, 1]^d to the provided box bounds."""
    bounds = validate_bounds(bounds)
    unit_samples = np.asarray(unit_samples, dtype=float)

    if unit_samples.ndim != 2:
        raise ValueError("unit_samples must have shape (n_points, dim).")
    if unit_samples.shape[1] != bounds.shape[0]:
        raise ValueError("unit_samples dimension does not match bounds dimension.")

    return bounds[:, 0] + unit_samples * (bounds[:, 1] - bounds[:, 0])


def normalize_to_unit_cube(x, bounds):
    """Map points from box bounds to [0, 1]^d."""
    bounds = validate_bounds(bounds)
    x = np.asarray(x, dtype=float)
    return (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def unnormalize_from_unit_cube(x_unit, bounds):
    """Map points from [0, 1]^d to box bounds."""
    return scale_unit_samples(np.asarray(x_unit, dtype=float), bounds)


def normalize_to_minus_one_one(x, bounds):
    """Map points from real box bounds to the normalized domain [-1, 1]^D."""
    bounds = validate_bounds(bounds)
    x = np.asarray(x, dtype=float)
    return 2.0 * (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) - 1.0


def denormalize_from_minus_one_one(x_norm, bounds):
    """Map normalized points in [-1, 1]^D back to real box bounds."""
    bounds = validate_bounds(bounds)
    x_norm = np.asarray(x_norm, dtype=float)
    return bounds[:, 0] + 0.5 * (x_norm + 1.0) * (bounds[:, 1] - bounds[:, 0])


def make_random_embedding_matrix(
    original_dim,
    embedded_dim,
    rng=None,
    seed=None,
    normalize_columns=False,
):
    """Create a Gaussian projection matrix A with shape (D, d)."""
    original_dim = int(original_dim)
    embedded_dim = int(embedded_dim)
    if original_dim <= 0 or embedded_dim <= 0:
        raise ValueError("original_dim and embedded_dim must be positive.")
    if embedded_dim > original_dim:
        raise ValueError("embedded_dim must be less than or equal to original_dim.")

    rng = get_rng(rng=rng, seed=seed)
    A = rng.normal(0.0, 1.0, size=(original_dim, embedded_dim))

    if normalize_columns:
        norms = np.linalg.norm(A, axis=0, keepdims=True)
        A = A / np.maximum(norms, 1e-12)

    return A


def embed_original_to_embedded(x, projection_matrix, method="least_squares"):
    """Map original-space points x into embedded coordinates z.

    For a REMBO projection x ~= A z, the inverse is generally not exact because
    A maps a low-dimensional z into a high-dimensional x.  This helper returns
    the least-squares embedded coordinate z that best reconstructs x through A.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    A = np.asarray(projection_matrix, dtype=float)
    if A.ndim != 2:
        raise ValueError("projection_matrix must have shape (original_dim, embedded_dim).")
    if x.shape[1] != A.shape[0]:
        raise ValueError("x dimension must match projection_matrix original_dim.")

    method = str(method).lower()
    if method in {"least_squares", "lstsq"}:
        z, *_ = np.linalg.lstsq(A, x.T, rcond=None)
        return z.T
    if method in {"pinv", "pseudo_inverse", "pseudoinverse"}:
        return x @ np.linalg.pinv(A).T

    raise ValueError("method must be one of: least_squares, lstsq, pinv, pseudo_inverse.")


project_original_to_embedded = embed_original_to_embedded


def project_embedded_to_original(
    z,
    projection_matrix,
    original_bounds,
    n_keep_dims=0,
    return_clip_mask=False,
):
    """Project embedded z-space points into the clipped normalized original domain."""
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(1, -1)

    A = np.asarray(projection_matrix, dtype=float)
    z_projected = z[:, int(n_keep_dims) :]
    x_projected = z_projected @ A.T

    if int(n_keep_dims) > 0:
        x_keep = z[:, : int(n_keep_dims)]
        x_projected = np.hstack((x_keep, x_projected))

    original_bounds = validate_bounds(original_bounds)
    lower = original_bounds[:, 0]
    upper = original_bounds[:, 1]
    clip_mask = (x_projected < lower) | (x_projected > upper)
    clipped = np.clip(x_projected, lower, upper)

    if return_clip_mask:
        return clipped, clip_mask
    return clipped


def to_numpy(x):
    """Convert tensor-like input to a detached CPU NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_tensor(x, dtype=torch.double, device=None):
    """Convert array-like input to a torch tensor."""
    if isinstance(x, torch.Tensor):
        return x.detach().clone().to(dtype=dtype, device=device)
    return torch.as_tensor(x, dtype=dtype, device=device)


def ensure_2d_tensor(x, dtype=torch.double, device=None):
    """Convert input to a 2D torch tensor."""
    tensor = to_tensor(x, dtype=dtype, device=device)
    if tensor.ndim == 1:
        tensor = tensor.view(1, -1)
    if tensor.ndim != 2:
        raise ValueError("Input must be convertible to a 2D tensor.")
    return tensor


def select_best_kernel_by_mll(kernel_names, model_builder, score_getter=None):
    """Fit candidate GP kernels and return the one with best marginal likelihood.

    Parameters
    ----------
    kernel_names:
        Iterable of kernel-name strings to try.
    model_builder:
        Callable receiving ``kernel_name`` and returning a fitted model.
    score_getter:
        Optional callable receiving the fitted model and returning a scalar
        score.  By default, ``model.kernel_fit_score`` is used.

    Returns
    -------
    tuple
        ``(best_model, best_kernel_name, best_score)``.
    """
    best_model = None
    best_kernel_name = None
    best_score = -float("inf")
    errors = []

    for kernel_name in kernel_names:
        try:
            model = model_builder(str(kernel_name).lower())
            if score_getter is None:
                score = getattr(model, "kernel_fit_score", None)
            else:
                score = score_getter(model)
            score = float(score)
            if not np.isfinite(score):
                score = -float("inf")
        except Exception as exc:
            errors.append(f"{kernel_name}: {exc}")
            continue

        if best_model is None or score > best_score:
            best_model = model
            best_kernel_name = str(kernel_name).lower()
            best_score = score

    if best_model is None:
        detail = "; ".join(errors) if errors else "no candidate kernels were provided"
        raise RuntimeError(f"All candidate kernels failed during GP fitting: {detail}")

    return best_model, best_kernel_name, best_score


def _posterior_mean_gradient_norm(model, point):
    """Return ||grad mean(point)|| for a fitted BoTorch/GPyTorch model."""
    z = torch.as_tensor(point, dtype=torch.double).view(1, -1).clone().detach()
    z.requires_grad_(True)

    model.eval()
    if hasattr(model, "likelihood"):
        model.likelihood.eval()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The input matches the stored training data.*")
        mean = model.posterior(z).mean.reshape(-1).sum()
    grad = torch.autograd.grad(mean, z, retain_graph=False, create_graph=False)[0]
    value = torch.linalg.norm(grad.reshape(-1), ord=2)
    return float(value.detach().cpu().item())


def _posterior_mean_gradient_norm_finite_difference(model, point, bounds, eps=1e-4):
    """Finite-difference fallback for models whose posterior is not autograd-safe."""
    point = np.asarray(point, dtype=float).reshape(-1)
    bounds = validate_bounds(bounds)
    eps = float(eps)
    grad = np.zeros(point.shape[0], dtype=float)

    def mean_at(x):
        tensor = torch.as_tensor(x, dtype=torch.double).view(1, -1)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The input matches the stored training data.*")
            return float(model.posterior(tensor).mean.reshape(-1)[0].detach().cpu().item())

    for dim_index in range(point.shape[0]):
        step = eps * max(1.0, bounds[dim_index, 1] - bounds[dim_index, 0])
        left = point.copy()
        right = point.copy()
        left[dim_index] = max(bounds[dim_index, 0], left[dim_index] - step)
        right[dim_index] = min(bounds[dim_index, 1], right[dim_index] + step)
        width = right[dim_index] - left[dim_index]
        if width <= 0:
            continue
        grad[dim_index] = (mean_at(right) - mean_at(left)) / width
    return float(np.linalg.norm(grad, ord=2))


def estimate_lipschitz_from_gp_mean(
    model,
    bounds,
    candidate_points=None,
    n_random=64,
    min_lipschitz=1e-8,
    rng=None,
    seed=None,
    finite_difference_eps=1e-4,
):
    """Estimate a Lipschitz constant from the GP posterior mean gradient.

    The estimate is the maximum gradient norm found on observed/candidate
    points and a small Latin-hypercube probe set inside ``bounds``.  It is a
    cheap estimator intended for per-iteration BO updates, not a global
    inner optimization.
    """
    bounds = validate_bounds(bounds)
    dim = bounds.shape[0]
    probe_points = []

    if candidate_points is not None:
        candidates = np.asarray(candidate_points, dtype=float)
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)
        if candidates.shape[1] == dim:
            probe_points.append(np.clip(candidates, bounds[:, 0], bounds[:, 1]))

    center = np.mean(bounds, axis=1, keepdims=True).T
    probe_points.append(center)

    if int(n_random) > 0:
        probe_points.append(
            latin_hypercube_sample(bounds, int(n_random), rng=rng, seed=seed)
        )

    points = np.vstack(probe_points)
    best_norm = 0.0
    for point in points:
        try:
            norm = _posterior_mean_gradient_norm(model, point)
        except Exception:
            norm = _posterior_mean_gradient_norm_finite_difference(
                model,
                point,
                bounds,
                eps=finite_difference_eps,
            )
        if np.isfinite(norm):
            best_norm = max(best_norm, float(norm))

    return max(float(min_lipschitz), best_norm)


def latin_hypercube(n_points, dim, rng=None, seed=None):
    """Draw Latin hypercube samples in [0, 1]^dim."""
    n_points = int(n_points)
    dim = int(dim)

    if n_points <= 0:
        raise ValueError("n_points must be positive.")
    if dim <= 0:
        raise ValueError("dim must be positive.")

    rng = get_rng(rng=rng, seed=seed)
    samples = np.empty((n_points, dim), dtype=float)
    centers = (1.0 + 2.0 * np.arange(n_points, dtype=float)) / (2.0 * n_points)

    for dim_index in range(dim):
        samples[:, dim_index] = centers[rng.permutation(n_points)]

    perturbation = rng.uniform(-1.0, 1.0, size=(n_points, dim)) / (2.0 * n_points)
    return np.clip(samples + perturbation, 0.0, 1.0)


def latin_hypercube_sample(bounds, n_points, rng=None, seed=None):
    """Draw Latin hypercube samples scaled to bounds shaped (dim, 2)."""
    bounds = validate_bounds(bounds)
    unit_samples = latin_hypercube(
        n_points=n_points,
        dim=bounds.shape[0],
        rng=rng,
        seed=seed,
    )
    return scale_unit_samples(unit_samples, bounds)


def uniform_sample(bounds, n_points, rng=None, seed=None):
    """Draw independent uniform random samples inside the box bounds."""
    bounds = validate_bounds(bounds)
    n_points = int(n_points)
    if n_points <= 0:
        raise ValueError("n_points must be positive.")

    rng = get_rng(rng=rng, seed=seed)
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_points, bounds.shape[0]))


def sobol_sample(bounds, n_points, scramble=True, seed=None):
    """Draw Sobol quasi-random samples inside the box bounds."""
    bounds = validate_bounds(bounds)
    n_points = int(n_points)
    if n_points <= 0:
        raise ValueError("n_points must be positive.")

    engine = torch.quasirandom.SobolEngine(
        dimension=bounds.shape[0],
        scramble=bool(scramble),
        seed=seed,
    )
    unit_samples = engine.draw(n_points).detach().cpu().numpy()
    return scale_unit_samples(unit_samples, bounds)


def center_sample(bounds, n_points=1):
    """Return the center of the box, repeated n_points times."""
    bounds = validate_bounds(bounds)
    n_points = int(n_points)
    if n_points <= 0:
        raise ValueError("n_points must be positive.")

    center = np.mean(bounds, axis=1)
    return np.tile(center, (n_points, 1))


def grid_sample(bounds, points_per_dim=3, max_points=None):
    """Draw a Cartesian grid inside the box bounds."""
    bounds = validate_bounds(bounds)
    points_per_dim = int(points_per_dim)
    if points_per_dim <= 0:
        raise ValueError("points_per_dim must be positive.")

    axes = [np.linspace(lower, upper, points_per_dim) for lower, upper in bounds]
    mesh = np.meshgrid(*axes, indexing="ij")
    samples = np.stack([axis.reshape(-1) for axis in mesh], axis=-1)

    if max_points is not None:
        max_points = int(max_points)
        if max_points <= 0:
            raise ValueError("max_points must be positive when provided.")
        samples = samples[:max_points]

    return samples


def sample_points(bounds, n_points, method="latin_hypercube", rng=None, seed=None, **kwargs):
    """Unified sampling interface."""
    method = str(method).lower()

    if method in {"latin_hypercube", "lhs"}:
        return latin_hypercube_sample(bounds, n_points, rng=rng, seed=seed)
    if method in {"uniform", "random"}:
        return uniform_sample(bounds, n_points, rng=rng, seed=seed)
    if method == "sobol":
        return sobol_sample(bounds, n_points, seed=seed, **kwargs)
    if method == "center":
        return center_sample(bounds, n_points=n_points)
    if method == "grid":
        points_per_dim = kwargs.get("points_per_dim", 3)
        return grid_sample(bounds, points_per_dim=points_per_dim, max_points=n_points)

    raise ValueError(
        f"Unknown sampling method {method!r}. Supported methods: "
        f"{sorted(SUPPORTED_SAMPLING_METHODS)}"
    )


class TreeNode:
    """Discrete partition-tree node for REMBO-UCB embedded cells."""

    def __init__(self, h, i, bounds_emb, original_boundaries):
        self.h = int(h)
        self.i = int(i)
        self.bounds_emb = validate_bounds(bounds_emb)
        self.C = 0
        self.mu_hat = 0.0
        self.B = float("inf")
        self.uncertainty = float("inf")

        self.local_dataset = []
        self.X_data = []
        self.Z_data = []
        self.y_data = []

        self.bounds_high = original_boundaries
        self.model = None
        self.probability_model = None
        self.children = None
        self.L_hat = None
        self.gp_kernel_name = None
        self.gp_kernel_score = -float("inf")

    def is_leaf(self):
        """Return True when the node has not been discretized further."""
        return self.children is None


def split_bounds_longest_dimension(bounds, n_splits):
    """Split a box along its longest dimension into equal-width child boxes."""
    bounds = validate_bounds(bounds)
    n_splits = int(n_splits)
    if n_splits <= 0:
        raise ValueError("n_splits must be positive.")

    longest_dim = int(np.argmax(bounds[:, 1] - bounds[:, 0]))
    dim_min = bounds[longest_dim, 0]
    dim_max = bounds[longest_dim, 1]
    step = (dim_max - dim_min) / n_splits

    child_bounds = []
    for split_index in range(n_splits):
        child = np.copy(bounds)
        child[longest_dim, 0] = dim_min + split_index * step
        child[longest_dim, 1] = dim_min + (split_index + 1) * step
        child_bounds.append(child)
    return child_bounds, longest_dim


def z_in_bounds(z, bounds, dim=None):
    """Return True if a z point lies inside bounds, optionally checking one dimension."""
    z = np.asarray(z, dtype=float).reshape(-1)
    bounds = validate_bounds(bounds)
    if dim is None:
        return bool(np.all(z >= bounds[:, 0]) and np.all(z <= bounds[:, 1]))
    dim = int(dim)
    return bool(bounds[dim, 0] <= z[dim] <= bounds[dim, 1])


def sample_embedded_uniform(bounds, n_points, rng=None, seed=None):
    """Draw uniform samples from an embedded cell using the package sampler."""
    return latin_hypercube_sample(bounds, n_points=n_points, rng=rng, seed=seed)


def global_optimization(
    objective_function,
    boundaries,
    batch_size=1,
    num_restarts=20,
    raw_samples=1024,
    method="botorch",
    direct_maxfun=500,
    direct_maxiter=200,
    local_maxfun=800,
):
    """Optimize a BoTorch acquisition function over box boundaries."""
    method = str(method).strip().lower().replace("_", "-")
    if method in {"direct", "direct-local", "direct+local"} and int(batch_size) == 1:
        try:
            from scipy.optimize import direct, minimize

            bounds_np = validate_bounds(boundaries)
            bounds_list = [(float(lo), float(hi)) for lo, hi in bounds_np]
            objective_function.to(dtype=torch.double)

            def scipy_objective(z):
                z_tensor = torch.as_tensor(np.asarray(z, dtype=float).reshape(1, -1), dtype=torch.double)
                with torch.no_grad():
                    value = objective_function(z_tensor)
                return -float(value.detach().cpu().reshape(-1)[0].item())

            best_x = None
            best_value = float("inf")
            result = direct(
                scipy_objective,
                bounds_list,
                maxfun=max(1, int(direct_maxfun)),
                maxiter=max(1, int(direct_maxiter)),
                locally_biased=True,
            )
            if np.isfinite(float(result.fun)):
                best_x = np.clip(np.asarray(result.x, dtype=float), bounds_np[:, 0], bounds_np[:, 1])
                best_value = float(result.fun)

            if best_x is not None and method in {"direct-local", "direct+local"}:
                local_result = minimize(
                    scipy_objective,
                    best_x,
                    method="Nelder-Mead",
                    bounds=bounds_list,
                    options={"maxfev": max(1, int(local_maxfun))},
                )
                if np.isfinite(float(local_result.fun)) and float(local_result.fun) < best_value:
                    best_x = np.clip(np.asarray(local_result.x, dtype=float), bounds_np[:, 0], bounds_np[:, 1])

            if best_x is not None:
                return torch.as_tensor(best_x.reshape(1, -1), dtype=torch.double)
        except Exception:
            pass

    from botorch.optim import optimize_acqf

    if isinstance(boundaries, torch.Tensor):
        bounds_tensor = boundaries.detach().clone().to(dtype=torch.double)
    else:
        bounds_tensor = bounds_to_botorch(boundaries, dtype=torch.double)

    if bounds_tensor.shape[0] != 2:
        bounds_tensor = bounds_tensor.T

    objective_function.to(dtype=torch.double)

    candidates, _ = optimize_acqf(
        acq_function=objective_function,
        bounds=bounds_tensor,
        q=int(batch_size),
        num_restarts=int(num_restarts),
        raw_samples=int(raw_samples),
        return_best_only=True,
        options={"dtype": torch.double},
    )

    return candidates.detach().to(dtype=torch.double)
