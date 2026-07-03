"""Kernel definitions for Bayesian-optimization optimizers.

This module owns normal kernels and embedded-space kernels.  When an optimizer
works in an embedded space, it can require the induced embedded kernel

    k_emb(z, z') = k(p_X(Az), p_X(Az')).

That rule is enforced in ``build_embedded_kernel`` and
``build_rembo_ucb_kernel`` so callers cannot accidentally fit an ordinary
z-space kernel to embedded data.
"""

import torch
import gpytorch
from gpytorch.constraints import Interval


SUPPORTED_BASE_KERNELS = {"rbf", "matern12", "matern32", "matern52"}
SUPPORTED_EMBEDDED_KERNELS = SUPPORTED_BASE_KERNELS


class InducedEmbeddingKernel(gpytorch.kernels.Kernel):
    """Kernel induced by projecting embedded points into the original domain."""

    has_lengthscale = False

    def __init__(self, base_kernel, projection_matrix, original_bounds=None, n_keep_dims=0):
        super().__init__()
        self.base_kernel = base_kernel
        self.n_keep_dims = int(n_keep_dims)

        A = torch.as_tensor(projection_matrix, dtype=torch.double)
        self.register_buffer("projection_matrix", A)

        if original_bounds is None:
            bounds = torch.stack(
                [
                    -torch.ones(A.shape[0], dtype=torch.double),
                    torch.ones(A.shape[0], dtype=torch.double),
                ],
                dim=-1,
            )
        else:
            bounds = torch.as_tensor(original_bounds, dtype=torch.double)

        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("original_bounds must have shape (D, 2).")

        self.register_buffer("original_bounds", bounds)

    def _project_to_original_domain(self, z):
        z = z.to(
            dtype=self.projection_matrix.dtype,
            device=self.projection_matrix.device,
        )
        if z.ndim == 1:
            z = z.view(1, -1)

        z_projected = z[..., self.n_keep_dims:]
        x_projected = z_projected.matmul(self.projection_matrix.transpose(-1, -2))

        if self.n_keep_dims > 0:
            x_keep = z[..., :self.n_keep_dims]
            x_projected = torch.cat([x_keep, x_projected], dim=-1)

        lower = self.original_bounds[:, 0].to(
            device=x_projected.device,
            dtype=x_projected.dtype,
        )
        upper = self.original_bounds[:, 1].to(
            device=x_projected.device,
            dtype=x_projected.dtype,
        )
        return torch.clamp(x_projected, lower, upper)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        projected_x1 = self._project_to_original_domain(x1)
        projected_x2 = self._project_to_original_domain(x2)
        return self.base_kernel(
            projected_x1,
            projected_x2,
            diag=diag,
            last_dim_is_batch=last_dim_is_batch,
            **params,
        )


def build_base_kernel(kernel_name="rbf", input_dim=None, lengthscale_bounds=(0.01, 50.0)):
    """Build a stationary base kernel on the normalized original domain."""
    kernel_name = str(kernel_name).lower()
    if kernel_name not in SUPPORTED_BASE_KERNELS:
        raise ValueError(
            f"Unknown kernel_name={kernel_name!r}. "
            f"Supported kernels: {sorted(SUPPORTED_BASE_KERNELS)}"
        )

    if lengthscale_bounds is None:
        lengthscale_constraint = None
    else:
        lower, upper = lengthscale_bounds
        lengthscale_constraint = Interval(float(lower), float(upper))

    if kernel_name == "rbf":
        return gpytorch.kernels.RBFKernel(
            ard_num_dims=input_dim,
            lengthscale_constraint=lengthscale_constraint,
        )
    if kernel_name == "matern12":
        return gpytorch.kernels.MaternKernel(
            nu=0.5,
            ard_num_dims=input_dim,
            lengthscale_constraint=lengthscale_constraint,
        )
    if kernel_name == "matern32":
        return gpytorch.kernels.MaternKernel(
            nu=1.5,
            ard_num_dims=input_dim,
            lengthscale_constraint=lengthscale_constraint,
        )

    return gpytorch.kernels.MaternKernel(
        nu=2.5,
        ard_num_dims=input_dim,
        lengthscale_constraint=lengthscale_constraint,
    )


def build_normal_kernel(kernel_name="rbf", input_dim=None, lengthscale_bounds=(0.01, 50.0)):
    """Build a normal non-embedded kernel for ordinary x-space GP models."""
    return gpytorch.kernels.ScaleKernel(
        build_base_kernel(
            kernel_name=kernel_name,
            input_dim=input_dim,
            lengthscale_bounds=lengthscale_bounds,
        )
    )


def build_embedded_kernel(
    kernel_name="rbf",
    projection_matrix=None,
    original_bounds=None,
    lengthscale_bounds=(0.01, 50.0),
    n_keep_dims=0,
):
    """Build the only allowed kernel family for embedded-space models."""
    if projection_matrix is None:
        raise ValueError(
            "Embedded kernels require projection_matrix. Use "
            "build_normal_kernel for non-embedded spaces."
        )
    kernel_name = str(kernel_name).lower()
    if kernel_name not in SUPPORTED_EMBEDDED_KERNELS:
        raise ValueError(
            f"Unknown embedded kernel_name={kernel_name!r}. "
            f"Allowed embedded kernels: {sorted(SUPPORTED_EMBEDDED_KERNELS)}"
        )

    base_input_dim = torch.as_tensor(projection_matrix).shape[0]
    base_kernel = build_base_kernel(
        kernel_name=kernel_name,
        input_dim=base_input_dim,
        lengthscale_bounds=lengthscale_bounds,
    )
    return gpytorch.kernels.ScaleKernel(
        InducedEmbeddingKernel(
            base_kernel=base_kernel,
            projection_matrix=projection_matrix,
            original_bounds=original_bounds,
            n_keep_dims=n_keep_dims,
        )
    )


def build_rembo_ucb_kernel(
    kernel_name="rbf",
    input_dim=None,
    lengthscale_bounds=(0.01, 50.0),
    projection_matrix=None,
    original_bounds=None,
    n_keep_dims=0,
    embedded=False,
):
    """Build the allowed kernel for REMBO-UCB-style embedded optimization."""
    if embedded:
        return build_embedded_kernel(
            kernel_name=kernel_name,
            projection_matrix=projection_matrix,
            original_bounds=original_bounds,
            lengthscale_bounds=lengthscale_bounds,
            n_keep_dims=n_keep_dims,
        )

    if projection_matrix is not None:
        return build_embedded_kernel(
            kernel_name=kernel_name,
            projection_matrix=projection_matrix,
            original_bounds=original_bounds,
            lengthscale_bounds=lengthscale_bounds,
            n_keep_dims=n_keep_dims,
        )

    return build_normal_kernel(
        kernel_name=kernel_name,
        input_dim=input_dim,
        lengthscale_bounds=lengthscale_bounds,
    )


def build_induced_kernel(
    kernel_name="rbf",
    input_dim=None,
    lengthscale_bounds=(0.01, 50.0),
    projection_matrix=None,
    original_bounds=None,
    n_keep_dims=0,
):
    """Backward-compatible name for the REMBO-UCB kernel builder."""
    return build_rembo_ucb_kernel(
        kernel_name=kernel_name,
        input_dim=input_dim,
        lengthscale_bounds=lengthscale_bounds,
        projection_matrix=projection_matrix,
        original_bounds=original_bounds,
        n_keep_dims=n_keep_dims,
        embedded=projection_matrix is not None,
    )
