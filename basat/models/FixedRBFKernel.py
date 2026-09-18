"""Fixed RBF covariance with explicitly reproducible numerical arithmetic.

The ``gpy`` distance/DSYRK calculation is adapted from GPy v1.10.0:
GPy/kern/src/stationary.py, GPy/kern/src/rbf.py, and GPy/util/linalg.py.
Copyright (c) 2012, the GPy authors. All rights reserved.
The complete BSD-3-Clause terms are in FixedRBFKernel_GPy_LICENSE.txt.
This module contains no benchmark, dataset, or acquisition policy.
"""

from __future__ import annotations

from typing import Any

import gpytorch
import numpy as np
import torch
from scipy.linalg import blas


class FixedRBFKernel(gpytorch.kernels.Kernel):
    """Evaluate a fixed CPU/float64 RBF using GPy-compatible arithmetic.

    Parameters
    ----------
    input_dimension : int
        Positive number of coordinates after any active-dimension selection.
    lengthscale : float or array-like
        Positive finite scale. With ``ard=True``, supply one scale per coordinate
        or a scalar to broadcast; otherwise supply exactly one scalar.
    variance : float, default 1.0
        Positive finite fixed signal variance.
    ard : bool, default False
        Scale coordinates before computing distances when true. When false,
        scale the completed Euclidean distance, matching GPy's non-ARD path.
    arithmetic : {"gpy"}, default "gpy"
        Named numerical evaluation protocol, not a change to the RBF equation.

    Notes
    -----
    This compatibility kernel deliberately preserves squared-norm/BLAS DSYRK,
    square-root, scaling, and exponential evaluation order. Parameters are
    immutable model buffers, not trainable hyperparameters. Only unbatched,
    finite CPU float64 inputs are supported. Input-gradient optimization and
    training-mode evaluation raise errors instead of silently dropping
    derivatives. Use ``GaussianProcess.condition_only()`` or algorithm GP
    training steps of zero. Native differentiable RBF kernels remain available
    through ``KernelConfig(name="rbf")``.
    ``kernel(X)`` uses self-covariance arithmetic, whereas explicitly supplied
    cross inputs ``kernel(X, X.clone())`` use the cross-distance path even if
    their values are equal. This distinction can change last-bit roundoff.

    Examples
    --------
    >>> import basat
    >>> kernel = basat.FixedRBFKernel(
    ...     input_dimension=2, lengthscale=[0.2, 5.0], variance=1.0, ard=True
    ... )
    >>> points = basat.array([[0.0, 0.0], [1.0, 2.0]])
    >>> covariance = kernel(points).to_dense()
    >>> covariance.shape
    torch.Size([2, 2])
    """

    has_lengthscale = False

    def __init__(
        self,
        *,
        input_dimension: int,
        lengthscale: Any = 1.0,
        variance: float = 1.0,
        ard: bool = False,
        arithmetic: str = "gpy",
    ) -> None:
        if (
            isinstance(input_dimension, bool)
            or not isinstance(input_dimension, int)
            or input_dimension < 1
        ):
            raise ValueError("input_dimension must be a positive integer.")
        if not isinstance(ard, bool) or arithmetic != "gpy":
            raise ValueError("FixedRBFKernel requires boolean ard and arithmetic='gpy'.")
        scales = torch.as_tensor(lengthscale, dtype=torch.float64).reshape(-1)
        if scales.requires_grad:
            raise ValueError("FixedRBFKernel does not support lengthscale gradients.")
        if ard and scales.numel() == 1:
            scales = scales.repeat(input_dimension)
        count = input_dimension if ard else 1
        if (
            scales.numel() != count
            or not bool(torch.isfinite(scales).all())
            or bool((scales <= 0).any())
        ):
            raise ValueError("lengthscale must contain the required positive finite scales.")
        signal = torch.as_tensor(variance, dtype=torch.float64)
        if signal.requires_grad:
            raise ValueError("FixedRBFKernel does not support variance gradients.")
        if signal.numel() != 1 or not bool(torch.isfinite(signal).all()) or float(signal) <= 0:
            raise ValueError("variance must be a positive finite scalar.")
        super().__init__()
        self.input_dimension = input_dimension
        self.ard = ard
        self.arithmetic = arithmetic
        self.register_buffer("fixed_lengthscale", scales.detach().clone())
        self.register_buffer("fixed_variance", signal.reshape(()).detach().clone())
        self.eval()

    def _validate_inputs(self, x1: torch.Tensor, x2: torch.Tensor) -> None:
        if self.training:
            raise ValueError(
                "FixedRBFKernel does not support training-mode evaluation; use condition_only()."
            )
        if (
            self.fixed_lengthscale.dtype != torch.float64
            or self.fixed_lengthscale.device.type != "cpu"
        ):
            raise ValueError("FixedRBFKernel supports CPU float64 buffers only.")
        for values in (x1, x2):
            if values.ndim != 2 or values.shape[-1] != self.input_dimension:
                raise ValueError("FixedRBFKernel requires unbatched (N, input_dimension) matrices.")
            if values.dtype != torch.float64 or values.device.type != "cpu":
                raise ValueError("FixedRBFKernel supports CPU float64 inputs only.")
            if values.requires_grad:
                raise ValueError("FixedRBFKernel does not support input gradients.")
            if not bool(torch.isfinite(values).all()):
                raise ValueError("FixedRBFKernel inputs must be finite.")

    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Any:
        """Retain the distinction between an omitted and explicit cross input."""
        params.setdefault("_basat_self_covariance", x2 is None)
        return super().__call__(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> torch.Tensor:
        """Return the covariance matrix, preserving the selected arithmetic."""
        self_covariance = bool(params.get("_basat_self_covariance", False))
        self._validate_inputs(x1, x2)
        if last_dim_is_batch:
            raise ValueError("FixedRBFKernel does not support last_dim_is_batch.")
        same = self_covariance and torch.equal(x1, x2)
        if diag and same:
            return self.fixed_variance.expand(x1.shape[0]).clone()
        if diag and x1.shape[0] != x2.shape[0]:
            raise ValueError("Diagonal cross-covariance requires equal row counts.")
        left, right = x1.numpy(), x2.numpy()
        scales = self.fixed_lengthscale.numpy()
        if self.ard:
            left, right = left / scales, right / scales
        left_squared = np.sum(np.square(left), axis=1)
        if same:
            # GPy's tdot_blas: upper DSYRK, mirror, then C-order storage.
            product = blas.dsyrk(
                alpha=1.0,
                a=np.asfortranarray(left),
                beta=0.0,
                c=np.zeros((len(left), len(left))),
                overwrite_c=1,
                trans=0,
                lower=0,
            )
            upper = np.triu_indices(len(left), k=1)
            product.T[upper] = product[upper]
            product = np.ascontiguousarray(product)
            squared_distance = -2.0 * product + (left_squared[:, None] + left_squared[None, :])
            np.fill_diagonal(squared_distance, 0.0)
        else:
            right_squared = np.sum(np.square(right), axis=1)
            squared_distance = -2.0 * np.dot(left, right.T) + (
                left_squared[:, None] + right_squared[None, :]
            )
        distance = np.sqrt(np.clip(squared_distance, 0, np.inf))
        if not self.ard:
            distance = distance / scales
        covariance = float(self.fixed_variance) * np.exp(-0.5 * distance**2)
        result = torch.from_numpy(covariance)
        return result.diagonal().clone() if diag else result


__all__ = ["FixedRBFKernel"]
