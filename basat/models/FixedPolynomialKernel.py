"""Fixed polynomial covariance in explicitly declared GPy operation order.

The NumPy expression and full-matrix diagonal path follow GPy 1.13.2
``kern/src/poly.py`` (Copyright 2014 James Hensman, BSD-3-Clause; the adjacent
``FixedRBFKernel_GPy_LICENSE.txt`` contains the license terms). No benchmark,
dataset, posterior or acquisition policy is embedded in this kernel.
"""
from __future__ import annotations

import math
from functools import lru_cache
from numbers import Integral, Real
from typing import Any

import gpytorch
import numpy as np
import torch


_DIAGONAL_CACHE_MAX_INPUT_BYTES = 128 * 1024


def _gpy_covariance(left_values: np.ndarray, right_values: np.ndarray,
                    self_covariance: bool, scale: np.ndarray, bias: np.ndarray,
                    variance: np.ndarray, order: int | float) -> np.ndarray:
    """Keep the source full-dot arithmetic identical on cache misses."""
    active = np.arange(left_values.shape[1])
    left = left_values[:, active].astype("float")
    right = left if self_covariance else right_values[:, active].astype("float")
    product = np.dot(left, right.T)
    base = (scale * product) + bias
    with np.errstate(invalid="ignore", over="ignore"):
        covariance = (base ** order) * variance
    if not np.isfinite(covariance).all():
        raise ValueError("FixedPolynomialKernel produced nonfinite covariance.")
    return covariance


@lru_cache(maxsize=8)
def _gpy_diagonal_bytes(left_shape: tuple[int, int], left_bytes: bytes,
                        right_shape: tuple[int, int], right_bytes: bytes,
                        self_covariance: bool, scale_bytes: bytes, bias_bytes: bytes,
                        variance_bytes: bytes, order_bytes: bytes,
                        integer_order: bool) -> bytes:
    """Cache immutable diagonal bytes, never a posterior or full Gram matrix.

    The key retains every input/parameter bit and the source sharing convention.
    Equal grids can therefore reuse the literal source calculation across fresh
    model instances without changing its operation order or returned ownership.
    """
    left = np.frombuffer(left_bytes, dtype=np.float64).reshape(left_shape)
    right = np.frombuffer(right_bytes, dtype=np.float64).reshape(right_shape)
    scale, bias, variance, raw_order = (
        np.frombuffer(value, dtype=np.float64).reshape(())
        for value in (scale_bytes, bias_bytes, variance_bytes, order_bytes)
    )
    order = int(raw_order) if integer_order else float(raw_order)
    covariance = _gpy_covariance(left, right, self_covariance, scale, bias, variance, order)
    return covariance.diagonal().copy().tobytes()


class FixedPolynomialKernel(gpytorch.kernels.Kernel):
    """Evaluate ``(scale * numpy.dot(X, Z.T) + bias)**order * variance``.

    Fixed parameters retain their original scale/bias/variance parameterization
    instead of being algebraically rewritten as an offset and outputscale.
    Integer and floating-point ``order`` inputs retain their respective NumPy
    power arithmetic. ``diag=True`` extracts a diagonal from the full dot-product
    covariance, matching GPy's ``Kdiag`` rather than a rowwise square reduction.

    Only finite, unbatched CPU float64 coordinates are supported. Gradients and
    training-mode evaluation are rejected, not silently detached. Parameters
    are immutable model buffers. A noninteger order is accepted as in GPy, but
    is not a general guarantee of positive semidefiniteness or real-valued
    covariance on arbitrary coordinates; nonfinite evaluations raise an error.
    A bounded process-local cache retains only immutable diagonal bytes for
    small, bitwise-identical inputs/parameters. Misses still evaluate the full
    source dot product; returned tensors never share cached mutable storage.

    Parameters
    ----------
    input_dimension : int
        Positive model-coordinate count, independent of any benchmark.
    scale, bias, variance : float
        Positive finite fixed polynomial parameters.
    order : int or float, default 6
        Finite power of at least one; its supplied integer/float kind is kept.
    arithmetic : {"gpy"}
        Explicit numerical operation-order protocol.
    """

    has_lengthscale = False

    def __init__(self, *, input_dimension: int, scale: float = 1.0, bias: float = 1.0,
                 variance: float = 1.0, order: int | float = 6, arithmetic: str = "gpy") -> None:
        if isinstance(input_dimension, bool) or not isinstance(input_dimension, int) or input_dimension < 1:
            raise ValueError("input_dimension must be a positive integer.")
        if arithmetic != "gpy":
            raise ValueError("FixedPolynomialKernel requires arithmetic='gpy'.")
        values = {}
        for name, value in (("scale", scale), ("bias", bias), ("variance", variance)):
            tensor = torch.as_tensor(value, dtype=torch.float64)
            if tensor.requires_grad or tensor.numel() != 1 or not bool(torch.isfinite(tensor).all()) or float(tensor) <= 0:
                raise ValueError(f"{name} must be a positive finite scalar without gradients.")
            values[name] = tensor.reshape(()).detach().clone()
        if isinstance(order, bool) or not isinstance(order, Real) or not math.isfinite(float(order)) or order < 1:
            raise ValueError("order must be a finite integer or float of at least one.")
        super().__init__()
        self.input_dimension = input_dimension
        self.arithmetic = arithmetic
        self.order = int(order) if isinstance(order, Integral) else float(order)
        for name, value in values.items():
            self.register_buffer("fixed_" + name, value)
        self.register_buffer("fixed_order", torch.tensor(float(order), dtype=torch.float64))
        self.register_buffer("fixed_order_is_integer", torch.tensor(isinstance(order, Integral), dtype=torch.bool))
        self.eval()

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor | None = None, diag: bool = False,
                 last_dim_is_batch: bool = False, **params: Any) -> Any:
        """Preserve an omitted self input versus explicit cross coordinates."""
        params.setdefault("_basat_self_covariance", x2 is None)
        return super().__call__(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False,
                last_dim_is_batch: bool = False, **params: Any) -> torch.Tensor:
        """Return the full covariance, or its full-dot-product diagonal."""
        self_covariance = bool(params.get("_basat_self_covariance", False))
        if self.training:
            raise ValueError("FixedPolynomialKernel rejects training mode; use condition_only().")
        if last_dim_is_batch:
            raise ValueError("FixedPolynomialKernel does not support last_dim_is_batch.")
        for name in ("fixed_scale", "fixed_bias", "fixed_variance", "fixed_order"):
            value = getattr(self, name)
            if value.dtype != torch.float64 or value.device.type != "cpu" or value.requires_grad:
                raise ValueError("FixedPolynomialKernel supports fixed CPU float64 buffers only.")
        for value in (x1, x2):
            if value.ndim != 2 or value.shape[-1] != self.input_dimension:
                raise ValueError("FixedPolynomialKernel requires unbatched (N, input_dimension) matrices.")
            if value.dtype != torch.float64 or value.device.type != "cpu" or value.requires_grad:
                raise ValueError("FixedPolynomialKernel requires CPU float64 inputs without gradients.")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("FixedPolynomialKernel inputs must be finite.")
        if diag and x1.shape[0] != x2.shape[0]:
            raise ValueError("Diagonal cross-covariance requires equal row counts.")
        integer_order = bool(self.fixed_order_is_integer)
        order = int(self.fixed_order) if integer_order else float(self.fixed_order)
        # GPy's default active_dims is arange(D), and its kernel wrapper uses
        # advanced column selection even when every coordinate is active. This
        # preserves that Fortran-ordered copy before NumPy/BLAS dot products.
        shared = self_covariance and torch.equal(x1, x2)
        if diag and (x1.numel() + x2.numel()) * x1.element_size() <= _DIAGONAL_CACHE_MAX_INPUT_BYTES:
            left_bytes = x1.numpy().tobytes()
            right_bytes = x2.numpy().tobytes()
            if right_bytes == left_bytes:
                right_bytes = left_bytes
            result_bytes = _gpy_diagonal_bytes(tuple(x1.shape), left_bytes, tuple(x2.shape),
                right_bytes, shared, self.fixed_scale.numpy().tobytes(), self.fixed_bias.numpy().tobytes(),
                self.fixed_variance.numpy().tobytes(), self.fixed_order.numpy().tobytes(), integer_order)
            return torch.from_numpy(np.frombuffer(result_bytes, dtype=np.float64).copy())
        covariance = _gpy_covariance(x1.numpy(), x2.numpy(), shared, self.fixed_scale.numpy(),
                                    self.fixed_bias.numpy(), self.fixed_variance.numpy(), order)
        result = torch.from_numpy(covariance)
        return result.diagonal().clone() if diag else result


__all__ = ["FixedPolynomialKernel"]
