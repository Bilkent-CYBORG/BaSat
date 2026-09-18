"""Validated GPyTorch kernel construction in algorithm model coordinates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import gpytorch
import torch


class KernelError(Exception):
    """Base class for Basat kernel construction errors."""


class InvalidKernelConfigError(KernelError, ValueError):
    """Raised when a kernel configuration contradicts the model dimension."""


class UnsupportedKernelError(KernelError, ValueError):
    """Raised when a named kernel is not implemented."""


@dataclass(frozen=True, slots=True)
class KernelConfig:
    """Declarative kernel configuration.

    Parameters
    ----------
    name:
        Kernel family.  Supported names include Matérn-1/2, Matérn-3/2,
        Matérn-5/2, RBF, rational quadratic, periodic, linear, polynomial,
        and spectral mixture aliases.
    ard_num_dims:
        Number of automatic-relevance-determination lengthscales in model
        coordinates.  It must match the modeled coordinate count.
    active_dims:
        Optional unique model-coordinate indices consumed by this kernel.
    use_outputscale:
        Wrap the base kernel in :class:`gpytorch.kernels.ScaleKernel`, except
        spectral mixtures whose mixture weights already provide scaling.
    lengthscale_prior, outputscale_prior:
        Optional GPyTorch prior instances.
    kwargs:
        Additional family-specific constructor arguments.
    """

    name: str = "matern52"
    ard_num_dims: int | None = None
    active_dims: tuple[int, ...] | None = None
    use_outputscale: bool = True
    lengthscale_prior: object | None = None
    outputscale_prior: object | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)


def build_kernel(config: KernelConfig, *, input_dimension: int) -> gpytorch.kernels.Kernel:
    """Build a GPyTorch kernel for an explicit model-space dimension.

    Parameters
    ----------
    config:
        Kernel definition.
    input_dimension:
        Surrogate-model dimension ``D_model``.  This value is never inferred
        from an environment.

    Returns
    -------
    gpytorch.kernels.Kernel
        Configured base or output-scaled kernel.

    Raises
    ------
    InvalidKernelConfigError
        If ARD, active dimensions, priors, or family arguments are invalid.
    UnsupportedKernelError
        If ``config.name`` is unknown.
    """

    if not isinstance(config, KernelConfig):
        raise InvalidKernelConfigError("config must be a KernelConfig instance.")
    if isinstance(input_dimension, bool) or not isinstance(input_dimension, int):
        raise InvalidKernelConfigError("input_dimension must be an integer.")
    if input_dimension < 1:
        raise InvalidKernelConfigError("input_dimension must be positive.")
    active_dims = config.active_dims
    if active_dims is not None:
        if not active_dims:
            raise InvalidKernelConfigError("active_dims cannot be empty.")
        if len(set(active_dims)) != len(active_dims):
            raise InvalidKernelConfigError("active_dims must not contain duplicates.")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= input_dimension
            for index in active_dims
        ):
            raise InvalidKernelConfigError(
                "active_dims contains an index outside the model coordinates."
            )
    modeled_dimension = input_dimension if active_dims is None else len(active_dims)
    if config.ard_num_dims is not None:
        if isinstance(config.ard_num_dims, bool) or not isinstance(config.ard_num_dims, int):
            raise InvalidKernelConfigError("ard_num_dims must be a positive integer.")
        if config.ard_num_dims != modeled_dimension:
            raise InvalidKernelConfigError(
                "ard_num_dims must equal the number of modeled coordinates "
                f"({modeled_dimension}), not an environment dimension."
            )
    if not isinstance(config.use_outputscale, bool):
        raise InvalidKernelConfigError("use_outputscale must be bool.")
    if not isinstance(config.name, str):
        raise InvalidKernelConfigError("name must be a string identifying a kernel family.")
    for label, prior in {
        "lengthscale_prior": config.lengthscale_prior,
        "outputscale_prior": config.outputscale_prior,
    }.items():
        if prior is not None and not isinstance(prior, gpytorch.priors.Prior):
            raise InvalidKernelConfigError(f"{label} must be a GPyTorch prior.")
    if not isinstance(config.kwargs, Mapping):
        raise InvalidKernelConfigError("kwargs must be a mapping.")

    common: dict[str, Any] = {"active_dims": active_dims}
    if config.ard_num_dims is not None:
        common["ard_num_dims"] = config.ard_num_dims
    if config.lengthscale_prior is not None:
        common["lengthscale_prior"] = config.lengthscale_prior
    kwargs = dict(config.kwargs)
    name = config.name.lower().replace("-", "").replace("_", "")
    try:
        if name in {"rbf", "se", "squaredexponential", "gaussian"}:
            base = gpytorch.kernels.RBFKernel(**common, **kwargs)
        elif name in {"matern12", "matern05", "exponential"}:
            base = gpytorch.kernels.MaternKernel(nu=0.5, **common, **kwargs)
        elif name in {"matern32", "matern15"}:
            base = gpytorch.kernels.MaternKernel(nu=1.5, **common, **kwargs)
        elif name in {"matern52", "matern25", "matern"}:
            base = gpytorch.kernels.MaternKernel(nu=2.5, **common, **kwargs)
        elif name in {"rq", "rationalquadratic"}:
            base = gpytorch.kernels.RQKernel(**common, **kwargs)
        elif name in {"periodic", "period"}:
            base = gpytorch.kernels.PeriodicKernel(**common, **kwargs)
        elif name == "linear":
            # Linear kernels do not define a lengthscale or ARD parameter.
            if config.ard_num_dims is not None or config.lengthscale_prior is not None:
                raise InvalidKernelConfigError(
                    "Linear kernels do not accept ARD or a lengthscale prior."
                )
            base = gpytorch.kernels.LinearKernel(active_dims=active_dims, **kwargs)
        elif name in {"polynomial", "poly"}:
            if config.ard_num_dims is not None or config.lengthscale_prior is not None:
                raise InvalidKernelConfigError(
                    "Polynomial kernels do not accept ARD or a lengthscale prior."
                )
            power = kwargs.pop("power", 2)
            if isinstance(power, bool) or not isinstance(power, int) or power < 1:
                raise InvalidKernelConfigError("Polynomial power must be a positive integer.")
            base = gpytorch.kernels.PolynomialKernel(power=power, active_dims=active_dims, **kwargs)
        elif name in {"spectralmixture", "spectral"}:
            if config.lengthscale_prior is not None:
                raise InvalidKernelConfigError(
                    "Spectral-mixture kernels do not accept a lengthscale prior."
                )
            num_mixtures = kwargs.pop("num_mixtures", 4)
            if (
                isinstance(num_mixtures, bool)
                or not isinstance(num_mixtures, int)
                or num_mixtures < 1
            ):
                raise InvalidKernelConfigError("num_mixtures must be a positive integer.")
            base = gpytorch.kernels.SpectralMixtureKernel(
                num_mixtures=num_mixtures,
                ard_num_dims=modeled_dimension,
                active_dims=active_dims,
                **kwargs,
            )
        else:
            raise UnsupportedKernelError(f"Unsupported kernel: {config.name!r}.")
    except UnsupportedKernelError:
        raise
    except InvalidKernelConfigError:
        raise
    except (TypeError, ValueError, RuntimeError) as exc:
        raise InvalidKernelConfigError(
            f"Invalid arguments for kernel {config.name!r}: {exc}"
        ) from exc
    if isinstance(base, gpytorch.kernels.SpectralMixtureKernel):
        if config.outputscale_prior is not None:
            raise InvalidKernelConfigError(
                "Spectral mixtures use mixture weights and do not accept outputscale_prior."
            )
        return base
    if not config.use_outputscale:
        if config.outputscale_prior is not None:
            raise InvalidKernelConfigError("outputscale_prior requires use_outputscale=True.")
        return base
    return gpytorch.kernels.ScaleKernel(
        base_kernel=base, outputscale_prior=config.outputscale_prior
    )


def make_kernel(
    config: KernelConfig, *, input_dimension: int,
    parameters: Mapping[str, Any] | None = None,
    dtype: str = "float64", device: str = "cpu",
) -> gpytorch.kernels.Kernel:
    """Construct and initialize a kernel without importing a backend library.

    Parameters
    ----------
    config : KernelConfig
        Family, ARD, active-coordinate, and output-scale configuration.
    input_dimension : int
        Number of coordinates in the surrogate input.
    parameters : mapping, optional
        Explicit numeric constrained parameters, such as ``lengthscale``,
        ``outputscale`` or polynomial ``offset``. Base-kernel parameters need
        no ``base_kernel.`` prefix. Assignment occurs *after* precision/device
        conversion, avoiding accidental float32 rounding of float64 settings.
    dtype, device : str
        Surrogate precision and computation device.

    Returns
    -------
    kernel
        A fresh covariance object accepted by Basat model configurations.

    Examples
    --------
    >>> kernel = basat.make_kernel(
    ...     basat.KernelConfig(name="rbf", ard_num_dims=2, use_outputscale=False),
    ...     input_dimension=2, parameters={"lengthscale": [0.2, 5.0]})
    """
    from .utils import resolve_device, resolve_dtype

    actual_dtype = resolve_dtype(dtype)
    actual_device = resolve_device(device)
    kernel = build_kernel(config, input_dimension=input_dimension).to(
        dtype=actual_dtype, device=actual_device
    )
    if parameters is not None and not isinstance(parameters, Mapping):
        raise InvalidKernelConfigError("parameters must be a mapping.")
    for name, value in (parameters or {}).items():
        if not isinstance(name, str) or name.startswith("raw_") or "." in name:
            raise InvalidKernelConfigError("Use constrained parameter names without module prefixes.")
        target = kernel
        if isinstance(kernel, gpytorch.kernels.ScaleKernel) and name != "outputscale":
            target = kernel.base_kernel
        if not hasattr(target, name):
            raise InvalidKernelConfigError(f"Kernel has no numeric parameter {name!r}.")
        numeric = torch.as_tensor(value, dtype=actual_dtype, device=actual_device)
        if not numeric.numel() or not torch.isfinite(numeric).all():
            raise InvalidKernelConfigError(f"Kernel parameter {name!r} must be finite and nonempty.")
        try:
            target.initialize(**{name: numeric})
        except (RuntimeError, ValueError, AttributeError) as exc:
            raise InvalidKernelConfigError(f"Invalid kernel parameter {name!r}: {exc}") from exc
    return kernel


__all__ = [
    "InvalidKernelConfigError",
    "KernelConfig",
    "KernelError",
    "UnsupportedKernelError",
    "build_kernel",
    "make_kernel",
]
