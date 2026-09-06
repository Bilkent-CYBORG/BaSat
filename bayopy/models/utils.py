"""Numerical, geometric, sampling, and visualization utilities for BayOPy.

The model layer uses PyTorch tensors as its canonical representation.  Conversion
to NumPy is intentionally explicit and is reserved for third-party libraries
whose public APIs require NumPy arrays.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from gpytorch.priors import (
    GammaPrior,
    HalfCauchyPrior,
    LogNormalPrior,
    NormalPrior,
    UniformPrior,
)

DEFAULT_DTYPE = torch.float32
DEFAULT_DEVICE = torch.device("cpu")


class ModelUtilityError(Exception):
    """Base class for model-utility contract violations."""


class UnsupportedDtypeError(ModelUtilityError, TypeError):
    """Raised when a tensor dtype is not supported by BayOPy."""


class DeviceUnavailableError(ModelUtilityError, RuntimeError):
    """Raised when a requested accelerator cannot be used."""


class InvalidBoundsError(ModelUtilityError, ValueError):
    """Raised when a box domain is malformed."""


class InvalidTensorShapeError(ModelUtilityError, ValueError):
    """Raised when a tensor is incompatible with its documented coordinates."""


class SamplingError(ModelUtilityError, ValueError):
    """Raised when a requested random design cannot be generated."""


class PriorConfigurationError(ModelUtilityError, ValueError):
    """Raised when prior parameters do not define a valid distribution."""


class ReductionError(ModelUtilityError, ValueError):
    """Raised when a dimensionality-reduction request is invalid."""


class MissingOptionalDependencyError(ModelUtilityError, ImportError):
    """Raised when a requested optional visualization dependency is absent."""


def resolve_dtype(dtype: torch.dtype | str | None) -> torch.dtype:
    """Resolve a supported floating-point dtype.

    Parameters
    ----------
    dtype:
        A PyTorch dtype, ``"float32"``, ``"float64"``, or ``None``.  ``None``
        selects :data:`DEFAULT_DTYPE`.

    Returns
    -------
    torch.dtype
        The canonical PyTorch dtype.

    Raises
    ------
    UnsupportedDtypeError
        If ``dtype`` is not single- or double-precision floating point.
    """

    if dtype is None:
        return DEFAULT_DTYPE
    if isinstance(dtype, torch.dtype):
        if dtype not in {torch.float32, torch.float64}:
            raise UnsupportedDtypeError(f"Unsupported dtype: {dtype}.")
        return dtype
    mapping = {
        "float": torch.float32,
        "float32": torch.float32,
        "torch.float32": torch.float32,
        "double": torch.float64,
        "float64": torch.float64,
        "torch.float64": torch.float64,
    }
    try:
        return mapping[dtype.lower()]
    except (AttributeError, KeyError) as exc:
        raise UnsupportedDtypeError(f"Unsupported dtype: {dtype!r}.") from exc


def resolve_device(device: torch.device | str | None) -> torch.device:
    """Resolve and validate a PyTorch execution device.

    CPU, CUDA, and Apple MPS are the supported device families.  Accelerator
    requests fail eagerly so that an experiment cannot silently move to CPU.

    Parameters
    ----------
    device:
        Device specification accepted by :class:`torch.device`, or ``None`` for
        :data:`DEFAULT_DEVICE`.

    Returns
    -------
    torch.device
        A validated device.

    Raises
    ------
    DeviceUnavailableError
        If the device is malformed, unsupported, or unavailable.
    """

    if device is None:
        return DEFAULT_DEVICE
    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise DeviceUnavailableError(f"Invalid device specification: {device!r}.") from exc
    if resolved.type not in {"cpu", "cuda", "mps"}:
        raise DeviceUnavailableError(
            f"Unsupported device family {resolved.type!r}; use CPU, CUDA, or MPS."
        )
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise DeviceUnavailableError("CUDA was requested but is unavailable.")
        index = torch.cuda.current_device() if resolved.index is None else resolved.index
        if index < 0 or index >= torch.cuda.device_count():
            raise DeviceUnavailableError(f"CUDA device index {index} is unavailable.")
        resolved = torch.device("cuda", index)
    if resolved.type == "mps":
        available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not available:
            raise DeviceUnavailableError("MPS was requested but is unavailable.")
    return resolved


def as_tensor(
    value: Any,
    *,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
    copy: bool = False,
) -> torch.Tensor:
    """Convert a value to a BayOPy floating-point tensor.

    Parameters
    ----------
    value:
        Array-like input.
    dtype, device:
        Requested tensor precision and location.
    copy:
        Clone the converted tensor when ``True``.

    Returns
    -------
    torch.Tensor
        Tensor with the requested dtype and device.
    """

    target_dtype = resolve_dtype(dtype)
    target_device = resolve_device(device)
    if isinstance(value, torch.Tensor):
        result = value.to(dtype=target_dtype, device=target_device)
    else:
        result = torch.as_tensor(value, dtype=target_dtype, device=target_device)
    return result.clone() if copy else result


def to_numpy(value: torch.Tensor) -> np.ndarray:
    """Return a detached CPU NumPy view of a tensor.

    Parameters
    ----------
    value:
        Tensor to convert.

    Returns
    -------
    numpy.ndarray
        A NumPy representation that never participates in autograd.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError("to_numpy() requires a torch.Tensor.")
    return value.detach().cpu().numpy()


@dataclass(frozen=True, slots=True)
class Bounds:
    """Axis-aligned action or model-space bounds.

    Parameters
    ----------
    lower, upper:
        One-dimensional finite vectors.  Every lower coordinate must be
        strictly smaller than its corresponding upper coordinate.

    Examples
    --------
    >>> import bayopy
    >>> bounds = bayopy.Bounds(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    >>> bounds.dimension
    2
    """

    lower: torch.Tensor
    upper: torch.Tensor

    def __post_init__(self) -> None:
        """Canonicalize both endpoints and enforce a finite nonempty box."""

        if isinstance(self.lower, torch.Tensor):
            dtype = resolve_dtype(self.lower.dtype)
            device = resolve_device(self.lower.device)
        else:
            dtype = DEFAULT_DTYPE
            device = DEFAULT_DEVICE
        lower = as_tensor(self.lower, dtype=dtype, device=device, copy=True)
        upper = as_tensor(self.upper, dtype=dtype, device=device, copy=True)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        validate_bounds(self)

    @property
    def dimension(self) -> int:
        """Number of coordinates in the box."""

        return int(self.lower.numel())

    @property
    def width(self) -> torch.Tensor:
        """Per-coordinate box widths."""

        return self.upper - self.lower

    @property
    def dtype(self) -> torch.dtype:
        """Floating-point precision of the bounds."""

        return self.lower.dtype

    @property
    def device(self) -> torch.device:
        """Device holding the bounds."""

        return self.lower.device

    def as_tensor(self) -> torch.Tensor:
        """Return bounds in BoTorch's ``(2, D)`` representation."""

        return torch.stack((self.lower, self.upper))

    def contains(self, points: Any, *, atol: float = 0.0) -> torch.Tensor:
        """Test whether each point lies in the closed box.

        The final axis of ``points`` is interpreted as the coordinate axis.  The
        returned tensor has the leading shape of ``points``.

        Parameters
        ----------
        points:
            Points with trailing dimension equal to :attr:`dimension`.
        atol:
            Finite non-negative absolute containment tolerance.

        Returns
        -------
        torch.Tensor
            Boolean mask over the leading point dimensions.
        """

        if isinstance(atol, bool) or not isinstance(atol, (int, float)):
            raise InvalidBoundsError("atol must be a finite non-negative number.")
        if not np.isfinite(atol) or atol < 0:
            raise InvalidBoundsError("atol must be a finite non-negative number.")
        tensor = as_tensor(points, dtype=self.dtype, device=self.device)
        _validate_point_dimension(tensor, self)
        return ((tensor >= self.lower - atol) & (tensor <= self.upper + atol)).all(dim=-1)

    def clamp(self, points: Any) -> torch.Tensor:
        """Project points coordinate-wise onto the closed box.

        Parameters
        ----------
        points:
            Points with trailing dimension equal to :attr:`dimension`.

        Returns
        -------
        torch.Tensor
            Coordinate-wise projected points with the original shape.
        """

        tensor = as_tensor(points, dtype=self.dtype, device=self.device)
        _validate_point_dimension(tensor, self)
        return torch.maximum(torch.minimum(tensor, self.upper), self.lower)

    def to(
        self,
        *,
        dtype: torch.dtype | str | None = None,
        device: torch.device | str | None = None,
    ) -> Bounds:
        """Return a copy on the requested dtype and device.

        Parameters
        ----------
        dtype:
            Optional floating tensor dtype.
        device:
            Optional available Torch device.

        Returns
        -------
        Bounds
            Validated copied endpoints on the requested backend.
        """

        target_dtype = self.dtype if dtype is None else resolve_dtype(dtype)
        target_device = self.device if device is None else resolve_device(device)
        return Bounds(
            self.lower.to(dtype=target_dtype, device=target_device),
            self.upper.to(dtype=target_dtype, device=target_device),
        )


def validate_bounds(bounds: Bounds) -> None:
    """Validate the structural and numerical invariants of a box.

    Parameters
    ----------
    bounds:
        Candidate :class:`Bounds` instance.

    Raises
    ------
    InvalidBoundsError
        If endpoints are empty, mismatched, non-finite, or not strictly ordered.
    """

    if not isinstance(bounds, Bounds):
        raise InvalidBoundsError("bounds must be a Bounds instance.")
    if bounds.lower.ndim != 1:
        raise InvalidBoundsError("lower must be one-dimensional.")
    if bounds.upper.ndim != 1:
        raise InvalidBoundsError("upper must be one-dimensional.")
    if bounds.lower.shape != bounds.upper.shape:
        raise InvalidBoundsError("lower and upper must have equal shapes.")
    if bounds.lower.numel() == 0:
        raise InvalidBoundsError("bounds must contain at least one coordinate.")
    if bounds.lower.dtype != bounds.upper.dtype or bounds.lower.device != bounds.upper.device:
        raise InvalidBoundsError("lower and upper must use the same dtype and device.")
    if not torch.isfinite(bounds.lower).all():
        raise InvalidBoundsError("lower contains non-finite values.")
    if not torch.isfinite(bounds.upper).all():
        raise InvalidBoundsError("upper contains non-finite values.")
    if not torch.all(bounds.lower < bounds.upper):
        raise InvalidBoundsError(
            "Every lower bound must be less than the corresponding upper bound."
        )


def _validate_point_dimension(points: torch.Tensor, bounds: Bounds) -> None:
    """Validate the trailing coordinate dimension shared by box operations."""

    if points.ndim == 0 or points.shape[-1] != bounds.dimension:
        raise InvalidTensorShapeError(
            f"Expected trailing dimension {bounds.dimension}, got shape {tuple(points.shape)}."
        )


def normalize(X: torch.Tensor, bounds: Bounds) -> torch.Tensor:
    """Map points from ``bounds`` to the unit box.

    Parameters
    ----------
    X:
        Points with trailing dimension ``bounds.dimension``.
    bounds:
        Source coordinate box.

    Returns
    -------
    torch.Tensor
        Unit-box points with the same leading shape as ``X``.
    """

    tensor = as_tensor(X, dtype=bounds.dtype, device=bounds.device)
    _validate_point_dimension(tensor, bounds)
    return (tensor - bounds.lower) / bounds.width


def unnormalize(Z: torch.Tensor, bounds: Bounds) -> torch.Tensor:
    """Map points from the unit box to ``bounds``.

    Parameters
    ----------
    Z:
        Unit coordinates with trailing dimension ``bounds.dimension``.
    bounds:
        Destination coordinate box.

    Returns
    -------
    torch.Tensor
        Physical points with the same leading shape as ``Z``.
    """

    tensor = as_tensor(Z, dtype=bounds.dtype, device=bounds.device)
    _validate_point_dimension(tensor, bounds)
    return bounds.lower + tensor * bounds.width


def to_symmetric_box(X: torch.Tensor, bounds: Bounds) -> torch.Tensor:
    """Map points from ``bounds`` to ``[-1, 1]^D``.

    Parameters
    ----------
    X:
        Physical points with trailing environment/model dimension.
    bounds:
        Source coordinate box.

    Returns
    -------
    torch.Tensor
        Symmetric-box coordinates with the same shape as ``X``.
    """

    return 2.0 * normalize(X, bounds) - 1.0


def from_symmetric_box(X_symmetric: torch.Tensor, bounds: Bounds) -> torch.Tensor:
    """Map points from ``[-1, 1]^D`` back to ``bounds``.

    Parameters
    ----------
    X_symmetric:
        Symmetric-box points with the box's trailing dimension.
    bounds:
        Destination physical coordinate box.

    Returns
    -------
    torch.Tensor
        Physical coordinates with the same shape as ``X_symmetric``.
    """

    return unnormalize((X_symmetric + 1.0) / 2.0, bounds)


def make_generator(*, seed: int, device: torch.device | str | None = None) -> torch.Generator:
    """Create an independently seeded PyTorch random generator.

    Parameters
    ----------
    seed:
        Non-negative integer seed.
    device:
        Device on which random draws will be produced.

    Returns
    -------
    torch.Generator
        Independently seeded generator owned by the caller.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SamplingError("seed must be a non-negative integer.")
    target_device = resolve_device(device)
    try:
        generator = torch.Generator(device=target_device)
        generator.manual_seed(seed)
    except RuntimeError as exc:
        raise SamplingError(f"Cannot create a generator on {target_device}.") from exc
    return generator


def sobol_sample(
    bounds: Bounds,
    n: int,
    *,
    seed: int,
    scramble: bool = True,
) -> torch.Tensor:
    """Draw a deterministic scrambled Sobol design inside ``bounds``.

    Sobol points are produced on CPU by PyTorch and then moved to the bounds'
    device.  This gives identical designs across supported execution devices.

    Parameters
    ----------
    bounds:
        Destination sampling box.
    n:
        Positive number of design points.
    seed:
        Non-negative Sobol scrambling seed.
    scramble:
        Whether to apply deterministic Owen scrambling.

    Returns
    -------
    torch.Tensor
        Design with shape ``(n, bounds.dimension)``.
    """

    _validate_sample_count(n)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SamplingError("seed must be a non-negative integer.")
    try:
        engine = torch.quasirandom.SobolEngine(
            dimension=bounds.dimension, scramble=scramble, seed=seed
        )
        unit = engine.draw(n, dtype=bounds.dtype).to(bounds.device)
    except (RuntimeError, ValueError) as exc:
        raise SamplingError("Unable to generate the requested Sobol design.") from exc
    return unnormalize(unit, bounds)


def uniform_sample(
    bounds: Bounds,
    n: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw independent uniform points inside ``bounds``.

    Parameters
    ----------
    bounds:
        Destination sampling box.
    n:
        Positive number of points.
    generator:
        Caller-owned generator compatible with the bounds device.

    Returns
    -------
    torch.Tensor
        Random design with shape ``(n, bounds.dimension)``.
    """

    _validate_sample_count(n)
    if not isinstance(generator, torch.Generator):
        raise SamplingError("generator must be a torch.Generator.")
    try:
        unit = torch.rand(
            (n, bounds.dimension),
            dtype=bounds.dtype,
            device=bounds.device,
            generator=generator,
        )
    except RuntimeError as exc:
        raise SamplingError(
            "The random generator must be compatible with the bounds device."
        ) from exc
    return unnormalize(unit, bounds)


def _validate_sample_count(n: int) -> None:
    """Validate a requested design size."""

    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise SamplingError("n must be a positive integer.")


@dataclass(frozen=True, slots=True)
class TrustRegion:
    """A bounded local search region.

    ``center``, ``lower``, and ``upper`` share model or environment coordinates;
    the owning algorithm determines which coordinate system is used.

    Parameters
    ----------
    center:
        One-dimensional local-search center.
    lower, upper:
        Coordinate-aligned clipped trust-region bounds.
    length:
        Positive normalized side-length parameter owned by the algorithm.
    """

    center: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    length: float

    def __post_init__(self) -> None:
        """Validate shared shape, containment, finiteness, and positive length."""

        if self.center.ndim != 1:
            raise InvalidTensorShapeError("Trust-region center must be one-dimensional.")
        if self.lower.shape != self.center.shape or self.upper.shape != self.center.shape:
            raise InvalidTensorShapeError(
                "Trust-region center, lower, and upper must have equal shapes."
            )
        if not torch.isfinite(self.center).all():
            raise InvalidTensorShapeError("Trust-region center must be finite.")
        if not torch.all(self.lower <= self.center) or not torch.all(self.center <= self.upper):
            raise InvalidBoundsError("Trust-region center must lie inside its bounds.")
        if not np.isfinite(self.length) or self.length <= 0:
            raise InvalidBoundsError("Trust-region length must be finite and positive.")


def normal_prior(loc: float = 0.0, scale: float = 1.0) -> NormalPrior:
    """Construct a normal GPyTorch prior.

    Parameters
    ----------
    loc:
        Finite distribution location.
    scale:
        Finite positive standard deviation.

    Returns
    -------
    gpytorch.priors.NormalPrior
        Validated normal prior.
    """

    _require_finite("loc", loc)
    _require_positive("scale", scale)
    return NormalPrior(loc, scale)


def lognormal_prior(loc: float = 0.0, scale: float = 1.0) -> LogNormalPrior:
    """Construct a log-normal GPyTorch prior.

    Parameters
    ----------
    loc:
        Finite log-space location.
    scale:
        Finite positive log-space standard deviation.

    Returns
    -------
    gpytorch.priors.LogNormalPrior
        Validated log-normal prior.
    """

    _require_finite("loc", loc)
    _require_positive("scale", scale)
    return LogNormalPrior(loc, scale)


def gamma_prior(concentration: float = 1.0, rate: float = 1.0) -> GammaPrior:
    """Construct a gamma prior using concentration-rate parameters.

    Parameters
    ----------
    concentration:
        Finite positive shape parameter.
    rate:
        Finite positive rate parameter.

    Returns
    -------
    gpytorch.priors.GammaPrior
        Validated gamma prior.
    """

    _require_positive("concentration", concentration)
    _require_positive("rate", rate)
    return GammaPrior(concentration, rate)


def half_cauchy_prior(scale: float = 1.0) -> HalfCauchyPrior:
    """Construct a half-Cauchy GPyTorch prior.

    Parameters
    ----------
    scale:
        Finite positive scale.

    Returns
    -------
    gpytorch.priors.HalfCauchyPrior
        Validated half-Cauchy prior.
    """

    _require_positive("scale", scale)
    return HalfCauchyPrior(scale)


def uniform_prior(low: float = 0.0, high: float = 1.0) -> UniformPrior:
    """Construct a continuous uniform GPyTorch prior.

    Parameters
    ----------
    low, high:
        Finite endpoints satisfying ``low < high``.

    Returns
    -------
    gpytorch.priors.UniformPrior
        Validated continuous uniform prior.
    """

    _require_finite("low", low)
    _require_finite("high", high)
    if low >= high:
        raise PriorConfigurationError("low must be smaller than high.")
    return UniformPrior(low, high)


def _require_finite(name: str, value: float) -> None:
    """Validate a finite scalar prior parameter."""

    if isinstance(value, bool) or not np.isfinite(value):
        raise PriorConfigurationError(f"{name} must be finite.")


def _require_positive(name: str, value: float) -> None:
    """Validate a positive scalar prior parameter."""

    _require_finite(name, value)
    if value <= 0:
        raise PriorConfigurationError(f"{name} must be positive.")


def _validate_reduction_input(X: Any, n_components: int) -> torch.Tensor:
    """Convert and validate data shared by reduction helpers."""

    tensor = X if isinstance(X, torch.Tensor) else as_tensor(X)
    if tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] < 1:
        raise ReductionError("X must be a non-empty two-dimensional array.")
    if not torch.is_floating_point(tensor):
        tensor = tensor.to(DEFAULT_DTYPE)
    if not torch.isfinite(tensor).all():
        raise ReductionError("X contains non-finite values.")
    if isinstance(n_components, bool) or not isinstance(n_components, int):
        raise ReductionError("n_components must be an integer.")
    if n_components < 1 or n_components > min(tensor.shape):
        raise ReductionError("n_components must lie between 1 and min(number of rows, columns).")
    return tensor


def reduce_pca(X: Any, *, n_components: int = 2) -> torch.Tensor:
    """Reduce a matrix using deterministic principal-component analysis.

    Parameters
    ----------
    X:
        Nonempty finite matrix with shape ``(N, D)``.
    n_components:
        Positive output dimension no larger than ``min(N, D)``.

    Returns
    -------
    torch.Tensor
        Centered principal-component scores with shape ``(N, n_components)``.
    """

    tensor = _validate_reduction_input(X, n_components)
    centered = tensor - tensor.mean(dim=0, keepdim=True)
    # Full SVD is deterministic on CPU and preserves the input tensor contract.
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:n_components].transpose(-1, -2)


def reduce_tsne(
    X: Any,
    *,
    n_components: int = 2,
    perplexity: float = 30.0,
    seed: int = 1,
    **kwargs: Any,
) -> torch.Tensor:
    """Reduce a matrix using scikit-learn t-SNE.

    Raises :class:`MissingOptionalDependencyError` when the ``viz`` extra is not
    installed.

    Parameters
    ----------
    X:
        Nonempty finite matrix with shape ``(N, D)``.
    n_components:
        Positive embedding dimension.
    perplexity:
        Positive value strictly smaller than ``N``.
    seed:
        scikit-learn random seed.
    **kwargs:
        Additional keyword arguments forwarded to ``sklearn.manifold.TSNE``.

    Returns
    -------
    torch.Tensor
        Embedding with shape ``(N, n_components)`` on the input device.
    """

    tensor = _validate_reduction_input(X, n_components)
    if not np.isfinite(perplexity) or perplexity <= 0 or perplexity >= tensor.shape[0]:
        raise ReductionError("perplexity must be positive and smaller than the sample count.")
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise MissingOptionalDependencyError(
            "t-SNE requires the 'viz' extra: pip install bayopy[viz]."
        ) from exc
    values = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=seed,
        **kwargs,
    ).fit_transform(to_numpy(tensor))
    return torch.as_tensor(values, dtype=tensor.dtype, device=tensor.device)


def reduce_umap(
    X: Any,
    *,
    n_components: int = 2,
    seed: int = 1,
    **kwargs: Any,
) -> torch.Tensor:
    """Reduce a matrix using UMAP from the optional ``viz`` extra.

    Parameters
    ----------
    X:
        Nonempty finite matrix with shape ``(N, D)``.
    n_components:
        Positive embedding dimension.
    seed:
        UMAP random seed.
    **kwargs:
        Additional keyword arguments forwarded to ``umap.UMAP``.

    Returns
    -------
    torch.Tensor
        Embedding with shape ``(N, n_components)`` on the input device.

    Raises
    ------
    MissingOptionalDependencyError
        If UMAP or one of its required import-time dependencies is absent.
    ReductionError
        If an installed UMAP backend cannot initialize or execute.  In
        particular, Numba requires at least one writable cache location.
    """

    tensor = _validate_reduction_input(X, n_components)
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise MissingOptionalDependencyError(
            "UMAP requires the 'viz' extra: pip install bayopy[viz]."
        ) from exc
    except Exception as exc:
        # UMAP compiles several functions at import time.  An installed backend
        # can therefore fail before construction, most notably when Numba has
        # no writable cache locator.  This is an execution failure, not a
        # missing dependency, and must never trigger a mathematical fallback.
        cache_hint = (
            " Set NUMBA_CACHE_DIR to a writable directory before importing UMAP."
            if "no locator available" in str(exc).lower()
            else ""
        )
        raise ReductionError(f"UMAP backend initialization failed: {exc}.{cache_hint}") from exc
    try:
        values = umap.UMAP(n_components=n_components, random_state=seed, **kwargs).fit_transform(
            to_numpy(tensor)
        )
    except Exception as exc:
        raise ReductionError(f"UMAP reduction failed: {exc}") from exc
    return torch.as_tensor(values, dtype=tensor.dtype, device=tensor.device)


def _get_axes(ax: Any = None) -> tuple[Any, Any]:
    """Create or reuse a Matplotlib axes without displaying it."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise MissingOptionalDependencyError(
            "Plotting requires the 'viz' extra: pip install bayopy[viz]."
        ) from exc
    if ax is None:
        return plt.subplots()
    return ax.figure, ax


def _values_1d(values: Any, *, name: str = "values") -> np.ndarray:
    """Convert a finite numeric sequence to a one-dimensional NumPy array."""

    if isinstance(values, torch.Tensor):
        array = to_numpy(values).reshape(-1)
    else:
        array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite sequence.")
    return array


def plot_iteration_values(
    values: Any, *, ax: Any = None, label: str | None = None
) -> tuple[Any, Any]:
    """Plot observed objective values by evaluation index.

    Parameters
    ----------
    values:
        Nonempty finite one-dimensional observation sequence.
    ax:
        Optional Matplotlib axes to reuse.
    label:
        Optional legend label.

    Returns
    -------
    tuple[Any, Any]
        Matplotlib figure and axes.
    """

    array = _values_1d(values)
    fig, axes = _get_axes(ax)
    axes.plot(np.arange(1, array.size + 1), array, marker="o", label=label)
    axes.set(xlabel="Evaluation", ylabel="Observation")
    if label:
        axes.legend()
    return fig, axes


def plot_best_so_far(
    values: Any,
    *,
    maximize: bool = True,
    ax: Any = None,
    label: str | None = None,
) -> tuple[Any, Any]:
    """Plot the incumbent observation after each evaluation.

    Parameters
    ----------
    values:
        Nonempty finite observation sequence.
    maximize:
        Use cumulative maxima when true, otherwise cumulative minima.
    ax:
        Optional Matplotlib axes to reuse.
    label:
        Optional legend label.

    Returns
    -------
    tuple[Any, Any]
        Matplotlib figure and axes.
    """

    array = _values_1d(values)
    incumbent = np.maximum.accumulate(array) if maximize else np.minimum.accumulate(array)
    fig, axes = _get_axes(ax)
    axes.plot(np.arange(1, array.size + 1), incumbent, label=label)
    axes.set(xlabel="Evaluation", ylabel="Best observation")
    if label:
        axes.legend()
    return fig, axes


def plot_regret(
    values: Any,
    *,
    optimum: float | None = None,
    maximize: bool = True,
    cumulative: bool = False,
    ax: Any = None,
    label: str | None = None,
) -> tuple[Any, Any]:
    """Plot supplied regret values or derive regret from a known optimum.

    Parameters
    ----------
    values:
        Observations when ``optimum`` is known, otherwise precomputed regrets.
    optimum:
        Optional finite optimum used to derive instantaneous regret.
    maximize:
        Select maximization or minimization regret orientation.
    cumulative:
        Plot cumulative rather than instantaneous regret.
    ax:
        Optional Matplotlib axes to reuse.
    label:
        Optional legend label.

    Returns
    -------
    tuple[Any, Any]
        Matplotlib figure and axes.
    """

    array = _values_1d(values)
    if optimum is not None:
        if not np.isfinite(optimum):
            raise ValueError("optimum must be finite.")
        regret = optimum - array if maximize else array - optimum
    else:
        regret = array
    if cumulative:
        regret = np.cumsum(regret)
    fig, axes = _get_axes(ax)
    axes.plot(np.arange(1, array.size + 1), regret, label=label)
    axes.set(xlabel="Evaluation", ylabel="Cumulative regret" if cumulative else "Regret")
    if label:
        axes.legend()
    return fig, axes


def plot_comparison(
    series: Mapping[str, Any], *, ax: Any = None, ylabel: str = "Value"
) -> tuple[Any, Any]:
    """Plot multiple named one-dimensional evaluation series.

    Parameters
    ----------
    series:
        Nonempty mapping from legend labels to finite sequences.
    ax:
        Optional Matplotlib axes to reuse.
    ylabel:
        Vertical-axis label.

    Returns
    -------
    tuple[Any, Any]
        Matplotlib figure and axes.
    """

    if not isinstance(series, Mapping) or not series:
        raise ValueError("series must be a non-empty mapping.")
    fig, axes = _get_axes(ax)
    for label, values in series.items():
        array = _values_1d(values, name=str(label))
        axes.plot(np.arange(1, array.size + 1), array, label=str(label))
    axes.set(xlabel="Evaluation", ylabel=ylabel)
    axes.legend()
    return fig, axes


def plot_search_trajectory(
    actions: Any,
    *,
    values: Any | None = None,
    reduction: str = "pca",
    ax: Any = None,
) -> tuple[Any, Any]:
    """Plot a one- or two-dimensional view of evaluated search coordinates.

    Parameters
    ----------
    actions:
        Nonempty matrix with shape ``(N, D)``.
    values:
        Optional length-``N`` colors; evaluation order is used when omitted.
    reduction:
        ``"pca"``, ``"tsne"``, or ``"umap"`` when ``D > 2``.
    ax:
        Optional Matplotlib axes to reuse.

    Returns
    -------
    tuple[Any, Any]
        Matplotlib figure and axes containing the trajectory and colorbar.
    """

    tensor = actions if isinstance(actions, torch.Tensor) else as_tensor(actions)
    if tensor.ndim != 2 or tensor.shape[0] == 0:
        raise ValueError("actions must be a non-empty two-dimensional array.")
    if tensor.shape[1] > 2:
        method = reduction.lower()
        if method == "pca":
            tensor = reduce_pca(tensor, n_components=2)
        elif method == "tsne":
            tensor = reduce_tsne(tensor, n_components=2)
        elif method == "umap":
            tensor = reduce_umap(tensor, n_components=2)
        else:
            raise ReductionError(f"Unsupported trajectory reduction: {reduction!r}.")
    coordinates = to_numpy(tensor)
    if coordinates.shape[1] == 1:
        coordinates = np.column_stack((coordinates[:, 0], np.zeros(coordinates.shape[0])))
    color = np.arange(coordinates.shape[0]) if values is None else _values_1d(values)
    if color.shape[0] != coordinates.shape[0]:
        raise ValueError("values and actions must have equal row counts.")
    fig, axes = _get_axes(ax)
    axes.plot(coordinates[:, 0], coordinates[:, 1], color="0.7", linewidth=1)
    artist = axes.scatter(coordinates[:, 0], coordinates[:, 1], c=color)
    axes.set(xlabel="Component 1", ylabel="Component 2")
    fig.colorbar(artist, ax=axes, label="Evaluation" if values is None else "Value")
    return fig, axes


def plot_stopping_metrics(metrics: Mapping[str, Any], *, ax: Any = None) -> tuple[Any, Any]:
    """Plot named stopping-rule diagnostics over evaluations.

    Parameters
    ----------
    metrics:
        Nonempty mapping from metric names to finite sequences.
    ax:
        Optional Matplotlib axes to reuse.

    Returns
    -------
    tuple[Any, Any]
        Matplotlib figure and axes.
    """

    return plot_comparison(metrics, ax=ax, ylabel="Stopping metric")


__all__ = [
    "DEFAULT_DEVICE",
    "DEFAULT_DTYPE",
    "Bounds",
    "DeviceUnavailableError",
    "InvalidBoundsError",
    "InvalidTensorShapeError",
    "MissingOptionalDependencyError",
    "ModelUtilityError",
    "PriorConfigurationError",
    "ReductionError",
    "SamplingError",
    "TrustRegion",
    "UnsupportedDtypeError",
    "as_tensor",
    "from_symmetric_box",
    "gamma_prior",
    "half_cauchy_prior",
    "lognormal_prior",
    "make_generator",
    "normal_prior",
    "normalize",
    "plot_best_so_far",
    "plot_comparison",
    "plot_iteration_values",
    "plot_regret",
    "plot_search_trajectory",
    "plot_stopping_metrics",
    "reduce_pca",
    "reduce_tsne",
    "reduce_umap",
    "resolve_device",
    "resolve_dtype",
    "sobol_sample",
    "to_numpy",
    "to_symmetric_box",
    "uniform_prior",
    "uniform_sample",
    "unnormalize",
    "validate_bounds",
]
