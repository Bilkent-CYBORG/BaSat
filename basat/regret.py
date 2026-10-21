"""Robust-satisficing regret and finite-domain fragility utilities.

The definitions in this module are independent of an optimizer and of a
plotting backend.  In particular, regret is evaluated from *noiseless*
objective values supplied by the caller; noisy feedback is not silently used
as a proxy for the latent objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def _finite_real(value: Any, name: str) -> float:
    """Return one finite real scalar, rejecting booleans and arrays."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar, not a Boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be a real scalar.") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _finite_p(value: Any) -> float:
    """Validate the exponent used by the finite-p RS-G definition."""
    if isinstance(value, bool):
        raise TypeError("p must be a real scalar, not a Boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("p must be a real scalar.") from error
    if math.isinf(result):
        raise ValueError(
            "p must be finite. p=inf is not evaluated by the finite-p RS-G "
            "formula: its limiting decision rule is convention-dependent and "
            "must be represented as a separate RS-2 policy, not as floating-point infinity."
        )
    if not math.isfinite(result) or result < 1:
        raise ValueError("p must be finite and at least one.")
    return result


def _numpy(value: Any, name: str) -> np.ndarray:
    """Convert NumPy/Torch data to a detached float64 NumPy array."""
    if isinstance(value, torch.Tensor):
        if value.is_complex():
            raise TypeError(f"{name} must be real-valued.")
        array = value.detach().cpu().numpy()
    else:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be numeric array-like data.") from error
        if array.dtype.kind not in "iuf":
            raise TypeError(f"{name} must be real-valued numeric data.")
    try:
        result = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be real-valued numeric data.") from error
    return result


def _epsilon(value: Any) -> float | tuple[float, ...]:
    """Canonicalize a scalar or one-dimensional nonnegative budget."""
    array = _numpy(value, "epsilon")
    if array.ndim == 0:
        result = float(array)
        if not math.isfinite(result) or result < 0:
            raise ValueError("epsilon must be finite and nonnegative.")
        return result
    if array.ndim != 1 or array.size == 0:
        raise ValueError("epsilon must be a scalar or a nonempty one-dimensional sequence.")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("epsilon values must be finite and nonnegative.")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class RSRegret:
    r"""Validated finite-:math:`p` robust-satisficing regret definition.

    Parameters
    ----------
    aspiration : float
        Satisficing threshold :math:`\tau` in objective units.
    optimal_fragility : float
        True optimal finite-domain fragility :math:`\kappa_{\tau,p}`.  This is
        a property of the comparison problem, not an estimate learned from the
        observations.
    epsilon : float or one-dimensional sequence
        Perturbation radius.  A sequence supplies one radius per round.
    p : float, default 1
        Finite RS-G exponent, at least one.  ``p=inf`` is rejected explicitly:
        the RS-2 limiting policy does not define a unique real-valued plug-in
        expression for all choices of radius and fragility.
    cumulative : bool, default True
        Whether :meth:`evaluate` returns the cumulative sum.  Set false for
        instantaneous regret, or call :meth:`instantaneous` directly.

    Notes
    -----
    For noiseless rewards :math:`f(x_t)`, instantaneous regret is

    .. math::

       [\tau - (\kappa_{\tau,p}\epsilon_t)^p - f(x_t)]_+.

    Examples
    --------
    >>> metric = RSRegret(aspiration=-10., optimal_fragility=2., epsilon=.5)
    >>> metric.evaluate([-12., -10., -13.]).tolist()
    [1.0, 1.0, 3.0]
    >>> metric.instantaneous([-12., -10., -13.]).tolist()
    [1.0, 0.0, 2.0]
    """

    aspiration: float
    optimal_fragility: float
    epsilon: float | tuple[float, ...]
    p: float = 1.0
    cumulative: bool = True

    def __post_init__(self) -> None:
        aspiration = _finite_real(self.aspiration, "aspiration")
        fragility = _finite_real(self.optimal_fragility, "optimal_fragility")
        if fragility < 0:
            raise ValueError("optimal_fragility must be nonnegative.")
        if not isinstance(self.cumulative, bool):
            raise TypeError("cumulative must be Boolean.")
        object.__setattr__(self, "aspiration", aspiration)
        object.__setattr__(self, "optimal_fragility", fragility)
        object.__setattr__(self, "epsilon", _epsilon(self.epsilon))
        object.__setattr__(self, "p", _finite_p(self.p))

    def _numpy_instantaneous(self, rewards: np.ndarray) -> np.ndarray:
        if rewards.ndim < 1 or rewards.size == 0:
            raise ValueError("noiseless_rewards must be a nonempty array with rounds on its last axis.")
        if not np.isfinite(rewards).all():
            raise ValueError("noiseless_rewards must contain only finite values.")
        count = rewards.shape[-1]
        epsilon = np.asarray(self.epsilon, dtype=np.float64)
        if epsilon.ndim == 1 and len(epsilon) != count:
            raise ValueError(
                "A per-round epsilon sequence must match the last dimension of noiseless_rewards."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            target = self.aspiration - np.power(self.optimal_fragility * epsilon, self.p)
        if not np.isfinite(target).all():
            raise ValueError("The configured fragility penalty overflows at the requested precision.")
        return np.maximum(target - rewards, 0.0)

    def instantaneous(self, noiseless_rewards: Any) -> np.ndarray | torch.Tensor:
        """Return per-round RS-G regret while preserving a Torch input backend.

        The final dimension is interpreted as time; leading dimensions may be
        trials or other batches.  Integer tensors are promoted to float64.
        """
        if isinstance(noiseless_rewards, torch.Tensor):
            if noiseless_rewards.is_complex():
                raise TypeError("noiseless_rewards must be real-valued.")
            rewards = noiseless_rewards.detach()
            if rewards.ndim < 1 or rewards.numel() == 0:
                raise ValueError(
                    "noiseless_rewards must be a nonempty array with rounds on its last axis."
                )
            if not rewards.is_floating_point():
                rewards = rewards.to(dtype=torch.float64)
            if not bool(torch.isfinite(rewards).all()):
                raise ValueError("noiseless_rewards must contain only finite values.")
            if isinstance(self.epsilon, tuple):
                if len(self.epsilon) != rewards.shape[-1]:
                    raise ValueError(
                        "A per-round epsilon sequence must match the last dimension of "
                        "noiseless_rewards."
                    )
                epsilon = torch.as_tensor(self.epsilon, dtype=rewards.dtype, device=rewards.device)
            else:
                epsilon = torch.as_tensor(self.epsilon, dtype=rewards.dtype, device=rewards.device)
            target = self.aspiration - (self.optimal_fragility * epsilon).pow(self.p)
            if not bool(torch.isfinite(target).all()):
                raise ValueError(
                    "The configured fragility penalty overflows at the requested precision."
                )
            return (target - rewards).clamp_min(0)

        numpy_rewards = _numpy(noiseless_rewards, "noiseless_rewards")
        return self._numpy_instantaneous(numpy_rewards)

    def evaluate(
        self, noiseless_rewards: Any, *, cumulative: bool | None = None,
    ) -> np.ndarray | torch.Tensor:
        """Evaluate instantaneous or cumulative regret from latent rewards.

        ``cumulative=None`` uses the immutable configuration's default.  An
        explicit Boolean overrides it for this call without mutating the
        metric.  NumPy/list inputs return a NumPy array; Torch inputs return a
        tensor on the same device and floating dtype.
        """
        use_cumulative = self.cumulative if cumulative is None else cumulative
        if not isinstance(use_cumulative, bool):
            raise TypeError("cumulative must be Boolean or None.")
        values = self.instantaneous(noiseless_rewards)
        if not use_cumulative:
            return values
        if isinstance(values, torch.Tensor):
            return torch.cumsum(values, dim=-1)
        return np.cumsum(values, axis=-1)

    def statistics(
        self, noiseless_reward_runs: Any, *, error_scale: float = 1.0,
    ) -> RSRegretStatistics:
        """Evaluate rectangular runs and return population mean, SD and bars."""
        values = self.evaluate(noiseless_reward_runs)
        return rs_regret_statistics(values, error_scale=error_scale)


@dataclass(frozen=True)
class FiniteDomainFragility:
    """Immutable result of an exhaustive physical finite-domain calculation."""

    aspiration: float
    p: float
    per_center: tuple[float, ...]
    optimal_fragility: float
    optimal_indices: tuple[int, ...]
    safe_indices: tuple[int, ...]


def finite_domain_optimal_fragility(
    actions: Any, objective_values: Any, *, aspiration: float, p: float = 1.0,
) -> FiniteDomainFragility:
    r"""Compute true finite-domain :math:`p`-fragility using Euclidean distance.

    Every row of ``actions`` is both a candidate center and a possible
    perturbation.  For center :math:`x_i`, the exhaustive finite-domain value is

    .. math::

       \max_{j:f_j<\tau}\frac{(\tau-f_j)^{1/p}}{\lVert x_i-x_j\rVert_2}.

    Centers below the aspiration are assigned infinity because their own
    zero-distance perturbation violates the condition.  A safe center that is
    duplicated at zero distance from an unsafe row is likewise infinite; this
    guards inconsistent tabular objectives instead of dividing by zero.  When
    every row is safe, all center fragilities are zero.

    ``p`` must be finite.  Although Proposition-style decision rules may call
    their limit ``p=inf`` or RS-2, directly substituting infinity into the
    numerical formula is ambiguous (notably at zero distance and at unit
    penalties), so callers must implement that distinct policy explicitly.

    Parameters accept NumPy arrays, Torch tensors, or ordinary sequences.  The
    returned tuples are backend-neutral and cannot be mutated through this
    frozen result object.

    Examples
    --------
    >>> result = finite_domain_optimal_fragility(
    ...     [[0.], [1.], [2.]], [-1., 1., -3.], aspiration=0., p=1.)
    >>> result.per_center
    (inf, 3.0, inf)
    >>> result.optimal_fragility, result.optimal_indices
    (3.0, (1,))
    """
    tau = _finite_real(aspiration, "aspiration")
    exponent = _finite_p(p)
    points = _numpy(actions, "actions")
    values = _numpy(objective_values, "objective_values")
    if points.ndim != 2 or not all(points.shape):
        raise ValueError("actions must be a nonempty two-dimensional array.")
    if values.ndim != 1 or values.shape[0] != points.shape[0]:
        raise ValueError("objective_values must be a vector with one value per action.")
    if not np.isfinite(points).all() or not np.isfinite(values).all():
        raise ValueError("actions and objective_values must contain only finite values.")

    safe = values >= tau
    unsafe = ~safe
    fragility = np.full(points.shape[0], np.inf, dtype=np.float64)
    if not np.any(unsafe):
        fragility[:] = 0.0
    elif np.any(safe):
        unsafe_points = points[unsafe]
        gaps = np.power(tau - values[unsafe], 1.0 / exponent)
        safe_indices = np.flatnonzero(safe)
        differences = points[safe_indices, None, :] - unsafe_points[None, :, :]
        distances = np.linalg.norm(differences, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.divide(
                gaps[None, :], distances,
                out=np.full_like(distances, np.inf), where=distances > 0,
            )
        fragility[safe_indices] = ratios.max(axis=1)

    finite = np.isfinite(fragility)
    if np.any(finite):
        optimum = float(fragility[finite].min())
        optimal = tuple(int(i) for i in np.flatnonzero(fragility == optimum))
    else:
        optimum = math.inf
        optimal = ()
    return FiniteDomainFragility(
        aspiration=tau,
        p=exponent,
        per_center=tuple(float(item) for item in fragility),
        optimal_fragility=optimum,
        optimal_indices=optimal,
        safe_indices=tuple(int(i) for i in np.flatnonzero(safe)),
    )


@dataclass(frozen=True)
class RSRegretStatistics:
    """Immutable population summary of equal-length RS-regret runs.

    ``std`` is the population standard deviation (``ddof=0``).  ``error`` is
    exactly ``std * error_scale``; no confidence-interval interpretation is
    implied.  Use :meth:`to_plot_series` to feed Basat's ordinary plotting API
    without importing Matplotlib in experiment code.
    """

    runs: tuple[tuple[float, ...], ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    error: tuple[float, ...]
    error_scale: float

    @property
    def run_count(self) -> int:
        """Number of independent trajectories in the population."""
        return len(self.runs)

    @property
    def round_count(self) -> int:
        """Number of rounds in every trajectory."""
        return len(self.mean)

    def to_plot_series(self, *, x: Any = None, **style: Any) -> Any:
        """Create a :class:`basat.PlotSeries` with measured half widths.

        The import is lazy and this module never imports Matplotlib directly.
        When ``x`` is omitted, one-based round coordinates are used.
        """
        from .plotting import PlotSeries

        coordinates = np.arange(1, self.round_count + 1) if x is None else x
        return PlotSeries(
            values=np.asarray(self.mean),
            x=coordinates,
            error=np.asarray(self.error),
            **style,
        )


def rs_regret_statistics(regret_runs: Any, *, error_scale: float = 1.0) -> RSRegretStatistics:
    """Aggregate rectangular regret trajectories with population uncertainty.

    Parameters
    ----------
    regret_runs : array-like, shape (runs, rounds)
        Instantaneous or cumulative RS-regret trajectories.  Values must be
        finite and nonnegative.
    error_scale : float, default 1
        Multiplier applied to population SD.  The AdveRS figure's ``SD/2``
        convention is obtained with ``error_scale=0.5``.

    Examples
    --------
    >>> summary = rs_regret_statistics([[1., 2.], [3., 4.]], error_scale=.5)
    >>> summary.mean, summary.std, summary.error
    ((2.0, 3.0), (1.0, 1.0), (0.5, 0.5))
    """
    scale = _finite_real(error_scale, "error_scale")
    if scale < 0:
        raise ValueError("error_scale must be nonnegative.")
    values = _numpy(regret_runs, "regret_runs")
    if values.ndim != 2 or not all(values.shape):
        raise ValueError("regret_runs must be a nonempty rectangular runs-by-rounds array.")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("regret_runs must contain finite nonnegative values.")
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    error = std * scale
    return RSRegretStatistics(
        runs=tuple(tuple(float(item) for item in row) for row in values),
        mean=tuple(float(item) for item in mean),
        std=tuple(float(item) for item in std),
        error=tuple(float(item) for item in error),
        error_scale=scale,
    )


__all__ = [
    "FiniteDomainFragility",
    "RSRegret",
    "RSRegretStatistics",
    "finite_domain_optimal_fragility",
    "rs_regret_statistics",
]
