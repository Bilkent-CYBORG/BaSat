"""Acquisition construction and continuous or finite-candidate optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from botorch.acquisition import (
    LogExpectedImprovement,
    UpperConfidenceBound,
    qLogExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from torch.quasirandom import SobolEngine

from basat.warnings import translate_backend_warnings


class AcquisitionError(Exception):
    """Base class for acquisition construction and optimization errors."""


class InvalidAcquisitionConfigError(AcquisitionError, ValueError):
    """Raised when acquisition parameters are invalid or incomplete."""


class DiscreteAcquisitionError(AcquisitionError, ValueError):
    """Raised when a finite candidate table violates its row contract."""


class NoAvailableCandidateError(DiscreteAcquisitionError):
    """Raised when a finite acquisition request cannot select ``q`` rows."""


def _optimize_acqf_with_basat_warnings(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Run BoTorch optimization while exposing only Basat-owned backend warnings."""

    with translate_backend_warnings(operation="continuous acquisition optimization"):
        return optimize_acqf(**kwargs)


@dataclass(frozen=True, slots=True)
class AcquisitionConfig:
    """Configuration shared by Basat's standard acquisitions.

    Parameters
    ----------
    name:
        ``"EI"`` or ``"UCB"`` (case-insensitive).
    beta:
        Non-negative UCB exploration coefficient.
    xi:
        Non-negative improvement offset added to EI's incumbent threshold.
    mc_samples:
        Number of Sobol-QMC posterior samples for batch acquisitions.
    seed:
        Seed for the independent acquisition sampler.

    Examples
    --------
    >>> import basat
    >>> acquisition = basat.AcquisitionConfig(
    ...     name="EI", beta=2.0, xi=0.0, mc_samples=512, seed=42
    ... )
    """

    name: str = "EI"
    beta: float = 2.0
    xi: float = 0.0
    mc_samples: int = 512
    seed: int = 1


@dataclass(frozen=True, slots=True)
class DiscreteAcquisitionCandidates:
    """One-to-one mapping from finite environment rows to model coordinates.

    Parameters
    ----------
    environment_actions:
        Available actions of shape ``(M, D_env)``.
    model_inputs:
        Algorithm-specific representations of shape ``(M, D_model)``.
    dataset_indices:
        Hidden-table row identifiers of shape ``(M,)``.  No observations are
        stored in this learner-visible mapping.
    """

    environment_actions: torch.Tensor
    model_inputs: torch.Tensor
    dataset_indices: torch.Tensor


@dataclass(frozen=True, slots=True)
class DiscreteAcquisitionResult:
    """Selected finite rows and their aligned coordinate representations.

    Parameters
    ----------
    selected_rows:
        Positions in the supplied finite candidate table.
    environment_actions:
        Selected complete actions shaped ``(q, D_env)``.
    model_inputs:
        Aligned selected representations shaped ``(q, D_model)``.
    dataset_indices:
        Stable selected hidden-table identifiers shaped ``(q,)``.
    acquisition_values:
        One acquisition score for each selected row.
    """

    selected_rows: torch.Tensor
    environment_actions: torch.Tensor
    model_inputs: torch.Tensor
    dataset_indices: torch.Tensor
    acquisition_values: torch.Tensor


def _underlying_model(model: Any) -> Any:
    """Accept either a BoTorch model or Basat's GaussianProcess wrapper."""

    wrapped = getattr(model, "model", None)
    if wrapped is not None and hasattr(wrapped, "posterior"):
        if hasattr(model, "is_fit") and not model.is_fit:
            raise InvalidAcquisitionConfigError("The GaussianProcess must be fit first.")
        return wrapped
    if hasattr(model, "posterior") and hasattr(model, "num_outputs"):
        return model
    raise InvalidAcquisitionConfigError(
        "model must expose the BoTorch Model contract or contain a fitted model."
    )


def _resolve_batch_request(
    *,
    batch_size: int | None,
    q_batch: int | None,
    q: int | None,
    error_type: type[Exception] = InvalidAcquisitionConfigError,
) -> tuple[str, int]:
    """Resolve independent, joint, and legacy acquisition batch arguments."""

    if q_batch is not None and q is not None:
        raise error_type("Choose exactly one of batch_size or q_batch (legacy q cannot be mixed).")
    if q_batch is not None:
        if batch_size not in (None, 1):
            raise error_type("Choose exactly one of batch_size or q_batch.")
        name, value, mode = "q_batch", q_batch, "joint"
    elif q is not None:
        if batch_size not in (None, 1):
            raise error_type("Legacy q cannot be mixed with batch_size.")
        name, value, mode = "q", q, "joint"
    elif batch_size is not None:
        name, value, mode = "batch_size", batch_size, "independent"
    else:
        name, value, mode = "batch_size", 1, "independent"
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise error_type(f"{name} must be a positive integer.")
    return mode, value


def build_acquisition(
    *,
    model: Any,
    config: AcquisitionConfig,
    batch_size: int | None = 1,
    q_batch: int | None = None,
    q: int | None = None,
    best_f: torch.Tensor | float | None = None,
) -> Any:
    """Build an independent one-point or jointly scored q-acquisition.

    The returned object consumes model-space coordinates.  This function has no
    access to ``D_env`` and cannot manufacture finite environment actions.

    Parameters
    ----------
    model:
        A fitted Basat GP wrapper or compatible BoTorch model.
    config:
        Acquisition name, exploration parameters, and Monte Carlo seed.
    batch_size:
        Number of independently scored points.  The returned acquisition stays
        analytic and one-point, irrespective of this count.
    q_batch:
        Number of points whose acquisition value is evaluated jointly.  For
        example, EI becomes qEI when ``q_batch > 1``.
    q:
        Deprecated compatibility alias for ``q_batch``.
    best_f:
        Finite scalar incumbent required by expected improvement.

    Returns
    -------
    Any
        A BoTorch analytic or Monte Carlo acquisition object.

    Raises
    ------
    InvalidAcquisitionConfigError
        If the model, configuration, batch size, or incumbent is invalid.
    """

    if not isinstance(config, AcquisitionConfig):
        raise InvalidAcquisitionConfigError("config must be an AcquisitionConfig instance.")
    mode, count = _resolve_batch_request(
        batch_size=batch_size,
        q_batch=q_batch,
        q=q,
    )
    acquisition_q = count if mode == "joint" else 1
    if not isinstance(config.mc_samples, int) or isinstance(config.mc_samples, bool):
        raise InvalidAcquisitionConfigError("mc_samples must be a positive integer.")
    if config.mc_samples < 1:
        raise InvalidAcquisitionConfigError("mc_samples must be a positive integer.")
    if not isinstance(config.seed, int) or isinstance(config.seed, bool) or config.seed < 0:
        raise InvalidAcquisitionConfigError("seed must be a non-negative integer.")
    if not torch.isfinite(torch.tensor(float(config.beta))) or config.beta < 0:
        raise InvalidAcquisitionConfigError("beta must be finite and non-negative.")
    if not torch.isfinite(torch.tensor(float(config.xi))) or config.xi < 0:
        raise InvalidAcquisitionConfigError("xi must be finite and non-negative.")
    botorch_model = _underlying_model(model)
    if not isinstance(config.name, str):
        raise InvalidAcquisitionConfigError("name must be a string identifying EI or UCB.")
    name = config.name.lower().replace("-", "").replace("_", "")
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([config.mc_samples]), seed=config.seed)
    if name in {"ei", "expectedimprovement"}:
        if best_f is None:
            raise InvalidAcquisitionConfigError("EI requires best_f.")
        threshold = (
            torch.as_tensor(
                best_f,
                dtype=getattr(botorch_model, "train_targets", torch.tensor(0.0)).dtype,
                device=getattr(botorch_model, "train_targets", torch.tensor(0.0)).device,
            )
            + config.xi
        )
        if threshold.numel() != 1 or not torch.isfinite(threshold).all():
            raise InvalidAcquisitionConfigError("best_f must be one finite scalar.")
        if acquisition_q == 1:
            return LogExpectedImprovement(model=botorch_model, best_f=threshold)
        return qLogExpectedImprovement(model=botorch_model, best_f=threshold, sampler=sampler)
    if name in {"ucb", "upperconfidencebound"}:
        if acquisition_q == 1:
            return UpperConfidenceBound(model=botorch_model, beta=config.beta)
        return qUpperConfidenceBound(model=botorch_model, beta=config.beta, sampler=sampler)
    raise InvalidAcquisitionConfigError(
        f"Unsupported acquisition {config.name!r}; expected EI or UCB."
    )


def validate_discrete_candidates(candidates: DiscreteAcquisitionCandidates) -> None:
    """Validate a confidentiality-safe finite candidate row table.

    Parameters
    ----------
    candidates:
        Complete environment/model rows and stable hidden-table indices.

    Raises
    ------
    DiscreteAcquisitionError
        If shapes, dtypes, devices, indices, or coordinates are invalid.
    NoAvailableCandidateError
        If the candidate table has no rows.
    """

    if not isinstance(candidates, DiscreteAcquisitionCandidates):
        raise DiscreteAcquisitionError(
            "candidates must be a DiscreteAcquisitionCandidates instance."
        )
    environment_actions = candidates.environment_actions
    model_inputs = candidates.model_inputs
    dataset_indices = candidates.dataset_indices
    if environment_actions.ndim != 2:
        raise DiscreteAcquisitionError("environment_actions must be two-dimensional.")
    if model_inputs.ndim != 2:
        raise DiscreteAcquisitionError("model_inputs must be two-dimensional.")
    if dataset_indices.ndim != 1:
        raise DiscreteAcquisitionError("dataset_indices must be one-dimensional.")
    row_count = environment_actions.shape[0]
    if model_inputs.shape[0] != row_count or dataset_indices.shape[0] != row_count:
        raise DiscreteAcquisitionError("Candidate row counts do not match.")
    if row_count == 0:
        raise NoAvailableCandidateError("No finite candidate is available.")
    if model_inputs.shape[1] == 0 or environment_actions.shape[1] == 0:
        raise DiscreteAcquisitionError("Candidate coordinate dimensions cannot be empty.")
    if not torch.is_floating_point(environment_actions) or not torch.is_floating_point(
        model_inputs
    ):
        raise DiscreteAcquisitionError(
            "environment_actions and model_inputs must be floating-point tensors."
        )
    if not torch.isfinite(environment_actions).all() or not torch.isfinite(model_inputs).all():
        raise DiscreteAcquisitionError("Candidate coordinates must be finite.")
    if dataset_indices.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise DiscreteAcquisitionError("dataset_indices must use an integer dtype.")
    if (dataset_indices < 0).any():
        raise DiscreteAcquisitionError("dataset_indices cannot be negative.")
    devices = {environment_actions.device, model_inputs.device, dataset_indices.device}
    if len(devices) != 1:
        raise DiscreteAcquisitionError("All candidate tensors must be on the same device.")


def optimize_acquisition_continuous(
    *,
    acquisition: Any,
    bounds: torch.Tensor,
    batch_size: int | None = 1,
    q_batch: int | None = None,
    q: int | None = None,
    num_restarts: int,
    raw_samples: int,
    constraints: Any = None,
    sampling_method: Literal["paper", "sobol", "uniform"] = "paper",
    sampling_generator: torch.Generator | None = None,
    sampling_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimize an acquisition over an algorithm-defined continuous box.

    ``bounds`` are in model/search coordinates and may therefore have a
    different dimensionality from the environment.

    Parameters
    ----------
    acquisition:
        Callable BoTorch acquisition in model coordinates.
    bounds:
        Floating tensor with shape ``(2, D_model)``.
    batch_size:
        Number of points selected by their separate one-point scores.
    q_batch:
        Positive joint batch size.
    q:
        Deprecated compatibility alias for ``q_batch``.
    num_restarts:
        Number of multistart local solves.
    raw_samples:
        Minimum number of initialization samples; it is raised to at least
        ``num_restarts`` when required by BoTorch.
    constraints:
        Optional BoTorch inequality-constraint specification.
    sampling_method:
        ``paper`` retains local BoTorch optimization. ``sobol`` or ``uniform``
        instead scores a distinct raw candidate design directly. Joint-q mode
        scores complete sampled sets together.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Optimizing candidates and their acquisition values.

    Raises
    ------
    AcquisitionError
        If validation or the continuous optimizer fails.
    """

    if not isinstance(bounds, torch.Tensor) or bounds.ndim != 2 or bounds.shape[0] != 2:
        raise AcquisitionError("bounds must be a floating tensor of shape (2, D_model).")
    if bounds.shape[1] == 0 or not torch.is_floating_point(bounds):
        raise AcquisitionError("bounds must contain floating model coordinates.")
    if not torch.isfinite(bounds).all() or not torch.all(bounds[0] < bounds[1]):
        raise AcquisitionError("bounds must be finite with lower < upper.")
    mode, count = _resolve_batch_request(
        batch_size=batch_size,
        q_batch=q_batch,
        q=q,
        error_type=AcquisitionError,
    )
    for name, value in {
        "num_restarts": num_restarts,
        "raw_samples": raw_samples,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise AcquisitionError(f"{name} must be a positive integer.")
    if sampling_method not in {"paper", "sobol", "uniform"}:
        raise AcquisitionError("sampling_method must be 'paper', 'sobol', or 'uniform'.")
    if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int) or sampling_seed < 0:
        raise AcquisitionError("sampling_seed must be a non-negative integer.")
    if sampling_method != "paper":
        if constraints is not None:
            raise AcquisitionError(
                "Explicit sampled acquisition candidates do not support constraints."
            )
        pool_size = max(raw_samples, count * 16)
        draw_count = pool_size * count if mode == "joint" else pool_size
        if sampling_method == "sobol":
            # A joint q-batch is one point in q * D dimensions. Grouping
            # consecutive D-dimensional Sobol points introduces dependencies
            # and is not a Sobol design over the joint search space.
            dimension = bounds.shape[1] * (count if mode == "joint" else 1)
            engine = SobolEngine(dimension=dimension, scramble=True, seed=sampling_seed)
            unit = engine.draw(pool_size, dtype=bounds.dtype).to(bounds.device)
            unit = unit.reshape(draw_count, bounds.shape[1])
        else:
            if not isinstance(sampling_generator, torch.Generator):
                raise AcquisitionError("uniform acquisition sampling requires sampling_generator.")
            unit = torch.rand(
                (draw_count, bounds.shape[1]),
                dtype=bounds.dtype,
                device=bounds.device,
                generator=sampling_generator,
            )
        pool = bounds[0] + (bounds[1] - bounds[0]) * unit
        pool = _distinct_rows_in_order(pool)
        if pool.shape[0] < draw_count:
            raise AcquisitionError("The sampler returned duplicate acquisition candidates.")
        with torch.no_grad():
            if mode == "joint":
                sets = pool.reshape(pool_size, count, bounds.shape[1])
                values = acquisition(sets).reshape(-1)
                if values.numel() != pool_size or not torch.isfinite(values).all():
                    raise AcquisitionError(
                        "The acquisition must return one finite score per sampled batch."
                    )
                position = int(torch.argmax(values).item())
                return sets[position], values[position]
            values = acquisition(pool.unsqueeze(-2)).reshape(-1)
            if values.numel() != pool.shape[0] or not torch.isfinite(values).all():
                raise AcquisitionError(
                    "The acquisition must return one finite score per sampled point."
                )
            selected = torch.argsort(values, descending=True, stable=True)[:count]
            return pool[selected], values[selected]
    try:
        # BoTorch selects restart seeds without replacement, so its raw design
        # must contain at least one point per requested restart.  Treat the
        # public ``raw_samples`` value as a minimum sampling budget rather than
        # silently reducing the explicitly requested multistart count.
        effective_raw_samples = max(raw_samples, num_restarts)
        # SciPy's box-constrained L-BFGS-B default permits only 20 line-search
        # steps. Smooth GP acquisitions can need more near flat optima or box
        # boundaries; a larger budget prevents recoverable ``ABNORMAL`` exits
        # without filtering BoTorch warnings or changing the selected solver.
        optimizer_options: dict[str, bool | float | int | str] | None = (
            None if constraints is not None else {"maxls": 100}
        )
        if mode == "joint" or count == 1:
            return _optimize_acqf_with_basat_warnings(
                acq_function=acquisition,
                bounds=bounds,
                q=count,
                num_restarts=num_restarts,
                raw_samples=effective_raw_samples,
                options=optimizer_options,
                inequality_constraints=constraints,
                sequential=False,
            )
        if constraints is not None:
            raise AcquisitionError(
                "Independent continuous batches with constraints are not supported; "
                "use q_batch or batch_size=1."
            )

        # Keep the acquisition one-point.  Combine its optimized maximizer with
        # a deterministic Sobol candidate pool, score every row independently,
        # and retain the best distinct rows.  The pool prevents a multimethod
        # local optimizer from returning the same maximizer ``count`` times.
        best_point, _ = _optimize_acqf_with_basat_warnings(
            acq_function=acquisition,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=effective_raw_samples,
            options=optimizer_options,
            sequential=False,
        )
        pool_size = max(effective_raw_samples, count * 16)
        engine = SobolEngine(dimension=bounds.shape[1], scramble=True, seed=sampling_seed)
        unit = engine.draw(pool_size, dtype=bounds.dtype).to(bounds.device)
        pool = bounds[0] + (bounds[1] - bounds[0]) * unit
        pool = _distinct_rows_in_order(torch.cat((best_point.reshape(1, -1), pool), dim=0))
        if pool.shape[0] < count:
            raise AcquisitionError("Too few distinct acquisition candidates for batch_size.")
        with torch.no_grad():
            values = acquisition(pool.unsqueeze(-2)).reshape(-1)
        if values.numel() != pool.shape[0] or not torch.isfinite(values).all():
            raise AcquisitionError("The acquisition must return one finite score per point.")
        order = torch.argsort(values, descending=True, stable=True)[:count]
        return pool[order], values[order]
    except (RuntimeError, ValueError) as exc:
        raise AcquisitionError(f"Continuous acquisition optimization failed: {exc}") from exc


def _distinct_rows_in_order(rows: torch.Tensor) -> torch.Tensor:
    """Remove duplicates without rearranging randomly sampled joint batches."""

    unique, inverse = torch.unique(rows, dim=0, return_inverse=True)
    first = torch.full((unique.shape[0],), rows.shape[0], device=rows.device, dtype=torch.long)
    first.scatter_reduce_(
        0, inverse, torch.arange(rows.shape[0], device=rows.device), reduce="amin"
    )
    return rows[torch.sort(first).values]


def _best_position(values: torch.Tensor, dataset_indices: torch.Tensor) -> int:
    """Resolve score ties by the smallest stable hidden-table index."""

    flat = values.reshape(-1)
    if flat.numel() != dataset_indices.numel():
        raise DiscreteAcquisitionError("The acquisition must return one scalar per candidate row.")
    if not torch.isfinite(flat).all():
        raise DiscreteAcquisitionError("The acquisition returned a non-finite score.")
    maximum = flat.max()
    tied = torch.nonzero(flat == maximum, as_tuple=False).reshape(-1)
    tied_indices = dataset_indices[tied]
    return int(tied[torch.argmin(tied_indices)].item())


def _optimize_discrete_q1(
    *, acquisition: Any, candidates: DiscreteAcquisitionCandidates
) -> DiscreteAcquisitionResult:
    """Evaluate a sequential acquisition on each available finite row."""

    X = candidates.model_inputs.unsqueeze(-2)
    with torch.no_grad():
        values = acquisition(X).reshape(-1)
    best_position = _best_position(values, candidates.dataset_indices)
    selected_rows = torch.tensor(
        [best_position], device=candidates.model_inputs.device, dtype=torch.long
    )
    return DiscreteAcquisitionResult(
        selected_rows=selected_rows,
        environment_actions=candidates.environment_actions[selected_rows],
        model_inputs=candidates.model_inputs[selected_rows],
        dataset_indices=candidates.dataset_indices[selected_rows],
        acquisition_values=values[selected_rows],
    )


def _optimize_discrete_independent(
    *, acquisition: Any, candidates: DiscreteAcquisitionCandidates, batch_size: int
) -> DiscreteAcquisitionResult:
    """Select distinct finite rows by independent one-point acquisition score."""

    row_count = candidates.model_inputs.shape[0]
    if batch_size > row_count:
        raise NoAvailableCandidateError(
            f"batch_size={batch_size} is larger than the available candidate set ({row_count})."
        )
    with torch.no_grad():
        values = acquisition(candidates.model_inputs.unsqueeze(-2)).reshape(-1)
    if values.numel() != row_count:
        raise DiscreteAcquisitionError("The acquisition must return one scalar per candidate row.")
    if not torch.isfinite(values).all():
        raise DiscreteAcquisitionError("The acquisition returned a non-finite score.")

    # Stable two-key ordering: score descending, then hidden dataset index
    # ascending for exact score ties.
    by_index = torch.argsort(candidates.dataset_indices, stable=True)
    by_score = torch.argsort(values[by_index], descending=True, stable=True)
    selected_rows = by_index[by_score[:batch_size]]
    return DiscreteAcquisitionResult(
        selected_rows=selected_rows,
        environment_actions=candidates.environment_actions[selected_rows],
        model_inputs=candidates.model_inputs[selected_rows],
        dataset_indices=candidates.dataset_indices[selected_rows],
        acquisition_values=values[selected_rows],
    )


def _optimize_discrete_q_batch(
    *, acquisition: Any, candidates: DiscreteAcquisitionCandidates, q: int
) -> DiscreteAcquisitionResult:
    """Greedily condition a q-acquisition over the remaining finite rows."""

    row_count = candidates.model_inputs.shape[0]
    if q > row_count:
        raise NoAvailableCandidateError(
            f"q={q} is larger than the available candidate set ({row_count})."
        )
    setter = getattr(acquisition, "set_X_pending", None)
    if not callable(setter):
        raise DiscreteAcquisitionError(
            "Batch finite optimization requires a q-acquisition with set_X_pending()."
        )
    base_pending = getattr(acquisition, "X_pending", None)
    remaining = torch.arange(row_count, device=candidates.model_inputs.device, dtype=torch.long)
    selected_rows: list[torch.Tensor] = []
    selected_values: list[torch.Tensor] = []
    try:
        for _ in range(q):
            if selected_rows:
                pending = candidates.model_inputs[torch.stack(selected_rows)]
                if base_pending is not None:
                    pending = torch.cat((base_pending, pending), dim=-2)
                setter(pending)
            X = candidates.model_inputs[remaining].unsqueeze(-2)
            with torch.no_grad():
                values = acquisition(X).reshape(-1)
            local_position = _best_position(values, candidates.dataset_indices[remaining])
            row = remaining[local_position]
            selected_rows.append(row)
            selected_values.append(values[local_position])
            keep = torch.arange(remaining.shape[0], device=remaining.device) != local_position
            remaining = remaining[keep]
    finally:
        setter(base_pending)
    row_tensor = torch.stack(selected_rows)
    return DiscreteAcquisitionResult(
        selected_rows=row_tensor,
        environment_actions=candidates.environment_actions[row_tensor],
        model_inputs=candidates.model_inputs[row_tensor],
        dataset_indices=candidates.dataset_indices[row_tensor],
        acquisition_values=torch.stack(selected_values),
    )


def optimize_acquisition_discrete(
    *,
    acquisition: Any,
    candidates: DiscreteAcquisitionCandidates,
    batch_size: int | None = 1,
    q_batch: int | None = None,
    q: int | None = None,
) -> DiscreteAcquisitionResult:
    """Optimize an acquisition without generating unevaluable actions.

    For ``q > 1``, the supplied q-acquisition is conditioned on already selected
    rows at each greedy step.  This is intentionally different from taking the
    largest ``q`` independent one-point scores.

    Parameters
    ----------
    acquisition:
        One-point or q-aware acquisition evaluated in ``D_model`` coordinates.
    candidates:
        Complete finite rows linking model inputs to environment actions.
    batch_size:
        Number of rows selected by separate one-point scores.
    q_batch:
        Number of rows selected with pending-point-aware joint scoring.
    q:
        Deprecated compatibility alias for ``q_batch``.

    Returns
    -------
    DiscreteAcquisitionResult
        Selected row positions, indices, actions, model inputs, and scores.

    Raises
    ------
    DiscreteAcquisitionError
        If the acquisition or candidate table violates the finite-row contract.
    NoAvailableCandidateError
        If fewer than ``q`` rows are available.
    """

    validate_discrete_candidates(candidates)
    mode, count = _resolve_batch_request(
        batch_size=batch_size,
        q_batch=q_batch,
        q=q,
        error_type=ValueError,
    )
    if mode == "independent" and count > 1:
        return _optimize_discrete_independent(
            acquisition=acquisition,
            candidates=candidates,
            batch_size=count,
        )
    if count == 1:
        return _optimize_discrete_q1(acquisition=acquisition, candidates=candidates)
    return _optimize_discrete_q_batch(acquisition=acquisition, candidates=candidates, q=count)


__all__ = [
    "AcquisitionConfig",
    "AcquisitionError",
    "DiscreteAcquisitionCandidates",
    "DiscreteAcquisitionError",
    "DiscreteAcquisitionResult",
    "InvalidAcquisitionConfigError",
    "NoAvailableCandidateError",
    "build_acquisition",
    "optimize_acquisition_continuous",
    "optimize_acquisition_discrete",
    "validate_discrete_candidates",
]
