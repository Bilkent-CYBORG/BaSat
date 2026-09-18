"""Numerical primitives for explicit released tree-search protocols.

These helpers know neither objective functions nor experiment budgets. Random
streams are owned by the calling algorithm and are serialized with its state.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np
import torch


def random_state_dict(rng: np.random.RandomState) -> dict[str, Any]:
    """Return a JSON-compatible snapshot of an owned MT19937 stream."""
    snapshot: Any = rng.get_state()
    name, keys, position, has_gaussian, cached_gaussian = snapshot
    return {
        "name": name,
        "keys": keys.tolist(),
        "position": position,
        "has_gaussian": has_gaussian,
        "cached_gaussian": cached_gaussian,
    }


def restore_random_state(rng: np.random.RandomState, state: Mapping[str, Any]) -> None:
    """Validate and restore an MT19937 snapshot without touching global RNGs."""
    if state.get("name") != "MT19937" or len(state.get("keys", [])) != 624:
        raise ValueError("Invalid released-protocol MT19937 state.")
    if any(
        not isinstance(key, Integral) or isinstance(key, bool) or not 0 <= int(key) < 2**32
        for key in state["keys"]
    ):
        raise ValueError("MT19937 keys must be 624 unsigned 32-bit integers.")
    position, has_gaussian = state.get("position"), state.get("has_gaussian")
    if (
        not isinstance(position, Integral)
        or isinstance(position, bool)
        or not 0 <= int(position) <= 624
    ):
        raise ValueError("Invalid MT19937 position.")
    if (
        not isinstance(has_gaussian, Integral)
        or isinstance(has_gaussian, bool)
        or has_gaussian not in (0, 1)
    ):
        raise ValueError("Invalid MT19937 cached-normal flag.")
    if not np.isfinite(state.get("cached_gaussian", np.nan)):
        raise ValueError("Invalid MT19937 cached-normal value.")
    rng.set_state(
        (
            "MT19937",
            np.asarray(state["keys"], dtype=np.uint32),
            int(state["position"]),
            int(state["has_gaussian"]),
            float(state["cached_gaussian"]),
        )
    )


def validate_random_states(state: Mapping[str, Any]) -> None:
    """Check both source RNG snapshots using disposable, private streams."""
    restore_random_state(np.random.RandomState(0), state["source_numpy_rng"])
    torch.Generator().set_state(
        torch.as_tensor(state["source_torch_rng"], dtype=torch.uint8, device="cpu")
    )


def latin_hypercube(rng: np.random.RandomState, count: int, dimension: int) -> np.ndarray:
    """Stratified Latin design with the released center-then-perturb arithmetic."""
    centers = (1.0 + 2.0 * np.arange(count)) / (2.0 * count)
    design = np.empty((count, dimension))
    for coordinate in range(dimension):
        design[:, coordinate] = centers[rng.permutation(count)]
    design += rng.uniform(-1.0, 1.0, design.shape) / (2.0 * count)
    return design


def sobol_points(rng: np.random.RandomState, count: int, dimension: int) -> np.ndarray:
    """Draw the release's float32 Sobol sequence, returned as NumPy float64."""
    seed = int(rng.randint(500_000))
    return (
        torch.quasirandom.SobolEngine(dimension, scramble=True, seed=seed)
        .draw(count)
        .double()
        .numpy()
    )


def expected_improvement(
    gp: Any, candidates: np.ndarray, reference: np.ndarray, *, xi: float
) -> np.ndarray:
    """Pointwise EI with a fitted-posterior-mean incumbent and SciPy tails."""
    from scipy.stats import norm

    mean, std = gp.predict(candidates, return_std=True)
    improvement = mean - np.max(gp.predict(reference)) - xi
    with np.errstate(divide="ignore", invalid="ignore"):
        z = improvement / std
        scores = improvement * norm.cdf(z) + std * norm.pdf(z)
    return np.where(std == 0, 0.0, scores)
