"""Owned and checkpointable streams for explicit historical Ax protocols.

The optional initializer stream replaces unrecorded ``RandomState(None)``
entropy. It is independent of the source's NumPy/Torch proposal streams.
These helpers never read or modify process-global random states.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from ._tree_numerics import random_state_dict, restore_random_state


def export_streams(
    numpy_rng: np.random.RandomState,
    torch_generator: torch.Generator,
    initializer_seed_stream: np.random.RandomState | None = None,
) -> dict[str, Any]:
    """Copy streams, including cached normal draws, without advancing them."""
    state = {
        "numpy": random_state_dict(numpy_rng),
        "torch": torch_generator.get_state().detach().cpu().clone(),
    }
    if initializer_seed_stream is not None:
        state["initializer_seed_stream"] = random_state_dict(initializer_seed_stream)
    return state


def validate_streams(
    state: Mapping[str, Any],
) -> tuple[np.random.RandomState, torch.Generator, np.random.RandomState | None]:
    """Validate on disposable objects, permitting atomic imports by callers."""
    if not isinstance(state, Mapping) or set(state) not in (
        {"numpy", "torch"},
        {"numpy", "torch", "initializer_seed_stream"},
    ):
        raise ValueError("Expected numpy and torch streams, and optional initializer_seed_stream.")
    numpy_rng = np.random.RandomState(0)
    restore_random_state(numpy_rng, state["numpy"])
    value = torch.as_tensor(state["torch"], device="cpu")
    if (
        value.ndim != 1
        or value.is_floating_point()
        or value.dtype == torch.bool
        or bool(((value < 0) | (value > 255)).any())
    ):
        raise ValueError("Torch RNG state must be a one-dimensional byte sequence.")
    generator = torch.Generator(device="cpu")
    generator.set_state(value.to(dtype=torch.uint8).contiguous())
    initializer = None
    if "initializer_seed_stream" in state:
        initializer = np.random.RandomState(0)
        restore_random_state(initializer, state["initializer_seed_stream"])
    return numpy_rng, generator, initializer
