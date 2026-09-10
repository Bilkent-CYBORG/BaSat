"""BayOPy-owned numerical, reproducibility, and file helpers for user workflows.

Backend packages remain implementation details: tutorial users import BayOPy
alone. Arrays returned here are float64 Torch tensors unless specified otherwise.
"""

from __future__ import annotations

import importlib.metadata
import platform
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

_seed = 1


def set_seed(seed: int, *, threads: int | None = None) -> int:
    """Seed Python, NumPy, Torch (including CUDA), and the Pyro Torch backend.

    BayOPy's high-level factories use this seed when none is supplied. Explicit
    per-algorithm/config seeds take precedence. This does not retroactively reseed
    existing algorithm generators or promise identical accelerator arithmetic.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    if threads is not None:
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer.")
        torch.set_num_threads(threads)
    global _seed
    _seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def get_seed() -> int:
    """Return the default seed used by BayOPy workflow factories."""
    return _seed


def array(values: Any, *, dtype: str = "float64", device: str = "cpu") -> torch.Tensor:
    """Create a numeric array with explicit precision and device."""
    from .models.utils import resolve_dtype

    return torch.as_tensor(values, dtype=resolve_dtype(dtype), device=device)


def linspace(start: float, stop: float, count: int) -> torch.Tensor:
    """Return evenly spaced float64 coordinates, including both endpoints."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer.")
    return torch.linspace(start, stop, count, dtype=torch.float64)


def to_list(value: Any) -> Any:
    """Convert arrays and nested mappings to ordinary Python values."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_list(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_list(v) for v in value]
    return value


def square(value: Any) -> torch.Tensor:
    """Elementwise square of a BayOPy array or numeric sequence."""
    return torch.square(torch.as_tensor(value))


def sum(value: Any, *, axis: int | None = None) -> torch.Tensor:
    """Sum numeric elements, optionally along an axis."""
    return torch.sum(torch.as_tensor(value), dim=axis)


def mean(value: Any, *, axis: int | None = None) -> torch.Tensor:
    """Mean of numeric elements, optionally along an axis."""
    return torch.mean(array(value), dim=axis)


def sin(value: Any) -> torch.Tensor:
    """Elementwise sine in radians."""
    return torch.sin(array(value))


def runtime_info() -> dict[str, str]:
    """Report Python and installed backend versions without external imports."""
    result = {"python": platform.python_version()}
    for name in (
        "bayopy",
        "numpy",
        "torch",
        "gpytorch",
        "botorch",
        "scikit-learn",
        "matplotlib",
        "pyro-ppl",
        "cvxpy",
        "cma",
        "jupyterlab",
    ):
        result[name] = importlib.metadata.version(name)
    return result


def new_run_directory(parent: str = "results/bayopy_runs") -> str:
    """Create a unique run directory without overwriting earlier checkpoints."""
    path = Path(parent).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(prefix="run_", dir=path)


def join_path(directory: str, *parts: str) -> str:
    """Construct an output path portably."""
    return str(Path(directory).joinpath(*parts))


def save_dataset(dataset: Any, path: str) -> str:
    """Save an EnvironmentDataset as CSV or NPZ, preserving context and noise."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    values = {"actions": to_list(dataset.actions), "observations": to_list(dataset.observations)}
    for field in ("contexts", "observation_noise_variance"):
        value = getattr(dataset, field, None)
        if value is not None:
            values[field] = to_list(value)
    if target.suffix.lower() == ".npz":
        np.savez(target, **values)
    elif target.suffix.lower() == ".csv":
        actions = np.asarray(values["actions"])
        columns = [actions, np.asarray(values["observations"]).reshape(-1, 1)]
        names = [f"action_{i}" for i in range(actions.shape[1])] + ["observation"]
        if "contexts" in values:
            contexts = np.asarray(values["contexts"])
            columns.append(contexts)
            names.extend(f"context_{i}" for i in range(contexts.shape[1]))
        if "observation_noise_variance" in values:
            columns.append(np.asarray(values["observation_noise_variance"]).reshape(-1, 1))
            names.append("observation_noise_variance")
        np.savetxt(
            target, np.column_stack(columns), delimiter=",", header=",".join(names), comments=""
        )
    else:
        raise ValueError("Dataset output must end in .csv or .npz.")
    return str(target)


def find_path(path: str) -> str:
    """Find an existing file/directory from the current folder or its parents.

    Useful when the same notebook is launched from a project root or docs folder.
    This locates user-owned files; it never downloads or creates data.
    """
    from pathlib import Path

    requested = Path(path)
    for parent in (Path.cwd(), *Path.cwd().parents):
        candidate = parent / requested
        if candidate.exists():
            return str(candidate.resolve())
    raise FileNotFoundError(f"Cannot find {path!r} in the current folder or its parents.")


def save_csv(path: str, rows: Any, *, columns: Any) -> str:
    """Write an explicitly named rectangular table using Python values.

    Column roles belong to the caller; this helper never infers a benchmark or
    computes rewards. Use ``load_dataset(..., environment=...)`` to validate the
    resulting table against an environment's reserved column contract.
    """
    import csv
    import io

    names = list(columns)
    if not names or any(not isinstance(x, str) or not x.strip() for x in names):
        raise ValueError("columns must contain nonempty names.")
    if len(set(names)) != len(names):
        raise ValueError("CSV column names must be unique.")
    values = to_list(rows)
    if not isinstance(values, (list, tuple)) or any(
        not isinstance(row, (list, tuple)) or len(row) != len(names) for row in values
    ):
        raise ValueError("rows must be a rectangular table matching columns.")
    # Build before replacing the destination, so malformed input cannot truncate it.
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(names)
    writer.writerows(values)
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(buffer.getvalue(), encoding="utf-8")
    return str(target)
