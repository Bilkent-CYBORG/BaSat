"""Portable, validated loading of user-owned finite optimization datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .environments import EnvironmentDataset
from .models.utils import Bounds

DatasetFamily = Literal["stochastic", "contextual", "adversarial", "distributional"]


@dataclass(frozen=True, slots=True)
class DatasetSchema:
    """Column and semantic contract for a portable finite dataset.

    The zero-configuration convention is ``action_0``, ``action_1``, ...,
    ``observation``, and optional ``context_0``, ``context_1``, ... columns.

    Parameters
    ----------
    action_prefix:
        Prefix used for zero-based action columns in CSV and TSV files.
    observation_column:
        Name of the scalar reward/observation column in CSV and TSV files.
    context_prefix:
        Prefix used for zero-based context columns in CSV and TSV files.
    noise_variance_column:
        Optional name of the per-row observation-noise variance column.
    environment_family:
        Explicit environment semantics. When omitted, context columns select
        ``"contextual"`` and their absence selects ``"stochastic"``.
    lower_bounds, upper_bounds:
        Optional action-space bounds. Both must be supplied together; otherwise
        BayOPy infers bounds from the action columns.
    """

    action_prefix: str = "action_"
    observation_column: str = "observation"
    context_prefix: str = "context_"
    noise_variance_column: str | None = "observation_noise_variance"
    environment_family: DatasetFamily | None = None
    lower_bounds: tuple[float, ...] | None = None
    upper_bounds: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """Validated finite table plus inferred bounds and environment family.

    Attributes
    ----------
    dataset:
        Tensor-backed data accepted by BayOPy environments.
    bounds:
        Validated or inferred bounds of the action space.
    environment_family:
        Semantic family selected for the dataset.
    path:
        Absolute path from which the dataset was loaded.
    """

    dataset: EnvironmentDataset
    bounds: Bounds
    environment_family: DatasetFamily
    path: Path


def _numbered_columns(fieldnames: list[str], prefix: str) -> list[str]:
    """Return contiguous, numerically sorted columns having ``prefix``."""

    selected: list[tuple[int, str]] = []
    for name in fieldnames:
        if not name.startswith(prefix):
            continue
        suffix = name.removeprefix(prefix)
        if not suffix.isdigit():
            raise ValueError(f"Column {name!r} must end in a non-negative integer.")
        selected.append((int(suffix), name))
    selected.sort()
    indices = [index for index, _ in selected]
    if indices and indices != list(range(len(indices))):
        raise ValueError(f"Columns beginning with {prefix!r} must be contiguous from 0.")
    return [name for _, name in selected]


def _bounds(actions: torch.Tensor, schema: DatasetSchema) -> Bounds:
    """Validate explicit bounds or infer nondegenerate bounds from actions."""

    dimension = int(actions.shape[1])
    if (schema.lower_bounds is None) != (schema.upper_bounds is None):
        raise ValueError("lower_bounds and upper_bounds must be supplied together.")
    if schema.lower_bounds is not None and schema.upper_bounds is not None:
        if len(schema.lower_bounds) != dimension or len(schema.upper_bounds) != dimension:
            raise ValueError(f"Dataset bounds must contain exactly {dimension} values.")
        lower = torch.tensor(schema.lower_bounds, dtype=torch.float64)
        upper = torch.tensor(schema.upper_bounds, dtype=torch.float64)
    else:
        lower = actions.min(dim=0).values
        upper = actions.max(dim=0).values
        constant = upper <= lower
        lower = torch.where(constant, lower - 0.5, lower)
        upper = torch.where(constant, upper + 0.5, upper)
    if bool((upper <= lower).any()):
        raise ValueError("Every upper dataset bound must be greater than its lower bound.")
    return Bounds(lower, upper)


def _family(schema: DatasetSchema, contexts: torch.Tensor | None) -> DatasetFamily:
    """Resolve environment semantics and validate context compatibility."""

    family = schema.environment_family
    if family is None:
        return "contextual" if contexts is not None else "stochastic"
    if family in {"contextual", "distributional"} and contexts is None:
        raise ValueError(f"A {family} dataset requires context_0, context_1, ... columns.")
    if family in {"stochastic", "adversarial"} and contexts is not None:
        raise ValueError(f"A {family} dataset must not contain context columns.")
    return family


def _load_delimited(path: Path, schema: DatasetSchema) -> LoadedDataset:
    """Load a CSV or TSV table according to the portable column contract."""

    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError(f"Dataset {path} contains no data rows.")
    action_columns = _numbered_columns(fieldnames, schema.action_prefix)
    context_columns = _numbered_columns(fieldnames, schema.context_prefix)
    if not action_columns:
        raise ValueError(
            f"Dataset must contain {schema.action_prefix}0, {schema.action_prefix}1, ... columns."
        )
    if schema.observation_column not in fieldnames:
        raise ValueError(f"Dataset is missing {schema.observation_column!r}.")
    try:
        actions = torch.tensor(
            [[float(row[name]) for name in action_columns] for row in rows],
            dtype=torch.float64,
        )
        observations = torch.tensor(
            [float(row[schema.observation_column]) for row in rows], dtype=torch.float64
        )
        contexts = (
            torch.tensor(
                [[float(row[name]) for name in context_columns] for row in rows],
                dtype=torch.float64,
            )
            if context_columns
            else None
        )
        variance = None
        if schema.noise_variance_column and schema.noise_variance_column in fieldnames:
            variance = torch.tensor(
                [float(row[schema.noise_variance_column]) for row in rows],
                dtype=torch.float64,
            )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Dataset {path} contains a nonnumeric required value.") from error
    tensors = [actions, observations, *([] if contexts is None else [contexts])]
    if variance is not None:
        tensors.append(variance)
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError(f"Dataset {path} contains NaN or infinite required values.")
    if variance is not None and bool((variance < 0).any()):
        raise ValueError("observation_noise_variance cannot be negative.")
    family = _family(schema, contexts)
    dataset = EnvironmentDataset(
        actions=actions,
        observations=observations,
        contexts=contexts,
        observation_noise_variance=variance,
        metadata={"source_path": str(path)},
    )
    return LoadedDataset(
        dataset=dataset, bounds=_bounds(actions, schema), environment_family=family, path=path
    )


def _load_npz(path: Path, schema: DatasetSchema) -> LoadedDataset:
    """Load a compressed NumPy dataset according to the array contract."""

    with np.load(path, allow_pickle=False) as values:
        if "actions" not in values or "observations" not in values:
            raise ValueError("NPZ datasets require 'actions' and 'observations' arrays.")
        actions = torch.as_tensor(values["actions"], dtype=torch.float64)
        observations = torch.as_tensor(values["observations"], dtype=torch.float64)
        contexts = (
            torch.as_tensor(values["contexts"], dtype=torch.float64)
            if "contexts" in values
            else None
        )
        variance = (
            torch.as_tensor(values["observation_noise_variance"], dtype=torch.float64)
            if "observation_noise_variance" in values
            else None
        )
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] < 1:
        raise ValueError("NPZ actions must have nonempty shape (N,D).")
    if observations.reshape(-1).shape[0] != actions.shape[0]:
        raise ValueError("NPZ actions must be (N,D) and observations must contain N values.")
    if contexts is not None and (contexts.ndim != 2 or contexts.shape[0] != actions.shape[0]):
        raise ValueError("NPZ contexts must have shape (N,D_context).")
    if variance is not None and variance.reshape(-1).shape[0] != actions.shape[0]:
        raise ValueError("NPZ observation_noise_variance must contain N values.")
    if variance is not None:
        variance = variance.reshape(-1)
    tensors = [actions, observations, *([] if contexts is None else [contexts])]
    if variance is not None:
        tensors.append(variance)
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError(f"Dataset {path} contains NaN or infinite required values.")
    if variance is not None and bool((variance < 0).any()):
        raise ValueError("observation_noise_variance cannot be negative.")
    family = _family(schema, contexts)
    dataset = EnvironmentDataset(
        actions=actions,
        observations=observations.reshape(-1),
        contexts=contexts,
        observation_noise_variance=variance,
        metadata={"source_path": str(path)},
    )
    return LoadedDataset(
        dataset=dataset, bounds=_bounds(actions, schema), environment_family=family, path=path
    )


def load_dataset(
    path: str | Path,
    *,
    schema: DatasetSchema | None = None,
) -> LoadedDataset:
    """Load and validate a user-owned CSV, TSV, or NPZ finite dataset.

    Parameters
    ----------
    path:
        Path to the dataset. CSV and TSV use the column convention documented
        by :class:`DatasetSchema`; NPZ uses ``actions``, ``observations``, and
        optional ``contexts`` and ``observation_noise_variance`` arrays.
    schema:
        Optional names, semantics, and bounds. The conventional schema is used
        when this argument is omitted.

    Returns
    -------
    LoadedDataset
        The validated environment data, bounds, family, and resolved path.

    Examples
    --------
    Load a user-created file whose columns are ``action_0``, ``action_1``, and
    ``observation``:

    >>> import bayopy
    >>> loaded = bayopy.load_dataset("./my_measurements.csv")
    >>> environment = bayopy.StochasticEnvironment(
    ...     bounds=loaded.bounds, dataset=loaded.dataset, seed=42
    ... )
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {source}")
    selected_schema = DatasetSchema() if schema is None else schema
    suffix = source.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return _load_delimited(source, selected_schema)
    if suffix == ".npz":
        return _load_npz(source, selected_schema)
    raise ValueError("Dataset path must end in .csv, .tsv, or .npz.")


__all__ = ["DatasetFamily", "DatasetSchema", "LoadedDataset", "load_dataset"]
