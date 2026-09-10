"""Portable, validated loading of user-owned finite optimization datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Literal

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

    def __post_init__(self) -> None:
        """Reject ambiguous column conventions and unknown source families."""

        for name in ("action_prefix", "context_prefix", "observation_column"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if self.action_prefix.startswith(self.context_prefix) or self.context_prefix.startswith(
            self.action_prefix
        ):
            raise ValueError("action_prefix and context_prefix must not overlap.")
        if self.noise_variance_column is not None and (
            not isinstance(self.noise_variance_column, str) or not self.noise_variance_column
        ):
            raise ValueError("noise_variance_column must be a non-empty string or None.")
        if self.noise_variance_column == self.observation_column:
            raise ValueError("Observation and noise variance columns must be distinct.")
        for name in ("observation_column", "noise_variance_column"):
            value = getattr(self, name)
            if value is not None and value.startswith((self.action_prefix, self.context_prefix)):
                raise ValueError(f"{name} must not overlap action or context coordinate columns.")
        if self.environment_family not in {
            None,
            "stochastic",
            "contextual",
            "adversarial",
            "distributional",
        }:
            raise ValueError("Unknown environment_family in dataset schema.")


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
    context_support: torch.Tensor | None = None
    reference_distribution: torch.Tensor | None = None
    sampling_distribution: torch.Tensor | None = None
    margin: float | None = None
    attack: Any | None = None

    def preview(self, rows: int = 3) -> str:
        """Return the header and first ``rows`` CSV/TSV records for display."""

        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
            raise ValueError("rows must be a positive integer.")
        if self.path.suffix.lower() not in {".csv", ".tsv"}:
            raise ValueError("Text previews require a CSV or TSV dataset.")
        delimiter = "\t" if self.path.suffix.lower() == ".tsv" else ","
        output = StringIO()
        writer = csv.writer(output, delimiter=delimiter, lineterminator="\n")
        with self.path.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream, delimiter=delimiter)
            for index, record in enumerate(reader):
                if index > rows:
                    break
                writer.writerow(record)
        return output.getvalue().rstrip("\n")

    def make_environment(self, **settings: Any) -> Any:
        """Construct this table's native environment with optional settings.

        ``settings`` accepts the corresponding environment constructor's
        options, for example ``dtype="float64", seed=42``.  Distribution
        probabilities and finite attacks are taken from the validated CSV;
        explicit settings may replace these providers.
        """

        from .environments import (
            AdversarialEnvironment,
            ContextualEnvironment,
            DistributionalEnvironment,
            StochasticEnvironment,
        )

        constructors = {
            "stochastic": StochasticEnvironment,
            "contextual": ContextualEnvironment,
            "distributional": DistributionalEnvironment,
            "adversarial": AdversarialEnvironment,
        }
        options: dict[str, Any] = {"bounds": self.bounds, "dataset": self.dataset}
        if self.environment_family == "distributional":
            options.update(
                context_support=self.context_support,
                reference_distribution=self.reference_distribution,
                sampling_distribution=self.sampling_distribution,
                margin=self.margin,
            )
        elif self.environment_family == "adversarial":
            options["attack"] = self.attack
        options.update(settings)
        return constructors[self.environment_family](**options)


@dataclass(frozen=True, slots=True)
class _FiniteAttack:
    """An explicit action-to-evaluated-action map from a user-owned table."""

    actions: torch.Tensor
    evaluated_actions: torch.Tensor

    def __call__(self, *, action: torch.Tensor, **_: Any) -> torch.Tensor:
        candidates = self.actions.to(dtype=action.dtype, device=action.device)
        matches = torch.isclose(candidates, action, atol=1e-8, rtol=0.0).all(dim=1)
        if not bool(matches.any()):
            raise ValueError("The adversarial CSV has no mapping for the requested action.")
        index = int(torch.nonzero(matches, as_tuple=False)[0, 0])
        evaluated = self.evaluated_actions[index].to(dtype=action.dtype, device=action.device)
        calculation_device = "cpu" if action.device.type == "mps" else action.device
        # Preserve an exact finite destination through subtraction and later
        # addition, including when the environment itself uses float32.
        return evaluated.to(device=calculation_device, dtype=torch.float64) - action.to(
            device=calculation_device, dtype=torch.float64
        )


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
    bounds = Bounds(lower, upper)
    if not bool(bounds.contains(actions).all()):
        raise ValueError("Dataset actions lie outside the supplied bounds.")
    return bounds


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
        if len(set(fieldnames)) != len(fieldnames):
            raise ValueError("Dataset column names must be unique.")
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
    if contexts is not None and (
        contexts.ndim != 2 or contexts.shape[0] != actions.shape[0] or contexts.shape[1] == 0
    ):
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


def _load_environment_table(path: Path, family: DatasetFamily) -> LoadedDataset:
    """Read the reserved-column, reward-last native environment contract."""

    if path.suffix.lower() not in {".csv", ".tsv"}:
        raise ValueError("The explicit environment contract requires a CSV or TSV file.")
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream, delimiter=delimiter)
        columns = next(reader, [])
        raw_rows = list(reader)
    if len(columns) != len(set(columns)):
        raise ValueError("Dataset column names must be unique.")
    if not columns or columns[-1] != "reward":
        raise ValueError("The final CSV column must be named 'reward'.")
    if not raw_rows or any(len(row) != len(columns) for row in raw_rows):
        raise ValueError("The CSV must contain nonempty rows matching its column count.")
    action_columns = _numbered_columns(columns, "action_")
    context_columns = _numbered_columns(columns, "context_")
    evaluated_columns = _numbered_columns(columns, "evaluated_action_")
    if not action_columns:
        raise ValueError("The CSV requires action_0, action_1, ... columns.")
    needs_context = family in {"contextual", "distributional"}
    if bool(context_columns) != needs_context:
        requirement = "requires" if needs_context else "must not contain"
        raise ValueError(f"A {family} CSV {requirement} context_0, context_1, ... columns.")
    if family == "adversarial":
        if len(evaluated_columns) != len(action_columns):
            raise ValueError("Adversarial CSV requires one evaluated_action_i per action_i.")
    elif evaluated_columns:
        raise ValueError("Only an adversarial CSV may contain evaluated_action_i columns.")
    allowed = set(action_columns + context_columns + evaluated_columns)
    allowed.update({"reward", "observation_noise_variance"})
    if family == "distributional":
        allowed.update({"reference_probability", "sampling_probability", "margin"})
        if "reference_probability" not in columns:
            raise ValueError("A distributional CSV requires reference_probability.")
    unexpected = sorted(set(columns) - allowed)
    if unexpected:
        raise ValueError(f"Unexpected columns in the {family} CSV: {unexpected}.")
    try:
        matrix = torch.tensor(
            [[float(value) for value in row] for row in raw_rows], dtype=torch.float64
        )
    except (ValueError, TypeError) as error:
        raise ValueError("All reserved CSV columns must contain numeric values.") from error
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("CSV values must be finite; NaN and infinity are invalid.")

    def values(names: list[str]) -> torch.Tensor:
        return matrix[:, [columns.index(name) for name in names]]

    actions = values(action_columns)
    contexts = values(context_columns) if context_columns else None
    variance = (
        values(["observation_noise_variance"]).flatten()
        if "observation_noise_variance" in columns
        else None
    )
    if variance is not None and bool((variance < 0).any()):
        raise ValueError("observation_noise_variance cannot be negative.")
    support = reference = sampling = None
    if contexts is not None:
        support = torch.tensor(
            list(dict.fromkeys(tuple(row) for row in contexts.tolist())), dtype=torch.float64
        )
    margin = None
    attack = None
    if family == "distributional":
        assert contexts is not None
        context_keys = [tuple(row) for row in contexts.tolist()]
        unique_contexts = list(dict.fromkeys(context_keys))
        support = torch.tensor(unique_contexts, dtype=torch.float64)
        action_keys = [tuple(row) for row in actions.tolist()]
        pairs = set(zip(action_keys, context_keys, strict=True))
        if len(pairs) != len(set(action_keys)) * len(unique_contexts):
            raise ValueError("Distributional CSV must be a complete action-context grid.")

        def probabilities(column: str) -> torch.Tensor:
            per_row = values([column]).flatten()
            result = []
            for context in unique_contexts:
                indices = [i for i, key in enumerate(context_keys) if key == context]
                entries = per_row[indices]
                if not bool(torch.isclose(entries, entries[0], atol=1e-10, rtol=0).all()):
                    raise ValueError(f"{column} must be identical for a context across actions.")
                result.append(float(entries[0]))
            probability = torch.tensor(result, dtype=torch.float64)
            if bool((probability < 0).any()) or abs(float(probability.sum()) - 1.0) > 1e-8:
                raise ValueError(f"{column} must be nonnegative and sum to one over contexts.")
            return probability

        reference = probabilities("reference_probability")
        sampling = (
            probabilities("sampling_probability")
            if "sampling_probability" in columns
            else reference.clone()
        )
        margin = 0.0
        if "margin" in columns:
            margins = values(["margin"]).flatten()
            if bool((margins < 0).any()) or not bool((margins == margins[0]).all()):
                raise ValueError("margin must be a single nonnegative value repeated on every row.")
            margin = float(margins[0])
    elif family == "adversarial":
        evaluated = values(evaluated_columns)
        nominal_keys = [tuple(row) for row in actions.tolist()]
        evaluated_keys = [tuple(row) for row in evaluated.tolist()]
        if set(nominal_keys) != set(evaluated_keys):
            raise ValueError("Adversarial nominal and evaluated finite action sets must match.")
        mapping: dict[tuple[float, ...], tuple[float, ...]] = {}
        for nominal, actual in zip(nominal_keys, evaluated_keys, strict=True):
            if nominal in mapping and mapping[nominal] != actual:
                raise ValueError("An adversarial nominal action must have one evaluated mapping.")
            mapping[nominal] = actual
        attack = _FiniteAttack(actions=actions.clone(), evaluated_actions=evaluated.clone())
        # Native adversarial tables look up reward at the actually evaluated coordinate.
        actions = evaluated
    dataset = EnvironmentDataset(
        actions=actions,
        observations=values(["reward"]).flatten(),
        contexts=contexts,
        observation_noise_variance=variance,
        metadata={
            "source_path": str(path),
            "environment_family": family,
            "column_contract": "reward-last-v1",
        },
    )
    return LoadedDataset(
        dataset=dataset,
        bounds=_bounds(actions, DatasetSchema()),
        environment_family=family,
        path=path,
        context_support=support,
        reference_distribution=reference,
        sampling_distribution=sampling,
        margin=margin,
        attack=attack,
    )


def load_dataset(
    path: str | Path,
    *,
    schema: DatasetSchema | None = None,
    environment: DatasetFamily | None = None,
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
    environment:
        Select the strict native CSV/TSV contract: numbered ``action_i``
        columns and a final ``reward`` column. Contextual and distributional
        files additionally require ``context_i``. Distributional files require
        a complete action-context grid and ``reference_probability`` repeated
        consistently per context; optional ``sampling_probability`` and
        ``margin`` configure the hidden sampling provider and public ambiguity
        radius. Adversarial files require ``evaluated_action_i`` and associate
        each reward with that evaluated coordinate. The nominal and evaluated
        finite action sets must agree. This option cannot be combined with
        a custom ``schema``. Without it the existing configurable schema and
        NPZ array contract remain available.

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
    if environment is not None:
        if schema is not None:
            raise ValueError("Use either environment's reserved columns or a custom schema.")
        if environment not in {"stochastic", "contextual", "distributional", "adversarial"}:
            raise ValueError("Unknown dataset environment; expected one of the four families.")
        return _load_environment_table(source, environment)
    selected_schema = DatasetSchema() if schema is None else schema
    if not isinstance(selected_schema, DatasetSchema):
        raise TypeError("schema must be a DatasetSchema instance or None.")
    suffix = source.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return _load_delimited(source, selected_schema)
    if suffix == ".npz":
        return _load_npz(source, selected_schema)
    raise ValueError("Dataset path must end in .csv, .tsv, or .npz.")


__all__ = ["DatasetFamily", "DatasetSchema", "LoadedDataset", "load_dataset"]
