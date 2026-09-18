"""Explicit long-form sensor-table processing; datasets stay in user projects."""

from __future__ import annotations

import gzip
import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class SensorTextSchema:
    """Zero-based column positions in whitespace-delimited long-form data.

    ``date_column`` contains ISO YYYY-MM-DD, ``time_column`` contains HH:MM:SS,
    ``sensor_column`` an integer ID, and ``value_column`` the numeric reading.
    Location rows contain the corresponding ID and coordinate columns. IDs
    need not be consecutive or sorted. Gzip input is detected by '.gz'.
    """

    date_column: int = 0
    time_column: int = 1
    sensor_column: int = 3
    value_column: int = 4
    location_id_column: int = 0
    location_coordinate_columns: tuple[int, ...] = (1, 2)

    def __post_init__(self) -> None:
        columns = (
            self.date_column,
            self.time_column,
            self.sensor_column,
            self.value_column,
            self.location_id_column,
            *self.location_coordinate_columns,
        )
        if not self.location_coordinate_columns or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in columns
        ):
            raise ValueError("Schema columns must be nonnegative integer indices.")


@dataclass(frozen=True)
class SensorTable:
    """Processed values, action locations, missing-bin flags and provenance.

    No preprocessing statistics are fitted on the online rows. ``values`` is
    a (time, sensor) float64 tensor; ``missing`` records which original bins
    were imputed. Calibration and online values are exposed separately.
    """

    values: torch.Tensor
    locations: torch.Tensor
    sensor_ids: torch.Tensor
    missing: torch.Tensor
    calibration_rows: int
    interval_minutes: int
    metadata: dict[str, Any]

    @property
    def calibration_values(self) -> torch.Tensor:
        """Return the first declared calibration rows only."""
        return self.values[: self.calibration_rows].clone()

    @property
    def online_values(self) -> torch.Tensor:
        """Return held-out online rows, never used for parameter calibration."""
        return self.values[self.calibration_rows :].clone()

    def periodic_contexts(self, *, period_hours: float) -> torch.Tensor:
        """Return normalized periodic phases, one per processed time row."""
        period = period_hours * 60 / self.interval_minutes
        if period <= 0 or int(period) != period:
            raise ValueError("period_hours must span a positive integer number of bins.")
        return (torch.arange(len(self.values), dtype=torch.float64) % int(period)).reshape(
            -1, 1
        ) / period

    def periodic_embedding(self, *, period_hours: float) -> torch.Tensor:
        """Embed periodic phases on a circle whose circumference is period_hours."""
        phase = self.periodic_contexts(period_hours=period_hours).reshape(-1) * (2 * np.pi)
        return torch.stack((phase.cos(), phase.sin()), dim=1) * period_hours / (2 * np.pi)


def _download(path: str | Path, url: str | None, sha256: str | None) -> Path:
    target = Path(path).expanduser()
    if not target.is_file():
        if url is None:
            raise FileNotFoundError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            urllib.request.urlretrieve(url, temporary)
            if sha256 and hashlib.sha256(temporary.read_bytes()).hexdigest() != sha256:
                raise ValueError(f"Downloaded sensor source hash does not match: {target}")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256 and hashlib.sha256(target.read_bytes()).hexdigest() != sha256:
        raise ValueError(f"Sensor source hash does not match: {target}")
    return target


def load_sensor_table(
    path: str,
    *,
    locations_path: str,
    schema: SensorTextSchema,
    start_date: str,
    days: int,
    interval_minutes: int,
    calibration_hours: int,
    sensor_count: int,
    value_range: tuple[float, float],
    calibration_imputation: str = "linear",
    online_imputation: str = "forward_fill",
    url: str | None = None,
    locations_url: str | None = None,
    sha256: str | None = None,
    locations_sha256: str | None = None,
) -> SensorTable:
    """Read and preprocess an arbitrary dated long-form sensor dataset.

    Parameters
    ----------
    path, locations_path : str
        User-owned raw data and location table paths. Optional URLs download
        missing files; existing files are never replaced.
    schema : SensorTextSchema
        Explicit date, time, ID, value and location column definitions.
    start_date, days, interval_minutes, calibration_hours : date and integers
        Calendar window, bin width and initial calibration duration. Bins use
        the arithmetic mean of valid readings. Window starts at midnight.
    sensor_count : int
        Select this many sensors by count of nonempty calibration bins only;
        smaller sensor ID breaks ties. Return columns in sensor-ID order.
    value_range : (float, float)
        Inclusive admissible reading interval. Invalid/malformed rows are
        counted in provenance, not silently converted into valid values.
    calibration_imputation, online_imputation : str
        Currently 'linear' within the calibration interval and 'forward_fill'
        online. Online imputation never uses future observations.
    url, locations_url, sha256, locations_sha256 : str, optional
        Explicit source download URLs and optional integrity checks.

    Returns
    -------
    SensorTable
        Numeric data, coordinates, original missing flags and full processing
        metadata. The package contains no dataset files or prescribed dates.
    """
    if not isinstance(schema, SensorTextSchema):
        raise ValueError("schema must be SensorTextSchema.")
    if any(
        isinstance(n, bool) or not isinstance(n, int) or n < 1
        for n in (days, interval_minutes, calibration_hours, sensor_count)
    ):
        raise ValueError("Durations, bin width and sensor count must be positive integers.")
    if (
        1440 % interval_minutes
        or calibration_hours * 60 % interval_minutes
        or calibration_hours >= days * 24
    ):
        raise ValueError("Bins must divide a day and calibration must precede held-out rows.")
    if calibration_imputation != "linear" or online_imputation != "forward_fill":
        raise ValueError("Supported imputation is calibration='linear', online='forward_fill'.")
    if not np.isfinite(value_range).all() or value_range[0] > value_range[1]:
        raise ValueError("value_range must be finite and ordered.")
    source = _download(path, url, sha256)
    location_file = _download(locations_path, locations_url, locations_sha256)
    locations = np.loadtxt(location_file, ndmin=2)
    if not np.isfinite(locations).all():
        raise ValueError("Location IDs and coordinates must be finite.")
    raw_ids = locations[:, schema.location_id_column]
    ids = raw_ids.astype(int)
    if not np.array_equal(ids, raw_ids) or len(set(ids)) != len(ids):
        raise ValueError("Location sensor IDs must be unique integers.")
    if sensor_count > len(ids):
        raise ValueError("sensor_count exceeds the location table.")
    id_to_column = {int(value): i for i, value in enumerate(ids)}
    total = days * 1440 // interval_minutes
    calibration = calibration_hours * 60 // interval_minutes
    sums, counts = np.zeros((total, len(ids))), np.zeros((total, len(ids)), dtype=int)
    first = date.fromisoformat(start_date)
    end = first + timedelta(days=days)
    malformed = excluded = 0
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt") as stream:
        for line in stream:
            row = line.split()
            try:
                day_text = row[schema.date_column]
                if day_text < first.isoformat() or day_text >= end.isoformat():
                    continue
                sensor = int(row[schema.sensor_column])
                value = float(row[schema.value_column])
                hour, minute, second = row[schema.time_column].split(":")
                h, m, sec = int(hour), int(minute), float(second)
                if not (0 <= h < 24 and 0 <= m < 60 and 0 <= sec < 60):
                    raise ValueError("Invalid time.")
                day = (date.fromisoformat(day_text) - first).days
                slot = (day * 1440 + h * 60 + m) // interval_minutes
                if (
                    sensor in id_to_column
                    and np.isfinite(value)
                    and value_range[0] <= value <= value_range[1]
                ):
                    sums[slot, id_to_column[sensor]] += value
                    counts[slot, id_to_column[sensor]] += 1
                else:
                    excluded += 1
            except (ValueError, IndexError):
                malformed += 1
    coverage = (counts[:calibration] > 0).sum(axis=0)
    chosen = np.lexsort((ids, -coverage))[:sensor_count]
    chosen = chosen[np.argsort(ids[chosen])]
    values = np.divide(
        sums[:, chosen],
        counts[:, chosen],
        out=np.full((total, sensor_count), np.nan),
        where=counts[:, chosen] > 0,
    )
    missing = np.isnan(values)
    for column in range(sensor_count):
        observed = np.flatnonzero(np.isfinite(values[:calibration, column]))
        if not len(observed):
            raise ValueError("A selected sensor has no calibration observations.")
        values[:calibration, column] = np.interp(
            np.arange(calibration), observed, values[observed, column]
        )
        for row_index in range(calibration, total):
            if np.isnan(values[row_index, column]):
                values[row_index, column] = values[row_index - 1, column]
    coordinates = locations[chosen][:, schema.location_coordinate_columns]
    metadata = {
        "start_date": start_date,
        "end_date_exclusive": end.isoformat(),
        "interval_minutes": interval_minutes,
        "calibration_hours": calibration_hours,
        "sensor_ids": ids[chosen].tolist(),
        "value_range": value_range,
        "calibration_missing_fraction": float(missing[:calibration].mean()),
        "online_missing_fraction": float(missing[calibration:].mean()),
        "selection": "descending calibration-bin coverage, ascending ID tie-break",
        "bin_aggregation": "arithmetic mean",
        "calibration_imputation": calibration_imputation,
        "online_imputation": online_imputation,
        "malformed_rows": malformed,
        "excluded_rows": excluded,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "locations_sha256": hashlib.sha256(location_file.read_bytes()).hexdigest(),
    }
    return SensorTable(
        torch.from_numpy(values),
        torch.from_numpy(coordinates.copy()),
        torch.from_numpy(ids[chosen].copy()),
        torch.from_numpy(missing),
        calibration,
        interval_minutes,
        metadata,
    )
