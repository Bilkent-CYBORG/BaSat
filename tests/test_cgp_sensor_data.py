"""Paper-reconstruction preprocessing invariants using tiny Intel-format data."""

import gzip
import hashlib
import math

import pytest
import torch

import basat


def reading(day, hour, minute, sensor, value, second="00"):
    return f"2004-03-{day:02d} {hour:02d}:{minute:02d}:{second} 1 {sensor} {value} 50 100 3\n"


def sources(directory, lines, *, locations="7 1.5 2.5\n"):
    directory.mkdir(parents=True, exist_ok=True)
    data = directory / "data.txt.gz"
    coordinates = directory / "mote_locs.txt"
    data.write_bytes(gzip.compress("".join(lines).encode(), mtime=0))
    coordinates.write_text(locations)
    return data, coordinates


def load(paths, *, sensor_count=1, **overrides):
    data, locations = paths
    kwargs = dict(
        locations_path=str(locations), schema=basat.SensorTextSchema(),
        start_date="2004-03-01", days=2, interval_minutes=5,
        calibration_hours=24, sensor_count=sensor_count, value_range=(0.0, 50.0),
        calibration_imputation="linear", online_imputation="forward_fill",
        sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        locations_sha256=hashlib.sha256(locations.read_bytes()).hexdigest(),
    )
    kwargs.update(overrides)
    return basat.load_sensor_table(str(data), **kwargs)


def test_selection_uses_calibration_bin_coverage_and_id_ties_only(tmp_path):
    lines = [reading(1, 0, minute, sensor, 10.0 + sensor / 10)
             for sensor, minutes in ((40, (0, 5, 10)), (10, (0, 10)),
                                     (20, (0, 10)), (30, (0,)))
             for minute in minutes]
    # Repeated readings in the same bin must not improve a sensor's coverage.
    lines += [reading(1, 0, 0, 30, 15.0)] * 30
    # Excellent held-out coverage must not change selection at all.
    future = [reading(2, hour, minute, 30, 48.0)
              for hour in range(24) for minute in range(0, 60, 5)]
    locations = "40 4 40\n10 1 10\n30 3 30\n20 2 20\n"
    base = load(sources(tmp_path / "baseline", lines, locations=locations), sensor_count=2)
    augmented = load(sources(tmp_path / "online", lines + future, locations=locations), sensor_count=2)
    assert base.sensor_ids.tolist() == [10, 40]
    assert augmented.sensor_ids.tolist() == [10, 40]
    assert base.locations.tolist() == [[1.0, 10.0], [4.0, 40.0]]
    assert torch.equal(base.calibration_values, augmented.calibration_values)
    assert base.metadata["selection"] == "descending calibration-bin coverage, ascending ID tie-break"


def test_bins_are_chronological_means_of_only_valid_values(tmp_path):
    # Deliberately unsorted input, fractional seconds, and bin boundaries.
    lines = [reading(1, 0, 9, 7, 30.0, "59.999"),
             reading(1, 0, 4, 7, 8.0, "59.999"),
             reading(1, 0, 0, 7, 12.0),
             reading(1, 0, 10, 7, 40.0),
             reading(1, 0, 4, 7, "nan"),
             reading(1, 0, 4, 7, -1.0),
             reading(1, 0, 4, 7, 51.0),
             reading(1, 0, 15, 7, 0.0),
             reading(1, 0, 15, 7, 50.0, "01"),
             reading(1, 0, 0, 999, 21.0),
             reading(1, 0, 60, 7, 20.0),
             "2004-03-01 broken\n",
             "2004-02-29 23:59:59 1 7 50 0 0 0\n",
             reading(3, 0, 0, 7, 1.0)]
    table = load(sources(tmp_path, lines))
    assert table.values[:4, 0].tolist() == [10.0, 30.0, 40.0, 25.0]
    assert not bool(table.missing[:4].any())
    assert bool(table.missing[4:].all())
    assert table.metadata["excluded_rows"] == 4
    assert table.metadata["malformed_rows"] == 2
    assert table.metadata["bin_aggregation"] == "arithmetic mean"
    assert table.values.shape == (576, 1)
    assert table.calibration_rows == 288
    assert table.values.dtype == torch.float64
    assert table.locations.dtype == torch.float64
    assert table.sensor_ids.dtype == torch.int64
    assert table.missing.dtype == torch.bool
    assert table.metadata["end_date_exclusive"] == "2004-03-03"


def test_calibration_interpolation_is_linear_with_endpoint_carry(tmp_path):
    table = load(sources(tmp_path, [reading(1, 0, 5, 7, 10.0),
                                         reading(1, 0, 20, 7, 16.0)]))
    assert table.calibration_values[:7, 0].tolist() == [10.0, 10.0, 12.0, 14.0, 16.0, 16.0, 16.0]
    assert table.missing[:7, 0].tolist() == [True, False, True, True, False, True, True]
    assert bool((table.calibration_values[4:] == 16.0).all())
    assert bool((table.online_values == 16.0).all())
    assert table.metadata["calibration_missing_fraction"] == pytest.approx(286 / 288)
    assert table.metadata["online_missing_fraction"] == 1.0


def test_online_missing_data_never_uses_future_even_valid_outlier(tmp_path):
    calibration = [reading(1, 0, 0, 7, 10.0), reading(1, 23, 55, 7, 20.0)]
    past = calibration + [reading(2, 0, 15, 7, 25.0)]
    before = load(sources(tmp_path / "without_future", past))
    after = load(sources(tmp_path / "with_future", past + [reading(2, 0, 30, 7, 50.0)]))
    assert torch.equal(before.calibration_values, after.calibration_values)
    assert torch.equal(before.online_values[:6], after.online_values[:6])
    assert after.online_values[:8, 0].tolist() == [20., 20., 20., 25., 25., 25., 50., 50.]
    assert after.missing[288:296, 0].tolist() == [True, True, True, False, True, True, False, True]
    # Accessors give caller-owned copies; fitting must not alter the table.
    copied = after.calibration_values
    copied.zero_()
    assert after.calibration_values[0, 0] == 10.0
    copied = after.online_values
    copied.zero_()
    assert after.online_values[0, 0] == 20.0


def test_daily_contexts_repeat_after_288_rows_but_physical_time_is_not_reordered(tmp_path):
    table = load(sources(tmp_path, [reading(1, 0, 0, 7, 20.0)]))
    phase = table.periodic_contexts(period_hours=24)
    assert phase.shape == (576, 1)
    assert phase.dtype == torch.float64
    assert torch.equal(phase[:288], phase[288:])
    assert phase[0].item() == 0.0
    assert phase[1].item() == pytest.approx(1 / 288)
    assert phase[287].item() == pytest.approx(287 / 288)
    assert phase[288].item() == 0.0
    embedding = table.periodic_embedding(period_hours=24)
    radius = 24 / (2 * math.pi)
    assert embedding.shape == (576, 2)
    assert torch.equal(embedding[:288], embedding[288:])
    assert embedding[0].tolist() == pytest.approx([radius, 0.0])
    assert embedding[72].tolist() == pytest.approx([0.0, radius], abs=1e-14)
    assert torch.linalg.vector_norm(embedding[0] - embedding[1]).item() == pytest.approx(
        2 * radius * math.sin(math.pi / 288), abs=1e-14,
    )
    assert torch.linalg.vector_norm(embedding[0] - embedding[287]).item() == pytest.approx(
        2 * radius * math.sin(math.pi / 288), abs=1e-14,
    )


def test_sources_are_verified_and_provenance_records_exact_bytes(tmp_path, monkeypatch):
    paths = sources(tmp_path, [reading(1, 0, 0, 7, 20.0)])

    def no_download(*args, **kwargs):
        raise AssertionError("Verified existing source assets must not download.")

    monkeypatch.setattr("basat.sensor_data.urllib.request.urlretrieve", no_download)
    table = load(paths)
    assert table.metadata["source_sha256"] == hashlib.sha256(paths[0].read_bytes()).hexdigest()
    assert table.metadata["locations_sha256"] == hashlib.sha256(paths[1].read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="hash does not match"):
        load(paths, sha256="0" * 64)
    with pytest.raises(ValueError, match="hash does not match"):
        load(paths, locations_sha256="0" * 64)


def test_zero_calibration_coverage_cannot_be_rescued_by_online_data(tmp_path):
    paths = sources(tmp_path, [reading(2, 0, 0, 7, 20.0)])
    with pytest.raises(ValueError, match="no calibration observations"):
        load(paths)


@pytest.mark.parametrize("overrides", [
    {"online_imputation": "linear"}, {"calibration_imputation": "forward_fill"},
    {"calibration_hours": 48}, {"interval_minutes": 7},
    {"days": 0}, {"sensor_count": 2}, {"value_range": (50., 0.)},
])
def test_preprocessing_configuration_is_explicit_and_validated(tmp_path, overrides):
    paths = sources(tmp_path, [reading(1, 0, 0, 7, 20.0)])
    with pytest.raises(ValueError):
        load(paths, **overrides)
