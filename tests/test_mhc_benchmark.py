"""Network-free MHC source, encoding, availability and protocol regressions."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from basat.environments import ContextualEnvironment
from basat.mhc_benchmark import (
    MHC_POCKET_POSITIONS,
    MHC_PSEUDOSEQUENCES,
    VB_DESCRIPTORS,
    MHCBindingBenchmark,
)

PINNED_SHA256 = "c4f9bcb02c630e5f774d5d2e3612cc35aa24a7e0e14ef3bd743b639fa7b2c7a3"
EXPECTED_COUNTS = (254, 197, 104, 3089, 1443, 1447, 833)


def make_archive(rows, *, member="predictions_smm.txt"):
    table = io.StringIO(newline="")
    writer = csv.writer(table, delimiter="\t", lineterminator="\n")
    writer.writerow(["species", "allele", "length", "cv", "sequence", "inequality", "ic50", "smm"])
    writer.writerows(rows)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        stream.writestr(member, table.getvalue())
    return archive.getvalue()


@pytest.fixture(scope="module")
def synthetic_rows():
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    rows = [["mouse", "H-2 Kb", "9", "0", "AAAAAAAAA", "=", "20", "999999"]]
    # Deliberately not context order: archive order must win over task sorting.
    for context in (3, 5, 4, 2, 1, 0, 6):
        for index in range(EXPECTED_COUNTS[context]):
            peptide = "".join(alphabet[(index // 20 ** (position % 3) + position) % 20]
                              for position in range(9))
            rows.append(["human", MHCBindingBenchmark.SOURCE_ALLELES[context], "9", str(index % 5),
                         peptide, ("=", ">", "<")[index % 3], str(10.0 + index), "999999"])
    return rows


@pytest.fixture
def benchmark(synthetic_rows, monkeypatch, tmp_path):
    content = make_archive(synthetic_rows)
    path = tmp_path / "source.zip"
    path.write_bytes(content)
    monkeypatch.setattr(MHCBindingBenchmark, "SOURCE_SHA256", hashlib.sha256(content).hexdigest())

    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not access the network.")

    monkeypatch.setattr("basat.mhc_benchmark.urllib.request.urlopen", forbidden)
    return MHCBindingBenchmark(local_asset=path)


def test_pinned_definition_and_offline_full_table(benchmark):
    assert EXPECTED_COUNTS == MHCBindingBenchmark.EXPECTED_COUNTS
    assert benchmark.counts == EXPECTED_COUNTS
    assert benchmark.actions.shape == (7367, 45)
    assert benchmark.contexts.shape == (7367, 1)
    assert benchmark.dataset.observations.shape == (7367,)
    assert torch.equal(benchmark.source_row_indices, torch.arange(1, 7368))
    assert benchmark.contexts[0].item() == 3  # Source order, not context order.
    assert benchmark.maximize
    assert not benchmark.metadata["predictor_column_used"]
    assert not hasattr(benchmark, "observations")
    assert not hasattr(benchmark, "rewards")
    assert benchmark.metadata["pair_availability"].startswith("only measured")
    assert [int((benchmark.contexts == i).sum()) for i in range(7)] == list(EXPECTED_COUNTS)


def test_published_descriptor_table_and_position_major_encoding(benchmark):
    assert set(VB_DESCRIPTORS) == set("ACDEFGHIKLMNPQRSTVWY")
    assert VB_DESCRIPTORS["A"] == (0.008, 0.134, -0.475, -0.039, 0.181)
    assert VB_DESCRIPTORS["Q"] == (0.149, -0.184, 0.030, 0.035, -0.112)
    assert VB_DESCRIPTORS["W"] == (-0.296, -0.186, 0.389, 0.083, 0.297)
    # Freeze all 100 transcribed values, not only the three example rows.
    serialized = json.dumps(VB_DESCRIPTORS, sort_keys=True, separators=(",", ":"))
    expected_hash = "f48f3566bd17dde48e959245da7cf8a7d9074b582bebaaf3af1201bfbae8d5d8"
    assert hashlib.sha256(serialized.encode()).hexdigest() == expected_hash
    assert benchmark.metadata["features"]["table_sha256"] == expected_hash
    expected = torch.tensor([x for aa in benchmark.peptides[0] for x in VB_DESCRIPTORS[aa]],
                            dtype=torch.float64)
    assert torch.equal(benchmark.actions[0], expected)
    assert torch.all(benchmark.actions >= benchmark.bounds.lower)
    assert torch.all(benchmark.actions <= benchmark.bounds.upper)
    assert benchmark.bounds.dimension == 45


def test_censoring_and_numeric_ic50_reward_are_retained(benchmark):
    assert benchmark.qualifiers[:3] == ("=", ">", "<")
    expected = np.log10(500) - np.log10([10, 11, 12])
    np.testing.assert_allclose(benchmark.dataset.observations[:3].numpy(), expected, rtol=0, atol=0)
    assert "point surrogate" in benchmark.metadata["censoring"]["policy"]
    assert sum(benchmark.metadata["censoring"]["qualifier_counts"].values()) == 7367
    assert benchmark.dataset.observation_noise_variance is None
    noisy_model = benchmark.make_dataset(noise_variance=0.125)
    assert noisy_model.observation_noise_variance == 0.125
    assert torch.equal(noisy_model.observations, benchmark.dataset.observations)
    assert not noisy_model.metadata["observation_noise"]["synthetic_noise_added"]


@pytest.mark.parametrize("variance", [-1, float("nan"), float("inf"), [0.1, 0.2]])
def test_invalid_conditioning_noise_rejected(benchmark, variance):
    with pytest.raises(ValueError, match="finite nonnegative scalar"):
        benchmark.make_dataset(noise_variance=variance)


def test_pocket_distances_gram_and_schedule(benchmark):
    expected = torch.tensor([
        [0, 1, 2, 11, 12, 10, 13], [1, 0, 1, 12, 13, 11, 14],
        [2, 1, 0, 11, 13, 11, 13], [11, 12, 11, 0, 2, 2, 6],
        [12, 13, 13, 2, 0, 2, 8], [10, 11, 11, 2, 2, 0, 8],
        [13, 14, 13, 6, 8, 8, 0],
    ])
    assert len(MHC_POCKET_POSITIONS) == 34
    assert all(len(sequence) == 34 for sequence in MHC_PSEUDOSEQUENCES)
    assert torch.equal(benchmark.context_distances, expected)
    assert torch.equal(benchmark.context_gram, 1 - expected.double() / 14)
    assert torch.linalg.eigvalsh(benchmark.context_gram).min() > 0.038
    assert torch.equal(benchmark.context_support[:, 0], torch.arange(7, dtype=torch.float64))
    assert torch.equal(benchmark.context_schedule[:, 0], torch.arange(7).repeat_interleave(50))
    provider = benchmark.context_provider
    assert provider.queries_per_context == 50
    assert provider(round_index=0) == provider(round_index=49) == [0]
    assert provider(round_index=50) == [1]
    assert provider(round_index=349) == [6]
    for value in [-1, 350, 1.5, True]:
        with pytest.raises(ValueError, match="round_index"):
            provider(round_index=value)


def test_calibration_sample_exact_mt19937_1000_and_copy_isolation(benchmark):
    sample = benchmark.calibration_sample()
    expected = np.random.RandomState(2021).choice(7367, size=1000, replace=False)
    assert sample.seed == 2021
    assert sample.actions.shape == (1000, 45)
    assert sample.contexts.shape == (1000, 1)
    assert sample.observations.shape == (1000,)
    np.testing.assert_array_equal(sample.row_indices.numpy(), expected)
    assert sample.row_indices[:10].tolist() == [5717, 4664, 789, 6728, 7280, 7103, 7096, 2471, 2974, 5153]
    assert len(set(sample.row_indices.tolist())) == 1000
    assert torch.equal(sample.source_row_indices, benchmark.source_row_indices[sample.row_indices])
    assert torch.equal(sample.observations, benchmark.dataset.observations[sample.row_indices])
    sample.actions.zero_()
    sample.observations.zero_()
    assert not torch.equal(sample.actions, benchmark.calibration_sample().actions)
    assert not torch.equal(sample.observations, benchmark.calibration_sample().observations)
    assert benchmark.dataset.actions.shape[0] == 7367  # Calibration is not an online blacklist.


@pytest.mark.parametrize("kwargs", [{"seed": -1}, {"seed": True}, {"size": 0}, {"size": 7368}])
def test_invalid_calibration_configuration(benchmark, kwargs):
    with pytest.raises(ValueError):
        benchmark.calibration_sample(**kwargs)


def test_properties_and_metadata_do_not_expose_mutable_storage(benchmark):
    for attribute in ["actions", "contexts", "source_row_indices", "context_gram", "context_distances"]:
        original = getattr(benchmark, attribute)
        getattr(benchmark, attribute).zero_()
        assert torch.equal(getattr(benchmark, attribute), original)
    dataset = benchmark.dataset
    dataset.actions.zero_()
    dataset.observations.zero_()
    assert not torch.equal(dataset.actions, benchmark.dataset.actions)
    assert not torch.equal(dataset.observations, benchmark.dataset.observations)
    benchmark.metadata["selection"]["alleles"].clear()
    assert len(benchmark.metadata["selection"]["alleles"]) == 7


def test_only_measured_pairs_are_available_and_query_exclusion_is_pair_specific(benchmark):
    full = benchmark.make_dataset(noise_variance=0.125)
    selected = torch.cat([torch.nonzero(full.contexts[:, 0] == i).flatten()[:2] for i in range(7)]
                         + [torch.tensor([300])]).sort().values
    small = replace(full, actions=full.actions[selected], contexts=full.contexts[selected],
                    observations=full.observations[selected])
    env = ContextualEnvironment(
        bounds=benchmark.bounds, dataset=small,
        context_provider=benchmark.context_provider, dtype="float64", seed=11,
        allow_repeated_actions=False, add_noise_to_data=False,
    )
    info = env.before_action()
    assert set(info) == {"context", "candidate_actions", "candidate_dataset_indices"}
    indices = torch.nonzero(small.contexts[:, 0] == 0).flatten()
    assert torch.equal(info["candidate_actions"], small.actions[indices])
    assert torch.equal(info["candidate_dataset_indices"], indices)
    assert len(info["candidate_actions"]) == 2
    action = info["candidate_actions"][0]
    feedback = env.step(action)
    assert feedback["observation_noise_variance"] == 0.125
    assert feedback["observation"] == small.observations[indices[0]]
    assert len(env.available_actions(context=[0.0]).actions) == 1
    # The identical peptide in another measured context remains available.
    other_actions = env.available_actions(context=[1.0]).actions
    assert len(other_actions) == 2
    assert torch.any(torch.all(other_actions == action, dim=1))
    # Allele 3 has additional peptides; they are not imputed into allele 0.
    absent = benchmark.actions[300]
    assert not torch.any(torch.all(env.available_actions(context=[0.0]).actions == absent, dim=1))
    assert len(env.available_actions(context=[3.0]).actions) == 3


def test_schedule_is_part_of_environment_source_fingerprint(benchmark):
    full = benchmark.dataset
    small = replace(full, actions=full.actions[:3], contexts=full.contexts[:3],
                    observations=full.observations[:3])

    def environment(provider):
        return ContextualEnvironment(bounds=benchmark.bounds, dataset=small,
                                     context_provider=provider, dtype="float64", seed=1)

    provider = benchmark.context_provider
    original = environment(provider)._source_fingerprint()
    assert original == environment(benchmark.context_provider)._source_fingerprint()
    changed_count = replace(provider, queries_per_context=49)
    changed_schedule = replace(provider, schedule=(1, *provider.schedule[1:]))
    assert original != environment(changed_count)._source_fingerprint()
    assert original != environment(changed_schedule)._source_fingerprint()


def test_checksum_mismatch_and_missing_explicit_asset_never_download(monkeypatch, tmp_path):
    assert MHCBindingBenchmark.SOURCE_SHA256 == PINNED_SHA256
    path = tmp_path / "bad.zip"
    path.write_bytes(b"not a zip")

    def forbidden(*args, **kwargs):
        raise AssertionError("An explicit local source must not download.")

    monkeypatch.setattr("basat.mhc_benchmark.urllib.request.urlopen", forbidden)
    with pytest.raises(ValueError, match="SHA-256 checksum"):
        MHCBindingBenchmark(local_asset=path)
    assert path.read_bytes() == b"not a zip"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        MHCBindingBenchmark(local_asset=tmp_path / "missing.zip")


def test_verified_download_is_cached_and_cached_corruption_is_not_overwritten(
    synthetic_rows, monkeypatch, tmp_path,
):
    content = make_archive(synthetic_rows)
    monkeypatch.setattr(MHCBindingBenchmark, "SOURCE_SHA256", hashlib.sha256(content).hexdigest())
    calls = []

    class Response(io.BytesIO):
        def geturl(self):
            return MHCBindingBenchmark.SOURCE_URL

    def download(url, *, timeout):
        calls.append((url, timeout))
        return Response(content)

    monkeypatch.setattr("basat.mhc_benchmark.urllib.request.urlopen", download)
    cache = tmp_path / "cache"
    first = MHCBindingBenchmark(cache_directory=cache)
    cached = cache / MHCBindingBenchmark.SOURCE_FILENAME
    assert cached.read_bytes() == content
    second = MHCBindingBenchmark(cache_directory=cache)
    assert torch.equal(first.actions, second.actions)
    assert calls == [(MHCBindingBenchmark.SOURCE_URL, 60)]
    cached.write_bytes(b"corruption")
    with pytest.raises(ValueError, match="SHA-256 checksum"):
        MHCBindingBenchmark(cache_directory=cache)
    assert cached.read_bytes() == b"corruption"
    assert len(calls) == 1


@pytest.mark.parametrize("mutation, message", [
    (lambda rows: rows[1].__setitem__(6, "0"), "finite and positive"),
    (lambda rows: rows[1].__setitem__(6, "nan"), "finite and positive"),
    (lambda rows: rows[1].__setitem__(5, "~"), "censoring qualifier"),
    (lambda rows: rows[1].__setitem__(4, "XXXXXXXXX"), "Invalid MHC 9-mer"),
    (lambda rows: rows.append(rows[1]), "Duplicate measured"),
    (lambda rows: rows.pop(), "measured-row counts"),
])
def test_structural_source_validation(synthetic_rows, monkeypatch, tmp_path, mutation, message):
    rows = [list(row) for row in synthetic_rows]
    mutation(rows)
    content = make_archive(rows)
    path = tmp_path / "malformed.zip"
    path.write_bytes(content)
    monkeypatch.setattr(MHCBindingBenchmark, "SOURCE_SHA256", hashlib.sha256(content).hexdigest())
    with pytest.raises(ValueError, match=message):
        MHCBindingBenchmark(local_asset=path)


def test_exports_are_portable_and_contain_provenance_and_reference_tables(benchmark, tmp_path):
    exported = benchmark.export_artifacts(tmp_path / "artifacts")
    assert set(exported) == {"provenance", "arrays", "descriptors", "pseudosequences", "source_rows"}
    metadata = json.loads(Path(exported["provenance"]).read_text())
    assert metadata["measured_pairs"] == 7367
    assert metadata["reconstruction_limits"]
    assert "/Users/" not in json.dumps(metadata)
    with np.load(exported["arrays"], allow_pickle=False) as arrays:
        assert arrays["actions"].shape == (7367, 45)
        np.testing.assert_array_equal(arrays["observations"], benchmark.dataset.observations.numpy())
    with Path(exported["descriptors"]).open() as stream:
        assert len(list(csv.DictReader(stream))) == 20
    with Path(exported["pseudosequences"]).open() as stream:
        pockets = list(csv.DictReader(stream))
        assert [row["pseudosequence"] for row in pockets] == list(MHC_PSEUDOSEQUENCES)
    with Path(exported["source_rows"]).open() as stream:
        rows = list(csv.DictReader(stream))
        assert len(rows) == 7367
        assert rows[0]["source_row_index"] == "1"
        assert rows[1]["inequality"] == ">"


def test_real_pinned_archive_when_cached():
    path = Path(__file__).resolve().parents[1] / "data/reproducibility/mhc_i/predictions_smm.zip"
    if not path.is_file():
        pytest.skip("Historical archive not cached; test never downloads.")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PINNED_SHA256
    benchmark = MHCBindingBenchmark(local_asset=path)
    assert benchmark.counts == EXPECTED_COUNTS
    assert benchmark.metadata["censoring"]["qualifier_counts"] == {"=": 5124, ">": 1984, "<": 259}
    assert benchmark.source_row_indices[:3].tolist() == [1434, 1435, 1436]
    assert benchmark.peptides[:3] == ("AAAATCALV", "AAAKTPVIV", "AAEQRRSTI")
    assert benchmark.qualifiers[:3] == ("=", "=", ">")
    np.testing.assert_allclose(benchmark.dataset.observations[:3].numpy(),
                               np.log10(500) - np.log10([41.0052, 34725.6, 78125.0]), rtol=0, atol=0)
    assert benchmark.calibration_sample().actions.shape == (1000, 45)
