"""Recovered coauthor release: native measured-data loading, not curve tuning."""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from basat.mhc_benchmark import (
    AUTHOR_CONTEXT_GRAM,
    AUTHOR_VB_DESCRIPTORS,
    VB_DESCRIPTORS,
    MHCBindingBenchmark,
)


@pytest.fixture
def author_fixture(tmp_path, monkeypatch):
    """Small deterministic matched archives; never access an external service."""
    monkeypatch.setattr(MHCBindingBenchmark, "EXPECTED_COUNTS", (2,) * 7)
    monkeypatch.setattr(MHCBindingBenchmark, "AUTHOR_EXPECTED_COUNTS", (1,) * 7)
    table = io.StringIO(newline="")
    writer = csv.writer(table, delimiter="\t", lineterminator="\n")
    writer.writerow(["species", "allele", "length", "cv", "sequence", "inequality", "ic50", "smm"])
    author = []
    # IEDB source order deliberately differs from the author table order.
    for context in reversed(range(7)):
        for index in range(2):
            peptide = ("ACDEFGH"[context] + "Q" + "A" * 6 + "ST"[index])
            value = 10. ** (1 + context / 2 + index / 10)
            writer.writerow(["human", MHCBindingBenchmark.SOURCE_ALLELES[context], 9, index,
                             peptide, "<" if index == 0 else ">", value, -999999])
            if index == 1:
                feature = np.asarray([v for aa in peptide for v in AUTHOR_VB_DESCRIPTORS[aa]])
                feature /= np.linalg.norm(feature)
                author.append([np.log10(500) - np.log10(value), context, *feature])
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as archive:
        archive.writestr(MHCBindingBenchmark.SOURCE_MEMBER, table.getvalue())
    measured_bytes = zip_buffer.getvalue()
    measured_path = tmp_path / "measured.zip"
    measured_path.write_bytes(measured_bytes)
    monkeypatch.setattr(MHCBindingBenchmark, "SOURCE_SHA256", hashlib.sha256(measured_bytes).hexdigest())
    array = np.asarray(list(reversed(author)))
    csv_buffer = io.StringIO()
    np.savetxt(csv_buffer, array, delimiter=",")
    author_bytes = csv_buffer.getvalue().encode()
    author_path = tmp_path / "author.csv"
    author_path.write_bytes(author_bytes)
    monkeypatch.setattr(MHCBindingBenchmark, "AUTHOR_SOURCE_SHA256", hashlib.sha256(author_bytes).hexdigest())

    def forbidden(*args, **kwargs):
        raise AssertionError("Fixture tests must not download data.")

    monkeypatch.setattr("basat.mhc_benchmark.urllib.request.urlopen", forbidden)
    return author_path, measured_path, array


def load_fixture(fixture, **kwargs):
    author_path, measured_path, _ = fixture
    return MHCBindingBenchmark(
        source="author_nips11", local_asset=author_path, measured_asset=measured_path, **kwargs)


def test_author_encoding_preserves_csv_not_reconstructed_substitute(author_fixture):
    benchmark = load_fixture(author_fixture)
    expected = author_fixture[2]
    assert benchmark.source == "author_nips11"
    assert benchmark.counts == (1,) * 7
    np.testing.assert_array_equal(benchmark.actions.numpy(), expected[:, 2:])
    np.testing.assert_array_equal(benchmark.dataset.observations.numpy(), expected[:, 0])
    np.testing.assert_allclose(np.linalg.norm(benchmark.actions.numpy(), axis=1), 1, atol=1e-15)
    assert VB_DESCRIPTORS["Q"][2] == 0.03
    assert AUTHOR_VB_DESCRIPTORS["Q"][2] == -0.03
    assert all(benchmark.actions[:, 7] < 0)  # Q at the second peptide position.
    assert benchmark.peptides[0] == "AQAAAAAAT"
    assert benchmark.qualifiers == (">",) * 7
    assert benchmark.source_row_indices.tolist() == [13, 11, 9, 7, 5, 3, 1]
    assert benchmark.metadata["features"]["scaling"].startswith("each complete 45D")
    assert "-0.030" in benchmark.metadata["features"]["source_specific_difference"]
    assert torch.all(benchmark.actions >= benchmark.bounds.lower)
    assert torch.all(benchmark.actions <= benchmark.bounds.upper)


def test_author_matrix_is_exact_and_not_silently_repaired(author_fixture):
    benchmark = load_fixture(author_fixture)
    expected = torch.tensor(AUTHOR_CONTEXT_GRAM, dtype=torch.float64)
    assert torch.equal(benchmark.context_gram, expected)
    assert torch.equal(benchmark.author_context_gram, expected)
    assert float(torch.linalg.eigvalsh(expected).min()) == pytest.approx(-0.06180313615644345)
    assert not benchmark.metadata["contexts"]["gram_is_positive_semidefinite"]
    assert benchmark.metadata["contexts"]["gram_min_eigenvalue"] < -0.06
    assert torch.equal(benchmark.hamming_context_gram, 1 - benchmark.context_distances.double() / 14)
    assert torch.linalg.eigvalsh(benchmark.hamming_context_gram).min() > .038
    benchmark.context_gram.zero_()
    assert torch.equal(benchmark.context_gram, expected)


def test_author_base_observation_is_explicit_and_separate(author_fixture):
    benchmark = load_fixture(author_fixture)
    all_data = benchmark.make_dataset(noise_variance=.2)
    online = benchmark.make_dataset(noise_variance=.2, exclude_initial=True)
    base = benchmark.make_initial_data(noise_variance=.2)
    assert all_data.actions.shape == (7, 45)
    assert online.actions.shape == (6, 45)
    assert base.actions.shape == (1, 45)
    assert torch.equal(online.actions, all_data.actions[1:])
    assert torch.equal(base.actions, all_data.actions[:1])
    assert torch.equal(base.observations, all_data.observations[:1])
    assert base.contexts.item() == 0
    assert online.observation_noise_variance == base.observation_noise_variance == .2
    assert base.metadata == {}  # No unseen reward or measured-table provenance leak.
    assert online.metadata["online_subset"]["excluded_author_rows"] == [0]
    assert not benchmark.metadata["author_initialization"]["applied_automatically"]


def test_author_calibration_uses_selected_pool_without_implicit_warm_start(author_fixture):
    benchmark = load_fixture(author_fixture)
    sample = benchmark.calibration_sample(seed=2021, size=5)
    indices = np.random.RandomState(2021).choice(7, 5, replace=False)
    np.testing.assert_array_equal(sample.row_indices.numpy(), indices)
    assert torch.equal(sample.source_row_indices, benchmark.source_row_indices[indices])
    assert benchmark.dataset.actions.shape == (7, 45)
    assert benchmark.make_initial_data().actions.shape == (1, 45)


def test_author_exports_record_actual_descriptor_and_measured_identities(author_fixture, tmp_path):
    benchmark = load_fixture(author_fixture)
    exported = benchmark.export_artifacts(tmp_path / "artifacts")
    descriptor_rows = list(csv.DictReader(Path(exported["descriptors"]).open()))
    assert float(next(row for row in descriptor_rows if row["amino_acid"] == "Q")["E3"]) == -.03
    with np.load(exported["arrays"]) as arrays:
        np.testing.assert_array_equal(arrays["actions"], author_fixture[2][:, 2:])
        np.testing.assert_array_equal(arrays["context_gram"], AUTHOR_CONTEXT_GRAM)
    assert benchmark.metadata["measured_provenance"]["source_member"] == "predictions_smm.txt"


def test_bad_author_checksum_does_not_download_or_modify(author_fixture):
    author_path, _, _ = author_fixture
    author_path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA-256 checksum"):
        load_fixture(author_fixture)
    assert author_path.read_bytes() == b"corrupted"


@pytest.mark.parametrize("mutation,message", [
    (lambda x: x.__setitem__((0, 0), 99), "rewards do not match"),
    (lambda x: x.__setitem__((0, 2), 1), "unit Euclidean norm"),
    (lambda x: x.__setitem__((0, 9), -x[0, 9]), "recovered source encoding"),
    (lambda x: x.__setitem__((0, 1), .5), "Invalid context IDs"),
    (lambda x: x.__setitem__((0, 1), 1), "counts/order"),
])
def test_author_schema_encoding_and_rewards_validated(author_fixture, monkeypatch, mutation, message):
    path, _, values = author_fixture
    mutation(values)
    buf = io.StringIO()
    np.savetxt(buf, values, delimiter=",")
    content = buf.getvalue().encode()
    path.write_bytes(content)
    monkeypatch.setattr(MHCBindingBenchmark, "AUTHOR_SOURCE_SHA256", hashlib.sha256(content).hexdigest())
    with pytest.raises(ValueError, match=message):
        load_fixture(author_fixture)


def test_legacy_source_unchanged_and_no_author_initialization(author_fixture):
    _, measured_path, _ = author_fixture
    legacy = MHCBindingBenchmark(local_asset=measured_path)
    assert legacy.source == "historical_iedb"
    assert legacy.actions.shape == (14, 45)
    assert legacy.counts == (2,) * 7
    assert torch.linalg.eigvalsh(legacy.context_gram).min() > 0
    assert legacy.actions[0, 7] == .03
    with pytest.raises(ValueError, match="only supported"):
        legacy.make_dataset(exclude_initial=True)
    with pytest.raises(ValueError, match="Only author_nips11"):
        legacy.make_initial_data()


@pytest.mark.parametrize("kwargs", [{"source": "unknown"}, {"measured_asset": "not-used.zip"}])
def test_invalid_modes_rejected_before_io(kwargs):
    with pytest.raises(ValueError):
        MHCBindingBenchmark(**kwargs)


def test_full_pinned_author_asset_when_locally_available():
    root = Path(__file__).resolve().parents[1]
    author_path = root / "data/reproducibility/mhc_i_author_audit/mhc-nips11-v2.csv"
    measured_path = root / "data/reproducibility/mhc_i/predictions_smm.zip"
    if not author_path.is_file() or not measured_path.is_file():
        pytest.skip("Optional full checksum-pinned archives are not cached locally.")
    benchmark = MHCBindingBenchmark(
        source="author_nips11", local_asset=author_path, measured_asset=measured_path)
    assert benchmark.counts == (153, 118, 62, 1853, 865, 868, 499)
    assert benchmark.actions.shape == (4418, 45)
    assert benchmark.peptides[:2] == ("ESAERLKAY", "DTDISQLHH")
    assert benchmark.dataset.observations[0].item() == -1.6020599913279625
    assert benchmark.source_row_indices[:2].tolist() == [20023, 20129]
    assert benchmark.calibration_sample().row_indices[:10].tolist() == [
        525, 1177, 1520, 2693, 938, 4389, 889, 1686, 350, 4408]
    assert benchmark.make_dataset(exclude_initial=True).actions.shape == (4417, 45)
