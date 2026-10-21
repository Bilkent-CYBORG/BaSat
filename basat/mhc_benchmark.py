"""Auditable reconstruction of the seven-task MHC-I binding benchmark.

Measured IC50 values come from the historical IEDB evaluation table, not the
SMM predictor column. The default 7367-row protocol retains its explicitly
reconstructed encoding and pocket kernel. Opt-in ``author_nips11`` loads the
subsequently recovered, checksum-pinned 4418-row coauthor feature table. Its
published task matrix is indefinite and is never silently repaired. Neither
source requires a predictor executable or pickled model.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .environments.environment import EnvironmentData, EnvironmentDataset
from .models.utils import Bounds, resolve_device, resolve_dtype
from .workflow import ArtifactStore, save_csv

# Venkatarajan--Braun E1..E5, as published, without normalization or rounding
# beyond the three decimal places of the source table. Position-major encoding
# concatenates nine copies of a five-component amino-acid descriptor.
VB_DESCRIPTORS: dict[str, tuple[float, ...]] = {
    "A": (0.008, 0.134, -0.475, -0.039, 0.181),
    "C": (-0.132, 0.174, 0.070, 0.565, -0.374),
    "D": (0.303, -0.057, -0.014, 0.225, 0.156),
    "E": (0.221, -0.280, -0.315, 0.157, 0.303),
    "F": (-0.329, -0.023, 0.072, -0.002, 0.208),
    "G": (0.218, 0.562, -0.024, 0.018, 0.106),
    "H": (0.023, -0.177, 0.041, 0.280, -0.021),
    "I": (-0.353, 0.071, -0.088, -0.195, -0.107),
    "K": (0.243, -0.339, -0.044, -0.325, -0.027),
    "L": (-0.267, 0.018, -0.265, -0.274, 0.206),
    "M": (-0.239, -0.141, -0.155, 0.321, 0.077),
    "N": (0.255, 0.038, 0.117, 0.118, -0.055),
    "P": (0.173, 0.286, 0.407, -0.215, 0.384),
    "Q": (0.149, -0.184, 0.030, 0.035, -0.112),
    "R": (0.171, -0.361, 0.107, -0.258, -0.364),
    "S": (0.199, 0.238, -0.015, -0.068, -0.196),
    "T": (0.068, 0.147, -0.015, -0.132, -0.274),
    "V": (-0.274, 0.136, -0.187, -0.196, -0.299),
    "W": (-0.296, -0.186, 0.389, 0.083, 0.297),
    "Y": (-0.141, -0.057, 0.425, -0.096, -0.091),
}

# The recovered author CSV uses the same descriptors except Q's third entry,
# followed by normalization of the complete 45-dimensional peptide vector.
# Keep this source-specific difference separate from the published table above.
AUTHOR_VB_DESCRIPTORS = {**VB_DESCRIPTORS, "Q": (0.149, -0.184, -0.030, 0.035, -0.112)}
AUTHOR_CONTEXT_GRAM = (
    (1., .9, .85, .25, 0., .3, 0.),
    (.9, 1., .9, 0., 0., .25, 0.),
    (.85, .9, 1., .25, 0., .25, 0.),
    (.25, 0., .25, 1., .85, .85, .55),
    (0., 0., 0., .85, 1., .85, .45),
    (.3, .25, .25, .85, .85, 1., .45),
    (0., 0., 0., .55, .45, .45, 1.),
)

# These seven small reference strings are vendored here, not loaded from a
# developer's IEDB installation. Positions are one-based mature MHC numbering.
MHC_PSEUDOSEQUENCES = (
    "YSAMYEEKVAHTDENIAYLMFHYYTWAVQAYTWY",
    "YSAMYEEKVAHTDENIAYLMFHYYTWAVQAYTGY",
    "YSAMYEEKVAHTDENIAYLMFHYYTWAVLAYTGY",
    "YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY",
    "YFAMYGEKVAHTHVDTLYVRYHYYTWAEWAYTWY",
    "YFAMYGEKVAHTHVDTLYLRYHYYTWAVWAYTWY",
    "YYAMYRNNVAQTDVDTLYVRYHYYTWAVLAYTWY",
)
MHC_POCKET_POSITIONS = (
    7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81,
    84, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163, 167, 171,
)


@dataclass(frozen=True, slots=True)
class MHCCalibrationSample:
    """Explicitly selected calibration observations, in RNG draw order.

    ``row_indices`` index the filtered benchmark; ``source_row_indices`` index
    the original TSV body, excluding its header. Sampling does not consume
    environment evaluations or remove these pairs from online availability.
    These tensors are copies and do not expose mutable benchmark storage.
    """

    actions: torch.Tensor
    contexts: torch.Tensor
    observations: torch.Tensor
    row_indices: torch.Tensor
    source_row_indices: torch.Tensor
    seed: int


@dataclass(frozen=True)
class _MHCContextSchedule:
    """Stateless provider with explicit, environment-fingerprinted attributes."""

    schedule: tuple[int, ...]
    queries_per_context: int

    def __call__(self, *, round_index: int, generator: torch.Generator | None = None) -> list[int]:
        if (isinstance(round_index, bool) or not isinstance(round_index, int)
                or not 0 <= round_index < len(self.schedule)):
            raise ValueError(f"MHC round_index must be in [0, {len(self.schedule)}).")
        return [self.schedule[round_index]]


class MHCBindingBenchmark:
    """Historical measured, ragged peptide/allele table for contextual BO.

    The seven alleles have respectively 254, 197, 104, 3089, 1443, 1447 and
    833 measured 9-mers. Each dataset row is an available action/context pair;
    there is no rectangular completion, prediction substitution or imputation.
    Rows retain their order in the pinned source archive. Features concatenate
    the five published Venkatarajan--Braun descriptors at each peptide position.
    Context IDs follow :attr:`allele_names`; their covariance is ``1-D/14``
    for Hamming distances between the vendored 34-residue pseudosequences.

    Parameters
    ----------
    cache_directory : path, optional
        Directory for the checksum-pinned source ZIP, downloaded only if absent.
        An existing file with an incorrect checksum is rejected, not replaced.
    local_asset : path, optional
        Explicit source ZIP (historical mode) or author CSV (author mode) to
        verify instead of downloading. It must exist.
    source : {"historical_iedb", "author_nips11"}, optional
        Defaults to the original 7367-pair reconstruction. The opt-in author
        release contains 4418 ordered pairs and the actual normalized features.
        Its published context matrix is indefinite; it is exposed unchanged,
        not silently converted into a valid GP covariance.
    measured_asset : path, optional
        In author mode, optional local verified IEDB ZIP for matching author
        rows to measured peptides, censoring qualifiers and original row IDs.
    dtype, device : optional
        Floating tensor precision and device; defaults to float64 on CPU.

    Notes
    -----
    Reward is ``log10(500) - log10(ic50_nM)`` and is maximized. All inequality
    qualifiers are retained. For censored measurements, the reported numeric
    bound is used as a point surrogate, an explicitly documented approximation.
    The source's SMM predictions are never used as rewards. Give a dataset only
    to the environment; optimizers receive available features and observations
    through their usual ask/tell interface, not this benchmark object.

    Examples
    --------
    >>> benchmark = MHCBindingBenchmark()  # doctest: +SKIP
    >>> calibration = benchmark.calibration_sample()  # doctest: +SKIP
    >>> dataset = benchmark.make_dataset(noise_variance=0.1)  # doctest: +SKIP
    """

    SOURCE_URL = "https://tools.iedb.org/static/main/predictions_smm.zip"
    SOURCE_SHA256 = "c4f9bcb02c630e5f774d5d2e3612cc35aa24a7e0e14ef3bd743b639fa7b2c7a3"
    SOURCE_FILENAME = "predictions_smm.zip"
    SOURCE_MEMBER = "predictions_smm.txt"
    ALLELES = (
        "HLA-A*24:03", "HLA-A*24:02", "HLA-A*23:01", "HLA-A*02:01",
        "HLA-A*02:03", "HLA-A*02:02", "HLA-A*69:01",
    )
    SOURCE_ALLELES = (
        "HLA A*2403", "HLA A*2402", "HLA A*2301", "HLA A*0201",
        "HLA A*0203", "HLA A*0202", "HLA A*6901",
    )
    EXPECTED_COUNTS = (254, 197, 104, 3089, 1443, 1447, 833)
    AUTHOR_SOURCE_URL = (
        "https://mldata-static.ml.tu-berlin.de/repository/data/download/csv/"
        "mhc-nips11-v2/index.html"
    )
    AUTHOR_SOURCE_SHA256 = "f98882a4a823308d20168e584c0f2118c01ddef078954fa4fabf82df0978ec5e"
    AUTHOR_SOURCE_FILENAME = "mhc-nips11-v2.csv"
    AUTHOR_METADATA_URL = (
        "https://mldata-static.ml.tu-berlin.de/repository/data/viewslug/"
        "mhc-nips11-v2/index.html"
    )
    AUTHOR_EXPECTED_COUNTS = (153, 118, 62, 1853, 865, 868, 499)
    QUERIES_PER_CONTEXT = 50
    _MAX_ARCHIVE_BYTES = 2 * 1024 * 1024
    _MAX_TABLE_BYTES = 16 * 1024 * 1024

    def __init__(
        self, *, cache_directory: str | Path = "./data/reproducibility/mhc_i",
        local_asset: str | Path | None = None,
        source: str = "historical_iedb", measured_asset: str | Path | None = None,
        dtype: str | torch.dtype = "float64", device: str | torch.device = "cpu",
    ) -> None:
        if source not in {"historical_iedb", "author_nips11"}:
            raise ValueError("source must be 'historical_iedb' or 'author_nips11'.")
        if measured_asset is not None and source != "author_nips11":
            raise ValueError("measured_asset is only applicable to the author_nips11 source.")
        self.source = source
        self.dtype, self.device = resolve_dtype(dtype), resolve_device(device)
        if source == "author_nips11":
            self._load_author(cache_directory, local_asset, measured_asset)
            return
        source_path = (Path(local_asset).expanduser() if local_asset is not None
                       else Path(cache_directory).expanduser() / self.SOURCE_FILENAME)
        if local_asset is not None and not source_path.is_file():
            raise FileNotFoundError(f"MHC source archive does not exist: {source_path}")
        missing = not source_path.exists()
        if missing:
            with urllib.request.urlopen(self.SOURCE_URL, timeout=60) as response:
                if not response.geturl().startswith("https://"):
                    raise ValueError("MHC source download redirected outside HTTPS.")
                content = response.read(self._MAX_ARCHIVE_BYTES + 1)
        else:
            with source_path.open("rb") as stream:
                content = stream.read(self._MAX_ARCHIVE_BYTES + 1)
        if len(content) > self._MAX_ARCHIVE_BYTES:
            raise ValueError("MHC source archive exceeds the size limit.")
        if hashlib.sha256(content).hexdigest() != self.SOURCE_SHA256:
            raise ValueError(f"MHC source SHA-256 checksum does not match: {source_path}")
        records = self._parse_archive(content)
        # Only verified, structurally validated data may enter the persistent cache.
        if missing:
            ArtifactStore(str(source_path.parent))._write(source_path.name, content)
        self._source_path = source_path.resolve()
        self._archive_size = len(content)
        self._records = tuple(records)
        self._peptides = tuple(row[2] for row in records)
        self._qualifiers = tuple(row[3] for row in records)
        self._source_row_indices = torch.tensor(
            [row[0] for row in records], dtype=torch.int64, device=self.device)
        self._contexts = torch.tensor(
            [[row[1]] for row in records], dtype=self.dtype, device=self.device)
        self._ic50 = torch.tensor([row[4] for row in records], dtype=self.dtype, device=self.device)
        # Compute in float64 before any requested storage conversion.
        reward = np.log10(500.0) - np.log10(np.asarray([row[4] for row in records]))
        self._observations = torch.as_tensor(reward, dtype=self.dtype, device=self.device)
        encoded = [[value for aa in peptide for value in VB_DESCRIPTORS[aa]]
                   for peptide in self._peptides]
        self._actions = torch.tensor(encoded, dtype=self.dtype, device=self.device)
        descriptors = torch.tensor(list(VB_DESCRIPTORS.values()), dtype=self.dtype, device=self.device)
        self.bounds = Bounds(descriptors.amin(dim=0).repeat(9), descriptors.amax(dim=0).repeat(9))
        self._distances = torch.tensor(
            [[sum(a != b for a, b in zip(first, second, strict=True))
              for second in MHC_PSEUDOSEQUENCES] for first in MHC_PSEUDOSEQUENCES],
            dtype=torch.int64, device=self.device,
        )
        if int(self._distances.max()) != 14:
            raise ValueError("The vendored MHC pocket reference must have maximum distance 14.")
        self._context_gram = 1.0 - self._distances.to(dtype=self.dtype) / 14.0
        if torch.linalg.eigvalsh(self._context_gram.cpu().double()).min() < -1e-12:
            raise ValueError("The fixed seven-context covariance must be positive semidefinite.")

    def _load_author(
        self, cache_directory: str | Path, local_asset: str | Path | None,
        measured_asset: str | Path | None,
    ) -> None:
        """Load actual author features and verify their measured-data identities."""
        source_path = (Path(local_asset).expanduser() if local_asset is not None else
                       Path(cache_directory).expanduser() / self.AUTHOR_SOURCE_FILENAME)
        if local_asset is not None and not source_path.is_file():
            raise FileNotFoundError(f"MHC author source does not exist: {source_path}")
        missing = not source_path.exists()
        limit = 8 * 1024 * 1024
        if missing:
            with urllib.request.urlopen(self.AUTHOR_SOURCE_URL, timeout=60) as response:
                if not response.geturl().startswith("https://"):
                    raise ValueError("MHC author download redirected outside HTTPS.")
                content = response.read(limit + 1)
        else:
            with source_path.open("rb") as stream:
                content = stream.read(limit + 1)
        if len(content) > limit:
            raise ValueError("MHC author source exceeds the size limit.")
        if hashlib.sha256(content).hexdigest() != self.AUTHOR_SOURCE_SHA256:
            raise ValueError(f"MHC author source SHA-256 checksum does not match: {source_path}")
        rows = np.loadtxt(io.BytesIO(content), delimiter=",")
        if rows.shape != (sum(self.AUTHOR_EXPECTED_COUNTS), 47) or not np.isfinite(rows).all():
            raise ValueError("MHC author CSV must contain 4418 finite rows and 47 columns.")
        context_ids = rows[:, 1].astype(np.int64)
        if (not np.array_equal(context_ids, rows[:, 1]) or np.any(context_ids < 0)
                or np.any(context_ids >= len(self.ALLELES))):
            raise ValueError("Invalid context IDs in MHC author CSV.")
        counts = tuple(int(np.count_nonzero(context_ids == i)) for i in range(len(self.ALLELES)))
        if counts != self.AUTHOR_EXPECTED_COUNTS or np.any(np.diff(context_ids) < 0):
            raise ValueError("MHC author context counts/order do not match the released data.")
        features = rows[:, 2:]
        if not np.allclose(np.linalg.norm(features, axis=1), 1., atol=1e-12, rtol=0):
            raise ValueError("MHC author peptide features must have unit Euclidean norm.")

        # Decode only to establish provenance; optimization uses the exact CSV
        # floats, not a re-encoded substitute. All five-dimensional descriptor
        # directions are distinct, so peptide recovery is unambiguous.
        letters = np.asarray(list(AUTHOR_VB_DESCRIPTORS))
        descriptors = np.asarray(list(AUTHOR_VB_DESCRIPTORS.values()))
        directions = descriptors / np.linalg.norm(descriptors, axis=1)[:, None]
        blocks = features.reshape(-1, 9, 5)
        block_norms = np.linalg.norm(blocks, axis=2)
        if np.any(block_norms == 0):
            raise ValueError("MHC author CSV contains an invalid zero descriptor.")
        identities = (blocks / block_norms[:, :, None] @ directions.T).argmax(axis=2)
        peptides = tuple("".join(row) for row in letters[identities])
        reconstructed = descriptors[identities].reshape(-1, 45)
        reconstructed /= np.linalg.norm(reconstructed, axis=1)[:, None]
        if not np.allclose(reconstructed, features, atol=1e-12, rtol=0):
            raise ValueError("MHC author features do not match the recovered source encoding.")
        measured = MHCBindingBenchmark(
            cache_directory=cache_directory, local_asset=measured_asset,
            dtype="float64", device="cpu",
        )
        lookup = {(record[1], record[2]): i for i, record in enumerate(measured._records)}
        pairs = tuple(zip(context_ids.tolist(), peptides, strict=True))
        if len(set(pairs)) != len(pairs) or any(pair not in lookup for pair in pairs):
            raise ValueError("MHC author CSV contains duplicate or unmeasured peptide/allele pairs.")
        selected = np.asarray([lookup[pair] for pair in pairs])
        if not np.allclose(rows[:, 0], measured._observations.numpy()[selected], atol=1e-12, rtol=0):
            raise ValueError("MHC author rewards do not match measured IEDB IC50 values.")
        if missing:
            ArtifactStore(str(source_path.parent))._write(source_path.name, content)
        self._source_path = source_path.resolve()
        self._archive_size = len(content)
        self._records = tuple(measured._records[i] for i in selected)
        self._peptides = peptides
        self._qualifiers = tuple(record[3] for record in self._records)
        self._source_row_indices = torch.tensor(
            [record[0] for record in self._records], dtype=torch.int64, device=self.device)
        self._contexts = torch.tensor(rows[:, 1:2], dtype=self.dtype, device=self.device)
        self._ic50 = torch.tensor([record[4] for record in self._records], dtype=self.dtype, device=self.device)
        self._observations = torch.tensor(rows[:, 0], dtype=self.dtype, device=self.device)
        self._actions = torch.tensor(features, dtype=self.dtype, device=self.device)
        self.bounds = Bounds(self._actions.amin(dim=0), self._actions.amax(dim=0))
        self._distances = measured._distances.to(device=self.device)
        self._context_gram = self.author_context_gram

    def _parse_archive(self, content: bytes) -> list[tuple[int, int, str, str, float, str]]:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if archive.namelist() != [self.SOURCE_MEMBER]:
                raise ValueError("Unexpected files in the MHC source archive.")
            member = archive.getinfo(self.SOURCE_MEMBER)
            if member.file_size > self._MAX_TABLE_BYTES:
                raise ValueError("MHC source table exceeds the size limit.")
            # Read a single bounded member in memory: never extract archive paths.
            table = archive.read(member).decode("ascii")
        reader = csv.DictReader(io.StringIO(table), delimiter="\t")
        if reader.fieldnames != ["species", "allele", "length", "cv", "sequence",
                                 "inequality", "ic50", "smm"]:
            raise ValueError("Unexpected MHC source table columns.")
        context_ids = dict(zip(self.SOURCE_ALLELES, range(len(self.ALLELES)), strict=True))
        records: list[tuple[int, int, str, str, float, str]] = []
        seen: set[tuple[int, str]] = set()
        for source_index, row in enumerate(reader):
            if row["species"] != "human" or row["length"] != "9" or row["allele"] not in context_ids:
                continue
            context, peptide = context_ids[row["allele"]], row["sequence"]
            if len(peptide) != 9 or any(aa not in VB_DESCRIPTORS for aa in peptide):
                raise ValueError(f"Invalid MHC 9-mer at source row {source_index}.")
            qualifier = row["inequality"]
            if qualifier not in {"=", "<", ">"}:
                raise ValueError(f"Invalid MHC censoring qualifier at source row {source_index}.")
            ic50 = float(row["ic50"])
            if not np.isfinite(ic50) or ic50 <= 0:
                raise ValueError(f"IC50 must be finite and positive at source row {source_index}.")
            if (context, peptide) in seen:
                raise ValueError(f"Duplicate measured peptide/allele pair at source row {source_index}.")
            seen.add((context, peptide))
            records.append((source_index, context, peptide, qualifier, ic50, row["cv"]))
        counts = Counter(row[1] for row in records)
        if tuple(counts[i] for i in range(len(self.ALLELES))) != self.EXPECTED_COUNTS:
            raise ValueError("MHC measured-row counts do not match the historical seven-task table.")
        return records

    @property
    def maximize(self) -> bool:
        """True: larger transformed binding affinity is better."""
        return True

    @property
    def allele_names(self) -> tuple[str, ...]:
        """Seven canonical allele names, in experiment context order."""
        return self.ALLELES

    @property
    def counts(self) -> tuple[int, ...]:
        """Number of measured candidate pairs per context, in context order."""
        return self.AUTHOR_EXPECTED_COUNTS if self.source == "author_nips11" else self.EXPECTED_COUNTS

    @property
    def peptides(self) -> tuple[str, ...]:
        """Peptide strings aligned with the measured dataset rows."""
        return self._peptides

    @property
    def qualifiers(self) -> tuple[str, ...]:
        """Original IC50 inequality signs aligned with the dataset rows."""
        return self._qualifiers

    @property
    def source_row_indices(self) -> torch.Tensor:
        """Zero-based original TSV body indices (header excluded), as a copy."""
        return self._source_row_indices.clone()

    @property
    def actions(self) -> torch.Tensor:
        """Measured-row-aligned 45D features; normalized only in author mode."""
        return self._actions.clone()

    @property
    def contexts(self) -> torch.Tensor:
        """Measured-row-aligned context IDs, shape ``(N, 1)``, as a copy."""
        return self._contexts.clone()

    @property
    def context_support(self) -> torch.Tensor:
        """Seven scalar context IDs, shape ``(7, 1)``."""
        return torch.arange(len(self.ALLELES), dtype=self.dtype, device=self.device).reshape(-1, 1)

    @property
    def context_gram(self) -> torch.Tensor:
        """Source-selected context matrix, copied without numerical repair.

        Historical mode uses positive-definite ``1-Hamming/14``. Author mode
        returns the published matrix, whose minimum eigenvalue is approximately
        -0.061803: it is *not* by itself a valid covariance. Callers must not
        silently project or treat this as a validated kernel.
        """
        return self._context_gram.clone()

    @property
    def author_context_gram(self) -> torch.Tensor:
        """Exact, indefinite 7x7 matrix from the coauthor's dataset metadata."""
        return torch.tensor(AUTHOR_CONTEXT_GRAM, dtype=self.dtype, device=self.device)

    @property
    def hamming_context_gram(self) -> torch.Tensor:
        """Explicit positive-definite reconstructed pocket covariance ``1-D/14``.

        This is identical in both source modes. In author mode it is an
        alternative reconstruction policy, *not* a correction or projection of
        the recovered author's indefinite task matrix. Callers selecting this
        policy must record that distinction in their experiment configuration.
        """
        return 1.0 - self._distances.to(dtype=self.dtype) / 14.0

    @property
    def context_distances(self) -> torch.Tensor:
        """Integer Hamming distances between the seven pocket pseudosequences."""
        return self._distances.clone()

    @property
    def context_schedule(self) -> torch.Tensor:
        """350 sequential task IDs: 50 queries on each allele in paper order."""
        return self.context_support.repeat_interleave(self.QUERIES_PER_CONTEXT, dim=0)

    @property
    def context_provider(self) -> _MHCContextSchedule:
        """Stateless callable schedule for ``ContextualEnvironment``.

        Accepts keyword ``round_index`` (zero-based) and optional ``generator``;
        rejects rounds outside 0..349. Schedule values and queries per context
        are ordinary provider attributes included in environment fingerprints.
        """
        return _MHCContextSchedule(
            tuple(i for i in range(len(self.ALLELES)) for _ in range(self.QUERIES_PER_CONTEXT)),
            self.QUERIES_PER_CONTEXT,
        )

    @property
    def dataset(self) -> EnvironmentDataset:
        """Fresh measured dataset without a declared conditioning-noise variance."""
        return self.make_dataset()

    def make_dataset(
        self, *, noise_variance: float | torch.Tensor | None = None,
        exclude_initial: bool = False,
    ) -> EnvironmentDataset:
        """Build environment-only observations with optional model noise.

        ``noise_variance`` is a finite nonnegative scalar used for GP
        conditioning. It does not add noise to measurements or change rewards.
        Use ``add_noise_to_data=False`` in the consuming environment.

        In author mode, ``exclude_initial=True`` removes the released base-GP
        observation (row zero) from the online search pool. The separate
        :meth:`make_initial_data` returns that recorded observation. Historical
        mode has no author-identified base observation and rejects this option.
        """
        if not isinstance(exclude_initial, bool):
            raise ValueError("exclude_initial must be a boolean.")
        if exclude_initial and self.source != "author_nips11":
            raise ValueError("exclude_initial is only supported for author_nips11.")
        variance = None
        if noise_variance is not None:
            candidate = torch.as_tensor(noise_variance, dtype=self.dtype, device=self.device)
            if candidate.numel() != 1 or not torch.isfinite(candidate).all() or bool(candidate < 0):
                raise ValueError("noise_variance must be a finite nonnegative scalar.")
            variance = float(candidate.detach().cpu().reshape(()))
        metadata = self.metadata
        metadata["observation_noise"] = {
            "conditioning_variance": variance, "synthetic_noise_added": False,
        }
        if exclude_initial:
            metadata["online_subset"] = {"excluded_author_rows": [0], "measured_pairs": len(self._records) - 1}
        start = int(exclude_initial)
        return EnvironmentDataset(
            actions=self._actions[start:].clone(), contexts=self._contexts[start:].clone(),
            observations=self._observations[start:].clone(), observation_noise_variance=variance,
            metadata=metadata,
        )

    def make_initial_data(self, *, noise_variance: float | torch.Tensor | None = None) -> EnvironmentData:
        """Return the single released author base-GP observation, without queries.

        This is distinct from the later random hyperparameter calibration
        sample. The user must explicitly pass it as environment previous_data;
        this method never inserts any other measured or calibration rewards.
        """
        if self.source != "author_nips11":
            raise ValueError("Only author_nips11 identifies a base-GP initialization observation.")
        declared = self.make_dataset(noise_variance=noise_variance)
        return EnvironmentData(
            actions=self._actions[:1].clone(), contexts=self._contexts[:1].clone(),
            observations=self._observations[:1].clone(),
            observation_noise_variance=declared.observation_noise_variance,
        )

    def calibration_sample(self, *, seed: int = 2021, size: int = 1000) -> MHCCalibrationSample:
        """Sample distinct measured pairs using NumPy's legacy MT19937 stream.

        The default is exactly 1000 rows sampled without replacement with
        ``RandomState(2021).choice(N, 1000, replace=False)`` where ``N`` is 7367
        in historical mode or 4418 in author mode. Draw order is
        retained. This sample is for hyperparameter calibration only: it does
        not warm-start an optimizer, consume the online budget, or blacklist
        sampled rows from later evaluation.
        """
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("Calibration seed must be an integer in [0, 2**32).")
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= len(self._records):
            raise ValueError("Calibration size must be a positive integer no larger than the dataset.")
        indices = np.random.RandomState(seed).choice(len(self._records), size=size, replace=False)
        selected = torch.as_tensor(indices, dtype=torch.int64, device=self.device)
        return MHCCalibrationSample(
            actions=self._actions[selected].clone(), contexts=self._contexts[selected].clone(),
            observations=self._observations[selected].clone(), row_indices=selected.clone(),
            source_row_indices=self._source_row_indices[selected].clone(), seed=seed,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        """Portable provenance and reconstruction limits, without hidden rewards."""
        descriptor_json = json.dumps(VB_DESCRIPTORS, sort_keys=True, separators=(",", ":"))
        pocket_json = json.dumps(MHC_PSEUDOSEQUENCES, separators=(",", ":"))
        metadata = {
            "name": "MHC-I seven-task measured binding reconstruction", "schema_version": 1,
            "source_url": self.SOURCE_URL, "source_sha256": self.SOURCE_SHA256,
            "source_member": self.SOURCE_MEMBER, "source_archive_bytes": self._archive_size,
            "selection": {"species": "human", "peptide_length": 9,
                          "alleles": list(self.ALLELES), "source_alleles": list(self.SOURCE_ALLELES)},
            "counts": list(self.counts), "measured_pairs": len(self._records),
            "row_order": "original archive TSV order, filtered without sorting",
            "source_row_index": "zero-based TSV body index; physical line number = index + 2",
            "pair_availability": "only measured peptide/allele pairs; no rectangular completion",
            "reward": "log10(500) - log10(ic50_nM)", "maximize": True,
            "censoring": {"qualifier_counts": dict(sorted(Counter(self._qualifiers).items())),
                          "policy": "retain signs; use reported numeric IC50 bound as point surrogate"},
            "predictor_column_used": False,
            "features": {"dimension": 45, "descriptor": "Venkatarajan-Braun E1..E5",
                         "order": "position-major, positions 1..9, E1..E5 within each position",
                         "scaling": "none; published three-decimal descriptor values",
                         "table_sha256": hashlib.sha256(descriptor_json.encode()).hexdigest(),
                         "descriptor_doi": "10.1007/s00894-001-0058-5",
                         "transcription_source": "https://academic.oup.com/view-large/393796368"},
            "contexts": {"ids": list(range(7)), "schedule": "50 consecutive queries per allele",
                         "pocket_positions_one_based": list(MHC_POCKET_POSITIONS),
                         "pseudosequences": dict(zip(self.ALLELES, MHC_PSEUDOSEQUENCES, strict=True)),
                         "pseudosequence_table_sha256": hashlib.sha256(pocket_json.encode()).hexdigest(),
                         "pseudosequence_source": "NetMHCpan 2.8 data/MHC_pseudo.dat",
                         "pseudosequence_source_file_sha256":
                             "cc353c1187980edcd30bffa24b2be316063c6ede5979ce56c474e640c8c85d62",
                         "positions_source": "NetMHCpan 2.8 data/all_varcontacts.nlist",
                         "kernel": "1 - pocket_hamming_distance / 14",
                         "normalization": "maximum Hamming distance among these seven contexts"},
            "calibration_protocol": {
                "default_size": 1000, "default_seed": 2021,
                "rng": "numpy.random.RandomState / MT19937", "replacement": False,
                "order": "RNG draw order", "online_warm_start": False,
                "calibration_pairs_remain_available_online": True,
            },
            "references": {
                "contextual_gp_bandit": "https://www.ong-home.my/papers/krause11cgp-ucb.pdf",
                "companion_encoding": "https://link.springer.com/article/10.1186/1471-2105-11-S8-S7",
                "later_45d_description": "https://jmlr.org/papers/volume15/desautels14a/desautels14a.pdf",
                "historical_author_code":
                    "https://github.com/cwidmer/multitask/tree/b8ce676ecb778b17dc39d06878523299fd693bc3",
            },
            "reconstruction_limits": [
                "Original 2011 feature matrix, calibration row indices and full optimizer settings unrecovered.",
                "45D published descriptor encoding is a documented reconstruction, not certified original bytes.",
                "34-residue pocket source and 1-D/14 scaling are explicit reconstruction choices.",
                "Original handling of censored IC50 measurements is not fully documented.",
                "Seed 2021, archive-order ties and calibration/online overlap policy are explicit new choices.",
            ],
            "dtype": str(self.dtype), "device": str(self.device),
        }
        if self.source == "author_nips11":
            author_descriptor_json = json.dumps(
                AUTHOR_VB_DESCRIPTORS, sort_keys=True, separators=(",", ":"))
            metadata.update({
                "name": "MHC-I recovered author NIPS-2011 dataset",
                "schema_version": 2,
                "source": self.source,
                "source_url": self.AUTHOR_SOURCE_URL,
                "source_sha256": self.AUTHOR_SOURCE_SHA256,
                "source_member": None,
                "source_format": "CSV without header: label, context, 45 features",
                "author_metadata_url": self.AUTHOR_METADATA_URL,
                "author_link_provenance": "https://www.ong-home.my/publications.html",
                "row_order": "exact author CSV order; not the original full IEDB archive order",
                "author_row_index": "zero-based CSV body index; physical line number = index + 1",
                "measured_provenance": {
                    "source_url": self.SOURCE_URL,
                    "source_sha256": self.SOURCE_SHA256,
                    "source_member": self.SOURCE_MEMBER,
                    "mapping": "all author rows matched to measured peptide/allele pairs and IC50 rewards",
                },
            })
            metadata["selection"]["author_description"] = (
                "first data point from split 0, and all data from splits 2,3,4")
            metadata["features"].update({
                "descriptor": "exact recovered author 45D PCA feature values",
                "scaling": "each complete 45D peptide vector divided by its Euclidean norm",
                "table_sha256": hashlib.sha256(author_descriptor_json.encode()).hexdigest(),
                "source_specific_difference": (
                    "author Q third component is -0.030, versus +0.030 in the published descriptor table"),
                "validation": "author features reconstructed independently within 1e-12; stored CSV floats used",
            })
            metadata["contexts"].update({
                "kernel": "exact author metadata Ky matrix, indefinite without an explicit numerical policy",
                "normalization": "none applied to the author matrix",
                "gram": [list(row) for row in AUTHOR_CONTEXT_GRAM],
                "gram_min_eigenvalue": float(np.linalg.eigvalsh(np.asarray(AUTHOR_CONTEXT_GRAM)).min()),
                "gram_is_positive_semidefinite": False,
                "pocket_data_usage": "historical pseudosequences retained for provenance only; not used for author Ky",
            })
            metadata["author_initialization"] = {
                "row_index": 0,
                "description": "base GP trained on first example; GP-UCB run on all the rest",
                "applied_automatically": False,
            }
            metadata["calibration_protocol"]["pool"] = (
                "recovered 4418-row table; original calibration selection and pool remain unrecovered")
            metadata["reconstruction_limits"] = [
                "Author online feature/reward table, row order, task matrix and base observation recovered.",
                "Exact original hyperparameter-calibration rows, fitted parameters and confidence schedule unrecovered.",
                "Published context matrix is indefinite; no covariance repair is silently applied.",
                "Seed 2021 and calibration over the released table remain explicitly selected reconstruction choices.",
            ]
        return metadata

    def export_artifacts(self, directory: str | Path) -> dict[str, str]:
        """Write auditable source rows, small reference tables and numeric arrays.

        These are evaluator-owned scientific artifacts, including all measured
        rewards; do not pass them to an optimizer. Only the explicitly named
        files in ``directory`` are replaced. The pinned source archive remains
        in its cache and is identified by its SHA-256 in the provenance JSON.
        """
        store = ArtifactStore(str(directory))
        paths = {"provenance": store.write_json("mhc_provenance.json", self.metadata)}
        paths["arrays"] = store.write_arrays(
            "mhc_measured_arrays.npz", actions=self._actions, contexts=self._contexts,
            observations=self._observations, ic50_nM=self._ic50,
            source_row_indices=self._source_row_indices, context_gram=self._context_gram,
            context_distances=self._distances, context_schedule=self.context_schedule,
        )
        paths["descriptors"] = save_csv(
            store.path("mhc_vb_descriptors.csv"),
            [[aa, *values] for aa, values in
             (AUTHOR_VB_DESCRIPTORS if self.source == "author_nips11" else VB_DESCRIPTORS).items()],
            columns=["amino_acid", "E1", "E2", "E3", "E4", "E5"],
        )
        paths["pseudosequences"] = save_csv(
            store.path("mhc_pseudosequences.csv"),
            [[i, allele, sequence] for i, (allele, sequence) in
             enumerate(zip(self.ALLELES, MHC_PSEUDOSEQUENCES, strict=True))],
            columns=["context_id", "allele", "pseudosequence"],
        )
        paths["source_rows"] = save_csv(
            store.path("mhc_measured_rows.csv"),
            [[i, source, context, self.ALLELES[context], peptide, qualifier, ic50, cv]
             for i, (source, context, peptide, qualifier, ic50, cv) in enumerate(self._records)],
            columns=["row_index", "source_row_index", "context_id", "allele", "peptide",
                     "inequality", "ic50_nM", "source_cv"],
        )
        return paths
