"""Basat-owned numerical, reproducibility, and file helpers for user workflows.

Backend packages remain implementation details: tutorial users import Basat
alone. Arrays returned here are float64 Torch tensors unless specified otherwise.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import json
import os
import platform
import random
import tempfile
import urllib.request
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import torch

_seed = 1


class NumericalFunction:
    """Evaluate a user callback with an explicitly chosen numerical backend.

    Parameters
    ----------
    function : callable
        Callback ``function(x, xp)``. ``x`` is a copied backend array and
        ``xp`` is the backend namespace. The callback defines the objective;
        Basat supplies neither a benchmark nor hidden benchmark information.
    backend : {"numpy"}, optional
        Arithmetic backend, currently NumPy only. Float32/float64 inputs keep
        their precision; outputs are not coerced or normalized by the wrapper.

    Examples
    --------
    >>> import basat
    >>> def objective(x, xp):
    ...     return -xp.sum((x - 0.25) ** 2)
    >>> function = basat.NumericalFunction(objective, backend="numpy")
    >>> float(function(basat.array([0.25, 0.25])))
    -0.0

    Notes
    -----
    Tensor inputs are detached and copied to CPU. The adapter is intended for
    black-box evaluations, not automatic differentiation. Environment history
    still uses the environment's storage dtype. Opt into
    ``StochasticEnvironment(dtype="float64", evaluation_dtype="proposal")`` when the callable
    must honor source algorithms' per-proposal mixed precision.
    """

    def __init__(self, function: Any, *, backend: str = "numpy") -> None:
        if not callable(function):
            raise TypeError("function must be a callable accepting (x, xp).")
        if backend != "numpy":
            raise ValueError("NumericalFunction currently supports backend='numpy' only.")
        self._function = function
        self._backend = backend

    @property
    def function(self) -> Any:
        """User-defined callback, kept separate from optimizer state."""
        return self._function

    @property
    def backend(self) -> str:
        """Explicit arithmetic backend name."""
        return self._backend

    def __call__(self, x: Any) -> Any:
        """Pass a copied, precision-preserving array and namespace to the callback."""
        if isinstance(x, torch.Tensor):
            values = x.detach().cpu().numpy().copy()
        else:
            values = np.array(x, copy=True)
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("NumericalFunction inputs must be numeric.")
        return self.function(values, np)


class RandomStream:
    """Independent, serializable random stream for user experiment protocols.

    This reproduces NumPy's legacy RandomState draw order without requiring a
    NumPy import in user code. ``engine='pcg64'`` instead matches NumPy's
    default_rng. It does not reseed algorithms or global RNGs.
    Array draws return CPU tensors (float64 or int64); scalar draws return
    ordinary Python numbers. Normal draws consume the generator even at zero
    scale, matching historical experiment runners.

    Examples
    --------
    >>> stream = basat.RandomStream(123)
    >>> state = stream.get_state()
    >>> first = stream.uniform()
    >>> stream.set_state(state)
    >>> stream.uniform() == first
    True
    """

    def __init__(self, seed: int, *, engine: str = "mt19937"):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        if engine not in {"mt19937", "pcg64"}:
            raise ValueError("engine must be 'mt19937' or 'pcg64'.")
        self._rng: Any = (np.random.RandomState(seed) if engine == "mt19937"
                          else np.random.default_rng(seed))

    @staticmethod
    def _value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.copy())
        return value.item() if isinstance(value, np.generic) else value

    def uniform(self, low: Any = 0.0, high: Any = 1.0, size: Any = None) -> Any:
        """Draw uniform numbers on [low, high) with explicit shape."""
        return self._value(self._rng.uniform(to_list(low), to_list(high), size))

    def normal(self, loc: Any = 0.0, scale: Any = 1.0, size: Any = None) -> Any:
        """Draw Gaussian numbers, preserving cached-pair state on resume."""
        return self._value(self._rng.normal(to_list(loc), to_list(scale), size))

    def randint(self, low: int, high: int | None = None, size: Any = None) -> Any:
        """Draw integer indices, excluding the upper bound."""
        method = (self._rng.integers if isinstance(self._rng, np.random.Generator)
                  else self._rng.randint)
        return self._value(method(low, high, size))

    def choice(self, values: Any, *, size: Any = None, replace: bool = True,
               probabilities: Any = None) -> Any:
        """Draw uniformly or from an explicit categorical distribution."""
        return self._value(self._rng.choice(to_list(values), size=size, replace=replace,
                                           p=to_list(probabilities)))

    def get_state(self) -> dict[str, Any]:
        """Return all RNG state as JSON-safe values, including a cached normal."""
        if isinstance(self._rng, np.random.Generator):
            return json.loads(json.dumps(self._rng.bit_generator.state))
        state: Any = self._rng.get_state()
        name, keys, position, has_gaussian, cached_gaussian = state
        return {"bit_generator": name, "keys": keys.tolist(), "position": position,
                "has_gaussian": has_gaussian, "cached_gaussian": cached_gaussian}

    def set_state(self, state: Any) -> None:
        """Restore a state emitted by get_state, without consuming a draw."""
        if isinstance(self._rng, np.random.Generator):
            if not isinstance(state, dict) or state.get("bit_generator") != "PCG64":
                raise ValueError("Expected a RandomStream PCG64 state.")
            # Validate on a temporary generator before changing live state.
            generator = np.random.default_rng()
            generator.bit_generator.state = state
            self._rng = generator
            return
        if not isinstance(state, dict) or state.get("bit_generator") != "MT19937":
            raise ValueError("Expected a RandomStream MT19937 state.")
        keys = np.asarray(state.get("keys"))
        if keys.shape != (624,) or not np.issubdtype(keys.dtype, np.integer) \
                or ((keys < 0) | (keys >= 2**32)).any():
            raise ValueError("Invalid MT19937 key array.")
        position, has_gaussian = state.get("position"), state.get("has_gaussian")
        cached = state.get("cached_gaussian")
        if not isinstance(position, int) or not 0 <= position <= 624 \
                or has_gaussian not in (0, 1) or not isinstance(cached, (int, float)) \
                or not np.isfinite(cached):
            raise ValueError("Invalid MT19937 position or Gaussian cache.")
        self._rng.set_state(("MT19937", keys.astype(np.uint32), position,
                             has_gaussian, float(cached)))


class ArtifactStore:
    """Portable, atomic experiment files rooted in the user's project.

    The store contains no benchmark knowledge. Paths are relative to ``root``;
    parent traversal and absolute child paths are rejected. JSON is portable
    data, not an executable model pickle. Array archives never allow pickle.
    Existing files are replaced atomically only on explicit write calls.
    """

    def __init__(self, root: str):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> str:
        """Resolve a child path without allowing escape from the store."""
        if Path(name).is_absolute():
            raise ValueError("Artifact names must be relative to the store.")
        result = (self.root / name).resolve()
        if not result.is_relative_to(self.root):
            raise ValueError("Artifact path escapes the store.")
        return str(result)

    def exists(self, name: str) -> bool:
        """Return whether a named artifact exists."""
        return Path(self.path(name)).exists()

    def paths(self, pattern: str) -> list[str]:
        """Return sorted relative file names matching a user-supplied glob."""
        self.path(pattern)
        return sorted(str(p.relative_to(self.root)) for p in self.root.glob(pattern)
                      if p.is_file() and p.resolve().is_relative_to(self.root))

    def read_json(self, name: str) -> Any:
        """Read ordinary JSON data."""
        return json.loads(Path(self.path(name)).read_text(encoding="utf-8"))

    def download_json(
        self, name: str, *, url: str, sha256: str, max_bytes: int = 64 * 1024 * 1024,
        allow_nonfinite: bool = False,
    ) -> Any:
        """Read a checksum-pinned JSON artifact, downloading it only if absent.

        The explicit HTTP(S) URL and 64-character SHA-256 identify immutable
        reference data. Existing files are verified and never overwritten on
        mismatch. A missing artifact is size-limited, verified and parsed as
        ordinary JSON before atomic persistence in this caller-owned store.
        No decoder imports or executes classes named by a JSON document.

        Non-finite numbers are rejected by default. Set ``allow_nonfinite=True``
        only for external reference artifacts whose published JSON contains
        ``NaN`` or infinities (including overflowing number literals). This
        preserves the checksum-pinned original bytes; it does not certify those
        values as valid scientific data. Callers must separately validate the
        fields they use. This option does not change :meth:`write_json`.

        Examples
        --------
        >>> reference = store.download_json("reference.json", url=pinned_url,
        ...                                  sha256=expected_sha256)  # doctest: +SKIP
        """
        if (not isinstance(url, str) or not url.startswith(("https://", "http://"))
                or not isinstance(sha256, str) or len(sha256) != 64
                or any(c not in "0123456789abcdefABCDEF" for c in sha256)):
            raise ValueError("JSON downloads require an HTTP(S) URL and a full SHA-256 checksum.")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer.")
        if not isinstance(allow_nonfinite, bool):
            raise ValueError("allow_nonfinite must be a boolean.")
        target = Path(self.path(name))
        missing = not target.exists()
        if missing:
            with urllib.request.urlopen(url, timeout=60) as response:
                if not response.geturl().startswith(("https://", "http://")):
                    raise ValueError("JSON download redirected outside HTTP(S).")
                content = response.read(max_bytes + 1)
        else:
            with target.open("rb") as stream:
                content = stream.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise ValueError("JSON artifact exceeds max_bytes.")
        if hashlib.sha256(content).hexdigest() != sha256.lower():
            raise ValueError(f"JSON checksum does not match: {target}")
        def invalid_constant(value: str) -> Any:
            raise ValueError(f"Non-finite JSON constant is not permitted: {value}")
        if allow_nonfinite:
            value = json.loads(content)
        else:
            value = json.loads(content, parse_constant=invalid_constant)
            # Also reject finite JSON number spellings that overflow the parser's
            # float (for example 1e999); parse_constant only catches NaN/Infinity.
            json.dumps(value, allow_nan=False)
        if missing:
            self._write(name, content)
        return value

    def sha256(self, name: str) -> str:
        """Return a file's SHA-256 checksum without changing the artifact.

        Parameters
        ----------
        name : str
            Relative file name within this store. The same containment rules
            as :meth:`path` apply, including to symbolic links.

        Returns
        -------
        str
            Lowercase hexadecimal digest of the exact stored bytes.

        Raises
        ------
        FileNotFoundError
            If the named file does not exist.
        ValueError
            If the path escapes the store.

        Examples
        --------
        >>> store = basat.ArtifactStore("./results/my_experiment")
        >>> path = store.write_json("progress.json", {"rounds": 10})
        >>> len(store.sha256("progress.json"))
        64

        Notes
        -----
        This checks bytes, not scientific correctness or authenticity. A
        cached validation result must also identify its protocol and code.
        Avoid concurrent writers while validating an artifact.
        """
        with Path(self.path(name)).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def _write(self, name: str, content: bytes) -> str:
        target = Path(self.path(name))
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return str(target)

    def write_json(self, name: str, value: Any) -> str:
        """Atomically save JSON-safe values or nested Basat tensors."""
        return self._write(name, json.dumps(to_list(value), indent=2, allow_nan=False).encode())

    def write_arrays(self, name: str, **arrays: Any) -> str:
        """Atomically save a compressed, non-pickled named numeric archive."""
        import io

        values = {k: np.asarray(to_list(v)) for k, v in arrays.items()}
        if set(values) & {"file", "allow_pickle"}:
            raise ValueError("Array names 'file' and 'allow_pickle' are reserved.")
        if any(v.dtype.hasobject for v in values.values()):
            raise ValueError("Array archives cannot contain Python objects.")
        buffer = io.BytesIO()
        save_archive: Any = np.savez_compressed
        save_archive(buffer, **values)
        return self._write(name, buffer.getvalue())

    def read_arrays(self, name: str, *, keys: Any = None) -> dict[str, torch.Tensor]:
        """Load named numeric arrays as tensors without executing pickle."""
        with np.load(self.path(name), allow_pickle=False) as archive:
            selected = archive.files if keys is None else keys
            return {key: torch.from_numpy(archive[key].copy()) for key in selected}


def load_array(
    path: str, *, url: str | None = None, sha256: str | None = None,
    dtype: str = "float64", device: str = "cpu",
) -> torch.Tensor:
    """Read user-owned NPY data, optionally downloading a pinned missing file.

    The dataset is stored only at the caller's path, never inside Basat.
    Existing data are verified, not silently overwritten on checksum failure.
    Downloads are validated before atomic replacement. Pickled/object arrays
    are forbidden. ``url`` must be HTTP(S); no download occurs when the file
    already exists. Set ``sha256`` for reproducible source data.
    """
    import io

    target = Path(path).expanduser()
    missing = not target.exists()
    if missing:
        if url is None or not url.startswith(("https://", "http://")):
            raise ValueError("A missing array needs an explicit HTTP(S) download URL.")
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
    else:
        content = target.read_bytes()
    if sha256 is not None and hashlib.sha256(content).hexdigest() != sha256.lower():
        raise ValueError(f"Array checksum does not match: {target}")
    values = np.load(io.BytesIO(content), allow_pickle=False)
    if not isinstance(values, np.ndarray) or values.dtype.hasobject:
        if hasattr(values, "close"):
            values.close()
        raise ValueError("load_array expects one non-pickled NPY array.")
    tensor = array(values, dtype=dtype, device=device)
    if missing:
        ArtifactStore(str(target.parent))._write(target.name, content)
    return tensor


def set_seed(seed: int, *, threads: int | None = None) -> int:
    """Seed Python, NumPy, Torch (including CUDA), and the Pyro Torch backend.

    Basat's high-level factories use this seed when none is supplied. Explicit
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
    """Return the default seed used by Basat workflow factories."""
    return _seed


def derive_seed(*components: int) -> int:
    """Derive a reproducible 32-bit seed from nonnegative integer identifiers.

    NumPy SeedSequence mixes the complete sequence, so independent trial,
    method and repetition identifiers need not be packed into decimal digits.
    This does not advance any live generator or change the global seed.

    Examples
    --------
    >>> stream = basat.RandomStream(basat.derive_seed(2026, 2, 4, 17), engine="pcg64")
    """
    if not components or any(isinstance(x, bool) or not isinstance(x, int) or x < 0
                             for x in components):
        raise ValueError("Seed components must be nonnegative integers.")
    return int(np.random.SeedSequence(list(components)).generate_state(1)[0])


def array(values: Any, *, dtype: str = "float64", device: str = "cpu") -> torch.Tensor:
    """Create a numeric array with explicit precision and device."""
    from .models.utils import resolve_dtype

    actual = {"int64": torch.int64, "int32": torch.int32, "bool": torch.bool}.get(dtype)
    return torch.as_tensor(values, dtype=actual or resolve_dtype(dtype), device=device)


def linspace(
    start: float, stop: float, count: int, *, arithmetic: str = "torch",
) -> torch.Tensor:
    """Return evenly spaced CPU float64 coordinates, including both endpoints.

    Parameters
    ----------
    start, stop : float
        First and last coordinates; for ``count=1`` only ``start`` is returned.
    count : int
        Positive number of coordinates.
    arithmetic : {"torch", "numpy"}, optional
        Coordinate construction backend. The default preserves Torch's
        arithmetic. Choose ``"numpy"`` to reproduce a NumPy-generated grid
        bit for bit; algebraically equivalent grids can differ in their last
        floating-point bits. Both choices return a Torch tensor.

    Examples
    --------
    >>> basat.linspace(0.0, 1.0, 3).tolist()
    [0.0, 0.5, 1.0]
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer.")
    if arithmetic not in {"torch", "numpy"}:
        raise ValueError("arithmetic must be 'torch' or 'numpy'.")
    if arithmetic == "numpy":
        return torch.from_numpy(np.linspace(start, stop, count, dtype=np.float64))
    return torch.linspace(start, stop, count, dtype=torch.float64)


def normal_probabilities(support: Any, mean: float, variance: float) -> torch.Tensor:
    """Normalize Gaussian point densities over a finite one-dimensional support.

    Parameters
    ----------
    support : array-like of shape (n,)
        Nonempty finite coordinates. Each row receives a probability mass;
        repeated coordinates remain distinct rows with equal masses.
    mean : float
        Finite Gaussian mean.
    variance : float
        Strictly positive finite Gaussian variance, not standard deviation.

    Returns
    -------
    probabilities : CPU float64 tensor of shape (n,)
        SciPy standard-normal densities at the standardized coordinates,
        divided by their NumPy sum. This arithmetic matches common source
        experiment protocols. These are normalized point densities, **not**
        probabilities integrated over bins. Unevenly spaced supports do not
        implicitly receive quadrature weights.

    Raises
    ------
    ValueError
        If inputs are invalid or all densities underflow to zero.

    Examples
    --------
    >>> p = basat.normal_probabilities([-1.0, 0.0, 1.0], mean=0.0, variance=1.0)
    >>> bool(p[0] == p[2])
    True
    """
    from scipy.stats import norm

    coordinates = np.asarray(to_list(support), dtype=np.float64)
    center, spread = np.asarray(mean, dtype=np.float64), np.asarray(variance, dtype=np.float64)
    if coordinates.ndim != 1 or coordinates.size == 0 or not np.isfinite(coordinates).all():
        raise ValueError("support must be a nonempty finite one-dimensional array.")
    if center.ndim != 0 or not np.isfinite(center):
        raise ValueError("mean must be a finite scalar.")
    if spread.ndim != 0 or not np.isfinite(spread) or spread <= 0:
        raise ValueError("variance must be a strictly positive finite scalar.")
    density = norm.pdf((coordinates - center) / np.sqrt(spread))
    total = density.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Gaussian point densities have no finite positive mass on this support.")
    return torch.from_numpy(density / total)


def _probability_vector(value: Any, *, name: str) -> np.ndarray:
    probabilities = np.asarray(to_list(value), dtype=np.float64)
    if (probabilities.ndim != 1 or not len(probabilities)
            or not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.isclose(probabilities.sum(), 1.0, rtol=1e-12, atol=1e-12)):
        raise ValueError(f"{name} must be a finite nonnegative vector summing to one.")
    # Validate, but never renormalize or perturb explicitly supplied source data.
    return probabilities


def expected_values(values: Any, probabilities: Any) -> torch.Tensor:
    """Compute finite-distribution expectations with explicit support ordering.

    Parameters
    ----------
    values : array-like of shape (n,) or (n, m)
        Finite values on ``n`` support rows, optionally for ``m`` alternatives.
    probabilities : array-like of shape (n,)
        Nonnegative probability masses in the same row order, summing to one.
        Values are validated but are not silently renormalized.

    Returns
    -------
    expectations : CPU float64 tensor of shape (m,) or (1,)
        NumPy row-vector/matrix dot products. The row-first operation order
        preserves common source experiment arithmetic. Gradients are not
        tracked; this is an evaluation metric, not a differentiable loss.

    Examples
    --------
    >>> basat.expected_values([[0.0, 4.0], [2.0, 0.0]], [0.25, 0.75]).tolist()
    [1.5, 1.0]
    """
    p = _probability_vector(probabilities, name="probabilities")
    table = np.asarray(to_list(values), dtype=np.float64)
    if table.ndim not in (1, 2) or table.shape[0] != len(p) or not np.isfinite(table).all():
        raise ValueError("values must be a finite (n,) or (n, m) table matching probabilities.")
    return torch.from_numpy(np.dot(p.reshape(1, -1), table).reshape(-1))


def mmd_distance(p: Any, q: Any, gram: Any) -> float:
    """Evaluate finite-support maximum mean discrepancy from a kernel Gram matrix.

    Parameters
    ----------
    p, q : array-like of shape (n,)
        Probability masses on the same ordered support. They are validated,
        not renormalized.
    gram : array-like of shape (n, n)
        Finite symmetric positive-semidefinite kernel matrix on that support.
        No jitter, projection, or diagonal regularization is applied.

    Returns
    -------
    distance : float
        ``sqrt((p-q).T @ gram @ (p-q))``, using NumPy column-vector dot
        products. A tiny negative squared distance caused by floating-point
        roundoff is clipped to zero with a Basat numerical warning. A
        materially indefinite kernel or negative squared distance is rejected.

    Examples
    --------
    >>> basat.mmd_distance([1.0, 0.0], [0.0, 1.0], [[1.0, 0.5], [0.5, 1.0]])
    1.0
    """
    import warnings

    from .warnings import BasatNumericalWarning

    first, second = _probability_vector(p, name="p"), _probability_vector(q, name="q")
    matrix = np.asarray(to_list(gram), dtype=np.float64)
    n = len(first)
    if second.shape != first.shape or matrix.shape != (n, n) or not np.isfinite(matrix).all():
        raise ValueError("p, q and finite square gram must have matching support sizes.")
    tolerance = 100 * np.finfo(np.float64).eps * n * max(1.0, float(np.abs(matrix).max()))
    if not np.allclose(matrix, matrix.T, rtol=0, atol=tolerance):
        raise ValueError("gram must be symmetric.")
    if np.linalg.eigvalsh(matrix).min() < -tolerance:
        raise ValueError("gram must be positive semidefinite.")
    difference = (first - second).reshape(-1, 1)
    squared = float(np.dot(difference.T, np.dot(matrix, difference)).item())
    if squared < -tolerance * max(1.0, float(np.dot(difference.T, difference).item())):
        raise ValueError("Squared MMD is materially negative.")
    if squared < 0:
        warnings.warn(
            f"Basat MMD metric: clipped roundoff-level squared distance {squared:.6g} "
            "to zero; the supplied Gram matrix was not modified.",
            BasatNumericalWarning, stacklevel=2,
        )
        squared = 0.0
    return float(np.sqrt(squared))


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
    """Elementwise square of a Basat array or numeric sequence."""
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


def runtime_info(*, include_source_hash: bool = False) -> dict[str, str]:
    """Report Python and installed backend versions without importing backends.

    Parameters
    ----------
    include_source_hash : bool, optional
        Add ``basat_source_sha256``, a deterministic digest of the installed
        Basat Python source files and their package-relative names. The default
        is False and preserves the version-only report. Both ordinary package
        directories and source-containing wheels imported as ZIPs are supported.

    Returns
    -------
    dict of str to str
        Version information and, when requested, the package source digest.

    Examples
    --------
    >>> information = basat.runtime_info(include_source_hash=True)
    >>> len(information["basat_source_sha256"])
    64

    Notes
    -----
    The optional hash reads package resources on disk, not a snapshot of code
    already loaded into a running interpreter. Do not edit or replace installed
    code during a validated run. Any Python-source change invalidates the hash,
    including changes unrelated to a particular experiment. It does not hash
    dependency binaries, hardware, or user benchmark code. Source-free packages
    raise an error instead of claiming an implementation fingerprint.
    """
    if not isinstance(include_source_hash, bool):
        raise TypeError("include_source_hash must be a boolean.")
    result = {"python": platform.python_version()}
    for name in (
        "basat",
        "numpy",
        "scipy",
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
    for name in ("jax", "jaxlib", "numpyro"):
        # Optional numerical inference dependencies, never imported here.
        with suppress(importlib.metadata.PackageNotFoundError):
            result[name] = importlib.metadata.version(name)
    if include_source_hash:
        entries: list[tuple[str, bytes]] = []

        def collect(directory: Any, prefix: str = "") -> None:
            for child in sorted(directory.iterdir(), key=lambda item: item.name):
                relative = prefix + child.name
                if child.is_dir() and child.name != "__pycache__":
                    collect(child, relative + "/")
                elif child.is_file() and child.name.endswith(".py"):
                    entries.append((relative, child.read_bytes()))

        collect(importlib.resources.files("basat"))
        if not entries:
            raise RuntimeError("Basat Python source resources are unavailable for hashing.")
        digest = hashlib.sha256()
        for relative, content in sorted(entries):
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        result["basat_source_sha256"] = digest.hexdigest()
    return result


def new_run_directory(parent: str = "results/basat_runs") -> str:
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
