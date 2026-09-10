"""BayOPy-owned warnings for backend numerical and optimization events."""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Literal

from botorch.exceptions.warnings import BadInitialCandidatesWarning, BotorchWarning
from gpytorch.utils.warnings import NumericalWarning


class BayOPyWarning(UserWarning):
    """Base class for warnings intentionally exposed by BayOPy."""


class BayOPyJitterWarning(BayOPyWarning):
    """A covariance factorization required explicitly reported diagonal jitter."""


class BayOPyAcquisitionWarning(BayOPyWarning):
    """Acquisition construction or optimization used a documented fallback."""


class BayOPyBackendWarning(BayOPyWarning):
    """A dependency warning was translated at BayOPy's backend boundary."""


class BayOPyNumericalWarning(BayOPyBackendWarning):
    """A non-jitter numerical stabilization or limitation was reported."""


class BayOPyExperimentWarning(BayOPyWarning):
    """One independently run algorithm failed while a comparison continued."""


_JITTER_PATTERN = re.compile(r"added jitter of\s+([^\s]+)", re.IGNORECASE)
_BACKEND_PATH_PARTS = (
    "/botorch/",
    "/gpytorch/",
    "/linear_operator/",
    "/pyro/",
    "/scipy/",
)


def _relay_warning(record: warnings.WarningMessage, *, operation: str) -> None:
    """Translate one captured dependency warning into a stable BayOPy warning."""

    category = record.category
    message = str(record.message)
    if issubclass(category, NumericalWarning):
        match = _JITTER_PATTERN.search(message)
        if match is None:
            warnings.warn(
                f"BayOPy numerical event during {operation}: {message}",
                BayOPyNumericalWarning,
                stacklevel=4,
            )
            return
        amount = match.group(1)
        warnings.warn(
            f"BayOPy added diagonal jitter={amount} during {operation} because the "
            "covariance matrix was not positive definite.",
            BayOPyJitterWarning,
            stacklevel=4,
        )
        return
    if issubclass(category, BadInitialCandidatesWarning):
        warnings.warn(
            f"BayOPy used random acquisition-optimizer restart points during {operation} "
            "because the sampled acquisition values were all zero. The optimization "
            "continues, but this usually indicates a locally flat acquisition surface.",
            BayOPyAcquisitionWarning,
            stacklevel=4,
        )
        return
    if issubclass(category, BotorchWarning):
        warnings.warn(
            f"BayOPy backend event during {operation}: {message}",
            BayOPyBackendWarning,
            stacklevel=4,
        )
        return
    if issubclass(category, BayOPyWarning):
        warnings.warn(message, category, stacklevel=4)
        return
    normalized_filename = record.filename.replace("\\", "/")
    if any(part in normalized_filename for part in _BACKEND_PATH_PARTS):
        warnings.warn(
            f"BayOPy backend event during {operation}: {message}",
            BayOPyBackendWarning,
            stacklevel=4,
        )
        return
    warnings.warn(message, category, stacklevel=4)


@contextmanager
def translate_backend_warnings(
    *,
    operation: str,
    capture: Sequence[Literal["botorch", "numerical"]] = ("botorch", "numerical"),
) -> Iterator[None]:
    """Expose dependency warnings through BayOPy's warning hierarchy.

    Only warnings emitted while executing the wrapped backend operation are
    captured. Every captured numerical-jitter retry is re-emitted separately,
    including its amount and the BayOPy operation where it occurred.
    """

    captured: tuple[warnings.WarningMessage, ...] = ()
    try:
        with warnings.catch_warnings(record=True) as records:
            if "botorch" in capture:
                warnings.simplefilter("always", BotorchWarning)
            if "numerical" in capture:
                warnings.simplefilter("always", NumericalWarning)
            try:
                yield
            finally:
                captured = tuple(records)
    finally:
        for record in captured:
            _relay_warning(record, operation=operation)


__all__ = [
    "BayOPyAcquisitionWarning",
    "BayOPyBackendWarning",
    "BayOPyExperimentWarning",
    "BayOPyJitterWarning",
    "BayOPyNumericalWarning",
    "BayOPyWarning",
    "translate_backend_warnings",
]
