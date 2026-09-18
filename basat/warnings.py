"""Basat-owned warnings for backend numerical and optimization events."""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Literal

from botorch.exceptions.warnings import BadInitialCandidatesWarning, BotorchWarning
from gpytorch.utils.warnings import NumericalWarning


class BasatWarning(UserWarning):
    """Base class for warnings intentionally exposed by Basat."""


class BasatJitterWarning(BasatWarning):
    """A covariance factorization required explicitly reported diagonal jitter."""


class BasatAcquisitionWarning(BasatWarning):
    """Acquisition construction or optimization used a documented fallback."""


class BasatBackendWarning(BasatWarning):
    """A dependency warning was translated at Basat's backend boundary."""


class BasatNumericalWarning(BasatBackendWarning):
    """A non-jitter numerical stabilization or limitation was reported."""


class BasatExperimentWarning(BasatWarning):
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
    """Translate one captured dependency warning into a stable Basat warning."""

    category = record.category
    message = str(record.message)
    if issubclass(category, NumericalWarning):
        match = _JITTER_PATTERN.search(message)
        if match is None:
            warnings.warn(
                f"Basat numerical event during {operation}: {message}",
                BasatNumericalWarning,
                stacklevel=4,
            )
            return
        amount = match.group(1)
        warnings.warn(
            f"Basat added diagonal jitter={amount} during {operation} because the "
            "covariance matrix was not positive definite.",
            BasatJitterWarning,
            stacklevel=4,
        )
        return
    if issubclass(category, BadInitialCandidatesWarning):
        warnings.warn(
            f"Basat used random acquisition-optimizer restart points during {operation} "
            "because the sampled acquisition values were all zero. The optimization "
            "continues, but this usually indicates a locally flat acquisition surface.",
            BasatAcquisitionWarning,
            stacklevel=4,
        )
        return
    if issubclass(category, BotorchWarning):
        warnings.warn(
            f"Basat backend event during {operation}: {message}",
            BasatBackendWarning,
            stacklevel=4,
        )
        return
    if issubclass(category, BasatWarning):
        warnings.warn(message, category, stacklevel=4)
        return
    normalized_filename = record.filename.replace("\\", "/")
    if any(part in normalized_filename for part in _BACKEND_PATH_PARTS):
        warnings.warn(
            f"Basat backend event during {operation}: {message}",
            BasatBackendWarning,
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
    """Expose dependency warnings through Basat's warning hierarchy.

    Only warnings emitted while executing the wrapped backend operation are
    captured. Every captured numerical-jitter retry is re-emitted separately,
    including its amount and the Basat operation where it occurred.
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
    "BasatAcquisitionWarning",
    "BasatBackendWarning",
    "BasatExperimentWarning",
    "BasatJitterWarning",
    "BasatNumericalWarning",
    "BasatWarning",
    "translate_backend_warnings",
]
