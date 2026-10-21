"""Optional analytic objectives, separate from benchmark-independent optimizers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .environments.environment import EnvironmentDataset
from .models.utils import Bounds, resolve_device, resolve_dtype


def _branin_value(x: torch.Tensor, active: tuple[int, int]) -> torch.Tensor:
    first, second = x[..., active[0]], x[..., active[1]]
    return ((second - (5.1 / (4 * math.pi**2)) * first.square()
             + (5 / math.pi) * first - 6).square()
            + 10 * (1 - 1 / (8 * math.pi)) * torch.cos(first) + 10)


class Branin:
    """Positive Branin objective with optional irrelevant coordinates.

    Parameters
    ----------
    dimension : int, default 2
        Ambient input dimension, at least two.
    active_dimensions : pair of int, default (0, 1)
        Ordered coordinates used as Branin's first and second variables.
    bounds : Bounds, optional
        Physical search box. Its active intervals must be exactly [-5, 10]
        and [0, 15]. Other coordinates may have any finite, nonempty interval.
        By default inactive coordinates have [0, 1] bounds. No hidden scaling,
        rotation, clipping, permutation or random draw is performed.
    dtype, device : optional
        Evaluation precision and device, default float64 on CPU.

    Notes
    -----
    Minimize this positive objective: ``maximize=False``. Its global minimum
    is ``1.25 / pi``. The function also evaluates points outside the box without
    clipping; the environment owns domain enforcement. Arbitrary leading batch
    dimensions are preserved. Algorithms receive only the bounds, not this
    objective's active coordinates or optimum.

    Examples
    --------
    >>> import basat
    >>> objective = basat.Branin(dimension=100, active_dimensions=(19, 64),
    ...     bounds=basat.Bounds([-5.] * 50 + [0.] * 50,
    ...                         [10.] * 50 + [15.] * 50))
    >>> environment = basat.StochasticEnvironment(
    ...     bounds=objective.bounds, function=objective, noise_std=0., dtype="float64")
    >>> objective.maximize
    False
    """

    def __init__(
        self, dimension: int = 2, *, active_dimensions: tuple[int, int] = (0, 1),
        bounds: Bounds | None = None, dtype: str | torch.dtype = "float64",
        device: str | torch.device = "cpu",
    ) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 2:
            raise ValueError("Branin dimension must be an integer of at least two.")
        if (not isinstance(active_dimensions, (tuple, list)) or len(active_dimensions) != 2
                or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < dimension
                       for i in active_dimensions) or active_dimensions[0] == active_dimensions[1]):
            raise ValueError("Branin needs two distinct active coordinate indices within its dimension.")
        self.dimension = dimension
        self.active_dimensions = (active_dimensions[0], active_dimensions[1])
        self.dtype, self.device = resolve_dtype(dtype), resolve_device(device)
        if bounds is None:
            lower = torch.zeros(dimension, dtype=self.dtype, device=self.device)
            upper = torch.ones(dimension, dtype=self.dtype, device=self.device)
            lower[list(self.active_dimensions)] = torch.tensor([-5., 0.], dtype=self.dtype, device=self.device)
            upper[list(self.active_dimensions)] = torch.tensor([10., 15.], dtype=self.dtype, device=self.device)
            bounds = Bounds(lower, upper)
        if not isinstance(bounds, Bounds) or bounds.dimension != dimension:
            raise ValueError("Branin bounds must match its ambient dimension.")
        self.bounds = Bounds(bounds.lower.to(dtype=self.dtype, device=self.device),
                             bounds.upper.to(dtype=self.dtype, device=self.device))
        for index, low, high in zip(self.active_dimensions, (-5., 0.), (10., 15.), strict=True):
            if float(self.bounds.lower[index]) != low or float(self.bounds.upper[index]) != high:
                raise ValueError("Branin's active physical bounds must be [-5, 10] and [0, 15].")
        # Keep the actual mathematical callback in the provider's state so
        # Environment's callable fingerprint includes its source definition.
        self._formula = _branin_value

    @property
    def optimal_value(self) -> float:
        """Known global minimum over the declared physical box."""
        return 1.25 / math.pi

    @property
    def maximize(self) -> bool:
        """False: positive Branin is a minimization objective."""
        return False

    @property
    def metadata(self) -> dict[str, Any]:
        """Portable objective definition; modifying it does not change the object."""
        return {"name": "Branin", "dimension": self.dimension, "effective_dimension": 2,
                "active_dimensions": list(self.active_dimensions), "maximize": False,
                "optimal_value": self.optimal_value, "input_coordinates": "physical",
                "bounds": {"lower": self.bounds.lower.tolist(), "upper": self.bounds.upper.tolist()},
                "dtype": str(self.dtype), "device": str(self.device)}

    def __call__(self, x: Any) -> torch.Tensor:
        """Evaluate physical inputs with shape ``(..., dimension)``."""
        values = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        if values.ndim < 1 or values.shape[-1] != self.dimension or not torch.isfinite(values).all():
            raise ValueError("Branin inputs must be finite and match its trailing ambient dimension.")
        return self._formula(values, self.active_dimensions)


def _advers_synthetic_value(
    x: torch.Tensor, *, interaction_coefficient: float,
) -> torch.Tensor:
    """Evaluate the two-dimensional polynomial used by the robust-BO studies."""

    first, second = x[..., 0], x[..., 1]
    return (
        -2 * first**6
        + 12.2 * first**5
        - 21.2 * first**4
        - 6.2 * first
        + 6.4 * first**3
        + 4.7 * first**2
        - second**6
        + 11 * second**5
        - 43.3 * second**4
        + 10 * second
        + 74.8 * second**3
        - 56.9 * second**2
        + 4.1 * first * second
        + interaction_coefficient * second**2 * first**2
        - 0.4 * second**2 * first
        - 0.4 * first**2 * second
    )


class AdveRSSynthetic:
    """Polynomial objective used by the released AdveRS synthetic experiment.

    Parameters
    ----------
    formula : {"released", "stableopt-paper"}, default "released"
        ``"released"`` uses the executable AdveRS notebook's coefficient
        ``0.05 * x**2 * y**2``. ``"stableopt-paper"`` exposes the earlier
        StableOPT paper's printed coefficient ``0.1`` for explicit audits; it
        is not silently substituted in an AdveRS source reproduction.
    resolution : int, default 32
        Number of points per axis returned by :meth:`grid`. Evaluation itself
        is continuous and does not round or clip inputs.
    dtype, device : optional
        Evaluation precision and device, default float64 on CPU.

    Notes
    -----
    The released AdveRS code maximizes this objective on a NumPy-style
    ``resolution x resolution`` grid spanning ``[-0.95, 3.2]`` by
    ``[-0.45, 4.4]``. The benchmark owns only the objective and domain;
    optimizers receive neither its formula nor hidden objective values.

    Examples
    --------
    >>> import basat
    >>> objective = basat.AdveRSSynthetic()
    >>> candidates = objective.grid()
    >>> candidates.shape
    torch.Size([1024, 2])
    >>> objective.maximize
    True
    """

    def __init__(
        self, *, formula: Literal["released", "stableopt-paper"] = "released",
        resolution: int = 32, dtype: str | torch.dtype = "float64",
        device: str | torch.device = "cpu",
    ) -> None:
        if formula not in {"released", "stableopt-paper"}:
            raise ValueError("formula must be 'released' or 'stableopt-paper'.")
        if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 2:
            raise ValueError("resolution must be an integer of at least two.")
        self.formula = formula
        self.resolution = resolution
        self.dtype, self.device = resolve_dtype(dtype), resolve_device(device)
        self.bounds = Bounds(
            torch.tensor([-0.95, -0.45], dtype=self.dtype, device=self.device),
            torch.tensor([3.2, 4.4], dtype=self.dtype, device=self.device),
        )
        self._interaction_coefficient = 0.05 if formula == "released" else 0.1

    @property
    def maximize(self) -> bool:
        """True: the robust-satisficing experiment maximizes this objective."""

        return True

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a portable, explicit benchmark definition."""

        return {
            "name": "AdveRSSynthetic",
            "formula": self.formula,
            "interaction_coefficient": self._interaction_coefficient,
            "source_repository": "https://github.com/Bilkent-CYBORG/AdveRS",
            "source_revision": "71a3fe89c7d051b7646d048fb6fc314e74a33095",
            "source_file": "synthetic.ipynb",
            "dimension": 2,
            "maximize": True,
            "bounds": {
                "lower": self.bounds.lower.tolist(),
                "upper": self.bounds.upper.tolist(),
            },
            "default_grid_resolution": self.resolution,
            "dtype": str(self.dtype),
            "device": str(self.device),
        }

    def grid(self, resolution: int | None = None) -> torch.Tensor:
        """Return the source-ordered finite grid using NumPy linspace arithmetic.

        Rows follow ``numpy.meshgrid`` with the first coordinate varying
        fastest, exactly as in the released notebook.
        """

        count = self.resolution if resolution is None else resolution
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError("resolution must be an integer of at least two.")
        first = np.linspace(-0.95, 3.2, count, dtype=np.float64)
        second = np.linspace(-0.45, 4.4, count, dtype=np.float64)
        first_grid, second_grid = np.meshgrid(first, second)
        values = np.column_stack((first_grid.ravel(), second_grid.ravel()))
        return torch.as_tensor(values, dtype=self.dtype, device=self.device)

    def __call__(self, x: Any) -> torch.Tensor:
        """Evaluate inputs with shape ``(..., 2)`` without clipping."""

        values = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        if values.ndim < 1 or values.shape[-1] != 2 or not torch.isfinite(values).all():
            raise ValueError("AdveRSSynthetic inputs must be finite with trailing dimension two.")
        return _advers_synthetic_value(
            values, interaction_coefficient=self._interaction_coefficient
        )


class RoBOSSynthetic:
    """Pinned finite payoff table used by the released RoBOS experiment.

    The hidden table has 50 context rows and eight action columns. Coordinates
    use NumPy's source arithmetic, and dataset rows are context-major with the
    action varying fastest. Reference and sampling probabilities are normalized
    Gaussian point densities, not integrated bin probabilities.

    Parameters
    ----------
    cache_directory : path-like, optional
        Project-relative cache for the checksum-pinned ``custom.npy`` download.
        Existing files are verified and never replaced on a checksum failure.
    local_asset : path-like, optional
        Explicit existing NPY file to verify and read without downloading or
        copying it. This file must have the published checksum.
    dtype, device : optional
        Storage and evaluation policy, default CPU float64. Source numerical
        reproduction requires these defaults. Epsilon is always computed from
        the original float64 probabilities and GPy-arithmetic context Gram.

    Notes
    -----
    This object supplies objective data, not learner observations. Pass only
    ``dataset`` to the environment; its complete rewards and hidden sampling
    probabilities must not be passed to the optimizer. The stored table is
    noiseless: its dataset does not declare observation noise variance.
    """

    SOURCE_REVISION = "96bf39400a0b2044e3c21e08b4fbeb8886584ee3"
    SOURCE_SHA256 = "737ab7d505b23466ce7017dd7878a28a9bf7894615931e2dd950f951e70d54e5"
    SOURCE_URL = (
        "https://raw.githubusercontent.com/Bilkent-CYBORG/RoBOS/"
        f"{SOURCE_REVISION}/functions/custom.npy"
    )

    def __init__(
        self, *, cache_directory: str | Path = "./data/reproducibility/robos",
        local_asset: str | Path | None = None, dtype: str | torch.dtype = "float64",
        device: str | torch.device = "cpu",
    ) -> None:
        from .models.FixedRBFKernel import FixedRBFKernel
        from .workflow import linspace, load_array, mmd_distance, normal_probabilities

        self.dtype, self.device = resolve_dtype(dtype), resolve_device(device)
        target = (
            Path(cache_directory).expanduser() / "custom.npy"
            if local_asset is None else Path(local_asset).expanduser()
        )
        if local_asset is not None and not target.is_file():
            raise FileNotFoundError(f"RoBOS local_asset must be an existing NPY file: {target}")
        table = load_array(
            str(target), url=self.SOURCE_URL if local_asset is None else None,
            sha256=self.SOURCE_SHA256, dtype="float64", device="cpu",
        )
        if table.shape != (50, 8) or not bool(torch.isfinite(table).all()):
            raise ValueError("RoBOS custom.npy must be a finite 50-context by 8-action table.")
        actions = linspace(-3.5, 3.5, 8, arithmetic="numpy")[:, None]
        contexts = linspace(-12.0, 12.0, 50, arithmetic="numpy")[:, None]
        reference = normal_probabilities(contexts[:, 0], mean=2.0, variance=1.0)
        sampling = normal_probabilities(contexts[:, 0], mean=0.0, variance=25.0)
        gram = FixedRBFKernel(input_dimension=1, lengthscale=5.0)(contexts).to_dense()
        self._epsilon = mmd_distance(sampling, reference, gram)
        self._asset_path = str(target.resolve())
        self._payoff_table = table.to(dtype=self.dtype, device=self.device).clone()
        self._actions = actions.to(dtype=self.dtype, device=self.device)
        self._contexts = contexts.to(dtype=self.dtype, device=self.device)
        self._reference = reference.to(dtype=self.dtype, device=self.device)
        self._sampling = sampling.to(dtype=self.dtype, device=self.device)
        self._context_gram = gram.to(dtype=self.dtype, device=self.device)
        self.bounds = Bounds(self._actions[0].clone(), self._actions[-1].clone())

    @property
    def actions(self) -> torch.Tensor:
        """Return the ordered eight-by-one finite action table."""
        return self._actions.detach().clone()

    @property
    def context_support(self) -> torch.Tensor:
        """Return the ordered 50-by-one finite context support."""
        return self._contexts.detach().clone()

    @property
    def payoff_table(self) -> torch.Tensor:
        """Return an evaluation-only copy of the context-by-action rewards."""
        return self._payoff_table.detach().clone()

    @property
    def reference_distribution(self) -> torch.Tensor:
        """Return learner-visible normalized N(2, 1) point probabilities."""
        return self._reference.detach().clone()

    @property
    def sampling_distribution(self) -> torch.Tensor:
        """Return hidden normalized N(0, 25) point probabilities."""
        return self._sampling.detach().clone()

    @property
    def context_gram(self) -> torch.Tensor:
        """Return the unit-amplitude, lengthscale-five context RBF Gram."""
        return self._context_gram.detach().clone()

    @property
    def epsilon(self) -> float:
        """Source-arithmetic MMD between sampling and reference probabilities."""
        return self._epsilon

    @property
    def maximize(self) -> bool:
        """True: the released experiment maximizes rewards."""
        return True

    @property
    def dataset(self) -> EnvironmentDataset:
        """Return a fresh complete hidden grid with no stored noise variance."""
        return EnvironmentDataset(
            actions=self._actions.repeat(len(self._contexts), 1),
            contexts=self._contexts.repeat_interleave(len(self._actions), dim=0),
            observations=self._payoff_table.reshape(-1).detach().clone(),
            metadata=self.metadata,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        """Return portable source provenance and an explicit domain definition."""
        return {
            "name": "RoBOSSynthetic", "source_repository": "https://github.com/Bilkent-CYBORG/RoBOS",
            "source_revision": self.SOURCE_REVISION, "source_path": "functions/custom.npy",
            "source_url": self.SOURCE_URL, "source_sha256": self.SOURCE_SHA256,
            "matrix_shape": [50, 8], "dataset_row_order": "context-major, action varies fastest",
            "action_coordinates": self._actions[:, 0].tolist(),
            "context_coordinates": self._contexts[:, 0].tolist(),
            "reference_normal": {"mean": 2.0, "variance": 1.0},
            "sampling_normal": {"mean": 0.0, "variance": 25.0},
            "probability_convention": "normalized Gaussian point densities",
            "mmd_kernel": {"name": "RBF", "variance": 1.0, "lengthscale": 5.0,
                           "arithmetic": "gpy", "computation_dtype": "float64"},
            "epsilon": self.epsilon, "maximize": True,
            "stored_observation_noise_variance": None,
            "dtype": str(self.dtype), "device": str(self.device),
        }

    def __call__(self, action: Any, context: Any) -> torch.Tensor:
        """Look up exact grid pairs with broadcastable ``(..., 1)`` coordinates.

        Off-grid coordinates are rejected, including values infinitesimally
        between stored coordinates; there is no interpolation or nearest-row
        rounding. Leading action/context dimensions follow Torch broadcasting.
        """
        values = []
        for name, raw in (("action", action), ("context", context)):
            value = torch.as_tensor(raw, device=self.device)
            if value.is_complex() or value.dtype == torch.bool:
                raise TypeError(f"RoBOS {name} must be real-valued coordinates.")
            value = torch.as_tensor(raw, dtype=self.dtype, device=self.device)
            if value.ndim < 1 or value.shape[-1] != 1 or not bool(torch.isfinite(value).all()):
                raise ValueError(f"RoBOS {name} must be finite with trailing dimension one.")
            values.append(value[..., 0])
        try:
            action_values, context_values = torch.broadcast_tensors(*values)
        except RuntimeError as error:
            raise ValueError("RoBOS action and context leading dimensions must broadcast.") from error
        action_matches = action_values[..., None] == self._actions[:, 0]
        context_matches = context_values[..., None] == self._contexts[:, 0]
        if not bool(action_matches.any(-1).all()) or not bool(context_matches.any(-1).all()):
            raise ValueError("RoBOS evaluation requires exact stored action and context coordinates.")
        return self._payoff_table[
            context_matches.to(torch.int64).argmax(-1),
            action_matches.to(torch.int64).argmax(-1),
        ].detach().clone()
