"""Numerical policy for the explicitly selected Ax 0.1.17 embedding baseline.

The qNEI sampling, pruning and initialization routines adapt BoTorch v0.3.1
(``acquisition/monte_carlo.py``, ``sampling/samplers.py``,
``optim/initializers.py`` and ``generation/gen.py``). Copyright (c) Facebook,
Inc. and its affiliates; MIT license, reproduced in
``_historical_ax01.LICENSE.txt``. They are internal numerical components, not
an external algorithm runner. Current native and Ax 0.2.2 policies are separate.

Historical unseeded NumPy initializer entropy cannot be recovered. An independent
checkpointed MT19937 seed stream explicitly reconstructs that entropy, one seed
per fresh initializer. This is a recorded convention, not a historical seed claim.
"""

from __future__ import annotations

import copy
import time
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import gpytorch
import numpy as np
import torch
from botorch.exceptions.warnings import BadInitialCandidatesWarning, OptimizationWarning
from botorch.optim.core import OptimizationStatus
from botorch.optim.fit import fit_gpytorch_mll_scipy
from botorch.optim.utils import sample_all_priors
from botorch.utils.sampling import draw_sobol_normal_samples
from scipy.optimize import minimize

from basat.models.GaussianProcess import GaussianProcess
from basat.warnings import translate_backend_warnings

from ._ax_source_random import export_streams, validate_streams

POLICY = "ax-0.1.17-botorch-0.3.1-raw-population-qnei-streams-v4"


def standardize_ax01_outcomes(values: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Ax 0.1.17 StandardizeY: NumPy population std (the tag has no ddof).

    This convention differs from later sample-standardized Ax releases. The
    raw outcomes are transformed before the standardized known-noise floor.
    """
    raw = values.detach().cpu().numpy().astype(np.float64)
    mean, std = float(np.mean(raw)), float(np.std(raw))
    if std < 1e-8:
        std = 1.0
    standardized = torch.as_tensor((raw - mean) / std, dtype=values.dtype, device=values.device)
    return standardized, mean, std


class Ax01StreamOwner:
    """Narrow random-stream hooks used only by the explicit Ax01 algorithms."""

    config: Any
    _ax01_streams: Ax01Streams | None
    _ax01_last_diagnostics: dict[str, Any]

    @property
    def _supports_shared_trial_streams(self) -> bool:
        return self.config.source_protocol == "ax_0_1_17"

    @property
    def _supports_ax017_evaluation(self) -> bool:
        return self.config.source_protocol == "ax_0_1_17"

    def _export_random_streams(self) -> dict[str, Any]:
        if self._ax01_streams is None:
            raise NotImplementedError("Shared streams require source_protocol='ax_0_1_17'.")
        return self._ax01_streams.export()

    def _import_random_streams(self, state: Mapping[str, Any]) -> None:
        if self._ax01_streams is None:
            raise NotImplementedError("Shared streams require source_protocol='ax_0_1_17'.")
        self._ax01_streams.restore(state)

    def _reset_shared_random_streams(self, state: Mapping[str, Any]) -> None:
        if getattr(self, "evaluations", 0) or getattr(self, "last_ask", None) is not None:
            raise ValueError("Projection reset is allowed only on a fresh source learner.")
        self._import_random_streams(state)
        assert self._ax01_streams is not None
        self._ax01_streams.initializer_seeds = []
        self._ax01_streams.evaluation_draws = 0
        self._ax01_streams.readback_draws = 0
        self._ax01_last_diagnostics = {}
        self._redraw_ax01_projection()

    def _redraw_ax01_projection(self) -> None:
        raise NotImplementedError

    def _consume_source_evaluation_draws(self, count: int) -> None:
        if self._ax01_streams is None:
            raise NotImplementedError("Evaluation draws require the Ax 0.1.17 source mode.")
        if type(count) is not int or count < 0:
            raise ValueError("Discarded evaluation draw count must be a nonnegative integer.")
        self._ax01_streams.numpy.randn(count)
        self._ax01_streams.evaluation_draws += count

    def _consume_source_readback_draws(self, count: int) -> None:
        if self._ax01_streams is None:
            raise NotImplementedError("Readback draws require the Ax 0.1.17 source mode.")
        if type(count) is not int or count < 0:
            raise ValueError("Discarded readback draw count must be a nonnegative integer.")
        self._ax01_streams.numpy.randn(count)
        self._ax01_streams.readback_draws += count

    def _ax01_checkpoint(self) -> dict[str, Any]:
        assert self._ax01_streams is not None
        return {**self._ax01_streams.state_dict(),
                "last_diagnostics": copy.deepcopy(self._ax01_last_diagnostics)}

    def _validate_ax01_checkpoint(self, state: Mapping[str, Any]) -> None:
        streams = Ax01Streams(self.config.seed)
        streams.load_state_dict(state)
        if not isinstance(state.get("last_diagnostics"), Mapping):
            raise ValueError("Missing historical Ax01 diagnostics state.")

    def _load_ax01_checkpoint(self, state: Mapping[str, Any]) -> None:
        assert self._ax01_streams is not None
        self._ax01_streams.load_state_dict(state)
        self._ax01_last_diagnostics = copy.deepcopy(dict(state["last_diagnostics"]))


class Ax01Streams:
    """Own independent source NumPy, source Torch, and initializer seed streams."""

    def __init__(self, seed: int) -> None:
        self.numpy = np.random.RandomState(seed)
        self.torch = torch.Generator(device="cpu").manual_seed(seed)
        self.initializer = np.random.RandomState(seed)
        self.initializer_seeds: list[int] = []
        self.evaluation_draws = 0
        self.readback_draws = 0

    def export(self) -> dict[str, Any]:
        return export_streams(self.numpy, self.torch, self.initializer)

    def restore(self, state: Mapping[str, Any]) -> None:
        numpy_rng, generator, initializer = validate_streams(state)
        if initializer is None:
            raise ValueError("Ax 0.1.17 requires its separate initializer_seed_stream.")
        self.numpy, self.torch, self.initializer = numpy_rng, generator, initializer

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Scope library routines using global Torch, restoring callers on errors."""
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(self.torch.get_state())
            try:
                yield
            finally:
                self.torch.set_state(torch.random.get_rng_state())

    def uniform(self, dimension: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        seed = int(self.initializer.randint(0, 2**32, dtype=np.uint32))
        self.initializer_seeds.append(seed)
        values = np.random.RandomState(seed).uniform(size=(1, dimension))
        return torch.as_tensor(values, dtype=dtype, device=device)

    def state_dict(self) -> dict[str, Any]:
        return {"policy": POLICY, "streams": self.export(),
                "initializer_seeds": list(self.initializer_seeds),
                "evaluation_draws": self.evaluation_draws, "readback_draws": self.readback_draws}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("policy") != POLICY:
            raise ValueError("Ax 0.1.17 numerical/RNG policy changed; use a fresh output directory.")
        seeds = state.get("initializer_seeds")
        if not isinstance(seeds, list) or any(
            type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds
        ):
            raise ValueError("Invalid recorded Ax initializer seeds.")
        for key in ("evaluation_draws", "readback_draws"):
            if type(state.get(key)) is not int or state[key] < 0:
                raise ValueError("Invalid recorded Ax evaluation/readback draw count.")
        self.restore(state["streams"])
        self.initializer_seeds = list(seeds)
        self.evaluation_draws = state["evaluation_draws"]
        self.readback_draws = state["readback_draws"]


def fit_gp(gp: GaussianProcess) -> None:
    """Five default SciPy MAP attempts, restoring and resampling priors on retry."""
    assert gp.model is not None and gp.likelihood is not None
    gp.mll = gp._make_mll("map")
    gp.mll.train()
    original = copy.deepcopy(gp.model.state_dict())
    started = time.perf_counter()
    success = False
    with gpytorch.settings.fast_computations(log_prob=False):
        for attempt in range(1, 6):
            with warnings.catch_warnings(record=True) as captured:
                if attempt > 1:
                    gp.model.load_state_dict(original)
                    sample_all_priors(gp.model)
                result = fit_gpytorch_mll_scipy(gp.mll, method="L-BFGS-B", options={})
                if result.status != OptimizationStatus.SUCCESS and not any(
                    issubclass(w.category, OptimizationWarning) for w in captured
                ):
                    warnings.warn(str(result.message), OptimizationWarning, stacklevel=2)
            for record in captured:
                warnings.warn(record.message, record.category, stacklevel=2)
            if not any(issubclass(w.category, OptimizationWarning) for w in captured):
                success = True
                break
        if not success:
            warnings.warn("Fitting failed on all retries.", OptimizationWarning, stacklevel=2)
    gp.mll.eval()
    gp.is_fit = True
    gp.last_fit_diagnostics = {
        "method": "map", "success": success, "optimizer": "scipy_lbfgsb",
        "optimizer_options": {}, "attempts": attempt, "source_model_fit_policy": POLICY,
        "elapsed_seconds": time.perf_counter() - started,
        "training_rows": int(gp.model.train_targets.numel()),
        "fallback_reason": None, "retained_last_iterate": not success,
    }


class HistoricalQMC:
    """BoTorch 0.3.1 collapsed-batch fixed base samples, with its seed increment."""

    def __init__(self, count: int, seed: int | None = None) -> None:
        self.count = count
        self.seed = int(torch.randint(0, 1000000, (1,)).item()) if seed is None else seed
        self.base_samples: torch.Tensor | None = None

    def __call__(self, posterior: Any) -> torch.Tensor:
        # Historical event_shape includes the single output dimension. Modern
        # GPyTorchPosterior base_sample_shape omits it, hence the final squeeze.
        shape = torch.Size([self.count] + [1] * (posterior.mean.ndim - 2)
                           + list(posterior.mean.shape[-2:]))
        if self.base_samples is None or self.base_samples.shape[-2:] != shape[-2:]:
            samples = draw_sobol_normal_samples(
                d=shape[-2] * shape[-1], n=self.count, seed=self.seed,
                dtype=posterior.mean.dtype, device=posterior.mean.device,
            )
            self.seed += 1
            self.base_samples = samples.reshape(shape)
        else:
            self.base_samples = self.base_samples.reshape(shape)
        return posterior.rsample_from_base_samples(
            sample_shape=torch.Size([self.count]),
            base_samples=self.base_samples.squeeze(-1).expand(self.count, *posterior.mean.shape[:-1]),
        )


class CandidateFirstQNEI(torch.nn.Module):
    """Single-output historical qNEI: candidates precede the pruned baseline."""

    X_baseline: torch.Tensor

    def __init__(self, model: Any, baseline: torch.Tensor, seed: int, *, prune: bool = True,
                 objective_weight: float = 1.0) -> None:
        super().__init__()
        if objective_weight not in (-1.0, 1.0):
            raise ValueError("Historical single-output objective weight must be +1 or -1.")
        self.model = model
        self.objective_weight = objective_weight
        self.sampler = HistoricalQMC(512, seed)
        self.pruning_seed: int | None = None
        if prune:
            with torch.no_grad():
                posterior = model.posterior(baseline)
                pruning_sampler = HistoricalQMC(2048)
                self.pruning_seed = pruning_sampler.seed
                samples = self.objective_weight * pruning_sampler(posterior).squeeze(-1)
                self.pruned_indices = torch.unique(samples.argmax(dim=-1))
                baseline = baseline[self.pruned_indices]
        else:
            self.pruned_indices = torch.arange(len(baseline), device=baseline.device)
        self.register_buffer("X_baseline", baseline)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 2:
            X = X.unsqueeze(0)
        q = X.shape[-2]
        baseline = self.X_baseline.expand(*X.shape[:-2], *self.X_baseline.shape[-2:])
        posterior = self.model.posterior(torch.cat((X, baseline), dim=-2))
        samples = self.objective_weight * self.sampler(posterior).squeeze(-1)
        improvement = samples[..., :q].max(-1).values - samples[..., q:].max(-1).values
        return improvement.clamp_min(0).mean(0)


def select_nonnegative_starts(X: torch.Tensor, values: torch.Tensor, count: int) -> torch.Tensor:
    """Source nonnegative acquisition start selection, including zero-value ties."""
    maximum, maximum_index = values.max(dim=0)
    if maximum <= 0:
        warnings.warn("All sampled acquisition values are nonpositive; selecting random starts.",
                      BadInitialCandidatesWarning, stacklevel=2)
        return X[torch.randperm(len(X), device=X.device)[:count]]
    positive = values > 0
    positive_count = int(positive.sum())
    if positive_count < count:
        remaining = (~positive).nonzero(as_tuple=False).flatten()
        order = torch.randperm(len(remaining), device=X.device)
        positive[remaining[order[:count - positive_count]]] = True
        return X[positive]
    alpha = 1e-4
    while int((values >= alpha * maximum).sum()) < count:
        alpha *= 0.1
    eligible = values >= alpha * maximum
    indices = torch.arange(len(X), device=X.device)[eligible]
    weights = torch.exp(values[eligible] / maximum - 1)
    chosen = indices[torch.multinomial(weights, count)]
    if maximum_index not in chosen:
        chosen[-1] = maximum_index
    return X[chosen]


def initial_conditions(acquisition: CandidateFirstQNEI, dimension: int,
                       dtype: torch.dtype) -> torch.Tensor:
    """1000 raw Sobol starts, multiplied up to four times only on zero values."""
    for factor in range(1, 5):
        with warnings.catch_warnings(record=True) as captured:
            # Seed=None deliberately advances the owning global Torch stream.
            points = torch.quasirandom.SobolEngine(dimension, scramble=True).draw(
                1000 * factor, dtype=dtype
            ).reshape(-1, 1, dimension)
            with torch.no_grad():
                # Historical batch_limit is fixed to the first pool size.
                values = torch.cat([acquisition(part) for part in points.split(1000)])
            starts = select_nonnegative_starts(points, values, 20)
        bad = any(issubclass(w.category, BadInitialCandidatesWarning) for w in captured)
        for record in captured:
            if not issubclass(record.category, BadInitialCandidatesWarning):
                warnings.warn(record.message, record.category, stacklevel=2)
        if not bad:
            return starts
    warnings.warn("Unable to find non-zero acquisition values; selecting random starts.",
                  BadInitialCandidatesWarning, stacklevel=2)
    return starts


def optimize_qnei(model: Any, baseline: torch.Tensor, *, objective_weight: float = 1.0
                  ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """One joint 20-start SLSQP call, default SciPy options and no retry."""
    seed = int(torch.randint(1, 10000, (1,)).item())
    acquisition = CandidateFirstQNEI(model, baseline, seed, objective_weight=objective_weight)
    starts = initial_conditions(acquisition, baseline.shape[-1], baseline.dtype)
    shape = starts.shape

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        X = torch.as_tensor(flat, dtype=starts.dtype).reshape(shape).requires_grad_(True)
        loss = -acquisition(X).sum()
        gradient = torch.autograd.grad(loss, X)[0]
        return float(loss.detach()), gradient.detach().reshape(-1).numpy().astype(np.float64)

    result = minimize(objective, starts.reshape(-1).numpy().astype(np.float64),
                      jac=True, method="SLSQP", bounds=[(0.0, 1.0)] * starts.numel(), options={})
    if not result.success:
        # Expose failure but retain the source's returned iterate; no modern
        # restart, alternative solver, or Adam substitution is introduced.
        warnings.warn(f"Historical SLSQP returned: {result.message}; retaining its iterate.",
                      OptimizationWarning, stacklevel=2)
    candidates = torch.as_tensor(result.x, dtype=starts.dtype).reshape(shape)
    if not torch.isfinite(candidates).all() or bool(((candidates < -1e-8) | (candidates > 1 + 1e-8)).any()):
        raise RuntimeError("Historical SLSQP returned invalid candidates.")
    candidates = candidates.clamp(0, 1)
    with torch.no_grad():
        values = acquisition(candidates)
    index = values.argmax()
    return candidates[index], values[index], {
        "policy": POLICY, "acquisition": "candidate_first_qNEI", "mc_samples": 512,
        "objective_weight": objective_weight, "training_outcome": "raw_population_standardized",
        "pruning_samples": 2048, "mc_seed": seed, "pruning_seed": acquisition.pruning_seed,
        "pruned_indices": acquisition.pruned_indices.tolist(), "raw_samples": 1000,
        "num_restarts": 20, "optimization_batch_limit": 20, "optimizer": "SLSQP",
        "optimizer_options": {}, "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message), "optimizer_nfev": int(result.nfev),
        "optimizer_nit": int(result.nit), "retry_on_optimization_warning": False,
    }


@contextmanager
def numerical_context(streams: Ax01Streams) -> Iterator[None]:
    with (streams.activate(),
          translate_backend_warnings(operation="Ax-0.1.17 embedding MAP/qNEI"),
          gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False)):
        yield
