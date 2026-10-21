"""Independent numerical implementation of the 2015 additive-GP equations.

This module receives physical-coordinate observations, never an objective. It
implements a documented paper-first protocol, not a translation of the authors'
MATLAB implementation: in particular the centered likelihood and the initial
fixed-bandwidth phase are intentional. NumPy and SciPy provide numerical
primitives; no external Bayesian-optimization implementation is invoked.
"""

from __future__ import annotations

import math
import warnings
from copy import deepcopy
from itertools import pairwise
from typing import Any

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import direct
from scipy.spatial.distance import cdist

from basat.warnings import BasatJitterWarning, BasatNumericalWarning


class _AcquisitionBudgetExhausted(Exception):
    """Internal control flow; only this event permits early DIRECT termination."""


class Additive2015Policy:
    """Sequential, checkpointable additive GP with a shared SE scale/bandwidth.

    Rewards supplied to :meth:`observe` are already oriented for maximization.
    The ten-point uniform design is counted as observations. The first 25 BO
    choices hold bandwidth fixed; joint refits precede choices 26, 51, etc.
    Calibration uses only the counted initial observations, unless the caller
    supplies noise and scale bounds. Common noise is a *variance*, and the SE
    scale multiplies the covariance (it is not squared again).
    """

    revision = "basat-paper2015-additive-v1-capped-direct"
    maximum_group_size: int
    initial_points: int
    refresh_cycle: int
    decomposition_candidates: int
    acquisition_evaluations: int
    hyperparameter_iterations: int

    def __init__(
        self,
        *,
        lower: Any,
        upper: Any,
        groups: Any = None,
        maximum_group_size: int = 5,
        learn_decomposition: bool = False,
        initial_points: int = 10,
        seed: int = 1,
        refresh_cycle: int = 25,
        decomposition_candidates: int = 5,
        acquisition_evaluations: int = 900,
        common_mean: float | str = 0.0,
        noise_variance: float | None = None,
        kernel_scale_bounds: Any = None,
        bandwidth_bounds: Any = (1e-5, 5.0),
        initial_bandwidth: float = 1e-5,
        initial_bandwidth_steps: int = 25,
        hyperparameter_iterations: int = 20,
    ) -> None:
        self.lower = np.asarray(lower, dtype=float).copy()
        self.upper = np.asarray(upper, dtype=float).copy()
        if (
            self.lower.ndim != 1
            or not self.lower.size
            or self.lower.shape != self.upper.shape
            or not np.all(np.isfinite(self.lower))
            or not np.all(np.isfinite(self.upper))
            or not np.all(self.lower < self.upper)
        ):
            raise ValueError("Additive2015 bounds must be finite, ordered vectors.")
        self.dimension = self.lower.size
        integer_settings = {
            "maximum_group_size": maximum_group_size,
            "initial_points": initial_points,
            "refresh_cycle": refresh_cycle,
            "decomposition_candidates": decomposition_candidates,
            "acquisition_evaluations": acquisition_evaluations,
            "hyperparameter_iterations": hyperparameter_iterations,
        }
        for name, value in integer_settings.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
            setattr(self, name, value)
        if initial_points < 2:
            raise ValueError("Additive2015 calibration requires at least two initial points.")
        if (
            isinstance(initial_bandwidth_steps, bool)
            or not isinstance(initial_bandwidth_steps, int)
            or initial_bandwidth_steps < 0
        ):
            raise ValueError("initial_bandwidth_steps must be a nonnegative integer.")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        if not isinstance(learn_decomposition, bool):
            raise ValueError("learn_decomposition must be bool.")
        if groups is not None and learn_decomposition:
            raise ValueError("Known groups and learned decomposition are mutually exclusive.")
        if isinstance(common_mean, str):
            if common_mean != "empirical":
                raise ValueError("common_mean must be finite or 'empirical'.")
        else:
            common_mean = float(common_mean)
            if not math.isfinite(common_mean):
                raise ValueError("common_mean must be finite or 'empirical'.")
        if noise_variance is not None:
            noise_variance = float(noise_variance)
            if not math.isfinite(noise_variance) or noise_variance < 0:
                raise ValueError("noise_variance must be finite and nonnegative.")
        self.bandwidth_bounds = self._positive_bounds(bandwidth_bounds, "bandwidth_bounds")
        requested_scale = (
            None
            if kernel_scale_bounds is None
            else self._positive_bounds(kernel_scale_bounds, "kernel_scale_bounds")
        )
        self.initial_bandwidth = float(initial_bandwidth)
        if not self.bandwidth_bounds[0] <= self.initial_bandwidth <= self.bandwidth_bounds[1]:
            raise ValueError("initial_bandwidth must lie within bandwidth_bounds.")
        self.initial_bandwidth_steps = initial_bandwidth_steps
        self.learn_decomposition = learn_decomposition
        self.common_mean = common_mean
        self.rng = np.random.RandomState(seed)
        # Draw the full design at once; preserve it and the continuing MT19937.
        self.initial_design = self.rng.uniform(
            self.lower, self.upper, size=(initial_points, self.dimension)
        )
        self.groups = self._partition() if groups is None else self._validate_groups(groups)
        self.config = {
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "groups": None if groups is None else [list(group) for group in self.groups],
            **integer_settings,
            "learn_decomposition": learn_decomposition,
            "seed": seed,
            "common_mean": common_mean,
            "noise_variance": noise_variance,
            "kernel_scale_bounds": None if requested_scale is None else list(requested_scale),
            "bandwidth_bounds": list(self.bandwidth_bounds),
            "initial_bandwidth": self.initial_bandwidth,
            "initial_bandwidth_steps": initial_bandwidth_steps,
        }
        self.X = np.empty((0, self.dimension), dtype=float)
        self.y = np.empty(0, dtype=float)
        self.pending: np.ndarray | None = None
        self.bandwidth = self.initial_bandwidth
        self.scale: float | None = None
        self.noise_variance = noise_variance
        self.kernel_scale_bounds = requested_scale
        self.calibration: dict[str, Any] | None = None
        self.last_fit_bo_step = 0
        self.fit_history: list[dict[str, Any]] = []
        self.last_acquisition: dict[str, Any] | None = None
        self.acquisition_history: list[dict[str, Any]] = []
        self.acquisition_evaluation_count = 0
        self.jitter_events: list[dict[str, Any]] = []
        self.conditioning_count = 0
        self.likelihood_count = 0
        self._cache: tuple[Any, np.ndarray, np.ndarray] | None = None

    @staticmethod
    def _positive_bounds(values: Any, name: str) -> tuple[float, float]:
        array = np.asarray(values, dtype=float)
        if array.shape != (2,) or not np.all(np.isfinite(array)) or not 0 < array[0] <= array[1]:
            raise ValueError(f"{name} must be two finite positive ordered values.")
        return float(array[0]), float(array[1])

    def _validate_groups(self, groups: Any) -> tuple[tuple[int, ...], ...]:
        result = tuple(tuple(group) for group in groups)
        flat = [index for group in result for index in group]
        if (
            not result
            or any(not group for group in result)
            or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in flat)
            or len(set(flat)) != len(flat)
            or any(i < 0 or i >= self.dimension for i in flat)
        ):
            raise ValueError("Groups must be nonempty disjoint in-range integer coordinates.")
        return tuple(tuple(int(i) for i in group) for group in result)

    def _partition(self) -> tuple[tuple[int, ...], ...]:
        order = self.rng.permutation(self.dimension)
        return tuple(
            tuple(int(i) for i in order[start : start + self.maximum_group_size])
            for start in range(0, self.dimension, self.maximum_group_size)
        )

    def _mean(self) -> float:
        return (
            float(np.mean(self.y)) if self.common_mean == "empirical" else float(self.common_mean)
        )

    @staticmethod
    def _distances(x: np.ndarray, groups: Any) -> list[np.ndarray]:
        return [cdist(x[:, group], x[:, group], metric="sqeuclidean") for group in groups]

    @staticmethod
    def _covariance(distances: list[np.ndarray], bandwidth: float, scale: float) -> np.ndarray:
        result = np.zeros_like(distances[0])
        for distance in distances:
            result += scale * np.exp((-0.5 / bandwidth**2) * distance)
        return result

    def _factor(self, covariance: np.ndarray, operation: str) -> np.ndarray:
        """Factor first without jitter; report every necessary stabilization."""
        try:
            return cholesky(covariance, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            scale = max(float(np.max(np.abs(np.diag(covariance)))), np.finfo(float).tiny)
            for attempt in range(8):
                jitter = scale * 1e-12 * 10**attempt
                event = {"operation": operation, "rows": len(covariance), "jitter": jitter}
                self.jitter_events.append(event)
                warnings.warn(
                    f"Basat Add-GP-UCB 2015 added diagonal jitter={jitter:.17g} "
                    f"during {operation} ({len(covariance)} rows; total on this attempt).",
                    BasatJitterWarning,
                    stacklevel=3,
                )
                stabilized = covariance.copy()
                stabilized.flat[:: len(covariance) + 1] += jitter
                try:
                    return cholesky(stabilized, lower=True, check_finite=False)
                except np.linalg.LinAlgError:
                    continue
        raise np.linalg.LinAlgError("Additive2015 covariance remained indefinite after jitter.")

    def _likelihood(self, distances: list[np.ndarray], bandwidth: float, scale: float) -> float:
        covariance = self._covariance(distances, bandwidth, scale)
        assert self.noise_variance is not None
        covariance.flat[:: len(self.y) + 1] += self.noise_variance
        factor = self._factor(covariance, "marginal likelihood")
        residual = self.y - self._mean()
        alpha = cho_solve((factor, True), residual, check_finite=False)
        self.likelihood_count += 1
        score = -0.5 * float(residual @ alpha)
        score -= float(np.log(np.diag(factor)).sum()) + 0.5 * len(self.y) * math.log(2 * math.pi)
        if not math.isfinite(score):
            raise FloatingPointError("Additive2015 marginal likelihood was nonfinite.")
        return score

    def _optimize_parameters(
        self, distances: list[np.ndarray], *, fixed_bandwidth: float | None = None
    ) -> tuple[float, float, float, dict[str, Any]]:
        assert self.kernel_scale_bounds is not None
        bounds = [self.bandwidth_bounds, self.kernel_scale_bounds]
        fixed = [fixed_bandwidth, None]
        for index, (lower, upper) in enumerate(bounds):
            if lower == upper:
                fixed[index] = lower
        free = [index for index, value in enumerate(fixed) if value is None]

        def decode(values: Any) -> tuple[float, float]:
            parameters = list(fixed)
            for index, value in zip(free, values, strict=True):
                parameters[index] = float(np.exp(value))
            return float(parameters[0]), float(parameters[1])  # type: ignore[arg-type]

        def objective(values: Any) -> float:
            bandwidth, scale = decode(values)
            return -self._likelihood(distances, bandwidth, scale)

        if not free:
            bandwidth, scale = decode([])
            score = self._likelihood(distances, bandwidth, scale)
            return bandwidth, scale, score, {"nfev": 1, "nit": 0, "status": "fixed"}
        # Large explicit safety limit: reaching it is an error, never a silent
        # replacement of the requested 20-iteration hyperparameter search.
        result = direct(
            objective,
            [tuple(np.log(bounds[index])) for index in free],
            locally_biased=False,
            maxiter=self.hyperparameter_iterations,
            maxfun=100_000,
            vol_tol=0.0,
            len_tol=0.0,
        )
        self._check_direct(result, "hyperparameter fitting")
        if result.status == 1:
            raise RuntimeError(
                "Additive2015 hyperparameter DIRECT reached its safety evaluation limit."
            )
        bandwidth, scale = decode(result.x)
        return bandwidth, scale, -float(result.fun), self._direct_metadata(result)

    @staticmethod
    def _check_direct(result: Any, operation: str) -> None:
        if result.status < 0 or not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"Additive2015 DIRECT failed during {operation}: {result.message}")

    @staticmethod
    def _direct_metadata(result: Any) -> dict[str, Any]:
        return {"nfev": int(result.nfev), "nit": int(result.nit), "status": int(result.status)}

    def _calibrate(self) -> None:
        scale = float(np.std(self.y[: self.initial_points], ddof=1))
        if not math.isfinite(scale):
            raise ValueError("Initial reward sample standard deviation is not finite.")
        if self.noise_variance is None:
            self.noise_variance = 0.01 * scale
        if self.kernel_scale_bounds is None:
            if scale <= 0:
                raise ValueError(
                    "Initial rewards are constant; explicitly provide positive kernel_scale_bounds."
                )
            self.kernel_scale_bounds = (0.03 * scale, 30.0 * scale)
        self.calibration = {
            "observations": self.initial_points,
            "sample_standard_deviation": scale,
            "noise_variance": self.noise_variance,
            "kernel_scale_bounds": list(self.kernel_scale_bounds),
            "additional_objective_evaluations": 0,
        }

    def _fit(self, bo_step: int, *, initial: bool) -> None:
        before = self.likelihood_count
        optimizers = []
        candidates: list[dict[str, Any]] = []
        fixed_bandwidth = self.initial_bandwidth if initial else None
        if self.learn_decomposition and not initial:
            # A provisional shared parameter fit on an orthogonal projection
            # avoids favoring the current coordinate partition in the search.
            rotation, _, _ = np.linalg.svd(
                self.rng.normal(size=(self.dimension, self.dimension)), full_matrices=False
            )
            sizes = [len(group) for group in self.groups]
            cuts = np.cumsum([0, *sizes])
            blocks = tuple(tuple(range(a, b)) for a, b in pairwise(cuts))
            provisional = self._distances(self.X @ rotation, blocks)
            bandwidth, scale, _, details = self._optimize_parameters(provisional)
            optimizers.append({"phase": "provisional", **details})
            del provisional
            for _ in range(self.decomposition_candidates):
                groups = self._partition()
                score = self._likelihood(self._distances(self.X, groups), bandwidth, scale)
                candidates.append({"groups": [list(group) for group in groups], "score": score})
            best = max(candidates, key=lambda item: item["score"])
            self.groups = self._validate_groups(best["groups"])
        distances = self._distances(self.X, self.groups)
        bandwidth, scale, score, details = self._optimize_parameters(
            distances, fixed_bandwidth=fixed_bandwidth
        )
        optimizers.append({"phase": "initial_amplitude" if initial else "final", **details})
        self.bandwidth, self.scale = bandwidth, scale
        self.last_fit_bo_step = bo_step
        self.fit_history.append(
            {
                "bo_step": bo_step,
                "training_rows": len(self.y),
                "kind": "initial_amplitude" if initial else "joint",
                "bandwidth": bandwidth,
                "scale": scale,
                "log_marginal_likelihood": score,
                "likelihood_evaluations": self.likelihood_count - before,
                "groups": [list(group) for group in self.groups],
                "candidate_decompositions": candidates,
                "optimizers": optimizers,
            }
        )
        self._cache = None

    def _condition(self) -> tuple[np.ndarray, np.ndarray]:
        if self.scale is None or self.noise_variance is None or not len(self.y):
            raise RuntimeError("Additive2015 posterior is unavailable before the initial fit.")
        key = (
            len(self.y),
            self.groups,
            self.bandwidth,
            self.scale,
            self.noise_variance,
            self._mean(),
        )
        if self._cache is None or self._cache[0] != key:
            covariance = self._covariance(
                self._distances(self.X, self.groups), self.bandwidth, self.scale
            )
            covariance.flat[:: len(self.y) + 1] += self.noise_variance
            factor = self._factor(covariance, "posterior conditioning")
            alpha = cho_solve((factor, True), self.y - self._mean(), check_finite=False)
            self._cache = (key, factor, alpha)
            self.conditioning_count += 1
        return self._cache[1], self._cache[2]

    def _points(self, points: Any, dimension: int) -> np.ndarray:
        result = np.asarray(points, dtype=float)
        if result.ndim == 1:
            result = result.reshape(1, -1)
        if result.ndim != 2 or result.shape[1] != dimension or not np.all(np.isfinite(result)):
            raise ValueError(f"Posterior points must be finite with shape (N, {dimension}).")
        return result

    def _posterior(
        self, cross: np.ndarray, prior: np.ndarray, mean: float, *, covariance: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        factor, alpha = self._condition()
        solved = solve_triangular(factor, cross.T, lower=True, check_finite=False)
        result_mean = mean + cross @ alpha
        uncertainty = prior - (solved.T @ solved if covariance else np.sum(solved**2, axis=0))
        diagonal = np.diag(uncertainty) if covariance else uncertainty
        tolerance = 1e-8 * max(float(self.scale or 0) * len(self.groups), np.finfo(float).tiny)
        if np.any(diagonal < -tolerance) or not np.all(np.isfinite(uncertainty)):
            raise FloatingPointError("Additive2015 posterior covariance is numerically invalid.")
        if np.any(diagonal < 0):
            warnings.warn(
                "Basat Add-GP-UCB 2015 clipped negative posterior variance from roundoff to zero.",
                BasatNumericalWarning,
                stacklevel=3,
            )
            if covariance:
                uncertainty = uncertainty.copy()
                np.fill_diagonal(uncertainty, np.maximum(diagonal, 0))
            else:
                uncertainty = np.maximum(uncertainty, 0)
        return result_mean, uncertainty

    def component_posterior(
        self, group_id: int, points: Any, *, covariance: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return one latent component's mean and variance/full covariance."""
        group = self.groups[group_id]
        query = self._points(points, len(group))
        if self.scale is None:
            raise RuntimeError("Additive2015 posterior is unavailable before the initial fit.")
        cross = self.scale * np.exp(
            -0.5 * cdist(query, self.X[:, group], "sqeuclidean") / self.bandwidth**2
        )
        prior = (
            self.scale * np.exp(-0.5 * cdist(query, query, "sqeuclidean") / self.bandwidth**2)
            if covariance
            else np.full(len(query), self.scale)
        )
        return self._posterior(cross, prior, 0.0, covariance=covariance)

    def combined_posterior(
        self, points: Any, *, covariance: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return latent total posterior, including posterior cross-component terms."""
        query = self._points(points, self.dimension)
        if self.scale is None:
            raise RuntimeError("Additive2015 posterior is unavailable before the initial fit.")
        cross = np.zeros((len(query), len(self.y)))
        for group in self.groups:
            cross += self.scale * np.exp(
                -0.5 * cdist(query[:, group], self.X[:, group], "sqeuclidean") / self.bandwidth**2
            )
        prior = (
            self._covariance(self._distances(query, self.groups), self.bandwidth, self.scale)
            if covariance
            else np.full(len(query), len(self.groups) * self.scale)
        )
        return self._posterior(cross, prior, self._mean(), covariance=covariance)

    def _choose(self) -> np.ndarray:
        action = 0.5 * (self.lower + self.upper)
        per_group = math.ceil(self.acquisition_evaluations / len(self.groups))
        components = []
        for group_id, group in enumerate(self.groups):
            beta = 0.2 * len(group) * math.log(2 * (len(self.y) + 1))
            evaluations = 0
            best_point = None
            best_value = float("inf")

            def objective(point: Any, group_id: int = group_id, beta: float = beta) -> float:
                nonlocal evaluations, best_point, best_value
                if evaluations >= per_group:
                    raise _AcquisitionBudgetExhausted
                mean, variance = self.component_posterior(group_id, point)
                value = -float(mean[0] + math.sqrt(beta * variance[0]))
                if not math.isfinite(value):
                    raise FloatingPointError("Additive2015 component acquisition was nonfinite.")
                evaluations += 1
                # Equal scores retain the first candidate, independently of
                # whichever rectangle the backend is inspecting when stopped.
                if value < best_value:
                    best_value = value
                    best_point = np.asarray(point, dtype=float).copy()
                return value

            try:
                result = direct(
                    objective,
                    [(float(self.lower[i]), float(self.upper[i])) for i in group],
                    maxfun=per_group,
                    maxiter=max(10_000, 10 * per_group),
                    locally_biased=False,
                    vol_tol=0.0,
                    len_tol=0.0,
                )
            except _AcquisitionBudgetExhausted:
                details = {
                    "nfev": evaluations,
                    "nit": None,
                    "status": "budget_exhausted",
                    "stopped_for_budget": True,
                }
            else:
                self._check_direct(result, "component acquisition")
                details = {
                    **self._direct_metadata(result),
                    "nfev": evaluations,
                    "stopped_for_budget": result.status == 1 or evaluations == per_group,
                }
            if best_point is None:
                raise RuntimeError("Additive2015 DIRECT returned without evaluating a candidate.")
            action[list(group)] = best_point
            components.append(
                {
                    "group": list(group),
                    "beta": beta,
                    "budget_policy": "hard_callback_cap",
                    **details,
                }
            )
        duplicate = bool(np.min(np.linalg.norm(self.X - action, axis=1)) / self.bandwidth < 1e-10)
        duplicate_attempts = 0
        if duplicate:
            original = action.copy()
            ignored = sorted(
                set(range(self.dimension)) - {i for group in self.groups for i in group}
            )
            for attempt in range(1, 101):
                duplicate_attempts = attempt
                action = np.clip(
                    original + self.rng.normal(size=self.dimension) * (0.1 * self.bandwidth),
                    self.lower,
                    self.upper,
                )
                # Keep every deliberately unmodeled coordinate deterministic;
                # still draw D normals so the owned RNG stream is unambiguous.
                action[ignored] = original[ignored]
                if np.min(np.linalg.norm(self.X - action, axis=1)) / self.bandwidth >= 1e-10:
                    break
            else:
                raise RuntimeError(
                    "Additive2015 could not perturb a duplicate to a distinct action."
                )
        self.last_acquisition = {
            "training_rows": len(self.y),
            "nominal_total_evaluations": self.acquisition_evaluations,
            "budget_policy": "hard_callback_cap",
            "allocated_per_group": per_group,
            "actual_total_evaluations": sum(item["nfev"] for item in components),
            "components": components,
            "duplicate_perturbed": duplicate,
            "duplicate_perturbation_attempts": duplicate_attempts,
        }
        accounting = {
            "bo_step": len(self.y) - self.initial_points + 1,
            "training_rows": len(self.y),
            "actual_total_evaluations": self.last_acquisition["actual_total_evaluations"],
            "components": [
                {key: item[key] for key in ("nfev", "nit", "stopped_for_budget")}
                for item in components
            ],
            "duplicate_perturbation_attempts": duplicate_attempts,
        }
        # Copy-on-append preserves the old list in the proposal transaction's
        # rollback snapshot. Re-reading a pending proposal never reaches here.
        self.acquisition_history = [*self.acquisition_history, accounting]
        self.acquisition_evaluation_count += accounting["actual_total_evaluations"]
        return action

    def propose(self) -> np.ndarray:
        """Return a pending action; retrying after interruption consumes no extra RNG."""
        if self.pending is not None:
            return self.pending.copy()
        # Roll back a partially interrupted parameter/decomposition/acquisition
        # operation, including RNG and diagnostics, without replaying observations.
        snapshot = dict(self.__dict__)
        snapshot["fit_history"] = deepcopy(self.fit_history)
        snapshot["jitter_events"] = deepcopy(self.jitter_events)
        rng_state = self.rng.get_state()
        try:
            if len(self.y) < self.initial_points:
                action = self.initial_design[len(self.y)].copy()
            else:
                if self.calibration is None:
                    self._calibrate()
                bo_step = len(self.y) - self.initial_points + 1
                if self.scale is None:
                    self._fit(bo_step, initial=self.initial_bandwidth_steps > 0)
                elif (
                    bo_step > self.initial_bandwidth_steps
                    and (bo_step - self.initial_bandwidth_steps - 1) % self.refresh_cycle == 0
                    and self.last_fit_bo_step != bo_step
                ):
                    self._fit(bo_step, initial=False)
                action = self._choose()
            self.pending = action.copy()
            return action.copy()
        except BaseException:
            self.__dict__.clear()
            self.__dict__.update(snapshot)
            self.rng.set_state(rng_state)
            raise

    def observe(self, action: Any, reward: float) -> None:
        """Commit exactly one finite reward for the pending physical action."""
        values = np.asarray(action, dtype=float)
        value = float(reward)
        if (
            self.pending is None
            or values.shape != (self.dimension,)
            or not np.array_equal(values, self.pending)
        ):
            raise ValueError("Observation action does not match the pending Additive2015 proposal.")
        if not math.isfinite(value):
            raise ValueError("Additive2015 rewards must be finite.")
        new_x = np.vstack((self.X, values))
        new_y = np.append(self.y, value)
        self.X, self.y = new_x, new_y
        self.pending = None
        self._cache = None

    def diagnostics(self) -> dict[str, Any]:
        """Return diagnostics without fitting, conditioning, or consuming randomness."""
        return deepcopy(
            {
                "policy_revision": self.revision,
                "evaluations": len(self.y),
                "groups": [list(group) for group in self.groups],
                "bandwidth": self.bandwidth,
                "kernel_scale": self.scale,
                "calibration": self.calibration,
                "fit_history": self.fit_history,
                "last_acquisition": self.last_acquisition,
                "acquisition_history": self.acquisition_history,
                "acquisition_evaluation_count": self.acquisition_evaluation_count,
                "conditioning_count": self.conditioning_count,
                "likelihood_count": self.likelihood_count,
                "jitter_events": self.jitter_events,
                "pending": self.pending is not None,
            }
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize history, fitted parameters and pending/RNG state, without a fit."""
        name, keys, position, has_gauss, cached = self.rng.get_state()
        return deepcopy(
            {
                "revision": self.revision,
                "config": self.config,
                "X": self.X.tolist(),
                "y": self.y.tolist(),
                "initial_design": self.initial_design.tolist(),
                "groups": [list(group) for group in self.groups],
                "pending": None if self.pending is None else self.pending.tolist(),
                "rng": [
                    name,
                    np.asarray(keys, dtype=np.uint32).tolist(),
                    position,
                    has_gauss,
                    cached,
                ],
                "bandwidth": self.bandwidth,
                "scale": self.scale,
                "noise_variance": self.noise_variance,
                "kernel_scale_bounds": self.kernel_scale_bounds,
                "calibration": self.calibration,
                "last_fit_bo_step": self.last_fit_bo_step,
                "fit_history": self.fit_history,
                "last_acquisition": self.last_acquisition,
                "acquisition_history": self.acquisition_history,
                "acquisition_evaluation_count": self.acquisition_evaluation_count,
                "jitter_events": self.jitter_events,
                "conditioning_count": self.conditioning_count,
                "likelihood_count": self.likelihood_count,
            }
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Validate and restore atomically; never refit a GP or replay an objective."""
        restored = deepcopy(state)
        if restored["revision"] != self.revision or restored["config"] != self.config:
            raise ValueError("Additive2015 checkpoint policy/configuration does not match.")
        x = np.asarray(restored["X"], dtype=float).reshape((-1, self.dimension))
        y = np.asarray(restored["y"], dtype=float)
        initial = np.asarray(restored["initial_design"], dtype=float)
        if (
            y.ndim != 1
            or len(x) != len(y)
            or initial.shape != (self.initial_points, self.dimension)
            or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y))
            or not np.all(np.isfinite(initial))
            or np.any(x < self.lower)
            or np.any(x > self.upper)
            or np.any(initial < self.lower)
            or np.any(initial > self.upper)
            or not np.array_equal(initial, self.initial_design)
            or not np.array_equal(
                x[: self.initial_points], initial[: min(len(y), self.initial_points)]
            )
        ):
            raise ValueError("Additive2015 checkpoint history/initial design is invalid.")
        groups = self._validate_groups(restored["groups"])
        if self.config["groups"] is not None:
            if [list(group) for group in groups] != self.config["groups"]:
                raise ValueError("Additive2015 checkpoint changed known groups.")
        elif sorted(i for group in groups for i in group) != list(range(self.dimension)) or [
            len(group) for group in groups
        ] != [
            min(self.maximum_group_size, self.dimension - start)
            for start in range(0, self.dimension, self.maximum_group_size)
        ]:
            raise ValueError("Additive2015 checkpoint learned partition is invalid.")
        pending = (
            None if restored["pending"] is None else np.asarray(restored["pending"], dtype=float)
        )
        if pending is not None and (
            pending.shape != (self.dimension,)
            or not np.all(np.isfinite(pending))
            or np.any(pending < self.lower)
            or np.any(pending > self.upper)
            or (len(y) < self.initial_points and not np.array_equal(pending, initial[len(y)]))
        ):
            raise ValueError("Additive2015 checkpoint pending action is invalid.")
        bandwidth = float(restored["bandwidth"])
        if not self.bandwidth_bounds[0] <= bandwidth <= self.bandwidth_bounds[1]:
            raise ValueError("Additive2015 checkpoint bandwidth is invalid.")
        scale = restored["scale"]
        noise = restored["noise_variance"]
        scale_bounds = restored["kernel_scale_bounds"]
        if scale_bounds is not None:
            scale_bounds = self._positive_bounds(scale_bounds, "checkpoint kernel_scale_bounds")
        if noise is not None and (not math.isfinite(float(noise)) or noise < 0):
            raise ValueError("Additive2015 checkpoint noise is invalid.")
        if scale is not None and (
            not math.isfinite(float(scale))
            or scale_bounds is None
            or not scale_bounds[0] <= scale <= scale_bounds[1]
            or noise is None
            or restored["calibration"] is None
            or len(y) < self.initial_points
        ):
            raise ValueError("Additive2015 checkpoint fitted scale/calibration is invalid.")
        for key in (
            "last_fit_bo_step",
            "conditioning_count",
            "likelihood_count",
            "acquisition_evaluation_count",
        ):
            if (
                isinstance(restored[key], bool)
                or not isinstance(restored[key], int)
                or restored[key] < 0
            ):
                raise ValueError(f"Additive2015 checkpoint {key} is invalid.")
        if restored["last_fit_bo_step"] > max(0, len(y) - self.initial_points + 1):
            raise ValueError("Additive2015 checkpoint fit is ahead of its observations.")
        if restored["calibration"] is not None:
            if len(y) < self.initial_points:
                raise ValueError("Additive2015 checkpoint calibration precedes initialization.")
            sample_sd = float(np.std(y[: self.initial_points], ddof=1))
            expected_noise = self.config["noise_variance"]
            if expected_noise is None:
                expected_noise = 0.01 * sample_sd
            expected_bounds = self.config["kernel_scale_bounds"]
            if expected_bounds is None:
                expected_bounds = [0.03 * sample_sd, 30.0 * sample_sd]
            expected_calibration = {
                "observations": self.initial_points,
                "sample_standard_deviation": sample_sd,
                "noise_variance": expected_noise,
                "kernel_scale_bounds": list(expected_bounds),
                "additional_objective_evaluations": 0,
            }
            if (
                restored["calibration"] != expected_calibration
                or noise != expected_noise
                or scale_bounds != tuple(expected_bounds)
            ):
                raise ValueError(
                    "Additive2015 checkpoint calibration does not match its initial data."
                )
        elif noise != self.config["noise_variance"] or scale_bounds != (
            None
            if self.config["kernel_scale_bounds"] is None
            else tuple(self.config["kernel_scale_bounds"])
        ):
            raise ValueError(
                "Additive2015 checkpoint unfitted parameters do not match configuration."
            )
        history = restored["fit_history"]
        if not isinstance(history, list) or not isinstance(restored["jitter_events"], list):
            raise ValueError("Additive2015 checkpoint diagnostic history must be a list.")
        if scale is None:
            if history or restored["last_fit_bo_step"] or len(y) > self.initial_points:
                raise ValueError("Additive2015 checkpoint is missing fitted parameters.")
        elif not history or (
            history[-1]["bo_step"] != restored["last_fit_bo_step"]
            or history[-1]["training_rows"]
            != self.initial_points + restored["last_fit_bo_step"] - 1
            or history[-1]["scale"] != scale
            or history[-1]["bandwidth"] != bandwidth
            or history[-1]["groups"] != [list(group) for group in groups]
        ):
            raise ValueError("Additive2015 checkpoint fitted parameters and fit history disagree.")
        acquisition_history = restored["acquisition_history"]
        proposal_count = max(0, len(y) - self.initial_points) + int(
            pending is not None and len(y) >= self.initial_points
        )
        if not isinstance(acquisition_history, list) or len(acquisition_history) != proposal_count:
            raise ValueError("Additive2015 checkpoint acquisition history count is inconsistent.")
        counted = 0
        group_cap = math.ceil(self.acquisition_evaluations / len(groups))
        for index, entry in enumerate(acquisition_history):
            if (
                entry["bo_step"] != index + 1
                or entry["training_rows"] != self.initial_points + index
                or not isinstance(entry["components"], list)
                or len(entry["components"]) != len(groups)
            ):
                raise ValueError("Additive2015 checkpoint acquisition history is inconsistent.")
            total = 0
            for component in entry["components"]:
                evaluations, iterations = component["nfev"], component["nit"]
                if (
                    isinstance(evaluations, bool)
                    or not isinstance(evaluations, int)
                    or not 1 <= evaluations <= group_cap
                    or (
                        iterations is not None
                        and (
                            isinstance(iterations, bool)
                            or not isinstance(iterations, int)
                            or iterations < 0
                        )
                    )
                    or not isinstance(component["stopped_for_budget"], bool)
                ):
                    raise ValueError("Additive2015 checkpoint component accounting is invalid.")
                total += evaluations
            if total != entry["actual_total_evaluations"]:
                raise ValueError(
                    "Additive2015 checkpoint per-proposal acquisition total is invalid."
                )
            counted += total
        if counted != restored["acquisition_evaluation_count"]:
            raise ValueError("Additive2015 checkpoint cumulative acquisition total is invalid.")
        if proposal_count and (
            restored["last_acquisition"]["training_rows"]
            != acquisition_history[-1]["training_rows"]
            or restored["last_acquisition"]["actual_total_evaluations"]
            != acquisition_history[-1]["actual_total_evaluations"]
        ):
            raise ValueError(
                "Additive2015 checkpoint latest acquisition accounting is inconsistent."
            )
        random = np.random.RandomState()
        name, keys, position, has_gauss, cached = restored["rng"]
        if (
            name != "MT19937"
            or not isinstance(keys, list)
            or len(keys) != 624
            or any(
                isinstance(key, bool) or not isinstance(key, int) or not 0 <= key < 2**32
                for key in keys
            )
            or isinstance(position, bool)
            or not isinstance(position, int)
            or not 0 <= position <= 624
            or has_gauss not in (0, 1)
            or not math.isfinite(float(cached))
        ):
            raise ValueError("Additive2015 checkpoint MT19937 state is invalid.")
        random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
        values = {
            "X": x,
            "y": y,
            "initial_design": initial,
            "groups": groups,
            "pending": pending,
            "rng": random,
            "bandwidth": bandwidth,
            "scale": scale,
            "noise_variance": noise,
            "kernel_scale_bounds": scale_bounds,
            "_cache": None,
        }
        for key in (
            "calibration",
            "last_fit_bo_step",
            "fit_history",
            "last_acquisition",
            "acquisition_history",
            "acquisition_evaluation_count",
            "jitter_events",
            "conditioning_count",
            "likelihood_count",
        ):
            values[key] = restored[key]
        self.__dict__.update(values)
