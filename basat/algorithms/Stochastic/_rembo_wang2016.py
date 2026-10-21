"""Independent, sequential implementation of the REMBO paper equations.

This module owns numerical state, never an objective function. The explicit
paper-first policy is not a literal execution of the released MATLAB Branin
demo (which uses UCB, unclipped actions and a DIRECT/Nelder--Mead optimizer).
NumPy, SciPy and pycma supply numerical primitives, not an external BO runner.
"""

from __future__ import annotations

import math
import warnings
from copy import deepcopy
from hashlib import sha256
from typing import Any

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import direct
from scipy.spatial.distance import cdist
from scipy.special import log_ndtr

from basat.warnings import BasatJitterWarning, BasatNumericalWarning


def log_expected_improvement(mean: Any, variance: Any, incumbent: float) -> np.ndarray:
    """Log EI for maximization, including zero variance and extreme left tails.

    The asymptotic branch evaluates the positive Mills-ratio remainder rather than
    subtracting nearly equal floating-point values. It changes neither EI's
    ordering nor its improvement threshold (which is exactly zero here).
    """
    mu, var = np.broadcast_arrays(np.asarray(mean, float), np.asarray(variance, float))
    if np.any(var < 0) or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(var)):
        raise ValueError("EI requires finite means and nonnegative finite variances.")
    answer = np.full(mu.shape, -np.inf)
    delta = mu - float(incumbent)
    deterministic = (var == 0) & (delta > 0)
    answer[deterministic] = np.log(delta[deterministic])
    active = var > 0
    sigma = np.sqrt(var[active])
    z = delta[active] / sigma
    log_phi = -0.5 * z * z - 0.5 * math.log(2 * math.pi)
    value = np.empty(z.shape)
    positive = z >= 0
    with np.errstate(divide="ignore"):
        value[positive] = np.logaddexp(
            np.log(z[positive]) + log_ndtr(z[positive]), log_phi[positive]
        )
    moderate = (z < 0) & (z >= -10)
    log_ratio = np.log(-z[moderate]) + log_ndtr(z[moderate]) - log_phi[moderate]
    value[moderate] = log_phi[moderate] + np.log(-np.expm1(log_ratio))
    tail = z < -10
    a = -z[tail]
    inverse_square = (1.0 / a) ** 2
    term = np.ones(a.shape)
    series = term.copy()
    remaining = np.ones(a.shape, dtype=bool)
    for k in range(1, 200):
        next_term = term * (-(2 * k + 1) * inverse_square)
        use = remaining & (np.abs(next_term) < np.abs(term))
        series[use] += next_term[use]
        remaining = use & (np.abs(next_term) > np.finfo(float).eps * np.abs(series))
        term = next_term
        if not np.any(remaining):
            break
    value[tail] = log_phi[tail] - 2 * np.log(a) + np.log(series)
    answer[active] = np.log(sigma) + value
    return answer


class REMBO2016Policy:
    """Checkpointable scalar-lengthscale REMBO with one zero-point initialization.

    All observations are physical-coordinate actions and maximization-oriented
    raw rewards. The GP operates on their *latent* coordinates, with zero mean,
    unit signal variance and fixed observation-noise variance. A completed
    observation can trigger a fit; asking, serializing and restoring do not
    silently refit a previously fitted model.
    """

    revision = "basat-wang2016-paper-v1"
    embedding_dimension: int
    refit_interval: int
    exploitation_count_limit: int
    acquisition_direct_evaluations: int
    acquisition_direct_iterations: int
    hyperparameter_direct_evaluations: int
    hyperparameter_direct_iterations: int
    cma_evaluations: int
    noise_variance: float
    initial_lengthscale: float
    lengthscale_lower: float
    lengthscale_upper: float
    t_sigma: float
    upper_bound_multiplier: float
    cma_tolfun: float

    def __init__(
        self,
        *,
        lower: Any,
        upper: Any,
        embedding_dimension: int,
        seed: int = 1,
        Y_lower: Any = None,
        Y_upper: Any = None,
        acquisition: str = "EI",
        ei_xi: float = 1e-4,
        noise_variance: float = 1e-10,
        initial_lengthscale: float = 0.1,
        lengthscale_lower: float = 0.01,
        lengthscale_upper: float = 50.0,
        t_sigma: float = 0.002,
        refit_interval: int = 20,
        exploitation_count_limit: int = 5,
        upper_bound_multiplier: float = 0.9,
        acquisition_direct_evaluations: int = 500,
        acquisition_direct_iterations: int = 200,
        hyperparameter_direct_evaluations: int = 400,
        hyperparameter_direct_iterations: int = 300,
        cma_evaluations: int = 2000,
        cma_tolfun: float = 1e-8,
    ) -> None:
        self.lower = np.asarray(lower, dtype=float).copy()
        self.upper = np.asarray(upper, dtype=float).copy()
        if (
            self.lower.ndim != 1
            or not self.lower.size
            or self.upper.shape != self.lower.shape
            or not np.all(np.isfinite(self.lower))
            or not np.all(np.isfinite(self.upper))
            or np.any(self.lower >= self.upper)
        ):
            raise ValueError("REMBO physical bounds must be finite ordered vectors.")
        integer_settings = {
            "embedding_dimension": embedding_dimension,
            "refit_interval": refit_interval,
            "exploitation_count_limit": exploitation_count_limit,
            "acquisition_direct_evaluations": acquisition_direct_evaluations,
            "acquisition_direct_iterations": acquisition_direct_iterations,
            "hyperparameter_direct_evaluations": hyperparameter_direct_evaluations,
            "hyperparameter_direct_iterations": hyperparameter_direct_iterations,
            "cma_evaluations": cma_evaluations,
        }
        for name, value in integer_settings.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
            setattr(self, name, value)
        if embedding_dimension > self.lower.size:
            raise ValueError("Embedding dimension cannot exceed physical dimension.")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer.")
        if (Y_lower is None) != (Y_upper is None):
            raise ValueError("Both latent bounds must be supplied together.")
        radius = math.sqrt(embedding_dimension)
        self.Y_lower = (
            np.full(embedding_dimension, -radius)
            if Y_lower is None
            else np.asarray(Y_lower, float).copy()
        )
        self.Y_upper = (
            np.full(embedding_dimension, radius)
            if Y_upper is None
            else np.asarray(Y_upper, float).copy()
        )
        if (
            self.Y_lower.shape != (embedding_dimension,)
            or self.Y_upper.shape != (embedding_dimension,)
            or not np.all(np.isfinite(self.Y_lower))
            or not np.all(np.isfinite(self.Y_upper))
            or np.any(self.Y_lower >= self.Y_upper)
            or np.any(self.Y_lower > 0)
            or np.any(self.Y_upper < 0)
        ):
            raise ValueError("Latent bounds must be ordered finite vectors containing zero.")
        float_settings = {
            "noise_variance": noise_variance,
            "initial_lengthscale": initial_lengthscale,
            "lengthscale_lower": lengthscale_lower,
            "lengthscale_upper": lengthscale_upper,
            "t_sigma": t_sigma,
            "upper_bound_multiplier": upper_bound_multiplier,
            "cma_tolfun": cma_tolfun,
        }
        for name, float_value in float_settings.items():
            if (
                isinstance(float_value, bool)
                or not math.isfinite(float(float_value))
                or float(float_value) <= 0
            ):
                raise ValueError(f"{name} must be positive and finite.")
            setattr(self, name, float(float_value))
        if not lengthscale_lower <= initial_lengthscale <= lengthscale_upper:
            raise ValueError("Initial lengthscale must lie within its positive bounds.")
        if not 0 < upper_bound_multiplier < 1:
            raise ValueError("upper_bound_multiplier must lie strictly between zero and one.")
        if acquisition not in {"EI", "UCB"}:
            raise ValueError("The paper policy supports EI or UCB.")
        if isinstance(ei_xi, bool) or not math.isfinite(float(ei_xi)) or ei_xi < 0:
            raise ValueError("ei_xi must be finite and nonnegative.")
        self.acquisition = acquisition
        self.ei_xi = float(ei_xi)
        self.seed = seed
        self.config = {
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "seed": seed,
            "acquisition": acquisition,
            "ei_xi": self.ei_xi,
            "Y_lower": self.Y_lower.tolist(),
            "Y_upper": self.Y_upper.tolist(),
            **integer_settings,
            **{k: float(v) for k, v in float_settings.items()},
            "numerical_compatibility": "pycma_owned_scalar_decoder_v1_halfwidth_std_cap",
            "numerical_acquisition": "negative_log_ei_released_branch"
            if acquisition == "EI"
            else "negative_ucb",
            "log_ei_floor": -1e100,
        }
        self.rng = np.random.RandomState(seed)
        self.A = self.rng.normal(size=(self.lower.size, embedding_dimension))
        self.Y = np.empty((0, embedding_dimension))
        self.X = np.empty((0, self.lower.size))
        self.quality = np.empty(0)
        self.lengthscale = self.initial_lengthscale
        self.exploitation_count = 0
        self.last_refit_observation: int | None = None
        self.pending_y: np.ndarray | None = None
        self.pending_action: np.ndarray | None = None
        self.pending_std: float | None = None
        self.pending_metadata: dict[str, Any] | None = None
        self.fit_history: list[dict[str, Any]] = []
        self.optimizer_history: list[dict[str, Any]] = []
        self.warning_history: list[dict[str, Any]] = []
        self.failed_operations: list[dict[str, Any]] = []
        self._L: np.ndarray | None = None
        self._alpha: np.ndarray | None = None
        self._factor_lengthscale: float | None = None
        self._factor_jitter = 0.0

    def _physical(self, y: np.ndarray) -> np.ndarray:
        unit = np.clip(self.A @ y, -1.0, 1.0)
        return self.lower + (unit + 1) * (self.upper - self.lower) / 2

    def _kernel(
        self, first: np.ndarray, second: np.ndarray, ell: float | None = None
    ) -> np.ndarray:
        scale = self.lengthscale if ell is None else ell
        return np.exp(-0.5 * cdist(first / scale, second / scale, metric="sqeuclidean"))

    def _warn(self, operation: str, message: str, *, jitter: float | None = None) -> None:
        self.warning_history.append(
            {
                "operation": operation,
                "message": message,
                "observations": len(self.quality),
                "jitter": jitter,
            }
        )
        category = BasatNumericalWarning if jitter is None else BasatJitterWarning
        warnings.warn(f"Basat REMBO ({operation}): {message}", category, stacklevel=3)

    def _cholesky(self, covariance: np.ndarray, operation: str) -> tuple[np.ndarray, float]:
        try:
            return cholesky(covariance, lower=True, check_finite=False), 0.0
        except np.linalg.LinAlgError:
            pass
        jitter = np.finfo(float).eps * max(1.0, float(np.max(np.diag(covariance))))
        for _ in range(9):
            self._warn(
                operation,
                f"adding diagonal jitter {jitter:.17g} to {len(covariance)}x{len(covariance)} GP covariance",
                jitter=jitter,
            )
            try:
                return cholesky(
                    covariance + jitter * np.eye(len(covariance)), lower=True, check_finite=False
                ), jitter
            except np.linalg.LinAlgError:
                jitter *= 10
        raise np.linalg.LinAlgError(
            "REMBO covariance remained non-positive definite after reported jitter attempts."
        )

    def _ensure_factor(self) -> None:
        n = len(self.quality)
        if (
            self._L is not None
            and len(self._L) == n
            and self._factor_lengthscale == self.lengthscale
        ):
            return
        if not n:
            self._L = self._alpha = None
            return
        K = self._kernel(self.Y, self.Y) + self.noise_variance * np.eye(n)
        self._L, self._factor_jitter = self._cholesky(K, "conditioning")
        self._alpha = cho_solve((self._L, True), self.quality, check_finite=False)
        self._factor_lengthscale = self.lengthscale

    def latent_posterior(self, Y: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return latent-function posterior mean and variance (noise excluded)."""
        points = np.atleast_2d(np.asarray(Y, float))
        if (
            points.ndim != 2
            or points.shape[1] != self.embedding_dimension
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("Posterior points must be a finite latent-coordinate matrix.")
        if not len(self.quality):
            return np.zeros(len(points)), np.ones(len(points))
        self._ensure_factor()
        cross = self._kernel(self.Y, points)
        v = solve_triangular(self._L, cross, lower=True, check_finite=False)
        mean = cross.T @ self._alpha
        variance = 1.0 - np.sum(v * v, axis=0)
        if np.any(variance < -1e-10):
            raise FloatingPointError("REMBO posterior has materially negative variance.")
        return mean, np.maximum(variance, 0.0)

    def acquisition_values(self, Y: Any) -> np.ndarray:
        """Return log EI or UCB; higher values are preferred.

        The released optional EI branch explicitly minimizes negative log EI.
        This numerical convention matters: DIRECT's subdivision and CMA's
        value-based stopping thresholds are not monotone-transform invariant.
        """
        if not len(self.quality):
            raise ValueError("Acquisition requires at least one observation.")
        mean, variance = self.latent_posterior(Y)
        if self.acquisition == "EI":
            values = log_expected_improvement(
                mean, variance, float(np.max(self.quality)) + self.ei_xi
            )
            return np.where(np.isneginf(values), -1e100, values)
        n = len(self.quality)
        beta = 2 * ((self.embedding_dimension / 2 + 2) * math.log(n) + math.log(math.pi**2 / 0.3))
        return mean + math.sqrt(beta) * np.sqrt(variance)

    def _likelihood(self, log_lengthscale: np.ndarray) -> float:
        ell = float(np.exp(log_lengthscale[0]))
        K = self._kernel(self.Y, self.Y, ell) + self.noise_variance * np.eye(len(self.Y))
        L, _ = self._cholesky(K, "hyperparameter likelihood")
        alpha = cho_solve((L, True), self.quality, check_finite=False)
        # The released optimizer uses twice NLL with the constant omitted.
        return float(self.quality @ alpha + 2 * np.log(np.diag(L)).sum())

    def _optimize(
        self, function: Any, lower: np.ndarray, upper: np.ndarray, *, stage: str
    ) -> np.ndarray:
        """Compare the best genuinely evaluated DIRECT and CMA candidates."""
        direct_budget = (
            self.acquisition_direct_evaluations
            if stage == "acquisition"
            else self.hyperparameter_direct_evaluations
        )
        direct_iterations = (
            self.acquisition_direct_iterations
            if stage == "acquisition"
            else self.hyperparameter_direct_iterations
        )
        winners: list[tuple[float, np.ndarray]] = []
        for method in ("DIRECT", "CMA"):
            record: dict[str, Any] = {
                "stage": stage,
                "method": method,
                "observations": len(self.quality),
                "nfev": 0,
                "nit": None,
                "status": "running",
                "max_evaluations": direct_budget if method == "DIRECT" else self.cma_evaluations,
                "max_iterations": direct_iterations if method == "DIRECT" else None,
            }
            best_value = math.inf
            best_point: np.ndarray | None = None

            def counted(point: Any, method: str = method, record: dict[str, Any] = record) -> float:
                nonlocal best_value, best_point
                x = np.asarray(point, float)
                if (
                    x.shape != lower.shape
                    or np.any(x < lower)
                    or np.any(x > upper)
                    or not np.all(np.isfinite(x))
                ):
                    raise FloatingPointError(f"{method} supplied an invalid optimizer point.")
                record["nfev"] += 1
                value = float(function(x))
                if not math.isfinite(value):
                    raise FloatingPointError(f"{stage} objective is not finite.")
                if value < best_value:
                    best_value, best_point = value, x.copy()
                return value

            captured: list[Any] = []
            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    if method == "DIRECT":
                        result = direct(
                            counted,
                            list(zip(lower, upper, strict=True)),
                            locally_biased=False,
                            maxfun=direct_budget,
                            maxiter=direct_iterations,
                        )
                        record.update(
                            nit=int(result.nit),
                            status=int(result.status),
                            message=str(result.message),
                        )
                        if result.status < 0:
                            raise RuntimeError(f"DIRECT failed: {result.message}")
                    else:
                        self._run_cma(counted, lower, upper, record)
                if best_point is None:
                    raise RuntimeError(f"{method} did not evaluate any candidate.")
                record["best_value"] = best_value
                winners.append((best_value, best_point))
            except BaseException as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                self.optimizer_history.append(record)
                for warning in captured:
                    if issubclass(warning.category, (BasatJitterWarning, BasatNumericalWarning)):
                        warnings.warn(str(warning.message), warning.category, stacklevel=3)
                    else:
                        self._warn(
                            f"{stage} {method}", f"{warning.category.__name__}: {warning.message}"
                        )
        return min(winners, key=lambda pair: pair[0])[1].copy()

    def _run_cma(
        self, counted: Any, lower: np.ndarray, upper: np.ndarray, record: dict[str, Any]
    ) -> None:
        import cma

        class ScalarDecoding(cma.transformations.DiagonalDecoding):
            """Repair pycma 4.4.4's size-one rejection, on this instance only."""

            scaling: Any

            def set_i(self, index: int, value: float) -> Any:
                if np.size(self.scaling) == 1 and index == 0:
                    self.scaling = np.atleast_1d(self.scaling)
                    self.scaling[0] = value
                    self.is_identity = False
                    return self
                return super().set_i(index, value)

        class ScalarSafeStrategy(cma.CMAEvolutionStrategy):
            sigma_vec: Any

            def _stds_into_limits(self, warn: bool = False) -> Any:
                if self.N == 1 and not isinstance(self.sigma_vec, ScalarDecoding):
                    decoder = ScalarDecoding(self.sigma_vec.scaling)
                    decoder.__dict__.update(deepcopy(self.sigma_vec.__dict__))
                    self.sigma_vec = decoder
                return super()._stds_into_limits(warn=warn)

        span = upper - lower
        x0 = self.rng.uniform(lower, upper)
        options = {
            "bounds": [lower.tolist(), upper.tolist()],
            "maxfevals": self.cma_evaluations,
            "tolfun": self.cma_tolfun,
            "verbose": -2,
            "verb_disp": 0,
            "verb_log": 0,
            "seed": np.nan,
            "randn": self.rng.randn,
            "CMA_stds": (span / np.max(span)).tolist(),
            "maxstd": (span / 2).tolist(),
        }
        # Keep the source half-width standard-deviation cap and all parent CMA
        # equations. No global class monkeypatch, dummy dimension, or fallback.
        optimizer = ScalarSafeStrategy(x0.tolist(), float(np.max(span) / 2), options=options)
        while not optimizer.stop():
            points, values = optimizer.ask_and_eval(counted)
            optimizer.tell(points, values)
        # Match pycma fmin's explicit final-mean evaluation; it can exceed the
        # nominal generation budget, so actual callback counts are retained.
        if optimizer.opts["eval_final_mean"]:
            mean = optimizer.to_phenotype(
                optimizer.mean,
                copy=True,
                into_bounds=optimizer.boundary_handler.repair,
                archive=optimizer.sent_solutions,
            )
            counted(mean)
        record.update(
            nit=int(optimizer.countiter),
            status="completed",
            termination={str(k): str(v) for k, v in optimizer.stop().items()},
            backend_version=cma.__version__,
            scalar_decoder_repair=bool(len(lower) == 1),
            maxstd=(span / 2).tolist(),
        )

    def _record_failure(
        self, snapshot: dict[str, Any], operation: str, error: BaseException
    ) -> None:
        attempts = deepcopy(self.optimizer_history[len(snapshot["optimizer_history"]) :])
        emitted = deepcopy(self.warning_history[len(snapshot["warning_history"]) :])
        self.__dict__.clear()
        self.__dict__.update(snapshot)
        self.failed_operations.append(
            {
                "operation": operation,
                "observations": len(self.quality),
                "error": f"{type(error).__name__}: {error}",
                "optimizer_attempts": attempts,
                "warnings": emitted,
            }
        )

    def propose(self) -> np.ndarray:
        """Return an idempotent pending physical point without evaluating it."""
        if self.pending_action is not None:
            return self.pending_action.copy()
        snapshot = deepcopy(self.__dict__)
        try:
            start = len(self.optimizer_history)
            if not len(self.quality):
                point = np.zeros(self.embedding_dimension)
                std = None
            else:

                def objective(y: np.ndarray) -> float:
                    return -float(self.acquisition_values(y)[0])

                point = self._optimize(objective, self.Y_lower, self.Y_upper, stage="acquisition")
                std = float(np.sqrt(self.latent_posterior(point)[1][0]))
            self.pending_y = point.copy()
            self.pending_action = self._physical(point)
            self.pending_std = std
            self.pending_metadata = {
                "training_observations": len(self.quality),
                "lengthscale": self.lengthscale,
                "lengthscale_upper": self.lengthscale_upper,
                "exploitation_count": self.exploitation_count,
                "optimizer_records": deepcopy(self.optimizer_history[start:]),
            }
            return self.pending_action.copy()
        except BaseException as error:
            self._record_failure(snapshot, "propose", error)
            raise

    def _refit(self, reason: str) -> None:
        previous = self.lengthscale
        start = len(self.optimizer_history)
        if self.lengthscale_upper == self.lengthscale_lower:
            self.lengthscale = self.lengthscale_lower
        else:
            point = self._optimize(
                self._likelihood,
                np.log([self.lengthscale_lower]),
                np.log([self.lengthscale_upper]),
                stage="hyperparameter",
            )
            self.lengthscale = float(
                np.clip(np.exp(point[0]), self.lengthscale_lower, self.lengthscale_upper)
            )
        self._L = self._alpha = None
        self._ensure_factor()
        self.last_refit_observation = len(self.quality)
        self.fit_history.append(
            {
                "observations": len(self.quality),
                "reason": reason,
                "previous_lengthscale": previous,
                "lengthscale": self.lengthscale,
                "lengthscale_upper": self.lengthscale_upper,
                "optimizer_records": deepcopy(self.optimizer_history[start:]),
            }
        )

    def observe(self, action: Any, quality: float) -> None:
        """Commit one pending observation, rolling back on a failed post-fit.

        A retry must supply the same pending action/reward after a failure; the
        numerical RNG and all model state are restored before the error escapes.
        """
        x = np.asarray(action, float)
        reward = float(quality)
        if self.pending_action is None or not np.array_equal(x, self.pending_action):
            raise ValueError("Observation must match the outstanding REMBO proposal exactly.")
        if self.pending_y is None:
            raise ValueError("Pending REMBO proposal has no latent coordinates.")
        if not math.isfinite(reward):
            raise ValueError("Observation quality must be finite.")
        snapshot = deepcopy(self.__dict__)
        try:
            previous_n = len(self.quality)
            old_Y = self.Y
            self.Y = np.vstack([self.Y, self.pending_y])
            self.X = np.vstack([self.X, x])
            self.quality = np.append(self.quality, reward)
            if self.pending_std is not None:
                self.exploitation_count = (
                    self.exploitation_count + 1 if self.pending_std < self.t_sigma else 0
                )
            threshold_fit = self.exploitation_count >= self.exploitation_count_limit
            scheduled_fit = previous_n > 0 and previous_n % self.refit_interval == 0
            if threshold_fit:
                self.lengthscale_upper = max(
                    self.upper_bound_multiplier * self.lengthscale, self.lengthscale_lower
                )
                self.exploitation_count = 0
            if scheduled_fit or threshold_fit:
                self._refit(
                    "scheduled+low_variance"
                    if scheduled_fit and threshold_fit
                    else "scheduled"
                    if scheduled_fit
                    else "low_variance"
                )
            elif (
                self._L is not None
                and len(self._L) == previous_n
                and self._factor_lengthscale == self.lengthscale
            ):
                cross = self._kernel(old_Y, self.Y[-1:]).ravel()
                v = solve_triangular(self._L, cross, lower=True, check_finite=False)
                diagonal = 1.0 + self.noise_variance + self._factor_jitter - float(v @ v)
                if diagonal <= 0:
                    self._warn(
                        "rank-one conditioning",
                        "non-positive append pivot; rebuilding the same GP covariance",
                    )
                    self._L = self._alpha = None
                    self._ensure_factor()
                else:
                    L = np.zeros((previous_n + 1, previous_n + 1))
                    L[:-1, :-1], L[-1, :-1], L[-1, -1] = self._L, v, math.sqrt(diagonal)
                    self._L = L
                    self._alpha = cho_solve((L, True), self.quality, check_finite=False)
            else:
                self._L = self._alpha = None
            self.pending_y = self.pending_action = None
            self.pending_std = None
            self.pending_metadata = None
        except BaseException as error:
            self._record_failure(snapshot, "observe", error)
            raise

    def diagnostics(self) -> dict[str, Any]:
        """Return a detached, non-fitting record of numerical work and settings."""
        return deepcopy(
            {
                "policy_revision": self.revision,
                "config": self.config,
                "observations": len(self.quality),
                "lengthscale": self.lengthscale,
                "lengthscale_upper": self.lengthscale_upper,
                "exploitation_count": self.exploitation_count,
                "last_refit_observation": self.last_refit_observation,
                "pending_std": self.pending_std,
                "fit_history": self.fit_history,
                "optimizer_history": self.optimizer_history,
                "optimizer_evaluations": sum(record["nfev"] for record in self.optimizer_history),
                "failed_optimizer_evaluations": sum(
                    record["nfev"]
                    for attempt in self.failed_operations
                    for record in attempt["optimizer_attempts"]
                ),
                "warnings": self.warning_history,
                "failed_operations": self.failed_operations,
            }
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize exact RNG, pending work and GP factors without fitting."""
        rng: Any = self.rng.get_state()
        state = self.diagnostics()
        state.update(
            {
                "A": self.A.tolist(),
                "Y": self.Y.tolist(),
                "X": self.X.tolist(),
                "quality": self.quality.tolist(),
                "pending_y": None if self.pending_y is None else self.pending_y.tolist(),
                "pending_action": None
                if self.pending_action is None
                else self.pending_action.tolist(),
                "pending_metadata": deepcopy(self.pending_metadata),
                "rng": [rng[0], rng[1].tolist(), int(rng[2]), int(rng[3]), float(rng[4])],
                "factor": None if self._L is None else self._L.tolist(),
                "alpha": None if self._alpha is None else self._alpha.tolist(),
                "factor_lengthscale": self._factor_lengthscale,
                "factor_jitter": self._factor_jitter,
                "factor_sha256": self._factor_digest(),
            }
        )
        return state

    def _factor_digest(self) -> str | None:
        """Integrity of saved numerical bytes, independent of conditioning."""
        if self._L is None or self._alpha is None:
            return None
        digest = sha256()
        for value in (
            self._L,
            self._alpha,
            np.asarray([self._factor_lengthscale, self._factor_jitter], float),
        ):
            array = np.asarray(value, dtype="<f8", order="C")
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Atomically validate and restore a checkpoint; no model fitting occurs."""
        state = deepcopy(state)
        if state.get("policy_revision") != self.revision or state.get("config") != self.config:
            raise ValueError("REMBO policy revision/configuration does not match the checkpoint.")
        candidate = deepcopy(self)
        n = state.get("observations")
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("Invalid REMBO observation count.")
        shapes = {
            "A": (self.lower.size, self.embedding_dimension),
            "Y": (n, self.embedding_dimension),
            "X": (n, self.lower.size),
            "quality": (n,),
        }
        for name, shape in shapes.items():
            array = np.asarray(state[name], float)
            if n == 0 and name in {"Y", "X"} and array.size == 0:
                array = array.reshape(shape)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"Invalid REMBO checkpoint {name}.")
            setattr(candidate, name, array.copy())
        if np.any(self.Y_lower > candidate.Y) or np.any(self.Y_upper < candidate.Y):
            raise ValueError("Checkpoint contains out-of-bounds latent observations.")
        if n and not np.allclose(
            candidate.X, np.array([candidate._physical(y) for y in candidate.Y]), rtol=0, atol=1e-13
        ):
            raise ValueError("Physical actions do not match checkpoint latent actions/projection.")
        for name in ("lengthscale", "lengthscale_upper", "factor_jitter"):
            value = float(state[name])
            if (
                not math.isfinite(value)
                or (name != "factor_jitter" and value <= 0)
                or (name == "factor_jitter" and value < 0)
            ):
                raise ValueError(f"Invalid REMBO checkpoint {name}.")
            setattr(candidate, "_factor_jitter" if name == "factor_jitter" else name, value)
        if (
            not self.lengthscale_lower
            <= candidate.lengthscale
            <= candidate.lengthscale_upper
            <= self.config["lengthscale_upper"]
        ):
            raise ValueError("Checkpoint lengthscale bounds are inconsistent.")
        count = state["exploitation_count"]
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 0 <= count < self.exploitation_count_limit
        ):
            raise ValueError("Invalid REMBO exploitation counter.")
        candidate.exploitation_count = count
        fitted = state["last_refit_observation"]
        if fitted is not None and (
            isinstance(fitted, bool) or not isinstance(fitted, int) or not 1 <= fitted <= n
        ):
            raise ValueError("Invalid REMBO fit boundary.")
        candidate.last_refit_observation = fitted
        for name in ("fit_history", "optimizer_history", "warnings", "failed_operations"):
            if not isinstance(state[name], list):
                raise ValueError(f"Invalid REMBO checkpoint {name}.")
            setattr(candidate, "warning_history" if name == "warnings" else name, state[name])
        pending_y, pending_action = state["pending_y"], state["pending_action"]
        if (pending_y is None) != (pending_action is None):
            raise ValueError("Incomplete REMBO pending state.")
        candidate.pending_y = None if pending_y is None else np.asarray(pending_y, float)
        candidate.pending_action = (
            None if pending_action is None else np.asarray(pending_action, float)
        )
        candidate.pending_std = state["pending_std"]
        candidate.pending_metadata = state["pending_metadata"]
        if pending_y is not None:
            assert candidate.pending_y is not None and candidate.pending_action is not None
            if (
                candidate.pending_y.shape != (self.embedding_dimension,)
                or not np.all(np.isfinite(candidate.pending_y))
                or np.any(candidate.pending_y < self.Y_lower)
                or np.any(candidate.pending_y > self.Y_upper)
                or candidate.pending_action.shape != self.lower.shape
                or not np.array_equal(
                    candidate._physical(candidate.pending_y), candidate.pending_action
                )
            ):
                raise ValueError("Invalid REMBO pending coordinates.")
            if (n > 0 and candidate.pending_std is None) or (
                candidate.pending_std is not None
                and (not math.isfinite(candidate.pending_std) or candidate.pending_std < 0)
            ):
                raise ValueError("Invalid REMBO pending predictive standard deviation.")
            if not isinstance(candidate.pending_metadata, dict):
                raise ValueError("Invalid REMBO pending metadata.")
        elif candidate.pending_std is not None or candidate.pending_metadata is not None:
            raise ValueError("Orphaned REMBO pending metadata.")
        rng = state["rng"]
        candidate.rng = np.random.RandomState()
        candidate.rng.set_state((rng[0], np.asarray(rng[1], np.uint32), rng[2], rng[3], rng[4]))
        candidate._L = None if state["factor"] is None else np.asarray(state["factor"], float)
        candidate._alpha = None if state["alpha"] is None else np.asarray(state["alpha"], float)
        candidate._factor_lengthscale = state["factor_lengthscale"]
        if (candidate._L is None) != (candidate._alpha is None):
            raise ValueError("Incomplete REMBO cached factor.")
        if state.get("factor_sha256") != candidate._factor_digest():
            raise ValueError("REMBO cached factor/solve integrity hash does not match.")
        if candidate._L is not None:
            assert candidate._alpha is not None
            if (
                candidate._L.shape != (n, n)
                or candidate._alpha.shape != (n,)
                or not np.all(np.isfinite(candidate._L))
                or not np.all(np.isfinite(candidate._alpha))
                or np.any(np.diag(candidate._L) <= 0)
                or np.any(np.triu(candidate._L, 1) != 0)
                or candidate._factor_lengthscale != candidate.lengthscale
            ):
                raise ValueError("Invalid REMBO cached factor/parameters.")
            # O(n^2) triangular checks; no cubic refactorization or optimization.
            restored_y = candidate._L @ (candidate._L.T @ candidate._alpha)
            residual = float(np.linalg.norm(restored_y - candidate.quality, ord=np.inf))
            scale = float(
                np.linalg.norm(candidate._L, ord=np.inf)
                * np.linalg.norm(candidate._L.T, ord=np.inf)
                * np.linalg.norm(candidate._alpha, ord=np.inf)
                + np.linalg.norm(candidate.quality, ord=np.inf)
            )
            tolerance = 50 * max(n, 1) * np.finfo(float).eps * max(scale, np.finfo(float).tiny)
            if not math.isfinite(residual) or residual > tolerance:
                raise ValueError("Cached REMBO solve is inconsistent with observations.")
        self.__dict__.clear()
        self.__dict__.update(candidate.__dict__)
