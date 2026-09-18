"""Opt-in numerical compatibility with the Cai et al. released code.

Source: caitree/GoodAction ff0ffe18a72e5eef824254ab630929708dd80455,
GP.py, GPBO.py and BO_methods.py. Adapted under the MIT license in
``_good_action_released_LICENSE.txt``. No external BO runner is imported.
The model receives only bounds and observations; it never sees an objective.

This deliberately retains discarded fitted parameters, conditional means
with prior-covariance draws, mixed precision, and resampled local objectives.
It is a released-code diagnostic, NOT a conditional-GP mathematical sampler.
"""

from __future__ import annotations

import copy
import threading
import warnings
from dataclasses import asdict
from typing import Any

import numpy as np
import scipy.linalg
import torch
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from torch.quasirandom import SobolEngine

from basat.models.GaussianProcess import GaussianProcess, GaussianProcessConfig, ModelStateError
from basat.models.GoodActionGaussianProcess import GoodActionGPConfig
from basat.models.utils import Bounds
from basat.warnings import BasatAcquisitionWarning, BasatNumericalWarning

_TORCH_STREAM_LOCK = threading.RLock()
_REVISION = "ff0ffe18a72e5eef824254ab630929708dd80455"


class ReleasedGoodActionModel(GaussianProcess):
    """Private observation-only source arithmetic and serialized random streams.

    ``fit``/``predict``/result inspection never advance proposal streams.
    Discarded-fit random starting points are consumed in ``propose`` only.
    The source's exact selected unit coordinates survive physical conversion,
    feedback and checkpoints; recomputing them would introduce rounding drift.
    """

    def __init__(self, config: GoodActionGPConfig, *, bounds: Bounds, seed: int):
        super().__init__(GaussianProcessConfig(
            mean="zero", noise_mode="fixed", noise_variance=config.regularizer,
            standardize_outcomes=True, dtype="float64", device="cpu"),
            input_dimension=bounds.dimension)
        self.source_config = config
        self.bounds = bounds.to(dtype=torch.float64, device="cpu")
        self.seed = seed
        self.dimension = bounds.dimension
        self.scaler = MinMaxScaler().fit(np.stack((
            self.bounds.lower.numpy(), self.bounds.upper.numpy())))
        self.reset()

    def reset(self) -> None:
        self.X_train = None
        self.y_train = None
        self.observation_noise_variance = None
        self.model = None
        self.is_fit = False
        self.unit_X = np.empty((0, self.dimension), dtype=np.float64)
        self.pending_unit: np.ndarray | None = None
        self.pending_action: np.ndarray | None = None
        self.initial_design: np.ndarray | None = None
        self.rng = np.random.RandomState(self.seed)
        self.torch_state = torch.Generator(device="cpu").manual_seed(self.seed).get_state()
        self.last_fit_diagnostics = {}
        self.last_decision: dict[str, Any] = {}

    def set_data(self, X: Any, y: Any, noise_variance: Any = None) -> None:
        values = torch.as_tensor(X, dtype=torch.float64, device="cpu").detach().clone()
        rewards = torch.as_tensor(y, dtype=torch.float64, device="cpu").detach().clone().reshape(-1)
        if (values.ndim != 2 or values.shape != (len(rewards), self.dimension)
                or not torch.isfinite(values).all() or not torch.isfinite(rewards).all()):
            raise ValueError("Released good-action data must be finite (N,D) actions and N rewards.")
        if not torch.all((values >= self.bounds.lower) & (values <= self.bounds.upper)):
            raise ValueError("Released good-action observations lie outside the bounds.")
        if noise_variance is not None and not torch.all(torch.as_tensor(noise_variance) == 0):
            raise ValueError("Released good-action compatibility currently requires noiseless rewards.")
        if self.pending_unit is not None:
            n = 0 if self.X_train is None else len(self.X_train)
            prefix_matches = self.X_train is None or torch.equal(values[:n], self.X_train)
            assert self.pending_action is not None
            if (len(values) != n + 1 or not prefix_matches
                    or not np.array_equal(values[-1].numpy(), self.pending_action)):
                raise ValueError("Released good-action feedback must append the exact proposed action.")
            unit = np.vstack((self.unit_X, self.pending_unit))
        elif self.X_train is not None and torch.equal(values, self.X_train):
            unit = self.unit_X.copy()
        else:
            unit = self.scaler.transform(values.numpy()) if len(values) else self.unit_X[:0].copy()
        self.X_train, self.y_train = values, rewards
        self.unit_X = unit
        self.pending_unit = self.pending_action = None
        self.is_fit = False

    def covariance(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        # sklearn preserves float32 when both arguments are float32. The
        # distance/sqrt/square order is intentional source arithmetic.
        distance = euclidean_distances(x, z)
        return self.source_config.initial_amplitude * np.exp(
            -np.square(distance) / self.source_config.initial_bandwidth)

    def condition_only(self) -> dict[str, Any]:
        if self.is_fit:
            return dict(self.last_fit_diagnostics)
        if self.y_train is None or len(self.y_train) < 2:
            raise ValueError("Released good-action inference needs at least two observations.")
        y = self.y_train.numpy()
        self.outcome_mean, self.outcome_std = float(np.mean(y)), float(np.std(y))
        if self.outcome_std == 0:
            raise ValueError("Released good-action standardization is undefined for constant rewards.")
        scaled = (y - self.outcome_mean) / self.outcome_std
        # The source preserves the first exact duplicate, in original order.
        _, first = np.unique(self.unit_X, axis=0, return_index=True)
        self.retained = np.isin(np.arange(len(y)), first)
        self.train_unit = self.unit_X[self.retained]
        retained_y = scaled[self.retained]
        self.retained_mean = float(np.mean(retained_y))
        self.retained_std = float(np.std(retained_y))
        if self.retained_std == 0:
            raise ValueError("Released GP retained unique rows have constant rewards.")
        self.targets = (retained_y - self.retained_mean) / self.retained_std
        matrix = self.covariance(self.train_unit, self.train_unit)
        matrix += np.eye(len(matrix)) * self.source_config.regularizer
        self.factor = scipy.linalg.cholesky(matrix, lower=True)
        temporary = np.linalg.solve(self.factor, self.targets)
        self.alpha = np.linalg.solve(self.factor.T, temporary)
        self.is_fit = True
        self.last_fit_diagnostics = {
            "method": "released_fixed_hyperparameters", "success": True,
            "training_rows": len(y), "deduplicated_training_rows": len(self.train_unit),
            "fitted_parameters_applied": False, "sampling_covariance": "prior",
            "source_revision": _REVISION, "regularizer": self.source_config.regularizer,
            "added_jitter": 0.0, "candidate_prior_dtype": "float32",
            "observation_dtype": "float64", "rng_policy": "isolated_shared_MT19937_and_Torch",
        }
        return dict(self.last_fit_diagnostics)

    def fit(self, *, maxiter: int = 200, **kwargs: Any) -> dict[str, Any]:
        del maxiter
        if kwargs:
            raise TypeError("Released fixed model has no additional fitting options.")
        return self.condition_only()

    def predict_unit(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.condition_only()
        points = np.asarray(points).reshape(-1, self.dimension)
        test = self.covariance(points, points) + np.eye(len(points)) * self.source_config.regularizer
        cross = self.covariance(points, self.train_unit)
        mean = np.dot(cross, self.alpha)
        solved = np.linalg.solve(self.factor, cross.T)
        variance = test - np.dot(solved.T, solved)
        return mean.reshape(-1, 1), np.diag(variance).reshape(-1, 1)

    def predict(self, X: Any, *, observation_noise: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        del observation_noise  # Released predict always includes its stated query regularizer.
        points = torch.as_tensor(X, dtype=torch.float64, device="cpu").detach().numpy()
        unit = self.scaler.transform(points.reshape(-1, self.dimension))
        mean, variance = self.predict_unit(unit)
        if np.any(variance < 0):
            warnings.warn("Basat released GP marginal variance is negative from rounding; "
                          "diagnostic standard deviations are floored at zero. No jitter was added.",
                          BasatNumericalWarning, stacklevel=2)
        scale = self.outcome_std * self.retained_std
        mean = self.outcome_mean + self.outcome_std * (self.retained_mean + self.retained_std * mean)
        return torch.from_numpy(mean.ravel()), torch.from_numpy(np.sqrt(np.maximum(variance, 0)).ravel() * scale)

    def sample_unit(self, points: np.ndarray, size: int) -> np.ndarray:
        mean, _ = self.predict_unit(points)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always", RuntimeWarning)
            values = self.rng.multivariate_normal(mean.ravel(), self.covariance(points, points), size)
        for record in records:
            warnings.warn(f"Basat released prior-covariance sample: {record.message}. "
                          "The released NumPy SVD rule was retained; no diagonal jitter was added.",
                          BasatNumericalWarning, stacklevel=2)
        return values.T[:, np.newaxis, :]

    def propose(self, method: str, threshold: float, initial_points: int) -> np.ndarray:
        n = 0 if self.y_train is None else len(self.y_train)
        if self.pending_unit is not None:
            raise ValueError("Released proposal already awaits feedback.")
        if n < initial_points:
            if self.initial_design is None:
                self.initial_design = self.rng.uniform(self.bounds.lower.numpy(),
                    self.bounds.upper.numpy(), (initial_points, self.dimension))
            action = self.initial_design[n].copy()
            self.pending_unit = self.scaler.transform(action.reshape(1, -1))[0]
            self.last_decision = {"branch": "initial_uniform", "restarts": 0, "objective_calls": 0}
        else:
            self.condition_only()
            if n % 3 == 0:
                # The source discards the fitted pair. Only the deterministic
                # fit's ten uniform starts affect later stochastic decisions.
                self.rng.uniform([.001, .05], [1., 1.5], (10, 2))
            epsilon = (threshold - self.outcome_mean) / self.outcome_std
            with _TORCH_STREAM_LOCK, torch.random.fork_rng(devices=[]):
                torch.set_rng_state(self.torch_state)
                try:
                    engine = SobolEngine(self.dimension, scramble=True)
                    unit = self._select(method, epsilon, engine)
                finally:
                    self.torch_state = torch.get_rng_state().clone()
            # Preserve dtype: the source's in-place inverse_transform can
            # return float32 actions for a float32 candidate.
            self.pending_unit = np.asarray(unit).copy()
            action = self.scaler.inverse_transform(np.reshape(unit, (1, -1)))[0]
        self.last_decision.update(source_unit_dtype=str(self.pending_unit.dtype),
                                  source_physical_action_dtype=str(np.asarray(action).dtype),
                                  returned_action_dtype="float64")
        self.pending_action = np.asarray(action, dtype=np.float64).copy()
        return self.pending_action.copy()

    def _select(self, method: str, epsilon: float, engine: SobolEngine) -> np.ndarray:
        unit_bounds = np.array([[0., 1.]] * self.dimension)
        report: dict[str, Any] = {"branch": method, "restarts": 0, "objective_calls": 0,
                                  "unsuccessful_restarts": 0, "sobol_blocks": []}
        self.last_decision = report

        def design(count: int) -> np.ndarray:
            report["sobol_blocks"].append(count)
            return engine.draw(count, dtype=torch.float32).cpu().numpy()

        def acquisition(x: np.ndarray) -> np.ndarray:
            if method == "sts":
                return self.sample_unit(x, 1)
            if method == "gs":
                return np.asarray((self.sample_unit(x, 100) >= epsilon).mean())
            mean, var = self.predict_unit(x)
            std = np.sqrt(var)
            difference = mean - epsilon
            z = difference / np.maximum(std, 1e-8)
            return z if method == "pg" else difference * norm.cdf(z) + std * norm.pdf(z)

        if method in {"gs", "sts"}:
            tries = design(100 * self.dimension)
            samples = self.sample_unit(tries, 1 if method == "sts" else 100)
            if method == "sts":
                good = np.where(samples >= epsilon)[0]
                report["sampled_good_count"] = len(good)
                if len(good):
                    report["branch"] = "sts_closest_good"
                    points = tries[good]
                    return points[np.argmin(np.linalg.norm(points - .5, axis=-1))]
                report["branch"] = "sts_fallback"
                tries = design(1000)
                values = acquisition(tries)
            elif np.all(samples < epsilon):
                report["branch"] = "gs_all_below"
                index, _, _ = np.unravel_index(samples.argmax(), samples.shape)
                return tries[index]
            else:
                report["branch"] = "gs_refinement"
                values = (samples >= epsilon).mean(axis=-1)
        else:
            tries = design(1000)
            values = acquisition(tries)
        best = tries[self.rng.choice(np.where(values == values.max())[0])]
        best_value = values.max()
        for initial in design(10):
            report["restarts"] += 1

            def objective(x: np.ndarray) -> float:
                report["objective_calls"] += 1
                return -float(acquisition(x.reshape(1, -1)).squeeze())

            with warnings.catch_warnings(record=True) as records:
                # The source's old (1,d) x0 is flattened for current SciPy,
                # preserving float32 dtype and all optimizer defaults.
                result = minimize(objective, initial.reshape(-1), bounds=unit_bounds,
                                  method="L-BFGS-B")
            for record in records:
                warnings.warn(f"Basat released acquisition refinement: {record.message}",
                              BasatNumericalWarning, stacklevel=2)
            if not result.success:
                report["unsuccessful_restarts"] += 1
                continue
            if -result.fun >= best_value:
                best, best_value = result.x, -result.fun
        report["score"] = float(best_value)
        if report["unsuccessful_restarts"]:
            warnings.warn("Basat released acquisition retained the source acceptance rule: "
                          f"{report['unsuccessful_restarts']} of 10 local restarts were unsuccessful. "
                          "The best accepted candidate is retained.",
                          BasatAcquisitionWarning, stacklevel=2)
        return best

    def state_dict(self) -> dict[str, Any]:
        numpy_state: Any = self.rng.get_state()
        return {
            "released_version": 1, "source_revision": _REVISION,
            "source_config": asdict(self.source_config), "seed": self.seed,
            "bounds": self.bounds.as_tensor().clone(),
            "X": None if self.X_train is None else self.X_train.clone(),
            "y": None if self.y_train is None else self.y_train.clone(),
            "unit_X": torch.from_numpy(self.unit_X.copy()),
            "pending_unit": None if self.pending_unit is None else torch.from_numpy(self.pending_unit.copy()),
            "pending_action": None if self.pending_action is None else torch.from_numpy(self.pending_action.copy()),
            "initial_design": None if self.initial_design is None else torch.from_numpy(self.initial_design.copy()),
            "numpy_state": {"kind": numpy_state[0], "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
                            "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4]},
            "torch_state": self.torch_state.clone(), "last_decision": copy.deepcopy(self.last_decision),
        }

    def load_state_dict(self, state: dict[str, Any], *, strict: bool = True) -> None:
        del strict
        try:
            if (state["released_version"] != 1 or state["source_revision"] != _REVISION
                    or state["source_config"] != asdict(self.source_config) or state["seed"] != self.seed
                    or not torch.equal(torch.as_tensor(state["bounds"]), self.bounds.as_tensor())):
                raise ValueError("Released model checkpoint identity differs.")
            # Build/validate in a disposable instance before replacing state.
            restored = ReleasedGoodActionModel(self.source_config, bounds=self.bounds, seed=self.seed)
            if state["X"] is not None:
                restored.set_data(state["X"], state["y"])
            unit = torch.as_tensor(state["unit_X"]).numpy().copy()
            n = 0 if restored.X_train is None else len(restored.X_train)
            if unit.shape != (n, self.dimension) or not np.isfinite(unit).all() or np.any((unit < 0) | (unit > 1)):
                raise ValueError("Invalid released unit-coordinate table.")
            if restored.X_train is not None and not np.allclose(self.scaler.inverse_transform(unit), restored.X_train.numpy(),
                                    rtol=0, atol=np.finfo(np.float32).eps * float((self.bounds.upper-self.bounds.lower).max()) * 2):
                raise ValueError("Released unit and physical observations disagree.")
            restored.unit_X = unit
            for name in ("pending_unit", "pending_action", "initial_design"):
                value = state[name]
                array = None if value is None else torch.as_tensor(value).numpy().copy()
                expected_shape = (3, self.dimension) if name == "initial_design" else (self.dimension,)
                if array is not None:
                    lower = 0.0 if name == "pending_unit" else self.bounds.lower.numpy()
                    upper = 1.0 if name == "pending_unit" else self.bounds.upper.numpy()
                    if (array.shape != expected_shape or not np.isfinite(array).all()
                            or np.any((array < lower) | (array > upper))):
                        raise ValueError(f"Invalid released {name}.")
                setattr(restored, name, array)
            if (restored.pending_unit is None) != (restored.pending_action is None):
                raise ValueError("Released pending proposal is incomplete.")
            if restored.pending_unit is not None:
                assert restored.pending_action is not None
                converted = self.scaler.inverse_transform(restored.pending_unit.reshape(1, -1))[0]
                if not np.allclose(converted, restored.pending_action, rtol=0,
                                   atol=np.finfo(np.float64).eps * float((self.bounds.upper-self.bounds.lower).max()) * 2):
                    raise ValueError("Released pending unit and physical actions disagree.")
            r = state["numpy_state"]
            keys = torch.as_tensor(r["keys"]).numpy()
            if (keys.shape != (624,) or not np.issubdtype(keys.dtype, np.integer)
                    or np.any((keys < 0) | (keys > 2**32-1))
                    or r["kind"] != "MT19937" or not isinstance(r["position"], int)
                    or not 0 <= r["position"] <= 624 or r["has_gauss"] not in {0, 1}
                    or not np.isfinite(r["cached_gaussian"])):
                raise ValueError("Invalid MT19937 keys.")
            restored.rng.set_state((r["kind"], keys.astype(np.uint32), r["position"],
                                   r["has_gauss"], r["cached_gaussian"]))
            generator = torch.Generator(device="cpu")
            generator.set_state(torch.as_tensor(state["torch_state"]))
            restored.torch_state = generator.get_state()
            restored.last_decision = copy.deepcopy(state["last_decision"])
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise ModelStateError(f"Invalid released good-action checkpoint: {error}") from error
        self.__dict__.update(restored.__dict__)
