"""Basat-owned sequential Add-GP-UCB source-policy state machine.

The numerical fitter is an attributed internal implementation. This module
never imports Dragonfly, starts an algorithm process, or sees an objective.
It receives observations only from Basat's ordinary ask/tell boundary.
"""

from __future__ import annotations

from argparse import Namespace
from copy import deepcopy

import numpy as np

from . import _additive_numerics as numeric


class AdditivePolicy:
    """Own action/reward history, RNG, fitting and component-UCB optimization.

    This private protocol reproduces the continuous Dragonfly-0.1.4 settings
    used by the ALEBO comparator. The source's asynchronous accounting is
    deliberately explicit: selecting a posterior draw precedes incorporation
    of the last returned observation. No benchmark or external runner is used.
    """

    revision = "basat-native-alebo-additive-v1"

    def __init__(self, bounds, seed, max_capital=49):
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("Additive policy bounds must have shape (D, 2).")
        self.dimension = self.bounds.shape[0]
        self.rng = np.random.RandomState(seed)
        self.cache = numeric.make_exact_kernel_cache()
        self.max_capital = max_capital
        self.init_capital = np.clip(
            5 * self.dimension,
            max(5.0, 0.025 * max_capital),
            max(5.0, 0.075 * max_capital),
        )
        self.initial_count = int(np.ceil(self.init_capital))
        self.pool_size = int(self.init_capital)
        self.initial_pool = []
        self.actions = []
        self.observations = []
        self.received = 0
        self.step = 0
        self.last_build = 0
        self.primary_fitter = None
        self.additive_fitter = None
        self.primary_gp = None
        self.additive_gp = None
        self.primary_fit_type = None
        self.additive_fit_type = None
        self.pending = None
        self.pending_unit = None
        self.build_rows = []

    def _draw_pool(self):
        self.initial_pool = list(numeric.latin_hc_sampling(self.dimension, self.pool_size))

    def _options(self, additive):
        return Namespace(
            hp_tune_criterion="ml-post_sampling",
            hp_tune_probs="0.3-0.7",
            ml_hp_tune_opt="default",
            post_hp_tune_method="slice",
            post_hp_tune_burn=-1,
            post_hp_tune_offset=25,
            mean_func=None,
            use_additive_gp=additive,
            rand_exp_sampling_replace=False,
        )

    def _build(self):
        x, y = self.actions[: self.received], self.observations[: self.received]
        self.primary_fitter = numeric.EuclideanGPFitter(x, y, self._options(False))
        self.primary_fitter.fit_gp_for_gp_bandit(17)
        self.additive_fitter = numeric.EuclideanGPFitter(x, y, self._options(True))
        self.additive_fitter.fit_gp_for_gp_bandit(17)
        self.primary_gp = self.additive_gp = None
        self.primary_fit_type = self.additive_fit_type = None
        self.build_rows.append(self.received)

    def _select_models(self):
        assert self.primary_fitter is not None and self.additive_fitter is not None
        primary = self.primary_fitter.get_next_gp()
        self.primary_fit_type, self.primary_method, self.primary_gp = primary
        additive = self.additive_fitter.get_next_gp()
        self.additive_fit_type, self.additive_method, self.additive_gp = additive
        x, y = self.actions[: self.received], self.observations[: self.received]
        # Source selects the additive GP, sets its data, then sets primary data.
        if self.additive_fit_type.endswith("sample_hps_with_probs"):
            self.additive_gp.set_data(x, y, build_posterior=True)
        if self.primary_fit_type.endswith("sample_hps_with_probs"):
            self.primary_gp.set_data(x, y, build_posterior=True)

    def _receive(self):
        if self.received >= len(self.observations):
            return
        x = self.actions[self.received :]
        y = self.observations[self.received :]
        self.received = len(self.observations)
        if self.primary_fit_type == "fitted_gp":
            assert self.primary_gp is not None
            self.primary_gp.add_data_multiple(x, y)
        if self.additive_fit_type == "fitted_gp":
            assert self.additive_gp is not None
            self.additive_gp.add_data_multiple(x, y)

    def _component_ucb(self):
        gp = self.additive_gp
        assert gp is not None
        kernel = gp.kernel
        # Source's choice still consumes one uniform draw for its singleton set.
        numeric.np.random.choice(["add_ucb"], p=np.array([1.0]))
        if self.dimension > 60:
            max_evals = np.clip(
                2 * min(5, self.dimension) ** 2 * np.sqrt(min(self.step, 1000)), 2000, 6e4
            )
        else:
            max_evals = np.clip(
                min(5, self.dimension) ** 2 * np.sqrt(min(self.step, 1000)), 1000, 3e4
            )
        max_evals //= len(kernel.kernel_list)
        scale = kernel.hyperparams["scale"]
        result = np.zeros(self.dimension)
        train = np.array(gp.X)
        for group, component in zip(kernel.groupings, kernel.kernel_list, strict=True):

            def acquisition(point, group=group, component=component):
                test = point.reshape((1, -1))
                beta = np.sqrt(0.2 * len(group) * np.log(2 * len(group) * self.step + 1))
                cross = scale * component(test, train[:, group])
                mean = cross.dot(gp.alpha) + np.array([0] * len(test))
                prior = scale * component(test, test)
                solved = numeric.solve_lower_triangular(gp.L, cross.T)
                covariance = prior - solved.T.dot(solved)
                covariance = numeric.get_post_covar_from_raw_covar(
                    covariance, gp.noise_var, kernel.is_guaranteed_psd()
                )
                return mean + beta * np.sqrt(np.diag(covariance))

            _, point, _ = numeric.pdoo_maximise(
                acquisition, np.array([[0.0, 1.0]] * len(group)), max_evals
            )
            result[group] = point
        return result

    def propose(self):
        """Return one physical-domain action without evaluating an objective."""
        if self.pending is not None:
            return self.pending.copy()
        if len(self.observations) >= max(self.initial_count, self.max_capital + 1):
            raise RuntimeError("The configured additive comparison budget is complete.")
        with numeric.local_random_state(self.rng, self.cache):
            if len(self.observations) < self.initial_count:
                if not self.initial_pool:
                    self._draw_pool()
                unit = self.initial_pool.pop(0)
                self.step += 1
            else:
                if self.primary_fitter is None:
                    # Source draws and discards a pool entry before discovering
                    # that the initialization capital has been exhausted.
                    if not self.initial_pool:
                        self._draw_pool()
                    self.initial_pool.pop(0)
                    self.step += 1
                    self.received = len(self.observations)
                    self._build()
                self._select_models()
                self._receive()
                unit = self._component_ucb()
            self.pending_unit = np.asarray(unit).copy()
            self.pending = numeric.map_to_bounds(unit, self.bounds)
        return self.pending.copy()

    def observe(self, value):
        """Record one returned quality value and update source scheduling."""
        if self.pending is None:
            raise RuntimeError("An observation requires a pending additive proposal.")
        assert self.pending_unit is not None
        # Preserve the optimizer's exact unit point: normalizing the physical
        # action back introduces avoidable floating-point round-trip drift.
        self.actions.append(self.pending_unit.copy())
        self.observations.append(float(value))
        self.pending = None
        self.pending_unit = None
        with numeric.local_random_state(self.rng, self.cache):
            if len(self.observations) > self.initial_count:
                self.step += 1
                if self.step - self.last_build >= 17:
                    self.last_build = self.step
                    self._build()
            if len(self.observations) >= self.max_capital + 1:
                self._receive()

    def progress(self):
        """Return portable progress; fitted GP objects are not serialized."""
        return {
            "revision": self.revision,
            "actions": np.asarray(self.actions).tolist(),
            "observations": list(self.observations),
            "pending": None if self.pending is None else self.pending.tolist(),
            "build_rows": list(self.build_rows),
            "rng_state": deepcopy(self.rng.get_state()),
        }
