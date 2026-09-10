"""Fully Bayesian Sparse Axis-Aligned Subspace Bayesian optimization.

SAASBO keeps full environment coordinates and induces sparsity through the
hierarchical half-Cauchy prior on inverse squared length scales.  This module
uses BoTorch's fully Bayesian GP representation and Pyro NUTS; it never
substitutes an MLE ARD model or a sparsity penalty for posterior sampling.
"""

from __future__ import annotations

import math
import random
import threading
import types
import warnings
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np
import torch

from bayopy.models.Acquisitions import DiscreteAcquisitionCandidates
from bayopy.models.utils import Bounds
from bayopy.warnings import translate_backend_warnings

from ..algorithm import Algorithm, resolve_batch_configuration
from ..helper import SourceRecord, to_quality
from .Exception.exception_SAASBO import (
    SAASBOCheckpointError,
    SAASBODivergenceError,
    SAASBOException,
    SAASBOMCMCError,
    SAASBOPriorError,
    SAASBOWarning,
)

_TORCH_NUM_THREADS_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class SAASBOConfig:
    """Configuration of the SAAS prior, NUTS, and averaged acquisition.

    Parameters
    ----------
    shrinkage_scale:
        Global half-Cauchy SAAS inverse-lengthscale prior scale.
    warmup_steps, posterior_samples, thinning, num_chains, max_tree_depth:
        Exact Pyro NUTS sampling controls.
    target_accept_prob:
        NUTS acceptance target in (0, 1). Larger values favor smaller integration
        steps and may reduce divergences at increased computational cost.
    initial_points:
        Optional initial-design count.
    initial_points_sampling_method:
        ``"paper"``, ``"sobol"``, or ``"uniform"`` initial-design sampling.
    acquisition, num_restarts, raw_samples, mc_samples, ucb_beta:
        Fully Bayesian EI/UCB and continuous optimization controls.
    max_r_hat, min_effective_sample_size, allow_divergences:
        Posterior diagnostic acceptance policy.  Set
        ``min_effective_sample_size=None`` only for explicitly non-inferential
        smoke runs that need to retain draws when a very short chain makes
        Pyro's ESS estimator unavailable or nonphysical. The runtime
        ``diagnostic_policy`` determines whether failed diagnostics warn or raise.
    batch_size, seed:
        Request size and checkpointed sampler/acquisition seed.
    """

    shrinkage_scale: float = 0.1
    warmup_steps: int = 512
    posterior_samples: int = 256
    thinning: int = 16
    num_chains: int = 1
    initial_points: int | None = None
    initial_points_sampling_method: Literal["paper", "sobol", "uniform"] = "paper"
    acquisition: Literal["EI", "UCB"] = "EI"
    max_tree_depth: int = 6
    target_accept_prob: float = field(default=0.8, kw_only=True)
    num_restarts: int = 10
    raw_samples: int = 256
    mc_samples: int = 128
    ucb_beta: float = 2.0
    max_r_hat: float = 1.1
    min_effective_sample_size: float | None = 1.0
    allow_divergences: bool = False
    batch_size: int | None = 1
    q_batch: int | None = None
    seed: int = 1


@dataclass(slots=True)
class SAASBOState:
    """Mutable training data, posterior draws, and sampler diagnostics.

    Attributes
    ----------
    X_train_unit, y_train:
        Normalized ``(N, D_env)`` inputs and raw learner-visible responses.
    posterior_hyperparameter_samples, mcmc_diagnostics:
        Retained Pyro draws and convergence evidence.
    posterior_valid, fit_generation:
        Cache-validity flag and completed NUTS fit count.
    outcome_mean, outcome_std:
        Response standardization used by the fully Bayesian model.
    """

    X_train_unit: torch.Tensor
    y_train: torch.Tensor
    posterior_hyperparameter_samples: dict[str, torch.Tensor] = field(default_factory=dict)
    mcmc_diagnostics: dict[str, Any] = field(default_factory=dict)
    posterior_valid: bool = False
    fit_generation: int = 0
    outcome_mean: torch.Tensor | None = None
    outcome_std: torch.Tensor | None = None


class SAASBO(Algorithm):
    """SAASBO with exact NUTS posterior draws and finite-row-safe acquisition.

    Install ``bayopy[mcmc]`` to fit this fully Bayesian model.

    Examples
    --------
    >>> import bayopy
    >>> bounds = bayopy.Bounds(lower=[-1.0] * 10, upper=[1.0] * 10)
    >>> algorithm = bayopy.SAASBO(
    ...     bounds=bounds,
    ...     config=bayopy.SAASBOConfig(
    ...         warmup_steps=512, posterior_samples=256, acquisition="EI", seed=7
    ...     ),
    ... )
    """

    CHECKPOINT_ADDED_CONFIG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **Algorithm.CHECKPOINT_ADDED_CONFIG_DEFAULTS, "target_accept_prob": 0.8,
    }
    ACQUISITION_POLICY = "configurable fully Bayesian EI or UCB"

    EXCEPTION_CLASS = SAASBOException
    native_environments = frozenset({"stochastic"})
    native_batching = True
    SOURCE = SourceRecord(
        paper_title="High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces",
        paper_year=2021,
        paper_url="https://proceedings.mlr.press/v161/eriksson21a.html",
        repository_url="https://github.com/martinjankowiak/saasbo",
        repository_status="author",
        notes="BoTorch/Pyro translation of the fully Bayesian SAAS model.",
    )

    def __init__(
        self,
        *,
        bounds: Bounds,
        config: SAASBOConfig | None = None,
        maximize: bool = True,
        dtype: torch.dtype | str = torch.float64,
        device: torch.device | str = "cpu",
        plot: bool = False,
        verbose: bool = False,
        verbose_setting: bool = False,
        verbose_iter_by_iter: bool = False,
        diagnostic_policy: Literal["raise", "warn"] = "warn",
        jit_compile: bool = False,
        mcmc_num_threads: int | None = None,
    ) -> None:
        """Initialize fully Bayesian SAAS optimization.

        Parameters
        ----------
        bounds:
            Physical action bounds in ``D_env`` coordinates.
        config:
            SAAS prior, NUTS, acquisition, batching, and seed settings. ``None``
            selects :class:`SAASBOConfig` defaults.
        maximize:
            Whether larger learner-visible observations are preferable.
        dtype, device:
            Tensor precision and execution device. Double precision is the
            default for stable fully Bayesian inference.
        plot, verbose, verbose_setting, verbose_iter_by_iter:
            Presentation flags; ``verbose`` also controls Pyro's progress bar.
        diagnostic_policy:
            Runtime handling for NUTS convergence diagnostics. Explicit
            ``"raise"`` enforces strict diagnostic acceptance. The default
            ``"warn"`` follows
            the source BoTorch fitting path by retaining nonempty, finite,
            loadable posterior draws while warning and invalidating source
            guarantees when divergences, low ESS, or poor R-hat are reported.
        jit_compile:
            Compile Pyro's potential function with TorchScript during NUTS.
            This is the method-preserving acceleration exposed by BoTorch's
            reference ``fit_fully_bayesian_model_nuts`` helper.  If the JIT
            path fails, BayOPy retries the same fit without JIT from the same
            Pyro seed, emits :class:`SAASBOWarning`, and invalidates source
            guarantees for that run.
        mcmc_num_threads:
            Optional CPU intra-op thread limit applied only while NUTS runs.
            Small SAAS GP factorizations can be faster with one thread because
            it avoids thread-pool overhead.  The prior, likelihood, samples,
            and all NUTS controls remain unchanged.  The previous process-wide
            setting is restored afterward under a module lock.

        Notes
        -----
        Construction does not import Pyro. Posterior fitting raises an
        actionable ``bayopy[mcmc]`` error if that optional capability is absent.
        """

        self.config = SAASBOConfig() if config is None else config
        self._validate_config(self.config)
        if diagnostic_policy not in {"raise", "warn"}:
            raise SAASBOMCMCError("diagnostic_policy must be 'raise' or 'warn'.")
        if not isinstance(jit_compile, bool):
            raise SAASBOMCMCError("jit_compile must be a boolean.")
        if mcmc_num_threads is not None and (
            isinstance(mcmc_num_threads, bool)
            or not isinstance(mcmc_num_threads, int)
            or mcmc_num_threads < 1
        ):
            raise SAASBOMCMCError("mcmc_num_threads must be a positive integer or None.")
        self.diagnostic_policy = diagnostic_policy
        self.jit_compile = jit_compile
        self.mcmc_num_threads = mcmc_num_threads
        super().__init__(
            bounds=bounds,
            batch_size=self.config.batch_size,
            q_batch=self.config.q_batch,
            maximize=maximize,
            dtype=dtype,
            device=device,
            seed=self.config.seed,
            initial_points_sampling_method=self.config.initial_points_sampling_method,
            plot=plot,
            verbose=verbose,
            verbose_setting=verbose_setting,
            verbose_iter_by_iter=verbose_iter_by_iter,
        )
        self.state = self._fresh_state()
        self._model: Any | None = None

    @staticmethod
    def _validate_config(config: SAASBOConfig) -> None:
        """Validate prior and sampler inputs before importing optional Pyro."""

        try:
            resolve_batch_configuration(batch_size=config.batch_size, q_batch=config.q_batch)
        except ValueError as error:
            raise SAASBOMCMCError(str(error)) from error
        if not math.isfinite(config.shrinkage_scale) or config.shrinkage_scale <= 0:
            raise SAASBOPriorError("shrinkage_scale must be finite and positive.")
        if (
            isinstance(config.target_accept_prob, bool)
            or not isinstance(config.target_accept_prob, (int, float))
            or not 0 < config.target_accept_prob < 1
        ):
            raise SAASBOMCMCError("target_accept_prob must lie strictly between 0 and 1.")
        integer_fields = {
            "warmup_steps": config.warmup_steps,
            "posterior_samples": config.posterior_samples,
            "thinning": config.thinning,
            "num_chains": config.num_chains,
            "max_tree_depth": config.max_tree_depth,
            "num_restarts": config.num_restarts,
            "raw_samples": config.raw_samples,
            "mc_samples": config.mc_samples,
            "seed": config.seed,
        }
        for name, value in integer_fields.items():
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise SAASBOMCMCError(f"{name} must be an integer >= {minimum}.")
        if config.posterior_samples // config.thinning < 1:
            raise SAASBOMCMCError("thinning must retain at least one posterior draw.")
        if config.initial_points is not None and (
            isinstance(config.initial_points, bool)
            or not isinstance(config.initial_points, int)
            or config.initial_points < 1
        ):
            raise SAASBOMCMCError("initial_points must be a positive integer or None.")
        if config.acquisition not in {"EI", "UCB"}:
            raise SAASBOMCMCError("acquisition must be 'EI' or 'UCB'.")
        if not math.isfinite(config.ucb_beta) or config.ucb_beta < 0:
            raise SAASBOMCMCError("ucb_beta must be finite and non-negative.")
        if not math.isfinite(config.max_r_hat) or config.max_r_hat < 1:
            raise SAASBOMCMCError("max_r_hat must be finite and at least one.")
        if config.min_effective_sample_size is not None and (
            isinstance(config.min_effective_sample_size, bool)
            or not math.isfinite(config.min_effective_sample_size)
            or config.min_effective_sample_size <= 0
        ):
            raise SAASBOMCMCError("min_effective_sample_size must be positive or None.")

    def _fresh_state(self) -> SAASBOState:
        """Create empty unit-cube training and posterior-sampler state."""

        return SAASBOState(
            X_train_unit=torch.empty(
                (0, self.action_dimension), dtype=self.dtype, device=self.device
            ),
            y_train=torch.empty((0,), dtype=self.dtype, device=self.device),
        )

    def _to_unit(self, actions: torch.Tensor) -> torch.Tensor:
        """Map physical ``D_env`` actions to SAAS model coordinates in ``[0, 1]``."""

        lower, upper = self._bounds_on_algorithm_device()
        return (actions - lower) / (upper - lower)

    def _from_unit(self, unit_actions: torch.Tensor) -> torch.Tensor:
        """Map unit-cube SAAS model inputs back to physical actions."""

        lower, upper = self._bounds_on_algorithm_device()
        return lower + unit_actions * (upper - lower)

    def _initialize_from_environment_impl(self, data: Any) -> None:
        """Import public history and invalidate any previously cached posterior."""

        actions = torch.as_tensor(data.actions, dtype=self.dtype, device=self.device)
        self.state.X_train_unit = self._to_unit(actions).detach().clone()
        self.state.y_train = (
            torch.as_tensor(data.observations, dtype=self.dtype, device=self.device)
            .reshape(-1)
            .detach()
            .clone()
        )
        self._invalidate_posterior()

    def _prepare_discrete_candidates_impl(
        self,
        *,
        candidate_actions: torch.Tensor,
        candidate_dataset_indices: torch.Tensor,
        round_info: Mapping[str, Any],
    ) -> DiscreteAcquisitionCandidates:
        """Normalize finite rows while preserving actions and dataset indices."""

        del round_info
        return DiscreteAcquisitionCandidates(
            environment_actions=candidate_actions,
            model_inputs=self._to_unit(candidate_actions),
            dataset_indices=candidate_dataset_indices,
        )

    def _standardized_outcomes(self) -> torch.Tensor:
        """Standardize rewards as required by the fully Bayesian model."""

        quality = to_quality(self.state.y_train, maximize=self.maximize)
        mean = quality.mean()
        # BoTorch's validation uses the sample standard deviation when at least
        # two observations exist; use the same convention to avoid a systematic
        # scale mismatch in small initial designs.
        std = quality.std(unbiased=quality.numel() > 1)
        if not torch.isfinite(std) or std <= torch.finfo(self.dtype).eps:
            std = torch.ones((), dtype=self.dtype, device=self.device)
        self.state.outcome_mean = mean.detach().clone()
        self.state.outcome_std = std.detach().clone()
        return (quality - mean) / std

    def _build_saas_model(self) -> Any:
        """Construct BoTorch's fully Bayesian GP with the configured SAAS scale."""

        try:
            from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
        except ImportError as error:  # pragma: no cover - core BoTorch dependency
            raise SAASBOMCMCError(
                "BoTorch fully Bayesian models are unavailable; reinstall BayOPy core dependencies."
            ) from error
        if self.state.X_train_unit.shape[0] < 1:
            raise SAASBOMCMCError("SAAS posterior fitting requires at least one observation.")
        y = self._standardized_outcomes().unsqueeze(-1)
        model = SaasFullyBayesianSingleTaskGP(self.state.X_train_unit, y)
        pyro_model: Any = model.pyro_model
        original = pyro_model.sample_lengthscale
        scale = float(self.config.shrinkage_scale)

        def sample_lengthscale(
            pyro_model: Any, dim: int, alpha: float = scale, **kwargs: Any
        ) -> Any:
            """Bind the configured SAAS shrinkage scale at the Pyro boundary.

            Parameters
            ----------
            pyro_model:
                Bound Pyro model instance; supplied by ``MethodType``.
            dim:
                Number of input dimensions requiring inverse lengthscales.
            alpha:
                Ignored caller value; BayOPy's validated configured scale is
                always used.
            **kwargs:
                Additional arguments forwarded to BoTorch's sampler.

            Returns
            -------
            Any
                The Pyro sample-site value produced by the original method.
            """

            del pyro_model, alpha
            return original(dim=dim, alpha=scale, **kwargs)

        pyro_model.sample_lengthscale = types.MethodType(sample_lengthscale, pyro_model)
        return model

    def _run_nuts(self, model: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Run Pyro NUTS and return retained, postprocessed posterior draws."""

        try:
            import pyro
            from pyro.infer import MCMC, NUTS
        except ImportError as error:
            raise SAASBOMCMCError(
                "SAASBO requires Pyro; install the 'bayopy[mcmc]' capability extra."
            ) from error
        except Exception as error:
            raise SAASBOMCMCError(f"The Pyro backend could not be initialized: {error}") from error
        model.train()

        def report_jit_fallback(error: Exception, *, phase: str) -> None:
            """Warn and record that same-seed eager NUTS replaced JIT execution."""

            warnings.warn(
                f"Pyro JIT NUTS {phase} failed; the posterior is being sampled "
                "by a same-seed eager NUTS retry. Source guarantees no longer "
                f"apply. JIT error: {error}",
                SAASBOWarning,
                stacklevel=3,
            )
            self.invalidate_source_guarantees("saasbo.nuts_jit_fallback")

        try:

            def make_mcmc(*, jit_compile: bool) -> Any:
                """Construct NUTS/MCMC with the requested execution backend."""

                nuts = NUTS(
                    model.pyro_model.sample,
                    full_mass=True,
                    ignore_jit_warnings=True,
                    jit_compile=jit_compile,
                    max_tree_depth=self.config.max_tree_depth,
                    target_accept_prob=self.config.target_accept_prob,
                )
                return MCMC(
                    nuts,
                    warmup_steps=self.config.warmup_steps,
                    num_samples=self.config.posterior_samples,
                    num_chains=self.config.num_chains,
                    disable_progbar=not self.verbose,
                )

            using_jit = self.jit_compile
            try:
                mcmc = make_mcmc(jit_compile=using_jit)
            except Exception as jit_initialization_error:
                if not using_jit:
                    raise
                try:
                    mcmc = make_mcmc(jit_compile=False)
                except Exception as eager_initialization_error:
                    raise SAASBOMCMCError(
                        "JIT NUTS initialization failed and eager initialization "
                        "also failed "
                        f"(JIT: {jit_initialization_error}; "
                        f"eager: {eager_initialization_error})."
                    ) from eager_initialization_error
                using_jit = False
                report_jit_fallback(jit_initialization_error, phase="initialization")
        except Exception as error:
            if isinstance(error, SAASBOMCMCError):
                raise
            raise SAASBOMCMCError(
                f"NUTS/MCMC initialization failed before sampling: {error}"
            ) from error
        drawn_seed = int(
            torch.randint(
                0,
                2**31 - 1,
                (1,),
                generator=self.generator,
                device=self.device,
            ).item()
        )
        fork_devices: list[int] = []
        if self.device.type == "cuda":
            fork_devices = [
                torch.cuda.current_device() if self.device.index is None else self.device.index
            ]
        thread_context = (
            _TORCH_NUM_THREADS_LOCK if self.mcmc_num_threads is not None else nullcontext()
        )
        # Pyro also seeds Python and NumPy. Preserve all ambient RNGs so a
        # posterior fit cannot alter another algorithm or the user's objective.
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        try:
            with thread_context:
                previous_num_threads: int | None = None
                try:
                    if self.mcmc_num_threads is not None:
                        previous_num_threads = torch.get_num_threads()
                        if previous_num_threads != self.mcmc_num_threads:
                            torch.set_num_threads(self.mcmc_num_threads)
                    with torch.random.fork_rng(devices=fork_devices):
                        pyro.set_rng_seed(drawn_seed)
                        try:
                            mcmc.run()
                        except Exception as jit_error:
                            if not using_jit:
                                raise
                            # A failed trace may have consumed arbitrary random
                            # draws. Re-seeding before the eager kernel makes
                            # the fallback identical to a direct eager attempt
                            # with this algorithm-owned seed.
                            pyro.set_rng_seed(drawn_seed)
                            try:
                                mcmc = make_mcmc(jit_compile=False)
                                mcmc.run()
                            except Exception as eager_error:
                                raise SAASBOMCMCError(
                                    "JIT NUTS failed and the same-seed eager retry also "
                                    f"failed (JIT: {jit_error}; eager: {eager_error})."
                                ) from eager_error
                            report_jit_fallback(jit_error, phase="execution")
                finally:
                    if (
                        previous_num_threads is not None
                        and torch.get_num_threads() != previous_num_threads
                    ):
                        torch.set_num_threads(previous_num_threads)
        except SAASBOMCMCError:
            raise
        except Exception as error:
            raise SAASBOMCMCError(f"NUTS failed before producing a posterior: {error}") from error
        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
        raw = mcmc.get_samples(group_by_chain=False)
        processed = model.pyro_model.postprocess_mcmc_samples(mcmc_samples=raw)
        retained = {
            name: value[:: self.config.thinning].detach().clone()
            for name, value in processed.items()
        }
        if self.config.posterior_samples < 4:
            diagnostics = {
                "unavailable": "Pyro requires at least four draws per chain to compute "
                "split-chain convergence diagnostics."
            }
        else:
            diagnostics = mcmc.diagnostics()
        return retained, diagnostics

    @staticmethod
    def _diagnostic_values(value: Any, key: str) -> list[float]:
        """Collect numeric values for a named key from nested Pyro diagnostics."""

        collected: list[float] = []
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                if str(nested_key).lower() == key.lower():
                    tensor = torch.as_tensor(nested_value).reshape(-1)
                    collected.extend(float(item) for item in tensor)
                else:
                    collected.extend(SAASBO._diagnostic_values(nested_value, key))
        elif isinstance(value, (list, tuple)):
            for item in value:
                collected.extend(SAASBO._diagnostic_values(item, key))
        return collected

    def _handle_mcmc_diagnostic(self, message: str, *, reason: str, allow: bool = False) -> None:
        """Raise or retain a usable posterior according to the runtime policy."""

        if self.diagnostic_policy == "raise" and not allow:
            raise SAASBODivergenceError(message)
        policy = "allow_divergences=True" if allow else "diagnostic_policy='warn'"
        warnings.warn(
            f"{message} Finite, loadable posterior draws were retained under "
            f"{policy}; source guarantees no longer apply.",
            SAASBOWarning,
            stacklevel=3,
        )
        self.invalidate_source_guarantees(reason)

    def _check_mcmc_diagnostics(self, diagnostics: Mapping[str, Any]) -> None:
        """Handle divergences, poor ESS, and multi-chain R-hat failures."""

        if diagnostics.get("unavailable"):
            self._handle_mcmc_diagnostic(
                str(diagnostics["unavailable"]), reason="saasbo.mcmc_diagnostics_unavailable"
            )
        divergences = diagnostics.get("divergences", {})
        divergence_count = 0
        if isinstance(divergences, Mapping):
            divergence_count = sum(len(value) for value in divergences.values())
        elif isinstance(divergences, (list, tuple)):
            divergence_count = len(divergences)
        if divergence_count:
            self._handle_mcmc_diagnostic(
                f"NUTS reported {divergence_count} divergent transitions.",
                reason="saasbo.mcmc_divergences",
                allow=self.config.allow_divergences,
            )
        ess_values = self._diagnostic_values(diagnostics, "n_eff")
        ess_floor = self.config.min_effective_sample_size
        if (
            ess_floor is not None
            and ess_values
            and any(not math.isfinite(value) or value < ess_floor for value in ess_values)
        ):
            self._handle_mcmc_diagnostic(
                "NUTS effective sample size is below the configured floor.",
                reason="saasbo.low_effective_sample_size",
            )
        if self.config.num_chains > 1:
            r_hats = self._diagnostic_values(diagnostics, "r_hat")
            if not r_hats or any(
                not math.isfinite(value) or value > self.config.max_r_hat for value in r_hats
            ):
                self._handle_mcmc_diagnostic(
                    "NUTS R-hat diagnostics do not show convergence.",
                    reason="saasbo.r_hat",
                )

    @staticmethod
    def _posterior_sample_problem(samples: Mapping[str, torch.Tensor]) -> str | None:
        """Return why retained posterior samples cannot safely be loaded, if any."""

        if not isinstance(samples, Mapping) or not samples:
            return "NUTS retained no usable posterior hyperparameter samples."
        for name, value in samples.items():
            if not isinstance(value, torch.Tensor):
                return f"NUTS posterior sample {name!r} is not a tensor."
            if value.ndim < 1 or value.shape[0] < 1 or value.numel() < 1:
                return f"NUTS posterior sample {name!r} retained no draws."
            if not torch.isfinite(value).all():
                return f"NUTS posterior sample {name!r} contains non-finite values."
        return None

    def _fit_posterior(self) -> Any:
        """Run NUTS, validate diagnostics, and cache retained hyperparameter draws.

        The Pyro seed is drawn from the algorithm-owned torch generator, so its
        advancement is part of run progress. Retained posterior samples are an
        in-memory training cache and are deliberately not checkpointed. No MLE
        fallback is used if sampling or diagnostics fail.
        """

        model = self._build_saas_model()
        samples, diagnostics = self._run_nuts(model)
        sample_problem = self._posterior_sample_problem(samples)
        if sample_problem is not None:
            raise SAASBOMCMCError(sample_problem)
        try:
            model.load_mcmc_samples(samples)
        except Exception as error:
            raise SAASBOMCMCError(f"NUTS posterior samples could not be loaded: {error}") from error
        self._check_mcmc_diagnostics(diagnostics)
        model.eval()
        self.state.posterior_hyperparameter_samples = samples
        self.state.mcmc_diagnostics = diagnostics
        self.state.posterior_valid = True
        self.state.fit_generation += 1
        self._model = model
        return model

    def _restore_posterior(self) -> Any:
        """Rebuild a model from retained in-memory MCMC draws."""

        model = self._build_saas_model()
        sample_problem = self._posterior_sample_problem(self.state.posterior_hyperparameter_samples)
        if sample_problem is not None:
            raise SAASBOCheckpointError(
                f"Stored SAAS posterior samples cannot be restored: {sample_problem}"
            )
        try:
            model.load_mcmc_samples(self.state.posterior_hyperparameter_samples)
        except Exception as error:
            raise SAASBOCheckpointError(
                f"Stored SAAS posterior samples cannot be restored: {error}"
            ) from error
        model.eval()
        self._model = model
        return model

    def _build_acquisition(self, model: Any) -> Any:
        """Build posterior-averaged qLogEI or qUCB with a deterministic sampler."""

        try:
            from botorch.acquisition.logei import qLogExpectedImprovement
            from botorch.acquisition.monte_carlo import qUpperConfidenceBound
            from botorch.sampling.normal import SobolQMCNormalSampler
        except ImportError as error:  # pragma: no cover - core BoTorch dependency
            raise SAASBOMCMCError("Required BoTorch acquisitions are unavailable.") from error
        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.config.mc_samples]),
            seed=self.config.seed + self.iteration,
        )
        if self.config.acquisition == "EI":
            assert self.state.outcome_mean is not None and self.state.outcome_std is not None
            standardized = (
                to_quality(self.state.y_train, maximize=self.maximize) - self.state.outcome_mean
            ) / self.state.outcome_std
            return qLogExpectedImprovement(
                model=model, best_f=float(standardized.max().item()), sampler=sampler
            )
        return qUpperConfidenceBound(
            model=model,
            beta=self.config.ucb_beta,
            sampler=sampler,
        )

    def _initial_count(self) -> int:
        """Return the configured or dimension-dependent warm-start row count."""

        return (
            max(2 * (self.action_dimension + 1), 4)
            if self.config.initial_points is None
            else self.config.initial_points
        )

    def _initial_design(
        self, round_info: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Choose deterministic Sobol/maximin initial points before NUTS fitting."""

        if "candidate_actions" in round_info:
            candidates = torch.as_tensor(
                round_info["candidate_actions"], dtype=self.dtype, device=self.device
            )
            indices = torch.as_tensor(
                round_info["candidate_dataset_indices"], dtype=torch.long, device=self.device
            )
            unit = self._to_unit(candidates)
            if self.state.X_train_unit.shape[0]:
                distances = torch.cdist(unit, self.state.X_train_unit).min(dim=1).values
            else:
                distances = torch.ones(unit.shape[0], dtype=self.dtype, device=self.device)
            order = torch.argsort(distances, descending=True, stable=True)[: self.request_size]
            return candidates[order], indices[order]
        unit_bounds = Bounds(
            torch.zeros(self.action_dimension, dtype=self.dtype, device=self.device),
            torch.ones(self.action_dimension, dtype=self.dtype, device=self.device),
        )
        return (
            self._from_unit(
                self._sample_design(
                    unit_bounds,
                    self.request_size,
                    phase="initial",
                    seed=self.config.seed + self.iteration,
                )
            ),
            None,
        )

    def _finite_acquisition(
        self, acquisition: Any, table: DiscreteAcquisitionCandidates
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Select finite rows using the configured independent or joint mode."""

        from bayopy.models.Acquisitions import optimize_acquisition_discrete

        result = optimize_acquisition_discrete(
            acquisition=acquisition,
            candidates=table,
            batch_size=self.batch_size,
            q_batch=self.q_batch,
        )
        return (
            result.environment_actions,
            result.dataset_indices,
            float(result.acquisition_values.sum().item()),
        )

    def _ask_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Select a physical action from initial design or Bayesian acquisition.

        Acquisition optimization occurs in full-dimensional unit-cube model
        coordinates. Finite mode scores only complete public rows and preserves
        their dataset indices; continuous results are denormalized before return.
        """

        if self.state.X_train_unit.shape[0] < self._initial_count():
            actions, dataset_indices = self._initial_design(round_info)
            model_inputs = self._to_unit(actions)
            acquisition_value = None
        else:
            model = self._model if self.state.posterior_valid and self._model is not None else None
            if model is None:
                model = (
                    self._restore_posterior()
                    if self.state.posterior_valid
                    else self._fit_posterior()
                )
            acquisition = self._build_acquisition(model)
            dataset_indices = None
            if "candidate_actions" in round_info:
                candidates = torch.as_tensor(
                    round_info["candidate_actions"], dtype=self.dtype, device=self.device
                )
                indices = torch.as_tensor(
                    round_info["candidate_dataset_indices"],
                    dtype=torch.long,
                    device=self.device,
                )
                table = self.prepare_discrete_candidates(
                    candidate_actions=candidates,
                    candidate_dataset_indices=indices,
                    round_info=round_info,
                )
                actions, dataset_indices, acquisition_value = self._finite_acquisition(
                    acquisition, table
                )
                model_inputs = self._to_unit(actions)
            else:
                try:
                    search_bounds = torch.stack(
                        (
                            torch.zeros(
                                self.action_dimension,
                                dtype=self.dtype,
                                device=self.device,
                            ),
                            torch.ones(
                                self.action_dimension,
                                dtype=self.dtype,
                                device=self.device,
                            ),
                        )
                    )
                    if (
                        self.acquisition_sampling_method == "paper"
                        and self.batch_size == 1
                        and self.q_batch is None
                    ):
                        import botorch.optim

                        with translate_backend_warnings(
                            operation="SAASBO continuous acquisition optimization"
                        ):
                            unit, value = botorch.optim.optimize_acqf(
                                acq_function=acquisition,
                                bounds=search_bounds,
                                q=1,
                                num_restarts=self.config.num_restarts,
                                raw_samples=self.config.raw_samples,
                                options={"maxls": 100},
                            )
                    else:
                        from bayopy.models.Acquisitions import (
                            optimize_acquisition_continuous,
                        )

                        unit, value = optimize_acquisition_continuous(
                            acquisition=acquisition,
                            bounds=search_bounds,
                            batch_size=self.batch_size,
                            q_batch=self.q_batch,
                            num_restarts=self.config.num_restarts,
                            raw_samples=self.config.raw_samples,
                            sampling_method=self.acquisition_sampling_method,
                            sampling_generator=self.generator,
                            sampling_seed=self.config.seed + self.iteration,
                        )
                except Exception as error:
                    raise SAASBOMCMCError(
                        f"Fully Bayesian acquisition optimization failed: {error}"
                    ) from error
                model_inputs = unit.detach()
                actions = self._from_unit(model_inputs)
                acquisition_value = float(value.reshape(-1)[0].item())
        self._last_acquisition_value = acquisition_value
        self._record_last_ask(
            actions=actions,
            model_inputs=model_inputs,
            round_info=round_info,
            algorithm_metadata={
                "dataset_indices": dataset_indices,
                "posterior_generation": self.state.fit_generation,
            },
        )
        return actions[0] if self.request_size == 1 else actions

    def _invalidate_posterior(self) -> None:
        """Discard posterior draws and standardization derived from stale data."""

        self.state.posterior_hyperparameter_samples = {}
        self.state.mcmc_diagnostics = {}
        self.state.posterior_valid = False
        self.state.outcome_mean = None
        self.state.outcome_std = None
        self._model = None

    def _tell_impl(
        self,
        *,
        action: torch.Tensor,
        feedback: Mapping[str, Any],
        round_info: Mapping[str, Any],
    ) -> None:
        """Append public observations in unit coordinates and stale the posterior."""

        del round_info
        observations = torch.as_tensor(
            feedback["observation"], dtype=self.dtype, device=self.device
        ).reshape(-1)
        self.state.X_train_unit = torch.cat((self.state.X_train_unit, self._to_unit(action)), dim=0)
        self.state.y_train = torch.cat((self.state.y_train, observations), dim=0)
        self._invalidate_posterior()

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Report posterior validity, retained draws, fit generation, and MCMC data."""

        return {
            "posterior_valid": self.state.posterior_valid,
            "retained_sample_count": (
                0
                if not self.state.posterior_hyperparameter_samples
                else min(
                    value.shape[0] for value in self.state.posterior_hyperparameter_samples.values()
                )
            ),
            "fit_generation": self.state.fit_generation,
            "mcmc_diagnostics": self.state.mcmc_diagnostics,
            "shrinkage_scale": self.config.shrinkage_scale,
            "diagnostic_policy": self.diagnostic_policy,
            "jit_compile": self.jit_compile,
            "mcmc_num_threads": self.mcmc_num_threads,
        }

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize observations and progress, excluding trained posterior draws."""

        return {
            "config": asdict(self.config),
            "diagnostic_policy": self.diagnostic_policy,
            "jit_compile": self.jit_compile,
            "mcmc_num_threads": self.mcmc_num_threads,
            "X_train_unit": self.state.X_train_unit,
            "y_train": self.state.y_train,
            "mcmc_diagnostics": self.state.mcmc_diagnostics,
            "fit_generation": self.state.fit_generation,
        }

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Validate checkpoint config, unit-cube rows, and posterior consistency."""

        required = {
            "config",
            "X_train_unit",
            "y_train",
            "mcmc_diagnostics",
            "fit_generation",
        }
        missing = required - set(state)
        if missing:
            raise SAASBOCheckpointError(f"SAASBO state is missing: {sorted(missing)}.")
        # Runtime diagnostic/JIT/thread policies do not define the objective
        # or stored observations. Resume deliberately runs fresh inference
        # using the current policies rather than restoring a fitted posterior.
        del strict
        X = torch.as_tensor(state["X_train_unit"])
        y = torch.as_tensor(state["y_train"])
        if X.ndim != 2 or X.shape[1] != self.action_dimension:
            raise SAASBOCheckpointError("Stored SAAS model inputs have an invalid shape.")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise SAASBOCheckpointError("Stored SAAS outcomes are not row-aligned.")
        if torch.any(X < 0) or torch.any(X > 1):
            raise SAASBOCheckpointError("Stored SAAS model inputs leave the unit cube.")

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Restore observations and force fresh posterior inference."""

        del strict
        self.state = SAASBOState(
            X_train_unit=torch.as_tensor(
                state["X_train_unit"], dtype=self.dtype, device=self.device
            ).clone(),
            y_train=torch.as_tensor(state["y_train"], dtype=self.dtype, device=self.device).clone(),
            posterior_hyperparameter_samples={},
            mcmc_diagnostics=dict(state["mcmc_diagnostics"]),
            posterior_valid=False,
            fit_generation=int(state["fit_generation"]),
            outcome_mean=None,
            outcome_std=None,
        )
        self._model = None

    def _reset_impl(self) -> None:
        """Clear training data, posterior draws, diagnostics, and cached model."""

        self.state = self._fresh_state()
        self._model = None


__all__ = ["SAASBO", "SAASBOConfig", "SAASBOState"]
