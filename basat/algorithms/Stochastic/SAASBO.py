"""Fully Bayesian Sparse Axis-Aligned Subspace Bayesian optimization.

SAASBO keeps full environment coordinates and induces sparsity through the
hierarchical half-Cauchy prior on inverse squared length scales.  This module
uses BoTorch's fully Bayesian GP representation and Pyro NUTS by default.
A native NumPyro backend implements the released MCTS-VS RBF SAAS
protocol. Neither backend substitutes MLE ARD or a sparsity penalty for NUTS.
"""

from __future__ import annotations

import gc
import math
import random
import threading
import time
import types
import warnings
from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np
import torch

from basat.models.Acquisitions import DiscreteAcquisitionCandidates
from basat.models.utils import Bounds
from basat.warnings import translate_backend_warnings

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
    inference_backend:
        ``"pyro"`` preserves the existing implementation. ``"numpyro"`` uses
        Basat's NumPyro implementation of the released MCTS-VS RBF SAAS
        model and requires ``source_protocol="mcts_vs_2022"``. Both backends
        are installed with Basat and consume observations only.
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
    source_protocol: Literal["saasbo", "mcts_vs_2022"] = "saasbo"
    inference_backend: Literal["pyro", "numpyro"] = "pyro"


@dataclass(slots=True)
class SAASBOState:
    """Mutable training data, posterior draws, and sampler diagnostics.

    Attributes
    ----------
    X_train_unit, y_train:
        Normalized ``(N, D_env)`` inputs and raw learner-visible responses.
    posterior_hyperparameter_samples, mcmc_diagnostics:
        Active posterior draws and convergence evidence.
    fit_diagnostics_history:
        Durable per-attempt training counts, backend, runtime, diagnostics and
        warning/error events. This history survives posterior invalidation;
        it contains no trained model or retained posterior draws.
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
    fit_diagnostics_history: list[dict[str, Any]] = field(default_factory=list)


class SAASBO(Algorithm):
    """SAASBO with exact NUTS posterior draws and finite-row-safe acquisition.

    Install ``basat[mcmc]`` to fit this fully Bayesian model.

    Examples
    --------
    >>> import basat
    >>> bounds = basat.Bounds(lower=[-1.0] * 10, upper=[1.0] * 10)
    >>> algorithm = basat.SAASBO(
    ...     bounds=bounds,
    ...     config=basat.SAASBOConfig(
    ...         warmup_steps=512, posterior_samples=256, acquisition="EI", seed=7
    ...     ),
    ... )
    """

    CHECKPOINT_ADDED_CONFIG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **Algorithm.CHECKPOINT_ADDED_CONFIG_DEFAULTS,
        "target_accept_prob": 0.8,
        "inference_backend": "pyro",
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
        release_inference_cache: bool = False,
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
            path fails, Basat retries the same fit without JIT from the same
            Pyro seed, emits :class:`SAASBOWarning`, and invalidates source
            guarantees for that run.
        mcmc_num_threads:
            Optional PyTorch CPU intra-op limit applied during Pyro NUTS only.
            Small SAAS GP factorizations can be faster with one thread because
            it avoids thread-pool overhead.  The prior, likelihood, samples,
            and all NUTS controls remain unchanged.  The previous process-wide
            setting is restored afterward under a module lock.
            NumPyro/XLA owns a separate thread pool; this setting does not limit
            it. Plan NumPyro worker concurrency separately.
        release_inference_cache:
            NumPyro-only runtime memory control, disabled by default. After a
            fitted proposal set is materialized and the posterior is invalidated,
            clear JAX's process-global compilation caches and run garbage
            collection. This can increase recompilation time and affect unrelated
            JAX computations; enable only in a dedicated worker. It changes no
            posterior draws, proposal queue, RNG state or inference budget.
            The current constructor setting takes precedence on resume, like
            the other runtime policies; it is not a mathematical configuration
            fingerprint and old checkpoints imply a recorded value of False.

        Notes
        -----
        Construction does not import an inference backend. Posterior fitting
        raises an actionable installation error if a required dependency
        is absent. ``jit_compile`` controls only Pyro;
        NumPyro compiles its numerical operations with JAX.
        """

        self.config = SAASBOConfig() if config is None else config
        self._validate_config(self.config)
        if diagnostic_policy not in {"raise", "warn"}:
            raise SAASBOMCMCError("diagnostic_policy must be 'raise' or 'warn'.")
        if not isinstance(jit_compile, bool):
            raise SAASBOMCMCError("jit_compile must be a boolean.")
        if not isinstance(release_inference_cache, bool):
            raise SAASBOMCMCError("release_inference_cache must be a boolean.")
        if release_inference_cache and self.config.inference_backend != "numpyro":
            raise SAASBOMCMCError("release_inference_cache is available only for NumPyro.")
        if mcmc_num_threads is not None and (
            isinstance(mcmc_num_threads, bool)
            or not isinstance(mcmc_num_threads, int)
            or mcmc_num_threads < 1
        ):
            raise SAASBOMCMCError("mcmc_num_threads must be a positive integer or None.")
        self.diagnostic_policy = diagnostic_policy
        self.jit_compile = jit_compile
        self.mcmc_num_threads = mcmc_num_threads
        self.release_inference_cache = release_inference_cache
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
        self._source_queue: torch.Tensor | None = None
        self._source_sobol_rng = np.random.RandomState(self.config.seed)
        self._active_fit_record: dict[str, Any] | None = None
        if release_inference_cache:
            warnings.warn(
                "NumPyro inference-cache release is enabled: JAX compilation caches "
                "are process-global, so clearing them can slow unrelated JAX work. "
                "Use a dedicated worker; inference settings and RNG state are unchanged.",
                SAASBOWarning,
                stacklevel=2,
            )
        if self.config.inference_backend == "numpyro" and (
            self.device.type != "cpu" or self.dtype != torch.float64
        ):
            raise SAASBOMCMCError("Released NumPyro SAAS currently requires float64 on CPU.")

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
        if config.source_protocol not in {"saasbo", "mcts_vs_2022"}:
            raise SAASBOMCMCError("Unknown SAAS source_protocol.")
        if config.inference_backend not in {"pyro", "numpyro"}:
            raise SAASBOMCMCError("inference_backend must be 'pyro' or 'numpyro'.")
        if config.inference_backend == "numpyro" and (
            config.source_protocol != "mcts_vs_2022"
            or config.acquisition != "EI"
            or config.num_chains != 1
            or config.batch_size not in (None, 1)
            or config.q_batch is not None
        ):
            raise SAASBOMCMCError(
                "Released NumPyro SAAS requires source_protocol='mcts_vs_2022', "
                "EI, one chain, and batch_size=1 (the source three-point program "
                "is queued internally)."
            )
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
        self.state.fit_diagnostics_history = []
        self._active_fit_record = None
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
        std = quality.std(
            unbiased=quality.numel() > 1 and self.config.source_protocol != "mcts_vs_2022"
        )
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
                "BoTorch fully Bayesian models are unavailable; reinstall Basat core dependencies."
            ) from error
        if self.state.X_train_unit.shape[0] < 1:
            raise SAASBOMCMCError("SAAS posterior fitting requires at least one observation.")
        y = self._standardized_outcomes().unsqueeze(-1)
        source_mode = self.config.source_protocol == "mcts_vs_2022"
        from botorch import settings as botorch_settings

        if source_mode:
            inputs = self.state.X_train_unit
            if not torch.isfinite(inputs).all() or not ((inputs >= 0) & (inputs <= 1)).all():
                raise SAASBOMCMCError("Source SAAS inputs must be finite and unit scaled.")
            population_std = y.std(unbiased=False)
            if (
                not torch.isfinite(y).all()
                or abs(float(y.mean())) > 1e-5
                or (float(population_std) > 1e-8 and abs(float(population_std) - 1.0) > 1e-5)
            ):
                raise SAASBOMCMCError("Source SAAS rewards must use population standardization.")
        with (
            translate_backend_warnings(operation="SAAS surrogate construction"),
            # The source uses ddof=0, incompatible with BoTorch's ddof=1 check.
            # Validate the actual source convention above instead of altering it.
            botorch_settings.validate_input_scaling(False) if source_mode else nullcontext(),
        ):
            model = SaasFullyBayesianSingleTaskGP(
                self.state.X_train_unit,
                y,
                train_Yvar=torch.full_like(y, 2e-6) if source_mode else None,
            )
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
                Ignored caller value; Basat's validated configured scale is
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
        if source_mode:
            import gpytorch
            import pyro

            # X is fixed for this entire NUTS fit. Cache its squared pairwise
            # coordinate differences, not any parameter-dependent covariance.
            # The released weighted-RBF formula is then one matrix-vector
            # product per leapfrog evaluation, with the identical likelihood.
            inputs = pyro_model.train_X
            delta_squared = (
                (inputs[:, None, :] - inputs[None, :, :])
                .square()
                .reshape(-1, self.action_dimension)
                .detach()
            )

            def sample(source_model: Any) -> None:
                tkwargs: dict[str, Any] = {"dtype": y.dtype, "device": y.device}
                variance = pyro.sample(
                    "outputscale",
                    pyro.distributions.LogNormal(
                        torch.tensor(0.0, **tkwargs), torch.tensor(10.0, **tkwargs)
                    ),
                )
                lengthscale = source_model.sample_lengthscale(dim=self.action_dimension, **tkwargs)
                count = source_model.train_X.shape[0]
                distance = (delta_squared @ lengthscale.square().reciprocal()).reshape(count, count)
                covariance = variance * torch.exp(-0.5 * distance)
                covariance = covariance + 2e-6 * torch.eye(count, **tkwargs)
                pyro.sample(
                    "Y",
                    pyro.distributions.MultivariateNormal(
                        torch.zeros(count, **tkwargs), covariance_matrix=covariance
                    ),
                    obs=source_model.train_Y.squeeze(-1),
                )

            original_postprocess = pyro_model.postprocess_mcmc_samples

            def postprocess(
                source_model: Any, mcmc_samples: dict[str, torch.Tensor]
            ) -> dict[str, torch.Tensor]:
                del source_model
                samples = original_postprocess(mcmc_samples)
                samples["mean"] = torch.zeros_like(samples["outputscale"])
                return samples

            def covariance_module(
                source_model: Any,
                *,
                use_scale_kernel: bool,
                batch_shape: torch.Size,
                **tkwargs: Any,
            ) -> Any:
                del source_model, use_scale_kernel
                return gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(
                        ard_num_dims=self.action_dimension, batch_shape=batch_shape
                    ),
                    batch_shape=batch_shape,
                ).to(**tkwargs)

            pyro_model.sample = types.MethodType(sample, pyro_model)
            pyro_model.postprocess_mcmc_samples = types.MethodType(postprocess, pyro_model)
            pyro_model._get_covar_module = types.MethodType(covariance_module, pyro_model)
        return model

    def _run_nuts(self, model: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Run Pyro NUTS and return retained, postprocessed posterior draws."""

        try:
            import pyro
            from pyro.infer import MCMC, NUTS
        except ImportError as error:
            raise SAASBOMCMCError(
                "SAASBO requires Pyro. Reinstall Basat from its project directory "
                "using 'python -m pip install .' so pip installs the required dependencies."
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
                    full_mass=self.config.source_protocol != "mcts_vs_2022",
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

        if self._active_fit_record is not None:
            self._active_fit_record["diagnostic_events"].append(
                {
                    "reason": reason,
                    "message": message,
                    "severity": "error"
                    if self.diagnostic_policy == "raise" and not allow
                    else "warning",
                }
            )
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

        if self._active_fit_record is not None:
            self._active_fit_record["diagnostics"] = deepcopy(dict(diagnostics))
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
        """Audit one inference attempt without retaining previous trained models."""
        started = time.perf_counter()
        record: dict[str, Any] = {
            "attempt": len(self.state.fit_diagnostics_history) + 1,
            "training_rows": int(self.state.y_train.numel()),
            "backend": self.config.inference_backend,
            "status": "running",
            "diagnostic_policy": self.diagnostic_policy,
            "release_inference_cache": self.release_inference_cache,
            "diagnostics": {},
            "diagnostic_events": [],
        }
        self.state.fit_diagnostics_history.append(record)
        self._active_fit_record = record
        try:
            model = self._fit_posterior_impl()
            record["status"] = "completed"
            record["fit_generation"] = self.state.fit_generation
            return model
        except BaseException as error:
            record["status"] = "failed" if isinstance(error, Exception) else "interrupted"
            record["error_type"] = type(error).__name__
            record["error"] = str(error)
            raise
        finally:
            record["elapsed_seconds"] = time.perf_counter() - started
            self._active_fit_record = None

    def _fit_posterior_impl(self) -> Any:
        """Run NUTS, validate diagnostics, and cache retained hyperparameter draws.

        The Pyro seed is drawn from the algorithm-owned torch generator, so its
        advancement is part of run progress. Retained posterior samples are an
        in-memory training cache and are deliberately not checkpointed. No MLE
        fallback is used if sampling or diagnostics fail.
        """

        if self.config.inference_backend == "numpyro":
            return self._fit_numpyro_posterior()
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

    def _fit_numpyro_posterior(self) -> Any:
        """Fit the native numerical model with the released minimization signs."""
        from basat.models.SAASNumPyro import SAASNumPyro

        quality = to_quality(self.state.y_train, maximize=self.maximize)
        values = -quality.detach().cpu().numpy()
        mean, std = float(values.mean()), float(values.std(ddof=0))
        if not math.isfinite(std) or std <= np.finfo(np.float64).eps:
            raise SAASBOMCMCError(
                "Released NumPyro SAAS cannot standardize constant/nonfinite outcomes."
            )
        standardized = (values - mean) / std
        self.state.outcome_mean = torch.tensor(-mean, dtype=self.dtype, device=self.device)
        self.state.outcome_std = torch.tensor(std, dtype=self.dtype, device=self.device)
        try:
            model = SAASNumPyro(
                shrinkage_scale=self.config.shrinkage_scale,
                warmup_steps=self.config.warmup_steps,
                posterior_samples=self.config.posterior_samples,
                thinning=self.config.thinning,
                num_chains=self.config.num_chains,
                max_tree_depth=self.config.max_tree_depth,
                target_accept_prob=self.config.target_accept_prob,
                verbose=self.verbose,
            ).fit(self.state.X_train_unit.detach().cpu().numpy(), standardized)
        except ImportError as error:
            raise SAASBOMCMCError(str(error)) from error
        except Exception as error:
            raise SAASBOMCMCError(
                f"NumPyro NUTS failed before producing a posterior: {error}"
            ) from error
        samples = {
            name: torch.tensor(np.asarray(value).copy(), dtype=self.dtype, device=self.device)
            for name, value in model.samples.items()
        }
        problem = self._posterior_sample_problem(samples)
        if problem is not None:
            raise SAASBOMCMCError(problem)
        self._check_mcmc_diagnostics(model.diagnostics)
        self.state.posterior_hyperparameter_samples = samples
        self.state.mcmc_diagnostics = model.diagnostics
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

        from basat.models.Acquisitions import optimize_acquisition_discrete

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

        if self.config.inference_backend == "numpyro" and "candidate_actions" in round_info:
            raise SAASBOMCMCError("Released NumPyro SAAS requires a continuous bounded domain.")
        if self.config.source_protocol == "mcts_vs_2022" and "candidate_actions" not in round_info:
            return self._ask_source_component(round_info=round_info)
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
                        from basat.models.Acquisitions import (
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

    def _ask_source_component(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Released E1 RBF-SaasGP and ranked pointwise EI over 1,024 points."""
        count = self._initial_count()
        rows = self.state.y_train.numel()
        if self._source_queue is not None and self._source_queue.shape[0]:
            points = self._source_queue[: self.request_size]
            self._source_queue = self._source_queue[self.request_size :]
            actions = self._from_unit(points)
            self._record_last_ask(actions=actions, model_inputs=points, round_info=round_info)
            return actions[0] if self.request_size == 1 else actions
        if rows < count and self.config.inference_backend == "numpyro":
            # The release constructs one full NumPy LHS before observing any
            # feedback. Reconstructing it from the immutable seed preserves its
            # exact arithmetic and does not consume the acquisition stream.
            rng = np.random.RandomState(self.config.seed)
            centers = (1.0 + 2.0 * np.arange(0.0, count)) / float(2 * count)
            design_np = np.zeros((count, self.action_dimension))
            for dimension in range(self.action_dimension):
                design_np[:, dimension] = centers[rng.permutation(count)]
            design_np += rng.uniform(-1.0, 1.0, design_np.shape) / float(2 * count)
            points = torch.tensor(
                design_np[rows : rows + self.request_size], dtype=self.dtype, device=self.device
            )
        elif rows < count:
            generator = torch.Generator(device=self.device).manual_seed(self.config.seed)
            design = torch.empty(count, self.action_dimension, dtype=self.dtype, device=self.device)
            for dimension in range(self.action_dimension):
                design[:, dimension] = torch.randperm(
                    count, generator=generator, device=self.device
                )
            design = (
                design
                + torch.rand(
                    design.shape, generator=generator, dtype=self.dtype, device=self.device
                )
            ) / count
            points = design[rows : rows + self.request_size]
            if points.shape[0] < self.request_size:
                points = torch.cat(
                    (
                        points,
                        torch.rand(
                            (self.request_size - points.shape[0], self.action_dimension),
                            generator=generator,
                            dtype=self.dtype,
                            device=self.device,
                        ),
                    )
                )
        elif self.config.inference_backend == "numpyro":
            from scipy.stats import qmc

            model = self._fit_posterior()
            # The original uses SciPy Sobol with seed=None. An owned MT19937
            # seed stream makes that otherwise entropy-driven step replayable;
            # it is recorded as an explicit reproducibility choice.
            candidates = qmc.Sobol(
                self.action_dimension, scramble=True, seed=self._source_sobol_rng
            ).random(1024)
            selected = model.select_candidates(candidates, count=3)
            points = torch.tensor(selected, dtype=self.dtype, device=self.device)
            self._source_queue = points[self.request_size :].detach().clone()
            points = points[: self.request_size]
        else:
            model = self._fit_posterior()
            points = (
                torch.quasirandom.SobolEngine(
                    self.action_dimension, scramble=True, seed=self.config.seed + self.iteration
                )
                .draw(1024)
                .to(dtype=self.dtype, device=self.device)
            )
            with torch.no_grad():
                posterior = model.posterior(points, observation_noise=True)
                mean = posterior.mean.squeeze(-1)
                std = posterior.variance.squeeze(-1).clamp_min(1e-12).sqrt()
                target = self._standardized_outcomes().max()
                # The released optimize_ei_random omits xi when calling ei;
                # the effective offset is zero, despite its unused xi argument.
                improvement = mean - target
                z = improvement / std
                values = (
                    improvement * (0.5 * (1 + torch.erf(z / math.sqrt(2))))
                    + std * torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)
                ).mean(0)
            points = points[torch.argsort(values, descending=True)[: max(3, self.request_size)]]
            self._source_queue = points[self.request_size :].detach().clone()
            points = points[: self.request_size]
        actions = self._from_unit(points)
        self._record_last_ask(actions=actions, model_inputs=points, round_info=round_info)
        return actions[0] if self.request_size == 1 else actions

    @property
    def source_iteration_complete(self) -> bool:
        """Whether the released initial design or queued EI triple is complete.

        A nominal 600-call run can stop after call 601 under a
        ``finish_iteration`` policy: ten initial points plus 197 triples.
        Pending feedback is never an iteration boundary.
        """
        if self.last_ask is not None:
            return False
        if self.config.source_protocol != "mcts_vs_2022":
            return True
        rows = self.state.y_train.numel()
        if 0 < rows < self._initial_count():
            return False
        return self._source_queue is None or not self._source_queue.shape[0]

    def _invalidate_posterior(self) -> None:
        """Discard posterior draws and standardization derived from stale data."""

        had_model = self._model is not None
        self.state.posterior_hyperparameter_samples = {}
        self.state.mcmc_diagnostics = {}
        self.state.posterior_valid = False
        self.state.outcome_mean = None
        self.state.outcome_std = None
        self._model = None
        if had_model and self.release_inference_cache:
            import jax

            started = time.perf_counter()
            jax.clear_caches()
            gc.collect()
            if self.state.fit_diagnostics_history:
                self.state.fit_diagnostics_history[-1]["inference_cache_released"] = True
                self.state.fit_diagnostics_history[-1]["cache_release_seconds"] = (
                    time.perf_counter() - started
                )

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
            "fit_diagnostics_count": len(self.state.fit_diagnostics_history),
            "last_fit_diagnostics": deepcopy(self.state.fit_diagnostics_history[-1])
            if self.state.fit_diagnostics_history
            else None,
            "shrinkage_scale": self.config.shrinkage_scale,
            "diagnostic_policy": self.diagnostic_policy,
            "jit_compile": self.jit_compile,
            "mcmc_num_threads": self.mcmc_num_threads,
            "release_inference_cache": self.release_inference_cache,
            "inference_backend": self.config.inference_backend,
            "source_iteration_complete": self.source_iteration_complete,
            "source_numerical_policy": (
                "numpyro-rbf-prng0-scipy-sobol-v1"
                if self.config.inference_backend == "numpyro"
                else None
            ),
            "source_sobol_seed_policy": (
                "algorithm-owned MT19937; released seed=None replaced for replay"
                if self.config.inference_backend == "numpyro"
                else None
            ),
            "outcome_standardization": (
                "population (ddof=0)"
                if self.config.source_protocol == "mcts_vs_2022"
                else "sample (ddof=1)"
            ),
        }

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize observations and progress, excluding trained posterior draws."""

        return {
            "config": asdict(self.config),
            "source_queue": self._source_queue,
            "source_sobol_rng_state": deepcopy(self._source_sobol_rng.get_state()),
            "source_numerical_policy": (
                "numpyro-rbf-prng0-scipy-sobol-v1"
                if self.config.inference_backend == "numpyro"
                else None
            ),
            "diagnostic_policy": self.diagnostic_policy,
            "jit_compile": self.jit_compile,
            "mcmc_num_threads": self.mcmc_num_threads,
            "release_inference_cache": self.release_inference_cache,
            "X_train_unit": self.state.X_train_unit,
            "y_train": self.state.y_train,
            "mcmc_diagnostics": self.state.mcmc_diagnostics,
            "fit_generation": self.state.fit_generation,
            "fit_diagnostics_history": deepcopy(self.state.fit_diagnostics_history),
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
        if not isinstance(state.get("release_inference_cache", False), bool):
            raise SAASBOCheckpointError("Stored SAAS cache-release runtime policy is invalid.")
        if self.config.inference_backend == "numpyro":
            if state.get("source_numerical_policy") != "numpyro-rbf-prng0-scipy-sobol-v1":
                raise SAASBOCheckpointError("NumPyro SAAS checkpoint source policy does not match.")
            try:
                probe = np.random.RandomState()
                probe.set_state(state["source_sobol_rng_state"])
            except (KeyError, TypeError, ValueError) as error:
                raise SAASBOCheckpointError("Invalid NumPyro SAAS Sobol RNG state.") from error
        queue = state.get("source_queue")
        if queue is not None:
            queue = torch.as_tensor(queue)
            if (
                queue.ndim != 2
                or queue.shape[1] != self.action_dimension
                or not torch.isfinite(queue).all()
                or torch.any((queue < 0) | (queue > 1))
            ):
                raise SAASBOCheckpointError("Invalid source SAAS pending candidate queue.")
        X = torch.as_tensor(state["X_train_unit"])
        y = torch.as_tensor(state["y_train"])
        if X.ndim != 2 or X.shape[1] != self.action_dimension:
            raise SAASBOCheckpointError("Stored SAAS model inputs have an invalid shape.")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise SAASBOCheckpointError("Stored SAAS outcomes are not row-aligned.")
        if torch.any(X < 0) or torch.any(X > 1):
            raise SAASBOCheckpointError("Stored SAAS model inputs leave the unit cube.")
        history = state.get("fit_diagnostics_history", [])
        if not isinstance(history, list):
            raise SAASBOCheckpointError("Stored SAAS fit diagnostics history must be a list.")
        for record in history:
            if (
                not isinstance(record, Mapping)
                or not isinstance(record.get("training_rows"), int)
                or isinstance(record["training_rows"], bool)
                or not 0 <= record["training_rows"] <= len(y)
                or record.get("backend") not in {"pyro", "numpyro"}
                or record.get("status") not in {"completed", "failed", "interrupted"}
                or not isinstance(record.get("diagnostics"), Mapping)
                or not isinstance(record.get("diagnostic_events"), list)
            ):
                raise SAASBOCheckpointError("Stored SAAS fit diagnostics record is invalid.")

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Restore observations and force fresh posterior inference."""

        del strict
        queue = state.get("source_queue")
        self._source_queue = (
            None
            if queue is None
            else torch.as_tensor(queue, dtype=self.dtype, device=self.device).clone()
        )
        self._source_sobol_rng = np.random.RandomState(self.config.seed)
        if "source_sobol_rng_state" in state:
            self._source_sobol_rng.set_state(state["source_sobol_rng_state"])
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
            fit_diagnostics_history=deepcopy(state.get("fit_diagnostics_history", [])),
        )
        self._model = None
        self._active_fit_record = None

    def _reset_impl(self) -> None:
        """Clear training data, posterior draws, diagnostics, and cached model."""

        self.state = self._fresh_state()
        self._source_queue = None
        self._source_sobol_rng = np.random.RandomState(self.config.seed)
        self._model = None
        self._active_fit_record = None


__all__ = ["SAASBO", "SAASBOConfig", "SAASBOState"]
