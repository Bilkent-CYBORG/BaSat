"""Robust-satisficing GP bandits under externally applied attacks.

One implementation exposes the RS1, RS2, and generalized RSG variants.  The
algorithm proposes a nominal action; the adversarial environment owns the
attack and returns the actual evaluated coordinate used to update the GP.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import Any, ClassVar, Literal

import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances

from basat.models.Acquisitions import DiscreteAcquisitionCandidates
from basat.models.GaussianProcess import GaussianProcess, GaussianProcessConfig
from basat.models.FixedKernelGaussianProcess import FixedKernelGaussianProcess
from basat.models.Kernels import KernelConfig
from basat.models.PolynomialFeatureGaussianProcess import PolynomialFeatureGaussianProcess
from basat.models.utils import Bounds
from basat.workflow import RandomStream

from ..algorithm import Algorithm, AlgorithmConfigurationError, resolve_batch_configuration
from ..helper import SourceRecord
from .Exception.exception_AdveRS import (
    AdveRSCheckpointError,
    AdveRSException,
    AdveRSInnerSolverError,
    AdveRSPerturbationError,
    AdveRSVariantError,
    AdveRSWarning,
)
from .StableOPT import MultiStartSolver, theoretical_beta


@dataclass(frozen=True, slots=True)
class PerturbationDomainSpec:
    """Explicit perturbation geometry for AdveRS inner problems.

    Parameters
    ----------
    radius:
        Optional non-negative radius in physical ``D_env`` coordinates.
        ``None`` makes the entire bounded action domain reachable.
    norm:
        ``"l2"`` or ``"linf"`` distance used by robust inner programs.
    distance_floor:
        Positive denominator floor used when a fragility ratio approaches the
        nominal action.
    """

    radius: float | None = None
    norm: Literal["l2", "linf"] = "l2"
    distance_floor: float = 1e-8


@dataclass(frozen=True, slots=True)
class FullDomainPerturbations(PerturbationDomainSpec):
    """Allow every feasible point in the configured action box.

    Attributes
    ----------
    radius:
        Fixed to ``None`` so the inner program searches the full action box.
    norm:
        Distance used by RS1, RS2, or RSG scores.
    distance_floor:
        Positive numerical floor for distance-based ratios.
    """

    radius: None = field(default=None, init=False)


@dataclass(frozen=True, slots=True)
class BallPerturbations(PerturbationDomainSpec):
    """Restrict perturbations to a finite-radius norm ball.

    Parameters
    ----------
    radius:
        Non-negative ball radius in physical action coordinates.
    norm:
        ``"l2"`` or ``"linf"`` distance defining the ball.
    distance_floor:
        Positive numerical floor for distance-based ratios.
    """

    radius: float = 0.1


SolverSpec = MultiStartSolver


_FEASIBILITY_TOLERANCE = 1e-8
_INNER_REFINEMENTS = 5
_CANDIDATE_REFINEMENTS = 10
_GP_RETRY_SEED_OFFSET = 104_729
_GP_RETAINED_HYPERPARAMETERS_REASON = "advers.gp_mle_retained_hyperparameters"
_SOURCE_PROTOCOL_VERSIONS = {"advers_2025": 1, "advers_2026_camera_ready": 2}


@dataclass(frozen=True, slots=True)
class AdveRSConfig:
    """Shared configuration for RS1, RS2, and generalized RSG.

    Parameters
    ----------
    variant:
        Robust-satisficing decision rule: ``"RS1"``, ``"RS2"``, or
        generalized ``"RSG"``.
    aspiration:
        Finite reward threshold that defines whether an action satisfies the
        robust objective.
    p:
        Positive RSG distance exponent. RS1 and RS2 require ``p=1``.
    beta:
        Non-negative scalar or round-indexed callable supplying the GP
        confidence multiplier.
    perturbation_domain:
        Reachable attack geometry in physical ``D_env`` coordinates.
    candidate_solver:
        Multistart solver used to select the outer nominal action.
    inner_solver:
        Multistart solver used to find worst or threshold-crossing points.
    decision_model:
        ``"ucb"`` decision surface, or ``"thompson"`` for RSG only.
    outer_raw_samples:
        Sobol candidates used to initialize outer action optimization.
    inner_raw_samples:
        Sobol candidates used to initialize robust inner optimization.
    radius_bisection_steps:
        Fixed bisection count for RS2 critical-radius calculations.
    solver_tolerance:
        Optimizer termination tolerance; feasibility is certified separately.
    initial_points:
        Sequential observations collected before model-based selection.
    fit_maxiter:
        Maximum GP hyperparameter-optimization iterations per refit. Set zero
        to condition on observations with fixed configured hyperparameters.
    kernel:
        Kernel settings for the GP over evaluated physical actions.
    gp_config:
        Optional complete surrogate settings, taking precedence over ``kernel``.
        Supports fixed observation noise, zero mean and unstandardized outcomes
        for reproducing a fixed-parameter paper protocol.
    finite_tie_break:
        ``"nominal"`` favors larger nominal UCB within RS2 radius ties.
        ``"first"`` selects the first tied finite row as in the author code.
    gp_backend:
        ``"standard"`` uses the general Gaussian process. ``"polynomial_features"``
        uses an exact finite polynomial basis with QR conditioning; it requires
        a fixed polynomial kernel, zero mean, unstandardized outcomes and
        ``fit_maxiter=0``. It is not a low-rank approximation.
        ``"fixed_gpy"`` explicitly uses fixed full CPU float64 conditioning
        with GPy-compatible LAPACK/marginal arithmetic and finite candidates;
        it requires explicit ``gp_config`` and ``fit_maxiter=0``.
    gp_training_regularization:
        Nonnegative training-only diagonal for ``"fixed_gpy"``, distinct from
        observation noise and conditional jitter. Other backends require zero.
    source_protocol:
        ``"native"`` retains the paper equations. ``"advers_2025"`` explicitly
        reproduces the released finite runner: standard deviation for the
        initial design, posterior variance thereafter, squared distance for
        RS2/RSG, and exact source ties. This inconsistent confidence rule does
        not inherit the paper's guarantees. Historical initialization with more
        than ``initial_points`` rows is treated as post-feedback state.
        ``"advers_2026_camera_ready"`` follows revision ``6de58579``: standard
        deviation at every round, Euclidean RS1/RSG distances, squared RS2
        distances, and the updated random fallback when every point satisfies
        the aspiration. ``beta`` is the squared confidence multiplier in both
        source protocols, so source ``beta=1`` corresponds to this ``beta=1``.
    batch_size:
        Requested proposal count. AdveRS version 1 requires ``1``.
    seed:
        Seed for algorithm-owned sampling and replayable decisions.
    """

    variant: Literal["RS1", "RS2", "RSG"]
    aspiration: float
    p: float = 1.0
    beta: Callable[[int], float] | float = theoretical_beta
    perturbation_domain: PerturbationDomainSpec = field(default_factory=FullDomainPerturbations)
    candidate_solver: SolverSpec = field(default_factory=MultiStartSolver)
    inner_solver: SolverSpec = field(default_factory=MultiStartSolver)
    decision_model: Literal["ucb", "thompson"] = "ucb"
    outer_raw_samples: int = 512
    inner_raw_samples: int = 256
    radius_bisection_steps: int = 30
    solver_tolerance: float = 1e-6
    initial_points: int = 1
    fit_maxiter: int = 100
    kernel: KernelConfig = field(default_factory=KernelConfig)
    gp_config: GaussianProcessConfig | None = None
    gp_backend: Literal["standard", "polynomial_features", "fixed_gpy"] = "standard"
    gp_training_regularization: float = 0.0
    finite_tie_break: Literal["nominal", "first"] = "nominal"
    source_protocol: Literal["native", "advers_2025", "advers_2026_camera_ready"] = "native"
    batch_size: int | None = 1
    q_batch: int | None = None
    seed: int = 1


@dataclass(slots=True)
class AdveRSState:
    """Mutable proposal, evaluation, and robust-satisficing history.

    Attributes
    ----------
    actions_train:
        GP input coordinates returned as evaluated actions, with shape
        ``(N, D_env)``.
    y_train:
        Learner-visible observations with shape ``(N,)``.
    observation_noise_variance:
        Known observation-noise variances with shape ``(N,)``; zeros represent
        feedback without a supplied variance.
    proposals:
        Nominal actions requested from the adversarial environment, each with
        shape ``(D_env,)``.
    evaluated_points:
        Actual learner-visible evaluation coordinates, each with shape
        ``(D_env,)``.
    external_perturbations:
        Public perturbation vectors or ``None`` when feedback omits them;
        latent environment attacks are never accessed.
    realized_distances:
        Distances from nominal proposals to evaluated points.
    optimistic_scores:
        Variant-specific scores used for the corresponding decisions.
    pessimistic_scores:
        Scores recomputed from the posterior lower-confidence surface after
        each observation.
    worst_points:
        Inner-solver certificate points associated with proposals, each with
        shape ``(D_env,)``.
    aspiration_satisfied:
        Whether each learner-visible observation met ``aspiration``.
    """

    actions_train: torch.Tensor
    y_train: torch.Tensor
    observation_noise_variance: torch.Tensor
    proposals: list[torch.Tensor]
    evaluated_points: list[torch.Tensor]
    external_perturbations: list[torch.Tensor | None]
    realized_distances: list[float]
    optimistic_scores: list[float]
    pessimistic_scores: list[float]
    worst_points: list[torch.Tensor]
    aspiration_satisfied: list[bool]


class AdveRS(Algorithm):
    """Adversarial robust-satisficing optimizer for RS1, RS2, or RSG.

    Examples
    --------
    >>> import basat
    >>> bounds = basat.Bounds(lower=[-1.0, -1.0], upper=[1.0, 1.0])
    >>> algorithm = basat.AdveRS(
    ...     bounds=bounds,
    ...     config=basat.AdveRSConfig(
    ...         variant="RS1", aspiration=0.0, beta=2.0, initial_points=3, seed=7
    ...     ),
    ... )
    """

    ACQUISITION_POLICY = "robust-satisficing UCB or Thompson decision rule"
    CHECKPOINT_ADDED_CONFIG_DEFAULTS: ClassVar[Mapping[str, Any]] = {
        **Algorithm.CHECKPOINT_ADDED_CONFIG_DEFAULTS,
        "gp_config": None,
        "gp_backend": "standard",
        "gp_training_regularization": 0.0,
        "finite_tie_break": "nominal",
        "source_protocol": "native",
    }

    EXCEPTION_CLASS = AdveRSException
    native_batching = False
    native_environments = frozenset({"adversarial"})
    required_capabilities = frozenset({"adversarial", "perturbed_actions"})
    SOURCE = SourceRecord(
        paper_title="Robust Satisficing Gaussian Process Bandits Under Adversarial Attacks",
        paper_year=2025,
        paper_url="https://proceedings.neurips.cc/paper_files/paper/2025/hash/4830a9b95a2f63fc4b3fe09abc18f045-Abstract-Conference.html",
        repository_url="https://github.com/Bilkent-CYBORG/AdveRS",
        repository_status="official",
        notes="RS1, RS2, and RSG share one Basat class selected by variant.",
    )

    def __init__(
        self,
        *,
        bounds: Bounds,
        config: AdveRSConfig,
        maximize: bool = True,
        dtype: torch.dtype | str = torch.float32,
        device: torch.device | str = "cpu",
        plot: bool = False,
        verbose: bool = False,
        verbose_setting: bool = False,
        verbose_iter_by_iter: bool = False,
    ) -> None:
        """Initialize one sequential robust-satisficing variant.

        Parameters
        ----------
        bounds:
            Physical action box with ``D_env`` coordinates.
        config:
            Immutable RS1, RS2, or RSG model, solver, and seed settings.
        maximize:
            Optimization direction. AdveRS v1 accepts only ``True``.
        dtype, device:
            Precision and device for all tensors and GP state.
        plot, verbose, verbose_setting, verbose_iter_by_iter:
            Presentation flags inherited from :class:`Algorithm`; they do not
            alter mathematical state.

        Raises
        ------
        AlgorithmConfigurationError
            If common or sequential configuration is invalid.
        AdveRSVariantError
            If variant, exponent, or decision-model settings conflict.
        AdveRSPerturbationError
            If the perturbation domain is invalid.
        AdveRSInnerSolverError
            If a configured solver is unsupported.
        """

        self.config = config
        self._validate_config(config)
        if not maximize:
            raise AlgorithmConfigurationError("AdveRS version 1 supports maximization only.")
        super().__init__(
            bounds=bounds,
            batch_size=config.batch_size,
            q_batch=config.q_batch,
            maximize=True,
            dtype=dtype,
            device=device,
            seed=config.seed,
            plot=plot,
            verbose=verbose,
            verbose_setting=verbose_setting,
            verbose_iter_by_iter=verbose_iter_by_iter,
        )
        model_class = (PolynomialFeatureGaussianProcess
                       if config.gp_backend == "polynomial_features" else GaussianProcess)
        gp_config = (replace(config.gp_config, dtype=self.dtype, device=self.device)
            if config.gp_config is not None
            else GaussianProcessConfig(kernel=config.kernel, dtype=self.dtype, device=self.device))
        self.gp = (FixedKernelGaussianProcess(gp_config, input_dimension=self.action_dimension,
            conditioning="full", arithmetic="gpy", training_regularization=config.gp_training_regularization)
            if config.gp_backend == "fixed_gpy" else model_class(gp_config, input_dimension=self.action_dimension))
        self.state = self._empty_state()
        self._source_rng = RandomStream(self.seed % 2**32)
        if config.source_protocol in _SOURCE_PROTOCOL_VERSIONS:
            if self.dtype != torch.float64 or self.device.type != "cpu":
                raise AlgorithmConfigurationError("The released AdveRS protocol requires CPU float64.")
        if config.source_protocol == "advers_2025":
            self.invalidate_source_guarantees("basat.advers.released_2025_variance_confidence")
            warnings.warn(
                "Basat AdveRS released-source protocol uses posterior variance as confidence "
                "width after the initial design and squared RS2/RSG distance. These are "
                "source-reproduction settings, not the paper's confidence guarantees.",
                AdveRSWarning, stacklevel=2,
            )

    def prepare_feedback(self, feedback: Mapping[str, Any], *, action: Any) -> dict[str, Any]:
        result = super().prepare_feedback(feedback, action=action)
        if result.get("perturbation") is None:
            nominal = torch.as_tensor(action, dtype=self.dtype, device=self.device)
            result["perturbation"] = (
                torch.as_tensor(result["evaluated_action"], dtype=self.dtype, device=self.device)
                - nominal
            )
            self._warn_environment_default(
                "perturbation",
                "evaluated_action - nominal_action (zero without an external attack)",
            )
        return result

    @staticmethod
    def _validate_config(config: AdveRSConfig) -> None:
        """Validate variant, perturbation geometry, and solver invariants."""

        if not isinstance(config, AdveRSConfig):
            raise AlgorithmConfigurationError("config must be AdveRSConfig.")
        if config.source_protocol not in {"native", *_SOURCE_PROTOCOL_VERSIONS}:
            raise AlgorithmConfigurationError("Unknown AdveRS source_protocol.")
        if config.source_protocol in _SOURCE_PROTOCOL_VERSIONS and (
            config.decision_model != "ucb" or config.perturbation_domain.radius is not None
            or config.perturbation_domain.norm != "l2"
        ):
            raise AlgorithmConfigurationError(
                "The released AdveRS protocol requires UCB and full-domain Euclidean geometry."
            )
        if config.gp_config is not None and not isinstance(config.gp_config, GaussianProcessConfig):
            raise AlgorithmConfigurationError("gp_config must be GaussianProcessConfig or None.")
        if config.gp_backend not in {"standard", "polynomial_features", "fixed_gpy"}:
            raise AlgorithmConfigurationError("Unknown gp_backend.")
        regularization = config.gp_training_regularization
        if isinstance(regularization, bool) or not isinstance(regularization, Real) or not math.isfinite(float(regularization)) or regularization < 0:
            raise AlgorithmConfigurationError("gp_training_regularization must be finite and nonnegative.")
        if config.gp_backend != "fixed_gpy" and regularization:
            raise AlgorithmConfigurationError("Explicit GP regularization requires gp_backend='fixed_gpy'.")
        if config.gp_backend == "fixed_gpy" and (config.fit_maxiter != 0 or config.gp_config is None):
            raise AlgorithmConfigurationError("fixed_gpy requires explicit gp_config and fit_maxiter=0.")
        if config.gp_backend == "polynomial_features" and config.fit_maxiter != 0:
            raise AlgorithmConfigurationError("Polynomial features require fit_maxiter=0.")
        batch_size, q_batch = resolve_batch_configuration(
            batch_size=config.batch_size, q_batch=config.q_batch
        )
        if config.source_protocol in _SOURCE_PROTOCOL_VERSIONS and batch_size != 1:
            raise AlgorithmConfigurationError("The released AdveRS protocol is sequential (batch_size=1).")
        if q_batch is not None and q_batch > 1:
            raise AlgorithmConfigurationError(
                "AdveRS does not define a joint q-acquisition; use batch_size for repeated "
                "independent robust-satisficing decisions."
            )
        if config.variant not in {"RS1", "RS2", "RSG"}:
            raise AdveRSVariantError("variant must be RS1, RS2, or RSG.")
        if not math.isfinite(config.aspiration):
            raise AlgorithmConfigurationError("aspiration must be finite.")
        if not math.isfinite(config.p) or config.p <= 0:
            raise AdveRSVariantError("p must be finite and positive.")
        if config.variant in {"RS1", "RS2"} and config.p != 1.0:
            raise AdveRSVariantError("RS1 and RS2 require p=1.")
        if config.decision_model not in {"ucb", "thompson"}:
            raise AdveRSVariantError("decision_model must be ucb or thompson.")
        if config.decision_model == "thompson" and config.variant != "RSG":
            raise AdveRSVariantError("Thompson decisions are supported only for RSG.")
        if (
            not isinstance(config.candidate_solver, MultiStartSolver)
            or config.candidate_solver.method != "multistart"
        ):
            raise AdveRSInnerSolverError(
                "candidate_solver must select the deterministic multistart solver."
            )
        if (
            not isinstance(config.inner_solver, MultiStartSolver)
            or config.inner_solver.method != "multistart"
        ):
            raise AdveRSInnerSolverError(
                "inner_solver must select the deterministic multistart solver."
            )
        domain = config.perturbation_domain
        if domain.norm not in {"l2", "linf"}:
            raise AdveRSPerturbationError("Perturbation norm must be l2 or linf.")
        if domain.radius is not None and (not math.isfinite(domain.radius) or domain.radius < 0):
            raise AdveRSPerturbationError("Perturbation radius must be non-negative.")
        if not math.isfinite(domain.distance_floor) or domain.distance_floor <= 0:
            raise AdveRSPerturbationError("distance_floor must be finite and positive.")
        for name in (
            "outer_raw_samples",
            "inner_raw_samples",
            "radius_bisection_steps",
            "initial_points",
            "fit_maxiter",
        ):
            value = getattr(config, name)
            minimum = 0 if name == "fit_maxiter" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise AlgorithmConfigurationError(f"{name} must be an integer >= {minimum}.")
        if not math.isfinite(config.solver_tolerance) or config.solver_tolerance <= 0:
            raise AlgorithmConfigurationError("solver_tolerance must be finite and positive.")
        if config.finite_tie_break not in {"nominal", "first"}:
            raise AlgorithmConfigurationError("finite_tie_break must be 'nominal' or 'first'.")

    def _empty_state(self) -> AdveRSState:
        """Create empty proposal and evaluated-action state in ``D_env``."""

        return AdveRSState(
            actions_train=torch.empty(
                (0, self.action_dimension), dtype=self.dtype, device=self.device
            ),
            y_train=torch.empty(0, dtype=self.dtype, device=self.device),
            observation_noise_variance=torch.empty(0, dtype=self.dtype, device=self.device),
            proposals=[],
            evaluated_points=[],
            external_perturbations=[],
            realized_distances=[],
            optimistic_scores=[],
            pessimistic_scores=[],
            worst_points=[],
            aspiration_satisfied=[],
        )

    def _initialize_from_environment_impl(self, data: Any) -> None:
        """Initialize the GP from public historical evaluated coordinates.

        For adversarial history, the requested action is not necessarily the
        observation location. The GP therefore consumes ``evaluated_actions``
        when provided and never reads latent attack state.
        """

        requested = torch.as_tensor(data.actions, dtype=self.dtype, device=self.device)
        evaluated_raw = getattr(data, "evaluated_actions", None)
        evaluated = (
            requested
            if evaluated_raw is None
            else torch.as_tensor(evaluated_raw, dtype=self.dtype, device=self.device)
        )
        y = torch.as_tensor(data.observations, dtype=self.dtype, device=self.device).reshape(-1)
        raw_variance = getattr(data, "observation_noise_variance", None)
        variance = (
            torch.zeros_like(y)
            if raw_variance is None
            else torch.as_tensor(raw_variance, dtype=self.dtype, device=self.device).reshape(-1)
        )
        self.state.actions_train = evaluated.detach().clone()
        self.state.y_train = y.detach().clone()
        self.state.observation_noise_variance = variance.detach().clone()
        if y.numel():
            self.gp.set_data(evaluated, y, variance)

    def _prepare_discrete_candidates_impl(
        self,
        *,
        candidate_actions: torch.Tensor,
        candidate_dataset_indices: torch.Tensor,
        round_info: Mapping[str, Any],
    ) -> DiscreteAcquisitionCandidates:
        """Preserve finite rows under AdveRS's identity coordinate map."""

        del round_info
        return DiscreteAcquisitionCandidates(
            candidate_actions.detach().clone(),
            candidate_actions.detach().clone(),
            candidate_dataset_indices.detach().clone(),
        )

    def _beta(self) -> float:
        """Resolve and validate the current nonnegative confidence multiplier."""

        raw = (
            self.config.beta(self.iteration + 1) if callable(self.config.beta) else self.config.beta
        )
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise AdveRSInnerSolverError("beta must resolve to a finite non-negative scalar.")
        return value

    def _gp_training_state_is_current(self) -> bool:
        """Return whether the fitted GP already represents the committed table."""

        return bool(
            self.gp.is_fit
            and self.gp.X_train is not None
            and self.gp.y_train is not None
            and self.gp.observation_noise is not None
            and torch.equal(self.gp.X_train, self.state.actions_train)
            and torch.equal(self.gp.y_train, self.state.y_train)
            and torch.equal(
                self.gp.observation_noise,
                self.state.observation_noise_variance,
            )
        )

    def _current_gp_model_state(self) -> dict[str, torch.Tensor] | None:
        """Copy fitted module values for a same-model MLE warm start."""

        if self.gp.model is None or not self.gp.is_fit:
            return None
        return {name: value.detach().clone() for name, value in self.gp.model.state_dict().items()}

    def _restore_compatible_gp_model_state(
        self, model_state: Mapping[str, torch.Tensor] | None
    ) -> None:
        """Restore shape-compatible module values after rebuilding training data."""

        if model_state is None or self.gp.model is None:
            return
        current = self.gp.model.state_dict()
        for name, value in model_state.items():
            if name in current and current[name].shape == value.shape:
                current[name] = value.to(
                    dtype=current[name].dtype,
                    device=current[name].device,
                )
        self.gp.model.load_state_dict(current, strict=False)

    @contextmanager
    def _owned_torch_rng(self, *, seed: int) -> Iterator[None]:
        """Seed GP fitting without mutating the caller's ambient RNG state."""

        cuda_devices: list[int] = []
        if self.device.type == "cuda":
            cuda_devices.append(
                torch.cuda.current_device() if self.device.index is None else self.device.index
            )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            yield

    def _initialize_response_scaled_gp(self) -> None:
        """Create a deterministic data-scaled start for the unchanged exact MLE."""

        model = self.gp.model
        if model is None:
            return
        targets = self.state.y_train
        mean = targets.mean()
        variance = (targets - mean).square().mean()
        signal_variance = max(
            float(variance.item()),
            float(self.gp.config.jitter),
            1e-4,
        )
        scale_modules = [
            module
            for module in model.modules()
            if hasattr(module, "raw_outputscale") and hasattr(module, "outputscale")
        ]
        scale_value = signal_variance ** (1.0 / max(len(scale_modules), 1))
        with torch.no_grad():
            mean_module = model.mean_module
            if hasattr(mean_module, "raw_constant"):
                mean_module.raw_constant.copy_(mean.to(mean_module.raw_constant))
            for module in scale_modules:
                module.outputscale = torch.as_tensor(
                    scale_value,
                    dtype=self.dtype,
                    device=self.device,
                )
            for module in model.modules():
                if not hasattr(module, "raw_noise") or not hasattr(module, "noise"):
                    continue
                constraint = getattr(module, "raw_noise_constraint", None)
                lower_bound = 0.0
                if constraint is not None and hasattr(constraint, "lower_bound"):
                    lower_bound = float(
                        torch.as_tensor(constraint.lower_bound).reshape(-1)[0].item()
                    )
                noise_value = max(
                    lower_bound + float(self.gp.config.jitter),
                    1e-4 * signal_variance,
                    10.0 * float(self.gp.config.jitter),
                )
                module.noise = torch.as_tensor(
                    noise_value,
                    dtype=self.dtype,
                    device=self.device,
                )

    def _retain_previous_hyperparameters(
        self,
        model_state: Mapping[str, torch.Tensor],
        fit_errors: list[Exception],
    ) -> None:
        """Certify and retain the preceding fit on an expanded training table."""

        self.gp.set_data(
            self.state.actions_train,
            self.state.y_train,
            self.state.observation_noise_variance,
        )
        self._restore_compatible_gp_model_state(model_state)
        if self.gp.model is None or self.gp.likelihood is None:
            raise RuntimeError("The rebuilt GP is unavailable for fallback certification.")
        if any(not torch.isfinite(parameter).all() for parameter in self.gp.model.parameters()):
            raise RuntimeError("The previous fitted GP contains non-finite parameters.")
        self.gp.model.eval()
        self.gp.likelihood.eval()
        self.gp.is_fit = True
        certificate_points = self.state.actions_train[-min(2, self.state.actions_train.shape[0]) :]
        mean, std = self.gp.predict(certificate_points)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise RuntimeError("The retained GP produced a non-finite posterior certificate.")
        self.gp.last_fit_diagnostics = {
            "method": "mle",
            "success": False,
            "usable_posterior": True,
            "initialization": "retained_previous_hyperparameters",
            "training_rows": int(self.state.actions_train.shape[0]),
            "input_dimension": self.action_dimension,
            "fit_errors": tuple(str(error) for error in fit_errors),
        }
        warnings.warn(
            "AdveRS GP MLE exhausted its warm-started and deterministic "
            "response-scaled attempts. The preceding finite fitted "
            "hyperparameters were retained on the expanded training table after "
            "a finite-posterior certificate; source guarantees no longer apply.",
            AdveRSWarning,
            stacklevel=3,
        )
        self.invalidate_source_guarantees(_GP_RETAINED_HYPERPARAMETERS_REASON)

    def _fit_model(self) -> None:
        """Fit the GP through deterministic, transactional exact-MLE attempts."""

        if self.state.y_train.numel() == 0:
            return
        if self._gp_training_state_is_current():
            # ``tell`` already fitted this exact table. Re-fitting it during the
            # following ``ask`` adds no information and can turn a valid fitted
            # state into a backend optimizer failure.
            return
        if self.config.fit_maxiter == 0:
            self.gp.set_data(
                self.state.actions_train,
                self.state.y_train,
                self.state.observation_noise_variance,
            )
            self.gp.condition_only()
            return
        previous_gp_state = self.gp.state_dict()
        warm_model_state = self._current_gp_model_state()
        fit_errors: list[Exception] = []
        for attempt in range(2):
            try:
                self.gp.set_data(
                    self.state.actions_train,
                    self.state.y_train,
                    self.state.observation_noise_variance,
                )
                if attempt == 0:
                    self._restore_compatible_gp_model_state(warm_model_state)
                else:
                    self._initialize_response_scaled_gp()
                with self._owned_torch_rng(
                    seed=(self.config.seed + self.iteration + attempt * _GP_RETRY_SEED_OFFSET)
                ):
                    self.gp.fit(maxiter=self.config.fit_maxiter)
            except Exception as error:
                fit_errors.append(error)
                continue
            if attempt:
                warnings.warn(
                    "AdveRS GP MLE exhausted its warm-started attempt; the same "
                    "exact MLE succeeded from one deterministic response-scaled "
                    "initialization. The GP, likelihood, robust objective, and "
                    "inner/outer solvers are unchanged.",
                    AdveRSWarning,
                    stacklevel=3,
                )
                self.gp.last_fit_diagnostics["initialization"] = (
                    "deterministic_response_scaled_retry"
                )
            else:
                self.gp.last_fit_diagnostics["initialization"] = "warm_start"
            return

        if warm_model_state is not None:
            try:
                self._retain_previous_hyperparameters(warm_model_state, fit_errors)
                return
            except Exception as fallback_error:
                fit_errors.append(fallback_error)
        try:
            self.gp.load_state_dict(previous_gp_state)
        except Exception as rollback_error:
            raise AdveRSInnerSolverError(
                f"AdveRS GP fitting failed and rollback failed: {rollback_error}",
                algorithm=self,
            ) from rollback_error
        details = "; ".join(
            f"attempt {index}: {error}" for index, error in enumerate(fit_errors, start=1)
        )
        raise AdveRSInnerSolverError(
            f"AdveRS GP fitting failed after deterministic attempts: {details}",
            algorithm=self,
        ) from fit_errors[-1]

    def _distance(self, center: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Measure configured perturbation distances in physical coordinates."""

        if self.config.source_protocol in _SOURCE_PROTOCOL_VERSIONS:
            distances = euclidean_distances(
                points.detach().numpy(), center.detach().numpy().reshape(1, -1),
                squared=(self.config.variant == "RS2" or (
                    self.config.source_protocol == "advers_2025" and self.config.variant == "RSG")),
            ).reshape(-1)
            return torch.as_tensor(distances, dtype=self.dtype, device=self.device)
        difference = points - center
        if self.config.perturbation_domain.norm == "linf":
            return difference.abs().amax(dim=-1)
        return torch.linalg.vector_norm(difference, dim=-1)

    def _normalized_lexicographic_key(self, point: torch.Tensor) -> tuple[float, ...]:
        """Return the stable continuous-domain secondary ordering key."""

        lower, upper = self._bounds_on_algorithm_device()
        normalized = (point - lower) / (upper - lower)
        return tuple(float(value) for value in normalized.detach().cpu().tolist())

    def _feasibility_residual(
        self,
        point: torch.Tensor,
        *,
        center: torch.Tensor | None = None,
        radius: float | None = None,
    ) -> float:
        """Independently recompute box and perturbation constraint residuals."""

        lower, upper = self._bounds_on_algorithm_device()
        residual = max(
            float((lower - point).clamp_min(0).max().item()),
            float((point - upper).clamp_min(0).max().item()),
            0.0,
        )
        if center is not None and radius is not None:
            residual = max(
                residual,
                float(self._distance(center, point.unsqueeze(0))[0].item()) - radius,
            )
        return max(residual, 0.0)

    def _maximum_feasible_radius(self, center: torch.Tensor) -> float:
        """Return the greatest configured distance from ``center`` in the box."""

        lower, upper = self._bounds_on_algorithm_device()
        farthest_coordinate_delta = torch.maximum(center - lower, upper - center)
        if self.config.perturbation_domain.norm == "linf":
            radius = float(farthest_coordinate_delta.max().item())
        else:
            radius = float(torch.linalg.vector_norm(farthest_coordinate_delta).item())
        configured = self.config.perturbation_domain.radius
        return radius if configured is None else min(radius, configured)

    def _continuous_multistart_minimize(
        self,
        *,
        solver: SolverSpec,
        objective: Callable[[torch.Tensor], float],
        raw_points: torch.Tensor,
        raw_values: torch.Tensor,
        restarts: int,
        center: torch.Tensor | None = None,
        radius: float | None = None,
        label: str,
    ) -> tuple[torch.Tensor, float, int]:
        """Refine deterministic starts and certify all returned coordinates."""

        if not isinstance(solver, MultiStartSolver) or solver.method != "multistart":
            raise AdveRSInnerSolverError(
                f"{label} does not support solver {getattr(solver, 'method', None)!r}.",
                algorithm=self,
            )
        finite = torch.nonzero(torch.isfinite(raw_values), as_tuple=False).reshape(-1)
        if finite.numel() == 0:
            raise AdveRSInnerSolverError(
                f"{label} raw search produced no finite objective.", algorithm=self
            )
        order = finite[torch.argsort(raw_values[finite], stable=True)]
        starts = order[: min(restarts, order.numel())]
        candidates: list[tuple[torch.Tensor, float]] = []

        lower, upper = self._bounds_on_algorithm_device()
        if center is not None and radius is not None:
            lower = torch.maximum(lower, center - radius)
            upper = torch.minimum(upper, center + radius)
        if torch.equal(lower, upper):
            point = lower.detach().clone()
            value = float(objective(point))
            residual = self._feasibility_residual(point, center=center, radius=radius)
            if not math.isfinite(value) or residual > _FEASIBILITY_TOLERANCE:
                raise AdveRSInnerSolverError(
                    f"{label} singleton domain failed independent certification.",
                    algorithm=self,
                )
            return point, value, 0
        scipy_bounds = list(
            zip(
                lower.detach().cpu().tolist(),
                upper.detach().cpu().tolist(),
                strict=True,
            )
        )
        constraints: list[dict[str, Any]] = []
        if (
            center is not None
            and radius is not None
            and self.config.perturbation_domain.norm == "l2"
        ):
            center_cpu = center.detach().cpu()

            def ball_constraint(array: Any) -> float:
                """Return the nonnegative Euclidean-ball feasibility slack."""

                point = torch.as_tensor(array, dtype=center_cpu.dtype)
                return float(radius - torch.linalg.vector_norm(point - center_cpu).item())

            constraints.append({"type": "ineq", "fun": ball_constraint})
        try:
            from scipy.optimize import minimize
        except ImportError as error:  # pragma: no cover - SciPy is a core dependency
            raise AdveRSInnerSolverError(
                f"{label} requires SciPy's SLSQP optimizer.", algorithm=self
            ) from error

        call_count = 0
        for position in starts:
            start = raw_points[int(position.item())]

            def scipy_objective(array: Any) -> float:
                """Evaluate the torch objective at one SciPy coordinate vector."""

                point = torch.as_tensor(array, dtype=self.dtype, device=self.device)
                value = float(objective(point))
                return value if math.isfinite(value) else float("inf")

            try:
                result = minimize(
                    scipy_objective,
                    start.detach().cpu().numpy(),
                    method="SLSQP",
                    bounds=scipy_bounds,
                    constraints=constraints,
                    options={"ftol": self.config.solver_tolerance, "maxiter": 200},
                )
            except (RuntimeError, TypeError, ValueError):
                call_count += 1
                continue
            call_count += 1
            point = torch.as_tensor(result.x, dtype=self.dtype, device=self.device)
            if not result.success:
                continue
            value = float(objective(point))
            residual = self._feasibility_residual(point, center=center, radius=radius)
            if math.isfinite(value) and residual <= _FEASIBILITY_TOLERANCE:
                candidates.append((point.detach().clone(), value))

        if not candidates:
            raise AdveRSInnerSolverError(
                f"{label} returned no finite point satisfying independently "
                "recomputed constraints.",
                algorithm=self,
            )
        best_value = min(value for _, value in candidates)
        tolerance = 1e-12 + 1e-9 * max(1.0, abs(best_value))
        tied = [item for item in candidates if item[1] <= best_value + tolerance]
        point, value = min(tied, key=lambda item: self._normalized_lexicographic_key(item[0]))
        return point, value, call_count

    def _inner_points(
        self,
        center: torch.Tensor,
        *,
        explicit: torch.Tensor | None,
        radius: float | None = None,
    ) -> torch.Tensor:
        """Construct the explicit or sampled perturbation neighborhood.

        ``center`` and returned rows are physical ``D_env`` coordinates. In
        finite-data mode only learner-visible candidate coordinates are used;
        observations and latent attacks remain inaccessible. Continuous rows
        are filtered by the configured norm before use.
        """

        lower, upper = self._bounds_on_algorithm_device()
        domain = self.config.perturbation_domain
        effective_radius = domain.radius if radius is None else radius
        if explicit is None:
            if effective_radius is None:
                local = self.bounds.to(dtype=self.dtype, device=self.device)
            elif effective_radius == 0:
                return center.unsqueeze(0)
            else:
                local = Bounds(
                    torch.maximum(lower, center - effective_radius),
                    torch.minimum(upper, center + effective_radius),
                )
            points = self._sample_design(
                local,
                self.config.inner_raw_samples,
                phase="acquisition",
                seed=self.seed + 32452843 * (self.iteration + 1),
            )
            points = torch.cat((center.unsqueeze(0), points), dim=0)
        else:
            # The queried center remains learner-visible after the environment
            # removes its finite row from future availability.
            points = torch.cat((center.unsqueeze(0), explicit), dim=0)
        if effective_radius is not None:
            points = points[
                self._distance(center, points) <= effective_radius + _FEASIBILITY_TOLERANCE
            ]
        if points.shape[0] == 0:
            raise AdveRSPerturbationError(
                "The explicit perturbation neighborhood is empty.", algorithm=self
            )
        return points

    def _optimistic_function(self, points: torch.Tensor) -> torch.Tensor:
        """Evaluate the configured UCB or one joint Thompson function sample."""

        if self.config.decision_model == "thompson":
            return self.gp.sample_posterior(points, 1, seed=self.seed + self.iteration).reshape(-1)
        return self.confidence_bounds(points)[0]

    def _pessimistic_function(self, points: torch.Tensor) -> torch.Tensor:
        """Evaluate the posterior lower confidence bound at physical points."""

        return self.confidence_bounds(points)[1]

    def confidence_bounds(self, points: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Return learner-visible UCB and LCB under the configured protocol.

        An external LCB attacker can use this method without duplicating the
        confidence formula or receiving any hidden objective evaluations.
        The 2025 source protocol uses standard deviation through
        ``initial_points`` training rows, then variance. Native and camera-ready
        source protocols always use standard deviation.
        """
        # Restoring a pending ask restores observations, not a fitted model.
        self._fit_model()
        points = torch.as_tensor(points, dtype=self.dtype, device=self.device)
        if self.config.gp_backend == "fixed_gpy":
            mean, std = self.gp.predict(points.unsqueeze(0) if points.ndim == 1 else points)
        else:
            mean, std = self.gp.predict(points.unsqueeze(-2))
            mean, std = mean.squeeze(-1), std.squeeze(-1)
        if (self.config.source_protocol == "advers_2025"
                and len(self.state.y_train) > self.config.initial_points):
            std = std.square()
        width = math.sqrt(self._beta()) * std
        return mean + width, mean - width

    def get_source_random_state(self) -> dict[str, Any]:
        """Return the source tie RNG in :class:`basat.RandomStream` JSON format."""
        if self.config.source_protocol not in _SOURCE_PROTOCOL_VERSIONS:
            raise AlgorithmConfigurationError("Source RNG requires a released AdveRS source_protocol.")
        return self._source_rng.get_state()

    def set_source_random_state(self, state: Mapping[str, Any]) -> None:
        """Restore or share an MT19937 source stream without consuming a draw."""
        if self.config.source_protocol not in _SOURCE_PROTOCOL_VERSIONS:
            raise AlgorithmConfigurationError("Source RNG requires a released AdveRS source_protocol.")
        self._source_rng.set_state(dict(state))

    def _source_finite_selection(
        self, candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, float, torch.Tensor, float]:
        """Apply the released finite equations and exact NumPy tie rules."""
        values = self._optimistic_function(candidates).detach().numpy().reshape(-1)
        points = candidates.detach().numpy()
        safe, unsafe = np.flatnonzero(values >= self.config.aspiration), np.flatnonzero(
            values < self.config.aspiration)
        if (len(safe) == 0 or (len(unsafe) == 0 and (
                self.config.variant == "RS2"
                or self.config.source_protocol == "advers_2026_camera_ready"))):
            position = int(self._source_rng.choice(len(candidates)))
            score = (-float("inf") if (
                (self.config.variant == "RS2" and len(safe) == 0)
                or (self.config.variant != "RS2" and len(unsafe) == 0)) else float("inf"))
            return candidates[position].clone(), score, candidates[position].clone(), float(values[position])
        if len(unsafe) == 0:
            raise AdveRSInnerSolverError(
                "The released RS1/RSG program has an empty violating set; its maximum is undefined. "
                "Use source_protocol='native' for the paper-defined nonnegative fragility.", algorithm=self,
            )
        distances = euclidean_distances(points[safe], points[unsafe],
                                        squared=(self.config.variant == "RS2" or (
                                            self.config.source_protocol == "advers_2025"
                                            and self.config.variant == "RSG")))
        if self.config.variant == "RS2":
            scores = np.min(distances, axis=1)
            local = int(np.argmax(scores))
            witness = int(np.argmin(distances[local]))
        else:
            exponent = 1. if self.config.variant == "RS1" else self.config.p
            ratios = np.power(self.config.aspiration - values[unsafe], 1 / exponent)[None, :] / distances
            scores = np.max(ratios, axis=1)
            local = int(np.argmin(scores))
            witness = int(np.argmax(ratios[local]))
        position = int(safe[local])
        return (candidates[position].clone(), float(scores[local]),
                candidates[int(unsafe[witness])].clone(), float(values[position]))

    def _fragility_from_values(
        self,
        center: torch.Tensor,
        points: torch.Tensor,
        values: torch.Tensor,
        *,
        exponent: float,
        nominal_value: float,
    ) -> tuple[float, torch.Tensor]:
        """Compute a finite-neighborhood fragility ratio and witness point.

        The ratio uses only supplied confidence values and distances. Stable
        row order resolves ties, preserving the caller's finite dataset-index
        ordering without consulting hidden observations.
        """

        if nominal_value < self.config.aspiration:
            return math.inf, center.detach().clone()
        distances = self._distance(center, points)
        valid = distances > self.config.perturbation_domain.distance_floor
        if not valid.any():
            return 0.0, center.detach().clone()
        deficits = (self.config.aspiration - values[valid]).clamp_min(0.0)
        ratios = deficits.pow(1.0 / exponent) / distances[valid]
        maximum = ratios.max()
        tolerance = 1e-12 + 1e-9 * max(1.0, abs(float(maximum.item())))
        position = int(torch.nonzero(ratios >= maximum - tolerance, as_tuple=False)[0].item())
        return float(ratios[position].item()), points[valid][position].detach().clone()

    def _continuous_fragility(
        self,
        center: torch.Tensor,
        *,
        exponent: float,
        value_function: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[float, torch.Tensor, float]:
        """Solve a continuous RS1/RSG ratio problem by certified multistart."""

        nominal = float(value_function(center.unsqueeze(0))[0].item())
        if nominal < self.config.aspiration:
            return math.inf, center.detach().clone(), nominal
        points = self._inner_points(center, explicit=None)
        values = value_function(points)
        distances = self._distance(center, points)
        valid = distances > self.config.perturbation_domain.distance_floor
        if not valid.any():
            return 0.0, center.detach().clone(), nominal
        valid_points = points[valid]
        deficits = (self.config.aspiration - values[valid]).clamp_min(0.0)
        ratios = deficits.pow(1.0 / exponent) / distances[valid]

        def negative_ratio(point: torch.Tensor) -> float:
            """Negate the fragility ratio for continuous minimization."""

            distance = float(self._distance(center, point.unsqueeze(0))[0].item())
            if distance <= self.config.perturbation_domain.distance_floor:
                return float("inf")
            value = float(value_function(point.unsqueeze(0))[0].item())
            deficit = max(self.config.aspiration - value, 0.0)
            return -(deficit ** (1.0 / exponent) / distance)

        radius = self.config.perturbation_domain.radius
        worst, negative_score, _ = self._continuous_multistart_minimize(
            solver=self.config.inner_solver,
            objective=negative_ratio,
            raw_points=valid_points,
            raw_values=-ratios,
            restarts=_INNER_REFINEMENTS,
            center=center if radius is not None else None,
            radius=radius,
            label="AdveRS fragility inner maximization",
        )
        return -negative_score, worst, nominal

    def _rs1_fragility(
        self, center: torch.Tensor, *, explicit: torch.Tensor | None = None
    ) -> tuple[float, torch.Tensor, float]:
        """Evaluate the RS1 unit-exponent optimistic fragility objective."""

        if explicit is None:
            return self._continuous_fragility(
                center,
                exponent=1.0,
                value_function=self._optimistic_function,
            )
        points = self._inner_points(center, explicit=explicit)
        values = self._optimistic_function(points)
        nominal = float(self._optimistic_function(center.unsqueeze(0))[0].item())
        score, worst = self._fragility_from_values(
            center, points, values, exponent=1.0, nominal_value=nominal
        )
        return score, worst, nominal

    def _rsg_fragility(
        self, center: torch.Tensor, *, explicit: torch.Tensor | None = None
    ) -> tuple[float, torch.Tensor, float]:
        """Evaluate generalized RSG fragility with exponent ``config.p``."""

        if explicit is None:
            return self._continuous_fragility(
                center,
                exponent=self.config.p,
                value_function=self._optimistic_function,
            )
        points = self._inner_points(center, explicit=explicit)
        values = self._optimistic_function(points)
        nominal = float(self._optimistic_function(center.unsqueeze(0))[0].item())
        score, worst = self._fragility_from_values(
            center, points, values, exponent=self.config.p, nominal_value=nominal
        )
        return score, worst, nominal

    def _rs2_critical_radius(
        self, center: torch.Tensor, *, explicit: torch.Tensor | None = None
    ) -> tuple[float, torch.Tensor, float]:
        """Return the largest aspiration-safe RS2 radius and its witness.

        Finite mode searches the exact public candidate graph. Continuous mode
        delegates to a fixed-step bisection whose inner minima are independently
        feasibility-certified.
        """

        nominal = float(self._optimistic_function(center.unsqueeze(0))[0].item())
        if nominal < self.config.aspiration:
            return 0.0, center.detach().clone(), nominal
        if explicit is not None:
            # A public finite table defines an exact graph; no continuous
            # interpolation or optimizer is permitted in this branch.
            points = self._inner_points(center, explicit=explicit)
            values = self._optimistic_function(points)
            distances = self._distance(center, points)
            violating = values < self.config.aspiration
            if violating.any():
                candidate_distances = distances[violating]
                minimum = candidate_distances.min()
                tolerance = 1e-12 + 1e-9 * max(1.0, abs(float(minimum.item())))
                position = int(
                    torch.nonzero(
                        candidate_distances <= minimum + tolerance,
                        as_tuple=False,
                    )[0].item()
                )
                return (
                    float(candidate_distances[position].item()),
                    points[violating][position].detach().clone(),
                    nominal,
                )
            position = int(torch.argmax(distances).item())
            return (
                float(distances[position].item()),
                points[position].detach().clone(),
                nominal,
            )

        radius, worst = self._continuous_critical_radius(
            center,
            value_function=self._optimistic_function,
        )
        return radius, worst, nominal

    def _continuous_minimum_in_ball(
        self,
        center: torch.Tensor,
        *,
        radius: float,
        value_function: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[float, torch.Tensor]:
        """Minimize a confidence function on one certified radius ball."""

        points = self._inner_points(center, explicit=None, radius=radius)
        values = value_function(points)
        if radius <= _FEASIBILITY_TOLERANCE:
            return float(values[0].item()), center.detach().clone()

        def objective(point: torch.Tensor) -> float:
            """Return the confidence value minimized inside the trial ball."""

            return float(value_function(point.unsqueeze(0))[0].item())

        point, value, _ = self._continuous_multistart_minimize(
            solver=self.config.inner_solver,
            objective=objective,
            raw_points=points,
            raw_values=values,
            restarts=_INNER_REFINEMENTS,
            center=center,
            radius=radius,
            label="AdveRS critical-radius inner minimization",
        )
        return value, point

    def _continuous_critical_radius(
        self,
        center: torch.Tensor,
        *,
        value_function: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[float, torch.Tensor]:
        """Run exactly ``radius_bisection_steps`` certified RS2 iterations."""

        lower_radius = 0.0
        upper_radius = self._maximum_feasible_radius(center)
        worst = center.detach().clone()
        for _ in range(self.config.radius_bisection_steps):
            radius = (lower_radius + upper_radius) / 2.0
            minimum, certificate_point = self._continuous_minimum_in_ball(
                center,
                radius=radius,
                value_function=value_function,
            )
            if minimum >= self.config.aspiration:
                lower_radius = radius
                worst = certificate_point
            else:
                upper_radius = radius
        # The loop count, not an optimizer status flag, controls the radius
        # resolution and therefore checkpoint-replay determinism.
        return lower_radius, worst

    def _solve_inner_perturbation_problem(
        self, center: torch.Tensor, *, explicit: torch.Tensor | None
    ) -> tuple[float, torch.Tensor, float]:
        """Dispatch the configured RS1, RS2, or RSG inner objective."""

        if self.config.variant == "RS1":
            return self._rs1_fragility(center, explicit=explicit)
        if self.config.variant == "RS2":
            return self._rs2_critical_radius(center, explicit=explicit)
        return self._rsg_fragility(center, explicit=explicit)

    def _candidate_table(
        self, round_info: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return stable finite rows or deterministic continuous outer starts.

        Finite rows are sorted by public dataset index so downstream ties are
        deterministic. The table contains coordinates only, never hidden
        response values.
        """

        if "candidate_actions" in round_info:
            table = self.prepare_discrete_candidates(
                candidate_actions=round_info["candidate_actions"],
                candidate_dataset_indices=round_info["candidate_dataset_indices"],
                round_info=round_info,
            )
            # Sorting once makes every exhaustive inner tie inherit the
            # smallest stable dataset-index rule without exposing responses.
            order = torch.argsort(table.dataset_indices, stable=True)
            return table.environment_actions[order], table.dataset_indices[order]
        return (
            self._sample_design(
                self.bounds,
                self.config.outer_raw_samples,
                phase=(
                    "initial"
                    if self.state.y_train.numel() < self.config.initial_points
                    else "acquisition"
                ),
                seed=self.seed + 49979687 * (self.iteration + 1),
            ),
            None,
        )

    def _select_outer_action(
        self, candidates: torch.Tensor, dataset_indices: torch.Tensor | None
    ) -> tuple[torch.Tensor, float, torch.Tensor, float]:
        """Select the nominal proposal under variant-specific stable tie rules.

        Returns the physical proposal, its robust-satisficing score, the inner
        witness point, and the nominal optimistic value. Continuous candidates
        are refined with independently certified multistart optimization.
        """

        scores: list[float] = []
        worst_points: list[torch.Tensor] = []
        nominal: list[float] = []
        candidate_iterator: Iterable[torch.Tensor]
        if self.config.source_protocol in _SOURCE_PROTOCOL_VERSIONS:
            if dataset_indices is None:
                raise AdveRSInnerSolverError(
                    "The released AdveRS protocol requires explicit finite candidate rows.", algorithm=self)
            return self._source_finite_selection(candidates)
        # Evaluate a finite confidence surface once. Recomputing the posterior
        # separately for every nominal row makes a finite 1024-point protocol
        # needlessly perform over a thousand identical GP predictions.
        vectorized = dataset_indices is not None and self.config.perturbation_domain.radius is None
        if vectorized:
            values = self._optimistic_function(candidates).detach()
            cached = getattr(self, "_finite_distance_cache", None)
            if cached is None or not torch.equal(candidates, cached[0]):
                distances = torch.cdist(
                    candidates,
                    candidates,
                    p=2.0 if self.config.perturbation_domain.norm == "l2" else float("inf"),
                )
                self._finite_distance_cache = (candidates.detach().clone(), distances)
            else:
                distances = cached[1]
            safe = values >= self.config.aspiration
            if self.config.variant == "RS2":
                masked = distances.masked_fill(safe.unsqueeze(0), float("inf"))
                score_tensor, witnesses = masked.min(dim=1)
                if safe.all():
                    score_tensor, witnesses = distances.max(dim=1)
                score_tensor = score_tensor.masked_fill(~safe, 0.0)
            else:
                exponent = 1.0 if self.config.variant == "RS1" else self.config.p
                deficit = (self.config.aspiration - values).clamp_min(0).pow(1 / exponent)
                ratios = deficit.unsqueeze(0) / distances.clamp_min(
                    self.config.perturbation_domain.distance_floor
                )
                ratios = ratios.masked_fill(
                    distances <= self.config.perturbation_domain.distance_floor, 0.0
                )
                score_tensor, witnesses = ratios.max(dim=1)
                score_tensor = score_tensor.masked_fill(~safe, float("inf"))
            nominal_tensor = values
            candidate_iterator = ()
        else:
            candidate_iterator = candidates
        for candidate in candidate_iterator:
            score, worst, nominal_value = self._solve_inner_perturbation_problem(
                candidate, explicit=candidates if dataset_indices is not None else None
            )
            scores.append(score)
            worst_points.append(worst)
            nominal.append(nominal_value)
        if not vectorized:
            score_tensor = torch.tensor(scores, dtype=self.dtype, device=self.device)
            nominal_tensor = torch.tensor(nominal, dtype=self.dtype, device=self.device)
        if dataset_indices is None:
            if self.config.variant == "RS2":
                raw_objective = -score_tensor

                def objective(point: torch.Tensor) -> float:
                    """Negate RS2 radius so candidate search maximizes it."""

                    score, _, _ = self._solve_inner_perturbation_problem(point, explicit=None)
                    return -score

            elif torch.isfinite(score_tensor).any():
                raw_objective = score_tensor

                def objective(point: torch.Tensor) -> float:
                    """Return finite RS1/RSG fragility for outer minimization."""

                    score, _, _ = self._solve_inner_perturbation_problem(point, explicit=None)
                    return score

            else:
                # No optimistic candidate reaches aspiration.  The documented
                # secondary rule then maximizes the nominal optimistic value.
                raw_objective = -nominal_tensor

                def objective(point: torch.Tensor) -> float:
                    """Negate nominal optimism when no action satisfies aspiration."""

                    return -float(self._optimistic_function(point.unsqueeze(0))[0].item())

            selected, _, _ = self._continuous_multistart_minimize(
                solver=self.config.candidate_solver,
                objective=objective,
                raw_points=candidates,
                raw_values=raw_objective,
                restarts=_CANDIDATE_REFINEMENTS,
                label="AdveRS outer candidate refinement",
            )
            score, worst, nominal_value = self._solve_inner_perturbation_problem(
                selected, explicit=None
            )
            return selected, score, worst, nominal_value

        if self.config.variant == "RS2":
            target = score_tensor.max()
            tolerance = 1e-12 + 1e-9 * max(1.0, abs(float(target.item())))
            tied = torch.nonzero(score_tensor >= target - tolerance, as_tuple=False).reshape(-1)
            if tied.numel() > 1 and self.config.finite_tie_break == "nominal":
                best_nominal = nominal_tensor[tied].max()
                nominal_tolerance = 1e-12 + 1e-9 * max(1.0, abs(float(best_nominal.item())))
                tied = tied[nominal_tensor[tied] >= best_nominal - nominal_tolerance]
        else:
            if not torch.isfinite(score_tensor).any():
                # When no optimistic action satisfies aspiration, select the
                # largest nominal UCB while retaining infinite fragility.
                target = nominal_tensor.max()
                tolerance = 1e-12 + 1e-9 * max(1.0, abs(float(target.item())))
                tied = torch.nonzero(nominal_tensor >= target - tolerance, as_tuple=False).reshape(
                    -1
                )
            else:
                target = score_tensor[torch.isfinite(score_tensor)].min()
                tolerance = 1e-12 + 1e-9 * max(1.0, abs(float(target.item())))
                tied = torch.nonzero(score_tensor <= target + tolerance, as_tuple=False).reshape(-1)
        if dataset_indices is not None and tied.numel() > 1:
            tied = tied[torch.argmin(dataset_indices[tied])].reshape(1)
        elif tied.numel() > 1:
            position = min(
                (int(item.item()) for item in tied),
                key=lambda item: self._normalized_lexicographic_key(candidates[item]),
            )
            tied = torch.tensor([position], dtype=torch.long, device=self.device)
        position = int(tied[0].item())
        if vectorized:
            return (
                candidates[position].detach().clone(),
                float(score_tensor[position]),
                candidates[witnesses[position]].detach().clone(),
                float(nominal_tensor[position]),
            )
        return (
            candidates[position].detach().clone(),
            scores[position],
            worst_points[position].detach().clone(),
            nominal[position],
        )

    def _ask_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Return one nominal adversarial-environment proposal in ``D_env``.

        The robust score and inner witness are snapshotted with the outstanding
        request, ensuring ``tell`` can commit exactly the decision that was
        evaluated even after an external perturbation.
        """

        if self.config.gp_backend == "fixed_gpy" and "candidate_actions" not in round_info:
            raise AlgorithmConfigurationError("fixed_gpy requires an explicit finite candidate_actions table.")
        candidates, indices = self._candidate_table(round_info)
        if self.state.y_train.numel() < self.config.initial_points:
            position = 0 if indices is None else int(torch.argmin(indices).item())
            action = candidates[position]
            score, worst, nominal = math.nan, action, math.nan
        else:
            self._fit_model()
            action, score, worst, nominal = self._select_outer_action(candidates, indices)
        self._last_acquisition_value = None if math.isnan(score) else float(score)
        self._record_last_ask(
            actions=action,
            round_info=round_info,
            model_inputs=action,
            algorithm_metadata={
                "score": score,
                "worst_point": worst,
                "nominal_optimistic_value": nominal,
            },
        )
        return action

    def _pessimistic_score(self, action: torch.Tensor, explicit: torch.Tensor | None) -> float:
        """Audit the selected action using the updated posterior lower bound."""

        if explicit is None and self.config.variant == "RS2":
            radius, _ = self._continuous_critical_radius(
                action,
                value_function=self._pessimistic_function,
            )
            return radius
        if explicit is None:
            score, _, _ = self._continuous_fragility(
                action,
                exponent=1.0 if self.config.variant == "RS1" else self.config.p,
                value_function=self._pessimistic_function,
            )
            return score
        points = self._inner_points(action, explicit=explicit)
        values = self._pessimistic_function(points)
        nominal = float(self._pessimistic_function(action.unsqueeze(0))[0].item())
        if self.config.variant == "RS2":
            distances = self._distance(action, points)
            violating = values < self.config.aspiration
            return (
                float(distances[violating].min().item())
                if violating.any()
                else float(distances.max().item())
            )
        score, _ = self._fragility_from_values(
            action,
            points,
            values,
            exponent=1.0 if self.config.variant == "RS1" else self.config.p,
            nominal_value=nominal,
        )
        return score

    def _tell_impl(
        self,
        *,
        action: torch.Tensor,
        feedback: Mapping[str, Any],
        round_info: Mapping[str, Any],
    ) -> None:
        """Commit public feedback while separating proposal and evaluation.

        The GP trains on ``evaluated_action`` because the observation belongs to
        that physical coordinate. The nominal proposal and optional public
        perturbation are retained separately for audit; no hidden attack value
        is requested from the environment.
        """

        assert self.last_ask is not None
        evaluated = feedback["evaluated_action"].reshape(1, -1)
        observation = feedback["observation"].reshape(-1)
        raw_variance = feedback.get("observation_noise_variance")
        variance = (
            torch.zeros_like(observation)
            if raw_variance is None
            else torch.as_tensor(raw_variance, dtype=self.dtype, device=self.device).reshape(-1)
        )
        previous_actions_train = self.state.actions_train
        previous_y_train = self.state.y_train
        previous_noise_variance = self.state.observation_noise_variance
        previous_gp_state = self.gp.state_dict()
        self.state.actions_train = torch.cat((previous_actions_train, evaluated))
        self.state.y_train = torch.cat((previous_y_train, observation))
        self.state.observation_noise_variance = torch.cat((previous_noise_variance, variance))
        external = feedback.get("perturbation")
        external_perturbation = (
            None
            if external is None
            else torch.as_tensor(external, dtype=self.dtype, device=self.device)
            .reshape(-1)
            .detach()
            .clone()
        )
        metadata = self.last_ask.algorithm_metadata
        # The updated posterior defines the pessimistic audit quantity.  It is
        # computed from learner-visible GP state, never from latent attacks.
        explicit = None
        if "candidate_actions" in round_info:
            explicit = torch.as_tensor(
                round_info["candidate_actions"], dtype=self.dtype, device=self.device
            )
        try:
            self._fit_model()
            pessimistic_score = self._pessimistic_score(action[0], explicit)
        except Exception:
            # ``Algorithm.tell`` deliberately leaves the request outstanding on
            # failure.  Keep AdveRS's concrete state equally retryable: neither
            # the pending observation nor its GP fit is committed until the
            # complete pessimistic audit succeeds.
            self.state.actions_train = previous_actions_train
            self.state.y_train = previous_y_train
            self.state.observation_noise_variance = previous_noise_variance
            self.gp.load_state_dict(previous_gp_state)
            raise

        self.state.proposals.append(action[0].detach().clone())
        self.state.evaluated_points.append(evaluated[0].detach().clone())
        self.state.external_perturbations.append(external_perturbation)
        self.state.realized_distances.append(float(self._distance(action[0], evaluated).item()))
        self.state.optimistic_scores.append(float(metadata["score"]))
        self.state.worst_points.append(
            torch.as_tensor(metadata["worst_point"], dtype=self.dtype, device=self.device)
        )
        self.state.pessimistic_scores.append(pessimistic_score)
        self.state.aspiration_satisfied.append(
            bool(float(observation[0].item()) >= self.config.aspiration)
        )

    def _recommendation_impl(self) -> tuple[torch.Tensor | None, float | None]:
        """Recommend the best recorded nominal proposal for the active variant."""

        if not self.state.proposals:
            return self._best_observed()
        scores = torch.tensor(self.state.optimistic_scores, dtype=self.dtype, device=self.device)
        finite = torch.isfinite(scores)
        if not finite.any():
            # Initial-design proposals have no acquisition score. An infinite
            # RS1/RSG fragility also supplies no finite certificate. Preserve
            # a nominal proposal while reporting that no score is available.
            return self.state.proposals[-1].detach().clone(), None
        indices = torch.nonzero(finite, as_tuple=False).reshape(-1)
        select = torch.argmax if self.config.variant == "RS2" else torch.argmin
        position = int(indices[select(scores[indices])].item())
        return self.state.proposals[position].detach().clone(), float(scores[position].item())

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Summarize variant scores, realized distance, and satisfaction rate."""

        return {
            "variant": self.config.variant,
            "source_protocol": self.config.source_protocol,
            "source_protocol_version": _SOURCE_PROTOCOL_VERSIONS.get(self.config.source_protocol),
            "decision_model": self.config.decision_model,
            "last_optimistic_score": self.state.optimistic_scores[-1]
            if self.state.optimistic_scores
            else None,
            "last_pessimistic_score": self.state.pessimistic_scores[-1]
            if self.state.pessimistic_scores
            else None,
            "last_realized_distance": self.state.realized_distances[-1]
            if self.state.realized_distances
            else None,
            "satisfaction_rate": (
                sum(self.state.aspiration_satisfied) / len(self.state.aspiration_satisfied)
                if self.state.aspiration_satisfied
                else None
            ),
        }

    def _result_metadata_impl(self) -> Mapping[str, Any]:
        """Return detached nominal, evaluated, and distance audit traces."""

        return {
            "proposals": [x.detach().clone() for x in self.state.proposals],
            "evaluated_points": [x.detach().clone() for x in self.state.evaluated_points],
            "realized_distances": tuple(self.state.realized_distances),
        }

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize mutable proposal, score, observation, and audit progress."""

        return {
            "variant": self.config.variant,
            "aspiration": self.config.aspiration,
            "p": self.config.p,
            "source_protocol": self.config.source_protocol,
            "source_protocol_version": _SOURCE_PROTOCOL_VERSIONS.get(self.config.source_protocol),
            "source_rng": self._source_rng.get_state(),
            "gp": self.gp.state_dict(),
            "state": {name: getattr(self.state, name) for name in self.state.__dataclass_fields__},
        }

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Validate immutable variant settings and checkpoint state sections."""

        del strict
        if (state.get("source_protocol", "native") != self.config.source_protocol
                or (self.config.source_protocol in _SOURCE_PROTOCOL_VERSIONS
                    and state.get("source_protocol_version") != _SOURCE_PROTOCOL_VERSIONS[self.config.source_protocol])):
            raise AdveRSCheckpointError("Stored source protocol/version is incompatible.", algorithm=self)
        if self.config.source_protocol in _SOURCE_PROTOCOL_VERSIONS:
            validator = RandomStream(0)
            validator.set_state(state.get("source_rng"))
        if (
            state.get("variant") != self.config.variant
            or state.get("aspiration") != self.config.aspiration
            or state.get("p") != self.config.p
        ):
            raise AdveRSCheckpointError(
                "Stored AdveRS configuration is incompatible.", algorithm=self
            )
        if not isinstance(state.get("state"), Mapping):
            raise AdveRSCheckpointError("AdveRS state is incomplete.", algorithm=self)

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Restore traces and the learned GP warm start for exact continuation."""

        del strict
        raw = state["state"]
        try:

            def tensor(value: Any) -> torch.Tensor:
                """Convert one checkpoint value to the configured tensor context."""

                return torch.as_tensor(value, dtype=self.dtype, device=self.device)

            self.state = AdveRSState(
                actions_train=tensor(raw["actions_train"]),
                y_train=tensor(raw["y_train"]),
                observation_noise_variance=tensor(raw["observation_noise_variance"]),
                proposals=[tensor(x) for x in raw["proposals"]],
                evaluated_points=[tensor(x) for x in raw["evaluated_points"]],
                external_perturbations=[
                    None if x is None else tensor(x) for x in raw["external_perturbations"]
                ],
                realized_distances=[float(x) for x in raw["realized_distances"]],
                optimistic_scores=[float(x) for x in raw["optimistic_scores"]],
                pessimistic_scores=[float(x) for x in raw["pessimistic_scores"]],
                worst_points=[tensor(x) for x in raw["worst_points"]],
                aspiration_satisfied=[bool(x) for x in raw["aspiration_satisfied"]],
            )
            if isinstance(state.get("gp"), dict):
                self.gp.load_state_dict(dict(state["gp"]))
            else:
                self.gp.reset()
            if self.config.source_protocol in _SOURCE_PROTOCOL_VERSIONS:
                self._source_rng.set_state(state["source_rng"])
            if self.config.source_protocol == "advers_2025":
                self.invalidate_source_guarantees("basat.advers.released_2025_variance_confidence")
        except Exception as error:
            raise AdveRSCheckpointError(
                f"Invalid AdveRS checkpoint: {error}", algorithm=self
            ) from error

    def _reset_impl(self) -> None:
        """Clear all mutable AdveRS and GP state."""

        self.state = self._empty_state()
        self.gp.reset()
        self._source_rng = RandomStream(self.seed % 2**32)


__all__ = [
    "AdveRS",
    "AdveRSConfig",
    "AdveRSState",
    "BallPerturbations",
    "FullDomainPerturbations",
    "PerturbationDomainSpec",
    "SolverSpec",
]
