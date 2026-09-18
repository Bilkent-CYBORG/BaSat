"""Robust Bayesian satisficing over finite context distributions.

RoBOS observes a reference distribution before acting and minimizes optimistic
fragility measured by the MMD induced by the configured context kernel.  The
true environment sampling distribution is never accepted by this API.
"""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from numbers import Integral, Real
from typing import Any, ClassVar, Literal

import gpytorch
import numpy as np
import torch

from basat.models.Acquisitions import DiscreteAcquisitionCandidates
from basat.models.FixedKernelGaussianProcess import FixedKernelGaussianProcess
from basat.models.GaussianProcess import GaussianProcess, GaussianProcessConfig
from basat.models.Kernels import KernelConfig, build_kernel
from basat.models.utils import Bounds, to_numpy

from ..algorithm import (
    Algorithm,
    AlgorithmConfigurationError,
    AlgorithmDataError,
    resolve_batch_configuration,
)
from ..helper import SourceRecord
from .Exception.exception_RoBOS import (
    RoBOSAspirationError,
    RoBOSCheckpointError,
    RoBOSContextSupportError,
    RoBOSDataCompatibilityError,
    RoBOSException,
    RoBOSFragilitySolverError,
    RoBOSMMDMatrixError,
    RoBOSReferenceDistributionError,
    RoBOSWarning,
)

_ABSOLUTE_TIE_TOLERANCE = 1e-12
_RELATIVE_TIE_TOLERANCE = 1e-9
_GP_RETRY_SEED_OFFSET = 104_729


def theoretical_beta(t: int, *, delta: float = 0.1) -> float:
    """Return an increasing confidence schedule for finite-context GP-UCB.

    Parameters
    ----------
    t:
        One-based optimization round; non-positive values are clamped to one.
    delta:
        Positive confidence-failure probability.

    Returns
    -------
    float
        Non-negative exploration coefficient for round ``t``.

    Raises
    ------
    ValueError
        If ``t`` is not an integer or ``delta`` is not a finite probability
        strictly between zero and one.
    """

    if isinstance(t, bool) or not isinstance(t, Integral):
        raise ValueError("t must be an integer.")
    if (
        isinstance(delta, bool)
        or not isinstance(delta, Real)
        or not math.isfinite(float(delta))
        or not 0.0 < float(delta) < 1.0
    ):
        raise ValueError("delta must be a finite probability strictly between zero and one.")
    time_index = max(int(t), 1)
    return float(2.0 * math.log(math.pi**2 * time_index**2 / (6.0 * float(delta)) + 1.0))


@dataclass(frozen=True, slots=True)
class RoBOSConfig:
    """Immutable finite-context RoBOS configuration.

    Parameters
    ----------
    aspiration:
        Finite reward threshold used in the optimistic fragility numerator.
    context_kernel:
        Kernel for context coordinates and the ``(K, K)`` MMD Gram matrix.
    action_kernel:
        Kernel operating on the ``D_action`` slice of joint GP inputs.
    joint_kernel_combination:
        ``"product"`` for source behavior or ``"custom"`` for a Basat
        extension using ``joint_kernel``.
    joint_kernel:
        Optional kernel over all ``D_action + D_context`` coordinates; required
        only when ``joint_kernel_combination="custom"``.
    beta:
        Non-negative scalar or round-indexed callable supplying the GP
        confidence multiplier.
    simplex_solver:
        SOCP, SLSQP, or enumeration solver for the worst simplex distribution.
        ``"source_cobyla"`` explicitly reproduces the released 2023 numerical
        protocol: one random simplex start, 1000 COBYLA evaluations, denominator
        ``MMD + 1e-9``, and first minimum finite-action tie. It is an approximate
        source-comparison mode, not a certificate of the mathematical optimum.
        SciPy versions before 1.16 use the original Powell implementation;
        later versions use PRIMA and may select different actions.
    distribution_floor:
        Excluded MMD-distance neighborhood around the reference distribution,
        preventing an undefined zero-distance fragility ratio.
    distribution_grid:
        Optional explicit simplex table with shape ``(G, K)`` used only by the
        enumeration solver. It contains candidate probabilities, not the
        environment's hidden sampling distribution.
    enumeration_resolution:
        Integer-composition resolution used when an enumeration grid is not
        supplied.
    action_raw_samples:
        Sobol candidates used to initialize continuous outer action search.
    action_restarts:
        Best outer action candidates refined by the continuous optimizer.
    action_maxiter:
        Maximum iterations for each outer action restart.
    initial_points:
        Sequential observations collected before robust model-based selection.
    fit_maxiter:
        Maximum joint-GP hyperparameter-optimization iterations per refit.
        Zero conditions at fixed configured hyperparameters.
    gp_config:
        Optional complete joint surrogate settings. Its kernel takes precedence
        for prediction; ``context_kernel`` still defines MMD geometry.
    gp_backend:
        ``"standard"`` uses the usual differentiable GP. ``"fixed_gpy"``
        opts into fixed, zero-mean CPU float64 full conditioning with GPy's
        lower-LAPACK/marginal-prediction arithmetic; it requires fit_maxiter=0.
    gp_training_regularization:
        Explicit training-only diagonal for ``"fixed_gpy"``, separate from
        observation noise and conditional jitter. GPy exact inference uses
        1e-8. Each new addition is reported as a Basat numerical warning.
    allow_semidefinite_mmd:
        Permit a numerically semidefinite context Gram matrix without adding
        a nugget. This reproduces the published dense RBF context grid. The
        default requires positive eigenvalues on the simplex tangent space.
    feasibility_tolerance:
        Tolerance for independently recomputed simplex, MMD, and action-bound
        residuals.
    optimizer_tolerance:
        Numerical termination tolerance passed to inner and outer optimizers.
    batch_size:
        Requested proposal count. RoBOS version 1 requires ``1``.
    seed:
        Seed for algorithm-owned candidate generation and deterministic replay.
    """

    aspiration: float
    context_kernel: KernelConfig | gpytorch.kernels.Kernel = field(default_factory=KernelConfig)
    action_kernel: KernelConfig | gpytorch.kernels.Kernel = field(default_factory=KernelConfig)
    joint_kernel_combination: Literal["product", "custom"] = "product"
    joint_kernel: gpytorch.kernels.Kernel | None = None
    beta: Callable[[int], float] | float = theoretical_beta
    simplex_solver: Literal["socp", "slsqp", "enumeration", "source_cobyla"] = "slsqp"
    distribution_floor: float = 0.0
    distribution_grid: torch.Tensor | np.ndarray | None = None
    enumeration_resolution: int = 20
    action_raw_samples: int = 256
    action_restarts: int = 10
    action_maxiter: int = 200
    initial_points: int = 1
    fit_maxiter: int = 100
    gp_config: GaussianProcessConfig | None = None
    gp_backend: Literal["standard", "fixed_gpy"] = "standard"
    gp_training_regularization: float = 0.0
    allow_semidefinite_mmd: bool = False
    feasibility_tolerance: float = 1e-8
    optimizer_tolerance: float = 1e-6
    batch_size: int | None = 1
    q_batch: int | None = None
    seed: int = 1


@dataclass(slots=True)
class RoBOSState:
    """Mutable joint-GP and fragility history.

    Attributes
    ----------
    actions_train:
        Evaluated physical actions with shape ``(N, D_action)``.
    contexts_train:
        Realized support contexts with shape ``(N, D_context)``.
    joint_inputs_train:
        Concatenated GP coordinates with shape
        ``(N, D_action + D_context)``.
    y_train:
        Learner-visible observations with shape ``(N,)``.
    observation_noise_variance:
        Known observation-noise variances with shape ``(N,)``; zeros denote
        feedback without a supplied variance.
    reference_history:
        Learner-visible reference simplex vectors, each with shape ``(K,)``;
        this never contains the hidden true sampling distribution.
    selected_actions:
        Nominal actions selected by RoBOS, each with shape ``(D_action,)``.
    selected_fragilities:
        Optimistic fragility values assigned to selected actions.
    reference_values:
        Expected UCB values under each public reference distribution.
    worst_distributions:
        Certified inner-solver simplex witnesses, each with shape ``(K,)``.
    realized_contexts:
        Learner-visible contexts returned after evaluation, each with shape
        ``(D_context,)``.
    aspiration_satisfied:
        Whether each observed reward met the configured aspiration threshold.
    """

    actions_train: torch.Tensor
    contexts_train: torch.Tensor
    joint_inputs_train: torch.Tensor
    y_train: torch.Tensor
    observation_noise_variance: torch.Tensor
    reference_history: list[torch.Tensor]
    selected_actions: list[torch.Tensor]
    selected_fragilities: list[float]
    reference_values: list[float]
    worst_distributions: list[torch.Tensor]
    realized_contexts: list[torch.Tensor]
    aspiration_satisfied: list[bool]


class _FiniteContextGPAlgorithm(Algorithm):
    """Shared finite-support joint-GP mechanics for RoBOS and DRBO."""

    require_tangent_pd: bool = False
    # Concrete subclasses provide structurally compatible configuration and
    # state dataclasses.  ``Any`` is intentional here: the shared mechanics
    # operate on their common tensor fields while their histories differ.
    config: Any
    state: Any
    contexts: torch.Tensor
    context_dimension: int
    gp: GaussianProcess
    _action_kernel_spec: KernelConfig | gpytorch.kernels.Kernel
    _context_kernel_spec: KernelConfig | gpytorch.kernels.Kernel
    _joint_kernel_combination: str
    _custom_joint_kernel: gpytorch.kernels.Kernel | None

    def prepare_round_info(self, round_info: Mapping[str, Any] | None) -> dict[str, Any]:
        """Fill only absent support, reference and ambiguity inputs."""
        info = super().prepare_round_info(round_info)
        if info.get("context_support") is None:
            if info.get("context") is not None:
                support = torch.as_tensor(
                    info["context"], dtype=self.dtype, device=self.device
                ).reshape(1, -1)
                info["context_support_is_dynamic"] = True
            elif self.contexts.numel():
                support = self.contexts.detach().clone()
            else:
                support = self._default_context().reshape(1, -1)
                info["context_support_is_dynamic"] = True
            self._warn_environment_default(
                "context_support",
                "the supplied current context as a singleton, or internal zero singleton",
            )
            info["context_support"] = support
        self.bind_environment_metadata(info)
        if info.get("reference_distribution") is None:
            reference = torch.full(
                (self.contexts.shape[0],),
                1.0 / self.contexts.shape[0],
                dtype=self.dtype,
                device=self.device,
            )
            if info.get("context") is not None:
                context = torch.as_tensor(
                    info["context"], dtype=self.dtype, device=self.device
                ).reshape(-1)
                matches = torch.isclose(self.contexts, context, rtol=0.0, atol=1e-8).all(dim=-1)
                if int(matches.sum()) == 1:
                    reference.zero_()
                    reference[matches] = 1.0
            info["reference_distribution"] = reference
            self._warn_environment_default("reference_distribution", str(reference.tolist()))
        if "margin" in self.required_round_info and info.get("margin") is None:
            info["margin"] = 0.0
            self._warn_environment_default("margin", "0")
        return info

    def prepare_feedback(self, feedback: Mapping[str, Any], *, action: Any) -> dict[str, Any]:
        result = super().prepare_feedback(feedback, action=action)
        if result.get("context") is None:
            # A concrete supported row keeps the finite-context update well-defined.
            result["context"] = self.contexts[0].clone()
            self._warn_environment_default("feedback.context", str(result["context"].tolist()))
        return result

    def _default_context(self) -> torch.Tensor:
        """Internal neutral context used only when the environment omits it."""
        dimension = getattr(self, "context_dimension", None)
        if not dimension:
            kernel = self.config.context_kernel
            dimension = getattr(kernel, "ard_num_dims", None) or 1
        return torch.zeros(int(dimension), dtype=self.dtype, device=self.device)

    def _initialize_finite_context_model(
        self,
        *,
        contexts: Any,
        action_kernel: KernelConfig | gpytorch.kernels.Kernel,
        context_kernel: KernelConfig | gpytorch.kernels.Kernel,
        joint_kernel_combination: str,
        joint_kernel: gpytorch.kernels.Kernel | None,
    ) -> None:
        """Freeze finite context support and construct a joint action-context GP.

        ``contexts`` is normalized to shape ``(K, D_context)``.  The resulting
        model coordinates concatenate physical actions and support rows, giving
        ``D_model = D_action + D_context``.
        """

        self._action_kernel_spec = action_kernel
        self._context_kernel_spec = context_kernel
        self._joint_kernel_combination = joint_kernel_combination
        self._custom_joint_kernel = joint_kernel
        if contexts is None:
            self.contexts = torch.empty((0, 0), dtype=self.dtype, device=self.device)
            self.context_dimension = 0
            self.model_input_dimension = self.action_dimension
            return
        support = torch.as_tensor(contexts, dtype=self.dtype, device=self.device)
        if support.ndim == 1:
            support = support.unsqueeze(-1)
        self.contexts = support.detach().clone()
        self._validate_context_support()
        self.context_dimension = int(self.contexts.shape[1])
        self.model_input_dimension = self.action_dimension + self.context_dimension
        gp_config = (
            replace(self.config.gp_config, dtype=self.dtype, device=self.device)
            if self.config.gp_config is not None
            else GaussianProcessConfig(
                kernel=self._build_joint_kernel(), dtype=self.dtype, device=self.device
            )
        )
        if self.config.gp_backend == "fixed_gpy":
            self.gp = FixedKernelGaussianProcess(
                gp_config,
                input_dimension=self.model_input_dimension,
                conditioning="full",
                arithmetic="gpy",
                training_regularization=self.config.gp_training_regularization,
            )
        else:
            self.gp = GaussianProcess(gp_config, input_dimension=self.model_input_dimension)

    @staticmethod
    def _validate_gp_backend(config: Any) -> None:
        if config.gp_backend not in {"standard", "fixed_gpy"}:
            raise AlgorithmConfigurationError("gp_backend must be 'standard' or 'fixed_gpy'.")
        regularization = config.gp_training_regularization
        if (
            isinstance(regularization, bool)
            or not isinstance(regularization, Real)
            or not math.isfinite(float(regularization))
            or regularization < 0
        ):
            raise AlgorithmConfigurationError(
                "gp_training_regularization must be finite and nonnegative."
            )
        if config.gp_backend == "standard" and regularization:
            raise AlgorithmConfigurationError(
                "Explicit GP regularization requires gp_backend='fixed_gpy'."
            )
        if config.gp_backend == "fixed_gpy" and (
            config.fit_maxiter != 0 or config.gp_config is None
        ):
            raise AlgorithmConfigurationError(
                "fixed_gpy requires explicit gp_config and zero GP fitting steps."
            )

    def _empty_state(self) -> Any:
        """Construct the concrete method's empty context-dependent history."""
        raise NotImplementedError

    def bind_environment_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Receive public finite support without accessing confidential distributions.

        The support and its row ordering are fixed for a run; references and
        realized contexts are delivered separately at their proper round times.
        """
        dynamic = bool(metadata.get("context_support_is_dynamic", False)) or getattr(
            self, "_dynamic_context_support", False
        )
        self._dynamic_context_support = dynamic or getattr(self, "_dynamic_context_support", False)
        support = metadata.get("context_support")
        if support is None:
            if self.contexts.numel():
                return
            support = self._default_context().reshape(1, -1)
            self._dynamic_context_support = True
            self._warn_environment_default("context_support", str(support.tolist()))
        if self.contexts.numel():
            if not self._support_matches(support):
                if dynamic:
                    candidate = torch.as_tensor(support, dtype=self.dtype, device=self.device)
                    if candidate.ndim != 2:
                        raise RoBOSContextSupportError(
                            "context_support must have shape (K, D_context).", algorithm=self
                        )
                    if candidate.shape[1] != self.context_dimension:
                        if self.state.y_train.numel():
                            raise RoBOSContextSupportError(
                                "Runtime context dimension changed.", algorithm=self
                            )
                        self._initialize_finite_context_model(
                            contexts=candidate,
                            action_kernel=self._action_kernel_spec,
                            context_kernel=self._context_kernel_spec,
                            joint_kernel_combination=self._joint_kernel_combination,
                            joint_kernel=self._custom_joint_kernel,
                        )
                        self.state = self._empty_state()
                        return
                    previous = self.contexts
                    self.contexts = candidate.detach().clone()
                    try:
                        self._validate_context_support()
                    except Exception:
                        self.contexts = previous
                        raise
                    # Only the integration support changes. Keep the fitted GP and
                    # all observations at their actual historical contexts.
                    return
                raise RoBOSContextSupportError(
                    "Environment context_support differs from the initialized support or its row order. "
                    "Create a fresh algorithm for a different support; do not duplicate support in config.",
                    algorithm=self,
                )
            return
        self._initialize_finite_context_model(
            contexts=support,
            action_kernel=self._action_kernel_spec,
            context_kernel=self._context_kernel_spec,
            joint_kernel_combination=self._joint_kernel_combination,
            joint_kernel=self._custom_joint_kernel,
        )
        self.state = self._empty_state()

    def _validate_context_support(self) -> None:
        """Validate finite, unique, non-overlapping context support rows.

        Exact support matching uses ``rtol=0`` and ``atol=1e-8``.  Support
        neighborhoods that can overlap under this tolerance are rejected so a
        realized context can never map ambiguously to two simplex coordinates.
        """

        support = getattr(self, "contexts", None)
        if not isinstance(support, torch.Tensor) or support.ndim != 2:
            raise RoBOSContextSupportError("contexts must have shape (K, D_context).")
        if support.shape[0] < 1 or support.shape[1] < 1:
            raise RoBOSContextSupportError(
                "At least one non-empty context-support row is required."
            )
        if not torch.isfinite(support).all():
            raise RoBOSContextSupportError("contexts must be finite.")
        for left in range(support.shape[0]):
            # Exact support lookup uses closed L-infinity balls of radius
            # 1e-8.  Two rows at most 2e-8 apart would make some realized
            # context match both rows, so reject the overlapping groups now.
            overlapping = ((support[left + 1 :] - support[left]).abs() <= 2e-8).all(dim=-1)
            if overlapping.any():
                raise RoBOSContextSupportError(
                    "contexts contain ambiguous tolerance-overlapping rows."
                )

    def _support_matches(self, support: Any) -> bool:
        """Return whether external support matches configured rows and ordering."""

        try:
            other = torch.as_tensor(support, dtype=self.dtype, device=self.device)
            if other.ndim == 1:
                other = other.unsqueeze(-1)
            return other.shape == self.contexts.shape and bool(
                torch.isclose(other, self.contexts, rtol=0.0, atol=1e-8).all()
            )
        except (TypeError, ValueError, RuntimeError):
            return False

    @staticmethod
    def _activate_custom_kernel(
        kernel: gpytorch.kernels.Kernel, dims: tuple[int, ...]
    ) -> gpytorch.kernels.Kernel:
        """Copy a custom kernel and assign explicit joint active dimensions."""

        result = copy.deepcopy(kernel)
        result.active_dims = torch.tensor(dims, dtype=torch.long)
        return result

    def _component_kernel(
        self,
        specification: KernelConfig | gpytorch.kernels.Kernel,
        dims: tuple[int, ...],
    ) -> gpytorch.kernels.Kernel:
        """Build one action/context component without rewriting ARD semantics."""

        assert self.model_input_dimension is not None
        if isinstance(specification, KernelConfig):
            if specification.ard_num_dims not in {None, len(dims)}:
                raise RoBOSMMDMatrixError(
                    "A component kernel's ard_num_dims must match its own "
                    "action or context coordinate dimension."
                )
            configured = replace(
                specification,
                active_dims=dims,
            )
            return build_kernel(configured, input_dimension=self.model_input_dimension)
        return self._activate_custom_kernel(specification, dims)

    def _build_joint_kernel(self) -> gpytorch.kernels.Kernel:
        """Build the product kernel or copy the configured custom joint kernel."""

        if self._joint_kernel_combination == "custom":
            if self._custom_joint_kernel is None:
                raise RoBOSMMDMatrixError(
                    "joint_kernel is required when joint_kernel_combination='custom'."
                )
            return copy.deepcopy(self._custom_joint_kernel)
        action_dims = tuple(range(self.action_dimension))
        assert self.model_input_dimension is not None
        context_dims = tuple(range(self.action_dimension, self.model_input_dimension))
        return gpytorch.kernels.ProductKernel(
            self._component_kernel(self._action_kernel_spec, action_dims),
            self._component_kernel(self._context_kernel_spec, context_dims),
        )

    def _pure_context_kernel(self) -> gpytorch.kernels.Kernel:
        """Build the context-only kernel used to define the MMD geometry."""

        specification = self._context_kernel_spec
        if isinstance(specification, KernelConfig):
            configured = replace(specification, active_dims=None)
            if configured.ard_num_dims is not None:
                configured = replace(configured, ard_num_dims=self.context_dimension)
            return build_kernel(configured, input_dimension=self.context_dimension).to(
                dtype=self.dtype, device=self.device
            )
        result = copy.deepcopy(specification).to(dtype=self.dtype, device=self.device)
        result.active_dims = None
        return result

    def _context_gram_matrix(self) -> torch.Tensor:
        """Return the symmetric ``(K, K)`` MMD Gram matrix.

        DRBO accepts positive-semidefinite matrices as seminorms. RoBOS sets
        ``require_tangent_pd`` and additionally requires strict positivity on
        every non-empty simplex tangent space so distinct distributions have
        nonzero MMD. A one-point simplex has no nonzero tangent direction and
        is therefore accepted as Basat's warned constant-context extension.

        Raises
        ------
        RoBOSMMDMatrixError
            If eigenvalue residuals exceed the configured feasibility tolerance
            or the RoBOS tangent-space condition fails.
        """

        kernel = self._pure_context_kernel()
        with torch.no_grad():
            gram = kernel(self.contexts).to_dense()
        gram = (gram + gram.transpose(-1, -2)) / 2.0
        eigenvalues = torch.linalg.eigvalsh(gram.double())
        tolerance = float(self.config.feasibility_tolerance)
        if float(eigenvalues.min().item()) < -tolerance:
            raise RoBOSMMDMatrixError("The context Gram matrix is not positive semidefinite.")
        if self.require_tangent_pd and not getattr(self.config, "allow_semidefinite_mmd", False):
            count = self.contexts.shape[0]
            if count > 1:
                directions = torch.eye(count, dtype=torch.float64)[:, :-1]
                directions[-1] = -1.0
                basis, _ = torch.linalg.qr(directions, mode="reduced")
                tangent = basis.transpose(-1, -2) @ gram.double().cpu() @ basis
                if float(torch.linalg.eigvalsh(tangent).min().item()) <= tolerance:
                    raise RoBOSMMDMatrixError(
                        "RoBOS requires positive definiteness on the simplex tangent space."
                    )
        return gram

    def context_gram_matrix(self) -> torch.Tensor:
        """Return a detached copy of the validated context MMD Gram matrix.

        Returns
        -------
        torch.Tensor
            Symmetric ``(K, K)`` matrix aligned with the bound context support.
            This is the same matrix used by the learner's distribution solver.
            External metrics should use it rather than recomputing an
            algebraically equivalent kernel with different floating-point
            arithmetic. No training rewards or hidden distributions are read.

        Notes
        -----
        Bind environment context support before calling this method. Mutating
        the returned tensor does not alter the algorithm's kernel or state.
        """
        if not self.contexts.numel():
            raise RoBOSContextSupportError(
                "Bind context support before requesting its Gram matrix."
            )
        return self._context_gram_matrix().detach().clone()

    def _joint_input(self, actions: Any, contexts: Any) -> torch.Tensor:
        """Concatenate broadcast-compatible action and context row matrices.

        Inputs have shapes ``(N, D_action)`` and ``(N, D_context)``; a singleton
        row on either side is broadcast.  The result has shape
        ``(N, D_action + D_context)`` in GP coordinates.
        """

        action_tensor = torch.as_tensor(actions, dtype=self.dtype, device=self.device)
        context_tensor = torch.as_tensor(contexts, dtype=self.dtype, device=self.device)
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        if context_tensor.ndim == 1:
            context_tensor = context_tensor.unsqueeze(0)
        if action_tensor.ndim != 2 or action_tensor.shape[1] != self.action_dimension:
            raise AlgorithmDataError("actions must have shape (N, D_env).")
        if context_tensor.ndim != 2 or context_tensor.shape[1] != self.context_dimension:
            raise AlgorithmDataError("contexts must have shape (N, D_context).")
        if action_tensor.shape[0] != context_tensor.shape[0]:
            if action_tensor.shape[0] == 1:
                action_tensor = action_tensor.expand(context_tensor.shape[0], -1)
            elif context_tensor.shape[0] == 1:
                context_tensor = context_tensor.expand(action_tensor.shape[0], -1)
            else:
                raise AlgorithmDataError("Action and context row counts do not align.")
        return torch.cat((action_tensor, context_tensor), dim=-1)

    def _current_gp_model_state(self) -> dict[str, torch.Tensor] | None:
        """Copy compatible GP module state for the next rebuilt training table.

        :meth:`GaussianProcess.set_data` necessarily constructs a fresh
        ``SingleTaskGP`` because the exact training table changes.  Keeping the
        preceding model's learned module values across that reconstruction is
        a numerical warm start only: the configured joint kernel and exact MLE
        objective remain unchanged.
        """

        if self.gp.model is None:
            return None
        return {name: value.detach().clone() for name, value in self.gp.model.state_dict().items()}

    def _restore_compatible_gp_model_state(
        self, model_state: Mapping[str, torch.Tensor] | None
    ) -> None:
        """Load shape-compatible warm-start values into the rebuilt joint GP."""

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
        """Seed GP-fit retries without changing the caller's ambient RNG."""

        cuda_devices: list[int] = []
        if self.device.type == "cuda":
            cuda_devices.append(
                torch.cuda.current_device() if self.device.index is None else self.device.index
            )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            yield

    def _initialize_response_scaled_gp(self) -> None:
        """Set a deterministic, data-scaled start for an otherwise identical MLE.

        This fallback is used only after a normal warm-started MLE exhausts its
        backend attempts.  It changes neither the GP family nor the optimized
        marginal likelihood.  Centering the constant mean and scaling each
        covariance factor prevents a large-offset response history from making
        every optimizer start numerically invalid.
        """

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

    def _match_contexts(self, contexts: torch.Tensor) -> None:
        """Require every realized context to match exactly one support row."""

        matches = torch.isclose(
            contexts.unsqueeze(1), self.contexts.unsqueeze(0), rtol=0.0, atol=1e-8
        ).all(dim=-1)
        if not matches.any(dim=1).all():
            raise RoBOSDataCompatibilityError(
                "A realized context is outside configured finite support.", algorithm=self
            )
        if (matches.sum(dim=1) != 1).any():
            raise RoBOSDataCompatibilityError(
                "A realized context ambiguously matches multiple support rows.",
                algorithm=self,
            )

    def _initialize_from_environment_impl(self, data: Any) -> None:
        """Load only learner-visible evaluated rows into the joint GP.

        Historical contexts are realized public feedback.  Hidden unqueried
        dataset rewards and true sampling distributions are never accessed.
        """

        actions = torch.as_tensor(data.actions, dtype=self.dtype, device=self.device)
        y = torch.as_tensor(data.observations, dtype=self.dtype, device=self.device).reshape(-1)
        if y.numel() == 0:
            return
        if not self.contexts.numel():
            if data.contexts is None:
                self.bind_environment_metadata({})
            else:
                historical = torch.as_tensor(data.contexts, dtype=self.dtype, device=self.device)
                if historical.ndim == 1:
                    historical = historical.unsqueeze(-1)
                # Only already observed context values are used to form support.
                self.bind_environment_metadata(
                    {
                        "context_support": torch.unique(historical, dim=0),
                        "context_support_is_dynamic": True,
                    }
                )
                self._warn_environment_default(
                    "context_support", "unique already-observed historical context rows"
                )
        if data.contexts is None:
            default = self.contexts[0]
            self._warn_environment_default("previous_data.contexts", str(default.tolist()))
            contexts = default.unsqueeze(0).expand(actions.shape[0], -1).clone()
        else:
            contexts = torch.as_tensor(data.contexts, dtype=self.dtype, device=self.device)
        if contexts.ndim == 1:
            contexts = contexts.unsqueeze(-1)
        if not getattr(self, "_dynamic_context_support", False):
            self._match_contexts(contexts)
        evaluated_raw = getattr(data, "evaluated_actions", None)
        evaluated = (
            actions
            if evaluated_raw is None
            else torch.as_tensor(evaluated_raw, dtype=self.dtype, device=self.device)
        )
        variance_raw = getattr(data, "observation_noise_variance", None)
        variance = (
            torch.zeros_like(y)
            if variance_raw is None
            else torch.as_tensor(variance_raw, dtype=self.dtype, device=self.device).reshape(-1)
        )
        self.state.actions_train = evaluated.detach().clone()
        self.state.contexts_train = contexts.detach().clone()
        self.state.joint_inputs_train = self._joint_input(evaluated, contexts)
        self.state.y_train = y.detach().clone()
        self.state.observation_noise_variance = variance.detach().clone()
        self.gp.set_data(self.state.joint_inputs_train, y, variance)

    def _prepare_discrete_candidates_impl(
        self,
        *,
        candidate_actions: torch.Tensor,
        candidate_dataset_indices: torch.Tensor,
        round_info: Mapping[str, Any],
    ) -> DiscreteAcquisitionCandidates:
        """Preserve exact finite actions and attach an expected-context model row.

        Robust scoring still evaluates every configured support context.  The
        expected-context joint row is bookkeeping only and does not reveal the
        environment's confidential sampling distribution.
        """

        self.bind_environment_metadata(round_info)
        reference = self._validate_reference(round_info["reference_distribution"])
        expected_context = reference @ self.contexts
        model_inputs = self._joint_input(candidate_actions, expected_context)
        return DiscreteAcquisitionCandidates(
            candidate_actions.detach().clone(),
            model_inputs,
            candidate_dataset_indices.detach().clone(),
        )

    def _validate_reference(self, value: Any) -> torch.Tensor:
        """Validate and tolerance-normalize a ``(K,)`` reference simplex vector."""

        try:
            reference = torch.as_tensor(value, dtype=self.dtype, device=self.device).reshape(-1)
        except (TypeError, ValueError, RuntimeError) as error:
            raise RoBOSReferenceDistributionError(
                "reference_distribution must be numeric.", algorithm=self
            ) from error
        if reference.shape != (self.contexts.shape[0],):
            raise RoBOSReferenceDistributionError(
                f"reference_distribution must have shape ({self.contexts.shape[0]},).",
                algorithm=self,
            )
        if not torch.isfinite(reference).all() or (reference < 0).any():
            raise RoBOSReferenceDistributionError(
                "reference_distribution must be finite and non-negative.", algorithm=self
            )
        total = reference.sum()
        if not torch.isclose(total, torch.ones_like(total), rtol=0.0, atol=1e-6):
            raise RoBOSReferenceDistributionError(
                "reference_distribution must sum to one within 1e-6.", algorithm=self
            )
        return reference / total

    def _beta(self) -> float:
        """Resolve the current finite non-negative confidence multiplier."""

        try:
            raw = (
                self.config.beta(self.iteration + 1)
                if callable(self.config.beta)
                else self.config.beta
            )
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise RoBOSFragilitySolverError(
                "beta must resolve to a numeric scalar.", algorithm=self
            ) from error
        if not math.isfinite(value) or value < 0:
            raise RoBOSFragilitySolverError("beta must be finite and non-negative.", algorithm=self)
        return value

    def _fit_model(self) -> None:
        """Fit the joint GP with warm-started, transactional exact MLE.

        The first attempt starts from the preceding round's fitted module
        values.  If the backend exhausts that attempt, one deterministic
        response-scaled initialization retries the same exact MLE.  Two
        failures restore the complete pre-fit GP so a later ``ask`` can be
        retried without retaining a partially optimized model.
        """

        if self.state.y_train.numel() == 0:
            return
        if self.config.fit_maxiter == 0:
            self.gp.set_data(
                self.state.joint_inputs_train,
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
                    self.state.joint_inputs_train,
                    self.state.y_train,
                    self.state.observation_noise_variance,
                )
                if attempt == 0:
                    self._restore_compatible_gp_model_state(warm_model_state)
                else:
                    self._initialize_response_scaled_gp()
                with self._owned_torch_rng(
                    seed=self.config.seed + self.iteration + attempt * _GP_RETRY_SEED_OFFSET
                ):
                    self.gp.fit(maxiter=self.config.fit_maxiter)
            except Exception as error:
                fit_errors.append(error)
                continue
            if attempt:
                exception_class = self.EXCEPTION_CLASS
                assert exception_class is not None
                warnings.warn(
                    f"{type(self).__name__}'s joint-GP MLE exhausted its warm-started "
                    "attempt; the same exact MLE succeeded from one deterministic "
                    "response-scaled initialization. The joint kernel, likelihood, "
                    "acquisition, action optimizer, and distribution solver are unchanged.",
                    exception_class.WARNING_TYPE,
                    stacklevel=3,
                )
                self.gp.last_fit_diagnostics["initialization"] = (
                    "deterministic_response_scaled_retry"
                )
            else:
                self.gp.last_fit_diagnostics["initialization"] = "warm_start"
            return

        try:
            self.gp.load_state_dict(previous_gp_state)
        except Exception as rollback_error:
            raise RoBOSFragilitySolverError(
                f"Joint GP fitting failed twice and rollback failed: {rollback_error}",
                algorithm=self,
            ) from rollback_error
        details = "; ".join(
            f"attempt {index}: {error}" for index, error in enumerate(fit_errors, start=1)
        )
        raise RoBOSFragilitySolverError(
            f"Joint GP fitting failed after two deterministic MLE attempts: {details}",
            algorithm=self,
        ) from fit_errors[-1]

    def _ucb_vector(self, action: torch.Tensor) -> torch.Tensor:
        """Return the ``(K,)`` UCB vector for one physical action."""

        joint = self._joint_input(action, self.contexts)
        mean, std = self.gp.predict(joint)
        quality_mean = mean if self.maximize else -mean
        return quality_mean + math.sqrt(self._beta()) * std

    def _ucb_matrix(self, actions: torch.Tensor) -> torch.Tensor:
        """Predict a context-major Cartesian grid once, retaining strided action columns."""
        count, support_count = len(actions), len(self.contexts)
        joint = torch.cat(
            (
                actions.unsqueeze(0).expand(support_count, -1, -1),
                self.contexts.unsqueeze(1).expand(-1, count, -1),
            ),
            dim=-1,
        ).reshape(support_count * count, -1)
        mean, std = self.gp.predict(joint)
        quality_mean = mean if self.maximize else -mean
        return (quality_mean + math.sqrt(self._beta()) * std).reshape(support_count, count)

    def _candidate_actions(
        self, round_info: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return exact data candidates or a deterministic continuous Sobol design."""

        if "candidate_actions" in round_info:
            table = self.prepare_discrete_candidates(
                candidate_actions=round_info["candidate_actions"],
                candidate_dataset_indices=round_info["candidate_dataset_indices"],
                round_info=round_info,
            )
            return table.environment_actions, table.dataset_indices
        return (
            self._sample_design(
                self.bounds,
                self.config.action_raw_samples,
                phase=(
                    "initial"
                    if self.state.y_train.numel() < self.config.initial_points
                    else "acquisition"
                ),
                seed=self.seed + 86028121 * (self.iteration + 1),
            ),
            None,
        )

    def _append_joint_feedback(
        self, feedback: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append public realized context, evaluated action, and observation.

        Returns the single context and evaluated-action rows plus the observation
        vector.  No audit-only true distribution is accepted or inferred.
        """

        context = torch.as_tensor(
            feedback["context"], dtype=self.dtype, device=self.device
        ).reshape(1, -1)
        self._match_contexts(context)
        evaluated = feedback["evaluated_action"].reshape(1, -1)
        observation = feedback["observation"].reshape(-1)
        raw_variance = feedback.get("observation_noise_variance")
        variance = (
            torch.zeros_like(observation)
            if raw_variance is None
            else torch.as_tensor(raw_variance, dtype=self.dtype, device=self.device).reshape(-1)
        )
        actions_train = torch.cat((self.state.actions_train, evaluated))
        contexts_train = torch.cat((self.state.contexts_train, context))
        joint_inputs_train = torch.cat(
            (self.state.joint_inputs_train, self._joint_input(evaluated, context))
        )
        y_train = torch.cat((self.state.y_train, observation))
        observation_noise_variance = torch.cat((self.state.observation_noise_variance, variance))
        previous_gp_state = self.gp.state_dict()
        warm_model_state = self._current_gp_model_state()
        try:
            self.gp.set_data(
                joint_inputs_train,
                y_train,
                observation_noise_variance,
            )
            self._restore_compatible_gp_model_state(warm_model_state)
        except Exception:
            self.gp.load_state_dict(previous_gp_state)
            raise
        self.state.actions_train = actions_train
        self.state.contexts_train = contexts_train
        self.state.joint_inputs_train = joint_inputs_train
        self.state.y_train = y_train
        self.state.observation_noise_variance = observation_noise_variance
        return context[0], evaluated[0], observation

    def _simplex_grid(self) -> torch.Tensor:
        """Return a validated ``(G, K)`` enumeration grid on the simplex.

        A user grid is used verbatim after residual checks.  Otherwise integer
        compositions at ``enumeration_resolution`` are generated, with a hard
        row cap that maps excessive grids to ``RoBOSFragilitySolverError``.
        """

        if self.config.distribution_grid is not None:
            try:
                grid = torch.as_tensor(
                    self.config.distribution_grid, dtype=self.dtype, device=self.device
                )
            except (TypeError, ValueError, RuntimeError) as error:
                raise RoBOSFragilitySolverError(
                    "distribution_grid must be numeric.", algorithm=self
                ) from error
        else:
            resolution = self.config.enumeration_resolution
            count = self.contexts.shape[0]
            compositions: list[list[int]] = []

            def visit(prefix: list[int], remaining: int, slots: int) -> None:
                """Recursively enumerate integer simplex compositions."""

                if len(compositions) > 200_000:
                    raise RoBOSFragilitySolverError(
                        "Generated simplex grid exceeds 200000 rows; supply distribution_grid.",
                        algorithm=self,
                    )
                if slots == 1:
                    compositions.append([*prefix, remaining])
                    return
                for value in range(remaining + 1):
                    visit([*prefix, value], remaining - value, slots - 1)

            visit([], resolution, count)
            grid = torch.tensor(compositions, dtype=self.dtype, device=self.device) / resolution
        if grid.ndim != 2 or grid.shape[1] != self.contexts.shape[0]:
            raise RoBOSFragilitySolverError(
                "distribution_grid has an invalid shape.", algorithm=self
            )
        sums = grid.sum(dim=1)
        if (
            not torch.isfinite(grid).all()
            or (grid < -self.config.feasibility_tolerance).any()
            or not torch.isclose(
                sums, torch.ones_like(sums), rtol=0.0, atol=self.config.feasibility_tolerance
            ).all()
        ):
            raise RoBOSFragilitySolverError(
                "distribution_grid contains invalid simplex rows.", algorithm=self
            )
        return grid

    @staticmethod
    def _stable_position(
        scores: torch.Tensor,
        secondary: torch.Tensor,
        actions: torch.Tensor,
        indices: torch.Tensor | None,
        *,
        minimize: bool,
    ) -> int:
        """Resolve robust-score ties by secondary score, row index, then action.

        Dataset ties use the smallest stable source index.  Continuous ties use
        the lexicographically smallest action; coordinate-wise normalization is
        monotone and therefore preserves that ordering.
        """

        finite = torch.isfinite(scores)
        if finite.any():
            target = scores[finite].min() if minimize else scores[finite].max()
            tolerance = _ABSOLUTE_TIE_TOLERANCE + _RELATIVE_TIE_TOLERANCE * max(
                1.0, abs(float(target.item()))
            )
            comparison = scores <= target + tolerance if minimize else scores >= target - tolerance
            tied = torch.nonzero(finite & comparison, as_tuple=False).reshape(-1)
        else:
            target = scores.min() if minimize else scores.max()
            tied = torch.nonzero(scores == target, as_tuple=False).reshape(-1)
        if tied.numel() > 1:
            best_secondary = secondary[tied].max()
            secondary_tolerance = _ABSOLUTE_TIE_TOLERANCE + _RELATIVE_TIE_TOLERANCE * max(
                1.0, abs(float(best_secondary.item()))
            )
            tied = tied[secondary[tied] >= best_secondary - secondary_tolerance]
        if indices is not None and tied.numel() > 1:
            tied = tied[torch.argmin(indices[tied])].reshape(1)
        if tied.numel() > 1:
            rows = to_numpy(actions[tied])
            order = np.lexsort(tuple(rows[:, i] for i in reversed(range(rows.shape[1]))))
            tied = tied[int(order[0])].reshape(1)
        return int(tied[0].item())


class RoBOS(_FiniteContextGPAlgorithm):
    """Minimum optimistic MMD-fragility action selection.

    Examples
    --------
    >>> import basat
    >>> bounds = basat.Bounds(lower=[-1.0, 0.0], upper=[1.0, 1.0])
    >>> algorithm = basat.RoBOS(
    ...     bounds=bounds,
    ...     config=basat.RoBOSConfig(
    ...         aspiration=0.0, initial_points=3, seed=7
    ...     ),
    ... )
    """

    ACQUISITION_POLICY = "minimum optimistic MMD fragility"
    CHECKPOINT_ADDED_CONFIG_DEFAULTS: ClassVar[Mapping[str, Any]] = {
        **_FiniteContextGPAlgorithm.CHECKPOINT_ADDED_CONFIG_DEFAULTS,
        "gp_config": None,
        "gp_backend": "standard",
        "gp_training_regularization": 0.0,
        "allow_semidefinite_mmd": False,
    }

    EXCEPTION_CLASS = RoBOSException
    native_batching = False
    native_environments = frozenset({"distributional"})
    required_round_info = frozenset({"reference_distribution"})
    required_feedback = frozenset({"observation", "evaluated_action", "context"})
    required_capabilities = frozenset({"distributional", "finite_context_support"})
    require_tangent_pd = True
    SOURCE = SourceRecord(
        paper_title="Robust Bayesian Satisficing",
        paper_year=2023,
        paper_url="https://proceedings.neurips.cc/paper_files/paper/2023/hash/daa098aa8e1fc718943ff1ab7b5b30c9-Abstract-Conference.html",
        repository_url="https://github.com/Bilkent-CYBORG/RoBOS",
        repository_status="official",
        notes="Finite-context RoBOS with strict MMD/simplex certificates.",
    )

    def __init__(
        self,
        *,
        bounds: Bounds,
        config: RoBOSConfig,
        maximize: bool = True,
        dtype: torch.dtype | str = torch.float32,
        device: torch.device | str = "cpu",
        plot: bool = False,
        verbose: bool = False,
        verbose_setting: bool = False,
        verbose_iter_by_iter: bool = False,
    ) -> None:
        """Initialize finite-context robust Bayesian satisficing.

        Parameters
        ----------
        bounds:
            Physical ``D_action``-dimensional action box.
        config:
            Immutable context support, aspiration, kernels, confidence schedule,
            inner/outer solver controls, batching, and seed configuration.
        maximize:
            Must be ``True`` in version 1.
        dtype, device:
            Tensor precision and execution device for GP and solver boundaries.
        plot, verbose, verbose_setting, verbose_iter_by_iter:
            Inherited presentation controls that do not change mathematics.
        """

        self.config = config
        self._validate_config(config)
        if not maximize:
            raise AlgorithmConfigurationError("RoBOS version 1 supports maximization only.")
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
        try:
            self._initialize_finite_context_model(
                contexts=None,
                action_kernel=config.action_kernel,
                context_kernel=config.context_kernel,
                joint_kernel_combination=config.joint_kernel_combination,
                joint_kernel=config.joint_kernel,
            )
        except RoBOSContextSupportError:
            raise
        except (TypeError, ValueError, RuntimeError) as error:
            raise RoBOSContextSupportError(str(error), algorithm=self) from error
        if config.joint_kernel_combination == "custom":
            self.invalidate_source_guarantees("basat.custom_robos_joint_kernel")
        self.state = self._empty_state()
        self._source_solver_rng = np.random.RandomState(self.seed % (2**32))
        self._source_solver_diagnostics: dict[str, Any] | None = None
        self._source_solver_starts: list[np.ndarray] = []
        self._source_solver_action_diagnostics: list[dict[str, Any]] = []
        if config.simplex_solver == "source_cobyla":
            self.invalidate_source_guarantees("basat.robos_source_cobyla_approximation")

    def get_source_solver_state(self) -> dict[str, Any]:
        """Return the JSON-safe MT19937 stream used by source COBYLA starts.

        A reproduction controller may exchange this state with its own random
        stream to preserve a released experiment's interleaved random draws.
        No benchmark values or hidden context distributions are accepted.
        Normal algorithm checkpoints include this state automatically.
        """
        raw_state: Any = self._source_solver_rng.get_state()
        name, keys, position, has_gauss, cached = raw_state
        return {
            "bit_generator": name,
            "keys": keys.tolist(),
            "position": int(position),
            "has_gaussian": int(has_gauss),
            "cached_gaussian": float(cached),
        }

    def set_source_solver_state(self, state: Mapping[str, Any]) -> None:
        """Restore a stream returned by :meth:`get_source_solver_state`.

        The state is copied and validated before replacing the algorithm-owned
        stream. This does not alter NumPy's global random state or the native
        acquisition generator. Malformed states raise ``RoBOSCheckpointError``.
        """
        try:
            if not isinstance(state, Mapping) or state.get("bit_generator") != "MT19937":
                raise ValueError("Expected an MT19937 mapping.")
            keys = np.asarray(state["keys"])
            position, has_gauss = state["position"], state["has_gaussian"]
            cached = float(state["cached_gaussian"])
            if (
                keys.shape != (624,)
                or not np.issubdtype(keys.dtype, np.integer)
                or (keys < 0).any()
                or (keys > np.iinfo(np.uint32).max).any()
                or isinstance(position, bool)
                or not isinstance(position, Integral)
                or not 0 <= int(position) <= 624
                or has_gauss not in (0, 1)
                or not math.isfinite(cached)
            ):
                raise ValueError("Invalid MT19937 state fields.")
            restored = np.random.RandomState()
            restored.set_state(
                ("MT19937", keys.astype(np.uint32), int(position), int(has_gauss), cached)
            )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise RoBOSCheckpointError(
                f"Invalid source solver RNG state: {error}", algorithm=self
            ) from error
        self._source_solver_rng = restored

    def set_source_solver_starts(self, starts: Any | None) -> None:
        """Set one-use simplex initial guesses for the source COBYLA solver.

        Parameters
        ----------
        starts : array-like of shape (N, K), or None
            One probability vector per upcoming numerical fragility solve,
            aligned with the finite context support. For :meth:`ask`, provide
            exactly one row per candidate action, in candidate-table order.
            ``None`` clears unused guesses and restores algorithm-owned random
            starts. Supplied guesses do not advance the internal random stream.

        Notes
        -----
        This source-only numerical interface permits reproduction controllers
        to preserve interleaved random-start draws. It accepts no benchmark
        rewards, hidden context probabilities, or selected actions. Each guess
        is consumed once; unused guesses are included in checkpoints. Do not
        reuse a start batch after resuming a checkpoint that already consumed it.
        """
        if self.config.simplex_solver != "source_cobyla":
            raise AlgorithmConfigurationError(
                "Explicit source solver starts require simplex_solver='source_cobyla'."
            )
        if starts is None:
            self._source_solver_starts = []
            return
        try:
            array = np.asarray(to_numpy(torch.as_tensor(starts, dtype=torch.float64)), dtype=float)
            expected = int(self.contexts.shape[0])
            if (
                array.ndim != 2
                or array.shape[0] == 0
                or array.shape[1] == 0
                or (expected and array.shape[1] != expected)
                or not np.isfinite(array).all()
                or (array < 0).any()
                or np.max(np.abs(array.sum(axis=1) - 1.0)) > self.config.feasibility_tolerance
            ):
                raise ValueError("Expected finite nonnegative (N, K) rows summing to one.")
        except (TypeError, ValueError, RuntimeError) as error:
            raise AlgorithmConfigurationError(f"Invalid source solver starts: {error}") from error
        self._source_solver_starts = [row.copy() for row in array]

    def evaluate_fragility(
        self, upper_bounds: Any, reference_distribution: Any, gram_matrix: Any | None = None
    ) -> tuple[float, torch.Tensor, dict[str, float]]:
        """Evaluate the configured fragility program on supplied finite values.

        Parameters
        ----------
        upper_bounds : array-like of shape (K,)
            A finite value per context. During acquisition these are optimistic
            posterior bounds; a separate metric-only instance may evaluate
            known test values without passing them to the learning algorithm.
        reference_distribution : array-like of shape (K,)
            Nonnegative reference probabilities summing to one.
        gram_matrix : array-like of shape (K, K), optional
            Symmetric positive-semidefinite MMD Gram matrix. If omitted, use
            this instance's configured context kernel and context support.

        Returns
        -------
        value : float
            Signed fragility, possibly positive infinity in native modes.
        distribution : torch.Tensor
            Maximizing or source-approximate context distribution.
        residuals : dict
            Independently measured simplex feasibility residuals. Source COBYLA
            is an approximate historical numerical protocol, not a certificate.

        Notes
        -----
        This does not fit a GP, select an action, or add training observations.
        Source mode consumes one supplied or algorithm-generated solver start.
        """
        values = torch.as_tensor(upper_bounds, dtype=self.dtype, device=self.device)
        reference = torch.as_tensor(reference_distribution, dtype=self.dtype, device=self.device)
        if (
            values.ndim != 1
            or values.numel() == 0
            or reference.shape != values.shape
            or not bool(torch.isfinite(values).all())
            or not bool(torch.isfinite(reference).all())
            or bool((reference < 0).any())
            or abs(float(reference.sum()) - 1.0) > self.config.feasibility_tolerance
        ):
            raise AlgorithmConfigurationError(
                "Fragility requires aligned finite vectors and simplex probabilities."
            )
        gram = (
            self._context_gram_matrix()
            if gram_matrix is None
            else torch.as_tensor(gram_matrix, dtype=self.dtype, device=self.device)
        )
        if (
            gram.shape != (values.numel(), values.numel())
            or not bool(torch.isfinite(gram).all())
            or not torch.allclose(gram, gram.T, atol=self.config.feasibility_tolerance, rtol=0.0)
        ):
            raise RoBOSMMDMatrixError(
                "Fragility Gram matrix must be finite, square, and symmetric.", algorithm=self
            )
        if float(torch.linalg.eigvalsh(gram).min()) < -self.config.feasibility_tolerance:
            raise RoBOSMMDMatrixError(
                "Fragility Gram matrix must be positive semidefinite.", algorithm=self
            )
        return self._solve_fragility_program(values, reference, gram)

    @staticmethod
    def _validate_config(config: RoBOSConfig) -> None:
        """Validate sequential, kernel, aspiration, simplex, and solver controls."""

        if not isinstance(config, RoBOSConfig):
            raise AlgorithmConfigurationError("config must be RoBOSConfig.")
        if config.gp_config is not None and not isinstance(config.gp_config, GaussianProcessConfig):
            raise AlgorithmConfigurationError("gp_config must be GaussianProcessConfig or None.")
        _FiniteContextGPAlgorithm._validate_gp_backend(config)
        if not isinstance(config.allow_semidefinite_mmd, bool):
            raise AlgorithmConfigurationError("allow_semidefinite_mmd must be bool.")
        _, q_batch = resolve_batch_configuration(
            batch_size=config.batch_size, q_batch=config.q_batch
        )
        if q_batch is not None and q_batch > 1:
            raise AlgorithmConfigurationError(
                "RoBOS does not define a joint q-acquisition; use batch_size for repeated "
                "independent decisions under the same reference distribution."
            )
        if (
            isinstance(config.aspiration, bool)
            or not isinstance(config.aspiration, Real)
            or not math.isfinite(config.aspiration)
        ):
            raise RoBOSAspirationError("aspiration must be finite.")
        if config.joint_kernel_combination not in {"product", "custom"}:
            raise AlgorithmConfigurationError("joint_kernel_combination must be product or custom.")
        if config.joint_kernel_combination == "custom" and config.joint_kernel is None:
            raise AlgorithmConfigurationError(
                "joint_kernel is required for joint_kernel_combination='custom'."
            )
        if config.joint_kernel_combination == "product" and config.joint_kernel is not None:
            raise AlgorithmConfigurationError(
                "joint_kernel is accepted only for joint_kernel_combination='custom'."
            )
        if config.simplex_solver not in {"socp", "slsqp", "enumeration", "source_cobyla"}:
            raise AlgorithmConfigurationError("Unknown simplex_solver.")
        if config.distribution_grid is not None and config.simplex_solver != "enumeration":
            raise AlgorithmConfigurationError(
                "distribution_grid is accepted only for simplex_solver='enumeration'."
            )
        if isinstance(config.beta, str) or (
            not callable(config.beta)
            and (
                isinstance(config.beta, bool)
                or not isinstance(config.beta, Real)
                or not math.isfinite(config.beta)
                or config.beta < 0
            )
        ):
            raise AlgorithmConfigurationError(
                "beta must be callable or a finite non-negative scalar."
            )
        if (
            isinstance(config.distribution_floor, bool)
            or not isinstance(config.distribution_floor, Real)
            or not math.isfinite(config.distribution_floor)
            or config.distribution_floor < 0
        ):
            raise AlgorithmConfigurationError("distribution_floor must be non-negative.")
        if (
            isinstance(config.enumeration_resolution, bool)
            or not isinstance(config.enumeration_resolution, int)
            or config.enumeration_resolution < 1
        ):
            raise AlgorithmConfigurationError("enumeration_resolution must be positive.")
        for name in (
            "action_raw_samples",
            "action_restarts",
            "action_maxiter",
            "initial_points",
            "fit_maxiter",
        ):
            value = getattr(config, name)
            minimum = 0 if name == "fit_maxiter" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise AlgorithmConfigurationError(f"{name} must be an integer >= {minimum}.")
        if any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            for value in (config.feasibility_tolerance, config.optimizer_tolerance)
        ):
            raise AlgorithmConfigurationError("Solver tolerances must be finite.")
        if config.feasibility_tolerance <= 0 or config.optimizer_tolerance <= 0:
            raise AlgorithmConfigurationError("Solver tolerances must be positive.")
        if config.simplex_solver == "socp" and config.distribution_floor > 0:
            raise AlgorithmConfigurationError(
                "The SOCP bisection cannot encode a positive MMD distance floor; "
                "use simplex_solver='slsqp' or 'enumeration'."
            )
        if config.simplex_solver == "source_cobyla" and config.distribution_floor != 0:
            raise AlgorithmConfigurationError(
                "source_cobyla reproduces MMD+1e-9 and does not accept a distribution_floor."
            )

    def _empty_state(self) -> RoBOSState:
        """Create empty tensor histories in their explicit coordinate systems."""

        assert self.model_input_dimension is not None
        return RoBOSState(
            actions_train=torch.empty(
                (0, self.action_dimension), dtype=self.dtype, device=self.device
            ),
            contexts_train=torch.empty(
                (0, self.context_dimension), dtype=self.dtype, device=self.device
            ),
            joint_inputs_train=torch.empty(
                (0, self.model_input_dimension), dtype=self.dtype, device=self.device
            ),
            y_train=torch.empty(0, dtype=self.dtype, device=self.device),
            observation_noise_variance=torch.empty(0, dtype=self.dtype, device=self.device),
            reference_history=[],
            selected_actions=[],
            selected_fragilities=[],
            reference_values=[],
            worst_distributions=[],
            realized_contexts=[],
            aspiration_satisfied=[],
        )

    def _mmd_distance(
        self, distribution: torch.Tensor, reference: torch.Tensor, gram: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate ``sqrt((w-reference)^T M (w-reference))`` for simplex rows.

        ``distribution`` may have shape ``(K,)`` or ``(..., K)``; ``reference``
        has shape ``(K,)`` and ``gram`` has shape ``(K, K)``.
        """

        difference = distribution - reference
        squared = torch.einsum("...i,ij,...j->...", difference, gram, difference)
        return squared.clamp_min(0.0).sqrt()

    def _solver_residuals(
        self, distribution: torch.Tensor, reference: torch.Tensor, gram: torch.Tensor
    ) -> dict[str, float]:
        """Recompute simplex, nonnegativity, MMD, and distance-floor residuals."""

        distance = float(self._mmd_distance(distribution, reference, gram).item())
        return {
            "simplex_sum": abs(float(distribution.sum().item()) - 1.0),
            "nonnegativity": max(0.0, -float(distribution.min().item())),
            "distance": distance,
            "distance_floor": max(0.0, self.config.distribution_floor - distance),
        }

    def _certify_fragility_distribution(
        self,
        distribution: torch.Tensor,
        reference: torch.Tensor,
        gram: torch.Tensor,
        *,
        require_distance: bool,
    ) -> dict[str, float]:
        """Independently verify simplex and distance-floor feasibility."""

        residuals = self._solver_residuals(distribution, reference, gram)
        tolerance = self.config.feasibility_tolerance
        names = ["simplex_sum", "nonnegativity"]
        if require_distance:
            names.append("distance_floor")
        if any(residuals[name] > tolerance for name in names):
            raise RoBOSFragilitySolverError(
                f"Fragility solution failed independent residual checks: {residuals}.",
                algorithm=self,
            )
        if require_distance and residuals["distance"] <= self.config.distribution_floor:
            raise RoBOSFragilitySolverError(
                "Fragility solution did not leave the excluded MMD neighborhood.",
                algorithm=self,
            )
        return residuals

    def _ratio(
        self,
        distributions: torch.Tensor,
        ucb: torch.Tensor,
        reference: torch.Tensor,
        gram: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate optimistic fragility ratios on ``(..., K)`` distributions.

        Rows inside the excluded MMD neighborhood receive ``-inf`` and cannot
        be selected by an inner solver.
        """

        denominator = self._mmd_distance(distributions, reference, gram)
        numerator = self.config.aspiration - distributions @ ucb
        floor = max(self.config.distribution_floor, self.config.feasibility_tolerance)
        return torch.where(
            denominator > floor,
            numerator / denominator,
            torch.full_like(numerator, -torch.inf),
        )

    def _solve_fragility_program(
        self, ucb: torch.Tensor, reference: torch.Tensor, gram: torch.Tensor
    ) -> tuple[float, torch.Tensor, dict[str, float]]:
        """Maximize optimistic fragility over the finite-context simplex.

        Parameters
        ----------
        ucb, reference:
            Aligned ``(K,)`` UCB and reference-distribution vectors.
        gram:
            Certified ``(K, K)`` MMD Gram matrix.

        Returns
        -------
        fragility, worst_distribution, residuals:
            Signed optimistic ratio, a certified ``(K,)`` simplex vector, and
            independently recomputed feasibility diagnostics.

        Raises
        ------
        RoBOSFragilitySolverError
            If enumeration, deterministic multistart SLSQP, or SOCP
            produces no independently feasible solution.
        """

        if self.config.simplex_solver == "source_cobyla":
            return self._solve_fragility_source_cobyla(ucb, reference, gram)
        if reference.numel() == 1 and float(ucb[0].detach()) >= self.config.aspiration:
            return (
                0.0,
                reference.detach().clone(),
                self._certify_fragility_distribution(
                    reference, reference, gram, require_distance=False
                ),
            )
        if float((reference @ ucb).detach()) < self.config.aspiration:
            # At the reference distribution the aspiration deficit is
            # positive and the MMD is zero, so fragility is infinite. An
            # optimizer approaching this singularity is not a valid finite
            # certificate. Keep the mathematical infinity for outer selection.
            return (
                math.inf,
                reference.detach().clone(),
                self._certify_fragility_distribution(
                    reference, reference, gram, require_distance=False
                ),
            )
        # Equation (7) allows NEGATIVE optimistic fragility even though true
        # fragility is non-negative. When all deficits are non-positive, the
        # maximum ratio is attained at a simplex vertex: convexity of the MMD
        # norm bounds any mixture's ratio by its best vertex ratio. Clipping
        # this value to zero changes the early exploration policy.
        if bool((ucb >= self.config.aspiration).all()):
            vertices = (
                self._simplex_grid()
                if self.config.simplex_solver == "enumeration"
                else torch.eye(reference.numel(), dtype=self.dtype, device=self.device)
            )
            ratios = self._ratio(vertices, ucb, reference, gram)
            position = int(torch.argmax(ratios))
            worst = vertices[position].detach().clone()
            return (
                float(ratios[position].detach()),
                worst,
                self._certify_fragility_distribution(worst, reference, gram, require_distance=True),
            )
        if self.config.simplex_solver == "enumeration":
            grid = self._simplex_grid()
            ratios = self._ratio(grid, ucb, reference, gram)
            eligible = torch.isfinite(ratios)
            if not eligible.any():
                raise RoBOSFragilitySolverError(
                    "distribution_grid contains no row outside the MMD distance floor.",
                    algorithm=self,
                )
            eligible_positions = torch.nonzero(eligible, as_tuple=False).reshape(-1)
            position = int(eligible_positions[torch.argmax(ratios[eligible_positions])].item())
            worst = grid[position]
        elif self.config.simplex_solver == "socp":
            worst = self._solve_fragility_socp(ucb, reference, gram)
        else:
            try:
                from scipy.optimize import minimize
            except ImportError as error:  # pragma: no cover - SciPy is a declared core dependency
                raise RoBOSFragilitySolverError("SLSQP requires scipy.", algorithm=self) from error
            u = to_numpy(ucb.double())
            ref = to_numpy(reference.double())
            matrix = to_numpy(gram.double())
            floor = max(self.config.distribution_floor, self.config.feasibility_tolerance)

            def objective(w: np.ndarray) -> float:
                """Return negative clipped fragility for one SciPy simplex row."""

                difference = w - ref
                denominator = math.sqrt(max(float(difference @ matrix @ difference), 0.0))
                if denominator <= floor:
                    return 1e12
                return -max(0.0, (self.config.aspiration - float(w @ u)) / denominator)

            starts = [ref, np.full_like(ref, 1.0 / ref.size)]
            starts.extend(np.eye(ref.size))
            best_distribution: torch.Tensor | None = None
            best_value = -math.inf
            for start in starts:
                try:
                    result = minimize(
                        objective,
                        start,
                        method="SLSQP",
                        bounds=[(0.0, 1.0)] * ref.size,
                        constraints={
                            "type": "eq",
                            "fun": lambda w: float(w.sum() - 1.0),
                        },
                        options={
                            "ftol": min(
                                self.config.optimizer_tolerance,
                                self.config.feasibility_tolerance / 10.0,
                            ),
                            "maxiter": 1000,
                        },
                    )
                except (FloatingPointError, RuntimeError, ValueError):
                    continue
                try:
                    result_x = np.asarray(result.x, dtype=float)
                    result_fun = float(result.fun)
                except (TypeError, ValueError):
                    continue
                if (
                    not result.success
                    or result_x.shape != ref.shape
                    or not math.isfinite(result_fun)
                    or not np.isfinite(result_x).all()
                ):
                    continue
                candidate = torch.as_tensor(result_x, dtype=self.dtype, device=self.device)
                try:
                    self._certify_fragility_distribution(
                        candidate, reference, gram, require_distance=True
                    )
                except RoBOSFragilitySolverError:
                    continue
                candidate_value = float(
                    self._ratio(candidate.unsqueeze(0), ucb, reference, gram)[0].item()
                )
                if not math.isfinite(candidate_value):
                    continue
                if candidate_value > best_value:
                    best_distribution = candidate.detach().clone()
                    best_value = candidate_value
                elif candidate_value == best_value and best_distribution is not None:
                    if tuple(to_numpy(candidate)) < tuple(to_numpy(best_distribution)):
                        best_distribution = candidate.detach().clone()
            if best_distribution is None:
                raise RoBOSFragilitySolverError(
                    "No SLSQP fragility solution converged.", algorithm=self
                )
            worst = best_distribution
        residuals = self._certify_fragility_distribution(
            worst, reference, gram, require_distance=True
        )
        value = float(self._ratio(worst.unsqueeze(0), ucb, reference, gram)[0].item())
        if not math.isfinite(value):
            raise RoBOSFragilitySolverError(
                "The independently recomputed fragility is non-finite.",
                algorithm=self,
            )
        return max(0.0, value), worst.detach().clone(), residuals

    def _solve_fragility_source_cobyla(
        self, ucb: torch.Tensor, reference: torch.Tensor, gram: torch.Tensor
    ) -> tuple[float, torch.Tensor, dict[str, float]]:
        """Evaluate the pinned author's single-start regularized ratio solver.

        This intentionally does not substitute the exact negative-vertex or
        infinite-reference shortcuts: the released notebook optimizes its
        regularized ratio for every action. Feasibility and termination are
        independently reported; finite un-converged source output is retained
        with an algorithm warning, never described as a certified optimum.
        """
        import scipy
        from scipy.optimize import minimize

        u = to_numpy(ucb.double()).reshape(-1)
        ref = to_numpy(reference.double()).reshape(-1)
        matrix = to_numpy(gram.double())
        if self._source_solver_starts:
            initial = self._source_solver_starts[0]
            if initial.shape != ref.shape:
                raise AlgorithmConfigurationError(
                    "Source solver start does not match context cardinality."
                )
            self._source_solver_starts.pop(0)
        else:
            cumulative = np.r_[0.0, np.sort(self._source_solver_rng.rand(len(ref) - 1)), 1.0]
            initial = np.diff(cumulative)

        def ratio(probabilities: np.ndarray) -> float:
            difference = (ref - probabilities).reshape(-1, 1)
            squared = float((difference.T @ (matrix @ difference)).item())
            if squared < 0:
                # A negative quadratic form is not an admissible real MMD.
                # Preserve the source objective's NaN rather than silently
                # changing it to a clipped norm during optimization.
                return math.nan
            expected = float(np.dot(u.reshape(1, -1), probabilities.reshape(-1, 1)).item())
            return float((self.config.aspiration - expected) / (math.sqrt(squared) + 1e-9))

        result = minimize(
            lambda probabilities: -ratio(probabilities),
            initial,
            method="COBYLA",
            constraints=[
                {"type": "ineq", "fun": lambda probabilities: probabilities},
                {"type": "ineq", "fun": lambda probabilities: np.sum(probabilities) - 1},
                {"type": "ineq", "fun": lambda probabilities: 1 - np.sum(probabilities)},
            ],
            options={"maxiter": 1000},
        )
        probabilities = np.asarray(result.x, dtype=float)
        value = ratio(probabilities)
        if (
            probabilities.shape != ref.shape
            or not np.isfinite(probabilities).all()
            or not math.isfinite(value)
        ):
            raise RoBOSFragilitySolverError(
                "Source COBYLA returned a non-finite fragility or malformed distribution.",
                algorithm=self,
            )
        residuals = {
            "simplex_sum": abs(float(probabilities.sum()) - 1.0),
            "nonnegativity": max(0.0, -float(probabilities.min())),
            "distance": 0.0,
        }
        feasible = max(residuals.values()) <= self.config.feasibility_tolerance
        self._source_solver_diagnostics = {
            "solver": "source_cobyla",
            "scipy_version": scipy.__version__,
            "backend": "powell_fortran"
            if tuple(int(v) for v in scipy.__version__.split(".")[:2]) < (1, 16)
            else "prima",
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "evaluations": int(result.nfev),
            "regularizer": 1e-9,
            "feasible": feasible,
            "residuals": dict(residuals),
            "global_optimum_certified": False,
        }
        if not result.success or not feasible:
            warnings.warn(
                "Basat RoBOS source COBYLA retained approximate author-protocol output: "
                f"status={result.status}, evaluations={result.nfev}, "
                f"simplex residual={residuals['simplex_sum']:.3g}, "
                f"negative mass={residuals['nonnegativity']:.3g}. {result.message}",
                RoBOSWarning,
                stacklevel=2,
            )
        return (
            value,
            torch.as_tensor(probabilities, dtype=self.dtype, device=self.device),
            residuals,
        )

    def _solve_fragility_socp(
        self, ucb: torch.Tensor, reference: torch.Tensor, gram: torch.Tensor
    ) -> torch.Tensor:
        """Solve positive fragility with an exact homogeneous SOCP transform.

        Put ``z = t*w`` and require ``tau*t - ucb@z = 1``. Minimizing
        ``||M**.5 (z-t*reference)||`` then gives the reciprocal of the
        maximal positive deficit-to-distance ratio. This avoids dozens of
        separate bisection programs per finite action without approximating
        the probability simplex or changing the fragility objective.
        """
        try:
            import cvxpy as cp
        except ImportError as error:
            raise RoBOSFragilitySolverError("SOCP requires cvxpy.", algorithm=self) from error
        u = to_numpy(ucb.double())
        ref = to_numpy(reference.double())
        matrix = to_numpy(gram.double())
        key = (ref.tobytes(), matrix.tobytes())
        cached = getattr(self, "_socp_problem_cache", None)
        if cached is None or cached[0] != key:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            factor = np.sqrt(np.clip(eigenvalues, 0.0, None))[:, None] * eigenvectors.T
            z = cp.Variable(ref.size, nonneg=True)
            scale = cp.Variable(nonneg=True)
            coefficients = cp.Parameter(ref.size)
            problem = cp.Problem(
                cp.Minimize(cp.norm(factor @ (z - scale * ref), 2)),
                [cp.sum(z) == scale, self.config.aspiration * scale - coefficients @ z == 1],
            )
            self._socp_problem_cache = (key, z, scale, coefficients, problem)
        else:
            _, z, scale, coefficients, problem = cached
        coefficients.value = u
        tolerance = min(self.config.optimizer_tolerance, self.config.feasibility_tolerance / 10)
        try:
            problem.solve(solver="CLARABEL", tol_gap_abs=tolerance, tol_feas=tolerance)
        except Exception as error:
            raise RoBOSFragilitySolverError(
                f"Fragility SOCP failed: {error}", algorithm=self
            ) from error
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RoBOSFragilitySolverError(
                f"Fragility SOCP returned {problem.status}.", algorithm=self
            )
        if scale.value is None or z.value is None or float(scale.value) <= 0:
            raise RoBOSFragilitySolverError(
                "Fragility SOCP returned no positive scale.", algorithm=self
            )
        worst = torch.as_tensor(
            np.asarray(z.value) / float(scale.value), dtype=self.dtype, device=self.device
        )
        self._certify_fragility_distribution(worst, reference, gram, require_distance=True)
        return worst

    def _solve_fragility_socp_bisection(
        self, ucb: torch.Tensor, reference: torch.Tensor, gram: torch.Tensor
    ) -> torch.Tensor:
        """Solve the quasiconcave ratio by certified SOCP feasibility bisection.

        CVXPY operates on NumPy ``(K,)`` vectors and the factorization of the
        ``(K, K)`` MMD matrix.  Every bisection candidate is independently
        checked against the simplex and ratio constraints at
        ``feasibility_tolerance`` before it can update the bracket.

        Raises
        ------
        RoBOSFragilitySolverError
            If CVXPY is unavailable, the ratio is directionally degenerate, no
            feasible bracket exists, or the certified solve returns no vector.
        """

        try:
            import cvxpy as cp
        except (
            ImportError
        ) as error:  # pragma: no cover - optional dependency availability is host-specific
            raise RoBOSFragilitySolverError(
                "SOCP requires the 'robust' extra: pip install basat[robust].",
                algorithm=self,
            ) from error
        u = to_numpy(ucb.double())
        ref = to_numpy(reference.double())
        matrix = to_numpy(gram.double())
        reference_deficit = self.config.aspiration - float(ref @ u)
        if abs(reference_deficit) <= self.config.feasibility_tolerance:
            raise RoBOSFragilitySolverError(
                "SOCP ratio is directionally degenerate at the reference; use SLSQP or enumeration.",
                algorithm=self,
            )
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        factor = np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
        variable = cp.Variable(ref.size)
        kappa = cp.Parameter(nonneg=True)
        distance = cp.norm(factor @ (variable - ref), 2)
        constraints = [
            variable >= 0,
            cp.sum(variable) == 1,
            self.config.aspiration - u @ variable >= kappa * distance,
        ]
        problem = cp.Problem(cp.Minimize(0), constraints)

        def feasible(value: float) -> tuple[bool, np.ndarray | None]:
            """Solve and independently certify SOCP feasibility at one ratio."""

            kappa.value = value
            try:
                problem.solve(
                    solver="CLARABEL",
                    tol_gap_abs=min(
                        self.config.optimizer_tolerance,
                        self.config.feasibility_tolerance / 10,
                    ),
                    tol_feas=min(
                        self.config.optimizer_tolerance,
                        self.config.feasibility_tolerance / 10,
                    ),
                )
            except Exception:
                return False, None
            if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or variable.value is None:
                return False, None
            candidate = np.asarray(variable.value, dtype=float).reshape(-1)
            if candidate.shape != ref.shape or not np.isfinite(candidate).all():
                return False, None
            sum_residual = abs(float(candidate.sum()) - 1.0)
            nonnegative_residual = max(0.0, -float(candidate.min()))
            difference = candidate - ref
            candidate_distance = math.sqrt(max(float(difference @ matrix @ difference), 0.0))
            ratio_residual = max(
                0.0,
                value * candidate_distance - (self.config.aspiration - float(u @ candidate)),
            )
            tolerance = self.config.feasibility_tolerance
            certified = (
                sum_residual <= tolerance
                and nonnegative_residual <= tolerance
                and ratio_residual <= tolerance
            )
            return certified, candidate if certified else None

        lower, upper = 0.0, 1.0
        best: np.ndarray | None = None
        is_feasible, candidate = feasible(lower)
        if not is_feasible:
            raise RoBOSFragilitySolverError("RoBOS SOCP is infeasible at kappa=0.", algorithm=self)
        best = candidate
        for _ in range(60):
            is_feasible, candidate = feasible(upper)
            if not is_feasible:
                break
            best = candidate
            lower, upper = upper, 2.0 * upper
        else:
            raise RoBOSFragilitySolverError(
                "RoBOS SOCP failed to bracket fragility.", algorithm=self
            )
        for _ in range(60):
            if upper - lower <= self.config.optimizer_tolerance:
                break
            middle = (lower + upper) / 2.0
            is_feasible, candidate = feasible(middle)
            if is_feasible:
                lower, best = middle, candidate
            else:
                upper = middle
        if best is None:
            raise RoBOSFragilitySolverError("RoBOS SOCP returned no distribution.", algorithm=self)
        return torch.as_tensor(best, dtype=self.dtype, device=self.device)

    def _optimistic_fragility(
        self, action: torch.Tensor, reference: torch.Tensor, gram: torch.Tensor
    ) -> tuple[float, torch.Tensor, float, dict[str, float]]:
        """Evaluate one action's UCB vector and optimistic fragility.

        If reference expected UCB is below aspiration, fragility is ``+inf`` and
        the inner program is intentionally skipped.
        """

        ucb = self._ucb_vector(action)
        reference_value = float((reference @ ucb).item())
        if (
            reference_value < self.config.aspiration
            and self.config.simplex_solver != "source_cobyla"
        ):
            return (
                math.inf,
                reference.detach().clone(),
                reference_value,
                {
                    "simplex_sum": 0.0,
                    "nonnegativity": 0.0,
                    "distance": 0.0,
                },
            )
        fragility, worst, residuals = self._solve_fragility_program(ucb, reference, gram)
        return fragility, worst, reference_value, residuals

    def _select_minimum_fragility_action(
        self,
        actions: torch.Tensor,
        reference: torch.Tensor,
        indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, float, torch.Tensor, float, dict[str, float]]:
        """Score finite actions and select minimum fragility with stable ties.

        ``actions`` has shape ``(N, D_action)``.  Ties prefer larger reference
        expected UCB, then the smallest dataset index for finite data.
        """

        gram = self._context_gram_matrix()
        values: list[float] = []
        worst: list[torch.Tensor] = []
        references: list[float] = []
        residuals: list[dict[str, float]] = []
        if self._source_solver_starts and len(self._source_solver_starts) != len(actions):
            raise AlgorithmConfigurationError(
                "Source solver starts must have exactly one row per candidate action."
            )
        if any(row.shape != (reference.numel(),) for row in self._source_solver_starts):
            raise AlgorithmConfigurationError(
                "Source solver starts do not match context cardinality."
            )
        self._source_solver_action_diagnostics = []
        source_upper = None
        if self.config.simplex_solver == "source_cobyla":
            # The released runner predicts its context-major Cartesian grid
            # once, then passes strided (K,) columns to COBYLA. Besides avoiding
            # repeated posterior calls, this preserves the source's numerical
            # prediction order and storage layout for its sensitive optimizer.
            source_upper = self._ucb_matrix(actions)
        for index, action in enumerate(actions):
            if source_upper is None:
                value, distribution, reference_value, certificate = self._optimistic_fragility(
                    action, reference, gram
                )
            else:
                upper = source_upper[:, index]
                reference_value = float((reference @ upper).detach())
                value, distribution, certificate = self._solve_fragility_program(
                    upper, reference, gram
                )
            values.append(value)
            worst.append(distribution)
            references.append(reference_value)
            residuals.append(certificate)
            if self._source_solver_diagnostics is not None:
                self._source_solver_action_diagnostics.append(
                    copy.deepcopy(self._source_solver_diagnostics)
                )
        score_tensor = torch.tensor(values, dtype=self.dtype, device=self.device)
        reference_tensor = torch.tensor(references, dtype=self.dtype, device=self.device)
        position = (
            int(torch.argmin(score_tensor))
            if self.config.simplex_solver == "source_cobyla"
            else self._stable_position(
                score_tensor, reference_tensor, actions, indices, minimize=True
            )
        )
        certificate = dict(residuals[position])
        certificate["outer_bounds"] = self._action_bound_residual(actions[position])
        return (
            actions[position],
            values[position],
            worst[position],
            references[position],
            certificate,
        )

    def _action_bound_residual(self, action: torch.Tensor) -> float:
        """Return the independently recomputed box-feasibility violation."""

        lower, upper = self._bounds_on_algorithm_device()
        violation = torch.maximum(lower - action, action - upper).clamp_min(0.0)
        return float(violation.max().item())

    def _refine_continuous_action(
        self,
        raw_actions: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, float, torch.Tensor, float, dict[str, float]]:
        """Refine deterministic RoBOS starts over the continuous action box.

        The inner fragility program is rerun after every successful local solve.
        Raw Sobol points rank the deterministic starts but are not a fallback:
        failure of every restart raises the algorithm-specific solver error.
        """

        try:
            from scipy.optimize import minimize
        except ImportError as error:  # pragma: no cover - SciPy is a core dependency
            raise RoBOSFragilitySolverError(
                "Continuous RoBOS action refinement requires scipy.", algorithm=self
            ) from error

        gram = self._context_gram_matrix()
        raw_fragilities: list[float] = []
        raw_references: list[float] = []
        for action in raw_actions:
            fragility, _, reference_value, _ = self._optimistic_fragility(action, reference, gram)
            raw_fragilities.append(fragility)
            raw_references.append(reference_value)

        lower, upper = self._bounds_on_algorithm_device()
        normalized = (raw_actions - lower) / (upper - lower)
        order = sorted(
            range(raw_actions.shape[0]),
            key=lambda position: (
                0 if math.isfinite(raw_fragilities[position]) else 1,
                raw_fragilities[position] if math.isfinite(raw_fragilities[position]) else 0.0,
                -raw_references[position],
                tuple(float(x) for x in normalized[position].tolist()),
            ),
        )
        scipy_bounds = list(zip(to_numpy(lower), to_numpy(upper), strict=True))
        accepted_actions: list[torch.Tensor] = []
        accepted_fragilities: list[float] = []
        accepted_worst: list[torch.Tensor] = []
        accepted_references: list[float] = []
        accepted_residuals: list[dict[str, float]] = []

        def objective(candidate: np.ndarray) -> float:
            """Return fragility or a deterministic penalty for one outer action."""

            point = torch.as_tensor(candidate, dtype=self.dtype, device=self.device)
            try:
                fragility, _, _, _ = self._optimistic_fragility(point, reference, gram)
            except RoBOSFragilitySolverError:
                return 1e12
            return fragility if math.isfinite(fragility) else 1e12

        for position in order[: min(self.config.action_restarts, len(order))]:
            start = to_numpy(raw_actions[position].double())
            try:
                result = minimize(
                    objective,
                    start,
                    method="Powell",
                    bounds=scipy_bounds,
                    options={
                        "xtol": self.config.optimizer_tolerance,
                        "ftol": self.config.optimizer_tolerance,
                        "maxiter": self.config.action_maxiter,
                    },
                )
            except (FloatingPointError, RuntimeError, ValueError):
                continue
            try:
                result_x = np.asarray(result.x, dtype=float)
                result_fun = float(result.fun)
            except (TypeError, ValueError):
                continue
            if (
                not result.success
                or result_x.shape != (self.action_dimension,)
                or not math.isfinite(result_fun)
                or not np.isfinite(result_x).all()
            ):
                continue
            action = torch.as_tensor(result_x, dtype=self.dtype, device=self.device)
            bound_residual = self._action_bound_residual(action)
            if bound_residual > self.config.feasibility_tolerance:
                continue
            try:
                fragility, worst, reference_value, certificate = self._optimistic_fragility(
                    action, reference, gram
                )
            except RoBOSFragilitySolverError:
                continue
            if not math.isfinite(reference_value):
                continue
            certificate = dict(certificate)
            certificate["outer_bounds"] = bound_residual
            accepted_actions.append(action.detach().clone())
            accepted_fragilities.append(fragility)
            accepted_worst.append(worst)
            accepted_references.append(reference_value)
            accepted_residuals.append(certificate)

        if not accepted_actions:
            raise RoBOSFragilitySolverError(
                "Every continuous RoBOS action-optimizer restart failed certification.",
                algorithm=self,
            )
        actions = torch.stack(accepted_actions)
        fragilities = torch.tensor(accepted_fragilities, dtype=self.dtype, device=self.device)
        references = torch.tensor(accepted_references, dtype=self.dtype, device=self.device)
        selected = self._stable_position(fragilities, references, actions, None, minimize=True)
        return (
            actions[selected],
            accepted_fragilities[selected],
            accepted_worst[selected],
            accepted_references[selected],
            accepted_residuals[selected],
        )

    def _ask_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Choose one action using only the public reference distribution.

        Data-backed rounds retain exact candidate actions.  Continuous rounds
        refine deterministic starts and require successful certified optimizer
        output; the true sampling distribution is never read.
        """

        self.bind_environment_metadata(round_info)
        reference = self._validate_reference(round_info["reference_distribution"])
        actions, indices = self._candidate_actions(round_info)
        if self.config.simplex_solver == "source_cobyla" and indices is None:
            raise AlgorithmConfigurationError(
                "source_cobyla requires an explicit finite candidate action table; "
                "the released source does not define continuous action refinement."
            )
        residuals: dict[str, float]
        if self.state.y_train.numel() < self.config.initial_points:
            position = 0 if indices is None else int(torch.argmin(indices).item())
            action = actions[position]
            fragility, worst, reference_value, residuals = (
                math.nan,
                reference,
                math.nan,
                {},
            )
        else:
            self._fit_model()
            if indices is None:
                action, fragility, worst, reference_value, residuals = (
                    self._refine_continuous_action(actions, reference)
                )
            else:
                action, fragility, worst, reference_value, residuals = (
                    self._select_minimum_fragility_action(actions, reference, indices)
                )
        self._last_acquisition_value = None if math.isnan(fragility) else -fragility
        self._record_last_ask(
            actions=action,
            round_info=round_info,
            model_inputs=self._joint_input(action, reference @ self.contexts)[0],
            algorithm_metadata={
                "reference": reference,
                "fragility": fragility,
                "worst_distribution": worst,
                "reference_value": reference_value,
                "solver_residuals": residuals,
            },
        )
        return action

    def _tell_impl(
        self,
        *,
        action: torch.Tensor,
        feedback: Mapping[str, Any],
        round_info: Mapping[str, Any],
    ) -> None:
        """Append realized public feedback and robust decision diagnostics.

        The realized context is stored with the evaluated action and observation.
        Audit-only true sampling probabilities are neither required nor copied.
        """

        del round_info
        assert self.last_ask is not None
        context, _, observation = self._append_joint_feedback(feedback)
        metadata = self.last_ask.algorithm_metadata
        self.state.reference_history.append(
            torch.as_tensor(metadata["reference"], dtype=self.dtype, device=self.device)
        )
        self.state.selected_actions.append(action[0].detach().clone())
        self.state.selected_fragilities.append(float(metadata["fragility"]))
        self.state.reference_values.append(float(metadata["reference_value"]))
        self.state.worst_distributions.append(
            torch.as_tensor(metadata["worst_distribution"], dtype=self.dtype, device=self.device)
        )
        self.state.realized_contexts.append(context.detach().clone())
        self.state.aspiration_satisfied.append(
            bool(float(observation[0].item()) >= self.config.aspiration)
        )

    def _recommendation_impl(self) -> tuple[torch.Tensor | None, float | None]:
        """Return the previously selected action with smallest finite fragility."""

        if not self.state.selected_actions:
            return self._best_observed()
        scores = torch.tensor(self.state.selected_fragilities, dtype=self.dtype, device=self.device)
        finite = torch.isfinite(scores)
        if finite.any():
            indices = torch.nonzero(finite, as_tuple=False).reshape(-1)
            position = int(indices[torch.argmin(scores[indices])].item())
        else:
            return self._best_observed()
        return self.state.selected_actions[position].detach().clone(), float(
            scores[position].item()
        )

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Return solver, fragility, reference-value, and satisfaction diagnostics."""

        return {
            "context_count": int(self.contexts.shape[0]),
            "simplex_solver": self.config.simplex_solver,
            "source_solver": copy.deepcopy(self._source_solver_diagnostics),
            "source_solver_actions": copy.deepcopy(self._source_solver_action_diagnostics),
            "source_prediction_layout": "context_major_batched"
            if self.config.simplex_solver == "source_cobyla"
            else None,
            "source_solver_queued_starts": len(self._source_solver_starts),
            "last_fragility": self.state.selected_fragilities[-1]
            if self.state.selected_fragilities
            else None,
            "last_reference_value": self.state.reference_values[-1]
            if self.state.reference_values
            else None,
            "satisfaction_rate": (
                sum(self.state.aspiration_satisfied) / len(self.state.aspiration_satisfied)
                if self.state.aspiration_satisfied
                else None
            ),
        }

    def _result_metadata_impl(self) -> Mapping[str, Any]:
        """Return detached reference and worst-distribution histories."""

        return {
            "reference_history": [x.detach().clone() for x in self.state.reference_history],
            "worst_distributions": [x.detach().clone() for x in self.state.worst_distributions],
        }

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize context support, aspiration, and mutable progress."""

        return {
            "contexts": self.contexts,
            "context_support_is_dynamic": getattr(self, "_dynamic_context_support", False),
            "gp": self.gp.state_dict(),
            "aspiration": self.config.aspiration,
            "source_solver_state": self.get_source_solver_state(),
            "source_solver_starts": [row.tolist() for row in self._source_solver_starts],
            "state": {name: getattr(self.state, name) for name in self.state.__dataclass_fields__},
        }

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Validate source-critical support, aspiration, state, and GP payloads."""

        del strict
        if not self.contexts.numel() or state.get("context_support_is_dynamic", False):
            self.bind_environment_metadata(
                {
                    "context_support": state.get("contexts"),
                    "context_support_is_dynamic": state.get("context_support_is_dynamic", False),
                }
            )
        if (
            not self._support_matches(state.get("contexts"))
            or state.get("aspiration") != self.config.aspiration
        ):
            raise RoBOSCheckpointError(
                "Stored RoBOS configuration is incompatible.", algorithm=self
            )
        if not isinstance(state.get("state"), Mapping):
            raise RoBOSCheckpointError("RoBOS state is incomplete.", algorithm=self)

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Restore RoBOS histories and reset the joint GP for a fresh fit."""

        del strict
        raw = state["state"]
        try:

            def tensor(value: Any) -> torch.Tensor:
                """Convert one checkpoint value to the algorithm tensor policy."""

                return torch.as_tensor(value, dtype=self.dtype, device=self.device)

            self.state = RoBOSState(
                actions_train=tensor(raw["actions_train"]),
                contexts_train=tensor(raw["contexts_train"]),
                joint_inputs_train=tensor(raw["joint_inputs_train"]),
                y_train=tensor(raw["y_train"]),
                observation_noise_variance=tensor(raw["observation_noise_variance"]),
                reference_history=[tensor(x) for x in raw["reference_history"]],
                selected_actions=[tensor(x) for x in raw["selected_actions"]],
                selected_fragilities=[float(x) for x in raw["selected_fragilities"]],
                reference_values=[float(x) for x in raw["reference_values"]],
                worst_distributions=[tensor(x) for x in raw["worst_distributions"]],
                realized_contexts=[tensor(x) for x in raw["realized_contexts"]],
                aspiration_satisfied=[bool(x) for x in raw["aspiration_satisfied"]],
            )
            if isinstance(state.get("gp"), dict):
                self.gp.load_state_dict(dict(state["gp"]))
            else:
                self.gp.reset()
            if state.get("source_solver_state") is not None:
                self.set_source_solver_state(state["source_solver_state"])
            if self.config.simplex_solver == "source_cobyla":
                self.set_source_solver_starts(state.get("source_solver_starts") or None)
        except Exception as error:
            raise RoBOSCheckpointError(
                f"Invalid RoBOS checkpoint: {error}", algorithm=self
            ) from error

    def _reset_impl(self) -> None:
        """Clear robust histories and reset the joint GP."""

        self.state = self._empty_state()
        self.gp.reset()
        self._source_solver_rng = np.random.RandomState(self.seed % (2**32))
        self._source_solver_diagnostics = None
        self._source_solver_starts = []
        self._source_solver_action_diagnostics = []


__all__ = ["RoBOS", "RoBOSConfig", "RoBOSState", "theoretical_beta"]
