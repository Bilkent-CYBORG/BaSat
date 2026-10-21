"""Ordinary stochastic objective environment."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Any, Literal

import torch

from basat.models.utils import Bounds

from .environment import (
    Environment,
    EnvironmentData,
    EnvironmentDataError,
    EnvironmentDataset,
    InvalidActionError,
    NoiseProvider,
    _Evaluation,
)


class StochasticEnvironment(Environment):
    """Evaluate noisy objectives without context or adversarial perturbation.

    Function-backed observations follow ``y = f(x) + noise``.  Data-backed
    observations reveal an exact hidden row and, by default, add no further
    noise to the stored response.

    Examples
    --------
    >>> import basat
    >>> def objective(x):
    ...     return -(x.square().sum())
    >>> bounds = basat.Bounds(lower=[-1.0, -1.0], upper=[1.0, 1.0])
    >>> environment = basat.StochasticEnvironment(
    ...     bounds=bounds, function=objective, noise_std=0.0,
    ...     dtype="float64", device="cpu", seed=42,
    ... )
    """

    environment_type = "stochastic"
    capabilities = frozenset({"stochastic"})

    def __init__(
        self,
        *,
        bounds: Bounds,
        function: Callable[..., Any] | None = None,
        dataset: EnvironmentDataset | None = None,
        previous_data: EnvironmentData | None = None,
        dtype: torch.dtype | str = torch.float32,
        device: torch.device | str = "cpu",
        noise_std: float = 0.0,
        noise_fn: NoiseProvider | Callable[..., Any] | None = None,
        add_noise_to_data: bool = False,
        allow_repeated_actions: bool = False,
        data_row_selection: Literal["first", "random"] = "first",
        evaluation_dtype: Literal["storage", "proposal"] = "storage",
        seed: int = 1,
    ) -> None:
        """Initialize a function-backed or finite stochastic environment.

        ``evaluation_dtype='storage'`` uses ``dtype`` for both callable input
        and stored history, as usual. Opt-in ``'proposal'`` instead honors an
        algorithm's recorded float32/float64 evaluation precision while still
        storing actions and scalar rewards in ``dtype``. It requires float64
        storage so float32 and float64 coordinates both survive validation.
        It is function-only;
        missing precision metadata falls back to the storage dtype. The policy
        forms part of checkpoint identity and cannot change on resume.
        """

        if evaluation_dtype not in {"storage", "proposal"}:
            raise EnvironmentDataError("evaluation_dtype must be 'storage' or 'proposal'.")
        self._evaluation_dtype = evaluation_dtype
        self._proposal_input_dtype: torch.dtype | None = None

        super().__init__(
            bounds=bounds,
            function=function,
            dataset=dataset,
            previous_data=previous_data,
            dtype=dtype,
            device=device,
            noise_std=noise_std,
            noise_fn=noise_fn,
            add_noise_to_data=add_noise_to_data,
            allow_repeated_actions=allow_repeated_actions,
            data_row_selection=data_row_selection,
            seed=seed,
        )
        if evaluation_dtype == "proposal" and self.dtype != torch.float64:
            raise EnvironmentDataError("Proposal evaluation precision requires dtype='float64' storage.")
        if self.dataset is not None and self.dataset.contexts is not None:
            raise EnvironmentDataError("A stochastic finite dataset must not contain contexts.")
        if self.dataset is not None and evaluation_dtype == "proposal":
            raise EnvironmentDataError("Proposal evaluation precision is only valid for a callable objective.")
        initial = self.history.initial_data
        if initial is not None and initial.contexts is not None:
            raise EnvironmentDataError("Stochastic previous_data must not contain contexts.")
        self._freeze_source_fingerprint()

    @property
    def evaluation_dtype(self) -> Literal["storage", "proposal"]:
        """Immutable callable precision policy; stored history keeps ``dtype``."""
        return self._evaluation_dtype

    def _compute_source_fingerprint(self) -> str:
        fingerprint = super()._compute_source_fingerprint()
        if self.evaluation_dtype == "storage":
            # Preserve the identity of existing default-policy checkpoints.
            return fingerprint
        return hashlib.sha256((fingerprint + "\0evaluation_dtype=proposal").encode()).hexdigest()

    def step(
        self, action: Any, *, proposal_metadata: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Evaluate proposals using their recorded precision when opted in.

        ``proposal_metadata['evaluation_dtype']`` may be ``'float32'`` or
        ``'float64'``. The legacy source-specific metadata key
        ``source_physical_action_dtype`` is also accepted. Metadata never
        changes the default storage policy or the precision of stored records.
        A proposal claiming float32 must already be exactly representable in
        float32: this mechanism changes arithmetic, not requested coordinates.
        """
        proposal_dtype = None
        if self.evaluation_dtype == "proposal":
            if proposal_metadata is not None and not isinstance(proposal_metadata, Mapping):
                raise EnvironmentDataError("proposal_metadata must be a mapping or None.")
            metadata = {} if proposal_metadata is None else proposal_metadata
            name = metadata.get("evaluation_dtype", metadata.get("source_physical_action_dtype"))
            if name is not None:
                if name not in ("float32", "float64"):
                    raise EnvironmentDataError("Proposal evaluation_dtype must be 'float32' or 'float64'.")
                proposal_dtype = torch.float32 if name == "float32" else torch.float64
        previous_dtype = self._proposal_input_dtype
        self._proposal_input_dtype = proposal_dtype
        try:
            return super().step(action)
        finally:
            self._proposal_input_dtype = previous_dtype

    def _evaluate_one(self, action: torch.Tensor) -> _Evaluation:
        """Prepare one stochastic evaluation without committing history."""

        if self.source_mode == "function":
            function_action = action
            if self._proposal_input_dtype is not None:
                function_action = action.to(dtype=self._proposal_input_dtype)
                if not torch.equal(function_action.to(dtype=action.dtype), action):
                    raise InvalidActionError("Proposal precision metadata would change the requested coordinates.")
            latent = self.evaluate_function(function_action)
            realized_noise = self.sample_noise(action=action)
            observation = latent + realized_noise
            variance = (
                torch.as_tensor(
                    self.noise_std**2,
                    dtype=self.dtype,
                    device=self.device,
                )
                if self.noise_fn is None
                else None
            )
            return _Evaluation(
                action=action,
                evaluated_action=action,
                observation=observation,
                observation_noise_variance=variance,
                metadata={"source_mode": "function", **(
                    {"evaluation_dtype": str(function_action.dtype).removeprefix("torch.")}
                    if self.evaluation_dtype == "proposal" else {})},
                audit={
                    "latent_observation": latent,
                    "realized_noise": realized_noise,
                    **({"function_input_dtype": str(function_action.dtype)}
                       if self.evaluation_dtype == "proposal" else {}),
                },
            )

        self._assert_action_available(action)
        stored, stored_variance, row = self.lookup_dataset_observation(action=action)
        observation, added_noise = self.make_data_observation(stored, action=action)
        variance = stored_variance
        if self.add_noise_to_data:
            if self.noise_fn is None:
                added_variance = torch.as_tensor(
                    self.noise_std**2,
                    dtype=self.dtype,
                    device=self.device,
                )
                variance = added_variance if variance is None else variance + added_variance
            else:
                variance = None
        return _Evaluation(
            action=action,
            evaluated_action=action,
            observation=observation,
            observation_noise_variance=variance,
            dataset_index=row,
            metadata={"source_mode": "data"},
            audit={"stored_observation": stored, "realized_noise": added_noise},
        )


__all__ = ["StochasticEnvironment"]
