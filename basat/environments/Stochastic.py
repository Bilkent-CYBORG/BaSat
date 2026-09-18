"""Ordinary stochastic objective environment."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import torch

from basat.models.utils import Bounds

from .environment import (
    Environment,
    EnvironmentData,
    EnvironmentDataError,
    EnvironmentDataset,
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
        seed: int = 1,
    ) -> None:
        """Initialize a function-backed or finite stochastic environment."""

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
        if self.dataset is not None and self.dataset.contexts is not None:
            raise EnvironmentDataError("A stochastic finite dataset must not contain contexts.")
        initial = self.history.initial_data
        if initial is not None and initial.contexts is not None:
            raise EnvironmentDataError("Stochastic previous_data must not contain contexts.")
        self._freeze_source_fingerprint()

    def _evaluate_one(self, action: torch.Tensor) -> _Evaluation:
        """Prepare one stochastic evaluation without committing history."""

        if self.source_mode == "function":
            latent = self.evaluate_function(action)
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
                metadata={"source_mode": "function"},
                audit={
                    "latent_observation": latent,
                    "realized_noise": realized_noise,
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
