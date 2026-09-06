"""Adversarial action-perturbation environment."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol, runtime_checkable

import torch

from bayopy.models.utils import Bounds

from .environment import (
    Environment,
    EnvironmentConfigurationError,
    EnvironmentData,
    EnvironmentDataError,
    EnvironmentDataset,
    InvalidActionError,
    NoiseProvider,
    _Evaluation,
)


@runtime_checkable
class AttackProvider(Protocol):
    """Protocol for reproducible adversarial perturbation providers."""

    def __call__(
        self,
        *,
        action: torch.Tensor,
        round_index: int,
        generator: torch.Generator,
    ) -> Any:
        """Return a perturbation vector with shape ``(D_env,)``."""


class AdversarialEnvironment(Environment):
    """Apply an attack before evaluating the requested action.

    The requested action remains in ``action`` while ``evaluated_action`` is
    ``action + perturbation``.  No clipping or interpolation is performed.  In
    data-backed mode, both the nominal action and the attacked coordinate must
    satisfy their respective exact finite-table contracts.

    Examples
    --------
    >>> import torch
    >>> import bayopy
    >>> bounds = bayopy.Bounds(lower=[-1.0], upper=[1.0])
    >>> def attack(*, action, round_index, generator):
    ...     return torch.zeros_like(action)
    >>> environment = bayopy.AdversarialEnvironment(
    ...     bounds=bounds, attack=attack,
    ...     function=lambda x: -(x.square().sum()),
    ...     dtype="float64", device="cpu", noise_std=0.0, seed=42,
    ... )
    """

    environment_type = "adversarial"
    provided_feedback = frozenset({"observation", "evaluated_action", "perturbation"})
    capabilities = frozenset({"adversarial", "perturbed_actions"})

    def __init__(
        self,
        *,
        bounds: Bounds,
        attack: AttackProvider | Callable[..., Any],
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
        """Initialize the source and its required reproducible attack provider."""

        if not callable(attack):
            raise EnvironmentConfigurationError("attack must be callable.")
        self.attack = attack
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
            raise EnvironmentDataError("An adversarial finite dataset must not contain contexts.")
        initial = self.history.initial_data
        if initial is not None and initial.contexts is not None:
            raise EnvironmentDataError("Adversarial previous_data must not contain contexts.")
        self._freeze_source_fingerprint()

    def _attack_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve and validate one perturbation and evaluated action."""

        raw = self._invoke_provider(
            self.attack,
            primary_name="action",
            primary_value=action.detach().clone(),
            round_index=self.round,
            generator=self.generator,
        )
        try:
            perturbation = (
                torch.as_tensor(raw, dtype=self.dtype, device=self.device).detach().clone()
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise InvalidActionError("attack must return a numeric vector.") from exc
        if perturbation.shape != action.shape:
            raise InvalidActionError(f"attack must return shape ({self.action_dimension},).")
        if not bool(torch.isfinite(perturbation).all()):
            raise InvalidActionError("attack returned a non-finite perturbation.")
        evaluated_action = self.validate_action(action + perturbation)
        return perturbation, evaluated_action

    def _evaluate_one(self, action: torch.Tensor) -> _Evaluation:
        """Prepare an attacked evaluation without committing history."""

        if self.source_mode == "data":
            self._assert_action_available(action)
        perturbation, evaluated_action = self._attack_action(action)

        if self.source_mode == "function":
            latent = self.evaluate_function(evaluated_action)
            realized_noise = self.sample_noise(action=evaluated_action)
            observation = latent + realized_noise
            variance = (
                torch.as_tensor(self.noise_std**2, dtype=self.dtype, device=self.device)
                if self.noise_fn is None
                else None
            )
            row = None
            audit: dict[str, Any] = {
                "latent_observation": latent,
                "realized_noise": realized_noise,
                "attack_perturbation": perturbation,
            }
        else:
            stored, variance, row = self.lookup_dataset_observation(action=evaluated_action)
            observation, added_noise = self.make_data_observation(stored, action=evaluated_action)
            if self.add_noise_to_data:
                if self.noise_fn is None:
                    added_variance = torch.as_tensor(
                        self.noise_std**2, dtype=self.dtype, device=self.device
                    )
                    variance = added_variance if variance is None else variance + added_variance
                else:
                    variance = None
            audit = {
                "stored_observation": stored,
                "realized_noise": added_noise,
                "attack_perturbation": perturbation,
            }
        return _Evaluation(
            action=action,
            evaluated_action=evaluated_action,
            observation=observation,
            feedback_extra={"perturbation": perturbation},
            observation_noise_variance=variance,
            dataset_index=row,
            metadata={"source_mode": self.source_mode},
            audit=audit,
        )


__all__ = ["AdversarialEnvironment", "AttackProvider"]
