"""Context-revealing Bayesian optimization environment."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol, runtime_checkable

import torch

from basat.models.utils import Bounds

from .environment import (
    Environment,
    EnvironmentConfigurationError,
    EnvironmentData,
    EnvironmentDataError,
    EnvironmentDataset,
    EnvironmentProtocolError,
    NoAvailableActionError,
    NoiseProvider,
    _clone_tree,
    _Evaluation,
)


@runtime_checkable
class ContextProvider(Protocol):
    """Protocol for reproducible round-context providers."""

    def __call__(
        self,
        *,
        round_index: int,
        generator: torch.Generator,
    ) -> Any:
        """Return one context vector before the learner acts."""


class ContextualEnvironment(Environment):
    """Reveal a context before selecting and evaluating an action.

    In data-backed mode, omitting ``context_provider`` samples uniformly from
    unique contexts that still have at least one available action.  Only action
    coordinates available at the revealed context are exposed.

    Examples
    --------
    >>> import torch
    >>> import basat
    >>> def context_provider(*, round_index, generator):
    ...     return torch.tensor([round_index % 2], dtype=torch.float64)
    >>> def objective(action, context):
    ...     return -((action[0] - context[0]).square())
    >>> bounds = basat.Bounds(lower=[-1.0], upper=[1.0])
    >>> environment = basat.ContextualEnvironment(
    ...     bounds=bounds, context_provider=context_provider, function=objective,
    ...     dtype="float64", device="cpu", noise_std=0.0, seed=42,
    ... )
    """

    environment_type = "contextual"
    provided_round_info = frozenset({"context"})
    provided_feedback = frozenset({"observation", "evaluated_action", "context"})
    capabilities = frozenset({"contextual", "revealed_context"})

    def __init__(
        self,
        *,
        bounds: Bounds,
        context_provider: ContextProvider | Callable[..., Any] | None = None,
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
        """Initialize a contextual source and freeze any known context dimension."""

        if context_provider is not None and not callable(context_provider):
            raise EnvironmentConfigurationError("context_provider must be callable or None.")
        if function is not None and context_provider is None:
            raise EnvironmentConfigurationError(
                "A function-backed contextual environment requires context_provider."
            )
        self.context_provider = context_provider
        self.current_context: torch.Tensor | None = None
        self._frozen_context_dimension: int | None = None
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
        if self.dataset is not None:
            if self.dataset.contexts is None:
                raise EnvironmentDataError("A contextual finite dataset requires contexts.")
            self._frozen_context_dimension = int(self.dataset.contexts.shape[1])
        initial = self.history.initial_data
        if initial is not None and initial.observations.shape[0] > 0:
            if initial.contexts is None:
                raise EnvironmentDataError(
                    "Contextual previous_data requires one context per observation."
                )
            dimension = int(initial.contexts.shape[1])
            if (
                self._frozen_context_dimension is not None
                and dimension != self._frozen_context_dimension
            ):
                raise EnvironmentDataError(
                    "Previous and finite-data context dimensions do not match."
                )
            self._frozen_context_dimension = dimension
        self._freeze_source_fingerprint()

    @property
    def context_dimension(self) -> int | None:
        """Return the frozen context dimension, if it has been established."""

        return self._frozen_context_dimension

    def _freeze_context(self, value: Any) -> torch.Tensor:
        """Validate one context and freeze its dimension on first use."""

        context = self._validate_context(value)
        if self._frozen_context_dimension is None:
            self._frozen_context_dimension = int(context.shape[0])
        elif context.shape[0] != self._frozen_context_dimension:
            raise EnvironmentDataError(
                f"context must have shape ({self._frozen_context_dimension},)."
            )
        return context

    def _unique_dataset_contexts(self) -> list[torch.Tensor]:
        """Return stable unique finite contexts."""

        assert self.dataset is not None and self.dataset.contexts is not None
        contexts: list[torch.Tensor] = []
        for context in self.dataset.contexts:
            if not any(
                bool(self._row_match(existing.unsqueeze(0), context)[0]) for existing in contexts
            ):
                contexts.append(context.detach().clone())
        return contexts

    def _resolve_context(self) -> torch.Tensor:
        """Obtain a provider context or sample an available finite context."""

        if self.context_provider is not None:
            raw = self._invoke_provider(
                self.context_provider,
                primary_name=None,
                primary_value=None,
                round_index=self.round,
                generator=self.generator,
            )
            return self._freeze_context(raw)
        if self.dataset is None:
            raise EnvironmentProtocolError("No context source is configured.")
        viable: list[torch.Tensor] = []
        for context in self._unique_dataset_contexts():
            try:
                self.available_actions(context=context)
            except NoAvailableActionError:
                continue
            viable.append(context)
        if not viable:
            raise NoAvailableActionError("No finite context has an available action.")
        generator_device = getattr(self.generator, "device", torch.device("cpu"))
        selected = int(
            torch.randint(
                len(viable),
                (),
                generator=self.generator,
                device=generator_device,
            ).item()
        )
        return self._freeze_context(viable[selected])

    def before_action(self) -> dict[str, Any]:
        """Reveal the current context and context-compatible finite actions."""

        if self._pending_round_info is not None:
            return _clone_tree(self._pending_round_info)
        if self.current_context is None:
            self.current_context = self._resolve_context()
        info: dict[str, Any] = {"context": self.current_context.detach().clone()}
        if self.source_mode == "data":
            available = self.available_actions(context=self.current_context)
            info.update(
                candidate_actions=available.actions,
                candidate_dataset_indices=available.dataset_indices,
            )
        self._pending_round_info = _clone_tree(info)
        return _clone_tree(info)

    def _evaluate_one(self, action: torch.Tensor) -> _Evaluation:
        """Evaluate one action at the already revealed context."""

        if self.current_context is None or self._pending_round_info is None:
            raise EnvironmentProtocolError(
                "Call before_action() to reveal the context before step()."
            )
        context = self.current_context.detach().clone()
        if self.source_mode == "function":
            latent = self.evaluate_function(action, context)
            realized_noise = self.sample_noise(action=action, context=context)
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
            }
        else:
            self._assert_action_available(action, context=context)
            stored, variance, row = self.lookup_dataset_observation(action=action, context=context)
            observation, added_noise = self.make_data_observation(
                stored, action=action, context=context
            )
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
            }
        return _Evaluation(
            action=action,
            evaluated_action=action,
            observation=observation,
            context=context,
            observation_noise_variance=variance,
            dataset_index=row,
            metadata={"source_mode": self.source_mode},
            audit=audit,
        )

    def _after_step(self) -> None:
        """Clear the revealed context after its single authorized evaluation."""

        self.current_context = None

    def _extra_state_dict(self) -> dict[str, Any]:
        """Serialize the pending context, frozen dimension, and provider state."""

        provider_state = None
        state_method = getattr(self.context_provider, "state_dict", None)
        if callable(state_method):
            provider_state = _clone_tree(state_method())
        return {
            "current_context": _clone_tree(self.current_context),
            "frozen_context_dimension": self._frozen_context_dimension,
            "provider_state": provider_state,
        }

    def _load_extra_state_dict(self, state: Any) -> None:
        """Restore contextual state after dimension and provider checks."""

        if not isinstance(state, Mapping):
            raise EnvironmentDataError("contextual extra_state must be a mapping.")
        dimension = state.get("frozen_context_dimension")
        if dimension is not None and (
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
        ):
            raise EnvironmentDataError("frozen_context_dimension is invalid.")
        if (
            self._frozen_context_dimension is not None
            and dimension is not None
            and self._frozen_context_dimension != dimension
        ):
            raise EnvironmentConfigurationError(
                "Checkpoint context dimension does not match configuration."
            )
        self._frozen_context_dimension = dimension or self._frozen_context_dimension
        current = state.get("current_context")
        self.current_context = None if current is None else self._freeze_context(current)
        provider_state = state.get("provider_state")
        if provider_state is not None:
            load_method = getattr(self.context_provider, "load_state_dict", None)
            if not callable(load_method):
                raise EnvironmentConfigurationError(
                    "Checkpoint contains provider state but context_provider cannot restore it."
                )
            load_method(_clone_tree(provider_state))

    def _reset_extra_state(self) -> None:
        """Clear transient contextual state while preserving source configuration."""

        self.current_context = None


__all__ = ["ContextProvider", "ContextualEnvironment"]
