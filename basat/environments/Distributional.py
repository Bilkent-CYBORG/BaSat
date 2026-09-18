"""Distribution-shift environment with finite context support."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol, runtime_checkable

import torch

from basat.models.utils import Bounds, as_tensor

from .environment import (
    Environment,
    EnvironmentConfigurationError,
    EnvironmentData,
    EnvironmentDataError,
    EnvironmentDataset,
    EnvironmentProtocolError,
    NoiseProvider,
    _clone_tree,
    _Evaluation,
)


@runtime_checkable
class DistributionProvider(Protocol):
    """Protocol for round-dependent discrete probability providers."""

    def __call__(
        self,
        *,
        round_index: int,
        generator: torch.Generator,
    ) -> Any:
        """Return a probability vector over ``context_support``."""


@runtime_checkable
class MarginProvider(Protocol):
    """Protocol for round-dependent ambiguity margins."""

    def __call__(
        self,
        *,
        reference_distribution: torch.Tensor,
        round_index: int,
        generator: torch.Generator,
    ) -> Any:
        """Return a finite, non-negative scalar margin."""


class DistributionalEnvironment(Environment):
    """Reveal a reference distribution, then privately sample a context.

    ``context_support`` has shape ``(S, D_context)`` and both probability
    vectors have length ``S``.  The true ``sampling_distribution`` is resolved
    only after an action is selected and is stored solely in audit state.
    Data-backed operation requires a complete action-context grid.

    Examples
    --------
    >>> import torch
    >>> import basat
    >>> support = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    >>> bounds = basat.Bounds(lower=[-1.0], upper=[1.0])
    >>> environment = basat.DistributionalEnvironment(
    ...     bounds=bounds, reference_distribution=[0.5, 0.5],
    ...     context_support=support,
    ...     function=lambda action, context: -((action[0] - context[0]).square()),
    ...     dtype="float64", device="cpu", noise_std=0.0, seed=42,
    ... )
    """

    environment_type = "distributional"
    provided_round_info = frozenset({"reference_distribution", "margin"})
    provided_feedback = frozenset({"observation", "evaluated_action", "context"})
    capabilities = frozenset({"distributional", "finite_context_support"})

    def __init__(
        self,
        *,
        bounds: Bounds,
        reference_distribution: Any | Callable[..., Any],
        sampling_distribution: Any | Callable[..., Any] | None = None,
        context_support: Any | None = None,
        margin: float | Callable[..., Any] | None = None,
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
        """Initialize providers, support, and the complete-grid data contract."""

        self.reference_distribution_provider = reference_distribution
        self.sampling_distribution_provider = sampling_distribution
        self.margin_provider = margin
        self.context_support: torch.Tensor | None = None
        self.current_reference_distribution: torch.Tensor | None = None
        self.current_margin: torch.Tensor | None = None
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
        if context_support is None:
            if self.dataset is None or self.dataset.contexts is None:
                raise EnvironmentConfigurationError(
                    "context_support is required unless it can be inferred from dataset contexts."
                )
            support = self._stable_unique_rows(self.dataset.contexts)
        else:
            support = self._coerce_support(context_support)
        self._validate_tolerance_equivalence(support, "context support")
        self.context_support = support
        if self.dataset is not None:
            if self.dataset.contexts is None:
                raise EnvironmentDataError("A distributional finite dataset requires contexts.")
            self._validate_complete_grid()
        initial = self.history.initial_data
        if initial is not None and initial.observations.shape[0] > 0:
            if initial.contexts is None:
                raise EnvironmentDataError("Distributional previous_data requires contexts.")
            if initial.contexts.shape[1] != self.context_support.shape[1]:
                raise EnvironmentDataError(
                    "Previous-data and support context dimensions do not match."
                )
        self._freeze_source_fingerprint()

    @property
    def context_dimension(self) -> int:
        """Return the dimension of each supported context."""

        assert self.context_support is not None
        return int(self.context_support.shape[1])

    def _coerce_support(self, value: Any) -> torch.Tensor:
        """Convert context support to a unique finite ``S x D_context`` tensor."""

        try:
            support = as_tensor(value, dtype=self.dtype, device=self.device, copy=True)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise EnvironmentDataError("context_support must be numeric.") from exc
        if support.ndim == 1:
            support = support.unsqueeze(-1)
        if support.ndim != 2 or support.shape[0] == 0 or support.shape[1] == 0:
            raise EnvironmentDataError("context_support must have shape (S, D_context) with S > 0.")
        if not bool(torch.isfinite(support).all()):
            raise EnvironmentDataError("context_support contains non-finite values.")
        if self._stable_unique_rows(support).shape[0] != support.shape[0]:
            raise EnvironmentDataError("context_support contains duplicate contexts.")
        return support

    def _stable_unique_rows(self, matrix: torch.Tensor) -> torch.Tensor:
        """Return first-occurrence rows under the environment match tolerance."""

        rows: list[torch.Tensor] = []
        for row in matrix:
            if not any(bool(self._row_match(existing.unsqueeze(0), row)[0]) for existing in rows):
                rows.append(row.detach().clone())
        return torch.stack(rows)

    def _validate_complete_grid(self) -> None:
        """Require every exposed action under every declared context."""

        assert self.dataset is not None
        assert self.dataset.contexts is not None
        assert self.context_support is not None
        # Extra undeclared contexts make the hidden sampling domain ambiguous.
        for context in self.dataset.contexts:
            if not bool(self._row_match(self.context_support, context).any()):
                raise EnvironmentDataError("Dataset contains a context outside context_support.")
        action_indices = self._collapse_action_indices(
            torch.arange(self.dataset.actions.shape[0], device=self.device)
        )
        for index in action_indices.tolist():
            action = self.dataset.actions[index]
            for context in self.context_support:
                if self.find_dataset_indices(action=action, context=context).numel() == 0:
                    raise EnvironmentDataError(
                        "Distributional dataset is not a complete action-context grid."
                    )

    def _resolve_value(self, provider: Any, *, action: torch.Tensor | None = None) -> Any:
        """Resolve a constant or reproducible stochastic provider value."""

        if callable(provider):
            return self._invoke_provider(
                provider,
                primary_name=None,
                primary_value=None,
                action=_clone_tree(action),
                reference_distribution=_clone_tree(self.current_reference_distribution),
                round_index=self.round,
                generator=self.generator,
            )
        return _clone_tree(provider)

    def _validate_probability(self, value: Any, name: str) -> torch.Tensor:
        """Validate and normalize one probability vector over the support."""

        assert self.context_support is not None
        try:
            probability = as_tensor(
                value, dtype=self.dtype, device=self.device, copy=True
            ).squeeze()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise EnvironmentDataError(f"{name} must be numeric.") from exc
        if probability.ndim != 1 or probability.shape[0] != self.context_support.shape[0]:
            raise EnvironmentDataError(
                f"{name} must have shape ({self.context_support.shape[0]},)."
            )
        if not bool(torch.isfinite(probability).all()) or bool((probability < 0).any()):
            raise EnvironmentDataError(f"{name} must be finite and non-negative.")
        total = probability.sum()
        if not bool(torch.isclose(total, torch.ones_like(total), rtol=0.0, atol=1e-6)):
            raise EnvironmentDataError(f"{name} must sum to one within 1e-6.")
        return probability / total

    def _resolve_reference_distribution(self) -> torch.Tensor:
        """Resolve the learner-visible reference probability vector."""

        return self._validate_probability(
            self._resolve_value(self.reference_distribution_provider),
            "reference_distribution",
        )

    def _resolve_sampling_distribution(self, action: torch.Tensor) -> torch.Tensor:
        """Resolve the hidden action-dependent sampling distribution."""

        provider = self.sampling_distribution_provider
        value = (
            self.current_reference_distribution
            if provider is None
            else self._resolve_value(provider, action=action)
        )
        return self._validate_probability(value, "sampling_distribution")

    def _resolve_margin(self) -> torch.Tensor | None:
        """Resolve an optional learner-visible non-negative ambiguity margin."""

        if self.margin_provider is None:
            return None
        value = self._resolve_value(self.margin_provider)
        try:
            result = as_tensor(value, dtype=self.dtype, device=self.device, copy=True).squeeze()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise EnvironmentDataError("margin must be numeric.") from exc
        if result.ndim != 0 or not bool(torch.isfinite(result)) or bool(result < 0):
            raise EnvironmentDataError("margin must be a finite non-negative scalar.")
        return result

    def before_action(self) -> dict[str, Any]:
        """Reveal reference ambiguity information but not the true distribution."""

        if self._pending_round_info is not None:
            return _clone_tree(self._pending_round_info)
        assert self.context_support is not None
        self.current_reference_distribution = self._resolve_reference_distribution()
        self.current_margin = self._resolve_margin()
        info: dict[str, Any] = {
            "context_support": self.context_support.detach().clone(),
            "reference_distribution": self.current_reference_distribution.clone(),
        }
        if self.current_margin is not None:
            info["margin"] = self.current_margin.clone()
        if self.source_mode == "data":
            available = self.available_actions()
            info.update(
                candidate_actions=available.actions,
                candidate_dataset_indices=available.dataset_indices,
            )
        self._pending_round_info = _clone_tree(info)
        return _clone_tree(info)

    def _sample_context(self, sampling_distribution: torch.Tensor) -> tuple[int, torch.Tensor]:
        """Draw one hidden support index with the environment-owned generator."""

        assert self.context_support is not None
        generator_device = getattr(self.generator, "device", torch.device("cpu"))
        probability = sampling_distribution.to(generator_device)
        index = int(
            torch.multinomial(probability, 1, replacement=True, generator=self.generator).item()
        )
        return index, self.context_support[index].detach().clone()

    def _evaluate_one(self, action: torch.Tensor) -> _Evaluation:
        """Privately sample a context and prepare one uncommitted evaluation."""

        if self.current_reference_distribution is None or self._pending_round_info is None:
            raise EnvironmentProtocolError("Call before_action() before a distributional step().")
        if self.source_mode == "data":
            self._assert_action_available(action)
        true_distribution = self._resolve_sampling_distribution(action)
        context_index, context = self._sample_context(true_distribution)
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
            stored, variance, row = self.lookup_dataset_observation(action=action, context=context)
            observation, added_noise = self.make_data_observation(
                stored, action=action, context=context
            )
            if self.add_noise_to_data:
                if self.noise_fn is None:
                    extra_variance = torch.as_tensor(
                        self.noise_std**2, dtype=self.dtype, device=self.device
                    )
                    variance = extra_variance if variance is None else variance + extra_variance
                else:
                    variance = None
            audit = {
                "stored_observation": stored,
                "realized_noise": added_noise,
            }
        audit.update(
            true_sampling_distribution=true_distribution,
            realized_context_index=context_index,
        )
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
        """Clear round-specific public distribution information after evaluation."""

        self.current_reference_distribution = None
        self.current_margin = None

    def _extra_state_dict(self) -> dict[str, Any]:
        """Serialize support, pending round values, and stateful providers."""

        assert self.context_support is not None
        provider_states: dict[str, Any] = {}
        for name, provider in (
            ("reference", self.reference_distribution_provider),
            ("sampling", self.sampling_distribution_provider),
            ("margin", self.margin_provider),
        ):
            state_method = getattr(provider, "state_dict", None)
            if callable(state_method):
                provider_states[name] = _clone_tree(state_method())
        return {
            "context_support": self.context_support.detach().cpu().clone(),
            "current_reference_distribution": _clone_tree(self.current_reference_distribution),
            "current_margin": _clone_tree(self.current_margin),
            "provider_states": provider_states,
        }

    def _load_extra_state_dict(self, state: Any) -> None:
        """Restore distributional state without exposing private audit values."""

        if not isinstance(state, Mapping):
            raise EnvironmentDataError("distributional extra_state must be a mapping.")
        assert self.context_support is not None
        support = self._coerce_support(state.get("context_support"))
        if not torch.equal(support, self.context_support):
            raise EnvironmentConfigurationError(
                "Checkpoint context_support does not match configuration."
            )
        reference = state.get("current_reference_distribution")
        self.current_reference_distribution = (
            None
            if reference is None
            else self._validate_probability(reference, "current_reference_distribution")
        )
        margin = state.get("current_margin")
        if margin is None:
            self.current_margin = None
        else:
            raw_provider = self.margin_provider
            self.margin_provider = margin
            try:
                self.current_margin = self._resolve_margin()
            finally:
                self.margin_provider = raw_provider
        providers = {
            "reference": self.reference_distribution_provider,
            "sampling": self.sampling_distribution_provider,
            "margin": self.margin_provider,
        }
        provider_states = state.get("provider_states", {})
        if not isinstance(provider_states, Mapping):
            raise EnvironmentDataError("provider_states must be a mapping.")
        for name, provider_state in provider_states.items():
            provider = providers.get(name)
            load_method = getattr(provider, "load_state_dict", None)
            if not callable(load_method):
                raise EnvironmentConfigurationError(
                    f"Checkpoint {name} provider state cannot be restored."
                )
            load_method(_clone_tree(provider_state))

    def _reset_extra_state(self) -> None:
        """Clear pending distributional values while retaining configured providers."""

        self.current_reference_distribution = None
        self.current_margin = None


__all__ = [
    "DistributionProvider",
    "DistributionalEnvironment",
    "MarginProvider",
]
