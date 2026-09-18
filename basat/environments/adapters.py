"""Explicit, learner-visible adapters for neutral cross-environment execution.

Adapters in this module enrich only public round information and feedback.  The
wrapped environment remains the sole owner of objective evaluation, finite
candidate lookup, noise, and audit state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Literal

import torch

from .environment import (
    MATCH_ATOL,
    Environment,
    EnvironmentConfigurationError,
    EnvironmentData,
    EnvironmentHistory,
    EnvironmentRecord,
    _clone_tree,
)


class StochasticContextAdapter:
    """Expose a non-distributional objective through a constant public context.

    ``mode="constant_context"`` supplies ``c0`` before selection for CGP-UCB.
    ``mode="distribution_mean_context"`` supplies the public reference mean
    before selection when CGP-UCB is paired with a distributional source.
    ``mode="singleton_distribution"`` supplies a point-mass reference
    distribution and zero ambiguity margin for DRBO/RoBOS.  In both cases the
    source remains the sole owner of evaluation. Provided current contexts,
    references, margins and evaluated actions take precedence over defaults.
    """

    package_version = "0.1.0"
    compatibility_status = "adapted"

    def __init__(
        self,
        source_environment: Any,
        *,
        constant_context: Any,
        mode: Literal["constant_context", "distribution_mean_context", "singleton_distribution"],
        context_support: Any | None = None,
    ) -> None:
        """Initialize a detached public view over a source environment.

        Parameters
        ----------
        source_environment:
            Environment that continues to own every evaluation.
        constant_context:
            Finite non-empty context vector exposed by the adapter.
        mode:
            Constant contextual or point-mass-distribution specialization.
        context_support:
            Optional finite support already configured on a distributional
            algorithm.  The point mass is placed on ``constant_context``.
        """
        source_type = getattr(source_environment, "environment_type", None)
        if not isinstance(source_type, str) or not source_type:
            raise EnvironmentConfigurationError(
                "The adapted source must declare a non-empty environment_type."
            )
        if mode not in {
            "constant_context",
            "distribution_mean_context",
            "singleton_distribution",
        }:
            raise EnvironmentConfigurationError("Unknown stochastic adapter mode.")
        context = (
            torch.as_tensor(
                constant_context,
                dtype=source_environment.dtype,
                device=source_environment.device,
            )
            .detach()
            .clone()
        )
        if context.ndim == 0:
            context = context.reshape(1)
        if context.ndim != 1 or context.numel() < 1 or not bool(torch.isfinite(context).all()):
            raise EnvironmentConfigurationError(
                "constant_context must be a finite one-dimensional value."
            )

        self.source_environment = source_environment
        self.environment_type = source_type
        self.constant_context = context
        self.mode = mode
        self.adapter_name = mode
        support = context.unsqueeze(0)
        if context_support is not None:
            support = (
                torch.as_tensor(
                    context_support,
                    dtype=source_environment.dtype,
                    device=source_environment.device,
                )
                .detach()
                .clone()
            )
            if (
                support.ndim != 2
                or support.shape[0] < 1
                or support.shape[1] != context.numel()
                or not bool(torch.isfinite(support).all())
            ):
                raise EnvironmentConfigurationError(
                    "context_support must be a finite non-empty (K, D_context) tensor."
                )
            matches = torch.isclose(
                support,
                context.unsqueeze(0),
                rtol=0.0,
                atol=MATCH_ATOL,
            ).all(dim=-1)
        self._context_support = support
        self.context_support_is_dynamic = (
            mode == "singleton_distribution"
            and "context" in getattr(source_environment, "provided_round_info", ())
            and getattr(source_environment, "context_support", None) is None
        )
        matches = torch.isclose(
            support,
            context.unsqueeze(0),
            rtol=0.0,
            atol=MATCH_ATOL,
        ).all(dim=-1)
        self._point_mass_index = (
            int(torch.nonzero(matches, as_tuple=False)[0].item()) if bool(matches.any()) else 0
        )
        if mode == "singleton_distribution":
            has_context = "context" in getattr(source_environment, "provided_round_info", ())
            has_support = getattr(source_environment, "context_support", None) is not None
            self.adapter_assumption_message = (
                "Missing distributional inputs use internal defaults: "
                + (
                    "context_support is the currently supplied context as a singleton; "
                    if has_context and not has_support
                    else "context_support is the zero context as a singleton when absent; "
                )
                + "reference_distribution is a point mass on the supplied context, or uniform "
                "over supplied support when no current context is provided; margin=0. "
                "Environment-provided values override each default. Actual contexts and "
                "evaluated actions remain attached to their observed rewards."
            )
        elif mode == "distribution_mean_context":
            self.adapter_assumption_message = (
                "Missing pre-action context: use the public reference-distribution mean "
                "as the contextual decision proxy. The realized distributional context is "
                "unavailable before selection; this proxy does not recover native contextual guarantees."
            )
        else:
            self.adapter_assumption_message = (
                f"Missing pre-action context: internal context={context.tolist()}. "
                "Any environment-provided current context takes precedence."
            )
        self._pending_round_info: dict[str, Any] | None = None
        self._sync_history()

        source_round_info = frozenset(
            getattr(source_environment, "provided_round_info", frozenset())
        )
        source_feedback = frozenset(getattr(source_environment, "provided_feedback", frozenset()))
        source_capabilities = frozenset(getattr(source_environment, "capabilities", frozenset()))
        if mode in {"constant_context", "distribution_mean_context"}:
            self.provided_round_info = source_round_info | {"context"}
            self.capabilities = source_capabilities | {"contextual", "revealed_context"}
        else:
            self.provided_round_info = source_round_info | {
                "reference_distribution",
                "margin",
                "context_support",
            }
            self.capabilities = source_capabilities | {
                "distributional",
                "finite_context_support",
            }
        self.provided_feedback = source_feedback | {"context"}

    def __getattr__(self, name: str) -> Any:
        """Delegate source configuration and audit properties unchanged."""

        source = self.__dict__.get("source_environment")
        if source is None:
            raise AttributeError(name)
        return getattr(source, name)

    @property
    def context_support(self) -> torch.Tensor:
        """Return the configured support carrying the explicit point mass."""

        return self._context_support.detach().clone()

    def _adapt_data(self, data: EnvironmentData) -> EnvironmentData:
        """Copy learner-visible data and attach the constant context rows."""

        rows = int(data.actions.shape[0])
        contexts = self.constant_context.unsqueeze(0).expand(rows, -1).clone()
        if data.contexts is not None and self.mode != "distribution_mean_context":
            contexts = data.contexts.detach().clone()
        return EnvironmentData(
            actions=data.actions.detach().clone(),
            observations=data.observations.detach().clone(),
            evaluated_actions=(
                None if data.evaluated_actions is None else data.evaluated_actions.detach().clone()
            ),
            contexts=contexts,
            observation_noise_variance=(
                None
                if data.observation_noise_variance is None
                else data.observation_noise_variance.detach().clone()
            ),
            metadata=_clone_tree(data.metadata),
        )

    def _visible_context(self, source_info: Mapping[str, Any]) -> torch.Tensor:
        """Return the constant context or the public reference-distribution mean."""

        if source_info.get("context") is not None:
            return (
                torch.as_tensor(source_info["context"], dtype=self.dtype, device=self.device)
                .detach()
                .clone()
            )
        if self.mode != "distribution_mean_context":
            return self.constant_context.detach().clone()
        raw_reference = source_info.get("reference_distribution")
        try:
            reference = torch.as_tensor(
                raw_reference, dtype=self.dtype, device=self.device
            ).reshape(-1)
        except (TypeError, ValueError, RuntimeError) as error:
            raise EnvironmentConfigurationError(
                "The distribution-mean adapter requires a numeric reference distribution."
            ) from error
        if (
            reference.shape[0] != self._context_support.shape[0]
            or not bool(torch.isfinite(reference).all())
            or bool((reference < 0).any())
            or not bool(
                torch.isclose(
                    reference.sum(),
                    torch.ones((), dtype=self.dtype, device=self.device),
                    rtol=0.0,
                    atol=1e-6,
                )
            )
        ):
            raise EnvironmentConfigurationError(
                "The distribution-mean adapter requires an aligned probability simplex."
            )
        return reference @ self._context_support

    def _round_info(self, source_info: Mapping[str, Any]) -> dict[str, Any]:
        """Add only the neutral fields selected by the adapter mode."""

        info = dict(_clone_tree(source_info))
        if self.mode in {"constant_context", "distribution_mean_context"}:
            info["context"] = self._visible_context(source_info)
            if source_info.get("context") is None:
                info["context_is_default"] = True
        else:
            current = self._visible_context(source_info)
            support = source_info.get("context_support")
            if support is None and self.context_support_is_dynamic:
                support = current.reshape(1, -1)
            if support is not None:
                self._context_support = (
                    torch.as_tensor(support, dtype=self.dtype, device=self.device).detach().clone()
                )
            reference = torch.full(
                (self._context_support.shape[0],),
                1.0 / self._context_support.shape[0],
                dtype=self.dtype,
                device=self.device,
            )
            if source_info.get("context") is not None:
                matches = torch.isclose(
                    self._context_support, current, rtol=0.0, atol=MATCH_ATOL
                ).all(dim=-1)
                if int(matches.sum()) == 1:
                    reference.zero_()
                    reference[matches] = 1.0
            info.setdefault("context", current)
            info.setdefault("context_support", self._context_support.detach().clone())
            info.setdefault("reference_distribution", reference)
            info.setdefault("margin", 0.0)
            info["context_support_is_dynamic"] = self.context_support_is_dynamic
        return info

    def _adapt_record(
        self,
        record: EnvironmentRecord,
        *,
        round_info: Mapping[str, Any] | None = None,
    ) -> EnvironmentRecord:
        """Copy one revealed source record into the adapted public history."""

        visible_round_info = (
            self._round_info(record.round_info) if round_info is None else round_info
        )
        visible_context = (
            torch.as_tensor(
                record.context
                if record.context is not None and self.mode != "distribution_mean_context"
                else visible_round_info.get("context", self.constant_context),
                dtype=self.dtype,
                device=self.device,
            )
            .detach()
            .clone()
        )
        return replace(
            record,
            round_info=_clone_tree(visible_round_info),
            context=visible_context,
        )

    def _sync_history(self) -> None:
        """Rebuild adapted history from already revealed source records."""

        source_history = self.source_environment.history
        initial = source_history.initial_data
        self.history = EnvironmentHistory(
            initial_data=None if initial is None else self._adapt_data(initial),
            records=[self._adapt_record(record) for record in source_history.records],
        )

    @property
    def records(self) -> tuple[EnvironmentRecord, ...]:
        """Return the adapted learner-visible records."""

        return tuple(self.history.records)

    def before_action(self) -> dict[str, Any]:
        """Return source candidates plus the explicit neutral adapter fields."""

        if self._pending_round_info is None:
            self._pending_round_info = self._round_info(self.source_environment.before_action())
        return _clone_tree(self._pending_round_info)

    def step(self, action: Any) -> dict[str, Any]:
        """Evaluate in the source and augment only learner-visible feedback.

        Parameters
        ----------
        action:
            One source action or synchronous source-action batch.
        """

        round_info = self.before_action()
        record_start = len(self.source_environment.history.records)
        feedback = dict(self.source_environment.step(action))
        new_records = self.source_environment.history.records[record_start:]
        self.history.records.extend(
            self._adapt_record(record, round_info=round_info) for record in new_records
        )
        rows = len(new_records)
        visible_context = (
            torch.as_tensor(
                round_info.get("context", self.constant_context),
                dtype=self.dtype,
                device=self.device,
            )
            .detach()
            .clone()
        )
        if feedback.get("context") is None or self.mode == "distribution_mean_context":
            feedback["context"] = (
                visible_context
                if rows == 1 and torch.as_tensor(action).ndim == 1
                else visible_context.unsqueeze(0).expand(rows, -1).clone()
            )
        self._pending_round_info = None
        return _clone_tree(feedback)

    def get_history_data(self) -> EnvironmentData:
        """Return source-visible observations paired with the constant context."""

        return self._adapt_data(self.source_environment.get_history_data())

    def available_actions(self, *, context: Any | None = None) -> Any:
        """Preserve the source finite candidate set without inspecting responses.

        Parameters
        ----------
        context:
            Optional context, which must equal the configured constant context.
        """

        if context is not None:
            candidate = torch.as_tensor(context, dtype=self.dtype, device=self.device).reshape(-1)
            expected = self._visible_context(self.before_action())
            if candidate.shape != expected.shape or not bool(
                torch.isclose(
                    candidate,
                    expected,
                    rtol=0.0,
                    atol=MATCH_ATOL,
                ).all()
            ):
                raise EnvironmentConfigurationError(
                    "The adapter exposes only its configured constant context."
                )
        return self.source_environment.available_actions()

    def validate_action(self, action: Any) -> torch.Tensor:
        """Validate an action using the unmodified source bounds.

        Parameters
        ----------
        action:
            Candidate source action to validate.
        """

        return self.source_environment.validate_action(action)

    def last_record(self) -> dict[str, Any]:
        """Return the source environment's audit-only last-record data."""

        return self.source_environment.last_record()

    def state_dict(self) -> dict[str, Any]:
        """Serialize the adapter configuration and complete source state."""

        return {
            "schema_version": 1,
            "package_version": self.package_version,
            "class_identity": f"{type(self).__module__}.{type(self).__qualname__}",
            "environment_type": self.environment_type,
            "adapter_name": self.adapter_name,
            "round": self.round,
            "constant_context": self.constant_context.detach().cpu().clone(),
            "context_support": self._context_support.detach().cpu().clone(),
            "pending_round_info": _clone_tree(self._pending_round_info),
            "source_state": self.source_environment.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any], *, strict: bool = True) -> None:
        """Restore adapter and source state after strict identity checks.

        Parameters
        ----------
        state:
            Mapping returned by :meth:`state_dict`.
        strict:
            Reserved forward-compatibility flag; identity remains strict.
        """

        del strict
        expected = {
            "schema_version": 1,
            "package_version": self.package_version,
            "class_identity": f"{type(self).__module__}.{type(self).__qualname__}",
            "environment_type": self.environment_type,
            "adapter_name": self.adapter_name,
        }
        if not isinstance(state, Mapping) or any(state.get(k) != v for k, v in expected.items()):
            raise EnvironmentConfigurationError(
                "Checkpoint adapter identity does not match this configuration."
            )
        context = torch.as_tensor(
            state.get("constant_context"), dtype=self.dtype, device=self.device
        )
        if context.shape != self.constant_context.shape or not torch.equal(
            context, self.constant_context
        ):
            raise EnvironmentConfigurationError(
                "Checkpoint constant context does not match this adapter."
            )
        support = torch.as_tensor(
            state.get("context_support"), dtype=self.dtype, device=self.device
        )
        if self.context_support_is_dynamic:
            self._context_support = support.detach().clone()
        elif support.shape != self._context_support.shape or not torch.equal(
            support, self._context_support
        ):
            raise EnvironmentConfigurationError(
                "Checkpoint context support does not match this adapter."
            )
        source_state = state.get("source_state")
        if not isinstance(source_state, Mapping):
            raise EnvironmentConfigurationError("Checkpoint source_state must be a mapping.")
        self.source_environment.load_state_dict(source_state)
        pending = state.get("pending_round_info")
        if pending is not None and not isinstance(pending, Mapping):
            raise EnvironmentConfigurationError(
                "Checkpoint pending_round_info must be a mapping or None."
            )
        self._pending_round_info = None if pending is None else _clone_tree(pending)
        self._sync_history()

    def reset(self, seed: int | None = None) -> None:
        """Reset the source and rebuild the adapted view of previous data.

        Parameters
        ----------
        seed:
            Optional replacement source-environment seed.
        """

        self.source_environment.reset(seed=seed)
        self._pending_round_info = None
        self._sync_history()


class UnperturbedAdapter:
    """Explicit zero-external-attack view for AdveRS on other source families.

    The source still owns evaluation, candidates, context sampling and state. No
    new attack is applied. The robust learner can retain its configured uncertainty
    neighborhood, but source guarantees do not transfer to this experiment.
    """

    adapter_name = "zero_external_attack"
    adapter_assumption_message = (
        "Missing external-attack metadata: evaluated_action defaults to the nominal "
        "action and perturbation defaults to evaluated_action - nominal_action (zero "
        "when no attack is reported). Environment-provided values take precedence. "
        "The source is evaluated unchanged. "
        "source context/distribution information is ignored by this learner. "
        "This is a cross-environment extension, not an adversarial source guarantee."
    )

    def __init__(self, source: Any) -> None:
        self.source_environment = source
        self.capabilities = source.capabilities | {"adversarial", "perturbed_actions"}
        self.provided_feedback = source.provided_feedback | {"evaluated_action", "perturbation"}

    def __getattr__(self, name: str) -> Any:
        source = self.__dict__.get("source_environment")
        if source is None:
            raise AttributeError(name)
        return getattr(source, name)

    def step(self, action: Any) -> dict[str, Any]:
        feedback = dict(self.source_environment.step(action))
        value = torch.as_tensor(action, dtype=self.dtype, device=self.device)
        if feedback.get("evaluated_action") is None:
            feedback["evaluated_action"] = value.detach().clone()
        if feedback.get("perturbation") is None:
            feedback["perturbation"] = (
                torch.as_tensor(feedback["evaluated_action"], dtype=self.dtype, device=self.device)
                - value
            )
        return feedback


def adapt_environment_for_algorithm(
    environment: Any,
    algorithm: Any,
    *,
    constant_context: Any | None = None,
) -> Any:
    """Return an explicit constant-context adapter for supported pairings.

    Parameters
    ----------
    environment:
        Candidate environment supplied to :class:`basat.experiment.Experiment`.
    algorithm:
        Selected algorithm whose declared contract determines adaptation.
    constant_context:
        Retained compatibility argument. Algorithm-owned internal defaults are
        used whenever the environment omits context; this argument is ignored.

    Returns
    -------
    Any
        Original environment or an explicit :class:`StochasticContextAdapter`.
    """

    if isinstance(environment, (StochasticContextAdapter, UnperturbedAdapter)):
        return environment
    environment_type = getattr(environment, "environment_type", None)
    from basat.algorithms.Adversarial.AdveRS import AdveRS
    from basat.algorithms.Contextual.CGP_UCB import CGP_UCB
    from basat.algorithms.Distributional.DRBO import DRBO
    from basat.algorithms.Distributional.RoBOS import RoBOS

    provided = frozenset(getattr(environment, "provided_round_info", ()))
    if isinstance(algorithm, AdveRS) and environment_type != "adversarial":
        return UnperturbedAdapter(environment)
    if isinstance(algorithm, CGP_UCB) and (
        "context" not in provided
        or not algorithm.required_capabilities.issubset(getattr(environment, "capabilities", ()))
    ):
        support = getattr(environment, "context_support", None)
        if support is not None and "reference_distribution" in provided:
            support = torch.as_tensor(support, dtype=environment.dtype, device=environment.device)
            adapted = StochasticContextAdapter(
                environment,
                constant_context=torch.zeros(
                    support.shape[1], dtype=environment.dtype, device=environment.device
                ),
                mode="constant_context",
                context_support=support,
            )
            adapted.adapter_assumption_message += (
                " The distributional realization is unknown before selection; its actual "
                "context is retained for GP training after feedback."
            )
            return adapted
        return StochasticContextAdapter(
            environment,
            constant_context=algorithm._default_context(),
            mode="constant_context",
        )
    if isinstance(algorithm, (DRBO, RoBOS)) and (
        environment_type != "distributional" or not algorithm.required_round_info.issubset(provided)
    ):
        support = getattr(environment, "context_support", None)
        # Obtain only already public pre-action context, never a hidden future realization.
        public = environment.before_action() if "context" in provided else {}
        context = public.get(
            "context", algorithm._default_context() if support is None else support[0]
        )
        adapted = StochasticContextAdapter(
            environment,
            constant_context=context,
            mode="singleton_distribution",
            context_support=support,
        )
        if support is not None:
            adapted._point_mass_index = 0
        return adapted
    return environment


__all__ = ["StochasticContextAdapter", "adapt_environment_for_algorithm"]

# These explicit delegating adapters implement the public Environment protocol.
Environment.register(StochasticContextAdapter)
Environment.register(UnperturbedAdapter)
