"""Sequential finite actions at each row of an externally supplied context table."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from ..models.utils import Bounds, resolve_device, resolve_dtype
from .environment import (
    Environment,
    EnvironmentConfigurationError,
    EnvironmentProtocolError,
    NoAvailableActionError,
    _clone_tree,
    _Evaluation,
)


class SequentialContextualTableEnvironment(Environment):
    """Reveal a context and evaluate distinct actions sequentially at that row.

    Parameters
    ----------
    actions : array-like, shape (M, D)
        Distinct finite action coordinates, visible to the learner.
    contexts : array-like, shape (T, C)
        Context sequence. Context values may recur on different rows.
    observations : array-like, shape (T, M)
        Hidden response table. Only the chosen action's response is returned.
    queries_per_context : int
        Number of distinct, sequentially observed actions before advancing to
        the next row. This is not a batch acquisition.
    observation_offset, noise_variance : float
        Subtract the supplied calibration-only offset from returned rewards.
        Report the supplied nonnegative likelihood variance without drawing
        or adding synthetic noise.
    dtype, device, seed : tensor settings
        Environment precision, device, and reproducibility seed.

    Notes
    -----
    The full table is hashed for checkpoint identity but never serialized
    into the learner state. Checkpoints retain revealed data and row progress.
    Repeated coordinates are forbidden within one row and allowed on later
    rows. A round is exactly one sensor/action query.

    Examples
    --------
    >>> import basat
    >>> environment = basat.SequentialContextualTableEnvironment(
    ...     actions=[[0.0], [1.0]], contexts=[[0.0], [1.0]],
    ...     observations=[[1.0, 2.0], [3.0, 1.0]], queries_per_context=2,
    ...     observation_offset=0.0, noise_variance=0.01)
    """

    environment_type = "contextual"
    provided_round_info = frozenset({"context"})
    provided_feedback = frozenset({"observation", "evaluated_action", "context"})
    capabilities = frozenset({"contextual", "revealed_context", "finite_actions"})

    def __init__(
        self,
        *,
        actions: Any,
        contexts: Any,
        observations: Any,
        queries_per_context: int,
        observation_offset: float,
        noise_variance: float,
        dtype: Any = "float64",
        device: Any = "cpu",
        seed: int = 1,
    ):
        tensor_options: dict[str, Any] = {
            "dtype": resolve_dtype(dtype),
            "device": resolve_device(device),
        }
        self._actions = torch.as_tensor(actions, **tensor_options).detach().clone()
        self._contexts = torch.as_tensor(contexts, **tensor_options).detach().clone()
        self._observations = torch.as_tensor(observations, **tensor_options).detach().clone()
        if (
            self._actions.ndim != 2
            or self._contexts.ndim != 2
            or not len(self._actions)
            or not len(self._contexts)
            or self._observations.shape != (len(self._contexts), len(self._actions))
            or not all(
                torch.isfinite(a).all() for a in [self._actions, self._contexts, self._observations]
            )
        ):
            raise EnvironmentConfigurationError(
                "Supply finite matching action/context/response tables."
            )
        if len(torch.unique(self._actions, dim=0)) != len(self._actions):
            raise EnvironmentConfigurationError("Action coordinates must be unique.")
        if (
            isinstance(queries_per_context, bool)
            or not isinstance(queries_per_context, int)
            or not 1 <= queries_per_context <= len(self._actions)
        ):
            raise EnvironmentConfigurationError("queries_per_context must lie between 1 and M.")
        if (
            not torch.isfinite(torch.tensor([observation_offset, noise_variance])).all()
            or noise_variance < 0
        ):
            raise EnvironmentConfigurationError("Offset must be finite and variance nonnegative.")
        self.queries_per_context = queries_per_context
        self.observation_offset, self.noise_variance = (
            float(observation_offset),
            float(noise_variance),
        )
        lower, upper = self._actions.amin(0), self._actions.amax(0)
        if bool((lower == upper).any()):
            raise EnvironmentConfigurationError(
                "Each action coordinate must span a nonzero bound interval."
            )
        super().__init__(
            bounds=Bounds(lower=lower, upper=upper),
            function=self._read_table,
            dtype=dtype,
            device=device,
            allow_repeated_actions=True,
            seed=seed,
        )
        self._freeze_source_fingerprint()

    @property
    def context_dimension(self) -> int:
        """Return the declared learner-visible context dimension."""
        return int(self._contexts.shape[1])

    @property
    def maximum_rounds(self) -> int:
        """Return the finite table's total number of sequential queries."""
        return len(self._contexts) * self.queries_per_context

    def _available_indices(self) -> torch.Tensor:
        used = self.round % self.queries_per_context
        excluded = set()
        if used:
            excluded = {
                int(record.audit["action_index"]) for record in self.history.records[-used:]
            }
        return torch.tensor(
            [i for i in range(len(self._actions)) if i not in excluded],
            dtype=torch.long,
            device=self.device,
        )

    def before_action(self) -> dict[str, Any]:
        """Reveal the current context and still-unqueried finite actions only."""
        if self.round >= self.maximum_rounds:
            raise NoAvailableActionError("All contextual table rows have been evaluated.")
        if self._pending_round_info is None:
            indices = self._available_indices()
            self._pending_round_info = {
                "context": self._contexts[self.round // self.queries_per_context].clone(),
                "candidate_actions": self._actions[indices].clone(),
                "candidate_dataset_indices": indices,
            }
        return _clone_tree(self._pending_round_info)

    def _read_table(self, action: Any) -> torch.Tensor:
        if self._pending_round_info is None:
            raise EnvironmentProtocolError("Reveal context with before_action() first.")
        matches = (self._actions == action).all(1).nonzero().reshape(-1)
        available = self._available_indices()
        if len(matches) != 1 or not bool((available == matches[0]).any()):
            raise NoAvailableActionError("Action is absent or already queried at the current row.")
        return self._observations[self.round // self.queries_per_context, int(matches[0])]

    def _evaluate_one(self, action: torch.Tensor) -> _Evaluation:
        value = self._read_table(action)
        index = int((self._actions == action).all(1).nonzero().reshape(-1)[0])
        return _Evaluation(
            action=action,
            evaluated_action=action,
            observation=value - self.observation_offset,
            context=self._contexts[self.round // self.queries_per_context].clone(),
            observation_noise_variance=torch.tensor(
                self.noise_variance, dtype=self.dtype, device=self.device
            ),
            dataset_index=None,
            metadata={"source_mode": "sequential_table"},
            audit={
                "stored_observation": value.clone(),
                "action_index": index,
                "context_row": self.round // self.queries_per_context,
            },
        )

    def step(self, action: Any) -> dict[str, Any]:
        """Evaluate one action; simultaneous batches would violate sequential feedback."""
        if torch.as_tensor(action).ndim != 1:
            raise EnvironmentProtocolError(
                "Sequential contextual tables require one action per step."
            )
        return super().step(action)

    def _compute_source_fingerprint(self) -> str:
        digest = hashlib.sha256()
        self._update_fingerprint_digest(
            digest,
            {
                "class": f"{type(self).__module__}.{type(self).__qualname__}",
                "actions": self._actions,
                "contexts": self._contexts,
                "observations": self._observations,
                "queries_per_context": self.queries_per_context,
                "offset": self.observation_offset,
                "variance": self.noise_variance,
            },
        )
        return digest.hexdigest()
