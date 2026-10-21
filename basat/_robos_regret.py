"""Expected-payoff regret for the released, approximate RoBOS metric oracle.

This source-reproduction metric deliberately accepts signed numerical oracle
outputs. It does not change the nonnegative mathematical fragility contract of
``RSRegret`` and never interprets a numerical source result as a certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .regret import RSRegretStatistics, rs_regret_statistics


def _numeric(value: Any, name: str) -> np.ndarray:
    """Validate finite real data without accepting booleans or numeric strings."""
    if isinstance(value, torch.Tensor):
        if value.is_complex() or value.dtype == torch.bool:
            raise TypeError(f"{name} must be real-valued numeric data.")
        result = value.detach().to(dtype=torch.float64, device="cpu").numpy()
    else:
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be numeric array-like data.") from error
        if raw.dtype.kind not in "iuf":
            raise TypeError(f"{name} must be real-valued numeric data.")
        result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _scalar(value: Any, name: str) -> float:
    raw = _numeric(value, name)
    if raw.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    return float(raw)


@dataclass(frozen=True)
class RoBOSSourceRegret:
    """Cumulative released-source regret from expected true-distribution rewards.

    At round ``t`` the instantaneous quantity is
    ``max(aspiration - source_kappa[t] * epsilon - expected_rewards[t], 0)``.
    ``source_kappa`` is a finite, possibly negative approximate numerical oracle
    value. It is retained without clipping and is not an optimal-fragility
    certificate. Only this linear, p=1 source formula is implemented.

    Rewards must already be expectations over the hidden true distribution;
    realized-context rewards or noisy learner observations are not substitutes.
    The caller supplies metric-only data; evaluating this object never invokes
    an optimizer, evaluates an objective, or consumes randomness.
    """

    aspiration: float
    epsilon: float
    cumulative: bool = True

    def __post_init__(self) -> None:
        aspiration, epsilon = _scalar(self.aspiration, "aspiration"), _scalar(self.epsilon, "epsilon")
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative.")
        if not isinstance(self.cumulative, bool):
            raise TypeError("cumulative must be Boolean.")
        object.__setattr__(self, "aspiration", aspiration)
        object.__setattr__(self, "epsilon", epsilon)

    def instantaneous(
        self, expected_rewards: Any, *, source_kappa: Any,
    ) -> np.ndarray | torch.Tensor:
        """Score scalar, shared per-round, or exactly aligned oracle values.

        Rewards have time on the final axis. Kappa may be scalar, a vector
        matching that axis, or an array with exactly the reward shape, such as
        one independent oracle trajectory per repetition. Other broadcasting
        is rejected to avoid silently assigning a trial's value to its rounds.
        A Torch reward input retains its device and floating dtype.
        """
        rewards = _numeric(expected_rewards, "expected_rewards")
        kappa = _numeric(source_kappa, "source_kappa")
        if rewards.ndim < 1 or rewards.size == 0:
            raise ValueError("expected_rewards must be nonempty with rounds on the last axis.")
        if (kappa.ndim != 0 and kappa.shape != rewards.shape
                and kappa.shape != (rewards.shape[-1],)):
            raise ValueError("source_kappa must be scalar, match the last round axis, or match expected_rewards.")
        if isinstance(expected_rewards, torch.Tensor):
            values = expected_rewards.detach()
            if not values.is_floating_point():
                values = values.to(dtype=torch.float64)
            oracle = torch.as_tensor(kappa, dtype=values.dtype, device=values.device)
            deficit = self.aspiration - oracle * self.epsilon - values
            if not bool(torch.isfinite(deficit).all()):
                raise ValueError("The configured source regret overflows at the requested precision.")
            return deficit.clamp_min(0)
        with np.errstate(over="ignore", invalid="ignore"):
            deficit = self.aspiration - kappa * self.epsilon - rewards
        if not np.isfinite(deficit).all():
            raise ValueError("The configured source regret overflows at the requested precision.")
        return np.maximum(deficit, 0.0)

    def evaluate(
        self, expected_rewards: Any, *, source_kappa: Any, cumulative: bool | None = None,
    ) -> np.ndarray | torch.Tensor:
        """Evaluate source regret, accumulating only along the final time axis."""
        accumulate = self.cumulative if cumulative is None else cumulative
        if not isinstance(accumulate, bool):
            raise TypeError("cumulative must be Boolean or None.")
        values = self.instantaneous(expected_rewards, source_kappa=source_kappa)
        if not accumulate:
            return values
        if isinstance(values, torch.Tensor):
            result = values.cumsum(dim=-1)
            finite = bool(torch.isfinite(result).all())
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                result = np.cumsum(values, axis=-1)
            finite = bool(np.isfinite(result).all())
        if not finite:
            raise ValueError("Cumulative source regret overflows at the requested precision.")
        return result

    def statistics(
        self, expected_reward_runs: Any, *, source_kappa: Any, error_scale: float = 1.0,
    ) -> RSRegretStatistics:
        """Return ordinary Basat population statistics for rectangular runs."""
        return rs_regret_statistics(
            self.evaluate(expected_reward_runs, source_kappa=source_kappa), error_scale=error_scale,
        )


__all__ = ["RoBOSSourceRegret"]
