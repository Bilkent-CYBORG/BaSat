"""Expected Improvement over Good with a fixed satisfaction threshold."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import torch

from bayopy.algorithms.helper import SourceRecord

from .Exception.exception_EG import EGCheckpointError, EGException, EGPosteriorError
from .PG import PG, GoodActionConfig, GoodActionState


class EG(PG):
    """Maximize expected latent improvement above a fixed good-action level.

    Examples
    --------
    >>> import bayopy
    >>> bounds = bayopy.Bounds(lower=[-1.0, -1.0], upper=[1.0, 1.0])
    >>> algorithm = bayopy.EG(
    ...     bounds=bounds,
    ...     config=bayopy.GoodActionConfig(threshold=0.5, initial_points=5, seed=7),
    ... )
    """

    ACQUISITION_POLICY = "expected improvement above the good-action threshold"

    EXCEPTION_CLASS = EGException
    POSTERIOR_ERROR_TYPE = EGPosteriorError
    CHECKPOINT_ERROR_TYPE = EGCheckpointError
    SOURCE = SourceRecord(
        paper_title="Lenient Regret and Good-Action Identification in Gaussian Process Bandits",
        paper_year=2021,
        paper_url="https://proceedings.mlr.press/v139/cai21c.html",
        repository_url="https://github.com/caitree/GoodAction",
        repository_status="author",
        notes="Uses fixed-threshold EI, not incumbent expected improvement.",
    )

    @staticmethod
    def _expected_improvement_over_good(
        mean: torch.Tensor,
        std: torch.Tensor,
        threshold: float,
        *,
        std_floor: float = 1e-12,
    ) -> torch.Tensor:
        """Evaluate the stable fixed-threshold Gaussian improvement formula."""

        safe_std = std.clamp_min(std_floor)
        difference = mean - threshold
        z = difference / safe_std
        normal = torch.distributions.Normal(
            torch.zeros((), dtype=mean.dtype, device=mean.device),
            torch.ones((), dtype=mean.dtype, device=mean.device),
        )
        value = difference * normal.cdf(z) + safe_std * torch.exp(normal.log_prob(z))
        deterministic = std <= std_floor
        return torch.where(deterministic, difference.clamp_min(0.0), value)

    def _score_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
        """Score candidates by expected improvement above the fixed threshold."""

        mean, std = self.gp.predict(candidates)
        return self._expected_improvement_over_good(
            mean,
            std,
            self.config.threshold,
            std_floor=self.config.posterior_std_floor,
        )

    def _continuous_acquisition_kind(self) -> Literal["pg", "eg"]:
        """Use fixed-threshold expected improvement during box optimization."""

        return "eg"

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Extend shared diagnostics with the fixed-threshold EI identity."""

        diagnostics = dict(super()._diagnostics_impl())
        diagnostics["acquisition"] = "expected_improvement_over_fixed_threshold"
        return diagnostics


__all__ = ["EG", "GoodActionConfig", "GoodActionState"]
