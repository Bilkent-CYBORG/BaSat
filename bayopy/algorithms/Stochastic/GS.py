"""Good-Action Search using one-step Gaussian-process fantasy lookahead."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from bayopy.algorithms.helper import SourceRecord
from bayopy.models.utils import Bounds

from .Exception.exception_GS import (
    GSCheckpointError,
    GSException,
    GSFantasyError,
    GSMaximumSamplingError,
)
from .PG import PG, GoodActionConfig, GoodActionState


class GS(PG):
    """Select evaluations likely to reveal that a good action exists.

    Temporary fantasy observations are used only inside acquisition scoring.
    They are never appended to environment history or GP training data.

    Examples
    --------
    >>> import bayopy
    >>> bounds = bayopy.Bounds(lower=[-1.0, -1.0], upper=[1.0, 1.0])
    >>> algorithm = bayopy.GS(
    ...     bounds=bounds,
    ...     config=bayopy.GoodActionConfig(threshold=0.5, initial_points=5, seed=7),
    ... )
    """

    ACQUISITION_POLICY = "good-action search with fantasy observations"

    EXCEPTION_CLASS = GSException
    POSTERIOR_ERROR_TYPE = GSFantasyError
    CHECKPOINT_ERROR_TYPE = GSCheckpointError
    SOURCE = SourceRecord(
        paper_title="Lenient Regret and Good-Action Identification in Gaussian Process Bandits",
        paper_year=2021,
        paper_url="https://proceedings.mlr.press/v139/cai21c.html",
        repository_url="https://github.com/caitree/GoodAction",
        repository_status="author",
        notes="Clean-room fantasy/max-value Monte Carlo implementation.",
    )

    def __init__(
        self,
        *,
        bounds: Bounds,
        config: GoodActionConfig,
        maximize: bool = True,
        dtype: torch.dtype | str = torch.float32,
        device: torch.device | str = "cpu",
        plot: bool = False,
        verbose: bool = False,
        verbose_setting: bool = False,
        verbose_iter_by_iter: bool = False,
    ) -> None:
        """Initialize shared PG state and GS Monte Carlo diagnostics."""

        super().__init__(
            bounds=bounds,
            config=config,
            maximize=maximize,
            dtype=dtype,
            device=device,
            plot=plot,
            verbose=verbose,
            verbose_setting=verbose_setting,
            verbose_iter_by_iter=verbose_iter_by_iter,
        )
        self._last_tie_statistics: torch.Tensor | None = None
        self._last_mc_standard_errors: torch.Tensor | None = None

    @staticmethod
    def _condition_on_hypothetical(
        mean: torch.Tensor,
        covariance: torch.Tensor,
        candidate_index: int,
        hypothetical: torch.Tensor,
        predictive_variance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact Gaussian conditional moments without mutating the GP."""

        denominator = predictive_variance.clamp_min(torch.finfo(mean.dtype).eps)
        cross = covariance[:, candidate_index]
        gain = cross / denominator
        conditioned_mean = mean.unsqueeze(0) + (hypothetical - mean[candidate_index]).unsqueeze(
            -1
        ) * gain.unsqueeze(0)
        conditioned_covariance = covariance - torch.outer(cross, cross) / denominator
        conditioned_covariance = 0.5 * (
            conditioned_covariance + conditioned_covariance.transpose(-1, -2)
        )
        return conditioned_mean, conditioned_covariance

    @staticmethod
    def _stable_cholesky(covariance: torch.Tensor) -> torch.Tensor:
        """Factor a PSD fantasy covariance with bounded deterministic jitter."""

        identity = torch.eye(covariance.shape[-1], dtype=covariance.dtype, device=covariance.device)
        jitter = 1e-10 if covariance.dtype == torch.float64 else 1e-6
        for _ in range(6):
            factor, info = torch.linalg.cholesky_ex(covariance + jitter * identity)
            if int(info.max().item()) == 0:
                return factor
            jitter *= 10.0
        raise GSMaximumSamplingError(
            "Fantasy posterior covariance is not numerically positive semidefinite."
        )

    def _sample_posterior_maxima(
        self,
        conditioned_mean: torch.Tensor,
        conditioned_covariance: torch.Tensor,
    ) -> torch.Tensor:
        """Draw posterior maximum values for every hypothetical observation."""

        factor = self._stable_cholesky(conditioned_covariance)
        standard = torch.randn(
            (
                conditioned_mean.shape[0],
                self.config.maximum_value_samples,
                conditioned_mean.shape[1],
            ),
            dtype=self.dtype,
            device=self.device,
            generator=self.generator,
        )
        samples = conditioned_mean.unsqueeze(1) + torch.matmul(standard, factor.transpose(-1, -2))
        return samples.max(dim=-1).values

    def _gs_score_one_candidate(
        self,
        *,
        candidate_index: int,
        mean: torch.Tensor,
        covariance: torch.Tensor,
        predictive_variance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return GS score, maximum tie statistic, and Monte Carlo standard error."""

        scale = predictive_variance.sqrt().clamp_min(self.config.posterior_std_floor)
        hypothetical = mean[candidate_index] + scale * torch.randn(
            (self.config.lookahead_samples,),
            dtype=self.dtype,
            device=self.device,
            generator=self.generator,
        )
        conditioned_mean, conditioned_covariance = self._condition_on_hypothetical(
            mean,
            covariance,
            candidate_index,
            hypothetical,
            predictive_variance,
        )
        maxima = self._sample_posterior_maxima(conditioned_mean, conditioned_covariance)
        indicators = (maxima >= self.config.threshold).to(self.dtype)
        score = indicators.mean()
        sample_count = indicators.numel()
        standard_error = torch.sqrt((score * (1.0 - score)).clamp_min(0.0) / max(sample_count, 1))
        return score, maxima.max(), standard_error

    def _score_candidate_set(
        self, candidates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate fantasy lookahead on a common candidate representation."""

        try:
            latent = self.gp.posterior(candidates, observation_noise=False)
            predictive = self.gp.posterior(candidates, observation_noise=True)
            mean = latent.mean.squeeze(-1)
            covariance = latent.distribution.covariance_matrix
            predictive_variance = predictive.variance.squeeze(-1)
        except Exception as error:
            raise GSFantasyError(
                f"Unable to construct the GS posterior: {error}", algorithm=self
            ) from error
        scores = torch.empty(candidates.shape[0], dtype=self.dtype, device=self.device)
        ties = torch.empty_like(scores)
        errors = torch.empty_like(scores)
        for index in range(candidates.shape[0]):
            scores[index], ties[index], errors[index] = self._gs_score_one_candidate(
                candidate_index=index,
                mean=mean,
                covariance=covariance,
                predictive_variance=predictive_variance[index],
            )
        return scores, ties, errors

    def _score_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
        """Cache fantasy scores, tie statistics, and Monte Carlo uncertainty."""

        scores, ties, errors = self._score_candidate_set(candidates)
        self._last_tie_statistics = ties.detach().clone()
        self._last_mc_standard_errors = errors.detach().clone()
        return scores

    def _supports_continuous_optimization(self) -> bool:
        """Keep GS on its reproducible finite fantasy representation.

        The guide permits continuous refinement only when the fantasy score is
        differentiable and supported. This Monte Carlo implementation draws
        discrete good-action indicators, so a gradient refinement would not be
        mathematically meaningful.
        """

        return False

    def _select_candidate(
        self,
        candidates: torch.Tensor,
        scores: torch.Tensor,
        dataset_indices: torch.Tensor | None,
    ) -> int:
        """Apply GS primary/tie scores followed by stable finite-row ordering."""

        del candidates
        if self._last_tie_statistics is None:
            return super()._select_candidate(
                self._last_candidates if self._last_candidates is not None else torch.empty(0),
                scores,
                dataset_indices,
            )
        primary = self._last_tie_statistics if torch.all(scores <= 1e-12) else scores
        maximum = torch.max(primary)
        tolerance = 1e-12 + 1e-9 * abs(float(maximum.item()))
        tied = torch.nonzero(primary >= maximum - tolerance, as_tuple=False).flatten()
        if dataset_indices is not None:
            return int(tied[torch.argmin(dataset_indices[tied])].item())
        return int(tied[0].item())

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Report fantasy sample sizes and the largest Monte Carlo error."""

        diagnostics = dict(super()._diagnostics_impl())
        diagnostics.update(
            {
                "acquisition": "good_action_search_fantasy",
                "lookahead_samples": self.config.lookahead_samples,
                "maximum_value_samples": self.config.maximum_value_samples,
                "maximum_mc_standard_error": (
                    None
                    if self._last_mc_standard_errors is None
                    else float(self._last_mc_standard_errors.max().item())
                ),
            }
        )
        return diagnostics

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Add GS tie and Monte Carlo diagnostics to shared checkpoint state."""

        state = dict(super()._state_dict_impl())
        state["last_tie_statistics"] = (
            None
            if self._last_tie_statistics is None
            else self._last_tie_statistics.detach().clone()
        )
        state["last_mc_standard_errors"] = (
            None
            if self._last_mc_standard_errors is None
            else self._last_mc_standard_errors.detach().clone()
        )
        return state

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Require GS-specific diagnostic arrays in checkpoint state."""

        super()._validate_state_dict_impl(state, strict=strict)
        for key in ("last_tie_statistics", "last_mc_standard_errors"):
            if key not in state:
                raise GSCheckpointError(f"GS state is missing {key!r}.", algorithm=self)

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Restore GS Monte Carlo diagnostics after shared state restoration."""

        super()._load_state_dict_impl(state, strict=strict)
        tie = state["last_tie_statistics"]
        error = state["last_mc_standard_errors"]
        self._last_tie_statistics = (
            None
            if tie is None
            else torch.as_tensor(tie, dtype=self.dtype, device=self.device).detach().clone()
        )
        self._last_mc_standard_errors = (
            None
            if error is None
            else torch.as_tensor(error, dtype=self.dtype, device=self.device).detach().clone()
        )

    def _reset_impl(self) -> None:
        """Reset shared state and discard GS-only Monte Carlo diagnostics."""

        super()._reset_impl()
        self._last_tie_statistics = None
        self._last_mc_standard_errors = None


__all__ = ["GS", "GoodActionConfig", "GoodActionState"]
