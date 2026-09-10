"""Good-Action Search using the authors' posterior-sample acquisition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from bayopy.algorithms.helper import SourceRecord
from bayopy.models.utils import Bounds

from .Exception.exception_GS import (
    GSCheckpointError,
    GSException,
    GSMaximumSamplingError,
    GSPosteriorError,
)
from .PG import PG, GoodActionConfig, GoodActionState


class GS(PG):
    """Rank actions by their sampled posterior probability of being good.

    This follows ``gs`` and ``gs_acq`` in the authors' ``BO_methods.py``:
    count threshold exceedances at each candidate and, if no sample meets the
    threshold, select the largest sampled value. The paper's discussion of
    future global maxima is not the acquisition used by that implementation.

    Examples
    --------
    >>> import bayopy
    >>> bounds = bayopy.Bounds(lower=[-1.0, -1.0], upper=[1.0, 1.0])
    >>> algorithm = bayopy.GS(
    ...     bounds=bounds,
    ...     config=bayopy.GoodActionConfig(threshold=0.5, initial_points=5, seed=7),
    ... )
    """

    ACQUISITION_POLICY = "posterior-sample good-action probability"

    EXCEPTION_CLASS = GSException
    POSTERIOR_ERROR_TYPE = GSPosteriorError
    CHECKPOINT_ERROR_TYPE = GSCheckpointError
    SOURCE = SourceRecord(
        paper_title="Lenient Regret and Good-Action Identification in Gaussian Process Bandits",
        paper_year=2021,
        paper_url="https://proceedings.mlr.press/v139/cai21c.html",
        repository_url="https://github.com/caitree/GoodAction",
        repository_status="author",
        notes="Author-code GS posterior exceedance fraction and sampled-maximum fallback; "
        "deterministic finite-candidate search replaces noisy local refinement.",
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

    def _score_candidate_set(
        self, candidates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate each candidate's chance of meeting the reward threshold.

        Averaging a *global* maximum event over hypothetical observations is
        independent of the evaluated candidate by total expectation. Such a
        score ranks Monte Carlo error rather than actions. The authors' code
        instead estimates the candidate-specific posterior exceedance event.
        The existing two Monte Carlo budgets determine the total number of
        draws, preserving their effect on accuracy and checkpoint settings.
        """

        sample_count = self.config.lookahead_samples * self.config.maximum_value_samples
        try:
            samples = self.gp.sample_posterior(
                candidates,
                num_samples=sample_count,
                seed=self.config.seed + self.iteration,
                observation_noise=False,
            ).reshape(sample_count, candidates.shape[0])
        except Exception as error:
            raise GSPosteriorError(
                f"Unable to sample the GS posterior: {error}", algorithm=self
            ) from error
        if not torch.isfinite(samples).all():
            raise GSMaximumSamplingError(
                "GS posterior samples contain non-finite values.", algorithm=self
            )
        scores = (samples >= self.config.threshold).to(self.dtype).mean(dim=0)
        ties = samples.max(dim=0).values
        errors = torch.sqrt((scores * (1.0 - scores)).clamp_min(0.0) / sample_count)
        return scores, ties, errors

    def _score_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
        """Cache exceedance scores, fallback values, and Monte Carlo uncertainty."""

        scores, ties, errors = self._score_candidate_set(candidates)
        self._last_tie_statistics = ties.detach().clone()
        self._last_mc_standard_errors = errors.detach().clone()
        return scores

    def _supports_continuous_optimization(self) -> bool:
        """Keep GS on its reproducible finite posterior-sample representation.

        Discrete threshold indicators have no useful gradient for local
        refinement, so candidates are ranked directly.
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
        """Report posterior sample count and the largest Monte Carlo error."""

        diagnostics = dict(super()._diagnostics_impl())
        diagnostics.update(
            {
                "acquisition": "good_action_search_posterior_probability",
                "posterior_samples": (
                    self.config.lookahead_samples * self.config.maximum_value_samples
                ),
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
