"""Satisficing Thompson Sampling on a reproducible action representation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from bayopy.algorithms.helper import SourceRecord
from bayopy.models.utils import Bounds

from .Exception.exception_Satisficing_TS import (
    Satisficing_TSCheckpointError,
    Satisficing_TSException,
    Satisficing_TSPosteriorSampleError,
    Satisficing_TSReferenceError,
)
from .PG import PG, GoodActionConfig, GoodActionState


class Satisficing_TS(PG):
    """Choose the closest sampled-good action, with Thompson fallback.

    Examples
    --------
    >>> import bayopy
    >>> bounds = bayopy.Bounds(lower=[-1.0, -1.0], upper=[1.0, 1.0])
    >>> algorithm = bayopy.Satisficing_TS(
    ...     bounds=bounds,
    ...     config=bayopy.GoodActionConfig(
    ...         threshold=0.5, reference_point=[0.0, 0.0], initial_points=5, seed=7
    ...     ),
    ... )
    """

    ACQUISITION_POLICY = "satisficing Thompson sampling"

    EXCEPTION_CLASS = Satisficing_TSException
    POSTERIOR_ERROR_TYPE = Satisficing_TSPosteriorSampleError
    CHECKPOINT_ERROR_TYPE = Satisficing_TSCheckpointError
    SOURCE = SourceRecord(
        paper_title="Lenient Regret and Good-Action Identification in Gaussian Process Bandits",
        paper_year=2021,
        paper_url="https://proceedings.mlr.press/v139/cai21c.html",
        repository_url="https://github.com/caitree/GoodAction",
        repository_status="author",
        notes="Joint posterior sample and stable closest-good finite representation.",
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
        """Initialize STS and validate or derive its reference action."""

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
        configured = self.config.reference_point
        if configured is None:
            domain = self.config.candidate_domain
            if isinstance(domain, Bounds):
                candidate_bounds = domain.to(dtype=self.dtype, device=self.device)
                lower, upper = candidate_bounds.lower, candidate_bounds.upper
            else:
                lower, upper = self._bounds_on_algorithm_device()
            self.reference_point = (lower + upper) / 2.0
        else:
            reference = torch.as_tensor(configured, dtype=self.dtype, device=self.device)
            if reference.shape != (self.action_dimension,):
                raise Satisficing_TSReferenceError(
                    "reference_point must have shape (D_env,).", algorithm=self
                )
            lower, upper = self._bounds_on_algorithm_device()
            if not torch.all((reference >= lower) & (reference <= upper)):
                raise Satisficing_TSReferenceError(
                    "reference_point lies outside the action bounds.", algorithm=self
                )
            self.reference_point = reference.detach().clone()
        self._last_function_sample: torch.Tensor | None = None
        self._last_fallback = False
        self._last_sample_seed: int | None = None

    def _supports_continuous_optimization(self) -> bool:
        """STS selects from one joint sample on a finite representation."""

        return False

    def _draw_function_sample(self, candidates: torch.Tensor) -> torch.Tensor:
        """Draw one joint latent posterior function sample."""

        seed = self.config.seed + self.iteration
        try:
            values = self.gp.sample_posterior(
                candidates,
                num_samples=1,
                seed=seed,
                observation_noise=False,
            ).reshape(-1)
        except Exception as error:
            raise Satisficing_TSPosteriorSampleError(
                f"Joint Thompson sample failed: {error}", algorithm=self
            ) from error
        if values.shape != (candidates.shape[0],) or not torch.isfinite(values).all():
            raise Satisficing_TSPosteriorSampleError(
                "Joint Thompson sample has an invalid shape or non-finite value.",
                algorithm=self,
            )
        self._last_sample_seed = seed
        return values

    def _good_mask(self, sample: torch.Tensor) -> torch.Tensor:
        """Identify posterior sample values satisfying the aspiration level."""

        return sample >= self.config.threshold

    def _select_closest_good(
        self,
        candidates: torch.Tensor,
        sample: torch.Tensor,
        good_mask: torch.Tensor,
        dataset_indices: torch.Tensor | None,
    ) -> int:
        """Apply distance, sampled-value, then stable-index tie breaking."""

        good_positions = torch.nonzero(good_mask, as_tuple=False).flatten()
        distances = torch.linalg.vector_norm(
            candidates[good_positions] - self.reference_point, dim=-1
        )
        minimum = distances.min()
        distance_tolerance = max(
            1e-12 + 1e-9 * abs(float(minimum.item())),
            10.0 * torch.finfo(distances.dtype).eps,
        )
        distance_ties = good_positions[distances <= minimum + distance_tolerance]
        if distance_ties.numel() > 1:
            tied_samples = sample[distance_ties]
            maximum = tied_samples.max()
            distance_ties = distance_ties[
                tied_samples >= maximum - 1e-12 - 1e-9 * abs(float(maximum.item()))
            ]
        if distance_ties.numel() > 1 and dataset_indices is not None:
            return int(distance_ties[torch.argmin(dataset_indices[distance_ties])].item())
        return int(distance_ties[0].item())

    @staticmethod
    def _thompson_fallback(sample: torch.Tensor, dataset_indices: torch.Tensor | None) -> int:
        """Return the stable maximizer of a joint posterior sample."""

        maximum = sample.max()
        tied = torch.nonzero(
            sample >= maximum - 1e-12 - 1e-9 * abs(float(maximum.item())),
            as_tuple=False,
        ).flatten()
        if dataset_indices is not None:
            return int(tied[torch.argmin(dataset_indices[tied])].item())
        return int(tied[0].item())

    def _ask_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Choose an initial, closest sampled-good, or Thompson fallback action."""

        candidates, dataset_indices = self._candidate_set(round_info)
        self._last_candidates = candidates.detach().clone()
        self._last_candidate_indices = (
            None if dataset_indices is None else dataset_indices.detach().clone()
        )
        if self.state.y_train.numel() < self._initial_design_size():
            position = int(self.state.y_train.numel()) % candidates.shape[0]
            sample = torch.full(
                (candidates.shape[0],), float("nan"), dtype=self.dtype, device=self.device
            )
            fallback = False
            good_count = 0
        else:
            self._fit_model()
            sample = self._draw_function_sample(candidates)
            good_mask = self._good_mask(sample)
            good_count = int(good_mask.sum().item())
            fallback = good_count == 0
            position = (
                self._thompson_fallback(sample, dataset_indices)
                if fallback
                else self._select_closest_good(candidates, sample, good_mask, dataset_indices)
            )
            self._last_acquisition_value = float(sample[position].item())
        self._last_function_sample = sample.detach().clone()
        self._last_fallback = fallback
        selected = candidates[position]
        self._record_last_ask(
            actions=selected,
            model_inputs=selected,
            round_info=round_info,
            algorithm_metadata={
                "candidate_position": position,
                "dataset_index": (
                    None if dataset_indices is None else int(dataset_indices[position].item())
                ),
                "sample_seed": self._last_sample_seed,
                "fallback": fallback,
                "sampled_good_count": good_count,
            },
        )
        return selected

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Report the reference action and most recent sampling decision."""

        diagnostics = dict(super()._diagnostics_impl())
        diagnostics.update(
            {
                "acquisition": "satisficing_thompson_sample",
                "reference_point": self.reference_point.detach().clone(),
                "last_fallback": self._last_fallback,
                "last_sample_seed": self._last_sample_seed,
            }
        )
        return diagnostics

    def _result_metadata_impl(self) -> Mapping[str, Any]:
        """Describe whether the latest selection required Thompson fallback."""

        metadata = dict(super()._result_metadata_impl())
        metadata.update(
            {
                "last_selection_used_thompson_fallback": self._last_fallback,
                "reference_point": self.reference_point.detach().clone(),
            }
        )
        return metadata

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize STS reference, joint sample, fallback, and seed state."""

        state = dict(super()._state_dict_impl())
        state.update(
            {
                "reference_point_runtime": self.reference_point.detach().clone(),
                "last_function_sample": (
                    None
                    if self._last_function_sample is None
                    else self._last_function_sample.detach().clone()
                ),
                "last_fallback": self._last_fallback,
                "last_sample_seed": self._last_sample_seed,
            }
        )
        return state

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Validate STS-specific checkpoint fields and reference dimension."""

        super()._validate_state_dict_impl(state, strict=strict)
        required = {
            "reference_point_runtime",
            "last_function_sample",
            "last_fallback",
            "last_sample_seed",
        }
        if missing := required - set(state):
            raise Satisficing_TSCheckpointError(
                f"STS state is missing {sorted(missing)}.", algorithm=self
            )
        reference = torch.as_tensor(
            state["reference_point_runtime"], dtype=self.dtype, device=self.device
        )
        if reference.shape != (self.action_dimension,):
            raise Satisficing_TSCheckpointError(
                "STS checkpoint reference dimension mismatch.", algorithm=self
            )

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Restore the exact STS reference and posterior-sample diagnostics."""

        super()._load_state_dict_impl(state, strict=strict)
        self.reference_point = (
            torch.as_tensor(state["reference_point_runtime"], dtype=self.dtype, device=self.device)
            .detach()
            .clone()
        )
        sample = state["last_function_sample"]
        self._last_function_sample = (
            None
            if sample is None
            else torch.as_tensor(sample, dtype=self.dtype, device=self.device).detach().clone()
        )
        self._last_fallback = bool(state["last_fallback"])
        sample_seed = state["last_sample_seed"]
        self._last_sample_seed = None if sample_seed is None else int(sample_seed)

    def _reset_impl(self) -> None:
        """Clear posterior-sample diagnostics while retaining the reference."""

        super()._reset_impl()
        self._last_function_sample = None
        self._last_fallback = False
        self._last_sample_seed = None


__all__ = ["GoodActionConfig", "GoodActionState", "Satisficing_TS"]
