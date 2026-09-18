"""Hashing-Embedding Subspace Bayesian Optimization (HeSBO)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch

from basat.algorithms.algorithm import (
    Algorithm,
    AlgorithmConfigurationError,
    resolve_batch_configuration,
)
from basat.algorithms.helper import LastAskRecord, SourceRecord, to_quality
from basat.models.Acquisitions import (
    AcquisitionConfig,
    DiscreteAcquisitionCandidates,
    build_acquisition,
    optimize_acquisition_continuous,
    optimize_acquisition_discrete,
)
from basat.models.GaussianProcess import GaussianProcess, GaussianProcessConfig
from basat.models.Kernels import KernelConfig
from basat.models.utils import Bounds, from_symmetric_box, to_symmetric_box

from .Exception.exception_HeSBO import (
    HeSBOCheckpointError,
    HeSBODataCompatibilityError,
    HeSBOEmptyBucketError,
    HeSBOException,
    HeSBOHistoricalDataError,
)
from .REMBO import (
    _AX02_MODEL_FIT_POLICY,
    _AX_UNIFORM_INITIALIZATION,
    _ax01_embedding_proposal,
    _ax_uniform_initial_point,
)


@dataclass(frozen=True, slots=True)
class HeSBOConfig:
    """Configuration of the immutable hash/sign embedding.

    Parameters
    ----------
    finite_candidate_mode:
        "project" scores actual finite rows in compressed coordinates (a warned
        extension); "exact" requires original embedding geometry. Continuous
        proposal generation is unchanged.
    embedding_dimension:
        Reduced signed-hash dimension ``D_model``.
    source_protocol:
        ``"hesbo"`` preserves native settings. ``"ax_0_1_17"`` selects the
        released comparison baseline: uniform initialization, unit latent inputs,
        sample-standardized targets, fixed standardized noise 1e-7, Matérn 5/2
        ARD MAP, and 512-sample qNEI with 20 SLSQP starts (1000 raw candidates).
        The latter accepts sequential ``acquisition.name="EI"`` only, the
        source paper's shorthand for its effective noisy-EI implementation.
        ``"ax_0_2_2"`` uses the same GP priors and qNEI but the newer 1e-6
        standardized noise floor and unconstrained L-BFGS-B acquisition solve.
        Its GP fitter uses five total SciPy-default attempts, prior resampling
        after a failed attempt, and no Adam fallback, matching BoTorch 0.5.1.
    acquisition:
        EI/UCB construction and Monte Carlo settings.
    initial_points:
        Optional initial-design size.
    initial_points_sampling_method:
        ``"paper"``, ``"sobol"``, or ``"uniform"`` initial-design sampling.
    hash_family:
        Immutable paper hash/sign construction selector.
    batch_size, seed:
        Sequential request size and embedding/random seed.
    """

    embedding_dimension: int
    source_protocol: Literal["hesbo", "ax_0_1_17", "ax_0_2_2"] = "hesbo"
    finite_candidate_mode: Literal["project", "exact"] = field(default="project", kw_only=True)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    initial_points: int | None = None
    initial_points_sampling_method: Literal["paper", "sobol", "uniform"] = "paper"
    hash_family: Literal["paper"] = "paper"
    batch_size: int | None = 1
    q_batch: int | None = None
    seed: int = 1


@dataclass(slots=True)
class HeSBOState:
    """Mutable HeSBO map, embedded data, and pending request state.

    Attributes
    ----------
    hash_index, sign, bucket_members:
        Immutable ambient-to-reduced signed-hash map and inverse buckets.
    Y_train, X_nominal_symmetric, y_train:
        Aligned embedded, nominal ambient, and response history.
    last_ask_Y, last_ask_X:
        Pending embedded and physical coordinates.
    """

    hash_index: torch.Tensor
    sign: torch.Tensor
    bucket_members: tuple[torch.Tensor, ...]
    Y_train: torch.Tensor
    X_nominal_symmetric: torch.Tensor
    y_train: torch.Tensor
    last_ask_Y: torch.Tensor | None
    last_ask_X: torch.Tensor | None


class HeSBO(Algorithm):
    """Optimize through a sparse signed hash map from ``R^d`` to ``R^D``.

    Examples
    --------
    >>> import basat
    >>> bounds = basat.Bounds(lower=[-1.0] * 6, upper=[1.0] * 6)
    >>> algorithm = basat.HeSBO(
    ...     bounds=bounds,
    ...     config=basat.HeSBOConfig(embedding_dimension=2, seed=7),
    ... )
    """

    ACQUISITION_POLICY = "configurable EI or UCB in the hash embedding"

    EXCEPTION_CLASS = HeSBOException
    native_batching = False
    native_environments = frozenset({"stochastic"})
    SOURCE = SourceRecord(
        paper_title="A Framework for Bayesian Optimization in Embedded Subspaces",
        paper_year=2019,
        paper_url="https://proceedings.mlr.press/v97/nayebi19a.html",
        repository_url="https://github.com/aminnayebi/HesBO",
        repository_status="author",
        notes="Signed-hash continuous search with a warned finite-row compression extension.",
    )

    def __init__(
        self,
        *,
        bounds: Bounds,
        config: HeSBOConfig,
        maximize: bool = True,
        dtype: torch.dtype | str = torch.float32,
        device: torch.device | str = "cpu",
        plot: bool = False,
        verbose: bool = False,
        verbose_setting: bool = False,
        verbose_iter_by_iter: bool = False,
    ) -> None:
        """Initialize HeSBO and draw its immutable hash/sign map.

        Parameters
        ----------
        bounds:
            Physical action bounds in ``D_env`` coordinates.
        config:
            Embedded dimension, acquisition, initial design, and seed.
        maximize:
            Optimize larger observations when ``True``.
        dtype, device:
            Tensor precision and execution device.
        plot, verbose, verbose_setting, verbose_iter_by_iter:
            Presentation flags inherited from :class:`Algorithm`.

        Raises
        ------
        AlgorithmConfigurationError
            If the embedded dimension or sequential source contract is invalid.
        """

        self.config = config
        self._validate_config(config, bounds.dimension)
        super().__init__(
            bounds=bounds,
            batch_size=config.batch_size,
            q_batch=config.q_batch,
            maximize=maximize,
            dtype=dtype,
            device=device,
            seed=config.seed,
            initial_points_sampling_method=config.initial_points_sampling_method,
            plot=plot,
            verbose=verbose,
            verbose_setting=verbose_setting,
            verbose_iter_by_iter=verbose_iter_by_iter,
        )
        hash_index, sign, buckets = self._draw_hash_and_sign()
        self.state = HeSBOState(
            hash_index=hash_index,
            sign=sign,
            bucket_members=buckets,
            Y_train=torch.empty(
                (0, config.embedding_dimension), dtype=self.dtype, device=self.device
            ),
            X_nominal_symmetric=torch.empty(
                (0, self.action_dimension), dtype=self.dtype, device=self.device
            ),
            y_train=torch.empty((0,), dtype=self.dtype, device=self.device),
            last_ask_Y=None,
            last_ask_X=None,
        )
        self.model_input_dimension = config.embedding_dimension
        self.gp = GaussianProcess(
            GaussianProcessConfig(
                kernel=KernelConfig(name="rbf", ard_num_dims=config.embedding_dimension),
                mean="constant",
                noise_mode="learned",
                dtype=self.dtype,
                device=self.device,
            ),
            input_dimension=config.embedding_dimension,
        )

    @staticmethod
    def _validate_config(config: HeSBOConfig, D: int) -> None:
        """Validate immutable HeSBO settings against ambient dimension ``D``."""

        if not isinstance(config, HeSBOConfig):
            raise AlgorithmConfigurationError("config must be HeSBOConfig.")
        if config.source_protocol not in {"hesbo", "ax_0_1_17", "ax_0_2_2"}:
            raise AlgorithmConfigurationError("Unknown HeSBO source_protocol.")
        if config.source_protocol.startswith("ax_") and (
            config.acquisition.name != "EI" or config.batch_size != 1 or config.q_batch is not None
        ):
            raise AlgorithmConfigurationError("Historical Ax HeSBO requires sequential EI/NEI.")
        if config.finite_candidate_mode not in {"project", "exact"}:
            raise AlgorithmConfigurationError("finite_candidate_mode must be project or exact.")
        if (
            isinstance(config.embedding_dimension, bool)
            or not isinstance(config.embedding_dimension, int)
            or not 1 <= config.embedding_dimension <= D
        ):
            raise AlgorithmConfigurationError("HeSBO requires 1 <= embedding_dimension <= D.")
        if not isinstance(config.acquisition, AcquisitionConfig):
            raise AlgorithmConfigurationError("acquisition must be AcquisitionConfig.")
        if config.initial_points is not None and (
            isinstance(config.initial_points, bool)
            or not isinstance(config.initial_points, int)
            or config.initial_points < 1
        ):
            raise AlgorithmConfigurationError("initial_points must be positive or None.")
        if config.hash_family != "paper":
            raise AlgorithmConfigurationError(
                "HeSBO version 1 implements only the paper hash family."
            )
        resolve_batch_configuration(batch_size=config.batch_size, q_batch=config.q_batch)

    def _draw_hash_and_sign(self) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        """Draw seeded uniform buckets and independent Rademacher signs."""

        d = self.config.embedding_dimension
        hash_index = torch.randint(
            d,
            (self.action_dimension,),
            device=self.device,
            generator=self.generator,
        )
        sign = (
            2
            * torch.randint(
                2,
                (self.action_dimension,),
                device=self.device,
                generator=self.generator,
            )
            - 1
        ).to(self.dtype)
        buckets = tuple(
            torch.nonzero(hash_index == index, as_tuple=False).flatten() for index in range(d)
        )
        return hash_index, sign, buckets

    @property
    def empty_buckets(self) -> tuple[int, ...]:
        """Return embedded coordinates that affect no ambient coordinate."""

        return tuple(
            index for index, members in enumerate(self.state.bucket_members) if members.numel() == 0
        )

    def _map_y_to_action(self, Y: torch.Tensor) -> torch.Tensor:
        """Apply ``x_i = s(i) y_h(i)`` and map to physical bounds."""

        X = Y[:, self.state.hash_index] * self.state.sign.unsqueeze(0)
        if bool((X.abs() > 1.0 + 1e-12).any()):
            raise HeSBODataCompatibilityError("HeSBO map left the symmetric box.", algorithm=self)
        return from_symmetric_box(X, self.bounds.to(dtype=self.dtype, device=self.device))

    def _metadata_Y(self, metadata: Mapping[str, Any], rows: int) -> torch.Tensor | None:
        """Read optional exact ``(rows, D_model)`` coordinates from metadata.

        Raises
        ------
        HeSBODataCompatibilityError
            If a supplied coordinate table has an incompatible shape.
        """

        nested = metadata.get("initial_metadata")
        if isinstance(nested, Mapping):
            metadata = {**nested, **metadata}
        for key in ("hesbo_Y", "embedded_coordinates", "model_inputs"):
            if key in metadata:
                value = torch.as_tensor(metadata[key], dtype=self.dtype, device=self.device)
                if value.shape != (rows, self.config.embedding_dimension):
                    raise HeSBODataCompatibilityError(
                        f"{key} must have shape ({rows}, {self.config.embedding_dimension}).",
                        algorithm=self,
                    )
                return value
        return None

    def _recover_y_if_exact(
        self,
        actions: torch.Tensor,
        *,
        metadata: Mapping[str, Any] | None = None,
        historical: bool = False,
    ) -> torch.Tensor:
        """Compress bucket coordinates, or verify the exact map in strict mode."""

        if self.config.finite_candidate_mode == "project":
            X = to_symmetric_box(actions, self.bounds.to(dtype=self.dtype, device=self.device))
            # Least-squares coordinates of the signed-hash expansion. Empty
            # buckets carry no information and receive the minimum-norm value 0.
            return torch.stack(
                [
                    (X[:, members] * self.state.sign[members]).mean(dim=-1)
                    if members.numel()
                    else X.new_zeros(X.shape[0])
                    for members in self.state.bucket_members
                ],
                dim=-1,
            )
        supplied = self._metadata_Y(metadata or {}, actions.shape[0])
        X = to_symmetric_box(actions, self.bounds.to(dtype=self.dtype, device=self.device))
        if supplied is None:
            if self.empty_buckets:
                message = (
                    "Empty HeSBO buckets make y unidentifiable without stored embedded coordinates."
                )
                if historical:
                    raise HeSBOHistoricalDataError(message, algorithm=self)
                raise HeSBOEmptyBucketError(message, algorithm=self)
            columns: list[torch.Tensor] = []
            for members in self.state.bucket_members:
                signed = X[:, members] * self.state.sign[members]
                reference = signed[:, :1]
                if not torch.isclose(signed, reference, rtol=0.0, atol=1e-8).all():
                    message = "Ambient coordinates in one HeSBO bucket are inconsistent."
                    if historical:
                        raise HeSBOHistoricalDataError(message, algorithm=self)
                    raise HeSBODataCompatibilityError(message, algorithm=self)
                columns.append(reference[:, 0])
            supplied = torch.stack(columns, dim=-1)
        if not torch.isclose(self._map_y_to_action(supplied), actions, rtol=0.0, atol=1e-8).all():
            message = "Embedded coordinates do not exactly reproduce HeSBO actions."
            if historical:
                raise HeSBOHistoricalDataError(message, algorithm=self)
            raise HeSBODataCompatibilityError(message, algorithm=self)
        return supplied.detach().clone()

    def _initialize_from_environment_impl(self, data: Any) -> None:
        """Encode learner-visible historical rows in signed-hash coordinates."""

        actions = torch.as_tensor(data.actions, dtype=self.dtype, device=self.device)
        if actions.shape[0] == 0:
            return
        Y = self._recover_y_if_exact(
            actions,
            metadata=data.metadata if isinstance(data.metadata, Mapping) else None,
            historical=True,
        )
        self.state.Y_train = Y
        self.state.X_nominal_symmetric = to_symmetric_box(
            actions, self.bounds.to(dtype=self.dtype, device=self.device)
        )
        self.state.y_train = (
            torch.as_tensor(data.observations, dtype=self.dtype, device=self.device)
            .detach()
            .clone()
        )
        self._set_gp_data()

    def _prepare_discrete_candidates_impl(
        self,
        *,
        candidate_actions: torch.Tensor,
        candidate_dataset_indices: torch.Tensor,
        round_info: Mapping[str, Any],
    ) -> DiscreteAcquisitionCandidates:
        """Represent available rows in signed-hash model coordinates."""

        Y = self._recover_y_if_exact(candidate_actions, metadata=round_info)
        return DiscreteAcquisitionCandidates(
            environment_actions=candidate_actions.detach().clone(),
            model_inputs=Y,
            dataset_indices=candidate_dataset_indices.detach().clone(),
        )

    def _set_gp_data(self) -> None:
        """Synchronize non-empty HeSBO training tensors with the GP wrapper."""

        if self.state.y_train.numel():
            self.gp.set_data(
                self.state.Y_train,
                to_quality(self.state.y_train, maximize=self.maximize),
            )

    def _perturb_continuous_batch_record(self, record: LastAskRecord, offset: int) -> LastAskRecord:
        """Keep duplicate-batch displacements inside the signed-hash image."""

        if record.model_inputs is None:
            raise HeSBODataCompatibilityError("HeSBO batch proposal has no latent coordinates.")
        Y = record.model_inputs.detach().clone()
        coordinate = offset % self.config.embedding_dimension
        step = 1e-4 * (1 + offset // self.config.embedding_dimension)
        direction = -1.0 if Y[0, coordinate] > 0 else 1.0
        Y[0, coordinate] = (Y[0, coordinate] + direction * step).clamp(-1.0, 1.0)
        actions = self._map_y_to_action(Y)
        return LastAskRecord(
            actions=actions,
            model_inputs=Y,
            round_info_snapshot=record.round_info_snapshot,
            algorithm_metadata=record.algorithm_metadata,
        )

    def _fit_embedded_gp(self) -> None:
        """Fit the GP over the current ``D_model`` signed-hash coordinates."""

        self._set_gp_data()
        self.gp.fit(maxiter=50)

    def _initial_design_size(self) -> int:
        """Return the configured or dimension-derived initial-design size."""

        return self.config.initial_points or max(2 * self.config.embedding_dimension, 4)

    def _ask_impl(self, *, round_info: Mapping[str, Any]) -> torch.Tensor:
        """Select the next continuous or finite HeSBO action.

        Raises
        ------
        HeSBODataCompatibilityError
            If candidate representation, GP fitting, or acquisition optimization fails.
        HeSBOEmptyBucketError
            If an inverse finite representation is unidentifiable without metadata.
        """

        if self.config.source_protocol.startswith("ax_"):
            if "candidate_actions" in round_info:
                raise HeSBODataCompatibilityError(
                    "Historical Ax comparison requires a continuous domain."
                )
            count = self.state.y_train.numel()
            lower = -torch.ones(
                self.config.embedding_dimension, dtype=self.dtype, device=self.device
            )
            upper = -lower
            if count < self._initial_design_size():
                unit = _ax_uniform_initial_point(
                    count=count,
                    dimension=self.config.embedding_dimension,
                    seed=self.config.seed,
                    dtype=self.dtype,
                    device=self.device,
                )
                Y = lower + unit * 2
            else:
                Y, self.gp, ax_value = _ax01_embedding_proposal(
                    self.state.Y_train,
                    to_quality(self.state.y_train, maximize=self.maximize),
                    lower,
                    upper,
                    self.config.seed + self.iteration,
                    protocol=self.config.source_protocol,
                )
                self._last_acquisition_value = float(ax_value.sum().item())
            action = self._map_y_to_action(Y)
            X = to_symmetric_box(action, self.bounds.to(dtype=self.dtype, device=self.device))
            self.state.last_ask_Y, self.state.last_ask_X = Y.detach().clone(), X.detach().clone()
            self._record_last_ask(
                actions=action,
                model_inputs=Y,
                round_info=round_info,
                algorithm_metadata={
                    "hash_index": self.state.hash_index,
                    "sign": self.state.sign,
                    "source_protocol": self.config.source_protocol,
                },
            )
            return action
        table: DiscreteAcquisitionCandidates | None = None
        if "candidate_actions" in round_info:
            table = self.prepare_discrete_candidates(
                candidate_actions=round_info["candidate_actions"],
                candidate_dataset_indices=round_info["candidate_dataset_indices"],
                round_info=round_info,
            )
        if self.state.y_train.numel() < self._initial_design_size():
            if table is None:
                Y = self._sample_design(
                    Bounds(
                        -torch.ones(
                            self.config.embedding_dimension, dtype=self.dtype, device=self.device
                        ),
                        torch.ones(
                            self.config.embedding_dimension, dtype=self.dtype, device=self.device
                        ),
                    ),
                    self.request_size,
                    phase="initial",
                    seed=self.config.seed + self.iteration,
                )
                action = self._map_y_to_action(Y)
            else:
                if table.dataset_indices.numel() < self.request_size:
                    raise HeSBODataCompatibilityError(
                        "The finite table has fewer rows than the requested HeSBO batch.",
                        algorithm=self,
                    )
                positions = torch.argsort(table.dataset_indices, stable=True)[: self.request_size]
                Y = table.model_inputs[positions]
                action = table.environment_actions[positions]
            value = None
        else:
            try:
                self._fit_embedded_gp()
                quality = self.state.y_train if self.maximize else -self.state.y_train
                acquisition = build_acquisition(
                    model=self.gp,
                    config=self.config.acquisition,
                    batch_size=self.batch_size,
                    q_batch=self.q_batch,
                    best_f=(
                        quality.max()
                        if self.config.acquisition.name.lower().startswith("e")
                        else None
                    ),
                )
                if table is None:
                    Y, score = optimize_acquisition_continuous(
                        acquisition=acquisition,
                        bounds=torch.stack(
                            (
                                -torch.ones(
                                    self.config.embedding_dimension,
                                    dtype=self.dtype,
                                    device=self.device,
                                ),
                                torch.ones(
                                    self.config.embedding_dimension,
                                    dtype=self.dtype,
                                    device=self.device,
                                ),
                            )
                        ),
                        batch_size=self.batch_size,
                        q_batch=self.q_batch,
                        num_restarts=10,
                        raw_samples=256,
                        sampling_method=self.acquisition_sampling_method,
                        sampling_generator=self.generator,
                        sampling_seed=self.config.seed + self.iteration,
                    )
                    action = self._map_y_to_action(Y)
                    value = float(score.item())
                else:
                    result = optimize_acquisition_discrete(
                        acquisition=acquisition,
                        candidates=table,
                        batch_size=self.batch_size,
                        q_batch=self.q_batch,
                    )
                    Y = result.model_inputs
                    action = result.environment_actions
                    value = float(result.acquisition_values.sum().item())
            except (HeSBODataCompatibilityError, HeSBOEmptyBucketError):
                raise
            except Exception as error:
                raise HeSBODataCompatibilityError(
                    f"HeSBO acquisition failed: {error}", algorithm=self
                ) from error
            self._last_acquisition_value = value
        X = to_symmetric_box(action, self.bounds.to(dtype=self.dtype, device=self.device))
        self.state.last_ask_Y = Y.detach().clone()
        self.state.last_ask_X = X.detach().clone()
        self._record_last_ask(
            actions=action,
            model_inputs=Y,
            round_info=round_info,
            algorithm_metadata={"hash_index": self.state.hash_index, "sign": self.state.sign},
        )
        return action

    def _tell_impl(
        self, *, action: torch.Tensor, feedback: Mapping[str, Any], round_info: Mapping[str, Any]
    ) -> None:
        """Append feedback using the exact embedded coordinates retained by ``ask``."""

        del round_info
        if self.last_ask is None or self.last_ask.model_inputs is None:
            raise HeSBODataCompatibilityError("HeSBO tell is missing ask state.", algorithm=self)
        pending_Y = self.last_ask.model_inputs
        pending_X = to_symmetric_box(action, self.bounds.to(dtype=self.dtype, device=self.device))
        self.state.Y_train = torch.cat((self.state.Y_train, pending_Y), dim=0)
        self.state.X_nominal_symmetric = torch.cat(
            (self.state.X_nominal_symmetric, pending_X), dim=0
        )
        self.state.y_train = torch.cat((self.state.y_train, feedback["observation"]), dim=0)
        self.state.last_ask_Y = None
        self.state.last_ask_X = None
        self._set_gp_data()

    def _bucket_diagnostics(self) -> Mapping[str, Any]:
        """Summarize signed-hash bucket occupancy and coordinate fractions."""

        sizes = tuple(int(members.numel()) for members in self.state.bucket_members)
        return {
            "bucket_sizes": sizes,
            "empty_buckets": self.empty_buckets,
            "largest_bucket": max(sizes),
            "bucket_fractions": tuple(size / self.action_dimension for size in sizes),
        }

    def _diagnostics_impl(self) -> Mapping[str, Any]:
        """Return bucket statistics and learner-visible training-row count."""

        return {
            **self._bucket_diagnostics(),
            "training_rows": int(self.state.y_train.numel()),
            "source_model_fit_policy": _AX02_MODEL_FIT_POLICY
            if self.config.source_protocol == "ax_0_2_2"
            else None,
        }

    def _state_dict_impl(self) -> Mapping[str, Any]:
        """Serialize the hash map, embedded history, and pending progress."""

        return {
            "source_model_fit_policy": _AX02_MODEL_FIT_POLICY
            if self.config.source_protocol == "ax_0_2_2"
            else None,
            "source_initialization": _AX_UNIFORM_INITIALIZATION
            if self.config.source_protocol.startswith("ax_")
            else None,
            "config": repr(asdict(self.config)),
            "hash_index": self.state.hash_index,
            "sign": self.state.sign,
            "Y_train": self.state.Y_train,
            "X_nominal_symmetric": self.state.X_nominal_symmetric,
            "y_train": self.state.y_train,
            "last_ask_Y": self.state.last_ask_Y,
            "last_ask_X": self.state.last_ask_X,
        }

    def _validate_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Validate required HeSBO checkpoint fields and ambient hash shape."""

        if (
            self.config.source_protocol == "ax_0_2_2"
            and state.get("source_model_fit_policy") != _AX02_MODEL_FIT_POLICY
        ):
            raise HeSBOCheckpointError(
                "Ax 0.2.2 requires the corrected source GP fitting policy. "
                "Preserve the old checkpoint and use a fresh output directory.",
                algorithm=self,
            )
        if (
            self.config.source_protocol.startswith("ax_")
            and state.get("source_initialization") != _AX_UNIFORM_INITIALIZATION
        ):
            raise HeSBOCheckpointError(
                "Historical Ax initialization requires the corrected uniform policy. "
                "Preserve the old checkpoint and start this source protocol in a fresh output directory.",
                algorithm=self,
            )

        required = {
            "config",
            "hash_index",
            "sign",
            "Y_train",
            "X_nominal_symmetric",
            "y_train",
            "last_ask_Y",
            "last_ask_X",
        }
        if missing := required - set(state):
            raise HeSBOCheckpointError(f"HeSBO state is missing {sorted(missing)}.", algorithm=self)
        if torch.as_tensor(state["hash_index"]).shape != (self.action_dimension,):
            raise HeSBOCheckpointError("HeSBO hash state has an invalid shape.", algorithm=self)
        del strict

    def _load_state_dict_impl(self, state: Mapping[str, Any], *, strict: bool) -> None:
        """Restore the HeSBO map and history, then reset the GP."""

        del strict
        self.state.hash_index = (
            torch.as_tensor(state["hash_index"], dtype=torch.long, device=self.device)
            .detach()
            .clone()
        )
        self.state.sign = (
            torch.as_tensor(state["sign"], dtype=self.dtype, device=self.device).detach().clone()
        )
        self.state.bucket_members = tuple(
            torch.nonzero(self.state.hash_index == index, as_tuple=False).flatten()
            for index in range(self.config.embedding_dimension)
        )
        for name in ("Y_train", "X_nominal_symmetric", "y_train"):
            setattr(
                self.state,
                name,
                torch.as_tensor(state[name], dtype=self.dtype, device=self.device).detach().clone(),
            )
        self.state.last_ask_Y = self._optional_tensor(state["last_ask_Y"])
        self.state.last_ask_X = self._optional_tensor(state["last_ask_X"])
        self.gp.reset()

    def _optional_tensor(self, value: Any) -> torch.Tensor | None:
        """Convert an optional checkpoint value to an owned algorithm tensor."""

        return (
            None
            if value is None
            else torch.as_tensor(value, dtype=self.dtype, device=self.device).detach().clone()
        )

    def _reset_impl(self) -> None:
        """Clear observations and redraw the seeded hash/sign embedding."""

        self.gp.reset()
        hash_index, sign, buckets = self._draw_hash_and_sign()
        self.state = HeSBOState(
            hash_index=hash_index,
            sign=sign,
            bucket_members=buckets,
            Y_train=torch.empty(
                (0, self.config.embedding_dimension), dtype=self.dtype, device=self.device
            ),
            X_nominal_symmetric=torch.empty(
                (0, self.action_dimension), dtype=self.dtype, device=self.device
            ),
            y_train=torch.empty((0,), dtype=self.dtype, device=self.device),
            last_ask_Y=None,
            last_ask_X=None,
        )


__all__ = ["HeSBO", "HeSBOConfig", "HeSBOState"]
