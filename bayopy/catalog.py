"""Algorithm discovery and construction without experiment-specific defaults."""

from __future__ import annotations

import dataclasses
from typing import Any

from . import algorithms as methods
from .algorithms.algorithm import Algorithm
from .environments.adapters import adapt_environment_for_algorithm
from .models import AcquisitionConfig, KernelConfig
from .workflow import get_seed

ALGORITHM_FAMILIES = {
    "stochastic": (
        "REMBO",
        "HeSBO",
        "ALEBO",
        "ADD_GP_UCB",
        "TuRBO",
        "Dropout",
        "MCTS_VS",
        "SAASBO",
        "AG_VS",
        "LA_MCTS",
        "PG",
        "EG",
        "GS",
        "Satisficing_TS",
    ),
    "contextual": ("CGP_UCB",),
    "distributional": ("DRBO", "RoBOS"),
    "adversarial": ("StableOPT", "AdveRS"),
}


def algorithm_names(family: str | None = None) -> tuple[str, ...]:
    """List public methods by implementation family (StableOPT is robust/adversarial)."""
    if family is None:
        return tuple(name for names in ALGORITHM_FAMILIES.values() for name in names)
    if family not in ALGORITHM_FAMILIES:
        raise ValueError(f"Unknown algorithm family {family!r}.")
    return ALGORITHM_FAMILIES[family]


def make_algorithm(
    name: str,
    *,
    bounds: Any,
    contexts: Any = None,
    seed: int | None = None,
    kernel: str | KernelConfig | None = None,
    maximize: bool = True,
    dtype: str = "float64",
    device: str = "cpu",
    plot: bool = False,
    verbose: bool = False,
    verbose_setting: bool = False,
    verbose_iter_by_iter: bool = False,
    **settings: Any,
) -> Algorithm:
    """Construct one algorithm using its configuration-class defaults.

    Required scientific choices (for example effective dimension, threshold,
    context support or a decomposition) must be supplied by the caller. Omitted
    optional fields retain their dataclass defaults. This factory does not choose
    workloads, objectives, parameter bundles, or evaluation budgets.
    """
    if name not in algorithm_names():
        raise ValueError(f"Unknown algorithm {name!r}.")
    if "profile" in settings:
        raise TypeError("Parameter profiles were removed; supply configuration fields explicitly.")
    classes = {
        "REMBO": methods.REMBOConfig,
        "HeSBO": methods.HeSBOConfig,
        "ALEBO": methods.ALEBOConfig,
        "ADD_GP_UCB": methods.AddGPUCBConfig,
        "TuRBO": methods.TuRBOConfig,
        "Dropout": methods.DropoutConfig,
        "MCTS_VS": methods.MCTSVSConfig,
        "SAASBO": methods.SAASBOConfig,
        "AG_VS": methods.AG_VSConfig,
        "LA_MCTS": methods.LAMCTSConfig,
        "StableOPT": methods.StableOPTConfig,
        "AdveRS": methods.AdveRSConfig,
        "CGP_UCB": methods.CGPUCBConfig,
        "DRBO": methods.DRBOConfig,
        "RoBOS": methods.RoBOSConfig,
        **{n: methods.GoodActionConfig for n in ("PG", "EG", "GS", "Satisficing_TS")},
    }
    config_class = classes[name]
    fields = {f.name for f in dataclasses.fields(config_class)}
    values = dict(settings)
    values["seed"] = get_seed() if seed is None else seed
    if contexts is not None:
        if "contexts" not in fields:
            raise TypeError(f"{name} does not have a contexts configuration field.")
        values["contexts"] = contexts
    if kernel is not None:
        key = "kernel" if "kernel" in fields else "action_kernel"
        if key not in fields:
            raise ValueError(f"{name} does not expose a generic kernel override.")
        values[key] = KernelConfig(name=kernel) if isinstance(kernel, str) else kernel
    acquisition = values.get("acquisition")
    if isinstance(acquisition, str) and (
        name in {"HeSBO", "ALEBO", "AG_VS"} or (name == "TuRBO" and acquisition != "TS")
    ):
        values["acquisition"] = AcquisitionConfig(name=acquisition, seed=values["seed"])
    return getattr(methods, name)(
        bounds=bounds,
        config=config_class(**values),
        maximize=maximize,
        dtype=dtype,
        device=device,
        plot=plot,
        verbose=verbose,
        verbose_setting=verbose_setting,
        verbose_iter_by_iter=verbose_iter_by_iter,
    )


def adapt_environment(environment: Any, algorithm: Any) -> Any:
    """Return the same automatic runtime-default view used by Experiment.

    Calling this helper is optional. The source owns observations and all
    environment-provided inputs take precedence over missing-input defaults.
    """
    return adapt_environment_for_algorithm(environment, algorithm)
