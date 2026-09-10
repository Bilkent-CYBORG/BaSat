"""A single plotting interface for BayOPy experiments and ordinary numeric data."""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .workflow import to_list


@dataclass
class PlotResult:
    """Displayable plot with optional saved path; no plotting backend API required."""

    figure: Any
    axes: Any
    path: str | None = None

    def _repr_png_(self) -> bytes:
        stream = io.BytesIO()
        self.figure.savefig(stream, format="png", dpi=140, bbox_inches="tight")
        return stream.getvalue()


def show(value: Any) -> None:
    """Display a BayOPy plot in a notebook, or in a desktop plot window."""
    import matplotlib.pyplot as plt
    from IPython import get_ipython

    if get_ipython() is not None:
        from IPython.display import display

        display(value)
    elif isinstance(value, PlotResult):
        value.figure.show()
        plt.show()
    else:
        print(value)


def _series(data: Any) -> dict[str, tuple[np.ndarray, bool, float | None]]:
    trial_stats = getattr(data, "statistics", None)
    if trial_stats is None and hasattr(data, "trial_result"):
        trial_stats = data.trial_result.statistics
    if trial_stats is not None:
        return {k: (np.asarray(v.mean), v.maximize, None) for k, v in trial_stats.items() if v.mean}
    if hasattr(data, "histories"):
        return {k: (np.asarray(v, dtype=float), data.maximize, None) for k, v in data.histories.items()}
    if hasattr(data, "algorithms"):
        # Failed initialization and zero-budget children have no observations.
        # Their failure entries are added to the legend by plot() below.
        data = {
            label: algorithm
            for label, algorithm in data.algorithms.items()
            if algorithm.observed_values.numel() > 0
        }
    elif hasattr(data, "experiments") and data.experiments:
        data = data.experiments
    if not isinstance(data, Mapping):
        data = {getattr(data, "name", type(data).__name__): data}
    result = {}
    for label, item in data.items():
        algorithm = getattr(item, "algorithm", item)
        values = getattr(algorithm, "observed_values", algorithm)
        values = np.asarray(to_list(values), dtype=float).reshape(-1)
        maximize = getattr(algorithm, "maximize", True)
        initial_best = None
        if hasattr(algorithm, "observed_values") and hasattr(algorithm, "evaluations"):
            offset = max(0, len(values) - algorithm.evaluations)
            if offset:
                initial_best = float(np.max(values[:offset]) if maximize else np.min(values[:offset]))
                values = values[offset:]
            if not len(values):
                continue
        result[str(label)] = (values, maximize, initial_best)
    return result


def plot(
    data: Any,
    *,
    kind: str = "comparison",
    x: Any = None,
    title: str = "BayOPy experiment",
    titles: tuple[str, ...] | None = None,
    xlabel: str = "Evaluation",
    ylabel: str = "Reward",
    figsize: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    grid: bool = True,
    legend: bool = True,
    legend_columns: int = 4,
    legend_fontsize: float = 8.0,
    linewidth: float = 1.6,
    marker: str | None = None,
    uncertainty: str | None = "std95",
    errorbar_cycles: int = 5,
    capsize: float = 2.5,
    save_path: str | None = None,
    dpi: int = 160,
    show: bool = True,
) -> PlotResult:
    """Plot comparison/progress, incumbent, line, scatter, or histogram data.

    Pass an Experiment, Algorithm, or label-to-values mapping.
    Comparison plots show observed and best-so-far rewards with the algorithm's
    maximization/minimization direction. Axis limits do not fabricate observations
    after early stopping. ``show=False`` supports scripts and batch execution.
    """
    import matplotlib.pyplot as plt

    if kind not in {"comparison", "progress", "incumbent", "line", "scatter", "histogram"}:
        raise ValueError("Unknown plot kind.")
    trial_result = getattr(data, "trial_result", data)
    trial_stats = getattr(trial_result, "statistics", None)
    if uncertainty not in ("std95", "std", None):
        raise ValueError("uncertainty must be 'std95', 'std', or None.")
    if isinstance(errorbar_cycles, bool) or not isinstance(errorbar_cycles, int) or errorbar_cycles < 1:
        raise ValueError("errorbar_cycles must be a positive integer.")
    if trial_stats is not None and kind in {"histogram", "scatter"}:
        raise ValueError("Trial aggregates support line, incumbent, progress and comparison plots.")
    series = _series(data)
    failures = getattr(data, "failures", {})
    if trial_stats is not None:
        failures = {label: True for label, runs in trial_result.results.items() if any(r is None for r in runs)}
    if (not series and not failures) or any(
        not len(v) or not np.isfinite(v).all() for v, _, _ in series.values()
    ):
        raise ValueError("Plot series must contain finite, nonempty numeric values.")
    panels = 2 if kind in {"comparison", "progress"} else 1
    if titles is not None and len(titles) != panels:
        raise ValueError("titles must contain one title per panel.")
    if not np.isfinite(legend_fontsize) or legend_fontsize <= 0:
        raise ValueError("legend_fontsize must be finite and positive.")
    if legend_columns < 1:
        raise ValueError("legend_columns must be positive.")
    labels = list(dict.fromkeys((*series, *failures)))
    rows = (len(labels) + legend_columns - 1) // legend_columns
    size = figsize or ((13.5, 4.8 + rows * 0.25) if panels == 2 else (8.5, 4.8 + rows * 0.25))
    fig, axs = plt.subplots(1, panels, figsize=size, squeeze=False)
    axes = axs[0]
    colors = plt.get_cmap("tab20")
    styles = ("-", "--", "-.", ":")
    handles = []
    horizon = max((len(v) for v, _, _ in series.values()), default=1)
    if xlim is not None and np.isfinite(xlim[1]):
        horizon = max(horizon, int(np.ceil(xlim[1])))
    method_count = max(1, len(labels))
    interval = max(method_count, int(np.ceil(horizon / (errorbar_cycles * method_count))) * method_count)
    for i, (label, (values, maximize, initial_best)) in enumerate(series.items()):
        coordinates = np.arange(1, len(values) + 1) if x is None else np.asarray(to_list(x))
        if len(coordinates) != len(values):
            raise ValueError("x and each series must have the same length.")
        incumbent = np.maximum.accumulate(values) if maximize else np.minimum.accumulate(values)
        if initial_best is not None:
            incumbent = (np.maximum if maximize else np.minimum)(incumbent, initial_best)
        for j, ax in enumerate(axes):
            is_incumbent = j == 1 or kind == "incumbent"
            y = incumbent if is_incumbent else values
            if trial_stats is not None:
                stats = trial_stats[label]
                y = np.asarray(stats.incumbent_mean if is_incumbent else stats.mean)
            settings = dict(label=label, color=colors(i % 20))
            if kind == "histogram":
                _, _, patches = ax.hist(y, alpha=0.6, **settings)
                handle = patches[0]
            elif kind == "scatter":
                handle = ax.scatter(coordinates, y, **settings)
            else:
                (handle,) = ax.plot(
                    coordinates,
                    y,
                    linestyle=styles[(i // 10) % 4],
                    linewidth=linewidth,
                    marker=marker,
                    markevery=max(1, len(y) // 15),
                    **settings,
                )
            if trial_stats is not None and uncertainty is not None:
                spread = np.asarray([np.nan if v is None else v for v in
                                     (stats.incumbent_std if is_incumbent else stats.std)])
                positions = np.arange((i * interval) // method_count, len(y), interval, dtype=int)
                positions = positions[np.isfinite(spread[positions])]
                ax.errorbar(coordinates[positions], y[positions],
                            yerr=spread[positions] * (1.96 if uncertainty == "std95" else 1.0),
                            fmt="none", color=colors(i % 20), elinewidth=max(0.6, linewidth * 0.6),
                            capsize=capsize, alpha=0.8)
            if j == 0:
                handles.append(handle)
    from matplotlib.lines import Line2D

    legend_labels = [
        f"{label} [FAILED; partial]" if label in failures else label for label in series
    ]
    for label in failures:
        if label not in series:
            handles.append(Line2D([], [], color="gray", marker="x", linestyle="none"))
            legend_labels.append(f"{label} [FAILED]")
    if not series:
        axes[0].text(
            0.5,
            0.5,
            "No successful observations; see diagnostics.",
            ha="center",
            transform=axes[0].transAxes,
        )
    headings = titles or (
        ("Observed reward", "Best observed reward") if panels == 2
        else ("" if trial_stats is not None else title,)
    )
    for ax, heading in zip(axes, headings, strict=True):
        ax.set(title=heading, xlabel=xlabel, ylabel=ylabel, xscale=xscale, yscale=yscale)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(grid, alpha=0.2)
    if trial_stats is not None:
        note = "Mean across trials" if uncertainty is None else (
            "Mean ± 1.96 sample SD across trials (not a confidence interval)"
            if uncertainty == "std95" else "Mean ± sample SD across trials")
        fig.suptitle(title + "\n" + note, fontsize=11)
    elif panels == 2:
        fig.suptitle(title)
    bottom = min(0.35, 0.07 + 0.043 * rows) if legend else 0.03
    fig.tight_layout(rect=(0, bottom, 1, 0.91 if trial_stats is not None else (0.94 if panels == 2 else 1)))
    if legend:
        fig.legend(handles, legend_labels, loc="lower center", ncol=legend_columns, fontsize=legend_fontsize)
    path = None
    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=dpi, bbox_inches="tight")
        path = str(target)
    result = PlotResult(fig, axes, path)
    if show:
        globals()["show"](result)
    plt.close(fig)
    return result
