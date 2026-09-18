"""A single plotting interface for Basat experiments and ordinary numeric data."""

from __future__ import annotations

import colorsys
import io
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
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

    def save(self, path: str, *, dpi: int = 160) -> str:
        """Save atomically without importing or calling a plotting backend.

        Examples
        --------
        >>> chart = basat.plot({"reward": [1., 2.]}, kind="line", show=False)
        >>> chart.save("results/reward.png")
        'results/reward.png'
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=target.parent, suffix=target.suffix, delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            self.figure.savefig(temporary, dpi=dpi, bbox_inches="tight")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        self.path = str(target)
        return self.path


def _palette(count: int) -> list[Any]:
    """Distinct colors for any finite comparison, without a modulo-20 cycle."""
    import matplotlib.pyplot as plt

    if count <= 10:
        return [plt.get_cmap("tab10")(i) for i in range(count)]
    if count <= 20:
        return [plt.get_cmap("tab20")(i) for i in range(count)]
    return [colorsys.hls_to_rgb((i * 0.618033988749895) % 1, 0.46, 0.72)
            for i in range(count)]


def _symmetric_ema(x: np.ndarray, y: np.ndarray, grid: np.ndarray,
                   decay_steps: float) -> np.ndarray:
    """Resample using the two one-sided EMA measures in MCTS-VS plot_util.

    This retains that published plotting convention, including double weight
    for a sample exactly on a grid point. It is an optional display transform,
    not a modification of the stored measurements. Reference: MCTS-VS commit
    29d32b6, plot_tool/plot_util.py (one_sided_ema and symmetric_ema).
    """
    period = (grid[-1] - grid[0]) / (len(grid) - 1) * decay_steps
    fade = np.exp(-1.0 / decay_steps)

    def sweep(coordinates, samples, targets):
        sums = np.empty(len(targets))
        counts = np.empty(len(targets))
        total = mass = 0.0
        index = 0
        for step, target in enumerate(targets):
            total *= fade
            mass *= fade
            while index < len(coordinates) and coordinates[index] <= target:
                weight = np.exp(-(target - coordinates[index]) / period)
                total += weight * samples[index]
                mass += weight
                index += 1
            sums[step], counts[step] = total, mass
        return sums, counts

    left_sum, left_count = sweep(x, y, grid)
    right_sum, right_count = sweep(-x[::-1], y[::-1], -grid[::-1])
    mass = left_count + right_count[::-1]
    if np.any(mass < 1e-8):
        raise ValueError("The EMA grid contains points with insufficient observation weight.")
    return (left_sum + right_sum[::-1]) / mass


@dataclass(frozen=True)
class PlotSeries:
    """One measured curve or a collection of final observations.

    Parameters
    ----------
    values : array-like
        Actual y values. A two-dimensional array draws one line per column.
        For a violin panel these are the individual final observations.
    x : array-like, optional
        Explicit coordinates; otherwise use 1, ..., number of rows.
    error : array-like, optional
        Nonnegative *half widths*, already in the plotted metric's units.
        No confidence-interval or standard-error conversion is implicit.
    color, linestyle, marker : optional
        Source-paper styling. Colors are otherwise distinct and consistent
        across panels for the same label.
    error_every : int or sequence of int
        Spacing, or explicit zero-based indices, for displayed error bars.

    Examples
    --------
    >>> curve = basat.PlotSeries.from_runs(
    ...     [[1., 3.], [2., 4.]], uncertainty="sd", ddof=0, error_scale=0.5)
    >>> basat.to_list(curve.values)
    [1.5, 3.5]
    """

    values: Any
    x: Any = None
    error: Any = None
    color: Any = None
    linestyle: str = "-"
    marker: str | None = None
    linewidth: float = 1.6
    alpha: float = 1.0
    error_every: Any = 1
    markevery: Any = None
    capsize: float = 2.5

    @classmethod
    def from_runs(
        cls, runs: Any, *, transform: str = "none", maximize: bool = True,
        uncertainty: str | None = "sd", ddof: int = 0,
        error_scale: float = 1.0, resample: int | None = None,
        smoothing: str = "none", decay_steps: float = 1.0, **style: Any,
    ) -> PlotSeries:
        """Aggregate equal-length measured runs, transforming each run first.

        ``transform`` is ``none``, ``incumbent``, ``cumulative`` or
        ``running_mean``. Unequal lengths must be selected/truncated explicitly
        by the caller; missing endpoints are never fabricated. ``sd`` denotes
        standard deviation and ``sem`` divides it by sqrt(number of runs).
        For the papers' SD/2 bars use ``ddof=0, error_scale=0.5``.

        ``resample`` optionally creates a uniformly spaced display grid over
        the supplied ``x`` range (or evaluations 1 through N). With
        ``smoothing="symmetric_ema"``, each run is smoothed before aggregation,
        using ``decay_steps`` in units of the new grid spacing. For the Song
        2022 plotting convention use ``resample=512, decay_steps=1.0,
        uncertainty="sem", ddof=0``. The default performs no resampling or
        smoothing. Measurements are never extrapolated or changed in place.
        """
        values = np.asarray(to_list(runs), dtype=float)
        if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
            raise ValueError("runs must be a finite, nonempty rectangular array.")
        if transform == "incumbent":
            values = (np.maximum if maximize else np.minimum).accumulate(values, axis=1)
        elif transform == "cumulative":
            values = np.cumsum(values, axis=1)
        elif transform == "running_mean":
            values = np.cumsum(values, axis=1) / np.arange(1, values.shape[1] + 1)
        elif transform != "none":
            raise ValueError("Unknown per-run metric transform.")
        if uncertainty not in {None, "sd", "sem"}:
            raise ValueError("uncertainty must be 'sd', 'sem', or None.")
        if isinstance(ddof, bool) or not isinstance(ddof, int) or ddof < 0:
            raise ValueError("ddof must be a nonnegative integer.")
        if not np.isfinite(error_scale) or error_scale < 0:
            raise ValueError("error_scale must be finite and nonnegative.")
        if smoothing not in {"none", "symmetric_ema"}:
            raise ValueError("smoothing must be 'none' or 'symmetric_ema'.")
        if not np.isfinite(decay_steps) or decay_steps <= 0:
            raise ValueError("decay_steps must be positive and finite.")
        if resample is None and smoothing != "none":
            raise ValueError("Smoothing requires an explicit resample grid size.")
        if resample is not None:
            if isinstance(resample, bool) or not isinstance(resample, int) or resample < 2:
                raise ValueError("resample must be an integer of at least two.")
            x = np.asarray(to_list(style.get("x", np.arange(1, values.shape[1] + 1))),
                           dtype=float)
            if (x.shape != (values.shape[1],) or len(x) < 2
                    or not np.isfinite(x).all() or np.any(np.diff(x) <= 0)):
                raise ValueError("Resampling requires finite strictly increasing x coordinates.")
            grid = np.linspace(x[0], x[-1], resample)
            values = np.asarray([
                _symmetric_ema(x, row, grid, decay_steps)
                if smoothing == "symmetric_ema" else np.interp(grid, x, row)
                for row in values
            ])
            style["x"] = grid
        error = None
        if uncertainty is not None:
            if ddof >= values.shape[0]:
                raise ValueError("ddof must be smaller than the number of runs.")
            error = np.std(values, axis=0, ddof=ddof) * error_scale
            if uncertainty == "sem":
                error /= np.sqrt(values.shape[0])
        return cls(values=values.mean(axis=0), error=error, **style)


@dataclass(frozen=True)
class PlotPanel:
    """A backend-independent scientific subplot specification.

    ``kind`` is ``line``, ``scatter`` or ``violin``. Violin panels also display
    every supplied observation; constant/singleton samples use points alone.
    Axis ranges and error bars belong to the caller's scientific protocol.

    Examples
    --------
    >>> panel = basat.PlotPanel(
    ...     {"method A": basat.PlotSeries([1., 2.], x=[10, 20])},
    ...     xlabel="Queries", ylabel="Best reward")
    >>> chart = basat.plot_panels([panel], show=False)
    """

    series: Mapping[str, Any] = field(default_factory=dict)
    kind: str = "line"
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    xlim: tuple[float | None, float | None] | None = None
    ylim: tuple[float | None, float | None] | None = None
    xscale: str = "linear"
    yscale: str = "linear"
    grid: bool = True
    legend: bool = True
    legend_columns: int = 1
    legend_fontsize: float = 8.0
    horizontal_lines: tuple[float, ...] = ()
    tick_rotation: float = 0.0


def plot_panels(
    panels: Any, *, columns: int | None = None, title: str = "",
    figsize: tuple[float, float] | None = None, save_path: str | None = None,
    dpi: int = 160, show: bool = True,
) -> PlotResult:
    """Draw scientific panels entirely through Basat's public plotting API.

    Parameters
    ----------
    panels : sequence of PlotPanel
        Explicit curve, error-bar or distribution specifications. Input data
        and statistics are not altered to match a reference figure.
    columns : int, optional
        Number of panel columns; defaults to all panels in one row.
    title : str
        Overall figure title.
    save_path : str, optional
        Destination in the user's project, saved by atomic replacement.
    show : bool
        Display in a notebook/window. False supports background execution.

    Returns
    -------
    PlotResult
        Displayable figure with a Basat-owned ``save`` method.
    """
    import matplotlib.pyplot as plt

    specs = list(panels)
    if not specs or any(not isinstance(p, PlotPanel) for p in specs):
        raise ValueError("panels must contain one or more PlotPanel objects.")
    columns = len(specs) if columns is None else columns
    if isinstance(columns, bool) or not isinstance(columns, int) or columns < 1:
        raise ValueError("columns must be a positive integer.")
    labels = list(dict.fromkeys(label for panel in specs for label in panel.series))
    palette = dict(zip(labels, _palette(len(labels)), strict=True))
    rows = (len(specs) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, squeeze=False,
                            figsize=figsize or (6 * columns, 4.2 * rows),
                            layout="constrained")
    try:
        for ax, panel in zip(axes.flat, specs, strict=False):
            if panel.kind not in {"line", "scatter", "violin"}:
                raise ValueError("Panel kind must be 'line', 'scatter', or 'violin'.")
            for index, (label, entry) in enumerate(panel.series.items()):
                series = entry if isinstance(entry, PlotSeries) else PlotSeries(entry)
                y = np.asarray(to_list(series.values), dtype=float)
                if y.ndim not in {1, 2} or not y.size or not np.isfinite(y).all():
                    raise ValueError(f"{label}: values must be finite nonempty 1D/2D data.")
                color = palette[label] if series.color is None else series.color
                if panel.kind == "violin":
                    if y.ndim != 1 or series.error is not None:
                        raise ValueError("Violin data must be 1D individual observations.")
                    if len(y) > 1 and np.ptp(y) > 0:
                        parts = ax.violinplot([y], positions=[index], showextrema=True,
                                              showmedians=True)
                        for body in parts["bodies"]:
                            body.set_facecolor(color)
                        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
                            parts[key].set_color(color)
                    ax.scatter(np.full(len(y), index), y, color=color, alpha=series.alpha,
                               s=12, label=label)
                    continue
                x = (np.arange(1, len(y) + 1) if series.x is None
                     else np.asarray(to_list(series.x), dtype=float))
                if x.ndim != 1 or len(x) != len(y) or not np.isfinite(x).all():
                    raise ValueError(f"{label}: x must be finite and match the row count.")
                if panel.kind == "scatter":
                    if y.ndim != 1:
                        raise ValueError("Scatter values must be one dimensional.")
                    ax.scatter(x, y, color=color, alpha=series.alpha, label=label,
                               marker=series.marker or "o")
                else:
                    lines = ax.plot(x, y, color=color, linestyle=series.linestyle,
                                    linewidth=series.linewidth, marker=series.marker,
                                    markevery=series.markevery, alpha=series.alpha)
                    lines[0].set_label(label)
                if series.error is not None:
                    error = np.asarray(to_list(series.error), dtype=float)
                    if y.ndim != 1 or error.shape != y.shape or not np.isfinite(error).all() \
                            or (error < 0).any():
                        raise ValueError(f"{label}: error must contain finite nonnegative half widths.")
                    every = series.error_every
                    if isinstance(every, int) and not isinstance(every, bool) and every > 0:
                        selected = np.arange(0, len(y), every)
                    else:
                        raw = np.asarray(to_list(every))
                        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
                            raise ValueError("error_every must be positive or a list of integer indices.")
                        selected = raw.astype(int)
                        if ((selected < 0) | (selected >= len(y))).any():
                            raise ValueError("Error-bar index outside the measured curve.")
                    ax.errorbar(x[selected], y[selected], yerr=error[selected], fmt="none",
                                color=color, capsize=series.capsize, alpha=series.alpha)
            for value in panel.horizontal_lines:
                if not np.isfinite(value):
                    raise ValueError("Reference lines must be finite.")
                ax.axhline(value, color="0.4", linestyle="--", linewidth=1)
            ax.set(title=panel.title, xlabel=panel.xlabel, ylabel=panel.ylabel,
                   xscale=panel.xscale, yscale=panel.yscale)
            if panel.xlim is not None:
                ax.set_xlim(*panel.xlim)
            if panel.ylim is not None:
                ax.set_ylim(*panel.ylim)
            if panel.kind == "violin":
                ax.set_xticks(range(len(panel.series)), list(panel.series),
                              rotation=panel.tick_rotation,
                              ha="right" if panel.tick_rotation else "center")
            if panel.grid:
                ax.grid(True, alpha=0.2)
            else:
                ax.grid(False)
            if panel.legend and panel.series:
                ax.legend(fontsize=panel.legend_fontsize, ncol=panel.legend_columns)
        for ax in list(axes.flat)[len(specs):]:
            ax.set_visible(False)
        if title:
            fig.suptitle(title)
        result = PlotResult(fig, axes)
        if save_path is not None:
            result.save(save_path, dpi=dpi)
        if show:
            globals()["show"](result)
        return result
    finally:
        plt.close(fig)


def show(value: Any) -> None:
    """Display a Basat plot in a notebook, or in a desktop plot window."""
    import matplotlib.pyplot as plt
    from IPython import get_ipython

    if get_ipython() is not None:
        from IPython.display import display

        display(value)
    elif isinstance(value, PlotResult):
        if value.figure.canvas.required_interactive_framework is None:
            print("Basat plot created with a non-interactive backend; use chart.save(path).")
            return
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
        return {
            k: (np.asarray(v, dtype=float), data.maximize, None) for k, v in data.histories.items()
        }
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
                initial_best = float(
                    np.max(values[:offset]) if maximize else np.min(values[:offset])
                )
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
    title: str = "Basat experiment",
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
    if (
        isinstance(errorbar_cycles, bool)
        or not isinstance(errorbar_cycles, int)
        or errorbar_cycles < 1
    ):
        raise ValueError("errorbar_cycles must be a positive integer.")
    if trial_stats is not None and kind in {"histogram", "scatter"}:
        raise ValueError("Trial aggregates support line, incumbent, progress and comparison plots.")
    series = _series(data)
    failures = getattr(data, "failures", {})
    if trial_stats is not None:
        failures = {
            label: True
            for label, runs in trial_result.results.items()
            if any(r is None for r in runs)
        }
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
    colors = _palette(len(labels))
    styles = ("-", "--", "-.", ":")
    handles = []
    horizon = max((len(v) for v, _, _ in series.values()), default=1)
    if xlim is not None and np.isfinite(xlim[1]):
        horizon = max(horizon, int(np.ceil(xlim[1])))
    method_count = max(1, len(labels))
    interval = max(
        method_count, int(np.ceil(horizon / (errorbar_cycles * method_count))) * method_count
    )
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
            settings = dict(label=label, color=colors[i])
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
                spread = np.asarray(
                    [
                        np.nan if v is None else v
                        for v in (stats.incumbent_std if is_incumbent else stats.std)
                    ]
                )
                positions = np.arange((i * interval) // method_count, len(y), interval, dtype=int)
                positions = positions[np.isfinite(spread[positions])]
                ax.errorbar(
                    coordinates[positions],
                    y[positions],
                    yerr=spread[positions] * (1.96 if uncertainty == "std95" else 1.0),
                    fmt="none",
                    color=colors[i],
                    elinewidth=max(0.6, linewidth * 0.6),
                    capsize=capsize,
                    alpha=0.8,
                )
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
        ("Observed reward", "Best observed reward")
        if panels == 2
        else ("" if trial_stats is not None else title,)
    )
    for ax, heading in zip(axes, headings, strict=True):
        ax.set(title=heading, xlabel=xlabel, ylabel=ylabel, xscale=xscale, yscale=yscale)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if grid:
            ax.grid(True, alpha=0.2)
        else:
            ax.grid(False)
    if trial_stats is not None:
        note = (
            "Mean across trials"
            if uncertainty is None
            else (
                "Mean ± 1.96 sample SD across trials (not a confidence interval)"
                if uncertainty == "std95"
                else "Mean ± sample SD across trials"
            )
        )
        fig.suptitle(title + "\n" + note, fontsize=11)
    elif panels == 2:
        fig.suptitle(title)
    bottom = min(0.35, 0.07 + 0.043 * rows) if legend else 0.03
    fig.tight_layout(
        rect=(0, bottom, 1, 0.91 if trial_stats is not None else (0.94 if panels == 2 else 1))
    )
    if legend:
        fig.legend(
            handles,
            legend_labels,
            loc="lower center",
            ncol=legend_columns,
            fontsize=legend_fontsize,
        )
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
