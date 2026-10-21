"""A single plotting interface for Basat experiments and ordinary numeric data."""

from __future__ import annotations

import colorsys
import io
import os
import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from .warnings import BasatWarning
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


def _set_figure_font(figure: Any, family: str | None) -> None:
    """Style only this figure's text artists; never mutate global rcParams."""
    if family is None:
        return
    if not isinstance(family, str) or not family.strip():
        raise ValueError("font_family must be a nonempty font-family name or None.")
    from matplotlib.text import Text

    for artist in figure.findobj(match=Text):
        artist.set_fontfamily(family)


def _tick_positions(value: Any) -> np.ndarray | None:
    """Validate explicit ticks without changing the caller's axis limits."""
    if value is None:
        return None
    positions = np.asarray(to_list(value), dtype=float)
    if (positions.ndim != 1 or not np.isfinite(positions).all()
            or np.any(np.diff(positions) <= 0)):
        raise ValueError("xticks must be a finite increasing one-dimensional sequence.")
    return positions


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
    color, linestyle, marker, markerfacecolor : optional
        Source-paper styling. Colors are otherwise distinct and consistent
        across panels for the same label. ``markerfacecolor="none"`` draws
        open markers; the default preserves the plotting backend's fill.
    error_every : int or sequence of int
        Spacing, or explicit zero-based indices, for displayed error bars.
    error_style : {"bar", "band"}, default "bar"
        Draw symmetric error bars or a shaded band. A band uses all measured
        coordinates; ``error_every`` applies only to bars. On a log axis,
        nonpositive lower bounds are clipped for display only to an explicitly
        positive ``PlotPanel.ylim[0]``. Without that declared floor, invalid
        band segments are masked. Both cases issue a Basat warning and leave
        the measured values and half widths unchanged.
    error_alpha : float, default 0.2
        Shaded-band opacity, between zero and one. Bar opacity is unchanged.
    error_color, error_linewidth : optional
        Separate error-bar color (including an optional alpha channel) and
        line width. Defaults preserve the curve color and backend bar width.

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
    markerfacecolor: Any = None
    error_style: str = "bar"
    error_alpha: float = 0.2
    error_color: Any = None
    error_linewidth: float | None = None

    @classmethod
    def from_runs(
        cls, runs: Any, *, transform: str = "none", maximize: bool = True,
        optimum: float | None = None,
        uncertainty: str | None = "sd", ddof: int = 0,
        error_scale: float = 1.0, resample: int | None = None,
        smoothing: str = "none", decay_steps: float = 1.0, **style: Any,
    ) -> PlotSeries:
        """Aggregate equal-length measured runs, transforming each run first.

        ``transform`` is ``none``, ``incumbent``, ``simple_regret``,
        ``cumulative`` or ``running_mean``. Simple regret requires an explicit
        finite ``optimum``: optimum minus each run's running maximum when
        maximizing, or each run's running minimum minus optimum when minimizing.
        Regrets are computed before averaging and are never clipped, including
        exact zeros or negative values when observations exceed the reference.
        ``optimum`` is accepted only for this transform.

        Unequal lengths must be selected/truncated explicitly
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

        Examples
        --------
        >>> curve = PlotSeries.from_runs(
        ...     [[1., 3., 2.], [2., 2., 4.]], transform="simple_regret",
        ...     optimum=4., uncertainty="sem", ddof=0)
        >>> curve.values.tolist()
        [2.5, 1.5, 0.5]
        """
        values = np.asarray(to_list(runs), dtype=float)
        if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
            raise ValueError("runs must be a finite, nonempty rectangular array.")
        if transform == "simple_regret":
            reference = np.asarray(to_list(optimum))
            if (reference.ndim != 0 or reference.dtype.kind not in "iuf"
                    or not np.isfinite(reference).item()):
                raise ValueError("simple_regret requires a finite scalar optimum.")
            incumbent = (np.maximum if maximize else np.minimum).accumulate(values, axis=1)
            values = float(reference) - incumbent if maximize else incumbent - float(reference)
            if not np.isfinite(values).all():
                raise ValueError("simple_regret values must be finite.")
        elif optimum is not None:
            raise ValueError("optimum is only supported with transform='simple_regret'.")
        elif transform == "incumbent":
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
    ``xticks`` optionally gives explicit x-axis tick positions. The default
    leaves tick placement to the plotting backend. ``legend_title`` and
    ``legend_location`` expose source-paper legend presentation without
    changing any series or statistics.

    For log-y line/scatter panels, nonpositive measured centers are masked
    with a Basat warning, leaving gaps rather than epsilon-valued points.
    Symmetric error bars touching or crossing zero are omitted in full with
    a warning. The underlying ``PlotSeries.values`` and ``error`` stay intact.

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
    legend_title: str | None = None
    legend_location: str = "best"
    horizontal_lines: tuple[float, ...] = ()
    tick_rotation: float = 0.0
    xticks: Any = None


def plot_panels(
    panels: Any, *, columns: int | None = None, title: str = "",
    figsize: tuple[float, float] | None = None, save_path: str | None = None,
    dpi: int = 160, show: bool = True,
    font_family: str | None = None,
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
    font_family : str, optional
        Font family for this figure only, for example "serif". Global plotting
        defaults are not changed.

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
                if series.error_style not in {"bar", "band"}:
                    raise ValueError("error_style must be 'bar' or 'band'.")
                if series.error_linewidth is not None and (
                    isinstance(series.error_linewidth, (bool, np.bool_))
                    or not isinstance(series.error_linewidth, Real)
                    or not np.isfinite(series.error_linewidth)
                    or series.error_linewidth <= 0
                ):
                    raise ValueError("error_linewidth must be positive and finite or None.")
                if (isinstance(series.error_alpha, bool) or not np.isfinite(series.error_alpha)
                        or not 0 <= series.error_alpha <= 1):
                    raise ValueError("error_alpha must be a finite number between zero and one.")
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
                nonpositive = y <= 0 if panel.yscale == "log" else np.zeros(y.shape, dtype=bool)
                display_y = np.where(nonpositive, np.nan, y) if nonpositive.any() else y
                omitted_bars = 0
                if panel.kind == "scatter":
                    if y.ndim != 1:
                        raise ValueError("Scatter values must be one dimensional.")
                    ax.scatter(x, display_y, color=color, alpha=series.alpha, label=label,
                               marker=series.marker or "o",
                               **({} if series.markerfacecolor is None else
                                  {"facecolors": series.markerfacecolor}))
                else:
                    lines = ax.plot(x, display_y, color=color, linestyle=series.linestyle,
                                    linewidth=series.linewidth, marker=series.marker,
                                    markerfacecolor=series.markerfacecolor,
                                    markevery=series.markevery, alpha=series.alpha)
                    lines[0].set_label(label)
                if series.error is not None:
                    error = np.asarray(to_list(series.error), dtype=float)
                    if y.ndim != 1 or error.shape != y.shape or not np.isfinite(error).all() \
                            or (error < 0).any():
                        raise ValueError(f"{label}: error must contain finite nonnegative half widths.")
                    if series.error_style == "band":
                        low, high = y-error, y+error
                        valid = np.ones(y.shape, dtype=bool)
                        if panel.yscale == "log":
                            valid &= y > 0
                            crossings = valid & (low <= 0)
                            floor = panel.ylim[0] if panel.ylim is not None else None
                            if floor is not None and np.isfinite(floor) and floor > 0:
                                low = np.where(crossings, floor, low)
                                policy = f"clipped for display only to declared y-axis floor {floor:g}"
                            else:
                                valid &= low > 0
                                policy = "masked because no positive y-axis floor was declared"
                            if crossings.any():
                                warnings.warn(
                                    f"Basat log-y band for {label!r}: {int(crossings.sum())} "
                                    f"nonpositive lower-bound coordinate(s) {policy}. "
                                    "Raw values and half widths are unchanged; no epsilon is added.",
                                    BasatWarning, stacklevel=2,
                                )
                        ax.fill_between(x, low, high, where=valid, color=color,
                                        alpha=series.error_alpha, linewidth=0)
                    else:
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
                        if panel.yscale == "log":
                            valid = (y[selected] > 0) & (y[selected] - error[selected] > 0)
                            omitted_bars = int(np.count_nonzero(~valid))
                            selected = selected[valid]
                        if len(selected):
                            ax.errorbar(x[selected], y[selected], yerr=error[selected], fmt="none",
                                        color=color, ecolor=series.error_color,
                                        elinewidth=series.error_linewidth,
                                        capsize=series.capsize,
                                        alpha=None if series.error_color is not None else series.alpha)
                if nonpositive.any() or omitted_bars:
                    warnings.warn(
                        f"Basat log-y display for {label!r}: masked "
                        f"{np.count_nonzero(nonpositive)} nonpositive measured value(s); omitted "
                        f"{omitted_bars} selected error bar(s) touching or crossing zero. "
                        "Raw values and half widths are unchanged; no epsilon clipping is applied.",
                        BasatWarning, stacklevel=2,
                    )
            for value in panel.horizontal_lines:
                if not np.isfinite(value):
                    raise ValueError("Reference lines must be finite.")
                ax.axhline(value, color="0.4", linestyle="--", linewidth=1)
            ax.set(title=panel.title, xlabel=panel.xlabel, ylabel=panel.ylabel,
                   xscale=panel.xscale)
            if panel.yscale == "log":
                ax.set_yscale("log", nonpositive="mask")
            else:
                ax.set_yscale(panel.yscale)
            ticks = _tick_positions(panel.xticks)
            if ticks is not None:
                ax.set_xticks(ticks)
            if panel.xlim is not None:
                ax.set_xlim(*panel.xlim)
            if panel.ylim is not None:
                ax.set_ylim(*panel.ylim)
            if panel.kind == "violin" and panel.xticks is None:
                ax.set_xticks(range(len(panel.series)), list(panel.series),
                              rotation=panel.tick_rotation,
                              ha="right" if panel.tick_rotation else "center")
            if panel.grid:
                ax.grid(True, alpha=0.2)
            else:
                ax.grid(False)
            if panel.legend and panel.series:
                if panel.legend_title is not None and not isinstance(panel.legend_title, str):
                    raise ValueError("legend_title must be a string or None.")
                if not isinstance(panel.legend_location, str) or not panel.legend_location:
                    raise ValueError("legend_location must be a non-empty string.")
                ax.legend(
                    fontsize=panel.legend_fontsize,
                    ncol=panel.legend_columns,
                    title=panel.legend_title,
                    loc=panel.legend_location,
                )
        for ax in list(axes.flat)[len(specs):]:
            ax.set_visible(False)
        if title:
            fig.suptitle(title)
        _set_figure_font(fig, font_family)
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
    xticks: Any = None,
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
    styles: Mapping[str, Mapping[str, Any]] | None = None,
    horizontal_lines: tuple[float, ...] = (),
    font_family: str | None = None,
    error_scale: float = 0.5,
    save_path: str | None = None,
    dpi: int = 160,
    show: bool = True,
) -> PlotResult:
    """Plot comparison/progress, incumbent, line, scatter, or histogram data.

    Pass an Experiment, Algorithm, label-to-values mapping, or
    :class:`SuccessStatistics`. Success statistics use a single success-fraction
    panel, with ``error_scale`` times the population SD across group fractions
    (default SD/2). ``styles`` maps labels to PlotSeries style options for
    success curves, or color/linestyle/linewidth/marker/markevery/
    markerfacecolor/alpha for ordinary line and trial plots. Scatter supports
    color/marker/alpha, and histograms color/alpha. Styles cannot replace
    measured data. No algorithm-specific styling is embedded in the library.
    ``horizontal_lines`` gives finite y coordinates for gray dashed references.
    ``font_family="serif"`` styles this figure's text without global side effects.
    Partial success
    summaries always show their completed/planned group count in the title.
    ``xticks`` optionally specifies x-axis tick positions; None preserves
    the backend's automatic tick locator. It does not resample measurements.
    Comparison plots show observed and best-so-far rewards with the algorithm's
    maximization/minimization direction. Axis limits do not fabricate observations
    after early stopping. ``show=False`` supports scripts and batch execution.
    """
    from .success import SuccessStatistics

    reference_lines = np.asarray(horizontal_lines, dtype=float)
    if reference_lines.ndim != 1 or not np.isfinite(reference_lines).all():
        raise ValueError("horizontal_lines must be a finite one-dimensional sequence.")

    if isinstance(data, SuccessStatistics):
        if uncertainty not in ("std95", "std", None):
            raise ValueError("uncertainty must be 'std95', 'std', or None; success curves use error_scale times SD.")
        if kind not in {"comparison", "line", "success"}:
            raise ValueError("Success statistics support kind='success' or 'line'.")
        if x is not None and tuple(to_list(x)) != data.queries:
            raise ValueError("Success plot x must match the statistics query grid.")
        if not np.isfinite(error_scale) or error_scale < 0:
            raise ValueError("error_scale must be finite and nonnegative.")
        supplied = {} if styles is None else dict(styles)
        if set(supplied) - set(data.methods):
            raise ValueError("Success plot styles contain an unknown method label.")
        curves = {}
        for label, curve in data.methods.items():
            style = dict(supplied.get(label, {}))
            if set(style) & {"values", "x", "error"}:
                raise ValueError("Style options cannot replace success measurements or uncertainty.")
            defaults: dict[str, Any] = {"linewidth": linewidth, "marker": marker, "capsize": capsize}
            defaults.update(style)
            curves[label] = PlotSeries(
                values=curve.mean, x=data.queries,
                error=None if uncertainty is None else np.asarray(curve.std)*error_scale,
                **defaults,
            )
        heading = "Success fraction" if title == "Basat experiment" else title
        if not data.complete:
            heading += f" (partial: {len(data.included_groups)}/{data.group_count} groups)"
        if data.source_exclude_exact_budget:
            heading += f"\nDisplay omits hits at query {data.max_evaluations}"
        if titles is not None and len(titles) != 1:
            raise ValueError("Success statistics require exactly one panel title.")
        return plot_panels([PlotPanel(
            series=curves, title="" if titles is None else titles[0],
            xlabel="Queries" if xlabel == "Evaluation" else xlabel,
            ylabel="Success fraction" if ylabel == "Reward" else ylabel,
            xlim=(0, data.max_evaluations) if xlim is None else xlim,
            ylim=(0, 1) if ylim is None else ylim, xscale=xscale, yscale=yscale,
            grid=grid, legend=legend, legend_columns=legend_columns,
            legend_fontsize=legend_fontsize, xticks=xticks,
            horizontal_lines=tuple(reference_lines),
        )], title=heading, figsize=figsize or (6., 4.5), save_path=save_path,
            dpi=dpi, show=show, font_family=font_family)

    ticks = _tick_positions(xticks)
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
    supplied = {} if styles is None else {label: dict(style) for label, style in styles.items()}
    if set(supplied) - set(labels):
        raise ValueError("Plot styles contain an unknown method label.")
    allowed = ({"color", "alpha"} if kind == "histogram" else
               {"color", "alpha", "marker"} if kind == "scatter" else
               {"color", "alpha", "linestyle", "linewidth", "marker", "markevery", "markerfacecolor"})
    if any(set(style) - allowed for style in supplied.values()):
        raise ValueError(f"Styles for {kind} support only {sorted(allowed)}; measurements cannot be overridden.")
    rows = (len(labels) + legend_columns - 1) // legend_columns
    size = figsize or ((13.5, 4.8 + rows * 0.25) if panels == 2 else (8.5, 4.8 + rows * 0.25))
    fig, axs = plt.subplots(1, panels, figsize=size, squeeze=False)
    axes = axs[0]
    colors = _palette(len(labels))
    line_styles = ("-", "--", "-.", ":")
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
            style = dict(supplied.get(label, {}))
            color = style.get("color", colors[i])
            settings = dict(label=label, color=color)
            if kind == "histogram":
                _, _, patches = ax.hist(y, alpha=style.get("alpha", 0.6), **settings)
                handle = patches[0]
            elif kind == "scatter":
                handle = ax.scatter(coordinates, y, **settings,
                                    **{k: v for k, v in style.items() if k != "color"})
            else:
                (handle,) = ax.plot(
                    coordinates,
                    y,
                    linestyle=style.get("linestyle", line_styles[(i // 10) % 4]),
                    linewidth=style.get("linewidth", linewidth),
                    marker=style.get("marker", marker),
                    markevery=style.get("markevery", max(1, len(y) // 15)),
                    markerfacecolor=style.get("markerfacecolor"),
                    alpha=style.get("alpha"),
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
                    color=color,
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
            handles.append(Line2D([], [], color=supplied.get(label, {}).get("color", "gray"),
                                  marker="x", linestyle="none"))
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
        for value in reference_lines:
            ax.axhline(float(value), color="0.4", linestyle="--", linewidth=1)
        ax.set(title=heading, xlabel=xlabel, ylabel=ylabel, xscale=xscale, yscale=yscale)
        if ticks is not None:
            ax.set_xticks(ticks)
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
    _set_figure_font(fig, font_family)
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
    _set_figure_font(fig, font_family)
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
