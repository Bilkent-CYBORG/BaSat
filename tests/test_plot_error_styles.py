"""Independent error-bar styling preserves the curve and existing defaults."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.container import ErrorbarContainer

from basat.plotting import PlotPanel, PlotSeries, plot_panels

matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def render(series):
    visual = plot_panels([PlotPanel({"measured": series})], show=False)
    visual.figure.canvas.draw()
    axis = visual.axes.flat[0]
    containers = [item for item in axis.containers if isinstance(item, ErrorbarContainer)]
    assert len(containers) == 1
    _, caps, bars = containers[0].lines
    assert len(bars) == 1
    curve = next(line for line in axis.lines if line.get_label() == "measured")
    return curve, caps, bars[0]


def test_explicit_rgba_error_color_and_width_render_independently():
    color = (.1, .6, .3, .72)
    curve, caps, bars = render(PlotSeries(
        [1., 2., 3.], error=[.1, .2, .3], color="#9933cc", alpha=.25,
        linewidth=5., error_color=color, error_linewidth=2.75, capsize=4.,
    ))
    assert to_rgba(curve.get_color(), curve.get_alpha()) == to_rgba("#9933cc", .25)
    assert curve.get_linewidth() == 5.
    np.testing.assert_allclose(bars.get_colors(), [color])
    np.testing.assert_allclose(bars.get_linewidths(), [2.75])
    assert caps
    for cap in caps:
        assert to_rgba(cap.get_markeredgecolor(), cap.get_alpha()) == color


def test_defaults_preserve_curve_color_alpha_and_backend_bar_width():
    # The previous bar width came from Matplotlib, independently of curve width.
    with matplotlib.rc_context({"lines.linewidth": 3.25}):
        curve, caps, bars = render(PlotSeries(
            [1., 2.], error=[.2, .3], color="#cc4422", alpha=.4, linewidth=6.,
        ))
    expected = to_rgba(curve.get_color(), .4)
    np.testing.assert_allclose(bars.get_colors(), [expected])
    np.testing.assert_allclose(bars.get_linewidths(), [3.25])
    assert curve.get_linewidth() == 6.
    for cap in caps:
        assert to_rgba(cap.get_markeredgecolor(), cap.get_alpha()) == expected


def test_width_only_retains_curve_alpha_and_automatic_color():
    curve, caps, bars = render(PlotSeries(
        [1., 2.], error=[.1, .2], alpha=.35, error_linewidth=np.float64(1.125),
    ))
    expected = to_rgba(curve.get_color(), .35)
    np.testing.assert_allclose(bars.get_colors(), [expected])
    np.testing.assert_allclose(bars.get_linewidths(), [1.125])
    for cap in caps:
        assert to_rgba(cap.get_markeredgecolor(), cap.get_alpha()) == expected


def test_explicit_named_error_color_has_its_own_opacity():
    curve, caps, bars = render(PlotSeries(
        [1., 2.], error=[.1, .2], color="blue", alpha=.2, error_color="orange",
    ))
    assert curve.get_alpha() == .2
    np.testing.assert_allclose(bars.get_colors(), [to_rgba("orange")])
    for cap in caps:
        assert to_rgba(cap.get_markeredgecolor(), cap.get_alpha()) == to_rgba("orange")


def test_from_runs_forwards_styles_without_changing_statistics_or_bar_positions():
    runs = np.array([[1., 3., 5.], [3., 7., 9.]])
    original = runs.copy()
    series = PlotSeries.from_runs(
        runs, uncertainty="sd", ddof=0, error_scale=.5,
        x=[0., 25., 50.], error_every=[0, 2],
        color="blue", error_color=(1., 0., 0., .6), error_linewidth=2.,
    )
    _, _, bars = render(series)
    np.testing.assert_array_equal(runs, original)
    np.testing.assert_array_equal(series.values, [2., 5., 7.])
    np.testing.assert_array_equal(series.error, [.5, 1., 1.])
    np.testing.assert_array_equal(bars.get_segments(), [
        [[0., 1.5], [0., 2.5]], [[50., 6.], [50., 8.]],
    ])
    np.testing.assert_allclose(bars.get_colors(), [[1., 0., 0., .6]])
    np.testing.assert_allclose(bars.get_linewidths(), [2.])


def test_error_bar_style_options_do_not_restyle_uncertainty_bands():
    visual = plot_panels([PlotPanel({"measured": PlotSeries(
        [1., 2.], error=[.1, .2], error_style="band", error_alpha=.3,
        color="blue", error_color="red", error_linewidth=4.,
    )})], show=False)
    visual.figure.canvas.draw()
    axis = visual.axes.flat[0]
    assert not axis.containers
    assert len(axis.collections) == 1
    np.testing.assert_allclose(axis.collections[0].get_facecolors(), [to_rgba("blue", .3)])
    np.testing.assert_allclose(axis.collections[0].get_linewidths(), [0.])


@pytest.mark.parametrize("width", [
    0., -1., float("nan"), float("inf"), -float("inf"), True, False,
    np.bool_(True), np.bool_(False), [1.], [1., 2.], np.array([1., 2.]), "wide", 1j,
])
def test_invalid_error_linewidth_is_rejected_before_rendering(width):
    with pytest.raises(ValueError, match="error_linewidth"):
        render(PlotSeries([1., 2.], error=[.1, .2], error_linewidth=width))


@pytest.mark.parametrize("settings", [
    {}, {"error": [.1, .2], "error_style": "band"},
    {"error": [.1, .2], "error_every": np.array([], dtype=int)},
])
def test_invalid_width_is_rejected_even_without_visible_error_bars(settings):
    with pytest.raises(ValueError, match="error_linewidth"):
        plot_panels([PlotPanel({"measured": PlotSeries(
            [1., 2.], error_linewidth=0., **settings,
        )})], show=False)
