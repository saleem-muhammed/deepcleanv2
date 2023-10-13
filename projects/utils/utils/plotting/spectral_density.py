from typing import Optional

import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Bright6 as palette
from bokeh.plotting import figure

from utils.plotting.utils import save


def plot_psds(
    freqs: np.ndarray,
    pred: np.ndarray,
    strain: np.ndarray,
    clean: np.ndarray,
    asd: bool = True,
    interval: float = 50,
    fname: Optional[str] = None,
):
    lines = dict(
        x=freqs, pred=pred, strain=strain, cleaned=clean, ratio=clean / strain
    )
    label_map = dict(
        strain="Raw strain",
        pred="Noise Prediction",
        cleaned="Cleaned",
        ratio="Ratio",
    )
    keys = ["strain", "pred", "cleaned", "ratio"]
    errs = dict(x=np.concatenate([freqs, freqs[::-1]]))

    percentiles = [(100 - interval) / 2, 50, (100 + interval) / 2]
    color_map = {}
    for i, key in enumerate(keys):
        color_map[key] = palette[i]
        x = lines[key]
        if x.ndim == 2:
            low, mid, high = np.percentile(x, percentiles, axis=0)
            lines[key] = mid
            errs[key] = np.concatenate([low, high[::-1]])

    source = ColumnDataSource(lines)
    err_source = ColumnDataSource(errs)

    kwargs = dict(
        height=250,
        width=700,
        y_axis_type="log",
        tools="box_zoom,save,reset",
    )

    title = "Spectral Densities"
    if len(errs) > 1:
        title += f" with {int(interval)}% confidence interval"
    mean_ratio = lines["ratio"].mean()
    title += f", mean ratio: {mean_ratio:0.3f}"
    y_axis_label = r"$$\text{{{}[Hz}}^{{-{}}}\text{{]}}$$".format(
        "ASD" if asd else "PSD", r"\frac{1}{2}" if asd else 1
    )
    p1 = figure(y_axis_label=y_axis_label, title=title, **kwargs)
    p1.xaxis.major_tick_line_color = None
    p1.xaxis.minor_tick_line_color = None
    p1.xaxis.major_label_text_font_size = "1pt"
    p1.xaxis.major_label_text_color = None

    def _plot_line(p, y):
        color = color_map[y]
        legend_label = label_map[y]
        if y in errs:
            p.patch(
                "x",
                y,
                line_color=color,
                line_width=0.5,
                fill_color=color,
                fill_alpha=0.5,
                legend_label=legend_label,
                source=err_source,
            )

        r = p.line(
            "x",
            y,
            line_color=color,
            line_width=1.5,
            line_alpha=0.8,
            legend_label=legend_label,
            source=source,
        )
        return r

    tooltips = [("Frequency", "@x Hz"), ("Ratio", "@ratio")]
    for key in keys[:-1]:
        r = _plot_line(p1, key)
        tooltips.append((label_map[key], f"@{key}"))
    hover = HoverTool(renderers=[r], tooltips=tooltips, mode="vline")
    p1.add_tools(hover)

    p2 = figure(
        x_range=p1.x_range,
        y_axis_label=r"$$\text{Ratio (Cleaned / Raw)}$$",
        x_axis_label=r"$$\text{Frequency [Hz]}$$",
        **kwargs,
    )
    _plot_line(p2, "ratio")
    p2.legend.location = "bottom_right"

    grid = gridplot([p1, p2], ncols=1, toolbar_location="right")
    if fname is not None:
        save(grid, fname, title="DeepClean PSDs")
    return grid
