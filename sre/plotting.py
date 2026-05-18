"""Publication-quality plotting for DynamicSpectrum objects."""
from __future__ import annotations

from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sre.processing import percentile_clip


PUBLICATION_RC = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 0.8,
}


def plot_spectrum(ds, cmap: str = "inferno", vmin: Optional[float] = None,
                  vmax: Optional[float] = None, vmin_pct: float = 1.0,
                  vmax_pct: float = 99.0, log_freq: bool = False,
                  title: Optional[str] = None,
                  xlabel: str = "Time (UTC)",
                  ylabel: str = "Frequency (MHz)",
                  cbar_label: Optional[str] = None,
                  figsize=(8.0, 4.0),
                  publication_style: bool = True,
                  rasterized: bool = True):
    """Return a (fig, ax, im) tuple. Caller is responsible for `savefig`."""
    rc = PUBLICATION_RC if publication_style else {}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        if vmin is None or vmax is None:
            v_lo, v_hi = percentile_clip(ds.data, vmin_pct, vmax_pct)
            vmin = v_lo if vmin is None else vmin
            vmax = v_hi if vmax is None else vmax

        t_mpl = mdates.date2num(pd.to_datetime(ds.times).to_pydatetime())
        f = ds.frequencies
        # pcolormesh wants cell edges; derive them from centres.
        t_edges = _edges_from_centres(t_mpl)
        f_edges = _edges_from_centres(f)

        im = ax.pcolormesh(
            t_edges, f_edges, ds.data,
            cmap=cmap, vmin=vmin, vmax=vmax,
            shading="auto", rasterized=rasterized,
        )

        if log_freq:
            ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

        if title is None:
            title = f"{ds.instrument} — {pd.Timestamp(ds.times[0]).strftime('%Y-%m-%d')}"
        ax.set_title(title)

        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.set_label(cbar_label or ds.unit)
        cb.outline.set_linewidth(0.5)

    return fig


def _edges_from_centres(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    if len(c) == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[1:] + c[:-1])
    first = c[0] - (mid[0] - c[0])
    last = c[-1] + (c[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])
