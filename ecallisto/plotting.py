"""Publication-quality spectrogram plotting (matplotlib + Plotly)."""
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .processing import Spectrogram

CMAPS = [
    "magma", "plasma", "inferno", "viridis", "cividis",
    "turbo", "Greys_r", "Spectral_r", "RdYlBu_r", "twilight",
]


@dataclass
class PlotStyle:
    cmap: str = "magma"
    vmin_pct: float = 1.0
    vmax_pct: float = 99.0
    figsize: Tuple[float, float] = (9.0, 4.5)
    dpi: int = 150
    invert_freq: bool = True
    show_grid: bool = False
    log_freq: bool = False
    show_colourbar: bool = True
    title: Optional[str] = None
    xlabel: str = "Time (UT)"
    ylabel: str = "Frequency (MHz)"
    cbar_label: str = "Intensity (raw counts)"
    fontsize: int = 11
    line_width: float = 0.8


def render(spec: Spectrogram, style: PlotStyle) -> Figure:
    """Return a Matplotlib Figure ready to display or export."""
    plt.rcParams.update({
        "font.size": style.fontsize,
        "axes.linewidth": style.line_width,
        "xtick.major.width": style.line_width,
        "ytick.major.width": style.line_width,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })

    fig, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi)

    vmin = np.percentile(spec.data, style.vmin_pct)
    vmax = np.percentile(spec.data, style.vmax_pct)
    if vmax <= vmin:
        vmax = vmin + 1.0

    # Build edges in real units so pcolormesh gives proper axes.
    t_edges_dt = _edges_dt(spec)
    f_edges = _edges_1d(spec.freqs_mhz)

    qm = ax.pcolormesh(
        t_edges_dt, f_edges, spec.data,
        cmap=style.cmap, vmin=vmin, vmax=vmax, shading="auto", rasterized=True,
    )

    if style.invert_freq:
        ax.invert_yaxis()
    if style.log_freq:
        ax.set_yscale("log")
    if style.show_grid:
        ax.grid(True, which="both", alpha=0.25, linewidth=0.4)

    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_xlabel(style.xlabel)
    ax.set_ylabel(style.ylabel)
    title = style.title
    if title is None:
        title = f"{spec.station_code}  {spec.obs_start.strftime('%Y-%m-%d %H:%M:%S UT')}"
    ax.set_title(title)

    if style.show_colourbar:
        cb = fig.colorbar(qm, ax=ax, pad=0.02)
        cb.set_label(style.cbar_label)
        cb.ax.tick_params(width=style.line_width)

    fig.tight_layout()
    return fig


def _edges_1d(centres: np.ndarray) -> np.ndarray:
    """Cell-centre to cell-edge conversion (handles non-uniform spacing)."""
    c = np.asarray(centres, dtype=float)
    if c.size < 2:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    inner = (c[:-1] + c[1:]) / 2.0
    first = c[0] - (inner[0] - c[0])
    last = c[-1] + (c[-1] - inner[-1])
    return np.concatenate([[first], inner, [last]])


def _edges_dt(spec: Spectrogram) -> np.ndarray:
    from datetime import timedelta
    t_s_edges = _edges_1d(spec.times_s)
    return np.array([spec.obs_start + timedelta(seconds=float(t)) for t in t_s_edges])


def export(fig: Figure, fmt: str = "png", dpi: int = 300) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()
