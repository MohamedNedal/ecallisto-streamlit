"""
3D Earth globe with coastlines and country borders, plus station markers.

The Earth body is a Plotly Surface trace coloured like an ocean. Coastlines and
country borders are added as Scatter3d line traces on top, using a coarse
GSHHS-derived bundle shipped in data/earth.npz (~220 KB). Station markers live
at a slightly larger radius so they sit clearly above the surface.

The function returns the figure AND the index of the "available stations"
trace, so the calling Streamlit app can identify clicked markers by curveNumber.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from .stations import Station

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EARTH_NPZ = DATA_DIR / "earth.npz"


# ----- geometry helpers -----------------------------------------------------

def _latlon_to_xyz(lat_deg, lon_deg, r=1.0):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    return r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat)


def _split_antimeridian(lons: np.ndarray, lats: np.ndarray, threshold: float = 180.0):
    """Yield (lon, lat) sub-arrays split where longitude jumps by >= threshold."""
    if lons.size < 2:
        yield lons, lats
        return
    d = np.abs(np.diff(lons))
    breaks = np.flatnonzero(d >= threshold)
    start = 0
    for b in breaks:
        yield lons[start:b + 1], lats[start:b + 1]
        start = b + 1
    yield lons[start:], lats[start:]


# ----- Earth surface + lines ------------------------------------------------

def _earth_surface(
    radius: float = 1.0,
    ocean_colour: str = "#1b3a5b",
    land_colour: str = "#3d6b4a",
) -> go.Surface:
    """
    Earth body. If the bundled land-sea mask is available, colour land vs ocean;
    otherwise fall back to a uniform ocean tone.
    """
    if EARTH_NPZ.exists():
        npz = np.load(EARTH_NPZ)
        if "lsmask" in npz.files:
            mask = npz["lsmask"].astype(np.float32)
            lats = npz["lsmask_lat"].astype(np.float64)
            lons = npz["lsmask_lon"].astype(np.float64)
            LON, LAT = np.meshgrid(lons, lats)
            x, y, z = _latlon_to_xyz(LAT, LON, radius)
            return go.Surface(
                x=x, y=y, z=z,
                surfacecolor=mask,
                colorscale=[[0.0, ocean_colour], [0.5, ocean_colour],
                            [0.5, land_colour],  [1.0, land_colour]],
                cmin=0.0, cmax=1.0,
                showscale=False, opacity=1.0,
                lighting=dict(ambient=0.7, diffuse=0.5, specular=0.10,
                              roughness=0.95, fresnel=0.05),
                lightposition=dict(x=-100, y=-200, z=120),
                hoverinfo="skip", name="earth",
            )
    # fallback: uniform ocean sphere
    lat = np.linspace(-90, 90, 90)
    lon = np.linspace(-180, 180, 180)
    LON, LAT = np.meshgrid(lon, lat)
    x, y, z = _latlon_to_xyz(LAT, LON, radius)
    return go.Surface(
        x=x, y=y, z=z,
        surfacecolor=np.zeros_like(x),
        colorscale=[[0, ocean_colour], [1, ocean_colour]],
        showscale=False, opacity=1.0,
        lighting=dict(ambient=0.65, diffuse=0.55, specular=0.15,
                      roughness=0.85, fresnel=0.05),
        hoverinfo="skip", name="earth",
    )


def _line_trace_from_npz(
    key_xy: str,
    key_starts: str,
    radius: float,
    colour: str,
    width: float,
    name: str,
) -> Optional[go.Scatter3d]:
    """
    Pack many line segments into ONE Scatter3d trace by inserting NaNs between
    them. This keeps the trace count low — far better for Plotly performance.
    """
    if not EARTH_NPZ.exists():
        return None
    npz = np.load(EARTH_NPZ)
    xy = npz[key_xy]
    starts = npz[key_starts]
    xs, ys, zs = [], [], []
    nan = float("nan")
    for i in range(len(starts) - 1):
        s, e = starts[i], starts[i + 1]
        if e <= s:
            continue
        seg = xy[s:e]
        lons, lats = seg[:, 0], seg[:, 1]
        for ln, lt in _split_antimeridian(lons, lats):
            if ln.size < 2:
                continue
            x, y, z = _latlon_to_xyz(lt, ln, radius)
            xs.extend(x.tolist()); ys.extend(y.tolist()); zs.extend(z.tolist())
            xs.append(nan); ys.append(nan); zs.append(nan)
    if not xs:
        return None
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=colour, width=width),
        hoverinfo="skip", showlegend=False, name=name,
        connectgaps=False,
    )


def _graticule(radius: float = 1.001, colour: str = "rgba(255,255,255,0.10)") -> go.Scatter3d:
    """Faint meridian/parallel reference grid — packed into ONE trace."""
    nan = float("nan")
    xs, ys, zs = [], [], []
    lon_line = np.linspace(-180, 180, 181)
    for lat in np.arange(-60, 61, 30):
        x, y, z = _latlon_to_xyz(np.full_like(lon_line, lat, dtype=float), lon_line, radius)
        xs.extend(x.tolist()); ys.extend(y.tolist()); zs.extend(z.tolist())
        xs.append(nan); ys.append(nan); zs.append(nan)
    lat_line = np.linspace(-90, 90, 91)
    for lon in np.arange(-150, 181, 30):
        x, y, z = _latlon_to_xyz(lat_line, np.full_like(lat_line, lon, dtype=float), radius)
        xs.extend(x.tolist()); ys.extend(y.tolist()); zs.extend(z.tolist())
        xs.append(nan); ys.append(nan); zs.append(nan)
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=colour, width=1),
        hoverinfo="skip", showlegend=False, connectgaps=False,
    )


# ----- main entry -----------------------------------------------------------

@dataclass
class GlobeResult:
    fig: go.Figure
    available_trace_index: int            # which trace holds the red markers
    available_codes: List[str]            # ordered list (curveNumber → code)


def build_globe(
    available_codes: List[str],
    registry: Dict[str, Station],
    available_map: Optional[Dict[str, Station]] = None,
    show_countries: bool = True,
    show_graticule: bool = True,
    bg: str = "rgba(0,0,0,0)",
) -> GlobeResult:
    """
    Build the Plotly figure.

    available_codes : list of station codes that have files in the search window
                      (rendered as red markers).
    registry        : full station registry (drawn as faint grey context dots).
    available_map   : optional dict {code -> Station with observed_* populated}.

    The "available" markers are always added LAST, so the calling code can match
    click events by curveNumber == result.available_trace_index.
    """
    fig = go.Figure()

    # 1) Earth surface
    fig.add_trace(_earth_surface(radius=1.0))

    # 2) coastlines (dark, prominent) — single trace, NaN-separated
    coast = _line_trace_from_npz(
        "coast_xy", "coast_starts",
        radius=1.003, colour="#0d1b2a", width=1.6, name="coast",
    )
    if coast is not None:
        fig.add_trace(coast)

    # 3) country borders (lighter) — single trace
    if show_countries:
        borders = _line_trace_from_npz(
            "country_xy", "country_starts",
            radius=1.002, colour="rgba(20,30,45,0.55)", width=0.8, name="country",
        )
        if borders is not None:
            fig.add_trace(borders)

    # 4) faint graticule (optional) — one trace
    if show_graticule:
        fig.add_trace(_graticule(radius=1.0015))

    # 5) greyed-out 'no data' stations (context only)
    avail_set = set(available_codes or [])
    grey_codes = [c for c in registry if c not in avail_set]
    if grey_codes:
        xs, ys, zs, txts = [], [], [], []
        for c in grey_codes:
            s = registry[c]
            x, y, z = _latlon_to_xyz(s.lat, s.lon, 1.013)
            xs.append(x); ys.append(y); zs.append(z); txts.append(f"{s.code} (no data)")
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="markers",
            marker=dict(size=3, color="rgba(180,180,190,0.55)"),
            text=txts, hovertemplate="%{text}<extra></extra>",
            customdata=[["__nodata__", c] for c in grey_codes],
            name="no data", showlegend=False,
        ))

    # 6) available stations as red markers (ALWAYS LAST)
    ordered_codes: List[str] = []
    if available_codes:
        xs, ys, zs, labels, custom = [], [], [], [], []
        for c in available_codes:
            s = (available_map or registry).get(c) or registry.get(c)
            if s is None:
                continue
            x, y, z = _latlon_to_xyz(s.lat, s.lon, 1.017)
            xs.append(x); ys.append(y); zs.append(z)
            labels.append(s.code)
            custom.append(["__avail__", s.code, s.name, s.city, s.country, s.n_files])
            ordered_codes.append(s.code)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="markers+text",
            marker=dict(size=8, color="#e63946",
                        line=dict(width=1.2, color="white"),
                        symbol="circle"),
            text=labels,
            textposition="top center",
            textfont=dict(size=11, color="#ffe8a1"),
            customdata=custom,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "%{customdata[2]}<br>"
                "%{customdata[3]}, %{customdata[4]}<br>"
                "files in window: %{customdata[5]}"
                "<extra></extra>"
            ),
            name="available",
            showlegend=False,
        ))

    available_trace_index = len(fig.data) - 1 if ordered_codes else -1

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.9)),
            bgcolor=bg,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=bg,
        showlegend=False,
        height=640,
    )

    return GlobeResult(fig=fig, available_trace_index=available_trace_index, available_codes=ordered_codes)
