"""
e-CALLISTO Globe Explorer
=========================
A Streamlit app for browsing the e-CALLISTO solar radio spectrograph network,
fetching FITS files in a chosen UT window, and processing/visualising them.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import io
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path

import numpy as np
import streamlit as st

# 3D-friendly click handler.  Falls back to a dropdown if the package is missing.
try:
    from streamlit_plotly_events import plotly_events  # type: ignore
    HAVE_PLOTLY_EVENTS = True
except Exception:
    plotly_events = None  # type: ignore
    HAVE_PLOTLY_EVENTS = False

from ecallisto import fetch as ec_fetch
from ecallisto import globe as ec_globe
from ecallisto import plotting as ec_plot
from ecallisto import processing as ec_proc
from ecallisto.stations import load_stations, match_station, Station

st.set_page_config(
    page_title="e-CALLISTO Globe Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- session state ----------------------------------------------------------

def _ss_default(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


_ss_default("registry", load_stations())
_ss_default("entries", [])
_ss_default("by_station", {})             # keyed by registry station code (post-match)
_ss_default("available_stations", {})
_ss_default("selected_code", None)
_ss_default("download_dir", str(Path.home() / "ecallisto_data"))
_ss_default("loaded_spec", None)
_ss_default("processed_spec", None)
_ss_default("search_window", None)        # (start, end) datetimes


# --- sidebar: time window + search -----------------------------------------

with st.sidebar:
    st.header("Search window (UT)")

    default_start_date = datetime.now(timezone.utc).date() - timedelta(days=1)
    col_a, col_b = st.columns(2)
    with col_a:
        d_start = st.date_input("Start date", value=default_start_date, key="d_start")
        t_start = st.time_input("Start time (UT)", value=dtime(10, 0), key="t_start")
    with col_b:
        d_end = st.date_input("End date", value=default_start_date, key="d_end")
        t_end = st.time_input("End time (UT)", value=dtime(11, 0), key="t_end")

    start_dt = datetime.combine(d_start, t_start).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(d_end, t_end).replace(tzinfo=timezone.utc)

    min_overlap = st.number_input(
        "Min overlap with window (s)", min_value=1, max_value=900, value=30, step=1,
        help="A file must overlap the search window by at least this many seconds."
    )

    st.divider()
    st.header("Globe display")
    show_graticule = st.checkbox("Show lat/lon grid", value=True)
    show_countries = st.checkbox("Show country borders", value=True)

    if st.button("Search availability", type="primary", use_container_width=True):
        if end_dt <= start_dt:
            st.error("End must be after start.")
        else:
            with st.spinner("Querying e-CALLISTO server..."):
                entries = ec_fetch.search_availability(start_dt, end_dt, min_overlap_s=float(min_overlap))
            st.session_state["entries"] = entries
            st.session_state["search_window"] = (start_dt, end_dt)
            raw_by = ec_fetch.group_by_station(entries)

            # remap raw filename-codes to registry codes
            registry = st.session_state["registry"]
            by_station = {}
            available = {}
            for raw_code, files in raw_by.items():
                s = match_station(raw_code, registry)
                if s is None:
                    continue
                # accumulate (a registry code may receive files from multiple raw codes)
                by_station.setdefault(s.code, []).extend(files)
                available[s.code] = s

            # finalise per-station state
            for code, files in by_station.items():
                files.sort(key=lambda x: x.start)
                s = available[code]
                s.n_files = len(files)
                s.observed_time_start = files[0].start.isoformat()
                s.observed_time_end = files[-1].end.isoformat()
                s.file_urls = [f.url for f in files]

            st.session_state["by_station"] = by_station
            st.session_state["available_stations"] = available
            st.session_state["selected_code"] = None
            if not available:
                st.warning("No files found for the chosen window.")
            else:
                st.success(f"Found {len(entries)} files across {len(available)} stations.")

    st.divider()
    st.header("Local data folder")
    st.text_input("Download folder", key="download_dir")

    st.divider()
    st.header("Load a FITS file")
    up = st.file_uploader("Upload .fit / .fit.gz", type=["fit", "gz", "fits"])
    if up is not None:
        tmp = Path(st.session_state["download_dir"]).expanduser() / up.name
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(up.getbuffer())
        try:
            spec = ec_proc.load_fits(tmp)
            st.session_state["loaded_spec"] = spec
            st.session_state["processed_spec"] = spec
            st.success(f"Loaded {up.name}")
        except Exception as e:
            st.error(f"Failed to load: {e}")


# --- header -----------------------------------------------------------------

st.title("e-CALLISTO Globe Explorer")
st.caption(
    "Browse the global solar radio spectrograph network, fetch FITS files in a chosen "
    "UT window, and process/visualise them. Click a red dot on the globe to open the "
    "station details panel."
)


# --- two-column layout: globe (left) + station panel (right) ---------------

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Network globe")
    available_map = st.session_state["available_stations"]
    globe = ec_globe.build_globe(
        available_codes=list(available_map.keys()),
        registry=st.session_state["registry"],
        available_map=available_map,
        show_countries=show_countries,
        show_graticule=show_graticule,
    )

    if HAVE_PLOTLY_EVENTS and globe.available_trace_index >= 0:
        # plotly_events returns clicked points: [{curveNumber, pointNumber, x, y, z, ...}]
        clicks = plotly_events(
            globe.fig,
            click_event=True,
            select_event=False,
            hover_event=False,
            override_height=640,
            key="globe_click",
        )
        if clicks:
            c0 = clicks[0]
            if c0.get("curveNumber") == globe.available_trace_index:
                pn = c0.get("pointNumber")
                if isinstance(pn, int) and 0 <= pn < len(globe.available_codes):
                    st.session_state["selected_code"] = globe.available_codes[pn]
    else:
        # Fallback: native chart + dropdown
        st.plotly_chart(globe.fig, use_container_width=True)
        if globe.available_codes:
            choice = st.selectbox(
                "Pick a station:",
                options=["—"] + globe.available_codes,
                key="station_picker",
            )
            if choice != "—":
                st.session_state["selected_code"] = choice
        if not HAVE_PLOTLY_EVENTS:
            st.info(
                "Install `streamlit-plotly-events` to enable click-on-dot. "
                "For now use the dropdown."
            )

with right:
    st.subheader("Station details")
    code = st.session_state["selected_code"]
    if code is None:
        st.info("Click a red dot on the globe to see station details.")
    else:
        s: Station = available_map.get(code) or st.session_state["registry"].get(code)
        if s is None:
            st.error(f"Unknown station: {code}")
        else:
            st.markdown(f"### {s.code}")
            st.markdown(f"**{s.name}**  \n{s.city}, {s.country}")
            st.markdown(
                f"**Coordinates:** {s.lat:.4f}°, {s.lon:.4f}°  \n"
                f"**Altitude:** {s.altitude_m:.0f} m"
            )

            nom_freq = f"{s.freq_min_mhz}–{s.freq_max_mhz} MHz" if s.freq_min_mhz else "—"
            obs_freq = (
                f"{s.observed_freq_min_mhz:.1f}–{s.observed_freq_max_mhz:.1f} MHz"
                if s.observed_freq_min_mhz else "(load a file to read)"
            )
            st.markdown(
                f"**Frequency (nominal):** {nom_freq}  \n"
                f"**Frequency (observed):** {obs_freq}  \n"
                f"**Nominal Δt:** {s.nominal_cadence_s or '—'} s  \n"
                f"**Observed Δt:** {s.observed_cadence_s or '—'} s  \n"
                f"**Observed Δf:** {s.observed_freq_cadence_mhz or '—'} MHz"
            )

            files = st.session_state["by_station"].get(code, [])
            win = st.session_state["search_window"]
            if win:
                st.markdown(
                    f"**Search window:** "
                    f"{win[0].strftime('%Y-%m-%d %H:%M:%S')} – "
                    f"{win[1].strftime('%H:%M:%S')} UT"
                )
            st.markdown(f"**Files in window:** {len(files)}")

            if files:
                out_dir = Path(st.session_state["download_dir"]).expanduser()

                if st.button("Download all files", type="primary", use_container_width=True):
                    prog = st.progress(0.0, text="Starting download…")
                    def _cb(i, total, e):
                        prog.progress(i / total, text=f"{i}/{total}  {e.filename}")
                    try:
                        paths = ec_fetch.download_many(files, out_dir, progress_cb=_cb)
                        prog.empty()
                        st.success(f"Saved {len(paths)} files to {out_dir}")
                        try:
                            st.session_state["loaded_spec"] = ec_proc.load_fits(paths[0])
                            st.session_state["processed_spec"] = st.session_state["loaded_spec"]
                        except Exception as e:
                            st.warning(f"Downloaded but could not load first file: {e}")
                    except Exception as e:
                        st.error(f"Download failed: {e}")

                st.markdown("**Files in window** (click Download to fetch one):")
                # Per-file row: filename + UT range tooltip, then a Download button.
                for i, f in enumerate(files):
                    if win:
                        shown_start = max(f.start, win[0])
                        shown_end = min(f.end, win[1])
                    else:
                        shown_start, shown_end = f.start, f.end
                    in_win = f"{shown_start.strftime('%H:%M:%S')}–{shown_end.strftime('%H:%M:%S')} UT"
                    file_span = f"{f.start.strftime('%H:%M:%S')}–{f.end.strftime('%H:%M:%S')}"
                    tooltip = (
                        f"In-window: {in_win}\n"
                        f"File span: {file_span}\n"
                        f"URL: {f.url}"
                    )
                    row1, row2 = st.columns([5, 1])
                    with row1:
                        st.markdown(
                            f"<div title='{tooltip}' style='font-family:monospace;font-size:0.85em;'>"
                            f"<b>{f.filename}</b><br>"
                            f"<span style='color:#888'>{in_win}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with row2:
                        if st.button("Download", key=f"dl_{code}_{i}", help=tooltip):
                            try:
                                p = ec_fetch.download_file(f, out_dir)
                                st.success(f"Saved {p.name}")
                                st.session_state["loaded_spec"] = ec_proc.load_fits(p)
                                st.session_state["processed_spec"] = st.session_state["loaded_spec"]
                                st.rerun()
                            except Exception as e:
                                st.error(f"Download failed: {e}")


# --- spectrogram workspace --------------------------------------------------

st.divider()
st.subheader("Spectrogram workspace")

spec = st.session_state["loaded_spec"]
if spec is None:
    st.info("Download or upload a FITS file to start processing.")
    st.stop()

# refresh observed metadata for the selected station from the loaded file
if st.session_state["selected_code"]:
    s = st.session_state["available_stations"].get(st.session_state["selected_code"])
    if s is not None:
        s.observed_freq_min_mhz = float(spec.freqs_mhz.min())
        s.observed_freq_max_mhz = float(spec.freqs_mhz.max())
        s.observed_cadence_s = round(spec.cadence_s, 4)
        s.observed_freq_cadence_mhz = round(spec.freq_cadence_mhz, 4)

with st.expander("Header metadata", expanded=False):
    keys_of_interest = [
        "INSTRUME", "CONTENT", "DATE-OBS", "TIME-OBS", "CRVAL1", "CDELT1",
        "CRVAL2", "CDELT2", "OBS_LAT", "OBS_LON", "OBS_ALT", "ORIGIN",
        "TELESCOP", "ANTENNAID", "OBJECT",
    ]
    md = {k: spec.header.get(k) for k in keys_of_interest if k in spec.header}
    if md:
        st.json(md)
    else:
        st.code("\n".join(f"{k:10s} = {v!r}" for k, v in list(spec.header.items())[:30]))

# --- processing controls ---

st.markdown("#### Processing")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Downsample**")
    ds_t = st.number_input("Time factor", min_value=1, max_value=64, value=1, step=1)
    ds_f = st.number_input("Frequency factor", min_value=1, max_value=64, value=1, step=1)

with c2:
    st.markdown("**Crop**")
    t_min_s = float(spec.times_s.min())
    t_max_s = float(spec.times_s.max())
    f_min = float(spec.freqs_mhz.min())
    f_max = float(spec.freqs_mhz.max())
    t_range = st.slider(
        "Time (s from obs start)",
        min_value=t_min_s, max_value=t_max_s,
        value=(t_min_s, t_max_s),
        step=max(spec.cadence_s, 0.1),
    )
    f_range = st.slider(
        "Frequency (MHz)",
        min_value=f_min, max_value=f_max,
        value=(f_min, f_max),
        step=max(spec.freq_cadence_mhz, 0.1),
    )

with c3:
    st.markdown("**Background**")
    bg_method = st.selectbox("Method", list(ec_proc.BG_METHODS.keys()), index=0)
    quiet_w = (t_min_s, min(t_min_s + 30.0, t_max_s))
    run_med_window = 60.0
    decile_pct = 10.0
    poly_order = 2
    if bg_method.startswith("Quiet-Sun"):
        quiet_w = st.slider(
            "Quiet window (s)",
            min_value=t_min_s, max_value=t_max_s,
            value=(t_min_s, min(t_min_s + 30.0, t_max_s)),
            step=max(spec.cadence_s, 0.1),
        )
    elif bg_method.startswith("Running median"):
        run_med_window = st.number_input("Window (s)", min_value=1.0, max_value=600.0, value=60.0, step=1.0)
    elif bg_method.startswith("Lowest decile"):
        decile_pct = st.slider("Percentile", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
    elif bg_method.startswith("Polynomial"):
        poly_order = int(st.number_input("Polynomial order", min_value=1, max_value=6, value=2, step=1))

work = spec
work = ec_proc.downsample(work, factor_t=ds_t, factor_f=ds_f)
work = ec_proc.crop(work, t_start_s=t_range[0], t_end_s=t_range[1],
                    f_min_mhz=f_range[0], f_max_mhz=f_range[1])
work = ec_proc.apply_bg(
    work, bg_method,
    quiet_window_s=quiet_w,
    run_median_window_s=run_med_window,
    low_decile_percentile=decile_pct,
    poly_order=poly_order,
)
st.session_state["processed_spec"] = work

# --- visualisation controls ---

st.markdown("#### Visualisation")
v1, v2, v3, v4 = st.columns(4)
with v1:
    cmap = st.selectbox("Colour map", ec_plot.CMAPS, index=0)
    invert_freq = st.checkbox("Invert frequency axis", value=True)
with v2:
    vmin_pct = st.slider("vmin percentile", 0.0, 49.0, 1.0, step=0.5)
    vmax_pct = st.slider("vmax percentile", 50.0, 100.0, 99.0, step=0.5)
with v3:
    show_grid = st.checkbox("Grid", value=False)
    log_freq = st.checkbox("Log frequency", value=False)
    show_cb = st.checkbox("Colourbar", value=True)
with v4:
    fig_w = st.number_input("Width (in)", 4.0, 16.0, 9.0, step=0.5)
    fig_h = st.number_input("Height (in)", 2.5, 10.0, 4.5, step=0.5)

style = ec_plot.PlotStyle(
    cmap=cmap, vmin_pct=vmin_pct, vmax_pct=vmax_pct,
    figsize=(fig_w, fig_h), dpi=110,
    invert_freq=invert_freq, show_grid=show_grid, log_freq=log_freq,
    show_colourbar=show_cb,
    cbar_label=("Δ Intensity" if work.bg_subtracted else "Intensity (raw counts)"),
)
fig = ec_plot.render(work, style)
st.pyplot(fig, use_container_width=True)

# --- export -----------------------------------------------------------------

st.markdown("#### Export")
e1, e2, e3, e4 = st.columns(4)
with e1:
    fmt = st.selectbox("Format", ["png", "pdf", "eps", "svg"], index=1)
with e2:
    dpi_out = st.number_input("DPI", min_value=72, max_value=1200, value=300, step=50)
with e3:
    fname = st.text_input(
        "Filename",
        value=f"{work.station_code}_{work.obs_start.strftime('%Y%m%d_%H%M%S')}.{fmt}",
    )
with e4:
    pub_style = ec_plot.PlotStyle(
        cmap=cmap, vmin_pct=vmin_pct, vmax_pct=vmax_pct,
        figsize=(fig_w, fig_h), dpi=int(dpi_out),
        invert_freq=invert_freq, show_grid=show_grid, log_freq=log_freq,
        show_colourbar=show_cb,
        cbar_label=("Δ Intensity" if work.bg_subtracted else "Intensity (raw counts)"),
    )
    fig_pub = ec_plot.render(work, pub_style)
    payload = ec_plot.export(fig_pub, fmt=fmt, dpi=int(dpi_out))
    st.download_button(
        "Download figure",
        data=payload,
        file_name=fname,
        mime={
            "png": "image/png", "pdf": "application/pdf",
            "eps": "application/postscript", "svg": "image/svg+xml",
        }[fmt],
        use_container_width=True,
    )

buf = io.BytesIO()
np.savez_compressed(
    buf,
    data=work.data, freqs_mhz=work.freqs_mhz, times_s=work.times_s,
    obs_start_iso=work.obs_start.isoformat(), station=work.station_code,
    bg_subtracted=work.bg_subtracted,
)
buf.seek(0)
st.download_button(
    "Download processed array (.npz)",
    data=buf.getvalue(),
    file_name=f"{work.station_code}_{work.obs_start.strftime('%Y%m%d_%H%M%S')}.npz",
    mime="application/octet-stream",
)
