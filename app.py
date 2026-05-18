"""Solar Radio Explorer — Streamlit app.

Run with:    streamlit run app.py
"""
from __future__ import annotations

import io
from datetime import datetime, time as dt_time, date

import numpy as np
import pandas as pd
import streamlit as st

from sre.readers import INSTRUMENT_LABELS, READERS, read as read_instrument


st.set_page_config(
    page_title="Solar Radio Explorer",
    page_icon="☀",
    layout="wide",
)

st.title("Solar Radio Explorer")
st.caption(
    "Load, clean, and export solar radio dynamic spectra from ground- and "
    "space-based instruments. v0.1 — local files only."
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("ds_original", None)
ss.setdefault("ds", None)               # currently displayed spectrum (after processing)


def _instrument_kwargs_ui(key: str) -> dict:
    """Per-instrument optional kwargs (kept inline to keep app.py self-contained)."""
    with st.expander("Reader options", expanded=False):
        if key == "psp_fields":
            band = st.selectbox("Band", ["auto", "lfr", "hfr"], index=0)
            channel = st.text_input("Channel", value="V1V2")
            return {"band": band, "channel": channel}
        if key == "solo_rpw":
            sensor = st.selectbox("Sensor", ["AGC1", "AGC2"], index=0)
            product = st.selectbox("Product", ["auto", "tnr", "hfr"], index=0)
            return {"sensor": sensor, "product": product}
        if key == "wind_waves":
            product = st.selectbox("Product", ["auto", "rad1", "rad2", "tnr"], index=0)
            return {"product": product}
        if key == "stereo_waves":
            prefer = st.selectbox("Preferred PSD variable", ["flux", "voltage"], index=0)
            return {"prefer": prefer}
        if key == "lofar":
            beam = st.number_input("Beam index", min_value=0, value=0, step=1)
            return {"beam": int(beam)}
        if key == "orfees":
            pol = st.selectbox("Polarisation", ["I", "V"], index=0)
            return {"polarization": pol}
        if key == "nda":
            pol = st.selectbox("Polarisation", ["I", "V", "LL", "RR"], index=0)
            return {"polarization": pol}
        return {}


# ---------------------------------------------------------------------------
# Sidebar — instrument + load
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1 · Load")
    instr_key = st.selectbox(
        "Instrument",
        options=list(READERS.keys()),
        format_func=lambda k: INSTRUMENT_LABELS[k],
    )

    source = st.radio("File source", ["Upload", "Local path"], horizontal=True)
    file_obj = None
    file_path = None
    if source == "Upload":
        file_obj = st.file_uploader(
            "FITS / CDF / HDF5 file",
            type=["fits", "fit", "fts", "fits.gz", "fit.gz", "cdf", "h5", "hdf5"],
            accept_multiple_files=False,
        )
    else:
        file_path = st.text_input(
            "Absolute path",
            placeholder="/data/observations/2024-05-14/...",
        )

    st.caption("Optional on-load crop (UTC):")
    use_crop = st.checkbox("Crop on load", value=False)
    t_start, t_end = None, None
    if use_crop:
        col_a, col_b = st.columns(2)
        d_start = col_a.date_input("Start date", value=date.today(), key="d_start")
        t_start_ = col_b.time_input("Start time (UTC)", value=dt_time(0, 0), key="t_start")
        d_end = col_a.date_input("End date", value=date.today(), key="d_end")
        t_end_ = col_b.time_input("End time (UTC)", value=dt_time(23, 59), key="t_end")
        t_start = datetime.combine(d_start, t_start_)
        t_end = datetime.combine(d_end, t_end_)

    extra_kwargs = _instrument_kwargs_ui(instr_key)

    if st.button("Load", type="primary", use_container_width=True):
        target = file_obj if file_obj is not None else file_path
        if not target:
            st.error("Provide a file (upload or absolute path).")
        else:
            try:
                with st.spinner("Reading file…"):
                    ds = read_instrument(instr_key, target, **extra_kwargs)
                if use_crop:
                    ds = ds.crop(t_start=t_start, t_end=t_end)
                ss.ds_original = ds
                ss.ds = ds
                st.success(f"Loaded {ds.instrument}: {ds.shape[0]} freq × {ds.shape[1]} time samples.")
            except Exception as exc:                                # noqa: BLE001
                st.exception(exc)


# ---------------------------------------------------------------------------
# Main panel — only meaningful once a spectrum is loaded
# ---------------------------------------------------------------------------
if ss.ds is None:
    st.info("Pick an instrument and load a file in the sidebar to begin.")
else:
    ds = ss.ds
    left, right = st.columns([3, 1], gap="large")

    with right:
        st.subheader("Metadata")
        st.json(ds.summary(), expanded=False)
        if st.button("Reset to original", use_container_width=True):
            ss.ds = ss.ds_original
            st.rerun()

    with left:
        proc_tab, vis_tab, export_tab = st.tabs(["Processing", "Visualisation", "Export"])

        with proc_tab:
            st.markdown("**Crop**")
            c1, c2, c3, c4 = st.columns(4)
            t_lo, t_hi = ds.time_range
            f_lo, f_hi = ds.freq_range
            crop_t0 = c1.text_input("Time start (UTC)", value=str(t_lo))
            crop_t1 = c2.text_input("Time end (UTC)", value=str(t_hi))
            crop_f0 = c3.number_input("Freq min (MHz)", value=float(f_lo), format="%.3f")
            crop_f1 = c4.number_input("Freq max (MHz)", value=float(f_hi), format="%.3f")
            if st.button("Apply crop"):
                try:
                    ss.ds = ds.crop(t_start=crop_t0, t_end=crop_t1,
                                    f_min=crop_f0, f_max=crop_f1)
                    st.rerun()
                except Exception as exc:                                # noqa: BLE001
                    st.exception(exc)

            st.markdown("**Downsample**")
            d1, d2, d3 = st.columns(3)
            time_factor = d1.number_input("Time factor", min_value=1, value=1, step=1)
            freq_factor = d2.number_input("Freq factor", min_value=1, value=1, step=1)
            method = d3.selectbox("Reduction", ["mean", "median", "max"])
            if st.button("Apply downsample"):
                ss.ds = ds.downsample(time_factor=int(time_factor),
                                      freq_factor=int(freq_factor), method=method)
                st.rerun()

            st.markdown("**Background subtraction**")
            bg_method = st.selectbox(
                "Method",
                ["quiet_window", "median", "running_median", "constant"],
            )
            bg_kwargs: dict = {}
            if bg_method == "quiet_window":
                q1, q2 = st.columns(2)
                q_start = q1.text_input("Quiet start (UTC)", value=str(t_lo))
                q_end = q2.text_input("Quiet end (UTC)",
                                      value=str(t_lo + (t_hi - t_lo) * 0.05))
                bg_kwargs["t_quiet"] = (q_start, q_end)
            elif bg_method == "running_median":
                width = st.number_input("Window width (samples)",
                                        min_value=3, value=51, step=2)
                bg_kwargs["width"] = int(width)
            elif bg_method == "constant":
                value = st.number_input("Subtract value",
                                        value=float(np.nanmedian(ds.data)))
                bg_kwargs["value"] = float(value)
            if st.button("Apply background subtraction"):
                try:
                    ss.ds = ds.subtract_background(method=bg_method, **bg_kwargs)
                    st.rerun()
                except Exception as exc:                                # noqa: BLE001
                    st.exception(exc)

            if ds.history:
                st.caption("History:")
                for step in ds.history:
                    st.code(step, language="python")

        with vis_tab:
            v1, v2, v3 = st.columns(3)
            cmap = v1.selectbox(
                "Colour map",
                ["inferno", "viridis", "plasma", "magma", "cividis",
                 "Greys_r", "hot", "RdBu_r", "Spectral_r"],
                index=0,
            )
            log_freq = v2.checkbox("Log frequency", value=False)
            rasterised = v3.checkbox("Rasterise image (smaller PDF)", value=True)

            clip_pct = st.slider("Display percentile clip", 0.0, 100.0,
                                 value=(1.0, 99.0), step=0.5)
            title = st.text_input(
                "Title",
                value=f"{ds.instrument} — {pd.Timestamp(ds.times[0]).strftime('%Y-%m-%d')}",
            )

            fig = ds.plot(
                cmap=cmap, vmin_pct=clip_pct[0], vmax_pct=clip_pct[1],
                log_freq=log_freq, title=title, rasterized=rasterised,
            )
            st.pyplot(fig, use_container_width=True)

        with export_tab:
            st.markdown("**Figure**")
            fmt = st.selectbox("Format", ["pdf", "png", "eps", "svg"])
            dpi = st.number_input("DPI (raster only)", min_value=72, value=300, step=50)
            if st.button("Build figure for download"):
                buf = io.BytesIO()
                fig = ds.plot(cmap=cmap, vmin_pct=clip_pct[0], vmax_pct=clip_pct[1],
                              log_freq=log_freq, title=title, rasterized=rasterised)
                fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
                buf.seek(0)
                st.download_button(
                    f"Download .{fmt}",
                    data=buf,
                    file_name=f"{ds.instrument.replace(' ', '_')}_"
                              f"{pd.Timestamp(ds.times[0]).strftime('%Y%m%dT%H%M')}.{fmt}",
                    mime="application/octet-stream",
                )

            st.markdown("**Processed array**")
            d_fmt = st.selectbox("Array format", ["npz", "fits"])
            if st.button("Build array for download"):
                buf = io.BytesIO()
                if d_fmt == "npz":
                    np.savez_compressed(
                        buf,
                        data=ds.data,
                        times=ds.times.astype("datetime64[ns]").astype("int64"),
                        frequencies=ds.frequencies,
                        instrument=np.array(ds.instrument),
                        unit=np.array(ds.unit),
                    )
                else:
                    import tempfile, os
                    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
                        ds.save_fits(tmp.name)
                        with open(tmp.name, "rb") as f:
                            buf.write(f.read())
                    os.unlink(tmp.name)
                buf.seek(0)
                st.download_button(
                    f"Download .{d_fmt}",
                    data=buf,
                    file_name=f"{ds.instrument.replace(' ', '_')}_"
                              f"{pd.Timestamp(ds.times[0]).strftime('%Y%m%dT%H%M')}.{d_fmt}",
                    mime="application/octet-stream",
                )
