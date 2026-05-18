"""FITS loading and spectrogram processing for e-CALLISTO files."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import re
from astropy.io import fits
from dateutil import parser as _dateparser
from scipy.ndimage import median_filter

# YYYYMMDD_HHMMSS embedded in e-CALLISTO filenames (e.g. BIR_20240315_103000_01.fit.gz)
_FILENAME_TIMESTAMP_RE = re.compile(r"_(\d{8})_(\d{6})_")


def _parse_obs_start(header: dict, fallback_path: Optional[Path] = None) -> datetime:
    """
    Robust observation-start parser.

    Tries (in order):
      1. Combined DATE-OBS (FITS ISO form like '2024-03-15T10:00:00.000')
      2. DATE-OBS + TIME-OBS concatenated, parsed by dateutil
      3. YYYYMMDD_HHMMSS embedded in the source filename

    Never silently falls back to 'now' -- that was the root cause of the bug
    where downloaded spectra appeared with today's timestamp.
    """
    date_obs = header.get("DATE-OBS") or header.get("DATE_OBS")
    time_obs = header.get("TIME-OBS") or header.get("TIME_OBS")

    candidates = []
    if isinstance(date_obs, str) and "T" in date_obs:
        candidates.append(date_obs)
    if isinstance(date_obs, str) and isinstance(time_obs, str):
        candidates.append(f"{date_obs} {time_obs}")
        candidates.append(f"{date_obs}T{time_obs}")
    if isinstance(date_obs, str):
        candidates.append(date_obs)

    for s in candidates:
        s2 = s.strip()
        if not s2:
            continue
        try:
            dt = _dateparser.parse(s2)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError, OverflowError):
            continue

    if fallback_path is not None:
        m = _FILENAME_TIMESTAMP_RE.search(Path(fallback_path).name)
        if m:
            return datetime.strptime(m.group(1) + m.group(2),
                                     "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)

    raise ValueError(
        "Could not determine observation start: "
        f"DATE-OBS={date_obs!r}, TIME-OBS={time_obs!r}, file={fallback_path}"
    )


@dataclass
class Spectrogram:
    """In-memory CALLISTO spectrogram."""
    data: np.ndarray              # shape (n_freq, n_time), float32, calibrated raw counts
    freqs_mhz: np.ndarray         # shape (n_freq,), descending order kept as in file
    times_s: np.ndarray           # shape (n_time,), seconds since obs_start
    obs_start: datetime           # UTC
    header: Dict                  # copy of primary HDU header as plain dict
    station_code: str = ""
    source_path: Optional[Path] = None
    bg_subtracted: bool = False

    @property
    def t_axis(self) -> np.ndarray:
        return np.array([self.obs_start + timedelta(seconds=float(t)) for t in self.times_s])

    @property
    def cadence_s(self) -> float:
        if len(self.times_s) < 2:
            return float("nan")
        return float(np.median(np.diff(self.times_s)))

    @property
    def freq_cadence_mhz(self) -> float:
        if len(self.freqs_mhz) < 2:
            return float("nan")
        return float(np.median(np.abs(np.diff(self.freqs_mhz))))

    @property
    def freq_range_mhz(self) -> Tuple[float, float]:
        return float(self.freqs_mhz.min()), float(self.freqs_mhz.max())

    @property
    def time_range(self) -> Tuple[datetime, datetime]:
        return self.obs_start, self.obs_start + timedelta(seconds=float(self.times_s[-1]))


def load_fits(path: Path) -> Spectrogram:
    """
    Read an e-CALLISTO FITS file. Standard layout: primary HDU holds the
    2D image (freq x time), HDU 1 is a BINTABLE with TIME and FREQUENCY columns.
    """
    path = Path(path)
    with fits.open(path) as hdul:
        primary = hdul[0]
        data = np.asarray(primary.data, dtype=np.float32)
        header = dict(primary.header)

        # axes from BINTABLE if present
        times_s = None
        freqs_mhz = None
        if len(hdul) > 1 and isinstance(hdul[1], fits.BinTableHDU):
            tbl = hdul[1].data
            cols = [c.upper() for c in hdul[1].columns.names]
            if "TIME" in cols:
                times_s = np.asarray(tbl["TIME"]).ravel().astype(np.float64)
            if "FREQUENCY" in cols:
                freqs_mhz = np.asarray(tbl["FREQUENCY"]).ravel().astype(np.float64)

        # fallback to CRVAL/CDELT
        if times_s is None:
            n_t = data.shape[1]
            t0 = float(header.get("CRVAL1", 0.0))
            dt = float(header.get("CDELT1", 0.25))
            times_s = t0 + np.arange(n_t) * dt
        if freqs_mhz is None:
            n_f = data.shape[0]
            f0 = float(header.get("CRVAL2", 100.0))
            df = float(header.get("CDELT2", 1.0))
            freqs_mhz = f0 + np.arange(n_f) * df

        # observation start datetime -- robust parsing (see _parse_obs_start)
        obs_start = _parse_obs_start(header, fallback_path=path)

        # station code: prefer INSTRUME, fall back to filename head
        station = str(header.get("INSTRUME") or "").strip()
        if not station:
            station = path.name.split("_")[0]

    return Spectrogram(
        data=data,
        freqs_mhz=freqs_mhz,
        times_s=times_s,
        obs_start=obs_start,
        header=header,
        station_code=station,
        source_path=path,
    )


def concatenate(specs: List[Spectrogram]) -> Spectrogram:
    """
    Concatenate several spectrograms from the same station in time.
    Assumes identical frequency axes.
    """
    specs = sorted(specs, key=lambda s: s.obs_start)
    ref = specs[0]
    fa = ref.freqs_mhz
    blocks = [ref.data]
    times = [ref.times_s]
    for s in specs[1:]:
        # offset times so they continue from previous run
        offset = (s.obs_start - ref.obs_start).total_seconds()
        if s.data.shape[0] != fa.shape[0]:
            # if frequency axes differ, skip (could resample, but stay conservative)
            continue
        blocks.append(s.data)
        times.append(s.times_s + offset)
    return Spectrogram(
        data=np.concatenate(blocks, axis=1),
        freqs_mhz=fa,
        times_s=np.concatenate(times),
        obs_start=ref.obs_start,
        header=ref.header,
        station_code=ref.station_code,
        source_path=ref.source_path,
    )


# --- processing operations ---------------------------------------------------

def downsample(spec: Spectrogram, factor_t: int = 1, factor_f: int = 1) -> Spectrogram:
    """Block-mean downsampling. factor=1 is a no-op."""
    factor_t = max(1, int(factor_t))
    factor_f = max(1, int(factor_f))
    if factor_t == 1 and factor_f == 1:
        return spec
    d = spec.data
    n_f, n_t = d.shape
    # trim to multiple
    n_f2 = (n_f // factor_f) * factor_f
    n_t2 = (n_t // factor_t) * factor_t
    d = d[:n_f2, :n_t2]
    d = d.reshape(n_f2 // factor_f, factor_f, n_t2 // factor_t, factor_t).mean(axis=(1, 3))
    freqs = spec.freqs_mhz[:n_f2].reshape(-1, factor_f).mean(axis=1)
    times = spec.times_s[:n_t2].reshape(-1, factor_t).mean(axis=1)
    return Spectrogram(
        data=d.astype(np.float32),
        freqs_mhz=freqs,
        times_s=times,
        obs_start=spec.obs_start,
        header=spec.header,
        station_code=spec.station_code,
        source_path=spec.source_path,
        bg_subtracted=spec.bg_subtracted,
    )


def crop(
    spec: Spectrogram,
    t_start_s: Optional[float] = None,
    t_end_s: Optional[float] = None,
    f_min_mhz: Optional[float] = None,
    f_max_mhz: Optional[float] = None,
) -> Spectrogram:
    """Smooth boolean cropping on both axes."""
    t_mask = np.ones_like(spec.times_s, dtype=bool)
    if t_start_s is not None:
        t_mask &= spec.times_s >= t_start_s
    if t_end_s is not None:
        t_mask &= spec.times_s <= t_end_s

    f_mask = np.ones_like(spec.freqs_mhz, dtype=bool)
    if f_min_mhz is not None:
        f_mask &= spec.freqs_mhz >= f_min_mhz
    if f_max_mhz is not None:
        f_mask &= spec.freqs_mhz <= f_max_mhz

    return Spectrogram(
        data=spec.data[np.ix_(f_mask, t_mask)],
        freqs_mhz=spec.freqs_mhz[f_mask],
        times_s=spec.times_s[t_mask],
        obs_start=spec.obs_start,
        header=spec.header,
        station_code=spec.station_code,
        source_path=spec.source_path,
        bg_subtracted=spec.bg_subtracted,
    )


# --- background subtraction --------------------------------------------------

def bg_quiet_window(
    spec: Spectrogram, quiet_start_s: float, quiet_end_s: float, statistic: str = "median"
) -> Spectrogram:
    """Subtract a per-channel value computed inside [quiet_start_s, quiet_end_s]."""
    m = (spec.times_s >= quiet_start_s) & (spec.times_s <= quiet_end_s)
    if not m.any():
        return spec
    block = spec.data[:, m]
    base = (np.median if statistic == "median" else np.mean)(block, axis=1)
    return _apply_baseline(spec, base[:, None])


def bg_running_median(spec: Spectrogram, window_s: float = 60.0) -> Spectrogram:
    """Subtract a running median along the time axis (per channel)."""
    if spec.cadence_s <= 0 or np.isnan(spec.cadence_s):
        return spec
    win_px = max(3, int(round(window_s / spec.cadence_s)))
    if win_px % 2 == 0:
        win_px += 1
    base = median_filter(spec.data, size=(1, win_px), mode="nearest")
    return _apply_baseline(spec, base)


def bg_lowest_decile(spec: Spectrogram, percentile: float = 10.0) -> Spectrogram:
    """Subtract the chosen low percentile per frequency channel."""
    base = np.percentile(spec.data, percentile, axis=1, keepdims=True)
    return _apply_baseline(spec, base)


def bg_polynomial_detrend(spec: Spectrogram, order: int = 2) -> Spectrogram:
    """Fit a low-order polynomial in time per channel and subtract the fit."""
    t = spec.times_s.astype(np.float64)
    if t.size < order + 2:
        return spec
    base = np.empty_like(spec.data)
    for i, row in enumerate(spec.data):
        coef = np.polyfit(t, row, deg=order)
        base[i] = np.polyval(coef, t)
    return _apply_baseline(spec, base)


def _apply_baseline(spec: Spectrogram, base: np.ndarray) -> Spectrogram:
    return Spectrogram(
        data=(spec.data - base).astype(np.float32),
        freqs_mhz=spec.freqs_mhz,
        times_s=spec.times_s,
        obs_start=spec.obs_start,
        header=spec.header,
        station_code=spec.station_code,
        source_path=spec.source_path,
        bg_subtracted=True,
    )


# dispatcher used by the Streamlit UI
BG_METHODS = {
    "None": None,
    "Quiet-Sun window (median)": ("quiet", "median"),
    "Quiet-Sun window (mean)":   ("quiet", "mean"),
    "Running median (time)":     ("run_median", None),
    "Lowest decile per channel": ("low_decile", None),
    "Polynomial detrend (time)": ("poly", None),
}


def apply_bg(
    spec: Spectrogram,
    method: str,
    *,
    quiet_window_s: Tuple[float, float] = (0.0, 30.0),
    run_median_window_s: float = 60.0,
    low_decile_percentile: float = 10.0,
    poly_order: int = 2,
) -> Spectrogram:
    if method not in BG_METHODS or BG_METHODS[method] is None:
        return spec
    kind, sub = BG_METHODS[method]
    if kind == "quiet":
        return bg_quiet_window(spec, quiet_window_s[0], quiet_window_s[1], statistic=sub)
    if kind == "run_median":
        return bg_running_median(spec, window_s=run_median_window_s)
    if kind == "low_decile":
        return bg_lowest_decile(spec, percentile=low_decile_percentile)
    if kind == "poly":
        return bg_polynomial_detrend(spec, order=poly_order)
    return spec
