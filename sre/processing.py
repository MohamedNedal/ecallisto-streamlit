"""Numerical operations on dynamic spectra.

These functions operate on raw numpy arrays plus the time/frequency axes, so
they can be unit-tested without instantiating DynamicSpectrum objects. The
DynamicSpectrum class wraps each one and appends a history entry.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter


def _to_datetime64(t) -> np.datetime64:
    return np.datetime64(pd.Timestamp(t).to_datetime64())


def crop(data: np.ndarray, times: np.ndarray, frequencies: np.ndarray,
         t_start=None, t_end=None,
         f_min: Optional[float] = None, f_max: Optional[float] = None):
    """Crop in time and/or frequency. Bounds are inclusive."""
    t_mask = np.ones_like(times, dtype=bool)
    if t_start is not None:
        t_mask &= times >= _to_datetime64(t_start)
    if t_end is not None:
        t_mask &= times <= _to_datetime64(t_end)

    f_mask = np.ones_like(frequencies, dtype=bool)
    if f_min is not None:
        f_mask &= frequencies >= f_min
    if f_max is not None:
        f_mask &= frequencies <= f_max

    if not t_mask.any() or not f_mask.any():
        raise ValueError("crop window leaves no samples")

    return data[np.ix_(f_mask, t_mask)], times[t_mask], frequencies[f_mask]


def downsample(data: np.ndarray, times: np.ndarray, frequencies: np.ndarray,
               time_factor: int = 1, freq_factor: int = 1, method: str = "mean"):
    """Block-reduce by integer factors along each axis."""
    if time_factor < 1 or freq_factor < 1:
        raise ValueError("downsample factors must be ≥ 1")
    if time_factor == 1 and freq_factor == 1:
        return data, times, frequencies

    nf, nt = data.shape
    # Trim so dimensions are divisible by the factors.
    nf_keep = (nf // freq_factor) * freq_factor
    nt_keep = (nt // time_factor) * time_factor
    d = data[:nf_keep, :nt_keep]
    f = frequencies[:nf_keep]
    t = times[:nt_keep]

    d = d.reshape(nf_keep // freq_factor, freq_factor,
                  nt_keep // time_factor, time_factor)

    reducers = {"mean": np.nanmean, "median": np.nanmedian, "max": np.nanmax}
    if method not in reducers:
        raise ValueError(f"unknown method {method!r}; use one of {list(reducers)}")
    reduce = reducers[method]
    d = reduce(reduce(d, axis=1), axis=2)

    f = f.reshape(-1, freq_factor).mean(axis=1)
    # For datetime64, average via int64 cast to avoid overflow on ns scale.
    t_int = t.astype("datetime64[ns]").astype("int64").reshape(-1, time_factor).mean(axis=1)
    t = t_int.astype("int64").astype("datetime64[ns]")
    return d, t, f


def subtract_background(data: np.ndarray, times: Optional[np.ndarray] = None,
                        method: str = "quiet_window", **kwargs) -> np.ndarray:
    """Remove instrumental and quiescent Sun background from a dynamic spectrum.

    Methods:
      'quiet_window'  : subtract per-channel median computed over (t_quiet[0], t_quiet[1]).
                        Falls back to the global per-channel median if t_quiet is None.
      'median'        : subtract per-channel median of the full spectrum (robust default).
      'running_median': subtract a running per-channel median with window `width` samples.
      'constant'      : subtract a single scalar (`value`, defaults to global median).
    """
    if method == "quiet_window":
        t_quiet = kwargs.get("t_quiet")
        if t_quiet is None or times is None:
            bg = np.nanmedian(data, axis=1, keepdims=True)
        else:
            mask = (times >= _to_datetime64(t_quiet[0])) & (times <= _to_datetime64(t_quiet[1]))
            if not mask.any():
                raise ValueError("quiet window does not intersect data times")
            bg = np.nanmedian(data[:, mask], axis=1, keepdims=True)
        return data - bg

    if method == "median":
        return data - np.nanmedian(data, axis=1, keepdims=True)

    if method == "running_median":
        width = int(kwargs.get("width", 51))
        if width < 3:
            raise ValueError("running_median width must be ≥ 3")
        bg = median_filter(data, size=(1, width), mode="nearest")
        return data - bg

    if method == "constant":
        value = kwargs.get("value", float(np.nanmedian(data)))
        return data - value

    raise ValueError(f"unknown background method {method!r}")


def percentile_clip(data: np.ndarray, lo: float = 1.0, hi: float = 99.0):
    """Return (vmin, vmax) from finite-data percentiles, useful for plotting."""
    finite = np.isfinite(data)
    if not finite.any():
        return 0.0, 1.0
    return np.percentile(data[finite], [lo, hi]).tolist()
