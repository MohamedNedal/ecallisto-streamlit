"""Common container for solar radio dynamic spectra.

A DynamicSpectrum holds a 2-D array of intensity values together with the time
and frequency axes, instrument metadata, and a history of processing steps. All
readers return one of these; all processing operations consume and return one.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from sre import processing as P


@dataclass
class DynamicSpectrum:
    data: np.ndarray                       # shape (n_freq, n_time)
    times: np.ndarray                      # 1-D, dtype datetime64[ns] or numpy float seconds
    frequencies: np.ndarray                # 1-D, MHz
    instrument: str = "unknown"
    unit: str = "arbitrary"
    metadata: dict = field(default_factory=dict)
    history: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.data.ndim != 2:
            raise ValueError(f"data must be 2-D, got shape {self.data.shape}")
        nf, nt = self.data.shape
        # The most common confusion point: many readers emit (time, freq).
        # Standardise on (freq, time) here, and let readers transpose if needed.
        if len(self.frequencies) != nf or len(self.times) != nt:
            raise ValueError(
                f"axis lengths inconsistent: data is {self.data.shape}, "
                f"frequencies has {len(self.frequencies)}, times has {len(self.times)}"
            )
        # Ensure frequencies are monotonically increasing (some CDFs store HFR descending).
        if len(self.frequencies) > 1 and self.frequencies[0] > self.frequencies[-1]:
            self.frequencies = self.frequencies[::-1]
            self.data = self.data[::-1, :]

    @property
    def shape(self):
        return self.data.shape

    @property
    def freq_range(self):
        return float(self.frequencies.min()), float(self.frequencies.max())

    @property
    def time_range(self):
        return pd.Timestamp(self.times[0]), pd.Timestamp(self.times[-1])

    def copy(self) -> "DynamicSpectrum":
        return copy.deepcopy(self)

    def crop(self, t_start=None, t_end=None,
             f_min: Optional[float] = None, f_max: Optional[float] = None) -> "DynamicSpectrum":
        out = self.copy()
        out.data, out.times, out.frequencies = P.crop(
            self.data, self.times, self.frequencies,
            t_start=t_start, t_end=t_end, f_min=f_min, f_max=f_max,
        )
        out.history.append(
            f"crop(t_start={t_start}, t_end={t_end}, f_min={f_min}, f_max={f_max})"
        )
        return out

    def downsample(self, time_factor: int = 1, freq_factor: int = 1,
                   method: str = "mean") -> "DynamicSpectrum":
        out = self.copy()
        out.data, out.times, out.frequencies = P.downsample(
            self.data, self.times, self.frequencies,
            time_factor=time_factor, freq_factor=freq_factor, method=method,
        )
        out.history.append(
            f"downsample(time_factor={time_factor}, freq_factor={freq_factor}, method={method})"
        )
        return out

    def subtract_background(self, method: str = "quiet_window", **kwargs) -> "DynamicSpectrum":
        out = self.copy()
        out.data = P.subtract_background(
            self.data, times=self.times, method=method, **kwargs,
        )
        out.history.append(f"subtract_background(method={method}, {kwargs})")
        return out

    def to_dB(self, reference: Optional[float] = None) -> "DynamicSpectrum":
        out = self.copy()
        ref = reference if reference is not None else np.nanmedian(self.data)
        if ref <= 0:
            raise ValueError("reference must be positive; data may contain non-positive values")
        with np.errstate(divide="ignore", invalid="ignore"):
            out.data = 10.0 * np.log10(self.data / ref)
        out.unit = "dB"
        out.history.append(f"to_dB(reference={ref:.4g})")
        return out

    def plot(self, **kwargs):
        # Imported lazily so headless environments without matplotlib backends
        # can still load readers and process data.
        from sre.plotting import plot_spectrum
        return plot_spectrum(self, **kwargs)

    def summary(self) -> dict:
        t0, t1 = self.time_range
        f0, f1 = self.freq_range
        return {
            "instrument": self.instrument,
            "shape (freq, time)": self.shape,
            "time range (UTC)": f"{t0} → {t1}",
            "frequency range (MHz)": f"{f0:.3f} – {f1:.3f}",
            "unit": self.unit,
            "history": list(self.history),
            **{f"meta::{k}": v for k, v in self.metadata.items()},
        }

    def save_npz(self, path: str) -> None:
        np.savez_compressed(
            path,
            data=self.data,
            times=self.times.astype("datetime64[ns]").astype("int64"),
            frequencies=self.frequencies,
            instrument=np.array(self.instrument),
            unit=np.array(self.unit),
        )

    def save_fits(self, path: str) -> None:
        from astropy.io import fits
        hdu = fits.PrimaryHDU(self.data.astype(np.float32))
        hdu.header["INSTRUME"] = self.instrument
        hdu.header["BUNIT"] = self.unit
        hdu.header["NAXIS1"] = self.data.shape[1]
        hdu.header["NAXIS2"] = self.data.shape[0]
        hdu.header["DATE-OBS"] = str(pd.Timestamp(self.times[0]))
        hdu.header["DATE-END"] = str(pd.Timestamp(self.times[-1]))
        col_t = fits.Column(
            name="TIME_UTC", format="26A",
            array=np.array([str(pd.Timestamp(t)) for t in self.times]),
        )
        col_f = fits.Column(name="FREQ_MHZ", format="D", array=self.frequencies)
        tab_t = fits.BinTableHDU.from_columns([col_t], name="TIMES")
        tab_f = fits.BinTableHDU.from_columns([col_f], name="FREQS")
        fits.HDUList([hdu, tab_t, tab_f]).writeto(path, overwrite=True)


def ensure_freq_time(data: np.ndarray, frequencies: Sequence,
                     times: Sequence) -> np.ndarray:
    """If the caller hands us a (time, freq) array, transpose it."""
    nf, nt = len(frequencies), len(times)
    if data.shape == (nt, nf):
        return data.T
    if data.shape == (nf, nt):
        return data
    raise ValueError(
        f"cannot reconcile data shape {data.shape} with "
        f"n_freq={nf}, n_time={nt}"
    )
