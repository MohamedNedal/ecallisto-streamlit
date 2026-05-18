"""NenuFAR reader.

Targets UnDySPuTeD-style HDF5 dynamic-spectrum products:

  /WAVEFORM/data        : 2-D dataset (n_time, n_freq) or (n_freq, n_time)
  /WAVEFORM/time        : seconds since /WAVEFORM.attrs['DATE-OBS']
                          (or absolute UTC strings)
  /WAVEFORM/frequency   : channel frequencies (Hz unless attr 'UNITS' says otherwise)

If your local file uses different group/dataset names, set `paths=` to override.
"""
from __future__ import annotations

from typing import Optional

import h5py
import numpy as np
import pandas as pd

from sre.readers.base import resolve_path
from sre.spectrum import DynamicSpectrum, ensure_freq_time
from sre.utils import ensure_mhz


DEFAULT_PATHS = {
    "data": ["/WAVEFORM/data", "/data", "/Stokes/I"],
    "time": ["/WAVEFORM/time", "/time", "/timestamps"],
    "freq": ["/WAVEFORM/frequency", "/frequency", "/freq"],
}


def read(path, paths: Optional[dict] = None, **_) -> DynamicSpectrum:
    p = resolve_path(path)
    paths = paths or DEFAULT_PATHS
    with h5py.File(p, "r") as f:
        data_ds = _resolve(f, paths["data"])
        time_ds = _resolve(f, paths["time"])
        freq_ds = _resolve(f, paths["freq"])

        data = np.asarray(data_ds[:], dtype=np.float32)
        time_raw = np.asarray(time_ds[:])
        freq_raw = np.asarray(freq_ds[:], dtype=float)

        date_obs = data_ds.attrs.get("DATE-OBS", time_ds.attrs.get("DATE-OBS", b"1970-01-01"))
        if isinstance(date_obs, bytes):
            date_obs = date_obs.decode()
        freq_unit = freq_ds.attrs.get("UNITS", b"Hz")
        if isinstance(freq_unit, bytes):
            freq_unit = freq_unit.decode()

    if np.issubdtype(time_raw.dtype, np.number):
        t0 = pd.Timestamp(date_obs)
        times = (t0 + pd.to_timedelta(time_raw.astype(float), unit="s"))\
            .to_numpy().astype("datetime64[ns]")
    else:
        times = pd.to_datetime(time_raw.astype(str)).to_numpy().astype("datetime64[ns]")

    freqs = ensure_mhz(freq_raw, freq_unit)
    data = ensure_freq_time(data, freqs, times)

    return DynamicSpectrum(
        data=data,
        times=times,
        frequencies=freqs,
        instrument="NenuFAR",
        unit="arbitrary",
        metadata={"station": "Nançay (NenuFAR core)"},
    )


def _resolve(f, candidates):
    for c in candidates:
        if c in f:
            return f[c]
    raise KeyError(f"none of {candidates} exist in HDF5 file")
