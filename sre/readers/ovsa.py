"""OVSA / EOVSA dynamic-spectrum FITS reader.

EOVSA distributes daily dynamic spectra (e.g., from the IDB pipeline) as FITS
with the spectrogram in the primary HDU and time/frequency stored either in
WCS keywords or in BinTable extensions named TIMES / FREQS. This reader tries
the extensions first, then falls back to WCS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.io import fits

from sre.readers.base import resolve_path
from sre.spectrum import DynamicSpectrum, ensure_freq_time
from sre.utils import ensure_mhz


def read(path, **_) -> DynamicSpectrum:
    p = resolve_path(path)
    with fits.open(p) as hdul:
        header = hdul[0].header
        data = np.asarray(hdul[0].data, dtype=np.float32)
        if data is None or data.ndim != 2:
            raise ValueError(f"unexpected EOVSA FITS data shape {None if data is None else data.shape}")

        times = _times(hdul)
        freqs = _freqs(hdul)
        data = ensure_freq_time(data, freqs, times)

    return DynamicSpectrum(
        data=data,
        times=times,
        frequencies=freqs,
        instrument=header.get("TELESCOP", "EOVSA"),
        unit=header.get("BUNIT", "sfu"),
        metadata={"observatory": header.get("ORIGIN", "OVRO")},
    )


def _times(hdul):
    for name in ("TIMES", "TIME"):
        if name in hdul:
            tab = hdul[name].data
            if "JD" in (tab.dtype.names or []):
                jd = np.asarray(tab["JD"], dtype=float)
                # Julian Date → datetime64. Use pandas, which handles JD via offset.
                return pd.to_datetime(jd - 2440587.5, unit="D")\
                    .to_numpy().astype("datetime64[ns]")
            if "TIME_UTC" in (tab.dtype.names or []):
                return pd.to_datetime(tab["TIME_UTC"].astype(str))\
                    .to_numpy().astype("datetime64[ns]")
    h = hdul[0].header
    if "CRVAL1" in h and "CDELT1" in h:
        nt = hdul[0].data.shape[-1]
        t0 = pd.Timestamp(h.get("DATE-OBS", "1970-01-01"))
        return (t0 + pd.to_timedelta((np.arange(nt) - h.get("CRPIX1", 1) + 1) * h["CDELT1"],
                                     unit="s")).to_numpy().astype("datetime64[ns]")
    raise ValueError("could not derive EOVSA time axis")


def _freqs(hdul):
    for name in ("FREQS", "FREQ", "FREQUENCIES"):
        if name in hdul:
            tab = hdul[name].data
            if "FREQUENCY" in (tab.dtype.names or []):
                return ensure_mhz(np.asarray(tab["FREQUENCY"], dtype=float),
                                  hdul[name].header.get("TUNIT1", "GHz"))
            if "FREQ_GHZ" in (tab.dtype.names or []):
                return np.asarray(tab["FREQ_GHZ"], dtype=float) * 1.0e3
    h = hdul[0].header
    if "CRVAL2" in h and "CDELT2" in h:
        nf = hdul[0].data.shape[0]
        return ensure_mhz((np.arange(nf) - h.get("CRPIX2", 1) + 1) * h["CDELT2"] + h["CRVAL2"],
                          h.get("CUNIT2", "GHz"))
    raise ValueError("could not derive EOVSA frequency axis")
