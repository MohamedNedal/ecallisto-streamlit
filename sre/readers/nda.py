"""Nançay Decameter Array (NDA) reader.

CDPP distributes NDA NewRoutine data either as FITS or CDF. This reader handles
the FITS variant (NDA Routine spectra, ~10–80 MHz, 1-s cadence). For the CDF
variant, point sre.readers.nda.read at the .cdf file — it dispatches to a
cdflib path that reads:

  - Epoch              -> times
  - Frequency          -> frequencies (MHz)
  - LL  and  RR        -> stacked into Stokes I

If your file uses different variable or column names, edit the SCHEMA dict.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from sre.readers.base import resolve_path
from sre.spectrum import DynamicSpectrum
from sre.utils import cdf_epoch_to_datetime64, ensure_mhz


SCHEMA = {
    "cdf_epoch": "Epoch",
    "cdf_freq": "Frequency",
    "cdf_pols": ("LL", "RR"),
    "fits_time_col": "TIME",
    "fits_freq_key_crval": "CRVAL2",
    "fits_freq_key_cdelt": "CDELT2",
}


def read(path, polarization: str = "I", **_) -> DynamicSpectrum:
    p = resolve_path(path)
    suffix = p.suffix.lower()
    if suffix == ".cdf":
        return _read_cdf(p, polarization)
    return _read_fits(p)


def _read_cdf(path: Path, polarization: str) -> DynamicSpectrum:
    import cdflib
    cdf = cdflib.CDF(str(path))
    epoch = cdf.varget(SCHEMA["cdf_epoch"])
    times = cdf_epoch_to_datetime64(epoch)
    freq_raw = np.asarray(cdf.varget(SCHEMA["cdf_freq"]))
    freq_attrs = cdf.varattsget(SCHEMA["cdf_freq"])
    freqs = ensure_mhz(freq_raw, freq_attrs.get("UNITS", "MHz"))

    ll = np.asarray(cdf.varget(SCHEMA["cdf_pols"][0]))  # (n_time, n_freq) typical
    rr = np.asarray(cdf.varget(SCHEMA["cdf_pols"][1]))
    if polarization.upper() == "I":
        data = 0.5 * (ll + rr)
    elif polarization.upper() == "V":
        data = 0.5 * (ll - rr)
    elif polarization.upper() in ("LL", "L"):
        data = ll
    elif polarization.upper() in ("RR", "R"):
        data = rr
    else:
        raise ValueError(f"unknown polarization {polarization!r}")
    data = data.T  # to (freq, time)

    return DynamicSpectrum(
        data=data.astype(np.float32),
        times=times,
        frequencies=freqs,
        instrument="NDA Routine",
        unit="dB above background",
        metadata={"polarization": polarization, "station": "Nançay"},
    )


def _read_fits(path: Path) -> DynamicSpectrum:
    with fits.open(path) as hdul:
        hdu = hdul[0] if hdul[0].data is not None else hdul[1]
        data = np.asarray(hdu.data, dtype=np.float32)
        h = hdu.header
        if data.ndim != 2:
            raise ValueError(f"unexpected NDA FITS data shape {data.shape}")

        nf, nt = data.shape  # assume (freq, time)
        t0 = pd.Timestamp(h.get("DATE-OBS", "1970-01-01"))
        dt = float(h.get("CDELT1", 1.0))
        times = (t0 + pd.to_timedelta(np.arange(nt) * dt, unit="s")).to_numpy().astype("datetime64[ns]")

        f0 = float(h.get(SCHEMA["fits_freq_key_crval"], 10.0))
        df = float(h.get(SCHEMA["fits_freq_key_cdelt"], 0.175))
        freqs = ensure_mhz(f0 + np.arange(nf) * df, h.get("CUNIT2", "MHz"))

    return DynamicSpectrum(
        data=data,
        times=times,
        frequencies=freqs,
        instrument="NDA Routine",
        unit=h.get("BUNIT", "arbitrary"),
        metadata={"station": "Nançay"},
    )
