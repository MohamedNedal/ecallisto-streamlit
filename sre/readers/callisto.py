"""e-CALLISTO FITS reader.

Reads the standard CALLISTO FITS format directly with astropy, with no
dependency on radiospectra (whose public API has shifted between releases).

Format (per the e-CALLISTO documentation):

  HDU 0 (PrimaryHDU)
    DATA      : 2-D image, shape (n_freq, n_time), usually uint8 or int16.
    HEADER    : DATE-OBS ('YYYY/MM/DD'), TIME-OBS ('HH:MM:SS.SSS'),
                DATE-END, TIME-END, INSTRUME (station name),
                CONTENT, BUNIT, etc.

  HDU 1 (BinTableHDU)
    COLUMN TIME      : seconds since (DATE-OBS, TIME-OBS).
                       Sometimes stored as a 2-D record with shape (1, n_time).
    COLUMN FREQUENCY : channel frequencies in MHz, same shape convention.

Some station files (notably from older pipelines) omit HDU 1 and instead use
WCS-style CRVAL/CDELT/CRPIX keywords in the primary header — we fall back to
that when the BinTable is absent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.io import fits

from sre.readers.base import resolve_path
from sre.spectrum import DynamicSpectrum, ensure_freq_time


def read(path, **_) -> DynamicSpectrum:
    p = resolve_path(path)
    with fits.open(p) as hdul:
        primary = hdul[0]
        header = primary.header
        data = np.asarray(primary.data, dtype=np.float32)
        if data is None or data.ndim != 2:
            raise ValueError(
                f"unexpected CALLISTO data shape: "
                f"{None if data is None else data.shape}"
            )

        t0 = _start_time(header)
        axes = _axes_from_bintable(hdul, t0)
        if axes is None:
            axes = _axes_from_wcs(header, data.shape, t0)
        times, freqs = axes

    data = ensure_freq_time(data, freqs, times)
    station = (header.get("INSTRUME") or header.get("ORIGIN") or "e-CALLISTO").strip()

    return DynamicSpectrum(
        data=data,
        times=times,
        frequencies=freqs,
        instrument=f"e-CALLISTO ({station})",
        unit=header.get("BUNIT", "digital number"),
        metadata={
            "station": station,
            "content": header.get("CONTENT", "").strip(),
            "observatory": header.get("ORIGIN", "").strip(),
        },
    )


def _start_time(header) -> pd.Timestamp:
    # DATE-OBS uses 'YYYY/MM/DD' in older CALLISTO files; pandas handles both.
    date_part = header.get("DATE-OBS", "1970/01/01").replace("/", "-")
    time_part = header.get("TIME-OBS", "00:00:00")
    return pd.Timestamp(f"{date_part} {time_part}")


def _axes_from_bintable(hdul, t0: pd.Timestamp):
    if len(hdul) < 2 or not hasattr(hdul[1], "data") or hdul[1].data is None:
        return None
    tab = hdul[1].data
    names = set(tab.dtype.names or ())
    if "TIME" not in names or "FREQUENCY" not in names:
        return None

    # CALLISTO stores these as nested arrays of shape (1, N); squeeze to 1-D.
    t_raw = np.squeeze(np.asarray(tab["TIME"]).astype(float))
    f_raw = np.squeeze(np.asarray(tab["FREQUENCY"]).astype(float))

    times = (t0 + pd.to_timedelta(t_raw, unit="s")).to_numpy().astype("datetime64[ns]")
    return times, f_raw


def _axes_from_wcs(header, shape, t0: pd.Timestamp):
    nf, nt = shape
    cdelt_t = float(header.get("CDELT1", 0.25))
    crval_t = float(header.get("CRVAL1", 0.0))
    crpix_t = float(header.get("CRPIX1", 1.0))
    seconds = (np.arange(nt) + 1 - crpix_t) * cdelt_t + crval_t
    times = (t0 + pd.to_timedelta(seconds, unit="s")).to_numpy().astype("datetime64[ns]")

    # CDELT2 is in MHz for CALLISTO.
    cdelt_f = float(header.get("CDELT2", 1.0))
    crval_f = float(header.get("CRVAL2", 0.0))
    crpix_f = float(header.get("CRPIX2", 1.0))
    freqs = (np.arange(nf) + 1 - crpix_f) * cdelt_f + crval_f
    return times, freqs