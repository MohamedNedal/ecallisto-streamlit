"""LOFAR / I-LOFAR dynamic-spectrum FITS reader.

Targets the dynspec FITS produced by the LOFAR Solar KSP pipeline (and the
matching format used at I-LOFAR). Schema assumed:

  Primary HDU:    2-D image data, shape (n_time, n_freq) or (n_freq, n_time).
                  Header carries DATE-OBS, plus CRVAL/CDELT/CRPIX for both axes.
  Extensions:     Optional BinTable named 'TIMES' (column TIME or TIME_UTC) and
                  'FREQS' / 'FREQUENCIES' (column FREQ or FREQUENCY).

Falls back to WCS-style CRVALn/CDELTn parsing if the BinTable extensions are
absent. If your local pipeline emits something different, adjust the section
marked SCHEMA below or write a sibling reader.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.io import fits

from sre.readers.base import resolve_path
from sre.spectrum import DynamicSpectrum, ensure_freq_time
from sre.utils import ensure_mhz


def read(path, beam: int = 0, **_) -> DynamicSpectrum:
    p = resolve_path(path)
    with fits.open(p) as hdul:
        data, times, freqs, header = _parse(hdul, beam=beam)

    data = ensure_freq_time(data, freqs, times)
    return DynamicSpectrum(
        data=data.astype(np.float32),
        times=times,
        frequencies=freqs,
        instrument=header.get("TELESCOP", "LOFAR"),
        unit=header.get("BUNIT", "arbitrary"),
        metadata={
            "antenna_set": header.get("ANTENNAS", "unknown"),
            "target": header.get("OBJECT", "Sun"),
            "obs_id": header.get("OBS-ID", "unknown"),
            "beam": beam,
        },
    )


def _parse(hdul, beam: int):
    # SCHEMA: prefer named extensions if present.
    primary = hdul[0]
    header = primary.header

    data = primary.data
    if data is None:
        # Some pipelines push the image into HDU 1.
        data = hdul[1].data
        header = hdul[1].header
    data = np.squeeze(np.asarray(data))
    if data.ndim == 3:
        # (n_beam, n_freq, n_time) → pick the requested beam.
        data = data[beam]
    if data.ndim != 2:
        raise ValueError(f"unexpected LOFAR data shape {data.shape}")

    times = _times_from_extensions(hdul) or _times_from_wcs(header, data.shape)
    freqs = _freqs_from_extensions(hdul) or _freqs_from_wcs(header, data.shape)
    if times is None or freqs is None:
        raise ValueError(
            "could not derive time/frequency axes from FITS — expected "
            "BinTable extensions named TIMES/FREQS or WCS keywords"
        )
    return data, times, freqs, header


def _times_from_extensions(hdul):
    for name in ("TIMES", "TIME"):
        if name in hdul:
            tab = hdul[name].data
            col = _first_existing(tab, ("TIME_UTC", "TIME", "EPOCH"))
            if col is not None:
                if np.issubdtype(col.dtype, np.number):
                    # Seconds since DATE-OBS.
                    t0 = pd.Timestamp(hdul[0].header.get("DATE-OBS", "1970-01-01"))
                    return (t0 + pd.to_timedelta(np.asarray(col, dtype=float), unit="s"))\
                        .to_numpy().astype("datetime64[ns]")
                return pd.to_datetime(np.asarray(col, dtype=str)).to_numpy().astype("datetime64[ns]")
    return None


def _freqs_from_extensions(hdul):
    for name in ("FREQS", "FREQUENCIES", "FREQ"):
        if name in hdul:
            tab = hdul[name].data
            col = _first_existing(tab, ("FREQ_MHZ", "FREQUENCY", "FREQ"))
            unit_hdr = hdul[name].header.get("TUNIT1", "MHz")
            if col is not None:
                return ensure_mhz(np.asarray(col, dtype=float), unit_hdr)
    return None


def _times_from_wcs(header, shape):
    if "CRVAL1" not in header or "CDELT1" not in header:
        return None
    nf, nt = shape
    crval = float(header["CRVAL1"])
    cdelt = float(header["CDELT1"])
    crpix = float(header.get("CRPIX1", 1.0))
    t0 = pd.Timestamp(header.get("DATE-OBS", "1970-01-01"))
    offsets = (np.arange(nt) + 1 - crpix) * cdelt + crval
    return (t0 + pd.to_timedelta(offsets, unit="s")).to_numpy().astype("datetime64[ns]")


def _freqs_from_wcs(header, shape):
    if "CRVAL2" not in header or "CDELT2" not in header:
        return None
    nf, nt = shape
    crval = float(header["CRVAL2"])
    cdelt = float(header["CDELT2"])
    crpix = float(header.get("CRPIX2", 1.0))
    unit = header.get("CUNIT2", "MHz")
    freqs = (np.arange(nf) + 1 - crpix) * cdelt + crval
    return ensure_mhz(freqs, unit)


def _first_existing(table, names):
    cols = set(table.dtype.names or [])
    for n in names:
        if n in cols:
            return table[n]
    return None
