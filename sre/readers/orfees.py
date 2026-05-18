"""ORFEES reader.

ORFEES (Observation Radio Frequence pour l'Etude des Eruptions Solaires) data
from obs-nancay.fr are FITS with a primary header (no image) and a BinTable
extension carrying one row per integration. The dynamic spectrum is stored in
the column 'STOKESI' (and 'STOKESV') with shape (n_time, n_freq_total) across
five concatenated frequency bands.

Frequency channels are listed in the primary header as keywords FREQ_B1, FREQ_B2,
... or in extension headers (varies by year). This reader covers the standard
post-2015 schema; if you have an older file, set `freq_cards` manually.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.io import fits

from sre.readers.base import resolve_path
from sre.spectrum import DynamicSpectrum


def read(path, polarization: str = "I", freq_cards=None, **_) -> DynamicSpectrum:
    p = resolve_path(path)
    with fits.open(p) as hdul:
        primary = hdul[0].header
        tab = hdul[1]
        rec = tab.data

        col = "STOKESI" if polarization.upper() == "I" else "STOKESV"
        if col not in (rec.dtype.names or []):
            raise KeyError(f"column {col!r} not in ORFEES FITS; available: {rec.dtype.names}")
        data = np.asarray(rec[col], dtype=np.float32)        # (n_time, n_freq)

        # Time column may be named 'TIME' (seconds of day) or 'EPOCH'.
        if "TIME" in (rec.dtype.names or []):
            sod = np.asarray(rec["TIME"], dtype=float)
            date_obs = pd.Timestamp(primary.get("DATE-OBS", "1970-01-01"))
            # DATE-OBS in ORFEES is the UT date only; combine with seconds of day.
            date_floor = pd.Timestamp(date_obs.date())
            times = (date_floor + pd.to_timedelta(sod, unit="s"))\
                .to_numpy().astype("datetime64[ns]")
        else:
            times = pd.to_datetime(rec["EPOCH"]).to_numpy().astype("datetime64[ns]")

        if freq_cards is None:
            freq_cards = _collect_freq_cards(primary, tab.header)
        freqs = np.concatenate([np.asarray(v, dtype=float) for v in freq_cards])

        if data.shape[1] != freqs.size:
            raise ValueError(
                f"ORFEES freq axis mismatch: data has {data.shape[1]} channels, "
                f"header lists {freqs.size}"
            )

    return DynamicSpectrum(
        data=data.T,                                          # (n_freq, n_time)
        times=times,
        frequencies=freqs,
        instrument="ORFEES",
        unit="arbitrary (raw counts)",
        metadata={
            "polarization": polarization,
            "station": "Nançay",
            "n_bands": len(freq_cards),
        },
    )


def _collect_freq_cards(primary, ext):
    """Pick up FREQ_B1..FREQ_Bn arrays from whichever header carries them."""
    cards = []
    for n in range(1, 12):
        key = f"FREQ_B{n}"
        for hdr in (primary, ext):
            if key in hdr:
                cards.append(np.atleast_1d(hdr[key]))
                break
    if not cards:
        raise KeyError(
            "no FREQ_Bn cards found in ORFEES headers; "
            "pass freq_cards=[arr1, arr2, ...] explicitly"
        )
    return cards
