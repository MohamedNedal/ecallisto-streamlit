"""Time and frequency utilities."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def cdf_epoch_to_datetime64(epoch_array) -> np.ndarray:
    """Convert CDF_EPOCH / CDF_TT2000 / CDF_EPOCH16 to numpy datetime64[ns].

    cdflib already returns ISO-format strings via `cdfepoch.encode`, but for
    large arrays it is faster to delegate to cdflib's `to_datetime` which
    yields a list of python datetimes we can hand to pandas.
    """
    import cdflib
    dt = cdflib.cdfepoch.to_datetime(epoch_array)
    return pd.to_datetime(dt).to_numpy().astype("datetime64[ns]")


def ensure_mhz(freq: np.ndarray, unit: str) -> np.ndarray:
    u = unit.lower().strip()
    if u in ("mhz", "megahertz"):
        return freq
    if u in ("hz", "hertz"):
        return freq / 1.0e6
    if u in ("khz", "kilohertz"):
        return freq / 1.0e3
    if u in ("ghz", "gigahertz"):
        return freq * 1.0e3
    raise ValueError(f"unknown frequency unit {unit!r}")


def to_pandas_timestamps(times: Iterable) -> np.ndarray:
    return pd.to_datetime(list(times)).to_numpy().astype("datetime64[ns]")
