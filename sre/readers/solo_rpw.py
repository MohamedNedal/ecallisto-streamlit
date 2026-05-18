"""Solar Orbiter / RPW reader.

Targets the L2 survey products available from SOAR / CDAWeb:

  solo_l2_rpw-tnr-surv_*.cdf       (4 kHz – 1 MHz)
  solo_l2_rpw-hfr-surv_*.cdf       (375 kHz – 16 MHz)

Variables:
  Epoch                : CDF_TT2000
  FREQUENCY            : channel frequencies (kHz, shape (n_freq,) or 2-D)
  SENSOR_CONFIG        : antenna config (for diagnostics)
  AGC1 / AGC2          : received voltage PSD per antenna (V^2/Hz)
  TNR_BAND             : (TNR only) which of the four sub-bands each record uses
                         — TNR records sweep through bands A/B/C/D; the reader
                         stitches the four into a single freq axis per timestep.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sre.readers.base import resolve_path
from sre.spectrum import DynamicSpectrum
from sre.utils import cdf_epoch_to_datetime64, ensure_mhz


def read(path, sensor: str = "AGC1", product: str = "auto", **_) -> DynamicSpectrum:
    """Load a Solar Orbiter / RPW CDF.

    Parameters
    ----------
    sensor : 'AGC1' | 'AGC2'
        Which antenna AGC to use.
    product : 'tnr' | 'hfr' | 'auto'
    """
    import cdflib

    p = resolve_path(path)
    product = _detect_product(p.name) if product == "auto" else product.lower()
    cdf = cdflib.CDF(str(p))

    epoch = cdf.varget("Epoch")
    times = cdf_epoch_to_datetime64(epoch)

    freq_raw = np.asarray(cdf.varget("FREQUENCY"), dtype=float)  # kHz
    if freq_raw.ndim == 1:
        freqs = ensure_mhz(freq_raw, "kHz")
        data = np.asarray(cdf.varget(sensor), dtype=np.float32)  # (n_time, n_freq)
        return DynamicSpectrum(
            data=data.T, times=times, frequencies=freqs,
            instrument=f"SolO/RPW-{product.upper()}", unit="V^2 Hz^-1",
            metadata={"sensor": sensor, "product": product.upper()},
        )

    # 2-D FREQUENCY: per-record band selection (typical of TNR sweep)
    return _stitch_bands(cdf, sensor, product, times, freq_raw)


def _stitch_bands(cdf, sensor, product, times, freq_raw) -> DynamicSpectrum:
    import cdflib  # noqa: F401
    raw = np.asarray(cdf.varget(sensor), dtype=np.float32)  # (n_time, n_freq_per_rec)
    # Build a global, unique, sorted frequency axis from the union of values.
    unique_f_khz = np.unique(np.round(freq_raw.ravel(), decimals=3))
    unique_f_khz = unique_f_khz[unique_f_khz > 0]
    freqs = ensure_mhz(unique_f_khz, "kHz")

    f_to_idx = {f: i for i, f in enumerate(unique_f_khz)}
    out = np.full((freqs.size, times.size), np.nan, dtype=np.float32)
    for i_rec in range(times.size):
        for j_ch, fkhz in enumerate(np.round(freq_raw[i_rec], decimals=3)):
            if fkhz <= 0:
                continue
            out[f_to_idx[fkhz], i_rec] = raw[i_rec, j_ch]

    return DynamicSpectrum(
        data=out, times=times, frequencies=freqs,
        instrument=f"SolO/RPW-{product.upper()}",
        unit="V^2 Hz^-1",
        metadata={"sensor": sensor, "product": product.upper(), "stitched_bands": True},
    )


def _detect_product(name: str) -> str:
    low = name.lower()
    if "hfr" in low:
        return "hfr"
    if "tnr" in low:
        return "tnr"
    raise ValueError(f"cannot infer product from filename {name!r}")
