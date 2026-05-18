"""WIND / Waves reader.

Targets the L2 60-second average CDFs distributed via CDAWeb:

  wi_l2_wav_rad1_*.cdf   (20 – 1040 kHz)
  wi_l2_wav_rad2_*.cdf   (1.075 – 13.825 MHz)
  wi_l2_wav_tnr_*.cdf    (4 – 256 kHz)

Variables:
  Epoch         : CDF_EPOCH
  Frequency     : channel frequencies (kHz)
  E_VOLTAGE_RAD1 / E_VOLTAGE_RAD2 / E_VOLTAGE_TNR : voltage PSD in dB above background

The dB-above-background scaling is canonical for WAVES; we expose it directly.
"""
from __future__ import annotations

import numpy as np

from sre.readers.base import resolve_path
from sre.spectrum import DynamicSpectrum
from sre.utils import cdf_epoch_to_datetime64, ensure_mhz


PRODUCTS = {"rad1": "E_VOLTAGE_RAD1", "rad2": "E_VOLTAGE_RAD2", "tnr": "E_VOLTAGE_TNR"}


def read(path, product: str = "auto", **_) -> DynamicSpectrum:
    import cdflib

    p = resolve_path(path)
    product = _detect_product(p.name) if product == "auto" else product.lower()
    cdf = cdflib.CDF(str(p))

    times = cdf_epoch_to_datetime64(cdf.varget("Epoch"))
    freqs_khz = np.asarray(cdf.varget("Frequency"), dtype=float)
    if freqs_khz.ndim == 2:
        freqs_khz = freqs_khz[0]
    freqs = ensure_mhz(freqs_khz, "kHz")

    var = PRODUCTS[product]
    data = np.asarray(cdf.varget(var), dtype=np.float32)        # (n_time, n_freq)

    return DynamicSpectrum(
        data=data.T,
        times=times,
        frequencies=freqs,
        instrument=f"WIND/WAVES {product.upper()}",
        unit="dB above background",
        metadata={"product": product.upper()},
    )


def _detect_product(name: str) -> str:
    low = name.lower()
    for key in PRODUCTS:
        if key in low:
            return key
    raise ValueError(
        f"cannot infer WIND/Waves product from filename {name!r}; "
        f"pass product='rad1' | 'rad2' | 'tnr'"
    )
