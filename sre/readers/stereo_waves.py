"""STEREO / Waves (SWAVES) reader.

Targets the L3 average CDF products on CDAWeb:

  sta_l3_wav_lfr_*.cdf
  sta_l3_wav_hfr_*.cdf
  stb_l3_wav_lfr_*.cdf
  stb_l3_wav_hfr_*.cdf

Variables:
  Epoch           : CDF_EPOCH
  FREQUENCY       : channel frequencies (kHz)
  PSD_FLUX        : flux density (W m^-2 Hz^-1) — preferred for science
  PSD_V2_SP       : raw voltage PSD if FLUX absent
"""
from __future__ import annotations

import numpy as np

from sre.readers.base import resolve_path
from sre.spectrum import DynamicSpectrum
from sre.utils import cdf_epoch_to_datetime64, ensure_mhz


def read(path, prefer: str = "flux", **_) -> DynamicSpectrum:
    import cdflib

    p = resolve_path(path)
    cdf = cdflib.CDF(str(p))
    info = cdf.cdf_info()
    zvars = set(info.zVariables + info.rVariables)

    times = cdf_epoch_to_datetime64(cdf.varget("Epoch"))
    freq_khz = np.asarray(cdf.varget("FREQUENCY"), dtype=float)
    if freq_khz.ndim == 2:
        freq_khz = freq_khz[0]
    freqs = ensure_mhz(freq_khz, "kHz")

    data_var = None
    if prefer == "flux" and "PSD_FLUX" in zvars:
        data_var, unit = "PSD_FLUX", "W m^-2 Hz^-1"
    elif "PSD_V2_SP" in zvars:
        data_var, unit = "PSD_V2_SP", "V^2 Hz^-1"
    elif "PSD_FLUX" in zvars:
        data_var, unit = "PSD_FLUX", "W m^-2 Hz^-1"
    else:
        raise KeyError(f"no PSD variable in SWAVES CDF; available: {sorted(zvars)}")

    data = np.asarray(cdf.varget(data_var), dtype=np.float32)  # (n_time, n_freq)

    # Spacecraft inferred from filename (sta vs stb).
    sc = "STEREO-A" if "sta_" in p.name.lower() else ("STEREO-B" if "stb_" in p.name.lower() else "STEREO")
    return DynamicSpectrum(
        data=data.T,
        times=times,
        frequencies=freqs,
        instrument=f"{sc}/SWAVES",
        unit=unit,
        metadata={"data_variable": data_var},
    )
