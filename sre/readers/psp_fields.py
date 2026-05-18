"""Parker Solar Probe / FIELDS RFS reader.

Targets the level-2 RFS LFR and HFR survey products:

  psp_fld_l2_rfs_lfr_*.cdf   (10.5 kHz – 1.7 MHz)
  psp_fld_l2_rfs_hfr_*.cdf   (1.3   –  19.2 MHz)

Variables of interest:
  epoch_lfr   /  epoch_hfr             : CDF_TT2000 timestamps
  frequency_lfr_stokes / frequency_hfr_stokes : channel centres (Hz)
  psp_fld_l2_rfs_lfr_auto_averages_ch0_V1V2 / equivalent for HFR
                                        : auto-spectral PSD (V^2/Hz)

Choose channel name with `channel` kwarg. Default = 'V1V2' (the dipole most
commonly used for solar bursts).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sre.readers.base import resolve_path
from sre.spectrum import DynamicSpectrum
from sre.utils import cdf_epoch_to_datetime64, ensure_mhz


def read(path, band: str = "auto", channel: str = "V1V2", **_) -> DynamicSpectrum:
    """Load a PSP/FIELDS RFS CDF.

    Parameters
    ----------
    band : 'lfr' | 'hfr' | 'auto'
        Force LFR or HFR; 'auto' detects from the filename.
    channel : str
        Dipole / channel suffix, e.g. 'V1V2', 'V3V4'.
    """
    import cdflib

    p = resolve_path(path)
    band = _detect_band(p.name) if band == "auto" else band.lower()
    cdf = cdflib.CDF(str(p))

    epoch_var = f"epoch_{band}_auto_averages_ch0_{channel}"
    if epoch_var not in cdf.cdf_info().rVariables + cdf.cdf_info().zVariables:
        # Fall back to the generic epoch name used in older releases.
        epoch_var = f"epoch_{band}"
    freq_var = f"frequency_{band}_auto_averages_ch0_{channel}"
    data_var = f"psp_fld_l2_rfs_{band}_auto_averages_ch0_{channel}"

    times = cdf_epoch_to_datetime64(cdf.varget(epoch_var))
    freq_hz = np.asarray(cdf.varget(freq_var), dtype=float)
    # Some releases store frequency as 2-D (one row per record); collapse.
    if freq_hz.ndim == 2:
        freq_hz = freq_hz[0]
    freqs = ensure_mhz(freq_hz, "Hz")

    psd = np.asarray(cdf.varget(data_var), dtype=np.float32)  # (n_time, n_freq)

    return DynamicSpectrum(
        data=psd.T,
        times=times,
        frequencies=freqs,
        instrument=f"PSP/FIELDS RFS-{band.upper()}",
        unit="V^2 Hz^-1",
        metadata={"channel": channel, "band": band.upper()},
    )


def _detect_band(name: str) -> str:
    low = name.lower()
    if "hfr" in low:
        return "hfr"
    if "lfr" in low:
        return "lfr"
    raise ValueError(f"cannot infer band from filename {name!r}; pass band='lfr' or 'hfr'")
