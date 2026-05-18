"""Reader registry and dispatch.

Every reader module exposes a top-level `read(path, **kwargs) -> DynamicSpectrum`.
The registry below maps the canonical instrument key to that function. Adding a
new instrument means dropping a module in this package and registering it here.
"""
from __future__ import annotations

from typing import Callable

from sre.spectrum import DynamicSpectrum

from . import (
    callisto, lofar, nda, orfees, nenufar, ovsa,
    psp_fields, solo_rpw, stereo_waves, wind_waves,
)


READERS: dict[str, Callable[..., DynamicSpectrum]] = {
    "callisto": callisto.read,
    "lofar": lofar.read,
    "ilofar": lofar.read,                  # I-LOFAR uses the same dynspec format
    "nda": nda.read,
    "orfees": orfees.read,
    "nenufar": nenufar.read,
    "ovsa": ovsa.read,
    "psp_fields": psp_fields.read,
    "solo_rpw": solo_rpw.read,
    "stereo_waves": stereo_waves.read,
    "wind_waves": wind_waves.read,
}


INSTRUMENT_LABELS: dict[str, str] = {
    "callisto": "e-CALLISTO",
    "lofar": "LOFAR (Solar KSP dynspec)",
    "ilofar": "I-LOFAR",
    "nda": "Nançay Decameter Array (NDA)",
    "orfees": "ORFEES",
    "nenufar": "NenuFAR",
    "ovsa": "OVSA / EOVSA",
    "psp_fields": "Parker Solar Probe / FIELDS",
    "solo_rpw": "Solar Orbiter / RPW",
    "stereo_waves": "STEREO / Waves",
    "wind_waves": "WIND / Waves",
}


def read(instrument: str, path: str, **kwargs) -> DynamicSpectrum:
    key = instrument.lower().replace("-", "_").replace("/", "_")
    if key not in READERS:
        raise KeyError(
            f"no reader for {instrument!r}. "
            f"Known instruments: {sorted(READERS)}"
        )
    return READERS[key](path, **kwargs)


__all__ = ["READERS", "INSTRUMENT_LABELS", "read"]
