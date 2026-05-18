"""Station registry: loads curated coordinates and merges in observed FITS metadata."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATIONS_JSON = DATA_DIR / "stations.json"


@dataclass
class Station:
    code: str
    name: str
    city: str
    country: str
    lat: float
    lon: float
    altitude_m: float = 0.0
    freq_min_mhz: Optional[float] = None
    freq_max_mhz: Optional[float] = None
    nominal_cadence_s: Optional[float] = None
    # populated when a file becomes available
    observed_freq_min_mhz: Optional[float] = None
    observed_freq_max_mhz: Optional[float] = None
    observed_cadence_s: Optional[float] = None
    observed_freq_cadence_mhz: Optional[float] = None
    observed_time_start: Optional[str] = None
    observed_time_end: Optional[str] = None
    n_files: int = 0
    file_urls: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return asdict(self)


def load_stations(path: Path = STATIONS_JSON) -> Dict[str, Station]:
    """Load the curated stations.json into a {code: Station} dict."""
    with open(path) as f:
        raw = json.load(f)
    out: Dict[str, Station] = {}
    for entry in raw["stations"]:
        s = Station(**entry)
        out[s.code.upper()] = s
    return out


def match_station(file_station_code: str, registry: Dict[str, Station]) -> Optional[Station]:
    """
    e-CALLISTO filenames use station tags that don't always match our dict keys
    one-to-one (e.g. 'BIR' vs 'BIR-A2'). Try exact then case-insensitive
    prefix/substring match.
    """
    key = file_station_code.upper()
    if key in registry:
        return registry[key]
    # exact uppercase
    for k, v in registry.items():
        if k.upper() == key:
            return v
    # prefix / contained
    candidates = [v for k, v in registry.items() if k.upper().startswith(key) or key.startswith(k.upper())]
    if candidates:
        # prefer shortest-name match (closer)
        return sorted(candidates, key=lambda s: len(s.code))[0]
    return None
