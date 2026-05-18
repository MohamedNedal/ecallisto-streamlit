"""Query the e-CALLISTO data server and download FITS files."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# e-CALLISTO public data root. Directory listing is browsable.
DATA_ROOT = "http://soleil.i4ds.ch/solarradio/data/2002-20yy_Callisto/"

# Filename pattern: STATION_YYYYMMDD_HHMMSS_NN.fit[.gz]
# Station code can include letters, digits, and dashes/underscores.
FILE_RE = re.compile(
    r"^(?P<station>[A-Za-z0-9\-]+)_"
    r"(?P<date>\d{8})_"
    r"(?P<time>\d{6})_"
    r"(?P<idx>\d{2,3})\.fit(?:\.gz)?$"
)

# Each e-CALLISTO file typically covers ~15 minutes.
DEFAULT_FILE_DURATION = timedelta(minutes=15)


@dataclass
class FileEntry:
    station: str            # raw code from filename
    start: datetime         # UTC
    end: datetime           # UTC (start + DEFAULT_FILE_DURATION)
    url: str
    filename: str


def _list_directory(url: str, session: requests.Session, timeout: int = 30) -> List[str]:
    """Return the filenames in an Apache directory listing."""
    r = session.get(url, timeout=timeout)
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    out = []
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        if href.endswith(".fit") or href.endswith(".fit.gz"):
            out.append(href)
    return out


def _daterange(start: datetime, end: datetime) -> Iterable[datetime]:
    """Iterate UTC date boundaries spanned by [start, end]."""
    d = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    last = datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
    while d <= last:
        yield d
        d += timedelta(days=1)


def search_availability(
    start: datetime,
    end: datetime,
    session: Optional[requests.Session] = None,
    min_overlap_s: float = 1.0,
) -> List[FileEntry]:
    """
    Walk the daily directories spanned by [start, end] (UTC) and return file
    entries whose [file_start, file_end] window overlaps the request by at
    least `min_overlap_s` seconds.

    A file is *kept* iff:
        overlap = min(file_end, end) - max(file_start, start) >= min_overlap_s
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    if end <= start:
        return []

    sess = session or requests.Session()
    results: List[FileEntry] = []
    min_overlap = timedelta(seconds=min_overlap_s)

    for day in _daterange(start, end):
        day_url = f"{DATA_ROOT}{day.year:04d}/{day.month:02d}/{day.day:02d}/"
        names = _list_directory(day_url, sess)
        for name in names:
            m = FILE_RE.match(name)
            if not m:
                continue
            try:
                dt = datetime.strptime(m["date"] + m["time"], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            file_end = dt + DEFAULT_FILE_DURATION
            overlap = min(file_end, end) - max(dt, start)
            if overlap < min_overlap:
                continue
            results.append(FileEntry(
                station=m["station"],
                start=dt,
                end=file_end,
                url=urljoin(day_url, name),
                filename=name,
            ))
    return results


def group_by_station(entries: List[FileEntry]) -> Dict[str, List[FileEntry]]:
    by: Dict[str, List[FileEntry]] = {}
    for e in entries:
        by.setdefault(e.station.upper(), []).append(e)
    for v in by.values():
        v.sort(key=lambda x: x.start)
    return by


def download_file(
    entry: FileEntry,
    out_dir: Path,
    session: Optional[requests.Session] = None,
    overwrite: bool = False,
    timeout: int = 60,
) -> Path:
    """Download a single FITS file. Returns the local path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    local = out_dir / entry.filename
    if local.exists() and not overwrite:
        return local

    sess = session or requests.Session()
    with sess.get(entry.url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(local, "wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
    return local


def download_many(
    entries: List[FileEntry],
    out_dir: Path,
    progress_cb=None,
    overwrite: bool = False,
) -> List[Path]:
    """Download a list of entries sequentially. progress_cb(done, total, entry)."""
    sess = requests.Session()
    paths: List[Path] = []
    total = len(entries)
    for i, e in enumerate(entries, start=1):
        p = download_file(e, out_dir, session=sess, overwrite=overwrite)
        paths.append(p)
        if progress_cb:
            progress_cb(i, total, e)
    return paths
