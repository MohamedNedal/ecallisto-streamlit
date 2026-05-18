"""Base helpers for readers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from sre.spectrum import DynamicSpectrum


def resolve_path(path) -> Path:
    """Accept str, Path, or a Streamlit UploadedFile (which has .name + .read())."""
    if hasattr(path, "name") and hasattr(path, "read") and not isinstance(path, (str, Path)):
        return _spool_upload(path)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _spool_upload(uploaded) -> Path:
    """Write a Streamlit UploadedFile to disk *with its original basename*.

    Many readers in this package sniff the filename for band/product/spacecraft
    (e.g. 'lfr' vs 'hfr' in PSP/FIELDS, 'rad1' vs 'rad2' in WIND/Waves,
    'sta_' vs 'stb_' in SWAVES). Using NamedTemporaryFile would mangle that to
    something like 'tmpkpd5952j.cdf' and break the readers. Instead we make a
    fresh temp *directory* and drop the file inside it under its real name.
    """
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="sre_upload_"))
    out = tmp_dir / uploaded.name
    payload = uploaded.getbuffer() if hasattr(uploaded, "getbuffer") else uploaded.read()
    with open(out, "wb") as f:
        f.write(payload)
    return out