"""Minimal programmatic example — load, process, plot, export.

Run with the path to one of your local files, e.g.

    python examples/quickstart.py /data/callisto/BLEN7_20170910_063000_59.fit.gz callisto
"""
from __future__ import annotations

import sys

from sre.readers import read


def main(path: str, instrument: str) -> None:
    ds = read(instrument, path)
    print("Loaded:")
    for k, v in ds.summary().items():
        print(f"  {k:25s} {v}")

    ds_clean = ds.subtract_background(method="quiet_window")
    ds_clean = ds_clean.downsample(time_factor=2, freq_factor=2)

    fig = ds_clean.plot(cmap="inferno", vmin_pct=5, vmax_pct=99)
    out = "quickstart_output.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nFigure written to {out}")

    ds_clean.save_fits("quickstart_output.fits")
    print("Processed array written to quickstart_output.fits")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} <path> <instrument-key>")
    main(sys.argv[1], sys.argv[2])
