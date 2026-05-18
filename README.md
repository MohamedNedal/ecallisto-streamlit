# e-CALLISTO Globe Explorer

A Streamlit application for browsing the e-CALLISTO solar radio spectrograph
network, fetching FITS files within a chosen UT window, and processing /
visualising dynamic spectra.

## Features

- UT date/time range picker, with live search against the e-CALLISTO HTTP
  archive at `soleil.i4ds.ch/solarradio/data/2002-20yy_Callisto/`.
- 3D wireframe globe (Plotly Scatter3d). Stations with files in the window are
  drawn as red dots, all others as faint grey dots. Hovering shows the station
  name; clicking opens the details panel.
- Station details panel: code, name, city/country, coordinates, altitude,
  nominal frequency range, nominal cadence, observed UT range in the search
  window, number of files. Once a file is loaded the panel also shows the
  *observed* frequency range and time/frequency cadence from the FITS header.
- One-click download of one or all files into a user-chosen folder.
- File loader (upload `.fit` / `.fit.gz`) and quick preview.
- Processing pipeline: block-mean downsampling in time and frequency, smooth
  cropping on both axes, and a choice of background-subtraction methods:
  - Quiet-Sun window (median or mean)
  - Running median along the time axis
  - Lowest decile per channel
  - Polynomial detrend per channel
- Publication-style visualisation: colour map, vmin/vmax percentile sliders,
  inverted/log frequency axis, grid, colourbar, figure size, DPI.
- Export the figure as PNG / PDF / EPS / SVG and the processed array as `.npz`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Project layout

```
.
├── app.py                  # Streamlit entry point
├── requirements.txt
├── data/
│   └── stations.json       # Curated station registry (extend as needed)
└── ecallisto/
    ├── __init__.py
    ├── stations.py         # Registry loader + station matching
    ├── fetch.py            # Directory walk + FITS download
    ├── globe.py            # Wireframe globe + station markers
    ├── processing.py       # Load FITS, downsample, crop, background
    └── plotting.py         # Matplotlib spectrogram + export
```

## A note on station coordinates

e-CALLISTO FITS headers reliably carry per-file metadata (frequency axis,
cadence, observed time range, instrument short code) but they do **not**
consistently carry geographic coordinates. The globe therefore relies on the
curated `data/stations.json` for lat/lon, and pulls per-file metadata from the
FITS header when files are loaded. Extending the JSON is the easiest way to add
new stations.

## Suggested next features

- Multi-station synchronous plotting: stitched composite spectrogram from
  overlapping stations to extend the frequency coverage (e.g. BIR 10–90 MHz +
  HUMAIN 45–870 MHz).
- Burst-type tagging and labelled time markers from the e-CALLISTO daily
  burst-list text files.
- Frequency–drift fitter for type II/III bursts with manual point selection,
  including an electron-density-model overlay (Newkirk, Saito, Mann) to
  translate the drift into a radial speed.
- RFI mask editor: per-channel toggle plus auto-flag from short time-scale
  statistics.
- Side-by-side comparison with Solar Orbiter / RPW or Wind / WAVES quick-look
  spectra fetched via `sunpy.net.Fido`.
- GOES SXR overlay on the time axis to flag flares co-temporal with bursts.
- Local solar elevation per station: a small per-station "sun above horizon"
  badge in the details panel, computed from station lat/lon + UT.
- Light/Dark theme toggle, and optional Mercator 2D map as an alternative to
  the globe.
- Caching layer (`requests-cache` or `st.cache_data`) for the directory walk so
  repeated searches in the same window are instant.
- Batch mode: queue many UT windows from a CSV and run the full pipeline
  unattended.
