# Solar Radio Explorer (SRE)

A Streamlit application for loading, inspecting, cleaning, and exporting solar radio
dynamic spectra from ground- and space-based instruments. Built for researchers and
students working on solar radio bursts.

## Supported instruments (v0.1)

Ground-based: e-CALLISTO, LOFAR / I-LOFAR (Solar KSP dynspec FITS), Nançay Decameter
Array (NDA), ORFEES, NenuFAR (BST/Pulsar HDF5), OVSA / EOVSA.

Space-based: Parker Solar Probe / FIELDS (RFS LFR+HFR), Solar Orbiter / RPW (TNR+HFR),
STEREO / Waves (SWAVES), WIND / Waves (RAD1+RAD2+TNR).

v0.1 assumes the data files are already on disk. Remote fetching from CDAWeb,
LOFAR LTA, soleil.i4ds.ch, etc. is planned for v0.2.

## Installation

```bash
git clone https://github.com/<your-user>/solar-radio-explorer.git
cd solar-radio-explorer
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested against Python 3.10–3.12.

## Running the app

```bash
streamlit run app.py
```

This opens the app at `http://localhost:8501`. Workflow inside the app:

1. Pick the instrument from the sidebar.
2. Point to the data file (drag-and-drop upload, or paste an absolute path if the file
   is larger than Streamlit's upload limit).
3. Optionally set the time window of interest (UTC) for cropping on load.
4. Click *Load*. The dynamic spectrum and metadata appear on the main panel.
5. Apply downsampling, cropping, and background subtraction from the *Processing*
   panel. Each step is logged and reversible.
6. Tweak colour map, dynamic range, axis labels in the *Visualisation* panel.
7. Export the figure (PNG / PDF / EPS / SVG) and/or the processed array (NPZ / FITS)
   from the *Export* panel.

## Programmatic use

Every reader returns a `DynamicSpectrum` object, which can also be used outside the
app. See `examples/quickstart.py`.

```python
from sre.readers import read

ds = read("callisto", "BLEN7_20170910_063000_59.fit.gz")
ds = ds.crop(t_start="2017-09-10 06:35", t_end="2017-09-10 06:45")
ds = ds.subtract_background(method="quiet_window",
                            t_quiet=("2017-09-10 06:31", "2017-09-10 06:33"))
ds.plot(cmap="inferno", vmin_pct=5, vmax_pct=99).savefig("burst.pdf")
```

## File format assumptions

Each reader documents the FITS / CDF / HDF5 schema it expects. If your local pipeline
emits a variant, the easiest path is to copy the relevant reader module, adjust the
keyword/variable names, and register it in `sre/readers/__init__.py`.

| Instrument | Container | Notes |
|---|---|---|
| e-CALLISTO | FITS | Uses `radiospectra.CallistoSpectrogram` |
| LOFAR / I-LOFAR | FITS | LOFAR Solar KSP dynspec; expects `TIME` and `FREQ` BinTable columns |
| NDA | FITS / CDF | CDPP routine spectra |
| ORFEES | FITS | obs-nancay.fr archive; BinTable with `STOKESI` |
| NenuFAR | HDF5 | UnDySPuTeD-style `(time, freq)` dataset |
| OVSA / EOVSA | FITS | EOVSA archive format |
| PSP / FIELDS | CDF | RFS LFR + HFR (`psp_fld_l2_rfs_*`) |
| Solar Orbiter / RPW | CDF | `solo_l2_rpw-tnr-surv`, `solo_l2_rpw-hfr-surv` |
| STEREO / Waves | CDF | `stereo_l3_waves` (CDAWeb) |
| WIND / Waves | CDF | `wi_l2_wav_rad1`, `wi_l2_wav_rad2`, `wi_l2_wav_tnr` |

## Architecture

```
sre/
├── spectrum.py        DynamicSpectrum container + crop/downsample/background ops
├── readers/           One reader per instrument; common interface
│   ├── base.py
│   └── <instrument>.py
├── processing.py      Standalone numerical routines (background, downsample, crop)
├── plotting.py        Matplotlib figure builder with publication defaults
└── utils.py           Time/frequency helpers
app.py                 Streamlit UI
```

## Contributing

Pull requests welcome, especially additional reader variants or remote-fetch backends.
Each reader should expose a `read(path, **kwargs) -> DynamicSpectrum` function and be
registered in `sre/readers/__init__.py::READERS`.

## Licence

MIT.
