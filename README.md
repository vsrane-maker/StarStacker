# StarStacker

Astrophotography frame stacking pipeline with optional AI-assisted stages.

## Planned architecture

```
Raw Frames -> Preprocessing -> [AI Frame Rejection] -> Calibration
    -> [AI-Assisted Alignment] -> Normalization -> Stacking
    -> [AI Denoising] -> Output
```

Bracketed stages are optional and AI-assisted; the user can opt into them or
run the classical pipeline path instead.

**Implemented so far:** Raw Frames, Preprocessing.

## Project layout

```
starstacker/
  io/               Raw frame loading (FITS, DSLR RAW) + RawFrame data model
  preprocessing/    Debayer, hot-pixel removal, bit-depth normalization
  pipeline.py       Top-level orchestrator; stages beyond preprocessing are stubs
scripts/
  run_pipeline.py   CLI: load + preprocess a directory of raw frames
data/
  raw/{lights,darks,flats,bias}/   Drop input frames here (gitignored)
  processed/                        Pipeline output (gitignored)
tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
python scripts/run_pipeline.py --raw-dir data/raw/lights --output-dir data/processed
```

## Tests

```bash
pytest
```

## Supported input formats

- FITS: `.fits`, `.fit`, `.fts`
- DSLR/mirrorless RAW: `.cr2`, `.cr3`, `.nef`, `.arw`, `.dng`

Frame type (light/dark/flat/bias) is inferred from the FITS `IMAGETYP`
header when present, otherwise from the containing directory name (e.g.
`data/raw/darks/`).
