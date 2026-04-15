"""
Step07a — Build arc difference image (slits-on minus slits-off)

PURPOSE
-------
Construct a full-frame arc-lamp difference image to isolate emission lines
from the calibration lamp while removing continuum/background signal.

The result is the canonical input for all subsequent wavelength calibration
steps (Step07b onward).

This step mirrors the quartz-difference logic used in Step05.

METHOD
------
Given two arc exposures:
  - ARC_SLITS_ON  : arc lamp exposure with slit pattern illuminated
  - ARC_SLITS_OFF : arc exposure representing background/continuum

we compute:

    ArcDiff = ARC_SLITS_ON - ARC_SLITS_OFF

This enhances emission lines and suppresses continuum/background structure.

INPUTS
------
From Step03.5 (rowstripe-corrected frames):

    config.ST03P5_ROWSTRIPE / config.ARC_SLITS_ON
    config.ST03P5_ROWSTRIPE / config.ARC_SLITS_OFF

These must:
  - be on the same detector geometry (same orientation, size)
  - already include bias correction, CR cleaning, and rowstripe correction
  - be aligned (no relative shifts)

OUTPUT
------
Written to:

    config.MASTER_ARC_DIFF

Typical filename:
    ArcDiff_<ON>_minus_<OFF>.fits

This is the canonical full-frame arc image used by:
  - Step07b (rectification into slitlets)
  - Step07c–07h (wavelength solution derivation)

HEADER CONTENT
--------------
The output FITS header includes:

    STAGE     = '07a'
    ARCDIFF   = 'ON-OFF'
    ARC_ON    = name of slits-on exposure
    ARC_OFF   = name of slits-off exposure
    ARC_ON_P  = full path to slits-on file
    ARC_OFF_P = full path to slits-off file

Optional:
    ROT180    = True  (if Step03.5 enforces orientation)

ASSUMPTIONS
-----------
- Both arc frames are already processed through Step03.5 and share identical geometry
- Subtraction is valid (no scaling needed between ON and OFF frames)
- No additional normalization or filtering is applied at this stage

NOTES
-----
- This step must preserve exact pixel geometry; no resampling is allowed
- Any geometric transformations must be deferred to Step07b (rectification)
- The resulting image should show sharp emission lines with minimal background

RUN
---
From repo root:

    PYTHONPATH=. python pipeline/step07_wavecal/step07a_make_arc_diff.py

"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from astropy.io import fits


THIS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR.parent.parent  # ...
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import config 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(Path(__file__).stem)

import numpy as np


def main():
#    config.ensure_dirs()

    in_dir = Path(config.ST03P5_ROWSTRIPE)
    arc_on  = in_dir / config.ARC_SLITS_ON
    arc_off = in_dir / config.ARC_SLITS_OFF

    if not arc_on.exists():
        raise FileNotFoundError(arc_on)
    if not arc_off.exists():
        raise FileNotFoundError(arc_off)

    with fits.open(arc_on) as hdul:
        lines = hdul[0].data.astype(np.float32)
        hdr = hdul[0].header.copy()

    with fits.open(arc_off) as hdul:
        background = hdul[0].data.astype(np.float32)

    if lines.shape != background.shape:
        raise ValueError(f"Shape mismatch: {lines.shape} vs {background.shape}")

    diff = (lines - background).astype(np.float32)

    hdr = hdr.copy()
    hdr.add_history(f"Arc subtraction: {arc_on.name} - {arc_off.name}")
    hdr["STAGE"] = ("07a", "Pipeline stage")
    hdr["ARCDIFF"] = ("ON-OFF", "Arc diff convention: slits_on - slits_off")

    hdr["ARC_ON"]  = (arc_on.name,  "Arc slits-on exposure")
    hdr["ARC_OFF"] = (arc_off.name, "Arc slits-off exposure")

    hdr["ARC_ON_P"]  = (str(config.ARC_SLITS_ON),  "Full path to arc ON")
    hdr["ARC_OFF_P"] = (str(config.ARC_SLITS_OFF), "Full path to arc OFF")

    out_dir = Path(config.ST07_WAVECAL)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = Path(config.MASTER_ARC_DIFF)
    fits.PrimaryHDU(diff, header=hdr).writeto(out_file, overwrite=True)

    print("Wrote:", out_file)

if __name__ == "__main__":
    main()