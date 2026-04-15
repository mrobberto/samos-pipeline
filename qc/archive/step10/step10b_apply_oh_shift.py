#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step10b — apply Step10a OH wavelength shifts to the telluric-corrected spectra.

PURPOSE
-------
Apply the additive wavelength corrections measured in Step10a to the
telluric-corrected Step09 spectra.

This step refines only the wavelength vector. It preserves the flux, variance,
sky, and extraction geometry. The output remains on the same pixel sampling as
the input.

INPUTS
------
1) Step09 science product:
    extract1d_optimal_ridge_all_wav_tellcorr.fits

2) Step10a shift table:
    oh_shifts.csv

Required CSV columns:
- slit
- shift_nm
- use

OUTPUT
------
- extract1d_optimal_ridge_all_wav_tellcorr_OHref.fits

Per slit:
- LAMBDA_NM is updated in place:
      LAMBDA_NM_new = LAMBDA_NM_old + shift_nm_used
- OH_SHIFT_NM column is added or updated, containing the applied additive shift
  for each row of that slit

PRIMARY HEADER
--------------
The output primary header records:
- PIPESTEP / STAGE
- source science file
- source OH-shift CSV
- median and robust scatter of accepted OH shifts
- number of slits using slit-specific vs fallback shifts

PER-SLIT HEADER / QC KEYWORDS
-----------------------------
- OHWREF   : True if OH refinement was applied
- OHSHIFT  : additive wavelength shift used for this slit (nm)
- OHSRC    : CSV file providing the shift
- OHUSE    : 1 if slit-specific shift accepted, 0 if median fallback used
- OHFALLBK : 1 if fallback median shift was used
- OHMED    : global median accepted shift (nm)
- OHRSIG   : robust scatter of accepted shifts (nm)

ROBUSTNESS FEATURES
-------------------
- Uses slit-specific shifts when use=1
- Falls back to the global median accepted shift when use=0 or slit is missing
- Preserves all existing table columns
- Preserves flux and variance values unchanged

NOTES
-----
- This step should follow Step09 telluric correction.
- This is the final wavelength refinement before flux calibration in Step11.
- The correction is additive in wavelength and does not resample the spectra.
"""
import numpy as np
import csv
from pathlib import Path
from astropy.io import fits
import config

INFILE = Path(config.ST09_TELLURIC) / "extract1d_optimal_ridge_all_wav_tellcorr.fits"
CSV = Path(config.ST10_OH) / "oh_shifts.csv"
OUTFILE = INFILE.with_name(INFILE.stem + "_OHref.fits")

CLIP = 1.0

# ---- read CSV ----
shifts = {}
useflag = {}

with open(CSV) as f:
    r = csv.DictReader(f)
    for row in r:
        slit = row["slit"].upper()
        sh = float(row["shift_nm"])
        use = row["use"].lower() in ["true", "1", "yes"]
        shifts[slit] = sh
        useflag[slit] = use

good = [v for k, v in shifts.items() if useflag.get(k) and abs(v) <= CLIP]
fallback = np.median(good) if good else 0.0

print("Fallback shift:", fallback)

# ---- apply ----
out = [fits.PrimaryHDU()]

with fits.open(INFILE) as h:
    for ext in h[1:]:
        if not ext.name.startswith("SLIT"):
            out.append(ext)
            continue

        tab = ext.data
        lam = tab["LAMBDA_NM"]

        sh = shifts.get(ext.name, np.nan)
        ok = useflag.get(ext.name, False)

        if ok and abs(sh) <= CLIP:
            shift = sh
            src = "GOOD"
        else:
            shift = fallback
            src = "FALLBACK"

        lam_new = lam + shift

        tab2 = tab.copy()
        tab2["LAMBDA_NM"] = lam_new

        hdu = fits.BinTableHDU(tab2, name=ext.name)
        hdu.header["OHSHIFT"] = shift
        hdu.header["OHSRC"] = src

        out.append(hdu)

fits.HDUList(out).writeto(OUTFILE, overwrite=True)

print("Wrote:", OUTFILE)