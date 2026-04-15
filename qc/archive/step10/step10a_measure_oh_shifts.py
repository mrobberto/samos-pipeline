#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step10a — measure per-slit OH wavelength zero-point shifts.

PURPOSE
-------
Measure residual wavelength zero-point offsets after Step09 by comparing the
sky-line residual structure across slits in the telluric-corrected spectra.

This step does not modify the science spectra. It derives one additive
wavelength shift per slit, in nm, to be applied by Step10b.

INPUT
-----
- Step09 product:
    extract1d_optimal_ridge_all_wav_tellcorr.fits

Expected columns per slit:
- LAMBDA_NM
- SKY
and, when present, one or more science-flux columns used only for optional QC

OUTPUT
------
- oh_shifts.csv

Required columns:
- slit       : SLIT### extension name
- shift_nm   : additive wavelength correction to apply in Step10b
- use        : 1 if the slit-specific shift is accepted, 0 if Step10b should
               fall back to the global median shift

Optional diagnostic columns may also be written when available, such as:
- corr / r           : similarity metric or cross-correlation strength
- nwin               : number of OH windows contributing
- shift_std_nm       : scatter across windows
- ref_slit           : reference slit used for relative registration

SCIENTIFIC METHOD
-----------------
1) Select wavelength regions containing OH sky residual structure.

2) Compare the per-slit sky spectrum to a reference sky spectrum.

3) Measure the relative wavelength offset in nm from the OH features.

4) Combine window-level estimates into a single per-slit shift.

5) Mark unreliable shifts with use=0 so that Step10b can fall back to the
   robust global median shift.

INTERPRETATION
--------------
- Positive shift_nm means the wavelength vector must be increased:
      LAMBDA_NM_corrected = LAMBDA_NM + shift_nm
- Negative shift_nm means the wavelength vector must be decreased.

NOTES
-----
- This step refines wavelength only; it does not alter flux values.
- The CSV is the authoritative Step10a product and should be consumed by
  Step10b.
- The file is intentionally lightweight and easy to inspect by eye.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
import config

# ---- CONFIG ----
INFILE = Path(config.ST09_TELLURIC) / "extract1d_optimal_ridge_all_wav_tellcorr.fits"
OUTCSV = Path(config.ST10_OH) / "oh_shifts.csv"

WINDOWS = [
    (780, 805),
    (806, 825),
    (845, 875),
    (875, 905),
    (905, 930),
]

GRID_STEP = 0.01
MAX_SHIFT = 2.0
MIN_CORR = 0.18
MIN_NWIN = 2

# ---- UTILS ----
def robust_z(y):
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med))
    sig = 1.4826 * mad if mad > 0 else np.nanstd(y)
    if not np.isfinite(sig) or sig == 0:
        sig = 1.0
    return (y - med) / sig

def interp(lam, flux, grid):
    m = np.isfinite(lam) & np.isfinite(flux)
    if m.sum() < 20:
        return np.full_like(grid, np.nan)
    s = np.argsort(lam[m])
    return np.interp(grid, lam[m][s], flux[m][s], left=np.nan, right=np.nan)

def xcorr_shift(grid, a, b):
    dlam = np.nanmedian(np.diff(grid))
    maxlag = int(MAX_SHIFT / dlam)

    best = (np.nan, -np.inf)
    for lag in range(-maxlag, maxlag + 1):
        if lag < 0:
            x, y = a[-lag:], b[:len(b)+lag]
        elif lag > 0:
            x, y = a[:-lag], b[lag:]
        else:
            x, y = a, b

        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 50:
            continue

        xx = robust_z(x[ok])
        yy = robust_z(y[ok])
        corr = np.nanmean(xx * yy)

        if corr > best[1]:
            best = (lag, corr)

    lag, peak = best
    if not np.isfinite(peak):
        return np.nan, np.nan

    return -lag * dlam, peak

# ---- MAIN ----
with fits.open(INFILE) as h:
    slits = [x.name for x in h[1:] if x.name.startswith("SLIT")]

    ref = slits[len(slits)//2]  # simple default

    lam_ref = h[ref].data["LAMBDA_NM"]
    sky_ref = h[ref].data["SKY"]

    rows = []

    for s in slits:
        lam = h[s].data["LAMBDA_NM"]
        sky = h[s].data["SKY"]

        shifts, weights = [], []

        for lo, hi in WINDOWS:
            grid = np.arange(lo, hi, GRID_STEP)

            a = interp(lam_ref, sky_ref, grid)
            b = interp(lam, sky, grid)

            sh, corr = xcorr_shift(grid, a, b)
            if np.isfinite(sh) and np.isfinite(corr):
                shifts.append(sh)
                weights.append(max(corr, 0))

        if len(shifts) < MIN_NWIN:
            rows.append((s, 0.0, False))
            continue

        shifts = np.array(shifts)
        weights = np.array(weights)

        shift = np.median(shifts)
        use = np.nanmedian(weights) > MIN_CORR

        rows.append((s, float(shift), bool(use)))

df = pd.DataFrame(rows, columns=["slit", "shift_nm", "use"])
df.to_csv(OUTCSV, index=False)

print("Wrote:", OUTCSV)