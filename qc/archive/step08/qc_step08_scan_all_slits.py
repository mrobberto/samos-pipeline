#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step08 QC — scan all slits and rank suspicious cases.

Defaults to the cleaned Step08a parity product and the latest matching
TRACECOORDS MEF for the same set.
"""

import argparse
from pathlib import Path
from glob import glob
import numpy as np
from astropy.io import fits
import config

def robust_median(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.nan if x.size == 0 else float(np.median(x))

def window_masks(y, lo_frac=(0.15, 0.35), hi_frac=(0.65, 0.85)):
    y = np.asarray(y)
    y0, y1 = np.nanmin(y), np.nanmax(y)
    if not np.isfinite(y0) or not np.isfinite(y1) or y1 <= y0:
        m = np.isfinite(y)
        return m, m
    lo0 = y0 + lo_frac[0] * (y1 - y0)
    lo1 = y0 + lo_frac[1] * (y1 - y0)
    hi0 = y0 + hi_frac[0] * (y1 - y0)
    hi1 = y0 + hi_frac[1] * (y1 - y0)
    lo = (y >= lo0) & (y <= lo1)
    hi = (y >= hi0) & (y <= hi1)
    return lo, hi

def compute_box_nosub(trace_img, x0, wobj):
    img = np.asarray(trace_img, float)
    x0 = np.asarray(x0, float)
    ny_img, nx = img.shape
    ny = min(ny_img, x0.size)
    x = np.arange(nx, dtype=float)
    box = np.full(ny_img, np.nan, float)
    for y in range(ny):
        if not np.isfinite(x0[y]):
            continue
        row = img[y]
        dx = x - x0[y]
        m = np.isfinite(row) & (np.abs(dx) <= wobj)
        if np.any(m):
            box[y] = np.nansum(row[m])
    return box

def pick_tracecoords(set_tag):
    matches = sorted(glob(str(Path(config.ST06_SCIENCE) / f"*_{set_tag}_tracecoords.fits")))
    return Path(matches[-1]) if matches else None

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True, choices=["EVEN", "ODD", "even", "odd"])
ap.add_argument("--extract", type=Path, default=None)
ap.add_argument("--tracecoords", type=Path, default=None)
ap.add_argument("--wobj", type=int, default=3)
ap.add_argument("--min_rows", type=int, default=500)
ap.add_argument("--top", type=int, default=30)
args = ap.parse_args()

set_tag = args.set.upper()
extract1d_fits = args.extract if args.extract else Path(config.ST08_EXTRACT1D) / f"extract1d_optimal_ridge_{set_tag.lower()}.fits"
tracecoords = args.tracecoords if args.tracecoords else pick_tracecoords(set_tag)

if not extract1d_fits.exists():
    raise FileNotFoundError(extract1d_fits)
tr_hdul = fits.open(tracecoords) if tracecoords else None

with fits.open(extract1d_fits) as hdul:
    slit_exts = [hdu.name for hdu in hdul[1:] if hdu.name.startswith("SLIT")]
    rows = []
    for slit in slit_exts:
        d = hdul[slit].data
        y = np.asarray(d["YPIX"], float)
        if y.size < args.min_rows:
            continue
        lo, hi = window_masks(y)

        sky = np.asarray(d["SKY"], float)
        flux = np.asarray(d["FLUX"], float)
        x0 = np.asarray(d["X0"], float)
        nobj = np.asarray(d["NOBJ"], float)
        nsky = np.asarray(d["NSKY"], float)

        SKY_lo = robust_median(sky[lo]); SKY_hi = robust_median(sky[hi])
        FLUX_lo = robust_median(flux[lo]); FLUX_hi = robust_median(flux[hi])

        R_sky = SKY_hi / SKY_lo if np.isfinite(SKY_lo) and SKY_lo != 0 else np.nan
        R_flux = FLUX_hi / FLUX_lo if np.isfinite(FLUX_lo) and FLUX_lo != 0 else np.nan

        noise_ratio = np.nan
        if tr_hdul is not None and slit in [h.name.upper() for h in tr_hdul[1:]]:
            img = np.asarray(tr_hdul[slit].data, float)
            box = compute_box_nosub(img, x0, args.wobj)
            m = np.isfinite(box) & np.isfinite(flux)
            if np.any(m):
                sb = np.nanstd(box[m]); so = np.nanstd(flux[m])
                if np.isfinite(sb) and sb > 0:
                    noise_ratio = so / sb

        rows.append({
            "slit": slit,
            "R_sky": R_sky,
            "R_flux": R_flux,
            "SKY_lo": SKY_lo, "SKY_hi": SKY_hi,
            "FLUX_lo": FLUX_lo, "FLUX_hi": FLUX_hi,
            "med_nobj": robust_median(nobj),
            "med_nsky": robust_median(nsky),
            "noise_ratio": noise_ratio,
        })

    if tr_hdul is not None:
        tr_hdul.close()

def score(r):
    rs = r["R_sky"]; rf = r["R_flux"]
    if not np.isfinite(rs): rs = 1.0
    if not np.isfinite(rf): rf = 1.0
    return rs / max(rf, 1e-6)

rows_sorted = sorted(rows, key=score, reverse=True)

print(f"\nFile: {extract1d_fits}")
if tracecoords:
    print(f"TRACECOORDS: {tracecoords}")
print("\nTop suspicious slits (ranked by R_sky / R_flux):\n")
print(f"{'SLIT':8s} {'R_sky':>8s} {'R_flux':>8s} {'SKYlo':>9s} {'SKYhi':>9s} {'FLO':>9s} {'FHI':>9s} {'NOBJ':>6s} {'NSKY':>6s} {'nratio':>8s}")
print("-"*92)
for r in rows_sorted[:args.top]:
    print(f"{r['slit']:8s} {r['R_sky']:8.2f} {r['R_flux']:8.2f} "
          f"{r['SKY_lo']:9.4g} {r['SKY_hi']:9.4g} {r['FLUX_lo']:9.4g} {r['FLUX_hi']:9.4g} "
          f"{r['med_nobj']:6.1f} {r['med_nsky']:6.1f} {r['noise_ratio']:8.3f}")

bad = [r for r in rows if np.isfinite(r["R_sky"]) and np.isfinite(r["R_flux"]) and (r["R_sky"] > 2.0) and (r["R_flux"] < 0.6)]
print(f"\nFlagged (R_sky>2 AND R_flux<0.6): {len(bad)} / {len(rows)} slits")
if len(bad) > 0:
    print("Worst flagged:", ", ".join([b["slit"] for b in sorted(bad, key=score, reverse=True)[:10]]))
