#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:11:33 2026

@author: robberto

QC script that scans all SLIT* extensions in your Step08 Extract1D MEF and prints a ranked list of “most suspicious” slits based on:

Sky baseline inflation: R_sky = median(SKY upper window) / median(SKY lower window)

Flux dip: R_flux = median(FLUX upper) / median(FLUX lower)

Extraction health: finite fractions, median NSKY/NOBJ, noise ratio opt/box (optional; uses TRACECOORDS)

It automatically chooses “lower” and “upper” windows as fractions of the spectrum so it works for different ny.


RETURNS DIAGNOSTIC INCLUDING E.G.
Flagged (R_sky>2 AND R_flux<0.6): 4 / 32 slits
Worst flagged: SLIT020, SLIT022, SLIT062, SLIT032

to run

runfile("QC_step08_scan_all_slits.py", \
args="../../reduced/08_extract1d/Extract1D_optimal_ridgeguided_POOLSKY.fits \
  --tracecoords ../../reduced/06_science/FinalScience_dolidze_ADUperS_reg_pixflatcorr_clipped_EVEN_tracecoords.fits \
  --wobj 3 --top 25")

"""

#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from astropy.io import fits

def robust_median(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.nan if x.size == 0 else float(np.median(x))

def window_masks(y, lo_frac=(0.15, 0.35), hi_frac=(0.65, 0.85)):
    """Return boolean masks for lower/upper windows based on fractional y-range."""
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

def mad_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extract1d_fits", type=Path, help="Step08 Extract1D MEF")
    ap.add_argument("--tracecoords", type=Path, default=None,
                    help="Optional TRACECOORDS MEF to compute box + noise ratio")
    ap.add_argument("--wobj", type=int, default=3, help="half-width for box extraction (px)")
    ap.add_argument("--min_rows", type=int, default=500, help="minimum rows to consider a slit")
    ap.add_argument("--top", type=int, default=30, help="how many slits to print")
    args = ap.parse_args()

    with fits.open(args.extract1d_fits) as hdul:
        slit_exts = [hdu.name for hdu in hdul[1:] if hdu.name.startswith("SLIT")]
        rows = []

        # If provided, open tracecoords once
        tr_hdul = fits.open(args.tracecoords) if args.tracecoords else None

        for slit in slit_exts:
            d = hdul[slit].data
            y   = d["YPIX"]
            sky = d["SKY"].astype(float)
            flx = d["FLUX"].astype(float)
            x0  = d["X0"].astype(float) if "X0" in d.names else None
            nobj = d["NOBJ"].astype(float) if "NOBJ" in d.names else None
            nsky = d["NSKY"].astype(float) if "NSKY" in d.names else None
            varf = d["VAR"].astype(float) if "VAR" in d.names else None

            if len(y) < args.min_rows:
                continue

            lo, hi = window_masks(y)

            sky_lo = robust_median(sky[lo])
            sky_hi = robust_median(sky[hi])
            flx_lo = robust_median(flx[lo])
            flx_hi = robust_median(flx[hi])

            r_sky  = sky_hi / sky_lo if np.isfinite(sky_lo) and sky_lo != 0 else np.nan
            r_flux = flx_hi / flx_lo if np.isfinite(flx_lo) and flx_lo != 0 else np.nan

            frac_flux = float(np.isfinite(flx).mean())
            frac_sky  = float(np.isfinite(sky).mean())
            med_nobj  = robust_median(nobj) if nobj is not None else np.nan
            med_nsky  = robust_median(nsky) if nsky is not None else np.nan
            sky_mad   = mad_sigma(sky)

            noise_ratio = np.nan
            if tr_hdul is not None and x0 is not None and slit in tr_hdul:
                img = np.asarray(tr_hdul[slit].data, float)
                box = compute_box_nosub(img, x0, args.wobj)
                # detrend with a running median (simple, no scipy)
                def running_median(a, win=301):
                    a = np.asarray(a, float)
                    out = np.full_like(a, np.nan)
                    half = win//2
                    for i in range(len(a)):
                        lo_i = max(0, i-half)
                        hi_i = min(len(a), i+half+1)
                        out[i] = np.nanmedian(a[lo_i:hi_i])
                    return out
                if np.isfinite(box).sum() > 100 and np.isfinite(flx).sum() > 100:
                    tb = running_median(box, 301)
                    tf = running_median(flx, 301)
                    rb = box - tb
                    rf = flx - tf
                    m = np.isfinite(rb) & np.isfinite(rf)
                    if m.sum() > 100:
                        noise_ratio = float(np.nanstd(rf[m]) / np.nanstd(rb[m]))

            rows.append(dict(
                slit=slit,
                R_sky=r_sky,
                R_flux=r_flux,
                SKY_lo=sky_lo, SKY_hi=sky_hi,
                FLUX_lo=flx_lo, FLUX_hi=flx_hi,
                frac_flux=frac_flux,
                frac_sky=frac_sky,
                med_nobj=med_nobj,
                med_nsky=med_nsky,
                sky_mad=sky_mad,
                noise_ratio=noise_ratio,
            ))

        if tr_hdul is not None:
            tr_hdul.close()

    # Rank by “badness”: high sky inflation + low flux ratio
    def score(r):
        rs = r["R_sky"]
        rf = r["R_flux"]
        if not np.isfinite(rs): rs = 1.0
        if not np.isfinite(rf): rf = 1.0
        return (rs) / max(rf, 1e-6)

    rows_sorted = sorted(rows, key=score, reverse=True)

    print(f"\nFile: {args.extract1d_fits}")
    if args.tracecoords:
        print(f"TRACECOORDS: {args.tracecoords}  (box/noise_ratio enabled)")
    print("\nTop suspicious slits (ranked by R_sky / R_flux):\n")
    print(f"{'SLIT':8s} {'R_sky':>8s} {'R_flux':>8s} {'SKYlo':>9s} {'SKYhi':>9s} {'FLO':>9s} {'FHI':>9s} {'NOBJ':>6s} {'NSKY':>6s} {'nratio':>8s}")
    print("-"*92)
    for r in rows_sorted[:args.top]:
        print(f"{r['slit']:8s} {r['R_sky']:8.2f} {r['R_flux']:8.2f} "
              f"{r['SKY_lo']:9.4g} {r['SKY_hi']:9.4g} {r['FLUX_lo']:9.4g} {r['FLUX_hi']:9.4g} "
              f"{r['med_nobj']:6.1f} {r['med_nsky']:6.1f} {r['noise_ratio']:8.3f}")

    # Quick count of “bad”
    bad = [r for r in rows if np.isfinite(r["R_sky"]) and np.isfinite(r["R_flux"]) and (r["R_sky"] > 2.0) and (r["R_flux"] < 0.6)]
    print(f"\nFlagged (R_sky>2 AND R_flux<0.6): {len(bad)} / {len(rows)} slits")
    if len(bad) > 0:
        print("Worst flagged:", ", ".join([b["slit"] for b in sorted(bad, key=score, reverse=True)[:10]]))

if __name__ == "__main__":
    main()