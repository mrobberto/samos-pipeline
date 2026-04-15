#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:57:21 2026

@author: robberto
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QC for Step08c wavelength attachment.

What it does
------------
1) Opens extract1d_optimal_ridge_all_wav.fits
2) For each SLIT###:
   - reads LAMBDA_NM
   - uses FLUX_APCORR if present, else FLUX
   - normalizes spectrum locally
3) Produces:
   - overlay plot around O2 B band  (685-690 nm)
   - overlay plot around O2 A band  (758-770 nm)
   - summary plot of A-band centroid/minimum vs slit number
4) Prints scatter statistics for A-band alignment

Usage
-----
From repo root:

PYTHONPATH=. python ./pipeline/step08_extract1d/qc_step08c_wavelength_alignment.py

Optional:
PYTHONPATH=. python ./pipeline/step08_extract1d/qc_step08c_wavelength_alignment.py \
    --in ./_Run8_Science_2026_01/SAMI/Dolidze25/reduced/08_extract1d/extract1d_optimal_ridge_all_wav.fits

"""

#from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


# ----------------------------
# defaults
# ----------------------------
BAND_B = (685.0, 690.0)
BAND_A = (758.0, 770.0)

# local continuum windows used for normalization
CONT_B = ((680.0, 684.0), (691.0, 695.0))
CONT_A = ((752.0, 757.0), (772.0, 780.0))

# expected telluric centers (for visual reference only)
O2_B_REF = 686.7
O2_A_REF = 760.5


def parse_args():
    ap = argparse.ArgumentParser(description="QC Step08c wavelength alignment")
    ap.add_argument(
        "--in",
        dest="infile",
        type=str,
        default="",
        help="Input wavelength-attached Extract1D FITS",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Directory for PNG outputs (default: same as input file)",
    )
    ap.add_argument(
        "--maxslits",
        type=int,
        default=0,
        help="Optional max number of slits to plot (0 = all)",
    )
    return ap.parse_args()


def is_slit_ext(hdu) -> bool:
    return (hdu.name or "").upper().startswith("SLIT")


def slit_num(name: str) -> int:
    m = re.match(r"SLIT(\d+)", (name or "").upper())
    return int(m.group(1)) if m else 10**9


def robust_median(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.median(x))


def mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def get_flux_and_var(tab) -> tuple[np.ndarray, np.ndarray]:
    names = [n.upper() for n in tab.columns.names]

    if "FLUX_APCORR" in names:
        flux = np.asarray(tab.data["FLUX_APCORR"], float)
    else:
        flux = np.asarray(tab.data["FLUX"], float)

    if "VAR_APCORR" in names:
        var = np.asarray(tab.data["VAR_APCORR"], float)
    elif "VAR" in names:
        var = np.asarray(tab.data["VAR"], float)
    else:
        var = np.full_like(flux, np.nan, dtype=float)

    return flux, var


def local_continuum(lam: np.ndarray, flux: np.ndarray, windows) -> float:
    mask = np.zeros(lam.shape, dtype=bool)
    for lo, hi in windows:
        mask |= (lam >= lo) & (lam <= hi)

    vals = flux[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size < 5:
        return np.nan
    return float(np.median(vals))


def normalize_band(lam: np.ndarray, flux: np.ndarray, band, cont_windows):
    ok = np.isfinite(lam) & np.isfinite(flux)
    lam = lam[ok]
    flux = flux[ok]

    if lam.size < 10:
        return None, None

    cont = local_continuum(lam, flux, cont_windows)
    if not np.isfinite(cont) or cont == 0:
        return None, None

    m = (lam >= band[0]) & (lam <= band[1])
    if m.sum() < 5:
        return None, None

    return lam[m], flux[m] / cont


def estimate_absorption_minimum(lam: np.ndarray, fnorm: np.ndarray) -> tuple[float, float]:
    """
    Return:
      lam_min  = wavelength at minimum normalized flux
      depth    = 1 - min(fnorm)
    """
    ok = np.isfinite(lam) & np.isfinite(fnorm)
    if ok.sum() < 5:
        return np.nan, np.nan

    lam = lam[ok]
    fnorm = fnorm[ok]

    i = np.argmin(fnorm)
    return float(lam[i]), float(1.0 - fnorm[i])


def weighted_mean_and_scatter(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() < 2:
        return np.nan, np.nan
    x = x[ok]
    w = w[ok]
    mu = float(np.sum(w * x) / np.sum(w))
    sig = float(np.sqrt(np.sum(w * (x - mu) ** 2) / np.sum(w)))
    return mu, sig


def main():
    args = parse_args()

    st08 = Path(config.ST08_EXTRACT1D)
    infile = Path(args.infile) if args.infile else (st08 / "extract1d_optimal_ridge_all_wav.fits")
    if not infile.exists():
        raise FileNotFoundError(infile)

    outdir = Path(args.outdir) if args.outdir else infile.parent
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []

    with fits.open(infile, memmap=False) as hdul:
        slit_hdus = [h for h in hdul[1:] if is_slit_ext(h)]
        slit_hdus = sorted(slit_hdus, key=lambda h: slit_num(h.name))
        if args.maxslits and args.maxslits > 0:
            slit_hdus = slit_hdus[:args.maxslits]

        for h in slit_hdus:
            names = [n.upper() for n in h.columns.names]
            if "LAMBDA_NM" not in names:
                continue

            slit = h.name.upper()
            sid = slit_num(slit)

            lam = np.asarray(h.data["LAMBDA_NM"], float)
            flux, var = get_flux_and_var(h)

            # B band
            lam_b, fn_b = normalize_band(lam, flux, BAND_B, CONT_B)

            # A band
            lam_a, fn_a = normalize_band(lam, flux, BAND_A, CONT_A)
            a_min, a_depth = (np.nan, np.nan)
            if lam_a is not None and fn_a is not None:
                a_min, a_depth = estimate_absorption_minimum(lam_a, fn_a)

            rows.append(
                dict(
                    slit=slit,
                    slitid=sid,
                    lam=lam,
                    flux=flux,
                    var=var,
                    lam_b=lam_b,
                    fn_b=fn_b,
                    lam_a=lam_a,
                    fn_a=fn_a,
                    a_min=a_min,
                    a_depth=a_depth,
                )
            )

    if not rows:
        raise RuntimeError("No usable SLIT extensions with LAMBDA_NM found.")

    # ----------------------------
    # Plot 1: B-band overlays
    # ----------------------------
    plt.figure(figsize=(10, 6))
    nplot_b = 0
    for r in rows:
        if r["lam_b"] is None:
            continue
        plt.plot(r["lam_b"], r["fn_b"], alpha=0.55, linewidth=1.0)
        nplot_b += 1
    plt.axvline(O2_B_REF, linestyle="--", linewidth=1.0)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized flux")
    plt.title(f"Step08c QC — O2 B band overlay ({nplot_b} slits)")
    plt.ylim(0.4, 1.1)
    plt.grid(True, alpha=0.25)
    out_b = outdir / "QC_step08c_O2_B_overlay.png"
    plt.tight_layout()
    plt.savefig(out_b, dpi=150)
    plt.close()

    # ----------------------------
    # Plot 2: A-band overlays
    # ----------------------------
    plt.figure(figsize=(10, 6))
    nplot_a = 0
    for r in rows:
        if r["lam_a"] is None:
            continue
        plt.plot(r["lam_a"], r["fn_a"], alpha=0.55, linewidth=1.0)
        nplot_a += 1
    plt.axvline(O2_A_REF, linestyle="--", linewidth=1.0)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized flux")
    plt.title(f"Step08c QC — O2 A band overlay ({nplot_a} slits)")
    plt.ylim(0.2, 1.1)
    plt.grid(True, alpha=0.25)
    out_a = outdir / "QC_step08c_O2_A_overlay.png"
    plt.tight_layout()
    plt.savefig(out_a, dpi=150)
    plt.close()

    # ----------------------------
    # Plot 3: A-band minima vs slit
    # ----------------------------
    slitid = np.array([r["slitid"] for r in rows], dtype=int)
    a_min = np.array([r["a_min"] for r in rows], dtype=float)
    a_depth = np.array([r["a_depth"] for r in rows], dtype=float)

    ok = np.isfinite(a_min) & np.isfinite(a_depth) & (a_depth > 0)
    mu, sig = weighted_mean_and_scatter(a_min[ok], a_depth[ok])

    plt.figure(figsize=(10, 5))
    plt.scatter(slitid[ok], a_min[ok], s=np.clip(150 * a_depth[ok], 10, 120), alpha=0.8)
    if np.isfinite(mu):
        plt.axhline(mu, linestyle="--", linewidth=1.2, label=f"weighted mean = {mu:.3f} nm")
    if np.isfinite(O2_A_REF):
        plt.axhline(O2_A_REF, linestyle=":", linewidth=1.0, label=f"ref = {O2_A_REF:.3f} nm")
    plt.xlabel("Slit ID")
    plt.ylabel("A-band minimum wavelength (nm)")
    ttl = "Step08c QC — O2 A-band minimum by slit"
    if np.isfinite(sig):
        ttl += f"   scatter={sig:.3f} nm"
    plt.title(ttl)
    plt.grid(True, alpha=0.25)
    plt.legend()
    out_sc = outdir / "QC_step08c_O2_A_minima_vs_slit.png"
    plt.tight_layout()
    plt.savefig(out_sc, dpi=150)
    plt.close()

    # ----------------------------
    # Print summary
    # ----------------------------
    print()
    print("Step08c QC summary")
    print(f"  input file: {infile}")
    print(f"  slits read: {len(rows)}")
    print(f"  B-band plotted: {nplot_b}")
    print(f"  A-band plotted: {nplot_a}")
    if np.isfinite(mu):
        print(f"  A-band weighted mean minimum: {mu:.4f} nm")
    if np.isfinite(sig):
        print(f"  A-band weighted scatter:      {sig:.4f} nm")
    else:
        print("  A-band weighted scatter:      NaN")

    good = sorted(
        [(r["slit"], r["a_min"], r["a_depth"]) for r in rows if np.isfinite(r["a_min"])],
        key=lambda t: t[0]
    )

    print()
    print("Per-slit A-band minima")
    for slit, amin, depth in good:
        print(f"  {slit}: A_min={amin:8.3f} nm   depth={depth:6.3f}")

    print()
    print("Wrote:")
    print(f"  {out_b}")
    print(f"  {out_a}")
    print(f"  {out_sc}")


if __name__ == "__main__":
    main()