#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 09:08:58 2026

@author: robberto
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QC08_7plots_batch.py

Batch version of the detailed Step08 viewer.
Creates one PNG per slit in an output folder.

Usage example:
runfile(
    "QC08_7plots_batch.py",
    args="--set EVEN "
         "--file08 ../../reduced/08_extract1d/Extract1D_optimal_ridgeguided_POOLSKY_EVEN.fits "
         "--file06 ../../reduced/06_science/FinalScience_dolidze_ADUperS_pixflatcorr_clipped_EVEN_tracecoords.fits"
)
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
import config


def first_existing(paths):
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def robust_limits(img, p_lo=2.0, p_hi=98.0):
    arr = np.asarray(img, float)
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return 0.0, 1.0
    lo = np.nanpercentile(v, p_lo)
    hi = np.nanpercentile(v, p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        med = np.nanmedian(v)
        sig = np.nanstd(v)
        if not np.isfinite(sig) or sig <= 0:
            sig = 1.0
        return med - 2 * sig, med + 4 * sig
    return float(lo), float(hi)


def infer_file06_from_file08(file08: Path, traceset: str) -> Path | None:
    st06 = Path(config.ST06_SCIENCE)
    tag = traceset.upper()
    candidates = [
        st06 / f"FinalScience_dolidze_ADUperS_pixflatcorr_clipped_{tag}_tracecoords.fits",
        *sorted(st06.glob(f"*_{tag}_tracecoords.fits")),
    ]
    return first_existing(candidates)


def pick_latest_step08(traceset: str) -> Path | None:
    st08 = Path(config.ST08_EXTRACT1D)
    tag = traceset.upper()
    candidates = sorted(st08.glob(f"*{tag}.fits"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def finite_median(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan


def build_longitudinal_profiles(img, x0):
    ny, nx = img.shape
    xx = np.arange(nx, dtype=float)
    dx_max = int(nx // 2)
    dx_grid = np.arange(-dx_max, dx_max + 1, dtype=float)

    stack = []
    cover = []

    for y in range(ny):
        if y >= len(x0) or not np.isfinite(x0[y]):
            continue
        row = np.asarray(img[y], float)
        valid = np.isfinite(row)
        if valid.sum() < 5:
            continue
        dx = xx - x0[y]
        interp = np.interp(dx_grid, dx[valid], row[valid], left=np.nan, right=np.nan)
        stack.append(interp)
        cover.append(np.isfinite(interp))

    if len(stack) == 0:
        return dx_grid, None, None, None, None

    stack = np.asarray(stack, float)
    cover = np.asarray(cover, bool)

    prof_med = np.nanmedian(stack, axis=0)
    prof_mean = np.nanmean(stack, axis=0)
    prof_std = np.nanstd(stack, axis=0)
    cover_frac = np.mean(cover, axis=0)

    return dx_grid, prof_med, prof_mean, prof_std, cover_frac


def make_one_png(file08, file06, slit, outpng):
    with fits.open(file08) as h08:
        tab = h08[slit].data
        hdr08 = h08[slit].header.copy()

    with fits.open(file06) as h06:
        img = np.asarray(h06[slit].data, float)

    ypix = np.asarray(tab["YPIX"], float)
    flux = np.asarray(tab["FLUX"], float) if "FLUX" in tab.names else None
    sky = np.asarray(tab["SKY"], float) if "SKY" in tab.names else None
    x0 = np.asarray(tab["X0"], float) if "X0" in tab.names else None
    nsky = np.asarray(tab["NSKY"], float) if "NSKY" in tab.names else None
    skysig = np.asarray(tab["SKYSIG"], float) if "SKYSIG" in tab.names else None
    apfrac = np.asarray(tab["APLOSS_FRAC"], float) if "APLOSS_FRAC" in tab.names else None
    edgeflag = np.asarray(tab["EDGEFLAG"], int) if "EDGEFLAG" in tab.names else None
    flux_apcorr = np.asarray(tab["FLUX_APCORR"], float) if "FLUX_APCORR" in tab.names else None
    trxleft = np.asarray(tab["TRXLEFT"], float) if "TRXLEFT" in tab.names else None
    trxright = np.asarray(tab["TRXRIGHT"], float) if "TRXRIGHT" in tab.names else None

    ny, nx = img.shape
    xx = np.arange(nx, dtype=float)

    WOBJ = hdr08.get("WOBJ", 3.0)
    GAP = hdr08.get("GAP", 1.0)
    ridge_mode = hdr08.get("RIDGEMOD", hdr08.get("RIDGEAUT", "NA"))
    nedge1 = hdr08.get("NEDGE1", -1)
    nedge2 = hdr08.get("NEDGE2", -1)

    obj_mask = np.zeros_like(img, dtype=bool)
    sky_mask = np.zeros_like(img, dtype=bool)

    if x0 is not None:
        for iy in range(min(ny, len(x0))):
            if not np.isfinite(x0[iy]):
                continue
            dx = xx - x0[iy]
            obj_mask[iy] = np.isfinite(img[iy]) & (np.abs(dx) <= WOBJ)
            sky_mask[iy] = np.isfinite(img[iy]) & (np.abs(dx) >= (WOBJ + GAP))

    obj_vis = np.where(obj_mask, img, np.nan)
    sky_vis = np.where(sky_mask, img, np.nan)

    good_flux_rows = int(np.sum(np.isfinite(flux))) if flux is not None else 0
    med_apfrac = finite_median(apfrac) if apfrac is not None else np.nan
    med_nsky = finite_median(nsky) if nsky is not None else np.nan
    med_skysig = finite_median(skysig) if skysig is not None else np.nan

    dx_grid, prof_med, prof_mean, prof_std, cover_frac = build_longitudinal_profiles(img, x0)

    v1, v2 = robust_limits(img, 5, 98)

    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(4, 2, hspace=0.34, wspace=0.24)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img, origin="lower", aspect="auto", cmap="gray", vmin=v1, vmax=v2)
    if x0 is not None:
        ax1.plot(x0, ypix, "r-", lw=1.3)
        ax1.plot(x0 - WOBJ, ypix, "b--", lw=1.0)
        ax1.plot(x0 + WOBJ, ypix, "b--", lw=1.0)
    if trxleft is not None and trxright is not None:
        ax1.plot(trxleft, ypix, color="cyan", lw=0.9, alpha=0.9)
        ax1.plot(trxright, ypix, color="cyan", lw=0.9, alpha=0.9)
    ax1.set_title("2D slit image + ridge/aperture")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    ax2 = fig.add_subplot(gs[0, 1])
    if flux is not None:
        ax2.plot(ypix, flux, lw=1.0, label="FLUX")
    if sky is not None:
        ax2.plot(ypix, sky, lw=1.0, label="SKY")
    if flux_apcorr is not None:
        ax2.plot(ypix, flux_apcorr, lw=1.0, label="FLUX_APCORR")
    ax2.set_title("Extracted quantities")
    ax2.set_xlabel("Y")
    ax2.set_ylabel("Value")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    if apfrac is not None:
        ax3.plot(ypix, apfrac, lw=1.0)
    if edgeflag is not None:
        ax3_t = ax3.twinx()
        ax3_t.plot(ypix, edgeflag, color="tab:red", lw=0.9, alpha=0.8)
        ax3_t.set_ylabel("EDGEFLAG", color="tab:red")
        ax3_t.tick_params(axis="y", labelcolor="tab:red")
    ax3.set_title("Aperture loss and edge flags")
    ax3.set_xlabel("Y")
    ax3.set_ylabel("APLOSS_FRAC")
    ax3.set_ylim(0, 1.05)

    ax4 = fig.add_subplot(gs[1, 1])
    if nsky is not None:
        ax4.plot(ypix, nsky, lw=1.0)
    if skysig is not None:
        ax4_t = ax4.twinx()
        ax4_t.plot(ypix, skysig, color="tab:orange", lw=0.9, alpha=0.85)
        ax4_t.set_ylabel("SKYSIG", color="tab:orange")
        ax4_t.tick_params(axis="y", labelcolor="tab:orange")
    ax4.set_title("Sky diagnostics")
    ax4.set_xlabel("Y")
    ax4.set_ylabel("NSKY")

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.imshow(obj_vis, origin="lower", aspect="auto", cmap="gray", vmin=v1, vmax=v2)
    if x0 is not None:
        ax5.plot(x0, ypix, "r-", lw=1.0)
    ax5.set_title("Object-region pixels")
    ax5.set_xlabel("X")
    ax5.set_ylabel("Y")

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.imshow(sky_vis, origin="lower", aspect="auto", cmap="gray", vmin=v1, vmax=v2)
    if x0 is not None:
        ax6.plot(x0, ypix, "r-", lw=1.0)
    ax6.set_title("Sky-region pixels")
    ax6.set_xlabel("X")
    ax6.set_ylabel("Y")

    ax7 = fig.add_subplot(gs[3, 0])
    if prof_med is not None:
        ax7.plot(dx_grid, prof_med, lw=2.0, label="median")
        ax7.plot(dx_grid, prof_mean, lw=1.0, alpha=0.7, label="mean")
        ax7.fill_between(dx_grid, prof_med - prof_std, prof_med + prof_std, alpha=0.20)
        ax7.axvline(0.0, color="r", ls="--", lw=1.0)
    ax7.set_title("Ridge-aligned cross-dispersion profile")
    ax7.set_xlabel("ΔX from ridge (pix)")
    ax7.set_ylabel("Flux")
    ax7.legend(fontsize=8)

    ax8 = fig.add_subplot(gs[3, 1])
    if cover_frac is not None:
        ax8.plot(dx_grid, cover_frac, lw=2.0)
        ax8.axvline(0.0, color="r", ls="--", lw=1.0)
    ax8.set_title("Coverage fraction in aligned profile")
    ax8.set_xlabel("ΔX from ridge (pix)")
    ax8.set_ylabel("Coverage fraction")
    ax8.set_ylim(0.0, 1.05)

    fig.suptitle(
        f"{slit}  |  ridge={ridge_mode}  |  "
        f"good_rows={good_flux_rows}  NEDGE1={nedge1}  NEDGE2={nedge2}  "
        f"med(APFR)={med_apfrac:.3f}  med(NSKY)={med_nsky:.1f}  med(SKYSIG)={med_skysig:.4f}",
        fontsize=13
    )
    fig.savefig(outpng, dpi=150, bbox_inches="tight")
    plt.close(fig)


ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True, choices=["EVEN", "ODD", "even", "odd"])
ap.add_argument("--file08", default=None)
ap.add_argument("--file06", default=None)
ap.add_argument("--outdir", default=None)
args = ap.parse_args()

set_tag = args.set.upper()

file08 = first_existing([
    Path(args.file08) if args.file08 else None,
    pick_latest_step08(set_tag),
])
if file08 is None:
    raise FileNotFoundError(f"No Step08 file found for set {set_tag}")

file06 = first_existing([
    Path(args.file06) if args.file06 else None,
    infer_file06_from_file08(file08, set_tag),
])
if file06 is None:
    raise FileNotFoundError(f"No Step06 tracecoords file found for set {set_tag}")

outdir = Path(args.outdir) if args.outdir else (file08.parent / f"{file08.stem}_QC08_png")
outdir.mkdir(parents=True, exist_ok=True)

with fits.open(file08) as h08:
    slits = [(h.name or "").upper() for h in h08[1:] if (h.name or "").upper().startswith("SLIT") and h.data is not None]

print("Using Step08:", file08)
print("Using Step06:", file06)
print("Output dir  :", outdir)
print(f"Found {len(slits)} slits")

for slit in slits:
    outpng = outdir / f"{file08.stem}_{slit}_7plots.png"
    make_one_png(file08, file06, slit, outpng)
    print("[OK] wrote", outpng)

print("[DONE]")