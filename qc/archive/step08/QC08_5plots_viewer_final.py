#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 23:25:07 2026

@author: robberto

QC08_5plots_viewer_final.py

Detailed single-slit QC viewer for Step08.

Usage examples:
runfile("QC08_5plots_viewer_final.py", args="--set EVEN --slit SLIT010")
runfile("QC08_5plots_viewer_final.py", args="--set ODD --slit SLIT029")
runfile("QC08_5plots_viewer_final.py", args="--set EVEN --slit SLIT010 --file08 /full/path/to/step08.fits")
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
import config


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
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
    candidates = sorted(st08.glob(f"*_{tag}.fits"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def finite_median(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan


# ------------------------------------------------------------
# args
# ------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True, choices=["EVEN", "ODD", "even", "odd"])
ap.add_argument("--slit", required=True, help="e.g. SLIT010")
ap.add_argument("--file08", default=None, help="Optional explicit Step08 FITS")
ap.add_argument("--file06", default=None, help="Optional explicit Step06 tracecoords FITS")
args = ap.parse_args()

SET_TAG = args.set.upper()
SLIT = args.slit.strip().upper()

file08 = first_existing([
    Path(args.file08) if args.file08 else None,
    pick_latest_step08(SET_TAG),
])
if file08 is None:
    raise FileNotFoundError(f"No Step08 file found for set {SET_TAG}")

file06 = first_existing([
    Path(args.file06) if args.file06 else None,
    infer_file06_from_file08(file08, SET_TAG),
])
if file06 is None:
    raise FileNotFoundError(f"No Step06 tracecoords file found for set {SET_TAG}")

print("Using Step08:", file08)
print("Using Step06:", file06)
print("Slit:", SLIT)

# ------------------------------------------------------------
# load data
# ------------------------------------------------------------
with fits.open(file08) as h08:
    if SLIT not in [h.name.upper() for h in h08[1:]]:
        raise KeyError(f"{SLIT} not found in {file08.name}")
    tab = h08[SLIT].data
    hdr08 = h08[SLIT].header.copy()

with fits.open(file06) as h06:
    if SLIT not in [h.name.upper() for h in h06[1:]]:
        raise KeyError(f"{SLIT} not found in {file06.name}")
    img = np.asarray(h06[SLIT].data, float)
    hdr06 = h06[SLIT].header.copy()

# ------------------------------------------------------------
# columns
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# approximate object/sky masks for visualization
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# summary metrics
# ------------------------------------------------------------
good_flux_rows = int(np.sum(np.isfinite(flux))) if flux is not None else 0
med_apfrac = finite_median(apfrac) if apfrac is not None else np.nan
med_nsky = finite_median(nsky) if nsky is not None else np.nan
med_skysig = finite_median(skysig) if skysig is not None else np.nan
frac_edge = float(np.mean(edgeflag > 0)) if edgeflag is not None and len(edgeflag) else np.nan

print()
print("====================================================")
print("QC08 detailed viewer")
print("====================================================")
print("Step08 file :", file08)
print("Step06 file :", file06)
print("Slit        :", SLIT)
print(f"RIDGEMOD    = {ridge_mode}")
print(f"good rows   = {good_flux_rows}")
print(f"NEDGE1      = {nedge1}")
print(f"NEDGE2      = {nedge2}")
print(f"median APFR = {med_apfrac:.4f}" if np.isfinite(med_apfrac) else "median APFR = nan")
print(f"median NSKY = {med_nsky:.2f}" if np.isfinite(med_nsky) else "median NSKY = nan")
print(f"median SKYSIG = {med_skysig:.4f}" if np.isfinite(med_skysig) else "median SKYSIG = nan")
print(f"edge frac   = {frac_edge:.4f}" if np.isfinite(frac_edge) else "edge frac   = nan")
print("====================================================")

# ------------------------------------------------------------
# plotting
# ------------------------------------------------------------
v1, v2 = robust_limits(img, 5, 98)

fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(3, 2, hspace=0.32, wspace=0.22)

# Panel 1: image + ridge + aperture + edges
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(img, origin="lower", aspect="auto", cmap="gray", vmin=v1, vmax=v2)
if x0 is not None:
    ax1.plot(x0, ypix, "r-", lw=1.3, label="ridge")
    ax1.plot(x0 - WOBJ, ypix, "b--", lw=1.0, label="obj aperture")
    ax1.plot(x0 + WOBJ, ypix, "b--", lw=1.0)
if trxleft is not None and trxright is not None:
    ax1.plot(trxleft, ypix, color="cyan", lw=0.9, alpha=0.9, label="trace edges")
    ax1.plot(trxright, ypix, color="cyan", lw=0.9, alpha=0.9)
ax1.set_title("2D slit image + ridge/aperture")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.legend(fontsize=8, loc="upper right")

# Panel 2: flux / sky / apcorr
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

# Panel 3: APLOSS + EDGEFLAG
ax3 = fig.add_subplot(gs[1, 0])
if apfrac is not None:
    ax3.plot(ypix, apfrac, lw=1.0, label="APLOSS_FRAC")
if edgeflag is not None:
    ax3_t = ax3.twinx()
    ax3_t.plot(ypix, edgeflag, color="tab:red", lw=0.9, alpha=0.8, label="EDGEFLAG")
    ax3_t.set_ylabel("EDGEFLAG", color="tab:red")
    ax3_t.tick_params(axis="y", labelcolor="tab:red")
ax3.set_title("Aperture loss and edge flags")
ax3.set_xlabel("Y")
ax3.set_ylabel("APLOSS_FRAC")
ax3.set_ylim(0, 1.05)

# Panel 4: NSKY + SKYSIG
ax4 = fig.add_subplot(gs[1, 1])
if nsky is not None:
    ax4.plot(ypix, nsky, lw=1.0, label="NSKY")
if skysig is not None:
    ax4_t = ax4.twinx()
    ax4_t.plot(ypix, skysig, color="tab:orange", lw=0.9, alpha=0.85, label="SKYSIG")
    ax4_t.set_ylabel("SKYSIG", color="tab:orange")
    ax4_t.tick_params(axis="y", labelcolor="tab:orange")
ax4.set_title("Sky diagnostics")
ax4.set_xlabel("Y")
ax4.set_ylabel("NSKY")

# Panel 5: object pixels actually used
ax5 = fig.add_subplot(gs[2, 0])
ax5.imshow(obj_vis, origin="lower", aspect="auto", cmap="gray", vmin=v1, vmax=v2)
if x0 is not None:
    ax5.plot(x0, ypix, "r-", lw=1.0)
ax5.set_title("Object-region pixels")
ax5.set_xlabel("X")
ax5.set_ylabel("Y")

# Panel 6: sky pixels actually used
ax6 = fig.add_subplot(gs[2, 1])
ax6.imshow(sky_vis, origin="lower", aspect="auto", cmap="gray", vmin=v1, vmax=v2)
if x0 is not None:
    ax6.plot(x0, ypix, "r-", lw=1.0)
ax6.set_title("Sky-region pixels")
ax6.set_xlabel("X")
ax6.set_ylabel("Y")

fig.suptitle(
    f"{SLIT}  |  set={SET_TAG}  |  ridge={ridge_mode}  |  "
    f"good_rows={good_flux_rows}  NEDGE1={nedge1}  NEDGE2={nedge2}  "
    f"med(APFR)={med_apfrac:.3f}",
    fontsize=13
)
if args.show_plots:
            plt.show()
        else:
            plt.close()