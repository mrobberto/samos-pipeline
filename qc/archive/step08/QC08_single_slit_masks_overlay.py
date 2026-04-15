#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 07:33:05 2026

@author: robberto
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QC08_single_slit_masks_overlay.py

Overlay of object aperture, sky regions, and invalid/masked pixels
for one Step08 slit on the corresponding Step06 tracecoords image.

Usage example:
runfile(
    "QC08_single_slit_masks_overlay.py",
    args="--file08 ../../reduced/08_extract1d/Extract1D_optimal_ridgeguided_POOLSKY_EVEN.fits "
         "--file06 ../../reduced/06_science/FinalScience_dolidze_ADUperS_pixflatcorr_clipped_EVEN_tracecoords.fits "
         "--slit SLIT044"
)
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits


def robust_limits(img, p_lo=5.0, p_hi=98.0):
    v = np.asarray(img, float)
    v = v[np.isfinite(v)]
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


ap = argparse.ArgumentParser()
ap.add_argument("--file08", required=True, help="Step08 extracted FITS")
ap.add_argument("--file06", required=True, help="Step06 tracecoords FITS")
ap.add_argument("--slit", required=True, help="e.g. SLIT044")
args = ap.parse_args()

file08 = Path(args.file08)
file06 = Path(args.file06)
slit = args.slit.strip().upper()

with fits.open(file08) as h08:
    if slit not in [h.name.upper() for h in h08[1:]]:
        raise KeyError(f"{slit} not found in {file08}")
    tab = h08[slit].data
    hdr08 = h08[slit].header.copy()

with fits.open(file06) as h06:
    if slit not in [h.name.upper() for h in h06[1:]]:
        raise KeyError(f"{slit} not found in {file06}")
    img = np.asarray(h06[slit].data, float)
    hdr06 = h06[slit].header.copy()

ypix = np.asarray(tab["YPIX"], int)
x0 = np.asarray(tab["X0"], float)

ny, nx = img.shape
xx = np.arange(nx, dtype=float)

# Parameters from header if present
WOBJ = float(hdr08.get("WOBJ", 3.0))
GAP = float(hdr08.get("GAP", 1.0))

# Build masks
finite_mask = np.isfinite(img)
invalid_mask = ~finite_mask

obj_mask = np.zeros_like(img, dtype=bool)
sky_mask = np.zeros_like(img, dtype=bool)

for i, y in enumerate(ypix):
    if y < 0 or y >= ny:
        continue
    if not np.isfinite(x0[i]):
        continue

    dx = xx - x0[i]

    # object aperture
    obj = finite_mask[y] & (np.abs(dx) <= WOBJ)

    # same side-agnostic sky definition used only for visualization
    sky = finite_mask[y] & (np.abs(dx) >= (WOBJ + GAP))

    obj_mask[y] = obj
    sky_mask[y] = sky

# exclusive masks for display priority
sky_only = sky_mask & (~obj_mask) & finite_mask
obj_only = obj_mask & finite_mask

# image limits
vmin, vmax = robust_limits(img)

# Base grayscale image
fig, ax = plt.subplots(figsize=(8, 10))
ax.imshow(img, origin="lower", aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)

# Invalid / masked pixels in red
red = np.zeros((ny, nx, 4), float)
red[..., 0] = 1.0
red[..., 3] = 0.35 * invalid_mask.astype(float)
ax.imshow(red, origin="lower", aspect="auto")

# Sky pixels in cyan
cyan = np.zeros((ny, nx, 4), float)
cyan[..., 1] = 1.0
cyan[..., 2] = 1.0
cyan[..., 3] = 0.28 * sky_only.astype(float)
ax.imshow(cyan, origin="lower", aspect="auto")

# Object pixels in yellow
yellow = np.zeros((ny, nx, 4), float)
yellow[..., 0] = 1.0
yellow[..., 1] = 1.0
yellow[..., 3] = 0.40 * obj_only.astype(float)
ax.imshow(yellow, origin="lower", aspect="auto")

# Ridge and aperture guides
ax.plot(x0, ypix, color="red", lw=1.4, label="ridge X0")
ax.plot(x0 - WOBJ, ypix, color="yellow", lw=1.0, ls="--", label="obj aperture")
ax.plot(x0 + WOBJ, ypix, color="yellow", lw=1.0, ls="--")
ax.plot(x0 - (WOBJ + GAP), ypix, color="cyan", lw=1.0, ls=":", label="sky start")
ax.plot(x0 + (WOBJ + GAP), ypix, color="cyan", lw=1.0, ls=":")

# Optional trace edges if present
if "TRXLEFT" in tab.names and "TRXRIGHT" in tab.names:
    trxleft = np.asarray(tab["TRXLEFT"], float)
    trxright = np.asarray(tab["TRXRIGHT"], float)
    ax.plot(trxleft, ypix, color="lime", lw=0.9, alpha=0.9, label="valid trace edges")
    ax.plot(trxright, ypix, color="lime", lw=0.9, alpha=0.9)

# Summary
frac_invalid = float(np.mean(invalid_mask))
frac_obj = float(np.mean(obj_only))
frac_sky = float(np.mean(sky_only))

title = (
    f"{slit}\n"
    f"yellow=obj  cyan=sky  red=invalid/masked\n"
    f"WOBJ={WOBJ:.1f}  GAP={GAP:.1f}  "
    f"invalid={frac_invalid:.3f}  obj={frac_obj:.3f}  sky={frac_sky:.3f}"
)

ax.set_title(title)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
if args.show_plots:
            plt.show()
        else:
            plt.close()