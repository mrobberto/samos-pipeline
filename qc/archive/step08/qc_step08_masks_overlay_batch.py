#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step08 QC — batch mask overlays.

Create one PNG per slit showing the Step06 TRACECOORDS image with the Step08
ridge, object aperture, sky region, and invalid pixels overlaid.
"""

import argparse
import math
from pathlib import Path
from glob import glob
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

def pick_tracecoords(set_tag):
    matches = sorted(glob(str(Path(config.ST06_SCIENCE) / f"*_{set_tag}_tracecoords.fits")))
    return Path(matches[-1]) if matches else None

def robust_limits(img, p_lo=5.0, p_hi=98.0):
    v = np.asarray(img, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo = np.nanpercentile(v, p_lo)
    hi = np.nanpercentile(v, p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        med = np.nanmedian(v); sig = np.nanstd(v)
        if not np.isfinite(sig) or sig <= 0: sig = 1.0
        return med - 2 * sig, med + 4 * sig
    return float(lo), float(hi)

def get_slit_names(hdul):
    out = []
    for h in hdul[1:]:
        nm = (h.name or "").upper()
        if nm.startswith("SLIT") and h.data is not None:
            out.append(nm)
    return out

def build_masks(img, tab, hdr08):
    ypix = np.asarray(tab["YPIX"], int)
    x0 = np.asarray(tab["X0"], float)
    ny, nx = img.shape
    xx = np.arange(nx, dtype=float)
    wobj = float(hdr08.get("WOBJ", 3.0))
    gap = float(hdr08.get("GAP", 1.0))
    finite_mask = np.isfinite(img)
    invalid_mask = ~finite_mask
    obj_mask = np.zeros_like(img, dtype=bool)
    sky_mask = np.zeros_like(img, dtype=bool)
    for i, y in enumerate(ypix):
        if y < 0 or y >= ny or not np.isfinite(x0[i]):
            continue
        dx = xx - x0[i]
        obj_mask[y] = finite_mask[y] & (np.abs(dx) <= wobj)
        sky_mask[y] = finite_mask[y] & (np.abs(dx) >= (wobj + gap))
    sky_only = sky_mask & (~obj_mask) & finite_mask
    obj_only = obj_mask & finite_mask
    return ypix, x0, invalid_mask, sky_only, obj_only

def add_overlay_to_axis(ax, img, tab, hdr08, slit):
    ypix, x0, invalid_mask, sky_only, obj_only = build_masks(img, tab, hdr08)
    vmin, vmax = robust_limits(img)
    ny, nx = img.shape
    ax.imshow(img, origin="lower", aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)

    red = np.zeros((ny, nx, 4), float); red[...,0] = 1.0; red[...,3] = 0.35 * invalid_mask.astype(float)
    cyan = np.zeros((ny, nx, 4), float); cyan[...,1] = 1.0; cyan[...,2] = 1.0; cyan[...,3] = 0.28 * sky_only.astype(float)
    yellow = np.zeros((ny, nx, 4), float); yellow[...,0] = 1.0; yellow[...,1] = 1.0; yellow[...,3] = 0.40 * obj_only.astype(float)
    ax.imshow(red, origin="lower", aspect="auto")
    ax.imshow(cyan, origin="lower", aspect="auto")
    ax.imshow(yellow, origin="lower", aspect="auto")
    if np.isfinite(x0).any():
        ax.plot(x0, ypix, color="red", lw=0.9)
    ax.set_title(slit, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

def make_overlay_figure(img, tab, hdr08, slit):
    fig, ax = plt.subplots(figsize=(8, 10))
    add_overlay_to_axis(ax, img, tab, hdr08, slit)
    fig.suptitle(f"Step08 mask overlay — {slit}", fontsize=11)
    return fig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["EVEN", "ODD", "even", "odd"])
    ap.add_argument("--file08", default=None)
    ap.add_argument("--file06", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--slits", nargs="*", default=None)
    ap.add_argument("--montage", action="store_true")
    ap.add_argument("--montage_ncols", type=int, default=4)
    ap.add_argument("--montage_nrows", type=int, default=4)
    args = ap.parse_args()

    set_tag = args.set.upper()
    file08 = first_existing([Path(args.file08) if args.file08 else None,
                             Path(config.ST08_EXTRACT1D) / f"extract1d_optimal_ridge_{set_tag.lower()}.fits"])
    file06 = first_existing([Path(args.file06) if args.file06 else None, pick_tracecoords(set_tag)])
    if file08 is None:
        raise FileNotFoundError(f"No Step08a file found for {set_tag}")
    if file06 is None:
        raise FileNotFoundError(f"No Step06 TRACECOORDS file found for {set_tag}")

    outdir = Path(args.outdir) if args.outdir else (Path(config.ST08_EXTRACT1D) / "qc_step08" / f"{file08.stem}_mask_overlays")
    outdir.mkdir(parents=True, exist_ok=True)

    with fits.open(file08) as h08, fits.open(file06) as h06:
        slits08 = get_slit_names(h08)
        slits06 = get_slit_names(h06)
        common = sorted(set(slits08).intersection(slits06))
        if args.slits:
            requested = [s.strip().upper() for s in args.slits]
            common = [s for s in common if s in requested]
        if not common:
            raise RuntimeError("No common SLIT### extensions found.")

        for slit in common:
            tab = h08[slit].data
            hdr08 = h08[slit].header.copy()
            img = np.asarray(h06[slit].data, float)
            fig = make_overlay_figure(img, tab, hdr08, slit)
            outpng = outdir / f"{file08.stem}_{slit}_mask_overlay.png"
            fig.tight_layout()
            fig.savefig(outpng, dpi=140, bbox_inches="tight")
            plt.close(fig)

        if args.montage:
            npp = args.montage_ncols * args.montage_nrows
            n_pages = math.ceil(len(common) / npp)
            for ipage in range(n_pages):
                subset = common[ipage * npp:(ipage + 1) * npp]
                fig = plt.figure(figsize=(args.montage_ncols * 3.2, args.montage_nrows * 3.0), dpi=150)
                gs = fig.add_gridspec(args.montage_nrows, args.montage_ncols, hspace=0.15, wspace=0.08)
                for i, slit in enumerate(subset):
                    r = i // args.montage_ncols
                    c = i % args.montage_ncols
                    ax = fig.add_subplot(gs[r, c])
                    tab = h08[slit].data
                    hdr08 = h08[slit].header.copy()
                    img = np.asarray(h06[slit].data, float)
                    add_overlay_to_axis(ax, img, tab, hdr08, slit)
                fig.suptitle(f"{file08.stem} — mask overlays — page {ipage + 1}/{n_pages}", fontsize=11)
                outpng = outdir / f"{file08.stem}_mask_overlay_montage_page{ipage + 1:02d}.png"
                fig.savefig(outpng, dpi=150, bbox_inches="tight")
                plt.close(fig)

    print("[DONE] all outputs under", outdir)

if __name__ == "__main__":
    main()
