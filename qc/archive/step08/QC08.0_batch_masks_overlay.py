#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 07:51:06 2026

@author: robberto

QC08_batch_masks_overlay.py

Batch viewer for Step08 single-slit mask overlays.
Creates one PNG per slit plus optional montage pages.

Shows, for each slit:
- grayscale Step06 tracecoords image
- ridge X0
- object aperture
- sky regions
- invalid/masked pixels
- optional valid trace edges

Usage examples
--------------
runfile(
    "QC08_batch_masks_overlay.py",
    args="--file08 ../../reduced/08_extract1d/Extract1D_optimal_ridgeguided_POOLSKY_EVEN.fits "
         "--file06 ../../reduced/06_science/FinalScience_dolidze_ADUperS_pixflatcorr_clipped_EVEN_tracecoords.fits"
)

runfile(
    "QC08_batch_masks_overlay.py",
    args="--file08 ../../reduced/08_extract1d/Extract1D_optimal_ridgeguided_POOLSKY_ODD.fits "
         "--file06 ../../reduced/06_science/FinalScience_dolidze_ADUperS_pixflatcorr_clipped_ODD_tracecoords.fits "
         "--outdir ../../reduced/08_extract1d/QC08_masks_ODD --montage"
)
"""

import argparse
import math
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
        if y < 0 or y >= ny:
            continue
        if not np.isfinite(x0[i]):
            continue

        dx = xx - x0[i]
        obj = finite_mask[y] & (np.abs(dx) <= wobj)
        sky = finite_mask[y] & (np.abs(dx) >= (wobj + gap))

        obj_mask[y] = obj
        sky_mask[y] = sky

    sky_only = sky_mask & (~obj_mask) & finite_mask
    obj_only = obj_mask & finite_mask

    return {
        "ypix": ypix,
        "x0": x0,
        "wobj": wobj,
        "gap": gap,
        "finite_mask": finite_mask,
        "invalid_mask": invalid_mask,
        "obj_only": obj_only,
        "sky_only": sky_only,
    }


def make_overlay_figure(img, tab, hdr08, slit, figsize=(6, 8)):
    masks = build_masks(img, tab, hdr08)

    ypix = masks["ypix"]
    x0 = masks["x0"]
    wobj = masks["wobj"]
    gap = masks["gap"]
    invalid_mask = masks["invalid_mask"]
    obj_only = masks["obj_only"]
    sky_only = masks["sky_only"]

    vmin, vmax = robust_limits(img)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, origin="lower", aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)

    # invalid/masked -> red
    red = np.zeros((img.shape[0], img.shape[1], 4), float)
    red[..., 0] = 1.0
    red[..., 3] = 0.35 * invalid_mask.astype(float)
    ax.imshow(red, origin="lower", aspect="auto")

    # sky -> cyan
    cyan = np.zeros((img.shape[0], img.shape[1], 4), float)
    cyan[..., 1] = 1.0
    cyan[..., 2] = 1.0
    cyan[..., 3] = 0.28 * sky_only.astype(float)
    ax.imshow(cyan, origin="lower", aspect="auto")

    # object -> yellow
    yellow = np.zeros((img.shape[0], img.shape[1], 4), float)
    yellow[..., 0] = 1.0
    yellow[..., 1] = 1.0
    yellow[..., 3] = 0.40 * obj_only.astype(float)
    ax.imshow(yellow, origin="lower", aspect="auto")

    # ridge/aperture guides
    ax.plot(x0, ypix, color="red", lw=1.2, label="ridge X0")
    ax.plot(x0 - wobj, ypix, color="yellow", lw=0.9, ls="--", label="obj aperture")
    ax.plot(x0 + wobj, ypix, color="yellow", lw=0.9, ls="--")
    ax.plot(x0 - (wobj + gap), ypix, color="cyan", lw=0.9, ls=":", label="sky start")
    ax.plot(x0 + (wobj + gap), ypix, color="cyan", lw=0.9, ls=":")

    # optional trace edges
    if "TRXLEFT" in tab.names and "TRXRIGHT" in tab.names:
        trxleft = np.asarray(tab["TRXLEFT"], float)
        trxright = np.asarray(tab["TRXRIGHT"], float)
        ax.plot(trxleft, ypix, color="lime", lw=0.8, alpha=0.9, label="valid trace edges")
        ax.plot(trxright, ypix, color="lime", lw=0.8, alpha=0.9)

    ridge_mode = hdr08.get("RIDGEMOD", hdr08.get("RIDGEAUT", "NA"))
    nedge1 = hdr08.get("NEDGE1", 0)
    nedge2 = hdr08.get("NEDGE2", 0)
    apfrac = np.asarray(tab["APLOSS_FRAC"], float) if "APLOSS_FRAC" in tab.names else None
    med_ap = np.nanmedian(apfrac) if apfrac is not None and np.isfinite(apfrac).any() else np.nan

    title = (
        f"{slit} | ridge={ridge_mode} | NEDGE1={nedge1} NEDGE2={nedge2} | "
        f"med(APFR)={med_ap:.3f}" if np.isfinite(med_ap)
        else f"{slit} | ridge={ridge_mode} | NEDGE1={nedge1} NEDGE2={nedge2}"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(fontsize=7, loc="upper right")

    return fig


def add_overlay_to_axis(ax, img, tab, hdr08, slit):
    masks = build_masks(img, tab, hdr08)

    ypix = masks["ypix"]
    x0 = masks["x0"]
    wobj = masks["wobj"]
    gap = masks["gap"]
    invalid_mask = masks["invalid_mask"]
    obj_only = masks["obj_only"]
    sky_only = masks["sky_only"]

    vmin, vmax = robust_limits(img)

    ax.imshow(img, origin="lower", aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)

    red = np.zeros((img.shape[0], img.shape[1], 4), float)
    red[..., 0] = 1.0
    red[..., 3] = 0.25 * invalid_mask.astype(float)
    ax.imshow(red, origin="lower", aspect="auto")

    cyan = np.zeros((img.shape[0], img.shape[1], 4), float)
    cyan[..., 1] = 1.0
    cyan[..., 2] = 1.0
    cyan[..., 3] = 0.20 * sky_only.astype(float)
    ax.imshow(cyan, origin="lower", aspect="auto")

    yellow = np.zeros((img.shape[0], img.shape[1], 4), float)
    yellow[..., 0] = 1.0
    yellow[..., 1] = 1.0
    yellow[..., 3] = 0.28 * obj_only.astype(float)
    ax.imshow(yellow, origin="lower", aspect="auto")

    ax.plot(x0, ypix, color="red", lw=0.8)
    ax.plot(x0 - wobj, ypix, color="yellow", lw=0.6, ls="--")
    ax.plot(x0 + wobj, ypix, color="yellow", lw=0.6, ls="--")
    ax.plot(x0 - (wobj + gap), ypix, color="cyan", lw=0.6, ls=":")
    ax.plot(x0 + (wobj + gap), ypix, color="cyan", lw=0.6, ls=":")

    if "TRXLEFT" in tab.names and "TRXRIGHT" in tab.names:
        trxleft = np.asarray(tab["TRXLEFT"], float)
        trxright = np.asarray(tab["TRXRIGHT"], float)
        ax.plot(trxleft, ypix, color="lime", lw=0.5, alpha=0.9)
        ax.plot(trxright, ypix, color="lime", lw=0.5, alpha=0.9)

    ridge_mode = hdr08.get("RIDGEMOD", hdr08.get("RIDGEAUT", "NA"))
    ax.set_title(f"{slit}\n{ridge_mode}", fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file08", required=True, help="Step08 extracted FITS")
    ap.add_argument("--file06", required=True, help="Step06 tracecoords FITS")
    ap.add_argument("--outdir", default=None, help="Output directory for PNGs")
    ap.add_argument("--slits", nargs="*", default=None, help="Optional subset of slits")
    ap.add_argument("--montage", action="store_true", help="Also write montage pages")
    ap.add_argument("--montage-ncols", type=int, default=4)
    ap.add_argument("--montage-nrows", type=int, default=3)
    args = ap.parse_args()

    file08 = Path(args.file08)
    file06 = Path(args.file06)

    if args.outdir is None:
        outdir = file08.parent / f"{file08.stem}_mask_overlays"
    else:
        outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with fits.open(file08) as h08, fits.open(file06) as h06:
        slits08 = get_slit_names(h08)
        slits06 = get_slit_names(h06)
        common = sorted(set(slits08).intersection(slits06))

        if args.slits:
            requested = [s.strip().upper() for s in args.slits]
            common = [s for s in common if s in requested]

        if not common:
            raise RuntimeError("No common SLIT### extensions found between Step08 and Step06 files.")

        print(f"Found {len(common)} common slits")

        # one PNG per slit
        for slit in common:
            tab = h08[slit].data
            hdr08 = h08[slit].header.copy()
            img = np.asarray(h06[slit].data, float)

            fig = make_overlay_figure(img, tab, hdr08, slit)
            outpng = outdir / f"{file08.stem}_{slit}_mask_overlay.png"
            fig.tight_layout()
            fig.savefig(outpng, dpi=140, bbox_inches="tight")
            plt.close(fig)
            print("[OK] wrote", outpng)

        # optional montage pages
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
                print("[OK] wrote", outpng)

    print("[DONE] all outputs under", outdir)


if __name__ == "__main__":
    main()