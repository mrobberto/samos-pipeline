#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step06c QC — quicklook export for TRACECOORDS slitlets.

PURPOSE
-------
Create lightweight JPG quicklooks of the Step06c TRACECOORDS slitlets for fast
visual inspection.

This is the first-pass QC layer for Step06c. It is meant to answer:
- do the slitlet images look sensible?
- are the slit widths and trace footprints plausible?
- are there obvious edge truncations, broken slitlets, or empty extensions?

INPUTS
------
From config.ST06_SCIENCE:

- science_tracecoords_even.fits / science_tracecoords_odd.fits
or, if older naming is still present, the newest file matching:
- *_{EVEN,ODD}_tracecoords.fits

When --reg is given, the script prefers the file whose primary header has:
    REGFLAT = True

OUTPUTS
-------
Written under config.ST06_SCIENCE:

- qc_step06c_even/
- qc_step06c_odd/

For each run:
- one JPG per SLIT### extension
- one montage JPG for the first N slits

DISPLAY
-------
- optional Y-binning for compact display
- robust percentile stretch
- TRACECOORDS axes:
    X = slit spatial direction
    Y = dispersion direction

NOTES
-----
- This script does not modify pipeline products.
- It is intentionally visual / lightweight.
- Use the batch and single-slit QC scripts for deeper quantitative analysis.

run standalone as
----
PYTHONPATH=. python qc/step06/qc_step06c_quicklooks_final.py \
  --traceset EVEN \
  "/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_Run8_Science_2026_01/SAMI/Dolidze25/reduced/06_science/FinalScience_dolidze_ADUperS_pixflatcorr_EVEN_tracecoords.fits"

PYTHONPATH=. python qc/step06/qc_step06c_quicklooks_final.py \
  --traceset ODD \
  "/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_Run8_Science_2026_01/SAMI/Dolidze25/reduced/06_science/FinalScience_dolidze_ADUperS_pixflatcorr_ODD_tracecoords.fits"
"""

import argparse
from pathlib import Path
import math

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config


def bin_y(img: np.ndarray, ybin: int) -> np.ndarray:
    if ybin <= 1:
        return img

    ny, nx = img.shape
    ny2 = (ny // ybin) * ybin
    if ny2 < ybin:
        return img

    a = img[:ny2, :].reshape(ny2 // ybin, ybin, nx)

    num = np.nansum(a, axis=1)
    den = np.sum(np.isfinite(a), axis=1)

    out = np.full((ny2 // ybin, nx), np.nan, dtype=float)
    good = den > 0
    out[good] = num[good] / den[good]

    return out


def robust_limits(img: np.ndarray, p_lo=5.0, p_hi=99.5) -> tuple[float, float]:
    v = img[np.isfinite(img)]
    if v.size == 0:
        return 0.0, 1.0
    lo = np.nanpercentile(v, p_lo)
    hi = np.nanpercentile(v, p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        med = np.nanmedian(v)
        return med - 1.0, med + 1.0
    return float(lo), float(hi)


def get_slit_hdus(hdul: fits.HDUList) -> list[fits.ImageHDU]:
    slits = []
    for h in hdul[1:]:
        nm = (h.header.get("EXTNAME") or "").strip()
        if nm.startswith("SLIT") and h.data is not None:
            slits.append(h)
    return slits


def save_one_jpg(img: np.ndarray, outjpg: Path, title: str, p_lo: float, p_hi: float):
    vmin, vmax = robust_limits(img, p_lo=p_lo, p_hi=p_hi)

    fig = plt.figure(figsize=(6.5, 4.0), dpi=150)
    ax = fig.add_subplot(111)
    ax.imshow(img, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X (TRACECOORDS spatial)")
    ax.set_ylabel("Y (binned dispersion)")
    fig.tight_layout()
    fig.savefig(outjpg, format="jpg")
    plt.close(fig)


def save_montage(images: list[np.ndarray], titles: list[str], outjpg: Path,
                 ncols: int = 6, p_lo: float = 5.0, p_hi: float = 99.5):
    if not images:
        return
    n = len(images)
    ncols = max(1, ncols)
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(ncols * 2.2, nrows * 2.0), dpi=170)
    for i, (img, ttl) in enumerate(zip(images, titles), start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        vmin, vmax = robust_limits(img, p_lo=p_lo, p_hi=p_hi)
        ax.imshow(img, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(ttl, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(outjpg.stem, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outjpg, format="jpg")
    plt.close(fig)


def pick_tracecoords_file(traceset: str, want_reg: bool) -> Path:
    st06 = Path(config.ST06_SCIENCE)

    # Prefer cleaned names if present
    preferred = st06 / f"science_tracecoords_{traceset.lower()}.fits"
    if preferred.exists():
        # if explicit reg requested, require REGFLAT match
        try:
            reg = bool(fits.getheader(preferred, 0).get("REGFLAT", False))
            if reg == want_reg:
                return preferred
        except Exception:
            if not want_reg:
                return preferred

    files = sorted(st06.glob(f"*_{traceset}_tracecoords.fits"), key=lambda p: p.stat().st_mtime)

    for p in files[::-1]:
        try:
            reg = bool(fits.getheader(p, 0).get("REGFLAT", False))
        except Exception:
            continue
        if reg == want_reg:
            return p

    raise FileNotFoundError(
        f"No {traceset} tracecoords file with REGFLAT={want_reg} found in {config.ST06_SCIENCE}"
    )


def main():
    ap = argparse.ArgumentParser(description="Step06c QC — export TRACECOORDS quicklooks")
    ap.add_argument("--traceset", required=True, choices=["EVEN", "ODD", "even", "odd"],
                    help="Trace set to inspect.")
    ap.add_argument("--reg", action="store_true",
                    help="Use the registered Step06c output (REGFLAT=True).")
    ap.add_argument("fitsfile", nargs="?", default=None,
                    help="Optional explicit Step06c MEF file.")
    ap.add_argument("--outdir", default=None,
                    help="Output directory root (default: config.ST06_SCIENCE/qc_step06c_<set>)")
    ap.add_argument("--ybin", type=int, default=12,
                    help="Bin factor along Y to compress length")
    ap.add_argument("--p-lo", type=float, default=5.0,
                    help="Lower percentile for display stretch")
    ap.add_argument("--p-hi", type=float, default=99.5,
                    help="Upper percentile for display stretch")
    ap.add_argument("--max-slits", type=int, default=9999,
                    help="Max number of slits to export")
    ap.add_argument("--montage", action="store_true", default=True)
    ap.add_argument("--montage-n", type=int, default=24,
                    help="How many slits in the montage")
    ap.add_argument("--montage-cols", type=int, default=6,
                    help="Columns in montage")
    args = ap.parse_args()

    traceset = args.traceset.upper()

    if args.fitsfile is not None:
        fitsfile = Path(args.fitsfile)
    else:
        fitsfile = pick_tracecoords_file(traceset, args.reg)

    if not fitsfile.exists():
        raise FileNotFoundError(fitsfile)

    outroot = Path(args.outdir) if args.outdir else (Path(config.ST06_SCIENCE) / f"qc_step06c_{traceset.lower()}")
    outroot.mkdir(parents=True, exist_ok=True)

    tag = fitsfile.stem
    subdir = outroot / tag
    subdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Reading:", fitsfile)
    with fits.open(fitsfile) as hdul:
        slits = get_slit_hdus(hdul)

        if not slits:
            raise RuntimeError("No SLIT### extensions found.")

        slits = slits[: args.max_slits]

        montage_imgs = []
        montage_titles = []

        for h in slits:
            ext = (h.header.get("EXTNAME") or "SLIT???").strip()
            img = np.array(h.data, dtype=float)

            img2 = bin_y(img, args.ybin)

            outjpg = subdir / f"{tag}_{ext}_ybin{args.ybin:02d}.jpg"
            title = f"{tag}  {ext}  ybin={args.ybin}"
            save_one_jpg(img2, outjpg, title, args.p_lo, args.p_hi)

            if args.montage and len(montage_imgs) < args.montage_n:
                montage_imgs.append(img2)
                montage_titles.append(ext)

        if args.montage:
            mout = outroot / f"{tag}_montage_ybin{args.ybin:02d}.jpg"
            save_montage(montage_imgs, montage_titles, mout,
                         ncols=args.montage_cols, p_lo=args.p_lo, p_hi=args.p_hi)
            print("[OK] montage:", mout)

    print("[OK] wrote JPGs under:", subdir)


if __name__ == "__main__":
    main()
