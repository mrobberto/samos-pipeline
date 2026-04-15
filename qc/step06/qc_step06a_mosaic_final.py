#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step06a QC — inspect the baseline FinalScience mosaic.

PURPOSE
-------
Validate the full-frame science mosaic produced by Step06a before any
pixel-flat correction or slit rectification is applied.

This QC is intentionally lightweight. It is meant to answer:

1) Does the combined science mosaic look sensible?
2) Is the ADU/s normalization plausible?
3) Are there obvious combine artifacts, empty regions, or bad headers?

INPUT
-----
From config.ST06_SCIENCE:

- FinalScience_*_ADUperS.fits
  or, if rate normalization was disabled:
- FinalScience_*_ADU.fits

OUTPUTS
-------
Written under config.ST06_SCIENCE:

- qc_step06a/
    - step06a_mosaic_summary.png
    - step06a_mosaic_report.txt

CHECKS PERFORMED
----------------
- full-frame image quicklook with robust stretch
- histogram of finite pixel values
- row and column median profiles
- finite-pixel fraction
- header audit:
    EXPTIME, NCOMBINE, UNIT, BUNIT, SCIENCE files

INTERPRETATION
--------------
Good behavior:
- image structure is plausible and continuous
- no large blank regions or obvious combine seams
- row/column medians are smooth
- finite-pixel fraction is high
- header provenance is present and sensible

Warning signs:
- large NaN regions
- abrupt seams or discontinuities
- suspicious normalization
- inconsistent EXPTIME / combine bookkeeping

run from xterm, samos-pipeline
> PYTHONPATH=. python pipeline/step06_science_rectify/step06a_make_final_science.py 

"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
import config


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def pct(x: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    lo_v = float(np.nanpercentile(x, lo))
    hi_v = float(np.nanpercentile(x, hi))
    if not np.isfinite(lo_v) or not np.isfinite(hi_v) or hi_v <= lo_v:
        med = float(np.nanmedian(x))
        return med - 1.0, med + 1.0
    return lo_v, hi_v


def first_existing(paths):
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            return p
    return None


def pick_science_file(st06: Path, explicit: str | None = None) -> Path:
    return first_existing([
        Path(explicit) if explicit else None,
        *sorted(st06.glob("FinalScience*_ADUperS.fits"), key=lambda p: p.stat().st_mtime, reverse=True),
        *sorted(st06.glob("FinalScience*_ADU*.fits"), key=lambda p: p.stat().st_mtime, reverse=True),
    ])


def write_report(path: Path, infile: Path, hdr, img: np.ndarray) -> None:
    finite = np.isfinite(img)
    vals = img[finite]
    row_med = np.nanmedian(img, axis=1)
    col_med = np.nanmedian(img, axis=0)

    with path.open("w") as f:
        f.write("Step06a QC — baseline FinalScience mosaic\n")
        f.write("=" * 60 + "\n")
        f.write(f"Input : {infile}\n\n")

        f.write("Image statistics:\n")
        f.write(f" shape                    = {img.shape}\n")
        f.write(f" finite pixel fraction    = {finite.mean():.6f}\n")
        if vals.size:
            f.write(f" median                   = {np.nanmedian(vals):.6e}\n")
            f.write(f" robust sigma             = {robust_sigma(vals):.6e}\n")
            f.write(f" p1, p99                  = {np.nanpercentile(vals,1):.6e}, {np.nanpercentile(vals,99):.6e}\n")
            f.write(f" min, max                 = {np.nanmin(vals):.6e}, {np.nanmax(vals):.6e}\n")
        f.write("\n")

        f.write("Row / column medians:\n")
        f.write(f" row median robust sigma  = {robust_sigma(row_med):.6e}\n")
        f.write(f" col median robust sigma  = {robust_sigma(col_med):.6e}\n\n")

        f.write("Header audit:\n")
        for k in ["EXPTIME", "NCOMBINE", "BUNIT", "UNIT", "SCIENCE", "SCI1", "SCI2", "SCI3", "SCI4"]:
            if k in hdr:
                f.write(f" {k:10s} = {hdr[k]}\n")


def main():
    ap = argparse.ArgumentParser(description="Step06a QC — inspect baseline FinalScience mosaic")
    ap.add_argument("--infile", default=None, help="Optional explicit FinalScience FITS")
    ap.add_argument("--outdir", default=None, help="Optional output directory (default: config.ST06_SCIENCE/qc_step06a)")
    ap.add_argument("--p-lo", type=float, default=2.0, help="Lower percentile for image stretch")
    ap.add_argument("--p-hi", type=float, default=98.0, help="Upper percentile for image stretch")
    args = ap.parse_args()

    st06 = Path(config.ST06_SCIENCE)
    infile = pick_science_file(st06, args.infile)
    if infile is None:
        raise FileNotFoundError(f"No FinalScience file found in {st06}")

    img = fits.getdata(infile).astype(float)
    hdr = fits.getheader(infile)

    outdir = Path(args.outdir) if args.outdir else (st06 / "qc_step06a")
    outdir.mkdir(parents=True, exist_ok=True)

    png_path = outdir / "step06a_mosaic_summary.png"
    txt_path = outdir / "step06a_mosaic_report.txt"

    v1, v2 = pct(img, args.p_lo, args.p_hi)
    finite = np.isfinite(img)
    vals = img[finite]
    row_med = np.nanmedian(img, axis=1)
    col_med = np.nanmedian(img, axis=0)

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    im = ax[0, 0].imshow(img, origin="lower", vmin=v1, vmax=v2, cmap="gray")
    ax[0, 0].set_title("FinalScience mosaic")
    plt.colorbar(im, ax=ax[0, 0], fraction=0.046)

    ax[0, 1].hist(vals, bins=200, histtype="step")
    ax[0, 1].set_title("Histogram of finite pixels")
    ax[0, 1].set_xlabel("Value")

    ax[1, 0].plot(row_med, lw=0.8)
    ax[1, 0].set_title("Row median profile")
    ax[1, 0].set_xlabel("Row")
    ax[1, 0].set_ylabel("Median")

    ax[1, 1].plot(col_med, lw=0.8)
    ax[1, 1].set_title("Column median profile")
    ax[1, 1].set_xlabel("Column")
    ax[1, 1].set_ylabel("Median")

    fig.suptitle(f"Step06a QC — {infile.name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    write_report(txt_path, infile, hdr, img)

    print("\n====================================================")
    print("Step06a QC — baseline FinalScience mosaic")
    print("====================================================")
    print("Input  :", infile)
    print("Output :", png_path)
    print("Report :", txt_path)
    print()
    print(f"shape                 = {img.shape}")
    print(f"finite pixel fraction = {finite.mean():.6f}")
    if vals.size:
        print(f"median                = {np.nanmedian(vals):.6e}")
        print(f"robust sigma          = {robust_sigma(vals):.6e}")
        print(f"p1, p99               = {np.nanpercentile(vals,1):.6e}, {np.nanpercentile(vals,99):.6e}")
    for k in ["EXPTIME", "NCOMBINE", "BUNIT", "UNIT"]:
        if k in hdr:
            print(f"{k:10s} = {hdr[k]}")
    print("====================================================")
    print(f"[DONE] Wrote {png_path}")
    print(f"[DONE] Wrote {txt_path}")


if __name__ == "__main__":
    main()
