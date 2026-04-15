#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step06c QC — single-slit diagnostic inspection for TRACECOORDS slitlets.

PURPOSE
-------
Provide a deeper QC view of one Step06c TRACECOORDS slitlet.

This script is intended to catch problems that may not be obvious from the
quicklook mosaics alone, such as:
- partially blank rows
- truncated slit width
- finite-footprint shifts across Y
- suspicious edge slits
- broken or sparsely populated slitlets

INPUTS
------
From config.ST06_SCIENCE:

- science_tracecoords_even.fits / science_tracecoords_odd.fits
or, if older naming is present, the newest file matching:
- *_{EVEN,ODD}_tracecoords.fits

When --reg is given, the script prefers a file whose primary header has:
    REGFLAT = True

OUTPUTS
-------
Written under config.ST06_SCIENCE:

- qc_step06c_even/
- qc_step06c_odd/

Each run writes:
- one PNG for the selected slit
- one text report

CHECKS PERFORMED
----------------
For one SLIT### extension:

1) Image quicklook with robust stretch

2) Finite-footprint diagnostics per row:
   - leftmost valid X
   - rightmost valid X
   - width = number of finite pixels

3) Blank-row fraction

4) Width statistics:
   - median / robust sigma
   - min / max finite width

5) Header audit:
   - EXTNAME, TRACESET, SLITID
   - YMIN / Y0DET / XREF when available
   - REGFLAT / FLATDX when available

INTERPRETATION
--------------
Good behavior:
- most rows have a continuous finite footprint
- median width is stable with Y
- few or no blank rows
- left and right footprint boundaries vary smoothly

Warning signs:
- many blank rows
- abrupt jumps in left/right valid X
- strong narrowing toward one end
- irregular footprint suggesting truncation or mis-rectification

run standalone as 
----
PYTHONPATH=. python qc/step06/qc_step06c_single_final.py \
  --traceset EVEN --slit SLIT018 \
  "/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_Run8_Science_2026_01/SAMI/Dolidze25/reduced/06_science/FinalScience_dolidze_ADUperS_pixflatcorr_EVEN_tracecoords.fits"

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


def first_existing(paths):
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            return p
    return None


def pick_tracecoords_file(traceset: str, want_reg: bool) -> Path:
    st06 = Path(config.ST06_SCIENCE)

    preferred = st06 / f"science_tracecoords_{traceset.lower()}.fits"
    if preferred.exists():
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


def get_slit_hdu(hdul: fits.HDUList, slit_name: str):
    slit_name = slit_name.strip().upper()
    for h in hdul[1:]:
        ext = (h.header.get("EXTNAME") or h.name or "").strip().upper()
        if ext == slit_name:
            return h
    raise KeyError(f"Slit {slit_name} not found.")


def compute_row_footprint(img: np.ndarray):
    ny, nx = img.shape
    finite = np.isfinite(img)

    x_left = np.full(ny, np.nan, float)
    x_right = np.full(ny, np.nan, float)
    width = np.zeros(ny, int)

    for y in range(ny):
        xs = np.where(finite[y])[0]
        if xs.size == 0:
            continue
        x_left[y] = float(xs[0])
        x_right[y] = float(xs[-1])
        width[y] = int(xs.size)

    blank_frac = float(np.mean(width == 0))
    return x_left, x_right, width, blank_frac


def write_report(path: Path, infile: Path, slit: str, hdr, width: np.ndarray,
                 x_left: np.ndarray, x_right: np.ndarray, blank_frac: float):
    finite_w = width[width > 0]

    with path.open("w") as f:
        f.write(f"Step06c QC — single slit diagnostic: {slit}\n")
        f.write("=" * 68 + "\n")
        f.write(f"Input file : {infile}\n")
        f.write(f"Slit       : {slit}\n\n")

        f.write("Footprint statistics:\n")
        f.write(f" rows total                = {len(width)}\n")
        f.write(f" blank row fraction        = {blank_frac:.6f}\n")
        if finite_w.size:
            f.write(f" median finite width       = {np.nanmedian(finite_w):.3f}\n")
            f.write(f" robust sigma(width)       = {robust_sigma(finite_w):.3f}\n")
            f.write(f" min/max finite width      = {np.nanmin(finite_w)}, {np.nanmax(finite_w)}\n")
        if np.isfinite(x_left).any():
            f.write(f" median x_left             = {np.nanmedian(x_left):.3f}\n")
            f.write(f" median x_right            = {np.nanmedian(x_right):.3f}\n")
            f.write(f" robust sigma(x_left)      = {robust_sigma(x_left):.3f}\n")
            f.write(f" robust sigma(x_right)     = {robust_sigma(x_right):.3f}\n")
        f.write("\nHeader audit:\n")
        for k in ["EXTNAME", "TRACESET", "SLITID", "YMIN", "Y0DET", "XREF", "REGFLAT", "FLATDX"]:
            if k in hdr:
                f.write(f" {k:10s} = {hdr[k]}\n")


def main():
    ap = argparse.ArgumentParser(description="Step06c QC — single-slit TRACECOORDS diagnostics")
    ap.add_argument("--traceset", required=True, choices=["EVEN", "ODD", "even", "odd"],
                    help="Trace set to inspect.")
    ap.add_argument("--slit", required=True, help="Slit extension to inspect, e.g. SLIT018")
    ap.add_argument("--reg", action="store_true", help="Use the registered Step06c output (REGFLAT=True).")
    ap.add_argument("fitsfile", nargs="?", default=None, help="Optional explicit Step06c MEF file.")
    ap.add_argument("--outdir", default=None, help="Optional output directory root")
    ap.add_argument("--p-lo", type=float, default=5.0, help="Lower percentile for image stretch")
    ap.add_argument("--p-hi", type=float, default=99.5, help="Upper percentile for image stretch")
    args = ap.parse_args()

    traceset = args.traceset.upper()
    slit = args.slit.strip().upper()

    if args.fitsfile is not None:
        fitsfile = Path(args.fitsfile)
    else:
        fitsfile = pick_tracecoords_file(traceset, args.reg)

    if not fitsfile.exists():
        raise FileNotFoundError(fitsfile)

    outroot = Path(args.outdir) if args.outdir else (Path(config.ST06_SCIENCE) / f"qc_step06c_{traceset.lower()}")
    outroot.mkdir(parents=True, exist_ok=True)

    tag = fitsfile.stem
    png_path = outroot / f"{tag}_{slit}_single_qc.png"
    txt_path = outroot / f"{tag}_{slit}_single_qc_report.txt"

    with fits.open(fitsfile) as hdul:
        h = get_slit_hdu(hdul, slit)
        img = np.array(h.data, dtype=float)
        hdr = h.header.copy()

    x_left, x_right, width, blank_frac = compute_row_footprint(img)
    yy = np.arange(img.shape[0], dtype=float)
    vmin, vmax = robust_limits(img, p_lo=args.p_lo, p_hi=args.p_hi)

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    ax[0, 0].imshow(img, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax[0, 0].set_title(f"{slit} image")
    ax[0, 0].set_xlabel("X")
    ax[0, 0].set_ylabel("Y")

    ax[0, 1].plot(width, yy, lw=0.8)
    ax[0, 1].set_title("Finite width per row")
    ax[0, 1].set_xlabel("Width (pixels)")
    ax[0, 1].set_ylabel("Y")

    ax[1, 0].plot(x_left, yy, lw=0.8, label="x_left")
    ax[1, 0].plot(x_right, yy, lw=0.8, label="x_right")
    ax[1, 0].set_title("Finite-footprint boundaries")
    ax[1, 0].set_xlabel("X")
    ax[1, 0].set_ylabel("Y")
    ax[1, 0].legend()

    finite_w = width[width > 0]
    ax[1, 1].hist(finite_w, bins=max(10, min(50, len(finite_w)//10 if len(finite_w) else 10)), histtype="step")
    ax[1, 1].set_title("Histogram of finite row width")
    ax[1, 1].set_xlabel("Width")
    ax[1, 1].set_ylabel("N rows")

    fig.suptitle(f"Step06c single-slit QC — {tag} — {slit}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    write_report(txt_path, fitsfile, slit, hdr, width, x_left, x_right, blank_frac)

    print("\n====================================================")
    print("Step06c single-slit QC")
    print("====================================================")
    print("Input :", fitsfile)
    print("Slit  :", slit)
    print("PNG   :", png_path)
    print("Report:", txt_path)
    print()
    print(f"blank row fraction   = {blank_frac:.6f}")
    if finite_w.size:
        print(f"median finite width  = {np.nanmedian(finite_w):.3f}")
        print(f"robust sigma(width)  = {robust_sigma(finite_w):.3f}")
        print(f"min/max finite width = {np.nanmin(finite_w)}, {np.nanmax(finite_w)}")
    for k in ["TRACESET", "SLITID", "YMIN", "Y0DET", "XREF", "REGFLAT", "FLATDX"]:
        if k in hdr:
            print(f"{k:10s} = {hdr[k]}")
    print("====================================================")
    print(f"[DONE] Wrote {png_path}")
    print(f"[DONE] Wrote {txt_path}")


if __name__ == "__main__":
    main()
