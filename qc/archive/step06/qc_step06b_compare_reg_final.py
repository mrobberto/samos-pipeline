#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step06b QC — compare registered vs non-registered flat correction.

PURPOSE
-------
Compare the two Step06b branches for one trace set:

- non-registered flat correction
- registered flat correction

This QC is intended to show whether flat registration changes the corrected
science frame in a beneficial, neutral, or harmful way.

It is especially useful when evaluating whether REGFLAT=True introduces
spectral deformation or edge artifacts relative to the default non-registered
branch.

INPUTS
------
From config.ST06_SCIENCE:

- science_pixflatcorr_even.fits / science_pixflatcorr_odd.fits
- science_pixflatcorr_reg_even.fits / science_pixflatcorr_reg_odd.fits

Optional reference products:
- Step06a FinalScience mosaic
- Step04 trace mask (preferred *_mask_reg, fallback *_mask)

OUTPUTS
-------
Written under config.ST06_SCIENCE:

- qc_step06b_even/
- qc_step06b_odd/

Each QC directory contains:
- one comparison PNG
- one text report

CHECKS PERFORMED
----------------
- direct image comparison: non-reg vs reg
- difference image: reg - nonreg
- ratio image: reg / nonreg
- histogram of masked pixel differences
- header audit:
    REGFLAT, FLATDX, TRACESET, FLCLIPLO, FLCLIPHI

INTERPRETATION
--------------
Good behavior:
- reg and non-reg look nearly identical
- reg - nonreg is small and structureless
- ratio is close to unity inside the mask

Warning signs:
- coherent residual structure along traces
- edge distortions or sky truncation
- large FLATDX or inconsistent headers
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


def pct(x: np.ndarray, lo: float, hi: float):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    return float(np.nanpercentile(x, lo)), float(np.nanpercentile(x, hi))


def first_existing(paths):
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            return p
    return None


def pick_mask_file(st04: Path, traceset: str, explicit: str | None = None) -> Path:
    base = "Even_traces" if traceset == "EVEN" else "Odd_traces"
    return first_existing([
        Path(explicit) if explicit else None,
        st04 / f"{base}_mask_reg.fits",
        st04 / f"{base}_mask.fits",
    ])


def write_report(path: Path, traceset: str, noreg_path: Path, reg_path: Path, mask_path: Path,
                 hdr_noreg, hdr_reg, vals_diff: np.ndarray, vals_ratio: np.ndarray) -> None:
    with path.open("w") as f:
        f.write(f"Step06b reg-vs-noreg QC for {traceset}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Non-reg : {noreg_path}\n")
        f.write(f"Reg     : {reg_path}\n")
        f.write(f"Mask    : {mask_path}\n\n")

        f.write("Inside mask:\n")
        f.write(f" N pixels                 = {len(vals_diff)}\n")
        f.write(f" median(reg-noreg)        = {np.nanmedian(vals_diff):.6e}\n")
        f.write(f" robust_sigma(reg-noreg)  = {robust_sigma(vals_diff):.6e}\n")
        f.write(f" p1, p99(reg-noreg)       = {np.nanpercentile(vals_diff,1):.6e}, {np.nanpercentile(vals_diff,99):.6e}\n\n")

        f.write(f" median(reg/noreg)        = {np.nanmedian(vals_ratio):.6f}\n")
        f.write(f" robust_sigma(reg/noreg)  = {robust_sigma(vals_ratio):.6f}\n")
        f.write(f" p1, p99(reg/noreg)       = {np.nanpercentile(vals_ratio,1):.6f}, {np.nanpercentile(vals_ratio,99):.6f}\n\n")

        f.write("Header audit (non-reg):\n")
        for k in ["TRACESET", "REGFLAT", "FLATDX", "FLCLIPLO", "FLCLIPHI", "PIXFLAT", "MASKFILE"]:
            if k in hdr_noreg:
                f.write(f" {k:10s} = {hdr_noreg[k]}\n")
        f.write("\nHeader audit (reg):\n")
        for k in ["TRACESET", "REGFLAT", "FLATDX", "FLCLIPLO", "FLCLIPHI", "PIXFLAT", "MASKFILE"]:
            if k in hdr_reg:
                f.write(f" {k:10s} = {hdr_reg[k]}\n")


def main():
    ap = argparse.ArgumentParser(description="Step06b QC — compare reg vs non-reg flat correction")
    ap.add_argument("--traceset", default="EVEN", choices=["EVEN", "ODD", "even", "odd"])
    ap.add_argument("--noreg", default=None, help="Explicit non-registered Step06b file")
    ap.add_argument("--reg", default=None, help="Explicit registered Step06b file")
    ap.add_argument("--mask", default=None, help="Explicit Step04 mask file")
    ap.add_argument("--outdir", default=None, help="Optional output directory")
    args = ap.parse_args()

    traceset = args.traceset.upper()
    suffix = traceset.lower()

    st04 = Path(config.ST04_TRACES)
    st06 = Path(config.ST06_SCIENCE)

    noreg_path = first_existing([
        Path(args.noreg) if args.noreg else None,
        st06 / f"science_pixflatcorr_{suffix}.fits",
    ])
    reg_path = first_existing([
        Path(args.reg) if args.reg else None,
        st06 / f"science_pixflatcorr_reg_{suffix}.fits",
    ])
    mask_path = pick_mask_file(st04, traceset, args.mask)

    if noreg_path is None or reg_path is None or mask_path is None:
        raise FileNotFoundError("Could not resolve non-reg, reg, or mask file.")

    img_noreg = fits.getdata(noreg_path).astype(float)
    img_reg = fits.getdata(reg_path).astype(float)
    mask = fits.getdata(mask_path).astype(bool)
    hdr_noreg = fits.getheader(noreg_path)
    hdr_reg = fits.getheader(reg_path)

    diff = np.full_like(img_noreg, np.nan)
    ratio = np.full_like(img_noreg, np.nan)

    good = mask & np.isfinite(img_noreg) & np.isfinite(img_reg)
    diff[good] = img_reg[good] - img_noreg[good]

    good_ratio = good & (img_noreg != 0)
    ratio[good_ratio] = img_reg[good_ratio] / img_noreg[good_ratio]

    vals_diff = diff[mask & np.isfinite(diff)]
    vals_ratio = ratio[mask & np.isfinite(ratio)]

    if vals_diff.size == 0 or vals_ratio.size == 0:
        raise RuntimeError("No finite masked pixels available for reg-vs-noreg QC.")

    n1, n2 = pct(img_noreg[mask], 2, 98)
    r1, r2 = pct(img_reg[mask], 2, 98)
    d1, d2 = pct(vals_diff, 1, 99)
    q1, q2 = pct(vals_ratio, 1, 99)

    outdir = Path(args.outdir) if args.outdir else (st06 / f"qc_step06b_{suffix}")
    outdir.mkdir(parents=True, exist_ok=True)

    png_path = outdir / f"step06b_{suffix}_reg_compare.png"
    txt_path = outdir / f"step06b_{suffix}_reg_compare_report.txt"

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))

    ax[0, 0].imshow(img_noreg, origin="lower", vmin=n1, vmax=n2, cmap="gray")
    ax[0, 0].set_title("Non-reg")

    ax[0, 1].imshow(img_reg, origin="lower", vmin=r1, vmax=r2, cmap="gray")
    ax[0, 1].set_title("Reg")

    im = ax[0, 2].imshow(diff, origin="lower", vmin=d1, vmax=d2, cmap="gray")
    ax[0, 2].set_title("Reg - Non-reg")
    plt.colorbar(im, ax=ax[0, 2])

    im2 = ax[1, 0].imshow(ratio, origin="lower", vmin=q1, vmax=q2, cmap="gray")
    ax[1, 0].set_title("Reg / Non-reg")
    plt.colorbar(im2, ax=ax[1, 0])

    ax[1, 1].hist(vals_diff, bins=150, histtype="step")
    ax[1, 1].axvline(0.0, color="gray")
    ax[1, 1].set_title("Histogram: Reg - Non-reg")

    ax[1, 2].hist(vals_ratio, bins=150, histtype="step")
    ax[1, 2].axvline(1.0, color="gray")
    ax[1, 2].set_title("Histogram: Reg / Non-reg")

    plt.suptitle(f"Step06b reg-vs-noreg QC — {traceset}")
    plt.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    write_report(txt_path, traceset, noreg_path, reg_path, mask_path, hdr_noreg, hdr_reg, vals_diff, vals_ratio)

    print("\n====================================================")
    print(f"Step06b reg-vs-noreg QC for {traceset}")
    print("====================================================")
    print("Non-reg :", noreg_path)
    print("Reg     :", reg_path)
    print("Mask    :", mask_path)
    print()
    print("Inside mask:")
    print(f" N pixels                = {len(vals_diff)}")
    print(f" median(reg-noreg)       = {np.nanmedian(vals_diff):.6e}")
    print(f" robust_sigma(reg-noreg) = {robust_sigma(vals_diff):.6e}")
    print(f" median(reg/noreg)       = {np.nanmedian(vals_ratio):.6f}")
    print(f" robust_sigma(reg/noreg) = {robust_sigma(vals_ratio):.6f}")
    print()
    print("Header audit:")
    for lab, hdr in [("NONREG", hdr_noreg), ("REG", hdr_reg)]:
        print(f" {lab}")
        for k in ["TRACESET", "REGFLAT", "FLATDX", "FLCLIPLO", "FLCLIPHI"]:
            if k in hdr:
                print(f"   {k:10s} = {hdr[k]}")
    print("====================================================")
    print(f"[DONE] Wrote {png_path}")
    print(f"[DONE] Wrote {txt_path}")


if __name__ == "__main__":
    main()
