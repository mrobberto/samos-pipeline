#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step06b QC — inspect full-frame pixel-flat correction.

PURPOSE
-------
Validate the Step06b full-frame science product after pixel-flat correction.

This QC compares:
- the baseline FinalScience mosaic from Step06a
- the Step06b corrected frame
- the Step05 pixel flat
- the Step04 trace mask used to restrict the correction

It is intended to answer three questions:

1) Is the flat correction being applied only inside the slit footprint?
2) Does raw / corrected behave consistently with the applied flat?
3) Are the clipping / registration header keywords and provenance sensible?

INPUTS
------
From the pipeline directories:

- Step06a science:
    config.ST06_SCIENCE / FinalScience_*_ADUperS.fits

- Step06b corrected science:
    config.ST06_SCIENCE / science_pixflatcorr_even.fits
    config.ST06_SCIENCE / science_pixflatcorr_odd.fits
    or, when --reg is used:
    config.ST06_SCIENCE / science_pixflatcorr_reg_even.fits
    config.ST06_SCIENCE / science_pixflatcorr_reg_odd.fits

- Step05 pixel flat:
    config.ST05_FLATCORR / pixflat_even.fits
    config.ST05_FLATCORR / pixflat_odd.fits

- Step04 mask:
    prefer Even_traces_mask_reg.fits / Odd_traces_mask_reg.fits
    fallback Even_traces_mask.fits   / Odd_traces_mask.fits

OUTPUTS
-------
Written under config.ST06_SCIENCE:

- qc_step06b_even/
- qc_step06b_odd/

Each QC directory contains:
- one summary PNG
- one text report

CHECKS PERFORMED
----------------
- image comparison: raw, flat, corrected, corrected-with-mask
- raw/corrected ratio image
- histogram of raw/corrected and flat values inside the mask
- summary statistics inside the mask
- audit of key Step06b header provenance:
    TRACESET, REGFLAT, FLATDX, FLCLIPLO, FLCLIPHI

INTERPRETATION
--------------
Good behavior:
- raw/corrected is close to the flat inside the mask
- flat median is near unity
- no strong artifacts outside the trace footprint
- clipping bounds and registration bookkeeping are sensible

Warning signs:
- large discrepancy between raw/corrected and flat
- strong structure outside the mask
- excessive spread in ratio values
- wrong trace set / wrong mask / inconsistent headers
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
    return float(np.nanpercentile(x, lo)), float(np.nanpercentile(x, hi))


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
        st06 / "FinalScience_dolidze_ADUperS.fits",
        *sorted(st06.glob("FinalScience*_ADUperS*.fits"), key=lambda p: p.stat().st_mtime, reverse=True),
    ])


def pick_corr_file(st06: Path, traceset: str, use_reg: bool, explicit: str | None = None) -> Path:
    suffix = traceset.lower()
    preferred = f"science_pixflatcorr_reg_{suffix}.fits" if use_reg else f"science_pixflatcorr_{suffix}.fits"
    return first_existing([
        Path(explicit) if explicit else None,
        st06 / preferred,
        *sorted(st06.glob(f"*pixflatcorr*_{suffix}.fits"), key=lambda p: p.stat().st_mtime, reverse=True),
    ])


def pick_flat_file(st05: Path, traceset: str, explicit: str | None = None) -> Path:
    suffix = traceset.lower()
    return first_existing([
        Path(explicit) if explicit else None,
        st05 / f"pixflat_{suffix}.fits",
    ])


def pick_mask_file(st04: Path, traceset: str, explicit: str | None = None) -> Path:
    base = "Even_traces" if traceset == "EVEN" else "Odd_traces"
    return first_existing([
        Path(explicit) if explicit else None,
        st04 / f"{base}_mask_reg.fits",
        st04 / f"{base}_mask.fits",
    ])


def write_report(path: Path, traceset: str, use_reg: bool, science_path: Path, corr_path: Path,
                 flat_path: Path, mask_path: Path, hdr_corr, vals_ratio: np.ndarray, vals_flat: np.ndarray) -> None:
    with path.open("w") as f:
        f.write(f"Step06b QC summary for {traceset} (reg={use_reg})\n")
        f.write("=" * 60 + "\n")
        f.write(f"Science   : {science_path}\n")
        f.write(f"Corrected : {corr_path}\n")
        f.write(f"Flat      : {flat_path}\n")
        f.write(f"Mask      : {mask_path}\n\n")

        f.write("Inside mask:\n")
        f.write(f" N pixels                    = {len(vals_ratio)}\n")
        f.write(f" median(raw/corrected)       = {np.nanmedian(vals_ratio):.6f}\n")
        f.write(f" robust_sigma(raw/corrected) = {robust_sigma(vals_ratio):.6f}\n")
        f.write(f" p1, p99(raw/corrected)      = {np.nanpercentile(vals_ratio,1):.6f}, {np.nanpercentile(vals_ratio,99):.6f}\n\n")

        f.write(f" median(flat)                = {np.nanmedian(vals_flat):.6f}\n")
        f.write(f" robust_sigma(flat)          = {robust_sigma(vals_flat):.6f}\n")
        f.write(f" p1, p99(flat)               = {np.nanpercentile(vals_flat,1):.6f}, {np.nanpercentile(vals_flat,99):.6f}\n\n")

        f.write("Header audit:\n")
        for k in ["TRACESET", "REGFLAT", "FLATDX", "FLCLIPLO", "FLCLIPHI", "SCIENCE", "PIXFLAT", "MASKFILE"]:
            if k in hdr_corr:
                f.write(f" {k:10s} = {hdr_corr[k]}\n")


def main():
    ap = argparse.ArgumentParser(description="Step06b QC — inspect full-frame pixel-flat correction")
    ap.add_argument("--traceset", default="EVEN", choices=["EVEN", "ODD", "even", "odd"])
    ap.add_argument("--reg", action="store_true", help="Inspect the registered Step06b output")
    ap.add_argument("--science", default=None, help="Optional explicit Step06a science FITS")
    ap.add_argument("--corr", default=None, help="Optional explicit Step06b corrected FITS")
    ap.add_argument("--flat", default=None, help="Optional explicit Step05 flat FITS")
    ap.add_argument("--mask", default=None, help="Optional explicit Step04 mask FITS")
    ap.add_argument("--outdir", default=None, help="Optional output directory (default: config.ST06_SCIENCE/qc_step06b_<set>)")
    args = ap.parse_args()

    traceset = args.traceset.upper()
    use_reg = bool(args.reg)

    st04 = Path(config.ST04_PIXFLAT)
    st05 = Path(config.ST05_FLATCORR)
    st06 = Path(config.ST06_SCIENCE)

    science_path = pick_science_file(st06, args.science)
    corr_path = pick_corr_file(st06, traceset, use_reg, args.corr)
    flat_path = pick_flat_file(st05, traceset, args.flat)
    mask_path = pick_mask_file(st04, traceset, args.mask)

    if science_path is None or corr_path is None or flat_path is None or mask_path is None:
        raise FileNotFoundError("Could not resolve one or more required files.")

    sci = fits.getdata(science_path).astype(float)
    corr = fits.getdata(corr_path).astype(float)
    flat = fits.getdata(flat_path).astype(float)
    mask = fits.getdata(mask_path).astype(bool)
    hdr_corr = fits.getheader(corr_path)

    ratio = np.full_like(sci, np.nan)
    good = mask & np.isfinite(sci) & np.isfinite(corr) & (corr != 0)
    ratio[good] = sci[good] / corr[good]

    vals_ratio = ratio[mask & np.isfinite(ratio)]
    vals_flat = flat[mask & np.isfinite(flat)]

    if vals_ratio.size == 0 or vals_flat.size == 0:
        raise RuntimeError("No finite masked pixels available for QC statistics.")

    v1, v2 = pct(sci[mask], 2, 98)
    c1, c2 = pct(corr[mask], 2, 98)
    f1, f2 = pct(vals_flat, 1, 99)
    r1, r2 = pct(vals_ratio, 1, 99)

    outdir = Path(args.outdir) if args.outdir else (st06 / f"qc_step06b_{traceset.lower()}")
    outdir.mkdir(parents=True, exist_ok=True)

    tag = "reg" if use_reg else "noreg"
    png_path = outdir / f"step06b_{traceset.lower()}_{tag}_summary.png"
    txt_path = outdir / f"step06b_{traceset.lower()}_{tag}_report.txt"

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))

    ax[0, 0].imshow(sci, origin="lower", vmin=v1, vmax=v2, cmap="gray")
    ax[0, 0].set_title("Raw science")

    im1 = ax[0, 1].imshow(flat, origin="lower", vmin=f1, vmax=f2, cmap="gray")
    ax[0, 1].set_title("Pixel flat")
    plt.colorbar(im1, ax=ax[0, 1])

    ax[0, 2].imshow(corr, origin="lower", vmin=c1, vmax=c2, cmap="gray")
    ax[0, 2].set_title("Corrected")

    tmp = corr.copy()
    tmp[~mask] = np.nan
    ax[1, 0].imshow(tmp, origin="lower", vmin=c1, vmax=c2, cmap="gray")
    ax[1, 0].set_title("Corrected (mask)")

    im2 = ax[1, 1].imshow(ratio, origin="lower", vmin=r1, vmax=r2, cmap="gray")
    ax[1, 1].set_title("Raw / Corrected")
    plt.colorbar(im2, ax=ax[1, 1])

    ax[1, 2].hist(vals_ratio, bins=150, histtype="step", label="raw/corr")
    ax[1, 2].hist(vals_flat, bins=150, histtype="step", label="flat")
    ax[1, 2].axvline(1.0, color="gray")
    ax[1, 2].legend()
    ax[1, 2].set_title("Histogram")

    plt.suptitle(f"Step06b QC — {traceset} | reg={use_reg}")
    plt.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    write_report(txt_path, traceset, use_reg, science_path, corr_path, flat_path, mask_path,
                 hdr_corr, vals_ratio, vals_flat)

    print("\n====================================================")
    print(f"Step06b QC summary for {traceset} (reg={use_reg})")
    print("====================================================")
    print("Science   :", science_path)
    print("Corrected :", corr_path)
    print("Flat      :", flat_path)
    print("Mask      :", mask_path)
    print()
    print("Inside mask:")
    print(f" N pixels                     = {len(vals_ratio)}")
    print(f" median(raw/corrected)        = {np.nanmedian(vals_ratio):.6f}")
    print(f" robust_sigma(raw/corrected)  = {robust_sigma(vals_ratio):.6f}")
    print(f" p1, p99(raw/corrected)       = {np.nanpercentile(vals_ratio,1):.6f}, {np.nanpercentile(vals_ratio,99):.6f}")
    print()
    print(f" median(flat)                 = {np.nanmedian(vals_flat):.6f}")
    print(f" robust_sigma(flat)           = {robust_sigma(vals_flat):.6f}")
    print(f" p1, p99(flat)                = {np.nanpercentile(vals_flat,1):.6f}, {np.nanpercentile(vals_flat,99):.6f}")
    print()
    print("Header:")
    for k in ["TRACESET", "REGFLAT", "FLATDX", "FLCLIPLO", "FLCLIPHI"]:
        if k in hdr_corr:
            print(f" {k:10s} = {hdr_corr[k]}")
    print("====================================================")
    print(f"[DONE] Wrote {png_path}")
    print(f"[DONE] Wrote {txt_path}")


if __name__ == "__main__":
    main()
