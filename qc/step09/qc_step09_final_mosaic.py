#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
QC Step09 final mosaic in wavelength space.

Default behavior
----------------
- reads the merged official Step09 product
- plots one panel per slit
- default signal = STELLAR
- overlays telluric guides
- uses global robust scaling across all slits

Typical use
-----------
PYTHONPATH=. python qc/step09/qc_step09_final_mosaic.py

Optional examples
-----------------
PYTHONPATH=. python qc/step09/qc_step09_final_mosaic.py --column OH_MODEL
PYTHONPATH=. python qc/step09/qc_step09_final_mosaic.py --column RESID_POSTOH
"""


import argparse
import math
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

import config


TELLURIC_LINES = [686.7, 760.5]
TELLURIC_WINDOWS = [685.3, 687.9, 758.5, 762.5]


def slit_num(name: str) -> int:
    m = re.match(r"SLIT(\d+)", str(name).upper())
    return int(m.group(1)) if m else 9999


def finite_arr(x):
    x = np.asarray(x, float)
    if np.ma.isMaskedArray(x):
        x = x.filled(np.nan)
    return np.ravel(x)


def parse_args():
    p = argparse.ArgumentParser(description="Step09 final mosaic QC in wavelength space")
    p.add_argument(
        "--infile",
        type=Path,
        default=None,
        help="Merged Step09 file (default: config.ST09_OH_REFINE/extract1d_optimal_ridge_all_wav_ohclean.fits)",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: config.ST09_OH_REFINE/qc_step09)",
    )
    p.add_argument(
        "--column",
        type=str,
        default="STELLAR",
        help="Column to plot: STELLAR, OBJ_PRESKY, OH_MODEL, RESID_POSTOH, etc.",
    )
    p.add_argument(
        "--ncols",
        type=int,
        default=6,
        help="Number of mosaic columns",
    )
    p.add_argument(
        "--show-pref",
        action="store_true",
        help="Append STEP09_PREF to each slit title when available",
    )
    p.add_argument(
        "--png-name",
        type=str,
        default=None,
        help="Optional output PNG filename",
    )
    p.add_argument(
        "--pdf-name",
        type=str,
        default=None,
        help="Optional output PDF filename",
    )
    return p.parse_args()


def main():
    args = parse_args()

    st09 = Path(config.ST09_OH_REFINE)
    infile = args.infile if args.infile else st09 / "extract1d_optimal_ridge_all_wav_ohclean.fits"
    outdir = args.outdir if args.outdir else st09 / "qc_step09"
    outdir.mkdir(parents=True, exist_ok=True)

    if not infile.exists():
        raise FileNotFoundError(f"Missing input file: {infile}")

    plot_col = args.column.strip().upper()

    slit_data = {}

    with fits.open(infile) as hdul:
        for hdu in hdul[1:]:
            name = str(hdu.name).strip().upper()
            if not name.startswith("SLIT") or hdu.data is None:
                continue

            cols = [c.upper() for c in hdu.data.names]
            if "LAMBDA_NM" not in cols:
                continue
            if plot_col not in cols:
                continue

            lam = finite_arr(hdu.data["LAMBDA_NM"])
            sig = finite_arr(hdu.data[plot_col])

            pref = None
            if "STEP09_PREF" in cols:
                raw = hdu.data["STEP09_PREF"][0]
                pref = raw.decode(errors="ignore").strip() if hasattr(raw, "decode") else str(raw).strip()

            slit_data[name] = {
                "lam": lam,
                "sig": sig,
                "pref": pref,
            }

    slits = sorted(slit_data.keys(), key=slit_num)
    if not slits:
        raise RuntimeError(f"No valid SLIT* extensions with column {plot_col} found in {infile}")

    all_lam = []
    all_sig = []
    for slit in slits:
        lam = slit_data[slit]["lam"]
        sig = slit_data[slit]["sig"]
        good = np.isfinite(lam) & np.isfinite(sig)
        if np.any(good):
            all_lam.append(lam[good])
            all_sig.append(sig[good])

    if not all_sig:
        raise RuntimeError(f"No finite data found for column {plot_col}")

    all_lam = np.concatenate(all_lam)
    all_sig = np.concatenate(all_sig)

    x_lo, x_hi = np.nanpercentile(all_lam, [1, 99])
    y_lo, y_hi = np.nanpercentile(all_sig, [5, 95])

    ncols = max(1, int(args.ncols))
    nrows = math.ceil(len(slits) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.set_visible(False)

    for ax, slit in zip(axes, slits):
        ax.set_visible(True)
        lam = slit_data[slit]["lam"]
        sig = slit_data[slit]["sig"]
        pref = slit_data[slit]["pref"]

        good = np.isfinite(lam) & np.isfinite(sig)
        if not np.any(good):
            continue

        ax.plot(lam, sig, color="k", lw=0.8)
        ax.axhline(0.0, color="0.7", lw=0.5)

        for x in TELLURIC_LINES:
            ax.axvline(x, color="cyan", lw=0.8, ls="--", alpha=0.7)
        for x in TELLURIC_WINDOWS:
            ax.axvline(x, color="cyan", lw=0.4, ls=":", alpha=0.45)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

        title = slit
        if args.show_pref and pref:
            title = f"{slit} ({pref})"
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    png_name = args.png_name or f"all_{plot_col.lower()}_wavelength_montage.png"
    pdf_name = args.pdf_name or f"all_{plot_col.lower()}_wavelength_montage.pdf"

    fig.suptitle(f"Step09 final mosaic: {plot_col} (wavelength space, global scale)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_png = outdir / png_name
    out_pdf = outdir / pdf_name
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_pdf)
    plt.close(fig)

    print("Wrote:", out_png)
    print("Wrote:", out_pdf)


if __name__ == "__main__":
    main()
