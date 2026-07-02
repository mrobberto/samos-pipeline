#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC Step09 — final sky-subtracted spectra mosaic.

Creates a paper-style per-slit mosaic from the final Step09 product,
normally using STELLAR_CONSENSUS.

Example
-------
PYTHONPATH=. python qc/step09/qc_step09_final_mosaic.py \
  --in ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/09_abab/extract1d_optimal_ridge_all_wav_step09_abab_preferred_consensus.fits \
  --outdir ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/09_abab/qc_step09 \
  --column STELLAR_CONSENSUS
"""

import argparse
from pathlib import Path
import math
import re

import numpy as np
from astropy.io import fits

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Telluric markers retained for visual registration
O2_B_REF = 686.7
O2_A_REF = 760.5
BAND_B = (685.0, 690.0)
BAND_A = (758.0, 770.0)


def parse_args():
    p = argparse.ArgumentParser(description="Step09 final spectra mosaic QC")
    p.add_argument("--in", dest="infile", required=True, help="Input Step09 final FITS")
    p.add_argument("--outdir", required=True, help="Output QC directory")
    p.add_argument("--column", default="STELLAR_CONSENSUS", help="Column to plot")
    p.add_argument("--maxslits", type=int, default=0, help="0 = all slits")
    p.add_argument("--xlo", type=float, default=590.0)
    p.add_argument("--xhi", type=float, default=960.0)
    p.add_argument("--ncol", type=int, default=8)
    return p.parse_args()


def is_slit_ext(hdu):
    return (hdu.name or "").upper().startswith("SLIT")


def slit_num(name):
    m = re.match(r"SLIT(\d+)", (name or "").upper())
    return int(m.group(1)) if m else 10**9


def robust_norm(flux):
    vals = flux[np.isfinite(flux)]
    if vals.size < 10:
        return flux

    med = np.nanmedian(vals)
    if np.isfinite(med) and med != 0:
        return flux / med

    scale = np.nanpercentile(np.abs(vals), 75)
    if np.isfinite(scale) and scale > 0:
        return flux / scale

    return flux


def safe_ylim(flux):
    vals = flux[np.isfinite(flux)]
    if vals.size < 10:
        return (-1, 1)

    lo, hi = np.nanpercentile(vals, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return (-1, 1)

    pad = 0.15 * (hi - lo)
    return lo - pad, hi + pad


def main():
    args = parse_args()

    infile = Path(args.infile)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not infile.exists():
        raise FileNotFoundError(infile)

    rows = []

    with fits.open(infile, memmap=False) as hdul:
        slit_hdus = sorted(
            [h for h in hdul[1:] if is_slit_ext(h)],
            key=lambda h: slit_num(h.name),
        )

        if args.maxslits and args.maxslits > 0:
            slit_hdus = slit_hdus[:args.maxslits]

        for h in slit_hdus:
            names = h.columns.names

            if "LAMBDA_NM" not in names:
                continue

            if args.column not in names:
                print(f"WARNING: {h.name} missing column {args.column}; skipped")
                continue

            lam = np.asarray(h.data["LAMBDA_NM"], float)
            flux = np.asarray(h.data[args.column], float)

            ok = np.isfinite(lam) & np.isfinite(flux)
            if ok.sum() < 10:
                continue

            rows.append(
                dict(
                    slit=h.name.upper(),
                    lam=lam[ok],
                    flux=flux[ok],
                )
            )

    if not rows:
        raise RuntimeError(f"No usable slit spectra found for column {args.column}")

    n = len(rows)
    ncol = int(args.ncol)
    nrow = math.ceil(n / ncol)

    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(4 * ncol, 2.5 * nrow),
        sharex=True,
        sharey=False,
    )

    axes = np.ravel(axes)

    for i, r in enumerate(rows):
        ax = axes[i]

        lam = r["lam"]
        flux = robust_norm(r["flux"])
        """
        ax.plot(lam, flux, color="k", linewidth=0.8)
        """
        # robust clipping
        vals = flux[np.isfinite(flux)]
        if vals.size > 10:
            lo, hi = np.percentile(vals, [1, 99])
            flux_plot = np.clip(flux, lo, hi)
        else:
            flux_plot = flux
        
        ax.plot(lam, flux_plot, color="k", linewidth=0.7)


        # Telluric visual references
        ax.axvline(O2_B_REF, color="red", linestyle="--", linewidth=0.8)
        ax.axvline(O2_A_REF, color="red", linestyle="--", linewidth=0.8)
        ax.axvspan(BAND_B[0], BAND_B[1], color="red", alpha=0.08)
        ax.axvspan(BAND_A[0], BAND_A[1], color="red", alpha=0.08)

        ax.set_xlim(args.xlo, args.xhi)
        ax.set_ylim(*safe_ylim(flux))
        ax.set_title(r["slit"], fontsize=8)
        ax.grid(True, alpha=0.20)

    for j in range(len(rows), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Step09 QC — final sky-subtracted spectra ({args.column})",
        fontsize=16,
    )

    out_png = outdir / "QC_step09_final_spectra_subplots.png"
    out_pdf = outdir / "QC_step09_final_spectra_subplots.pdf"

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)

    print()
    print("Step09 final mosaic QC")
    print("  input :", infile)
    print("  column:", args.column)
    print("  slits :", len(rows))
    print("Wrote:")
    print(" ", out_png)
    print(" ", out_pdf)


if __name__ == "__main__":
    main()
