#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC Step09 — final sky-subtracted spectra mosaic.

Creates a paper-style per-slit mosaic from the final Step11 product,
normally using STELLAR_CONSENSUS.

Example
-------
PYTHONPATH=. python ./qc/step12/qc_step12_final_mosaic.py \
  --in     ./products/Run8_Dolidze25/reduced/11_fluxcal/extract1d_fluxcal.fits \
  --outdir ./products/Run8_Dolidze25/qc/12_finalcal \
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
    p = argparse.ArgumentParser(description="Step12 final stellar-response spectra mosaic QC")
    p.add_argument("--column", default="FLUX_FLAM_STELLARRESP", help="Column to plot")
    p.add_argument("--in", dest="infile", required=True, help="Input Step09 final FITS")
    p.add_argument("--outdir", required=True, help="Output QC directory")
    p.add_argument("--maxslits", type=int, default=0, help="0 = all slits")
    p.add_argument("--xlo", type=float, default=590.0)
    p.add_argument("--xhi", type=float, default=960.0)
    p.add_argument("--ncol", type=int, default=8)
    p.add_argument("--photcat", default=None, help="Optional SkyMapper photometry CSV")
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

def abmag_to_flam_cgs(mag_ab, lam_nm):
    fnu_cgs = 3631.0 * 10 ** (-0.4 * mag_ab) * 1e-23
    c_A_s = 2.99792458e18
    lam_A = lam_nm * 10.0
    return float(fnu_cgs * c_A_s / lam_A**2)


def main():
    args = parse_args()

    infile = Path(args.infile)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not infile.exists():
        raise FileNotFoundError(infile)

    rows = []
    
    phot = None
    if args.photcat is not None:
        import pandas as pd
        phot = pd.read_csv(args.photcat)
        phot["slit"] = phot["slit"].astype(str).str.upper().str.strip()
        

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

            col = None
            
            # --- Preferred display logic ---
            display_mode = None
            
            if (
                "FLUX_FLAM_STELLARRESP" in names and
                np.isfinite(np.asarray(h.data["FLUX_FLAM_STELLARRESP"], float)).sum() > 0
            ):
                flux = np.asarray(h.data["FLUX_FLAM_STELLARRESP"], float)
                display_mode = "FLAM_CORRECTED"
            
            else:
                base_col = None
            
                for c in [
                    "FLUX_FLAM",
                    "FLUX_TELLCOR_O2",
                    "STELLAR_CONSENSUS",
                    "FLUX",
                ]:
                    if c in names and np.isfinite(np.asarray(h.data[c], float)).sum() > 0:
                        base_col = c
                        break
            
                if base_col is None:
                    print(f"WARNING: {h.name} has no usable display spectrum; skipped")
                    continue
            
                flux = np.asarray(h.data[base_col], float).copy()
            
                if "RESP_STELLAR_MASTER" in names:
                    resp = np.asarray(h.data["RESP_STELLAR_MASTER"], float)
                    good = np.isfinite(resp) & (resp > 0)
                    flux[good] *= resp[good]
            
                display_mode = f"{base_col}_CORRECTED"
            
            lam = np.asarray(h.data["LAMBDA_NM"], float)
            ok = np.isfinite(lam) & np.isfinite(flux)
            if ok.sum() < 10:
                continue

            rows.append(
                dict(
                    slit=h.name.upper(),
                    lam=lam[ok],
                    flux=flux[ok],
                    mode=display_mode,
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
        flux = r["flux"]

        mode = r["mode"]
        physical = "FLAM" in mode
        line_color = "k" if physical else "crimson"
        ylabel = r"$f_\lambda$" if physical else r"ADU s$^{-1}$ $\AA^{-1}$"

        # robust clipping
        vals = flux[np.isfinite(flux)]
        if vals.size > 10:
            lo, hi = np.percentile(vals, [1, 99])
            flux_plot = np.clip(flux, lo, hi)
        else:
            flux_plot = flux
        
        #ax.plot(lam, flux_plot, color=line_color, linewidth=0.7)
        ax.scatter(
            lam,
            flux_plot,
            s=1,
            c=line_color,
            linewidths=0,
            rasterized=True,
        )
        
        if physical and phot is not None:
            prow = phot.loc[phot["slit"] == r["slit"]]
        
            if len(prow) == 1:
                prow = prow.iloc[0]
                for band, lam_eff in {"r": 616.0, "i": 779.0, "z": 916.0}.items():
                    mag_col = f"{band}_mag"
                    if mag_col in prow.index and np.isfinite(prow[mag_col]):
                        fphot = abmag_to_flam_cgs(float(prow[mag_col]), lam_eff)
                        ax.scatter(
                            [lam_eff], [fphot],
                            s=38,
                            marker="o",
                            facecolor="gold",
                            edgecolor="k",
                            linewidth=0.8,
                            zorder=10,
                        )
                        ax.text(
                            lam_eff + 4,
                            fphot,
                            band,
                            fontsize=7,
                            color="k",
                            va="center",
                            ha="left",
                            zorder=11,
                        )

                        


        # Telluric visual references
        ax.axvline(O2_B_REF, color="red", linestyle="--", linewidth=0.8)
        ax.axvline(O2_A_REF, color="red", linestyle="--", linewidth=0.8)
        ax.axvspan(BAND_B[0], BAND_B[1], color="red", alpha=0.08)
        ax.axvspan(BAND_A[0], BAND_A[1], color="red", alpha=0.08)

        ax.set_xlim(args.xlo, args.xhi)
        ax.set_ylim(*safe_ylim(flux))
        unit_tag = "FLAM corr." if physical else "rel. corr."
        ax.set_title(f"{r['slit']}  {unit_tag}", fontsize=8)
        ax.grid(True, alpha=0.20)

    for j in range(len(rows), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Step12 final stellar-response corrected spectra ({args.column})",
        fontsize=16,
    )
    
    out_png = outdir / "QC_step12_final_spectra_subplots.png"
    out_pdf = outdir / "QC_step12_final_spectra_subplots.pdf"

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle(
        "Step12 final spectra: navy = physical FLAM, red = relative ADU/s/A; gold points = SkyMapper",
        fontsize=16,
    )
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)

    print()
    print("Step10 final mosaic QC")
    print("  input :", infile)
    print("  column:", args.column)
    print("  slits :", len(rows))
    print("Wrote:")
    print(" ", out_png)
    print(" ", out_pdf)


if __name__ == "__main__":
    main()
