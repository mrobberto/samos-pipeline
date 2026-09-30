#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC Step12 — final spectrophotometric spectra mosaic.

Creates a paper-style per-slit mosaic from the stored Step12 final product.
The default plotted column is FLUX_FLAM_STELLARRESP.

Important
---------
This QC displays the column already stored in the final FITS product. It does
not reconstruct or re-apply RESP_STELLAR_MASTER. Therefore --column selects
an actual stored data column rather than requesting a new correction.

SkyMapper r/i/z points, when requested, are shown only as visual broadband
anchors at the same pivot wavelengths used by the Step12 response calibration.
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

PIVOT_R_NM = 613.8440330950057
PIVOT_I_NM = 776.79762950059
PIVOT_Z_NM = 914.5992987637427


def parse_args():
    p = argparse.ArgumentParser(description="Step12 final spectrophotometric spectra mosaic QC")
    p.add_argument("--column", default="FLUX_FLAM_STELLARRESP", help="Column to plot")
    p.add_argument("--in", dest="infile", required=True, help="Input Step12 final FITS")
    p.add_argument("--outdir", required=True, help="Output QC directory")
    p.add_argument("--maxslits", type=int, default=0, help="0 = all slits")
    p.add_argument("--xlo", type=float, default=590.0)
    p.add_argument("--xhi", type=float, default=960.0)
    p.add_argument("--ncol", type=int, default=8)
    p.add_argument("--photcat", default=None, help="Optional SkyMapper photometry CSV")
    p.add_argument(
        "--yscale",
        choices=["global", "perpanel"],
        default="perpanel",
        help="Common or independent vertical scale",
    )
    p.add_argument(
        "--ymode",
        choices=["linear", "log"],
        default="linear",
        help="Linear or logarithmic vertical axis",
    )
    p.add_argument("--ylo", type=float, default=None)
    p.add_argument("--yhi", type=float, default=None)
    p.add_argument(
        "--title",
        default="Final calibrated spectra",
        help="Overall title; use empty string for no title",
    )
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

            # Display exactly the requested stored column. Do not silently
            # reconstruct a Step12 correction from another spectrum column.
            col = str(args.column).strip()
            if col not in names:
                print(f"WARNING: {h.name} lacks requested column {col}; skipped")
                continue

            flux = np.asarray(h.data[col], float)
            if np.isfinite(flux).sum() == 0:
                print(f"WARNING: {h.name} has no finite values in {col}; skipped")
                continue

            display_mode = col

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

    global_ylim = None

    if args.yscale == "global":
        yvals = []

        for r in rows:
            lam = np.asarray(r["lam"], float)
            flux = np.asarray(r["flux"], float)

            good = (
                np.isfinite(lam)
                & np.isfinite(flux)
                & (lam >= args.xlo)
                & (lam <= args.xhi)
            )

            if args.ymode == "log":
                good &= flux > 0

            if np.any(good):
                yvals.append(flux[good])

            # Include SkyMapper anchors in the common scale.
            if phot is not None:
                prow = phot.loc[phot["slit"] == r["slit"]]

                if len(prow) == 1:
                    prow = prow.iloc[0]

                    for band, lam_eff in {
                        "r": PIVOT_R_NM,
                        "i": PIVOT_I_NM,
                        "z": PIVOT_Z_NM,
                    }.items():
                        mag_col = f"{band}_mag"

                        if (
                            args.xlo <= lam_eff <= args.xhi
                            and mag_col in prow.index
                            and np.isfinite(prow[mag_col])
                        ):
                            fphot = abmag_to_flam_cgs(
                                float(prow[mag_col]),
                                lam_eff,
                            )

                            if np.isfinite(fphot):
                                yvals.append(np.asarray([fphot]))

        if yvals:
            yy = np.concatenate(yvals)
            yy = yy[np.isfinite(yy)]

            if args.ymode == "log":
                yy = yy[yy > 0]

                lo = (
                    args.ylo
                    if args.ylo is not None
                    else np.nanpercentile(yy, 1.0)
                )
                hi = (
                    args.yhi
                    if args.yhi is not None
                    else np.nanpercentile(yy, 99.5)
                )

            else:
                lo, hi = np.nanpercentile(yy, [1.0, 99.0])

                if args.ylo is not None:
                    lo = args.ylo
                if args.yhi is not None:
                    hi = args.yhi

            global_ylim = (float(lo), float(hi))

    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(4 * ncol, 2.5 * nrow),
        sharex=True,
        sharey=(args.yscale == "global"),
    )

    axes = np.ravel(axes)

    for i, r in enumerate(rows):
        ax = axes[i]

        lam = r["lam"]
        flux = r["flux"]

        mode = r["mode"]
        physical = mode in {"FLUX_FLAM", "FLUX_FLAM_STELLARRESP"}
        line_color = "k" if physical else "crimson"
        ylabel = r"$f_\lambda$" if physical else r"ADU s$^{-1}$ $\AA^{-1}$"

        # robust clipping
        vals = flux[np.isfinite(flux)]
        if vals.size > 10:
            lo, hi = np.percentile(vals, [1, 99])
            flux_plot = np.clip(flux, lo, hi)
        else:
            flux_plot = flux
        
        ax.plot(lam, flux_plot, color=line_color, linewidth=0.7)
        
        if physical and phot is not None:
            prow = phot.loc[phot["slit"] == r["slit"]]
        
            if len(prow) == 1:
                prow = prow.iloc[0]
                for band, lam_eff in {"r": PIVOT_R_NM, "i": PIVOT_I_NM, "z": PIVOT_Z_NM}.items():
                    mag_col = f"{band}_mag"
                    if mag_col in prow.index and np.isfinite(prow[mag_col]):
                        fphot = abmag_to_flam_cgs(float(prow[mag_col]), lam_eff)
                        ax.scatter(
                            [lam_eff], [fphot],
                            s=70,
                            marker="o",
                            facecolor="gold",
                            edgecolor="k",
                            linewidth=1.0,
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

        if args.ymode == "log":
            ax.set_yscale("log")

        if global_ylim is not None:
            ax.set_ylim(*global_ylim)

        elif args.ymode == "log":
            vals = np.asarray(flux, float)
            vals = vals[np.isfinite(vals) & (vals > 0)]

            if vals.size:
                lo = (
                    args.ylo
                    if args.ylo is not None
                    else np.nanpercentile(vals, 1.0)
                )
                hi = (
                    args.yhi
                    if args.yhi is not None
                    else np.nanpercentile(vals, 99.5)
                )
                ax.set_ylim(lo, hi)

        else:
            lo, hi = safe_ylim(flux)

            if args.ylo is not None:
                lo = args.ylo
            if args.yhi is not None:
                hi = args.yhi

            ax.set_ylim(lo, hi)

        ax.set_title(
            f"{r['slit']}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.20)

    for j in range(len(rows), len(axes)):
        axes[j].axis("off")

    out_png = outdir / "QC_step12_final_spectra_subplots.png"
    out_pdf = outdir / "QC_step12_final_spectra_subplots.pdf"

    if args.title.strip():
        plt.tight_layout(rect=[0, 0, 1, 0.965])
        fig.suptitle(args.title, fontsize=16)
    else:
        plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)

    print()
    print("Step12 final mosaic QC")
    print("  input :", infile)
    print("  column:", args.column)
    print("  slits :", len(rows))
    print("Wrote:")
    print(" ", out_png)
    print(" ", out_pdf)


if __name__ == "__main__":
    main()
