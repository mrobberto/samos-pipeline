#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QC Step09 — empirical resolving power from fitted OH lines.

Uses the Step09 OH component supertable plus the wavelength-attached spectra
to estimate:

    FWHM_nm = FWHM_pix * d(lambda)/d(pixel)
    R       = lambda / FWHM_nm

Outputs:
- CSV table of per-line resolving power
- PNG/PDF plots of R(lambda)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--components", type=Path, required=True,
                   help="Step09 OH component supertable CSV")
    p.add_argument("--extract", type=Path, required=True,
                   help="FITS file with LAMBDA_NM and YPIX per slit")
    p.add_argument("--outdir", type=Path, required=True,
                   help="Output directory")
    p.add_argument("--min-fwhm-pix", type=float, default=1.0)
    p.add_argument("--max-fwhm-pix", type=float, default=20.0)
    p.add_argument("--min-r", type=float, default=200.0)
    p.add_argument("--max-r", type=float, default=20000.0)
    return p.parse_args()


def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def local_dispersion_nm_per_pix(hdul, slit, lam_nm, ypix=None):
    """
    Estimate local d(lambda)/d(pixel) from the slit wavelength vector.
    If ypix is provided, evaluate near that pixel.
    Otherwise evaluate near lam_nm.
    """
    if slit not in hdul:
        return np.nan

    d = hdul[slit].data
    if d is None:
        return np.nan

    cols = d.columns.names
    if "LAMBDA_NM" not in cols:
        return np.nan

    lam = np.asarray(d["LAMBDA_NM"], float)

    if "YPIX" in cols:
        pix = np.asarray(d["YPIX"], float)
    else:
        pix = np.arange(len(lam), dtype=float)

    ok = np.isfinite(lam) & np.isfinite(pix)
    if ok.sum() < 20:
        return np.nan

    lam = lam[ok]
    pix = pix[ok]

    order = np.argsort(pix)
    pix = pix[order]
    lam = lam[order]

    dlam_dpix = np.gradient(lam, pix)

    if ypix is not None and np.isfinite(ypix):
        idx = np.nanargmin(np.abs(pix - ypix))
    else:
        idx = np.nanargmin(np.abs(lam - lam_nm))

    return float(abs(dlam_dpix[idx]))


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.components)
    
    # ------------------------------------------------------------
    # Case 1: global Step09 line supertable
    # Already contains wavelength and sigma in nm.
    # ------------------------------------------------------------
    if {"LAMBDA_MED_NM", "SIGMA_MED_NM"}.issubset(df.columns):
        out = df.copy()
    
        out["lambda_nm"] = out["LAMBDA_MED_NM"].astype(float)
        out["sigma_nm"] = out["SIGMA_MED_NM"].astype(float)
        out["fwhm_nm"] = 2.354820045 * out["sigma_nm"]
        out["R"] = out["lambda_nm"] / out["fwhm_nm"]
    
        good = (
            np.isfinite(out["lambda_nm"]) &
            np.isfinite(out["fwhm_nm"]) &
            (out["fwhm_nm"] > 0) &
            (out["R"] > args.min_r) &
            (out["R"] < args.max_r)
        )
        out = out.loc[good].copy()
    
        outcsv = args.outdir / "qc_step09_oh_resolution_table.csv"
        out.to_csv(outcsv, index=False)
    
        print("[OK] Wrote:", outcsv)
        print("Rows:", len(out))
        print(out["R"].describe())
    
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(out["lambda_nm"], out["R"], s=20, alpha=0.7)
    
        bins = np.linspace(out["lambda_nm"].min(), out["lambda_nm"].max(), 15)
        xmed, ymed = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (out["lambda_nm"] >= lo) & (out["lambda_nm"] < hi)
            if m.sum() >= 2:
                xmed.append(0.5 * (lo + hi))
                ymed.append(np.nanmedian(out.loc[m, "R"]))
    
        if xmed:
            ax.plot(xmed, ymed, lw=2, label="running median")
    
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(r"Resolving power $R = \lambda / \Delta\lambda$")
        ax.set_title("Step09 QC — resolving power from OH line supertable")
        ax.grid(True, alpha=0.25)
        ax.legend()
    
        outpng = args.outdir / "qc_step09_oh_resolution_vs_lambda.png"
        outpdf = args.outdir / "qc_step09_oh_resolution_vs_lambda.pdf"
        fig.tight_layout()
        fig.savefig(outpng, dpi=150)
        fig.savefig(outpdf)
        plt.close(fig)
    
        print("[OK] Wrote:", outpng)
        print("[OK] Wrote:", outpdf)
        return
    


    slit_col = find_col(df, ["slit", "SLIT", "slit_id"])
    lam_col = find_col(df, ["lambda_nm", "lam_nm", "lambda", "center_nm"])
    ypix_col = find_col(df, ["ypix", "YPIX", "mu_pix", "center_pix", "pix"])
    fwhm_pix_col = find_col(df, ["fwhm_pix", "FWHM_PIX", "fwhm", "fwhm_px"])
    sigma_pix_col = find_col(df, ["sigma_pix", "SIGMA_PIX", "sigma", "sigma_px"])
    fwhm_nm_col = find_col(df, ["fwhm_nm", "FWHM_NM"])

    if slit_col is None or lam_col is None:
        raise KeyError(f"Need slit and wavelength columns. Found: {list(df.columns)}")

    if fwhm_pix_col is None and sigma_pix_col is None and fwhm_nm_col is None:
        raise KeyError(
            "Need one of fwhm_pix, sigma_pix, or fwhm_nm in the component table. "
            f"Found: {list(df.columns)}"
        )

    rows = []

    with fits.open(args.extract) as hdul:
        for _, r in df.iterrows():
            slit = str(r[slit_col]).strip().upper()
            if not slit.startswith("SLIT"):
                continue

            lam_nm = float(r[lam_col])
            if not np.isfinite(lam_nm):
                continue

            ypix = None
            if ypix_col is not None:
                ypix = float(r[ypix_col])

            if fwhm_nm_col is not None:
                fwhm_nm = float(r[fwhm_nm_col])
                fwhm_pix = np.nan
                disp = np.nan
            else:
                if fwhm_pix_col is not None:
                    fwhm_pix = float(r[fwhm_pix_col])
                else:
                    sigma_pix = float(r[sigma_pix_col])
                    fwhm_pix = 2.354820045 * sigma_pix

                if not np.isfinite(fwhm_pix):
                    continue

                if not (args.min_fwhm_pix <= fwhm_pix <= args.max_fwhm_pix):
                    continue

                disp = local_dispersion_nm_per_pix(hdul, slit, lam_nm, ypix=ypix)
                if not np.isfinite(disp) or disp <= 0:
                    continue

                fwhm_nm = fwhm_pix * disp

            if not np.isfinite(fwhm_nm) or fwhm_nm <= 0:
                continue

            R = lam_nm / fwhm_nm

            if not (args.min_r <= R <= args.max_r):
                continue

            rows.append({
                "slit": slit,
                "lambda_nm": lam_nm,
                "ypix": ypix if ypix is not None else np.nan,
                "fwhm_pix": fwhm_pix,
                "disp_nm_per_pix": disp,
                "fwhm_nm": fwhm_nm,
                "R": R,
            })

    out = pd.DataFrame(rows)
    outcsv = args.outdir / "qc_step09_oh_resolution_table.csv"
    out.to_csv(outcsv, index=False)

    print("[OK] Wrote:", outcsv)
    print("Rows:", len(out))

    if out.empty:
        print("No valid lines after filtering.")
        return

    print()
    print("Resolving power summary:")
    print(out["R"].describe())

    # ------------------------------------------------------------
    # Plot R versus wavelength
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(out["lambda_nm"], out["R"], s=12, alpha=0.45)

    # running median
    bins = np.linspace(np.nanmin(out["lambda_nm"]), np.nanmax(out["lambda_nm"]), 18)
    xmed, ymed = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (out["lambda_nm"] >= lo) & (out["lambda_nm"] < hi)
        if m.sum() >= 3:
            xmed.append(0.5 * (lo + hi))
            ymed.append(np.nanmedian(out.loc[m, "R"]))

    if xmed:
        ax.plot(xmed, ymed, lw=2, label="running median")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"Resolving power $R = \lambda / \Delta\lambda$")
    ax.set_title("Step09 QC — empirical resolving power from OH lines")
    ax.grid(True, alpha=0.25)
    ax.legend()

    outpng = args.outdir / "qc_step09_oh_resolution_vs_lambda.png"
    outpdf = args.outdir / "qc_step09_oh_resolution_vs_lambda.pdf"

    fig.tight_layout()
    fig.savefig(outpng, dpi=150)
    fig.savefig(outpdf)
    plt.close(fig)

    print("[OK] Wrote:", outpng)
    print("[OK] Wrote:", outpdf)

    # ------------------------------------------------------------
    # Per-slit median R
    # ------------------------------------------------------------
    per_slit = (
        out.groupby("slit")
        .agg(
            n_lines=("R", "size"),
            R_median=("R", "median"),
            R_std=("R", "std"),
            lambda_min=("lambda_nm", "min"),
            lambda_max=("lambda_nm", "max"),
        )
        .reset_index()
        .sort_values("R_median")
    )

    out_slit = args.outdir / "qc_step09_oh_resolution_per_slit.csv"
    per_slit.to_csv(out_slit, index=False)
    print("[OK] Wrote:", out_slit)


if __name__ == "__main__":
    main()