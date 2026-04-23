#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step11 QC — multi-panel overview of flux-calibrated spectra.

PURPOSE
-------
Create a compact PDF grid of flux-calibrated spectra for fast visual review of
the final Step11 product.

DEFAULT INPUTS
--------------
- final spectrum product:
    config.ST11_FLUXCAL / extract1d_fluxcal.fits

Optional summary CSV:
- config.ST11_FLUXCAL / step11_fluxcal_summary.csv

OUTPUT
------
- config.ST11_FLUXCAL / qc_step11 / qc_step11_fluxcal_grid.pdf
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

import config

FITS_IN = Path(config.ST11_FLUXCAL) / "extract1d_fluxcal.fits"
SUMCSV  = Path(config.ST11_FLUXCAL) / "step11_fluxcal_summary.csv"
OUTPDF  = Path(config.ST11_FLUXCAL) / "qc_step11" / "qc_step11_fluxcal_grid.pdf"

XLIM_NM = (600, 980)
NCOL = 4
NROW = 8
PER_PAGE = NCOL * NROW

def _get_col(tbl, *names):
    cols = tbl.columns.names
    for n in names:
        if n in cols:
            return n
    return None

def _read(lamhdu):
    d = lamhdu.data
    if d is None or not hasattr(d, "columns"):
        return None
    lamc = _get_col(d, "LAMBDA_NM")
    ycol = _get_col(d, "FLUX_FLAM", "FLAM", "FLUX_CAL_FLAM", "FLUX_TELLCOR_O2", "FLUX")
    if lamc is None or ycol is None:
        return None
    lam = np.asarray(d[lamc], float)
    y   = np.asarray(d[ycol], float)
    m = np.isfinite(lam) & np.isfinite(y)
    return lam[m], y[m], (lamc, ycol)

def main():
    OUTPDF.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMCSV) if SUMCSV.exists() else None

    with fits.open(FITS_IN) as hdul:
        hmap = {h.name.strip().upper(): h for h in hdul[1:]}
        slits = list(hmap.keys())

        if summary is not None and "slit" in summary.columns:
            summary["slit"] = summary["slit"].astype(str).str.strip().str.upper()
            good = np.ones(len(summary), dtype=bool)
            if "qcflag" in summary.columns:
                good = summary["qcflag"].astype(str).str.upper().isin(["GOOD", "OK"])
            if "S" in summary.columns:
                order = np.argsort(-np.abs(pd.to_numeric(summary["S"], errors="coerce").fillna(0).to_numpy(float)))
                summary = summary.iloc[order]
            slits = summary.loc[good, "slit"].tolist() or summary["slit"].tolist()

        with PdfPages(OUTPDF) as pdf:
            for i0 in range(0, len(slits), PER_PAGE):
                batch = slits[i0:i0+PER_PAGE]
                fig, axes = plt.subplots(NROW, NCOL, figsize=(11, 14), sharex=True)
                axes = axes.ravel()

                for ax, slit in zip(axes, batch):
                    hdu = hmap.get(slit)
                    if hdu is None:
                        ax.axis("off")
                        continue
                    parsed = _read(hdu)
                    if parsed is None:
                        ax.axis("off")
                        continue

                    lam, y, cols = parsed
                    ax.plot(lam, y, lw=0.8)
                    ax.set_xlim(*XLIM_NM)
                    ax.text(0.02, 0.92, slit, transform=ax.transAxes, fontsize=8, va="top")
                    ax.axvspan(686.0, 689.0, alpha=0.08)
                    ax.axvspan(760.0, 770.0, alpha=0.08)

                for ax in axes[len(batch):]:
                    ax.axis("off")

                fig.suptitle("Step11 QC — flux-calibrated spectra", y=0.995, fontsize=12)
                fig.text(0.5, 0.005, "Wavelength (nm)", ha="center")
                fig.tight_layout(rect=[0.03, 0.02, 0.98, 0.985])
                pdf.savefig(fig)
                plt.close(fig)

    print("[DONE] Wrote:", OUTPDF)

if __name__ == "__main__":
    main()
