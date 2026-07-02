#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAMOS Step12d QC: Stellar-Response Construction
===============================================

Purpose
-------
Generate diagnostic plots for Step12d, which derives an ensemble
stellar-response correction from flux-calibrated stellar spectra,
Step09 continuum estimates, and SkyMapper r/i/z photometry.

This QC verifies that:

1. Individual stellar response curves are smooth and broadly consistent.
2. The ensemble median response is well defined over the trusted wavelength
   range.
3. The fitted reddened blackbody continua pass through the SkyMapper
   photometric anchors.
4. The Step09 continuum estimates provide a sensible smooth representation
   of the observed spectra.

Inputs
------
step12d_stellar_response_per_slit.fits
    Per-slit Step09 continuum, fitted blackbody model, and response curve.

step12d_stellar_response_master.fits
    Ensemble median response and percentile envelope.

step12d_stellar_response_summary.csv
    Fit statistics and acceptance flags.

SkyMapper photometry catalog
    Used only for overplotting r/i/z photometric anchors.

Outputs
-------
qc_step12d_stellar_response.pdf

PDF Contents
------------
Page 1:
    Individual accepted response curves, ensemble median response, and
    16--84 percentile envelope.

Subsequent pages:
    Per-slit comparisons of:
        - Step11 FLUX_FLAM spectrum,
        - scaled Step09 continuum,
        - fitted reddened blackbody model,
        - SkyMapper r/i/z photometric fluxes.

Notes
-----
This QC is primarily diagnostic for the construction of the master response.
The final application to all spectra is validated separately by
qc_step12de_comprehensive.py.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import config

def abmag_to_flam_cgs(mag_ab, lam_nm):
    fnu_cgs = 3631.0 * 10 ** (-0.4 * mag_ab) * 1e-23
    c_A_s = 2.99792458e18
    lam_A = lam_nm * 10.0
    return float(fnu_cgs * c_A_s / lam_A**2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=str(config.ST12_FINALCAL / "step12d_stellar_response"))
    ap.add_argument("--outdir", default=str(config.QC_DIR / "12_finalcal"))
    ap.add_argument("--outpdf", default=str(config.NAME_QC_STEP12D_RESPONSE_PDF))
    args = ap.parse_args()

    indir = Path(args.indir)
    per = indir / "step12d_stellar_response_per_slit.fits"
    master = indir / "step12d_stellar_response_master.fits"
    summary_csv = indir / "step12d_stellar_response_summary.csv"
    #outpdf = Path(args.outpdf) if args.outpdf else indir / "qc_step12d_stellar_response.pdf"
    outpdf = Path(args.outdir) / args.outpdf
    
    # after loading df row / slit row
    phot = pd.read_csv(config.STEP12_PHOTCAT)
    phot["slit"] = phot["slit"].astype(str).str.upper().str.strip()

    df = pd.read_csv(summary_csv)

    with fits.open(per) as hp, fits.open(master) as hm, PdfPages(outpdf) as pdf:
        dm = hm["MASTER_RESPONSE"].data
        lm = np.asarray(dm["LAMBDA_NM"], float)
        rm = np.asarray(dm["RESP_MASTER"], float)
        r16 = np.asarray(dm["RESP_P16"], float)
        r84 = np.asarray(dm["RESP_P84"], float)

        # Page 1: response stack
        fig, ax = plt.subplots(figsize=(10, 6))

        for slit in df.loc[df["accepted"] == True, "slit"]:
            if slit not in hp:
                continue
            d = hp[slit].data
            lam = np.asarray(d["LAMBDA_NM"], float)
            r = np.asarray(d["RESP_BB"], float)
            m = np.isfinite(lam) & np.isfinite(r) & (r > 0)
            ax.plot(lam[m], r[m], color="0.7", lw=0.7, alpha=0.5)

        ax.plot(lm, rm, color="k", lw=2.0, label="median")
        ax.fill_between(lm, r16, r84, color="0.8", alpha=0.5, label="16–84%")
        ax.axhline(1, color="0.4", ls="--")
        ax.set_xlim(590, 960)
        ax.set_ylim(0, np.nanpercentile(r84[np.isfinite(r84)], 98) * 1.2)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("BB-derived response, normalized near i")
        ax.set_title("Step12d stellar-response stack")
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        # Pages: per slit fits
        accepted = df[df["accepted"] == True]["slit"].tolist()
        per_page = 6

        for i0 in range(0, len(accepted), per_page):
            batch = accepted[i0:i0+per_page]
            fig, axes = plt.subplots(3, 2, figsize=(11, 12), sharex=True)
            axes = axes.ravel()

            for ax, slit in zip(axes, batch):
                d = hp[slit].data
                lam = np.asarray(d["LAMBDA_NM"], float)
                flux = np.asarray(d["FLUX_FLAM"], float)
                cont = np.asarray(d["CONT09_SMOOTH"], float)
                bb = np.asarray(d["BB_MODEL"], float)
                resp = np.asarray(d["RESP_BB"], float)

                m = np.isfinite(lam) & np.isfinite(flux) & np.isfinite(cont) & np.isfinite(bb)
                m &= (lam >= 590) & (lam <= 960)

                ax.plot(lam[m], flux[m], color="0.65", lw=0.6, label="FLUX_FLAM")
                ax.plot(lam[m], cont[m], color="C0", lw=1.0, label="Step09 cont scaled")
                ax.plot(lam[m], bb[m], color="crimson", lw=1.0, label="BB+Av fit")
                ax.set_yscale("log")

                y = np.concatenate([flux[m], cont[m], bb[m]])
                y = y[np.isfinite(y) & (y > 0)]
                if y.size:
                    hi = np.nanpercentile(y, 98)
                    lo = max(hi / 300, np.nanpercentile(y, 5) / 2)
                    ax.set_ylim(lo, hi * 1.5)

                row = df[df["slit"] == slit].iloc[0]
                ax.set_title(f"{slit}  T={row.teff:.0f}K  Av={row.av:.2f}  rms={row.rmsdex:.3f}")
                ax.set_xlim(590, 960)
                
                rowp = phot.loc[phot["slit"] == slit]

                if len(rowp) == 1:
                    rowp = rowp.iloc[0]
                
                    xphot, yphot = [], []
                    for b, lam_eff in {"r": 620.0, "i": 760.0, "z": 900.0}.items():
                        mag_col = f"{b}_mag"
                        if mag_col in rowp.index and np.isfinite(rowp[mag_col]):
                            xphot.append(lam_eff)
                            yphot.append(abmag_to_flam_cgs(float(rowp[mag_col]), lam_eff))
                
                    if xphot:
                        ax.scatter(xphot, yphot, s=45, color="crimson",
                                   edgecolor="k", zorder=10)
                        for x, y, lab in zip(xphot, yphot, ["r", "i", "z"]):
                            ax.text(x, y, f" {lab}", fontsize=9,
                                    ha="left", va="bottom", color="black")
                            

            for ax in axes[len(batch):]:
                ax.set_axis_off()

            fig.suptitle("Step12d BB continuum fits", y=0.98)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

    print("[OK] Wrote", outpdf)


if __name__ == "__main__":
    main()