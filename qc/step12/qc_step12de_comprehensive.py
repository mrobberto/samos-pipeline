#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAMOS Step12d/e QC: Comprehensive Final-Calibration Validation
=============================================================

Purpose
-------
Generate a comprehensive validation PDF for the Step12d/e stellar-response
final calibration.

This QC compares the original Step11 spectra, the Step12d stellar-response
model products, and the final Step12e corrected spectra slit by slit.

It is intended as the main closeout diagnostic for the final calibration
stage.

Calibration Context
-------------------
Step12d derives a smooth ensemble response curve from stellar continua.
Step12e applies that master response to all spectra and optionally applies
a scalar photometric normalization when SkyMapper photometry is available.

This QC verifies both parts:

1. The common response correction behaves smoothly and consistently.
2. Corrected spectra align plausibly with the external photometric anchors.
3. Spectra lacking physical FLUX_FLAM calibration are still displayed in
   relative ADU/s/Angstrom units when possible.

Inputs
------
step12d_stellar_response_per_slit.fits
    Per-slit blackbody fits, continua, and response curves.

step12d_stellar_response_master.fits
    Master stellar-response correction.

step12d_stellar_response_summary.csv
    Per-slit fit diagnostics.

extract1d_finalcal_stellarresp.fits
    Final Step12e product containing:
        RESP_STELLAR_MASTER
        FLUX_FLAM_STELLARRESP
        VAR_FLAM2_STELLARRESP
        NORM_STELLARRESP
        HAS_PHOTNORM
        NORM_BAND

SkyMapper photometry catalog
    Used for overplotting r/i/z photometric anchors.

Outputs
-------
qc_step12de_comprehensive.pdf

PDF Contents
------------
Page 1:
    Master stellar-response correction, individual contributing response
    curves, and 16--84 percentile envelope.

Subsequent pages:
    One panel per slit showing, when available:
        - original FLUX_FLAM spectrum,
        - Step09 continuum,
        - fitted reddened blackbody target,
        - Step12e corrected spectrum,
        - SkyMapper r/i/z photometry,
        - applied response curve on the secondary y-axis.

Plotting Conventions
--------------------
Physical flux-calibrated spectra are plotted in FLUX_FLAM units.

Slits lacking finite FLUX_FLAM but retaining telluric-corrected spectra are
shown in relative ADU/s/Angstrom units using FLUX_TELLCOR_O2.

Corrected spectra are color-coded by calibration status:
    navy:
        master response plus photometric normalization
    crimson:
        shape-only or relative correction

Notes
-----
This QC is intentionally inclusive: all slit extensions in the final Step12e
product are shown, including those without Step12d stellar fits. This makes
the PDF useful both for scientific validation and for identifying missing or
non-physical flux-calibration cases.
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
    ap.add_argument("--final", default=str(config.ST12_FINALCAL / config.EXTRACT1D_FINALCAL_STELLARRESP))
    ap.add_argument("--outpdf", default=None)
    args = ap.parse_args()

    indir = Path(args.indir)
    final_fits = Path(args.final)

    per = indir / "step12d_stellar_response_per_slit.fits"
    master = indir / "step12d_stellar_response_master.fits"
    summary_csv = indir / "step12d_stellar_response_summary.csv"

    outpdf = Path(args.outpdf) if args.outpdf else config.QC_DIR / "12_finalcal" / "qc_step12de_comprehensive.pdf"

    phot = pd.read_csv(config.STEP12_PHOTCAT)
    phot["slit"] = phot["slit"].astype(str).str.upper().str.strip()

    df = pd.read_csv(summary_csv)
    with fits.open(final_fits) as htmp:
        accepted = [
            h.name for h in htmp[1:]
            if h.name.startswith("SLIT")
        ]

    with fits.open(per) as hp, fits.open(master) as hm, fits.open(final_fits) as hf, PdfPages(outpdf) as pdf:

        dm = hm["MASTER_RESPONSE"].data
        lm = np.asarray(dm["LAMBDA_NM"], float)
        rm = np.asarray(dm["RESP_MASTER"], float)
        r16 = np.asarray(dm["RESP_P16"], float)
        r84 = np.asarray(dm["RESP_P84"], float)

        # ------------------------------------------------------------
        # Page 1: master response
        # ------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))

        for slit in accepted:
            if slit not in hp:
                continue
            d = hp[slit].data
            lam = np.asarray(d["LAMBDA_NM"], float)
            r = np.asarray(d["RESP_BB"], float)
            m = np.isfinite(lam) & np.isfinite(r) & (r > 0)
            ax.plot(lam[m], r[m], color="0.75", lw=0.6, alpha=0.5)

        ax.plot(lm, rm, color="k", lw=2.2, label="median response")
        ax.fill_between(lm, r16, r84, color="0.85", alpha=0.7, label="16–84%")
        ax.axhline(1, color="0.4", ls="--")
        ax.set_xlim(590, 960)
        ax.set_ylim(0, np.nanpercentile(r84[np.isfinite(r84)], 98) * 1.25)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Response")
        ax.set_title("Step12d/e master stellar-response correction")
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        # ------------------------------------------------------------
        # Per-slit comprehensive pages
        # ------------------------------------------------------------
        per_page = 4

        for i0 in range(0, len(accepted), per_page):
            batch = accepted[i0:i0+per_page]

            fig, axes = plt.subplots(
                per_page, 1,
                figsize=(11, 3.0 * per_page),
                sharex=True
            )

            axes = np.atleast_1d(axes)

            for ax, slit in zip(axes, batch):

                if slit not in hf:
                    ax.text(0.5, 0.5, f"{slit}\nmissing", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue
                dfinal = hf[slit].data
                
                # plot in ADU/s/Å spectra, if FLUX_FLAM is missing
                has_flam = np.any(
                    np.isfinite(np.asarray(dfinal["FLUX_FLAM"], float)) &
                    (np.asarray(dfinal["LAMBDA_NM"], float) >= 590) &
                    (np.asarray(dfinal["LAMBDA_NM"], float) <= 960)
                )
                
                if has_flam:
                    flux = np.asarray(dfinal["FLUX_FLAM"], float)
                    flux_label = "FLUX_FLAM"
                    unit_label = r"$f_\lambda$"
                else:
                    flux = np.asarray(dfinal["FLUX_TELLCOR_O2"], float)
                    flux_label = "FLUX_TELLCOR_O2 relative"
                    unit_label = "ADU s$^{-1}$ Å$^{-1}$"
                    

                has_step12d = slit in hp
                
                if has_step12d and has_flam:
                    d = hp[slit].data
                
                    lam = np.asarray(d["LAMBDA_NM"], float)
                    flux = np.asarray(d["FLUX_FLAM"], float)
                    cont = np.asarray(d["CONT09_SMOOTH"], float)
                    bb = np.asarray(d["BB_MODEL"], float)
                else:
                    d = dfinal
                
                    lam = np.asarray(d["LAMBDA_NM"], float)
                
                    if has_flam:
                        flux = np.asarray(d["FLUX_FLAM"], float)
                    else:
                        flux = np.asarray(d["FLUX_TELLCOR_O2"], float)
                
                    cont = np.full_like(flux, np.nan)
                    bb = np.full_like(flux, np.nan)
                    

                lamf = np.asarray(dfinal["LAMBDA_NM"], float)
                if has_flam:
                    fcorr = np.asarray(dfinal["FLUX_FLAM_STELLARRESP"], float)
                else:
                    resp_tmp = np.asarray(dfinal["RESP_STELLAR_MASTER"], float)
                    fcorr = flux * resp_tmp
                    
                resp_master = np.asarray(dfinal["RESP_STELLAR_MASTER"], float)

                m_flux = (
                    np.isfinite(lam) &
                    np.isfinite(flux) &
                    (flux > 0) &
                    (lam >= 590) &
                    (lam <= 960)
                )
                
                m_model = (
                    m_flux &
                    np.isfinite(cont) &
                    np.isfinite(bb) &
                    (cont > 0) &
                    (bb > 0)
                )

                mf = (
                    np.isfinite(lamf) &
                    np.isfinite(fcorr) &
                    (fcorr > 0) &
                    (lamf >= 590) &
                    (lamf <= 960)
                )

                ax.plot(lam[m_flux], flux[m_flux], color="0.65", lw=0.7, label=flux_label)

                if np.any(m_model):
                    ax.plot(lam[m_model], cont[m_model], color="C0", lw=1.0, alpha=0.7, label="Step09 continuum")
                    ax.plot(lam[m_model], bb[m_model], color="crimson", lw=1.0, alpha=0.8, label="BB+Av target")
    
                has_photnorm = False
                if "HAS_PHOTNORM" in dfinal.names:
                    has_photnorm = bool(np.nanmedian(np.asarray(dfinal["HAS_PHOTNORM"], float)))
                
                corr_color = "navy" if has_photnorm else "crimson"
                corr_label = "shape+phot corrected" if has_photnorm else "shape-only corrected"
                
                ax.plot(
                    lamf[mf],
                    fcorr[mf],
                    color=corr_color,
                    lw=1.2,
                    label=corr_label,
                )


                # SkyMapper points
                rowp = phot.loc[phot["slit"] == slit]
                xphot, yphot = [], []

                if len(rowp) == 1:
                    rowp = rowp.iloc[0]
                    for b, leff in {"r": 620.0, "i": 760.0, "z": 900.0}.items():
                        mag_col = f"{b}_mag"
                        if mag_col in rowp.index and np.isfinite(rowp[mag_col]):
                            xphot.append(leff)
                            yphot.append(abmag_to_flam_cgs(float(rowp[mag_col]), leff))

                if xphot:
                    ax.scatter(xphot, yphot, s=55, color="crimson", edgecolor="k", zorder=10)
                    for x, y, lab in zip(xphot, yphot, ["r", "i", "z"]):
                        ax.text(x + 4, y, lab, fontsize=10, va="center")

                ax.set_yscale("log")
                ax.set_xlim(590, 960)

                # y-scale anchored to minima/maxima of original+corrected
                yscale = []
                if np.any(m_flux):
                    yscale.append(flux[m_flux])
                if np.any(m_model):
                    yscale.append(cont[m_model])
                    yscale.append(bb[m_model])    
                if np.any(mf):
                    yscale.append(fcorr[mf])
                if yphot:
                    yscale.append(np.asarray(yphot, float))


                if len(yscale) > 0:
                    yscale = np.concatenate(yscale)
                    yscale = yscale[np.isfinite(yscale) & (yscale > 0)]
                
                    if yscale.size:
                        ylo = np.nanpercentile(yscale, 2)
                        yhi = np.nanpercentile(yscale, 98)
                        ax.set_ylim(ylo / 1.5, yhi * 1.7)
                    else:
                        ax.text(
                            0.5, 0.5,
                            f"{slit}\nno positive plotted flux",
                            ha="center", va="center",
                            transform=ax.transAxes
                        )
                else:
                    ax.text(
                        0.5, 0.5,
                        f"{slit}\nno plotted arrays",
                        ha="center", va="center",
                        transform=ax.transAxes
                        )

                row_match = df[df["slit"] == slit]

                if len(row_match) == 1:
                    row = row_match.iloc[0]
                    title_left = (
                        f"{slit}  T={row.teff:.0f}K  Av={row.av:.2f}  "
                        f"resp_med={row.resp_med:.2f}"
                    )
                else:
                    title_left = f"{slit}  no Step12d stellar fit"
                    
                
                norm_txt = ""
                if "NORM_STELLARRESP" in dfinal.names:
                    norm_val = np.nanmedian(np.asarray(dfinal["NORM_STELLARRESP"], float))
                    norm_txt = f"  norm={norm_val:.2f}"

                ax.set_title(
                    f"{title_left}{norm_txt}",
                    fontsize=10
                )

                if not has_flam:
                    corr_color = "crimson"
                    corr_label = "relative shape-only corrected"
                
                ax.set_ylabel(unit_label)

                # inset response
                ax2 = ax.twinx()
                mr = (
                    np.isfinite(lamf) &
                    np.isfinite(resp_master) &
                    (lamf >= 590) &
                    (lamf <= 960)
                )
                ax2.plot(lamf[mr], resp_master[mr], color="k", lw=0.7, alpha=0.35)
                ax2.set_ylim(0, max(3, np.nanpercentile(resp_master[mr], 98) * 1.1))
                ax2.set_ylabel("R", fontsize=8)
                ax2.tick_params(axis="y", labelsize=7)

            for ax in axes[len(batch):]:
                ax.set_axis_off()

            axes[-1].set_xlabel("Wavelength [nm]")

            handles, labels = axes[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            fig.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

            fig.suptitle("Step12d/e comprehensive stellar-response validation", y=0.995)
            fig.tight_layout(rect=[0, 0, 0.88, 0.98])

            pdf.savefig(fig)
            plt.close(fig)

    print("[OK] Wrote", outpdf)


if __name__ == "__main__":
    main()
