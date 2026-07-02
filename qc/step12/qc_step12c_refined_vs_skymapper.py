#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
QC Step 12c — Refined spectra versus SkyMapper photometry.

This script generates multi-page diagnostic plots comparing the spectra
after illumination correction (Step 12b) and after photometric refinement
(Step 12c) against external SkyMapper photometry.

Purpose
-------
The goal of this QC is to verify that the final spectrophotometric
calibration:

  (1) preserves the spectral shape established by the illumination correction,
  (2) is consistent with external broadband photometry,
  (3) does not introduce spurious distortions.

Method
------
For each slit:

1. Input spectra:
   - Step12b: FLUX_FLAM_ILLUMCORR (illumination-corrected spectrum)
   - Step12c: FLUX_FLAM_REFINED (photometrically refined spectrum)

2. The spectra are plotted together:
   - gray: illumination-corrected spectrum (instrument-corrected)
   - blue: refined spectrum (final product)

3. External photometry:
   SkyMapper r, i, z magnitudes are converted to flux density and plotted
   at their effective wavelengths:
       r ≈ 620 nm, i ≈ 750 nm, z ≈ 870 nm

4. Comparison:
   The refined spectrum is expected to pass through (or closely match)
   the photometric points, while maintaining a smooth correction relative
   to the illumination-corrected spectrum.

5. Response diagnostic:
   The mean value of the refinement function R(λ) is displayed for each slit:
       ⟨R⟩ = median(RESP_STEP12c)

Plot characteristics
-------------------
- Wavelength range is restricted to the calibrated domain (typically 540–1050 nm)
- Logarithmic y-scale is used to accommodate dynamic range
- The y-range is limited to ~3 dex to suppress edge artifacts
- Only positive, finite values are plotted

Outputs
-------
A multi-page PDF:

- config.QC12_DIR / qc_step12c_refined_vs_skymapper.pdf

Each page contains a grid of slit spectra with:
  - Step12b vs Step12c comparison
  - SkyMapper photometric anchors
  - refinement strength indicator ⟨R⟩

Interpretation
--------------
A successful calibration shows:

- Step12c spectra closely matching the photometric points
- smooth deviations between Step12b and Step12c (no oscillations)
- no systematic offsets across r/i/z bands
- stable behavior across slits

Failure modes identifiable in this QC include:

- mismatched normalization (spectra offset from photometry)
- incorrect wavelength orientation (spectra mirrored vs photometry)
- overfitting or oscillatory corrections
- edge instabilities (excessive amplification at boundaries)

Dependencies
------------
- Step12b output: config.EXTRACT1D_ILLUMCORR
- Step12c output: refined spectra FITS (edge-matched version)
- SkyMapper photometry catalog (CSV)

Notes
-----
This QC complements the Step11d diagnostics by incorporating the
illumination correction (Step12b) and provides the final validation
of the spectrophotometric pipeline.


Run
---
samos) itsd-osx83:samos-pipeline robberto$ PYTHONPATH=. python qc/step12/qc_step12c_refined_vs_skymapper.py \
>   --input-fits "/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_Run8_Science_2026_01/SAMI/Dolidze25/reduced/12_finalcal/extract1d_optimal_ridge_all_wav_ohclean_tellcorr_illumcorr.fits" \
>   --edge-fits config.EXTRACT1D_FINALCAL \
>   --phot-csv "/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_Run8_Science_2026_01/SAMI/Dolidze25/reduced/11_fluxcal/slit_trace_radec_skymapper_all.csv"
[DONE] Wrote /Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_Run8_Science_2026_01/SAMI/Dolidze25/reduced/qc/12_finalcal/qc_step12c_refined_vs_skymapper.pdf

also should work
PYTHONPATH=. python qc/step12/qc_step12c_refined_vs_skymapper.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits

import config


LAM_EFF = {"r": 620.0, "i": 750.0, "z": 870.0}


def abmag_to_flam_cgs(mag_ab: float, lam_nm: float) -> float:
    fnu_cgs = 3631.0 * 10 ** (-0.4 * mag_ab) * 1e-23
    c_A_s = 2.99792458e18
    lam_A = lam_nm * 10.0
    return float(fnu_cgs * c_A_s / (lam_A ** 2))


def robust_ylim(y, qlo: float = 2, qhi: float = 98, pad: float = 0.10):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (-1, 1)
    lo, hi = np.nanpercentile(y, [qlo, qhi])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        med = np.nanmedian(y)
        return (med - 1, med + 1)
    d = hi - lo
    return lo - pad * d, hi + pad * d


def default_paths():
    st12 = Path(config.ST12_FINALCAL)
    qc_dir = Path(getattr(config, "QC12_DIR", Path(config.ST12_FINALCAL) / "qc_step12"))

    return {
        "input_fits": getattr(
            config,
            "EXTRACT1D_STEP12C_INPUT",
            config.ST11_FLUXCAL / "Extract1D_fluxcal.fits",
        ),
        "edge_fits": config.EXTRACT1D_FINALCAL,
        "phot_csv": config.STEP12_PHOTCAT,
        "outpdf": qc_dir / "qc_step12c_refined_vs_skymapper.pdf",
    }

def parse_args():
    d = default_paths()
    p = argparse.ArgumentParser(description="QC Step11d refined spectra versus SkyMapper")
    p.add_argument("--input-fits", type=Path, default=d["input_fits"])
    p.add_argument("--edge-fits", type=Path, default=d["edge_fits"])
    p.add_argument("--phot-csv", type=Path, default=d["phot_csv"])
    p.add_argument("--outpdf", type=Path, default=d["outpdf"])
    p.add_argument("--ncol", type=int, default=2)
    p.add_argument("--nrow", type=int, default=4)
    p.add_argument("--w-sub-cm", type=float, default=8.0)
    p.add_argument("--h-sub-cm", type=float, default=5.8)
    p.add_argument("--xmin", type=float, default=590.0)
    p.add_argument("--xmax", type=float, default=960.0)
    return p.parse_args()


def main():
    args = parse_args()

    input_file = Path(args.input_fits)
    edge_file = Path(args.edge_fits)
    phot_csv = Path(args.phot_csv)
    outpdf = Path(args.outpdf)
    
    if not input_file.exists():
        raise FileNotFoundError(input_file)
    if not edge_file.exists():
        raise FileNotFoundError(edge_file)
        
    outpdf.parent.mkdir(parents=True, exist_ok=True)

    phot = pd.read_csv(phot_csv)
    if "slit" not in phot.columns:
        raise KeyError(f"Expected column 'slit' in {phot_csv}; found {list(phot.columns)}")
    phot["slit"] = phot["slit"].astype(str).str.strip().str.upper()
    slits = phot["slit"].dropna().astype(str).str.strip().str.upper().tolist()

    ncol = args.ncol
    nrow = args.nrow
    per_page = ncol * nrow

    cm = 1 / 2.54
    fig_w = ncol * args.w_sub_cm * cm
    fig_h = nrow * args.h_sub_cm * cm

    with fits.open(input_file) as hin, fits.open(edge_file) as hedge, PdfPages(outpdf) as pdf:
        for i0 in range(0, len(slits), per_page):
            batch = slits[i0:i0 + per_page]

            fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h), sharex=True)
            axes = np.atleast_1d(axes).ravel()

            for ax, slit in zip(axes, batch):
                if slit not in hin or slit not in hedge:
                    ax.text(0.5, 0.5, f"{slit}\nmissing in one FITS", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue
                row = phot.loc[phot["slit"] == slit]
                if len(row) != 1:
                    ax.text(0.5, 0.5, f"{slit}\nno unique phot row", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue
                row = row.iloc[0]
                
                tab_in = hin[slit].data
                tab_edge = hedge[slit].data
                
                cols_in = list(tab_in.names)
                cols_edge = list(tab_edge.names)
                
                need_edge = {"LAMBDA_NM", "FLUX_FLAM_REFINED"}
                
                input_flux_col = None
                for cand in ["FLUX_FLAM", "FLUX_FLAM_ILLUMCORR", "FLUX_TELLCOR_O2"]:
                    if cand in cols_in:
                        input_flux_col = cand
                        break
                
                if input_flux_col is None or "LAMBDA_NM" not in cols_in:
                    ax.text(
                        0.5, 0.5,
                        f"{slit}\nmissing input flux column",
                        ha="center", va="center", transform=ax.transAxes
                    )
                    ax.set_axis_off()
                    continue
                                
                if not need_edge.issubset(cols_edge):
                    ax.text(
                        0.5, 0.5,
                        f"{slit}\nmissing refined columns:\n{sorted(need_edge - set(cols_edge))}",
                        ha="center", va="center", transform=ax.transAxes
                    )
                    ax.set_axis_off()
                    continue
                
                lam0 = np.asarray(tab_in["LAMBDA_NM"], float)
                flam0 = np.asarray(tab_in[input_flux_col], float)
                flam_edge = np.asarray(tab_edge["FLUX_FLAM_REFINED"], float)
                resp = np.asarray(tab_edge["RESP_STEP12c"], float) if "RESP_STEP12c" in cols_edge else np.full_like(flam_edge, np.nan)
                
                m = np.isfinite(lam0)
                
                lam0 = lam0[m]
                flam0 = flam0[m]
                flam_edge = flam_edge[m]
                resp = resp[m]


                if lam0.size == 0:
                    ax.text(0.5, 0.5, f"{slit}\nno finite data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue

                order = np.argsort(lam0)
                lam0 = lam0[order]
                flam0 = flam0[order]
                flam_edge = flam_edge[order]
                resp = resp[order]
                """
                ax.plot(lam0, flam0, lw=0.8, label="Step11c")
                #REMOVE TO AVOID ORANGE LINE
                #ax.plot(lam0, flam_full, lw=1.0, label="full r/i/z refined")
                ax.plot(lam0, flam_edge, lw=1.0, label="r_short/i/z_short refined")
                
                ax.plot(lam0, flam0, color="0.5", lw=0.8, label="Step12b illum-corrected")
                ax.plot(lam0, flam_edge, color="C0", lw=1.2, label="Step12c refined")
                """
                good0 = np.isfinite(lam0) & np.isfinite(flam0) & (flam0 > 0)
                good1 = np.isfinite(lam0) & np.isfinite(flam_edge) & (flam_edge > 0)
                
                ax.plot(lam0[good0], flam0[good0], color="0.5", lw=0.8, label=input_flux_col)
                ax.plot(lam0[good1], flam_edge[good1], color="C0", lw=1.2, label="Step12c refined")
                ax.set_yscale("log")
                
                xphot, yphot = [], []
                for b in ["r", "i", "z"]:
                    mag_col = f"{b}_mag"
                    if mag_col in row.index and np.isfinite(row[mag_col]):
                        lam_eff = LAM_EFF[b]
                        flam_eff = abmag_to_flam_cgs(float(row[mag_col]), lam_eff)
                        xphot.append(lam_eff)
                        yphot.append(flam_eff)
                        ax.text(lam_eff, flam_eff, f" {b}", fontsize=12,
                                ha="center", va="bottom", color="black")

                if xphot:
                    ax.scatter(xphot, yphot, s=40, color="crimson", edgecolor="k",
                                zorder=5, label="SkyMapper")

                vals = [
                    flam0[np.isfinite(flam0)],
                    flam_edge[np.isfinite(flam_edge)],
                ]
                if len(yphot):
                    vals.append(np.array(yphot, float))

                med_r = np.nanmedian(resp) if np.any(np.isfinite(resp)) else np.nan
                ax.set_title(f"{slit}   ⟨R⟩={med_r:.2f}", fontsize=10)
                ax.set_xlim(args.xmin, args.xmax)
                ymin, ymax = robust_ylim(np.concatenate(vals))
                ax.set_xlim(args.xmin, args.xmax)
                ax.set_ylim(max(ymin, 0), ymax)
                ax.set_xlabel("Wavelength (nm)", fontsize=9)
                ax.set_ylabel(r"$f_\lambda$  (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)", fontsize=9)
                ax.tick_params(axis="x", which="both", labelbottom=True)
                
                SHOW_LEGEND = False
                if SHOW_LEGEND:
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys(), fontsize=8)
                

                ylim_vals = np.concatenate([
                    flam0[good0],
                    flam_edge[good1],
                ])
                
                ylim_vals = ylim_vals[np.isfinite(ylim_vals) & (ylim_vals > 0)]
                
                if ylim_vals.size:
                
                    # robust central dynamic range
                    ylo = np.nanpercentile(ylim_vals, 5)
                    yhi = np.nanpercentile(ylim_vals, 99)
                
                    # avoid pathological compression
                    ylo = max(ylo * 0.7, 1e-30)
                    yhi = yhi * 1.5
                
                    # keep reasonable log span
                    if yhi / ylo > 300:
                        ylo = yhi / 300
                
                    ax.set_ylim(ylo, yhi)
    


            for ax in axes[len(batch):]:
                ax.set_axis_off()

            fig.suptitle(
                f"Step12c refined spectra with SkyMapper photometry  ({i0+1}–{i0+len(batch)} / {len(slits)})",
                y=0.97,
                fontsize=14,
            )
            """
            fig.text(0.5, 0.04, "Wavelength (nm)", ha="center")
            fig.text(0.02, 0.5, r"$f_\lambda$  [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]",
                     va="center", rotation="vertical")
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.dd(handles, labels, loc="upper right", fontsize=8)
            """
            fig.subplots_adjust(
                left=0.14,
                right=0.97,
                bottom=0.12,
                top=0.93,
                wspace=0.25,
                hspace=0.45,
            )
            """
            fig.tight_layout(rect=[0.03, 0.08, 0.97, 0.95])
            """
            plt.subplots_adjust(wspace=0.35)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[DONE] Wrote {outpdf}")


if __name__ == "__main__":
    main()
