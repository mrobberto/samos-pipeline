#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
qc_step11d_refined_vs_skymapper.py

Multipage QC for Step11d refined spectra versus SkyMapper photometry.

PYTHONPATH=. python qc/step11/qc_step11d_refined_vs_skymapper.py \  
--full-fits "/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_Run8_Science_2026_01/SAMI/Dolidze25/reduced/11_fluxcal/Extract1D_fluxcal_refined_perstar_full.fits"    \
--edge-fits "/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_Run8_Science_2026_01/SAMI/Dolidze25/reduced/11_fluxcal/Extract1D_fluxcal_refined_perstar_edge_matched.fits"

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
    st11 = Path(config.ST11_FLUXCAL)
    qc_dir = st11 / "qc_step11"
    return {
        "full_fits": st11 / "Extract1D_fluxcal_refined_perstar_full.fits",
        "edge_fits": st11 / "Extract1D_fluxcal_refined_perstar_edge_matched.fits",
        "phot_csv": st11 / "slit_trace_radec_skymapper_all.csv",
        "outpdf": qc_dir / "qc_step11d_refined_vs_skymapper.pdf",
    }

def parse_args():
    d = default_paths()
    p = argparse.ArgumentParser(description="QC Step11d refined spectra versus SkyMapper")
    p.add_argument("--full-fits", type=Path, default=d["full_fits"])
    p.add_argument("--edge-fits", type=Path, default=d["edge_fits"])
    p.add_argument("--phot-csv", type=Path, default=d["phot_csv"])
    p.add_argument("--outpdf", type=Path, default=d["outpdf"])
    p.add_argument("--ncol", type=int, default=2)
    p.add_argument("--nrow", type=int, default=4)
    p.add_argument("--w-sub-cm", type=float, default=8.0)
    p.add_argument("--h-sub-cm", type=float, default=5.8)
    p.add_argument("--xmin", type=float, default=540.0)
    p.add_argument("--xmax", type=float, default=930.0)
    return p.parse_args()


def main():
    args = parse_args()

    full_file = Path(args.full_fits)
    edge_file = Path(args.edge_fits)
    phot_csv = Path(args.phot_csv)
    outpdf = Path(args.outpdf)

    if not full_file.exists():
        raise FileNotFoundError(full_file)
    if not edge_file.exists():
        raise FileNotFoundError(edge_file)
    if not phot_csv.exists():
        raise FileNotFoundError(phot_csv)

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

    with fits.open(full_file) as hfull, fits.open(edge_file) as hedge, PdfPages(outpdf) as pdf:
        for i0 in range(0, len(slits), per_page):
            batch = slits[i0:i0 + per_page]

            fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h), sharex=True)
            axes = np.atleast_1d(axes).ravel()

            for ax, slit in zip(axes, batch):
                if slit not in hfull or slit not in hedge:
                    ax.text(0.5, 0.5, f"{slit}\nmissing in one FITS", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue
                row = phot.loc[phot["slit"] == slit]
                if len(row) != 1:
                    ax.text(0.5, 0.5, f"{slit}\nno unique phot row", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue
                row = row.iloc[0]
                
                tab_full = hfull[slit].data
                tab_edge = hedge[slit].data
                
                cols_full = list(tab_full.names)
                cols_edge = list(tab_edge.names)
                
                need = {"LAMBDA_NM", "FLUX_FLAM", "FLUX_FLAM_REFINED"}
                if not need.issubset(cols_full) or not need.issubset(cols_edge):
                    ax.text(0.5, 0.5, f"{slit}\nmissing required columns", ha="center", va="center",
                            transform=ax.transAxes)
                    ax.set_axis_off()
                    continue
                
                lam0 = np.asarray(tab_edge["LAMBDA_NM"], float)
                flam0 = np.asarray(tab_edge["FLUX_FLAM"], float)
                flam_full = np.asarray(tab_full["FLUX_FLAM_REFINED"], float)
                flam_edge = np.asarray(tab_edge["FLUX_FLAM_REFINED"], float)
                resp = np.asarray(tab_edge["RESP_STEP11D"], float) if "RESP_STEP11D" in cols_edge else np.full_like(flam_edge, np.nan)
                
                m = np.isfinite(lam0) & np.isfinite(flam0) & np.isfinite(flam_full) & np.isfinite(flam_edge)
                lam0 = lam0[m]
                flam0 = flam0[m]
                flam_full = flam_full[m]
                flam_edge = flam_edge[m]
                resp = resp[m]


                if lam0.size == 0:
                    ax.text(0.5, 0.5, f"{slit}\nno finite data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue

                order = np.argsort(lam0)
                lam0 = lam0[order]
                flam0 = flam0[order]
                flam_full = flam_full[order]
                flam_edge = flam_edge[order]
                resp = resp[order]
                """
                ax.plot(lam0, flam0, lw=0.8, label="Step11c")
                #REMOVE TO AVOID ORANGE LINE
                #ax.plot(lam0, flam_full, lw=1.0, label="full r/i/z refined")
                ax.plot(lam0, flam_edge, lw=1.0, label="r_short/i/z_short refined")
                """
                ax.plot(lam0, flam0, color="0.5", lw=0.8, label="Step11c")   # gray
                ax.plot(lam0, flam_edge, color="C0", lw=1.2, label="Refined")  # blue

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
                    flam_full[np.isfinite(flam_full)],
                    flam_edge[np.isfinite(flam_edge)],
                ]
                if len(yphot):
                    vals.append(np.array(yphot, float))

                med_r = np.nanmedian(resp) if np.any(np.isfinite(resp)) else np.nan
                ax.set_title(f"{slit}   ⟨R⟩={med_r:.2f}", fontsize=10)
                ax.set_xlim(args.xmin, args.xmax)
                ymin, ymax = robust_ylim(np.concatenate(vals))
                ax.set_ylim(max(ymin, 0), ymax)
                ax.set_xlabel("Wavelength (nm)", fontsize=9)
                ax.set_ylabel(r"$f_\lambda$  (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)", fontsize=9)
                ax.tick_params(axis="x", which="both", labelbottom=True)

            for ax in axes[len(batch):]:
                ax.set_axis_off()

            fig.suptitle(
                f"Step11d refined spectra with SkyMapper photometry  ({i0+1}–{i0+len(batch)} / {len(slits)})",
                y=0.97,
                fontsize=14,
            )
            fig.text(0.5, 0.04, "Wavelength (nm)", ha="center")
            fig.text(0.02, 0.5, r"$f_\lambda$  [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]",
                     va="center", rotation="vertical")

            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper right", fontsize=8)
            
            fig.subplots_adjust(
                left=0.10,
                right=0.97,
                bottom=0.12,
                top=0.93,
                wspace=0.25,
                hspace=0.45,
            )
            """
            fig.tight_layout(rect=[0.03, 0.08, 0.97, 0.95])
            """
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[DONE] Wrote {outpdf}")


if __name__ == "__main__":
    main()
