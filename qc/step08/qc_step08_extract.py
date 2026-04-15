#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC for Step08 optimal extraction.

Standalone diagnostic script for reviewing TRACECOORDS slit images together
with Step08 extraction products.

Inputs
------
- Step06 TRACECOORDS MEF:   *_{EVEN,ODD}_tracecoords.fits
- Step08a extraction MEF:   extract1d_optimal_ridge_{even,odd}.fits

Outputs
-------
PNG quicklooks written to:
  config.ST08_EXTRACT1D / "qc_step08"

Products per slit:
- panel 1: TRACECOORDS image with ridge and object aperture overlay
- panel 2: per-row diagnostics (X0, NOBJ, NSKY, EDGEFLAG)
- panel 3: extracted FLUX, SKY, and aperture-corrected FLUX
- panel 4: summary text block with key metadata / flags
"""

from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import config


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="QC for SAMOS Step08 extraction")
parser.add_argument("--set", choices=["EVEN", "ODD"], required=True,
                    help="Slit set to inspect")
parser.add_argument("--slit", default=None,
                    help="Specific slit name, e.g. SLIT018. Default: all slits")
parser.add_argument("--max", type=int, default=None,
                    help="Maximum number of slits to plot")
args = parser.parse_args()
SET_TAG = args.set.upper()
SLIT_FILTER = args.slit.upper() if args.slit else None


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------
trace_pattern = str(Path(config.ST06_SCIENCE) / f"*_{SET_TAG}_tracecoords.fits")
trace_matches = sorted(glob(trace_pattern))
if not trace_matches:
    raise FileNotFoundError(f"No TRACECOORDS file found for {SET_TAG}: {trace_pattern}")
TRACE_FITS = Path(trace_matches[-1])

EXTRACT_FITS = Path(config.ST08_EXTRACT1D) / f"extract1d_optimal_ridge_{SET_TAG.lower()}.fits"
if not EXTRACT_FITS.exists():
    raise FileNotFoundError(f"Extraction product not found: {EXTRACT_FITS}")

OUT_DIR = Path(config.ST08_EXTRACT1D) / "qc_step08"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W_OBJ = 3.0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_slit_image_hdus(hdul: fits.HDUList) -> dict[str, fits.ImageHDU]:
    out = {}
    for hdu in hdul[1:]:
        name = (hdu.name or "").upper()
        if isinstance(hdu, fits.ImageHDU) and name.startswith("SLIT") and hdu.data is not None:
            if np.asarray(hdu.data).ndim == 2:
                out[name] = hdu
    return out


def get_slit_table_hdus(hdul: fits.HDUList) -> dict[str, fits.BinTableHDU]:
    out = {}
    for hdu in hdul[1:]:
        name = (hdu.name or "").upper()
        if isinstance(hdu, fits.BinTableHDU) and name.startswith("SLIT"):
            out[name] = hdu
    return out


def robust_limits(img: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(img, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(vals, 5))
    vmax = float(np.nanpercentile(vals, 99.5))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        med = float(np.nanmedian(vals))
        sig = float(np.nanstd(vals)) if vals.size > 1 else 1.0
        return med - sig, med + 3 * sig
    return vmin, vmax


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    with fits.open(TRACE_FITS, memmap=False) as htrace, fits.open(EXTRACT_FITS, memmap=False) as hext:
        img_map = get_slit_image_hdus(htrace)
        tab_map = get_slit_table_hdus(hext)

        slit_names = sorted(set(img_map).intersection(tab_map))
        if SLIT_FILTER is not None:
            slit_names = [s for s in slit_names if s == SLIT_FILTER]
        if args.max is not None:
            slit_names = slit_names[:args.max]

        if not slit_names:
            raise RuntimeError("No matching slits found between TRACECOORDS and extraction product.")

        for slit in slit_names:
            img_hdu = img_map[slit]
            tab_hdu = tab_map[slit]

            img = np.asarray(img_hdu.data, float)
            data = tab_hdu.data
            ypix = np.asarray(data["YPIX"], dtype=float)
            x0 = np.asarray(data["X0"], dtype=float)
            flux = np.asarray(data["FLUX"], dtype=float)
            flux_apcorr = np.asarray(data["FLUX_APCORR"], dtype=float)
            sky = np.asarray(data["SKY"], dtype=float)
            nobj = np.asarray(data["NOBJ"], dtype=float)
            nsky = np.asarray(data["NSKY"], dtype=float)
            edgeflag = np.asarray(data["EDGEFLAG"], dtype=float)
            trxleft = np.asarray(data["TRXLEFT"], dtype=float)
            trxright = np.asarray(data["TRXRIGHT"], dtype=float)

            fig = plt.figure(figsize=(12, 10))
            gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], width_ratios=[1.25, 1.0])

            # Panel 1: image + ridge
            ax1 = fig.add_subplot(gs[0, 0])
            vmin, vmax = robust_limits(img)
            ax1.imshow(img, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
            ok = np.isfinite(x0) & np.isfinite(ypix)
            ax1.plot(x0[ok], ypix[ok], linewidth=1.2)
            ax1.plot((x0 - W_OBJ)[ok], ypix[ok], linestyle="--", linewidth=0.9)
            ax1.plot((x0 + W_OBJ)[ok], ypix[ok], linestyle="--", linewidth=0.9)
            valid_left = np.isfinite(trxleft) & np.isfinite(ypix)
            valid_right = np.isfinite(trxright) & np.isfinite(ypix)
            ax1.plot(trxleft[valid_left], ypix[valid_left], linestyle=":", linewidth=0.8)
            ax1.plot(trxright[valid_right], ypix[valid_right], linestyle=":", linewidth=0.8)
            bad = edgeflag >= 2
            if np.any(bad):
                ax1.scatter(x0[bad], ypix[bad], s=8)
            ax1.set_title(f"{slit} TRACECOORDS + ridge")
            ax1.set_xlabel("X (tracecoords px)")
            ax1.set_ylabel("YPIX")

            # Panel 2: row diagnostics
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(x0, ypix, label="X0")
            ax2.set_xlabel("X0 / counts")
            ax2.set_ylabel("YPIX")
            ax2.set_title("Per-row geometry")
            ax2b = ax2.twiny()
            ax2b.plot(nobj, ypix, linestyle="--", label="NOBJ")
            ax2b.plot(nsky, ypix, linestyle=":", label="NSKY")
            if np.any(edgeflag > 0):
                ef = np.where(edgeflag > 0, edgeflag, np.nan)
                ax2b.plot(ef, ypix, linewidth=0.8, label="EDGEFLAG")
            lines = ax2.get_lines() + ax2b.get_lines()
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc="best", fontsize=8)

            # Panel 3: spectra-like diagnostics
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(ypix, flux, label="FLUX")
            ax3.plot(ypix, flux_apcorr, label="FLUX_APCORR")
            ax3.plot(ypix, sky, label="SKY")
            ax3.set_xlabel("YPIX")
            ax3.set_ylabel("Value")
            ax3.set_title("Extraction diagnostics")
            ax3.legend(loc="best", fontsize=8)

            # Panel 4: summary text
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.axis("off")
            good_rows = int(np.sum(np.isfinite(flux)))
            apcorr_rows = int(np.sum(np.isfinite(flux_apcorr) & np.isfinite(flux)))
            nedge1 = int(np.sum(edgeflag == 1))
            nedge2 = int(np.sum(edgeflag == 2))
            med_nobj = float(np.nanmedian(nobj)) if np.isfinite(nobj).any() else np.nan
            med_nsky = float(np.nanmedian(nsky)) if np.isfinite(nsky).any() else np.nan
            summary = [
                f"Slit: {slit}",
                f"RIDGEMOD: {tab_hdu.header.get('RIDGEMOD', 'NA')}",
                f"TRACESET: {tab_hdu.header.get('TRACESET', 'NA')}",
                f"RA: {tab_hdu.header.get('RA', 'NA')}",
                f"DEC: {tab_hdu.header.get('DEC', 'NA')}",
                "",
                f"Good rows: {good_rows}",
                f"APCORR rows: {apcorr_rows}",
                f"Median NOBJ: {med_nobj:.2f}" if np.isfinite(med_nobj) else "Median NOBJ: NA",
                f"Median NSKY: {med_nsky:.2f}" if np.isfinite(med_nsky) else "Median NSKY: NA",
                f"EDGEFLAG==1: {nedge1}",
                f"EDGEFLAG==2: {nedge2}",
                "",
                f"S08RSKY: {tab_hdu.header.get('S08RSKY', 'NA')}",
                f"S08RFLX: {tab_hdu.header.get('S08RFLX', 'NA')}",
                f"S08BAD: {tab_hdu.header.get('S08BAD', 'NA')}",
                f"S08EMP: {tab_hdu.header.get('S08EMP', 'NA')}",
            ]
            ax4.text(0.02, 0.98, "\n".join(summary), va="top", ha="left", family="monospace")

            fig.suptitle(f"Step08 QC — {slit} ({SET_TAG})")
            fig.tight_layout()
            out_png = OUT_DIR / f"qc_step08_{SET_TAG.lower()}_{slit.lower()}.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] Wrote {out_png}")


if __name__ == "__main__":
    main()
