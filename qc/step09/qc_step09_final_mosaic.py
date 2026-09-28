#!/usr/bin/env python3
"""
qc_step09_final_mosaic.py

Create a mosaic of 1D spectra from a multi-extension FITS file.

Typical use cases:
  - Step09 product before telluric correction
  - Step10 validated telluric-corrected product

Example
-------
PYTHONPATH=. python qc/step09/qc_step09_final_mosaic.py \
  --in products/Run8_Dolidze25/reduced/08_extract1d/extract1d_optimal_ridge_all_varfix_wavfix_OHref_skyclean099.fits \
  --outdir products/Run8_Dolidze25/reduced/09_abab/qc_step09 \
  --column STELLAR_CONSENSUS \
  --xlo 590 \
  --xhi 960 \
  --ncol 6 \
  --yscale global \
  --ymode log \
  --ylo 0.001 \
  --title "Spectra after background subtraction"

PYTHONPATH=. python qc/step09/qc_step09_final_mosaic.py \
  --in products/Run8_Dolidze25/reduced/10_telluric/extract1d_skyclean099_tellcorr_validated.fits \
  --outdir products/Run8_Dolidze25/reduced/10_telluric/qc_step10 \
  --column FLUX_TELLCOR_O2 \
  --xlo 590 \
  --xhi 960 \
  --ncol 6 \
  --yscale global \
  --ymode log \
  --ylo 0.001 \
  --title "Spectra after validated telluric correction"
"""

import os
import math
import argparse

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


# ----------------------------
# Helpers
# ----------------------------

WAVE_CANDIDATES = [
    "LAMBDA_FINAL",
    "LAMBDA_NM",
    "WAVELENGTH",
    "LAMBDA",
    "WAVE",
    "WAV_NM",
]

# O2 B and A bands: central markers and shaded intervals
O2_B_CENTER = 687.0
O2_A_CENTER = 760.0
O2_B_SPAN = (684.5, 689.5)
O2_A_SPAN = (758.0, 763.5)


def find_wave_column(colnames):
    """Return the first matching wavelength column name."""
    upper_map = {c.upper(): c for c in colnames}
    for cand in WAVE_CANDIDATES:
        if cand in upper_map:
            return upper_map[cand]
    return None


def finite_positive(arr):
    arr = np.asarray(arr)
    return arr[np.isfinite(arr) & (arr > 0)]


def compute_global_ylim(spectra, xlo, xhi, ymode="log", user_ylo=None, user_yhi=None):
    vals = []
    for wave, flux in spectra:
        m = np.isfinite(wave) & np.isfinite(flux)
        if xlo is not None:
            m &= (wave >= xlo)
        if xhi is not None:
            m &= (wave <= xhi)

        if not np.any(m):
            continue

        yy = flux[m]
        if ymode == "log":
            yy = yy[np.isfinite(yy) & (yy > 0)]
        else:
            yy = yy[np.isfinite(yy)]

        if yy.size:
            vals.append(yy)

    if not vals:
        if ymode == "log":
            return (user_ylo if user_ylo is not None else 1e-3,
                    user_yhi if user_yhi is not None else 1.0)
        else:
            return (user_ylo if user_ylo is not None else -1.0,
                    user_yhi if user_yhi is not None else 1.0)

    vals = np.concatenate(vals)

    if ymode == "log":
        ylo = np.nanmin(vals) if user_ylo is None else user_ylo
        yhi = np.nanmax(vals) if user_yhi is None else user_yhi
        # give a small margin upward
        yhi *= 1.15
    else:
        ylo = np.nanpercentile(vals, 1) if user_ylo is None else user_ylo
        yhi = np.nanpercentile(vals, 99) if user_yhi is None else user_yhi
        pad = 0.05 * (yhi - ylo if yhi > ylo else 1.0)
        ylo -= pad
        yhi += pad

    return ylo, yhi


def load_spectra(fits_path, flux_column):
    """
    Read spectra from a multi-extension FITS file.

    Returns
    -------
    items : list of dict
        Each item contains:
          name, wave, flux, header
    """
    items = []

    with fits.open(fits_path) as hdul:
        for hdu in hdul[1:]:
            if not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                continue
            if hdu.data is None:
                continue

            colnames = list(hdu.columns.names)
            if flux_column not in colnames:
                continue

            wave_col = find_wave_column(colnames)
            if wave_col is None:
                raise RuntimeError(
                    f"No wavelength column found in extension {hdu.name}. "
                    f"Tried: {WAVE_CANDIDATES}"
                )

            wave = np.asarray(hdu.data[wave_col], dtype=float)
            flux = np.asarray(hdu.data[flux_column], dtype=float)

            # Flatten common FITS-table vector-column shapes
            wave = np.ravel(wave)
            flux = np.ravel(flux)

            extname = hdu.header.get("EXTNAME", hdu.name)
            items.append(
                {
                    "name": extname,
                    "wave": wave,
                    "flux": flux,
                    "header": hdu.header,
                }
            )

    return items


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True, help="Input FITS file")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--column", required=True, help="Flux column to plot")
    parser.add_argument("--xlo", type=float, default=None, help="Minimum wavelength")
    parser.add_argument("--xhi", type=float, default=None, help="Maximum wavelength")
    parser.add_argument("--ncol", type=int, default=6, help="Number of columns in mosaic")
    parser.add_argument(
        "--yscale",
        choices=["global", "perpanel"],
        default="global",
        help="Y-axis scaling mode",
    )
    parser.add_argument(
        "--ymode",
        choices=["linear", "log"],
        default="log",
        help="Use linear or logarithmic y-axis",
    )
    parser.add_argument("--ylo", type=float, default=None, help="Optional lower y-limit")
    parser.add_argument("--yhi", type=float, default=None, help="Optional upper y-limit")
    parser.add_argument(
        "--title",
        default="Spectra after background subtraction",
        help="Figure title",
    )
    parser.add_argument(
        "--outfile-root",
        default=None,
        help="Root name for output files (default derived from column name)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    items = load_spectra(args.infile, args.column)
    if len(items) == 0:
        raise RuntimeError(f"No usable spectra found in {args.infile} for column {args.column}")

    nspec = len(items)
    ncol = max(1, args.ncol)
    nrow = math.ceil(nspec / ncol)

    # Precompute y limits if global scaling
    spectra_for_ylim = [(it["wave"], it["flux"]) for it in items]
    global_ylim = None
    if args.yscale == "global":
        global_ylim = compute_global_ylim(
            spectra_for_ylim,
            args.xlo,
            args.xhi,
            ymode=args.ymode,
            user_ylo=args.ylo,
            user_yhi=args.yhi,
        )

    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(2.9 * ncol, 2.0 * nrow + 0.8),
        squeeze=False,
        sharex=True,
        sharey=(args.yscale == "global"),
    )

    axes = axes.ravel()

    for i, it in enumerate(items):
        ax = axes[i]
        wave = it["wave"]
        flux = it["flux"]
        name = it["name"]

        m = np.isfinite(wave) & np.isfinite(flux)
        if args.xlo is not None:
            m &= (wave >= args.xlo)
        if args.xhi is not None:
            m &= (wave <= args.xhi)

        xx = wave[m]
        yy = flux[m]

        if args.ymode == "log":
            good = np.isfinite(yy) & (yy > 0)
            xx = xx[good]
            yy = yy[good]

        ax.plot(xx, yy, "k.", ms=0.7, alpha=0.65, rasterized=True)

        # O2 B and A band markers
        ax.axvspan(*O2_B_SPAN, color="deepskyblue", alpha=0.10, zorder=0)
        ax.axvspan(*O2_A_SPAN, color="deepskyblue", alpha=0.10, zorder=0)
        ax.axvline(O2_B_CENTER, color="dodgerblue", ls=":", lw=0.8, alpha=0.9)
        ax.axvline(O2_A_CENTER, color="dodgerblue", ls=":", lw=0.8, alpha=0.9)

        ax.set_title(name, fontsize=7, pad=2)

        if args.ymode == "log":
            ax.set_yscale("log")

        if args.yscale == "global":
            ax.set_ylim(*global_ylim)
        else:
            # per-panel scaling
            if args.ymode == "log":
                ypos = finite_positive(yy)
                if ypos.size:
                    ylo = args.ylo if args.ylo is not None else np.nanmin(ypos)
                    yhi = args.yhi if args.yhi is not None else np.nanmax(ypos) * 1.15
                    ax.set_ylim(ylo, yhi)
            else:
                if yy.size:
                    ylo = args.ylo if args.ylo is not None else np.nanpercentile(yy, 1)
                    yhi = args.yhi if args.yhi is not None else np.nanpercentile(yy, 99)
                    pad = 0.05 * (yhi - ylo if yhi > ylo else 1.0)
                    ax.set_ylim(ylo - pad, yhi + pad)

        ax.grid(True, alpha=0.18, lw=0.5)

        if args.xlo is not None or args.xhi is not None:
            ax.set_xlim(args.xlo, args.xhi)

        # labels only on outer panels
        row = i // ncol
        col = i % ncol
        if row == nrow - 1:
            ax.tick_params(axis="x", labelsize=7)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        if col == 0:
            ax.tick_params(axis="y", labelsize=7)
        else:
            ax.tick_params(axis="y", labelleft=False)

    # Hide unused axes
    for j in range(nspec, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(args.title, fontsize=12, y=0.995)
    fig.supxlabel("Wavelength (nm)", fontsize=10)
    ylabel = "Signal (log scale)" if args.ymode == "log" else "Signal"
    fig.supylabel(ylabel, fontsize=10)

    fig.tight_layout(rect=(0.02, 0.03, 1.0, 0.97))

    if args.outfile_root is None:
        root = f"qc_mosaic_{args.column.lower()}"
    else:
        root = args.outfile_root

    png_path = os.path.join(args.outdir, root + ".png")
    pdf_path = os.path.join(args.outdir, root + ".pdf")

    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")


if __name__ == "__main__":
    main()