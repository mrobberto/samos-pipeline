#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC Step12a — Offset-stacked slit illumination profiles.

This script visualizes the individual slit illumination profiles derived
in Step12a before they are combined into the master illumination function.
It provides a diagnostic view of the consistency and scatter of the
per-slit profiles across the detector.

Purpose
-------
The goal of this QC is to assess:

  (1) the stability of the illumination profile across slits,
  (2) the presence of outliers or pathological slits,
  (3) the smoothness and coherence of the large-scale response.

Method
------
For each parity (EVEN / ODD):

1. Input:
   The raw per-slit profiles from Step12a:
       - ILLUM1D_RAW_EVEN
       - ILLUM1D_RAW_ODD

   Each slit provides:
       - PROFILE_RAW(Y)
       - VALID mask
       - XCENTER(Y) trace geometry

2. Normalization:
   Each slit profile is normalized using the median value within a
   reference Y range (default: 1500–2600), corresponding to a
   well-illuminated, high-S/N region.

3. Smoothing:
   A Gaussian-smoothed version of each normalized profile is computed
   to highlight the large-scale behavior and suppress pixel-scale noise.

4. Sorting:
   Slits are ordered by their median detector X position (XCENTER),
   effectively mapping their spatial location in the field.

5. Visualization:
   Profiles are plotted as an offset stack:
     - thin lines: raw normalized profiles
     - thick lines: smoothed profiles
     - vertical offset applied for visual separation

   Each slit is labeled with its ID and approximate field position.

Outputs
-------
Two multi-page PDF files:

  - qc_step12a_offset_even.pdf
  - qc_step12a_offset_odd.pdf

Each page shows all slit profiles for one parity.

Interpretation
--------------
A well-behaved dataset should show:

  - consistent large-scale profile shapes across slits
  - small scatter after normalization
  - smooth trends in the smoothed curves
  - no strong discontinuities or anomalous profiles

Deviations may indicate:

  - problematic slits (bad extraction, contamination, low S/N)
  - spatial variations in throughput
  - residual flat-fielding issues
  - edge effects at detector boundaries

Notes
-----
- The normalization region is critical: it defines the relative scaling
  between slits and should lie within a stable, well-illuminated region.

- The illumination profiles include the spectral shape of the quartz lamp
  and therefore represent an empirical instrumental response, not a
  pure flat field.

- This QC validates the inputs used to construct the master illumination
  profile applied in Step12b.

Dependencies
------------
- Step12a raw outputs (ILLUM1D_RAW_* files)

This script is intended for visual inspection and diagnostic validation,
and is not part of the automated calibration pipeline.
"""
from __future__ import annotations

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import config


def gaussian_smooth_1d(y, sigma=8.0):
    if sigma <= 0:
        return y.copy()

    radius = max(1, int(4 * sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()

    y_in = np.asarray(y, dtype=float)
    valid = np.isfinite(y_in)
    y0 = np.where(valid, y_in, 0.0)

    num = np.convolve(y0, k, mode="same")
    den = np.convolve(valid.astype(float), k, mode="same")

    out = np.full_like(y0, np.nan, dtype=float)
    m = den > 0
    out[m] = num[m] / den[m]
    return out


def collect_profiles(rawfile, y0=1500, y1=2600):
    rows = []
    with fits.open(rawfile) as hdul:
        for hdu in hdul[1:]:
            name = hdu.name
            ypix = np.asarray(hdu.data["YPIX"], dtype=float)
            prof = np.asarray(hdu.data["PROFILE_RAW"], dtype=float)
            valid = np.asarray(hdu.data["VALID"], dtype=bool)
            xcen = np.asarray(hdu.data["XCENTER"], dtype=float)

            good = valid & np.isfinite(prof)
            if np.count_nonzero(good) == 0:
                continue

            prof_use = prof.copy()
            prof_use[~good] = np.nan

            mnorm = (ypix >= y0) & (ypix <= y1) & np.isfinite(prof_use)
            if np.count_nonzero(mnorm) == 0:
                continue

            norm = np.nanmedian(prof_use[mnorm])
            if not np.isfinite(norm) or norm == 0:
                continue

            prof_norm = prof_use / norm
            xmed = np.nanmedian(xcen)

            rows.append({
                "name": name,
                "ypix": ypix,
                "prof_norm": prof_norm,
                "prof_smooth": gaussian_smooth_1d(prof_norm, sigma=8.0),
                "xmed": xmed,
            })

    rows.sort(key=lambda r: r["xmed"])
    return rows


def plot_offset_stack(rows, title, outfile, offset_step=0.12):
    with PdfPages(outfile) as pdf:
        fig, ax = plt.subplots(figsize=(11, 14))

        for i, row in enumerate(rows):
            off = i * offset_step
            y = row["ypix"]
            p = row["prof_norm"]
            s = row["prof_smooth"]

            ax.plot(y, p + off, linewidth=0.8, alpha=0.35)
            ax.plot(y, s + off, linewidth=1.5)

            ax.text(
                y[-1] + 25,
                off,
                f'{row["name"]}   x~{row["xmed"]:.1f}',
                fontsize=7,
                va="center",
            )

        ax.set_title(title)
        ax.set_xlabel("Detector row Y")
        ax.set_ylabel("Normalized profile + offset")
        ax.grid(alpha=0.2)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main():
    config.ensure_directories()

    even_rows = collect_profiles(config.ILLUM1D_RAW_EVEN)
    odd_rows = collect_profiles(config.ILLUM1D_RAW_ODD)

    plot_offset_stack(
        even_rows,
        "Step12a EVEN normalized slit profiles, offset and sorted by field position",
        config.QC12_DIR / "qc_step12a_offset_even.pdf",
    )

    plot_offset_stack(
        odd_rows,
        "Step12a ODD normalized slit profiles, offset and sorted by field position",
        config.QC12_DIR / "qc_step12a_offset_odd.pdf",
    )

    print("Wrote:")
    print(config.QC12_DIR / "qc_step12a_offset_even.pdf")
    print(config.QC12_DIR / "qc_step12a_offset_odd.pdf")


if __name__ == "__main__":
    main()