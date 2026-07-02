#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC Step12a — Raw and normalized slit illumination profiles.

This script visualizes the individual slit illumination profiles derived
in Step12a and compares their statistical behavior before and after
normalization. It provides a diagnostic assessment of the consistency
of the per-slit profiles and the robustness of the master illumination
function.

Purpose
-------
The goal of this QC is to evaluate:

  (1) the intrinsic scatter of raw slit profiles,
  (2) the effectiveness of the normalization procedure,
  (3) the stability of the master illumination profile,
  (4) the number of contributing slits as a function of detector row.

Method
------
For each parity (EVEN / ODD):

1. Input:
   Raw slit profiles from Step12a:
       - PROFILE_RAW(Y)
       - PROFILE_NORM(Y) (if present)
       - VALID mask

2. Raw profiles:
   All slit-collapse profiles are overplotted in detector coordinates.
   These show the combined effects of:
     - instrumental throughput
     - slit-to-slit variations
     - quartz lamp spectral shape

3. Normalized profiles:
   Profiles are normalized (either using Step12a normalization or
   recomputed here over a reference Y range) to remove slit-to-slit
   scaling differences.

4. Statistical summary:
   At each detector row:
     - mean profile (thin line)
     - median profile (thick line)
     - number of contributing slits (secondary axis)

5. Master comparison:
   The median profile approximates the master illumination function
   used in Step12a.

Outputs
-------
Two multi-page PDF files:

  - config.QC_STEP12A_RAW_EVEN_PDF
  - config.QC_STEP12A_RAW_ODD_PDF

Each file contains:
  - raw profile overlay
  - normalized profile overlay
  - statistical summaries (mean, median, N_used)

Interpretation
--------------
A well-behaved dataset should show:

  - large scatter in raw profiles (expected due to slit throughput differences)
  - strong convergence after normalization
  - smooth and stable median profile
  - high and relatively uniform N_used across the central detector region

Potential issues revealed by this QC include:

  - inconsistent normalization (profiles do not converge)
  - outlier slits dominating the mean
  - gaps or drops in N_used (missing or invalid data)
  - edge effects or low-S/N regions at detector boundaries

Notes
-----
- The normalization region is critical for aligning profiles; it should
  lie within a stable, well-illuminated portion of the detector.

- The illumination profiles include the spectral shape of the quartz
  lamp and therefore represent an empirical instrumental response.

- The median profile derived here forms the basis of the illumination
  correction applied in Step12b.

Dependencies
------------
- Step12a raw outputs (ILLUM1D_RAW_* files)

This script is intended for diagnostic validation and visual inspection,
and is not part of the automated calibration pipeline.
"""
from __future__ import annotations

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import config


def collect_profiles(rawfile):
    raw = []
    norm = []

    with fits.open(rawfile) as hdul:
        for hdu in hdul[1:]:
            cols = list(hdu.columns.names)

            # skip MASTER or any non-slit HDU
            if "PROFILE_RAW" not in cols:
                continue

            y = hdu.data["YPIX"]
            valid = hdu.data["VALID"]

            pr = hdu.data["PROFILE_RAW"]
            good_raw = valid & np.isfinite(pr)
            if np.any(good_raw):
                raw.append((hdu.name, y[good_raw], pr[good_raw]))

            if "PROFILE_NORM" in cols:
                pn = hdu.data["PROFILE_NORM"]
                good_norm = valid & np.isfinite(pn)
                if np.any(good_norm):
                    norm.append((hdu.name, y[good_norm], pn[good_norm]))

    return raw, norm


def normalize_profiles(profiles, y0=1500, y1=2600):
    out = []
    for name, x, y in profiles:
        m = (x >= y0) & (x <= y1) & np.isfinite(y)
        if np.count_nonzero(m) == 0:
            continue
        norm = np.nanmedian(y[m])
        if not np.isfinite(norm) or norm == 0:
            continue
        out.append((name, x, y / norm))
    return out

def get_master(rawfile):
    with fits.open(rawfile) as hdul:
        d = hdul["MASTER"].data
        return d["YPIX"], d["MASTER_RAW"], d["MASTER_SMOOTH"], d["MASTER_NORM"], d["N_USED"]

def plot_overlay(profiles, title, ylabel, pdf):
    fig, ax = plt.subplots(figsize=(10, 5))

    # assume all profiles are on the same Y grid
    yref = profiles[0][1]
    stack = []

    for name, y, p in profiles:
        ax.plot(y, p, alpha=0.2, linewidth=0.8)

        full = np.full_like(yref, np.nan, dtype=float)
        n = min(len(full), len(p))
        full[:n] = p[:n]
        stack.append(full)

    stack = np.vstack(stack)

    med = np.nanmedian(stack, axis=0)
    mean = np.nanmean(stack, axis=0)
    n_used = np.sum(np.isfinite(stack), axis=0)

    ax.plot(yref, mean, linewidth=2.0, label="mean")
    ax.plot(yref, med, linewidth=2.5, label="median")
    ax2 = ax.twinx()
    ax2.plot(yref, n_used, linewidth=1.0, alpha=0.35, label="N used")
    ax2.set_ylabel("N slits used")

    ax.set_title(title)
    ax.set_xlabel("Detector row Y")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")
    pdf.savefig(fig)
    plt.close(fig)
    


def write_qc(rawfile, outfile, parity_label):
    raw_profiles, norm_profiles = collect_profiles(rawfile)

    with PdfPages(outfile) as pdf:
        plot_overlay(
            raw_profiles,
            f"Step12a raw slit profiles {parity_label}",
            "Raw slit-collapse signal",
            pdf,
        )

        plot_overlay(
            norm_profiles,
            f"Step12a normalized slit profiles {parity_label}",
            "Normalized slit-collapse signal",
            pdf,
        )

def main():
    config.ensure_directories()

    write_qc(
        config.ILLUM1D_RAW_EVEN,
        config.QC_STEP12A_RAW_EVEN_PDF,
        "EVEN",
    )
    write_qc(
        config.ILLUM1D_RAW_ODD,
        config.QC_STEP12A_RAW_ODD_PDF,
        "ODD",
    )

    print("Wrote:")
    print(config.QC_STEP12A_RAW_EVEN_PDF)
    print(config.QC_STEP12A_RAW_ODD_PDF)


if __name__ == "__main__":
    main()