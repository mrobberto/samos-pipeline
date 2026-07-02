#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import config


def collect_profiles(rawfile):
    profiles = []
    with fits.open(rawfile) as hdul:
        for hdu in hdul[1:]:
            x = hdu.data["XPIX"]
            y = hdu.data["PROFILE_RAW"]
            v = hdu.data["VALID"]
            profiles.append((hdu.name, x[v], y[v]))
    return profiles


def plot_overlay(profiles, title, pdf_path):
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(10, 5))
        grid = None
        stack = []

        for name, x, y in profiles:
            ax.plot(x, y, alpha=0.2, linewidth=0.8)
            if grid is None:
                grid = x
            if len(x) == len(grid) and np.all(x == grid):
                stack.append(y)

        if stack:
            med = np.nanmedian(np.vstack(stack), axis=0)
            ax.plot(grid, med, linewidth=2.0)

        ax.set_title(title)
        ax.set_xlabel("Dispersion pixel")
        ax.set_ylabel("Raw slit-collapse signal")
        pdf.savefig(fig)
        plt.close(fig)


def main():
    config.ensure_directories()

    even_profiles = collect_profiles(config.ILLUM1D_RAW_EVEN)
    odd_profiles  = collect_profiles(config.ILLUM1D_RAW_ODD)

    plot_overlay(even_profiles, "Step12a raw slit profiles EVEN", config.QC_STEP12A_RAW_EVEN_PDF)
    plot_overlay(odd_profiles,  "Step12a raw slit profiles ODD",  config.QC_STEP12A_RAW_ODD_PDF)

    print("Wrote:")
    print(config.QC_STEP12A_RAW_EVEN_PDF)
    print(config.QC_STEP12A_RAW_ODD_PDF)


if __name__ == "__main__":
    main()