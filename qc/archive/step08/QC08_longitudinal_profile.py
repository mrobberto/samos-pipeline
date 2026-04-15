#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 07:26:14 2026

@author: robberto
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--file08", required=True)
ap.add_argument("--file06", required=True)
ap.add_argument("--slit", required=True)
args = ap.parse_args()

# -------------------------
# Load data
# -------------------------
h08 = fits.open(args.file08)
h06 = fits.open(args.file06)

tab = h08[args.slit].data
img = np.array(h06[args.slit].data, float)

x0 = tab["X0"]
ny, nx = img.shape

# -------------------------
# Build aligned grid
# -------------------------
x = np.arange(nx)

# choose symmetric range around ridge
dx_max = int(nx / 2)
dx_grid = np.arange(-dx_max, dx_max + 1)

stack = []
coverage = []

for y in range(ny):
    if not np.isfinite(x0[y]):
        continue

    dx = x - x0[y]

    row = img[y]

    # interpolate row onto dx_grid
    valid = np.isfinite(row)
    if valid.sum() < 5:
        continue

    try:
        interp = np.interp(dx_grid, dx[valid], row[valid], left=np.nan, right=np.nan)
    except:
        continue

    stack.append(interp)
    coverage.append(np.isfinite(interp))

stack = np.array(stack)
coverage = np.array(coverage)

# -------------------------
# Statistics
# -------------------------
profile_median = np.nanmedian(stack, axis=0)
profile_mean = np.nanmean(stack, axis=0)
profile_std = np.nanstd(stack, axis=0)

coverage_frac = np.mean(coverage, axis=0)

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# profile
ax[0].plot(dx_grid, profile_median, label="median", lw=2)
ax[0].plot(dx_grid, profile_mean, label="mean", lw=1, alpha=0.7)
ax[0].fill_between(dx_grid,
                   profile_median - profile_std,
                   profile_median + profile_std,
                   alpha=0.2, label="±1σ")

ax[0].axvline(0, color="r", linestyle="--", label="ridge")
ax[0].set_ylabel("Flux")
ax[0].set_title(f"{args.slit} — longitudinal profile")
ax[0].legend()

# coverage
ax[1].plot(dx_grid, coverage_frac, lw=2)
ax[1].set_ylabel("Coverage fraction")
ax[1].set_xlabel("ΔX (pixels)")
ax[1].set_ylim(0, 1.05)

plt.tight_layout()
if args.show_plots:
            plt.show()
        else:
            plt.close()