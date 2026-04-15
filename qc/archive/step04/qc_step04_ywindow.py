#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QC04 — Y-window inspector for trace images

Allows interactive inspection of a vertical slice of the detector
with configurable Y limits.

Useful for:
- checking residual background (Step03.5)
- inspecting trace quality near boundaries
- validating gap detection regions
"""

from __future__ import annotations
import argparse

import sys
from pathlib import Path

# --- make repo visible ---
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib
import config
importlib.reload(config)

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# USER PARAMETERS  ← EDIT THESE
# ============================================================================
YMIN = 1800
YMAX = 2100

FILE = config.ST04_TRACES / "Even_traces.fits"

SHOW_PROFILE = True      # show median X-profile
SHOW_FULL = True         # also show full frame for context

PCT = (5, 99)


# ============================================================================
# LOAD DATA
# ============================================================================
img = fits.getdata(FILE)

ny, nx = img.shape
print(f"[INFO] Image shape: {ny} x {nx}")

if YMIN < 0 or YMAX > ny:
    raise ValueError("Y window out of bounds")

img_slice = img[YMIN:YMAX, :]


# ============================================================================
# DISPLAY — MAIN SLICE
# ============================================================================
vmin = np.percentile(img_slice, PCT[0])
vmax = np.percentile(img_slice, PCT[1])

plt.figure(figsize=(12, 5))
plt.imshow(img_slice,
           origin="lower",
           aspect="auto",
           vmin=vmin,
           vmax=vmax)

plt.colorbar()
plt.title(f"Even_traces — Y range {YMIN}:{YMAX}")
plt.xlabel("X")
plt.ylabel("Y (local)")
plt.tight_layout()
if args.show_plots:
            plt.show()
        else:
            plt.close()


# ============================================================================
# DISPLAY — FULL FRAME WITH WINDOW OVERLAY
# ============================================================================
if SHOW_FULL:
    vmin_full = np.percentile(img, PCT[0])
    vmax_full = np.percentile(img, PCT[1])

    plt.figure(figsize=(8, 10))
    plt.imshow(img,
               origin="lower",
               aspect="auto",
               vmin=vmin_full,
               vmax=vmax_full)

    plt.axhline(YMIN, color="white", linestyle="--")
    plt.axhline(YMAX, color="white", linestyle="--")

    plt.title("Full frame with Y window overlay")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    if args.show_plots:
            plt.show()
        else:
            plt.close()


# ============================================================================
# DISPLAY — MEDIAN PROFILE IN WINDOW
# ============================================================================
if SHOW_PROFILE:
    profile = np.median(img_slice, axis=0)

    plt.figure(figsize=(12, 4))
    plt.plot(profile)

    # mark your background regions
    plt.axvline(1300, linestyle="--", label="left sideband")
    plt.axvline(2800, linestyle="--", label="right sideband")

    plt.title(f"Median X-profile (Y={YMIN}:{YMAX})")
    plt.xlabel("X")
    plt.ylabel("Signal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if args.show_plots:
            plt.show()
        else:
            plt.close()