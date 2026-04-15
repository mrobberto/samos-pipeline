#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC07f — inspect master arc products.

Input:
  - arc_master.fits   (Step07f)

Shows:
  1. master median and mean spectra
  2. aligned stack image
  3. coverage curve
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import config


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-plots", action="store_true",
                    help="Show plots interactively")
    ap.add_argument("--save-prefix", type=str, default=None,
                    help="Optional prefix for saving QC PNGs")
    return ap.parse_args()


def main(argv=None):
    args = parse_args()
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=str, default=None,
                    help="Input master arc FITS. Default: config.ST07_WAVECAL / arc_master.fits")
    ap.add_argument("--save-prefix", type=str, default=None,
                    help="Optional prefix for saving PNGs. If omitted, plots are shown only.")
    args = ap.parse_args(argv)

    master_fits = Path(args.master).expanduser() if args.master else (
        Path(config.ST07_WAVECAL) / "arc_master.fits"
    )
    if not master_fits.exists():
        raise FileNotFoundError(master_fits)

    with fits.open(master_fits) as h:
        master_median = np.asarray(h[0].data, float)
        master_mean = np.asarray(h["MASTER_MEAN"].data, float)
        stack_all = np.asarray(h["ALIGNED_STACK"].data, float)
        coverage_all = np.asarray(h["COVERAGE"].data, float)

    # Master spectrum
    plt.figure(figsize=(10, 4))
    plt.plot(master_median, label="median")
    plt.plot(master_mean, alpha=0.6, label="mean")
    plt.xlabel("Pixel")
    plt.ylabel("Flux")
    plt.title("QC07f: Master arc")
    plt.legend()
    plt.tight_layout()
    if args.save_prefix:
        plt.savefig(f"{args.save_prefix}_master_arc.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        if args.show_plots:
            plt.show()
        else:
            plt.close()

    # Aligned stack
    vmin = np.nanpercentile(stack_all, 5)
    vmax = np.nanpercentile(stack_all, 99)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        stack_all,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        cmap="gray",
    )
    plt.xlabel("Pixel")
    plt.ylabel("Stack row")
    plt.title("QC07f: Aligned stack")
    plt.tight_layout()
    if args.save_prefix:
        plt.savefig(f"{args.save_prefix}_aligned_stack.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        if args.show_plots:
            plt.show()
        else:
            plt.close()

    # Coverage
    plt.figure(figsize=(10, 3))
    plt.plot(coverage_all)
    plt.xlabel("Pixel")
    plt.ylabel("Coverage")
    plt.title("QC07f: Coverage")
    plt.tight_layout()
    if args.save_prefix:
        plt.savefig(f"{args.save_prefix}_coverage.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        if args.show_plots:
            plt.show()
        else:
            plt.close()

    bry = np.array([np.nanargmax(r) for r in stack_all if np.any(np.isfinite(r))], dtype=float)
    print("\nQC07f summary")
    print("  stack shape:", stack_all.shape)
    if bry.size:
        print(f"  brightline median pix: {np.nanmedian(bry):.1f}")
        print(f"  brightline RMS scatter: {np.nanstd(bry):.2f} px")
    print(f"  coverage median: {np.median(coverage_all)}  min: {np.nanmin(coverage_all)}  max: {np.nanmax(coverage_all)}")


if __name__ == "__main__":
    main()