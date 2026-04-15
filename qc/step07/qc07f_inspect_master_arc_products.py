#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC07f — inspect master arc products and shift bookkeeping.

Input:
  - config.MASTER_ARC_FITS   (Step07f)

Shows:
  1. master median and mean spectra
  2. aligned stack image
  3. coverage curve

Also prints:
  - bright-line alignment summary
  - SLITLIST bookkeeping summary
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


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=str, default=None)
    ap.add_argument("--save-prefix", type=str, default=None)
    ap.add_argument("--show-plots", action="store_true")
    return ap.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    master_fits = Path(args.master).expanduser() if args.master else Path(config.MASTER_ARC_FITS).expanduser()
    if not master_fits.exists():
        raise FileNotFoundError(master_fits)

    with fits.open(master_fits) as h:
        master_median = np.asarray(h[0].data, float)
        master_mean = np.asarray(h["MASTER_MEAN"].data, float)
        stack_all = np.asarray(h["ALIGNED_STACK"].data, float)
        coverage_all = np.asarray(h["COVERAGE"].data, float)
        slitlist = None
        if "SLITLIST" in h:
            slitlist = h["SLITLIST"].data

    bry = np.array([np.nanargmax(r) for r in stack_all if np.any(np.isfinite(r))], dtype=float)
    bry_med = float(np.nanmedian(bry)) if bry.size else np.nan

    # Master spectrum
    plt.figure(figsize=(10, 4))
    plt.plot(master_median, label="median")
    plt.plot(master_mean, alpha=0.6, label="mean")
    if np.isfinite(bry_med):
        plt.axvline(bry_med, linewidth=1, linestyle="--", label=f"brightline ~ {bry_med:.0f}")
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
    if np.isfinite(bry_med):
        plt.axvline(bry_med, linewidth=1, linestyle="--")
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
    if np.isfinite(bry_med):
        plt.axvline(bry_med, linewidth=1, linestyle="--")
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

    print("\nQC07f summary")
    print("  master FITS:", master_fits)
    print("  stack shape:", stack_all.shape)
    if bry.size:
        print(f"  brightline median pix: {np.nanmedian(bry):.1f}")
        print(f"  brightline RMS scatter: {np.nanstd(bry):.2f} px")
    print(f"  coverage median: {np.median(coverage_all)}  min: {np.nanmin(coverage_all)}  max: {np.nanmax(coverage_all)}")

    if slitlist is not None:
        print("\nSLITLIST summary")
        print("  rows:", len(slitlist))

        # decode possible bytes columns safely
        def _as_str(x):
            return x.decode().strip() if isinstance(x, (bytes, bytearray)) else str(x).strip()

        names = slitlist.dtype.names or ()
        print("  columns:", names)

        if "SHIFT_FINAL" in names:
            sf = np.asarray(slitlist["SHIFT_FINAL"], dtype=float)
            print(f"  SHIFT_FINAL range: {np.nanmin(sf):.0f} .. {np.nanmax(sf):.0f}")

        if "SHIFT_GLOBAL" in names:
            sg = np.asarray(slitlist["SHIFT_GLOBAL"], dtype=float)
            uniq = np.unique(sg[np.isfinite(sg)]).astype(int)
            print("  SHIFT_GLOBAL unique:", uniq.tolist())

        if "SHIFT_TO_MASTER" in names:
            sm = np.asarray(slitlist["SHIFT_TO_MASTER"], dtype=float)
            print(f"  SHIFT_TO_MASTER range: {np.nanmin(sm):.0f} .. {np.nanmax(sm):.0f}")

        print("\nFirst rows:")
        nshow = min(10, len(slitlist))
        for i in range(nshow):
            row = slitlist[i]
            vals = []
            for k in names:
                vals.append(f"{k}={_as_str(row[k])}")
            print("  " + ", ".join(vals))


if __name__ == "__main__":
    main()
