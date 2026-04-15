#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC07d — inspect bright-line alignment across slits.

Inputs
------
- Step07c 1D arc MEF:
    ArcDiff_*_pixflatcorr_clipped_1D_slitid_<EVEN|ODD>.fits

- Step07d shift CSV:
    Arc_shifts_initial_<EVEN|ODD>.csv

Shows
-----
1. raw normalized overlay
2. aligned normalized overlay
3. aligned stack image (vertical-line alignment check)
4. BRY vs slit number
5. SHIFT vs slit number

Run
---
    PYTHONPATH=. python qc/step07/qc07d_inspect_bright_line_alignment.py --set EVEN
    PYTHONPATH=. python qc/step07/qc07d_inspect_bright_line_alignment.py --set ODD
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import config


def list_slits(hdul):
    slits = []
    for hh in hdul[1:]:
        nm = (hh.header.get("EXTNAME") or "").strip().upper()
        if nm.startswith("SLIT") and len(nm) == 7:
            slits.append(nm)
    return sorted(set(slits), key=lambda s: int(s.replace("SLIT", "")))


def slit_num(s: str) -> int:
    return int(s.replace("SLIT", ""))


def read_shift_csv(path: Path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def default_arc1d_path(trace_set: str) -> Path:
    st07 = Path(config.ST07_WAVECAL).expanduser()
    if hasattr(config, "MASTER_ARC_DIFF"):
        stem = Path(config.MASTER_ARC_DIFF).stem
        return st07 / f"{stem}_pixflatcorr_clipped_1D_slitid_{trace_set}.fits"
    hits = sorted(st07.glob(f"ArcDiff*_pixflatcorr_clipped_1D_slitid_{trace_set}.fits"))
    if hits:
        return hits[0]
    return st07 / f"ArcDiff_pixflatcorr_clipped_1D_slitid_{trace_set}.fits"


def default_plot_ids(trace_set: str) -> list[int]:
    if trace_set.upper() == "EVEN":
        return [2, 8, 16, 24, 32, 48]
    return [1, 9, 19, 21, 31, 49]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["EVEN", "ODD"], default="EVEN")
    ap.add_argument("--arc1d", type=str, default=None)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--plot-all", action="store_true")
    ap.add_argument("--show-plots", action="store_true")
    ap.add_argument("--save-prefix", type=str, default=None)
    return ap.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    trace_set = args.set.upper()
    st07 = Path(config.ST07_WAVECAL).expanduser()

    arc1d_fits = Path(args.arc1d).expanduser() if args.arc1d else default_arc1d_path(trace_set)
    csv_path = Path(args.csv).expanduser() if args.csv else (st07 / f"Arc_shifts_initial_{trace_set}.csv")

    if not arc1d_fits.exists():
        raise FileNotFoundError(arc1d_fits)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    print("TRACE_SET  =", trace_set)
    print("ARC1D_FITS =", arc1d_fits)
    print("CSV        =", csv_path)

    rows = read_shift_csv(csv_path)
    if not rows:
        raise RuntimeError(f"No rows in {csv_path}")

    shift_map = {r["slit"].strip().upper(): int(float(r["SHIFT_vs_REF(px)"])) for r in rows}
    refslit = next((r["slit"].strip().upper() for r in rows if int(float(r["SHIFT_vs_REF(px)"])) == 0), None)

    with fits.open(arc1d_fits) as h:
        slits_present = list_slits(h)

        if args.plot_all:
            plot_slits = slits_present
        else:
            target_ids = default_plot_ids(trace_set)
            wanted = {f"SLIT{i:03d}" for i in target_ids}
            plot_slits = [s for s in slits_present if s in wanted]
            if not plot_slits:
                plot_slits = slits_present[: min(8, len(slits_present))]

        plt.figure(figsize=(10, 6))
        for s in plot_slits:
            if s not in h:
                continue
            f1d = h[s].data[0].astype(float)
            good = np.isfinite(f1d)
            if good.sum() < 10:
                continue
            scale = np.nanpercentile(f1d[good], 99)
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            plt.plot(f1d / scale, linewidth=0.8, alpha=0.7)
        plt.title(f"QC07d: Raw arc spectra ({trace_set}, normalized)")
        plt.xlabel("Detector Y (px)")
        plt.ylabel("Flux (norm)")
        plt.tight_layout()
        if args.show_plots:
            plt.show()
        else:
            plt.close()

        plt.figure(figsize=(10, 6))
        for s in plot_slits:
            if s not in h or s not in shift_map:
                continue
            f1d = h[s].data[0].astype(float)
            good = np.isfinite(f1d)
            if good.sum() < 10:
                continue
            scale = np.nanpercentile(f1d[good], 99)
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            sh = shift_map[s]
            f_shift = np.roll(f1d, -sh)
            plt.plot(f_shift / scale, linewidth=0.8, alpha=0.7)
        plt.title(f"QC07d: Aligned arc spectra ({trace_set}, ref={refslit})")
        plt.xlabel("Detector Y (px, shifted)")
        plt.ylabel("Flux (norm)")
        plt.tight_layout()
        if args.show_plots:
            plt.show()
        else:
            plt.close()

        ny = h[slits_present[0]].data.shape[1]
        stack = np.full((len(slits_present), ny), np.nan, dtype=float)
        for i, s in enumerate(slits_present):
            if s not in shift_map:
                continue
            f1d = h[s].data[0].astype(float)
            sh = shift_map[s]
            stack[i] = np.roll(f1d, -sh)

        disp = stack.copy()
        for i in range(disp.shape[0]):
            row = disp[i]
            good = np.isfinite(row)
            if good.sum() < 10:
                continue
            scale = np.nanpercentile(row[good], 99)
            if np.isfinite(scale) and scale > 0:
                disp[i, good] = row[good] / scale

        plt.figure(figsize=(10, 8))
        plt.imshow(disp, origin="lower", aspect="auto", interpolation="nearest")
        plt.title(f"QC07d: Vertical-line alignment check ({trace_set})")
        plt.xlabel("Detector Y (px, shifted)")
        plt.ylabel("Slit index")
        plt.tight_layout()
        if args.show_plots:
            plt.show()
        else:
            plt.close()

    slits = [r["slit"].strip().upper() for r in rows]
    x = np.array([slit_num(s) for s in slits], dtype=int)
    bry = np.array([float(r["BRY"]) for r in rows], dtype=float)
    shift = np.array([float(r["SHIFT_vs_REF(px)"]) for r in rows], dtype=float)

    plt.figure(figsize=(10, 4))
    plt.plot(x, bry, marker="o", linewidth=1)
    plt.title(f"QC07d: BRY vs slit number ({trace_set})")
    plt.xlabel("Slit number")
    plt.ylabel("BRY (detector Y)")
    plt.tight_layout()
    if args.show_plots:
          plt.show()
    else:
        plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(x, shift, marker="o", linewidth=1)
    plt.title(f"QC07d: SHIFT vs slit number ({trace_set}, ref={refslit})")
    plt.xlabel("Slit number")
    plt.ylabel("SHIFT (px)")
    plt.axhline(0, linewidth=1)
    plt.tight_layout()
    if args.show_plots:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    main()
