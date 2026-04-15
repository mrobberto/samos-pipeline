#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step07d — measure bright-line positions and relative shifts across slits.

Baseline-port version:
- preserves the original bright-line / initial-shift logic
- patched only for current config paths and canonical filenames

Inputs
------
- Step07c 1D arc MEF:
    ArcDiff_*_pixflatcorr_clipped_1D_slitid_<EVEN|ODD>.fits

Outputs
-------
- Arc_shifts_initial_<EVEN|ODD>.csv
- Arc_stack_aligned_<EVEN|ODD>.fits

Method
------
- find the brightest emission line in each slit using a running-sum spectrum
- restrict the search to rows with sufficient slit coverage
- measure BRY and SHIFT relative to a reference slit
- build an aligned stack FITS for downstream QC / refinement

Run
---
    PYTHONPATH=. python pipeline/step07_wavecal/step07d_find_line_shifts.py --traceset EVEN
    PYTHONPATH=. python pipeline/step07_wavecal/step07d_find_line_shifts.py --traceset ODD
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

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


def running_sum(x, n=3):
    x = np.asarray(x, float)
    k = np.ones(n, dtype=float)
    y = np.convolve(np.nan_to_num(x, nan=0.0), k, mode="same")
    return y


def robust_valid_region(npix_row, frac=0.5):
    npix = np.asarray(npix_row, float)
    good = np.isfinite(npix) & (npix > 0)
    if good.sum() < 50:
        return good
    med = np.median(npix[good])
    thr = frac * med
    return good & (npix >= thr)


def peak_metrics(g, idx):
    ny = len(g)
    peak = g[idx]
    w = 50
    a = max(0, idx - w)
    b = min(ny, idx + w + 1)
    local = g[a:b]
    baseline = np.median(local[np.isfinite(local)]) if np.isfinite(local).any() else 0.0
    prom = peak - baseline

    half = baseline + 0.5 * prom
    left = idx
    while left > a and g[left] > half:
        left -= 1
    right = idx
    while right < b - 1 and g[right] > half:
        right += 1
    width = right - left
    return float(peak), float(prom), int(width)


def choose_best_refslit(hdul, slits, frac=0.70, edge_pad=20, runsum_n=21):
    best = None
    best_score = -np.inf
    best_idx = None

    for slit in slits:
        f = hdul[slit].data[0].astype(float)
        npix = hdul[slit].data[1].astype(float)

        valid = robust_valid_region(npix, frac=frac)
        y = np.where(valid)[0]
        if y.size < (2 * edge_pad + 50):
            continue

        ylo = int(y.min()) + edge_pad
        yhi = int(y.max()) - edge_pad
        if yhi <= ylo:
            continue

        g = running_sum(f, runsum_n)
        idx = int(np.nanargmax(g[ylo:yhi + 1]) + ylo)
        peak, prom, wid = peak_metrics(g, idx)

        score = float(prom) * float(yhi - ylo)

        if np.isfinite(score) and score > best_score:
            best_score = score
            best = slit
            best_idx = idx

    if best is None:
        best = slits[len(slits) // 2]
        best_idx = None
        best_score = np.nan

    return best, best_idx, best_score


def default_arc1d_path(traceset: str) -> Path:
    st07 = Path(config.ST07_WAVECAL).expanduser()
    if hasattr(config, "MASTER_ARC_DIFF"):
        stem = Path(config.MASTER_ARC_DIFF).stem
        return st07 / f"{stem}_pixflatcorr_clipped_1D_slitid_{traceset}.fits"
    hits = sorted(st07.glob(f"ArcDiff*_pixflatcorr_clipped_1D_slitid_{traceset}.fits"))
    if hits:
        return hits[0]
    return st07 / f"ArcDiff_pixflatcorr_clipped_1D_slitid_{traceset}.fits"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--traceset", "--set", dest="trace_set", choices=["EVEN", "ODD"], default="EVEN",
                    help="Trace set to process")
    ap.add_argument("--arc1d", type=str, default=None,
                    help="Input Step07c arc1d FITS")
    ap.add_argument("--out-csv", type=str, default=None,
                    help="Output CSV")
    ap.add_argument("--out-stack", type=str, default=None,
                    help="Output aligned stack FITS")
    ap.add_argument("--refslit", type=str, default=None,
                    help="Optional explicit reference slit, e.g. SLIT024")
    ap.add_argument("--runsum", type=int, default=3,
                    help="Running-sum width for bright-line finding")
    ap.add_argument("--edge-pad", type=int, default=10,
                    help="Ignore this many rows near each edge of the valid region")
    ap.add_argument("--min-coverage-frac", type=float, default=0.5,
                    help="Require npix_per_row >= frac * median(npix_per_row)")
    ap.add_argument("--make-plots", action="store_true",
                    help="Also show legacy QC plots")
    args = ap.parse_args(argv)

    trace_set = args.trace_set.upper()
    st07 = Path(config.ST07_WAVECAL).expanduser()

    arc1d_fits = Path(args.arc1d).expanduser() if args.arc1d else default_arc1d_path(trace_set)
    out_csv = Path(args.out_csv).expanduser() if args.out_csv else (st07 / f"Arc_shifts_initial_{trace_set}.csv")
    out_stack_fits = Path(args.out_stack).expanduser() if args.out_stack else (st07 / f"Arc_stack_aligned_{trace_set}.fits")

    print("TRACE_SET   =", trace_set)
    print("ARC1D_FITS  =", arc1d_fits)
    print("OUT_CSV     =", out_csv)
    print("OUT_STACK   =", out_stack_fits)

    if not arc1d_fits.exists():
        raise FileNotFoundError(arc1d_fits)

    runsum_n = int(args.runsum)
    edge_pad = int(args.edge_pad)
    min_coverage_frac = float(args.min_coverage_frac)

    rows = []

    with fits.open(arc1d_fits) as h:
        slits = list_slits(h)
        if not slits:
            raise RuntimeError(f"No SLIT### extensions found in {arc1d_fits}")

        if args.refslit is not None:
            refslit = args.refslit.strip().upper()
            if refslit not in slits:
                raise RuntimeError(f"Requested ref slit {refslit} not found in {arc1d_fits}")
        else:
            refslit, idx_guess, score = choose_best_refslit(
                h, slits, frac=min_coverage_frac, edge_pad=edge_pad, runsum_n=runsum_n
            )

        print("Using REFSLIT =", refslit)

        f_ref = h[refslit].data[0].astype(float)
        np_ref = h[refslit].data[1].astype(float)

        valid_ref = robust_valid_region(np_ref, frac=min_coverage_frac)
        y_ref = np.where(valid_ref)[0]
        if y_ref.size == 0:
            raise RuntimeError("No valid rows for reference slit.")

        ylo_ref = int(y_ref.min()) + edge_pad
        yhi_ref = int(y_ref.max()) - edge_pad
        if yhi_ref <= ylo_ref:
            raise RuntimeError("Reference valid region too small; reduce --edge-pad or --min-coverage-frac.")

        g_ref = running_sum(f_ref, runsum_n)
        idx_ref = int(np.nanargmax(g_ref[ylo_ref:yhi_ref + 1]) + ylo_ref)
        peak_ref, prom_ref, wid_ref = peak_metrics(g_ref, idx_ref)

        print(f"Reference {refslit}: BRY={idx_ref}  peak={peak_ref:.3g}  prom={prom_ref:.3g}  width~{wid_ref}px")
        rows.append((refslit, idx_ref, 0, peak_ref, prom_ref, wid_ref, ylo_ref, yhi_ref))

        for slit in slits:
            if slit == refslit:
                continue

            f = h[slit].data[0].astype(float)
            npix = h[slit].data[1].astype(float)

            valid = robust_valid_region(npix, frac=min_coverage_frac)
            yy = np.where(valid)[0]
            if yy.size < 50:
                valid = (npix > 0)
                yy = np.where(valid)[0]

            if yy.size == 0:
                print("WARNING: no valid rows for", slit)
                continue

            ylo = int(yy.min()) + edge_pad
            yhi = int(yy.max()) - edge_pad
            if yhi <= ylo:
                ylo = int(yy.min())
                yhi = int(yy.max())

            g = running_sum(f, runsum_n)
            idx = int(np.nanargmax(g[ylo:yhi + 1]) + ylo)
            peak, prom, wid = peak_metrics(g, idx)

            shift = idx - idx_ref
            rows.append((slit, idx, shift, peak, prom, wid, ylo, yhi))

    rows_sorted = sorted(rows, key=lambda r: slit_num(r[0]))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slit", "BRY", "SHIFT_vs_REF(px)", "peak", "prominence", "width_px", "search_ylo", "search_yhi"])
        for r in rows_sorted:
            w.writerow(list(r))

    print("Wrote:", out_csv)

    with fits.open(arc1d_fits) as h:
        slits_present = list_slits(h)
        shift_map = {r[0]: int(r[2]) for r in rows_sorted}

        ny = h[slits_present[0]].data.shape[1]
        stack = np.full((len(slits_present), ny), np.nan, dtype=np.float32)

        for i, s in enumerate(slits_present):
            if s not in shift_map:
                continue
            f1d = h[s].data[0].astype(np.float32)
            sh = shift_map[s]
            stack[i] = np.roll(f1d, -sh)

        ph = fits.PrimaryHDU(stack)
        ph.header["STAGE"] = ("07d", "Pipeline stage")
        ph.header["TRACESET"] = (trace_set, "EVEN/ODD")
        ph.header["ARC1D"] = (arc1d_fits.name, "Source 1D arc MEF")
        ph.header["REFSLIT"] = (refslit, "Reference slit")
        ph.header["NSLITS"] = (stack.shape[0], "Number of slit rows in stack")
        ph.header["NY"] = (stack.shape[1], "Length of 1D spectra (Y pixels)")
        ph.header["SLITS"] = (",".join(slits_present[:20]) + ("..." if len(slits_present) > 20 else ""),
                              "First slits in stack order")

        out_stack_fits.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList([ph]).writeto(out_stack_fits, overwrite=True)
        print("Wrote:", out_stack_fits)

    if args.make_plots:
        slits = [r[0] for r in rows_sorted]
        bry = np.array([r[1] for r in rows_sorted], dtype=float)
        shift = np.array([r[2] for r in rows_sorted], dtype=float)

        with fits.open(arc1d_fits) as h:
            slits_present = list_slits(h)
            shift_map = {r[0]: int(r[2]) for r in rows_sorted}

            plt.figure(figsize=(10, 6))
            for s in slits_present:
                f1d = h[s].data[0].astype(float)
                good = np.isfinite(f1d)
                if good.sum() < 10:
                    continue
                scale = np.nanpercentile(f1d[good], 99)
                if scale == 0:
                    scale = 1.0
                plt.plot(f1d / scale, linewidth=0.8, alpha=0.7)
            plt.title("Step07d QC: Raw extracted arc spectra (normalized)")
            plt.xlabel("Detector Y (px)")
            plt.ylabel("Flux (norm)")
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 6))
            for s in slits_present:
                if s not in shift_map:
                    continue
                f1d = h[s].data[0].astype(float)
                good = np.isfinite(f1d)
                if good.sum() < 10:
                    continue
                scale = np.nanpercentile(f1d[good], 99)
                if scale == 0:
                    scale = 1.0
                sh = shift_map[s]
                f_shift = np.roll(f1d, -sh)
                plt.plot(f_shift / scale, linewidth=0.8, alpha=0.7)
            plt.title(f"Step07d QC: Aligned spectra by bright-line SHIFT (ref={refslit})")
            plt.xlabel("Detector Y (px) (rolled)")
            plt.ylabel("Flux (norm)")
            plt.tight_layout()
            plt.show()

        x = np.array([slit_num(s) for s in slits], dtype=int)

        plt.figure(figsize=(10, 4))
        plt.plot(x, bry, marker="o", linewidth=1)
        plt.title("BRY (bright-line Y position) vs slit number")
        plt.xlabel("Slit number")
        plt.ylabel("BRY (detector Y)")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(x, shift, marker="o", linewidth=1)
        plt.title(f"SHIFT vs slit number (relative to {refslit})")
        plt.xlabel("Slit number")
        plt.ylabel("SHIFT (px)")
        plt.axhline(0, linewidth=1)
        plt.tight_layout()
        plt.show()

        print("Step07d complete. Inspect the aligned overlay plot: lines should stack tightly.")


if __name__ == "__main__":
    main()
