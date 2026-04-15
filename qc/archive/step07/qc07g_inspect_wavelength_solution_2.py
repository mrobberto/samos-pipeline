#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC07g — inspect saved wavelength-solution products (read-only).

Reads the existing Step07g outputs and checks:
  1. master wavelength-solution metadata and peak matches
  2. residual histogram from saved PEAK_MATCHES
  3. selected slit spectra overplotted in wavelength space using the saved
     per-slit WVC* coefficients
  4. basic per-slit wavelength coverage summary

Inputs
------
- arc_master_wavesol.fits
- arc_wavesol_per_slit.fits
- Step07c arc1d MEFs:
    ArcDiff_*_pixflatcorr_clipped_1D_slitid_EVEN.fits
    ArcDiff_*_pixflatcorr_clipped_1D_slitid_ODD.fits

Run
---
    PYTHONPATH=. python qc/step07/qc07g_inspect_wavelength_solution.py
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


YWIN0 = int(config.WAVECAL_YWIN0)
FIRSTLEN = int(config.WAVECAL_FIRSTLEN)

QC_SLITS_DEFAULT = ["SLIT000", "SLIT024", "SLIT052", "SLIT001", "SLIT029", "SLIT051"]
QC_XLIM_DEFAULT = (740, 830)


def default_arc1d_path(trace_set: str) -> Path:
    wavecal_dir = Path(config.ST07_WAVECAL).expanduser()
    if hasattr(config, "MASTER_ARC_DIFF"):
        stem = Path(config.MASTER_ARC_DIFF).stem
        p = wavecal_dir / f"{stem}_pixflatcorr_clipped_1D_slitid_{trace_set}.fits"
        if p.exists():
            return p
    hits = sorted(wavecal_dir.glob(f"*_1D_slitid_{trace_set}.fits"))
    if hits:
        return hits[-1]
    return wavecal_dir / f"ArcDiff_pixflatcorr_clipped_1D_slitid_{trace_set}.fits"


def load_arc1d_flux_map(*fits_paths):
    arc1d = {}
    for fp in fits_paths:
        if fp is None or not fp.exists():
            continue
        with fits.open(fp) as h:
            for ext in h[1:]:
                name = str(ext.header.get("EXTNAME", "")).strip().upper()
                if name.startswith("SLIT"):
                    arc1d[name] = ext.data[0].astype(float)
    return arc1d


def read_poly_from_header(hdr, prefix="WVC"):
    coeffs = []
    i = 0
    while f"{prefix}{i}" in hdr:
        coeffs.append(float(hdr[f"{prefix}{i}"]))
        i += 1
    return np.array(coeffs, float) if coeffs else None


def eval_poly_low_to_high(coeffs, x):
    x = np.asarray(x, float)
    y = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coeffs):
        y += c * x**i
    return y


def norm_flux(f):
    f = np.asarray(f, float)
    m = np.isfinite(f)
    if not np.any(m):
        return f
    s = np.nanpercentile(f[m], 99)
    if not np.isfinite(s) or s == 0:
        s = 1.0
    return f / s


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
                    help="Saved master wavesol FITS. Default: ST07_WAVECAL/arc_master_wavesol.fits")
    ap.add_argument("--global-sol", type=str, default=None,
                    help="Saved per-slit wavesol FITS. Default: ST07_WAVECAL/arc_wavesol_per_slit.fits")
    ap.add_argument("--arc1d-even", type=str, default=None,
                    help="Override Step07c EVEN arc1d MEF")
    ap.add_argument("--arc1d-odd", type=str, default=None,
                    help="Override Step07c ODD arc1d MEF")
    ap.add_argument("--save-prefix", type=str, default=None,
                    help="Optional prefix for saving PNGs. If omitted, plots are shown only.")
    args = ap.parse_args(argv)

    wavecal_dir = Path(config.ST07_WAVECAL).expanduser()
    master_fits = Path(args.master).expanduser() if args.master else (wavecal_dir / "arc_master_wavesol.fits")
    global_fits = Path(args.global_sol).expanduser() if args.global_sol else (wavecal_dir / "arc_wavesol_per_slit.fits")
    arc1d_even = Path(args.arc1d_even).expanduser() if args.arc1d_even else default_arc1d_path("EVEN")
    arc1d_odd = Path(args.arc1d_odd).expanduser() if args.arc1d_odd else default_arc1d_path("ODD")

    if not master_fits.exists():
        raise FileNotFoundError(master_fits)
    if not global_fits.exists():
        raise FileNotFoundError(global_fits)
    if not arc1d_even.exists():
        raise FileNotFoundError(arc1d_even)
    if not arc1d_odd.exists():
        raise FileNotFoundError(arc1d_odd)

    print("MASTER_WAVESOL =", master_fits)
    print("GLOBAL_WAVESOL =", global_fits)
    print("ARC1D_EVEN     =", arc1d_even)
    print("ARC1D_ODD      =", arc1d_odd)

    with fits.open(master_fits) as h:
        mhdr = h[0].header
        peak_matches = h["PEAK_MATCHES"].data if "PEAK_MATCHES" in h else None

    with fits.open(global_fits) as h:
        ghdr = h[0].header
        slit_exts = [ext for ext in h[1:] if str(ext.name).strip().upper().startswith("SLIT")]

    print("\nMASTER solution summary")
    for k in ("YWIN0", "FIRSTLEN", "ORDER", "NMATCH", "RMSNM", "ANCH1NM", "ANCH2NM", "ANCHY1", "ANCHY2", "PROPCONV"):
        if k in mhdr:
            print(f"  {k} = {mhdr[k]}")
    coeffs_master = read_poly_from_header(mhdr, "WVC")
    if coeffs_master is not None:
        print("  WVC coeffs (low->high):", ", ".join(f"{c:.8g}" for c in coeffs_master))

    if peak_matches is not None:
        resid_nm = np.asarray(peak_matches["RESID_NM"], float)
        good = np.asarray(peak_matches["GOOD"], bool)
        print("\nPEAK_MATCHES summary")
        print("  rows:", len(peak_matches))
        print("  good:", int(np.sum(good)))
        if np.any(good):
            print(f"  residual RMS (good): {np.sqrt(np.mean(resid_nm[good]**2)):.4f} nm")
            print(f"  residual min/max (good): {np.nanmin(resid_nm[good]):.4f} / {np.nanmax(resid_nm[good]):.4f} nm")

        plt.figure(figsize=(6, 4))
        plt.hist(resid_nm[good], bins=20)
        plt.title("QC07g: Saved master residuals")
        plt.xlabel("Residual (nm)")
        plt.ylabel("Count")
        plt.tight_layout()
        if args.save_prefix:
            plt.savefig(f"{args.save_prefix}_master_residuals.png", dpi=150, bbox_inches="tight")
            plt.close()
        else:
            if args.show_plots:
            plt.show()
        else:
            plt.close()

    arc1d = load_arc1d_flux_map(arc1d_even, arc1d_odd)

    ywin0 = int(mhdr.get("YWIN0", YWIN0_DEFAULT))
    firstlen = int(mhdr.get("FIRSTLEN", FIRSTLEN_DEFAULT))
    ywin = np.arange(firstlen, dtype=float)

    print("\nPer-slit solution summary")
    print("  slit extensions:", len(slit_exts))

    cover_rows = []
    plt.figure(figsize=(10, 5))
    plotted = 0
    for slit_name in QC_SLITS_DEFAULT:
        slit_name = slit_name.upper()
        if slit_name not in arc1d:
            continue
        try:
            with fits.open(global_fits) as h:
                ext = h[slit_name]
                hdr = ext.header
        except Exception:
            continue

        coeffs = read_poly_from_header(hdr, "WVC")
        if coeffs is None:
            continue

        f_full = arc1d[slit_name]
        f = norm_flux(f_full[ywin0:ywin0 + firstlen])
        lam = eval_poly_low_to_high(coeffs, ywin)

        plt.plot(lam, f, lw=1, label=slit_name)
        plotted += 1

    plt.title(f"QC07g: saved slit spectra in wavelength space (plotted {plotted})")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux (normalized)")
    plt.xlim(QC_XLIM_DEFAULT[0], QC_XLIM_DEFAULT[1])
    plt.legend(fontsize=9)
    plt.tight_layout()
    if args.save_prefix:
        plt.savefig(f"{args.save_prefix}_slit_spectra_wavelength.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        if args.show_plots:
            plt.show()
        else:
            plt.close()

    with fits.open(global_fits) as h:
        for ext in slit_exts[: min(10, len(slit_exts))]:
            hdr = ext.header
            slit = str(ext.name).strip().upper()
            coeffs = read_poly_from_header(hdr, "WVC")
            if coeffs is None:
                continue
            lam = eval_poly_low_to_high(coeffs, ywin)
            cover_rows.append((slit, float(np.nanmin(lam)), float(np.nanmax(lam)), hdr.get("SHIFT_P", None)))

    print("\nFirst per-slit wavelength ranges")
    for slit, lmin, lmax, shiftp in cover_rows:
        print(f"  {slit}: {lmin:.2f} .. {lmax:.2f} nm   SHIFT_P={shiftp}")

    # Optional global range summary over all slits
    all_ranges = []
    with fits.open(global_fits) as h:
        for ext in slit_exts:
            coeffs = read_poly_from_header(ext.header, "WVC")
            if coeffs is None:
                continue
            lam = eval_poly_low_to_high(coeffs, ywin)
            all_ranges.append((float(np.nanmin(lam)), float(np.nanmax(lam))))
    if all_ranges:
        mins = np.array([r[0] for r in all_ranges], float)
        maxs = np.array([r[1] for r in all_ranges], float)
        print("\nGlobal wavelength coverage summary")
        print(f"  min(lambda) range across slits: {np.nanmin(mins):.2f} .. {np.nanmax(mins):.2f} nm")
        print(f"  max(lambda) range across slits: {np.nanmin(maxs):.2f} .. {np.nanmax(maxs):.2f} nm")


if __name__ == "__main__":
    main()
