#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
QC for Step09 preferred products across all slits in a single PDF.

This script does NOT recompute anything.
It reads each per-slit Step09 directory, loads the selected preferred FITS,
extracts the matching slit HDU, and plots:
1) original spectrum (OBJ_PRESKY)
2) cleaned spectrum (STELLAR_FINAL or STELLAR_P1)
3) removed flux = original - cleaned

One PDF page per slit.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
from astropy.table import Table


def robust_mask(*arrays):
    m = np.ones_like(np.asarray(arrays[0]), dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


def list_slit_dirs(root: Path):
    out = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.upper().startswith("SLIT"):
            out.append(p)
    return out


def load_selection_label(slit_dir: Path):
    sel = slit_dir / "step09_selection.txt"
    label = "UNKNOWN"
    if sel.exists():
        text = sel.read_text(errors="ignore").splitlines()
        for line in text:
            if line.startswith("PREFERRED ="):
                label = line.split("=", 1)[1].strip()
                break
    return label


def load_one_slit(slit_dir: Path):
    slit = slit_dir.name.upper()
    fits_path = slit_dir / "step09_preferred.fits"
    if not fits_path.exists():
        raise FileNotFoundError(f"{slit}: missing {fits_path.name}")

    with fits.open(fits_path) as hdul:
        if slit not in hdul:
            raise KeyError(f"{slit}: HDU not found in {fits_path.name}")
        tab = Table(hdul[slit].data)

    if "LAMBDA_NM" not in tab.colnames or "OBJ_PRESKY" not in tab.colnames:
        raise KeyError(f"{slit}: missing LAMBDA_NM or OBJ_PRESKY")

    lam = np.asarray(tab["LAMBDA_NM"], float)
    orig = np.asarray(tab["OBJ_PRESKY"], float)

    if "STELLAR_FINAL" in tab.colnames:
        clean = np.asarray(tab["STELLAR_FINAL"], float)
        clean_label = "STELLAR_FINAL"
    elif "STELLAR_P1" in tab.colnames:
        clean = np.asarray(tab["STELLAR_P1"], float)
        clean_label = "STELLAR_P1"
    else:
        raise KeyError(f"{slit}: no STELLAR_FINAL or STELLAR_P1 column")

    removed = orig - clean
    preferred = load_selection_label(slit_dir)
    return slit, lam, orig, clean, removed, clean_label, preferred


def finite_ylim(*arrays, qlo=0.5, qhi=99.5):
    vals = []
    for a in arrays:
        a = np.asarray(a, float)
        m = np.isfinite(a)
        if np.any(m):
            vals.append(a[m])
    if not vals:
        return (-1.0, 1.0)
    x = np.concatenate(vals)
    return np.nanpercentile(x, qlo), np.nanpercentile(x, qhi)


def plot_page(slit, lam, orig, clean, removed, clean_label, preferred):
    m = robust_mask(lam, orig, clean)

    fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(f"Step09 preferred QC - {slit}   ({preferred})", fontsize=15)

    ax = axs[0]
    ax.plot(lam[m], orig[m], color="0.7", lw=1.0, label="OBJ_PRESKY")
    ax.plot(lam[m], clean[m], color="tab:green", lw=1.4, label=clean_label)
    ax.set_ylabel("Signal")
    ax.set_ylim(*finite_ylim(orig, clean))
    ax.legend(loc="upper left", fontsize=9)

    m2 = robust_mask(lam, removed)
    ax2 = axs[1]
    ax2.plot(lam[m2], removed[m2], color="tab:red", lw=1.0, label="removed = original - cleaned")
    ax2.axhline(0.0, color="0.4", lw=0.8, ls="--")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Removed flux")
    ax2.set_ylim(*finite_ylim(removed))
    ax2.legend(loc="upper left", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def parse_args():
    p = argparse.ArgumentParser(description="QC all preferred Step09 slit products into one PDF")
    p.add_argument("--root", type=Path, required=True,
                   help="Root directory containing SLITxxx subdirectories")
    p.add_argument("--slits", nargs="*", default=None,
                   help="Optional subset, e.g. SLIT006 SLIT033")
    p.add_argument("--out-pdf", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Optional summary CSV")
    return p.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    out_pdf = args.out_pdf.resolve()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    if args.slits:
        slit_dirs = [root / s.strip().upper() for s in args.slits]
    else:
        slit_dirs = list_slit_dirs(root)

    rows = []
    failures = []

    with PdfPages(out_pdf) as pdf:
        for slit_dir in slit_dirs:
            slit = slit_dir.name.upper()
            print("QC:", slit)
            try:
                slit, lam, orig, clean, removed, clean_label, preferred = load_one_slit(slit_dir)
                fig = plot_page(slit, lam, orig, clean, removed, clean_label, preferred)
                pdf.savefig(fig)
                plt.close(fig)

                rows.append({
                    "slit": slit,
                    "preferred": preferred,
                    "clean_label": clean_label,
                    "n_finite_original": int(np.isfinite(orig).sum()),
                    "n_finite_cleaned": int(np.isfinite(clean).sum()),
                })
            except Exception as e:
                print("FAIL:", slit, repr(e))
                failures.append((slit, repr(e)))

    if args.out_csv:
        import csv
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["slit", "preferred", "clean_label", "n_finite_original", "n_finite_cleaned"])
            w.writeheader()
            for row in rows:
                w.writerow(row)

    print("WROTE", out_pdf)
    print("SUCCESS =", len(rows))
    print("FAILURES =", len(failures))
    if failures:
        print("Failed slits:")
        for slit, err in failures:
            print(" ", slit, err)


if __name__ == "__main__":
    main()
