#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QC Step08a — TRACECOORDS slit montage with optional extraction overlay.

Creates a montage of Step06 TRACECOORDS slit images, sorted by slit number,
optionally overplotting the Step08 ridge and object aperture.

Usage
-----
PYTHONPATH=. python qc/step08/qc_step08a_tracecoords_montage.py --set EVEN
PYTHONPATH=. python qc/step08/qc_step08a_tracecoords_montage.py --set ODD
PYTHONPATH=. python qc/step08/qc_step08a_tracecoords_montage.py --set ODD --no-overlay
PYTHONPATH=. python qc/step08/qc_step08a_tracecoords_montage.py --set EVEN --slit-min 20 --slit-max 40
"""

from __future__ import annotations

import argparse
import math
import re
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import config


def slit_num(name: str) -> int:
    m = re.match(r"SLIT(\d+)", str(name).upper())
    return int(m.group(1)) if m else 10**9


def robust_limits(img: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> tuple[float, float]:
    arr = np.asarray(img, float)
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return 0.0, 1.0
    lo = np.nanpercentile(v, p_lo)
    hi = np.nanpercentile(v, p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        med = np.nanmedian(v)
        sig = np.nanstd(v)
        if not np.isfinite(sig) or sig <= 0:
            sig = 1.0
        return med - 2 * sig, med + 4 * sig
    return float(lo), float(hi)


def pick_tracecoords(set_tag: str) -> Path:
    matches = sorted(glob(str(Path(config.ST06_SCIENCE) / f"*_{set_tag}_tracecoords.fits")))
    if not matches:
        raise FileNotFoundError(f"No TRACECOORDS file found for {set_tag}")
    return Path(matches[-1])


def pick_extract(set_tag: str) -> Path:
    path = Path(config.ST08_EXTRACT1D) / f"trace_analysis_optimal_ridge_{set_tag.lower()}.fits"
    if not path.exists():
        raise FileNotFoundError(f"No Step08a1 trace-analysis file found for {set_tag}: {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="QC Step08a TRACECOORDS montage")
    ap.add_argument("--set", required=True, choices=["EVEN", "ODD", "even", "odd"])
    ap.add_argument("--file06", default=None, help="Optional explicit Step06 TRACECOORDS FITS")
    ap.add_argument("--file08", default=None, help="Optional explicit Step08a extract FITS")
    ap.add_argument("--no-overlay", action="store_true", help="Do not overplot Step08 ridge/aperture")
    ap.add_argument("--slit-min", type=int, default=None)
    ap.add_argument("--slit-max", type=int, default=None)
    ap.add_argument("--ncols", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--outdir", default=None, help="Optional output directory")
    args = ap.parse_args()

    set_tag = args.set.upper()
    file06 = Path(args.file06) if args.file06 else pick_tracecoords(set_tag)
    file08 = None if args.no_overlay else (Path(args.file08) if args.file08 else pick_extract(set_tag))

    outdir = Path(args.outdir) if args.outdir else Path(config.ST08_EXTRACT1D) / "qc_step08"
    outdir.mkdir(parents=True, exist_ok=True)

    with fits.open(file06) as h06:
        slit_names = []
        imgs: dict[str, np.ndarray] = {}
        for h in h06[1:]:
            nm = (h.name or "").upper()
            if not nm.startswith("SLIT") or h.data is None:
                continue
            s = slit_num(nm)
            if args.slit_min is not None and s < args.slit_min:
                continue
            if args.slit_max is not None and s > args.slit_max:
                continue
            slit_names.append(nm)
            imgs[nm] = np.asarray(h.data, float)

    slit_names = sorted(slit_names, key=slit_num)
    if not slit_names:
        raise RuntimeError("No slits selected for montage.")

    tabs: dict[str, fits.FITS_rec] = {}
    hdrs08: dict[str, fits.Header] = {}
    if file08 is not None:
        with fits.open(file08) as h08:
            for h in h08[1:]:
                nm = (h.name or "").upper()
                if nm in slit_names and h.data is not None:
                    tabs[nm] = h.data
                    hdrs08[nm] = h.header.copy()

    n = len(slit_names)
    ncols = max(1, int(args.ncols))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 3.0 * nrows),
        squeeze=False,
    )

    for ax in axes.ravel():
        ax.set_visible(False)

    for ax, slit in zip(axes.ravel(), slit_names):
        ax.set_visible(True)

        img = imgs[slit]
        vmin, vmax = robust_limits(img)
        ax.imshow(img, origin="lower", aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(slit, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        if slit in tabs:
            tab = tabs[slit]
            names = [n.upper() for n in tab.names]

            if "YPIX" in names:
                ypix = np.asarray(tab["YPIX"], float)
            elif "YYPIX" in names:
                ypix = np.asarray(tab["YYPIX"], float)
            else:
                ypix = None

            x0 = np.asarray(tab["X0"], float) if "X0" in names else None

            if ypix is not None and x0 is not None:
                ok = np.isfinite(ypix) & np.isfinite(x0)
                ax.plot(x0[ok], ypix[ok], color="cyan", lw=1.1)

                wobj = float(hdrs08[slit].get("WOBJ", 3.0))
                ax.plot(x0[ok] - wobj, ypix[ok], color="yellow", lw=0.8, alpha=0.9)
                ax.plot(x0[ok] + wobj, ypix[ok], color="yellow", lw=0.8, alpha=0.9)

    stem = f"{set_tag.lower()}_tracecoords_montage"
    if file08 is not None:
        stem += "_overlay"

    outpng = outdir / f"{stem}.png"
    fig.suptitle(stem, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpng, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {outpng}")


if __name__ == "__main__":
    main()