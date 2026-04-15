#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QC Step08a — detector-frame slit montage.

Creates a montage of detector-frame slit images with the same tile ordering
and general layout as the Step08 TRACECOORDS montage, but using the curved
detector-frame slit cutouts instead.

This is intended as a visual / presentation companion to the TRACECOORDS QC,
not as the extraction authority.

Usage
-----
PYTHONPATH=. python qc/step08/qc_step08a_detector_montage.py --set EVEN
PYTHONPATH=. python qc/step08/qc_step08a_detector_montage.py --set ODD
PYTHONPATH=. python qc/step08/qc_step08a_detector_montage.py --set ODD --slit-min 21 --slit-max 41
PYTHONPATH=. python qc/step08/qc_step08a_detector_montage.py --set EVEN --pattern '*_detector*.fits'

Notes
-----
- The script searches for a detector-frame slit MEF in config.ST06_SCIENCE by
  default, with the requested parity tag in the filename.
- It expects one IMAGE extension per slit named SLIT###.
- It does not require Step08 products.
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


def candidate_patterns(set_tag: str, user_pattern: str | None = None) -> list[str]:
    tag = set_tag.upper()
    if user_pattern:
        return [user_pattern]

    return [
        f"*_{tag}_detector*.fits",
        f"*_{tag}_science*.fits",
        f"*_{tag}_slit*.fits",
        f"*_{tag}*.fits",
    ]


def is_slit_image_hdu(hdu) -> bool:
    name = (hdu.name or "").upper()
    return name.startswith("SLIT") and getattr(hdu, "data", None) is not None and np.asarray(hdu.data).ndim == 2


def pick_detector_mef(set_tag: str, user_pattern: str | None = None) -> Path:
    base = Path(config.ST06_SCIENCE)
    hits = []
    for pat in candidate_patterns(set_tag, user_pattern):
        found = sorted(glob(str(base / pat)))
        found = [p for p in found if "tracecoords" not in Path(p).name.lower()]
        if found:
            hits.extend(found)
            break

    if not hits:
        searched = [str(base / p) for p in candidate_patterns(set_tag, user_pattern)]
        raise FileNotFoundError(
            "No detector-frame slit MEF found for "
            f"{set_tag}. Tried patterns:\n  " + "\n  ".join(searched)
        )

    return Path(hits[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description="QC Step08a detector-frame slit montage")
    ap.add_argument("--set", required=True, choices=["EVEN", "ODD", "even", "odd"])
    ap.add_argument("--file", default=None, help="Explicit detector-frame slit FITS")
    ap.add_argument("--pattern", default=None, help="Optional glob pattern under config.ST06_SCIENCE")
    ap.add_argument("--slit-min", type=int, default=None)
    ap.add_argument("--slit-max", type=int, default=None)
    ap.add_argument("--ncols", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--outdir", default=None, help="Optional output directory")
    args = ap.parse_args()

    set_tag = args.set.upper()

    infile = Path(args.file) if args.file else pick_detector_mef(set_tag, args.pattern)
    if not infile.exists():
        raise FileNotFoundError(infile)

    outdir = Path(args.outdir) if args.outdir else Path(config.ST08_EXTRACT1D) / "qc_step08"
    outdir.mkdir(parents=True, exist_ok=True)

    slit_names: list[str] = []
    imgs: dict[str, np.ndarray] = {}

    with fits.open(infile) as hdul:
        for h in hdul[1:]:
            nm = (h.name or "").upper()
            if not is_slit_image_hdu(h):
                continue
            s = slit_num(nm)
            if args.slit_min is not None and s < args.slit_min:
                continue
            if args.slit_max is not None and s > args.slit_max:
                continue
            slit_names.append(nm)
            imgs[nm] = np.asarray(h.data, float)

    slit_names = sorted(set(slit_names), key=slit_num)
    if not slit_names:
        raise RuntimeError(f"No SLIT### image extensions found in {infile}")

    ncols = max(1, int(args.ncols))
    nrows = math.ceil(len(slit_names) / ncols)

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

    stem = f"{set_tag.lower()}_detector_montage"
    outpng = outdir / f"{stem}.png"

    fig.suptitle(stem, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpng, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Input : {infile}")
    print(f"Wrote : {outpng}")


if __name__ == "__main__":
    main()
