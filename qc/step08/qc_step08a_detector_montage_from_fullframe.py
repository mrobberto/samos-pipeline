#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QC Step08a — detector-frame slit montage from full science image + slit-id mask.

Builds a slit-by-slit detector-frame mosaic with the same tile ordering idea as
the TRACECOORDS montage, but starting from the full detector-frame Step06 science
image and a 2D slit-id image.

This is a display/advertising companion to the TRACECOORDS QC. Extraction is
still performed in TRACECOORDS.

Usage
-----
PYTHONPATH=. python qc/step08/qc_step08a_detector_montage_from_fullframe.py --set EVEN
PYTHONPATH=. python qc/step08/qc_step08a_detector_montage_from_fullframe.py --set ODD

Optional explicit inputs:
PYTHONPATH=. python qc/step08/qc_step08a_detector_montage_from_fullframe.py \
    --set EVEN \
    --science /path/to/FinalScience_dolidze_ADUperS_pixflatcorr_clipped_EVEN.fits \
    --slitid  /path/to/Even_traces_slitid.fits

Range restriction:
PYTHONPATH=. python qc/step08/qc_step08a_detector_montage_from_fullframe.py \
    --set EVEN --slit-min 20 --slit-max 40
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


def get_candidate_dirs() -> list[Path]:
    dirs = []
    for attr in [
        "ST04_TRACES",
        "ST04_PIXFLAT",
        "ST05_PIXFLAT",
        "ST06_SCIENCE",
        "ST08_EXTRACT1D",
    ]:
        if hasattr(config, attr):
            try:
                dirs.append(Path(getattr(config, attr)))
            except Exception:
                pass
    for p in list(dirs):
        dirs.append(p.parent)
        dirs.append(p.parent.parent if p.parent != p else p)
    out = []
    seen = set()
    for d in dirs:
        if d is not None:
            s = str(d.resolve()) if d.exists() else str(d)
            if s not in seen:
                seen.add(s)
                out.append(d)
    return out


def pick_science(set_tag: str) -> Path:
    base = Path(config.ST06_SCIENCE)
    pats = [
        f"FinalScience*_clipped_{set_tag}.fits",
        f"FinalScience*_{set_tag}.fits",
        f"*_{set_tag}.fits",
    ]
    hits: list[Path] = []
    for pat in pats:
        found = [Path(p) for p in sorted(glob(str(base / pat)))]
        found = [p for p in found if "tracecoords" not in p.name.lower()]
        if found:
            hits = found
            break
    if not hits:
        raise FileNotFoundError(f"No detector-frame Step06 science file found for {set_tag} in {base}")
    return hits[-1]


def pick_slitid(set_tag: str) -> Path:
    parity = set_tag.capitalize()
    all_hits: list[Path] = []
    patterns = [
        f"*{parity}*slitid*.fits",
        f"*{parity}*SLITID*.fits",
        f"*{parity}*traces_slitid*.fits",
        f"*{parity}*slit_id*.fits",
    ]
    for d in get_candidate_dirs():
        for pat in patterns:
            all_hits.extend(Path(p) for p in sorted(glob(str(d / pat))))
        for child in sorted(d.glob("*")):
            if child.is_dir():
                for pat in patterns:
                    all_hits.extend(Path(p) for p in sorted(glob(str(child / pat))))
    uniq = []
    seen = set()
    for p in all_hits:
        if p.exists():
            s = str(p.resolve())
            if s not in seen:
                seen.add(s)
                uniq.append(p)
    if not uniq:
        raise FileNotFoundError(f"No slit-id FITS found for {set_tag} under candidate reduced directories")
    return uniq[-1]


def load_primary_2d(path: Path) -> np.ndarray:
    with fits.open(path) as hdul:
        if hdul[0].data is not None and np.asarray(hdul[0].data).ndim == 2:
            return np.asarray(hdul[0].data, float)
        for h in hdul[1:]:
            if h.data is not None and np.asarray(h.data).ndim == 2:
                return np.asarray(h.data, float)
    raise RuntimeError(f"No 2D image found in {path}")


def infer_present_slits(labels: np.ndarray) -> list[int]:
    vals = labels[np.isfinite(labels)]
    if vals.size == 0:
        return []
    u = np.unique(vals.astype(int))
    return sorted(int(v) for v in u if v >= 0)


def cutout_bbox(mask: np.ndarray, pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return 0, 1, 0, 1
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(shape[1], int(xs.max()) + pad + 1)
    return y0, y1, x0, x1


def compute_center_and_edges(mask: np.ndarray):
    """
    For each detector row with slit pixels, return the slit center and edges.
    """
    ny, nx = mask.shape
    y_list, xc_list, xl_list, xr_list = [], [], [], []
    for y in range(ny):
        xs = np.where(mask[y])[0]
        if xs.size == 0:
            continue
        xl = float(xs.min())
        xr = float(xs.max())
        xc = 0.5 * (xl + xr)
        y_list.append(float(y))
        xc_list.append(xc)
        xl_list.append(xl)
        xr_list.append(xr)
    if not y_list:
        return None, None, None, None
    return (
        np.asarray(y_list, float),
        np.asarray(xc_list, float),
        np.asarray(xl_list, float),
        np.asarray(xr_list, float),
    )


def smooth_curve(y: np.ndarray, x: np.ndarray, order: int = 3) -> np.ndarray:
    """
    Smooth x(y) for display using a low-order polynomial fit.
    Falls back to the raw curve when too few valid points are available.
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < max(5, order + 1):
        return x
    yy = y[ok]
    xx = x[ok]
    ord_use = max(1, min(int(order), len(xx) - 1))
    try:
        coeff = np.polyfit(yy, xx, ord_use)
        xfit = np.polyval(coeff, y)
        return np.asarray(xfit, float)
    except Exception:
        return x


def main() -> None:
    ap = argparse.ArgumentParser(description="QC Step08a detector montage from full frame")
    ap.add_argument("--set", required=True, choices=["EVEN", "ODD", "even", "odd"])
    ap.add_argument("--science", default=None, help="Explicit Step06 detector-frame science FITS")
    ap.add_argument("--slitid", default=None, help="Explicit slit-id FITS")
    ap.add_argument("--slit-min", type=int, default=None)
    ap.add_argument("--slit-max", type=int, default=None)
    ap.add_argument("--pad", type=int, default=8, help="Padding (pixels) around each slit bbox")
    ap.add_argument("--ncols", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--show-mask-outline", action="store_true",
                    help="Overlay the slit-id region contour in magenta")
    ap.add_argument("--no-lines", action="store_true",
                    help="Disable curved center/edge overlays")
    ap.add_argument("--curve-order", type=int, default=3,
                    help="Polynomial order for displayed center/edge curves")
    args = ap.parse_args()

    set_tag = args.set.upper()
    science = Path(args.science) if args.science else pick_science(set_tag)
    slitid = Path(args.slitid) if args.slitid else pick_slitid(set_tag)

    sci = load_primary_2d(science)
    lbl = load_primary_2d(slitid)

    if sci.shape != lbl.shape:
        raise RuntimeError(
            f"Science image shape {sci.shape} does not match slit-id shape {lbl.shape}"
        )

    slit_ids = infer_present_slits(lbl)
    if args.slit_min is not None:
        slit_ids = [s for s in slit_ids if s >= args.slit_min]
    if args.slit_max is not None:
        slit_ids = [s for s in slit_ids if s <= args.slit_max]
    if not slit_ids:
        raise RuntimeError("No slit IDs remain after filtering")

    outdir = Path(args.outdir) if args.outdir else Path(config.ST08_EXTRACT1D) / "qc_step08"
    outdir.mkdir(parents=True, exist_ok=True)

    ncols = max(1, int(args.ncols))
    nrows = math.ceil(len(slit_ids) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 3.0 * nrows),
        squeeze=False,
    )

    for ax in axes.ravel():
        ax.set_visible(False)

    for ax, sid in zip(axes.ravel(), slit_ids):
        ax.set_visible(True)
        mask = np.asarray(lbl == sid)
        y0, y1, x0, x1 = cutout_bbox(mask, int(args.pad), sci.shape)
        cut = np.asarray(sci[y0:y1, x0:x1], float)
        cut_mask = np.asarray(mask[y0:y1, x0:x1], bool)

        vmin, vmax = robust_limits(cut)
        ax.imshow(cut, origin="lower", aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"SLIT{sid:03d}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        if args.show_mask_outline and np.any(cut_mask):
            yy, xx = np.mgrid[:cut_mask.shape[0], :cut_mask.shape[1]]
            ax.contour(xx, yy, cut_mask.astype(float), levels=[0.5], colors=["magenta"], linewidths=0.6)

        if (not args.no_lines) and np.any(cut_mask):
            y, xc, xl, xr = compute_center_and_edges(cut_mask)
            if y is not None:
                xc_s = smooth_curve(y, xc, order=args.curve_order)
                xl_s = smooth_curve(y, xl, order=args.curve_order)
                xr_s = smooth_curve(y, xr, order=args.curve_order)
                ax.plot(xc_s, y, color="cyan", lw=1.0)
                ax.plot(xl_s, y, color="yellow", lw=0.8)
                ax.plot(xr_s, y, color="yellow", lw=0.8)

    stem = f"{set_tag.lower()}_detector_montage"
    outpng = outdir / f"{stem}.png"

    fig.suptitle(stem, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpng, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Science : {science}")
    print(f"Slit-id : {slitid}")
    print(f"Wrote   : {outpng}")


if __name__ == "__main__":
    main()
