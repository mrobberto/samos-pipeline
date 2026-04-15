#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QC Step08a — ridge vs TRACECOORDS slit-envelope montage.

For each slit, show the TRACECOORDS slit image together with:
- slit-envelope center from the TRACECOORDS finite footprint (cyan dashed)
- slit-envelope left/right edges from the TRACECOORDS finite footprint (yellow)
- science ridgeline X0(y) from Step08a extraction (red solid)
- optional science aperture X0 ± WOBJ (red faint)

This is the core geometric QC for Step08a extraction in the actual rectified
frame used by the extractor.
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
    base = Path(config.ST06_SCIENCE)
    matches = sorted(glob(str(base / f"*_{set_tag}_tracecoords.fits")))
    if not matches:
        raise FileNotFoundError(f"No TRACECOORDS file found for {set_tag} in {base}")
    return Path(matches[-1])


def pick_extract(set_tag: str) -> Path:
    path = Path(config.ST08_EXTRACT1D) / f"trace_analysis_optimal_ridge_{set_tag.lower()}.fits"
    if not path.exists():
        raise FileNotFoundError(f"No Step08a1 trace-analysis file found for {set_tag}: {path}")
    return path


def get_tracecoords_images(hdul: fits.HDUList) -> dict[str, np.ndarray]:
    out = {}
    for h in hdul[1:]:
        nm = (h.name or "").upper()
        if nm.startswith("SLIT") and h.data is not None and np.asarray(h.data).ndim == 2:
            out[nm] = np.asarray(h.data, float)
    return out


def get_extract_tabs(hdul: fits.HDUList) -> dict[str, tuple]:
    out = {}
    for h in hdul[1:]:
        nm = (h.name or "").upper()
        if nm.startswith("SLIT") and h.data is not None:
            out[nm] = (h.data, h.header.copy())
    return out


def envelope_from_tracecoords(img: np.ndarray):
    """Return y, x_center, x_left, x_right from finite TRACECOORDS footprint."""
    fin = np.isfinite(img)
    ny, nx = img.shape
    y_list, xc_list, xl_list, xr_list = [], [], [], []
    for y in range(ny):
        xs = np.where(fin[y])[0]
        if xs.size == 0:
            continue
        xl = float(xs[0]) - 0.5
        xr = float(xs[-1]) + 0.5
        xc = 0.5 * (xl + xr)
        y_list.append(float(y))
        xc_list.append(xc)
        xl_list.append(xl)
        xr_list.append(xr)
    if not y_list:
        z = np.array([], dtype=float)
        return z, z, z, z
    return (
        np.asarray(y_list, float),
        np.asarray(xc_list, float),
        np.asarray(xl_list, float),
        np.asarray(xr_list, float),
    )


def smooth_curve(y: np.ndarray, x: np.ndarray, order: int = 3) -> np.ndarray:
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
        return np.asarray(np.polyval(coeff, y), float)
    except Exception:
        return x


def main() -> None:
    ap = argparse.ArgumentParser(description="QC Step08a ridge vs TRACECOORDS slit-envelope montage")
    ap.add_argument("--set", required=True, choices=["EVEN", "ODD", "even", "odd"])
    ap.add_argument("--file06", default=None, help="Explicit Step06 TRACECOORDS FITS")
    ap.add_argument("--file08", default=None, help="Explicit Step08a extract FITS")
    ap.add_argument("--slit-min", type=int, default=None)
    ap.add_argument("--slit-max", type=int, default=None)
    ap.add_argument("--ncols", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--no-aperture", action="store_true", help="Do not show X0 ± WOBJ")
    ap.add_argument("--curve-order", type=int, default=3, help="Polynomial order for displayed envelope curves")
    args = ap.parse_args()

    set_tag = args.set.upper()
    file06 = Path(args.file06) if args.file06 else pick_tracecoords(set_tag)
    file08 = Path(args.file08) if args.file08 else pick_extract(set_tag)

    outdir = Path(args.outdir) if args.outdir else Path(config.ST08_EXTRACT1D) / "qc_step08"
    outdir.mkdir(parents=True, exist_ok=True)

    with fits.open(file06) as h06, fits.open(file08) as h08:
        imgs = get_tracecoords_images(h06)
        tabs = get_extract_tabs(h08)

    slit_names = sorted(set(imgs.keys()) & set(tabs.keys()), key=slit_num)
    if args.slit_min is not None:
        slit_names = [s for s in slit_names if slit_num(s) >= args.slit_min]
    if args.slit_max is not None:
        slit_names = [s for s in slit_names if slit_num(s) <= args.slit_max]

    if not slit_names:
        raise RuntimeError("No common SLIT### entries found across TRACECOORDS and Step08a files")

    ncols = max(1, int(args.ncols))
    nrows = math.ceil(len(slit_names) / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.3 * ncols, 3.1 * nrows),
        squeeze=False,
    )

    for ax in axes.ravel():
        ax.set_visible(False)

    for ax, slit in zip(axes.ravel(), slit_names):
        ax.set_visible(True)

        img = imgs[slit]
        tab, hdr08 = tabs[slit]
        ny, nx = img.shape

        vmin, vmax = robust_limits(img)
        ax.imshow(img, origin="lower", aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(slit, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        names = [n.upper() for n in tab.names]
        if "YPIX" in names:
            ypix = np.asarray(tab["YPIX"], float)
        elif "YYPIX" in names:
            ypix = np.asarray(tab["YYPIX"], float)
        else:
            ypix = np.arange(ny, dtype=float)

        x0 = np.asarray(tab["X0"], float) if "X0" in names else np.full_like(ypix, np.nan, dtype=float)
        wobj = float(hdr08.get("WOBJ", 3.0))

        ye, xc, xl, xr = envelope_from_tracecoords(img)
        if ye.size:
            xc_s = smooth_curve(ye, xc, order=args.curve_order)
            xl_s = smooth_curve(ye, xl, order=args.curve_order)
            xr_s = smooth_curve(ye, xr, order=args.curve_order)
        # --- ridgeline ---
        ok = np.isfinite(ypix) & np.isfinite(x0)
        
        ax.plot(
            x0[ok], ypix[ok],
            color="red",
            lw=1.4,
            zorder=5
        )
        
        # aperture (lighter)
        if not args.no_aperture:
            ax.plot(
                x0[ok] - wobj, ypix[ok],
                color="red",
                lw=0.7,
                alpha=0.35,
                zorder=3
            )
            ax.plot(
                x0[ok] + wobj, ypix[ok],
                color="red",
                lw=0.7,
                alpha=0.35,
                zorder=3
            )
        
        # --- slit envelope / traces ---
        if ye.size:
            xc_s = smooth_curve(ye, xc, order=args.curve_order)
            xl_s = smooth_curve(ye, xl, order=args.curve_order)
            xr_s = smooth_curve(ye, xr, order=args.curve_order)
        
            # edges
            ax.plot(
                xl_s, ye,
                color="yellow",
                lw=1.2,
                zorder=6
            )
            ax.plot(
                xr_s, ye,
                color="yellow",
                lw=1.2,
                zorder=6
            )
        
            # centerline (most prominent)
            ax.plot(
                xc_s, ye,
                color="cyan",
                lw=1.7,
                ls="--",
                zorder=7
            )                
 
        if not args.no_aperture:
            ax.plot(x0[ok] - wobj, ypix[ok], color="red", lw=0.6, alpha=0.45)
            ax.plot(x0[ok] + wobj, ypix[ok], color="red", lw=0.6, alpha=0.45)

    stem = f"{set_tag.lower()}_ridge_vs_tracecoords_envelope_montage"
    outpng = outdir / f"{stem}.png"

    fig.suptitle(stem, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpng, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"TRACECOORDS : {file06}")
    print(f"STEP08a     : {file08}")
    print(f"Wrote       : {outpng}")


if __name__ == "__main__":
    main()
