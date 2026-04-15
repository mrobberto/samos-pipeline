#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QC Step08a — brightest-block cross-dispersion profile montage.

For each slit in TRACECOORDS:
- split the slit into blocks along Y (dispersion), default 20 rows
- collapse each block in X (cross-dispersion) with a robust median
- identify the block with the strongest compact peak
- plot that brightest block profile in a mosaic

Overplots:
- cyan dashed: slit-envelope center (from TRACECOORDS finite footprint)
- yellow: slit-envelope edges
- red solid: peak position of brightest block
- optional red dotted: Step08 ridgeline position at the block center
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
    fin = np.isfinite(img)
    ny, nx = img.shape
    xc = np.full(ny, np.nan, float)
    xl = np.full(ny, np.nan, float)
    xr = np.full(ny, np.nan, float)
    for y in range(ny):
        xs = np.where(fin[y])[0]
        if xs.size == 0:
            continue
        xl[y] = float(xs[0]) - 0.5
        xr[y] = float(xs[-1]) + 0.5
        xc[y] = 0.5 * (xl[y] + xr[y])
    return xc, xl, xr


def build_blocks(ny: int, block_rows: int):
    edges = list(range(0, ny, block_rows))
    if not edges or edges[-1] != ny:
        edges.append(ny)
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def robust_profile(block: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        prof = np.nanmedian(block, axis=0)
    return np.asarray(prof, float)


def profile_score(profile: np.ndarray) -> tuple[float, float, float]:
    p = np.asarray(profile, float)
    if not np.isfinite(p).any():
        return np.nan, np.nan, np.nan
    base = np.nanmedian(p)
    q = p - base
    q[~np.isfinite(q)] = 0.0
    q[q < 0] = 0.0
    if np.sum(q) <= 0:
        return 0.0, np.nan, 0.0
    x = np.arange(p.size, dtype=float)
    peak_x = float((x * q).sum() / q.sum())
    contrast = float(np.nanmax(q)) if q.size else 0.0
    var = float(((x - peak_x) ** 2 * q).sum() / q.sum()) if q.sum() > 0 else np.inf
    score = contrast / max(np.sqrt(var), 1.0)
    return score, peak_x, contrast


def interp_nan(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    x = np.arange(y.size)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return y
    out = y.copy()
    out[~ok] = np.interp(x[~ok], x[ok], y[ok])
    return out

def measure_fwhm(profile):
    p = np.asarray(profile, float)
    if not np.isfinite(p).any():
        return np.nan, np.nan, np.nan, np.nan

    peak_idx = np.nanargmax(p)
    peak_val = p[peak_idx]
    if not np.isfinite(peak_val) or peak_val <= 0:
        return np.nan, np.nan, np.nan, np.nan

    half = 0.5 * peak_val
    x = np.arange(len(p), dtype=float)

    # left crossing
    xl = np.nan
    for i in range(peak_idx - 1, -1, -1):
        if np.isfinite(p[i]) and np.isfinite(p[i + 1]) and p[i] <= half <= p[i + 1]:
            denom = (p[i + 1] - p[i])
            t = (half - p[i]) / denom if denom != 0 else 0.0
            xl = x[i] + t
            break

    # right crossing
    xr = np.nan
    for i in range(peak_idx, len(p) - 1):
        if np.isfinite(p[i]) and np.isfinite(p[i + 1]) and p[i] >= half >= p[i + 1]:
            denom = (p[i + 1] - p[i])
            t = (half - p[i]) / denom if denom != 0 else 0.0
            xr = x[i] + t
            break

    if not np.isfinite(xl) or not np.isfinite(xr) or xr <= xl:
        return peak_idx, peak_val, np.nan, np.nan

    return peak_idx, peak_val, xl, xr

def main() -> None:
    ap = argparse.ArgumentParser(description="QC Step08a brightest-block cross-dispersion profile montage")
    ap.add_argument("--set", required=True, choices=["EVEN", "ODD", "even", "odd"])
    ap.add_argument("--file06", default=None, help="Explicit Step06 TRACECOORDS FITS")
    ap.add_argument("--file08", default=None, help="Explicit Step08a extract FITS")
    ap.add_argument("--block-rows", type=int, default=20, help="Rows per Y block")
    ap.add_argument("--slit-min", type=int, default=None)
    ap.add_argument("--slit-max", type=int, default=None)
    ap.add_argument("--ncols", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--no-ridge-mark", action="store_true", help="Do not mark Step08 ridgeline at the best block center")
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
        figsize=(3.2 * ncols, 2.8 * nrows),
        squeeze=False,
    )

    for ax in axes.ravel():
        ax.set_visible(False)

    summaries = []

    for ax, slit in zip(axes.ravel(), slit_names):
        ax.set_visible(True)

        img = imgs[slit]
        tab, hdr08 = tabs[slit]
        ny, nx = img.shape
        x = np.arange(nx, dtype=float)

        names = [n.upper() for n in tab.names]
        if "YPIX" in names:
            ypix = np.asarray(tab["YPIX"], float)
        elif "YYPIX" in names:
            ypix = np.asarray(tab["YYPIX"], float)
        else:
            ypix = np.arange(ny, dtype=float)

        x0 = np.asarray(tab["X0"], float) if "X0" in names else np.full_like(ypix, np.nan, dtype=float)
        x0i = interp_nan(x0)

        xc_env, xl_env, xr_env = envelope_from_tracecoords(img)
        xci = interp_nan(xc_env)
        xli = interp_nan(xl_env)
        xri = interp_nan(xr_env)

        best = None
        for y0, y1 in build_blocks(ny, int(args.block_rows)):
            block = img[y0:y1, :]
            prof = robust_profile(block)
            score, peak_x, contrast = profile_score(prof)
            if not np.isfinite(score):
                continue
            ymid = 0.5 * (y0 + y1 - 1)
            item = {
                "y0": y0, "y1": y1, "ymid": ymid,
                "profile": prof,
                "score": score,
                "peak_x": peak_x,
                "contrast": contrast,
            }
            if best is None or item["score"] > best["score"]:
                best = item

        if best is None:
            ax.set_title(slit, fontsize=10)
            ax.text(0.5, 0.5, "no valid block", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        prof = best["profile"]
        base = np.nanmedian(prof[np.isfinite(prof)]) if np.isfinite(prof).any() else 0.0
        prof0 = prof - base

        ymid_idx = int(np.clip(round(best["ymid"]), 0, ny - 1))
        xc0 = xci[ymid_idx] if np.isfinite(xci[ymid_idx]) else np.nan
        xl0 = xli[ymid_idx] if np.isfinite(xli[ymid_idx]) else np.nan
        xr0 = xri[ymid_idx] if np.isfinite(xri[ymid_idx]) else np.nan
        ridge0 = x0i[ymid_idx] if np.isfinite(x0i[ymid_idx]) else np.nan

        ax.plot(x, prof0, color="black", lw=1.2, zorder=2)
        ax.axhline(0.0, color="0.7", lw=0.6, zorder=1)

        if np.isfinite(xl0):
            ax.axvline(xl0, color="yellow", lw=1.0, zorder=3)
        if np.isfinite(xr0):
            ax.axvline(xr0, color="yellow", lw=1.0, zorder=3)
        if np.isfinite(xc0):
            ax.axvline(xc0, color="cyan", lw=1.3, ls="--", zorder=4)

        if np.isfinite(best["peak_x"]):
            ax.axvline(best["peak_x"], color="red", lw=1.2, zorder=5)

        if (not args.no_ridge_mark) and np.isfinite(ridge0):
            ax.axvline(ridge0, color="red", lw=1.0, ls=":", alpha=0.8, zorder=4)

        # --- FWHM measurement ---
        peak_idx, peak_val, xh_left, xh_right = measure_fwhm(prof0)
        
        fwhm = (
            xh_right - xh_left
            if np.isfinite(xh_left) and np.isfinite(xh_right)
            else np.nan
        )
       
        # --- rejection logic ---
        reject = False
        
        if not np.isfinite(fwhm):
            reject = True
        elif np.isfinite(xl0) and np.isfinite(xr0):
            if xh_left < xl0 or xh_right > xr0:
                reject = True
        
        # Plot FWHM interval (magenta)        
        if np.isfinite(xh_left) and np.isfinite(xh_right):
            ax.axvline(xh_left, color="magenta", lw=1.0, ls="--", zorder=4)
            ax.axvline(xh_right, color="magenta", lw=1.0, ls="--", zorder=4)                
            
        # Add BIG red X for rejected slits
        if reject:
            ax.text(
                0.5, 0.5, "✕",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=46,
                color="red",
                alpha=0.35,
                weight="bold"
            )            


        dxc = best["peak_x"] - xc0 if (np.isfinite(best["peak_x"]) and np.isfinite(xc0)) else np.nan
        title = f"{slit}  y={best['y0']}-{best['y1']-1}"
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        if np.isfinite(dxc):
            ax.text(
                0.03, 0.92,
                f"Δx={dxc:+.1f}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.8)
            )

        summaries.append((slit, best["score"], best["peak_x"], xc0, dxc, best["contrast"]))
        
        # Annotate FWHM
        label = ""
        if np.isfinite(dxc):
            label += f"Δx={dxc:+.1f}  "
        if np.isfinite(fwhm):
            label += f"FWHM={fwhm:.1f}"
        
        if label:
            ax.text(
                0.03, 0.92,
                label,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.8)
            )
        
    stem = f"{set_tag.lower()}_crossdisp_brightest_block_montage"
    outpng = outdir / f"{stem}.png"

    fig.suptitle(stem, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpng, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"TRACECOORDS : {file06}")
    print(f"STEP08a     : {file08}")
    print(f"Wrote       : {outpng}")
    print(f"block_rows  : {args.block_rows}")
    print("Top 10 by block score:")
    for row in sorted(summaries, key=lambda t: (np.nan_to_num(t[1], nan=-np.inf)), reverse=True)[:10]:
        slit, score, peak_x, xc0, dxc, contrast = row
        print(f"  {slit}: score={score:.3f} peak_x={peak_x:.2f} center={xc0:.2f} Δx={dxc:+.2f} contrast={contrast:.3f}")


if __name__ == "__main__":
    main()
