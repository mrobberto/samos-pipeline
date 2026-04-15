#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:34:35 2026

@author: robberto
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
import config


def first_existing(paths):
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            return p
    return None


def robust_limits(img, p_lo=5, p_hi=99.5):
    v = np.asarray(img, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo = np.nanpercentile(v, p_lo)
    hi = np.nanpercentile(v, p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        med = np.nanmedian(v)
        return med - 1.0, med + 1.0
    return float(lo), float(hi)


def smooth1d(x, win=5):
    x = np.asarray(x, float)
    if win <= 1 or x.size < win:
        return x.copy()
    k = np.ones(win, float) / win
    return np.convolve(x, k, mode="same")


def eval_poly_from_header(hdr, prefix):
    coeffs = []
    i = 0
    while f"{prefix}{i}" in hdr:
        coeffs.append(float(hdr[f"{prefix}{i}"]))
        i += 1
    if not coeffs:
        raise KeyError(f"No coefficients found for prefix {prefix}")
    return np.array(coeffs, float)


def poly_eval(coeffs, y):
    # assume coefficients stored lowest order first: c0 + c1*y + c2*y^2 ...
    y = np.asarray(y, float)
    out = np.zeros_like(y, dtype=float)
    for i, c in enumerate(coeffs):
        out += c * y**i
    return out


def get_paths(traceset):
    st04 = Path(config.ST04_TRACES)
    st05 = Path(config.ST05_PIXFLAT)

    base = "Even_traces" if traceset == "EVEN" else "Odd_traces"
    suffix = traceset.lower()

    geom = st04 / f"{base}_geometry.fits"
    slitid = first_existing([
        st04 / f"{base}_slitid.fits",
        st04 / f"{base}_slitid_reg.fits",
    ])
    quartz = st05 / f"quartz_diff_{suffix}.fits"

    return geom, slitid, quartz


def get_slit_hdu(hdul, slit_name):
    slit_name = slit_name.upper()
    for h in hdul[1:]:
        nm = (h.header.get("EXTNAME") or h.name or "").strip().upper()
        if nm == slit_name:
            return h
    raise KeyError(f"Slit {slit_name} not found in geometry FITS")


def local_peak_measure(img, y, xpred, halfwin=12, smooth_win=5):
    ny, nx = img.shape
    if y < 0 or y >= ny or not np.isfinite(xpred):
        return np.nan

    x0 = int(np.floor(max(0, xpred - halfwin)))
    x1 = int(np.ceil(min(nx, xpred + halfwin + 1)))
    if x1 - x0 < 5:
        return np.nan

    prof = np.asarray(img[y, x0:x1], float)
    if not np.isfinite(prof).any():
        return np.nan

    # simple sideband background from edges of corridor
    nedge = max(2, min(4, (x1 - x0) // 4))
    edge_vals = np.r_[prof[:nedge], prof[-nedge:]]
    edge_vals = edge_vals[np.isfinite(edge_vals)]
    bkg = np.nanmedian(edge_vals) if edge_vals.size else 0.0

    s = smooth1d(np.nan_to_num(prof - bkg, nan=0.0), win=smooth_win)
    if np.nanmax(s) <= 0:
        return np.nan

    k = int(np.nanargmax(s))
    return float(x0 + k)


def main():
    ap = argparse.ArgumentParser(description="Diagnose one Step04 trace against quartz image")
    ap.add_argument("--traceset", required=True, choices=["EVEN", "ODD", "even", "odd"])
    ap.add_argument("--slit", required=True, help="e.g. SLIT018")
    ap.add_argument("--halfwin", type=int, default=12, help="local search half-width in X")
    ap.add_argument("--padx", type=int, default=20, help="extra cutout padding in X")
    ap.add_argument("--pady", type=int, default=30, help="extra cutout padding in Y")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    traceset = args.traceset.upper()
    slit = args.slit.upper()

    geom_path, slitid_path, quartz_path = get_paths(traceset)
    if not geom_path.exists():
        raise FileNotFoundError(geom_path)
    if slitid_path is None or not slitid_path.exists():
        raise FileNotFoundError(slitid_path)
    if not quartz_path.exists():
        raise FileNotFoundError(quartz_path)

    quartz = fits.getdata(quartz_path).astype(float)
    slitid = fits.getdata(slitid_path).astype(int)

    with fits.open(geom_path) as hdul:
        ghdu = get_slit_hdu(hdul, slit)
        hdr = ghdu.header

    # Read geometry coefficients
    pc = eval_poly_from_header(hdr, "PC")
    lc = eval_poly_from_header(hdr, "LC")
    rc = eval_poly_from_header(hdr, "RC")

    ymin = int(hdr.get("YMIN", 0))
    ymax = int(hdr.get("YMAX", quartz.shape[0] - 1))
    yy = np.arange(ymin, ymax + 1, dtype=float)

    xcen = poly_eval(pc, yy)
    xleft = poly_eval(lc, yy)
    xright = poly_eval(rc, yy)

    # If slitid map uses numeric IDs, try to get one from header; otherwise build cutout from geometry
    y0 = max(0, ymin - args.pady)
    y1 = min(quartz.shape[0], ymax + args.pady + 1)
    x0 = max(0, int(np.floor(np.nanmin(xleft))) - args.padx)
    x1 = min(quartz.shape[1], int(np.ceil(np.nanmax(xright))) + args.padx + 1)

    cut = quartz[y0:y1, x0:x1]

    # local peak remeasurement
    xpeak = np.full_like(yy, np.nan, dtype=float)
    for i, y in enumerate(yy.astype(int)):
        xpeak[i] = local_peak_measure(quartz, y, xcen[i], halfwin=args.halfwin)

    dx = xpeak - xcen

    outdir = Path(args.outdir) if args.outdir else (Path(config.ST04_TRACES) / f"qc_step04_diag_{traceset.lower()}")
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{slit}_diagnostic.png"
    txt = outdir / f"{slit}_diagnostic.txt"

    # choose sample rows for profile display
    frac_rows = [0.15, 0.50, 0.85]
    sample_rows = [int(ymin + f * (ymax - ymin)) for f in frac_rows]

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    vmin, vmax = robust_limits(cut, 5, 99.5)
    ax[0, 0].imshow(cut, origin="lower", aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
    ax[0, 0].plot(xcen - x0, yy - y0, color="cyan", lw=1.2, label="fit center")
    ax[0, 0].plot(xleft - x0, yy - y0, color="yellow", lw=0.8, label="fit left/right")
    ax[0, 0].plot(xright - x0, yy - y0, color="yellow", lw=0.8)
    ax[0, 0].plot(xpeak - x0, yy - y0, "r.", ms=2, label="local peak")
    ax[0, 0].set_title(f"{slit} quartz cutout + geometry")
    ax[0, 0].set_xlabel("X")
    ax[0, 0].set_ylabel("Y")
    ax[0, 0].legend(loc="best", fontsize=8)

    ax[0, 1].plot(xcen, yy, color="cyan", lw=1.2, label="fit center")
    ax[0, 1].plot(xpeak, yy, "r.", ms=2, label="local peak")
    ax[0, 1].set_title("Centerline comparison")
    ax[0, 1].set_xlabel("X")
    ax[0, 1].set_ylabel("Y")
    ax[0, 1].legend(loc="best", fontsize=8)

    ax[1, 0].plot(dx, yy, "k-", lw=0.8)
    ax[1, 0].axvline(0, color="gray", ls="--")
    ax[1, 0].set_title("Residual: local peak - fit center")
    ax[1, 0].set_xlabel("dx [pix]")
    ax[1, 0].set_ylabel("Y")

    for y in sample_rows:
        prof = quartz[y, x0:x1]
        ax[1, 1].plot(np.arange(x0, x1), prof, lw=0.9, label=f"y={y}")
        yf = y - ymin
        if 0 <= yf < len(xcen):
            ax[1, 1].axvline(xcen[int(yf)], color="cyan", alpha=0.5)
            if np.isfinite(xpeak[int(yf)]):
                ax[1, 1].axvline(xpeak[int(yf)], color="red", alpha=0.5)

    ax[1, 1].set_title("Sample row profiles")
    ax[1, 1].set_xlabel("X")
    ax[1, 1].set_ylabel("Quartz signal")
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(png, dpi=160)
    plt.close(fig)

    good = np.isfinite(dx)
    with txt.open("w") as f:
        f.write(f"Slit: {slit}\n")
        f.write(f"Geometry: {geom_path}\n")
        f.write(f"Quartz: {quartz_path}\n")
        f.write(f"Y range: {ymin}..{ymax}\n")
        f.write(f"N rows with peak measurement: {good.sum()}\n")
        if good.sum():
            f.write(f"Median dx = {np.nanmedian(dx):.3f} pix\n")
            f.write(f"RMS dx    = {np.nanstd(dx):.3f} pix\n")
            f.write(f"Min/max dx= {np.nanmin(dx):.3f}, {np.nanmax(dx):.3f} pix\n")

    print("[OK] wrote", png)
    print("[OK] wrote", txt)


if __name__ == "__main__":
    main()