#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qc_step06-overlay_Step04_trace_geometry_on_science _image.py

Purpose
-------
Visualize the Step04 quartz-derived trace geometry directly on the
corresponding Step06 science frame, in detector coordinates.

This QC is intended to diagnose quartz–science misalignment by showing
whether the fitted trace centerline and edges derived from the quartz
lamp still follow the actual science spectrum.

Inputs
------
- Step04 geometry:
    Even/Odd_traces_geometry.fits
- Step06 science frame:
    science_pixflatcorr_even.fits / science_pixflatcorr_odd.fits
    or another selected full-frame science product

Method
------
For each slit:
- read YMIN/YMAX and polynomial coefficients (PC*, LC*, RC*)
- evaluate centerline and edges in detector coordinates
- extract a geometry-centered cutout from the science image
- overplot:
    centerline (cyan)
    left/right edges (yellow)

Outputs
-------
Written under config.ST06_SCIENCE:

- qc_step06_overlay_even/
- qc_step06_overlay_odd/

Products include:
- one PNG per slit
- one montage PNG for the first N slits

Notes
-----
This script does not redetermine traces. It uses the Step04 geometry
exactly as stored, and checks its alignment against the science image.

run:
> PYTHONPATH=. python qc/step06/qc_step06c—overlay_Step04_trace_geometry_on_science_image.py --traceset EVEN
> PYTHONPATH=. python qc/step06/qc_step06c—overlay_Step04_trace_geometry_on_science_image.py --traceset ODD
"""

from __future__ import annotations

import argparse
import math
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


def robust_limits(img, p_lo=5.0, p_hi=99.5):
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


def eval_poly_from_header(hdr, prefix: str):
    coeffs = []
    i = 0
    while f"{prefix}{i}" in hdr:
        coeffs.append(float(hdr[f"{prefix}{i}"]))
        i += 1
    if not coeffs:
        return None
    return np.array(coeffs, float)


def poly_eval(coeffs, y):
    y = np.asarray(y, float)
    out = np.zeros_like(y, dtype=float)
    for i, c in enumerate(coeffs):
        out += c * y**i
    return out


def get_slit_hdus(hdul: fits.HDUList):
    out = []
    for h in hdul[1:]:
        nm = (h.header.get("EXTNAME") or h.name or "").strip().upper()
        if nm.startswith("SLIT"):
            out.append(h)
    return out


def pick_geometry_file(traceset: str) -> Path:
    st04 = Path(config.ST04_TRACES)
    base = "Even_traces" if traceset == "EVEN" else "Odd_traces"
    p = st04 / f"{base}_geometry.fits"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def pick_science_file(traceset: str, explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    st06 = Path(config.ST06_SCIENCE)
    suffix = traceset.lower()

    p = first_existing([
        st06 / f"science_pixflatcorr_{suffix}.fits",
        st06 / "FinalScience_dolidze_ADUperS.fits",
        *sorted(st06.glob("FinalScience*_ADUperS*.fits"), key=lambda q: q.stat().st_mtime, reverse=True),
    ])
    if p is None:
        raise FileNotFoundError(f"No science image found under {st06}")
    return p


def centered_cutout_from_geometry(img, hdr, xhalf=16, pady=20):
    ny, nx = img.shape

    ymin = int(hdr.get("YMIN", 0))
    ymax = int(hdr.get("YMAX", ny - 1))
    ymin = max(0, ymin)
    ymax = min(ny - 1, ymax)

    yy = np.arange(ymin, ymax + 1, dtype=float)

    pc = eval_poly_from_header(hdr, "PC")
    lc = eval_poly_from_header(hdr, "LC")
    rc = eval_poly_from_header(hdr, "RC")

    if pc is None:
        xref = float(hdr.get("XREF", nx / 2))
        xcen = np.full_like(yy, xref, dtype=float)
    else:
        xcen = poly_eval(pc, yy)

    xl = poly_eval(lc, yy) if lc is not None else None
    xr = poly_eval(rc, yy) if rc is not None else None

    if xl is not None and xr is not None:
        xmid = 0.5 * (xl + xr)
        width = np.nanmedian(xr - xl)
        if np.isfinite(width) and width > 2:
            xhalf = max(xhalf, int(np.ceil(0.6 * width)))
    else:
        xmid = xcen

    xcen_med = float(np.nanmedian(xmid)) if np.isfinite(xmid).any() else float(hdr.get("XREF", nx / 2))

    y0 = max(0, ymin - pady)
    y1 = min(ny, ymax + pady + 1)
    x0 = max(0, int(np.floor(xcen_med - xhalf)))
    x1 = min(nx, int(np.ceil(xcen_med + xhalf + 1)))

    cut = img[y0:y1, x0:x1].copy()

    overlay = {
        "yy": yy - y0,
        "xcen": xcen - x0,
        "xl": (xl - x0) if xl is not None else None,
        "xr": (xr - x0) if xr is not None else None,
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
    }
    return cut, overlay


def save_one(cut, overlay, outpng: Path, title: str, p_lo=5.0, p_hi=99.5):
    vmin, vmax = robust_limits(cut, p_lo=p_lo, p_hi=p_hi)

    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=150)
    ax.imshow(cut, origin="lower", aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)

    if overlay.get("xcen") is not None:
        ax.plot(overlay["xcen"], overlay["yy"], color="cyan", lw=1.0, label="center")
    if overlay.get("xl") is not None:
        ax.plot(overlay["xl"], overlay["yy"], color="yellow", lw=0.8, label="edges")
    if overlay.get("xr") is not None:
        ax.plot(overlay["xr"], overlay["yy"], color="yellow", lw=0.8)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X (detector)")
    ax.set_ylabel("Y (detector)")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(outpng)
    plt.close(fig)


def save_montage(images, titles, overlays, outpng: Path, ncols=6, p_lo=5.0, p_hi=99.5):
    if not images:
        return

    n = len(images)
    ncols = max(1, ncols)
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(ncols * 2.4, nrows * 2.3), dpi=170)
    for i, (img, ttl, ov) in enumerate(zip(images, titles, overlays), start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        vmin, vmax = robust_limits(img, p_lo=p_lo, p_hi=p_hi)
        ax.imshow(img, origin="lower", aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)

        if ov.get("xcen") is not None:
            ax.plot(ov["xcen"], ov["yy"], color="cyan", lw=0.6)
        if ov.get("xl") is not None:
            ax.plot(ov["xl"], ov["yy"], color="yellow", lw=0.4)
        if ov.get("xr") is not None:
            ax.plot(ov["xr"], ov["yy"], color="yellow", lw=0.4)

        ax.set_title(ttl, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(outpng.stem, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpng)
    plt.close(fig)

def slit_number(h):
    nm = (h.header.get("EXTNAME") or h.name or "").strip().upper()
    if nm.startswith("SLIT"):
        try:
            return int(nm.replace("SLIT", ""))
        except Exception:
            return 10**9
    return 10**9

def main():
    ap = argparse.ArgumentParser(description="Overlay Step04 trace geometry on Step06 science image")
    ap.add_argument("--traceset", required=True, choices=["EVEN", "ODD", "even", "odd"])
    ap.add_argument("--science", default=None, help="Optional explicit science full-frame FITS")
    ap.add_argument("--geometry", default=None, help="Optional explicit Step04 geometry FITS")
    ap.add_argument("--outdir", default=None, help="Optional output directory")
    ap.add_argument("--xhalf", type=int, default=16, help="Half-width of cutout in X")
    ap.add_argument("--pady", type=int, default=20, help="Padding in Y around YMIN/YMAX")
    ap.add_argument("--p-lo", type=float, default=5.0, help="Lower display percentile")
    ap.add_argument("--p-hi", type=float, default=99.5, help="Upper display percentile")
    ap.add_argument("--max-slits", type=int, default=9999, help="Max number of slits to export")
    ap.add_argument("--montage-n", type=int, default=24, help="How many slits in the montage")
    ap.add_argument("--montage-cols", type=int, default=6, help="Columns in montage")
    args = ap.parse_args()

    traceset = args.traceset.upper()

    geom_path = Path(args.geometry).expanduser() if args.geometry else pick_geometry_file(traceset)
    sci_path = pick_science_file(traceset, args.science)

    outroot = Path(args.outdir).expanduser() if args.outdir else (Path(config.ST06_SCIENCE) / f"qc_step06_overlay_{traceset.lower()}")
    outroot.mkdir(parents=True, exist_ok=True)

    tag = f"{sci_path.stem}_with_{geom_path.stem}"
    subdir = outroot / tag
    subdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Science :", sci_path)
    print("[INFO] Geometry:", geom_path)

    sci = fits.getdata(sci_path).astype(float)

    montage_imgs = []
    montage_titles = []
    montage_ovs = []

    with fits.open(geom_path) as ghdul:
        slits = sorted(get_slit_hdus(ghdul), key=slit_number)
        if not slits:
            raise RuntimeError("No SLIT### extensions found in geometry FITS")

        slits = slits[:args.max_slits]

        for h in slits:
            ext = (h.header.get("EXTNAME") or h.name or "SLIT???").strip().upper()

            cut, overlay = centered_cutout_from_geometry(
                sci,
                h.header,
                xhalf=args.xhalf,
                pady=args.pady,
            )

            outpng = subdir / f"{tag}_{ext}.png"
            title = f"{ext}  on  {sci_path.name}"
            save_one(cut, overlay, outpng, title, p_lo=args.p_lo, p_hi=args.p_hi)

            if len(montage_imgs) < args.montage_n:
                montage_imgs.append(cut)
                montage_titles.append(ext)
                montage_ovs.append(overlay)

    mout = outroot / f"{tag}_montage.png"
    save_montage(
        montage_imgs,
        montage_titles,
        montage_ovs,
        mout,
        ncols=args.montage_cols,
        p_lo=args.p_lo,
        p_hi=args.p_hi,
    )

    print("[OK] montage:", mout)
    print("[OK] wrote PNGs under:", subdir)


if __name__ == "__main__":
    main()