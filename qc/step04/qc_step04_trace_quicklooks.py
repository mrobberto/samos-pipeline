#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quicklook visualization of Step04 trace geometry.

This script generates full-frame and per-slit diagnostic plots for the
trace solutions derived from quartz-lamp data (Step04).

Key feature:
------------
Per-slit panels are **geometry-centered**, not mask-centered.
Each cutout is built from the fitted trace geometry (centerline and edges)
using YMIN/YMAX and polynomial coefficients (PC*, LC*, RC*), ensuring that
the illuminated ridge is visually centered even when the mask is asymmetric
or slightly offset.

Inputs:
-------
- config.ST04_TRACES:
    Even/Odd_traces_geometry.fits
    Even/Odd_traces_mask[_reg].fits
    Even/Odd_traces_slitid[_reg].fits

- config.ST05_PIXFLAT:
    quartz_diff_even.fits
    quartz_diff_odd.fits

Outputs:
--------
- Full-frame quicklooks (mask, slitid, quartz, overlay)
- Per-slit montage panels (geometry-centered)
- Optional overlays of centerline and edges

Notes:
------
- This is a visualization/QC tool; it does not validate segmentation completeness.
- For detailed validation of individual traces, use the single-slit diagnostic script.

run
---
PYTHONPATH=. python qc/step04/qc_step04_trace_quicklooks.py --traceset EVEN
PYTHONPATH=. python qc/step04/qc_step04_trace_quicklooks.py --traceset ODD
"""
import argparse
from pathlib import Path
import math

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

st04 = Path(config.ST04_TRACES)
st05 = Path(config.ST05_PIXFLAT)

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


def first_existing(paths):
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            return p
    return None


def pick_files(traceset):
    st04 = Path(config.ST04_TRACES)
    st05 = Path(config.ST05_PIXFLAT)

    base = "Even_traces" if traceset == "EVEN" else "Odd_traces"
    suffix = traceset.lower()

    mask = first_existing([
        st04 / f"{base}_mask_reg.fits",
        st04 / f"{base}_mask.fits",
    ])
    slitid = first_existing([
        st04 / f"{base}_slitid_reg.fits",
        st04 / f"{base}_slitid.fits",
    ])
    quartz = first_existing([
        st05 / f"quartz_diff_{suffix}.fits",
    ])

    return base, mask, slitid, quartz


def save_fullframe(img, outjpg, title, cmap="gray", p_lo=5.0, p_hi=99.5):
    vmin, vmax = robust_limits(img, p_lo, p_hi)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(outjpg, format="jpg")
    plt.close(fig)


def save_overlay(quartz, mask, outjpg, title, p_lo=5.0, p_hi=99.5):
    vmin, vmax = robust_limits(quartz, p_lo, p_hi)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.imshow(quartz, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    m = np.where(mask > 0, 1.0, np.nan)
    ax.imshow(m, origin="lower", cmap="autumn", alpha=0.35)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.tight_layout()
    fig.savefig(outjpg, format="jpg")
    plt.close(fig)


def bounding_box(mask, pad=8):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    y0 = max(0, ys.min() - pad)
    y1 = ys.max() + pad + 1
    x0 = max(0, xs.min() - pad)
    x1 = xs.max() + pad + 1
    return y0, y1, x0, x1


def save_montage(images, titles, outjpg, overlays=None, ncols=6, p_lo=5.0, p_hi=99.5):
    if not images:
        return

    n = len(images)
    nrows = math.ceil(n / ncols)
    fig = plt.figure(figsize=(ncols * 2.3, nrows * 2.1), dpi=170)

    if overlays is None:
        overlays = [None] * len(images)

    for i, (img, ttl, ov) in enumerate(zip(images, titles, overlays), start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        vmin, vmax = robust_limits(img, p_lo, p_hi)
        ax.imshow(img, origin="lower", aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)

        if ov is not None:
            if ov.get("xcen") is not None:
                ax.plot(ov["xcen"], ov["yy"], color="cyan", lw=0.7)
            if ov.get("xl") is not None:
                ax.plot(ov["xl"], ov["yy"], color="yellow", lw=0.5)
            if ov.get("xr") is not None:
                ax.plot(ov["xr"], ov["yy"], color="yellow", lw=0.5)

        ax.set_title(ttl, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(outjpg.stem, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outjpg, format="jpg")
    plt.close(fig)
    
def get_slit_hdu(hdul, slit_name: str):
    slit_name = slit_name.strip().upper()
    for h in hdul[1:]:
        nm = (h.header.get("EXTNAME") or h.name or "").strip().upper()
        if nm == slit_name:
            return h
    return None


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


def centered_cutout_from_geometry(quartz, hdr, xhalf=14, pady=20):
    ny, nx = quartz.shape

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

    if lc is not None and rc is not None:
        xl = poly_eval(lc, yy)
        xr = poly_eval(rc, yy)
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

    cut = quartz[y0:y1, x0:x1].copy()

    overlay = {
        "yy": yy - y0,
        "xcen": xcen - x0,
        "xl": (poly_eval(lc, yy) - x0) if lc is not None else None,
        "xr": (poly_eval(rc, yy) - x0) if rc is not None else None,
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
    }
    return cut, overlay    


def main():
    ap = argparse.ArgumentParser(description="Step04 trace quicklooks")
    ap.add_argument("--traceset", required=True, choices=["EVEN", "ODD", "even", "odd"])
    ap.add_argument("--mask", default=None)
    ap.add_argument("--slitid", default=None)
    ap.add_argument("--quartz", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--pad", type=int, default=8)
    ap.add_argument("--max-slits", type=int, default=24)
    args = ap.parse_args()

    traceset = args.traceset.upper()
    base, auto_mask, auto_slitid, auto_quartz = pick_files(traceset)

    mask_path = Path(args.mask) if args.mask else auto_mask
    slitid_path = Path(args.slitid) if args.slitid else auto_slitid
    quartz_path = Path(args.quartz) if args.quartz else auto_quartz

    if mask_path is None or not mask_path.exists():
        raise FileNotFoundError(f"Mask not found for {traceset}")
    if slitid_path is None or not slitid_path.exists():
        raise FileNotFoundError(f"SlitID not found for {traceset}")

    outdir = Path(args.outdir) if args.outdir else (Path(config.ST04_TRACES) / f"qc_step04_{traceset.lower()}")
    outdir.mkdir(parents=True, exist_ok=True)

    mask = fits.getdata(mask_path).astype(float)
    slitid = fits.getdata(slitid_path).astype(int)
    quartz = fits.getdata(quartz_path).astype(float) if quartz_path and quartz_path.exists() else None

    save_fullframe(mask, outdir / f"{traceset.lower()}_mask.jpg", f"{traceset} trace mask")
    save_fullframe(slitid, outdir / f"{traceset.lower()}_slitid.jpg", f"{traceset} slit-id map", cmap="viridis")

    if quartz is not None:
        save_fullframe(quartz, outdir / f"{traceset.lower()}_quartz.jpg", f"{traceset} quartz diff")
        save_overlay(quartz, mask, outdir / f"{traceset.lower()}_overlay.jpg", f"{traceset} quartz + mask")

    geom_path = st04 / f"{base}_geometry.fits"
    if not geom_path.exists():
        raise FileNotFoundError(f"Geometry file not found: {geom_path}")
    
    ids = sorted([i for i in np.unique(slitid) if i >= 0])[:args.max_slits]
    imgs, ttls, ovs = [], [], []
    
    with fits.open(geom_path) as ghdul:
        for sid in ids:
            slit_name = f"SLIT{sid:03d}"
            ghdu = get_slit_hdu(ghdul, slit_name)
            if ghdu is None:
                continue
    
            cut, overlay = centered_cutout_from_geometry(
                quartz,
                ghdu.header,
                xhalf=14,
                pady=args.pad,
            )
    
            imgs.append(cut)
            ttls.append(slit_name)
            ovs.append(overlay)
    
    save_montage(
        imgs,
        ttls,
        outdir / f"{traceset.lower()}_slit_montage.jpg",
        overlays=ovs,
    )    

    print("[OK] mask   :", mask_path)
    print("[OK] slitid :", slitid_path)
    if quartz_path:
        print("[OK] quartz :", quartz_path)
    print("[OK] wrote quicklooks under:", outdir)


if __name__ == "__main__":
    main()