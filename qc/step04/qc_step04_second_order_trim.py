"""
QC visualization of Step04 second-order trimming.

This script generates internal QC plots to inspect the per-slit second-order
removal performed in Step04.

Outputs
-------
1. Full-frame quartz difference image with:
   - final mask overlay
   - per-slit y_cut bars from *_gap_cuts.csv

2. Per-slit geometry-centered montage with:
   - quartz cutout
   - fitted center/edge overlays
   - y_cut line
   - optional shaded rejected region below the cut

3. Optional before/after comparison if *_mask_pretrim.fits exists:
   - pre-trim mask overlay
   - post-trim mask overlay
   - removed pixels only

Run
---
PYTHONPATH=. python qc/step04/qc_step04_second_order_trim.py --traceset EVEN
PYTHONPATH=. python qc/step04/qc_step04_second_order_trim.py --traceset ODD
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

st04 = Path(config.ST04_TRACES)


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
    base = "Even_traces" if traceset == "EVEN" else "Odd_traces"

    trace = first_existing([
        st04 / f"{base}.fits",
    ])
    mask = first_existing([
        st04 / f"{base}_mask_reg.fits",
        st04 / f"{base}_mask.fits",
    ])
    slitid = first_existing([
        st04 / f"{base}_slitid_reg.fits",
        st04 / f"{base}_slitid.fits",
    ])
    pretrim = first_existing([
        st04 / f"{base}_mask_pretrim.fits",
    ])
    geom = first_existing([
        st04 / f"{base}_geometry.fits",
    ])
    cuts = first_existing([
        st04 / f"{base}_gap_cuts.csv",
    ])
    return base, trace, mask, slitid, pretrim, geom, cuts


def load_gap_cuts(path: Path):
    out = {}
    if path is None or not path.exists():
        return out
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = int(row["slit_id"])
            raw = (row.get("y_cut", "") or "").strip()
            out[sid] = None if raw == "" else int(raw)
    return out


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


def overlay_mask(ax, quartz, mask, title, color="autumn", alpha=0.35, p_lo=5.0, p_hi=99.5):
    vmin, vmax = robust_limits(quartz, p_lo, p_hi)
    ax.imshow(quartz, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    m = np.where(mask > 0, 1.0, np.nan)
    ax.imshow(m, origin="lower", cmap=color, alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def draw_gap_bars_fullframe(ax, gap_cuts, geom_hdul, half_len=18, color="yellow", lw=1.0):
    for sid, ycut in sorted(gap_cuts.items()):
        if ycut is None:
            continue
        slit_name = f"SLIT{sid:03d}"
        ghdu = get_slit_hdu(geom_hdul, slit_name)
        if ghdu is None:
            continue
        xref = ghdu.header.get("XREF", None)
        if xref is None:
            continue
        x0 = float(xref) - half_len
        x1 = float(xref) + half_len
        ax.plot([x0, x1], [ycut, ycut], color=color, lw=lw)


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
        xl = None
        xr = None
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
        "xl": (xl - x0) if xl is not None else None,
        "xr": (xr - x0) if xr is not None else None,
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
    }
    return cut, overlay


def save_fullframe_main(quartz, mask, geom_path, gap_cuts, outjpg, title):
    with fits.open(geom_path) as ghdul:
        fig, ax = plt.subplots(figsize=(11, 8), dpi=170)
        overlay_mask(ax, quartz, mask, title)
        draw_gap_bars_fullframe(ax, gap_cuts, ghdul, half_len=18, color="yellow", lw=1.2)
        fig.tight_layout()
        fig.savefig(outjpg, format="jpg")
        plt.close(fig)


def save_prepost(quartz, mask_post, mask_pre, geom_path, gap_cuts, outjpg, title):
    removed = (mask_pre > 0) & ~(mask_post > 0)
    vmin, vmax = robust_limits(quartz, 5, 99.5)

    with fits.open(geom_path) as ghdul:
        fig, axs = plt.subplots(1, 3, figsize=(18, 7), dpi=170)

        for ax, m, ttl, cmap, alpha in [
            (axs[0], mask_pre, "pre-trim mask", "winter", 0.35),
            (axs[1], mask_post, "post-trim mask", "autumn", 0.35),
            (axs[2], removed, "removed pixels", "Reds", 0.55),
        ]:
            ax.imshow(quartz, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
            mm = np.where(m > 0, 1.0, np.nan)
            ax.imshow(mm, origin="lower", cmap=cmap, alpha=alpha)
            draw_gap_bars_fullframe(ax, gap_cuts, ghdul, half_len=18, color="yellow", lw=1.2)
            ax.set_title(ttl)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(outjpg, format="jpg")
        plt.close(fig)


def save_montage(images, titles, overlays, ycuts, outjpg, ncols=6, p_lo=5.0, p_hi=99.5, shade_cut=True):
    if not images:
        return

    n = len(images)
    nrows = math.ceil(n / ncols)
    fig = plt.figure(figsize=(ncols * 2.4, nrows * 2.2), dpi=180)

    for i, (img, ttl, ov, ycut) in enumerate(zip(images, titles, overlays, ycuts), start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        vmin, vmax = robust_limits(img, p_lo, p_hi)
        ax.imshow(img, origin="lower", aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)

        if ov.get("xcen") is not None:
            ax.plot(ov["xcen"], ov["yy"], color="cyan", lw=0.7)
        if ov.get("xl") is not None:
            ax.plot(ov["xl"], ov["yy"], color="yellow", lw=0.5)
        if ov.get("xr") is not None:
            ax.plot(ov["xr"], ov["yy"], color="yellow", lw=0.5)

        if ycut is not None:
            ycut_rel = ycut - ov["y0"]
            ax.axhline(ycut_rel, color="red", lw=0.8)
            if shade_cut:
                ax.axhspan(0, ycut_rel, color="red", alpha=0.12)

        ax.set_title(ttl, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(outjpg.stem, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outjpg, format="jpg")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="QC for Step04 second-order trimming")
    ap.add_argument("--traceset", required=True, choices=["EVEN", "ODD", "even", "odd"])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max-slits", type=int, default=24)
    ap.add_argument("--pad", type=int, default=14)
    args = ap.parse_args()

    traceset = args.traceset.upper()
    base, trace_path, mask_path, slitid_path, pretrim_path, geom_path, cuts_path = pick_files(traceset)

    if trace_path is None or not trace_path.exists():
        raise FileNotFoundError(f"Trace image not found for {traceset}")
    if mask_path is None or not mask_path.exists():
        raise FileNotFoundError(f"Mask not found for {traceset}")
    if slitid_path is None or not slitid_path.exists():
        raise FileNotFoundError(f"SlitID not found for {traceset}")
    if geom_path is None or not geom_path.exists():
        raise FileNotFoundError(f"Geometry file not found for {traceset}")
    if cuts_path is None or not cuts_path.exists():
        raise FileNotFoundError(f"Gap-cuts CSV not found for {traceset}")

    outdir = Path(args.outdir) if args.outdir else (st04 / f"qc_step04_second_order_{traceset.lower()}")
    outdir.mkdir(parents=True, exist_ok=True)

    quartz = fits.getdata(trace_path).astype(float)
    mask = fits.getdata(mask_path).astype(float)
    slitid = fits.getdata(slitid_path).astype(int)
    gap_cuts = load_gap_cuts(cuts_path)

    save_fullframe_main(
        quartz,
        mask,
        geom_path,
        gap_cuts,
        outdir / f"{traceset.lower()}_second_order_fullframe.jpg",
        f"{traceset} quartz difference + final mask + gap cuts",
    )

    if pretrim_path is not None and pretrim_path.exists():
        mask_pre = fits.getdata(pretrim_path).astype(float)
        save_prepost(
            quartz,
            mask,
            mask_pre,
            geom_path,
            gap_cuts,
            outdir / f"{traceset.lower()}_second_order_prepost.jpg",
            f"{traceset} second-order trim: pre/post/removed",
        )

    ids = sorted([i for i in np.unique(slitid) if i >= 0])[:args.max_slits]
    imgs, ttls, ovs, ycuts = [], [], [], []

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
            ycuts.append(gap_cuts.get(sid, None))

    save_montage(
        imgs,
        ttls,
        ovs,
        ycuts,
        outdir / f"{traceset.lower()}_second_order_montage.jpg",
        ncols=6,
        shade_cut=True,
    )

    print("[OK] trace  :", trace_path)
    print("[OK] mask   :", mask_path)
    print("[OK] slitid :", slitid_path)
    print("[OK] geom   :", geom_path)
    print("[OK] cuts   :", cuts_path)
    if pretrim_path and pretrim_path.exists():
        print("[OK] pretrim:", pretrim_path)
    print("[OK] wrote QC under:", outdir)


if __name__ == "__main__":
    main()