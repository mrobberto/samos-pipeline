#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC Step08a — detector-frame slit montage with Step04 geometry and Step08 ridge.

Builds a slit-by-slit detector-frame mosaic from the Step06b full-frame science
image. The cutouts, ordering, and geometry overlays are matched to the Step04
trace quicklook style, so the Step04, Step06b, and Step08a diagnostics can be
compared directly.

Each panel shows:
- grayscale background: Step06b pixel-flat-corrected science image
- cyan curve: Step04 quartz-derived slit centerline
- yellow curves: Step04 quartz-derived slit edges
- red curve: Step08a1 science ridge X0(y), mapped from TRACECOORDS back onto
  the detector-frame cutout

This is a QC/display product only. The actual ridge measurement and extraction
are performed in TRACECOORDS; this script maps those results back to detector
coordinates for visual validation.

Usage
-----
PYTHONPATH=. python qc/step08/qc_step08a_detector_montage_from_fullframe.py --set EVEN
PYTHONPATH=. python qc/step08/qc_step08a_detector_montage_from_fullframe.py --set ODD

Typical explicit usage:
PYTHONPATH=. python qc/step08/qc_step08a_detector_montage_from_fullframe.py \
    --set EVEN \
    --science /path/to/FinalScience_dolidze_ADUperS_pixflatcorr_EVEN.fits \
    --analysis /path/to/trace_analysis_optimal_ridge_even.fits \
    --max-slits 36 \
    --vmax-from-seed

Optional inputs:
- --science: Step06b detector-frame science image
- --slitid: Step04 slit-id image, if automatic discovery is not desired
- --analysis: Step08a1 trace-analysis FITS containing X0, TRXLEFT, TRXRIGHT,
  and seed metadata
- --vmax-from-seed: scale each panel using the science value at the ridge/seed
  position, useful for faint spectra
- --show-mask-outline: overlay the slit-id footprint for debugging only

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


def robust_limits(img: np.ndarray, p_lo: float = 5.0, p_hi: float = 99.5) -> tuple[float, float]:
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


def pick_geometry_file(set_tag: str) -> Path:
    base = "Even_traces" if set_tag == "EVEN" else "Odd_traces"
    p = Path(config.ST04_TRACES) / f"{base}_geometry.fits"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


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


def centered_cutout_from_geometry(img, hdr, xhalf=14, pady=8):
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

def plot_tracecoords_point(ax, x_tc, y_tc, y0det, overlay, ypix, trxleft, trxright,
                           marker="o", color="red", size=5, filled=False):
    if x_tc is None or y_tc is None:
        return None

    try:
        x_tc = float(x_tc)
        y_tc = float(y_tc)
    except Exception:
        return None

    if not np.isfinite(x_tc) or not np.isfinite(y_tc):
        return None

    yy_geom = overlay["yy"]
    xl_geom = overlay["xl"]
    xr_geom = overlay["xr"]

    if xl_geom is None or xr_geom is None:
        return None

    y_det = y0det + y_tc - overlay["y0"]

    trx_l = np.interp(y_tc, ypix, trxleft)
    trx_r = np.interp(y_tc, ypix, trxright)

    if not np.isfinite(trx_l) or not np.isfinite(trx_r) or trx_r <= trx_l:
        return None

    xl_det = np.interp(y_det, yy_geom, xl_geom)
    xr_det = np.interp(y_det, yy_geom, xr_geom)

    u = (x_tc - trx_l) / (trx_r - trx_l)
    u = np.clip(u, -0.5, 1.5)

    x_det = xl_det + u * (xr_det - xl_det)

    ax.plot(
        x_det, y_det,
        marker=marker,
        markersize=size,
        markerfacecolor=color if filled else "none",
        markeredgecolor=color,
        markeredgewidth=1.2,
        linestyle="None",
    )

    return x_det, y_det



def main() -> None:
    ap = argparse.ArgumentParser(description="QC Step08a detector montage from full frame")
    ap.add_argument("--set", required=True, choices=["EVEN", "ODD", "even", "odd"])
    ap.add_argument("--science", default=None, help="Explicit Step06 detector-frame science FITS")
    ap.add_argument("--slitid", default=None, help="Explicit slit-id FITS")
    ap.add_argument("--slit-min", type=int, default=None)
    ap.add_argument("--slit-max", type=int, default=None)
    ap.add_argument("--pad", type=int, default=8, help="Padding (pixels) around each slit bbox")
    ap.add_argument("--ncols", type=int, default=6)
    ap.add_argument("--max-slits", type=int, default=36)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--show-mask-outline", action="store_true",
                    help="Overlay the slit-id region contour in magenta")
    ap.add_argument("--no-lines", action="store_true",
                    help="Disable curved center/edge overlays")
    ap.add_argument("--curve-order", type=int, default=3,
                    help="Polynomial order for displayed center/edge curves")
    ap.add_argument("--analysis", default=None, help="Step08a1 trace_analysis_optimal_ridge_<set>.fits")
    ap.add_argument("--vmax-from-seed", action="store_true",
                help="Use flux at Step08 seed/ridge point as display vmax")
    args = ap.parse_args()

    set_tag = args.set.upper()
    science = Path(args.science) if args.science else pick_science(set_tag)
    slitid = Path(args.slitid) if args.slitid else pick_slitid(set_tag)
    geom_path = pick_geometry_file(set_tag)

    sci = load_primary_2d(science)
    lbl = load_primary_2d(slitid)
    analysis_path = Path(args.analysis).expanduser() if args.analysis else (
        Path(config.ST08_EXTRACT1D) / f"trace_analysis_optimal_ridge_{set_tag.lower()}.fits"
    )
    
    h08 = fits.open(analysis_path) if analysis_path.exists() else None

    ghdul = fits.open(geom_path)

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
    slit_ids = slit_ids[:args.max_slits]

    outdir = Path(args.outdir) if args.outdir else Path(config.ST08_EXTRACT1D) / "qc_step08"
    outdir.mkdir(parents=True, exist_ok=True)

    ncols = max(1, int(args.ncols))
    nrows = math.ceil(len(slit_ids) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.3 * ncols, 2.1 * nrows),
        squeeze=False,
    )

    for ax in axes.ravel():
        ax.set_visible(False)

    for ax, sid in zip(axes.ravel(), slit_ids):
        ax.set_visible(True)
        slit_name = f"SLIT{sid:03d}"
        ghdu = get_slit_hdu(ghdul, slit_name)
        if ghdu is None:
            ax.set_visible(False)
            continue
        
        cut, overlay = centered_cutout_from_geometry(
            sci,
            ghdu.header,
            xhalf=14,
            pady=int(args.pad),
        )
        
        y0, y1 = overlay["y0"], overlay["y1"]
        x0, x1 = overlay["x0"], overlay["x1"]
        
        mask = np.asarray(lbl == sid)
        cut_mask = np.asarray(mask[y0:y1, x0:x1], bool)
        
        # Optional display boost: use the seed / ridge value as local vmax
        peak_vmax = None
                                    
        # Step04 geometry: gold-standard overlay
        if not args.no_lines:
            if overlay.get("xcen") is not None:
                ax.plot(overlay["xcen"], overlay["yy"], color="cyan", lw=0.7)
            if overlay.get("xl") is not None:
                ax.plot(overlay["xl"], overlay["yy"], color="yellow", lw=0.5)
            if overlay.get("xr") is not None:
                ax.plot(overlay["xr"], overlay["yy"], color="yellow", lw=0.5)
        
        
        # --- Step08 science ridge from analysis FITS ---
        seed_display_x = None
        seed_display_y = None
        if h08 is not None and slit_name in h08:
            tab = h08[slit_name].data
            hdr08 = h08[slit_name].header
            
            seed_y = hdr08.get("S08YSED", None)
            seed_x = hdr08.get("S08XSED", None)
        
            ypix = np.asarray(tab["YPIX"], float)
            x0_ridge = np.asarray(tab["X0"], float)
            trxleft = np.asarray(tab["TRXLEFT"], float)
            trxright = np.asarray(tab["TRXRIGHT"], float)
        
            # detector-frame Step04 edges in cutout coordinates
            yy_geom = overlay["yy"]
            xl_geom = overlay["xl"]
            xr_geom = overlay["xr"]
        
        
            # TRACECOORDS row -> detector-cutout row
            y0det = float(hdr08.get("Y0DET", hdr08.get("YMIN", 0)))
            y_r_all = y0det + ypix - overlay["y0"]
            
            good = (
                np.isfinite(y_r_all)
                & np.isfinite(x0_ridge)
                & np.isfinite(trxleft)
                & np.isfinite(trxright)
                & (trxright > trxleft)
                & (xl_geom is not None)
                & (xr_geom is not None)
            )
            
            if np.any(good):
                y_r = y_r_all[good]
            
                u = (x0_ridge[good] - trxleft[good]) / (trxright[good] - trxleft[good])
                u = np.clip(u, -0.5, 1.5)
            
                xl_at_y = np.interp(y_r, yy_geom, xl_geom)
                xr_at_y = np.interp(y_r, yy_geom, xr_geom)
            
                x_r = xl_at_y + u * (xr_at_y - xl_at_y)
            
                ax.plot(x_r, y_r, color="red", lw=1.2)
                
                # --- compare seed position to final ridge at seed Y ---
                seed_y = hdr08.get("S08YSED", None)
                seed_x = hdr08.get("S08XSED", None)
                
                block_x = hdr08.get("S08XBLOK", None)
                block_y = hdr08.get("S08YBLOK", None)
                patch_x = hdr08.get("S08XPATC", None)
                patch_y = hdr08.get("S08YPATC", None)
                
                if seed_y is not None and seed_x is not None:
                    
                    try:
                        seed_y = float(seed_y)
                        seed_x = float(seed_x)
                
                        y0det = float(hdr08.get("Y0DET", hdr08.get("YMIN", 0)))
                        y_seed = y0det + seed_y - overlay["y0"]
                
                        # detector edges at seed row
                        xl_seed = np.interp(y_seed, yy_geom, xl_geom)
                        xr_seed = np.interp(y_seed, yy_geom, xr_geom)
                
                        # TRACECOORDS valid range at seed row
                        trx_l_seed = np.interp(seed_y, ypix, trxleft)
                        trx_r_seed = np.interp(seed_y, ypix, trxright)
                
                        if np.isfinite(trx_l_seed) and np.isfinite(trx_r_seed) and trx_r_seed > trx_l_seed:
                            # seed position
                            u_seed = (seed_x - trx_l_seed) / (trx_r_seed - trx_l_seed)
                            x_seed = xl_seed + u_seed * (xr_seed - xl_seed)
                            """
                            ax.plot(
                                x_seed, y_seed,
                                marker="o",
                                markersize=5,
                                markerfacecolor="none",
                                markeredgecolor="red",
                                markeredgewidth=1.2,
                            )
                            """
                            # final ridge position at same Y
                            x0_at_seed = np.interp(seed_y, ypix, x0_ridge)
                            u_ridge_seed = (x0_at_seed - trx_l_seed) / (trx_r_seed - trx_l_seed)
                            x_ridge_seed = xl_seed + u_ridge_seed * (xr_seed - xl_seed)
                
                            ax.plot(
                                x_ridge_seed, y_seed,
                                marker="o",
                                markersize=4,
                                markerfacecolor="orange",
                                markeredgecolor="red",
                                markeredgewidth=1.2,
                            )
                            
                            # --- define display point for scaling ---
                            seed_display_x = x_ridge_seed
                            seed_display_y = y_seed
                
                    except Exception:
                        pass
                    
        # --- scaling (AFTER seed/ridge computation) ---
        vmin, vmax = robust_limits(cut)
        
        if args.vmax_from_seed and seed_display_x is not None:
            xi = int(round(seed_display_x))
            yi = int(round(seed_display_y))
        
            if 0 <= yi < cut.shape[0] and 0 <= xi < cut.shape[1]:
                sv = float(cut[yi, xi])
                if np.isfinite(sv):
                    vmax = sv * 1.2
        
        # --- display ---
        ax.imshow(cut, origin="lower", aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(slit_name, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])        
                    
        
        # Optional mask outline, now only diagnostic
        if args.show_mask_outline and np.any(cut_mask):
            yy, xx = np.mgrid[:cut_mask.shape[0], :cut_mask.shape[1]]
            ax.contour(xx, yy, cut_mask.astype(float), levels=[0.5], colors=["magenta"], linewidths=0.4)
    
    stem = f"{set_tag.lower()}_detector_montage"
    outpng = outdir / f"{stem}.png"

    fig.suptitle(stem, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpng, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    ghdul.close()
    
    if h08 is not None:
        h08.close()
    
    print(f"Science : {science}")
    print(f"Slit-id : {slitid}")
    print(f"Wrote   : {outpng}")


if __name__ == "__main__":
    main()
