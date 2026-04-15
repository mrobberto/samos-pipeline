#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 10:46:39 2026

@author: robberto
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step04 companion QC: physical geometry validation

What it adds beyond the summary QC:
  1) Overlay fitted center/edges on the actual quartz trace image
  2) Raw-vs-fit residuals vs Y for center, left edge, right edge
  3) Width diagnostics vs Y
  4) Mask continuity / fragmentation summary
  5) Optional PNG per slit

Outputs:
  - Step04_QC_COMPANION_EVEN/...
  - Step04_QC_COMPANION_ODD/...
  - per-slit PNGs
  - summary CSV
  - summary text report

Run:
  %runfile '.../Step04_companion_QC.py' --wdir
"""

from pathlib import Path
import csv
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

import config


# ------------------------------------------------------------
# User options
# ------------------------------------------------------------
TRACE_SETS = ["EVEN", "ODD"]

MAKE_ALL_SLIT_PLOTS = True      # if False, only plot worst slits
N_WORST_TO_PLOT = 12            # used when MAKE_ALL_SLIT_PLOTS=False

IMG_HALF_WIDTH = 16             # x half-width around fitted center for overlay
FIG_DPI = 140

USE_REG_PRODUCTS_IF_PRESENT = True

# How to estimate raw center from trace pixels
TRACE_WEIGHTED_CENTER = True

# Plot limits for residual panels
RESID_YLIM = (-2.5, 2.5)

# If True, skip slits with no valid rows in slitid
SKIP_EMPTY_SLITS = False


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _poly_eval_power(coeff, y):
    y = np.asarray(y, float)
    out = np.zeros_like(y, dtype=float)
    p = np.ones_like(y, dtype=float)
    for c in coeff:
        out += float(c) * p
        p *= y
    return out


def _as_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _mad_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def _find_mask_and_slitid(st04: Path, base: str):
    mask_reg = st04 / f"{base}_mask_reg.fits"
    slitid_reg = st04 / f"{base}_slitid_reg.fits"

    if USE_REG_PRODUCTS_IF_PRESENT and mask_reg.exists() and slitid_reg.exists():
        return mask_reg, slitid_reg, "reg"

    return st04 / f"{base}_mask.fits", st04 / f"{base}_slitid.fits", "raw"


def _load_trace_set_products(trace_set: str):
    st04 = Path(config.ST04_PIXFLAT)

    base = "Even_traces" if trace_set.upper() == "EVEN" else "Odd_traces"

    trace_path = st04 / f"{base}.fits"
    geom_path = st04 / f"{base}_geometry.fits"
    mask_path, slitid_path, mask_tag = _find_mask_and_slitid(st04, base)

    if not trace_path.exists():
        raise FileNotFoundError(f"Missing trace image: {trace_path}")
    if not geom_path.exists():
        raise FileNotFoundError(f"Missing geometry file: {geom_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask file: {mask_path}")
    if not slitid_path.exists():
        raise FileNotFoundError(f"Missing slitid file: {slitid_path}")

    trace_img = fits.getdata(trace_path).astype(float)
    mask = fits.getdata(mask_path).astype(bool)
    slitid = fits.getdata(slitid_path).astype(int)

    return st04, base, trace_img, mask, slitid, geom_path, mask_tag


def _extract_coeffs_from_header(hdr):
    porder = int(hdr.get("PORDER", 5))
    eporder = int(hdr.get("EPORDER", porder))

    pc = np.array([hdr.get(f"PC{i}", np.nan) for i in range(porder + 1)], dtype=float)
    lc = np.array([hdr.get(f"LC{i}", np.nan) for i in range(eporder + 1)], dtype=float)
    rc = np.array([hdr.get(f"RC{i}", np.nan) for i in range(eporder + 1)], dtype=float)

    return pc, lc, rc


def _raw_row_measurements(trace_img, slitid, sid, y0, y1, weighted=True):
    """
    Reconstruct raw per-row measurements from the slitid map and trace image.

    Returns arrays over full detector Y:
      xcen_raw, xl_raw, xr_raw, width_raw, nseg, npx
    """
    ny, nx = trace_img.shape

    xcen_raw = np.full(ny, np.nan, float)
    xl_raw = np.full(ny, np.nan, float)
    xr_raw = np.full(ny, np.nan, float)
    width_raw = np.full(ny, np.nan, float)
    nseg = np.zeros(ny, dtype=int)
    npx = np.zeros(ny, dtype=int)

    yy = np.arange(max(0, y0), min(ny - 1, y1) + 1)

    for y in yy:
        xs = np.where(slitid[y] == sid)[0]
        if xs.size == 0:
            continue

        # fragmentation count
        jumps = np.where(np.diff(xs) > 1)[0]
        nseg[y] = int(len(jumps) + 1)
        npx[y] = int(xs.size)

        xl_raw[y] = float(xs.min())
        xr_raw[y] = float(xs.max())
        width_raw[y] = float(xr_raw[y] - xl_raw[y])

        if weighted:
            vals = trace_img[y, xs].astype(float)
            m = np.isfinite(vals)
            if m.sum() >= 3:
                vals = vals[m]
                xuse = xs[m].astype(float)

                base = np.nanmedian(vals)
                w = vals - base
                w[w < 0] = 0.0

                if np.isfinite(w).sum() >= 1 and np.nansum(w) > 0:
                    xcen_raw[y] = float(np.nansum(xuse * w) / np.nansum(w))
                else:
                    xcen_raw[y] = float(np.mean(xs))
            else:
                xcen_raw[y] = float(np.mean(xs))
        else:
            xcen_raw[y] = float(np.mean(xs))

    return xcen_raw, xl_raw, xr_raw, width_raw, nseg, npx


def _choose_worst_slits(summary_rows, n=12):
    """
    Rank by largest of center/edge MAD residuals and fragmentation.
    """
    score = []
    for r in summary_rows:
        vals = [
            abs(_as_float(r["MAD_C"], 0.0)),
            abs(_as_float(r["MAD_L"], 0.0)),
            abs(_as_float(r["MAD_R"], 0.0)),
            0.5 * _as_float(r["FRAG_ROWS"], 0.0),
            0.2 * _as_float(r["MISS_ROWS"], 0.0),
        ]
        score.append(np.nansum(vals))
    idx = np.argsort(score)[::-1]
    return [summary_rows[i]["SLIT"] for i in idx[:min(n, len(idx))]]


# ------------------------------------------------------------
# Per-slit plot
# ------------------------------------------------------------
def _make_slit_plot(
    out_png: Path,
    trace_img: np.ndarray,
    sid: int,
    slit_name: str,
    yplot: np.ndarray,
    xcen_fit: np.ndarray,
    xl_fit: np.ndarray,
    xr_fit: np.ndarray,
    xcen_raw: np.ndarray,
    xl_raw: np.ndarray,
    xr_raw: np.ndarray,
    width_raw: np.ndarray,
    nseg: np.ndarray,
    hdr,
):
    ny, nx = trace_img.shape

    good = np.isfinite(xcen_fit[yplot])
    if np.any(good):
        xmid = np.nanmedian(xcen_fit[yplot][good])
    else:
        xmid = 0.5 * nx

    xlo = max(0, int(np.floor(xmid - IMG_HALF_WIDTH)))
    xhi = min(nx, int(np.ceil(xmid + IMG_HALF_WIDTH + 1)))

    sub = trace_img[yplot.min():yplot.max() + 1, xlo:xhi]

    v1 = np.nanpercentile(sub, 5)
    v2 = np.nanpercentile(sub, 99)

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.28, wspace=0.22)

    # Panel 1: image overlay
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(
        sub,
        origin="lower",
        cmap="gray",
        aspect="auto",
        vmin=v1,
        vmax=v2,
        extent=[xlo, xhi - 1, yplot.min(), yplot.max()],
    )
    ax1.plot(xcen_fit[yplot], yplot, color="red", lw=1.2, label="fit center")
    ax1.plot(xl_fit[yplot], yplot, color="cyan", lw=1.0, label="fit left/right")
    ax1.plot(xr_fit[yplot], yplot, color="cyan", lw=1.0)
    ax1.plot(xcen_raw[yplot], yplot, ".", color="orange", ms=2.5, label="raw center")
    ax1.plot(xl_raw[yplot], yplot, ".", color="lime", ms=2.0, label="raw edges")
    ax1.plot(xr_raw[yplot], yplot, ".", color="lime", ms=2.0)
    ax1.set_title(f"{slit_name} overlay on quartz trace")
    ax1.set_xlabel("X pixel")
    ax1.set_ylabel("Y pixel")
    ax1.legend(loc="best", fontsize=8)

    # Panel 2: center residual
    ax2 = fig.add_subplot(gs[0, 1])
    rc = xcen_raw[yplot] - xcen_fit[yplot]
    ax2.plot(yplot, rc, "k.", ms=2.5)
    ax2.axhline(0.0, color="0.6", lw=0.8)
    ax2.set_ylim(*RESID_YLIM)
    ax2.set_title("center residual: raw - fit")
    ax2.set_xlabel("Y")
    ax2.set_ylabel("pix")

    # Panel 3: edge residuals
    ax3 = fig.add_subplot(gs[1, 0])
    rl = xl_raw[yplot] - xl_fit[yplot]
    rr = xr_raw[yplot] - xr_fit[yplot]
    ax3.plot(yplot, rl, ".", color="tab:blue", ms=2.5, label="left")
    ax3.plot(yplot, rr, ".", color="tab:red", ms=2.5, label="right")
    ax3.axhline(0.0, color="0.6", lw=0.8)
    ax3.set_ylim(*RESID_YLIM)
    ax3.set_title("edge residuals: raw - fit")
    ax3.set_xlabel("Y")
    ax3.set_ylabel("pix")
    ax3.legend(loc="best", fontsize=8)

    # Panel 4: width vs Y
    ax4 = fig.add_subplot(gs[1, 1])
    wfit = xr_fit[yplot] - xl_fit[yplot]
    ax4.plot(yplot, width_raw[yplot], ".", color="tab:green", ms=2.5, label="raw width")
    ax4.plot(yplot, wfit, "-", color="black", lw=1.0, label="fit width")
    ax4.set_title("slit width vs Y")
    ax4.set_xlabel("Y")
    ax4.set_ylabel("pix")
    ax4.legend(loc="best", fontsize=8)

    # Panel 5: fragmentation / occupancy
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(yplot, nseg[yplot], ".", color="tab:purple", ms=2.5, label="segments/row")
    ax5.set_title("mask fragmentation")
    ax5.set_xlabel("Y")
    ax5.set_ylabel("N segments")
    ax5.legend(loc="best", fontsize=8)

    # Panel 6: summary text
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")

    def _medmad(a):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return np.nan, np.nan
        return float(np.nanmedian(a)), float(_mad_sigma(a))

    med_rc, mad_rc = _medmad(rc)
    med_rl, mad_rl = _medmad(rl)
    med_rr, mad_rr = _medmad(rr)

    frag_rows = int(np.sum(nseg[yplot] > 1))
    miss_rows = int(np.sum(~np.isfinite(width_raw[yplot])))

    lines = [
        f"SLITID: {sid}",
        f"EXTNAME: {slit_name}",
        f"RA: {hdr.get('RA', '')}",
        f"DEC: {hdr.get('DEC', '')}",
        f"YMIN..YMAX: {hdr.get('YMIN', '')} .. {hdr.get('YMAX', '')}",
        f"XREF: {hdr.get('XREF', np.nan)}",
        "",
        f"center resid median / MAD : {med_rc:+.3f} / {mad_rc:.3f} pix",
        f"left resid   median / MAD : {med_rl:+.3f} / {mad_rl:.3f} pix",
        f"right resid  median / MAD : {med_rr:+.3f} / {mad_rr:.3f} pix",
        "",
        f"RMSC: {hdr.get('RMSC', np.nan):.3f}",
        f"RMSL: {hdr.get('RMSL', np.nan):.3f}",
        f"RMSR: {hdr.get('RMSR', np.nan):.3f}",
        "",
        f"fragmented rows (>1 segment): {frag_rows}",
        f"missing rows in slitid map:   {miss_rows}",
    ]
    ax6.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=9)

    fig.suptitle(f"Step04 companion QC — {slit_name}", fontsize=13)
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Main QC per set
# ------------------------------------------------------------
def qc_one_set(trace_set: str):
    st04, base, trace_img, mask, slitid, geom_path, mask_tag = _load_trace_set_products(trace_set)

    outdir = st04 / f"Step04_QC_COMPANION_{trace_set.upper()}"
    outdir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    with fits.open(geom_path) as hdul:
        for h in hdul[1:]:
            hdr = h.header
            sid = int(hdr.get("SLITID", -999))
            slit_name = h.name if h.name else f"SLIT{sid:03d}"

            y0 = int(hdr.get("YMIN", 0))
            y1 = int(hdr.get("YMAX", trace_img.shape[0] - 1))

            if y1 <= y0:
                summary_rows.append({
                    "SLIT": slit_name, "SLITID": sid, "YMIN": y0, "YMAX": y1,
                    "MAD_C": np.nan, "MAD_L": np.nan, "MAD_R": np.nan,
                    "FRAG_ROWS": np.nan, "MISS_ROWS": np.nan,
                    "MEDW_RAW": np.nan, "MEDW_FIT": np.nan,
                    "RMSC": hdr.get("RMSC", np.nan),
                    "RMSL": hdr.get("RMSL", np.nan),
                    "RMSR": hdr.get("RMSR", np.nan),
                })
                continue

            pc, lc, rc = _extract_coeffs_from_header(hdr)

            yy = np.arange(y0, y1 + 1, dtype=float)
            xcen_fit = np.full(trace_img.shape[0], np.nan, float)
            xl_fit = np.full(trace_img.shape[0], np.nan, float)
            xr_fit = np.full(trace_img.shape[0], np.nan, float)

            xcen_fit[y0:y1 + 1] = _poly_eval_power(pc, yy)
            xl_fit[y0:y1 + 1] = _poly_eval_power(lc, yy)
            xr_fit[y0:y1 + 1] = _poly_eval_power(rc, yy)

            xcen_raw, xl_raw, xr_raw, width_raw, nseg, npx = _raw_row_measurements(
                trace_img, slitid, sid, y0, y1, weighted=TRACE_WEIGHTED_CENTER
            )

            yplot = np.arange(y0, y1 + 1, dtype=int)

            if SKIP_EMPTY_SLITS and np.all(~np.isfinite(width_raw[yplot])):
                continue

            rcen = xcen_raw[yplot] - xcen_fit[yplot]
            rleft = xl_raw[yplot] - xl_fit[yplot]
            rright = xr_raw[yplot] - xr_fit[yplot]

            wfit = xr_fit[yplot] - xl_fit[yplot]

            summary_rows.append({
                "SLIT": slit_name,
                "SLITID": sid,
                "RA": hdr.get("RA", ""),
                "DEC": hdr.get("DEC", ""),
                "YMIN": y0,
                "YMAX": y1,
                "MAD_C": _mad_sigma(rcen),
                "MAD_L": _mad_sigma(rleft),
                "MAD_R": _mad_sigma(rright),
                "MED_C": np.nanmedian(rcen),
                "MED_L": np.nanmedian(rleft),
                "MED_R": np.nanmedian(rright),
                "FRAG_ROWS": int(np.sum(nseg[yplot] > 1)),
                "MISS_ROWS": int(np.sum(~np.isfinite(width_raw[yplot]))),
                "MEDW_RAW": np.nanmedian(width_raw[yplot]),
                "MEDW_FIT": np.nanmedian(wfit),
                "RMSC": hdr.get("RMSC", np.nan),
                "RMSL": hdr.get("RMSL", np.nan),
                "RMSR": hdr.get("RMSR", np.nan),
            })

    if MAKE_ALL_SLIT_PLOTS:
        slit_names_to_plot = [r["SLIT"] for r in summary_rows]
    else:
        slit_names_to_plot = _choose_worst_slits(summary_rows, n=N_WORST_TO_PLOT)

    with fits.open(geom_path) as hdul:
        for h in hdul[1:]:
            hdr = h.header
            sid = int(hdr.get("SLITID", -999))
            slit_name = h.name if h.name else f"SLIT{sid:03d}"

            if slit_name not in slit_names_to_plot:
                continue

            y0 = int(hdr.get("YMIN", 0))
            y1 = int(hdr.get("YMAX", trace_img.shape[0] - 1))
            if y1 <= y0:
                continue

            pc, lc, rc = _extract_coeffs_from_header(hdr)

            yy = np.arange(y0, y1 + 1, dtype=float)
            xcen_fit = np.full(trace_img.shape[0], np.nan, float)
            xl_fit = np.full(trace_img.shape[0], np.nan, float)
            xr_fit = np.full(trace_img.shape[0], np.nan, float)

            xcen_fit[y0:y1 + 1] = _poly_eval_power(pc, yy)
            xl_fit[y0:y1 + 1] = _poly_eval_power(lc, yy)
            xr_fit[y0:y1 + 1] = _poly_eval_power(rc, yy)

            xcen_raw, xl_raw, xr_raw, width_raw, nseg, npx = _raw_row_measurements(
                trace_img, slitid, sid, y0, y1, weighted=TRACE_WEIGHTED_CENTER
            )

            yplot = np.arange(y0, y1 + 1, dtype=int)

            out_png = outdir / f"{base}_{slit_name}_companionQC.png"
            _make_slit_plot(
                out_png=out_png,
                trace_img=trace_img,
                sid=sid,
                slit_name=slit_name,
                yplot=yplot,
                xcen_fit=xcen_fit,
                xl_fit=xl_fit,
                xr_fit=xr_fit,
                xcen_raw=xcen_raw,
                xl_raw=xl_raw,
                xr_raw=xr_raw,
                width_raw=width_raw,
                nseg=nseg,
                hdr=hdr,
            )

    # summary CSV
    csv_path = outdir / f"{base}_companionQC_summary.csv"
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)

    # text report
    txt_path = outdir / f"{base}_companionQC_report.txt"
    with open(txt_path, "w") as f:
        f.write(f"Step04 companion QC report — {trace_set.upper()}\n")
        f.write(f"Trace image / geometry base: {base}\n")
        f.write(f"Mask/slitid source: {mask_tag}\n")
        f.write(f"Number of slits in geometry file: {len(summary_rows)}\n\n")

        if summary_rows:
            madc = np.array([_as_float(r["MAD_C"]) for r in summary_rows], float)
            madl = np.array([_as_float(r["MAD_L"]) for r in summary_rows], float)
            madr = np.array([_as_float(r["MAD_R"]) for r in summary_rows], float)
            frag = np.array([_as_float(r["FRAG_ROWS"]) for r in summary_rows], float)

            f.write(f"Median MAD center residual : {np.nanmedian(madc):.3f} pix\n")
            f.write(f"Median MAD left residual   : {np.nanmedian(madl):.3f} pix\n")
            f.write(f"Median MAD right residual  : {np.nanmedian(madr):.3f} pix\n")
            f.write(f"Median fragmented rows/slit: {np.nanmedian(frag):.1f}\n\n")

            worst = _choose_worst_slits(summary_rows, n=min(12, len(summary_rows)))
            f.write("Worst slits by combined residual/fragmentation score:\n")
            for s in worst:
                rr = next(r for r in summary_rows if r["SLIT"] == s)
                f.write(
                    f"  {s:8s}  "
                    f"MAD_C={_as_float(rr['MAD_C']):5.2f}  "
                    f"MAD_L={_as_float(rr['MAD_L']):5.2f}  "
                    f"MAD_R={_as_float(rr['MAD_R']):5.2f}  "
                    f"FRAG={int(_as_float(rr['FRAG_ROWS'], 0))}  "
                    f"MISS={int(_as_float(rr['MISS_ROWS'], 0))}\n"
                )

    print(f"[DONE] {trace_set.upper()} companion QC written to: {outdir}")


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    for ts in TRACE_SETS:
        qc_one_set(ts)