#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper — score arc-line footprint cleanliness for Step07 reference-slit selection.

Purpose
-------
Inspect the 2D full-frame arc image together with the Step04 slit geometry and
measure whether a candidate bright line fills the slit band across X, or only
partially intrudes from a neighboring trace.

This is intended as a helper for Step07d/07e reference-slit choice.

Idea
----
For each slit:
1. use Step04 geometry to define the slit band (left/right yellow edges)
2. use the full-frame pixflat-corrected arc image (Step07b)
3. find the brightest line near the slit centerline
4. in a small Y neighborhood around that line:
   - estimate local background + sigma from sidebands
   - mark significant pixels (> bkg + nsig*sigma)
   - require the illuminated segment to cover a large fraction of the slit width
   - optionally require the segment to reach close to both edges
5. report a cleanliness score and PASS/FAIL flag

Outputs
-------
- CSV summary per slit
- optional diagnostic PNGs for selected slits

Run
---
PYTHONPATH=. python qc/step07/helper_arc_trace_fill_score.py --traceset EVEN
PYTHONPATH=. python qc/step07/helper_arc_trace_fill_score.py --traceset ODD
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import config


# ---------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------
DEFAULT_NSIG = 6.0
DEFAULT_Y_HALF = 2              # rows around candidate line => 2 => 5 rows total
DEFAULT_FILL_FRAC = 0.65        # segment must cover >= this fraction of slit width
DEFAULT_EDGE_TOL = 4.0          # significant segment must reach within this many px of both edges
DEFAULT_MIN_GOOD_ROWS = 3       # among (2*y_half+1) rows


def default_arc_path() -> Path:
    st07 = Path(config.ST07_WAVECAL).expanduser()
    if hasattr(config, "MASTER_ARC_DIFF"):
        stem = Path(config.MASTER_ARC_DIFF).stem
        p = st07 / f"{stem}_pixflatcorr_clipped.fits"
        if p.exists():
            return p
    hits = sorted(st07.glob("ArcDiff*_pixflatcorr_clipped.fits"))
    if hits:
        return hits[-1]
    return st07 / "ArcDiff_pixflatcorr_clipped.fits"


def default_geom_path(traceset: str) -> Path:
    st04 = Path(config.ST04_TRACES).expanduser()
    base = "Even_traces_geometry.fits" if traceset == "EVEN" else "Odd_traces_geometry.fits"
    return st04 / base


def poly_from_header(hdr, prefix):
    coeffs = []
    i = 0
    while f"{prefix}{i}" in hdr:
        coeffs.append(float(hdr[f"{prefix}{i}"]))
        i += 1
    return np.array(coeffs, float) if coeffs else None


def poly_eval(coeffs, y):
    out = np.zeros_like(y, dtype=float)
    for i, c in enumerate(coeffs):
        out += c * y**i
    return out


def robust_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return float(np.nanstd(x)) if x.size else 0.0
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    s = 1.4826 * mad
    if not np.isfinite(s) or s <= 0:
        s = np.nanstd(x)
    return float(s) if np.isfinite(s) else 0.0


def list_slit_exts(geom_fits: Path):
    with fits.open(geom_fits) as h:
        return [ext.header.get("EXTNAME", ext.name).strip().upper()
                for ext in h[1:]
                if (ext.header.get("EXTNAME", ext.name).strip().upper().startswith("SLIT"))]


def find_candidate_y(img, hdr):
    ny, nx = img.shape
    ymin = int(hdr.get("YMIN", 0))
    ymax = int(hdr.get("YMAX", ny - 1))
    ymin = max(0, ymin)
    ymax = min(ny - 1, ymax)
    yy = np.arange(ymin, ymax + 1, dtype=float)

    pc = poly_from_header(hdr, "PC")
    lc = poly_from_header(hdr, "LC")
    rc = poly_from_header(hdr, "RC")
    if pc is None or lc is None or rc is None:
        return None

    xc = poly_eval(pc, yy)
    xl = poly_eval(lc, yy)
    xr = poly_eval(rc, yy)

    prof = np.full_like(yy, np.nan, dtype=float)
    for i, y in enumerate(yy.astype(int)):
        xa = max(0, int(np.floor(xl[i])))
        xb = min(nx, int(np.ceil(xr[i] + 1)))
        if xb - xa < 3:
            continue
        row = img[y, xa:xb].astype(float)
        if not np.isfinite(row).any():
            continue
        prof[i] = np.nanmax(row)

    if not np.isfinite(prof).any():
        return None
    return int(yy[np.nanargmax(prof)])


def row_fill_metrics(img_row, xl, xr, nsig):
    """
    Measure whether a significant segment fills the slit band.
    Returns dict with:
      width_model, width_sig, fill_frac, left_gap, right_gap, pass_row
    """
    nx = img_row.size
    xa = max(0, int(np.floor(xl)))
    xb = min(nx, int(np.ceil(xr + 1)))
    if xb - xa < 4:
        return None

    vals = img_row[xa:xb].astype(float)
    if not np.isfinite(vals).any():
        return None

    # local background from the faintest half of pixels inside the band
    vv = vals[np.isfinite(vals)]
    if vv.size < 4:
        return None
    med = np.nanmedian(vv)
    lo = vv[vv <= med]
    bkg = float(np.nanmedian(lo)) if lo.size else float(med)
    sig = robust_sigma(lo if lo.size else vv)
    if not np.isfinite(sig) or sig <= 0:
        sig = max(1.0, float(np.nanstd(vv)))

    thr = bkg + nsig * sig
    on = np.isfinite(vals) & (vals > thr)
    if not np.any(on):
        return {
            "width_model": float(xr - xl),
            "width_sig": 0.0,
            "fill_frac": 0.0,
            "left_gap": np.inf,
            "right_gap": np.inf,
            "bkg": bkg,
            "sigma": sig,
            "thr": thr,
            "pass_row": False,
        }

    idx = np.where(on)[0]
    l = int(idx.min())
    r = int(idx.max())

    width_model = float(xr - xl)
    width_sig = float(r - l + 1)
    fill_frac = width_sig / width_model if width_model > 0 else 0.0

    left_gap = abs((xa + l) - xl)
    right_gap = abs(xr - (xa + r))

    return {
        "width_model": width_model,
        "width_sig": width_sig,
        "fill_frac": fill_frac,
        "left_gap": float(left_gap),
        "right_gap": float(right_gap),
        "bkg": bkg,
        "sigma": sig,
        "thr": thr,
        "pass_row": False,  # set later
    }


def score_one_slit(img, hdr, nsig, y_half, fill_frac_req, edge_tol, min_good_rows):
    ny, nx = img.shape
    y_cand = find_candidate_y(img, hdr)
    if y_cand is None:
        return None

    pc = poly_from_header(hdr, "PC")
    lc = poly_from_header(hdr, "LC")
    rc = poly_from_header(hdr, "RC")
    if pc is None or lc is None or rc is None:
        return None

    rows = []
    for y in range(max(0, y_cand - y_half), min(ny - 1, y_cand + y_half) + 1):
        yy = np.array([float(y)])
        xl = poly_eval(lc, yy)[0]
        xr = poly_eval(rc, yy)[0]
        metr = row_fill_metrics(img[y], xl, xr, nsig=nsig)
        if metr is None:
            continue
        ok = (
            (metr["fill_frac"] >= fill_frac_req) and
            (metr["left_gap"] <= edge_tol) and
            (metr["right_gap"] <= edge_tol)
        )
        metr["pass_row"] = bool(ok)
        metr["y"] = int(y)
        rows.append(metr)

    if not rows:
        return None

    good_rows = sum(1 for r in rows if r["pass_row"])
    frac_med = float(np.median([r["fill_frac"] for r in rows]))
    lg_med = float(np.median([r["left_gap"] for r in rows]))
    rg_med = float(np.median([r["right_gap"] for r in rows]))

    passed = good_rows >= int(min_good_rows)

    # composite cleanliness score
    score = frac_med - 0.1 * (lg_med + rg_med)

    return {
        "y_line": int(y_cand),
        "good_rows": int(good_rows),
        "n_rows": int(len(rows)),
        "fill_frac_med": frac_med,
        "left_gap_med": lg_med,
        "right_gap_med": rg_med,
        "score": float(score),
        "passed": bool(passed),
        "rows": rows,
    }


def save_diag(img, hdr, slit_name, result, outpng):
    y_line = result["y_line"]
    y_half = max(abs(r["y"] - y_line) for r in result["rows"]) if result["rows"] else 2

    ny, nx = img.shape
    ymin = max(0, y_line - 15)
    ymax = min(ny, y_line + 16)

    yy = np.arange(ymin, ymax, dtype=float)
    lc = poly_from_header(hdr, "LC")
    rc = poly_from_header(hdr, "RC")
    pc = poly_from_header(hdr, "PC")

    xl = poly_eval(lc, yy) if lc is not None else None
    xr = poly_eval(rc, yy) if rc is not None else None
    xc = poly_eval(pc, yy) if pc is not None else None

    x0 = int(max(0, np.floor(np.nanmin(xl) - 6))) if xl is not None else 0
    x1 = int(min(nx, np.ceil(np.nanmax(xr) + 7))) if xr is not None else nx

    cut = img[ymin:ymax, x0:x1].astype(float)
    vv = cut[np.isfinite(cut)]
    vmin = np.nanpercentile(vv, 5) if vv.size else 0
    vmax = np.nanpercentile(vv, 99.5) if vv.size else 1

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    ax.imshow(cut, origin="lower", aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)

    if xc is not None:
        ax.plot(xc - x0, yy - ymin, color="cyan", lw=1)
    if xl is not None:
        ax.plot(xl - x0, yy - ymin, color="yellow", lw=0.8)
    if xr is not None:
        ax.plot(xr - x0, yy - ymin, color="yellow", lw=0.8)

    ax.axhline(y_line - ymin, color="red", ls="--", lw=1)
    ax.set_title(
        f"{slit_name}  PASS={result['passed']}  fill={result['fill_frac_med']:.2f}  "
        f"gaps=({result['left_gap_med']:.1f},{result['right_gap_med']:.1f})"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.tight_layout()
    fig.savefig(outpng)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traceset", required=True, choices=["EVEN", "ODD", "even", "odd"])
    ap.add_argument("--arc", type=str, default=None)
    ap.add_argument("--geom", type=str, default=None)
    ap.add_argument("--nsig", type=float, default=DEFAULT_NSIG)
    ap.add_argument("--y-half", type=int, default=DEFAULT_Y_HALF)
    ap.add_argument("--fill-frac", type=float, default=DEFAULT_FILL_FRAC)
    ap.add_argument("--edge-tol", type=float, default=DEFAULT_EDGE_TOL)
    ap.add_argument("--min-good-rows", type=int, default=DEFAULT_MIN_GOOD_ROWS)
    ap.add_argument("--diag-slits", type=str, default="",
                    help="Comma-separated slit names for diagnostic PNGs, e.g. SLIT002,SLIT024")
    args = ap.parse_args()

    traceset = args.traceset.upper()
    arc_path = Path(args.arc).expanduser() if args.arc else default_arc_path()
    geom_path = Path(args.geom).expanduser() if args.geom else default_geom_path(traceset)

    if not arc_path.exists():
        raise FileNotFoundError(arc_path)
    if not geom_path.exists():
        raise FileNotFoundError(geom_path)

    outdir = Path(config.ST07_WAVECAL).expanduser() / f"helper_arc_fill_{traceset.lower()}"
    outdir.mkdir(parents=True, exist_ok=True)

    img = fits.getdata(arc_path).astype(float)

    results = []
    diag_set = {s.strip().upper() for s in args.diag_slits.split(",") if s.strip()}

    with fits.open(geom_path) as hg:
        for ext in hg[1:]:
            slit = (ext.header.get("EXTNAME") or ext.name).strip().upper()
            if not slit.startswith("SLIT"):
                continue

            res = score_one_slit(
                img, ext.header,
                nsig=args.nsig,
                y_half=args.y_half,
                fill_frac_req=args.fill_frac,
                edge_tol=args.edge_tol,
                min_good_rows=args.min_good_rows,
            )
            if res is None:
                continue

            results.append({
                "slit": slit,
                "y_line": res["y_line"],
                "good_rows": res["good_rows"],
                "n_rows": res["n_rows"],
                "fill_frac_med": res["fill_frac_med"],
                "left_gap_med": res["left_gap_med"],
                "right_gap_med": res["right_gap_med"],
                "score": res["score"],
                "passed": res["passed"],
            })

            if slit in diag_set:
                save_diag(img, ext.header, slit, res, outdir / f"{slit}_diag.png")

    results = sorted(results, key=lambda r: r["score"], reverse=True)

    csv_path = outdir / f"arc_fill_score_{traceset}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["slit", "y_line", "good_rows", "n_rows", "fill_frac_med",
                        "left_gap_med", "right_gap_med", "score", "passed"]
        )
        w.writeheader()
        w.writerows(results)

    print("ARC =", arc_path)
    print("GEOM =", geom_path)
    print("Wrote:", csv_path)
    print("\nTop passed slits:")
    shown = 0
    for r in results:
        if r["passed"]:
            print(f"  {r['slit']}: score={r['score']:.3f} fill={r['fill_frac_med']:.2f} "
                  f"gaps=({r['left_gap_med']:.1f},{r['right_gap_med']:.1f}) y={r['y_line']}")
            shown += 1
            if shown >= 10:
                break

    print("\nExample failed slits:")
    shown = 0
    for r in results:
        if not r["passed"]:
            print(f"  {r['slit']}: score={r['score']:.3f} fill={r['fill_frac_med']:.2f} "
                  f"gaps=({r['left_gap_med']:.1f},{r['right_gap_med']:.1f}) y={r['y_line']}")
            shown += 1
            if shown >= 10:
                break


if __name__ == "__main__":
    main()
