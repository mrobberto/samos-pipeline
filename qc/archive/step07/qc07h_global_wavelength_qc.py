#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
import config


def robust_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def robust_limits(img, p_lo=2.0, p_hi=98.0):
    arr = np.asarray(img, float)
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return 0.0, 1.0
    lo = np.nanpercentile(v, p_lo)
    hi = np.nanpercentile(v, p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        med = np.nanmedian(v)
        sig = robust_sigma(v)
        if not np.isfinite(sig) or sig <= 0:
            sig = 1.0
        return med - 2 * sig, med + 4 * sig
    return float(lo), float(hi)


def first_existing(paths):
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def norm1(y):
    y = np.asarray(y, float)
    if not np.any(np.isfinite(y)):
        return y * np.nan
    med = np.nanmedian(y)
    yy = y - med
    mx = np.nanmax(np.abs(yy))
    if not np.isfinite(mx) or mx <= 0:
        return yy
    return yy / mx


def weighted_centroid_1d(y, x0=None, halfwin=8):
    y = np.asarray(y, float)
    n = y.size
    x = np.arange(n, dtype=float)

    if x0 is None or not np.isfinite(x0):
        if not np.any(np.isfinite(y)):
            return np.nan
        x0 = float(np.nanargmax(y))

    i0 = int(round(x0))
    a = max(0, i0 - halfwin)
    b = min(n, i0 + halfwin + 1)

    seg = y[a:b].astype(float)
    xx = x[a:b]

    if not np.any(np.isfinite(seg)):
        return np.nan

    base = np.nanmedian(seg)
    w = seg - base
    w[~np.isfinite(w)] = 0.0
    w[w < 0] = 0.0

    s = np.sum(w)
    if s <= 0:
        return np.nan

    return float(np.sum(xx * w) / s)


def score_row(r):
    terms = [
        0.0 if not np.isfinite(r["LINE_RESID_NM"]) else abs(r["LINE_RESID_NM"]),
        0.0 if r["MONOTONIC_OK"] else 5.0,
        0.0 if not np.isfinite(r["FRAC_FINITE"]) else 2.0 * (1.0 - r["FRAC_FINITE"]),
    ]
    return float(np.sum(terms))


def pick_input_fits(explicit=None):
    st07 = Path(config.ST07_WAVECAL)
    return first_existing([
        Path(explicit).expanduser() if explicit else None,
        st07 / "arc_1d_wavelength_all.fits",
        *sorted(st07.glob("*_1D_wavelength_ALL.fits")),
        *sorted(st07.glob("*wavelength*ALL*.fits")),
    ])


def finish_figure(show_plots: bool, outfile: Path | None = None):
    if outfile is not None:
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Global quantitative QC for Step07 wavelength products")
    ap.add_argument("--fits", default=None, help="Optional explicit input wavelength-ALL FITS")
    ap.add_argument("--write-csv", action="store_true", help="Write ranked CSV summary")
    ap.add_argument("--n-show", type=int, default=12, help="How many worst slits to show")
    ap.add_argument("--show-plots", action="store_true", help="Show plots interactively")
    ap.add_argument("--outdir", default=None, help="Optional output directory")
    args = ap.parse_args()

    st07 = Path(config.ST07_WAVECAL)
    fits_path = pick_input_fits(args.fits)
    if fits_path is None:
        raise FileNotFoundError(f"Could not find Step07 wavelength-ALL FITS in {st07}")

    outdir = Path(args.outdir).expanduser() if args.outdir else (st07 / "qc_step07h")
    outdir.mkdir(parents=True, exist_ok=True)

    with fits.open(fits_path) as hdul:
        slits = []
        for h in hdul[1:]:
            nm = (h.header.get("EXTNAME") or h.name or "").strip().upper()
            if nm.startswith("SLIT") and h.data is not None:
                data = np.array(h.data, dtype=float)
                hdr = h.header.copy()
                slits.append((nm, data, hdr))

    if not slits:
        raise RuntimeError("No SLIT### extensions found")

    rows = []
    lam_min_all = []
    lam_max_all = []

    for nm, arr, hdr in slits:
        if arr.ndim < 2 or arr.shape[0] < 2:
            raise ValueError(f"{nm}: expected at least 2 rows [flux, lambda], got shape {arr.shape}")

        flux = np.asarray(arr[0], float)
        lam = np.asarray(arr[1], float)

        finite = np.isfinite(flux) & np.isfinite(lam)
        frac_finite = float(np.mean(finite)) if lam.size else np.nan

        lam_good = lam[np.isfinite(lam)]
        if lam_good.size > 1:
            dlam = np.diff(lam_good)
            monotonic_ok = bool(np.all(dlam > 0) or np.all(dlam < 0))
            lam_min = float(np.nanmin(lam_good))
            lam_max = float(np.nanmax(lam_good))
            lam_span = lam_max - lam_min
        else:
            monotonic_ok = False
            lam_min = np.nan
            lam_max = np.nan
            lam_span = np.nan

        lam_min_all.append(lam_min)
        lam_max_all.append(lam_max)

        rows.append({
            "SLIT": nm,
            "FLUX": flux,
            "LAMBDA": lam,
            "FRAC_FINITE": frac_finite,
            "MONOTONIC_OK": monotonic_ok,
            "LAM_MIN": lam_min,
            "LAM_MAX": lam_max,
            "LAM_SPAN": lam_span,
            "SLITID": hdr.get("SLITID", np.nan),
            "INDEX": hdr.get("INDEX", np.nan),
            "RA": hdr.get("RA", ""),
            "DEC": hdr.get("DEC", ""),
            "SHIFT_P": hdr.get("SHIFT_P", np.nan),
        })

    lam_min_global = np.nanmax(lam_min_all)
    lam_max_global = np.nanmin(lam_max_all)
    if not np.isfinite(lam_min_global) or not np.isfinite(lam_max_global) or lam_max_global <= lam_min_global:
        raise RuntimeError("No valid common wavelength overlap across slits")

    n_pix_med = int(np.nanmedian([len(r["LAMBDA"]) for r in rows]))
    n_grid = max(1000, n_pix_med)
    lam_grid = np.linspace(lam_min_global, lam_max_global, n_grid)

    stack = []
    for r in rows:
        flux = r["FLUX"]
        lam = r["LAMBDA"]

        m = np.isfinite(flux) & np.isfinite(lam)
        if np.sum(m) < 10:
            interp = np.full_like(lam_grid, np.nan, dtype=float)
        else:
            lam_use = lam[m]
            flux_use = flux[m]
            order = np.argsort(lam_use)
            lam_use = lam_use[order]
            flux_use = flux_use[order]

            uniq = np.diff(lam_use, prepend=lam_use[0] - 1e-9) > 0
            lam_use = lam_use[uniq]
            flux_use = flux_use[uniq]

            if lam_use.size < 10:
                interp = np.full_like(lam_grid, np.nan, dtype=float)
            else:
                interp = np.interp(lam_grid, lam_use, flux_use, left=np.nan, right=np.nan)

        r["FLUX_GRID"] = interp
        stack.append(interp)

    stack = np.asarray(stack, float)
    master = np.nanmedian(stack, axis=0)

    xpk = float(np.nanargmax(master))
    for r in rows:
        fg = r["FLUX_GRID"]
        cen = weighted_centroid_1d(fg, xpk, halfwin=8)
        if np.isfinite(cen):
            r["LINE_CEN_IDX"] = cen
            r["LINE_CEN_NM"] = np.interp(cen, np.arange(n_grid), lam_grid)
            r["LINE_RESID_NM"] = r["LINE_CEN_NM"] - np.interp(xpk, np.arange(n_grid), lam_grid)
        else:
            r["LINE_CEN_IDX"] = np.nan
            r["LINE_CEN_NM"] = np.nan
            r["LINE_RESID_NM"] = np.nan

        r["SCORE"] = score_row(r)

    rows_sorted = sorted(rows, key=lambda r: r["SCORE"], reverse=True)

    v1, v2 = robust_limits(stack, p_lo=2, p_hi=99)
    vr1, vr2 = robust_limits(stack - master[None, :], p_lo=2, p_hi=98)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(stack, origin="lower", aspect="auto", vmin=v1, vmax=v2, cmap="gray",
                     extent=[lam_grid[0], lam_grid[-1], 0, len(rows)])
    ax1.set_title("All slits on common wavelength grid")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Slit index")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(stack - master[None, :], origin="lower", aspect="auto", vmin=vr1, vmax=vr2, cmap="gray",
                     extent=[lam_grid[0], lam_grid[-1], 0, len(rows)])
    ax2.set_title("Residuals vs median master")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Slit index")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(lam_grid, master, color="black", lw=1.2)
    ax3.set_title("Median master spectrum")
    ax3.set_xlabel("Wavelength (nm)")
    ax3.set_ylabel("Flux")

    idx = np.arange(len(rows_sorted))
    line_res = np.array([r["LINE_RESID_NM"] for r in rows_sorted], float)
    frac_fin = np.array([r["FRAC_FINITE"] for r in rows_sorted], float)
    span = np.array([r["LAM_SPAN"] for r in rows_sorted], float)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(idx, line_res, "o", ms=3)
    ax4.axhline(0, color="0.6", lw=0.8)
    ax4.set_title("Line centroid residual")
    ax4.set_xlabel("Ranked slit index")
    ax4.set_ylabel("Residual (nm)")

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(idx, frac_fin, "o", ms=3)
    ax5.set_title("Finite fraction")
    ax5.set_xlabel("Ranked slit index")
    ax5.set_ylabel("Frac finite")

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(idx, span, "o", ms=3)
    ax6.set_title("Wavelength span per slit")
    ax6.set_xlabel("Ranked slit index")
    ax6.set_ylabel("Span (nm)")

    fig.suptitle(
        f"QC75 companion — global Step07 check  |  "
        f"N slits={len(rows)}  |  "
        f"common overlap = {lam_min_global:.2f}–{lam_max_global:.2f} nm  |  "
        f"line-resid sigma = {robust_sigma(line_res):.4f} nm",
        fontsize=13
    )
    finish_figure(args.show_plots, outdir / f"{fits_path.stem}_qc75_global_summary.png")

    nshow = min(args.n_show, len(rows_sorted))
    fig2 = plt.figure(figsize=(15, 3.2 * int(np.ceil(nshow / 3))))
    gs2 = fig2.add_gridspec(int(np.ceil(nshow / 3)), 3, hspace=0.35, wspace=0.25)

    master_n = norm1(master)

    for i, r in enumerate(rows_sorted[:nshow], start=1):
        fg_n = norm1(r["FLUX_GRID"])

        ax = fig2.add_subplot(gs2[(i - 1) // 3, (i - 1) % 3])
        ax.plot(lam_grid, fg_n, color="tab:blue", lw=1.0, label=r["SLIT"])
        ax.plot(lam_grid, master_n, color="black", lw=1.0, label="master")
        ax.set_title(
            f"{r['SLIT']}\n"
            f"resid={r['LINE_RESID_NM']:.4f} nm  "
            f"finite={r['FRAC_FINITE']:.3f}\n"
            f"mono={r['MONOTONIC_OK']}  span={r['LAM_SPAN']:.2f}",
            fontsize=8
        )
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("norm flux")
        if i == 1:
            ax.legend(fontsize=7)

    fig2.suptitle("Worst-slit overlays in wavelength space", fontsize=13)
    finish_figure(args.show_plots, outdir / f"{fits_path.stem}_qc75_worst_slits.png")

    print()
    print("====================================================")
    print("QC75 companion — global Step07 check")
    print("====================================================")
    print("File:", fits_path)
    print(f"N slits                  = {len(rows)}")
    print(f"Common overlap nm        = {lam_min_global:.6f} .. {lam_max_global:.6f}")
    print(f"Median span nm           = {np.nanmedian(span):.6f}")
    print(f"Line-residual sigma nm   = {robust_sigma(line_res):.6f}")
    print()

    print("Worst slits:")
    print("SLIT      SLITID  SHIFT_P  RESID_NM   FRAC_FIN   MONO   LAM_MIN   LAM_MAX   SPAN")
    for r in rows_sorted[:min(15, len(rows_sorted))]:
        print(
            f"{r['SLIT']:8s}  "
            f"{str(r['SLITID']):>6s}  "
            f"{str(r['SHIFT_P']):>7s}  "
            f"{r['LINE_RESID_NM']:9.4f}  "
            f"{r['FRAC_FINITE']:8.3f}  "
            f"{str(r['MONOTONIC_OK']):>5s}  "
            f"{r['LAM_MIN']:8.3f}  "
            f"{r['LAM_MAX']:8.3f}  "
            f"{r['LAM_SPAN']:8.3f}"
        )
    print("====================================================")

    if args.write_csv:
        out_csv = outdir / "QC75_companion_ranked.csv"
        keys = [k for k in rows_sorted[0].keys() if k not in ("FLUX", "LAMBDA", "FLUX_GRID")] if rows_sorted else []
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows_sorted:
                rr = {k: r[k] for k in keys}
                w.writerow(rr)
        print("Wrote:", out_csv)


if __name__ == "__main__":
    main()