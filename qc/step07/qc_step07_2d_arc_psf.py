#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QC Step07 — local 2D arc-line PSF / resolving power.

Measures arc-line widths from narrow local cuts in the 2D arc image,
rather than from full-slit 1D extractions.

Goal:
  Estimate local optical/instrumental LSF before full-slit averaging.

Outputs:
  qc_step07_2d_arc_psf_table.csv
  qc_step07_2d_arc_psf_R_vs_lambda.png
  qc_step07_2d_arc_psf_R_vs_lambda.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from astropy.io import fits

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.optimize import curve_fit

import config

QC_DIR = Path(config.PRODUCT_ROOT) / "qc" / "07_wavecal" / "07h"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arc2d", type=Path, default=Path(config.MASTER_ARC_DIFF_PIXFLATCORR_CLIPPED))
    p.add_argument("--wavesol", type=Path, default=Path(config.WAVESOL_FITS))
    p.add_argument("--geom-even", type=Path, default=Path(config.EVEN_TRACES_GEOM))
    p.add_argument("--geom-odd", type=Path, default=Path(config.ODD_TRACES_GEOM))
    p.add_argument("--outdir", type=Path, default=Path(QC_DIR) / "qc_step07")
    p.add_argument("--half-height", type=int, default=2, help="Spatial half-height of local cut")
    p.add_argument("--half-width", type=int, default=8, help="Spectral half-width for Gaussian fit")
    p.add_argument("--peak-percentile", type=float, default=99.5)
    p.add_argument("--max-lines-per-slit", type=int, default=12)
    p.add_argument("--max-slits", type=int, default=0, help="0 = all")
    return p.parse_args()


def slit_num(name):
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else -1


def is_slit_ext(hdu):
    return (hdu.name or "").upper().startswith("SLIT")


def gauss(x, a, x0, sig, c):
    return a * np.exp(-0.5 * ((x - x0) / sig) ** 2) + c


def poly_from_header(hdr):
    coeff = []
    for i in range(50):
        k = f"WVC{i}"
        if k in hdr:
            coeff.append(float(hdr[k]))
        else:
            break
    if len(coeff) < 2:
        raise RuntimeError("No WVC* polynomial found")
    return np.poly1d(coeff[::-1])


def trace_center_from_geom(geom_hdu, ypix):
    """
    First-pass robust trace center estimate.

    Supports either:
      - CENTER column/table
      - X0/XCENTER column
      - header fallback XREF
    """
    d = geom_hdu.data
    hdr = geom_hdu.header

    if d is not None and hasattr(d, "columns"):
        names = d.columns.names

        for cname in ["CENTER", "XCENTER", "X0"]:
            if cname in names:
                arr = np.asarray(d[cname], float)
                if arr.size == 1:
                    return float(arr[0])
                idx = int(np.clip(round(ypix), 0, arr.size - 1))
                return float(arr[idx])

    for key in ["XREF", "X0", "XCENTER"]:
        if key in hdr:
            return float(hdr[key])

    return np.nan


def fit_line_pixels(xpix, prof):
    ok = np.isfinite(xpix) & np.isfinite(prof)
    x = xpix[ok]
    y = prof[ok]

    if len(x) < 7:
        return None

    c0 = np.nanmedian(y)
    a0 = np.nanmax(y) - c0
    if not np.isfinite(a0) or a0 <= 0:
        return None

    x0 = x[np.nanargmax(y)]
    p0 = [a0, x0, 2.0, c0]

    try:
        popt, _ = curve_fit(
            gauss,
            x,
            y,
            p0=p0,
            bounds=([0, x0 - 5, 0.3, -np.inf], [np.inf, x0 + 5, 10.0, np.inf]),
            maxfev=10000,
        )
    except Exception:
        return None

    a, xc, sig, c = popt
    fwhm_pix = 2.354820045 * abs(sig)

    if not np.isfinite(fwhm_pix) or not (0.5 < fwhm_pix < 20):
        return None

    return xc, fwhm_pix, a, c


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with fits.open(args.wavesol) as hw:
        poly = poly_from_header(hw[0].header)
        ywin0 = float(hw[0].header.get("YWIN0", 0.0))

    rows = []

    with fits.open(args.arc2d, memmap=False) as ha, \
         fits.open(args.geom_even, memmap=False) as ge, \
         fits.open(args.geom_odd, memmap=False) as go:

        arc = np.asarray(ha[0].data, float)

        geom_map = {}
        for h in ge[1:]:
            if is_slit_ext(h):
                geom_map[h.name.upper()] = h
        for h in go[1:]:
            if is_slit_ext(h):
                geom_map[h.name.upper()] = h

        slit_names = sorted(geom_map.keys(), key=slit_num)
        if args.max_slits and args.max_slits > 0:
            slit_names = slit_names[:args.max_slits]

        ny, nx = arc.shape

        for slit in slit_names:
            gh = geom_map[slit]

            y0 = int(round(float(gh.header.get("YMIN", 0))))
            y1 = int(round(float(gh.header.get("YMAX", y0 + 4112))))
            y0 = max(0, y0)
            y1 = min(ny, y1)

            if y1 <= y0 + 20:
                continue

            # Collapse only a narrow spatial cut around local trace center.
            # Sample several y positions to avoid using the full slit.
            y_samples = np.linspace(y0 + 20, y1 - 20, 5).astype(int)

            for ycen in y_samples:
                xcen = trace_center_from_geom(gh, ycen - y0)
                if not np.isfinite(xcen):
                    continue

                # Local spatial slice around trace center.
                x0 = int(round(xcen))
                xlo = max(0, x0 - args.half_height)
                xhi = min(nx, x0 + args.half_height + 1)

                local = arc[y0:y1, xlo:xhi]
                spec = np.nanmedian(local, axis=1)

                ypix = np.arange(y0, y1, dtype=float)

                base = np.nanmedian(spec)
                sp = spec - base
                thresh = np.nanpercentile(sp, args.peak_percentile)

                peaks, _ = find_peaks(sp, height=thresh, distance=15)
                if len(peaks) == 0:
                    continue

                # keep strongest peaks only
                peaks = peaks[np.argsort(sp[peaks])[::-1]][:args.max_lines_per_slit]

                for pk in peaks:
                    lo = max(0, pk - args.half_width)
                    hi = min(len(spec), pk + args.half_width + 1)

                    fit = fit_line_pixels(ypix[lo:hi], spec[lo:hi])
                    if fit is None:
                        continue

                    yfit, fwhm_pix, amp, cont = fit

                    # Convert local pixel width to nm using wavelength polynomial.
                    # Here y_eff follows Step08c convention approximately.
                    y_eff = (yfit - ywin0)
                    lam = float(poly(y_eff))

                    dlam_dpix = float(abs(np.polyder(poly)(y_eff)))
                    fwhm_nm = fwhm_pix * dlam_dpix

                    if not np.isfinite(lam) or not np.isfinite(fwhm_nm) or fwhm_nm <= 0:
                        continue

                    R = lam / fwhm_nm

                    if not (100 < R < 30000):
                        continue

                    rows.append(dict(
                        slit=slit,
                        y_center=float(ycen),
                        x_center=float(xcen),
                        lambda_nm=lam,
                        fwhm_pix=fwhm_pix,
                        dlam_dpix=dlam_dpix,
                        fwhm_nm=fwhm_nm,
                        R=R,
                        amp=amp,
                        cont=cont,
                    ))

    df = pd.DataFrame(rows)
    outcsv = args.outdir / "qc_step07_2d_arc_psf_table.csv"
    df.to_csv(outcsv, index=False)

    print("[OK] Wrote:", outcsv)
    print("Rows:", len(df))

    if df.empty:
        return

    print(df["R"].describe())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["lambda_nm"], df["R"], s=10, alpha=0.35)

    bins = np.linspace(df["lambda_nm"].min(), df["lambda_nm"].max(), 18)
    xm, ym = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (df["lambda_nm"] >= lo) & (df["lambda_nm"] < hi)
        if m.sum() >= 5:
            xm.append(0.5 * (lo + hi))
            ym.append(np.nanmedian(df.loc[m, "R"]))

    if xm:
        ax.plot(xm, ym, lw=2, label="running median")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"Resolving power $R = \lambda/\Delta\lambda$")
    ax.set_title("Step07 QC — local 2D arc-line resolving power")
    ax.grid(True, alpha=0.25)
    ax.legend()

    outpng = args.outdir / "qc_step07_2d_arc_psf_R_vs_lambda.png"
    outpdf = args.outdir / "qc_step07_2d_arc_psf_R_vs_lambda.pdf"

    fig.tight_layout()
    fig.savefig(outpng, dpi=150)
    fig.savefig(outpdf)
    plt.close(fig)

    print("[OK] Wrote:", outpng)
    print("[OK] Wrote:", outpdf)


if __name__ == "__main__":
    main()