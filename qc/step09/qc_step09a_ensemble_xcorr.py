#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QC-only ensemble OH wavelength-shift diagnostic for corrected Step08c spectra.

Purpose
-------
Measure *relative* per-slit wavelength offsets from OH sky structure without
anchoring the result to one arbitrarily chosen slit.  This diagnostic does not
modify any wavelength vector and does not write a Step09b correction product.

Method
------
For each OH window, each slit SKY spectrum is interpolated onto a common grid,
high-pass filtered with a running median, and robustly normalized.  A robust
ensemble template is built from the median of all usable slits.  Individual
slits are cross-correlated against that template, shifted, and the template is
rebuilt iteratively.  Per-slit shifts are then combined across windows with a
correlation-weighted median.

The final per-slit corrections are centered so that the median high-quality
shift is exactly zero.  Therefore ``shift_rel_nm`` is a *differential* shift to
add to LAMBDA_NM if one later chooses to correct slit-to-slit offsets.  This QC
does not determine a global/absolute wavelength zero point.

Outputs
-------
- qc_step09a_ensemble_xcorr.csv
- qc_step09a_ensemble_xcorr.png

No FITS product is changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


WINDOWS_NM = [
    (780.0, 805.0),
    (806.0, 825.0),
    (845.0, 875.0),
    (875.0, 905.0),
    (905.0, 930.0),
    (930.0, 960.0),
]


def parse_args():
    p = argparse.ArgumentParser(description="QC ensemble-relative OH wavelength shifts")
    p.add_argument("--infile", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--grid-step-nm", type=float, default=0.01)
    p.add_argument("--max-shift-nm", type=float, default=1.5)
    p.add_argument("--hp-width", type=int, default=101)
    p.add_argument("--min-finite-frac", type=float, default=0.5)
    p.add_argument("--min-window-peak", type=float, default=0.15)
    p.add_argument("--min-nwin", type=int, default=3)
    p.add_argument("--min-r-med", type=float, default=0.20)
    p.add_argument("--max-window-scatter-nm", type=float, default=0.50)
    p.add_argument("--max-abs-rel-nm", type=float, default=1.0)
    p.add_argument("--iterations", type=int, default=3)
    return p.parse_args()


def robust_zscore(y):
    y = np.asarray(y, float)
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med))
    sig = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(y)
    if not np.isfinite(sig) or sig <= 0:
        sig = 1.0
    return (y - med) / sig


def highpass_running_median(y, k=101):
    y = np.asarray(y, float)
    if y.size == 0:
        return y
    k = max(int(k), 5)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(ypad, k)
    with np.errstate(all="ignore"):
        med = np.nanmedian(win, axis=1)
    return y - med


def interp_to_grid(lam, flux, grid):
    lam = np.asarray(lam, float)
    flux = np.asarray(flux, float)
    ok = np.isfinite(lam) & np.isfinite(flux)
    if ok.sum() < 20:
        return np.full_like(grid, np.nan, dtype=float)
    x = lam[ok]
    y = flux[ok]
    s = np.argsort(x)
    x = x[s]
    y = y[s]
    good = np.concatenate([[True], np.diff(x) > 0])
    x = x[good]
    y = y[good]
    if x.size < 20:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, x, y, left=np.nan, right=np.nan)


def xcorr_shift_nm(grid, ref, vec, max_shift_nm):
    """Return wavelength shift to ADD to vec wavelengths to align vec to ref."""
    ref = np.asarray(ref, float)
    vec = np.asarray(vec, float)
    ok = np.isfinite(ref) & np.isfinite(vec)
    if ok.sum() < 50:
        return np.nan, np.nan
    dlam = float(np.nanmedian(np.diff(grid)))
    max_lag = max(int(round(max_shift_nm / dlam)), 1)
    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    corr = np.full(lags.shape, np.nan, dtype=float)
    for j, lag in enumerate(lags):
        if lag < 0:
            a = ref[-lag:]
            b = vec[:len(vec) + lag]
            q = ok[-lag:] & ok[:len(ok) + lag]
        elif lag > 0:
            a = ref[:len(ref) - lag]
            b = vec[lag:]
            q = ok[:len(ok) - lag] & ok[lag:]
        else:
            a, b, q = ref, vec, ok
        if q.sum() < 50:
            continue
        x = a[q]
        y = b[q]
        x = x - np.nanmean(x)
        y = y - np.nanmean(y)
        sx = np.nanstd(x)
        sy = np.nanstd(y)
        if not np.isfinite(sx) or not np.isfinite(sy) or sx <= 0 or sy <= 0:
            continue
        corr[j] = float(np.nanmean((x / sx) * (y / sy)))
    if not np.any(np.isfinite(corr)):
        return np.nan, np.nan
    jb = int(np.nanargmax(corr))
    lag = float(lags[jb])
    peak = float(corr[jb])
    # sub-pixel parabola where possible
    if 0 < jb < len(corr) - 1 and np.all(np.isfinite(corr[jb-1:jb+2])):
        ym, y0, yp = corr[jb-1:jb+2]
        den = (ym - 2.0*y0 + yp)
        if np.isfinite(den) and abs(den) > 1e-12:
            delta = 0.5 * (ym - yp) / den
            if abs(delta) <= 1.0:
                lag += float(delta)
    return float(-lag * dlam), peak


def weighted_median(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return np.nan
    x = x[ok]
    w = w[ok]
    s = np.argsort(x)
    x = x[s]
    w = w[s]
    cw = np.cumsum(w)
    return float(x[np.searchsorted(cw, 0.5 * cw[-1])])


def sigma_mad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    return float(1.4826 * np.nanmedian(np.abs(x - med)))


def shifted_vector(grid, vec, shift_nm):
    """Evaluate vec after adding shift_nm to its wavelength coordinate."""
    # New wavelength = old wavelength + shift, so value at output grid g comes
    # from old coordinate g-shift.
    return np.interp(grid - shift_nm, grid, vec, left=np.nan, right=np.nan)


def main():
    args = parse_args()
    if not args.infile.exists():
        raise FileNotFoundError(args.infile)
    args.outdir.mkdir(parents=True, exist_ok=True)

    spectra = {}
    with fits.open(args.infile, memmap=False) as h:
        for ext in h[1:]:
            slit = (ext.name or "").upper()
            if not slit.startswith("SLIT") or ext.data is None:
                continue
            names = [n.upper() for n in ext.columns.names]
            if "LAMBDA_NM" not in names or "SKY" not in names:
                continue
            lam = np.asarray(ext.data["LAMBDA_NM"], float)
            sky = np.asarray(ext.data["SKY"], float)
            finite_frac = np.isfinite(sky).sum() / max(len(sky), 1)
            if finite_frac < args.min_finite_frac:
                continue
            spectra[slit] = (lam, sky)

    slits = sorted(spectra)
    if len(slits) < 3:
        raise RuntimeError("Too few slits with usable SKY spectra")

    # Store per-window final shift and peak for each slit.
    by_slit = {s: [] for s in slits}

    for iw, (lo, hi) in enumerate(WINDOWS_NM):
        grid = np.arange(lo, hi + args.grid_step_nm, args.grid_step_nm, dtype=float)
        raw = {}
        for s in slits:
            lam, sky = spectra[s]
            v = interp_to_grid(lam, sky, grid)
            q = np.isfinite(v)
            if q.sum() < int(args.min_finite_frac * len(grid)):
                continue
            hp = highpass_running_median(v, args.hp_width)
            z = np.full_like(hp, np.nan)
            z[q] = robust_zscore(hp[q])
            raw[s] = z
        if len(raw) < 3:
            continue

        stack = np.vstack([raw[s] for s in sorted(raw)])
        with np.errstate(all="ignore"):
            template = np.nanmedian(stack, axis=0)

        shifts = {s: 0.0 for s in raw}
        peaks = {s: np.nan for s in raw}
        for _ in range(max(args.iterations, 1)):
            new_shifts = {}
            new_peaks = {}
            for s, v in raw.items():
                sh, pk = xcorr_shift_nm(grid, template, v, args.max_shift_nm)
                new_shifts[s] = sh
                new_peaks[s] = pk
            good_center = np.array([
                sh for s, sh in new_shifts.items()
                if np.isfinite(sh) and np.isfinite(new_peaks[s]) and new_peaks[s] >= args.min_window_peak
            ], float)
            center = float(np.nanmedian(good_center)) if good_center.size else 0.0
            for s in new_shifts:
                if np.isfinite(new_shifts[s]):
                    new_shifts[s] -= center
            aligned = []
            for s, v in raw.items():
                sh = new_shifts[s]
                pk = new_peaks[s]
                if np.isfinite(sh) and np.isfinite(pk) and pk >= args.min_window_peak:
                    aligned.append(shifted_vector(grid, v, sh))
            if len(aligned) >= 3:
                with np.errstate(all="ignore"):
                    template = np.nanmedian(np.vstack(aligned), axis=0)
            shifts, peaks = new_shifts, new_peaks

        for s in slits:
            sh = shifts.get(s, np.nan)
            pk = peaks.get(s, np.nan)
            by_slit[s].append((iw, sh, pk))

    rows = []
    for s in slits:
        vals = [(sh, pk) for _, sh, pk in by_slit[s]
                if np.isfinite(sh) and np.isfinite(pk) and pk >= args.min_window_peak]
        if vals:
            sh = np.array([v[0] for v in vals], float)
            pk = np.array([v[1] for v in vals], float)
            comb = weighted_median(sh, np.clip(pk, 0, None))
            scat = sigma_mad(sh)
            rmed = float(np.nanmedian(pk))
            nwin = len(sh)
        else:
            comb = scat = rmed = np.nan
            nwin = 0
        rows.append(dict(slit=s, shift_raw_nm=comb, nwin=nwin,
                         window_scatter_nm=scat, r_med=rmed))

    df = pd.DataFrame(rows)
    pre_good = (
        np.isfinite(df.shift_raw_nm)
        & np.isfinite(df.r_med)
        & np.isfinite(df.window_scatter_nm)
        & (df.nwin >= args.min_nwin)
        & (df.r_med >= args.min_r_med)
        & (df.window_scatter_nm <= args.max_window_scatter_nm)
    )
    center = float(np.nanmedian(df.loc[pre_good, "shift_raw_nm"])) if pre_good.any() else 0.0
    df["shift_rel_nm"] = df["shift_raw_nm"] - center
    df["use"] = pre_good & (df.shift_rel_nm.abs() <= args.max_abs_rel_nm)
    df["ensemble_center_removed_nm"] = center

    outcsv = args.outdir / "qc_step09a_ensemble_xcorr.csv"
    df.to_csv(outcsv, index=False)

    used = df[df.use].copy()
    if len(used):
        med = float(np.nanmedian(used.shift_rel_nm))
        sig = sigma_mad(used.shift_rel_nm)
        print(f"Usable SKY slits: {len(df)}")
        print(f"Accepted ensemble-relative shifts: {len(used)}")
        print(f"Removed ensemble center: {center:+.4f} nm")
        print(f"Accepted shift_rel nm min/med/max = {used.shift_rel_nm.min():+.4f} / {med:+.4f} / {used.shift_rel_nm.max():+.4f}")
        print(f"Accepted sigma_MAD = {sig:.4f} nm")
        print("N |shift_rel| >= 0.25/0.50/0.75 nm =",
              *[int((used.shift_rel_nm.abs() >= q).sum()) for q in (0.25, 0.50, 0.75)])
        print("Largest accepted residuals:")
        print(used.assign(abs_shift=used.shift_rel_nm.abs()).sort_values("abs_shift", ascending=False)
              [["slit", "shift_rel_nm", "r_med", "nwin", "window_scatter_nm"]]
              .head(15).to_string(index=False))
    else:
        print("No shifts passed ensemble QC thresholds")

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    ax.axhline(0.0, linewidth=1.0)
    ax.scatter(x[~df.use], df.loc[~df.use, "shift_rel_nm"], marker="x", label="rejected")
    ax.scatter(x[df.use], df.loc[df.use, "shift_rel_nm"], label="accepted")
    ax.set_xticks(x)
    ax.set_xticklabels(df.slit, rotation=90, fontsize=7)
    ax.set_ylabel("Ensemble-relative OH shift to add [nm]")
    ax.set_xlabel("Slit")
    ax.set_title("Step09a QC — ensemble-relative OH wavelength shifts (median centered to zero)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    outpng = args.outdir / "qc_step09a_ensemble_xcorr.png"
    fig.savefig(outpng, dpi=160)
    plt.close(fig)

    print("Wrote:", outcsv)
    print("Wrote:", outpng)
    print("NOTE: differential QC only; no absolute/global wavelength correction is inferred or applied.")


if __name__ == "__main__":
    main()
