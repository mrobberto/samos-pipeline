#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step09a — measure ensemble-relative OH wavelength zero-point shifts.

This stage refines *relative* slit-to-slit wavelength registration using OH sky
emission.  It does not establish an absolute wavelength zero point: the
ensemble median correction is explicitly removed so that the global Step07/08c
arc calibration is preserved.

The reference spectrum is a robust ensemble template built from all usable SKY
spectra, rather than one arbitrarily chosen slit.  Per-window shifts are
measured by cross-correlation, the ensemble template is rebuilt iteratively,
and the final per-slit correction is the correlation-weighted median across OH
windows.

Only shifts passing the configured quality criteria are marked ``use=True``.
Rejected or unavailable shifts are *not* intended to be replaced by an
ensemble median in Step09b; those slits retain their Step07/08c wavelength
vectors unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from astropy.io import fits
import config


DEFAULT_WINDOWS_NM = [
    (780.0, 805.0),
    (806.0, 825.0),
    (845.0, 875.0),
    (875.0, 905.0),
    (905.0, 930.0),
    (930.0, 960.0),
]


def norm_slit(s: str) -> str:
    return str(s).strip().upper()


def is_slit(name: str) -> bool:
    return norm_slit(name).startswith("SLIT")


def slit_num(name: str) -> int:
    try:
        return int(norm_slit(name).replace("SLIT", ""))
    except Exception:
        return 10**9


def robust_zscore(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med))
    sig = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(y)
    if not np.isfinite(sig) or sig <= 0:
        sig = 1.0
    return (y - med) / sig


def sigma_mad(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    return float(1.4826 * np.nanmedian(np.abs(x - med)))


def highpass_running_median(y: np.ndarray, k: int = 101) -> np.ndarray:
    y = np.asarray(y, float)
    n = y.size
    if n == 0:
        return y
    k = max(int(k), 5)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(ypad, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(win, axis=1)
    return y - med


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return np.nan
    x = x[ok]
    w = w[ok]
    s = np.argsort(x)
    x = x[s]
    w = w[s]
    cw = np.cumsum(w)
    return float(x[np.searchsorted(cw, 0.5 * cw[-1])])


def interp_to_grid(lam: np.ndarray, flux: np.ndarray, grid: np.ndarray) -> np.ndarray:
    lam = np.asarray(lam, float)
    flux = np.asarray(flux, float)
    ok = np.isfinite(lam) & np.isfinite(flux)
    if ok.sum() < 20:
        return np.full_like(grid, np.nan, dtype=float)
    s = np.argsort(lam[ok])
    x = lam[ok][s]
    y = flux[ok][s]
    good = np.concatenate([[True], np.diff(x) > 0])
    x = x[good]
    y = y[good]
    if x.size < 20:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, x, y, left=np.nan, right=np.nan)


def xcorr_shift_nm(
    grid: np.ndarray,
    ref: np.ndarray,
    vec: np.ndarray,
    max_shift_nm: float,
) -> tuple[float, float]:
    """Return wavelength shift to ADD to vec wavelengths to align vec to ref."""
    ref = np.asarray(ref, float)
    vec = np.asarray(vec, float)
    ok = np.isfinite(ref) & np.isfinite(vec)
    if ok.sum() < 50:
        return np.nan, np.nan

    dlam = float(np.nanmedian(np.diff(grid)))
    if not np.isfinite(dlam) or dlam <= 0:
        return np.nan, np.nan

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

        x = a[q] - np.nanmean(a[q])
        y = b[q] - np.nanmean(b[q])
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

    # Sub-pixel parabolic refinement around the discrete maximum.
    if 0 < jb < len(corr) - 1 and np.all(np.isfinite(corr[jb - 1:jb + 2])):
        ym, y0, yp = corr[jb - 1:jb + 2]
        den = ym - 2.0 * y0 + yp
        if np.isfinite(den) and abs(den) > 1e-12:
            delta = 0.5 * (ym - yp) / den
            if abs(delta) <= 1.0:
                lag += float(delta)

    return float(-lag * dlam), peak


def shifted_vector(grid: np.ndarray, vec: np.ndarray, shift_nm: float) -> np.ndarray:
    """Evaluate vec after adding shift_nm to its wavelength coordinate."""
    return np.interp(grid - shift_nm, grid, vec, left=np.nan, right=np.nan)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, default="", help="Input extraction FITS (default: config.EXTRACT1D_WAV)")
    ap.add_argument("--outcsv", type=str, default="", help="Output CSV (default: config.OH_SHIFT_CSV)")
    ap.add_argument("--skip_bad", action="store_true", help="Skip slits with S08BAD=1")
    ap.add_argument("--skip_empty", action="store_true", help="Skip slits with S08EMP=1")
    ap.add_argument("--max_shift_nm", type=float, default=1.5, help="Search range +/- nm per window")
    ap.add_argument("--grid_step_nm", type=float, default=0.01, help="Linear grid step in nm")
    ap.add_argument("--min_finite_frac", type=float, default=0.5, help="Min finite SKY fraction")
    ap.add_argument("--hp_width", type=int, default=101, help="Running-median width for continuum removal")
    ap.add_argument("--min_window_peak", type=float, default=0.15, help="Min peak corr for a window")
    ap.add_argument("--min_nwin", type=int, default=3, help="Min number of accepted OH windows")
    ap.add_argument("--min_r_med", type=float, default=0.20, help="Min median window correlation")
    ap.add_argument("--max_shift_std_nm", type=float, default=0.50, help="Max robust per-window shift scatter (nm)")
    ap.add_argument("--max_abs_shift_nm", type=float, default=1.0, help="Hard max |ensemble-relative shift| (nm)")
    ap.add_argument("--iterations", type=int, default=3, help="Template alignment/rebuild iterations")
    return ap.parse_args()


def main():
    args = parse_args()

    st08 = Path(config.ST08_EXTRACT1D)
    st09 = Path(config.ST09_OH_REFINE)
    st09.mkdir(parents=True, exist_ok=True)

    infile = Path(args.infile) if args.infile else Path(
        getattr(config, "EXTRACT1D_WAV", st08 / "extract1d_optimal_ridge_all_wav.fits")
    )
    if not infile.exists():
        raise FileNotFoundError(infile)

    outcsv = Path(args.outcsv) if args.outcsv else Path(
        getattr(config, "OH_SHIFT_CSV", st09 / "oh_shifts.csv")
    )
    outcsv.parent.mkdir(parents=True, exist_ok=True)

    print("INFILE =", infile)
    print("OUTCSV =", outcsv)
    print("METHOD = ENSEMBLE_REL")

    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_slits: list[str] = []

    with fits.open(infile, memmap=False) as h:
        for ext in h[1:]:
            slit = norm_slit(ext.name)
            if not is_slit(slit) or ext.data is None:
                continue
            all_slits.append(slit)
            hdr = ext.header
            if args.skip_bad and int(hdr.get("S08BAD", 0)) == 1:
                continue
            if args.skip_empty and int(hdr.get("S08EMP", 0)) == 1:
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

    all_slits = sorted(set(all_slits), key=slit_num)
    usable_slits = sorted(spectra, key=slit_num)
    if len(usable_slits) < 3:
        raise RuntimeError("Too few slits with usable SKY spectra")

    by_slit: dict[str, list[tuple[float, float]]] = {s: [] for s in usable_slits}

    for lo, hi in DEFAULT_WINDOWS_NM:
        grid = np.arange(lo, hi + args.grid_step_nm, args.grid_step_nm, dtype=float)
        raw: dict[str, np.ndarray] = {}

        for s in usable_slits:
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            template = np.nanmedian(np.vstack([raw[s] for s in sorted(raw, key=slit_num)]), axis=0)

        shifts: dict[str, float] = {s: 0.0 for s in raw}
        peaks: dict[str, float] = {s: np.nan for s in raw}

        for _ in range(max(args.iterations, 1)):
            new_shifts: dict[str, float] = {}
            new_peaks: dict[str, float] = {}
            for s, v in raw.items():
                sh, pk = xcorr_shift_nm(grid, template, v, args.max_shift_nm)
                new_shifts[s] = sh
                new_peaks[s] = pk

            good_center = np.array([
                sh for s, sh in new_shifts.items()
                if np.isfinite(sh)
                and np.isfinite(new_peaks[s])
                and new_peaks[s] >= args.min_window_peak
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
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    template = np.nanmedian(np.vstack(aligned), axis=0)

            shifts, peaks = new_shifts, new_peaks

        for s in usable_slits:
            sh = shifts.get(s, np.nan)
            pk = peaks.get(s, np.nan)
            if np.isfinite(sh) and np.isfinite(pk) and pk >= args.min_window_peak:
                by_slit[s].append((sh, pk))

    rows = []
    for s in all_slits:
        vals = by_slit.get(s, [])
        if vals:
            sh = np.array([v[0] for v in vals], float)
            pk = np.array([v[1] for v in vals], float)
            raw_shift = weighted_median(sh, np.clip(pk, 0.0, None))
            scatter = sigma_mad(sh)
            rmed = float(np.nanmedian(pk))
            nwin = int(len(sh))
        else:
            raw_shift = scatter = rmed = np.nan
            nwin = 0

        rows.append({
            "slit": s,
            "shift_raw_nm": raw_shift,
            "nwin": nwin,
            "shift_std_nm": scatter,
            "r": rmed,
        })

    df = pd.DataFrame(rows)
    pre_good = (
        np.isfinite(df["shift_raw_nm"])
        & np.isfinite(df["r"])
        & np.isfinite(df["shift_std_nm"])
        & (df["nwin"] >= args.min_nwin)
        & (df["r"] >= args.min_r_med)
        & (df["shift_std_nm"] <= args.max_shift_std_nm)
    )

    center = float(np.nanmedian(df.loc[pre_good, "shift_raw_nm"])) if pre_good.any() else 0.0
    df["shift_nm"] = df["shift_raw_nm"] - center
    df["use"] = pre_good & (df["shift_nm"].abs() <= args.max_abs_shift_nm)
    df["ref_slit"] = "ENSEMBLE"
    df["fallback_objraw"] = False
    df["method"] = "ENSEMBLE_REL"
    df["center_removed_nm"] = center

    cols = [
        "slit", "shift_nm", "use", "r", "ref_slit", "fallback_objraw",
        "nwin", "shift_std_nm", "shift_raw_nm", "method", "center_removed_nm",
    ]
    df[cols].to_csv(outcsv, index=False)

    used = df[df["use"] & np.isfinite(df["shift_nm"])].copy()
    print("Usable SKY slits:", len(usable_slits), "/", len(all_slits))
    print("Accepted ensemble-relative shifts:", len(used), "/", len(all_slits))
    print(f"Removed ensemble center: {center:+.4f} nm")
    if len(used):
        x = used["shift_nm"].to_numpy(float)
        print(
            "Accepted shift_nm min/med/max = "
            f"{np.nanmin(x):+.4f} / {np.nanmedian(x):+.4f} / {np.nanmax(x):+.4f} nm"
        )
        print(f"Accepted sigma_MAD = {sigma_mad(x):.4f} nm")
    print("Wrote:", outcsv)
    print("NOTE: shifts are differential only; global Step07/08c zero point is preserved.")


if __name__ == "__main__":
    main()
