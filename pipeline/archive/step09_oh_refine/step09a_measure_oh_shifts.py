#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step09a — measure per-slit OH wavelength zero-point shifts from sky emission lines.

CLEANED FOR NEW PIPELINE SEMANTICS
----------------------------------
Pipeline meaning is now:
  Step09 = OH refine
  Step10 = telluric

Science logic preserved from the prior working Step10a script.
Only plumbing/default-path behavior has been updated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


def highpass_running_median(y: np.ndarray, k: int = 101) -> np.ndarray:
    y = np.asarray(y, float)
    n = y.size
    if n == 0:
        return y
    k = int(k)
    if k < 5:
        return y.copy()
    if k % 2 == 0:
        k += 1
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        win = sliding_window_view(ypad, k)
        med = np.nanmedian(win, axis=1)
    except Exception:
        med = np.empty(n, float)
        for i in range(n):
            med[i] = np.nanmedian(ypad[i:i + k])
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
    dx = np.diff(x)
    good = np.concatenate([[True], dx > 0])
    x = x[good]
    y = y[good]
    if x.size < 20:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, x, y, left=np.nan, right=np.nan)


def xcorr_shift_nm(grid: np.ndarray, a: np.ndarray, b: np.ndarray, max_shift_nm: float) -> tuple[float, float]:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
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
            a1 = a[-lag:]
            b1 = b[:len(b) + lag]
            ok1 = ok[-lag:] & ok[:len(ok) + lag]
        elif lag > 0:
            a1 = a[:len(a) - lag]
            b1 = b[lag:]
            ok1 = ok[:len(ok) - lag] & ok[lag:]
        else:
            a1 = a
            b1 = b
            ok1 = ok

        if ok1.sum() < 50:
            continue

        x = a1[ok1]
        y = b1[ok1]
        x = x - np.nanmean(x)
        y = y - np.nanmean(y)
        sx = np.nanstd(x)
        sy = np.nanstd(y)
        if not np.isfinite(sx) or not np.isfinite(sy) or sx <= 0 or sy <= 0:
            continue
        x /= sx
        y /= sy
        corr[j] = float(np.nanmean(x * y))

    if not np.any(np.isfinite(corr)):
        return np.nan, np.nan

    jbest = int(np.nanargmax(corr))
    lag_best = int(lags[jbest])
    peak = float(corr[jbest])
    return float(-lag_best * dlam), peak


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, default="", help="Input extraction FITS (default: config.EXTRACT1D_WAV)")
    ap.add_argument("--outcsv", type=str, default="", help="Output CSV (default: config.OH_SHIFT_CSV)")
    ap.add_argument("--ref", type=str, default="SLIT027", help="Reference slit, or AUTO")
    ap.add_argument("--auto_ref", action="store_true", help="Auto-pick reference slit with strongest OH")
    ap.add_argument("--use_objraw_fallback", action="store_true", help="Fallback to OBJ_RAW if SKY is too empty")
    ap.add_argument("--skip_bad", action="store_true", help="Skip slits with S08BAD=1")
    ap.add_argument("--skip_empty", action="store_true", help="Skip slits with S08EMP=1")
    ap.add_argument("--max_shift_nm", type=float, default=2.0, help="Search range +/- nm per window")
    ap.add_argument("--grid_step_nm", type=float, default=0.01, help="Linear grid step in nm")
    ap.add_argument("--min_finite_frac", type=float, default=0.5, help="Min finite frac in window to use")
    ap.add_argument("--hp_width", type=int, default=101, help="Running-median width for continuum removal")
    ap.add_argument("--min_peak_corr", type=float, default=0.18, help="Min median peak corr across windows")
    ap.add_argument("--min_nwin", type=int, default=2, help="Min number of OH windows required")
    ap.add_argument("--max_shift_std_nm", type=float, default=0.15, help="Max std of per-window shifts")
    ap.add_argument("--max_abs_shift_nm", type=float, default=1.2, help="Hard max |shift| (nm) to accept")
    ap.add_argument("--reject_edge_shift_nm", type=float, default=1.95, help="Reject if |shift| near search boundary")
    return ap.parse_args()


def main():
    args = parse_args()

    st08 = Path(config.ST08_EXTRACT1D)
    st09 = Path(config.ST09_OH_REFINE)
    st09.mkdir(parents=True, exist_ok=True)

    infile = Path(args.infile) if args.infile else Path(getattr(config, "EXTRACT1D_WAV", st08 / "extract1d_optimal_ridge_all_wav.fits"))
    if not infile.exists():
        raise FileNotFoundError(infile)

    outcsv = Path(args.outcsv) if args.outcsv else Path(getattr(config, "OH_SHIFT_CSV", st09 / "oh_shifts.csv"))

    ref_slit = norm_slit(args.ref)
    want_auto = args.auto_ref or (ref_slit == "AUTO")

    print("INFILE =", infile)
    print("OUTCSV =", outcsv)
    print("REF    =", ref_slit)

    with fits.open(infile) as h:
        slits = sorted([x.name.upper() for x in h[1:] if is_slit(x.name)], key=slit_num)
        if not slits:
            raise RuntimeError(f"No SLIT* extensions found in {infile}")

        def _skip_slit(slitname: str):
            hdr = h[slitname].header
            if args.skip_bad and int(hdr.get("S08BAD", 0)) == 1:
                return True
            if args.skip_empty and int(hdr.get("S08EMP", 0)) == 1:
                return True
            return False

        def _get_sky(tab):
            sky = np.asarray(tab["SKY"], float)
            if args.use_objraw_fallback and "OBJ_RAW" in tab.columns.names:
                fsky = np.isfinite(sky).sum() / max(sky.size, 1)
                if fsky < 0.5:
                    sky = np.asarray(tab["OBJ_RAW"], float)
            return sky

        if want_auto or (ref_slit not in slits):
            scores = []
            for s in slits:
                if _skip_slit(s):
                    continue
                tab = h[s].data
                if "LAMBDA_NM" not in tab.columns.names or "SKY" not in tab.columns.names:
                    continue
                lam = np.asarray(tab["LAMBDA_NM"], float)
                sky = _get_sky(tab)
                score = 0.0
                nwin_ok = 0
                for lo, hi in DEFAULT_WINDOWS_NM:
                    grid = np.arange(lo, hi + args.grid_step_nm, args.grid_step_nm, dtype=float)
                    v = interp_to_grid(lam, sky, grid)
                    ok = np.isfinite(v)
                    if ok.sum() < int(args.min_finite_frac * grid.size):
                        continue
                    vv = highpass_running_median(v, k=args.hp_width)
                    z = robust_zscore(vv[ok])
                    if np.any(np.isfinite(z)):
                        score += float(np.nanmedian(np.abs(z)))
                    nwin_ok += 1
                scores.append((score, nwin_ok, s))
            if not scores:
                raise RuntimeError("Could not select OH reference slit")
            scores_ok = [t for t in scores if t[1] >= max(args.min_nwin, 1)]
            scores_use = scores_ok if scores_ok else scores
            scores_use.sort(key=lambda t: (t[0], t[1]))
            ref_slit = scores_use[-1][2]
            print(f"[INFO] Auto reference slit = {ref_slit}")
        elif ref_slit not in slits:
            raise KeyError(f"Reference slit {ref_slit} not found")

        tref = h[ref_slit].data
        lam_ref = np.asarray(tref["LAMBDA_NM"], float)
        sky_ref = _get_sky(tref)
        rows = []

        for s in slits:
            if _skip_slit(s):
                rows.append({"slit": s, "shift_nm": 0.0, "use": False, "r": np.nan, "ref_slit": ref_slit,
                             "fallback_objraw": False, "nwin": 0, "shift_std_nm": np.nan})
                continue

            tab = h[s].data
            if "LAMBDA_NM" not in tab.columns.names or "SKY" not in tab.columns.names:
                rows.append({"slit": s, "shift_nm": 0.0, "use": False, "r": np.nan, "ref_slit": ref_slit,
                             "fallback_objraw": False, "nwin": 0, "shift_std_nm": np.nan})
                continue

            lam = np.asarray(tab["LAMBDA_NM"], float)
            sky0 = np.asarray(tab["SKY"], float)
            sky = _get_sky(tab)
            use_fallback = bool(args.use_objraw_fallback and ("OBJ_RAW" in tab.columns.names) and
                                (np.isfinite(sky0).sum() / max(sky0.size, 1) < 0.5))

            shifts = []
            weights = []
            for lo, hi in DEFAULT_WINDOWS_NM:
                grid = np.arange(lo, hi + args.grid_step_nm, args.grid_step_nm, dtype=float)
                a = interp_to_grid(lam_ref, sky_ref, grid)
                b = interp_to_grid(lam, sky, grid)
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() < int(args.min_finite_frac * grid.size):
                    continue
                a_hp = highpass_running_median(a, k=args.hp_width)
                b_hp = highpass_running_median(b, k=args.hp_width)
                az = np.full_like(a_hp, np.nan)
                bz = np.full_like(b_hp, np.nan)
                az[ok] = robust_zscore(a_hp[ok])
                bz[ok] = robust_zscore(b_hp[ok])
                sh_nm, peak = xcorr_shift_nm(grid, az, bz, max_shift_nm=args.max_shift_nm)
                if np.isfinite(sh_nm) and np.isfinite(peak):
                    shifts.append(sh_nm)
                    weights.append(max(peak, 0.0))

            shifts = np.asarray(shifts, float)
            weights = np.asarray(weights, float)
            if shifts.size == 0:
                use = False
                sh_final = 0.0
                r_final = np.nan
                sh_std = np.nan
                nwin = 0
            else:
                nwin = int(np.isfinite(shifts).sum())
                sh_std = float(np.nanstd(shifts)) if nwin > 1 else 0.0
                w = np.clip(weights.astype(float), 0.0, None)
                sh_final = weighted_median(shifts, w)
                r_final = float(np.nanmedian(w)) if w.size else np.nan
                good_corr = np.isfinite(r_final) and (r_final >= args.min_peak_corr)
                rigid = np.isfinite(sh_std) and (sh_std <= args.max_shift_std_nm)
                within = (abs(sh_final) <= args.max_abs_shift_nm) or good_corr
                not_edge = np.isfinite(sh_final) and (abs(sh_final) < args.reject_edge_shift_nm)
                use = bool((nwin >= max(args.min_nwin, 1)) and np.isfinite(sh_final) and not_edge and within and (good_corr or rigid))

            rows.append({
                "slit": s,
                "shift_nm": float(sh_final),
                "use": bool(use),
                "r": float(r_final) if np.isfinite(r_final) else np.nan,
                "ref_slit": ref_slit,
                "fallback_objraw": bool(use_fallback),
                "nwin": int(nwin),
                "shift_std_nm": float(sh_std) if np.isfinite(sh_std) else np.nan,
            })

    df_out = pd.DataFrame(rows)[["slit", "shift_nm", "use", "r", "ref_slit", "fallback_objraw", "nwin", "shift_std_nm"]]
    df_out.to_csv(outcsv, index=False)
    print("Wrote:", outcsv)
    print("Use==True:", int(df_out["use"].sum()), "/", len(df_out))


if __name__ == "__main__":
    main()
