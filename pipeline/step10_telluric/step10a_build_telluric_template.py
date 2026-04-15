#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step10a — build empirical O2 telluric templates from extracted 1D spectra.

Pipeline meaning
----------------
  Step09 = OH refine
  Step10 = telluric
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
from astropy.io import fits
import config


def parse_args():
    p = argparse.ArgumentParser(description="Build empirical O2 telluric templates")
    p.add_argument(
        "--infile",
        type=Path,
        default=None,
        help="Input extracted MEF from Step09 "
             "(default: ST09 merged ABAB product)",
    )
    p.add_argument(
        "--outfile",
        type=Path,
        default=None,
        help="Output telluric template FITS "
             "(default: ST10_TELLURIC/telluric_O2_template.fits)",
    )
    return p.parse_args()


ST10 = Path(config.ST10_TELLURIC)
ST10.mkdir(parents=True, exist_ok=True)

DEFAULT_INFILE = Path(
    getattr(
        config,
        "EXTRACT1D_OHCLEAN",
        Path(config.ST09_OH_REFINE) / "extract1d_optimal_ridge_all_wav_ohclean.fits",
    )
)
DEFAULT_OUTFILE = Path(
    getattr(
        config,
        "TELLURIC_TEMPLATE",
        ST10 / "telluric_O2_template.fits",
    )
)

B_LO, B_HI = 682.0, 692.0
B_SB1 = (682.0, 684.0)
B_SB2 = (690.0, 692.0)
A_LO, A_HI = 752.5, 768.5
A_SB1 = (752.5, 754.5)
A_SB2 = (766.5, 768.5)
SNR_MIN = 8.0
MAX_SLITS = 30
NGRID_B = 800
NGRID_A = 900


def finite(x):
    return np.isfinite(x)


def pick_flux_column(cols):
    cols_u = {c.upper(): c for c in cols}
    preferred = [
        "STELLAR",          # canonical Step09 merged science column
        "FLUX_TELLCOR_O2",  # allow rebuilding from already corrected product if desired
        "OBJ_SKYSUB",
        "OBJ_RAW",
        "FLUX_ADU_S",
        "FLUX",
        "FLUX_APCORR",
    ]
    for key in preferred:
        if key in cols_u:
            return cols_u[key]
    return None

def pick_var_column(cols):
    preferred = ["VAR", "VAR_ADU_S2", "VAR_APCORR", "VAR_TELLCOR_O2"]
    cols_u = {c.upper(): c for c in cols}
    for key in preferred:
        if key in cols_u:
            return cols_u[key]
    return None


def fit_cont_sidebands(x, y, sb1, sb2, order=1):
    m = finite(x) & finite(y) & (((x > sb1[0]) & (x < sb1[1])) | ((x > sb2[0]) & (x < sb2[1])))
    if m.sum() < 20:
        return None
    xs, ys = x[m], y[m]
    p = None
    for _ in range(3):
        p = np.polyfit(xs, ys, order)
        r = ys - np.polyval(p, xs)
        med = np.median(r)
        sig = 1.4826 * np.median(np.abs(r - med))
        if not np.isfinite(sig) or sig <= 0:
            break
        good = np.abs(r - med) < 3 * sig
        if good.sum() < 15:
            break
        xs, ys = xs[good], ys[good]
    return p


def window_vec(lam, flux, var, lo, hi, sb1, sb2, grid):
    m = finite(lam) & finite(flux) & (lam > lo) & (lam < hi)
    if m.sum() < 60:
        return None, np.nan
    x = lam[m].astype(float)
    y = flux[m].astype(float)
    v = var[m].astype(float) if var is not None else None
    s = np.argsort(x)
    x = x[s]
    y = y[s]
    if v is not None:
        v = v[s]
    dx = np.diff(x)
    good = np.concatenate([[True], dx > 0])
    x = x[good]
    y = y[good]
    if v is not None:
        v = v[good]
    if x.size < 20:
        return None, np.nan
    p = fit_cont_sidebands(x, y, sb1, sb2, order=1)
    if p is None:
        return None, np.nan
    cont = np.polyval(p, x)
    if not finite(cont).all() or np.nanmedian(cont) == 0:
        return None, np.nan
    yn = y / cont
    snr = np.nan
    if v is not None:
        msb = finite(v) & (v > 0) & (((x > sb1[0]) & (x < sb1[1])) | ((x > sb2[0]) & (x < sb2[1])))
        if msb.sum() >= 20:
            cont_level = np.nanmedian(y[msb])
            sig = np.sqrt(np.nanmedian(v[msb]))
            if np.isfinite(cont_level) and np.isfinite(sig) and sig > 0:
                snr = cont_level / sig
    vec = np.interp(grid, x, yn, left=1.0, right=1.0)
    return vec, snr


def robust_median_stack(arr2d):
    arr2d = np.asarray(arr2d, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(arr2d, axis=0)
    x = np.arange(med.size)
    good = np.isfinite(med)
    if good.sum() >= 2 and (~good).any():
        med[~good] = np.interp(x[~good], x[good], med[good])
    elif good.sum() == 1:
        med[~good] = med[good][0]
    elif good.sum() == 0:
        med[:] = 1.0
    return med


def to_optical_depth(T):
    T = np.clip(T, 0.02, 1.5)
    return -np.log(T)


def estimate_shift_corr(vec, ref, grid, max_shift_nm=1.0):
    s = 1.0 - vec
    sr = 1.0 - ref
    s1 = np.gradient(s, grid)
    sr1 = np.gradient(sr, grid)
    s1 -= np.nanmean(s1)
    sr1 -= np.nanmean(sr1)
    s1 /= (np.nanstd(s1) + 1e-12)
    sr1 /= (np.nanstd(sr1) + 1e-12)
    dlam = grid[1] - grid[0]
    max_k = int(np.round(max_shift_nm / dlam))
    ks = np.arange(-max_k, max_k + 1)
    corr = []
    for k in ks:
        if k < 0:
            a, b = s1[-k:], sr1[:len(sr1) + k]
        elif k > 0:
            a, b = s1[:-k], sr1[k:]
        else:
            a, b = s1, sr1
        corr.append(np.nan if len(a) < 50 else np.nanmean(a * b))
    corr = np.array(corr)
    if not np.isfinite(corr).any():
        return 0.0, np.nan
    imax = np.nanargmax(corr)
    shift_nm = ks[imax] * dlam
    if 0 < imax < len(corr) - 1:
        y1, y2, y3 = corr[imax - 1], corr[imax], corr[imax + 1]
        denom = (y1 - 2 * y2 + y3)
        if abs(denom) > 1e-12:
            delta = 0.5 * (y1 - y3) / denom
            shift_nm = (ks[imax] + delta) * dlam
    return float(shift_nm), float(corr[imax])


def align_vectors(vecs_raw, grid, label, max_shift_nm=1.0, min_peak=0.15):
    ref = robust_median_stack(np.array(vecs_raw[:5]))
    aligned = []
    shifts = []
    for v in vecs_raw:
        dlam, peak = estimate_shift_corr(v, ref, grid, max_shift_nm=max_shift_nm)
        if (not np.isfinite(peak)) or (peak < min_peak):
            continue
        v_shift = np.interp(grid, grid + dlam, v, left=np.nan, right=np.nan)
        if np.isfinite(v_shift).sum() < 0.8 * v_shift.size:
            continue
        aligned.append(v_shift)
        shifts.append(dlam)
    if len(aligned) < 5:
        raise SystemExit(f"Too few aligned {label}-band slits after correlation gating: {len(aligned)}")
    print(f"[{label}-band xcorr] used={len(aligned)}/{len(vecs_raw)}  shift_nm median={float(np.nanmedian(shifts)):+.4f}  std={float(np.nanstd(shifts)):.4f}")
    return aligned, shifts


def main():
    args = parse_args()

    infile = args.infile if args.infile else DEFAULT_INFILE
    outfile = args.outfile if args.outfile else DEFAULT_OUTFILE
    outfile.parent.mkdir(parents=True, exist_ok=True)

    if not infile.exists():
        raise FileNotFoundError(infile)

    print("INFILE =", infile)
    print("OUTFILE =", outfile)

    gridB = np.linspace(B_LO, B_HI, NGRID_B)
    gridA = np.linspace(A_LO, A_HI, NGRID_A)
    vecsB_raw = []
    vecsA_raw = []
    keep = 0
    
    with fits.open(infile) as hdul:
        for hdu in hdul[1:]:
            d = hdu.data
            if d is None or not hasattr(d, "columns"):
                continue
            cols = d.columns.names
            if "LAMBDA_NM" not in cols:
                continue
            flux_col = pick_flux_column(cols)
            if flux_col is None:
                continue
            lam = np.asarray(d["LAMBDA_NM"], float)
            flux = np.asarray(d[flux_col], float)
            var = np.asarray(d[pick_var_column(cols)], float) if pick_var_column(cols) is not None else None
            vB, snrB = window_vec(lam, flux, var, B_LO, B_HI, B_SB1, B_SB2, gridB)
            vA, snrA = window_vec(lam, flux, var, A_LO, A_HI, A_SB1, A_SB2, gridA)
            if (vA is None) and (vB is None):
                continue
            okA = (vA is not None) and (not np.isfinite(snrA) or snrA >= SNR_MIN)
            okB = (vB is not None) and (not np.isfinite(snrB) or snrB >= SNR_MIN)
            if not (okA or okB):
                continue
            if okB:
                vecsB_raw.append(vB)
            if okA:
                vecsA_raw.append(vA)
            keep += 1
            if keep >= MAX_SLITS:
                break

    if len(vecsA_raw) < 5 or len(vecsB_raw) < 5:
        raise SystemExit(f"Not enough good slits to build template: A={len(vecsA_raw)} B={len(vecsB_raw)}")

    vecsA, shiftsA = align_vectors(vecsA_raw, gridA, "A", max_shift_nm=1.0, min_peak=0.15)
    vecsB, shiftsB = align_vectors(vecsB_raw, gridB, "B", max_shift_nm=1.0, min_peak=0.12)
    stackB = robust_median_stack(np.array(vecsB))
    stackA = robust_median_stack(np.array(vecsA))
    tauB = to_optical_depth(stackB)
    tauA = to_optical_depth(stackA)

    ph = fits.Header()
    ph["PIPESTEP"] = "STEP10"
    ph["STAGE"] = "10a"
    ph["SRCFILE"] = infile.name
    ph["SNRMIN"] = float(SNR_MIN)
    ph["MAXSLITS"] = int(MAX_SLITS)
    ph["TMPLTYPE"] = "EMPIRICAL_O2"
    ph["TMPLVER"] = "FINAL_CLEAN_XCORR"
    ph["NA_RAW"] = len(vecsA_raw)
    ph["NB_RAW"] = len(vecsB_raw)
    ph["NA_USE"] = len(vecsA)
    ph["NB_USE"] = len(vecsB)
    ph["A_SHMED"] = float(np.nanmedian(shiftsA))
    ph["A_SHSTD"] = float(np.nanstd(shiftsA))
    ph["B_SHMED"] = float(np.nanmedian(shiftsB))
    ph["B_SHSTD"] = float(np.nanstd(shiftsB))

    hB = fits.BinTableHDU.from_columns([
        fits.Column(name="LAMBDA_NM", format="E", array=gridB.astype(np.float32)),
        fits.Column(name="T_MED", format="E", array=stackB.astype(np.float32)),
        fits.Column(name="TAU_O2", format="E", array=tauB.astype(np.float32)),
    ], name="O2_BAND")

    hA = fits.BinTableHDU.from_columns([
        fits.Column(name="LAMBDA_NM", format="E", array=gridA.astype(np.float32)),
        fits.Column(name="T_MED", format="E", array=stackA.astype(np.float32)),
        fits.Column(name="TAU_O2", format="E", array=tauA.astype(np.float32)),
    ], name="O2_ABAND")

    hA.header["NSLITS"] = len(vecsA)
    hB.header["NSLITS"] = len(vecsB)

    fits.HDUList([fits.PrimaryHDU(header=ph), hB, hA]).writeto(outfile, overwrite=True)
    print(f"Wrote {outfile}")
    print(f"Used slits (cap {MAX_SLITS}): A={len(vecsA)}  B={len(vecsB)}")
    hA = fits.BinTableHDU.from_columns([
        fits.Column(name="LAMBDA_NM", format="E", array=gridA.astype(np.float32)),
        fits.Column(name="T_MED", format="E", array=stackA.astype(np.float32)),
        fits.Column(name="TAU_O2", format="E", array=tauA.astype(np.float32)),
    ], name="O2_ABAND")
    hA.header["NSLITS"] = len(vecsA)
    hB.header["NSLITS"] = len(vecsB)
    fits.HDUList([fits.PrimaryHDU(header=ph), hB, hA]).writeto(outfile, overwrite=True)
    print(f"Wrote {outfile}")
    print(f"Used slits (cap {MAX_SLITS}): A={len(vecsA)}  B={len(vecsB)}")


if __name__ == "__main__":
    main()
