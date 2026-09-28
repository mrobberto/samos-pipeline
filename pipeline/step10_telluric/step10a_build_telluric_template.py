#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step10a — build empirical O2 telluric templates from extracted 1D spectra.

Pipeline meaning
----------------
  Step09 = OH refine
  Step10 = telluric
  
PYTHONPATH=. python pipeline/step10_telluric/step10a_build_telluric_template.py \
  --infile products/Run8_Dolidze25/reduced/09_oh_refine/extract1d_optimal_ridge_all_wav_abswav_OHref.fits \
  --outfile ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/10_telluric/telluric_O2_template.fits  
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

DEFAULT_INFILE = Path(config.EXTRACT1D_OHREF)
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
A_LO, A_HI = 752.5, 771.5  #range of template window
A_SB1 = (752.0, 754.5)
A_SB2 = (769.5, 771.5)  
SNR_MIN = 3.0
MAX_SLITS = 62
NGRID_B = 800
NGRID_A = 1100
TAU_SCALE_A = 1.20
TAU_SCALE_B = 1.10



def finite(x):
    return np.isfinite(x)


def pick_flux_column(cols):
    cols_u = {c.upper(): c for c in cols}
    preferred = [
        "STELLAR_CONSENSUS",
        "STELLAR",
        "OBJ_PRESKY",
        "FLUX",
    ]
    for key in preferred:
        if key in cols_u:
            return cols_u[key]
    return None

def pick_var_column(cols):
    preferred = ["VAR_APCORR", "VAR", "VAR_ADU_S2", "VAR_TELLCOR_O2"]
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
        return None, np.nan, np.nan
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
        return None, np.nan, np.nan
    p = fit_cont_sidebands(x, y, sb1, sb2, order=1)
    if p is None:
        return None, np.nan, np.nan
    cont = np.polyval(p, x)
    if not finite(cont).all() or np.nanmedian(cont) == 0:
        return None, np.nan, np.nan
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
    depth = 1.0 - np.nanmin(vec)
    return vec, snr, depth


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
    T = np.clip(T, 0.02, 1.0)
    return -np.log(T)


def estimate_shift_corr(vec, ref, grid,
                        max_shift_nm=1.0,
                        use_gradient=True):
    s = 1.0 - vec
    sr = 1.0 - ref
    
    if use_gradient:
        s1 = np.gradient(s, grid)
        sr1 = np.gradient(sr, grid)
    else:
        s1 = s.copy()
        sr1 = sr.copy()
    
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


def align_vectors(vecs_raw, grid, label,
                  max_shift_nm=1.0,
                  min_peak=0.15,
                  use_gradient=True,
                  n_iter=3):
    # Iteratively register all correlation-qualified normalized spectra.
    # Seed from the robust median of all eligible candidates, not the first 5.
    ref = robust_median_stack(np.asarray(vecs_raw, float))

    for iteration in range(1, n_iter + 1):
        aligned = []
        shifts = []

        for v in vecs_raw:
            dlam, peak = estimate_shift_corr(
                v, ref, grid,
                max_shift_nm=max_shift_nm,
                use_gradient=use_gradient,
            )
            if (not np.isfinite(peak)) or (peak < min_peak):
                continue

            v_shift = np.interp(
                grid, grid + dlam, v,
                left=np.nan, right=np.nan,
            )
            if np.isfinite(v_shift).sum() < 0.8 * v_shift.size:
                continue

            aligned.append(v_shift)
            shifts.append(dlam)

        if len(aligned) < 5:
            raise SystemExit(
                f"Too few aligned {label}-band slits after correlation "
                f"gating at iteration {iteration}: {len(aligned)}"
            )

        new_ref = robust_median_stack(np.asarray(aligned, float))
        delta = float(np.nanmedian(np.abs(new_ref - ref)))

        print(
            f"[{label} iter {iteration}] "
            f"used={len(aligned)}/{len(vecs_raw)}  "
            f"shift_nm median={float(np.nanmedian(shifts)):+.4f}  "
            f"std={float(np.nanstd(shifts)):.4f}  "
            f"template median |delta|={delta:.6g}"
        )

        ref = new_ref

    # Final registration against the converged common template.
    aligned = []
    shifts = []

    for v in vecs_raw:
        dlam, peak = estimate_shift_corr(
            v, ref, grid,
            max_shift_nm=max_shift_nm,
            use_gradient=use_gradient,
        )
        if (not np.isfinite(peak)) or (peak < min_peak):
            continue

        v_shift = np.interp(
            grid, grid + dlam, v,
            left=np.nan, right=np.nan,
        )
        if np.isfinite(v_shift).sum() < 0.8 * v_shift.size:
            continue

        aligned.append(v_shift)
        shifts.append(dlam)

    if len(aligned) < 5:
        raise SystemExit(
            f"Too few final aligned {label}-band slits after correlation "
            f"gating: {len(aligned)}"
        )

    print(
        f"[{label} FINAL] "
        f"used={len(aligned)}/{len(vecs_raw)}  "
        f"shift_nm median={float(np.nanmedian(shifts)):+.4f}  "
        f"std={float(np.nanstd(shifts)):.4f}"
    )

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

            # Only genuine detected-source slitlets may contribute to the
            # empirical telluric template.
            s08use = int(hdu.header.get("S08USE", 0))
            s08good = int(hdu.header.get("S08GOOD", s08use))
            s08clas = str(hdu.header.get("S08CLAS", "")).strip().upper()
            if (
                s08use != 1
                or s08good != 1
                or s08clas in {"EMPTY", "NOSEED"}
            ):
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
            vB, snrB, depthB = window_vec(lam, flux, var, B_LO, B_HI, B_SB1, B_SB2, gridB)
            vA, snrA, depthA = window_vec(lam, flux, var, A_LO, A_HI, A_SB1, A_SB2, gridA)
            if (vA is None) and (vB is None):
                continue
            okA = (vA is not None) and np.isfinite(depthA) and (0.08 < depthA < 0.80)
            okB = (vB is not None) and np.isfinite(depthB) and (0.03 < depthB < 0.60)
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


    vecsA, shiftsA = align_vectors(vecsA_raw, gridA, "A", max_shift_nm=4.0, min_peak=0.15)
    vecsB, shiftsB = align_vectors(
        vecsB_raw,
        gridB,
        "B",
        max_shift_nm=1.0,
        min_peak=0.08,
        use_gradient=False,
    )
    
    # Use all correlation-qualified aligned spectra.
    # Do not rank/select by normalized telluric depth: depth = 1-min(T)
    # is noise-biased, especially in the B band.
    
    arrA = np.asarray(vecsA, float)
    arrB = np.asarray(vecsB, float)
    
    stackA = robust_median_stack(arrA)
    stackB = robust_median_stack(arrB)
    
    stackA = np.clip(stackA, 0.02, 1.0)
    stackB = np.clip(stackB, 0.02, 1.0)
    
    # Retain the robust, iteratively registered ensemble median
    # at its native spectral sampling; no post-stack smoothing.
    
    tauA = to_optical_depth(stackA)
    tauB = to_optical_depth(stackB)
    
    
    tauA = TAU_SCALE_A * tauA
    tauB = TAU_SCALE_B * tauB
    
    stackA = np.exp(-tauA)
    stackB = np.exp(-tauB)

    ph = fits.Header()
    ph["PIPESTEP"] = "STEP10"
    ph["STAGE"] = "10a"
    ph["SRCFILE"] = infile.name
    ph["SNRMIN"] = float(SNR_MIN)
    ph["MAXSLITS"] = int(MAX_SLITS)
    ph["TMPLTYPE"] = "EMPIRICAL_O2"
    ph["TMPLVER"] = "FINAL_CLEAN_XCORR"
    ph["A_LO"] = float(A_LO)
    ph["A_HI"] = float(A_HI)
    ph["B_LO"] = float(B_LO)
    ph["B_HI"] = float(B_HI)
    ph["TSA"] = float(TAU_SCALE_A)
    ph["TSB"] = float(TAU_SCALE_B)
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


if __name__ == "__main__":
    main()
