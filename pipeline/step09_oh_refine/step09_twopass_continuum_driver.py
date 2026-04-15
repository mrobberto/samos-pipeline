#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import median_filter

from astropy.io import fits
from astropy.table import Table

from step09_continuum_moving_population import (
    read_fits_slit,
    parse_ranges,
    fit_bidirectional_continuum,
    robust_rms,
    robust_ylim,
    in_any_ranges,
    norm_slit,
)


def robust_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    sig = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(x)
    return float(sig)

def add_or_replace_column(table_hdu, name, array):
    tab = Table(table_hdu.data)
    if name in tab.colnames:
        tab[name] = np.asarray(array)
    else:
        tab[name] = np.asarray(array)
    return fits.BinTableHDU(tab, name=table_hdu.name, header=table_hdu.header)

def read_fits_table_column(tab, colname):
    arr = np.asarray(tab[colname])
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    arr = np.asarray(arr, dtype=float)
    arr = np.ravel(arr)
    return arr


def build_broad_negative_bias(
    lam,
    resid,
    protected_mask=None,
    smooth_pix=81,
):
    resid = np.asarray(resid, float)

    # keep only negative part
    neg = np.where(resid < 0, resid, 0.0)

    if protected_mask is not None:
        neg = neg.copy()
        neg[protected_mask] = 0.0

    # smooth strongly so only broad bias survives
    bias = median_filter(neg, size=smooth_pix, mode="nearest")

    # keep only negative correction
    bias = np.where(bias < 0, bias, 0.0)

    return bias

def build_structured_positive_line_model(
    lam,
    resid,
    halfwin_pix=9,
    sigma_thresh=1.8,
    grow_pix=4,
    merge_gap_pix=6,
    smooth_pix=5,
):
    """
    Build a structured narrow positive-line model from a continuum residual.

    Strategy
    --------
    1. Estimate local baseline with a median filter.
    2. Identify positive excess above threshold.
    3. Grow detections to include line wings.
    4. Merge nearby detections into groups.
    5. Within each group, keep only positive excess and smooth lightly.

    Returns
    -------
    line_model : ndarray
        Narrow positive model, same shape as resid.
    """
    lam = np.asarray(lam, float)
    resid = np.asarray(resid, float)

    # Local baseline: broad enough to avoid following narrow lines
    baseline = median_filter(resid, size=2 * halfwin_pix + 1, mode="nearest")
    excess = resid - baseline

    sig = robust_sigma(excess)
    if not np.isfinite(sig) or sig <= 0:
        return np.zeros_like(resid)

    # Initial peak mask
    mask = excess > (sigma_thresh * sig)

    # Grow to capture wings
    if grow_pix > 0:
        grown = mask.copy()
        for k in range(1, grow_pix + 1):
            grown[:-k] |= mask[k:]
            grown[k:] |= mask[:-k]
        mask = grown

    # Merge nearby groups
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.zeros_like(resid)

    groups = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i - prev <= merge_gap_pix:
            prev = i
        else:
            groups.append((start, prev))
            start = i
            prev = i
    groups.append((start, prev))

    line_model = np.zeros_like(resid)

    for i0, i1 in groups:
        sl = slice(max(0, i0), min(len(resid), i1 + 1))
        local = excess[sl].copy()
        local = np.where(local > 0, local, 0.0)

        # Light smoothing inside the group
        if smooth_pix > 1 and len(local) >= smooth_pix:
            local = median_filter(local, size=smooth_pix, mode="nearest")

        line_model[sl] = np.maximum(line_model[sl], local)

    return line_model

def make_qc(
    slit,
    lam,
    y,
    cont1,
    resid1,
    line1,
    y_clean1,
    cont2,
    resid2,
    cont_final,
    telluric_ranges,
    forest_ranges,
    out_png,
    show=False,
):
    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(f"Two-pass continuum + line cleanup - {slit}", fontsize=16)

    mfor = in_any_ranges(lam, forest_ranges) & (~in_any_ranges(lam, telluric_ranges))
    mtel = in_any_ranges(lam, telluric_ranges)
    mnorm = (~mfor) & (~mtel)

    yfin = y[np.isfinite(y)]
    ylo = float(np.nanmin(yfin)) if len(yfin) else -1.0
    yhi = float(np.nanmax(yfin)) if len(yfin) else 1.0

    print(cont_final[0:100])
    # Panel 1
    ax1 = fig.add_subplot(221)
    if np.any(mnorm):
        ax1.fill_between(lam[mnorm], ylo, yhi, color="tab:green", alpha=0.05, step="mid")
    if np.any(mfor):
        ax1.fill_between(lam[mfor], ylo, yhi, color="tab:orange", alpha=0.05, step="mid")
    if np.any(mtel):
        ax1.fill_between(lam[mtel], ylo, yhi, color="tab:red", alpha=0.05, step="mid")
    ax1.plot(lam, y, lw=0.8, color="0.5", label="original")
    ax1.plot(lam, cont1, lw=1.0, color="tab:blue", label="continuum pass1")
    ax1.plot(lam, cont2, lw=1.2, color="k", label="continuum pass2")
    ax1.plot(lam,cont_final,lw=1.5,color='tab:red',label="continuum final")
    ax1.set_title("Original spectrum and continua")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Signal")
    ax1.set_ylim(*robust_ylim(np.r_[y, cont1, cont2], q=(0.5, 99.5)))
    ax1.legend(fontsize=8)

    # Panel 2
    ax2 = fig.add_subplot(222)
    ax2.plot(lam, resid1, lw=0.8, color="tab:blue", label="residual pass1")
    ax2.plot(lam, resid2, lw=0.8, color="k", label="residual pass2")
    ax2.axhline(0.0, lw=0.8, ls="--", color="0.5")
    ax2.set_title(
        f"Residuals  pass1 RMSrob={robust_rms(resid1):.6f}   "
        f"pass2 RMSrob={robust_rms(resid2):.6f}"
    )
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Residual")
    ax2.set_ylim(*robust_ylim(np.r_[resid1, resid2], q=(0.5, 99.5)))
    ax2.legend(fontsize=8)

    # Panel 3
    ax3 = fig.add_subplot(223)
    ax3.plot(lam, resid1, lw=0.6, color="0.6", label="residual pass1")
    ax3.plot(lam, line1, lw=1.0, color="tab:red", label="positive line model")
    ax3.plot(lam, y_clean1, lw=0.8, color="tab:green", label="line-cleaned spectrum")
    ax3.set_title("First-pass line model and cleaned spectrum")
    ax3.set_xlabel("Wavelength (nm)")
    ax3.set_ylabel("Signal")
    ax3.set_ylim(*robust_ylim(np.r_[resid1, line1, y_clean1], q=(0.5, 99.5)))
    ax3.legend(fontsize=8)

    # Panel 4
    ax4 = fig.add_subplot(224)
    ax4.axis("off")
    txt = [
        f"slit              = {slit}",
        f"n points          = {len(lam)}",
        f"RMS robust pass1  = {robust_rms(resid1):.6f}",
        f"RMS robust pass2  = {robust_rms(resid2):.6f}",
        f"line flux removed = {np.nansum(line1):.6f}",
        "",
        "Interpretation:",
        "- pass1 continuum is only a seed",
        "- line1 models narrow positive excess",
        "- pass2 continuum is fitted on line-cleaned spectrum",
    ]
    ax4.text(0.02, 0.98, "\n".join(txt), va="top", ha="left",
             family="monospace", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

def fill_nan_linear(y):
    y = np.asarray(y, float)
    x = np.arange(len(y))
    m = np.isfinite(y)
    if m.sum() == 0:
        return np.zeros_like(y, dtype=float)
    if m.sum() == 1:
        return np.full_like(y, y[m][0], dtype=float)
    out = y.copy()
    out[~m] = np.interp(x[~m], x[m], y[m])
    return out

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-pass continuum + simple positive-line cleanup for one slit"
    )
    p.add_argument("--extract", type=Path, default=None)
    p.add_argument("--slit", type=str, required=True)

    p.add_argument("--telluric-ranges", type=str,
                   default="655.5:657.5,685:690,758:770")
    p.add_argument("--forest-ranges", type=str,
                   default="630:640,680:690,735:745,760:800,820:850,875:910")

    p.add_argument("--window-nm", type=float, default=20.0)
    p.add_argument("--stride-nm", type=float, default=20.0)
    p.add_argument("--method", type=str, default="pchip", choices=["pchip", "linear"])
    p.add_argument("--passes", type=int, default=2)

    p.add_argument("--line-halfwin-pix", type=int, default=5)

    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--qc-png", type=Path, default=None)
    p.add_argument("--show", action="store_true")
    p.add_argument("--line-sigma-thresh", type=float, default=1.8)
    p.add_argument("--line-grow-pix", type=int, default=4)
    p.add_argument("--line-merge-gap-pix", type=int, default=6)
    p.add_argument("--line-smooth-pix", type=int, default=5)
    
    p.add_argument("--in-fits", type=Path, default=None)
    p.add_argument("--out-fits", type=Path, default=None)
    p.add_argument("--source-col", type=str, default="OBJ_PRESKY")
    p.add_argument("--continuum-col-name", type=str, default="CONTINUUM")
    p.add_argument("--resid-col-name", type=str, default="RESID")
    return p.parse_args()

def model_oh_lines_locally(
    lam,
    resid,
    halfwin_pix=9,
    sigma_thresh=2.0,
    max_iter=3,
    max_sigma_pix=3.5,
):
    """
    Build a positive-only local OH line model from residual spectrum.
    """

    lam = np.asarray(lam, float)
    resid = np.asarray(resid, float)

    model = np.zeros_like(resid)
    work = resid.copy()

    for _ in range(max_iter):

        sig = robust_sigma(work)
        if not np.isfinite(sig) or sig <= 0:
            break

        # global candidate list
        peaks = np.where(work > (sigma_thresh * sig))[0]
        if len(peaks) == 0:
            break

        for p in peaks:

            # --- local tuning around the telluric A-band neighborhood ---
            local_sigma_thresh = sigma_thresh
            local_halfwin_pix = halfwin_pix
            local_max_sigma_pix = max_sigma_pix

            if 745.0 <= lam[p] <= 780.0:
                # A-band neighborhood
                local_sigma_thresh = 1.6
                local_halfwin_pix = halfwin_pix + 3
                local_max_sigma_pix = max_sigma_pix + 1.0
    
            # skip if this peak is not significant under the local rule
            if work[p] <= local_sigma_thresh * sig:
                continue

            i0 = max(0, p - local_halfwin_pix)
            i1 = min(len(work), p + local_halfwin_pix + 1)

            x = np.arange(i0, i1, dtype=float)
            y = work[i0:i1]

            if len(y) < 5:
                continue

            amp = y.max()
            if not np.isfinite(amp) or amp <= 0:
                continue
            
            local_floor = np.nanmedian(y)
            if amp < local_floor + 1.0 * sig:
                continue

            weights = np.clip(y, 0, None)
            if np.sum(weights) <= 0:
                continue

            mu = np.sum(x * weights) / np.sum(weights)
            var = np.sum(weights * (x - mu) ** 2) / np.sum(weights)
            sigma_pix = np.sqrt(var)

            if not np.isfinite(sigma_pix):
                continue
            if sigma_pix <= 0:
                continue
            if sigma_pix > local_max_sigma_pix:
                continue

            g = amp * np.exp(-0.5 * ((x - mu) / sigma_pix) ** 2)

            model[i0:i1] += g
            work[i0:i1] -= g

    model = np.clip(model, 0, None)
    return model


def main():
    args = parse_args()

    if args.in_fits is None and args.extract is None:
        raise ValueError("Provide either --extract or --in-fits")

    slit = norm_slit(args.slit)

    telluric_ranges = parse_ranges(args.telluric_ranges)
    forest_ranges = parse_ranges(args.forest_ranges)
    
    # --------------------------------------------------
    # MODE 1: legacy CSV / PNG mode
    # --------------------------------------------------
    if args.in_fits is None:
        lam, y, sig_name = read_fits_slit(args.extract, slit)

        protected_mask = in_any_ranges(lam, telluric_ranges)

        # Pass 1 continuum
        cont1, resid1, cont1_fwd, cont1_bwd, xb1, yb1, keep1, win1 = fit_bidirectional_continuum(
            lam, y,
            telluric_ranges=telluric_ranges,
            forest_ranges=forest_ranges,
            window_nm=args.window_nm,
            stride_nm=args.stride_nm,
            method=args.method,
            n_passes=args.passes,
        )

        bias1 = build_broad_negative_bias(
            lam,
            resid1,
            protected_mask=protected_mask,
            smooth_pix=81,
        )

        cont1_corr = cont1 + bias1
        resid1_corr = y - cont1_corr

        line1 = model_oh_lines_locally(
            lam,
            resid1_corr,
            halfwin_pix=args.line_halfwin_pix,
            sigma_thresh=args.line_sigma_thresh,
        )

        line1 = np.minimum(line1, np.maximum(resid1_corr, 0.0))
        y_clean1 = y - line1

        cont2, resid2, cont2_fwd, cont2_bwd, xb2, yb2, keep2, win2 = fit_bidirectional_continuum(
            lam, y_clean1,
            telluric_ranges=telluric_ranges,
            forest_ranges=forest_ranges,
            window_nm=args.window_nm,
            stride_nm=args.stride_nm,
            method=args.method,
            n_passes=args.passes,
        )
        
        cont1f = fill_nan_linear(cont1_corr)
        cont2f = fill_nan_linear(cont2)
        
        cont1s = median_filter(cont1f, size=41, mode="nearest")
        cont2s = median_filter(cont2f, size=41, mode="nearest")
        
#        cont_final = np.fmin(cont1s, cont2s)
        cont_final = cont2s
        
        out_dir = args.extract.resolve().parent
        out_csv = args.out_csv or (out_dir / f"{args.extract.stem}_{slit}_twopass_continuum.csv")
        out_png = args.qc_png or (out_dir / f"qc_{args.extract.stem}_{slit}_twopass_continuum.png")

        df = pd.DataFrame({
            "LAMBDA_NM": lam,
            "SIGNAL_ORIG": y,
            "CONT1": cont1_corr,
            "RESID1": resid1_corr,
            "LINE1": line1,
            "SIGNAL_CLEAN1": y_clean1,
            "CONT2": cont2,
            "RESID2": resid2,
            "CONT_FINAL": cont_final,
        })
        df.to_csv(out_csv, index=False)

        make_qc(
            slit, lam, y, cont1, resid1, line1, y_clean1,
            cont2, resid2, cont_final,
            telluric_ranges, forest_ranges, out_png, show=args.show,
        )

        print("SLIT      =", slit)
        print("SIGNAL    =", sig_name)
        print("OUT CSV   =", out_csv)
        print("QC PNG    =", out_png)
        print("RMS PASS1 =", robust_rms(resid1))
        print("RMS PASS2 =", robust_rms(resid2))
        print("LINE FLUX =", np.nansum(line1))
        return

    # --------------------------------------------------
    # MODE 2: FITS plumbing mode
    # --------------------------------------------------
    with fits.open(args.in_fits) as hdul:
        out_hdus = [fits.PrimaryHDU(header=hdul[0].header)]

        for hdu in hdul[1:]:
            name = str(hdu.name).strip().upper()

            if not name.startswith("SLIT"):
                out_hdus.append(hdu.copy())
                continue

            if name != slit:
                out_hdus.append(hdu.copy())
                continue

            tab = Table(hdu.data)

            if "LAMBDA_NM" not in tab.colnames:
                raise KeyError(f"{name}: missing LAMBDA_NM column")
            if args.source_col not in tab.colnames:
                raise KeyError(
                    f"{name}: missing source column {args.source_col}. "
                    f"Available columns: {list(tab.colnames)}"
                )

            # --- read and sanitize like the working path ---
            lam_full = read_fits_table_column(tab, "LAMBDA_NM")
            y_full = read_fits_table_column(tab, args.source_col)

            m = np.isfinite(lam_full) & np.isfinite(y_full)
            if m.sum() < 10:
                raise ValueError(f"{name}: not enough finite points in LAMBDA_NM / {args.source_col}")

            lam = lam_full[m]
            y = y_full[m]

            # mimic read_fits_slit behavior as closely as possible
            order = np.argsort(lam)
            lam = lam[order]
            y = y[order]

            protected_mask = in_any_ranges(lam, telluric_ranges)

            # ------------------------------
            # science block = saved working version
            # ------------------------------
            cont1, resid1, cont1_fwd, cont1_bwd, xb1, yb1, keep1, win1 = fit_bidirectional_continuum(
                lam, y,
                telluric_ranges=telluric_ranges,
                forest_ranges=forest_ranges,
                window_nm=args.window_nm,
                stride_nm=args.stride_nm,
                method=args.method,
                n_passes=args.passes,
            )

            bias1 = build_broad_negative_bias(
                lam,
                resid1,
                protected_mask=protected_mask,
                smooth_pix=81,
            )

            cont1_corr = cont1 + bias1
            resid1_corr = y - cont1_corr

            line1 = model_oh_lines_locally(
                lam,
                resid1_corr,
                halfwin_pix=args.line_halfwin_pix,
                sigma_thresh=args.line_sigma_thresh,
            )

            line1 = np.minimum(line1, np.maximum(resid1_corr, 0.0))
            y_clean1 = y - line1

            cont2, resid2, cont2_fwd, cont2_bwd, xb2, yb2, keep2, win2 = fit_bidirectional_continuum(
                lam, y_clean1,
                telluric_ranges=telluric_ranges,
                forest_ranges=forest_ranges,
                window_nm=args.window_nm,
                stride_nm=args.stride_nm,
                method=args.method,
                n_passes=args.passes,
            )

            cont1f = fill_nan_linear(cont1_corr)
            cont2f = fill_nan_linear(cont2)

            cont1s = median_filter(cont1f, size=41, mode="nearest")
            cont2s = median_filter(cont2f, size=41, mode="nearest")

            # safer choice for FITS mode
            cont_final = cont2s

            # ------------------------------
            # now map back to full slit length
            # ------------------------------
            inv_order = np.argsort(order)

            cont1_corr_uns = cont1_corr[inv_order]
            resid1_corr_uns = resid1_corr[inv_order]
            line1_uns = line1[inv_order]
            y_clean1_uns = y_clean1[inv_order]
            cont2_uns = cont2[inv_order]
            resid2_uns = resid2[inv_order]
            cont_final_uns = cont_final[inv_order]
            resid_final_uns = (y - cont_final)[inv_order]

            cont1_full = np.full_like(lam_full, np.nan, dtype=float)
            resid1_full = np.full_like(lam_full, np.nan, dtype=float)
            line1_full = np.full_like(lam_full, np.nan, dtype=float)
            y_clean1_full = np.full_like(lam_full, np.nan, dtype=float)
            cont2_full = np.full_like(lam_full, np.nan, dtype=float)
            resid2_full = np.full_like(lam_full, np.nan, dtype=float)
            cont_final_full = np.full_like(lam_full, np.nan, dtype=float)
            resid_final_full = np.full_like(lam_full, np.nan, dtype=float)

            cont1_full[m] = cont1_corr_uns
            resid1_full[m] = resid1_corr_uns
            line1_full[m] = line1_uns
            y_clean1_full[m] = y_clean1_uns
            cont2_full[m] = cont2_uns
            resid2_full[m] = resid2_uns
            cont_final_full[m] = cont_final_uns
            resid_final_full[m] = resid_final_uns

            print(name, "input finite      =", np.isfinite(y_full).sum(), "/", len(y_full))
            print(name, "CONT1 corr finite =", np.isfinite(cont1_full).sum(), "/", len(cont1_full))
            print(name, "CONT2 finite      =", np.isfinite(cont2_full).sum(), "/", len(cont2_full))
            print(name, "CONT_FINAL finite =", np.isfinite(cont_final_full).sum(), "/", len(cont_final_full))

            new_hdu = add_or_replace_column(hdu, args.continuum_col_name, cont_final_full)
            new_hdu = add_or_replace_column(new_hdu, args.resid_col_name, resid_final_full)
            new_hdu = add_or_replace_column(new_hdu, "CONT1", cont1_full)
            new_hdu = add_or_replace_column(new_hdu, "RESID1", resid1_full)
            new_hdu = add_or_replace_column(new_hdu, "LINE1", line1_full)
            new_hdu = add_or_replace_column(new_hdu, "SIGNAL_CLEAN1", y_clean1_full)
            new_hdu = add_or_replace_column(new_hdu, "CONT2", cont2_full)
            new_hdu = add_or_replace_column(new_hdu, "RESID2", resid2_full)

            out_hdus.append(new_hdu)

        args.out_fits.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList(out_hdus).writeto(args.out_fits, overwrite=True)
        print("WROTE FITS =", args.out_fits)
        print(name, "input finite      =", np.isfinite(y_full).sum(), "/", len(y_full))
        print(name, "CONT1 corr finite =", np.isfinite(cont1_full).sum(), "/", len(cont1_full))

if __name__ == "__main__":
    main()