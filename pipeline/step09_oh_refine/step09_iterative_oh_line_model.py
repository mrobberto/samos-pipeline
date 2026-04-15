#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 14:03:25 2026

@author: robberto
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table

from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

from step09_continuum_moving_population import (
    read_fits_slit,
    norm_slit,
    parse_ranges,
    robust_rms,
    robust_ylim,
)


SIGMA_THRESH_BRIGHT = 3.0
SIGMA_THRESH_FAINT  = 1.4
SIGMA_THRESH_CLEAN  = 1.2

PROM_THRESH_BRIGHT  = 1.5
PROM_THRESH_FAINT   = 0.7
PROM_THRESH_CLEAN   = 0.6

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def robust_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if np.isfinite(mad) and mad > 0:
        return float(1.4826 * mad)
    return float(np.nanstd(x))

def safe_trapz(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 2:
        return 0.0
    return float(np.trapz(y[m], x[m]))

def gaussian_from_flux(lam, lam0, sigma_nm, flux_int):
    lam = np.asarray(lam, float)
    if sigma_nm <= 0:
        return np.zeros_like(lam)
    amp = flux_int / (sigma_nm * np.sqrt(2.0 * np.pi))
    return amp * np.exp(-0.5 * ((lam - lam0) / sigma_nm) ** 2)

def interp_fine_grid(lam_win, y_win, oversample=8):
    lam_win = np.asarray(lam_win, float)
    y_win = np.asarray(y_win, float)
    if len(lam_win) < 3:
        return lam_win, y_win
    nfine = max(len(lam_win) * oversample, len(lam_win))
    lf = np.linspace(lam_win.min(), lam_win.max(), nfine)
    yf = np.interp(lf, lam_win, y_win)
    return lf, yf

def read_fits_table_column(tab, colname):
    arr = np.asarray(tab[colname])
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    arr = np.asarray(arr, dtype=float)
    arr = np.ravel(arr)
    return arr

# ------------------------------------------------------------
# data models
# ------------------------------------------------------------

@dataclass
class PeakCandidate:
    peak_id: str
    lam_peak: float
    ipix: int
    height: float
    prominence: float
    score: float
    pass_found: str

@dataclass
class LineComponent:
    line_id: str
    lam_init: float
    lam_fit: float
    sigma_nm: float
    flux_int: float
    window_lo: float
    window_hi: float
    iteration: int
    phase: str
    accepted: bool
    quality_flag: str

# ------------------------------------------------------------
# continuum
# ------------------------------------------------------------

def infer_slit_from_filename(path: Path):
    """
    Try to extract SLITXXX from filename.
    Returns normalized slit (e.g. 'SLIT006') or None.
    """
    name = path.name.upper()
    import re
    m = re.search(r"SLIT\s*0*([0-9]+)", name)
    if m:
        return f"SLIT{int(m.group(1)):03d}"
    return None


def read_continuum(args_continuum_csv, lam, slit_expected=None, strict=True):
    df = pd.read_csv(args_continuum_csv)

    # --- find continuum column ---
    cont_col = None
    for cand in ["CONT_FINAL", "CONT2", "CONT1"]:
        if cand in df.columns:
            cont_col = cand
            break
    if cont_col is None:
        raise KeyError(
            f"No usable continuum column found in {args_continuum_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"USING CONTINUUM COLUMN = {cont_col}")

    # ------------------------------------------------------------
    # NEW: slit consistency check
    # ------------------------------------------------------------
    slit_from_file = infer_slit_from_filename(Path(args_continuum_csv))

    if slit_expected is not None and slit_from_file is not None:
        if slit_from_file != slit_expected:
            msg = (
                f"[CONTINUUM MISMATCH]\n"
                f"  requested slit : {slit_expected}\n"
                f"  continuum file : {args_continuum_csv}\n"
                f"  inferred slit  : {slit_from_file}\n"
            )
            if strict:
                raise ValueError(msg)
            else:
                print("WARNING:", msg)

    elif slit_expected is not None:
        print(f"WARNING: Could not infer slit from continuum filename: {args_continuum_csv}")

    # ------------------------------------------------------------
    # interpolate continuum
    # ------------------------------------------------------------
    cont = np.interp(
        lam,
        np.asarray(df["LAMBDA_NM"], float),
        np.asarray(df[cont_col], float),
    )

    return cont, cont_col
# ------------------------------------------------------------
# candidate detection
# ------------------------------------------------------------

def detect_candidate_peaks(
    lam,
    resid,
    sigma_thresh=SIGMA_THRESH_BRIGHT,
    prominence_sigma=PROM_THRESH_BRIGHT,
    min_sep_pix=4,
    pass_found="bright",
):
    lam = np.asarray(lam, float)
    resid = np.asarray(resid, float)

    m = np.isfinite(lam) & np.isfinite(resid)
    if m.sum() < 5:
        return []

    y = resid.copy()
    y[~m] = 0.0

    sig = robust_sigma(y[m])
    if not np.isfinite(sig) or sig <= 0:
        return []

    peaks, props = find_peaks(
        y,
        height=sigma_thresh * sig,
        prominence=prominence_sigma * sig,
        distance=min_sep_pix,
    )

    out = []
    for k, p in enumerate(peaks):
        h = float(props["peak_heights"][k])
        prom = float(props["prominences"][k])
        score = h * np.sqrt(max(prom, 0.0))
        out.append(
            PeakCandidate(
                peak_id=f"{pass_found.upper()}_{k:04d}",
                lam_peak=float(lam[p]),
                ipix=int(p),
                height=h,
                prominence=prom,
                score=score,
                pass_found=pass_found,
            )
        )
    if pass_found == "bright":
        out.sort(key=lambda z: z.height, reverse=True)
    else:
        out.sort(key=lambda z: z.score, reverse=True)
    return out

def suppress_near_existing(cands, components, sep_nm=0.25):
    if len(components) == 0:
        return cands
    keep = []
    lam_used = np.array([c.lam_fit for c in components if c.accepted], float)
    for c in cands:
        if lam_used.size == 0 or np.min(np.abs(lam_used - c.lam_peak)) > sep_nm:
            keep.append(c)
    return keep

# ------------------------------------------------------------
# local fit
# ------------------------------------------------------------

def estimate_initial_sigma_nm(lam, resid, ipix, halfwin_pix=5, sigma_min_nm=0.06, sigma_max_nm=0.40):
    n = len(lam)
    i0 = max(0, ipix - halfwin_pix)
    i1 = min(n, ipix + halfwin_pix + 1)
    x = lam[i0:i1]
    y = np.clip(resid[i0:i1], 0, None)
    if len(x) < 3 or np.nansum(y) <= 0:
        dlam = np.nanmedian(np.diff(lam))
        if not np.isfinite(dlam) or dlam <= 0:
            return 0.12
        return float(np.clip(1.5 * dlam, sigma_min_nm, sigma_max_nm))
    mu = np.nansum(x * y) / np.nansum(y)
    var = np.nansum(y * (x - mu) ** 2) / np.nansum(y)
    sig = np.sqrt(max(var, 0.0))
    return float(np.clip(sig, sigma_min_nm, sigma_max_nm))

def fit_single_line_local(
    lam,
    resid,
    lam_init,
    window_half_nm=0.8,
    delta_max_nm=0.18,
    sigma_min_nm=0.05,
    sigma_max_nm=0.45,
    oversample=8,
):
    m = (lam >= lam_init - window_half_nm) & (lam <= lam_init + window_half_nm) & np.isfinite(resid)
    if m.sum() < 5:
        return None

    lam_win = lam[m]
    y_win = resid[m]

    # local constant background from lower envelope
    c0 = float(np.nanmedian(np.clip(y_win, None, np.nanpercentile(y_win, 40))))
    y_loc = y_win - c0

    ip = int(np.nanargmax(y_loc))
    sigma0 = estimate_initial_sigma_nm(lam_win, y_loc, ip)
    flux0 = max(safe_trapz(np.clip(y_loc, 0, None), lam_win), 1e-6)

    lf, yf = interp_fine_grid(lam_win, y_loc, oversample=oversample)

    def resid_fun(p):
        lam0, sig_nm, flux_int = p
        mod = gaussian_from_flux(lf, lam0, sig_nm, flux_int)
        return yf - mod

    p0 = np.array([lam_init, sigma0, flux0], float)
    lo = np.array([lam_init - delta_max_nm, sigma_min_nm, 0.0], float)
    hi = np.array([lam_init + delta_max_nm, sigma_max_nm, np.inf], float)

    try:
        sol = least_squares(resid_fun, p0, bounds=(lo, hi), method="trf")
    except Exception:
        return None

    if not sol.success:
        return None

    lam_fit, sigma_fit, flux_fit = [float(v) for v in sol.x]
    model_native = gaussian_from_flux(lam_win, lam_fit, sigma_fit, flux_fit)
    resid_post = y_loc - model_native

    rms_before = robust_rms(y_loc)
    rms_after = robust_rms(resid_post)
    neg_area = safe_trapz(np.clip(-resid_post, 0, None), lam_win)
    pos_left = safe_trapz(np.clip(resid_post, 0, None), lam_win)

    accepted = True
    flags = []

    if rms_after > 0.998 * rms_before:
        flags.append("weak_improvement")
    if neg_area > 2.0 * max(pos_left, 1e-8):
        flags.append("oversubtracted")
    if flux_fit <= 0:
        flags.append("nonpositive_flux")
        accepted = False

    qflag = "ok" if len(flags) == 0 else ",".join(flags)
    
    full_model = np.zeros_like(lam, dtype=float)
    full_model[m] = model_native

    return dict(
        lam_fit=lam_fit,
        sigma_nm=sigma_fit,
        flux_int=flux_fit,
        window_lo=float(lam_win.min()),
        window_hi=float(lam_win.max()),
        rms_before=float(rms_before),
        rms_after=float(rms_after),
        neg_area=float(neg_area),
        pos_leftover=float(pos_left),
        accepted=accepted,
        quality_flag=qflag,
        model_full=full_model,
    )

# ------------------------------------------------------------
# iterative modeling
# ------------------------------------------------------------

def build_model_from_components(lam, components):
    model = np.zeros_like(lam, dtype=float)
    for c in components:
        if c.accepted:
            model += gaussian_from_flux(lam, c.lam_fit, c.sigma_nm, c.flux_int)
    return np.clip(model, 0.0, None)

def add_or_replace_column(table_hdu, name, array):
    tab = Table(table_hdu.data)
    if name in tab.colnames:
        tab[name] = np.asarray(array)
    else:
        tab[name] = np.asarray(array)
    return fits.BinTableHDU(tab, name=table_hdu.name, header=table_hdu.header)


def fit_ranked_lines_pass(
    lam,
    resid0,
    components_locked,
    candidates,
    top_n,
    iteration,
    phase,
):
    new_components = []

    use = candidates[:top_n]
    for i, cand in enumerate(use):
        model_locked = build_model_from_components(lam, components_locked + new_components)
        work = resid0 - model_locked

        fit = fit_single_line_local(
            lam=lam,
            resid=work,
            lam_init=cand.lam_peak,
            window_half_nm=1.0 if phase == "bright" else 0.7,
            delta_max_nm=0.25 if phase == "bright" else 0.15,
            sigma_min_nm=0.05,
            sigma_max_nm=0.45,
            oversample=8,
        )

        if fit is None:
            continue
        
        new_components.append(
            LineComponent(
                line_id=f"{phase.upper()}_{iteration:02d}_{i:04d}",
                lam_init=float(cand.lam_peak),
                lam_fit=float(fit["lam_fit"]),
                sigma_nm=float(fit["sigma_nm"]),
                flux_int=float(fit["flux_int"]),
                window_lo=float(fit["window_lo"]),
                window_hi=float(fit["window_hi"]),
                iteration=int(iteration),
                phase=str(phase),
                accepted=bool(fit["accepted"]),
                quality_flag=str(fit["quality_flag"]),
            )
        )
    return new_components

def cleanup_residual_lines(
    lam,
    resid0,
    existing_components,
    iteration,
    n_cleanup=60,
):
    model_now = build_model_from_components(lam, existing_components)
    resid_now = resid0 - model_now

    cleanup_peaks = detect_candidate_peaks(
        lam, resid_now,
        sigma_thresh=SIGMA_THRESH_CLEAN,
        prominence_sigma=PROM_THRESH_CLEAN,
        min_sep_pix=2,
        pass_found="cleanup",
    )
    """
    cleanup_peaks = [
        pk for pk in cleanup_peaks
        if is_new_peak(
            pk.lam_peak,
            lam=lam,
            master_resid=resid0,
            components=existing_components,
            tol_nm=0.04,
            residual_sigma_thresh=1.5,
        )
    ]
    """    
    
    cleanup_peaks.sort(key=lambda z: z.height, reverse=True)

    new_cleanup = fit_ranked_lines_pass(
        lam=lam,
        resid0=resid0,
        components_locked=existing_components,
        candidates=cleanup_peaks,
        top_n=n_cleanup,
        iteration=iteration,
        phase="cleanup",
    )

    return new_cleanup

def refit_existing_lines(
    lam,
    resid0,
    components,
    fixed_components,
    iteration,
    phase,
):

    out = []
    for i, c in enumerate(components):
        others_same = [z for j, z in enumerate(components) if j != i and z.accepted]
        model_same = build_model_from_components(lam, others_same)
        model_fixed = build_model_from_components(lam, fixed_components)
        work = resid0 - model_same - model_fixed
        
        fit = fit_single_line_local(
            lam=lam,
            resid=work,
            lam_init=c.lam_fit,
            window_half_nm=max(0.6, 2.0 * c.sigma_nm),
            delta_max_nm=0.08,
            sigma_min_nm=max(0.04, 0.70 * c.sigma_nm),
            sigma_max_nm=min(0.30, 1.15 * c.sigma_nm),
            oversample=8,
        )

        if fit is None:
            out.append(c)
            continue

        out.append(
            LineComponent(
                line_id=f"{phase.upper()}_REFIT_{iteration:02d}_{i:04d}",
                lam_init=float(c.lam_fit),
                lam_fit=float(fit["lam_fit"]),
                sigma_nm=float(fit["sigma_nm"]),
                flux_int=float(fit["flux_int"]),
                window_lo=float(fit["window_lo"]),
                window_hi=float(fit["window_hi"]),
                iteration=int(iteration),
                phase=f"{phase}_refit",
                accepted=bool(fit["accepted"]),
                quality_flag=str(fit["quality_flag"]),
            )
        )
    return out

def is_new_peak(lam_peak, lam, master_resid, components, tol_nm=0.05, residual_sigma_thresh=1.5):
    """
    A peak is considered 'already represented' only if:
    1) there is an accepted component nearby, AND
    2) the current search residual near that wavelength is already small.
    """
    current_model = build_model_from_components(lam, components)
    search_resid = master_resid - current_model

    # local residual significance near the candidate peak
    mloc = np.abs(lam - lam_peak) <= 0.10
    if mloc.sum() < 3:
        return True

    local_sig = robust_sigma(search_resid[mloc])
    if not np.isfinite(local_sig) or local_sig <= 0:
        local_sig = robust_sigma(search_resid)

    local_peak = np.nanmax(search_resid[mloc])

    # if a strong residual peak is still there, it is NOT already explained
    if np.isfinite(local_peak) and np.isfinite(local_sig):
        if local_peak > residual_sigma_thresh * local_sig:
            return True  # genuinely new / still active

    # otherwise check proximity to existing accepted fitted lines
    for c in components:
        if not c.accepted:
            continue
        tol = max(tol_nm, 0.5 * c.sigma_nm)
        if abs(c.lam_fit - lam_peak) < tol:
            return False

    return True


def iterative_bright_faint_model(
    lam,
    resid0,
    slit="UNKNOWN",
    n_bright=50,
    n_faint=50,
    max_cycles=4,
):

    bright_components = []
    faint_components = []

    prev_rms = robust_rms(resid0)

    # --------------------------------------------------
    # INITIAL BRIGHT DETECTION: ONCE ONLY
    # --------------------------------------------------
    bright_peaks = detect_candidate_peaks(
        lam, resid0,
        sigma_thresh=SIGMA_THRESH_BRIGHT,
        prominence_sigma=PROM_THRESH_BRIGHT,
        min_sep_pix=4,
        pass_found="bright",
    )

    # --------------------------------------------------
    # DEBUG QC: show the 20 brightest initially detected peaks
    # --------------------------------------------------
    top20 = bright_peaks[:20]
    
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    
    ax.plot(lam, resid0, lw=0.8, color="k", label="MASTER residual")
    
    for pk in top20:
        ypk = np.interp(pk.lam_peak, lam, resid0)
        ax.plot(pk.lam_peak, ypk, marker="*", ms=12, color="red")
    
    ax.set_title(f"Top 20 brightest detected peaks - initial bright pass")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Residual")
    ax.set_ylim(*robust_ylim(resid0, q=(0.5, 99.5)))
    ax.legend(fontsize=8)
    
    dbg_png = Path(f"debug_top20_peaks_{slit}.png")
    fig.tight_layout()
    fig.savefig(dbg_png, dpi=150)
    plt.close(fig)
    
    print("DEBUG TOP20 PNG =", dbg_png)
    print("TOP20 PEAKS =", [round(pk.lam_peak, 3) for pk in top20])


    # bright pass ranked by pure peak height
    bright_peaks.sort(key=lambda z: z.height, reverse=True)

    print(f"INITIAL bright candidates = {len(bright_peaks)}")

    bright_components = fit_ranked_lines_pass(
        lam=lam,
        resid0=resid0,
        components_locked=[],
        candidates=bright_peaks,
        top_n=n_bright,
        iteration=0,
        phase="bright",
    )
    bright_components = refit_existing_lines(lam, resid0, bright_components, [], 0, "bright")

    print(f"INITIAL accepted bright comps = {sum(c.accepted for c in bright_components)}")

    # --------------------------------------------------
    # ITERATIONS: refit bright, discover new faint, refit faint, refit bright
    # --------------------------------------------------
    for it in range(1, max_cycles + 1):

        # refit bright on original residual minus current faint model
        bright_components = refit_existing_lines(lam, resid0, bright_components, faint_components, it, "bright")

        model_bright = build_model_from_components(lam, bright_components)
        model_faint = build_model_from_components(lam, faint_components)
        resid_now = resid0 - model_bright - model_faint

        # detect NEW faint peaks only
        faint_peaks = detect_candidate_peaks(
            lam, resid_now,
            sigma_thresh=SIGMA_THRESH_FAINT,
            prominence_sigma=PROM_THRESH_FAINT,
            min_sep_pix=2,
            pass_found="faint",
        )

        # only genuinely new peaks
        faint_peaks = [
            pk for pk in faint_peaks
            if is_new_peak(
                pk.lam_peak,
                lam=lam,
                master_resid=resid0,
                components=bright_components + faint_components,
                tol_nm=0.05,
                residual_sigma_thresh=1.5,
            )
        ]
        
        # faint pass ranked by composite score
        faint_peaks.sort(key=lambda z: z.score, reverse=True)

        print(f"ITER {it} faint candidates = {len(faint_peaks)}")

        new_faint = fit_ranked_lines_pass(
            lam=lam,
            resid0=resid0,
            components_locked=bright_components + faint_components,
            candidates=faint_peaks,
            top_n=n_faint,
            iteration=it,
            phase="faint",
        )

        # add only this cycle's new faint lines
        faint_components.extend(new_faint)
        print(f"ITER {it} accepted new faint comps = {sum(c.accepted for c in new_faint)}")

        # refit faint set against original residual minus bright model
        faint_components = refit_existing_lines(lam, resid0, faint_components, bright_components, it, "faint")

        # refit bright again now that faint changed
        bright_components = refit_existing_lines(lam, resid0, bright_components, faint_components, it, "bright")

        model_now = build_model_from_components(lam, bright_components + faint_components)
        resid_now = resid0 - model_now
        rms_now = robust_rms(resid_now)

        frac_improve = (prev_rms - rms_now) / max(prev_rms, 1e-12)
        print(f"ITER {it}: RMS {prev_rms:.6f} -> {rms_now:.6f}  frac_improve={frac_improve:.4f}")

        if it >= 2 and frac_improve < 0.003 and len(faint_peaks) == 0:
            break

        prev_rms = rms_now

    # --------------------------------------------------
    # FINAL CLEANUP PASS ON REMAINING RESIDUAL
    # --------------------------------------------------
    cleanup_components = cleanup_residual_lines(
        lam=lam,
        resid0=resid0,
        existing_components=bright_components + faint_components,
        iteration=max_cycles + 1,
        n_cleanup=60,
    )

    print(f"FINAL cleanup accepted comps = {sum(c.accepted for c in cleanup_components)}")

    faint_components.extend(cleanup_components)
    faint_components = refit_existing_lines(lam, resid0, faint_components, bright_components, max_cycles + 1, "cleanup")
    bright_components = refit_existing_lines(lam, resid0, bright_components, faint_components, max_cycles + 1, "bright")    

    all_components = bright_components + faint_components
    model_final = build_model_from_components(lam, all_components)
    resid_final = resid0 - model_final
    return model_final, resid_final, all_components
# ------------------------------------------------------------
# outputs
# ------------------------------------------------------------

def components_to_table(components):
    rows = []
    for c in components:
        rows.append(dict(
            LINE_ID=c.line_id,
            LAM_INIT=c.lam_init,
            LAM_FIT=c.lam_fit,
            SIGMA_NM=c.sigma_nm,
            FLUX_INT=c.flux_int,
            WINDOW_LO=c.window_lo,
            WINDOW_HI=c.window_hi,
            ITERATION=c.iteration,
            PHASE=c.phase,
            ACCEPTED=int(c.accepted),
            QUALITY_FLAG=c.quality_flag,
        ))
    return pd.DataFrame(rows)

def make_qc(slit, lam, flux, cont, oh_model, flux_ohsub, resid_pre, resid_post, comp_df, out_png):
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f"Iterative OH subtraction - {slit}", fontsize=16)

    ax1 = fig.add_subplot(221)
    ax1.plot(lam, flux, lw=0.7, color="0.6", label="OBJ_PRESKY")
    ax1.plot(lam, cont, lw=1.2, color="tab:blue", label="continuum")
    ax1.plot(lam, flux_ohsub, lw=0.8, color="tab:green", label="OH-subtracted")
    ax1.set_title("Spectrum")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Signal")
    ax1.set_ylim(*robust_ylim(np.r_[flux, cont, flux_ohsub], q=(0.5, 99.5)))
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(222)
    ax2.plot(lam, resid_pre, lw=0.7, color="tab:orange", label="resid pre")
    ax2.plot(lam, resid_post, lw=0.7, color="k", label="resid post")
    ax2.axhline(0.0, color="0.5", ls="--", lw=0.8)
    ax2.set_title(f"Residuals   RMS pre={robust_rms(resid_pre):.6f}   post={robust_rms(resid_post):.6f}")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Residual")
    ax2.set_ylim(*robust_ylim(np.r_[resid_pre, resid_post], q=(0.5, 99.5)))
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(223)
    ax3.plot(lam, resid_pre, lw=0.5, color="0.8", label="resid pre")
    ax3.plot(lam, oh_model, lw=0.9, color="tab:red", label="OH model")
    ax3.set_title("OH model")
    ax3.set_xlabel("Wavelength (nm)")
    ax3.set_ylabel("Model")
    ax3.set_ylim(*robust_ylim(np.r_[resid_pre, oh_model], q=(0.5, 99.5)))
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(224)
    ax4.axis("off")
    nacc = int(comp_df["ACCEPTED"].sum()) if len(comp_df) else 0
    txt = [
        f"slit            = {slit}",
        f"accepted comps  = {nacc}",
        f"OH flux removed = {safe_trapz(oh_model, lam):.6f}",
        f"RMS pre         = {robust_rms(resid_pre):.6f}",
        f"RMS post        = {robust_rms(resid_post):.6f}",
    ]
    ax4.text(0.03, 0.97, "\n".join(txt), va="top", ha="left", family="monospace", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

# ------------------------------------------------------------
# CLI / main
# ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Iterative bright/faint OH line modeling")
    p.add_argument("--extract", type=Path, default=None)
    p.add_argument("--slit", type=str, required=True)
    p.add_argument("--continuum-csv", type=Path, default=None)

    p.add_argument("--n-bright", type=int, default=50)
    p.add_argument("--n-faint", type=int, default=80)
    p.add_argument("--max-cycles", type=int, default=4)

    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--comp-table-csv", type=Path, default=None)
    p.add_argument("--qc-png", type=Path, default=None)
    
    p.add_argument("--in-fits", type=Path, default=None,
               help="Optional FITS input carrying spectrum + continuum columns")
    p.add_argument("--out-fits", type=Path, default=None,
                   help="Optional FITS output with OH columns added")
    p.add_argument("--source-col", type=str, default="OBJ_PRESKY",
                   help="Input source column when reading --in-fits")
    p.add_argument("--continuum-col", type=str, default=None,
                   help="Continuum column name when reading --in-fits")
    p.add_argument("--oh-col-name", type=str, default="OH_MODEL",
                   help="Output FITS column for OH model")
    p.add_argument("--stellar-col-name", type=str, default="STELLAR",
                   help="Output FITS column for line-cleaned stellar spectrum")
    p.add_argument("--resid-col-name", type=str, default="RESID_POSTOH",
                   help="Output FITS column for residual after OH subtraction")

    return p.parse_args()

def main():
    args = parse_args()
    slit = norm_slit(args.slit)

    # ------------------------------
    # MODE 1 — legacy CSV workflow
    # ------------------------------
    if args.in_fits is None:

        if args.extract is None or args.continuum_csv is None:
            raise ValueError("...")

        lam, flux, sig_name = read_fits_slit(args.extract, slit)

        cont, cont_col = read_continuum(
            args.continuum_csv,
            lam,
            slit_expected=slit,
            strict=True,
        )

        resid_pre = flux - cont

        oh_model, resid_post, components = iterative_bright_faint_model(
            lam=lam,
            resid0=resid_pre,
            slit=slit,
            n_bright=args.n_bright,
            n_faint=args.n_faint,
            max_cycles=args.max_cycles,
        )

        flux_ohsub = flux - oh_model

        # existing CSV / QC code

        return

    # ------------------------------
    # MODE 2 — FITS pipeline mode
    # ------------------------------
    if args.continuum_col is None:
        raise ValueError("--continuum-col required in FITS mode")

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

            lam_full = read_fits_table_column(tab, "LAMBDA_NM")
            flux_full = read_fits_table_column(tab, args.source_col)
            cont_full = read_fits_table_column(tab, args.continuum_col)
            
            m = np.isfinite(lam_full) & np.isfinite(flux_full) & np.isfinite(cont_full)
            if m.sum() < 10:
                raise ValueError(
                    f"{name}: not enough finite points in "
                    f"LAMBDA_NM / {args.source_col} / {args.continuum_col}"
                )
            
            lam = lam_full[m]
            flux = flux_full[m]
            cont = cont_full[m]
            
            order = np.argsort(lam)
            lam = lam[order]
            flux = flux[order]
            cont = cont[order]
            
            resid_pre = flux - cont

            oh_model, resid_post, components = iterative_bright_faint_model(
                lam=lam,
                resid0=resid_pre,
                slit=name,
                n_bright=args.n_bright,
                n_faint=args.n_faint,
                max_cycles=args.max_cycles,
            )

            stellar = flux - oh_model
            
            inv_order = np.argsort(order)

            oh_uns = oh_model[inv_order]
            stellar_uns = stellar[inv_order]
            resid_uns = resid_post[inv_order]
            
            oh_full = np.full_like(lam_full, np.nan, dtype=float)
            stellar_full = np.full_like(lam_full, np.nan, dtype=float)
            resid_full = np.full_like(lam_full, np.nan, dtype=float)
            
            oh_full[m] = oh_uns
            stellar_full[m] = stellar_uns
            resid_full[m] = resid_uns

            new_hdu = add_or_replace_column(hdu, args.oh_col_name, oh_full)
            new_hdu = add_or_replace_column(new_hdu, args.stellar_col_name, stellar_full)
            new_hdu = add_or_replace_column(new_hdu, args.resid_col_name, resid_full)

            out_hdus.append(new_hdu)
            
            print(name, "finite rows for OH =", m.sum(), "/", len(m))
            print(name, "resid_pre finite   =", np.isfinite(resid_pre).sum(), "/", len(resid_pre))

        args.out_fits.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList(out_hdus).writeto(args.out_fits, overwrite=True)

        print("WROTE FITS =", args.out_fits)

if __name__ == "__main__":
    main()