#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step10b — apply empirical O2 telluric correction to extracted spectra.

Pipeline meaning
----------------
  Step09 = OH refine
  Step10 = telluric
  
PYTHONPATH=. python pipeline/step10_telluric/step10b_apply_telluric.py \
  --infile ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/09_abab/extract1d_optimal_ridge_all_wav_step09_abab_preferred_consensus.fits \
  --template ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/10_telluric/telluric_O2_template.fits \
  --outfile ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/10_telluric/extract1d_optimal_ridge_all_wav_step09_abab_preferred_consensus_tellcorr.fits  
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
import config


def parse_args():
    p = argparse.ArgumentParser(description="Apply empirical O2 telluric correction")
    p.add_argument(
        "--infile",
        type=Path,
        default=None,
        help="Input extracted MEF from Step09 "
             "(default: ST09 merged ABAB product)",
    )
    p.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Telluric template FITS "
             "(default: ST10_TELLURIC/telluric_O2_template.fits)",
    )
    p.add_argument(
        "--outfile",
        type=Path,
        default=None,
        help="Output telluric-corrected MEF "
             "(default: ST10_TELLURIC/extract1d_optimal_ridge_all_wav_ohclean_tellcorr.fits)",
    )
    p.add_argument(
        "--max-shift-nm",
        type=float,
        default=1.0,
        help="Maximum accepted absolute O2 template shift in either band (nm)",
    )
    p.add_argument(
        "--max-obj-ratio",
        type=float,
        default=0.95,
        help="Require fitted objective / no-telluric objective below this value",
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
DEFAULT_INFILE = Path(
    getattr(
        config,
        "EXTRACT1D_STEP09_CONSENSUS",
        Path(config.ST09_OH_REFINE)
        / "extract1d_optimal_ridge_all_wav_step09_abab_preferred_consensus.fits",
    )
)
DEFAULT_OUTFILE = Path(
    getattr(
        config,
        "EXTRACT1D_TELLCOR",
        ST10 / "extract1d_optimal_ridge_all_wav_step09_abab_preferred_consensus_tellcorr.fits",
    )
)

DEFAULT_TEMPLATE = Path(
    getattr(
        config,
        "TELLURIC_TEMPLATE",
        ST10 / "telluric_O2_template.fits",
    )
)

A_CONT_L = (750.5, 758.0)
A_CONT_R = (770.5, 774.5)
B_CONT_L = (678.0, 684.0)
B_CONT_R = (694.0, 704.0)
SHIFT_NM_GRID_A = np.linspace(-6.0, +6.0, 481)
SHIFT_NM_GRID_B = np.linspace(-4.0, +4.0, 321)
USE_CONTINUUM_TWEAK = False
T_MIN, T_MAX = 0.02, 1.5
TAU_MIN_FRAC = 0.02
TAU_WEIGHT_POWER = 2.0
A_LO, A_HI = 748.0, 772.0
B_LO, B_HI = 680.0, 695.0
B_CONT_L = (678.0, 684.0)
B_CONT_R = (694.0, 704.0)

A_CORE_WEIGHT = 4.0
A_RED_WING_WEIGHT = 0.4
A_CORE_RANGE = (759.5, 762.5)
A_RED_WING_RANGE = (764.0, 772.0)



def finite(x):
    return np.isfinite(x)


def pick_flux_column(cols):
    cols_u = {c.upper(): c for c in cols}
    preferred = [
        "STELLAR_CONSENSUS",
        "STELLAR",
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
    preferred = ["VAR_APCORR", "VAR", "VAR_ADU_S2"]
    cols_u = {c.upper(): c for c in cols}
    for key in preferred:
        if key in cols_u:
            return cols_u[key]
    return None


def interp_tau(lam_grid, tau_grid, lam_eval):
    return np.interp(lam_eval, lam_grid, tau_grid, left=np.nan, right=np.nan)


def weighted_lstsq(y, X, w):
    m = finite(y) & finite(w) & (w > 0) & finite(X).all(axis=1)
    if m.sum() < X.shape[1] + 10:
        return None
    yy = y[m]
    XX = X[m]
    ww = np.sqrt(w[m])[:, None]
    beta, *_ = np.linalg.lstsq(XX * ww, yy * ww[:, 0], rcond=None)
    return beta


def fit_one_window_weighted(lam, flux_norm, tau_lam, tau_tau, lo, hi, shift_nm):
    m = finite(lam) & finite(flux_norm) & (lam > lo) & (lam < hi)
    x = lam[m]
    f = flux_norm[m]
    if x.size < 10:
        return None
    y = np.log(np.clip(f, 1e-6, 1e6))
    tau = interp_tau(tau_lam, tau_tau, x - shift_nm)
    if not finite(tau).any():
        return None
    tau_pk = np.nanmax(tau_tau)
    if (not np.isfinite(tau_pk)) or (tau_pk <= 0):
        return None
    tau_norm = tau / tau_pk
    inband = finite(tau_norm) & (tau_norm > TAU_MIN_FRAC)
    if inband.sum() < 10:
        return None
    x = x[inband]
    y = y[inband]
    tau = tau[inband]
    tau_norm = tau_norm[inband]
    x0 = np.nanmedian(x)
    
    w = np.clip(np.clip(tau_norm, 0.0, 1.0) ** TAU_WEIGHT_POWER, 1e-4, None)

    # Extra metric weighting for the O2 A-band:
    # lock the fit to the sharp core and downweight the broad red-side residuals.
    if 748.0 <= lo <= 752.0 and 768.0 <= hi <= 774.0:
        core = (x > A_CORE_RANGE[0]) & (x < A_CORE_RANGE[1])
        red_wing = (x > A_RED_WING_RANGE[0]) & (x < A_RED_WING_RANGE[1])
    
        w[core] *= A_CORE_WEIGHT
        w[red_wing] *= A_RED_WING_WEIGHT
    
    
    if USE_CONTINUUM_TWEAK:
        X = np.vstack([np.ones_like(x), (x - x0), -tau]).T
        beta = weighted_lstsq(y, X, w)
        if beta is None:
            return None
        c0, c1, a = beta
        model = c0 + c1 * (x - x0) - a * tau
    else:
        X = np.vstack([np.ones_like(x), -tau]).T
        beta = weighted_lstsq(y, X, w)
        if beta is None:
            return None
        c0, a = beta
        c1 = 0.0
        model = c0 - a * tau
    resid = y - model
    wrms = np.sqrt(np.nansum(w * resid ** 2) / np.nansum(w))
    return {"a": float(a), "c0": float(c0), "c1": float(c1), "wrms": float(wrms), "n": int(inband.sum()), "w_sum": float(np.nansum(w))}


def fit_cont_sidebands(lam, flux, side_left, side_right):
    lam = np.asarray(lam, float)
    flux = np.asarray(flux, float)
    mL = np.isfinite(lam) & np.isfinite(flux) & (lam >= side_left[0]) & (lam <= side_left[1])
    mR = np.isfinite(lam) & np.isfinite(flux) & (lam >= side_right[0]) & (lam <= side_right[1])
    m = mL | mR
    if m.sum() < 4:
        return np.full_like(lam, np.nan, dtype=float)
    x = lam[m]
    y = flux[m]
    p = np.polyfit(x, y, 1)
    return np.polyval(p, lam)


def normalize_band_local(lam, flux, band_lo, band_hi, side_left, side_right):
    """
    Fit a local linear continuum using sidebands, then divide the spectrum by it.
    The normalized spectrum is used only for fitting the telluric template.
    The final correction is still applied multiplicatively to the original flux.
    """
    cont = fit_cont_sidebands(lam, flux, side_left, side_right)

    m = (
        np.isfinite(lam)
        & np.isfinite(flux)
        & np.isfinite(cont)
        & (cont != 0)
        & (lam > band_lo)
        & (lam < band_hi)
    )

    if m.sum() < 5:
        return np.array([]), np.array([])

    fnorm = flux[m] / cont[m]

    # remove remaining scalar offset so the local continuum is near unity
    edge = (
        ((lam[m] < band_lo + 0.20 * (band_hi - band_lo)) |
         (lam[m] > band_hi - 0.20 * (band_hi - band_lo)))
        & np.isfinite(fnorm)
    )

    if edge.sum() >= 5:
        scale = np.nanmedian(fnorm[edge])
        if np.isfinite(scale) and scale != 0:
            fnorm = fnorm / scale

    return lam[m], fnorm

def build_transmission_decoupled(lam, shift_A, a_A, shift_B, a_B, lamA, tauA, lamB, tauB):
    T = np.ones_like(lam, dtype=float)
    mA = finite(lam) & (lam > A_LO) & (lam < A_HI) & np.isfinite(a_A) & np.isfinite(shift_A)
    if mA.any():
        tau_A = interp_tau(lamA, tauA, lam[mA] - shift_A)
        T[mA] = np.exp(-a_A * np.nan_to_num(tau_A, nan=0.0))
    mB = finite(lam) & (lam > B_LO) & (lam < B_HI) & np.isfinite(a_B) & np.isfinite(shift_B)
    if mB.any():
        tau_B = interp_tau(lamB, tauB, lam[mB] - shift_B)
        T[mB] = np.exp(-a_B * np.nan_to_num(tau_B, nan=0.0))
    return np.clip(T, T_MIN, T_MAX)


def best_band_solution(lam_obs, fnorm, lamT, tauT, lo, hi, shift_grid):
    if lam_obs.size < 5:
        return None
    best = None
    shift_max = max(abs(shift_grid[0]), abs(shift_grid[-1]))
    for sh in shift_grid:
        fit = fit_one_window_weighted(lam_obs, fnorm, lamT, tauT, lo, hi, sh)
        if fit is None:
            continue
        if (best is None) or (fit["wrms"] < best["obj"]):
            best = {"obj": float(fit["wrms"]), "fit": fit, "shift": float(sh), "a": float(fit["a"]), "n": int(fit["n"]), "w_sum": float(fit["w_sum"])}
    if best is None:
        return None
    return best

def best_band_solution_grid_metric(lam_obs, fnorm, lamT, tauT, shift_grid, a_grid, band="A"):
    if lam_obs.size < 20:
        return None

    weight = np.ones_like(lam_obs, dtype=float)

    if band.upper() == "A":
        weight[(lam_obs > 759.5) & (lam_obs < 762.5)] = 4.0
        weight[(lam_obs > 764.0) & (lam_obs < 772.0)] = 0.4
    
    elif band.upper() == "B":
        # O2 B-band: emphasize the coherent absorption core,
        # but do not chase noisy continuum structure.
        weight[(lam_obs > 686.6) & (lam_obs < 688.4)] = 4.0
        weight[(lam_obs > 688.4) & (lam_obs < 691.5)] = 1.5
        weight[(lam_obs < 685.5) | (lam_obs > 692.0)] = 0.3
    

    best = None

    for sh in shift_grid:
        tau = interp_tau(lamT, tauT, lam_obs - sh)
        if not np.isfinite(tau).any():
            continue

        for a in a_grid:
            T = np.exp(-a * tau)

            good = (
                np.isfinite(fnorm)
                & np.isfinite(T)
                & (T > 0.02)
                & np.isfinite(weight)
                & (weight > 0)
            )

            if good.sum() < 40:
                continue

            corr = fnorm[good] / T[good]
            med = np.nanmedian(corr)

            if not np.isfinite(med) or med == 0:
                continue

            corr = corr / med
            resid = corr - 1.0
            wg = weight[good]

            obj = np.sqrt(np.nansum(wg * resid**2) / np.nansum(wg))

            if best is None or obj < best["obj"]:
                best = {
                    "obj": float(obj),
                    "shift": float(sh),
                    "a": float(a),
                    "n": int(good.sum()),
                    "w_sum": float(np.nansum(wg)),
                    "fit": {
                        "a": float(a),
                        "c0": 0.0,
                        "c1": 0.0,
                        "wrms": float(obj),
                        "n": int(good.sum()),
                        "w_sum": float(np.nansum(wg)),
                    },
                }

    return best


def accept_band_solution(sol, null_sol, max_shift_nm, max_obj_ratio):
    """Return (accepted, objective_ratio, reason) for one O2 band."""
    if sol is None:
        return False, np.nan, "NO_FIT"
    if not np.isfinite(sol.get("a", np.nan)) or sol["a"] <= 0:
        return False, np.nan, "NONPOS_AMP"
    if not np.isfinite(sol.get("shift", np.nan)):
        return False, np.nan, "BAD_SHIFT"
    if abs(sol["shift"]) > max_shift_nm:
        return False, np.nan, "SHIFT"
    if (
        null_sol is None
        or not np.isfinite(null_sol.get("obj", np.nan))
        or null_sol["obj"] <= 0
        or not np.isfinite(sol.get("obj", np.nan))
    ):
        return False, np.nan, "NO_NULL"
    ratio = float(sol["obj"] / null_sol["obj"])
    if ratio >= max_obj_ratio:
        return False, ratio, "WEAK_GAIN"
    return True, ratio, "OK"


def add_or_replace_column(tab, name, data, fmt="E"):
    name_u = name.upper()
    cols = []
    for i, colname in enumerate(tab.names):
        if colname.upper() == name_u:
            continue
        cols.append(fits.Column(name=colname, format=tab.columns[i].format, array=tab[colname]))
    cols.append(fits.Column(name=name_u, format=fmt, array=data))
    return fits.BinTableHDU.from_columns(cols).data


def main():
    args = parse_args()

    infile = args.infile if args.infile else DEFAULT_INFILE
    template = args.template if args.template else DEFAULT_TEMPLATE
    outfile = args.outfile if args.outfile else DEFAULT_OUTFILE
    outfile.parent.mkdir(parents=True, exist_ok=True)

    if not infile.exists():
        raise FileNotFoundError(infile)
    if not template.exists():
        raise FileNotFoundError(f"Missing {template} (run Step10a first)")

    with fits.open(template) as h:
        tabB = h["O2_BAND"].data
        lamB = np.asarray(tabB["LAMBDA_NM"], float)
        tauB = np.asarray(tabB["TAU_O2"], float)

        tabA = h["O2_ABAND"].data
        lamA = np.asarray(tabA["LAMBDA_NM"], float)
        tauA = np.asarray(tabA["TAU_O2"], float)

    print("INFILE  =", infile)
    print("TEMPLATE=", template)
    print("OUTFILE =", outfile)
    
    with fits.open(infile) as hdutmp:
        phdr = hdutmp[0].header.copy()

    out_hdus = [fits.PrimaryHDU(header=phdr)]
    out_hdus[0].header["PIPESTEP"] = "STEP10"
    out_hdus[0].header["STAGE"] = "10b"
    out_hdus[0].header["SRCFILE"] = infile.name
    out_hdus[0].header["TEMPLATE"] = template.name
    out_hdus[0].header["TELLCOR"] = "O2_ABDECW"
    out_hdus[0].header["AWRCORE"] = (float(A_CORE_WEIGHT), "Step10 A-band core fit weight")
    out_hdus[0].header["AWRRED"] = (float(A_RED_WING_WEIGHT), "Step10 A-band red-wing fit weight")
    out_hdus[0].header["TELMSH"] = (float(args.max_shift_nm), "Maximum accepted abs O2 shift (nm)")
    out_hdus[0].header["TELOBJR"] = (float(args.max_obj_ratio), "Maximum fitted/null objective ratio")
    
    n_seen = n_written = n_ok = n_fail = n_varcorr = n_okA = n_okB = n_okAB = 0
    with fits.open(infile) as hdul:
        for hdu in hdul[1:]:
            d = hdu.data
            cols = getattr(d, "columns", None)
            if d is None or cols is None or "LAMBDA_NM" not in d.columns.names:
                continue
            flux_col = pick_flux_column(d.columns.names)
            if flux_col is None:
                continue
            var_col = pick_var_column(d.columns.names)
            n_seen += 1
            lam = np.asarray(d["LAMBDA_NM"], float)
            flux = np.asarray(d[flux_col], float)
            var = np.asarray(d[var_col], float) if var_col is not None else None
            lamA_obs, fnA = normalize_band_local(lam, flux, A_LO, A_HI, A_CONT_L, A_CONT_R)
            lamB_obs, fnB = normalize_band_local(lam, flux, B_LO, B_HI, B_CONT_L, B_CONT_R)
            solA = best_band_solution_grid_metric(
                lamA_obs,
                fnA,
                lamA,
                tauA,
                SHIFT_NM_GRID_A,
                np.linspace(0.0, 1.5, 241),
                band="A",
            )
                        
            solB = best_band_solution_grid_metric(
                lamB_obs,
                fnB,
                lamB,
                tauB,
                SHIFT_NM_GRID_B,
                np.linspace(0.0, 1.5, 241),
                band="B",
            )
                        
            # Validate each fitted band against a no-telluric baseline.
            nullA = best_band_solution_grid_metric(
                lamA_obs,
                fnA,
                lamA,
                tauA,
                np.array([0.0]),
                np.array([0.0]),
                band="A",
            )
            nullB = best_band_solution_grid_metric(
                lamB_obs,
                fnB,
                lamB,
                tauB,
                np.array([0.0]),
                np.array([0.0]),
                band="B",
            )

            okA, ratio_A, reason_A = accept_band_solution(
                solA, nullA, args.max_shift_nm, args.max_obj_ratio
            )
            okB, ratio_B, reason_B = accept_band_solution(
                solB, nullB, args.max_shift_nm, args.max_obj_ratio
            )

            print(
                hdu.name,
                "A:",
                None if solA is None else (
                    solA["shift"], solA["a"], solA["obj"], ratio_A, reason_A
                ),
                "B:",
                None if solB is None else (
                    solB["shift"], solB["a"], solB["obj"], ratio_B, reason_B
                ),
            )

            if not (okA or okB):
                tab2 = add_or_replace_column(d, "FLUX_TELLCOR_O2", np.asarray(flux, np.float32), fmt="E")
                if var is not None:
                    tab2 = add_or_replace_column(tab2, "VAR_TELLCOR_O2", np.asarray(var, np.float32), fmt="E")
                out_hdu = fits.BinTableHDU(data=tab2, name=hdu.name)
                for k, v in hdu.header.items():
                    if k not in out_hdu.header:
                        out_hdu.header[k] = v
                out_hdu.header["TELL_OK"] = False
                out_hdu.header["TELL_OKA"] = False
                out_hdu.header["TELL_OKB"] = False
                out_hdu.header["TELL_IN"] = str(flux_col)
                out_hdu.header["TELLVAR"] = (str(var_col) if var_col is not None else "NONE")
                out_hdu.header["TELL_RA"] = (float(ratio_A) if np.isfinite(ratio_A) else -999.0, "A fitted/null objective ratio")
                out_hdu.header["TELL_RB"] = (float(ratio_B) if np.isfinite(ratio_B) else -999.0, "B fitted/null objective ratio")
                out_hdu.header["TELREJA"] = (str(reason_A), "A-band acceptance status")
                out_hdu.header["TELREJB"] = (str(reason_B), "B-band acceptance status")
                out_hdu.header["TELLBAND"] = "NONE"
                out_hdus.append(out_hdu)
                n_written += 1
                n_fail += 1
                continue

            shift_A = solA["shift"] if okA else np.nan
            shift_B = solB["shift"] if okB else np.nan
            a_A = solA["a"] if okA else np.nan
            a_B = solB["a"] if okB else np.nan
            T = build_transmission_decoupled(lam, shift_A, a_A, shift_B, a_B, lamA, tauA, lamB, tauB)
            flux_corr = flux / T
            tab2 = add_or_replace_column(d, "FLUX_TELLCOR_O2", np.asarray(flux_corr, np.float32), fmt="E")
            if var is not None:
                var_corr = var / (T ** 2)
                tab2 = add_or_replace_column(tab2, "VAR_TELLCOR_O2", np.asarray(var_corr, np.float32), fmt="E")
                n_varcorr += 1
            out_hdu = fits.BinTableHDU(data=tab2, name=hdu.name)
            for k, v in hdu.header.items():
                if k not in out_hdu.header:
                    out_hdu.header[k] = v
            if okA:
                n_okA += 1
            if okB:
                n_okB += 1
            if okA and okB:
                n_okAB += 1
            vals = []
            shifts = []
            objs = []
            bands = []
            if okA:
                vals.append(a_A); shifts.append(shift_A); objs.append(solA["obj"]); bands.append("A")
            if okB:
                vals.append(a_B); shifts.append(shift_B); objs.append(solB["obj"]); bands.append("B")
            out_hdu.header["TELL_OK"] = True
            out_hdu.header["TELL_OKA"] = bool(okA)
            out_hdu.header["TELL_OKB"] = bool(okB)
            out_hdu.header["TELL_IN"] = str(flux_col)
            out_hdu.header["TELLVAR"] = (str(var_col) if var_col is not None else "NONE")
            out_hdu.header["TELL_SH"] = float(np.nanmedian(shifts)) if shifts else -999.0
            out_hdu.header["TELL_SHA"] = float(shift_A) if okA else -999.0
            out_hdu.header["TELL_SHB"] = float(shift_B) if okB else -999.0
            out_hdu.header["TELL_A"] = float(np.nanmedian(vals)) if vals else -999.0
            out_hdu.header["TELL_AA"] = float(a_A) if okA else -999.0
            out_hdu.header["TELL_AB"] = float(a_B) if okB else -999.0
            out_hdu.header["TELL_OBJ"] = float(np.nanmean(objs)) if objs else -999.0
            out_hdu.header["TELL_RA"] = (float(ratio_A) if np.isfinite(ratio_A) else -999.0, "A fitted/null objective ratio")
            out_hdu.header["TELL_RB"] = (float(ratio_B) if np.isfinite(ratio_B) else -999.0, "B fitted/null objective ratio")
            out_hdu.header["TELREJA"] = (str(reason_A), "A-band acceptance status")
            out_hdu.header["TELREJB"] = (str(reason_B), "B-band acceptance status")
            out_hdu.header["TELLBAND"] = "+".join(bands) if bands else "NONE"
            out_hdu.header["TELNOTE"] = ("O2 weighted AB dec", "Telluric note")
            out_hdus.append(out_hdu)
            n_written += 1
            n_ok += 1

    if n_written == 0:
        raise RuntimeError(f"Step10 telluric: wrote 0 slit extensions. Input: {infile}")

    fits.HDUList(out_hdus).writeto(outfile, overwrite=True)
    print(f"Wrote {outfile}")
    print(f"Slits seen={n_seen} written={n_written} TELL_OK={n_ok} pass-through={n_fail} var_corrected={n_varcorr}")
    print(f"Band fits: OK_A={n_okA} OK_B={n_okB} OK_A+B={n_okAB}")

if __name__ == "__main__":
    main()
