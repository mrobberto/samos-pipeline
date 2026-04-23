#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step11d — Empirical Flux Calibration Refinement (Per-Slit)

This step refines the Step11c flux calibration by enforcing consistency with
broadband SkyMapper photometry on a slit-by-slit basis.

Philosophy
----------
Step11c establishes the absolute flux scale and remains the baseline calibrated
product. Step11d applies an additional smooth multiplicative correction to
reduce residual large-scale response mismatches, especially toward the spectral
edges, while preserving the per-slit nature of the spectra.

The refinement is empirical: it uses external photometric constraints rather
than an instrumental response model. It is therefore intended as a broadband
shape correction, not a replacement for the underlying extraction or absolute
flux calibration.

Method
------
For each slit:

1. Read the Step11c calibrated spectrum from `FLUX_FLAM`.
2. Read the matched SkyMapper r, i, z photometry.
3. Construct the photometric constraints in one of two modes:

   - `full`
       Use the standard SkyMapper r/i/z bandpasses unchanged.

   - `edge_matched`
       Use truncated bandpasses to reduce sensitivity to low-throughput or
       contaminated spectral edges:
         * r_short: keep only λ >= 600 nm
         * i: unchanged
         * z_short: keep only λ <= 930 nm

4. In `edge_matched` mode, derive synthetic magnitudes for `r_short` and
   `z_short` from the catalog r/i/z photometry by interpolating the source SED
   in f_nu and integrating it over the truncated bandpasses.
5. Solve for a smooth quadratic multiplicative response function R_11d(λ) such
   that the band-integrated spectrum matches the target broadband constraints.
6. Apply the correction only if the solution satisfies quality criteria
   (positivity, conditioning, smoothness, and band residual checks). Otherwise,
   retain the original Step11c spectrum.

Formally,

    F_lambda_refined(λ) = R_11d(λ) * F_lambda_11c(λ)

where `R_11d(λ)` is a per-slit quadratic function of wavelength.

Outputs
-------
Per-slit refined product:
- `Extract1D_fluxcal_refined_perstar_<bandpass_mode>.fits`

Diagnostic products:
- `Extract1D_fluxcal_step11d_summary.csv`
- `Extract1D_fluxcal_step11d_debug.csv`
- `Extract1D_fluxcal_step11d_metadata.json`

Optional master products (if master mode is used):
- `Extract1D_fluxcal_refined_master_<bandpass_mode>.fits`
- `Extract1D_fluxcal_step11d_master_response.fits`

Key output columns
------------------
- `FLUX_FLAM`
    Step11c calibrated spectrum in physical flux-density units
    (erg s^-1 cm^-2 A^-1)

- `VAR_FLAM2`
    Propagated variance associated with `FLUX_FLAM`

- `RESP_STEP11D`
    Dimensionless multiplicative refinement response

- `FLUX_FLAM_REFINED`
    Refined spectrum after applying `RESP_STEP11D`

- `VAR_FLAM2_REFINED`
    Variance after propagation through the multiplicative refinement

Notes
-----
- The `edge_matched` mode uses fixed cuts at 600 nm and 930 nm by design.
  These cuts are intended to suppress edge regions affected by grating/coating
  roll-off or contamination, while keeping the broadband constraints tied to
  the reliable spectral domain.
- The refinement is constrained by integrated band fluxes, not by pointwise
  matching of flux densities.
- This implementation has been checked against the single-slit validation
  script `1slittester.py`, which serves as the reference solver for debugging
  and numerical verification.

Typical usage
-------------
PYTHONPATH=. python pipeline/step11_fluxcal/step11d_refine_fluxcal.py \
  --id-col slit \
  --r-col r_mag \
  --i-col i_mag \
  --z-col z_mag \
  --mode perstar \
  --bandpass-mode edge_matched
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table

LAM_EFF_NM = {"r": 620.0, "i": 750.0, "z": 870.0}
R_SHORT_CUT_NM = 600.0
Z_SHORT_CUT_NM = 930.0


# ----------------------------------------------------------------------
# Basic helpers (kept in cgs to match the tested one-slit implementation)
# ----------------------------------------------------------------------
def trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def load_two_col_ascii(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, comments="#", usecols=(0, 1))
    wave = np.asarray(arr[:, 0], float)
    tran = np.asarray(arr[:, 1], float)
    order = np.argsort(wave)
    return wave[order], np.clip(tran[order], 0.0, None)


def interp_bandpass(wave_spec_nm: np.ndarray,
                    wave_band_nm: np.ndarray,
                    tran_band: np.ndarray) -> np.ndarray:
    return np.interp(wave_spec_nm, wave_band_nm, tran_band, left=0.0, right=0.0)


def abmag_to_fnu_cgs(mag_ab: float) -> float:
    return 3631.0 * 10 ** (-0.4 * mag_ab) * 1e-23


def fnu_to_abmag(fnu_cgs: float) -> float:
    return -2.5 * np.log10(fnu_cgs / 3631e-23)


def abmag_to_flam_cgs(mag_ab: float, lam_nm: float) -> float:
    fnu_cgs = abmag_to_fnu_cgs(mag_ab)
    c_A_s = 2.99792458e18
    lam_A = lam_nm * 10.0
    return float(fnu_cgs * c_A_s / (lam_A ** 2))


def _to_float_or_nan(v) -> float:
    try:
        if np.ma.is_masked(v):
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def _is_finite_mag(v: float) -> bool:
    return np.isfinite(v) and (-50.0 < v < 99.0)



# ----------------------------------------------------------------------
# Filter shaping and synthetic photometry
# ----------------------------------------------------------------------
def make_r_short_filter(wr: np.ndarray, tr: np.ndarray, lam_min: float = R_SHORT_CUT_NM) -> Tuple[np.ndarray, np.ndarray]:
    m = wr >= lam_min
    wrs = wr[m].copy()
    trs = tr[m].copy()

    if wrs.size == 0 or wrs[0] > lam_min:
        tmin = np.interp(lam_min, wr, tr, left=0.0, right=0.0)
        wrs = np.insert(wrs, 0, lam_min)
        trs = np.insert(trs, 0, tmin)

    return wrs, np.clip(trs, 0.0, None)


def make_z_short_filter(wz: np.ndarray, tz: np.ndarray, lam_max: float) -> Tuple[np.ndarray, np.ndarray]:
    m = wz <= lam_max
    wzs = wz[m].copy()
    tzs = tz[m].copy()

    if wzs.size == 0 or wzs[-1] < lam_max:
        tmax = np.interp(lam_max, wz, tz, left=0.0, right=0.0)
        wzs = np.append(wzs, lam_max)
        tzs = np.append(tzs, tmax)

    return wzs, np.clip(tzs, 0.0, None)


def effective_lambda_nm(w: np.ndarray, t: np.ndarray) -> float:
    m = np.isfinite(w) & np.isfinite(t) & (t > 0)
    if np.count_nonzero(m) < 2:
        return np.nan
    return trapz(w[m] * t[m], w[m]) / trapz(t[m], w[m])


def predict_fnu_from_photometry(lam_query_nm: np.ndarray,
                                lam_pts_nm: np.ndarray,
                                mags_ab: np.ndarray) -> np.ndarray:
    lam_pts_nm = np.asarray(lam_pts_nm, float)
    mags_ab = np.asarray(mags_ab, float)
    m = np.isfinite(lam_pts_nm) & np.isfinite(mags_ab)
    lam_pts_nm = lam_pts_nm[m]
    mags_ab = mags_ab[m]

    fnu_pts = abmag_to_fnu_cgs(mags_ab)
    logfnu = np.log10(fnu_pts)

    order = np.argsort(lam_pts_nm)
    lam_pts_nm = lam_pts_nm[order]
    logfnu = logfnu[order]

    return 10 ** np.interp(lam_query_nm, lam_pts_nm, logfnu)


def synth_abmag_from_fnu_model(wband: np.ndarray,
                               tband: np.ndarray,
                               lam_pts_nm: np.ndarray,
                               mags_ab: np.ndarray) -> float:
    m = np.isfinite(wband) & np.isfinite(tband) & (tband > 0)
    w = np.asarray(wband[m], float)
    t = np.asarray(tband[m], float)

    if w.size < 2:
        return np.nan

    fnu = predict_fnu_from_photometry(w, lam_pts_nm, mags_ab)

    c_A_s = 2.99792458e18
    w_A = w * 10.0
    flam = fnu * c_A_s / (w_A ** 2)

    num = trapz(flam * t * w, w)
    den = trapz(t * w, w)
    if den <= 0:
        return np.nan
    flam_band = num / den

    lam_eff = effective_lambda_nm(w, t)
    if not np.isfinite(lam_eff):
        return np.nan

    lam_eff_A = lam_eff * 10.0
    fnu_eff = flam_band * lam_eff_A ** 2 / c_A_s
    if fnu_eff <= 0 or not np.isfinite(fnu_eff):
        return np.nan

    return fnu_to_abmag(fnu_eff)


def observed_band_target_ab(mag_ab: float,
                            wave_nm: np.ndarray,
                            tran: np.ndarray) -> float:
    fnu = abmag_to_fnu_cgs(mag_ab)
    c_ang_s = 2.99792458e18
    wave_ang = wave_nm * 10.0
    flam = fnu * c_ang_s / (wave_ang ** 2)
    return trapz(flam * tran * wave_nm, wave_nm)

def choose_scaled_coordinate(wave_nm: np.ndarray,
                             lambda0_nm: Optional[float] = None,
                             scale_nm: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
    if lambda0_nm is None:
        lambda0_nm = float(0.5 * (np.nanmin(wave_nm) + np.nanmax(wave_nm)))
    if scale_nm is None:
        scale_nm = float(0.5 * (np.nanmax(wave_nm) - np.nanmin(wave_nm)))
    x = (wave_nm - lambda0_nm) / scale_nm
    return x, float(lambda0_nm), float(scale_nm)

def scaled_to_micron_coeffs(a: float,
                            b: float,
                            c: float,
                            lambda0_nm: float,
                            scale_nm: float) -> Tuple[float, float, float]:
    """
    Convert R = a*x^2 + b*x + c with
        x = (lambda_nm - lambda0_nm) / scale_nm
    into
        R(lambda_um) = A_um * lambda_um^2 + B_um * lambda_um + C_um
    where lambda_um is wavelength in microns.
    """
    alpha = 1000.0 / scale_nm
    beta = -lambda0_nm / scale_nm

    A_um = a * alpha**2
    B_um = 2.0 * a * alpha * beta + b * alpha
    C_um = a * beta**2 + b * beta + c
    return float(A_um), float(B_um), float(C_um)



def build_design_row(flux: np.ndarray,
                     wave_nm: np.ndarray,
                     tran: np.ndarray,
                     x: np.ndarray) -> np.ndarray:
    w = tran * wave_nm
    return np.array([
        trapz(flux * x**2 * w, wave_nm),
        trapz(flux * x    * w, wave_nm),
        trapz(flux        * w, wave_nm),
    ], dtype=float)

def coverage_fraction(wb: np.ndarray,
                      tb: np.ndarray,
                      lam_min: float,
                      lam_max: float) -> float:
    mfull = np.isfinite(wb) & np.isfinite(tb) & (tb > 0)
    mcov = mfull & (wb >= lam_min) & (wb <= lam_max)
    if np.count_nonzero(mfull) < 2 or np.count_nonzero(mcov) < 2:
        return np.nan
    num = trapz(tb[mcov] * wb[mcov], wb[mcov])
    den = trapz(tb[mfull] * wb[mfull], wb[mfull])
    return num / den if den > 0 else np.nan

def synth_band_flam_mean(lam_nm: np.ndarray,
                         flam: np.ndarray,
                         band: Tuple[np.ndarray, np.ndarray]) -> float:
    """
    Return band-averaged f_lambda in the same units as the input flam.
    This is directly comparable to the plotted spectrum y-axis.
    """
    wb, tb = band
    ti = interp_bandpass(lam_nm, wb, tb)
    m = np.isfinite(lam_nm) & np.isfinite(flam) & (ti > 0)
    if np.count_nonzero(m) < 3:
        return np.nan

    num = trapz(flam[m] * ti[m] * lam_nm[m], lam_nm[m])
    den = trapz(ti[m] * lam_nm[m], lam_nm[m])
    return num / den if den > 0 else np.nan

# ----------------------------------------------------------------------
# One-slit exact solver: this is the heart of the rebuild
# ----------------------------------------------------------------------
def solve_one_slit_exact(
    wave_nm: np.ndarray,
    flux: np.ndarray,
    mags_full: Dict[str, float],
    bandpasses_full: Dict[str, Tuple[np.ndarray, np.ndarray]],
    bandpass_mode: str = "edge_matched",
) -> Tuple[
    np.ndarray, np.ndarray, float, float,
    Dict[str, float],
    Dict[str, Tuple[np.ndarray, np.ndarray]],
    Dict[str, float],
    np.ndarray,
]:
    """
    Ground-truth slit-by-slit solve, following the tested one-slit logic.

    Returns
    -------
    coeff, response, lambda0_nm, scale_nm, residuals, bandpasses_local, mags_work
    """
    wave_nm = np.asarray(wave_nm, float)
    flux = np.asarray(flux, float)

    order = np.argsort(wave_nm)
    wave_nm = wave_nm[order]
    flux = flux[order]

    wr, tr = bandpasses_full["r"]
    wi, ti = bandpasses_full["i"]
    wz, tz = bandpasses_full["z"]

    if bandpass_mode == "edge_matched":
        wrs, trs = make_r_short_filter(wr, tr, lam_min=R_SHORT_CUT_NM)
        wzs, tzs = make_z_short_filter(wz, tz, lam_max=Z_SHORT_CUT_NM)
        """
        lam_min = float(np.nanmin(wave_nm))
        lam_max = float(np.nanmax(wave_nm))
        
        r_cut = max(600.0, lam_min + 10.0)
        z_cut = min(930.0, lam_max - 10.0)
        
        wrs, trs = make_r_short_filter(wr, tr, lam_min=r_cut)
        wzs, tzs = make_z_short_filter(wz, tz, lam_max=z_cut)
        """

        lam_pts = np.array([LAM_EFF_NM["r"], LAM_EFF_NM["i"], LAM_EFF_NM["z"]], float)
        mag_pts = np.array([
            mags_full.get("r", np.nan),
            mags_full.get("i", np.nan),
            mags_full.get("z", np.nan),
        ], float)

        m_rshort = synth_abmag_from_fnu_model(wrs, trs, lam_pts, mag_pts)
        m_zshort = synth_abmag_from_fnu_model(wzs, tzs, lam_pts, mag_pts)

        bandpasses_local = {
            "r": (wrs, trs),
            "i": (wi, ti),
            "z": (wzs, tzs),
        }
        mags_work = {
            "r": m_rshort,
            "i": mags_full.get("i", np.nan),
            "z": m_zshort,
        }
        
    elif bandpass_mode == "full":
        bandpasses_local = dict(bandpasses_full)
        mags_work = dict(mags_full)
    else:
        raise ValueError(f"Unknown bandpass_mode={bandpass_mode}")

    x, lambda0_nm, scale_nm = choose_scaled_coordinate(wave_nm)

    rows, rhs, used = [], [], []
    for band in ["r", "i", "z"]:
        mag = mags_work[band]
        if not np.isfinite(mag):
            continue
        wb, tb = bandpasses_local[band]
        tran = interp_bandpass(wave_nm, wb, tb)
        if np.nanmax(tran) <= 0:
            continue
        rows.append(build_design_row(flux, wave_nm, tran, x))
        rhs.append(observed_band_target_ab(mag, wave_nm, tran))
        used.append(band)

    if len(rows) < 3:
        raise RuntimeError("Need at least 3 valid bands to solve for a quadratic refinement response.")

    M = np.vstack(rows)
    y = np.asarray(rhs, float)
    coeff = np.linalg.solve(M, y)
    response = coeff[0] * x**2 + coeff[1] * x + coeff[2]

    pred = M @ coeff
    residuals = {band: float((yp - yo) / yo) if yo != 0 else np.nan
                 for band, yp, yo in zip(used, pred, y)}
    
    print("\n[CHECK SHORT MAGS]")
    print("r_full:", mags_full.get("r"))
    print("r_short:", m_rshort)
    print("Δr:", m_rshort - mags_full.get("r"))
    
    print("z_full:", mags_full.get("z"))
    print("z_short:", m_zshort)
    print("Δz:", m_zshort - mags_full.get("z"))

    return coeff, response, lambda0_nm, scale_nm, residuals, bandpasses_local, mags_work, order


# ----------------------------------------------------------------------
# Quality control
# ----------------------------------------------------------------------
@dataclass
class PerStarFit:
    slit_id: str
    coeff_a: float
    coeff_b: float
    coeff_c: float
    coeff_um_a2: float
    coeff_um_b1: float
    coeff_um_c0: float
    lambda0_nm: float
    scale_nm: float
    cond: float
    rms_rel_band: float
    n_good_bands: int
    accepted: bool
    reason: str
    band_residual_g: float
    band_residual_r: float
    band_residual_i: float
    band_residual_z: float


def evaluate_quality(response: np.ndarray,
                     residuals: Dict[str, float],
                     response_positive_frac_min: float,
                     cond_max: float,
                     band_rms_rel_max: float,
                     response_median_min: float,
                     response_median_max: float,
                     cond: float) -> Tuple[bool, str, float]:
    positive_frac = float(np.mean(np.isfinite(response) & (response > 0)))
    rels = np.array([residuals.get("g", np.nan),
                     residuals.get("r", np.nan),
                     residuals.get("i", np.nan),
                     residuals.get("z", np.nan)], dtype=float)
    rms_rel = float(np.sqrt(np.nanmean(rels**2)))
    med = float(np.nanmedian(response)) if np.any(np.isfinite(response)) else np.nan

    reasons = []
    ok = True

    if positive_frac < response_positive_frac_min:
        ok = False
        reasons.append(f"positive_frac={positive_frac:.3f} < {response_positive_frac_min:.3f}")

    if (not np.isfinite(cond)) or cond > cond_max:
        ok = False
        reasons.append(f"cond={cond:.3e} > {cond_max:.3e}")

    if (not np.isfinite(rms_rel)) or rms_rel > band_rms_rel_max:
        ok = False
        reasons.append(f"band_rms_rel={rms_rel:.3e} > {band_rms_rel_max:.3e}")

    if (not np.isfinite(med)) or (med < response_median_min) or (med > response_median_max):
        ok = False
        reasons.append(f"median_response={med:.3g} outside [{response_median_min:.3g}, {response_median_max:.3g}]")

    return ok, "accepted" if ok else "; ".join(reasons), rms_rel


# ----------------------------------------------------------------------
# FITS / table plumbing
# ----------------------------------------------------------------------
def find_table_hdus(hdul: fits.HDUList) -> List[int]:
    return [idx for idx, hdu in enumerate(hdul) if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU))]


def get_column_name(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    cset = {c.upper(): c for c in columns}
    for cand in candidates:
        if cand.upper() in cset:
            return cset[cand.upper()]
    return None


def normalize_slit_id(value) -> str:
    s = str(value).strip().upper()
    if s == "":
        return s

    if s.startswith("SLIT"):
        tail = s[4:].strip()
        try:
            return f"SLIT{int(tail):03d}"
        except Exception:
            return s

    try:
        return f"SLIT{int(float(s)):03d}"
    except Exception:
        return s


def infer_slit_id_from_hdu(hdu, idx: int) -> str:
    hdr = hdu.header
    for key in ("EXTNAME", "NAME", "OBJECT"):
        if key in hdr:
            val = str(hdr[key]).strip()
            if val:
                return val
    for key in ("SLITID", "SLIT"):
        if key in hdr:
            try:
                return f"SLIT{int(hdr[key]):03d}"
            except Exception:
                val = str(hdr[key]).strip()
                if val:
                    return val
    return f"HDU{idx:03d}"


def read_phot_table(path: str | Path) -> Table:
    path = str(path)
    if path.lower().endswith((".fits", ".fit", ".fz")):
        return Table.read(path, format="fits")
    if path.lower().endswith(".ecsv"):
        return Table.read(path, format="ascii.ecsv")
    return Table.read(path)


def build_phot_lookup(tab: Table, id_col: str, r_col: str, i_col: str, z_col: str) -> Dict[str, Dict[str, float]]:
    out = {}
    for row in tab:
        sid = normalize_slit_id(row[id_col])
        out[sid] = {
            "r": _to_float_or_nan(row[r_col]),
            "i": _to_float_or_nan(row[i_col]),
            "z": _to_float_or_nan(row[z_col]),
        }
    return out


def choose_wave_column(tab: Table, wave_col: Optional[str]) -> str:
    cols = list(tab.colnames)
    if wave_col is None:
        wave_col = get_column_name(cols, ["LAMBDA_NM", "WAVELENGTH_NM", "WAVE_NM", "LAMBDA"])
    if wave_col is None:
        raise RuntimeError(f"Could not find wavelength column among {cols}")
    return wave_col


def choose_flux_var_columns(tab: Table, flux_col: Optional[str], var_col: Optional[str]) -> Tuple[str, Optional[str]]:
    cols = list(tab.colnames)
    if flux_col is None:
        flux_col = get_column_name(cols, ["FLUX_FLAM", "FLUX_CAL_FLAM"])
    if flux_col is None:
        raise RuntimeError(f"Could not find physical input flux column among {cols}")
    if var_col is None:
        var_col = get_column_name(cols, ["VAR_FLAM2", "VAR_CAL_FLAM2"])
    return flux_col, var_col


# ----------------------------------------------------------------------
# Main processing
# ----------------------------------------------------------------------
def process_file(infile: Path,
                 phot_lookup: Dict[str, Dict[str, float]],
                 bandpasses: Dict[str, Tuple[np.ndarray, np.ndarray]],
                 outdir: Path,
                 flux_col: Optional[str],
                 var_col: Optional[str],
                 wave_col: Optional[str],
                 mode: str,
                 response_positive_frac_min: float,
                 cond_max: float,
                 band_rms_rel_max: float,
                 response_median_min: float,
                 response_median_max: float,
                 combine_method: str,
                 allowed_slits: Optional[set[str]] = None,
                 bandpass_mode: str = "edge_matched") -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    with fits.open(infile) as hdul:
        table_hdus = find_table_hdus(hdul)

        hdul_out_master = fits.HDUList([fits.PrimaryHDU(header=hdul[0].header.copy())])
        hdul_out_perstar = fits.HDUList([fits.PrimaryHDU(header=hdul[0].header.copy())])
        per_star_rows: List[PerStarFit] = []
        debug_rows: List[dict] = []
        accepted_responses: List[np.ndarray] = []
        accepted_slits: List[str] = []
        common_wave = common_lambda0 = common_scale = None

        for idx in table_hdus:
            hdu = hdul[idx]
            tab = Table(hdu.data)

            slit_id = str(infer_slit_id_from_hdu(hdu, idx)).strip().upper()
            if allowed_slits is not None and slit_id not in allowed_slits:
                continue
            if slit_id not in phot_lookup:
                continue

            local_wave_col = choose_wave_column(tab, wave_col)
            local_flux_col, local_var_col = choose_flux_var_columns(tab, flux_col, var_col)

            wave_full = np.asarray(tab[local_wave_col], dtype=float)
            flux_full = np.asarray(tab[local_flux_col], dtype=float)
            good = np.isfinite(wave_full) & np.isfinite(flux_full)

            if np.count_nonzero(good) < 20:
                continue

            wave_nm = np.asarray(wave_full[good], float)
            flux = np.asarray(flux_full[good], float)
            
            mags_full = phot_lookup[slit_id]

            try:
                coeff, response, lambda0_nm, scale_nm, residuals, bandpasses_local, mags_work, order = solve_one_slit_exact(
                    wave_nm=wave_nm,
                    flux=flux,
                    mags_full=mags_full,
                    bandpasses_full=bandpasses,
                    bandpass_mode=bandpass_mode,
                )
            except Exception as exc:
                per_star_rows.append(
                    PerStarFit(
                        slit_id,
                        np.nan, np.nan, np.nan,          # coeff_a, coeff_b, coeff_c
                        np.nan, np.nan, np.nan,          # coeff_um_a2, coeff_um_b1, coeff_um_c0
                        np.nan, np.nan,                  # lambda0_nm, scale_nm
                        np.nan, np.nan,                  # cond, rms_rel_band
                        0,                               # n_good_bands
                        False,                           # accepted
                        f"solve_failed: {exc}",          # reason
                        np.nan, np.nan, np.nan, np.nan   # band_residual_g/r/i/z
                    )
                )
                continue
            wave_nm_sorted = np.asarray(wave_nm, float)[order]
            flux_sorted = np.asarray(flux, float)[order]
            response_sorted = response
            flux_refined_sorted = flux_sorted * response_sorted

            n_good_bands = int(sum(_is_finite_mag(mags_work[b]) for b in ("r", "i", "z")))

            # build condition number on the exact working solve
            x, _, _ = choose_scaled_coordinate(wave_nm_sorted, lambda0_nm, scale_nm)
            rows = []
            for band in ("r", "i", "z"):
                if not _is_finite_mag(mags_work[band]):
                    continue
                wb, tb = bandpasses_local[band]
                tran = interp_bandpass(wave_nm_sorted, wb, tb)
                rows.append(build_design_row(flux_sorted, wave_nm_sorted, tran, x))
                
            
            cond = float(np.linalg.cond(np.vstack(rows))) if len(rows) >= 3 else np.nan
            accepted, reason, rms_rel = evaluate_quality(
                response=response,
                residuals=residuals,
                response_positive_frac_min=response_positive_frac_min,
                cond_max=cond_max,
                band_rms_rel_max=band_rms_rel_max,
                response_median_min=response_median_min,
                response_median_max=response_median_max,
                cond=cond,
            )
            
            lam_min = float(np.nanmin(wave_nm_sorted))
            lam_max = float(np.nanmax(wave_nm_sorted))
                        
            # use the exact working bands returned by the solver
            bandpasses_local = bandpasses_local
                        
            # --- synthetic fluxes (Step11c) ---
            syn_r = synth_band_flam_mean(wave_nm_sorted, flux_sorted, bandpasses_local["r"])
            syn_i = synth_band_flam_mean(wave_nm_sorted, flux_sorted, bandpasses_local["i"])
            syn_z = synth_band_flam_mean(wave_nm_sorted, flux_sorted, bandpasses_local["z"])
            
            # --- interpolated transmissions ---
            tran_r = interp_bandpass(wave_nm_sorted, *bandpasses_local["r"])
            tran_i = interp_bandpass(wave_nm_sorted, *bandpasses_local["i"])
            tran_z = interp_bandpass(wave_nm_sorted, *bandpasses_local["z"])
            
            # --- targets ---
            tgt_r = observed_band_target_ab(mags_work["r"], wave_nm_sorted, tran_r) if _is_finite_mag(mags_work["r"]) else np.nan
            tgt_i = observed_band_target_ab(mags_work["i"], wave_nm_sorted, tran_i) if _is_finite_mag(mags_work["i"]) else np.nan
            tgt_z = observed_band_target_ab(mags_work["z"], wave_nm_sorted, tran_z) if _is_finite_mag(mags_work["z"]) else np.nan
            
            tgt_r_flam = abmag_to_flam_cgs(mags_work["r"], LAM_EFF_NM["r"]) if _is_finite_mag(mags_work["r"]) else np.nan
            tgt_i_flam = abmag_to_flam_cgs(mags_work["i"], LAM_EFF_NM["i"]) if _is_finite_mag(mags_work["i"]) else np.nan
            tgt_z_flam = abmag_to_flam_cgs(mags_work["z"], LAM_EFF_NM["z"]) if _is_finite_mag(mags_work["z"]) else np.nan

            # --- refined flux ---
            syn_r_ref = synth_band_flam_mean(wave_nm_sorted, flux_refined_sorted, bandpasses_local["r"])
            syn_i_ref = synth_band_flam_mean(wave_nm_sorted, flux_refined_sorted, bandpasses_local["i"])
            syn_z_ref = synth_band_flam_mean(wave_nm_sorted, flux_refined_sorted, bandpasses_local["z"])
            
            A_um, B_um, C_um = scaled_to_micron_coeffs(
                float(coeff[0]), float(coeff[1]), float(coeff[2]),
                float(lambda0_nm), float(scale_nm)
            )    
                    

            
            debug_rows.append({
                "slit_id": slit_id,
                "lam_min_nm": lam_min,
                "lam_max_nm": lam_max,
            
                "mag_r_cat": mags_full.get("r", np.nan),
                "mag_i_cat": mags_full.get("i", np.nan),
                "mag_z_cat": mags_full.get("z", np.nan),
            
                "mag_r_used": mags_work.get("r", np.nan),
                "mag_i_used": mags_work.get("i", np.nan),
                "mag_z_used": mags_work.get("z", np.nan),
            
                "cov_r": coverage_fraction(*bandpasses_local["r"], lam_min, lam_max),
                "cov_i": coverage_fraction(*bandpasses_local["i"], lam_min, lam_max),
                "cov_z": coverage_fraction(*bandpasses_local["z"], lam_min, lam_max),
            
                "flux_r_step11c": syn_r,
                "flux_i_step11c": syn_i,
                "flux_z_step11c": syn_z,
            
                "flux_r_target": tgt_r,
                "flux_i_target": tgt_i,
                "flux_z_target": tgt_z,
            
                "flux_r_refined": syn_r_ref,
                "flux_i_refined": syn_i_ref,
                "flux_z_refined": syn_z_ref,
            
                "ratio_r_target_over_step11c": tgt_r_flam / syn_r if np.isfinite(tgt_r_flam) and np.isfinite(syn_r) and syn_r != 0 else np.nan,
                "ratio_i_target_over_step11c": tgt_i_flam / syn_i if np.isfinite(tgt_i_flam) and np.isfinite(syn_i) and syn_i != 0 else np.nan,
                "ratio_z_target_over_step11c": tgt_z_flam / syn_z if np.isfinite(tgt_z_flam) and np.isfinite(syn_z) and syn_z != 0 else np.nan,
            
                "coeff_a": float(coeff[0]),
                "coeff_b": float(coeff[1]),
                "coeff_c": float(coeff[2]),
                "lambda0_nm": float(lambda0_nm),
                "scale_nm": float(scale_nm),
                "cond": float(cond),
            
                "resp_median": float(np.nanmedian(response)),
                "resp_min": float(np.nanmin(response)),
                "resp_max": float(np.nanmax(response)),
            
                "resid_r": float(residuals.get("r", np.nan)),
                "resid_i": float(residuals.get("i", np.nan)),
                "resid_z": float(residuals.get("z", np.nan)),
            
                "accepted": bool(accepted),
                "reason": reason,
                
                "coeff_um_a2": A_um,
                "coeff_um_b1": B_um,
                "coeff_um_c0": C_um,
            })

            if slit_id in ("SLIT000","SLIT036"):
                print("\n=== DEBUG "+slit_id+" ===")
                print("lambda range:", lam_min, lam_max)
                print("catalog mags:", mags_full)
                print("used mags   :", mags_work)
                print("coverage    :", {
                    "r": coverage_fraction(*bandpasses_local["r"], lam_min, lam_max),
                    "i": coverage_fraction(*bandpasses_local["i"], lam_min, lam_max),
                    "z": coverage_fraction(*bandpasses_local["z"], lam_min, lam_max),
                })
                print("step11c <f_lambda> :", {"r": syn_r, "i": syn_i, "z": syn_z})
                print("target  f_lambda   :", {"r": tgt_r_flam, "i": tgt_i_flam, "z": tgt_z_flam})
                print("refined <f_lambda> :", {"r": syn_r_ref, "i": syn_i_ref, "z": syn_z_ref})
                print("target/step11c :", {
                    "r": tgt_r_flam / syn_r if np.isfinite(tgt_r_flam) and np.isfinite(syn_r) and syn_r != 0 else np.nan,
                    "i": tgt_i_flam / syn_i if np.isfinite(tgt_i_flam) and np.isfinite(syn_i) and syn_i != 0 else np.nan,
                    "z": tgt_z_flam / syn_z if np.isfinite(tgt_z_flam) and np.isfinite(syn_z) and syn_z != 0 else np.nan,
                })
                print("coeff:", coeff)
                print("cond :", cond)
                print("resp median/min/max:", np.nanmedian(response), np.nanmin(response), np.nanmax(response))
                print("residuals:", residuals)
                print("accepted:", accepted, reason)
                
                


                print("poly in x:     ", coeff)
                print("poly in micron:", (A_um, B_um, C_um))
                print("R(lambda_um) = "
                      f"{A_um:.6f} * lambda_um^2 + "
                      f"{B_um:.6f} * lambda_um + "
                      f"{C_um:.6f}")    
                
            per_star_rows.append(PerStarFit(
                slit_id,
                float(coeff[0]), float(coeff[1]), float(coeff[2]),
                float(A_um), float(B_um), float(C_um),
                float(lambda0_nm), float(scale_nm), float(cond), float(rms_rel),
                int(n_good_bands), bool(accepted), reason, np.nan,
                float(residuals.get("r", np.nan)),
                float(residuals.get("i", np.nan)),
                float(residuals.get("z", np.nan))
                )
            )

            if accepted:
                if common_wave is None:
                    common_wave = wave_nm.copy()
                    common_lambda0 = lambda0_nm
                    common_scale = scale_nm
                if common_wave.shape == wave_nm.shape and np.allclose(common_wave, wave_nm, rtol=0, atol=1e-6):
                    accepted_responses.append(response.copy())
                    accepted_slits.append(slit_id)

            if mode in ("perstar", "both"):
                new_tab = tab.copy()

                response_write = np.empty_like(response)
                response_write[order] = response
                
                full_response = np.full(len(tab), np.nan, dtype=float)
                full_response[good] = response_write
                new_tab["RESP_STEP11D"] = full_response

                # Keep the same units as the input FLUX_FLAM
                full_flux_ref = np.asarray(tab[local_flux_col], dtype=float).copy()
                if accepted:
                    full_flux_ref[good] *= response_write
                new_tab["FLUX_FLAM_REFINED"] = full_flux_ref

                if local_var_col is not None:
                    full_var_ref = np.asarray(tab[local_var_col], dtype=float).copy()
                    if accepted:
                        full_var_ref[good] *= response_write**2
                    new_tab["VAR_FLAM2_REFINED"] = full_var_ref


                hdr = hdu.header.copy()
                hdr["STEP11D"] = ("REFINE", "Flux refinement relative to FLUX_FLAM")
                hdr["D_MODE"] = ("PERSTAR", "Refinement calibration mode")
                hdr["D_FLUX"] = (local_flux_col, "Input physical flux column")
                hdr["D_A"] = (float(coeff[0]), "Quadratic coeff A in scaled basis")
                hdr["D_B"] = (float(coeff[1]), "Quadratic coeff B in scaled basis")
                hdr["D_C"] = (float(coeff[2]), "Quadratic coeff C in scaled basis")
                hdr["D_L0"] = (float(lambda0_nm), "Scaled basis lambda0 [nm]")
                hdr["D_SCL"] = (float(scale_nm), "Scaled basis scale [nm]")
                hdr["DCOND"] = (float(cond), "Condition number of solve")
                hdr["NDBAND"] = (int(n_good_bands), "Number of photometric bands used")
                hdr["DACCPT"] = (int(accepted), "1 if refined solution accepted")
                hdul_out_perstar.append(fits.BinTableHDU(new_tab, header=hdr, name=hdu.name))

        master_response = None
        if accepted_responses:
            stack = np.vstack(accepted_responses)
            if combine_method == "median":
                master_response = np.nanmedian(stack, axis=0)
            else:
                master_response = np.nanmean(stack, axis=0)

        if mode in ("master", "both"):
            if master_response is None or common_wave is None:
                raise RuntimeError("No accepted calibration stars available for master refinement response.")

            for idx in table_hdus:
                hdu = hdul[idx]
                tab = Table(hdu.data)
                local_wave_col = choose_wave_column(tab, wave_col)
                local_flux_col, local_var_col = choose_flux_var_columns(tab, flux_col, var_col)

                wave_full = np.asarray(tab[local_wave_col], dtype=float)
                response_full = np.full(len(tab), np.nan, dtype=float)

                good = np.isfinite(wave_full)
                if np.count_nonzero(good) > 0:
                    wave_good = np.asarray(wave_full[good], dtype=float)

                    m = np.isfinite(common_wave) & np.isfinite(master_response)
                    wref = np.asarray(common_wave[m], dtype=float)
                    rref = np.asarray(master_response[m], dtype=float)

                    order = np.argsort(wref)
                    wref = wref[order]
                    rref = rref[order]

                    keep = np.concatenate(([True], np.diff(wref) > 0))
                    wref = wref[keep]
                    rref = rref[keep]

                    if wave_good.shape == wref.shape and np.allclose(wave_good, wref, rtol=0, atol=1e-6):
                        interp_resp = rref.copy()
                    else:
                        interp_resp = np.interp(wave_good, wref, rref, left=np.nan, right=np.nan)

                    response_full[good] = interp_resp

                new_tab = tab.copy()
                new_tab["RESP_STEP11D"] = response_full

                # Keep the same units as the input FLUX_FLAM
                full_flux_ref = np.asarray(tab[local_flux_col], dtype=float).copy()
                gmask = np.isfinite(full_flux_ref) & np.isfinite(response_full)
                full_flux_ref[gmask] *= response_full[gmask]
                new_tab["FLUX_FLAM_REFINED"] = full_flux_ref

                if local_var_col is not None:
                    full_var_ref = np.asarray(tab[local_var_col], dtype=float).copy()
                    gmaskv = np.isfinite(full_var_ref) & np.isfinite(response_full)
                    full_var_ref[gmaskv] *= response_full[gmaskv] ** 2
                    new_tab["VAR_FLAM2_REFINED"] = full_var_ref

                hdr = hdu.header.copy()
                hdr["STEP11D"] = ("REFINE", "Flux refinement relative to FLUX_FLAM")
                hdr["D_MODE"] = ("MASTER", "Refinement calibration mode")
                hdr["D_FLUX"] = (local_flux_col, "Input physical flux column")
                hdr["D_L0"] = (float(common_lambda0), "Scaled basis lambda0 [nm]")
                hdr["D_SCL"] = (float(common_scale), "Scaled basis scale [nm]")
                hdr["NDSTAR"] = (int(len(accepted_slits)), "Accepted refinement stars")
                hdul_out_master.append(fits.BinTableHDU(new_tab, header=hdr, name=hdu.name))
                
        for idx, hdu in enumerate(hdul[1:], start=1):
            if idx not in table_hdus:
                if mode in ("master", "both"):
                    hdul_out_master.append(hdu.copy())
                if mode in ("perstar", "both"):
                    hdul_out_perstar.append(hdu.copy())

        stem = infile.stem
        if mode in ("master", "both"):
            hdul_out_master.writeto(outdir / f"{stem}_refined_master_{bandpass_mode}.fits", overwrite=True)
        if mode in ("perstar", "both"):
            hdul_out_perstar.writeto(outdir / f"{stem}_refined_perstar_{bandpass_mode}.fits", overwrite=True)

        if master_response is not None and common_wave is not None:
            cols = [
                fits.Column(name="LAMBDA_NM", array=np.asarray(common_wave, dtype=np.float64), format="D"),
                fits.Column(name="RESP_MASTER", array=np.asarray(master_response, dtype=np.float64), format="D"),
            ]
            hdu_resp = fits.BinTableHDU.from_columns(cols, name="MASTER_RESPONSE")
            hdr = hdu_resp.header
            hdr["NSTAR"] = int(len(accepted_slits))
            hdr["METHOD"] = combine_method
            hdr["D_L0"] = float(common_lambda0)
            hdr["D_SCL"] = float(common_scale)
            fits.HDUList([fits.PrimaryHDU(), hdu_resp]).writeto(
                outdir / f"{stem}_step11d_master_response.fits",
                overwrite=True
            )

        csv_path = outdir / f"{stem}_step11d_summary.csv"
        fieldnames = [
            "slit_id",
            "coeff_a", "coeff_b", "coeff_c",
            "coeff_um_a2", "coeff_um_b1", "coeff_um_c0",
            "lambda0_nm", "scale_nm",
            "cond", "rms_rel_band", "n_good_bands", "accepted", "reason",
            "band_residual_g", "band_residual_r", "band_residual_i", "band_residual_z"
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_star_rows:
                writer.writerow(asdict(row))
                
        debug_csv_path = outdir / f"{stem}_step11d_debug.csv"
        
        print("[DEBUG] debug_rows length =", len(debug_rows))
        
        if not debug_rows:
            print("[WARNING] debug_rows is empty — writing empty debug file")
        
        # Always define fieldnames safely
        debug_fieldnames = list(debug_rows[0].keys()) if debug_rows else ["empty"]
        
        with open(debug_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=debug_fieldnames)
            writer.writeheader()
            if debug_rows:
                writer.writerows(debug_rows)
        
        print("[OK] Wrote", debug_csv_path)


        meta = {
            "input_file": str(infile),
            "output_dir": str(outdir),
            "mode": mode,
            "combine_method": combine_method,
            "n_input_rows": len(per_star_rows),
            "n_accepted_rows": int(sum(r.accepted for r in per_star_rows)),
            "accepted_slits": accepted_slits,
            "band_set": ["r", "i", "z"],
            "input_flux_column_default": flux_col or "FLUX_FLAM",
            "bandpass_mode": bandpass_mode,
        }
        with open(outdir / f"{stem}_step11d_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step11d refinement of FLUX_FLAM using SkyMapper AB photometry.")
    p.add_argument("--infile", default=None, help="Input Step11c FITS file containing FLUX_FLAM")
    p.add_argument("--phot-cat", default=None, help="Photometric catalog table")
    p.add_argument("--r-band", default=None, help="ASCII response curve for r")
    p.add_argument("--i-band", default=None, help="ASCII response curve for i")
    p.add_argument("--z-band", default=None, help="ASCII response curve for z")
    p.add_argument("--outdir", default=None, help="Output directory")
    p.add_argument("--id-col", default="SLITID", help="Identifier column in photometric table")
    p.add_argument("--r-col", required=True, help="AB magnitude column for r")
    p.add_argument("--i-col", required=True, help="AB magnitude column for i")
    p.add_argument("--z-col", required=True, help="AB magnitude column for z")
    p.add_argument("--combine-method", choices=["median", "mean"], default="median")
    p.add_argument("--flux-col", default=None, help="Override input physical flux column name (default: FLUX_FLAM)")
    p.add_argument("--var-col", default=None, help="Override input physical variance column name (default: VAR_FLAM2)")
    p.add_argument("--wave-col", default=None, help="Override wavelength column name (default: LAMBDA_NM)")
    p.add_argument("--response-positive-frac-min", type=float, default=0.90)
    p.add_argument("--cond-max", type=float, default=1e14)
    p.add_argument("--band-rms-rel-max", type=float, default=0.10)
    p.add_argument("--response-median-min", type=float, default=0.5)
    p.add_argument("--response-median-max", type=float, default=2.5)
    p.add_argument("--slit-list-file", default=None,
                   help="CSV file with ranked slits (e.g. step11d_rank_calibrators output)")
    p.add_argument("--top-n", type=int, default=None,
                   help="Use top N slits from slit-list-file")
    p.add_argument("--bandpass-mode", choices=["full", "edge_matched"], default="edge_matched",
                   help="Use original r/i/z filters or edge-matched r_short/i/z_short")
    p.add_argument("--mode", choices=["master", "perstar", "both"], default="perstar")
    return p


def main() -> None:
    import config

    args = build_argparser().parse_args()

    st11 = Path(config.ST11_FLUXCAL)

    infile = Path(args.infile) if args.infile else Path(config.EXTRACT1D_FLUXCAL)
    phot_cat = Path(args.phot_cat) if args.phot_cat else Path(config.STEP11_PHOTCAT)
    outdir = Path(args.outdir) if args.outdir else st11

    r_band = Path(args.r_band) if args.r_band else Path(config.FILTER_R)
    i_band = Path(args.i_band) if args.i_band else Path(config.FILTER_I)
    z_band = Path(args.z_band) if args.z_band else Path(config.FILTER_Z)

    if not infile.exists():
        raise FileNotFoundError(f"Missing Step11c input FITS: {infile}")
    if not phot_cat.exists():
        raise FileNotFoundError(f"Missing Step11 photometry catalog: {phot_cat}")

    phot_tab = read_phot_table(phot_cat)
    phot_lookup = build_phot_lookup(phot_tab, args.id_col, args.r_col, args.i_col, args.z_col)

    bandpasses = {
        "r": load_two_col_ascii(r_band),
        "i": load_two_col_ascii(i_band),
        "z": load_two_col_ascii(z_band),
    }

    print("INFILE   =", infile)
    print("PHOT_CAT =", phot_cat)
    print("OUTDIR   =", outdir)
    print("R_BAND   =", r_band)
    print("I_BAND   =", i_band)
    print("Z_BAND   =", z_band)
    print("MODE     =", args.mode)
    print("BANDPASS =", args.bandpass_mode)

    allowed_slits = None
    if args.slit_list_file:
        import pandas as pd

        df = pd.read_csv(args.slit_list_file)
        if "slit_id" not in df.columns:
            raise KeyError("Expected 'slit_id' column in slit-list-file")

        df["slit_id"] = df["slit_id"].astype(str).str.strip().str.upper()
        if "accepted" in df.columns:
            df = df[df["accepted"] == True]
        if "rank_score" in df.columns:
            df = df.sort_values(by="rank_score", ascending=False)
        if args.top_n is not None:
            df = df.head(args.top_n)

        allowed_slits = set(df["slit_id"].tolist())
        print(f"[INFO] Using {len(allowed_slits)} slits from {args.slit_list_file}")
        print("[INFO] Slits:", sorted(allowed_slits))

    process_file(
        infile=infile,
        phot_lookup=phot_lookup,
        bandpasses=bandpasses,
        outdir=outdir,
        flux_col=args.flux_col,
        var_col=args.var_col,
        wave_col=args.wave_col,
        mode=args.mode,
        response_positive_frac_min=args.response_positive_frac_min,
        cond_max=args.cond_max,
        band_rms_rel_max=args.band_rms_rel_max,
        response_median_min=args.response_median_min,
        response_median_max=args.response_median_max,
        combine_method=args.combine_method,
        allowed_slits=allowed_slits,
        bandpass_mode=args.bandpass_mode,
    )

    stem = infile.stem
    print("[OK] Step11d finished")
    if args.mode in ("master", "both"):
        print("[OK] Wrote", outdir / f"{stem}_refined_master.fits")
        print("[OK] Wrote", outdir / f"{stem}_step11d_master_response.fits")
    if args.mode in ("perstar", "both"):
        print("[OK] Wrote", outdir / f"{stem}_refined_perstar_{args.bandpass_mode}.fits")
    print("[OK] Wrote", outdir / f"{stem}_step11d_summary.csv")
    print("[OK] Wrote", outdir / f"{stem}_step11d_metadata.json")
    print("[OK] Wrote", outdir / f"{stem}_step11d_debug.csv")
    
#    print("lambda_eff r_short =", effective_lambda_nm(wrs, trs))
#    print("lambda_eff z_short =", effective_lambda_nm(wzs, tzs))


if __name__ == "__main__":
    main()