from pathlib import Path
import textwrap

# Create SkyMapper bandpass files in nm from SVO ASCII tables (originally in Angstrom)
bands_angstrom = {
    "g": "4000 0.000 4050 0.003 4100 0.006 4150 0.078 4200 0.301 4250 0.506 4300 0.612 4350 0.676 4400 0.716 4450 0.753 4500 0.784 4550 0.813 4600 0.839 4650 0.856 4700 0.876 4750 0.896 4800 0.918 4850 0.939 4900 0.953 4950 0.971 5000 0.981 5050 0.990 5100 0.995 5150 1.000 5200 0.997 5250 0.988 5300 0.980 5350 0.960 5400 0.937 5450 0.908 5500 0.874 5550 0.828 5600 0.775 5650 0.719 5700 0.658 5750 0.594 5800 0.524 5850 0.460 5900 0.391 5950 0.329 6000 0.271 6050 0.219 6100 0.173 6150 0.135 6200 0.103 6250 0.077 6300 0.057 6350 0.042 6400 0.031 6450 0.020 6500 0.014 6550 0.012 6600 0.007 6650 0.005 6700 0.000",
    "r": "4800 0.000 4850 0.002 4900 0.008 4950 0.012 5000 0.021 5050 0.034 5100 0.051 5150 0.070 5200 0.097 5250 0.141 5300 0.218 5350 0.354 5400 0.572 5450 0.772 5500 0.951 5550 0.932 5600 0.953 5650 0.962 5700 0.926 5750 0.943 5800 0.947 5850 0.937 5900 0.946 5950 0.947 6000 0.927 6050 0.919 6100 0.898 6150 0.895 6200 0.926 6250 0.945 6300 0.966 6350 0.974 6400 0.947 6450 0.970 6500 1.000 6550 0.928 6600 0.907 6650 0.986 6700 0.990 6750 0.923 6800 0.858 6850 0.782 6900 0.849 6950 0.631 7000 0.210 7050 0.093 7100 0.071 7150 0.052 7200 0.017 7250 0.006 7300 0.000",
    "i": "6800 0.000 6850 0.001 6900 0.003 6950 0.015 7000 0.087 7050 0.262 7100 0.480 7150 0.633 7200 0.707 7250 0.777 7300 0.851 7350 0.934 7400 0.980 7450 0.993 7500 1.000 7550 0.936 7600 0.709 7650 0.716 7700 0.917 7750 0.959 7800 0.945 7850 0.934 7900 0.927 7950 0.923 8000 0.931 8050 0.929 8100 0.902 8150 0.826 8200 0.784 8250 0.733 8300 0.643 8350 0.565 8400 0.576 8450 0.734 8500 0.554 8550 0.123 8600 0.028 8650 0.009 8700 0.004 8750 0.002 8800 0.000",
    "z": "8100 0.000 8150 0.008 8200 0.019 8250 0.046 8300 0.102 8350 0.203 8400 0.347 8450 0.502 8500 0.653 8550 0.782 8600 0.876 8650 0.935 8700 0.979 8750 0.995 8800 1.000 8850 0.982 8900 0.946 8950 0.845 9000 0.773 9050 0.764 9100 0.719 9150 0.699 9200 0.708 9250 0.661 9300 0.493 9350 0.348 9400 0.361 9450 0.350 9500 0.341 9550 0.350 9600 0.368 9650 0.407 9700 0.432 9750 0.424 9800 0.402 9850 0.378 9900 0.355 9950 0.334 10000 0.312 10050 0.291 10100 0.270 10150 0.250 10200 0.225 10250 0.203 10300 0.181 10350 0.159 10400 0.134 10450 0.112 10500 0.091 10600 0.048 10700 0.000",
}

for band, seq in bands_angstrom.items():
    vals = list(map(float, seq.split()))
    wl_a = vals[0::2]
    thr = vals[1::2]
    wl_nm = [w / 10.0 for w in wl_a]
    out = Path(f"./skymapper_{band}_nm.txt")
    with out.open("w") as f:
        f.write("# wavelength_nm throughput\n")
        for w, t in zip(wl_nm, thr):
            f.write(f"{w:8.1f} {t:0.6f}\n")

script = r'''#!/usr/bin/env python3
"""
step11a_absolute_flux_from_skymapper_griz.py

Standalone SAMOS step11 script for smooth absolute flux calibration using SkyMapper
AB photometry and extracted 1D spectra.

Compared with the earlier prototype:
- adds optional g band support
- if 4 bands (g,r,i,z) are available, solves for the quadratic response by
  least squares (over-constrained 4x3 system)
- if only 3 bands are available, solves exactly
- writes calibrated FITS products and response summaries

Model
-----
F_cal(lambda) = F_in(lambda) * P(lambda)

with
    P(lambda) = A*x^2 + B*x + C
    x = (lambda_nm - lambda0_nm) / scale_nm

For each band b, the predicted linear broadband quantity is
    Y_b = ∫ F_in(lambda) * P(lambda) * T_b(lambda) * lambda d lambda

For AB photometry, the matched observed quantity is
    Y_b(obs) = c * <f_nu>_b * ∫ T_b(lambda) / lambda d lambda
where
    <f_nu>_b = 3631 Jy * 10^(-0.4 m_b)

Input bandpass files are expected as two-column ASCII:
    wavelength_nm  throughput
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

C_M_PER_S = 299792458.0
JY_TO_W_M2_HZ = 1e-26


@dataclass
class PerStarFit:
    slit_id: str
    coeff_a: float
    coeff_b: float
    coeff_c: float
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


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapz(y, x))


def abmag_to_fnu_si(mag_ab: float) -> float:
    return 3631.0 * 10.0 ** (-0.4 * mag_ab) * JY_TO_W_M2_HZ


def robust_median_and_mad(arr: np.ndarray) -> Tuple[float, float]:
    good = np.isfinite(arr)
    if not np.any(good):
        return np.nan, np.nan
    med = float(np.nanmedian(arr[good]))
    mad = float(np.nanmedian(np.abs(arr[good] - med)))
    return med, mad


def _to_float_or_nan(v) -> float:
    try:
        return float(v)
    except Exception:
        return np.nan


def _is_finite_mag(v: float) -> bool:
    return np.isfinite(v) and (-50.0 < v < 99.0)


def load_two_col_ascii(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, comments="#", usecols=(0, 1))
    wave = np.asarray(arr[:, 0], dtype=float)
    tran = np.asarray(arr[:, 1], dtype=float)
    order = np.argsort(wave)
    return wave[order], tran[order]


def interp_bandpass(wave_spec_nm: np.ndarray,
                    wave_band_nm: np.ndarray,
                    tran_band: np.ndarray) -> np.ndarray:
    return np.interp(wave_spec_nm, wave_band_nm, tran_band, left=0.0, right=0.0)


def observed_band_target_ab(mag_ab: float,
                            wave_nm: np.ndarray,
                            tran: np.ndarray) -> float:
    fnu = abmag_to_fnu_si(mag_ab)
    c_nm_s = C_M_PER_S * 1e9
    norm = trapz(tran / wave_nm, wave_nm)
    return c_nm_s * fnu * norm


def choose_scaled_coordinate(wave_nm: np.ndarray,
                             lambda0_nm: Optional[float] = None,
                             scale_nm: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
    if lambda0_nm is None:
        lambda0_nm = float(0.5 * (np.nanmin(wave_nm) + np.nanmax(wave_nm)))
    if scale_nm is None:
        scale_nm = float(0.5 * (np.nanmax(wave_nm) - np.nanmin(wave_nm)))
    x = (wave_nm - lambda0_nm) / scale_nm
    return x, float(lambda0_nm), float(scale_nm)


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


def solve_one_star(wave_nm: np.ndarray,
                   flux: np.ndarray,
                   mags_ab: Dict[str, float],
                   bandpasses_interp: Dict[str, np.ndarray],
                   bands_priority: Sequence[str] = ("g", "r", "i", "z"),
                   lambda0_nm: Optional[float] = None,
                   scale_nm: Optional[float] = None
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, Dict[str, float], int]:
    x, lambda0_nm, scale_nm = choose_scaled_coordinate(wave_nm, lambda0_nm, scale_nm)

    rows = []
    rhs = []
    used_bands = []
    for band in bands_priority:
        mag = mags_ab.get(band, np.nan)
        if not _is_finite_mag(mag):
            continue
        tran = bandpasses_interp[band]
        if np.nanmax(tran) <= 0:
            continue
        rows.append(build_design_row(flux, wave_nm, tran, x))
        rhs.append(observed_band_target_ab(mag, wave_nm, tran))
        used_bands.append(band)

    if len(rows) < 3:
        raise RuntimeError("Need at least 3 valid bands to solve for a quadratic response.")

    M = np.vstack(rows)
    y = np.asarray(rhs, dtype=float)
    cond = float(np.linalg.cond(M))

    if len(rows) == 3:
        coeff = np.linalg.solve(M, y)
    else:
        coeff, _, _, _ = np.linalg.lstsq(M, y, rcond=None)

    response = coeff[0] * x**2 + coeff[1] * x + coeff[2]
    pred = M @ coeff

    residuals = {}
    for band, yp, yo in zip(used_bands, pred, y):
        residuals[band] = float((yp - yo) / yo) if yo != 0 else np.nan

    return coeff, response, y, lambda0_nm, scale_nm, cond, residuals, len(used_bands)


def evaluate_quality(response: np.ndarray,
                     cond: float,
                     residuals: Dict[str, float],
                     response_positive_frac_min: float,
                     cond_max: float,
                     band_rms_rel_max: float) -> Tuple[bool, str, float]:
    positive_frac = float(np.mean(np.isfinite(response) & (response > 0)))
    rels = np.array([residuals.get("g", np.nan),
                     residuals.get("r", np.nan),
                     residuals.get("i", np.nan),
                     residuals.get("z", np.nan)], dtype=float)
    rms_rel = float(np.sqrt(np.nanmean(rels**2)))

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

    return ok, "accepted" if ok else "; ".join(reasons), rms_rel


def find_table_hdus(hdul: fits.HDUList) -> List[int]:
    out = []
    for idx, hdu in enumerate(hdul):
        if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
            out.append(idx)
    return out


def get_column_name(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    cset = {c.upper(): c for c in columns}
    for cand in candidates:
        if cand.upper() in cset:
            return cset[cand.upper()]
    return None


def infer_slit_id_from_hdu(hdu, idx: int) -> str:
    hdr = hdu.header
    for key in ("SLITID", "SLIT", "EXTNAME", "NAME", "OBJECT"):
        if key in hdr:
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


def build_phot_lookup(tab: Table,
                      id_col: str,
                      g_col: Optional[str],
                      r_col: str,
                      i_col: str,
                      z_col: str) -> Dict[str, Dict[str, float]]:
    out = {}
    for row in tab:
        sid = str(row[id_col]).strip()
        out[sid] = {
            "g": _to_float_or_nan(row[g_col]) if g_col and g_col in tab.colnames else np.nan,
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


def choose_flux_var_columns(tab: Table,
                            flux_col: Optional[str],
                            var_col: Optional[str]) -> Tuple[str, Optional[str]]:
    cols = list(tab.colnames)
    if flux_col is None:
        flux_col = get_column_name(cols, ["FLUX_APCORR", "FLUX", "OBJ_PRESKY", "OBJ"])
    if flux_col is None:
        raise RuntimeError(f"Could not find flux column among {cols}")
    if var_col is None:
        var_col = get_column_name(cols, ["VAR_APCORR", "VAR"])
    return flux_col, var_col


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
                 combine_method: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    with fits.open(infile) as hdul:
        table_hdus = find_table_hdus(hdul)
        hdul_out_master = fits.HDUList([fits.PrimaryHDU(header=hdul[0].header.copy())])
        hdul_out_perstar = fits.HDUList([fits.PrimaryHDU(header=hdul[0].header.copy())])

        per_star_rows: List[PerStarFit] = []
        accepted_responses = []
        accepted_slits = []
        common_wave = None
        common_lambda0 = None
        common_scale = None

        for idx in table_hdus:
            hdu = hdul[idx]
            tab = Table(hdu.data)
            slit_id = infer_slit_id_from_hdu(hdu, idx)

            if slit_id not in phot_lookup:
                continue

            local_wave_col = choose_wave_column(tab, wave_col)
            local_flux_col, local_var_col = choose_flux_var_columns(tab, flux_col, var_col)

            wave_full = np.asarray(tab[local_wave_col], dtype=float)
            flux_full = np.asarray(tab[local_flux_col], dtype=float)

            good = np.isfinite(wave_full) & np.isfinite(flux_full)
            if np.count_nonzero(good) < 20:
                continue

            wave_nm = wave_full[good]
            flux = flux_full[good]

            mags = phot_lookup[slit_id]
            band_interp = {
                band: interp_bandpass(wave_nm, *bandpasses[band]) for band in ("g", "r", "i", "z")
            }

            try:
                coeff, response, _, lambda0_nm, scale_nm, cond, residuals, n_good_bands = solve_one_star(
                    wave_nm=wave_nm,
                    flux=flux,
                    mags_ab=mags,
                    bandpasses_interp=band_interp,
                )
            except Exception as exc:
                per_star_rows.append(PerStarFit(
                    slit_id=slit_id,
                    coeff_a=np.nan, coeff_b=np.nan, coeff_c=np.nan,
                    lambda0_nm=np.nan, scale_nm=np.nan, cond=np.nan,
                    rms_rel_band=np.nan, n_good_bands=0, accepted=False,
                    reason=f"solve_failed: {exc}",
                    band_residual_g=np.nan, band_residual_r=np.nan,
                    band_residual_i=np.nan, band_residual_z=np.nan,
                ))
                continue

            accepted, reason, rms_rel = evaluate_quality(
                response=response,
                cond=cond,
                residuals=residuals,
                response_positive_frac_min=response_positive_frac_min,
                cond_max=cond_max,
                band_rms_rel_max=band_rms_rel_max,
            )

            per_star_rows.append(PerStarFit(
                slit_id=slit_id,
                coeff_a=float(coeff[0]), coeff_b=float(coeff[1]), coeff_c=float(coeff[2]),
                lambda0_nm=float(lambda0_nm), scale_nm=float(scale_nm),
                cond=float(cond), rms_rel_band=float(rms_rel),
                n_good_bands=int(n_good_bands),
                accepted=bool(accepted), reason=reason,
                band_residual_g=float(residuals.get("g", np.nan)),
                band_residual_r=float(residuals.get("r", np.nan)),
                band_residual_i=float(residuals.get("i", np.nan)),
                band_residual_z=float(residuals.get("z", np.nan)),
            ))

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
                full_response = np.full(len(tab), np.nan, dtype=float)
                full_response[good] = response
                new_tab["RESP_STEP11"] = full_response

                full_flux = np.asarray(tab[local_flux_col], dtype=float).copy()
                full_flux[good] *= response
                new_tab[f"{local_flux_col}_ABSCAL"] = full_flux

                if local_var_col is not None:
                    full_var = np.asarray(tab[local_var_col], dtype=float).copy()
                    full_var[good] *= response**2
                    new_tab[f"{local_var_col}_ABSCAL"] = full_var

                hdr = hdu.header.copy()
                hdr["STEP11"] = ("ABSFLUX", "Absolute flux calibration applied")
                hdr["ABMODE"] = ("PERSTAR", "Absolute flux calibration mode")
                hdr["AB_A"] = (float(coeff[0]), "Quadratic coeff A in scaled basis")
                hdr["AB_B"] = (float(coeff[1]), "Quadratic coeff B in scaled basis")
                hdr["AB_C"] = (float(coeff[2]), "Quadratic coeff C in scaled basis")
                hdr["AB_L0"] = (float(lambda0_nm), "Scaled basis lambda0 [nm]")
                hdr["AB_SCL"] = (float(scale_nm), "Scaled basis scale [nm]")
                hdr["ABCOND"] = (float(cond), "Condition number of solve")
                hdr["NABAND"] = (int(n_good_bands), "Number of photometric bands used")
                hdul_out_perstar.append(fits.BinTableHDU(new_tab, header=hdr, name=hdu.name))

        master_response = None
        if accepted_responses:
            stack = np.vstack(accepted_responses)
            if combine_method == "median":
                master_response = np.nanmedian(stack, axis=0)
            elif combine_method == "mean":
                master_response = np.nanmean(stack, axis=0)
            else:
                raise ValueError(f"Unsupported combine method: {combine_method}")

        if mode in ("master", "both"):
            if master_response is None or common_wave is None:
                raise RuntimeError("No accepted calibration stars available for master response.")

            for idx in table_hdus:
                hdu = hdul[idx]
                tab = Table(hdu.data)
                local_wave_col = choose_wave_column(tab, wave_col)
                local_flux_col, local_var_col = choose_flux_var_columns(tab, flux_col, var_col)

                wave_full = np.asarray(tab[local_wave_col], dtype=float)
                response_full = np.full(len(tab), np.nan, dtype=float)
                good = np.isfinite(wave_full)

                if wave_full.shape == common_wave.shape and np.allclose(wave_full, common_wave, rtol=0, atol=1e-6):
                    response_full[good] = master_response
                else:
                    response_full[good] = np.interp(wave_full[good], common_wave, master_response, left=np.nan, right=np.nan)

                new_tab = tab.copy()
                new_tab["RESP_STEP11"] = response_full

                full_flux = np.asarray(tab[local_flux_col], dtype=float).copy()
                gmask = np.isfinite(full_flux) & np.isfinite(response_full)
                full_flux[gmask] *= response_full[gmask]
                new_tab[f"{local_flux_col}_ABSCAL"] = full_flux

                if local_var_col is not None:
                    full_var = np.asarray(tab[local_var_col], dtype=float).copy()
                    gmask = np.isfinite(full_var) & np.isfinite(response_full)
                    full_var[gmask] *= response_full[gmask]**2
                    new_tab[f"{local_var_col}_ABSCAL"] = full_var

                hdr = hdu.header.copy()
                hdr["STEP11"] = ("ABSFLUX", "Absolute flux calibration applied")
                hdr["ABMODE"] = ("MASTER", "Absolute flux calibration mode")
                hdr["AB_L0"] = (float(common_lambda0), "Scaled basis lambda0 [nm]")
                hdr["AB_SCL"] = (float(common_scale), "Scaled basis scale [nm]")
                hdr["NABSTAR"] = (int(len(accepted_slits)), "Accepted calibration stars")
                hdul_out_master.append(fits.BinTableHDU(new_tab, header=hdr, name=hdu.name))

        for idx, hdu in enumerate(hdul[1:], start=1):
            if idx not in table_hdus:
                if mode in ("master", "both"):
                    hdul_out_master.append(hdu.copy())
                if mode in ("perstar", "both"):
                    hdul_out_perstar.append(hdu.copy())

        stem = infile.stem
        if mode in ("master", "both"):
            hdul_out_master.writeto(outdir / f"{stem}_abscal_master.fits", overwrite=True)
        if mode in ("perstar", "both"):
            hdul_out_perstar.writeto(outdir / f"{stem}_abscal_perstar.fits", overwrite=True)

        if master_response is not None and common_wave is not None:
            cols = [
                fits.Column(name="LAMBDA_NM", array=np.asarray(common_wave, dtype=np.float64), format="D"),
                fits.Column(name="RESP_MASTER", array=np.asarray(master_response, dtype=np.float64), format="D"),
            ]
            hdu_resp = fits.BinTableHDU.from_columns(cols, name="MASTER_RESPONSE")
            hdr = hdu_resp.header
            hdr["NSTAR"] = int(len(accepted_slits))
            hdr["METHOD"] = combine_method
            hdr["AB_L0"] = float(common_lambda0)
            hdr["AB_SCL"] = float(common_scale)
            fits.HDUList([fits.PrimaryHDU(), hdu_resp]).writeto(
                outdir / f"{stem}_master_response.fits", overwrite=True
            )

        csv_path = outdir / f"{stem}_abscal_summary.csv"
        fieldnames = [
            "slit_id", "coeff_a", "coeff_b", "coeff_c", "lambda0_nm", "scale_nm",
            "cond", "rms_rel_band", "n_good_bands", "accepted", "reason",
            "band_residual_g", "band_residual_r", "band_residual_i", "band_residual_z"
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_star_rows:
                writer.writerow(asdict(row))

        meta = {
            "input_file": str(infile),
            "output_dir": str(outdir),
            "mode": mode,
            "combine_method": combine_method,
            "n_calibration_rows": len(per_star_rows),
            "n_accepted_rows": int(sum(r.accepted for r in per_star_rows)),
            "accepted_slits": accepted_slits,
            "band_set": ["g", "r", "i", "z"],
        }
        with open(outdir / f"{stem}_abscal_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone step11 absolute flux calibration from SkyMapper AB photometry with optional g-band."
    )
    p.add_argument("--infile", required=True)
    p.add_argument("--phot-cat", required=True)
    p.add_argument("--id-col", default="SLITID")
    p.add_argument("--g-col", default=None, help="AB magnitude column for g (optional)")
    p.add_argument("--r-col", required=True)
    p.add_argument("--i-col", required=True)
    p.add_argument("--z-col", required=True)
    p.add_argument("--g-band", default=None, help="ASCII response curve for g (optional)")
    p.add_argument("--r-band", required=True)
    p.add_argument("--i-band", required=True)
    p.add_argument("--z-band", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--mode", choices=["master", "perstar", "both"], default="master")
    p.add_argument("--combine-method", choices=["median", "mean"], default="median")
    p.add_argument("--flux-col", default=None)
    p.add_argument("--var-col", default=None)
    p.add_argument("--wave-col", default=None)
    p.add_argument("--response-positive-frac-min", type=float, default=0.85)
    p.add_argument("--cond-max", type=float, default=1e10)
    p.add_argument("--band-rms-rel-max", type=float, default=0.03,
                   help="Max relative RMS across used bands; for 4-band LS fit this should be small but non-zero.")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    bandpasses = {}
    if args.g_band is not None:
        bandpasses["g"] = load_two_col_ascii(args.g_band)
    else:
        bandpasses["g"] = (np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    bandpasses["r"] = load_two_col_ascii(args.r_band)
    bandpasses["i"] = load_two_col_ascii(args.i_band)
    bandpasses["z"] = load_two_col_ascii(args.z_band)

    phot_tab = read_phot_table(args.phot_cat)
    phot_lookup = build_phot_lookup(
        phot_tab,
        id_col=args.id_col,
        g_col=args.g_col,
        r_col=args.r_col,
        i_col=args.i_col,
        z_col=args.z_col,
    )

    process_file(
        infile=Path(args.infile),
        phot_lookup=phot_lookup,
        bandpasses=bandpasses,
        outdir=Path(args.outdir),
        flux_col=args.flux_col,
        var_col=args.var_col,
        wave_col=args.wave_col,
        mode=args.mode,
        response_positive_frac_min=args.response_positive_frac_min,
        cond_max=args.cond_max,
        band_rms_rel_max=args.band_rms_rel_max,
        combine_method=args.combine_method,
    )


if __name__ == "__main__":
    main()
'''
Path("./step11a_absolute_flux_from_skymapper_griz.py").write_text(script)
print("Created:")
for name in [
    "./skymapper_g_nm.txt",
    "./skymapper_r_nm.txt",
    "./skymapper_i_nm.txt",
    "./skymapper_z_nm.txt",
    "./step11a_absolute_flux_from_skymapper_griz.py",
]:
    print(name)
