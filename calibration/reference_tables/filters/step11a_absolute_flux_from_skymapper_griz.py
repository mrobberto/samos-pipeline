#!/usr/bin/env python3
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
