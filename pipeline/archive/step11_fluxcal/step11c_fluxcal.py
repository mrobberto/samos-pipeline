#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step11 — photometric (r/i/z) flux calibration / zero-point anchoring.

FINAL CLEAN VERSION
-------------------
This is a plumbing-only port of the original working script:
    Step11.0_fluxcal_photanchor_v2_use_tellcorr.py

Science behavior intentionally preserved:
- reads extracted 1D MEF
- prefers telluric-corrected flux when present
- anchors spectra to photometric r/i/z measurements
- writes FLUX_FLAM (and VAR_FLAM2 when variance exists)
- writes summary CSV and QA plot

Updated plumbing:
- current path discovery under config tree
- current default filenames/output locations
- defaults aimed at the restored OH-refined + telluric-corrected flow
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from astropy.constants import c
import config


# -----------------------------
# Defaults (nm)
# -----------------------------
DEFAULT_WINDOWS_NM = {
    "r": (560.0, 700.0),
    "i": (700.0, 820.0),
    "z": (820.0, 950.0),
}

LAMBDA_EFF_NM = {"r": 620.0, "i": 750.0, "z": 870.0}
VEGA_TO_AB_DEFAULT = {"r": 0.16, "i": 0.37, "z": 0.54}


def _hdr_set_float(header, key, val, comment="", blank=-999.0):
    """Set a FITS header float value, avoiding NaN/Inf."""
    try:
        v = float(val)
    except Exception:
        v = blank
    if not np.isfinite(v):
        v = blank
    header[key] = (v, comment)


def abmag_to_fnu_jy(m_ab: float) -> float:
    return 3631.0 * 10.0 ** (-0.4 * m_ab)


def fnu_jy_to_flambda_cgs(fnu_jy: float, lam_nm: np.ndarray) -> np.ndarray:
    lam = (lam_nm * u.nm).to(u.AA)
    fnu = (fnu_jy * u.Jy).to(u.erg / u.s / u.cm**2 / u.Hz)
    flam = (fnu * c / lam**2).to(u.erg / u.s / u.cm**2 / u.AA)
    return flam.value


def load_filter_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.genfromtxt(path, comments="#", delimiter=None)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Filter file {path} must have at least 2 columns (wavelength, throughput).")
    lam = arr[:, 0].astype(float)
    thr = arr[:, 1].astype(float)
    lam_nm = lam / 10.0 if np.nanmedian(lam) > 2000 else lam
    thr = np.clip(thr, 0.0, None)
    return lam_nm, thr


def synth_counts_in_band(lam_nm: np.ndarray, counts: np.ndarray, bandpass: tuple[np.ndarray, np.ndarray] | None,
                         window_nm: tuple[float, float]) -> float:
    good = np.isfinite(lam_nm) & np.isfinite(counts)
    lam_nm = lam_nm[good]
    counts = counts[good]
    if lam_nm.size < 10:
        return np.nan

    if bandpass is None:
        lo, hi = window_nm
        m = (lam_nm >= lo) & (lam_nm <= hi)
        if m.sum() < 10:
            return np.nan
        return float(np.nanmedian(counts[m]))

    lam_f, thr = bandpass
    thr_i = np.interp(lam_nm, lam_f, thr, left=0.0, right=0.0)
    m = thr_i > 0
    if m.sum() < 10:
        return np.nan
    num = np.trapz(counts[m] * thr_i[m], lam_nm[m])
    den = np.trapz(thr_i[m], lam_nm[m])
    if den <= 0:
        return np.nan
    return float(num / den)


def synth_abmag_from_flux(lam_nm: np.ndarray, flam_cgs: np.ndarray,
                          bandpass: tuple[np.ndarray, np.ndarray] | None,
                          window_nm: tuple[float, float],
                          lam_eff_nm: float) -> float:
    good = np.isfinite(lam_nm) & np.isfinite(flam_cgs)
    lam_nm = lam_nm[good]
    flam = flam_cgs[good]
    if lam_nm.size < 10:
        return np.nan

    if bandpass is None:
        lo, hi = window_nm
        m = (lam_nm >= lo) & (lam_nm <= hi)
        if m.sum() < 10:
            return np.nan
        flam_band = float(np.nanmedian(flam[m]))
    else:
        lam_f, thr = bandpass
        thr_i = np.interp(lam_nm, lam_f, thr, left=0.0, right=0.0)
        m = thr_i > 0
        if m.sum() < 10:
            return np.nan
        num = np.trapz(flam[m] * thr_i[m], lam_nm[m])
        den = np.trapz(thr_i[m], lam_nm[m])
        if den <= 0:
            return np.nan
        flam_band = float(num / den)

    lam_eff_AA = (lam_eff_nm * u.nm).to(u.AA)
    flam_q = (flam_band * u.erg / u.s / u.cm**2 / u.AA)
    fnu = (flam_q * lam_eff_AA**2 / c).to(u.erg / u.s / u.cm**2 / u.Hz)
    fnu_jy = fnu.to(u.Jy).value
    if fnu_jy <= 0 or not np.isfinite(fnu_jy):
        return np.nan
    return float(-2.5 * np.log10(fnu_jy / 3631.0))


def parse_keyvals(s: str) -> dict[str, float]:
    out = {}
    for part in s.split(","):
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out


def parse_windows(s: str) -> dict[str, tuple[float, float]]:
    out = {}
    for part in s.split(","):
        k, rng = part.split("=")
        lo, hi = rng.split("-")
        out[k.strip()] = (float(lo), float(hi))
    return out


def find_filter_files(filters_dir: Path) -> dict[str, Path]:
    cand = {}
    for b in ["r", "i", "z"]:
        for name in [f"{b}.dat", f"sloan_{b}.dat", f"{b}.txt", f"sloan_{b}.txt"]:
            p = filters_dir / name
            if p.exists():
                cand[b] = p
                break
    return cand


def find_latest(path: Path, patterns):
    hits = []
    for pat in patterns:
        hits.extend(path.glob(pat))
    if not hits:
        return None
    hits = sorted(set(hits), key=lambda p: p.stat().st_mtime)
    return hits[-1]


def parse_args():
    p = argparse.ArgumentParser(description="SAMOS Step11 photometric flux calibration (r/i/z).")
    p.add_argument("extract_fits", type=Path, nargs="?", help="Input extracted 1D MEF")
    p.add_argument("phot_csv", type=Path, nargs="?", help="CSV with slit and r/i/z mags")
    p.add_argument("--out-fits", type=Path, default=None, help="Output MEF")
    p.add_argument("--out-summary", type=Path, default=None, help="Output summary CSV")
    p.add_argument("--qa-plot", type=Path, default=None, help="Output QA plot")
    p.add_argument("--mag-system", choices=["ab", "vega", "auto"], default="auto")
    p.add_argument("--vega-offsets", type=str, default=None,
                   help="Override Vega->AB offsets as 'r=0.16,i=0.37,z=0.54'")
    p.add_argument("--mode", choices=["gray", "tilt"], default="gray")
    p.add_argument("--filters-dir", type=Path, default=None,
                   help="Optional directory containing filter curves")
    p.add_argument("--windows-nm", type=str, default=None,
                   help="Override rectangular windows as 'r=560-700,i=700-820,z=820-950'")
    p.add_argument("--max-band-dispersion", type=float, default=0.20)
    p.add_argument("--min-overlap-points", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()

    st10 = Path(config.ST10_TELLURIC)
    st11 = Path(config.ST11_FLUXCAL)
    st11.mkdir(parents=True, exist_ok=True)

    # Default input: restored flow product
    extract_fits = args.extract_fits
    if extract_fits is None:
        preferred = [
            st10 / "extract1d_optimal_ridge_all_wav_OHref_tellcorr.fits",
            st10 / "extract1d_optimal_ridge_all_wav_tellcorr_OHref_tellcorr.fits",
        ]
        extract_fits = None
        for p in preferred:
            if p.exists():
                extract_fits = p
                break
    if extract_fits is None or not Path(extract_fits).exists():
        raise FileNotFoundError("No suitable Step10 telluric-corrected FITS found")
    
    phot_csv = args.phot_csv
    if phot_csv is None:
        phot_csv = st11 / "slit_trace_radec_skymapper_all.csv"
    if not Path(phot_csv).exists():
        raise FileNotFoundError(phot_csv)
    
    out_fits = args.out_fits or (st11 / "Extract1D_fluxcal.fits")
    out_summary = args.out_summary or (st11 / "Step11_fluxcal_summary.csv")
    qa_plot = args.qa_plot or (st11 / "Step11_fluxcal_QA.png")

    mag_system = args.mag_system
    if mag_system == "auto":
        mag_system = "ab"

    vega_offsets = dict(VEGA_TO_AB_DEFAULT)
    if args.vega_offsets:
        vega_offsets.update(parse_keyvals(args.vega_offsets))

    windows = dict(DEFAULT_WINDOWS_NM)
    if args.windows_nm:
        windows.update(parse_windows(args.windows_nm))

    phot = pd.read_csv(phot_csv)
    cols = {c.lower(): c for c in phot.columns}
    if "slit" not in cols:
        raise SystemExit("phot_csv must contain a 'slit' column")
    slit_col = cols["slit"]

    def col(name):
        return cols.get(name.lower())

    rcol, icol, zcol = col("r_mag"), col("i_mag"), col("z_mag")
    rerr, ierr, zerr = col("r_err"), col("i_err"), col("z_err")
    if rcol is None and icol is None and zcol is None:
        raise SystemExit("phot_csv must contain at least one of r_mag/i_mag/z_mag columns")

    bandpass = {b: None for b in ["r", "i", "z"]}
    if args.filters_dir:
        fls = find_filter_files(args.filters_dir)
        if fls:
            for b, path in fls.items():
                bandpass[b] = load_filter_curve(path)
        else:
            print("[WARN] --filters-dir provided but no filter files found; using rectangular windows.")

    print("EXTRACT_FITS =", extract_fits)
    print("PHOT_CSV     =", phot_csv)
    print("OUT_FITS     =", out_fits)
    print("OUT_SUMMARY  =", out_summary)
    print("QA_PLOT      =", qa_plot)

    with fits.open(extract_fits) as hdul:
        out_hdus = [fits.PrimaryHDU(header=hdul[0].header)]
        out_hdus[0].header["PIPESTEP"] = "STEP11"
        out_hdus[0].header["STAGE"] = "11"
        out_hdus[0].header["SRCFILE"] = str(Path(extract_fits).name)

        summary_rows = []

        for hdu in hdul[1:]:
            slit = (hdu.name or "").strip()
            if not slit.startswith("SLIT"):
                continue

            data = hdu.data
            if data is None:
                continue

            names = list(data.columns.names)

            lam_col = None
            for cand in ["LAMBDA_NM", "LAMBDA", "WAVE_NM", "WAVELENGTH_NM", "WAVELENGTH"]:
                if cand in names:
                    lam_col = cand
                    break
            if lam_col is None:
                low = {n.lower(): n for n in names}
                for cand in ["lambda_nm", "lambda", "wave_nm", "wavelength_nm", "wavelength"]:
                    if cand in low:
                        lam_col = low[cand]
                        break
            if lam_col is None:
                raise RuntimeError(f"{slit}: could not find wavelength column in {names}")

            flux_col = None
            for cand in ["FLUX_TELLCOR_O2", "FLUX_TELLCOR", "FLUX_TELLCORR_O2", "FLUX_TELLCORR",
                         "FLUX_O2CORR", "FLUX_ADU_S", "FLUX", "COUNTS", "COUNTS_S", "ADU_S"]:
                if cand in names:
                    flux_col = cand
                    break
            if flux_col is None:
                low = {n.lower(): n for n in names}
                for cand in ["flux_tellcor_o2", "flux_tellcor", "flux_tellcorr_o2", "flux_tellcorr",
                             "flux_o2corr", "flux_adu_s", "flux", "counts", "counts_s", "adu_s"]:
                    if cand in low:
                        flux_col = low[cand]
                        break
            if flux_col is None:
                raise RuntimeError(f"{slit}: could not find flux column in {names}")

            var_col = None
            for cand in ["VAR_TELLCOR_O2", "VAR_ADU_S2", "VAR", "VARIANCE", "SIGMA2"]:
                if cand in names:
                    var_col = cand
                    break
            if var_col is None:
                low = {n.lower(): n for n in names}
                for cand in ["var_tellcor_o2", "var_adu_s2", "var", "variance", "sigma2"]:
                    if cand in low:
                        var_col = low[cand]
                        break

            lam_nm = np.asarray(data[lam_col], float)
            counts = np.asarray(data[flux_col], float)
            var = np.asarray(data[var_col], float) if var_col else None

            row = phot.loc[phot[slit_col] == slit]
            if len(row) != 1:
                m_r = m_i = m_z = np.nan
                e_r = e_i = e_z = np.nan
            else:
                row = row.iloc[0]

                def _to_float(x):
                    try:
                        v = pd.to_numeric(x, errors="coerce")
                        return float(v) if np.isfinite(v) else np.nan
                    except Exception:
                        return np.nan

                m_r = _to_float(row[rcol]) if rcol else np.nan
                m_i = _to_float(row[icol]) if icol else np.nan
                m_z = _to_float(row[zcol]) if zcol else np.nan
                e_r = _to_float(row[rerr]) if rerr else np.nan
                e_i = _to_float(row[ierr]) if ierr else np.nan
                e_z = _to_float(row[zerr]) if zerr else np.nan

            if mag_system == "vega":
                if np.isfinite(m_r): m_r += vega_offsets["r"]
                if np.isfinite(m_i): m_i += vega_offsets["i"]
                if np.isfinite(m_z): m_z += vega_offsets["z"]

            S_b = {}
            C_b = {}
            F_b = {}

            for b, m in [("r", m_r), ("i", m_i), ("z", m_z)]:
                if not np.isfinite(m):
                    S_b[b] = np.nan
                    C_b[b] = np.nan
                    F_b[b] = np.nan
                    continue

                Cband = synth_counts_in_band(lam_nm, counts, bandpass[b], windows[b])
                if not np.isfinite(Cband):
                    S_b[b] = np.nan
                    C_b[b] = np.nan
                    F_b[b] = np.nan
                    continue

                fnu_jy = abmag_to_fnu_jy(m)
                flam = fnu_jy_to_flambda_cgs(fnu_jy, np.array([LAMBDA_EFF_NM[b]]))[0]
                S_b[b] = flam / Cband
                C_b[b] = Cband
                F_b[b] = flam

            Sb_vals = np.array([S_b["r"], S_b["i"], S_b["z"]], float)
            good_bands = np.isfinite(Sb_vals)
            nband = int(good_bands.sum())

            cal_mode = "NONE"
            S = np.nan
            alpha = np.nan
            flag = "NOMAG" if nband == 0 else "OK"

            if nband == 0:
                flam_cal = np.full_like(counts, np.nan, dtype=float)
                var_cal = np.full_like(counts, np.nan, dtype=float) if var is not None else None

            elif args.mode == "gray" or nband < 2:
                cal_mode = "GRAY"
                S = float(np.nanmedian(Sb_vals[good_bands]))
                disp = float(np.nanstd(Sb_vals[good_bands]) / np.nanmedian(Sb_vals[good_bands])) if nband >= 2 else 0.0
                if disp > args.max_band_dispersion:
                    flag = f"DISP>{args.max_band_dispersion:.2f}"
                flam_cal = S * counts
                var_cal = (S**2) * var if var is not None else None

            else:
                cal_mode = "TILT"
                lam0 = 750.0
                xs = []
                ys = []
                ws = []
                for b, m, e in [("r", m_r, e_r), ("i", m_i, e_i), ("z", m_z, e_z)]:
                    if not np.isfinite(S_b[b]):
                        continue
                    xs.append(np.log(LAMBDA_EFF_NM[b] / lam0))
                    ys.append(np.log(S_b[b]))
                    if np.isfinite(e) and e > 0:
                        ws.append(1.0 / (0.4 * np.log(10.0) * e)**2)
                    else:
                        ws.append(1.0)
                xs = np.array(xs)
                ys = np.array(ys)
                W = np.diag(ws)
                A = np.vstack([np.ones_like(xs), xs]).T
                beta = np.linalg.inv(A.T @ W @ A) @ (A.T @ W @ ys)
                logS, alpha = beta[0], beta[1]
                S = float(np.exp(logS))
                alpha = float(alpha)
                flam_cal = counts * S * (lam_nm / lam0) ** alpha
                var_cal = var * (S**2) * (lam_nm / lam0) ** (2 * alpha) if var is not None else None

            dm = {}
            for b, mcat in [("r", m_r), ("i", m_i), ("z", m_z)]:
                if not np.isfinite(mcat) or (cal_mode == "NONE"):
                    dm[b] = np.nan
                    continue
                msyn = synth_abmag_from_flux(lam_nm, flam_cal, bandpass[b], windows[b], LAMBDA_EFF_NM[b])
                dm[b] = float(msyn - mcat) if np.isfinite(msyn) else np.nan

            cols_out = []
            for name in names:
                cols_out.append(fits.Column(name=name, array=data[name], format=hdu.columns[name].format, unit=hdu.columns[name].unit))
            cols_out.append(fits.Column(name="FLUX_FLAM", array=flam_cal.astype(np.float32), format="E", unit="erg/s/cm^2/Angstrom"))
            if var_cal is not None:
                cols_out.append(fits.Column(name="VAR_FLAM2", array=var_cal.astype(np.float32), format="E", unit="(erg/s/cm^2/Angstrom)^2"))

            hdu_out = fits.BinTableHDU.from_columns(cols_out, name=slit)
            hdu_out.header["FLUXCAL"] = (cal_mode, "Step11 photometric flux calibration mode")
            hdu_out.header["FLUXIN"] = (str(flux_col)[:68], "Input flux column used")
            if np.isfinite(S):
                hdu_out.header["SCALE"] = (S, "Multiplicative scale (counts -> f_lambda)")
            if np.isfinite(alpha):
                hdu_out.header["ALPHA"] = (alpha, "Tilt exponent around 750 nm (if TILT)")
            hdu_out.header["NBAND"] = (nband, "Number of bands used (r,i,z)")
            _hdr_set_float(hdu_out.header, "DMR", dm.get("r", np.nan), "m_syn - m_cat (r)")
            _hdr_set_float(hdu_out.header, "DMI", dm.get("i", np.nan), "m_syn - m_cat (i)")
            _hdr_set_float(hdu_out.header, "DMZ", dm.get("z", np.nan), "m_syn - m_cat (z)")
            hdu_out.header["QCFLAG"] = (flag, "Calibration QC flag")

            out_hdus.append(hdu_out)

            summary_rows.append({
                "slit": slit,
                "cal_mode": cal_mode,
                "qcflag": flag,
                "nband": nband,
                "S": S,
                "alpha": alpha,
                "S_r": S_b["r"], "S_i": S_b["i"], "S_z": S_b["z"],
                "C_r": C_b["r"], "C_i": C_b["i"], "C_z": C_b["z"],
                "F_r": F_b["r"], "F_i": F_b["i"], "F_z": F_b["z"],
                "dm_r": dm["r"], "dm_i": dm["i"], "dm_z": dm["z"],
                "m_r_ab": m_r, "m_i_ab": m_i, "m_z_ab": m_z,
                "mag_system_used": mag_system,
                "flux_input_col": flux_col,
            })

        fits.HDUList(out_hdus).writeto(out_fits, overwrite=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_summary, index=False)

    try:
        ok = summary["cal_mode"].isin(["GRAY", "TILT"])
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.hist(summary.loc[ok, "dm_r"].dropna(), bins=30)
        ax1.set_title("Δm_r (syn - cat)")
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.hist(summary.loc[ok, "dm_i"].dropna(), bins=30)
        ax2.set_title("Δm_i (syn - cat)")
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.hist(summary.loc[ok, "dm_z"].dropna(), bins=30)
        ax3.set_title("Δm_z (syn - cat)")
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.hist(np.log10(summary.loc[ok, "S"].dropna()), bins=30)
        ax4.set_title("log10(S)  [counts → fλ]")
        fig.tight_layout()
        fig.savefig(qa_plot, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] QA plot failed: {e}")

    print("[OK] Wrote:", out_fits)
    print("[OK] Wrote:", out_summary)
    print("[OK] Wrote:", qa_plot)


if __name__ == "__main__":
    main()
