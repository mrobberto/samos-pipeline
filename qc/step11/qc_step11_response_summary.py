#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qc_step11_response_summary.py

Comprehensive quality-control (QC) diagnostics for Step11 absolute flux calibration.

----------------------------------------------------------------------
OVERVIEW
----------------------------------------------------------------------

This script evaluates the stability and quality of the flux calibration
derived in Step11 by reconstructing and analyzing per-slit instrumental
response functions.

The QC operates on:
    (1) the summary table of per-slit polynomial fits
    (2) the extracted 1D spectra (for wavelength grids)
    (3) an optional master response function

and produces a multi-page PDF summarizing calibration performance.

----------------------------------------------------------------------
METHOD
----------------------------------------------------------------------

Each slit response function is parameterized as:

    R_s(λ) = a_s x^2 + b_s x + c_s

with:

    x = (λ - λ0) / Δλ

where (a_s, b_s, c_s) are the coefficients derived in Step11.

For QC purposes, the script:

1. Reconstructs R_s(λ) on the native wavelength grid of each slit
2. Interpolates all responses onto a common wavelength grid
3. Normalizes each response by its median value:
       R̃_s(λ) = R_s(λ) / median(R_s)
4. Compares all responses to the master response

----------------------------------------------------------------------
QUALITY CONTROL PRODUCTS
----------------------------------------------------------------------

The output PDF includes:

Page 1:
    - Summary statistics (accepted/rejected slits)
    - Rejection reasons (e.g., negative response, poor fit)

Page 2:
    - Acceptance fraction
    - Distribution of condition numbers
    - Number of valid photometric bands

Page 3:
    - Distribution of polynomial coefficients (a, b, c)
    - Separation between accepted and rejected solutions

Page 4:
    - Master response function

Page 5:
    - Per-slit response curves ("spaghetti plot")
      * Accepted solutions plotted as solid lines
      * Rejected solutions plotted as dashed lines

Page 6:
    - Ensemble response:
        * Median response
        * 16th–84th percentile envelope
        * Comparison with master response

----------------------------------------------------------------------
INTERPRETATION
----------------------------------------------------------------------

This QC provides the primary validation of the flux calibration:

- A tight bundle of accepted curves indicates a stable calibration
- Large dispersion indicates photometric or spectral inconsistencies
- Rejected curves identify pathological solutions (e.g., negative response)

The master response is considered robust if it lies within the envelope
of accepted slit responses.

----------------------------------------------------------------------
INPUTS
----------------------------------------------------------------------

--summary-csv
    CSV file containing per-slit calibration results from Step11.
    Required columns:
        slit_id, coeff_a, coeff_b, coeff_c, lambda0_nm, scale_nm,
        accepted, reason, (optional: cond, n_good_bands)

--extract
    FITS file containing extracted 1D spectra (Step08 output),
    used to recover wavelength grids for each slit.

--master-response (optional)
    FITS file containing the master response function, with columns:
        LAMBDA_NM, RESP_MASTER

--outpdf
    Output multi-page PDF with QC diagnostics.

----------------------------------------------------------------------
ASSUMPTIONS AND LIMITATIONS
----------------------------------------------------------------------

- Wavelength grids may differ across slits; interpolation is required
- np.interp requires strictly increasing wavelength arrays:
  input grids are sorted and deduplicated internally
- Response normalization removes absolute scaling and emphasizes shape
- Edge regions may be sparsely sampled and are masked accordingly

----------------------------------------------------------------------
OUTPUT
----------------------------------------------------------------------

A PDF file summarizing the calibration quality and providing the final
diagnostic plots used for validation and publication.


----------------------------------------------------------------------
TO RUN
----------------------------------------------------------------------
PYTHONPATH=. python qc/step11/qc_step11_response_summary.py \
  --summary-csv ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/11_fluxcal/extract1d_optimal_ridge_all_wav_abscal_summary.csv \
  --master-response ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/11_fluxcal/extract1d_optimal_ridge_all_wav_master_response.fits \
  --extract ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/08_extract1d/extract1d_optimal_ridge_all_wav.fits \
  --outpdf ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/11_fluxcal/qc_step11/qc_step11_response_summary.pdf
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from astropy.io import fits
from astropy.table import Table


def _norm_key(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _norm_slit(s: str) -> str:
    s = str(s).strip().upper()
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return f"SLIT{int(digits):03d}"
    return s


def _read_summary_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "slit_id" not in df.columns:
        raise KeyError(f"Missing 'slit_id' in {path}")
    df["slit_id"] = df["slit_id"].map(_norm_slit)
    if "accepted" in df.columns:
        def as_bool(x):
            s = str(x).strip().upper()
            return s in {"TRUE", "T", "1", "YES"}
        df["accepted"] = df["accepted"].map(as_bool)
    else:
        df["accepted"] = False
    if "reason" not in df.columns:
        df["reason"] = ""
    for c in ["coeff_a", "coeff_b", "coeff_c", "lambda0_nm", "scale_nm", "cond", "n_good_bands"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _choose_wave_column(tab: Table) -> str:
    cols = list(tab.colnames)
    cmap = {_norm_key(c): c for c in cols}
    for cand in ["LAMBDA_NM", "WAVELENGTH_NM", "WAVE_NM", "LAMBDA"]:
        k = _norm_key(cand)
        if k in cmap:
            return cmap[k]
    raise KeyError(f"Could not find wavelength column among {cols}")


def _infer_slit_id_from_hdu(hdu, idx: int) -> str:
    hdr = hdu.header
    for key in ("EXTNAME", "NAME", "OBJECT"):
        if key in hdr:
            val = str(hdr[key]).strip()
            if val:
                return _norm_slit(val)
    for key in ("SLITID", "SLIT"):
        if key in hdr:
            try:
                return f"SLIT{int(hdr[key]):03d}"
            except Exception:
                val = str(hdr[key]).strip()
                if val:
                    return _norm_slit(val)
    return f"HDU{idx:03d}"


def _find_table_hdus(hdul: fits.HDUList) -> List[int]:
    return [i for i, hdu in enumerate(hdul) if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU))]


def _read_master_response(path: Optional[Path]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if path is None:
        return None, None
    path = Path(path)
    if not path.exists():
        return None, None
    with fits.open(path) as hdul:
        for hdu in hdul[1:]:
            if getattr(hdu, "data", None) is None or not hasattr(hdu, "columns"):
                continue
            cols = list(hdu.columns.names)
            if "LAMBDA_NM" in cols and "RESP_MASTER" in cols:
                return np.asarray(hdu.data["LAMBDA_NM"], float), np.asarray(hdu.data["RESP_MASTER"], float)
    return None, None


def _reason_family(reason: str) -> str:
    s = str(reason)
    if "solve_failed" in s:
        return "solve_failed"
    if "positive_frac" in s:
        return "negative response"
    if "band_rms_rel" in s:
        return "bad residuals"
    if "cond=" in s:
        return "ill-conditioned"
    if s.strip().lower() in {"accepted", ""}:
        return "accepted"
    return "other"


def _text_page(pdf: PdfPages, title: str, lines: Sequence[str]) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.03, 0.97, title + "\n\n" + "\n".join(lines),
            va="top", ha="left", family="monospace", fontsize=11)
    pdf.savefig(fig)
    plt.close(fig)


def _build_wave_map(extract_fits: Path) -> Dict[str, np.ndarray]:
    wave_map: Dict[str, np.ndarray] = {}
    with fits.open(extract_fits) as hdul:
        for idx in _find_table_hdus(hdul):
            hdu = hdul[idx]
            slit = _infer_slit_id_from_hdu(hdu, idx)
            try:
                tab = Table(hdu.data)
                wave_col = _choose_wave_column(tab)
            except Exception:
                continue
            wave = np.asarray(tab[wave_col], float)
            good = np.isfinite(wave)
            if np.count_nonzero(good) < 2:
                continue
            wave_map[slit] = wave[good]
    return wave_map


def _build_common_wave_from_map(wave_map: Dict[str, np.ndarray], step_nm: float = 1.0) -> np.ndarray:
    if not wave_map:
        return np.array([], dtype=float)
    mins, maxs = [], []
    for w in wave_map.values():
        if w.size >= 20:
            mins.append(np.nanmin(w))
            maxs.append(np.nanmax(w))
    if not mins or not maxs:
        return np.array([], dtype=float)
    wmin = float(np.nanmin(mins))
    wmax = float(np.nanmax(maxs))
    if not np.isfinite(wmin) or not np.isfinite(wmax) or wmax <= wmin:
        return next(iter(wave_map.values())).copy()
    n = int(np.floor((wmax - wmin) / step_nm)) + 1
    if n < 20:
        return np.linspace(wmin, wmax, 200)
    return wmin + step_nm * np.arange(n, dtype=float)


def _reconstruct_response_on_grid(wave_native: np.ndarray, coeff_a: float, coeff_b: float, coeff_c: float, lambda0_nm: float, scale_nm: float) -> np.ndarray:
    x = (wave_native - float(lambda0_nm)) / float(scale_nm)
    return float(coeff_a) * x**2 + float(coeff_b) * x + float(coeff_c)


def _rebuild_response_curves(extract_fits: Path, summary_df: pd.DataFrame, step_nm: float = 1.0):
    stats = {
        "summary_rows": int(len(summary_df)),
        "rows_with_coeffs": 0,
        "rows_with_wave": 0,
        "rows_interpolated": 0,
    }
    work = summary_df.copy()
    req = ["coeff_a", "coeff_b", "coeff_c", "lambda0_nm", "scale_nm"]
    good_coeff = np.ones(len(work), dtype=bool)
    for c in req:
        if c not in work.columns:
            good_coeff &= False
        else:
            good_coeff &= np.isfinite(work[c].to_numpy(float))
    work = work[good_coeff].copy()
    stats["rows_with_coeffs"] = int(len(work))
    if work.empty:
        return np.array([], dtype=float), [], stats

    wave_map = _build_wave_map(extract_fits)
    if not wave_map:
        return np.array([], dtype=float), [], stats
    common_wave = _build_common_wave_from_map(wave_map, step_nm=step_nm)
    if common_wave.size == 0:
        return np.array([], dtype=float), [], stats

    curves = []
    for _, row in work.iterrows():
        slit = row["slit_id"]
        if slit not in wave_map:
            continue
        wave_native = wave_map[slit]
        stats["rows_with_wave"] += 1
        resp_native = _reconstruct_response_on_grid(
            wave_native=wave_native,
            coeff_a=row["coeff_a"],
            coeff_b=row["coeff_b"],
            coeff_c=row["coeff_c"],
            lambda0_nm=row["lambda0_nm"],
            scale_nm=row["scale_nm"],
        )
        good = np.isfinite(wave_native) & np.isfinite(resp_native)
        if np.count_nonzero(good) < 2:
            continue
        
        w = np.asarray(wave_native[good], float)
        r = np.asarray(resp_native[good], float)
        
        # np.interp requires increasing x
        order = np.argsort(w)
        w = w[order]
        r = r[order]
        
        # remove duplicate wavelength samples if present
        keep = np.concatenate(([True], np.diff(w) > 0))
        w = w[keep]
        r = r[keep]
        
        if w.size < 2:
            continue
        
        resp_common = np.interp(common_wave, w, r, left=np.nan, right=np.nan)
        curves.append((slit, bool(row["accepted"]), resp_common))
        
        stats["rows_interpolated"] += 1
    return common_wave, curves, stats


def make_qc_pdf(summary_csv: Path, outpdf: Path, master_response_fits: Optional[Path] = None, extract_fits: Optional[Path] = None) -> None:
    df = _read_summary_csv(summary_csv)
    n_total = len(df)
    n_acc = int(df["accepted"].sum())
    n_rej = int((~df["accepted"]).sum())
    families = Counter(_reason_family(r) for r in df["reason"].fillna(""))
    reason_lines = [f"{k:>16s} : {v}" for k, v in families.most_common()]
    master_wave, master_resp = _read_master_response(master_response_fits)

    curve_wave = np.array([], dtype=float)
    curves = []
    rebuild_stats = {}
    if extract_fits is not None:
        try:
            curve_wave, curves, rebuild_stats = _rebuild_response_curves(Path(extract_fits), df, step_nm=1.0)
        except Exception as exc:
            rebuild_stats = {"rebuild_error": str(exc)}

    outpdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(outpdf) as pdf:
        lines = [
            f"Summary CSV : {summary_csv}",
            f"Master resp : {master_response_fits}",
            f"Extract FITS: {extract_fits}",
            "",
            f"Total rows     : {n_total}",
            f"Accepted       : {n_acc}",
            f"Rejected       : {n_rej}",
            f"Accepted frac  : {n_acc / n_total:.3f}" if n_total else "Accepted frac  : n/a",
            "",
            "Reason families:",
            *reason_lines,
            "",
        ]
        if rebuild_stats:
            lines += ["Rebuild stats:"]
            for k, v in rebuild_stats.items():
                lines.append(f"  {k:>16s} : {v}")
        _text_page(pdf, "Step11 response-summary QC", lines)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        ax = axes.ravel()
        ax[0].bar(["accepted", "rejected"], [n_acc, n_rej])
        ax[0].set_title("Calibration-star acceptance")
        ax[0].set_ylabel("N")
        fam_names = list(families.keys())
        fam_vals = [families[k] for k in fam_names]
        ax[1].bar(range(len(fam_names)), fam_vals)
        ax[1].set_xticks(range(len(fam_names)))
        ax[1].set_xticklabels(fam_names, rotation=35, ha="right")
        ax[1].set_title("Rejection / status reasons")
        ax[1].set_ylabel("N")
        cond = pd.to_numeric(df.get("cond", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        cond = cond[np.isfinite(cond)]
        if cond.size:
            ax[2].hist(cond, bins=20)
            ax[2].set_title("Condition number")
            ax[2].set_xlabel("cond")
            ax[2].set_ylabel("N")
        else:
            ax[2].axis("off")
        ng = pd.to_numeric(df.get("n_good_bands", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        ng = ng[np.isfinite(ng)]
        if ng.size:
            bins = np.arange(-0.5, np.nanmax(ng) + 1.5, 1)
            ax[3].hist(ng, bins=bins)
            ax[3].set_title("Number of valid bands")
            ax[3].set_xlabel("n_good_bands")
            ax[3].set_ylabel("N")
        else:
            ax[3].axis("off")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        ax = axes.ravel()
        acc = df["accepted"].to_numpy(bool)
        rej = ~acc
        for j, cname in enumerate(["coeff_a", "coeff_b", "coeff_c"]):
            vals = pd.to_numeric(df.get(cname, pd.Series(dtype=float)), errors="coerce").to_numpy(float)
            a = vals[acc & np.isfinite(vals)]
            r = vals[rej & np.isfinite(vals)]
            if a.size or r.size:
                ax[j].hist(r, bins=20, alpha=0.6, label="rejected")
                ax[j].hist(a, bins=20, alpha=0.6, label="accepted")
                ax[j].set_title(cname)
                ax[j].legend(fontsize=8)
            else:
                ax[j].axis("off")
        coeff_a = pd.to_numeric(df.get("coeff_a", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        coeff_b = pd.to_numeric(df.get("coeff_b", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        good = np.isfinite(coeff_a) & np.isfinite(coeff_b)
        if np.any(good):
            ax[3].scatter(coeff_a[rej & good], coeff_b[rej & good], s=16, alpha=0.7, label="rejected")
            ax[3].scatter(coeff_a[acc & good], coeff_b[acc & good], s=16, alpha=0.7, label="accepted")
            ax[3].set_title("coeff_a vs coeff_b")
            ax[3].set_xlabel("coeff_a")
            ax[3].set_ylabel("coeff_b")
            ax[3].legend(fontsize=8)
        else:
            ax[3].axis("off")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        if master_wave is not None and master_resp is not None:
            okm = np.isfinite(master_wave) & np.isfinite(master_resp)
            if np.any(okm):
                medm = np.nanmedian(master_resp[okm])
                master_plot = master_resp[okm] / medm if np.isfinite(medm) and medm != 0 else master_resp[okm]
                okm = np.isfinite(master_wave) & np.isfinite(master_resp)
                if np.any(okm):
                    medm = np.nanmedian(master_resp[okm])
                    master_plot = master_resp[okm] / medm if np.isfinite(medm) and medm != 0 else master_resp[okm]
                    ax.plot(master_wave[okm], master_plot, lw=2.0, color="k", label="master")
                ax.axhline(0.0, lw=0.8, ls="--", color="0.4")
                ax.set_xlabel("Wavelength (nm)")
                ax.set_ylabel("Response")
                ax.set_title("Step11 master response")
                ax.legend()
            else:
                ax.text(0.5, 0.5, "Master response has no finite samples", ha="center", va="center")
                ax.set_axis_off()
        else:
            ax.text(0.5, 0.5, "No master-response FITS provided or readable", ha="center", va="center")
            ax.set_axis_off()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        if curve_wave.size and curves:
            nacc, nrej = 0, 0
            for slit, accepted, resp in curves:
                ok = np.isfinite(resp)
                if not np.any(ok):
                    continue
        
                resp_plot = resp[ok]
                med = np.nanmedian(resp_plot)
                if np.isfinite(med) and med != 0:
                    resp_plot = resp_plot / med
        
                if accepted:
                    ax.plot(curve_wave[ok], resp_plot, alpha=0.25, lw=0.8, color="C0")
                    nacc += 1
                else:
                    ax.plot(curve_wave[ok], resp_plot, alpha=0.20, lw=0.7, ls="--", color="C1")
                    nrej += 1
        
            if master_wave is not None and master_resp is not None:
                okm = np.isfinite(master_wave) & np.isfinite(master_resp)
                if np.any(okm):
                    medm = np.nanmedian(master_resp[okm])
                    master_plot = master_resp[okm] / medm if np.isfinite(medm) and medm != 0 else master_resp[okm]
                    ax.plot(master_wave[okm], master_plot, color="k", lw=2.0, label="master")
        
            ax.axhline(1.0, lw=0.8, ls="--", color="0.4")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Normalized response")
            ax.set_title(f"Per-star response curves (accepted={nacc}, rejected={nrej})")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "Could not reconstruct per-star response curves", ha="center", va="center")
            ax.set_axis_off()
    
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        if curve_wave.size and curves:
            acc_arrs = []
            for slit, accepted, resp in curves:
                if not accepted:
                    continue
                ok = np.isfinite(resp)
                if not np.any(ok):
                    continue
            
                resp_full = np.full_like(resp, np.nan, dtype=float)
                med = np.nanmedian(resp[ok])
                if np.isfinite(med) and med != 0:
                    resp_full[ok] = resp[ok] / med
                else:
                    resp_full[ok] = resp[ok]
            
                acc_arrs.append(resp_full)
            if acc_arrs:
                stack = np.vstack(acc_arrs)
                valid_frac = np.sum(np.isfinite(stack), axis=0) / stack.shape[0]
                mask = valid_frac > 0.3   # keep wavelengths where ≥30% stars exist
                wave_plot = curve_wave[mask]
                stack_plot = stack[:, mask]
                
                p16 = np.nanpercentile(stack_plot, 16, axis=0)
                p50 = np.nanpercentile(stack_plot, 50, axis=0)
                p84 = np.nanpercentile(stack_plot, 84, axis=0)
                ax.fill_between(wave_plot, p16, p84, alpha=0.3, label="16-84%")
                ax.plot(wave_plot, p50, lw=1.6, label="median accepted")
                if master_wave is not None and master_resp is not None:
                    okm = np.isfinite(master_wave) & np.isfinite(master_resp)
                    ax.plot(master_wave[okm], master_resp[okm], lw=2.0, color="k", label="master")
                ax.axhline(0.0, lw=0.8, ls="--", color="0.4")
                ax.set_xlabel("Wavelength (nm)")
                ax.set_ylabel("Response")
                ax.set_title("Accepted-star response envelope")
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, "No accepted response curves available", ha="center", va="center")
                ax.set_axis_off()
        else:
            ax.text(0.5, 0.5, "No reconstructed curve stack available", ha="center", va="center")
            ax.set_axis_off()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Step11 calibration-response QC summary.")
    p.add_argument("--summary-csv", required=True, help="Step11 abscal summary CSV")
    p.add_argument("--outpdf", required=True, help="Output PDF")
    p.add_argument("--master-response", default=None, help="Optional master response FITS")
    p.add_argument("--extract", default=None, help="Optional extract FITS used to rebuild response curves")
    return p.parse_args()


def main():
    args = parse_args()
    make_qc_pdf(
        summary_csv=Path(args.summary_csv),
        outpdf=Path(args.outpdf),
        master_response_fits=Path(args.master_response) if args.master_response else None,
        extract_fits=Path(args.extract) if args.extract else None,
    )


if __name__ == "__main__":
    main()
