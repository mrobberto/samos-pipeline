#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step09h — consensus validation / restoration after ABAB OH subtraction.

Purpose
-------
ABAB fits and subtracts OH-like features slit by slit. Some fitted features may
actually be astrophysical features unique to one object. This script compares
the ABAB-subtracted OH_MODEL across all slits and restores features that do not
belong to the common sky-line population.

Input
-----
Merged Step09 ABAB product, expected columns per slit:
    LAMBDA_NM
    OBJ_PRESKY
    OH_MODEL
    STELLAR or RESID_POSTOH

Output
------
Same MEF with added columns:
    OH_MODEL_CONSENSUS
    RESTORE_MODEL
    STELLAR_CONSENSUS
    RESID_POSTOH_CONSENSUS
    CONSENSUS_FLAG

Interpretation
--------------
    OH_MODEL_CONSENSUS = accepted common-sky part of OH_MODEL
    RESTORE_MODEL      = rejected object-like part restored to source
    STELLAR_CONSENSUS  = OBJ_PRESKY - OH_MODEL_CONSENSUS
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from scipy.signal import find_peaks
from scipy.ndimage import binary_dilation


OH_WINDOWS = [
    (780.0, 805.0),
    (806.0, 825.0),
    (845.0, 875.0),
    (875.0, 905.0),
    (905.0, 930.0),
    (930.0, 960.0),
]


def norm_slit(name: str) -> str:
    name = str(name).strip().upper()
    digits = "".join(ch for ch in name if ch.isdigit())
    return f"SLIT{int(digits):03d}" if digits else name


def slit_num(name: str) -> int:
    try:
        return int(norm_slit(name).replace("SLIT", ""))
    except Exception:
        return 10**9


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


def robust_rms(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    return float(np.sqrt(np.nanmedian((x - med) ** 2)))


def choose_col(tab, candidates):
    names = {c.upper(): c for c in tab.colnames}
    for c in candidates:
        if c.upper() in names:
            return names[c.upper()]
    raise KeyError(f"Missing any of columns: {candidates}")


def interp_to_grid(lam, y, grid):
    lam = np.asarray(lam, float)
    y = np.asarray(y, float)
    ok = np.isfinite(lam) & np.isfinite(y)
    if ok.sum() < 20:
        return np.full_like(grid, np.nan, dtype=float)
    order = np.argsort(lam[ok])
    x = lam[ok][order]
    z = y[ok][order]
    good = np.concatenate([[True], np.diff(x) > 0])
    x = x[good]
    z = z[good]
    if len(x) < 20:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, x, z, left=np.nan, right=np.nan)


def detect_oh_model_peaks(grid, model, min_prom_sigma=3.0, min_height_sigma=2.0, min_sep_nm=0.25):
    model = np.asarray(model, float)
    finite = np.isfinite(model)
    if finite.sum() < 20:
        return []

    sig = robust_sigma(model[finite])
    if not np.isfinite(sig) or sig <= 0:
        return []

    dlam = float(np.nanmedian(np.diff(grid)))
    min_sep_pix = max(1, int(round(min_sep_nm / dlam)))

    y = model.copy()
    y[~finite] = 0.0

    peaks, props = find_peaks(
        y,
        height=min_height_sigma * sig,
        prominence=min_prom_sigma * sig,
        distance=min_sep_pix,
    )

    out = []
    for k, p in enumerate(peaks):
        out.append(
            dict(
                ipix=int(p),
                lambda_nm=float(grid[p]),
                height=float(props["peak_heights"][k]),
                prominence=float(props["prominences"][k]),
            )
        )
    return out


def build_peak_catalog(slits, grid, models, args):
    rows = []
    for slit in slits:
        peaks = detect_oh_model_peaks(
            grid,
            models[slit],
            min_prom_sigma=args.min_prom_sigma,
            min_height_sigma=args.min_height_sigma,
            min_sep_nm=args.min_sep_nm,
        )
        for j, pk in enumerate(peaks):
            rows.append(
                dict(
                    slit=slit,
                    peak_id=f"{slit}_P{j:04d}",
                    lambda_nm=pk["lambda_nm"],
                    ipix=pk["ipix"],
                    height=pk["height"],
                    prominence=pk["prominence"],
                )
            )
    return pd.DataFrame(rows)


def classify_peaks(catalog, slits, args):
    """
    For every peak, count how many other slits have a peak nearby.
    If enough matches exist, classify as common sky.
    Otherwise classify as object-like / non-consensus.
    """
    if len(catalog) == 0:
        catalog["n_match"] = []
        catalog["class"] = []
        return catalog

    lam_all = np.asarray(catalog["lambda_nm"], float)
    slit_all = np.asarray(catalog["slit"].values)

    n_match = []
    classes = []

    for _, row in catalog.iterrows():
        lam0 = float(row["lambda_nm"])
        slit0 = row["slit"]

        other = slit_all != slit0
        close = np.abs(lam_all - lam0) <= args.match_tol_nm
        matched_slits = set(slit_all[other & close])

        nm = len(matched_slits)
        n_match.append(nm)

        frac = nm / max(len(slits) - 1, 1)

        if nm >= args.min_match and frac >= args.min_match_frac:
            classes.append("SKY_COMMON")
        elif nm <= args.max_unique_match:
            classes.append("OBJECTLIKE_UNIQUE")
        else:
            classes.append("AMBIGUOUS")

    out = catalog.copy()
    out["n_match"] = n_match
    out["match_frac"] = np.asarray(n_match, float) / max(len(slits) - 1, 1)
    out["class"] = classes
    return out


def make_reject_mask(grid, model, peak_rows, args):
    """
    Build wavelength mask around rejected peaks.
    In version 1, reject a fixed half-width around non-consensus peaks.
    """
    mask = np.zeros_like(grid, dtype=bool)

    if len(peak_rows) == 0:
        return mask

    for _, row in peak_rows.iterrows():
        lam0 = float(row["lambda_nm"])
        mask |= np.abs(grid - lam0) <= args.restore_half_width_nm

    if args.restore_grow_pix > 0:
        mask = binary_dilation(mask, iterations=args.restore_grow_pix)

    return mask


def add_or_replace(tab: Table, name: str, values):
    values = np.asarray(values)
    if name in tab.colnames:
        tab[name] = values
    else:
        tab[name] = values


def parse_args():
    p = argparse.ArgumentParser(
        description="Consensus validation/restoration for Step09 ABAB OH subtraction"
    )

    p.add_argument("--in-fits", type=Path, required=True)
    p.add_argument("--out-fits", type=Path, required=True)
    p.add_argument("--summary-csv", type=Path, default=None)
    p.add_argument("--peak-csv", type=Path, default=None)

    p.add_argument("--grid-step-nm", type=float, default=0.02)

    p.add_argument("--min-prom-sigma", type=float, default=3.0)
    p.add_argument("--min-height-sigma", type=float, default=2.0)
    p.add_argument("--min-sep-nm", type=float, default=0.25)

    p.add_argument("--match-tol-nm", type=float, default=0.20)
    p.add_argument("--min-match", type=int, default=6)
    p.add_argument("--min-match-frac", type=float, default=0.10)
    p.add_argument("--max-unique-match", type=int, default=1)

    p.add_argument("--restore-half-width-nm", type=float, default=0.35)
    p.add_argument("--restore-grow-pix", type=int, default=2)

    p.add_argument(
        "--restore-ambiguous",
        action="store_true",
        help="Also restore AMBIGUOUS peaks. Default restores only OBJECTLIKE_UNIQUE.",
    )
    p.add_argument(
    "--lambda-reference-fits",
    type=Path,
    default=None,
    help="Reference MEF whose per-slit LAMBDA_NM columns should be preserved in the output.",
)

    return p.parse_args()


def main():
    args = parse_args()

    if not args.in_fits.exists():
        raise FileNotFoundError(args.in_fits)

    ref_lambda = {}

    if args.lambda_reference_fits is not None:
        if not args.lambda_reference_fits.exists():
            raise FileNotFoundError(args.lambda_reference_fits)
    
        with fits.open(args.lambda_reference_fits) as href:
            for h in href[1:]:
                name = norm_slit(h.name)
                if not name.startswith("SLIT"):
                    continue
                tab_ref = Table(h.data)
                if "LAMBDA_NM" in tab_ref.colnames:
                    ref_lambda[name] = np.asarray(tab_ref["LAMBDA_NM"], float)
                    
    args.out_fits.parent.mkdir(parents=True, exist_ok=True)

    with fits.open(args.in_fits) as hdul:
        slits = sorted(
            [h.name for h in hdul[1:] if str(h.name).upper().startswith("SLIT")],
            key=slit_num,
        )

        if len(slits) == 0:
            raise RuntimeError("No SLIT extensions found")

        # Common wavelength grid
        all_lam = []
        for slit in slits:
            tab = Table(hdul[slit].data)
            lam_col = choose_col(tab, ["LAMBDA_NM", "WAVELENGTH_NM", "LAMBDA"])
            all_lam.append(np.asarray(tab[lam_col], float))

        lo = max(np.nanmin(x) for x in all_lam)
        hi = min(np.nanmax(x) for x in all_lam)
        grid = np.arange(lo, hi, args.grid_step_nm)

        models = {}
        objpresky = {}
        stellar_abab = {}

        for slit in slits:
            tab = Table(hdul[slit].data)
            lam_col = choose_col(tab, ["LAMBDA_NM", "WAVELENGTH_NM", "LAMBDA"])
            obj_col = choose_col(tab, ["OBJ_PRESKY"])
            oh_col = choose_col(tab, ["OH_MODEL", "OH_MODEL_FINAL", "OH_MODEL_P1"])
            stellar_col = choose_col(tab, ["STELLAR", "STELLAR_FINAL", "STELLAR_P1", "RESID_POSTOH"])

            lam = np.asarray(tab[lam_col], float)
            objpresky[slit] = interp_to_grid(lam, np.asarray(tab[obj_col], float), grid)
            models[slit] = interp_to_grid(lam, np.asarray(tab[oh_col], float), grid)
            stellar_abab[slit] = interp_to_grid(lam, np.asarray(tab[stellar_col], float), grid)

        catalog = build_peak_catalog(slits, grid, models, args)
        catalog = classify_peaks(catalog, slits, args)

        if args.peak_csv is None:
            args.peak_csv = args.out_fits.with_suffix("").with_name(args.out_fits.stem + "_peaks.csv")
        catalog.to_csv(args.peak_csv, index=False)
        print("Wrote peak catalog:", args.peak_csv)

        out_hdus = [fits.PrimaryHDU(header=hdul[0].header.copy())]
        summary_rows = []

        for slit in slits:
            in_hdu = hdul[slit]
            tab = Table(in_hdu.data)
            
            slit_norm = norm_slit(slit)

            if slit_norm in ref_lambda:
                lam_ref = ref_lambda[slit_norm]
            
                if len(lam_ref) != len(tab):
                    raise RuntimeError(
                        f"{slit}: reference LAMBDA_NM length mismatch: "
                        f"{len(lam_ref)} vs {len(tab)}"
                    )
            
                # -------------------------------------------------
                # Detect accidental wavelength drift
                # BEFORE restoring canonical grid
                # -------------------------------------------------
                
                if "LAMBDA_NM" in tab.colnames:
                
                    lam_current = np.asarray(tab["LAMBDA_NM"], float)
                
                    diff_before = np.nanmedian(lam_current - lam_ref)
                
                    if abs(diff_before) > 1e-6:
                        print(
                            f"[WARN] {slit}: Step09 wavelength drift detected "
                            f"(median Δλ = {diff_before:.6f} nm). "
                            f"Restoring Step08 wavelength grid."
                            )
                    tab["LAMBDA_NM"] = lam_ref
                else:
                    tab.add_column(lam_ref, name="LAMBDA_NM")
            
                        
            lam_col = choose_col(tab, ["LAMBDA_NM", "WAVELENGTH_NM", "LAMBDA"])
            obj_col = choose_col(tab, ["OBJ_PRESKY"])
            oh_col = choose_col(tab, ["OH_MODEL", "OH_MODEL_FINAL", "OH_MODEL_P1"])

            lam_native = np.asarray(tab[lam_col], float)
            obj_native = np.asarray(tab[obj_col], float)
            oh_native = np.asarray(tab[oh_col], float)

            slit_peaks = catalog[catalog["slit"] == slit].copy()

            if args.restore_ambiguous:
                reject = slit_peaks[slit_peaks["class"].isin(["OBJECTLIKE_UNIQUE", "AMBIGUOUS"])]
            else:
                reject = slit_peaks[slit_peaks["class"] == "OBJECTLIKE_UNIQUE"]

            reject_mask_grid = make_reject_mask(grid, models[slit], reject, args)
            reject_mask_native = np.interp(
                lam_native,
                grid,
                reject_mask_grid.astype(float),
                left=0.0,
                right=0.0,
            ) > 0.5

            restore_model = np.where(reject_mask_native, oh_native, 0.0)
            oh_consensus = oh_native - restore_model
            stellar_consensus = obj_native - oh_consensus
            resid_consensus = stellar_consensus.copy()

            flag = np.zeros(len(tab), dtype=np.int16)
            flag[reject_mask_native] = 1

            add_or_replace(tab, "OH_MODEL_CONSENSUS", oh_consensus.astype("f4"))
            add_or_replace(tab, "RESTORE_MODEL", restore_model.astype("f4"))
            add_or_replace(tab, "STELLAR_CONSENSUS", stellar_consensus.astype("f4"))
            add_or_replace(tab, "RESID_POSTOH_CONSENSUS", resid_consensus.astype("f4"))
            add_or_replace(tab, "CONSENSUS_FLAG", flag)

            hdr = in_hdu.header.copy()
            hdr["S9HCONS"] = (True, "Step09h consensus restoration applied")
            hdr["S9HREJ"] = (int(len(reject)), "Rejected/restored non-consensus peaks")
            hdr["S9HPEAK"] = (int(len(slit_peaks)), "Detected OH_MODEL peaks")

            out_hdus.append(fits.BinTableHDU(tab, header=hdr, name=slit))

            summary_rows.append(
                dict(
                    slit=slit,
                    n_peaks=int(len(slit_peaks)),
                    n_sky_common=int(np.sum(slit_peaks["class"] == "SKY_COMMON")),
                    n_objectlike_unique=int(np.sum(slit_peaks["class"] == "OBJECTLIKE_UNIQUE")),
                    n_ambiguous=int(np.sum(slit_peaks["class"] == "AMBIGUOUS")),
                    restored_flux=float(np.nansum(restore_model)),
                    original_oh_flux=float(np.nansum(oh_native)),
                    restored_frac=float(
                        np.nansum(restore_model) / np.nansum(oh_native)
                        if np.nansum(oh_native) != 0
                        else np.nan
                    ),
                    rms_abab=float(robust_rms(obj_native - oh_native)),
                    rms_consensus=float(robust_rms(stellar_consensus)),
                )
            )

        fits.HDUList(out_hdus).writeto(args.out_fits, overwrite=True)
        print("Wrote:", args.out_fits)

    summary = pd.DataFrame(summary_rows)
    if args.summary_csv is None:
        args.summary_csv = args.out_fits.with_suffix("").with_name(args.out_fits.stem + "_summary.csv")
    summary.to_csv(args.summary_csv, index=False)
    print("Wrote summary:", args.summary_csv)


if __name__ == "__main__":
    main()