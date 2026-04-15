#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step07e — refine initial arc shifts using windowed cross-correlation.

This is the plumbing-updated version of the old Step07.25 refinement script.

PURPOSE
-------
Starting from the initial bright-line shifts measured in Step07d, refine the
per-slit shifts with a local cross-correlation against a reference slit inside
a limited Y window around the selected bright line.

This step is intentionally conservative:
- it starts from SHIFT0 from Step07d
- it only searches a small extra lag range (+/- MAX_DSHIFT)
- it writes a refined shift table and a final aligned stack

INPUTS
------
- Step07c 1D arc MEF:
    ArcDiff_*_pixflatcorr_clipped_1D_slitid_<EVEN|ODD>.fits

- Step07d initial shifts CSV:
    Arc_shifts_initial_<EVEN|ODD>.csv

OUTPUTS
-------
- Arc_shifts_final_<EVEN|ODD>.csv
- Arc_stack_aligned_final_<EVEN|ODD>.fits

METHOD
------
For each slit:
1. roll the 1D spectrum by SHIFT0 from Step07d
2. define a window around the reference bright line
3. cross-correlate against the similarly aligned reference slit
4. search only +/- MAX_DSHIFT pixels
5. write SHIFT_FINAL = SHIFT0 + DSHIFT

NOTES
-----
- This script preserves the old Step07.25 science logic.
- Plumbing has been updated to the restored pipeline naming and config model.
- By default the reference slit is inherited from Step07d as the row with
  SHIFT_vs_REF(px) == 0 in the initial CSV.
- You may still override the reference slit explicitly with --refslit.

RUN
---
    PYTHONPATH=. python pipeline/step07_wavecal/step07e_refine_stack_arc.py --set EVEN
    PYTHONPATH=. python pipeline/step07_wavecal/step07e_refine_stack_arc.py --set ODD
"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits

import config


# =============================================================================
# Tunable parameters
# =============================================================================
WIN_HALF = 250       # window is [center-WIN_HALF, center+WIN_HALF]
MAX_DSHIFT = 10      # search +/- this many pixels around 0
EDGE_PAD = 20        # avoid edges inside valid region
MIN_COVERAGE_FRAC = 0.70


# =============================================================================
# Helpers
# =============================================================================
def slit_num(s: str) -> int:
    return int(s.replace("SLIT", ""))


def list_slits(hdul: fits.HDUList) -> list[str]:
    slits = []
    for h in hdul[1:]:
        nm = (h.header.get("EXTNAME") or "").strip().upper()
        if nm.startswith("SLIT") and len(nm) == 7:
            slits.append(nm)
    return sorted(set(slits), key=slit_num)


def robust_valid_region(npix_per_row: np.ndarray, frac: float) -> np.ndarray:
    """Rows considered valid if npix_per_row >= frac * median_nonzero."""
    v = npix_per_row[np.isfinite(npix_per_row) & (npix_per_row > 0)]
    if v.size == 0:
        return np.ones_like(npix_per_row, dtype=bool)
    med = np.median(v)
    return npix_per_row >= (frac * med)


def best_lag_xcorr(ref: np.ndarray, sig: np.ndarray, max_lag: int) -> tuple[int, float]:
    """
    Find lag in [-max_lag, +max_lag] that maximizes dot(ref, roll(sig, -lag)).
    Returns (best_lag, best_corr).
    """
    r = ref - np.nanmedian(ref)
    s = sig - np.nanmedian(sig)

    r = np.where(np.isfinite(r), r, 0.0)
    s = np.where(np.isfinite(s), s, 0.0)

    best_corr = -np.inf
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        ss = np.roll(s, -lag)
        corr = float(np.dot(r, ss))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return best_lag, best_corr


def default_arc1d_path(trace_set: str) -> Path:
    st07 = Path(config.ST07_WAVECAL).expanduser()
    if hasattr(config, "MASTER_ARC_DIFF"):
        stem = Path(config.MASTER_ARC_DIFF).stem
        return st07 / f"{stem}_pixflatcorr_clipped_1D_slitid_{trace_set}.fits"
    hits = sorted(st07.glob(f"ArcDiff*_pixflatcorr_clipped_1D_slitid_{trace_set}.fits"))
    if hits:
        return hits[0]
    return st07 / f"ArcDiff_pixflatcorr_clipped_1D_slitid_{trace_set}.fits"


def default_shift0_csv(trace_set: str) -> Path:
    return Path(config.ST07_WAVECAL).expanduser() / f"Arc_shifts_initial_{trace_set}.csv"


def infer_refslit_from_shift0(csv_path: Path) -> str:
    df = read_shift0_csv(csv_path)
    zero = df[df["SHIFT0"] == 0]
    if len(zero) >= 1:
        return str(zero.iloc[0]["SLIT"]).upper()
    raise RuntimeError(f"Could not infer reference slit from {csv_path}: no row with SHIFT0 == 0")


def read_shift0_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read Step07d shift CSV and return standardized columns:
      SLIT, SHIFT0, BRY
    """
    import re
    df = pd.read_csv(csv_path)

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.lower())

    norm_map = {norm(c): c for c in df.columns}

    slit_col = None
    for cand in ["slit", "extname", "slitid"]:
        if cand in norm_map:
            slit_col = norm_map[cand]
            break
    if slit_col is None:
        raise RuntimeError(f"CSV missing SLIT column. Found columns={list(df.columns)} in {csv_path}")

    shift_col = None
    for cand in [
        "shiftvsrefpx",      # SHIFT_vs_REF(px)
        "shiftvsref",        # SHIFT_vs_REF
        "shift0",
        "shift",
        "dely",
        "dy",
    ]:
        if cand in norm_map:
            shift_col = norm_map[cand]
            break
    if shift_col is None:
        raise RuntimeError(f"CSV missing shift column. Found columns={list(df.columns)} in {csv_path}")

    bry_col = None
    for cand in ["bry", "idx", "brighty", "brightliney"]:
        if cand in norm_map:
            bry_col = norm_map[cand]
            break
    if bry_col is None:
        raise RuntimeError(f"CSV missing BRY column. Found columns={list(df.columns)} in {csv_path}")

    out = pd.DataFrame({
        "SLIT": df[slit_col].astype(str).str.upper(),
        "SHIFT0": df[shift_col].astype(float),
        "BRY": df[bry_col].astype(float),
    })

    out["SHIFT0"] = out["SHIFT0"].round().astype(int)
    out["BRY"] = out["BRY"].round().astype(int)
    return out


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["EVEN", "ODD"], default="EVEN")
    ap.add_argument("--arc1d", type=str, default=None,
                    help="Override Step07c 1D arc MEF")
    ap.add_argument("--shift0-csv", type=str, default=None,
                    help="Override Step07d initial shift CSV")
    ap.add_argument("--refslit", type=str, default=None,
                    help="Optional explicit reference slit, e.g. SLIT024")
    ap.add_argument("--out-csv", type=str, default=None,
                    help="Override final shift CSV")
    ap.add_argument("--out-stack", type=str, default=None,
                    help="Override final aligned stack FITS")
    args = ap.parse_args()

    trace_set = args.set.upper()
    wavecal_dir = Path(config.ST07_WAVECAL).expanduser()

    arc1d_fits = Path(args.arc1d).expanduser() if args.arc1d else default_arc1d_path(trace_set)
    shift0_csv = Path(args.shift0_csv).expanduser() if args.shift0_csv else default_shift0_csv(trace_set)

    out_csv = Path(args.out_csv).expanduser() if args.out_csv else (wavecal_dir / f"Arc_shifts_final_{trace_set}.csv")
    out_stack = Path(args.out_stack).expanduser() if args.out_stack else (wavecal_dir / f"Arc_stack_aligned_final_{trace_set}.fits")

    if not arc1d_fits.exists():
        raise FileNotFoundError(arc1d_fits)
    if not shift0_csv.exists():
        raise FileNotFoundError(shift0_csv)

    refslit = args.refslit.strip().upper() if args.refslit else infer_refslit_from_shift0(shift0_csv)

    print("ARC1D_FITS =", arc1d_fits)
    print("SHIFT0_CSV =", shift0_csv)
    print("REFSLIT    =", refslit)
    print("OUT_CSV    =", out_csv)
    print("OUT_STACK  =", out_stack)

    df0 = read_shift0_csv(shift0_csv)
    shift0 = {r.SLIT: int(r.SHIFT0) for r in df0.itertuples(index=False)}
    bry0 = {r.SLIT: int(r.BRY) for r in df0.itertuples(index=False)}

    with fits.open(arc1d_fits) as h:
        slits = list_slits(h)
        if not slits:
            raise RuntimeError(f"No SLIT### extensions found in {arc1d_fits}")

        if refslit not in slits:
            raise RuntimeError(f"REFSLIT={refslit} not in ARC1D. Available: {slits[:5]} ... {slits[-5:]}")
        if refslit not in shift0:
            raise RuntimeError(f"REFSLIT={refslit} not in SHIFT0 table. Check {shift0_csv}")

        # Reference data (apply SHIFT0 of refslit too, for robustness)
        ref_flux = h[refslit].data[0].astype(float)
        ref_np = h[refslit].data[1].astype(float)

        s0_ref = int(shift0[refslit])
        ref_flux0 = np.roll(ref_flux, -s0_ref)
        ref_np0 = np.roll(ref_np, -s0_ref)

        valid_ref = robust_valid_region(ref_np0, frac=MIN_COVERAGE_FRAC)
        y_ref = np.where(valid_ref)[0]
        if y_ref.size == 0:
            raise RuntimeError("No valid rows in reference slit (coverage).")

        ylo_valid = int(y_ref.min()) + EDGE_PAD
        yhi_valid = int(y_ref.max()) - EDGE_PAD
        if yhi_valid <= ylo_valid:
            raise RuntimeError("Reference valid region too small; reduce EDGE_PAD or coverage threshold.")

        center_ref = int(bry0[refslit] - s0_ref)
        center_ref = max(ylo_valid, min(yhi_valid, center_ref))

        y0 = max(ylo_valid, center_ref - WIN_HALF)
        y1 = min(yhi_valid, center_ref + WIN_HALF)
        if y1 <= y0 + 10:
            raise RuntimeError("Reference window too small; adjust WIN_HALF/EDGE_PAD.")

        ref_win = ref_flux0[y0:y1 + 1].copy()

        rows = []
        for slit in slits:
            if slit not in shift0 or slit not in bry0:
                continue

            f = h[slit].data[0].astype(float)
            npix = h[slit].data[1].astype(float)
            s0 = int(shift0[slit])

            # Coarse alignment using SHIFT0
            f0 = np.roll(f, -s0)
            np0 = np.roll(npix, -s0)

            valid = robust_valid_region(np0, frac=MIN_COVERAGE_FRAC)
            sig_win = f0[y0:y1 + 1].copy()

            dshift, corr = best_lag_xcorr(ref_win, sig_win, MAX_DSHIFT)
            s_final = int(s0 + dshift)

            rows.append({
                "slit": slit,
                "SHIFT0": s0,
                "DSHIFT": int(dshift),
                "SHIFT_FINAL": s_final,
                "BRY": int(bry0[slit]),
                "WIN_Y0": int(y0),
                "WIN_Y1": int(y1),
                "CORR": float(corr),
                "VALID_FRAC": float(np.mean(valid[y0:y1 + 1])) if (y1 >= y0) else np.nan,
            })

    df = pd.DataFrame(rows).sort_values("slit", key=lambda s: s.map(slit_num))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv, f"({len(df)} slits)")
    print(df[["slit", "SHIFT0", "DSHIFT", "SHIFT_FINAL", "CORR", "VALID_FRAC"]]
          .sort_values("CORR").head(10))

    with fits.open(arc1d_fits) as h:
        slits = list_slits(h)
        ny = h[slits[0]].data.shape[1]

        shift_final = {r.slit.upper(): int(r.SHIFT_FINAL) for r in df.itertuples(index=False)}

        stack = np.full((len(slits), ny), np.nan, dtype=np.float32)
        slit_ids = np.array([slit_num(s) for s in slits], dtype=np.int16)

        for i, s in enumerate(slits):
            if s not in shift_final:
                continue
            flux = h[s].data[0].astype(np.float32)
            stack[i] = np.roll(flux, -shift_final[s])

        ph = fits.PrimaryHDU(stack)
        ph.header["STAGE"] = ("07e", "Pipeline stage")
        ph.header["TRACESET"] = (trace_set, "EVEN/ODD")
        ph.header["ARC1D"] = (Path(arc1d_fits).name, "Source 1D arc MEF")
        ph.header["SHIFT0"] = (Path(shift0_csv).name, "Input SHIFT0 CSV")
        ph.header["REFSLIT"] = (refslit, "Reference slit")
        ph.header["NSLITS"] = (stack.shape[0], "Number of slits (rows)")
        ph.header["NY"] = (stack.shape[1], "Length of 1D spectra (Y pixels)")
        ph.header["WINHALF"] = (WIN_HALF, "Half-window for xcorr")
        ph.header["MAXDSH"] = (MAX_DSHIFT, "Max fine shift searched")
        ph.header["Y0WIN"] = (int(df["WIN_Y0"].iloc[0]) if len(df) else -1, "Window Y0")
        ph.header["Y1WIN"] = (int(df["WIN_Y1"].iloc[0]) if len(df) else -1, "Window Y1")

        hdu_ids = fits.ImageHDU(slit_ids, name="SLITIDS")

        out_stack.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList([ph, hdu_ids]).writeto(out_stack, overwrite=True)
        print("Wrote:", out_stack)


if __name__ == "__main__":
    main()
