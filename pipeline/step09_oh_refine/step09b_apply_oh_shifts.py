#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step09b — apply per-slit OH wavelength zero-point corrections.

CLEANED FOR NEW PIPELINE SEMANTICS
----------------------------------
Pipeline meaning is now:
  Step09 = OH refine
  Step10 = telluric

Science logic preserved from the prior working Step10b script.
Only plumbing/default-path behavior has been updated.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits
import config


def _normcol(s: str) -> str:
    return s.strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def find_latest(folder: Path, pattern: str) -> Optional[Path]:
    hits = sorted(folder.glob(pattern))
    return hits[-1] if hits else None


def read_oh_csv(csv_path: Path) -> Tuple[Dict[str, float], Dict[str, bool]]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        cols_norm = {_normcol(c): c for c in cols}

        def pick(*cands: str) -> Optional[str]:
            for c in cands:
                if c in cols_norm:
                    return cols_norm[c]
            return None

        slit_col = pick("slit", "extname")
        shift_col = pick("shiftnm", "shift_nm", "shiftnanometers", "shiftnmmed")
        use_col = pick("use", "useflag", "ok", "good")
        if slit_col is None or shift_col is None:
            raise RuntimeError(f"CSV missing required columns. Found: {cols}")

        shifts: Dict[str, float] = {}
        useflag: Dict[str, bool] = {}
        for row in reader:
            slit = (row.get(slit_col, "") or "").strip().upper()
            if not slit or not slit.startswith("SLIT"):
                continue
            try:
                sh = float(row.get(shift_col, "nan"))
            except Exception:
                sh = float("nan")
            shifts[slit] = sh
            if use_col is None:
                useflag[slit] = np.isfinite(sh)
            else:
                v = (row.get(use_col, "") or "").strip().lower()
                useflag[slit] = v in ("1", "true", "t", "yes", "y", "ok")
        return shifts, useflag


def robust_stats(x: np.ndarray) -> Tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    return med, 1.4826 * mad


def add_or_replace_column(tab: fits.FITS_rec, name: str, data: np.ndarray, fmt: str = "E") -> fits.FITS_rec:
    name_u = name.upper()
    cols = []
    for i, colname in enumerate(tab.names):
        if colname.upper() == name_u:
            continue
        cols.append(fits.Column(name=colname, format=tab.columns[i].format, array=tab[colname]))
    cols.append(fits.Column(name=name_u, format=fmt, array=data))
    return fits.BinTableHDU.from_columns(cols).data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", default=None, help="Input MEF (default: config.EXTRACT1D_WAV)")
    p.add_argument("--csv", dest="csvfile", default=None, help="CSV of OH shifts (default: config.OH_SHIFT_CSV)")
    p.add_argument("--out", dest="outfile", default=None, help="Output MEF (default: config.EXTRACT1D_OHREF)")
    p.add_argument("--clip", dest="clip_nm", type=float, default=1.0, help="Hard clip threshold |shift_nm|<=clip")
    p.add_argument("--no-col", action="store_true", help="Do not add OH_SHIFT_NM column")
    return p.parse_args()


def main():
    args = parse_args()

    st08 = Path(config.ST08_EXTRACT1D)
    st09 = Path(config.ST09_OH_REFINE)
    st09.mkdir(parents=True, exist_ok=True)

    infile = Path(args.infile) if args.infile else Path(getattr(config, "EXTRACT1D_WAV", st08 / "extract1d_optimal_ridge_all_wav.fits"))
    if not infile.exists():
        infile = find_latest(st08, "*all_wav*.fits")
    if infile is None or not infile.exists():
        raise FileNotFoundError("Input MEF not found. Pass --in <file>.")

    csvfile = Path(args.csvfile) if args.csvfile else Path(getattr(config, "OH_SHIFT_CSV", st09 / "oh_shifts.csv"))
    if not csvfile.exists():
        raise FileNotFoundError(f"OH CSV not found: {csvfile}")

    outfile = Path(args.outfile) if args.outfile else Path(getattr(config, "EXTRACT1D_OHREF", st09 / "extract1d_optimal_ridge_all_wav_OHref.fits"))

    print("INFILE =", infile)
    print("CSV    =", csvfile)
    print("OUT    =", outfile)
    print("CLIP_NM=", args.clip_nm)

    shifts_nm, useflag = read_oh_csv(csvfile)
    good_vals = np.array([sh for slit, sh in shifts_nm.items() if np.isfinite(sh) and useflag.get(slit, False) and abs(sh) <= args.clip_nm], float)

    if good_vals.size == 0:
        fallback = 0.0
        med = float("nan")
        sig = float("nan")
        ngood = 0
        print("[WARN] No GOOD shifts after clipping; fallback=0.0 nm")
    else:
        med, sig = robust_stats(good_vals)
        fallback = med
        ngood = int(good_vals.size)
        print(f"[INFO] GOOD shifts after clip: {ngood}  median={med:+.4f} nm  robust_sigma~{sig:.4f} nm")

    out_hdus: List[fits.HDUBase] = []
    with fits.open(infile) as hdul:
        phdr = hdul[0].header.copy()
        out_hdus.append(fits.PrimaryHDU(header=phdr))

        nslits = 0
        n_good = 0
        n_fallback = 0
        n_clipped = 0

        for hdu in hdul[1:]:
            extname = (hdu.name or "").strip().upper()
            if not extname.startswith("SLIT") or hdu.data is None:
                out_hdus.append(hdu.copy())
                continue

            nslits += 1
            tab = hdu.data
            hdr = hdu.header.copy()
            colnames = [c.upper() for c in tab.names]
            if "LAMBDA_NM" not in colnames:
                out_hdus.append(hdu.copy())
                continue

            sh = float(shifts_nm.get(extname, float("nan")))
            ok = bool(useflag.get(extname, False)) and np.isfinite(sh)
            if ok and abs(sh) <= args.clip_nm:
                src = "GOOD"
                sh_app = float(sh)
                n_good += 1
            else:
                src = "FALLBACK"
                sh_app = float(fallback)
                n_fallback += 1
                if ok and np.isfinite(sh) and abs(sh) > args.clip_nm:
                    n_clipped += 1

            lam_new = np.asarray(tab["LAMBDA_NM"], dtype=np.float32) + np.float32(sh_app)
            tab2 = tab.copy()
            tab2["LAMBDA_NM"] = lam_new
            if not args.no_col:
                try:
                    tab2 = add_or_replace_column(tab2, "OH_SHIFT_NM", np.full(lam_new.shape, np.float32(sh_app), dtype=np.float32), fmt="E")
                except Exception:
                    pass

            hdr["OHSHIFT"] = (float(sh_app), "Applied OH wavelength shift (nm)")
            hdr["OHSRC"] = (src, "OH shift source: GOOD/FALLBACK")
            hdr["OHCLIP"] = (float(args.clip_nm), "Hard clip threshold |shift_nm|<=OHCLIP (nm)")
            hdr["OHCSV"] = (csvfile.name, "OH shifts CSV")
            out_hdus.append(fits.BinTableHDU(data=tab2, header=hdr, name=extname))

        out_hdus[0].header["OHREF"] = (True, "Wavelengths corrected using OH SKY shifts")
        out_hdus[0].header["OHCLIP"] = (float(args.clip_nm), "Hard clip threshold for GOOD OH shifts (nm)")
        out_hdus[0].header["OHNGOOD"] = (int(ngood), "Number of GOOD shifts used for fallback stats")
        if np.isfinite(med):
            out_hdus[0].header["OHMED"] = (float(med), "Median GOOD OH shift (nm)")
        if np.isfinite(sig):
            out_hdus[0].header["OHRSIG"] = (float(sig), "Robust sigma of GOOD OH shifts (nm)")
        out_hdus[0].header["OHCSV"] = (csvfile.name, "OH shifts CSV used")
        out_hdus[0].header["OHSRCG"] = (int(n_good), "Number of slits using GOOD OH shifts")
        out_hdus[0].header["OHSRCF"] = (int(n_fallback), "Number of slits using FALLBACK shift")
        out_hdus[0].header["OHSRCC"] = (int(n_clipped), "Number of slits CLIPPED from direct OH shifts")

    fits.HDUList(out_hdus).writeto(outfile, overwrite=True)
    print("Wrote:", outfile)
    print(f"Slits processed: {nslits}  GOOD={n_good}  CLIPPED={n_clipped}  FALLBACK={n_fallback}")


if __name__ == "__main__":
    main()
