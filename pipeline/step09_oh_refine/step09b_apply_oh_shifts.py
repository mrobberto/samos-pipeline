#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step09b — apply accepted ensemble-relative OH wavelength corrections.

Only shifts marked ``use=True`` by the ensemble Step09a estimator are applied.
Rejected, unavailable, or clipped shifts leave the Step07/08c wavelength vector
unchanged.  No median fallback correction is applied.

This preserves the absolute/global arc-lamp calibration while removing only
well-measured differential slit-to-slit residuals.
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


def read_oh_csv(csv_path: Path) -> Tuple[Dict[str, float], Dict[str, bool], dict]:
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
        method_col = pick("method")
        center_col = pick("centerremovednm", "center_removed_nm")
        if slit_col is None or shift_col is None:
            raise RuntimeError(f"CSV missing required columns. Found: {cols}")

        shifts: Dict[str, float] = {}
        useflag: Dict[str, bool] = {}
        methods = set()
        centers = []

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

            if method_col is not None:
                m = (row.get(method_col, "") or "").strip().upper()
                if m:
                    methods.add(m)
            if center_col is not None:
                try:
                    c = float(row.get(center_col, "nan"))
                except Exception:
                    c = float("nan")
                if np.isfinite(c):
                    centers.append(c)

    method = next(iter(methods)) if len(methods) == 1 else ""
    center = float(np.nanmedian(centers)) if centers else np.nan
    return shifts, useflag, {"method": method, "center_removed_nm": center}


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
    p.add_argument(
        "--allow-legacy-csv",
        action="store_true",
        help="Allow a CSV not tagged METHOD=ENSEMBLE_REL (unsafe for normal production use)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    st08 = Path(config.ST08_EXTRACT1D)
    st09 = Path(config.ST09_OH_REFINE)
    st09.mkdir(parents=True, exist_ok=True)

    infile = Path(args.infile) if args.infile else Path(
        getattr(config, "EXTRACT1D_WAV", st08 / "extract1d_optimal_ridge_all_wav.fits")
    )
    if not infile.exists():
        infile = find_latest(st08, "*all_wav*.fits")
    if infile is None or not infile.exists():
        raise FileNotFoundError("Input MEF not found. Pass --in <file>.")

    csvfile = Path(args.csvfile) if args.csvfile else Path(
        getattr(config, "OH_SHIFT_CSV", st09 / "oh_shifts.csv")
    )
    if not csvfile.exists():
        raise FileNotFoundError(f"OH CSV not found: {csvfile}")

    outfile = Path(args.outfile) if args.outfile else Path(
        getattr(config, "EXTRACT1D_OHREF", st09 / "extract1d_optimal_ridge_all_wav_OHref.fits")
    )

    print("INFILE =", infile)
    print("CSV    =", csvfile)
    print("OUT    =", outfile)
    print("CLIP_NM=", args.clip_nm)

    shifts_nm, useflag, meta = read_oh_csv(csvfile)
    method = str(meta.get("method", "")).upper()
    center_removed = float(meta.get("center_removed_nm", np.nan))

    if method != "ENSEMBLE_REL" and not args.allow_legacy_csv:
        raise RuntimeError(
            "OH CSV is not tagged METHOD=ENSEMBLE_REL. Refusing to apply legacy/single-reference shifts. "
            "Re-run Step09a or pass --allow-legacy-csv explicitly for a historical test."
        )

    good_vals = np.array([
        sh for slit, sh in shifts_nm.items()
        if np.isfinite(sh) and useflag.get(slit, False) and abs(sh) <= args.clip_nm
    ], float)

    med, sig = robust_stats(good_vals)
    ngood = int(good_vals.size)
    if ngood:
        print(f"[INFO] Accepted direct shifts after clip: {ngood}  median={med:+.4f} nm  robust_sigma~{sig:.4f} nm")
    else:
        print("[WARN] No accepted OH shifts after clipping; all wavelength vectors will remain unchanged")

    out_hdus: List[fits.HDUBase] = []
    with fits.open(infile) as hdul:
        phdr = hdul[0].header.copy()
        out_hdus.append(fits.PrimaryHDU(header=phdr))

        nslits = 0
        n_good = 0
        n_unchanged = 0
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
            requested = bool(useflag.get(extname, False)) and np.isfinite(sh)

            if requested and abs(sh) <= args.clip_nm:
                src = "ENSEMBLE"
                sh_app = float(sh)
                n_good += 1
            else:
                sh_app = 0.0
                if requested and abs(sh) > args.clip_nm:
                    src = "CLIPPED"
                    n_clipped += 1
                else:
                    src = "UNCHANGED"
                n_unchanged += 1

            lam_new = np.asarray(tab["LAMBDA_NM"], dtype=np.float32) + np.float32(sh_app)
            tab2 = tab.copy()
            tab2["LAMBDA_NM"] = lam_new
            if not args.no_col:
                try:
                    tab2 = add_or_replace_column(
                        tab2,
                        "OH_SHIFT_NM",
                        np.full(lam_new.shape, np.float32(sh_app), dtype=np.float32),
                        fmt="E",
                    )
                except Exception:
                    pass

            hdr["OHSHIFT"] = (float(sh_app), "Applied ensemble-relative OH shift (nm)")
            hdr["OHSRC"] = (src, "OH shift source: ENSEMBLE/UNCHANGED/CLIPPED")
            hdr["OHCLIP"] = (float(args.clip_nm), "Hard clip threshold |shift_nm|<=OHCLIP (nm)")
            hdr["OHCSV"] = (csvfile.name, "OH shifts CSV")
            hdr["OHMETH"] = (method or "LEGACY", "OH wavelength-refinement method")
            out_hdus.append(fits.BinTableHDU(data=tab2, header=hdr, name=extname))

        out_hdus[0].header["OHREF"] = (bool(n_good > 0), "Differential wavelengths refined using OH SKY shifts")
        out_hdus[0].header["OHMETH"] = (method or "LEGACY", "OH wavelength-refinement method")
        out_hdus[0].header["OHABS"] = (False, "OH correction is differential, not an absolute zero point")
        out_hdus[0].header["OHCLIP"] = (float(args.clip_nm), "Hard clip threshold for accepted OH shifts (nm)")
        out_hdus[0].header["OHNGOOD"] = (int(ngood), "Number of accepted direct OH shifts after clip")
        if np.isfinite(med):
            out_hdus[0].header["OHMED"] = (float(med), "Median accepted ensemble-relative OH shift (nm)")
        if np.isfinite(sig):
            out_hdus[0].header["OHRSIG"] = (float(sig), "Robust sigma of accepted ensemble-relative shifts (nm)")
        if np.isfinite(center_removed):
            out_hdus[0].header["OHCTR"] = (float(center_removed), "Global ensemble center removed in Step09a (nm)")
        out_hdus[0].header["OHCSV"] = (csvfile.name, "OH shifts CSV used")
        out_hdus[0].header["OHSRCG"] = (int(n_good), "Number of slits using accepted ensemble OH shifts")
        out_hdus[0].header["OHSRCU"] = (int(n_unchanged), "Number of slits left unchanged")
        out_hdus[0].header["OHSRCC"] = (int(n_clipped), "Number of accepted shifts clipped by hard threshold")
        out_hdus[0].header["OHSRCF"] = (0, "Median fallback shifts applied (disabled by design)")

    fits.HDUList(out_hdus).writeto(outfile, overwrite=True)
    print("Wrote:", outfile)
    print(f"Slits processed: {nslits}  ENSEMBLE={n_good}  CLIPPED={n_clipped}  UNCHANGED={n_unchanged}")
    print("Median fallback: DISABLED")


if __name__ == "__main__":
    main()
