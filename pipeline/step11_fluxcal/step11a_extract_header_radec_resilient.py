#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step11a — build slit RA/DEC table for flux calibration.

PURPOSE
-------
Create the authoritative slit coordinate table used by Step11b.

Unlike older versions, this script does not assume that the Step10 science MEF
always carries RA/DEC in every slit header. It uses a fallback chain:

1) RA/DEC from the input Step10 MEF slit headers
2) RA/DEC from Step04 geometry FITS slit headers
3) RA/DEC from the original radec_Even.csv / radec_Odd.csv tables, matched by
   slit parity and slit number ordering

INPUTS
------
- Step10 science MEF
- Step04 geometry FITS:
    Even_traces_geometry.fits
    Odd_traces_geometry.fits
- RA/DEC tables:
    radec_Even.csv
    radec_Odd.csv

OUTPUT
------
- slit_trace_radec_all.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "config"))

import numpy as np
import pandas as pd
from astropy.io import fits
import config


def norm_slit(name: str) -> str:
    s = str(name).strip().upper()
    m = re.search(r"(SLIT\d+)", s)
    return m.group(1) if m else s


def slitnum(name: str) -> int:
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else -1


def finite_float_or_nan(value):
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def load_radec_table(path: Path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")

        fieldmap = {k.strip().lower(): k for k in reader.fieldnames}

        def pick(*cands):
            for c in cands:
                if c in fieldmap:
                    return fieldmap[c]
            return None

        ra_key = pick("ra", "ra_deg", "radeg", "ra_hours", "rah")
        dec_key = pick("dec", "dec_deg", "decdeg", "decd")
        sid_key = pick("label", "slit", "slit_id", "slitid", "id")
        idx_key = pick("index", "idx", "targidx", "target_index", "targetid")

        if ra_key is None or dec_key is None:
            raise ValueError(f"{path} must contain RA/Dec columns. Found: {reader.fieldnames}")

        for r in reader:
            ra = (r.get(ra_key, "") or "").strip()
            dec = (r.get(dec_key, "") or "").strip()
            sid = None
            if sid_key is not None:
                raw = (r.get(sid_key, "") or "").strip()
                if raw != "":
                    try:
                        sid = int(float(raw))
                    except Exception:
                        sid = None
            idx = None
            if idx_key is not None:
                raw = (r.get(idx_key, "") or "").strip()
                if raw != "":
                    try:
                        idx = int(float(raw))
                    except Exception:
                        idx = None
            rows.append({"sid": sid, "idx": idx, "ra": ra, "dec": dec})
    return rows


def build_radec_lookup_from_csv() -> dict[str, tuple[float, float]]:
    out = {}
    for trace_set, fname in [("EVEN", "radec_Even.csv"), ("ODD", "radec_Odd.csv")]:
        path = Path(config.REGIONS) / fname
        if not path.exists():
            continue
        rows = load_radec_table(path)

        explicit = {int(r["sid"]): r for r in rows if r.get("sid") is not None}
        if explicit:
            for sid, r in explicit.items():
                out[f"SLIT{sid:03d}"] = (float(r["ra"]), float(r["dec"]))
            continue

        start = 0 if trace_set == "EVEN" else 1
        for j, r in enumerate(rows):
            sid = start + 2 * j
            out[f"SLIT{sid:03d}"] = (float(r["ra"]), float(r["dec"]))
    return out


def build_geom_lookup(path: Path) -> dict[str, tuple[float, float, float, float, float, float]]:
    out = {}
    if not path.exists():
        return out
    with fits.open(path) as h:
        for ext in h[1:]:
            slit = norm_slit(ext.name)
            if not slit.startswith("SLIT"):
                continue
            hdr = ext.header
            ra = hdr.get("RA")
            dec = hdr.get("DEC")
            if ra is None or dec is None:
                continue
            try:
                out[slit] = (
                    float(ra),
                    float(dec),
                    finite_float_or_nan(hdr.get("XREF")),
                    finite_float_or_nan(hdr.get("XLO")),
                    finite_float_or_nan(hdr.get("XHI")),
                    finite_float_or_nan(hdr.get("YMIN")),
                )
            except Exception:
                continue
    return out

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--infile",
        type=Path,
        default=Path(getattr(config, "EXTRACT1D_TELLCOR", Path(config.ST10_TELLURIC) / "extract1d_optimal_ridge_all_wav_ohclean_tellcorr.fits")),
        help="Input Step10 telluric-corrected MEF",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(config.ST11_FLUXCAL) / "slit_trace_radec_all.csv",
        help="Output CSV",
    )
    ap.add_argument(
        "--even-geom",
        type=Path,
        default=Path(config.ST04_PIXFLAT) / "Even_traces_geometry.fits",
    )
    ap.add_argument(
        "--odd-geom",
        type=Path,
        default=Path(config.ST04_PIXFLAT) / "Odd_traces_geometry.fits",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.infile.exists():
        raise FileNotFoundError(args.infile)

    geom_lookup = {}
    geom_lookup.update(build_geom_lookup(args.even_geom))
    geom_lookup.update(build_geom_lookup(args.odd_geom))
    csv_lookup = build_radec_lookup_from_csv()

    rows = []
    n_total = 0
    n_from_hdr = 0
    n_from_geom = 0
    n_from_csv = 0
    n_missing = 0

    with fits.open(args.infile) as h:
        for ext in h[1:]:
            slit = norm_slit(ext.name)
            if not slit.startswith("SLIT"):
                continue
            n_total += 1
            hdr = ext.header

            ra = hdr.get("RA")
            dec = hdr.get("DEC")
            src = None
            gxref = gxlo = gxhi = gymin = np.nan

            if ra is not None and dec is not None:
                try:
                    ra = float(ra)
                    dec = float(dec)
                    src = "HEADER"
                except Exception:
                    ra = dec = None

            if src is None and slit in geom_lookup:
                ra, dec, gxref, gxlo, gxhi, gymin = geom_lookup[slit]
                src = "GEOMETRY"

            if src is None and slit in csv_lookup:
                ra, dec = csv_lookup[slit]
                src = "CSV"

            if src is None:
                n_missing += 1
                continue

            x_center = hdr.get("XREF", hdr.get("X0", np.nan))
            if not np.isfinite(finite_float_or_nan(x_center)) and ext.data is not None:
                try:
                    cols = list(ext.data.columns.names)
                except Exception:
                    cols = []
                if "X0" in cols:
                    arr = np.asarray(ext.data["X0"], float)
                    if np.isfinite(arr).any():
                        x_center = float(np.nanmedian(arr))

            if not np.isfinite(finite_float_or_nan(x_center)):
                x_center = gxref

            xlo = hdr.get("XLO", gxlo)
            xhi = hdr.get("XHI", gxhi)
            ymin = hdr.get("YMIN", gymin)

            rows.append({
                "slit": slit,
                "SLITNUM": slitnum(slit),
                "RA": finite_float_or_nan(ra),
                "DEC": finite_float_or_nan(dec),
                "x_center": finite_float_or_nan(x_center),
                "xlo": finite_float_or_nan(xlo),
                "xhi": finite_float_or_nan(xhi),
                "YMIN": finite_float_or_nan(ymin),
                "SRC": src,
            })

            if src == "HEADER":
                n_from_hdr += 1
            elif src == "GEOMETRY":
                n_from_geom += 1
            elif src == "CSV":
                n_from_csv += 1

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"No slit coordinates found in {args.infile}. "
            f"Checked {n_total} slit extensions; none had usable header, geometry, or CSV coordinates."
        )

    df = df.sort_values("SLITNUM").reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("[OK] Wrote:", args.out)
    print(f"Rows kept: {len(df)} / {n_total}")
    print(f"Recovered from headers : {n_from_hdr}")
    print(f"Recovered from geometry: {n_from_geom}")
    print(f"Recovered from CSV     : {n_from_csv}")
    print(f"Missing                : {n_missing}")


if __name__ == "__main__":
    main()
