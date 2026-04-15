#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step11a — extract slit RA/Dec and geometry metadata.

PURPOSE
-------
Extract per-slit sky coordinates and basic geometric metadata from the
Step10 output MEF. These are used to associate each slit with external
photometric catalogs (Step11b) and for downstream flux calibration.

INPUT
-----
- Step10 product:
    extract1d_optimal_ridge_all_wav_tellcorr_OHref.fits

Expected per-slit header keywords:
- RA, DEC   : sky coordinates of the slit center (degrees)

Optional header metadata (copied when present):
- XREF      : reference X position of slit center (pixel)
- XLO, XHI  : slit boundaries in X (pixel)
- YMIN      : minimum detector Y (pixel)

NOTES
-----
- The extraction centroid X0 is not required in the header.
  In Step08 it is usually stored as a table column, not a header keyword.

OUTPUT
------
- slit_trace_radec_all.csv

Columns:
- slit      : SLIT### identifier
- SLITNUM   : numeric slit ID
- RA, DEC   : sky coordinates (deg)
- x_center  : XREF if available, otherwise X0 header, otherwise median table X0
- xlo, xhi  : slit bounds if available
- YMIN      : detector Y minimum if available

ROBUSTNESS
----------
- Slits without valid RA/DEC are skipped
- Missing optional metadata is filled with NaN
- Output is sorted by slit number
- Default input/output paths are explicit and do not depend on optional config aliases
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--infile",
        type=Path,
        default=Path(config.ST10_OH) / "extract1d_optimal_ridge_all_wav_tellcorr_OHref.fits",
        help="Input MEF (Step10 output)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(config.ST11_FLUXCAL) / "slit_trace_radec_all.csv",
        help="Output CSV",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    if not args.infile.exists():
        raise FileNotFoundError(args.infile)

    rows = []
    n_total = 0
    n_missing_radec = 0

    with fits.open(args.infile) as h:
        for ext in h[1:]:
            slit = norm_slit(ext.name)
            if not slit.startswith("SLIT"):
                continue

            n_total += 1
            hdr = ext.header
            ra = hdr.get("RA")
            dec = hdr.get("DEC")

            if ra is None or dec is None:
                n_missing_radec += 1
                continue

            x_center = hdr.get("XREF", hdr.get("X0", np.nan))
            if (x_center is None or not np.isfinite(finite_float_or_nan(x_center))) and ext.data is not None:
                try:
                    cols = list(ext.data.columns.names)
                except Exception:
                    cols = []
                if "X0" in cols:
                    arr = np.asarray(ext.data["X0"], float)
                    if np.isfinite(arr).any():
                        x_center = float(np.nanmedian(arr))
                    else:
                        x_center = np.nan

            rows.append({
                "slit": slit,
                "SLITNUM": slitnum(slit),
                "RA": finite_float_or_nan(ra),
                "DEC": finite_float_or_nan(dec),
                "x_center": finite_float_or_nan(x_center),
                "xlo": finite_float_or_nan(hdr.get("XLO")),
                "xhi": finite_float_or_nan(hdr.get("XHI")),
                "YMIN": finite_float_or_nan(hdr.get("YMIN")),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            f"No valid slit RA/DEC found in {args.infile}. "
            f"Checked {n_total} slit extensions; {n_missing_radec} lacked RA/DEC."
        )

    df = df.sort_values("SLITNUM").reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("[OK] Wrote:", args.out)
    print("Rows kept:", len(df), f"out of {n_total} slit extensions")
    print("Missing RA/DEC:", n_missing_radec)
    print("First/last slit:", df["slit"].iloc[0], df["slit"].iloc[-1])
    print("RA range:", df["RA"].min(), "to", df["RA"].max())
    print("DEC range:", df["DEC"].min(), "to", df["DEC"].max())


if __name__ == "__main__":
    main()
