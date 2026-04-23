#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step11a — Fetch SkyMapper photometry for calibration stars

Inputs:
- RA/DEC CSV files (e.g. regions/radec_even.csv, radec_odd.csv)

Outputs:
- SkyMapper photometry cache:
  config/reference_tables/photometry/skymapper_cache/skymapper_calstars_<tag>.csv

Columns:
- SLITID, RA, DEC, g_mag, r_mag, i_mag, z_mag
- plus quality flags

Notes:
- Uses SkyMapper DR4 cone search
- Applies basic quality cuts
"""

import argparse
import pandas as pd
import numpy as np
import urllib.parse
import time
from pathlib import Path


SKYMAPPER_URL = "https://skymapper.anu.edu.au/sm-cone/public/query"


# --------------------------------------------------
# Query function
# --------------------------------------------------

def query_skymapper(ra_deg, dec_deg, radius_arcsec=1.5):
    params = {
        "RA": ra_deg,
        "DEC": dec_deg,
        "SR": radius_arcsec / 3600.0,
        "RESPONSEFORMAT": "CSV",
        "VERB": 3,
        "CATALOG": "dr4.master",
    }

    url = SKYMAPPER_URL + "?" + urllib.parse.urlencode(params)

    try:
        df = pd.read_csv(url)
    except Exception:
        return None

    if df is None or len(df) == 0:
        return None

    return df


# --------------------------------------------------
# Select best match
# --------------------------------------------------

def select_best_source(df, ra, dec):
    if df is None or len(df) == 0:
        return None

    # distance on sky (approx)
    dra = (df["raj2000"] - ra) * np.cos(np.deg2rad(dec))
    ddec = (df["dej2000"] - dec)
    dist = np.sqrt(dra**2 + ddec**2)

    df = df.copy()
    df["dist"] = dist

    # basic quality cuts (relaxed first)
    mask = (
        (df.get("flags", 0) == 0) &
        (df.get("nimaflags", 0) == 0)
    )

    if np.any(mask):
        df = df[mask]

    if len(df) == 0:
        return None

    # pick closest
    row = df.sort_values("dist").iloc[0]

    return row


# --------------------------------------------------
# Process RA/DEC table
# --------------------------------------------------

def process_catalog(cat_path, slit_col, ra_col, dec_col):
    df = pd.read_csv(cat_path)

    results = []

    for _, row in df.iterrows():
        slit = str(row[slit_col])
        ra = float(row[ra_col])
        dec = float(row[dec_col])

        sm = query_skymapper(ra, dec)

        best = select_best_source(sm, ra, dec)

        if best is None:
            continue

        results.append({
            "SLITID": slit,
            "RA": ra,
            "DEC": dec,
            "g_mag": best.get("g_psf", np.nan),
            "r_mag": best.get("r_psf", np.nan),
            "i_mag": best.get("i_psf", np.nan),
            "z_mag": best.get("z_psf", np.nan),
            "flags": best.get("flags", np.nan),
            "nimaflags": best.get("nimaflags", np.nan),
            "dist_arcsec": best["dist"] * 3600.0,
        })

        # be gentle to server
        time.sleep(0.2)

    return pd.DataFrame(results)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--even", required=True)
    parser.add_argument("--odd", required=True)

    parser.add_argument("--slit-col", default="SLITID")
    parser.add_argument("--ra-col", default="RA")
    parser.add_argument("--dec-col", default="DEC")

    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    df_even = process_catalog(args.even, args.slit_col, args.ra_col, args.dec_col)
    df_odd = process_catalog(args.odd, args.slit_col, args.ra_col, args.dec_col)

    df_all = pd.concat([df_even, df_odd], ignore_index=True)

    # drop duplicates (if any)
    df_all = df_all.drop_duplicates(subset=["SLITID"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(args.out, index=False)

    print(f"Wrote {len(df_all)} calibration stars to {args.out}")


if __name__ == "__main__":
    main()