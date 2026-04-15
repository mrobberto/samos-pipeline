#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step11b — query SkyMapper photometry for slit positions.

PURPOSE
-------
Retrieve broadband SkyMapper DR4 photometry for each slit position so the
extracted spectra can be anchored to an external photometric scale in
Step11c.

For every slit coordinate, this step performs a local catalog search,
selects the closest SkyMapper source within the search radius, and
records r/i/z PSF magnitudes, uncertainties, and match separation.

INPUT
-----
- slit_trace_radec_all.csv

Required columns:
    slit, RA, DEC

OUTPUT
------
- slit_trace_radec_skymapper_all.csv

Adds columns:
    r_mag, i_mag, z_mag         : SkyMapper PSF magnitudes
    r_err, i_err, z_err         : magnitude uncertainties
    match_sep_arcsec            : angular separation to the adopted match

SCIENTIFIC METHOD
-----------------
For each slit:

1) Query SkyMapper DR4 within a fixed search radius (default: 2 arcsec).

2) If multiple sources are returned, select the nearest one in angular
   separation.

3) Record the matched PSF magnitudes and their uncertainties.

4) Preserve the match separation as a QC quantity for later inspection.

ASSUMPTIONS
-----------
- The nearest SkyMapper source corresponds to the slit target.
- PSF magnitudes are an appropriate photometric anchor for the objects of
  interest, or at least a useful first-order calibration reference.
- The SkyMapper catalog values are in the system intended for Step11c
  (typically treated as AB unless otherwise specified there).

ROBUSTNESS FEATURES
-------------------
- Returns NaN values when no match is found
- Uses a compact candidate list and nearest-neighbor selection
- Records match separation for downstream quality filtering

QUALITY CONTROL
---------------
Small match_sep_arcsec values indicate reliable associations.
Large separations may indicate:
- astrometric mismatch
- crowded-field ambiguity
- a target without a secure catalog counterpart

NOTES
-----
- This step depends on external catalog/network access through
  astroquery/Vizier.
- The output is the photometric anchor table consumed by Step11c.

DOES NOT DO
-----------
- perform morphology filtering
- resolve ambiguous blends beyond nearest-neighbor choice
- convert magnitudes between photometric systems
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

import config


SEARCH_RADIUS = 2.0  # arcsec
SKYMAPPER_CAT = "II/379/smssdr4"


def parse_args():
    ap = argparse.ArgumentParser(description="Query SkyMapper DR4 photometry for slit positions.")
    ap.add_argument(
        "--in",
        dest="infile",
        type=Path,
        default=Path(config.ST11_FLUXCAL) / "slit_trace_radec_all.csv",
        help="Input CSV with slit, RA, DEC columns",
    )
    ap.add_argument(
        "--out",
        dest="outfile",
        type=Path,
        default=Path(config.ST11_FLUXCAL) / "slit_trace_radec_skymapper_all.csv",
        help="Output CSV with SkyMapper photometry appended",
    )
    return ap.parse_args()


def query_skymapper(ra: float, dec: float) -> dict:
    """
    Query SkyMapper around one position and return the closest source.

    Returns:
      r_mag, r_err, i_mag, i_err, z_mag, z_err, match_sep_arcsec
    If no source is found, all values are NaN.
    """
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    viz = Vizier(columns=[
        "RAICRS", "DEICRS",
        "rPSF", "e_rPSF",
        "iPSF", "e_iPSF",
        "zPSF", "e_zPSF",
    ])
    viz.ROW_LIMIT = 5

    result = viz.query_region(coord, radius=SEARCH_RADIUS * u.arcsec, catalog=SKYMAPPER_CAT)

    if len(result) == 0:
        return {
            "r_mag": np.nan, "r_err": np.nan,
            "i_mag": np.nan, "i_err": np.nan,
            "z_mag": np.nan, "z_err": np.nan,
            "match_sep_arcsec": np.nan,
        }

    tab = result[0]
    ccat = SkyCoord(tab["RAICRS"], tab["DEICRS"], unit="deg")
    sep = coord.separation(ccat)
    idx = int(np.argmin(sep))
    row = tab[idx]

    return {
        "r_mag": float(row["rPSF"]) if np.isfinite(row["rPSF"]) else np.nan,
        "r_err": float(row["e_rPSF"]) if np.isfinite(row["e_rPSF"]) else np.nan,
        "i_mag": float(row["iPSF"]) if np.isfinite(row["iPSF"]) else np.nan,
        "i_err": float(row["e_iPSF"]) if np.isfinite(row["e_iPSF"]) else np.nan,
        "z_mag": float(row["zPSF"]) if np.isfinite(row["zPSF"]) else np.nan,
        "z_err": float(row["e_zPSF"]) if np.isfinite(row["e_zPSF"]) else np.nan,
        "match_sep_arcsec": float(sep[idx].arcsec),
    }


def main():
    args = parse_args()

    if not args.infile.exists():
        raise FileNotFoundError(args.infile)

    df = pd.read_csv(args.infile)

    required = {"slit", "RA", "DEC"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Input CSV missing required columns: {sorted(missing)}")

    r_mag, r_err = [], []
    i_mag, i_err = [], []
    z_mag, z_err = [], []
    sep_arcsec = []

    for k, row in df.iterrows():
        ra = float(row["RA"])
        dec = float(row["DEC"])

        phot = query_skymapper(ra, dec)

        r_mag.append(phot["r_mag"])
        r_err.append(phot["r_err"])
        i_mag.append(phot["i_mag"])
        i_err.append(phot["i_err"])
        z_mag.append(phot["z_mag"])
        z_err.append(phot["z_err"])
        sep_arcsec.append(phot["match_sep_arcsec"])

        print(
            f"{k+1}/{len(df)}  {row['slit']}  "
            f"RA={ra:.5f} DEC={dec:.5f}  "
            f"r={phot['r_mag']}  sep={phot['match_sep_arcsec']}"
        )

    df["r_mag"] = r_mag
    df["r_err"] = r_err
    df["i_mag"] = i_mag
    df["i_err"] = i_err
    df["z_mag"] = z_mag
    df["z_err"] = z_err
    df["match_sep_arcsec"] = sep_arcsec

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.outfile, index=False)

    print("Saved:", args.outfile.resolve())


if __name__ == "__main__":
    main()