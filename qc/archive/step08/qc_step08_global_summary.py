#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step08 QC — global summary for Step08a products.

Reads the cleaned Step08a parity product:
  extract1d_optimal_ridge_{even,odd}.fits
and prints a ranked list of the worst slits.
"""

import argparse
from pathlib import Path
import numpy as np
from astropy.io import fits
import config

def first_existing(paths):
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None

def finite_median(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True, choices=["EVEN", "ODD", "even", "odd"])
ap.add_argument("--file", default=None, help="Optional explicit Step08a FITS")
args = ap.parse_args()

set_tag = args.set.upper()
st08 = Path(config.ST08_EXTRACT1D)

file08 = first_existing([
    Path(args.file) if args.file else None,
    st08 / f"extract1d_optimal_ridge_{set_tag.lower()}.fits",
])

if file08 is None:
    raise FileNotFoundError(f"No Step08a file found for set {set_tag}")

rows = []
with fits.open(file08) as hdul:
    for h in hdul[1:]:
        nm = (h.name or "").upper()
        if not nm.startswith("SLIT") or h.data is None:
            continue
        d = h.data
        names = list(d.names)
        flux = np.asarray(d["FLUX"], float) if "FLUX" in names else None
        apfrac = np.asarray(d["APLOSS_FRAC"], float) if "APLOSS_FRAC" in names else None
        edge = np.asarray(d["EDGEFLAG"], int) if "EDGEFLAG" in names else None
        nsky = np.asarray(d["NSKY"], float) if "NSKY" in names else None
        skysig = np.asarray(d["SKYSIG"], float) if "SKYSIG" in names else None

        valid_frac = float(np.mean(np.isfinite(flux))) if flux is not None and flux.size else np.nan
        edge_frac = float(np.mean(edge > 0)) if edge is not None and edge.size else np.nan
        med_apfrac = finite_median(apfrac) if apfrac is not None else np.nan
        med_nsky = finite_median(nsky) if nsky is not None else np.nan
        med_skysig = finite_median(skysig) if skysig is not None else np.nan

        ridge_mode = h.header.get("RIDGEMOD", h.header.get("RIDGEAUT", "NA"))
        s08bad = h.header.get("S08BAD", 0)
        s08emp = h.header.get("S08EMP", 0)
        nedge1 = h.header.get("NEDGE1", 0)
        nedge2 = h.header.get("NEDGE2", 0)

        score = 0.0
        if np.isfinite(valid_frac):
            score += 3.0 * (1.0 - valid_frac)
        if np.isfinite(edge_frac):
            score += 2.0 * edge_frac
        if np.isfinite(med_apfrac):
            score += max(0.0, 0.9 - med_apfrac)
        score += 2.0 * int(bool(s08bad))
        score += 2.0 * int(bool(s08emp))

        rows.append({
            "SLIT": nm, "RIDGE": ridge_mode, "VALID_FRAC": valid_frac, "EDGE_FRAC": edge_frac,
            "MED_APFRAC": med_apfrac, "MED_NSKY": med_nsky, "MED_SKYSIG": med_skysig,
            "NEDGE1": nedge1, "NEDGE2": nedge2, "S08BAD": s08bad, "S08EMP": s08emp, "SCORE": score,
        })

rows = sorted(rows, key=lambda r: r["SCORE"], reverse=True)

print()
print("====================================================")
print(f"QC08 global summary — {set_tag}")
print("====================================================")
print("File:", file08)
print(f"N slits: {len(rows)}")
print("====================================================")
print("Worst slits:")
print("SLIT      RIDGE    VALID   EDGE   APFRAC   NSKY   SKYSIG   NEDGE1 NEDGE2 BAD EMP")
for r in rows[:15]:
    print(
        f"{r['SLIT']:8s}  "
        f"{str(r['RIDGE']):7s}  "
        f"{r['VALID_FRAC']:5.2f}  "
        f"{r['EDGE_FRAC']:5.2f}  "
        f"{r['MED_APFRAC']:7.3f}  "
        f"{r['MED_NSKY']:6.1f}  "
        f"{r['MED_SKYSIG']:7.3f}  "
        f"{int(r['NEDGE1']):6d} {int(r['NEDGE2']):6d} "
        f"{int(bool(r['S08BAD'])):3d} {int(bool(r['S08EMP'])):3d}"
    )
