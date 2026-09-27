#!/usr/bin/env python3
"""
QC for the Step09 empirical residual-SKY candidate.

Checks:
- wavelength vectors are unchanged;
- FLUX_APCORR is unchanged;
- STELLAR_CONSENSUS = FLUX_APCORR - SKYRES_MODEL;
- SKYRES_MODEL is zero wherever SKYRES_FLAG == 0;
- rejected windows remain unchanged;
- summarizes the magnitude of the applied correction.

No data are modified.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


def parse_args():
    p = argparse.ArgumentParser(description="Validate empirical residual-SKY candidate")
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--summary-csv", type=Path, required=True)
    return p.parse_args()


def finite_max_abs(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.max(np.abs(x))) if x.size else 0.0


def main():
    args = parse_args()
    summary = pd.read_csv(args.summary_csv)
    accepted = summary[summary["accepted"] == True].copy()

    rows = []

    with fits.open(args.before) as hb, fits.open(args.after) as ha:
        before_names = [h.name for h in hb[1:] if str(h.name).upper().startswith("SLIT")]
        after_names = [h.name for h in ha[1:] if str(h.name).upper().startswith("SLIT")]
        common = sorted(set(before_names) & set(after_names))

        for slit in common:
            b = hb[slit].data
            a = ha[slit].data

            lam_b = np.asarray(b["LAMBDA_NM"], float)
            lam_a = np.asarray(a["LAMBDA_NM"], float)
            src_b = np.asarray(b["FLUX_APCORR"], float)
            src_a = np.asarray(a["FLUX_APCORR"], float)
            stellar = np.asarray(a["STELLAR_CONSENSUS"], float)
            model = np.asarray(a["SKYRES_MODEL"], float)
            flag = np.asarray(a["SKYRES_FLAG"], int)

            inv_lam = finite_max_abs(lam_a - lam_b)
            inv_src = finite_max_abs(src_a - src_b)
            identity = finite_max_abs(stellar - (src_a - model))
            model_outside = finite_max_abs(model[flag == 0])

            q = (
                np.isfinite(src_a)
                & np.isfinite(stellar)
                & np.isfinite(model)
                & (flag > 0)
            )
            if q.any():
                denom = np.nanmedian(np.abs(src_a[q]))
                med_frac = (
                    float(np.nanmedian(np.abs(model[q])) / denom)
                    if np.isfinite(denom) and denom > 0
                    else np.nan
                )
                p95_frac = (
                    float(np.nanpercentile(np.abs(model[q]), 95) / denom)
                    if np.isfinite(denom) and denom > 0
                    else np.nan
                )
            else:
                med_frac = np.nan
                p95_frac = np.nan

            rows.append(
                dict(
                    slit=slit,
                    n_model_pix=int(q.sum()),
                    max_dlambda_nm=inv_lam,
                    max_dflux_apcorr=inv_src,
                    max_identity_error=identity,
                    max_model_where_flag0=model_outside,
                    median_model_frac=med_frac,
                    p95_model_frac=p95_frac,
                )
            )

    d = pd.DataFrame(rows)

    print("Common slit extensions:", len(d))
    print("Accepted slit/windows:", len(accepted))
    print("Accepted unique slits:", accepted["slit"].nunique() if len(accepted) else 0)
    print(
        "max |delta LAMBDA_NM| =",
        d["max_dlambda_nm"].max() if len(d) else np.nan,
    )
    print(
        "max |delta FLUX_APCORR| =",
        d["max_dflux_apcorr"].max() if len(d) else np.nan,
    )
    print(
        "max identity error STELLAR-(FLUX_APCORR-SKYRES_MODEL) =",
        d["max_identity_error"].max() if len(d) else np.nan,
    )
    print(
        "max |SKYRES_MODEL| where SKYRES_FLAG=0 =",
        d["max_model_where_flag0"].max() if len(d) else np.nan,
    )

    changed = d[d["n_model_pix"] > 0]
    print("Slits with modeled pixels:", len(changed))
    if len(changed):
        print(
            "median modeled-pixel |model|/median|source| = %.4f"
            % np.nanmedian(changed["median_model_frac"])
        )
        print(
            "median slit p95 |model|/median|source| = %.4f"
            % np.nanmedian(changed["p95_model_frac"])
        )
        print()
        print("Largest p95 fractional corrections:")
        print(
            changed.sort_values("p95_model_frac", ascending=False)
            .head(12)
            .to_string(index=False)
        )

    bad_identity = (
        (d["max_dlambda_nm"] > 0)
        | (d["max_dflux_apcorr"] > 0)
        | (d["max_identity_error"] > 1e-6)
        | (d["max_model_where_flag0"] > 0)
    )
    print()
    print("QC STATUS:", "PASS" if not bad_identity.any() else "FAIL")


if __name__ == "__main__":
    main()
