#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
step11d_rank_calibrators.py

Merge:
- Step11c continuum S/N diagnostics
- Step11d fit summary

and rank the best calibration slits for building the Step11d master response.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config


def parse_args():
    st11 = Path(config.ST11_FLUXCAL)
    p = argparse.ArgumentParser(description="Rank Step11d calibrator slits using continuum S/N + fit summary.")
    p.add_argument(
        "--snr-csv",
        type=Path,
        default=st11 / "Extract1D_fluxcal_continuum_snr.csv",
        help="Continuum S/N CSV from step11c_continuum_snr.py",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=st11 / "Extract1D_fluxcal_step11d_summary.csv",
        help="Step11d summary CSV",
    )
    p.add_argument(
        "--outcsv",
        type=Path,
        default=st11 / "Extract1D_fluxcal_step11d_ranked_calibrators.csv",
        help="Merged/ranked output CSV",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top-ranked slits to print",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.snr_csv.exists():
        raise FileNotFoundError(args.snr_csv)
    if not args.summary_csv.exists():
        raise FileNotFoundError(args.summary_csv)

    args.outcsv.parent.mkdir(parents=True, exist_ok=True)

    snr = pd.read_csv(args.snr_csv)
    summ = pd.read_csv(args.summary_csv)

    # Normalize slit column names
    if "slit" not in snr.columns:
        raise KeyError(f"{args.snr_csv} must contain column 'slit'")
    if "slit_id" not in summ.columns:
        raise KeyError(f"{args.summary_csv} must contain column 'slit_id'")

    snr["slit"] = snr["slit"].astype(str).str.strip().str.upper()
    summ["slit_id"] = summ["slit_id"].astype(str).str.strip().str.upper()

    # Normalize accepted column
    if "accepted" in summ.columns:
        summ["accepted"] = (
            summ["accepted"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map({"TRUE": True, "FALSE": False})
            .fillna(False)
        )

    merged = pd.merge(
        summ,
        snr,
        left_on="slit_id",
        right_on="slit",
        how="left",
        suffixes=("", "_snr"),
    )

    # Ranking score:
    # prioritize accepted slits with high continuum S/N and good conditioning
    # We keep it simple and transparent.
    merged["rank_score"] = (
        np.where(merged["accepted"], 1.0, 0.0) * (
            np.nan_to_num(merged.get("snr_median_bands", np.nan), nan=0.0)
            + 0.5 * np.nan_to_num(merged.get("snr_global", np.nan), nan=0.0)
        )
    )

    ranked = merged.sort_values(
        by=["accepted", "rank_score", "snr_median_bands", "snr_global"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    ranked.to_csv(args.outcsv, index=False)

    print("[OK] Wrote:", args.outcsv)
    print()

    cols_to_show = [
        "slit_id",
        "accepted",
        "reason",
        "n_good_bands",
        "cond",
        "snr_600_720",
        "snr_730_820",
        "snr_830_900",
        "snr_median_bands",
        "snr_global",
        "rank_score",
    ]
    cols_to_show = [c for c in cols_to_show if c in ranked.columns]

    print(f"Top {args.top_n} ranked calibrators:")
    print(ranked[cols_to_show].head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()