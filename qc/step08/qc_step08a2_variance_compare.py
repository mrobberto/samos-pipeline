#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC Step08a2 — compare a candidate extraction against the frozen reference.

Purpose
-------
Validate the Step08a2 count-rate variance patch without overwriting the
reference reduction. The script checks that geometry/background products are
unchanged and quantifies the change in the optimally extracted FLUX caused only
by the corrected inverse-variance weights.

Inputs
------
--reference  Existing/frozen Step08a2 FITS product.
--candidate  New Step08a2 FITS product generated with the variance patch.

Outputs
-------
- qc_step08a2_variance_compare.csv
- qc_step08a2_variance_compare.png
- terminal summary
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits


def slit_num(name: str) -> int:
    m = re.match(r"SLIT(\d+)", str(name).upper())
    return int(m.group(1)) if m else 10**9


def max_abs_diff(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return np.inf
    if a.dtype.kind in "iu" and b.dtype.kind in "iu":
        return float(np.max(np.abs(a.astype(np.int64) - b.astype(np.int64)))) if a.size else 0.0
    aa = np.asarray(a, float)
    bb = np.asarray(b, float)
    q = np.isfinite(aa) & np.isfinite(bb)
    mismatch_nan = np.any(np.isfinite(aa) != np.isfinite(bb))
    if mismatch_nan:
        return np.inf
    return float(np.nanmax(np.abs(aa[q] - bb[q]))) if np.any(q) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="QC corrected Step08a2 variance weighting")
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    outdir = args.outdir or args.candidate.parent / "qc_step08"
    outdir.mkdir(parents=True, exist_ok=True)

    invariant_cols = [
        "YPIX", "OBJ_PRESKY", "SKY", "X0", "NOBJ", "NSKY", "SKYSIG",
        "APLOSS_FRAC", "EDGEFLAG", "TRXLEFT", "TRXRIGHT",
    ]

    rows = []
    with fits.open(args.reference, memmap=False) as href, fits.open(args.candidate, memmap=False) as hnew:
        print("REFERENCE:", args.reference)
        print("CANDIDATE:", args.candidate)
        print(
            "candidate variance metadata:",
            "VARMOD=", hnew[0].header.get("VARMOD"),
            "VARUNIT=", hnew[0].header.get("VARUNIT"),
            "TOTEXP=", hnew[0].header.get("TOTEXP"),
            "NCOMBINE=", hnew[0].header.get("NCOMBINE"),
        )

        ref_slits = {h.name.upper() for h in href[1:] if h.name.upper().startswith("SLIT")}
        new_slits = {h.name.upper() for h in hnew[1:] if h.name.upper().startswith("SLIT")}
        common = sorted(ref_slits & new_slits, key=slit_num)
        if ref_slits != new_slits:
            print("WARNING: slit sets differ")
            print("  only reference:", sorted(ref_slits - new_slits, key=slit_num))
            print("  only candidate:", sorted(new_slits - ref_slits, key=slit_num))

        for slit in common:
            r = href[slit].data
            n = hnew[slit].data
            if r is None or n is None:
                continue

            diffs = {}
            for col in invariant_cols:
                if col in r.names and col in n.names:
                    diffs[col] = max_abs_diff(r[col], n[col])

            fr = np.asarray(r["FLUX"], float)
            fn = np.asarray(n["FLUX"], float)
            vn = np.asarray(n["VAR"], float)

            q = np.isfinite(fr) & np.isfinite(fn) & (np.abs(fr) > 0)
            if np.sum(q) >= 20:
                d_pct = 100.0 * (fn[q] / fr[q] - 1.0)
                med_signed = float(np.nanmedian(d_pct))
                med_abs = float(np.nanmedian(np.abs(d_pct)))
                p95_abs = float(np.nanpercentile(np.abs(d_pct), 95))
            else:
                med_signed = med_abs = p95_abs = np.nan

            qv = np.isfinite(vn) & (vn > 0) & np.isfinite(fn)
            med_snr = float(np.nanmedian(np.abs(fn[qv]) / np.sqrt(vn[qv]))) if np.any(qv) else np.nan

            rows.append(dict(
                slit=slit,
                n_flux_compare=int(np.sum(q)),
                median_dflux_pct=med_signed,
                median_abs_dflux_pct=med_abs,
                p95_abs_dflux_pct=p95_abs,
                n_positive_var=int(np.sum(qv)),
                median_candidate_snr=med_snr,
                max_invariant_diff=max(diffs.values()) if diffs else np.nan,
                **{f"diff_{k}": v for k, v in diffs.items()},
            ))

    df = pd.DataFrame(rows).sort_values("slit")
    csv = outdir / "qc_step08a2_variance_compare.csv"
    df.to_csv(csv, index=False)

    usable = df[np.isfinite(df["median_abs_dflux_pct"])]
    print("---")
    print("N common slits:", len(df))
    print("N flux-comparable slits:", len(usable))
    if len(usable):
        print("median of slit median |dF| = %.3f %%" % np.nanmedian(usable["median_abs_dflux_pct"]))
        print("median of slit p95 |dF|   = %.3f %%" % np.nanmedian(usable["p95_abs_dflux_pct"]))
        iw = usable["median_abs_dflux_pct"].idxmax()
        print("worst slit median |dF|    = %s %.3f %%" % (usable.loc[iw, "slit"], usable.loc[iw, "median_abs_dflux_pct"]))
    finite_inv = df["max_invariant_diff"].replace([np.inf, -np.inf], np.nan)
    print("max finite invariant-column difference =", np.nanmax(finite_inv) if np.any(np.isfinite(finite_inv)) else np.nan)
    if np.any(np.isinf(df["max_invariant_diff"])):
        print("WARNING: at least one invariant column has a shape/finite-mask mismatch")
    print("Wrote:", csv)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(df))
    ax.plot(x, df["median_abs_dflux_pct"], marker="o", ms=3, lw=1, label="median |dF| per slit")
    ax.plot(x, df["p95_abs_dflux_pct"], marker=".", ms=3, lw=1, label="p95 |dF| per slit")
    ax.set_xticks(x)
    ax.set_xticklabels(df["slit"], rotation=90, fontsize=7)
    ax.set_ylabel("Flux change (%)")
    ax.set_xlabel("Slit")
    ax.set_title("Step08a2 variance-weighting patch: reference vs candidate")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = outdir / "qc_step08a2_variance_compare.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)
    print("Wrote:", png)


if __name__ == "__main__":
    main()
