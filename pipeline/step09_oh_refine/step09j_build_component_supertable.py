#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 20:20:40 2026

@author: robberto
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step09j — build ABAB OH-component supertable.

Reads per-slit ABAB component CSVs, clusters components by wavelength,
and writes:
  1. all_components table
  2. line_supertable with detection counts and parameter statistics

Expected component columns from step09e:
  LINE_ID, LAM_INIT, LAM_FIT, SIGMA_NM, FLUX_INT,
  WINDOW_LO, WINDOW_HI, ITERATION, PHASE, ACCEPTED, QUALITY_FLAG
"""
#from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd


def norm_slit(text: str) -> str:
    text = str(text).upper()
    m = re.search(r"SLIT\s*0*([0-9]+)", text)
    if m:
        return f"SLIT{int(m.group(1)):03d}"
    return text.strip()


def robust_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if np.isfinite(mad) and mad > 0:
        return 1.4826 * mad
    return np.nanstd(x)


def weighted_median(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return np.nan
    x = x[ok]
    w = w[ok]
    idx = np.argsort(x)
    x = x[idx]
    w = w[idx]
    cw = np.cumsum(w)
    return float(x[np.searchsorted(cw, 0.5 * cw[-1])])


def infer_slit_from_path(path: Path) -> str:
    for part in path.parts[::-1]:
        if "SLIT" in part.upper():
            return norm_slit(part)
    return norm_slit(path.name)


def read_component_tables(root: Path, pattern: str) -> pd.DataFrame:
    files = sorted(root.rglob(pattern))
    rows = []

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        cols = {c.upper(): c for c in df.columns}
        if "LAM_FIT" not in cols:
            continue

        slit = infer_slit_from_path(f)

        d = df.copy()
        d["SLIT"] = slit
        d["SOURCE_CSV"] = str(f)

        # Normalize column names
        rename = {}
        for c in d.columns:
            cu = c.upper()
            if cu in [
                "LINE_ID", "LAM_INIT", "LAM_FIT", "SIGMA_NM", "FLUX_INT",
                "WINDOW_LO", "WINDOW_HI", "ITERATION", "PHASE",
                "ACCEPTED", "QUALITY_FLAG"
            ]:
                rename[c] = cu
        d = d.rename(columns=rename)

        rows.append(d)

    if not rows:
        raise RuntimeError(f"No component tables found under {root} with pattern {pattern}")

    allc = pd.concat(rows, ignore_index=True)

    # Required numeric cleanup
    for c in ["LAM_INIT", "LAM_FIT", "SIGMA_NM", "FLUX_INT", "WINDOW_LO", "WINDOW_HI"]:
        if c in allc.columns:
            allc[c] = pd.to_numeric(allc[c], errors="coerce")

    if "ACCEPTED" not in allc.columns:
        allc["ACCEPTED"] = 1
    allc["ACCEPTED"] = pd.to_numeric(allc["ACCEPTED"], errors="coerce").fillna(0).astype(int)

    if "PHASE" not in allc.columns:
        allc["PHASE"] = "UNKNOWN"
    if "QUALITY_FLAG" not in allc.columns:
        allc["QUALITY_FLAG"] = ""

    allc = allc[np.isfinite(allc["LAM_FIT"])].copy()
    return allc


def cluster_lines(df: pd.DataFrame, tol_nm: float) -> pd.DataFrame:
    """
    Greedy wavelength clustering of accepted components.

    Components sorted by LAM_FIT. A component joins the current cluster if it is
    within tol_nm of the current cluster median. Otherwise a new line is started.
    """
    d = df.sort_values("LAM_FIT").reset_index(drop=True).copy()

    line_ids = []
    current = []
    line_id = 0

    for i, row in d.iterrows():
        lam = float(row["LAM_FIT"])

        if not current:
            current = [lam]
            line_ids.append(line_id)
            continue

        med = float(np.nanmedian(current))
        if abs(lam - med) <= tol_nm:
            current.append(lam)
            line_ids.append(line_id)
        else:
            line_id += 1
            current = [lam]
            line_ids.append(line_id)

    d["GLOBAL_LINE_ID"] = [f"OH{j:04d}" for j in line_ids]
    return d


def summarize_clusters(clustered: pd.DataFrame, n_slits_total: int) -> pd.DataFrame:
    rows = []

    for gid, g in clustered.groupby("GLOBAL_LINE_ID", sort=True):
        lam = np.asarray(g["LAM_FIT"], float)
        sig = np.asarray(g["SIGMA_NM"], float) if "SIGMA_NM" in g else np.full(len(g), np.nan)
        flux = np.asarray(g["FLUX_INT"], float) if "FLUX_INT" in g else np.full(len(g), np.nan)

        # Use flux as weight only if positive and finite
        w = np.where(np.isfinite(flux) & (flux > 0), flux, 1.0)

        slits = sorted(set(g["SLIT"].astype(str)))
        phases = ",".join(sorted(set(g["PHASE"].astype(str))))
        qflags = ",".join(sorted(set(g["QUALITY_FLAG"].astype(str))))

        rows.append(dict(
            GLOBAL_LINE_ID=gid,
            LAMBDA_MED_NM=float(np.nanmedian(lam)),
            LAMBDA_WMED_NM=weighted_median(lam, w),
            LAMBDA_MEAN_NM=float(np.nanmean(lam)),
            LAMBDA_STD_NM=float(np.nanstd(lam)),
            LAMBDA_RSIG_NM=float(robust_sigma(lam)),
            N_DET=int(len(g)),
            N_SLITS=int(len(slits)),
            DET_FRAC=float(len(slits) / max(n_slits_total, 1)),
            SIGMA_MED_NM=float(np.nanmedian(sig)),
            SIGMA_RSIG_NM=float(robust_sigma(sig)),
            FLUX_MED=float(np.nanmedian(flux)),
            FLUX_RSIG=float(robust_sigma(flux)),
            FLUX_SUM=float(np.nansum(flux)),
            PHASES=phases,
            QUALITY_FLAGS=qflags,
            SLITS=",".join(slits),
        ))

    out = pd.DataFrame(rows)
    out = out.sort_values("LAMBDA_MED_NM").reset_index(drop=True)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Build ABAB OH component supertable")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--pattern", default="*components*.csv")
    p.add_argument("--out-prefix", type=Path, required=True)
    p.add_argument("--tol-nm", type=float, default=0.12)
    p.add_argument("--accepted-only", action="store_true", default=True)
    p.add_argument("--include-rejected", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    allc = read_component_tables(args.root, args.pattern)

    if args.include_rejected:
        use = allc.copy()
    else:
        use = allc[allc["ACCEPTED"] == 1].copy()

    n_slits_total = len(set(allc["SLIT"].astype(str)))

    clustered = cluster_lines(use, tol_nm=args.tol_nm)
    supertab = summarize_clusters(clustered, n_slits_total=n_slits_total)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    all_out = args.out_prefix.with_name(args.out_prefix.name + "_all_components.csv")
    clustered_out = args.out_prefix.with_name(args.out_prefix.name + "_clustered_components.csv")
    super_out = args.out_prefix.with_name(args.out_prefix.name + "_line_supertable.csv")

    allc.to_csv(all_out, index=False)
    clustered.to_csv(clustered_out, index=False)
    supertab.to_csv(super_out, index=False)

    print("Input components:", len(allc))
    print("Used components :", len(use))
    print("Total slits     :", n_slits_total)
    print("Global lines    :", len(supertab))
    print("Wrote:", all_out)
    print("Wrote:", clustered_out)
    print("Wrote:", super_out)


if __name__ == "__main__":
    main()
