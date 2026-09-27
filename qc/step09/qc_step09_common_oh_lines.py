#!/usr/bin/env python3
"""
QC-only diagnostic for Step09 residual sky structure.

Starting from a locally sky-subtracted extracted spectrum (default FLUX_APCORR),
identify narrow positive and negative residual peaks in each usable slit.  Build
an empirical list of recurrent positive features across the slit ensemble, then
ask how often significant negative residual peaks coincide with those recurrent
features.

The purpose is to distinguish plausible OH over/under-subtraction from arbitrary
stellar/telluric absorption before implementing a signed residual-sky model.
No FITS data are modified.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return np.nan
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return float(1.4826 * mad) if np.isfinite(mad) and mad > 0 else float(np.nanstd(x))


def cluster_positive(peaks: pd.DataFrame, tol_nm: float) -> pd.DataFrame:
    if len(peaks) == 0:
        return pd.DataFrame(columns=["cluster_id", "lambda_nm", "n_det", "n_slits", "slits"])

    d = peaks.sort_values("lambda_nm").reset_index(drop=True)
    groups: list[list[int]] = []
    current: list[int] = []

    for i, row in d.iterrows():
        lam = float(row["lambda_nm"])
        if not current:
            current = [i]
            continue
        med = float(np.nanmedian(d.loc[current, "lambda_nm"]))
        if abs(lam - med) <= tol_nm:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    if current:
        groups.append(current)

    rows = []
    for j, idx in enumerate(groups):
        g = d.loc[idx]
        slits = sorted(set(g["slit"].astype(str)))
        rows.append({
            "cluster_id": f"C{j:04d}",
            "lambda_nm": float(np.nanmedian(g["lambda_nm"])),
            "n_det": int(len(g)),
            "n_slits": int(len(slits)),
            "slits": ",".join(slits),
        })
    return pd.DataFrame(rows)


def parse_args():
    ap = argparse.ArgumentParser(description="QC common OH-like residual lines in Step09 input")
    ap.add_argument("--infile", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--source-col", default="FLUX_APCORR")
    ap.add_argument("--lam-min", type=float, default=780.0)
    ap.add_argument("--lam-max", type=float, default=950.0)
    ap.add_argument("--median-kernel", type=int, default=101)
    ap.add_argument("--sigma-thresh", type=float, default=3.0)
    ap.add_argument("--cluster-tol-nm", type=float, default=0.25)
    ap.add_argument("--common-min-slits", type=int, default=4)
    ap.add_argument("--match-tol-nm", type=float, default=0.30)
    return ap.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    pos_rows = []
    neg_rows = []
    slit_rows = []

    with fits.open(args.infile) as hdul:
        for hdu in hdul[1:]:
            slit = str(hdu.name or "").strip().upper()
            if not slit.startswith("SLIT") or hdu.data is None:
                continue
            if int(hdu.header.get("S08USE", 0)) != 1:
                continue
            names = {n.upper(): n for n in hdu.columns.names}
            if "LAMBDA_NM" not in names or args.source_col.upper() not in names:
                continue

            lam = np.asarray(hdu.data[names["LAMBDA_NM"]], float)
            flux = np.asarray(hdu.data[names[args.source_col.upper()]], float)
            q = (
                np.isfinite(lam) & np.isfinite(flux)
                & (lam >= args.lam_min) & (lam <= args.lam_max)
            )
            if int(q.sum()) < 100:
                continue

            fill = float(np.nanmedian(flux[q]))
            work = np.array(flux, copy=True)
            work[~np.isfinite(work)] = fill
            cont = median_filter(work, size=args.median_kernel, mode="nearest")
            resid = flux - cont
            sig = robust_sigma(resid[q])
            if not np.isfinite(sig) or sig <= 0:
                continue

            idx = np.where(q)[0]
            z = resid[idx] / sig
            ppos, _ = find_peaks(z, height=args.sigma_thresh, distance=2)
            pneg, _ = find_peaks(-z, height=args.sigma_thresh, distance=2)

            for p in ppos:
                ii = int(idx[p])
                pos_rows.append({"slit": slit, "lambda_nm": float(lam[ii]), "z": float(resid[ii] / sig)})
            for p in pneg:
                ii = int(idx[p])
                neg_rows.append({"slit": slit, "lambda_nm": float(lam[ii]), "z": float(resid[ii] / sig)})

            slit_rows.append({
                "slit": slit,
                "sigma": sig,
                "n_pos_peaks": int(len(ppos)),
                "n_neg_peaks": int(len(pneg)),
            })

    pos = pd.DataFrame(pos_rows, columns=["slit", "lambda_nm", "z"])
    neg = pd.DataFrame(neg_rows, columns=["slit", "lambda_nm", "z"])
    slits = pd.DataFrame(slit_rows)
    clusters = cluster_positive(pos, args.cluster_tol_nm)
    common = clusters[clusters["n_slits"] >= args.common_min_slits].copy()

    centers = common["lambda_nm"].to_numpy(float) if len(common) else np.array([], float)
    if len(neg):
        dist = np.array([
            np.min(np.abs(centers - lam)) if centers.size else np.inf
            for lam in neg["lambda_nm"].to_numpy(float)
        ])
        neg["nearest_common_nm"] = dist
        neg["matches_common"] = dist <= args.match_tol_nm
    else:
        neg["nearest_common_nm"] = []
        neg["matches_common"] = []

    nneg = int(len(neg))
    nmatch = int(neg["matches_common"].sum()) if nneg else 0
    frac = nmatch / nneg if nneg else np.nan

    if len(slits):
        nm = neg.groupby("slit")["matches_common"].agg(["sum", "count"]).reset_index()
        nm = nm.rename(columns={"sum": "n_neg_match_common", "count": "n_neg_total"})
        slits = slits.merge(nm, on="slit", how="left")
        slits[["n_neg_match_common", "n_neg_total"]] = slits[["n_neg_match_common", "n_neg_total"]].fillna(0).astype(int)
        slits["neg_match_frac"] = np.where(slits["n_neg_total"] > 0,
                                             slits["n_neg_match_common"] / slits["n_neg_total"],
                                             np.nan)

    pos_path = args.outdir / "qc_step09_common_positive_peaks.csv"
    neg_path = args.outdir / "qc_step09_negative_peaks.csv"
    line_path = args.outdir / "qc_step09_common_oh_lines.csv"
    slit_path = args.outdir / "qc_step09_common_oh_by_slit.csv"
    pos.to_csv(pos_path, index=False)
    neg.to_csv(neg_path, index=False)
    common.to_csv(line_path, index=False)
    slits.to_csv(slit_path, index=False)

    print(f"Usable slits: {len(slits)}")
    print(f"Positive >={args.sigma_thresh:.1f}sigma peaks: {len(pos)}")
    print(f"Negative >={args.sigma_thresh:.1f}sigma peaks: {len(neg)}")
    print(
        f"Common positive clusters (Nslit>={args.common_min_slits}, "
        f"tol={args.cluster_tol_nm:.2f} nm): {len(common)}"
    )
    print(
        f"Negative peaks within {args.match_tol_nm:.2f} nm of a common positive cluster: "
        f"{nmatch}/{nneg} ({100*frac:.1f}%)" if nneg else "No negative peaks"
    )
    if len(slits):
        print("Largest negative-peak counts:")
        print(slits.sort_values("n_neg_peaks", ascending=False).head(12).to_string(index=False))
    print("Wrote:", line_path)
    print("Wrote:", slit_path)
    print("QC only; no spectra were modified.")


if __name__ == "__main__":
    main()
