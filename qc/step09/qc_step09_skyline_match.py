#!/usr/bin/env python3
"""
QC-only diagnostic: compare positive residual features in FLUX_APCORR with
features actually present in the extracted local-sky spectra.

The script independently detects narrow positive peaks in:
  1) FLUX_APCORR (or --source-col), after broad median-filter continuum removal
  2) SKY, after the same broad filtering

It clusters detections by wavelength across slits and asks whether recurrent
positive residual features in the science spectra coincide with recurrent
features in the measured sky.  This provides a direct empirical test of whether
the Step09 residual-line cleanup should be constrained to observed sky lines.

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
    if np.isfinite(mad) and mad > 0:
        return float(1.4826 * mad)
    return float(np.nanstd(x))


def detect_positive_peaks(lam, y, q, kernel, thresh, slit):
    fill = float(np.nanmedian(y[q]))
    work = np.asarray(y, float).copy()
    work[~np.isfinite(work)] = fill
    cont = median_filter(work, size=kernel, mode="nearest")
    resid = np.asarray(y, float) - cont
    sig = robust_sigma(resid[q])
    if not np.isfinite(sig) or sig <= 0:
        return [], sig
    idx = np.where(q)[0]
    z = resid[idx] / sig
    pp, _ = find_peaks(z, height=thresh, distance=2)
    rows = []
    for p in pp:
        ii = int(idx[p])
        rows.append({
            "slit": slit,
            "lambda_nm": float(lam[ii]),
            "z": float(resid[ii] / sig),
        })
    return rows, sig


def cluster_peaks(peaks: pd.DataFrame, tol_nm: float) -> pd.DataFrame:
    cols = ["cluster_id", "lambda_nm", "n_det", "n_slits", "slits"]
    if len(peaks) == 0:
        return pd.DataFrame(columns=cols)

    d = peaks.sort_values("lambda_nm").reset_index(drop=True)
    groups = []
    current = []

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

    out = []
    for j, idx in enumerate(groups):
        g = d.loc[idx]
        ss = sorted(set(g["slit"].astype(str)))
        out.append({
            "cluster_id": f"C{j:04d}",
            "lambda_nm": float(np.nanmedian(g["lambda_nm"])),
            "n_det": int(len(g)),
            "n_slits": int(len(ss)),
            "slits": ",".join(ss),
        })
    return pd.DataFrame(out, columns=cols)


def nearest_distance(values, centers):
    if len(centers) == 0:
        return np.full(len(values), np.inf, float)
    c = np.asarray(centers, float)
    return np.asarray([np.min(np.abs(c - float(v))) for v in values], float)


def parse_args():
    ap = argparse.ArgumentParser(description="QC match science residual peaks to measured sky lines")
    ap.add_argument("--infile", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--source-col", default="FLUX_APCORR")
    ap.add_argument("--sky-col", default="SKY")
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

    science_rows = []
    sky_rows = []
    slit_rows = []

    with fits.open(args.infile) as hdul:
        for hdu in hdul[1:]:
            slit = str(hdu.name or "").strip().upper()
            if not slit.startswith("SLIT") or hdu.data is None:
                continue
            if int(hdu.header.get("S08USE", 0)) != 1:
                continue

            names = {n.upper(): n for n in hdu.columns.names}
            need = {"LAMBDA_NM", args.source_col.upper(), args.sky_col.upper()}
            if not need.issubset(names):
                continue

            lam = np.asarray(hdu.data[names["LAMBDA_NM"]], float)
            sci = np.asarray(hdu.data[names[args.source_col.upper()]], float)
            sky = np.asarray(hdu.data[names[args.sky_col.upper()]], float)

            q_sci = (
                np.isfinite(lam) & np.isfinite(sci)
                & (lam >= args.lam_min) & (lam <= args.lam_max)
            )
            q_sky = (
                np.isfinite(lam) & np.isfinite(sky)
                & (lam >= args.lam_min) & (lam <= args.lam_max)
            )
            if q_sci.sum() < 100 or q_sky.sum() < 100:
                continue

            srows, ssig = detect_positive_peaks(
                lam, sci, q_sci, args.median_kernel, args.sigma_thresh, slit
            )
            krows, ksig = detect_positive_peaks(
                lam, sky, q_sky, args.median_kernel, args.sigma_thresh, slit
            )
            science_rows.extend(srows)
            sky_rows.extend(krows)
            slit_rows.append({
                "slit": slit,
                "science_sigma": ssig,
                "sky_sigma": ksig,
                "n_science_pos": len(srows),
                "n_sky_pos": len(krows),
            })

    sci = pd.DataFrame(science_rows, columns=["slit", "lambda_nm", "z"])
    sky = pd.DataFrame(sky_rows, columns=["slit", "lambda_nm", "z"])
    slit = pd.DataFrame(slit_rows)

    sci_cl = cluster_peaks(sci, args.cluster_tol_nm)
    sky_cl = cluster_peaks(sky, args.cluster_tol_nm)
    sci_common = sci_cl[sci_cl["n_slits"] >= args.common_min_slits].copy()
    sky_common = sky_cl[sky_cl["n_slits"] >= args.common_min_slits].copy()

    sky_centers = sky_common["lambda_nm"].to_numpy(float) if len(sky_common) else np.array([], float)

    if len(sci):
        sci["nearest_common_sky_nm"] = nearest_distance(sci["lambda_nm"], sky_centers)
        sci["matches_common_sky"] = sci["nearest_common_sky_nm"] <= args.match_tol_nm

    if len(sci_common):
        sci_common["nearest_common_sky_nm"] = nearest_distance(
            sci_common["lambda_nm"], sky_centers
        )
        sci_common["matches_common_sky"] = (
            sci_common["nearest_common_sky_nm"] <= args.match_tol_nm
        )

    if len(sci) and len(slit):
        sm = sci.groupby("slit")["matches_common_sky"].agg(["sum", "count"]).reset_index()
        sm = sm.rename(columns={"sum": "n_science_match_sky", "count": "n_science_total"})
        slit = slit.merge(sm, on="slit", how="left")
        slit[["n_science_match_sky", "n_science_total"]] = (
            slit[["n_science_match_sky", "n_science_total"]].fillna(0).astype(int)
        )
        slit["science_match_frac"] = np.where(
            slit["n_science_total"] > 0,
            slit["n_science_match_sky"] / slit["n_science_total"],
            np.nan,
        )

    sci.to_csv(args.outdir / "qc_step09_science_positive_peaks_vs_sky.csv", index=False)
    sky.to_csv(args.outdir / "qc_step09_sky_positive_peaks.csv", index=False)
    sci_common.to_csv(args.outdir / "qc_step09_science_common_lines_vs_sky.csv", index=False)
    sky_common.to_csv(args.outdir / "qc_step09_common_sky_lines.csv", index=False)
    slit.to_csv(args.outdir / "qc_step09_skyline_match_by_slit.csv", index=False)

    n_sci_match = int(sci["matches_common_sky"].sum()) if len(sci) else 0
    n_scic_match = int(sci_common["matches_common_sky"].sum()) if len(sci_common) else 0

    print(f"Usable slits: {len(slit)}")
    print(f"Science positive >={args.sigma_thresh:.1f}sigma peaks: {len(sci)}")
    print(f"Sky positive >={args.sigma_thresh:.1f}sigma peaks: {len(sky)}")
    print(
        f"Common science clusters (Nslit>={args.common_min_slits}): {len(sci_common)}"
    )
    print(
        f"Common SKY clusters (Nslit>={args.common_min_slits}): {len(sky_common)}"
    )
    if len(sci):
        print(
            f"All science positive peaks within {args.match_tol_nm:.2f} nm of a common SKY line: "
            f"{n_sci_match}/{len(sci)} ({100*n_sci_match/len(sci):.1f}%)"
        )
    if len(sci_common):
        print(
            f"Common science clusters matching a common SKY line: "
            f"{n_scic_match}/{len(sci_common)} ({100*n_scic_match/len(sci_common):.1f}%)"
        )
    if len(slit):
        print("Lowest science-to-SKY match fractions:")
        show = slit.sort_values(
            ["science_match_frac", "n_science_total"],
            ascending=[True, False],
            na_position="last",
        ).head(12)
        print(show.to_string(index=False))
    print("QC only; no spectra were modified.")


if __name__ == "__main__":
    main()
