#!/usr/bin/env python3
"""
Step09 residual-sky cleanup using the measured local SKY spectrum.

This stage is designed to follow the Step09a/09b ensemble-relative wavelength
refinement.  It starts from the already local-sky-subtracted Step08 optimal
extraction (FLUX_APCORR by default) and models only recurrent residual sky
features.

Method
------
1. Build an empirical list of recurrent sky-line wavelengths from the SKY
   column across S08USE=1 slits.
2. In each of six OH-rich wavelength windows, split those recurrent lines into
   two interleaved folds.
3. Fit a signed scale factor alpha between the high-pass science spectrum and
   high-pass SKY template on one fold, validate on the other, then reverse.
4. Accept a slit/window only when both folds:
      - have the same sign,
      - have relative alpha disagreement <= 0.5,
      - have |alpha| <= 5,
      - use >= 6 fit pixels,
      - reduce the independent holdout RMS by at least 1% by default
        (holdout RMS ratio < 0.99; configurable).
5. Refit alpha using all recurrent sky lines in the accepted window and apply
   alpha * SKY_highpass only within +/-0.60 nm of those recurrent sky lines.

Rejected windows are left unchanged.  No median fallback is used.

Outputs preserve all input columns and add:
  SKYRES_MODEL        residual sky model actually applied
  SKYRES_FLAG         1 where an accepted model is active, else 0
  STELLAR_CONSENSUS   FLUX_APCORR - SKYRES_MODEL

The existing variance columns are preserved unchanged.  Uncertainty in the
empirical residual-sky model is not propagated here and is recorded in the
headers.

This is intentionally conservative: it does not fit arbitrary Gaussian
emission or absorption features in the stellar spectrum.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


OH_WINDOWS = [
    (780.0, 805.0),
    (806.0, 825.0),
    (845.0, 875.0),
    (875.0, 905.0),
    (905.0, 930.0),
    (930.0, 960.0),
]


def robust_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if np.isfinite(mad) and mad > 0:
        return float(1.4826 * mad)
    return float(np.nanstd(x))


def robust_rms(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    med = np.nanmedian(x)
    return float(np.sqrt(np.nanmedian((x - med) ** 2)))


def highpass(y, valid, kernel):
    y = np.asarray(y, float)
    out = np.full_like(y, np.nan, dtype=float)
    if valid.sum() < 5:
        return out
    fill = float(np.nanmedian(y[valid]))
    work = y.copy()
    work[~np.isfinite(work)] = fill
    cont = median_filter(work, size=kernel, mode="nearest")
    out = y - cont
    out[~valid] = np.nan
    return out


def detect_positive_peaks(lam, y, valid, kernel, sigma_thresh, slit):
    hp = highpass(y, valid, kernel)
    sig = robust_sigma(hp[valid])
    if not np.isfinite(sig) or sig <= 0:
        return []
    idx = np.where(valid)[0]
    z = hp[idx] / sig
    pp, _ = find_peaks(z, height=sigma_thresh, distance=2)
    rows = []
    for p in pp:
        ii = int(idx[p])
        rows.append(
            dict(
                slit=slit,
                lambda_nm=float(lam[ii]),
                z=float(hp[ii] / sig),
            )
        )
    return rows


def cluster_peaks(peaks, tol_nm):
    if len(peaks) == 0:
        return pd.DataFrame(
            columns=["cluster_id", "lambda_nm", "n_det", "n_slits", "slits"]
        )
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

    rows = []
    for j, idx in enumerate(groups):
        g = d.loc[idx]
        slits = sorted(set(g["slit"].astype(str)))
        rows.append(
            dict(
                cluster_id=f"SKY{j:04d}",
                lambda_nm=float(np.nanmedian(g["lambda_nm"])),
                n_det=int(len(g)),
                n_slits=int(len(slits)),
                slits=",".join(slits),
            )
        )
    return pd.DataFrame(rows)


def line_mask(lam, centers, halfwidth):
    m = np.zeros(len(lam), dtype=bool)
    for c in np.asarray(centers, float):
        m |= np.abs(lam - c) <= halfwidth
    return m


def fit_alpha(science_hp, sky_hp, fitmask, clip_sigma=3.5, maxiter=8):
    q = fitmask & np.isfinite(science_hp) & np.isfinite(sky_hp)
    if q.sum() < 6:
        return np.nan, int(q.sum())
    use = q.copy()
    alpha = np.nan
    for _ in range(maxiter):
        x = sky_hp[use]
        y = science_hp[use]
        den = float(np.nansum(x * x))
        if not np.isfinite(den) or den <= 0:
            return np.nan, int(use.sum())
        alpha_new = float(np.nansum(x * y) / den)
        resid = science_hp - alpha_new * sky_hp
        sig = robust_sigma(resid[use])
        alpha = alpha_new
        if not np.isfinite(sig) or sig <= 0:
            break
        new_use = q & (np.abs(resid) <= clip_sigma * sig)
        if new_use.sum() < 6 or np.array_equal(new_use, use):
            break
        use = new_use
    return alpha, int(use.sum())


def add_or_replace(tab, name, arr, dtype=None):
    arr = np.asarray(arr)
    if dtype is not None:
        arr = arr.astype(dtype)
    if name in tab.colnames:
        tab[name] = arr
    else:
        tab[name] = arr


def parse_args():
    p = argparse.ArgumentParser(
        description="Apply conservative empirical residual-SKY cleanup"
    )
    p.add_argument("--infile", type=Path, required=True)
    p.add_argument("--outfile", type=Path, required=True)
    p.add_argument("--summary-csv", type=Path, default=None)
    p.add_argument("--lines-csv", type=Path, default=None)
    p.add_argument("--source-col", default="FLUX_APCORR")
    p.add_argument("--sky-col", default="SKY")
    p.add_argument("--median-kernel", type=int, default=101)
    p.add_argument("--sky-peak-sigma", type=float, default=3.0)
    p.add_argument("--cluster-tol-nm", type=float, default=0.25)
    p.add_argument("--common-min-slits", type=int, default=4)
    p.add_argument("--line-halfwidth-nm", type=float, default=0.60)
    p.add_argument("--sky-sigma-min", type=float, default=2.0)
    p.add_argument("--clip-sigma", type=float, default=3.5)
    p.add_argument("--max-alpha", type=float, default=5.0)
    p.add_argument("--max-rel-disagree", type=float, default=0.50)
    p.add_argument("--min-fit-pixels", type=int, default=6)
    p.add_argument(
        "--max-holdout-ratio",
        type=float,
        default=0.99,
        help="Require each independent holdout RMS post/pre ratio to be below this value",
    )
    p.add_argument("--min-lines-per-fold", type=int, default=3)
    return p.parse_args()


def build_common_sky_lines(hdul, args):
    rows = []
    for hdu in hdul[1:]:
        slit = str(hdu.name or "").strip().upper()
        if not slit.startswith("SLIT") or hdu.data is None:
            continue
        if int(hdu.header.get("S08USE", 0)) != 1:
            continue
        names = {n.upper(): n for n in hdu.columns.names}
        need = {"LAMBDA_NM", args.sky_col.upper()}
        if not need.issubset(names):
            continue
        lam = np.asarray(hdu.data[names["LAMBDA_NM"]], float)
        sky = np.asarray(hdu.data[names[args.sky_col.upper()]], float)
        valid = (
            np.isfinite(lam)
            & np.isfinite(sky)
            & (lam >= 780.0)
            & (lam <= 960.0)
        )
        if valid.sum() < 100:
            continue
        rows.extend(
            detect_positive_peaks(
                lam,
                sky,
                valid,
                args.median_kernel,
                args.sky_peak_sigma,
                slit,
            )
        )

    peaks = pd.DataFrame(rows, columns=["slit", "lambda_nm", "z"])
    clusters = cluster_peaks(peaks, args.cluster_tol_nm)
    common = clusters[clusters["n_slits"] >= args.common_min_slits].copy()
    common = common.sort_values("lambda_nm").reset_index(drop=True)
    return peaks, clusters, common


def evaluate_window(lam, science_hp, sky_hp, sky_sig, centers, args):
    result = dict(
        n_common_lines=int(len(centers)),
        alpha0=np.nan,
        alpha1=np.nan,
        nfit0=0,
        nfit1=0,
        holdout0=np.nan,
        holdout1=np.nan,
        rel_disagree=np.nan,
        alpha_final=np.nan,
        nfit_final=0,
        accepted=False,
        reason="",
    )

    if len(centers) < 2 * args.min_lines_per_fold:
        result["reason"] = "TOO_FEW_LINES"
        return result, np.zeros(len(lam), bool)

    fold0 = np.asarray(centers[::2], float)
    fold1 = np.asarray(centers[1::2], float)
    if (
        len(fold0) < args.min_lines_per_fold
        or len(fold1) < args.min_lines_per_fold
    ):
        result["reason"] = "TOO_FEW_FOLD_LINES"
        return result, np.zeros(len(lam), bool)

    masks = []
    alphas = []
    nfits = []
    holdouts = []

    for train_centers, test_centers in ((fold0, fold1), (fold1, fold0)):
        train_line = line_mask(lam, train_centers, args.line_halfwidth_nm)
        test_line = line_mask(lam, test_centers, args.line_halfwidth_nm)

        if np.isfinite(sky_sig) and sky_sig > 0:
            train_fit = train_line & (
                np.abs(sky_hp) >= args.sky_sigma_min * sky_sig
            )
        else:
            train_fit = train_line

        alpha, nfit = fit_alpha(
            science_hp,
            sky_hp,
            train_fit,
            clip_sigma=args.clip_sigma,
        )

        pre = robust_rms(science_hp[test_line])
        if np.isfinite(alpha):
            post = robust_rms(
                science_hp[test_line] - alpha * sky_hp[test_line]
            )
        else:
            post = np.nan

        ratio = (
            float(post / pre)
            if np.isfinite(pre) and pre > 0 and np.isfinite(post)
            else np.nan
        )

        masks.append(train_line)
        alphas.append(alpha)
        nfits.append(nfit)
        holdouts.append(ratio)

    result["alpha0"] = alphas[0]
    result["alpha1"] = alphas[1]
    result["nfit0"] = nfits[0]
    result["nfit1"] = nfits[1]
    result["holdout0"] = holdouts[0]
    result["holdout1"] = holdouts[1]

    a0, a1 = alphas
    if not np.isfinite(a0) or not np.isfinite(a1):
        result["reason"] = "NONFINITE_ALPHA"
        return result, np.zeros(len(lam), bool)

    denom = max(0.5 * (abs(a0) + abs(a1)), 1e-12)
    rel = abs(a0 - a1) / denom
    result["rel_disagree"] = float(rel)

    if a0 * a1 <= 0:
        result["reason"] = "SIGN_DISAGREE"
        return result, np.zeros(len(lam), bool)
    if rel > args.max_rel_disagree:
        result["reason"] = "ALPHA_UNSTABLE"
        return result, np.zeros(len(lam), bool)
    if abs(a0) > args.max_alpha or abs(a1) > args.max_alpha:
        result["reason"] = "ALPHA_TOO_LARGE"
        return result, np.zeros(len(lam), bool)
    if nfits[0] < args.min_fit_pixels or nfits[1] < args.min_fit_pixels:
        result["reason"] = "TOO_FEW_PIXELS"
        return result, np.zeros(len(lam), bool)
    if (
        not np.isfinite(holdouts[0])
        or not np.isfinite(holdouts[1])
        or holdouts[0] >= args.max_holdout_ratio
        or holdouts[1] >= args.max_holdout_ratio
    ):
        result["reason"] = "NO_HOLDOUT_GAIN"
        return result, np.zeros(len(lam), bool)

    all_line = line_mask(lam, centers, args.line_halfwidth_nm)
    if np.isfinite(sky_sig) and sky_sig > 0:
        all_fit = all_line & (np.abs(sky_hp) >= args.sky_sigma_min * sky_sig)
    else:
        all_fit = all_line

    alpha_final, nfit_final = fit_alpha(
        science_hp,
        sky_hp,
        all_fit,
        clip_sigma=args.clip_sigma,
    )
    result["alpha_final"] = alpha_final
    result["nfit_final"] = nfit_final

    if not np.isfinite(alpha_final):
        result["reason"] = "FINAL_ALPHA_NONFINITE"
        return result, np.zeros(len(lam), bool)
    if abs(alpha_final) > args.max_alpha:
        result["reason"] = "FINAL_ALPHA_TOO_LARGE"
        return result, np.zeros(len(lam), bool)

    result["accepted"] = True
    result["reason"] = "ACCEPT"
    return result, all_line


def main():
    args = parse_args()

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_csv is None:
        args.summary_csv = args.outfile.with_name(
            args.outfile.stem + "_sky_summary.csv"
        )
    if args.lines_csv is None:
        args.lines_csv = args.outfile.with_name(
            args.outfile.stem + "_sky_lines.csv"
        )

    with fits.open(args.infile, memmap=False) as hdul:
        _, _, common = build_common_sky_lines(hdul, args)
        if len(common) == 0:
            raise RuntimeError("No recurrent empirical SKY lines were found.")

        common.to_csv(args.lines_csv, index=False)
        centers_all = common["lambda_nm"].to_numpy(float)

        primary = fits.PrimaryHDU(header=hdul[0].header.copy())
        primary.header["PIPESTEP"] = ("STEP09", "Pipeline step")
        primary.header["STAGE"] = ("09sky", "Empirical residual-sky cleanup")
        primary.header["S9METH"] = ("EMP_SKYT", "Empirical SKY-template method")
        primary.header["S9SRC"] = (args.source_col, "Science source column")
        primary.header["S9SKY"] = (args.sky_col, "Measured SKY template column")
        primary.header["S9NLINES"] = (int(len(common)), "Common SKY lines")
        primary.header["S9VARPR"] = (False, "Residual-sky model variance propagated")
        primary.header.add_history(
            "Step09 empirical residual-sky correction from measured SKY; "
            "rejected windows unchanged."
        )

        out_hdus = [primary]
        summary_rows = []
        n_accepted_total = 0
        slits_with_accept = set()

        for hdu in hdul[1:]:
            if hdu.data is None or not str(hdu.name or "").upper().startswith("SLIT"):
                out_hdus.append(hdu.copy())
                continue

            slit = str(hdu.name).strip().upper()
            tab = Table(hdu.data)
            names = {n.upper(): n for n in tab.colnames}

            required = {
                "LAMBDA_NM",
                args.source_col.upper(),
                args.sky_col.upper(),
            }
            if not required.issubset(names):
                out_hdus.append(hdu.copy())
                continue

            lam = np.asarray(tab[names["LAMBDA_NM"]], float)
            source = np.asarray(tab[names[args.source_col.upper()]], float)
            sky = np.asarray(tab[names[args.sky_col.upper()]], float)

            base = np.isfinite(lam) & np.isfinite(source) & np.isfinite(sky)
            science_hp = highpass(source, base, args.median_kernel)
            sky_hp = highpass(sky, base, args.median_kernel)
            sky_sig = robust_sigma(sky_hp[base])

            model = np.zeros(len(tab), dtype=float)
            flag = np.zeros(len(tab), dtype=np.int16)
            nacc = 0

            use_slit = int(hdu.header.get("S08USE", 0)) == 1

            for wlo, whi in OH_WINDOWS:
                centers = centers_all[
                    (centers_all >= wlo) & (centers_all <= whi)
                ]

                if use_slit:
                    result, apply_mask = evaluate_window(
                        lam,
                        science_hp,
                        sky_hp,
                        sky_sig,
                        centers,
                        args,
                    )
                else:
                    result = dict(
                        n_common_lines=int(len(centers)),
                        alpha0=np.nan,
                        alpha1=np.nan,
                        nfit0=0,
                        nfit1=0,
                        holdout0=np.nan,
                        holdout1=np.nan,
                        rel_disagree=np.nan,
                        alpha_final=np.nan,
                        nfit_final=0,
                        accepted=False,
                        reason="S08USE0",
                    )
                    apply_mask = np.zeros(len(tab), dtype=bool)

                if result["accepted"]:
                    alpha = float(result["alpha_final"])
                    this_model = alpha * sky_hp
                    q = (
                        apply_mask
                        & (lam >= wlo)
                        & (lam <= whi)
                        & np.isfinite(this_model)
                    )
                    model[q] += this_model[q]
                    flag[q] = 1
                    nacc += 1
                    n_accepted_total += 1
                    slits_with_accept.add(slit)

                summary_rows.append(
                    dict(
                        slit=slit,
                        s08use=int(hdu.header.get("S08USE", 0)),
                        wlo_nm=wlo,
                        whi_nm=whi,
                        **result,
                    )
                )

            stellar = source - model

            add_or_replace(tab, "SKYRES_MODEL", model, "f4")
            add_or_replace(tab, "SKYRES_FLAG", flag, "i2")
            add_or_replace(tab, "STELLAR_CONSENSUS", stellar, "f4")

            hdr = hdu.header.copy()
            hdr["S9METH"] = ("EMP_SKYT", "Empirical SKY-template method")
            hdr["S9SRC"] = (args.source_col, "Science source column")
            hdr["S9SKY"] = (args.sky_col, "Measured SKY template column")
            hdr["S9NWACC"] = (int(nacc), "Accepted residual-sky windows")
            hdr["S9CVREL"] = (float(args.max_rel_disagree), "Max fold alpha disagreement")
            hdr["S9AMAX"] = (float(args.max_alpha), "Max absolute alpha")
            hdr["S9HRMAX"] = (float(args.max_holdout_ratio), "Max holdout RMS post/pre ratio")
            hdr["S9LHW"] = (float(args.line_halfwidth_nm), "Sky-line halfwidth nm")
            hdr["S9VARPR"] = (False, "Residual-sky model variance propagated")

            out_hdus.append(
                fits.BinTableHDU(tab, header=hdr, name=slit)
            )

        summary = pd.DataFrame(summary_rows)
        summary.to_csv(args.summary_csv, index=False)

        primary.header["S9NACC"] = (
            int(n_accepted_total),
            "Accepted slit/window corrections",
        )
        primary.header["S9NSLIT"] = (
            int(len(slits_with_accept)),
            "Slits with >=1 accepted correction",
        )

        fits.HDUList(out_hdus).writeto(args.outfile, overwrite=True)

    print("Input:", args.infile)
    print("Output:", args.outfile)
    print("Common empirical SKY lines:", len(common))
    print(
        "Accepted slit/windows:",
        n_accepted_total,
        " across ",
        len(slits_with_accept),
        "slits",
    )
    if len(summary):
        acc = summary[summary["accepted"] == True]
        if len(acc):
            print(
                "Final alpha median [p16,p84] = "
                f"{np.nanmedian(acc['alpha_final']):+.4f} "
                f"[{np.nanpercentile(acc['alpha_final'],16):+.4f}, "
                f"{np.nanpercentile(acc['alpha_final'],84):+.4f}]"
            )
            print(
                "Accepted signs: positive=",
                int(np.sum(acc["alpha_final"] > 0)),
                " negative=",
                int(np.sum(acc["alpha_final"] < 0)),
            )
        print("Acceptance reasons:")
        print(summary["reason"].value_counts().to_string())
    print("Wrote summary:", args.summary_csv)
    print("Wrote sky lines:", args.lines_csv)
    print("NOTE: rejected windows and all non-OH wavelengths are unchanged.")
    print("NOTE: residual-sky model uncertainty is not propagated into VAR.")


if __name__ == "__main__":
    main()
