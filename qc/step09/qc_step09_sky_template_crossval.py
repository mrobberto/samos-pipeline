#!/usr/bin/env python3
"""
QC-only cross-validation of the Step09 measured-SKY template model.

For each usable slit and each OH window, recurrent empirical SKY lines are
split deterministically into two wavelength-interleaved folds.  A single
scale factor alpha is fitted on one fold and evaluated on the other, then the
roles are reversed.

This avoids judging the model on the same sky-line pixels used to fit alpha.
Both signed alpha and non-negative alpha=max(alpha,0) are evaluated.

No FITS data are modified.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import median_filter


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


def line_mask(lam, centers, halfwidth):
    m = np.zeros(len(lam), dtype=bool)
    for c in centers:
        m |= np.abs(lam - float(c)) <= halfwidth
    return m


def fit_alpha(science_hp, sky_hp, fitmask, clip_sigma=3.5, maxiter=8):
    q = fitmask & np.isfinite(science_hp) & np.isfinite(sky_hp)
    if q.sum() < 6:
        return np.nan, 0

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


def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-validate measured-SKY residual template subtraction"
    )
    p.add_argument("--infile", type=Path, required=True)
    p.add_argument("--sky-lines-csv", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--source-col", default="FLUX_APCORR")
    p.add_argument("--sky-col", default="SKY")
    p.add_argument("--median-kernel", type=int, default=101)
    p.add_argument("--line-halfwidth-nm", type=float, default=0.60)
    p.add_argument("--sky-sigma-min", type=float, default=2.0)
    p.add_argument("--clip-sigma", type=float, default=3.5)
    p.add_argument("--min-common-slits", type=int, default=4)
    p.add_argument("--min-lines-per-fold", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    lines = pd.read_csv(args.sky_lines_csv)
    need = {"lambda_nm", "n_slits"}
    if not need.issubset(lines.columns):
        raise RuntimeError(
            f"{args.sky_lines_csv} must contain columns {sorted(need)}"
        )

    centers_all = lines.loc[
        pd.to_numeric(lines["n_slits"], errors="coerce") >= args.min_common_slits,
        "lambda_nm",
    ].to_numpy(float)
    centers_all = np.sort(centers_all[np.isfinite(centers_all)])
    if centers_all.size == 0:
        raise RuntimeError("No common SKY lines passed the requested threshold.")

    rows = []

    with fits.open(args.infile) as hdul:
        for hdu in hdul[1:]:
            slit = str(hdu.name or "").strip().upper()
            if not slit.startswith("SLIT") or hdu.data is None:
                continue
            if int(hdu.header.get("S08USE", 0)) != 1:
                continue

            names = {n.upper(): n for n in hdu.columns.names}
            needed = {"LAMBDA_NM", args.source_col.upper(), args.sky_col.upper()}
            if not needed.issubset(names):
                continue

            lam = np.asarray(hdu.data[names["LAMBDA_NM"]], float)
            science = np.asarray(hdu.data[names[args.source_col.upper()]], float)
            sky = np.asarray(hdu.data[names[args.sky_col.upper()]], float)

            base = np.isfinite(lam) & np.isfinite(science) & np.isfinite(sky)
            science_hp = highpass(science, base, args.median_kernel)
            sky_hp = highpass(sky, base, args.median_kernel)
            sky_sig = robust_sigma(sky_hp[base])

            for wlo, whi in OH_WINDOWS:
                centers = centers_all[
                    (centers_all >= wlo) & (centers_all <= whi)
                ]
                if len(centers) < 2 * args.min_lines_per_fold:
                    continue

                fold0 = centers[::2]
                fold1 = centers[1::2]
                if (
                    len(fold0) < args.min_lines_per_fold
                    or len(fold1) < args.min_lines_per_fold
                ):
                    continue

                win = base & (lam >= wlo) & (lam <= whi)

                for fold_id, (train_centers, test_centers) in enumerate(
                    ((fold0, fold1), (fold1, fold0))
                ):
                    train_line = win & line_mask(
                        lam, train_centers, args.line_halfwidth_nm
                    )
                    test_line = win & line_mask(
                        lam, test_centers, args.line_halfwidth_nm
                    )

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
                        post_signed = robust_rms(
                            science_hp[test_line] - alpha * sky_hp[test_line]
                        )
                        alpha_pos = max(alpha, 0.0)
                        post_pos = robust_rms(
                            science_hp[test_line] - alpha_pos * sky_hp[test_line]
                        )
                    else:
                        alpha_pos = np.nan
                        post_signed = np.nan
                        post_pos = np.nan

                    rows.append(
                        {
                            "slit": slit,
                            "wlo_nm": wlo,
                            "whi_nm": whi,
                            "fold": fold_id,
                            "n_train_lines": int(len(train_centers)),
                            "n_test_lines": int(len(test_centers)),
                            "n_fit_pixels": int(nfit),
                            "alpha_signed": alpha,
                            "alpha_nonnegative": alpha_pos,
                            "test_rms_pre": pre,
                            "test_rms_post_signed": post_signed,
                            "test_rms_post_nonnegative": post_pos,
                        }
                    )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise RuntimeError("No usable cross-validation folds.")

    out["ratio_signed"] = (
        out["test_rms_post_signed"] / out["test_rms_pre"]
    )
    out["ratio_nonnegative"] = (
        out["test_rms_post_nonnegative"] / out["test_rms_pre"]
    )

    csv = args.outdir / "qc_step09_sky_template_crossval.csv"
    out.to_csv(csv, index=False)

    good = out[
        np.isfinite(out["ratio_signed"])
        & np.isfinite(out["ratio_nonnegative"])
        & (out["n_fit_pixels"] >= 6)
    ].copy()

    print(f"Usable cross-validation folds: {len(good)}")
    if len(good):
        rs = good["ratio_signed"].to_numpy(float)
        rp = good["ratio_nonnegative"].to_numpy(float)
        aa = good["alpha_signed"].to_numpy(float)

        print(
            "SIGNED holdout RMS post/pre median [p16,p84] = "
            f"{np.nanmedian(rs):.3f} "
            f"[{np.nanpercentile(rs,16):.3f}, {np.nanpercentile(rs,84):.3f}]"
        )
        print(
            "NONNEG holdout RMS post/pre median [p16,p84] = "
            f"{np.nanmedian(rp):.3f} "
            f"[{np.nanpercentile(rp,16):.3f}, {np.nanpercentile(rp,84):.3f}]"
        )
        print(
            "SIGNED improves holdout = "
            f"{int(np.sum(rs < 1))}/{len(good)} "
            f"({100*np.mean(rs < 1):.1f}%)"
        )
        print(
            "NONNEG improves holdout = "
            f"{int(np.sum(rp < 1))}/{len(good)} "
            f"({100*np.mean(rp < 1):.1f}%)"
        )
        print(
            "Signed alpha signs: "
            f"positive={int(np.sum(aa > 0))} "
            f"negative={int(np.sum(aa < 0))}"
        )
        print(
            f"|alpha_signed| > 5: {int(np.sum(np.abs(aa) > 5))}/{len(good)}"
        )

        pair = good.groupby(["slit", "wlo_nm", "whi_nm"]).agg(
            signed_med=("ratio_signed", "median"),
            nonneg_med=("ratio_nonnegative", "median"),
            nfold=("fold", "count"),
        ).reset_index()
        pair = pair[pair["nfold"] == 2].copy()

        print(f"Complete slit/window pairs: {len(pair)}")
        if len(pair):
            print(
                "Complete-pair median signed/nonnegative = "
                f"{np.nanmedian(pair['signed_med']):.3f} / "
                f"{np.nanmedian(pair['nonneg_med']):.3f}"
            )
            print()
            print("Best generalizing signed fits:")
            print(
                pair.sort_values("signed_med").head(12).to_string(index=False)
            )

    print(f"Wrote: {csv}")
    print("QC only; no spectra were modified.")


if __name__ == "__main__":
    main()
