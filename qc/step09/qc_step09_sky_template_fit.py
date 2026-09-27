#!/usr/bin/env python3
"""
QC-only diagnostic for Step09 residual sky subtraction.

Tests a conservative alternative to the current free Gaussian OH fitter:

  science residual ~= alpha * measured SKY residual

where alpha is fitted independently in broad OH windows, using only pixels
near empirically recurrent SKY lines.  Alpha is allowed to be positive or
negative, but the model basis is the measured SKY spectrum itself; arbitrary
stellar/telluric absorption features are therefore not fitted.

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
    if valid.sum() < 5:
        return np.full_like(y, np.nan, dtype=float)
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
    q = (
        fitmask
        & np.isfinite(science_hp)
        & np.isfinite(sky_hp)
    )
    if q.sum() < 6:
        return np.nan, q, 0

    use = q.copy()
    alpha = np.nan
    for _ in range(maxiter):
        x = sky_hp[use]
        y = science_hp[use]
        den = float(np.nansum(x * x))
        if not np.isfinite(den) or den <= 0:
            return np.nan, use, int(use.sum())
        new_alpha = float(np.nansum(x * y) / den)
        resid = science_hp - new_alpha * sky_hp

        sig = robust_sigma(resid[use])
        if not np.isfinite(sig) or sig <= 0:
            alpha = new_alpha
            break

        new_use = q & (np.abs(resid) <= clip_sigma * sig)
        alpha = new_alpha
        if np.array_equal(new_use, use):
            break
        if new_use.sum() < 6:
            break
        use = new_use

    return alpha, use, int(use.sum())


def parse_args():
    p = argparse.ArgumentParser(
        description="QC signed measured-SKY template fit on common sky lines"
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
    centers_all = centers_all[np.isfinite(centers_all)]
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
                win = base & (lam >= wlo) & (lam <= whi)
                centers = centers_all[(centers_all >= wlo) & (centers_all <= whi)]
                lmask = win & line_mask(lam, centers, args.line_halfwidth_nm)

                if np.isfinite(sky_sig) and sky_sig > 0:
                    informative = lmask & (np.abs(sky_hp) >= args.sky_sigma_min * sky_sig)
                else:
                    informative = lmask

                alpha, used, nfit = fit_alpha(
                    science_hp,
                    sky_hp,
                    informative,
                    clip_sigma=args.clip_sigma,
                )

                model = np.zeros_like(science_hp)
                if np.isfinite(alpha):
                    model[lmask] = alpha * sky_hp[lmask]
                post = science_hp - model

                rows.append({
                    "slit": slit,
                    "wlo_nm": wlo,
                    "whi_nm": whi,
                    "n_common_lines": int(len(centers)),
                    "n_line_pixels": int(lmask.sum()),
                    "n_fit_pixels": int(nfit),
                    "alpha": alpha,
                    "rms_line_pre": robust_rms(science_hp[lmask]),
                    "rms_line_post": robust_rms(post[lmask]),
                    "rms_window_pre": robust_rms(science_hp[win]),
                    "rms_window_post": robust_rms(post[win]),
                })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise RuntimeError("No usable slit/window measurements.")

    out["line_ratio_post_pre"] = out["rms_line_post"] / out["rms_line_pre"]
    out["window_ratio_post_pre"] = out["rms_window_post"] / out["rms_window_pre"]
    out["line_improved"] = out["line_ratio_post_pre"] < 1.0
    out["window_improved"] = out["window_ratio_post_pre"] < 1.0

    csv = args.outdir / "qc_step09_sky_template_fit.csv"
    out.to_csv(csv, index=False)

    good = out[
        np.isfinite(out["alpha"])
        & np.isfinite(out["line_ratio_post_pre"])
        & (out["n_fit_pixels"] >= 6)
    ].copy()

    print(f"Usable slit/window fits: {len(good)}")
    if len(good):
        a = good["alpha"].to_numpy(float)
        lr = good["line_ratio_post_pre"].to_numpy(float)
        wr = good["window_ratio_post_pre"].to_numpy(float)

        print(
            "alpha median [p16,p84] = "
            f"{np.nanmedian(a):+.4f} "
            f"[{np.nanpercentile(a,16):+.4f}, {np.nanpercentile(a,84):+.4f}]"
        )
        print(
            f"alpha signs: positive={int(np.sum(a > 0))} "
            f"negative={int(np.sum(a < 0))} zero={int(np.sum(a == 0))}"
        )
        print(
            "Masked common-sky-line RMS post/pre median "
            f"[p16,p84] = {np.nanmedian(lr):.3f} "
            f"[{np.nanpercentile(lr,16):.3f}, {np.nanpercentile(lr,84):.3f}]"
        )
        print(
            "Full OH-window RMS post/pre median "
            f"[p16,p84] = {np.nanmedian(wr):.3f} "
            f"[{np.nanpercentile(wr,16):.3f}, {np.nanpercentile(wr,84):.3f}]"
        )
        print(
            f"Fits improving masked sky-line RMS: "
            f"{int(np.sum(lr < 1))}/{len(good)} "
            f"({100*np.mean(lr < 1):.1f}%)"
        )
        print(
            f"Fits improving full-window RMS: "
            f"{int(np.sum(wr < 1))}/{len(good)} "
            f"({100*np.mean(wr < 1):.1f}%)"
        )
        print()
        show = good.sort_values("line_ratio_post_pre").head(12)
        print("Largest masked-line improvements:")
        print(show[
            [
                "slit", "wlo_nm", "whi_nm", "n_common_lines",
                "n_fit_pixels", "alpha",
                "line_ratio_post_pre", "window_ratio_post_pre"
            ]
        ].to_string(index=False))

    print(f"Wrote: {csv}")
    print("QC only; no spectra were modified.")


if __name__ == "__main__":
    main()
