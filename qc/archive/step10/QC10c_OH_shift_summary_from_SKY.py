#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC10c_OH_shift_summary_from_SKY.py

Official Step10 QC for SAMOS OH wavelength zero-point refinement.

Purpose
-------
Summarize and visualize the OH-based slit-by-slit wavelength refinement used in
Step10, using the same SKY/LAMBDA_NM convention as the Step08/Step10 products.

Inputs
------
1) Pre-Step10 extracted MEF (default):
   ST08/extract1d_optimal_ridge_all_wav_tellcorr.fits
2) Post-Step10 extracted MEF (default):
   ST08/extract1d_optimal_ridge_all_wav_tellcorr_OHref.fits.fits
3) Step10.1 CSV (default):
   ST10/oh_shifts.csv

Outputs
-------
- ST10/QC10_OH_shift_summary.csv
- ST10/QC10_OH_shift_summary.pdf

What it reports
---------------
- Measured shift_nm from Step10.1 CSV
- Whether the slit-specific shift was accepted (GOOD) or fallback was used
- Applied shift inferred from the change in LAMBDA_NM between pre and post files
- Step08 flags (S08BAD, S08EMP)
- Summary statistics and diagnostic plots
- A few before/after SKY overlays in OH-rich windows

Run (Spyder)
------------
%runfile 'QC10c_OH_shift_summary_from_SKY.py' --wdir
or
%runfile 'QC10c_OH_shift_summary_from_SKY.py' --args='--clip 0.5 --nexamples 6'
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import config


DEFAULT_WINDOWS_NM = [
    (780.0, 805.0),
    (806.0, 825.0),
    (845.0, 875.0),
    (875.0, 905.0),
    (905.0, 930.0),
]


def norm_slit(s: str) -> str:
    return str(s).strip().upper()


def is_slit(name: str) -> bool:
    return norm_slit(name).startswith("SLIT")


def slit_num(name: str) -> int:
    try:
        return int(norm_slit(name).replace("SLIT", ""))
    except Exception:
        return 10**9


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if np.isfinite(mad) and mad > 0:
        return float(1.4826 * mad)
    sig = np.nanstd(x)
    return float(sig) if np.isfinite(sig) else np.nan


def highpass_running_median(y: np.ndarray, k: int = 101) -> np.ndarray:
    y = np.asarray(y, float)
    n = y.size
    if n == 0:
        return y.copy()
    k = int(k)
    if k < 5:
        return y.copy()
    if k % 2 == 0:
        k += 1
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        win = sliding_window_view(ypad, k)
        med = np.nanmedian(win, axis=1)
    except Exception:
        med = np.empty(n, float)
        for i in range(n):
            med[i] = np.nanmedian(ypad[i:i + k])
    return y - med


def interp_to_grid(lam: np.ndarray, flux: np.ndarray, grid: np.ndarray) -> np.ndarray:
    lam = np.asarray(lam, float)
    flux = np.asarray(flux, float)
    ok = np.isfinite(lam) & np.isfinite(flux)
    if ok.sum() < 20:
        return np.full_like(grid, np.nan, dtype=float)
    s = np.argsort(lam[ok])
    x = lam[ok][s]
    y = flux[ok][s]
    dx = np.diff(x)
    good = np.concatenate([[True], dx > 0])
    x = x[good]
    y = y[good]
    if x.size < 20:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, x, y, left=np.nan, right=np.nan)


def parse_args():
    ap = argparse.ArgumentParser()
    st08 = config.ST08_EXTRACT1D
    st10 = config.ST10_OH

    ap.add_argument("--pre", type=str, default=str(st08 / "extract1d_optimal_ridge_all_wav_tellcorr.fits.fits"))
    ap.add_argument("--post", type=str, default=str(st08 / "extract1d_optimal_ridge_all_wav_tellcorr_OHref.fits.fits"))
    ap.add_argument("--csv", type=str, default=str(st10 / "oh_shifts.csv"))
    ap.add_argument("--outcsv", type=str, default=str(st10 / "QC10_OH_shift_summary.csv"))
    ap.add_argument("--outpdf", type=str, default=str(st10 / "QC10_OH_shift_summary.pdf"))
    ap.add_argument("--clip", type=float, default=0.5, help="Same conservative clip used in patched Step10.2")
    ap.add_argument("--hp_width", type=int, default=101, help="High-pass width for overlay display")
    ap.add_argument("--nexamples", type=int, default=6, help="Number of slit overlays to include")
    return ap.parse_args()


def get_slit_arrays(hdu):
    d = hdu.data
    if d is None or not hasattr(d, "columns"):
        return None
    cols = set(d.columns.names)
    need = {"LAMBDA_NM", "SKY", "VAR"}
    if not need.issubset(cols):
        return None
    return {
        "lam": np.asarray(d["LAMBDA_NM"], float),
        "sky": np.asarray(d["SKY"], float),
        "var": np.asarray(d["VAR"], float),
    }


def infer_applied_shift_nm(lam_pre: np.ndarray, lam_post: np.ndarray) -> float:
    if lam_pre is None or lam_post is None:
        return np.nan
    ok = np.isfinite(lam_pre) & np.isfinite(lam_post)
    if ok.sum() < 10:
        return np.nan
    d = np.asarray(lam_post[ok] - lam_pre[ok], float)
    return float(np.nanmedian(d))


def classify_source(use: bool, measured_shift: float, clip_nm: float) -> str:
    if bool(use) and np.isfinite(measured_shift) and abs(float(measured_shift)) <= float(clip_nm):
        return "GOOD"
    return "FALLBACK"


def choose_reference_slit(df: pd.DataFrame) -> str:
    vals = df["ref_slit"].dropna().astype(str).str.upper()
    if len(vals) == 0:
        return "UNKNOWN"
    return vals.mode().iloc[0]


def collect_slit_metadata(hdul):
    meta = {}
    for h in hdul[1:]:
        name = norm_slit(h.name)
        if not is_slit(name):
            continue
        hdr = h.header
        meta[name] = {
            "S08BAD": int(hdr.get("S08BAD", 0)),
            "S08EMP": int(hdr.get("S08EMP", 0)),
            "Y0DET": hdr.get("Y0DET", np.nan),
            "SHIFT2M": hdr.get("SHIFT2M", np.nan),
        }
    return meta


def build_summary(pre_file: Path, post_file: Path, csv_file: Path, clip_nm: float) -> tuple[pd.DataFrame, dict]:
    df_csv = pd.read_csv(csv_file)
    df_csv["slit"] = df_csv["slit"].astype(str).str.upper()
    if "use" in df_csv.columns:
        df_csv["use"] = df_csv["use"].astype(bool)
    else:
        df_csv["use"] = False

    with fits.open(pre_file) as hpre, fits.open(post_file) as hpost:
        meta = collect_slit_metadata(hpre)
        slits_pre = {norm_slit(h.name): h for h in hpre[1:] if is_slit(h.name)}
        slits_post = {norm_slit(h.name): h for h in hpost[1:] if is_slit(h.name)}

        rows = []
        all_slits = sorted(set(slits_pre) | set(slits_post) | set(df_csv["slit"]), key=slit_num)
        fallback_values = []

        for slit in all_slits:
            rec = df_csv.loc[df_csv["slit"] == slit]
            measured_shift = float(rec["shift_nm"].iloc[0]) if len(rec) else np.nan
            use = bool(rec["use"].iloc[0]) if len(rec) else False
            r = float(rec["r"].iloc[0]) if (len(rec) and "r" in rec.columns and np.isfinite(rec["r"].iloc[0])) else np.nan
            nwin = int(rec["nwin"].iloc[0]) if (len(rec) and "nwin" in rec.columns and np.isfinite(rec["nwin"].iloc[0])) else np.nan
            shift_std_nm = float(rec["shift_std_nm"].iloc[0]) if (len(rec) and "shift_std_nm" in rec.columns and np.isfinite(rec["shift_std_nm"].iloc[0])) else np.nan
            ref_slit = str(rec["ref_slit"].iloc[0]).upper() if (len(rec) and "ref_slit" in rec.columns) else np.nan

            arr_pre = get_slit_arrays(slits_pre[slit]) if slit in slits_pre else None
            arr_post = get_slit_arrays(slits_post[slit]) if slit in slits_post else None
            applied_shift = infer_applied_shift_nm(
                arr_pre["lam"] if arr_pre else None,
                arr_post["lam"] if arr_post else None,
            )

            source = classify_source(use, measured_shift, clip_nm)
            if source == "FALLBACK" and np.isfinite(applied_shift):
                fallback_values.append(applied_shift)

            m = meta.get(slit, {})
            rows.append({
                "slit": slit,
                "measured_shift_nm": measured_shift,
                "use": bool(use),
                "source": source,
                "applied_shift_nm": applied_shift,
                "r": r,
                "nwin": nwin,
                "shift_std_nm": shift_std_nm,
                "ref_slit": ref_slit,
                "S08BAD": int(m.get("S08BAD", 0)),
                "S08EMP": int(m.get("S08EMP", 0)),
                "Y0DET": m.get("Y0DET", np.nan),
                "SHIFT2M": m.get("SHIFT2M", np.nan),
                "pre_exists": bool(arr_pre is not None),
                "post_exists": bool(arr_post is not None),
            })

    df = pd.DataFrame(rows)
    fallback_median = float(np.nanmedian(fallback_values)) if len(fallback_values) else np.nan

    stats = {
        "n_total": int(len(df)),
        "n_good": int((df["source"] == "GOOD").sum()),
        "n_fallback": int((df["source"] == "FALLBACK").sum()),
        "median_measured_nm": float(np.nanmedian(df["measured_shift_nm"])) if len(df) else np.nan,
        "robust_sigma_measured_nm": robust_sigma(df["measured_shift_nm"].values) if len(df) else np.nan,
        "median_applied_nm": float(np.nanmedian(df["applied_shift_nm"])) if len(df) else np.nan,
        "robust_sigma_applied_nm": robust_sigma(df["applied_shift_nm"].values) if len(df) else np.nan,
        "fallback_median_nm": fallback_median,
        "ref_slit": choose_reference_slit(df),
    }
    return df, stats


def make_summary_page(pdf, df: pd.DataFrame, stats: dict, pre_file: Path, post_file: Path, csv_file: Path, clip_nm: float):
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")

    lines = [
        "SAMOS Step10 QC — OH shift summary from SKY",
        "",
        f"Pre file : {pre_file}",
        f"Post file: {post_file}",
        f"CSV file : {csv_file}",
        "",
        f"Reference slit          : {stats['ref_slit']}",
        f"Total slits             : {stats['n_total']}",
        f"GOOD slits              : {stats['n_good']}",
        f"FALLBACK slits          : {stats['n_fallback']}",
        f"Clip threshold (nm)     : {clip_nm:.3f}",
        "",
        f"Median measured shift   : {stats['median_measured_nm']:+.4f} nm",
        f"Robust sigma measured   : {stats['robust_sigma_measured_nm']:.4f} nm",
        f"Median applied shift    : {stats['median_applied_nm']:+.4f} nm",
        f"Robust sigma applied    : {stats['robust_sigma_applied_nm']:.4f} nm",
        f"Fallback median applied : {stats['fallback_median_nm']:+.4f} nm",
        "",
        f"Step08 flagged S08BAD   : {int((df['S08BAD'] == 1).sum())}",
        f"Step08 flagged S08EMP   : {int((df['S08EMP'] == 1).sum())}",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)
    pdf.savefig(fig)
    plt.close(fig)


def make_shift_plots(pdf, df: pd.DataFrame, clip_nm: float):
    slnum = np.array([slit_num(s) for s in df["slit"]], int)
    good = df["source"] == "GOOD"
    fallback = df["source"] == "FALLBACK"

    fig = plt.figure(figsize=(11, 8.5))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    x = df["measured_shift_nm"].values
    x = x[np.isfinite(x)]
    if x.size:
        ax1.hist(x, bins=20)
    ax1.axvline(0.0, linestyle="--")
    ax1.axvline(+clip_nm, linestyle=":")
    ax1.axvline(-clip_nm, linestyle=":")
    ax1.set_xlabel("Measured shift (nm)")
    ax1.set_ylabel("N slits")
    ax1.set_title("Measured Step10.1 shifts")

    ax2.scatter(slnum[good], df.loc[good, "measured_shift_nm"], label="GOOD")
    ax2.scatter(slnum[fallback], df.loc[fallback, "measured_shift_nm"], label="FALLBACK")
    ax2.axhline(0.0, linestyle="--")
    ax2.axhline(+clip_nm, linestyle=":")
    ax2.axhline(-clip_nm, linestyle=":")
    ax2.set_xlabel("Slit number")
    ax2.set_ylabel("Measured shift (nm)")
    ax2.set_title("Measured shift by slit")
    ax2.legend(loc="best", fontsize=8)

    y = df["applied_shift_nm"].values
    y = y[np.isfinite(y)]
    if y.size:
        ax3.hist(y, bins=20)
    ax3.axvline(0.0, linestyle="--")
    ax3.set_xlabel("Applied shift (nm)")
    ax3.set_ylabel("N slits")
    ax3.set_title("Applied shifts inferred from LAMBDA_NM")

    ax4.scatter(slnum[good], df.loc[good, "applied_shift_nm"], label="GOOD")
    ax4.scatter(slnum[fallback], df.loc[fallback, "applied_shift_nm"], label="FALLBACK")
    ax4.axhline(0.0, linestyle="--")
    ax4.set_xlabel("Slit number")
    ax4.set_ylabel("Applied shift (nm)")
    ax4.set_title("Applied shift by slit")
    ax4.legend(loc="best", fontsize=8)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def make_table_pages(pdf, df: pd.DataFrame):
    cols = ["slit", "source", "measured_shift_nm", "applied_shift_nm", "r", "nwin", "S08BAD", "S08EMP"]
    worst = df.copy()
    worst["abs_meas"] = np.abs(worst["measured_shift_nm"])
    worst = worst.sort_values(["abs_meas", "slit"], ascending=[False, True]).head(15)

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title("Worst slits by |measured_shift_nm|", pad=12)

    cell_text = []
    for _, row in worst.iterrows():
        cell_text.append([
            row["slit"],
            row["source"],
            f"{row['measured_shift_nm']:+.3f}" if np.isfinite(row["measured_shift_nm"]) else "nan",
            f"{row['applied_shift_nm']:+.3f}" if np.isfinite(row["applied_shift_nm"]) else "nan",
            f"{row['r']:.3f}" if np.isfinite(row["r"]) else "nan",
            f"{int(row['nwin'])}" if np.isfinite(row["nwin"]) else "nan",
            f"{int(row['S08BAD'])}",
            f"{int(row['S08EMP'])}",
        ])

    tab = ax.table(cellText=cell_text, colLabels=cols, loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.4)
    pdf.savefig(fig)
    plt.close(fig)


def pick_example_slits(df: pd.DataFrame, nexamples: int) -> list[str]:
    examples = []
    good = df[df["source"] == "GOOD"].copy()
    fallback = df[df["source"] == "FALLBACK"].copy()

    if len(good):
        good = good.assign(abs_meas=np.abs(good["measured_shift_nm"]))
        examples += list(good.sort_values("abs_meas", ascending=False)["slit"].head(max(1, nexamples // 2)))
    if len(fallback):
        fallback = fallback.assign(abs_meas=np.abs(fallback["measured_shift_nm"]))
        examples += list(fallback.sort_values("abs_meas", ascending=False)["slit"].head(max(1, nexamples - len(examples))))

    # Deduplicate while preserving order
    out = []
    seen = set()
    for s in examples:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out[:nexamples]


def make_overlay_pages(pdf, pre_file: Path, post_file: Path, df: pd.DataFrame, ref_slit: str, hp_width: int, nexamples: int):
    examples = pick_example_slits(df, nexamples)
    if not examples:
        return

    with fits.open(pre_file) as hpre, fits.open(post_file) as hpost:
        slits_pre = {norm_slit(h.name): h for h in hpre[1:] if is_slit(h.name)}
        slits_post = {norm_slit(h.name): h for h in hpost[1:] if is_slit(h.name)}
        if ref_slit not in slits_post:
            ref_slit = next(iter(slits_post.keys()))

        ref_arr = get_slit_arrays(slits_post[ref_slit])
        if ref_arr is None:
            return

        for slit in examples:
            arr_pre = get_slit_arrays(slits_pre[slit]) if slit in slits_pre else None
            arr_post = get_slit_arrays(slits_post[slit]) if slit in slits_post else None
            if arr_pre is None or arr_post is None:
                continue

            fig = plt.figure(figsize=(11, 8.5))
            axes = [fig.add_subplot(3, 2, i + 1) for i in range(min(6, len(DEFAULT_WINDOWS_NM) + 1))]
            meta = df.loc[df["slit"] == slit].iloc[0]
            fig.suptitle(
                f"{slit}   source={meta['source']}   measured={meta['measured_shift_nm']:+.3f} nm   applied={meta['applied_shift_nm']:+.3f} nm   ref={ref_slit}",
                fontsize=12,
            )

            for ax, (lo, hi) in zip(axes, DEFAULT_WINDOWS_NM):
                grid = np.arange(lo, hi + 0.01, 0.01)
                yref = interp_to_grid(ref_arr["lam"], ref_arr["sky"], grid)
                ypre = interp_to_grid(arr_pre["lam"], arr_pre["sky"], grid)
                ypost = interp_to_grid(arr_post["lam"], arr_post["sky"], grid)

                yref = highpass_running_median(yref, hp_width)
                ypre = highpass_running_median(ypre, hp_width)
                ypost = highpass_running_median(ypost, hp_width)

                def _norm(y):
                    ok = np.isfinite(y)
                    if ok.sum() < 10:
                        return y
                    s = np.nanstd(y[ok])
                    return y / s if np.isfinite(s) and s > 0 else y

                ax.plot(grid, _norm(ypre), label="pre")
                ax.plot(grid, _norm(ypost), label="post")
                ax.plot(grid, _norm(yref), label=f"ref {ref_slit}")
                ax.set_title(f"{lo:.0f}–{hi:.0f} nm")
                ax.set_xlabel("Wavelength (nm)")
                ax.set_ylabel("HP SKY (norm)")
                ax.legend(loc="best", fontsize=8)

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)


def main():
    args = parse_args()
    pre_file = Path(args.pre)
    post_file = Path(args.post)
    csv_file = Path(args.csv)
    outcsv = Path(args.outcsv)
    outpdf = Path(args.outpdf)

    for p in [pre_file, post_file, csv_file]:
        if not p.exists():
            raise FileNotFoundError(p)

    outcsv.parent.mkdir(parents=True, exist_ok=True)
    outpdf.parent.mkdir(parents=True, exist_ok=True)

    print("PRE   =", pre_file)
    print("POST  =", post_file)
    print("CSV   =", csv_file)
    print("OUTCSV=", outcsv)
    print("OUTPDF=", outpdf)
    print("CLIP_NM=", args.clip)

    df, stats = build_summary(pre_file, post_file, csv_file, clip_nm=args.clip)
    df.to_csv(outcsv, index=False)

    with PdfPages(outpdf) as pdf:
        make_summary_page(pdf, df, stats, pre_file, post_file, csv_file, args.clip)
        make_shift_plots(pdf, df, args.clip)
        make_table_pages(pdf, df)
        make_overlay_pages(pdf, pre_file, post_file, df, stats["ref_slit"], args.hp_width, args.nexamples)

    print("Wrote:", outcsv)
    print("Wrote:", outpdf)
    print(f"Reference: {stats['ref_slit']}")
    print(f"Total slits: {stats['n_total']}")
    print(f"GOOD={stats['n_good']}  FALLBACK={stats['n_fallback']}")
    print(f"Median measured shift: {stats['median_measured_nm']:+.4f} nm")
    print(f"Robust sigma measured: {stats['robust_sigma_measured_nm']:.4f} nm")
    print(f"Median applied shift:  {stats['median_applied_nm']:+.4f} nm")
    print(f"Fallback median:      {stats['fallback_median_nm']:+.4f} nm")


if __name__ == "__main__":
    main()
