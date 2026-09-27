#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QC residual slit-to-slit wavelength offsets after Step08c.

This diagnostic reuses the Step10a O2-band normalization and correlation
machinery, but keeps slit identities and reports *relative* residual shifts
about a robust ensemble reference.  It does not modify any science product and
must not be used to impose a global wavelength zero point.

The intent is to decide whether any per-slit Step07i-style correction is still
justified after the Step08c detector-window mapping is fixed.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.step10_telluric.step10a_build_telluric_template import (
    A_HI,
    A_LO,
    A_SB1,
    A_SB2,
    B_HI,
    B_LO,
    B_SB1,
    B_SB2,
    NGRID_A,
    NGRID_B,
    estimate_shift_corr,
    robust_median_stack,
    window_vec,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="QC residual O2 cross-correlation shifts after corrected Step08c"
    )
    p.add_argument("--infile", type=Path, required=True,
                   help="Corrected Step08c wavelength-attached MEF")
    p.add_argument("--outdir", type=Path, required=True,
                   help="Directory for CSV/PNG diagnostics")
    return p.parse_args()


def slit_num(name: str) -> int:
    try:
        return int(name.upper().replace("SLIT", ""))
    except Exception:
        return 10**9


def science_flux(hdu):
    names = {n.upper(): n for n in hdu.columns.names}
    for key in ("FLUX_APCORR", "FLUX"):
        if key in names:
            x = np.asarray(hdu.data[names[key]], float)
            if np.isfinite(x).sum() >= 20:
                return x, key
    return None, ""


def science_var(hdu):
    names = {n.upper(): n for n in hdu.columns.names}
    for key in ("VAR_APCORR", "VAR", "VAR_ADU_S2"):
        if key in names:
            x = np.asarray(hdu.data[names[key]], float)
            if np.isfinite(x).sum() >= 20:
                return x
    return None


def collect_band_vectors(infile: Path):
    grid_a = np.linspace(A_LO, A_HI, NGRID_A)
    grid_b = np.linspace(B_LO, B_HI, NGRID_B)
    rows = []

    with fits.open(infile, memmap=False) as hdul:
        slits = [h for h in hdul[1:] if (h.name or "").upper().startswith("SLIT")]
        slits.sort(key=lambda h: slit_num(h.name))

        for h in slits:
            if h.data is None or not hasattr(h, "columns"):
                continue
            if int(h.header.get("S08USE", 1)) == 0:
                continue
            names = {n.upper(): n for n in h.columns.names}
            if "LAMBDA_NM" not in names:
                continue

            lam = np.asarray(h.data[names["LAMBDA_NM"]], float)
            flux, flux_col = science_flux(h)
            if flux is None:
                continue
            var = science_var(h)

            va, snra, da = window_vec(lam, flux, var, A_LO, A_HI, A_SB1, A_SB2, grid_a)
            vb, snrb, db = window_vec(lam, flux, var, B_LO, B_HI, B_SB1, B_SB2, grid_b)

            oka = va is not None and np.isfinite(da) and (0.08 < da < 0.80)
            okb = vb is not None and np.isfinite(db) and (0.03 < db < 0.60)
            if not (oka or okb):
                continue

            rows.append({
                "slit": h.name.upper(),
                "flux_col": flux_col,
                "vec_a": va if oka else None,
                "snr_a": float(snra) if np.isfinite(snra) else np.nan,
                "depth_a": float(da) if np.isfinite(da) else np.nan,
                "vec_b": vb if okb else None,
                "snr_b": float(snrb) if np.isfinite(snrb) else np.nan,
                "depth_b": float(db) if np.isfinite(db) else np.nan,
            })

    return rows, grid_a, grid_b


def iterative_relative_shifts(rows, vec_key, grid, max_shift_nm, min_peak, use_gradient):
    ids = [i for i, r in enumerate(rows) if r[vec_key] is not None]
    if len(ids) < 5:
        return {}

    ref = robust_median_stack(np.asarray([rows[i][vec_key] for i in ids], float))
    accepted = ids

    # Two ensemble-reference refinements.  The reference is always built from
    # all correlation-gated spectra, not from the first few slit IDs.
    for _ in range(2):
        aligned = []
        new_accepted = []
        for i in accepted:
            v = rows[i][vec_key]
            shift, peak = estimate_shift_corr(
                v, ref, grid,
                max_shift_nm=max_shift_nm,
                use_gradient=use_gradient,
            )
            if not np.isfinite(peak) or peak < min_peak:
                continue
            vv = np.interp(grid, grid + shift, v, left=np.nan, right=np.nan)
            if np.isfinite(vv).sum() < 0.8 * vv.size:
                continue
            aligned.append(vv)
            new_accepted.append(i)
        if len(aligned) < 5:
            return {}
        ref = robust_median_stack(np.asarray(aligned, float))
        accepted = new_accepted

    raw = {}
    for i in accepted:
        shift, peak = estimate_shift_corr(
            rows[i][vec_key], ref, grid,
            max_shift_nm=max_shift_nm,
            use_gradient=use_gradient,
        )
        if np.isfinite(peak) and peak >= min_peak:
            raw[i] = (float(shift), float(peak))

    if not raw:
        return {}
    center = float(np.nanmedian([v[0] for v in raw.values()]))
    return {i: (shift - center, peak) for i, (shift, peak) in raw.items()}


def mad_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def summarize(label, vals):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        print(f"{label}: no accepted shifts")
        return
    print(
        f"{label}: N={vals.size} median={np.median(vals):+.4f} nm "
        f"sigma_MAD={mad_sigma(vals):.4f} nm "
        f"min/max={np.min(vals):+.4f}/{np.max(vals):+.4f} nm "
        f"N(|dlam|>=0.5)={np.sum(np.abs(vals)>=0.5)} "
        f"N(|dlam|>=1.0)={np.sum(np.abs(vals)>=1.0)}"
    )


def main():
    args = parse_args()
    if not args.infile.exists():
        raise FileNotFoundError(args.infile)
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows, grid_a, grid_b = collect_band_vectors(args.infile)
    if not rows:
        raise RuntimeError("No usable S08USE=1 slit spectra with O2-band coverage")

    sh_a = iterative_relative_shifts(
        rows, "vec_a", grid_a, max_shift_nm=4.0, min_peak=0.15, use_gradient=True
    )
    sh_b = iterative_relative_shifts(
        rows, "vec_b", grid_b, max_shift_nm=1.0, min_peak=0.08, use_gradient=False
    )

    out_csv = args.outdir / "qc_step08c_residual_xcorr.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "slit", "flux_col",
            "depth_a", "snr_a", "shift_a_rel_nm", "peak_a",
            "depth_b", "snr_b", "shift_b_rel_nm", "peak_b",
            "a_minus_b_nm",
        ])
        for i, r in enumerate(rows):
            a = sh_a.get(i, (np.nan, np.nan))
            b = sh_b.get(i, (np.nan, np.nan))
            diff = a[0] - b[0] if np.isfinite(a[0]) and np.isfinite(b[0]) else np.nan
            w.writerow([
                r["slit"], r["flux_col"],
                r["depth_a"], r["snr_a"], a[0], a[1],
                r["depth_b"], r["snr_b"], b[0], b[1], diff,
            ])

    a_vals = [x[0] for x in sh_a.values()]
    b_vals = [x[0] for x in sh_b.values()]
    summarize("A-band relative xcorr", a_vals)
    summarize("B-band relative xcorr", b_vals)

    both = []
    for i in range(len(rows)):
        if i in sh_a and i in sh_b:
            both.append(sh_a[i][0] - sh_b[i][0])
    summarize("A-B residual difference", both)

    # Plot residual shift versus slit ID.  This is QC only; no correction table
    # is generated automatically.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    xa, ya, xb, yb = [], [], [], []
    for i, r in enumerate(rows):
        x = slit_num(r["slit"])
        if i in sh_a:
            xa.append(x); ya.append(sh_a[i][0])
        if i in sh_b:
            xb.append(x); yb.append(sh_b[i][0])
    if xa:
        ax.scatter(xa, ya, marker="o", label="O2 A band")
    if xb:
        ax.scatter(xb, yb, marker="x", label="O2 B band")
    ax.axhline(0.0, linewidth=1.0)
    ax.axhline(+0.5, linestyle="--", linewidth=0.8)
    ax.axhline(-0.5, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Slit ID")
    ax.set_ylabel("Relative wavelength shift (nm)")
    ax.set_title("Step08c QC — residual slit-to-slit O2 cross-correlation shifts")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_png = args.outdir / "qc_step08c_residual_xcorr.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"Input: {args.infile}")
    print(f"Usable S08USE=1 slits entering O2 QC: {len(rows)}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_png}")
    print("NOTE: shifts are ensemble-relative only; this QC does not set an absolute zero point or apply Step07i corrections.")


if __name__ == "__main__":
    main()
