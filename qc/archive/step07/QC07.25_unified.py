#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:39:38 2026

@author: robberto

QC07_25_unified.py

Unified QC for Step07.25 refine-shifts-and-stack.

Checks:
  1) initial vs final aligned stack
  2) line centering before/after refinement
  3) SHIFT0 / DSHIFT / SHIFT_FINAL / CORR diagnostics
  4) ranked worst slits
  5) optional CSV output

Usage:
runfile("QC07_25_unified.py", args="--set EVEN")
runfile("QC07_25_unified.py", args="--set ODD")
runfile("QC07_25_unified.py", args="--set EVEN --write-csv")
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
import config


def robust_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def robust_limits(img, p_lo=2.0, p_hi=98.0):
    arr = np.asarray(img, float)
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return 0.0, 1.0
    lo = np.nanpercentile(v, p_lo)
    hi = np.nanpercentile(v, p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        med = np.nanmedian(v)
        sig = robust_sigma(v)
        if not np.isfinite(sig) or sig <= 0:
            sig = 1.0
        return med - 2 * sig, med + 4 * sig
    return float(lo), float(hi)


def first_existing(paths):
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def parse_num(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_col(row, candidates):
    keys = list(row.keys())
    lowmap = {k.lower(): k for k in keys}
    for c in candidates:
        if c.lower() in lowmap:
            return lowmap[c.lower()]
    # contains fallback
    for c in candidates:
        for k in keys:
            if c.lower() in k.lower():
                return k
    return None


def weighted_centroid_1d(y, x0=None, halfwin=8):
    y = np.asarray(y, float)
    n = y.size
    x = np.arange(n, dtype=float)

    if x0 is None or not np.isfinite(x0):
        if not np.any(np.isfinite(y)):
            return np.nan
        x0 = float(np.nanargmax(y))

    i0 = int(round(x0))
    a = max(0, i0 - halfwin)
    b = min(n, i0 + halfwin + 1)

    seg = y[a:b].astype(float)
    xx = x[a:b]

    if not np.any(np.isfinite(seg)):
        return np.nan

    base = np.nanmedian(seg)
    w = seg - base
    w[~np.isfinite(w)] = 0.0
    w[w < 0] = 0.0

    s = np.sum(w)
    if s <= 0:
        return np.nan

    return float(np.sum(xx * w) / s)


def score_row(r):
    terms = [
        0.0 if not np.isfinite(r["FINAL_RESID"]) else abs(r["FINAL_RESID"]),
        0.0 if not np.isfinite(r["DELTA_IMPROVE"]) else max(0.0, -r["DELTA_IMPROVE"]),
        0.0 if not np.isfinite(r["CORR"]) else 2.0 * max(0.0, 1.0 - r["CORR"]),
        0.0 if not np.isfinite(r["DSHIFT"]) else 0.2 * abs(r["DSHIFT"]),
    ]
    return float(np.sum(terms))


ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True, choices=["EVEN", "ODD", "even", "odd"])
ap.add_argument("--csv-final", default=None)
ap.add_argument("--stack-initial", default=None)
ap.add_argument("--stack-final", default=None)
ap.add_argument("--n-show", type=int, default=12)
ap.add_argument("--write-csv", action="store_true")
args = ap.parse_args()

trace_set = args.set.upper()
st07 = Path(config.ST07_WAVECAL)

csv_final = first_existing([
    Path(args.csv_final) if args.csv_final else None,
    st07 / f"Arc_shifts_final_{trace_set}.csv",
])

stack_initial = first_existing([
    Path(args.stack_initial) if args.stack_initial else None,
    st07 / f"Arc_stack_aligned_{trace_set}.fits",
])

stack_final = first_existing([
    Path(args.stack_final) if args.stack_final else None,
    st07 / f"Arc_stack_final_even/odd.fits",
])

if csv_final is None:
    raise FileNotFoundError(f"Could not find Arc_shifts_final_{trace_set}.csv")
if stack_initial is None:
    raise FileNotFoundError(f"Could not find Arc_stack_aligned_{trace_set}.fits")
if stack_final is None:
    raise FileNotFoundError(f"Could not find Arc_stack_final_even/odd.fits")

rows = read_csv_rows(csv_final)
if not rows:
    raise RuntimeError(f"No rows in {csv_final}")

c_slit = find_col(rows[0], ["SLIT", "slit", "EXTNAME", "extname"])
c_shift0 = find_col(rows[0], ["SHIFT0", "shift0", "shift_initial", "initial_shift"])
c_dshift = find_col(rows[0], ["DSHIFT", "dshift", "delta_shift"])
c_shiftf = find_col(rows[0], ["SHIFT_FINAL", "shift_final", "shift"])
c_corr = find_col(rows[0], ["CORR", "corr", "xcorr", "correlation"])

csv_info = []
for r in rows:
    slit = str(r[c_slit]).strip().upper() if c_slit else "UNKNOWN"
    shift0 = parse_num(r[c_shift0]) if c_shift0 else np.nan
    dshift = parse_num(r[c_dshift]) if c_dshift else np.nan
    shiftf = parse_num(r[c_shiftf]) if c_shiftf else np.nan
    corr = parse_num(r[c_corr]) if c_corr else np.nan
    csv_info.append({
        "SLIT": slit,
        "SHIFT0": shift0,
        "DSHIFT": dshift,
        "SHIFT_FINAL": shiftf,
        "CORR": corr,
    })

stack0 = fits.getdata(stack_initial).astype(float)
stackf = fits.getdata(stack_final).astype(float)
hdrf = fits.getheader(stack_final)

if stack0.ndim != 2 or stackf.ndim != 2:
    raise ValueError("Initial/final stacks must be 2D arrays")
if stack0.shape != stackf.shape:
    raise ValueError(f"Stack shape mismatch: {stack0.shape} vs {stackf.shape}")

nslit, npix = stackf.shape
master0 = np.nanmedian(stack0, axis=0)
masterf = np.nanmedian(stackf, axis=0)

# common final line center from master
xpk_final = float(np.nanargmax(masterf))
xpk_initial = float(np.nanargmax(master0))

qc_rows = []
for i in range(min(nslit, len(csv_info))):
    slit = csv_info[i]["SLIT"]
    prof0 = np.asarray(stack0[i], float)
    proff = np.asarray(stackf[i], float)

    cen0 = weighted_centroid_1d(prof0, xpk_initial, halfwin=8)
    cenf = weighted_centroid_1d(proff, xpk_final, halfwin=8)

    resid0 = cen0 - xpk_initial if np.isfinite(cen0) else np.nan
    residf = cenf - xpk_final if np.isfinite(cenf) else np.nan

    qc_rows.append({
        "SLIT": slit,
        "SHIFT0": csv_info[i]["SHIFT0"],
        "DSHIFT": csv_info[i]["DSHIFT"],
        "SHIFT_FINAL": csv_info[i]["SHIFT_FINAL"],
        "CORR": csv_info[i]["CORR"],
        "INITIAL_CEN": cen0,
        "FINAL_CEN": cenf,
        "INITIAL_RESID": resid0,
        "FINAL_RESID": residf,
        "DELTA_IMPROVE": (abs(resid0) - abs(residf))
        if np.isfinite(resid0) and np.isfinite(residf) else np.nan,
    })

for r in qc_rows:
    r["SCORE"] = score_row(r)

qc_rows = sorted(qc_rows, key=lambda r: r["SCORE"], reverse=True)

# Figures
v0a, v0b = robust_limits(stack0, 2, 99)
vfa, vfb = robust_limits(stackf, 2, 99)
dra, drb = robust_limits(stackf - stack0, 2, 98)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(stack0, origin="lower", aspect="auto", vmin=v0a, vmax=v0b, cmap="gray")
ax1.set_title("Initial aligned stack")
ax1.set_xlabel("Pixel")
ax1.set_ylabel("Slit index")
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(stackf, origin="lower", aspect="auto", vmin=vfa, vmax=vfb, cmap="gray")
ax2.set_title("Final aligned stack")
ax2.set_xlabel("Pixel")
ax2.set_ylabel("Slit index")
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

ax3 = fig.add_subplot(gs[0, 2])
im3 = ax3.imshow(stackf - stack0, origin="lower", aspect="auto", vmin=dra, vmax=drb, cmap="gray")
ax3.set_title("Final - Initial")
ax3.set_xlabel("Pixel")
ax3.set_ylabel("Slit index")
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

idx = np.arange(len(qc_rows))
init_res = np.array([r["INITIAL_RESID"] for r in qc_rows], float)
fin_res = np.array([r["FINAL_RESID"] for r in qc_rows], float)
dshift = np.array([r["DSHIFT"] for r in qc_rows], float)
corr = np.array([r["CORR"] for r in qc_rows], float)

ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(idx, init_res, "o-", ms=3, lw=1, label="initial resid")
ax4.plot(idx, fin_res, "o-", ms=3, lw=1, label="final resid")
ax4.axhline(0, color="0.6", lw=0.8)
ax4.set_title("Line-centroid residual vs master")
ax4.set_xlabel("Ranked slit index")
ax4.set_ylabel("pix")
ax4.legend(fontsize=8)

ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(idx, dshift, "o", ms=3)
ax5.axhline(0, color="0.6", lw=0.8)
ax5.set_title("DSHIFT per slit")
ax5.set_xlabel("Ranked slit index")
ax5.set_ylabel("pix")

ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(idx, corr, "o", ms=3)
ax6.set_title("Correlation metric")
ax6.set_xlabel("Ranked slit index")
ax6.set_ylabel("CORR")

fig.suptitle(
    f"QC07.25 unified — {trace_set}  |  "
    f"median |initial resid|={np.nanmedian(np.abs(init_res)):.3f}  "
    f"median |final resid|={np.nanmedian(np.abs(fin_res)):.3f}",
    fontsize=13
)
if args.show_plots:
            plt.show()
        else:
            plt.close()

# Overlay plots for worst slits
nshow = min(args.n_show, len(qc_rows))
fig2 = plt.figure(figsize=(15, 3.2 * int(np.ceil(nshow / 3))))
gs2 = fig2.add_gridspec(int(np.ceil(nshow / 3)), 3, hspace=0.35, wspace=0.25)

for i, rr in enumerate(qc_rows[:nshow], start=1):
    slit = rr["SLIT"]
    j = next((k for k, x in enumerate(csv_info) if x["SLIT"] == slit), None)
    if j is None or j >= nslit:
        continue

    p0 = stack0[j]
    pf = stackf[j]

    ax = fig2.add_subplot(gs2[(i - 1) // 3, (i - 1) % 3])
    ax.plot(p0 / np.nanmax(np.abs(p0 - np.nanmedian(p0))) if np.any(np.isfinite(p0)) else p0,
            color="0.6", lw=0.8, label="initial")
    ax.plot(pf / np.nanmax(np.abs(pf - np.nanmedian(pf))) if np.any(np.isfinite(pf)) else pf,
            color="tab:blue", lw=1.0, label="final")
    ax.plot(masterf / np.nanmax(np.abs(masterf - np.nanmedian(masterf))),
            color="black", lw=1.0, label="master")
    ax.set_title(
        f"{slit}\n"
        f"S0={rr['SHIFT0']:.1f}  dS={rr['DSHIFT']:.1f}  Sf={rr['SHIFT_FINAL']:.1f}\n"
        f"ri={rr['INITIAL_RESID']:.2f}  rf={rr['FINAL_RESID']:.2f}  C={rr['CORR']:.3f}",
        fontsize=8
    )
    ax.set_xlabel("Pixel")
    ax.set_ylabel("norm flux")
    if i == 1:
        ax.legend(fontsize=7)

fig2.suptitle(f"QC07.25 worst-slit overlays — {trace_set}", fontsize=13)
if args.show_plots:
            plt.show()
        else:
            plt.close()

print()
print("====================================================")
print(f"QC07.25 unified summary — {trace_set}")
print("====================================================")
print("CSV final     :", csv_final)
print("Stack initial :", stack_initial)
print("Stack final   :", stack_final)
print()
print("Worst slits:")
print("SLIT      SHIFT0  DSHIFT  SHIFTF   CORR   RINIT   RFINAL   IMPROVE")
for r in qc_rows[:min(15, len(qc_rows))]:
    print(
        f"{r['SLIT']:8s}  "
        f"{r['SHIFT0']:6.1f}  "
        f"{r['DSHIFT']:6.1f}  "
        f"{r['SHIFT_FINAL']:6.1f}  "
        f"{r['CORR']:6.3f}  "
        f"{r['INITIAL_RESID']:7.3f}  "
        f"{r['FINAL_RESID']:7.3f}  "
        f"{r['DELTA_IMPROVE']:8.3f}"
    )
print("====================================================")

if args.write_csv:
    out_csv = st07 / f"QC07_25_ranked_{trace_set}.csv"
    keys = list(qc_rows[0].keys()) if qc_rows else []
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(qc_rows)
    print("Wrote:", out_csv)