#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:49:57 2026

@author: robberto

QC07_2_unified.py

Unified QC for Step07.2 line-shift finding.

Checks:
  1) aligned stack image
  2) per-slit line profiles, raw and shifted
  3) residuals against the stack median profile
  4) bright-line shift vs xcorr shift
  5) ranked outlier table

Usage examples:
runfile("QC07_2_unified.py", args="--traceset EVEN")
runfile("QC07_2_unified.py", args="--traceset ODD")
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


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
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


def norm1(x):
    x = np.asarray(x, float)
    if not np.any(np.isfinite(x)):
        return x * np.nan
    med = np.nanmedian(x)
    y = x - med
    mx = np.nanmax(np.abs(y))
    if not np.isfinite(mx) or mx <= 0:
        return y
    return y / mx


def integer_shift_1d(arr, dx):
    """Shift 1D array by integer pixels, filling with NaN."""
    arr = np.asarray(arr, float)
    out = np.full_like(arr, np.nan)
    dx = int(dx)
    if dx == 0:
        out[:] = arr
    elif dx > 0:
        out[dx:] = arr[:-dx]
    else:
        out[:dx] = arr[-dx:]
    return out


def xcorr_integer_shift(a, b):
    """Integer shift maximizing cross-correlation, b relative to a."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 5:
        return np.nan

    aa = a[good] - np.nanmedian(a[good])
    bb = b[good] - np.nanmedian(b[good])

    aa = np.nan_to_num(aa)
    bb = np.nan_to_num(bb)

    n = aa.size
    cc = np.fft.irfft(np.fft.rfft(aa) * np.conj(np.fft.rfft(bb)), n=n)
    dx = int(np.argmax(cc))
    if dx > n // 2:
        dx -= n
    return dx


def read_shift_csv(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def parse_num(x):
    try:
        return float(x)
    except Exception:
        return np.nan


# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--traceset", required=True, choices=["EVEN", "ODD", "even", "odd"])
ap.add_argument("--csv", default=None, help="Optional explicit Arc_shifts_initial_<SET>.csv")
ap.add_argument("--stack", default=None, help="Optional explicit Arc_stack_aligned_<SET>.fits")
ap.add_argument("--arc1d", default=None, help="Optional explicit Step07.1 input MEF")
ap.add_argument("--n-show", type=int, default=12, help="Number of slits to show in overlays")
ap.add_argument("--write-csv", action="store_true", help="Write ranked QC CSV")
args = ap.parse_args()

TRACESET = args.traceset.upper()
st07 = Path(config.ST07_WAVECAL)

shift_csv = first_existing([
    Path(args.csv) if args.csv else None,
    st07 / f"Arc_shifts_initial_{TRACESET}.csv",
])

stack_fits = first_existing([
    Path(args.stack) if args.stack else None,
    st07 / f"Arc_stack_aligned_{TRACESET}.fits",
])

arc1d_fits = first_existing([
    Path(args.arc1d) if args.arc1d else None,
    *sorted(st07.glob(f"arc_1d{TRACESET}.fits")),
])

if shift_csv is None:
    raise FileNotFoundError(f"Could not find Arc_shifts_initial_{TRACESET}.csv")
if stack_fits is None:
    raise FileNotFoundError(f"Could not find Arc_stack_aligned_{TRACESET}.fits")
if arc1d_fits is None:
    raise FileNotFoundError(f"Could not find arc_1d{TRACESET}.fits")

# ------------------------------------------------------------
# Load shift CSV
# ------------------------------------------------------------
rows = read_shift_csv(shift_csv)
if not rows:
    raise RuntimeError(f"No rows in {shift_csv}")

# expected useful columns from current Step07.2
# keep this tolerant to exact naming
for r in rows:
    r["_slit"] = str(r.get("slit", r.get("SLIT", r.get("extname", r.get("EXTNAME", ""))))).strip().upper()
    r["_shift"] = parse_num(r.get("shift_pix", r.get("SHIFT_PIX", r.get("shift", r.get("SHIFT", np.nan)))))
    r["_peakx"] = parse_num(r.get("peak_x", r.get("PEAK_X", r.get("xpeak", np.nan))))
    r["_prom"] = parse_num(r.get("prominence", r.get("PROMINENCE", np.nan)))
    r["_width"] = parse_num(r.get("width", r.get("WIDTH", np.nan)))
    r["_ok"] = str(r.get("ok", r.get("OK", "True"))).strip().lower() not in ("false", "0", "no")

rows = [r for r in rows if r["_slit"] != ""]
slit_names = [r["_slit"] for r in rows]

# ------------------------------------------------------------
# Load aligned stack
# ------------------------------------------------------------
stack = fits.getdata(stack_fits).astype(float)
hdr_stack = fits.getheader(stack_fits)

# interpret stack shape
# assume shape = (nslits, npix)
if stack.ndim != 2:
    raise ValueError(f"Expected 2D aligned stack in {stack_fits}, got shape {stack.shape}")

nstack, npix = stack.shape

# ------------------------------------------------------------
# Load Step07.1 input 1D products for raw overlays
# Support both image HDUs and tables
# ------------------------------------------------------------
raw_profiles = {}

with fits.open(arc1d_fits) as hdul:
    for h in hdul[1:]:
        nm = (h.header.get("EXTNAME") or h.name or "").strip().upper()
        if nm not in slit_names:
            continue

        data = h.data
        if data is None:
            continue

        if hasattr(data, "names") and data.names is not None:
            cols = list(data.names)
            col_flux = None
            for cand in ["FLUX", "ARC", "SPEC", "COUNTS", "VALUE"]:
                if cand in cols:
                    col_flux = cand
                    break
            if col_flux is None and len(cols) > 0:
                col_flux = cols[1] if len(cols) > 1 else cols[0]
            flux = np.asarray(data[col_flux], float)
        else:
            arr = np.asarray(data, float)
            if arr.ndim == 2:
                flux = np.nanmean(arr, axis=1)
            elif arr.ndim == 1:
                flux = arr
            else:
                continue

        raw_profiles[nm] = np.asarray(flux, float)

# ------------------------------------------------------------
# Build aligned/raw comparison arrays
# ------------------------------------------------------------
master = np.nanmedian(stack, axis=0)

qc_rows = []
for i, r in enumerate(rows):
    slit = r["_slit"]

    # aligned profile from stack
    if i < nstack:
        aligned = np.asarray(stack[i], float)
    else:
        aligned = np.full(npix, np.nan)

    raw = raw_profiles.get(slit, None)

    # xcorr diagnostic against master
    xcorr_dx = xcorr_integer_shift(master, aligned)
    dx_bright = r["_shift"]

    resid = aligned - master
    resid_sig = robust_sigma(resid)

    med_flux = float(np.nanmedian(aligned)) if np.any(np.isfinite(aligned)) else np.nan
    sig_flux = float(robust_sigma(aligned))
    frac_good = float(np.mean(np.isfinite(aligned))) if aligned.size else np.nan

    qc_rows.append({
        "SLIT": slit,
        "SHIFT_BRIGHT": dx_bright,
        "SHIFT_XCORR": xcorr_dx,
        "DELTA_SHIFT": xcorr_dx - dx_bright if np.isfinite(xcorr_dx) and np.isfinite(dx_bright) else np.nan,
        "PROMINENCE": r["_prom"],
        "WIDTH": r["_width"],
        "RESID_SIG": resid_sig,
        "MED_FLUX": med_flux,
        "SIG_FLUX": sig_flux,
        "FRAC_GOOD": frac_good,
        "OK": r["_ok"],
    })

# rank worst by mismatch + residual
def _score(rr):
    a = 0.0 if not np.isfinite(rr["DELTA_SHIFT"]) else abs(rr["DELTA_SHIFT"])
    b = 0.0 if not np.isfinite(rr["RESID_SIG"]) else rr["RESID_SIG"]
    c = 0.0 if not np.isfinite(rr["WIDTH"]) else 0.1 * rr["WIDTH"]
    d = 0.0 if rr["OK"] else 5.0
    return a + b + d + c

qc_rows = sorted(qc_rows, key=_score, reverse=True)

# ------------------------------------------------------------
# Figure 1: aligned stack + residuals
# ------------------------------------------------------------
v1, v2 = robust_limits(stack, p_lo=2, p_hi=99)
vr1, vr2 = robust_limits(stack - master[None, :], p_lo=2, p_hi=98)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(stack, origin="lower", aspect="auto", vmin=v1, vmax=v2)#, cmap="gray")
ax1.set_title(f"Aligned stack — {TRACESET}")
ax1.set_xlabel("Pixel")
ax1.set_ylabel("Slit index")
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(stack - master[None, :], origin="lower", aspect="auto", vmin=vr1, vmax=vr2)#, cmap="gray")
ax2.set_title("Residuals vs median master")
ax2.set_xlabel("Pixel")
ax2.set_ylabel("Slit index")
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(master, color="black", lw=1.2)
ax3.set_title("Median master profile")
ax3.set_xlabel("Pixel")
ax3.set_ylabel("Flux")

# shift diagnostics
ax4 = fig.add_subplot(gs[1, 0])
bright = np.array([r["SHIFT_BRIGHT"] for r in qc_rows], float)
xcorr = np.array([r["SHIFT_XCORR"] for r in qc_rows], float)
idx = np.arange(len(qc_rows))
ax4.plot(idx, bright, "o-", ms=3, lw=1, label="bright-line shift")
ax4.plot(idx, xcorr, "o-", ms=3, lw=1, label="xcorr shift")
ax4.set_title("Shift per slit")
ax4.set_xlabel("Ranked slit index")
ax4.set_ylabel("Shift (pix)")
ax4.legend(fontsize=8)

ax5 = fig.add_subplot(gs[1, 1])
delta = np.array([r["DELTA_SHIFT"] for r in qc_rows], float)
ax5.plot(idx, delta, "o", ms=3)
ax5.axhline(0, color="0.6", lw=0.8)
ax5.set_title("xcorr - bright-line shift")
ax5.set_xlabel("Ranked slit index")
ax5.set_ylabel("Δ shift (pix)")

ax6 = fig.add_subplot(gs[1, 2])
resid_sig = np.array([r["RESID_SIG"] for r in qc_rows], float)
ax6.plot(idx, resid_sig, "o", ms=3)
ax6.set_title("Residual sigma vs master")
ax6.set_xlabel("Ranked slit index")
ax6.set_ylabel("robust sigma")

fig.suptitle(
    f"QC07.2 unified — {TRACESET}  |  "
    f"N slits={len(qc_rows)}  |  "
    f"median Δshift={np.nanmedian(delta):.3f}  "
    f"σ(Δshift)={robust_sigma(delta):.3f}",
    fontsize=13
)
if args.show_plots:
    plt.show()
else:
    plt.close()

# ------------------------------------------------------------
# Figure 2: raw and shifted overlays for selected slits
# ------------------------------------------------------------
nshow = min(args.n_show, len(qc_rows))
chosen = qc_rows[:nshow]

fig2 = plt.figure(figsize=(15, 3.2 * int(np.ceil(nshow / 3))))
gs2 = fig2.add_gridspec(int(np.ceil(nshow / 3)), 3, hspace=0.35, wspace=0.25)

for i, rr in enumerate(chosen, start=1):
    slit = rr["SLIT"]
    raw = raw_profiles.get(slit, None)
    dx = rr["SHIFT_BRIGHT"]

    ax = fig2.add_subplot(gs2[(i - 1) // 3, (i - 1) % 3])

    if raw is not None:
        raw_n = norm1(raw)
        shifted_n = norm1(integer_shift_1d(raw, int(np.round(dx))) if np.isfinite(dx) else raw)
        master_n = norm1(master)

        xraw = np.arange(raw_n.size)
        xmst = np.arange(master_n.size)

        ax.plot(xraw, raw_n, lw=0.8, color="0.6", label="raw")
        ax.plot(xraw, shifted_n, lw=1.0, color="tab:blue", label="shifted")
        ax.plot(xmst, master_n, lw=1.0, color="black", label="master")
    else:
        ax.text(0.5, 0.5, "no raw profile", ha="center", va="center", transform=ax.transAxes)

    ax.set_title(
        f"{slit}\n"
        f"shift={rr['SHIFT_BRIGHT']:.1f}  xcorr={rr['SHIFT_XCORR']:.1f}\n"
        f"Δ={rr['DELTA_SHIFT']:.1f}  resid={rr['RESID_SIG']:.3f}",
        fontsize=8
    )
    ax.set_xlabel("Pixel")
    ax.set_ylabel("norm flux")
    if i == 1:
        ax.legend(fontsize=7)

fig2.suptitle(f"QC07.2 overlays — {TRACESET}", fontsize=13)
if args.show_plots:
    plt.show()
else:
    plt.close()

# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------
print()
print("====================================================")
print(f"QC07.2 unified summary — {TRACESET}")
print("====================================================")
print("Shift CSV :", shift_csv)
print("Stack FITS:", stack_fits)
print("Arc1D FITS:", arc1d_fits)
print()
print("Worst slits:")
print("SLIT      SHIFT   XCORR   DELTA   PROM   WIDTH   RESID_SIG   OK")
for rr in qc_rows[:min(15, len(qc_rows))]:
    print(
        f"{rr['SLIT']:8s}  "
        f"{rr['SHIFT_BRIGHT']:5.1f}  "
        f"{rr['SHIFT_XCORR']:5.1f}  "
        f"{rr['DELTA_SHIFT']:6.2f}  "
        f"{rr['PROMINENCE']:5.1f}  "
        f"{rr['WIDTH']:6.2f}  "
        f"{rr['RESID_SIG']:9.3f}  "
        f"{str(rr['OK']):>5s}"
    )
print("====================================================")

if args.write_csv:
    out_csv = st07 / f"QC07_2_ranked_{TRACESET}.csv"
    keys = list(qc_rows[0].keys()) if qc_rows else []
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(qc_rows)
    print("Wrote:", out_csv)