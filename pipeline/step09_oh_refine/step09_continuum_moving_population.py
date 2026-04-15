#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
step09_continuum_moving_population.py

Bidirectional moving-population continuum fitter for SAMOS 1D spectra.

Purpose
-------
Estimate the broad continuum in extracted 1D spectra while rejecting:
- narrow OH emission spikes
- deep narrow dips
- telluric absorption bands
- dense OH-forest bias

Method
------
1. Build local continuum anchors in moving wavelength windows.
2. In each window:
   - exclude telluric / protected regions completely
   - apply stronger rejection in OH-forest regions
   - reject high positive spikes
   - trim the extreme low tail
   - compute a robust local anchor
3. Interpolate anchors with PCHIP.
4. Solve both forward and backward in wavelength.
5. Blend the two continua smoothly to suppress edge artifacts.

Notes
-----
- The default exclusion range 655.5:657.5 protects Halpha at 656.3 nm.
- This script estimates only the broad continuum.
- OH subtraction is handled downstream.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.interpolate import PchipInterpolator


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    sig = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(x)
    return float(sig) if np.isfinite(sig) and sig > 0 else float(np.nanstd(x))


def robust_rms(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    return float(np.sqrt(np.nanmedian((x - med) ** 2)))


def robust_ylim(y, q=(1, 99), pad=0.08):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (-1.0, 1.0)
    lo, hi = np.nanpercentile(y, q)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        m = np.nanmedian(y) if y.size else 0.0
        return (m - 1.0, m + 1.0)
    d = hi - lo
    return (lo - pad * d, hi + pad * d)


def parse_ranges(text: str):
    text = str(text).strip()
    if not text:
        return []
    out = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Bad range chunk: {chunk!r}")
        a, b = chunk.split(":", 1)
        out.append((float(a), float(b)))
    return out


def in_any_ranges(lam: np.ndarray, ranges):
    m = np.zeros_like(lam, dtype=bool)
    for a, b in ranges:
        lo, hi = (a, b) if a <= b else (b, a)
        m |= (lam >= lo) & (lam <= hi)
    return m

def point_in_ranges(x, ranges):
    for a, b in ranges:
        lo, hi = (a, b) if a <= b else (b, a)
        if lo <= x <= hi:
            return True
    return False


def move_anchor_out_of_ranges(x_anchor, x_valid, ranges):
    """
    If x_anchor falls inside a protected range, move it to the nearest
    valid wavelength from x_valid that lies outside all protected ranges.
    """
    if not point_in_ranges(x_anchor, ranges):
        return float(x_anchor)

    x_valid = np.asarray(x_valid, float)
    good = np.isfinite(x_valid)
    x_valid = x_valid[good]
    if x_valid.size == 0:
        return np.nan

    safe = np.array([not point_in_ranges(xx, ranges) for xx in x_valid], dtype=bool)
    if not np.any(safe):
        return np.nan

    x_safe = x_valid[safe]
    j = np.argmin(np.abs(x_safe - x_anchor))
    return float(x_safe[j])

def choose_lambda_column(names):
    cols = list(names)
    low = {c.lower(): c for c in cols}
    for cand in ["LAMBDA_NM", "lambda_nm", "WAVELENGTH_NM", "WAVELENGTH", "LAMBDA", "lam", "lambda"]:
        if cand in cols:
            return cand
        if cand.lower() in low:
            return low[cand.lower()]
    raise KeyError("No wavelength column found")


def choose_signal_column(names):
    cols = list(names)
    low = {c.lower(): c for c in cols}
    for cand in ["OBJ_PRESKY", "OBJ_RAW", "SKY", "FLUX", "SIGNAL", "signal", "y", "Y"]:
        if cand in cols:
            return cand
        if cand.lower() in low:
            return low[cand.lower()]
    raise KeyError("No signal column found")


def norm_slit(s: str) -> str:
    s = str(s).strip().upper()
    digits = "".join(ch for ch in s if ch.isdigit())
    return f"SLIT{int(digits):03d}" if digits else s


def read_fits_slit(path: Path, slit: str):
    slit = norm_slit(slit)
    with fits.open(path) as h:
        hdu = None
        if slit in h:
            hdu = h[slit]
        else:
            sid = int("".join(ch for ch in slit if ch.isdigit()))
            for ext in h[1:]:
                try:
                    if int(ext.header.get("SLITID", -999)) == sid:
                        hdu = ext
                        break
                except Exception:
                    pass
        if hdu is None:
            raise KeyError(f"Could not find {slit} in {path}")
        tab = hdu.data
        lam_col = choose_lambda_column(tab.columns.names)
        sig_col = choose_signal_column(tab.columns.names)
        lam = np.ravel(np.asarray(tab[lam_col], float))
        y = np.ravel(np.asarray(tab[sig_col], float))
    m = np.isfinite(lam) & np.isfinite(y)
    lam, y = lam[m], y[m]
    o = np.argsort(lam)
    return lam[o], y[o], sig_col


def read_text_or_csv(path: Path, lam_col=None, y_col=None):
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, comment="#", delim_whitespace=True, header=None)
        if df.shape[1] < 2:
            raise ValueError(f"{path} does not look like a 2-column table")
        df = df.iloc[:, :2].copy()
        df.columns = ["lam", "y"]

    if lam_col is None or y_col is None:
        cols = list(df.columns)
        if "lam" in cols and "y" in cols:
            lam_col = "lam"
            y_col = "y"
        else:
            low = {c.lower(): c for c in cols}
            lam_col = lam_col or low.get("lambda_nm") or low.get("wavelength_nm") or low.get("lambda") or cols[0]
            y_col = y_col or low.get("signal") or low.get("flux") or low.get("sky") or cols[1]

    lam = np.asarray(df[lam_col], float)
    y = np.asarray(df[y_col], float)
    m = np.isfinite(lam) & np.isfinite(y)
    lam, y = lam[m], y[m]
    o = np.argsort(lam)
    return lam[o], y[o], str(y_col)


def local_anchor_from_window(
    lam_w, y_w, regime_w,
    prev_anchor=None,
    exclusion_ranges=None,
    low_trim_frac=0.08,
    high_clip_sigma=2.0,
    low_clip_sigma=3.5,
    niter=4,
):
    """
    Build one anchor from the local moving population.

    regime_w:
      0 = excluded / telluric / protected
      1 = normal
      2 = forest
    """
    lam_w = np.asarray(lam_w, float)
    y_w = np.asarray(y_w, float)
    regime_w = np.asarray(regime_w, int)

    valid = np.isfinite(lam_w) & np.isfinite(y_w) & (regime_w != 0)
    if np.sum(valid) < 8:
        return np.nan, np.nan, valid

    x = lam_w[valid]
    z = y_w[valid]
    r = regime_w[valid]

    order = np.argsort(z)
    z_sorted = z[order]
    n = len(z_sorted)
    i0 = int(np.floor(low_trim_frac * n))
    i1 = max(i0 + 3, int(np.floor(0.70 * n)))
    seed = np.nanmedian(z_sorted[i0:i1])

    keep = np.ones_like(z, dtype=bool)

    for _ in range(int(niter)):
        zz = z[keep]
        if len(zz) < 6:
            break

        med = np.nanmedian(zz)
        sig = robust_sigma(zz)
        if not np.isfinite(sig) or sig <= 0:
            break

        hi_k = np.where(r == 2, 0.3, high_clip_sigma)
        lo_k = np.where(r == 2, low_clip_sigma * 0.7, low_clip_sigma)

        if prev_anchor is None or not np.isfinite(prev_anchor):
            center = med if np.isfinite(med) else seed
        else:
            center = 0.75 * med + 0.25 * prev_anchor

        keep_new = (z <= center + hi_k * sig) & (z >= center - lo_k * sig)

        # Moderate upper-tail rejection.        
        if np.sum(keep_new) > 10:
            vals = z[keep_new]
        
            if np.mean(r[keep_new] == 2) > 0.5:
                # FOREST → very aggressive: keep only lower envelope
                cutoff_hi = np.nanpercentile(vals, 55)   # was 70 → too high
                cutoff_lo = np.nanpercentile(vals, 15)   # stabilize floor
            else:
                # NORMAL → moderate cleaning
                cutoff_hi = np.nanpercentile(vals, 80)
                cutoff_lo = np.nanpercentile(vals, 5)
        
            keep_new &= (z <= cutoff_hi) & (z >= cutoff_lo)
    

        # Trim only the extreme lowest tail.
        if np.sum(keep_new) >= 8:
            vals = z[keep_new]
            cutoff_lo = np.nanpercentile(vals, 100.0 * low_trim_frac)
            keep_new &= (z >= cutoff_lo)

        if np.all(keep_new == keep):
            keep = keep_new
            break
        keep = keep_new

    if np.sum(keep) < 4:
        keep[:] = True

    xk = x[keep]
    zk = z[keep]

    x_anchor = float(np.nanmedian(xk))
    
    if exclusion_ranges:
        x_anchor = move_anchor_out_of_ranges(x_anchor, xk, exclusion_ranges)
        if not np.isfinite(x_anchor):
            return np.nan, np.nan, valid
        
    forest_frac = np.mean(r[keep] == 2)

    if forest_frac > 0.5:
        y_anchor = float(np.nanpercentile(zk, 20))
    else:
        y_anchor = float(np.nanmedian(zk))

    keep_full = np.zeros_like(lam_w, dtype=bool)
    keep_full[np.where(valid)[0][keep]] = True
    return x_anchor, y_anchor, keep_full


def build_window_regime(lam_w, telluric_ranges, forest_ranges):
    regime = np.ones_like(lam_w, dtype=int)
    regime[in_any_ranges(lam_w, forest_ranges)] = 2
    regime[in_any_ranges(lam_w, telluric_ranges)] = 0
    return regime


def moving_population_anchors(
    lam, y,
    telluric_ranges,
    forest_ranges,
    window_nm=20.0,
    stride_nm=20.0,
    pass_id=1,
    guide_cont=None,
    reverse_walk=False,
):
    lam = np.asarray(lam, float)
    y = np.asarray(y, float)

    anchors = []
    keep_mask = np.zeros_like(lam, dtype=bool)
    rows = []

    lo = float(np.nanmin(lam))
    hi = float(np.nanmax(lam))

    centers = []
    if not reverse_walk:
        c = lo + 0.5 * window_nm
        while c <= hi - 0.5 * window_nm + 1e-9:
            centers.append(c)
            c += stride_nm
        if not centers or centers[-1] < hi - 0.5 * window_nm:
            centers.append(hi - 0.5 * window_nm)
    else:
        c = hi - 0.5 * window_nm
        while c >= lo + 0.5 * window_nm - 1e-9:
            centers.append(c)
            c -= stride_nm
        if not centers or centers[-1] > lo + 0.5 * window_nm:
            centers.append(lo + 0.5 * window_nm)

    prev_anchor = np.nan
    for iw, cen in enumerate(centers, start=1):
        wlo = cen - 0.5 * window_nm
        whi = cen + 0.5 * window_nm
        m = np.isfinite(lam) & np.isfinite(y) & (lam >= wlo) & (lam <= whi)
        if np.sum(m) < 8:
            continue

        lam_w = lam[m]
        y_w = y[m]

        if guide_cont is not None:
            y_loc = y_w - np.asarray(guide_cont[m], float)
        else:
            y_loc = y_w.copy()

        regime_w = build_window_regime(lam_w, telluric_ranges, forest_ranges)

        xa, ya_resid, keep_full = local_anchor_from_window(
            lam_w, y_loc, regime_w,
            prev_anchor=prev_anchor if np.isfinite(prev_anchor) else None,
            exclusion_ranges=telluric_ranges,
        )
        if not np.isfinite(xa) or not np.isfinite(ya_resid):
            continue

        if guide_cont is not None:
            ya = float(ya_resid + np.interp(xa, lam, guide_cont))
        else:
            ya = float(ya_resid)

        anchors.append((xa, ya))
        prev_anchor = ya if np.isfinite(ya) else prev_anchor

        idx_full = np.where(m)[0]
        keep_mask[idx_full[keep_full]] = True

        reg_counts = {
            "n_normal": int(np.sum(regime_w == 1)),
            "n_forest": int(np.sum(regime_w == 2)),
            "n_telluric": int(np.sum(regime_w == 0)),
        }
        rows.append({
            "pass": pass_id,
            "walk": "backward" if reverse_walk else "forward",
            "window_id": iw,
            "lam_lo_nm": float(wlo),
            "lam_hi_nm": float(whi),
            "x_anchor_nm": float(xa),
            "y_anchor": float(ya),
            "n_points": int(np.sum(m)),
            "n_kept": int(np.sum(keep_full)),
            **reg_counts,
        })

    if len(anchors) == 0:
        return np.array([]), np.array([]), keep_mask, pd.DataFrame(rows)

    anchors = np.asarray(anchors, float)
    order = np.argsort(anchors[:, 0])
    anchors = anchors[order]
    return anchors[:, 0], anchors[:, 1], keep_mask, pd.DataFrame(rows)


def continuum_from_anchors(lam, xb, yb, y, method="pchip"):
    lam = np.asarray(lam, float)
    xb = np.asarray(xb, float)
    yb = np.asarray(yb, float)
    y = np.asarray(y, float)

    good = np.isfinite(xb) & np.isfinite(yb)
    xb = xb[good]
    yb = yb[good]
    if len(xb) < 2:
        med = np.nanmedian(yb) if len(yb) else 0.0
        return np.full_like(lam, med, dtype=float)

    order = np.argsort(xb)
    xb = xb[order]
    yb = yb[order]

    xu, idx = np.unique(xb, return_index=True)
    xb = xu
    yb = yb[idx]

    if len(xb) < 2:
        med = np.nanmedian(yb)
        return np.full_like(lam, med if np.isfinite(med) else 0.0, dtype=float)

    if method == "linear":
        return np.interp(lam, xb, yb, left=yb[0], right=yb[-1])

    p = PchipInterpolator(xb, yb, extrapolate=False)
    cont = np.asarray(p(lam), float)

    mleft = lam < xb[0]
    if np.any(mleft) and len(xb) >= 2:
        slope_left = (yb[1] - yb[0]) / (xb[1] - xb[0])
        cont[mleft] = yb[0] + slope_left * (lam[mleft] - xb[0])

    mright = lam > xb[-1]
    if np.any(mright) and len(xb) >= 2:
        lam_r = lam[mright]
        y_r = y[mright]

        if len(lam_r) >= 6:
            y_rs = pd.Series(y_r).rolling(7, center=True, min_periods=1).median().to_numpy()
            slopes = np.diff(y_rs) / np.diff(lam_r)
            slopes = slopes[np.isfinite(slopes)]

            if len(slopes) >= 5:
                slo = np.nanpercentile(slopes, 5)
                shi = np.nanpercentile(slopes, 95)
                slopes_use = slopes[(slopes >= slo) & (slopes <= shi)]

                if len(slopes_use) >= 5:
                    cutoff = np.nanpercentile(slopes_use, 20)
                    low_group = slopes_use[slopes_use <= cutoff]
                    if len(low_group) >= 2:
                        slope_right = np.nanmean(low_group)
                    else:
                        slope_right = (yb[-1] - yb[-2]) / (xb[-1] - xb[-2])
                else:
                    slope_right = (yb[-1] - yb[-2]) / (xb[-1] - xb[-2])
            else:
                slope_right = (yb[-1] - yb[-2]) / (xb[-1] - xb[-2])

            cont_edge = yb[-1] + slope_right * (lam_r - xb[-1])
            floor = np.nanpercentile(y_r, 10)
            cont[mright] = np.maximum(cont_edge, floor)

    return np.asarray(cont, float)


def fit_moving_population_continuum(
    lam, y,
    telluric_ranges,
    forest_ranges,
    window_nm=20.0,
    stride_nm=20.0,
    method="pchip",
    n_passes=2,
    reverse_walk=False,
):
    xb1, yb1, keep1, win1 = moving_population_anchors(
        lam, y,
        telluric_ranges=telluric_ranges,
        forest_ranges=forest_ranges,
        window_nm=window_nm,
        stride_nm=stride_nm,
        pass_id=1,
        guide_cont=None,
        reverse_walk=reverse_walk,
    )
    cont1 = continuum_from_anchors(lam, xb1, yb1, y, method=method)

    if int(n_passes) <= 1:
        resid = y - cont1
        return cont1, resid, xb1, yb1, keep1, win1

    xb2, yb2, keep2, win2 = moving_population_anchors(
        lam, y,
        telluric_ranges=telluric_ranges,
        forest_ranges=forest_ranges,
        window_nm=window_nm,
        stride_nm=stride_nm,
        pass_id=2,
        guide_cont=cont1,
        reverse_walk=reverse_walk,
    )
    cont2 = continuum_from_anchors(lam, xb2, yb2, y, method=method)
    resid = y - cont2
    wins = pd.concat([win1, win2], ignore_index=True)
    return cont2, resid, xb2, yb2, keep2, wins


def fit_bidirectional_continuum(
    lam, y,
    telluric_ranges,
    forest_ranges,
    window_nm=20.0,
    stride_nm=20.0,
    method="pchip",
    n_passes=2,
):
    cont_fwd, _, xb_fwd, yb_fwd, keep_fwd, win_fwd = fit_moving_population_continuum(
        lam, y,
        telluric_ranges=telluric_ranges,
        forest_ranges=forest_ranges,
        window_nm=window_nm,
        stride_nm=stride_nm,
        method=method,
        n_passes=n_passes,
        reverse_walk=False,
    )

    cont_bwd, _, xb_bwd, yb_bwd, keep_bwd, win_bwd = fit_moving_population_continuum(
        lam, y,
        telluric_ranges=telluric_ranges,
        forest_ranges=forest_ranges,
        window_nm=window_nm,
        stride_nm=stride_nm,
        method=method,
        n_passes=n_passes,
        reverse_walk=True,
    )

    u = (lam - lam.min()) / (lam.max() - lam.min())
    w = 0.5 * (1.0 - np.cos(np.pi * u))

    cont = (1.0 - w) * cont_bwd + w * cont_fwd
    resid = y - cont

    keep_mask = keep_fwd | keep_bwd
    windows_df = pd.concat([win_fwd, win_bwd], ignore_index=True)

    return cont, resid, cont_fwd, cont_bwd, xb_fwd, yb_fwd, keep_mask, windows_df


def make_qc(
    lam, y, cont, resid, xb, yb, keep_mask, windows_df,
    telluric_ranges, forest_ranges,
    qc_png: Path, show=False, title="Moving-population continuum",
    cont_fwd=None, cont_bwd=None,
):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    mfor = in_any_ranges(lam, forest_ranges) & (~in_any_ranges(lam, telluric_ranges))
    mtel = in_any_ranges(lam, telluric_ranges)
    mnorm = (~mfor) & (~mtel)

    ylo, yhi = np.nanmin(y[np.isfinite(y)]), np.nanmax(y[np.isfinite(y)])

    ax1 = fig.add_subplot(221)
    if np.any(mnorm):
        ax1.fill_between(lam[mnorm], ylo, yhi, color="tab:green", alpha=0.05, step="mid", label="normal")
    if np.any(mfor):
        ax1.fill_between(lam[mfor], ylo, yhi, color="tab:orange", alpha=0.05, step="mid", label="forest")
    if np.any(mtel):
        ax1.fill_between(lam[mtel], ylo, yhi, color="tab:red", alpha=0.05, step="mid", label="telluric")
    ax1.plot(lam, y, lw=0.8, color="0.5", label="spectrum")
    if cont_fwd is not None:
        ax1.plot(lam, cont_fwd, lw=0.8, color="tab:blue", alpha=0.5, label="forward")
    if cont_bwd is not None:
        ax1.plot(lam, cont_bwd, lw=0.8, color="tab:orange", alpha=0.5, label="backward")
    ax1.plot(lam, cont, lw=1.2, color="k", label="continuum")
    if len(xb):
        ax1.scatter(xb, yb, s=18, color="tab:blue", label="anchors")
    ax1.set_title("Spectrum + moving-population continuum")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Signal")
    ax1.set_ylim(*robust_ylim(np.r_[y, cont], q=(0.5, 99.5)))
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(222)
    ax2.plot(lam, resid, lw=0.8, color="k")
    ax2.axhline(0.0, lw=0.8, ls="--", color="0.5")
    if np.any(mfor):
        ax2.fill_between(lam[mfor], -1e9, 1e9, color="tab:orange", alpha=0.04, step="mid")
    if np.any(mtel):
        ax2.fill_between(lam[mtel], -1e9, 1e9, color="tab:red", alpha=0.04, step="mid")
    ax2.set_title(f"Residual after continuum fit (RMSrob={robust_rms(resid):.6f})")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Residual")
    ax2.set_ylim(*robust_ylim(resid, q=(0.5, 99.5)))

    ax3 = fig.add_subplot(223)
    ax3.plot(lam, y, lw=0.5, color="0.8", label="all points")
    if np.any(keep_mask):
        ax3.plot(lam[keep_mask], y[keep_mask], lw=0, marker=".", ms=2.5, color="tab:blue", label="kept points")
    if len(xb):
        ax3.scatter(xb, yb, s=20, color="k", label="anchors")
    ax3.set_title("Points retained by the local population selector")
    ax3.set_xlabel("Wavelength (nm)")
    ax3.set_ylabel("Signal")
    ax3.set_ylim(*robust_ylim(y, q=(0.5, 99.5)))
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(224)
    txt = [
        f"n points = {len(lam)}",
        f"n anchors = {len(xb)}",
        f"RMS robust residual = {robust_rms(resid):.6f}",
        "",
        "region laws:",
        "  normal   = green",
        "  forest   = orange",
        "  telluric = red",
        "",
    ]
    if len(windows_df):
        txt.append("last windows:")
        for _, row in windows_df.tail(10).iterrows():
            txt.append(
                f"  {row.get('walk', 'dir')[0]} p{int(row['pass'])} w{int(row['window_id']):02d}: "
                f"{row['lam_lo_nm']:.0f}-{row['lam_hi_nm']:.0f}  "
                f"keep={int(row['n_kept'])}/{int(row['n_points'])}  "
                f"anchor={row['y_anchor']:.3f}"
            )
    ax4.axis("off")
    ax4.text(0.02, 0.98, "\n".join(txt), va="top", ha="left", family="monospace", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(qc_png, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="Bidirectional moving-population continuum fitter for SAMOS 1D spectra"
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--slit", type=str, default=None)
    p.add_argument("--lam-col", type=str, default=None)
    p.add_argument("--y-col", type=str, default=None)

    p.add_argument("--telluric-ranges", type=str, default="655.5:657.5,685:690,758:770")
    p.add_argument("--forest-ranges", type=str, default="630:640,680:690,735:745,760:800,820:850,875:910")

    p.add_argument("--window-nm", type=float, default=20.0)
    p.add_argument("--stride-nm", type=float, default=20.0)
    p.add_argument("--method", type=str, default="pchip", choices=["pchip", "linear"])
    p.add_argument("--passes", type=int, default=2)

    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--qc-png", type=Path, default=None)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    inp = Path(args.input)
    if inp.suffix.lower() == ".fits":
        if not args.slit:
            raise ValueError("--slit is required for FITS input")
        lam, y, sig_name = read_fits_slit(inp, args.slit)
        stem = f"{inp.stem}_{norm_slit(args.slit)}"
    else:
        lam, y, sig_name = read_text_or_csv(inp, lam_col=args.lam_col, y_col=args.y_col)
        stem = inp.stem

    telluric_ranges = parse_ranges(args.telluric_ranges)
    forest_ranges = parse_ranges(args.forest_ranges)

    cont, resid, cont_fwd, cont_bwd, xb, yb, keep_mask, windows_df = fit_bidirectional_continuum(
        lam, y,
        telluric_ranges=telluric_ranges,
        forest_ranges=forest_ranges,
        window_nm=args.window_nm,
        stride_nm=args.stride_nm,
        method=args.method,
        n_passes=args.passes,
    )

    out_dir = inp.resolve().parent
    out_csv = args.out_csv or (out_dir / f"{stem}_moving_population_continuum.csv")
    out_json = args.out_json or (out_dir / f"{stem}_moving_population_continuum_summary.json")
    qc_png = args.qc_png or (out_dir / f"qc_{stem}_moving_population_continuum.png")

    pd.DataFrame({
        "LAMBDA_NM": lam,
        "SIGNAL": y,
        "CONTINUUM": cont,
        "CONTINUUM_FWD": cont_fwd,
        "CONTINUUM_BWD": cont_bwd,
        "RESIDUAL": resid,
        "KEPT_POINT": keep_mask.astype(int),
    }).to_csv(out_csv, index=False)

    summary = {
        "input": str(inp.resolve()),
        "signal_name": sig_name,
        "n_points": int(len(lam)),
        "n_anchors": int(len(xb)),
        "telluric_ranges_nm": telluric_ranges,
        "forest_ranges_nm": forest_ranges,
        "window_nm": float(args.window_nm),
        "stride_nm": float(args.stride_nm),
        "method": args.method,
        "passes": int(args.passes),
        "rms_robust_resid": float(robust_rms(resid)),
    }
    out_json.write_text(json.dumps(summary, indent=2))
    windows_df.to_csv(out_dir / f"{stem}_moving_population_windows.csv", index=False)

    title = "Moving-population continuum fit"
    if inp.suffix.lower() == ".fits":
        title += f" — {norm_slit(args.slit)}"
    make_qc(
        lam, y, cont, resid, xb, yb, keep_mask, windows_df,
        telluric_ranges, forest_ranges, qc_png,
        show=args.show, title=title,
        cont_fwd=cont_fwd, cont_bwd=cont_bwd,
    )

    print("INPUT     =", inp)
    print("SIGNAL    =", sig_name)
    print("OUT CSV   =", out_csv)
    print("OUT JSON  =", out_json)
    print("QC        =", qc_png)
    print("RMSROB    =", robust_rms(resid))
    print("N ANCHORS =", len(xb))


if __name__ == "__main__":
    main()
