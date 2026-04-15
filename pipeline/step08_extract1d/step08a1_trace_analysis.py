#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step08a1 — trace analysis and slit-quality classification from TRACECOORDS slit images.

This stage does NOT perform the final 1D extraction. It:
- measures a ridge line X0(y)
- identifies the brightest compact Y-block
- classifies slit usability for point-source extraction
- writes an editable CSV table
- writes a contract-compatible FITS MEF that preserves the columns expected by
  the existing Step08 QC scripts.

Important design choice
-----------------------
To preserve compatibility with the current Step08 QC scripts, this stage writes
`extract1d_optimal_ridge_{even,odd}.fits` with the usual table schema, but the
spectral columns are placeholders at this stage. Step08a2 will read this file
and overwrite those columns with the real extracted spectra.
"""
from __future__ import annotations

import argparse
from glob import glob
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

import config

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="SAMOS Step08a1 trace analysis")
parser.add_argument("--set", choices=["EVEN", "ODD"], required=True, help="Slit set to process")
parser.add_argument("--block-rows", type=int, default=20, help="Rows per Y block for seed/profile analysis")
args = parser.parse_args()
SET_TAG = args.set.upper()

# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------
pattern = str(Path(config.ST06_SCIENCE) / f"*_{SET_TAG}_tracecoords.fits")
matches = sorted(glob(pattern))
if not matches:
    raise FileNotFoundError(f"No TRACECOORDS file found for {SET_TAG}: {pattern}")
IN_FITS = Path(matches[-1])

ST08 = Path(config.ST08_EXTRACT1D)
ST08.mkdir(parents=True, exist_ok=True)
OUT_FITS = ST08 / f"trace_analysis_optimal_ridge_{SET_TAG.lower()}.fits"
OUT_CSV = ST08 / f"step08a_slit_quality_{SET_TAG.lower()}.csv"

# -----------------------------------------------------------------------------
# Parameters (kept close to current script)
# -----------------------------------------------------------------------------
YBIN = 50
RIDGE_TRACK_GATE = 3
RIDGE_TRACK_JUMP = 1.25
RIDGE_TRACK_MIN_CONTRAST = 2e-4
RIDGE_FIT_ORDER = 3
RIDGE_OUTLIER_PIX = 1.5
RIDGE_MIN_BINS = 3
SMOOTH_WIN = 301
SMOOTH_POLY = 2
CENTER_SIGMA = 2.0
MAX_CENTER_OFFSET = 3.0
SKYONLY_MIN_VALID_FRAC = 0.20
W_OBJ = 3.0
GAP = 1.0
PROFILE_SIGMA = 2.0
GAIN_E_PER_ADU = float(config.GAIN_E_PER_ADU)
READ_NOISE_E = float(config.READNOISE_E)
SQRT2 = sqrt(2.0)
EXTENDED_WIDTH_SIGMA_MAX = 4.0

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def savgol_smooth(y: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    y = np.asarray(y, float)
    n = y.size
    if n < 5:
        return y.copy()
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < 5:
        return y.copy()
    x = np.arange(n, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < max(5, polyorder + 2):
        return y.copy()
    yfill = y.copy()
    yfill[~ok] = np.interp(x[~ok], x[ok], y[ok])
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(yfill, window_length=window, polyorder=polyorder, mode="interp")
    except Exception:
        half = window // 2
        out = np.full(n, np.nan, float)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            xx = x[lo:hi]
            yy = yfill[lo:hi]
            coeff = np.polyfit(xx, yy, polyorder)
            out[i] = np.polyval(coeff, x[i])
        return out

def set_hdr_float_safe(hdr, key, value, comment=""):
    if value is None or not np.isfinite(value):
        hdr[key] = ("UNDEF", comment)
    else:
        hdr[key] = (float(value), comment)
        

def valid_interval_from_row(row_valid: np.ndarray) -> tuple[float, float]:
    idx = np.where(row_valid)[0]
    if idx.size == 0:
        return np.nan, np.nan
    return float(idx[0]) - 0.5, float(idx[-1]) + 0.5


def estimate_trace_center(img: np.ndarray) -> float:
    ny, nx = img.shape
    fin = np.isfinite(img)
    centers = []
    for y in range(ny):
        xs = np.where(fin[y])[0]
        if xs.size > 5:
            centers.append(0.5 * (float(xs[0]) + float(xs[-1])))
    if not centers:
        return 0.5 * float(nx - 1)
    return float(np.median(centers))


def robust_profile(block: np.ndarray) -> np.ndarray:
    block = np.asarray(block, float)
    good_rows = np.any(np.isfinite(block), axis=1)
    if not np.any(good_rows):
        return np.full(block.shape[1], np.nan, float)
    with np.errstate(all="ignore"):
        p = np.nanmedian(block[good_rows], axis=0)
    return np.asarray(p, float)


def profile_peak_and_score(profile: np.ndarray) -> tuple[float, float, float, float]:
    p = np.asarray(profile, float)
    if not np.isfinite(p).any():
        return np.nan, np.nan, np.nan, np.nan
    base = np.nanmedian(p)
    q = p - base
    q[~np.isfinite(q)] = 0.0
    q[q < 0] = 0.0
    if q.sum() <= 0:
        return np.nan, np.nan, np.nan, np.nan
    x = np.arange(p.size, dtype=float)
    xpk = float((x * q).sum() / q.sum())
    peak = float(np.nanmax(q))
    var = float(((x - xpk) ** 2 * q).sum() / q.sum()) if q.sum() > 0 else np.inf
    width = np.sqrt(var) if np.isfinite(var) and var > 0 else np.nan
    score = peak / max(width, 1.0) if np.isfinite(width) else np.nan
    return xpk, peak, width, score


def find_brightest_seed_block(img: np.ndarray, block_rows: int = 20):
    ny, nx = img.shape
    best = None
    for y0 in range(0, ny, block_rows):
        y1 = min(ny, y0 + block_rows)
        block = img[y0:y1, :]
        if np.isfinite(block).sum() < 0.1 * block.size:
            continue
        prof = robust_profile(block)
        xpk, peak, width, score = profile_peak_and_score(prof)
        if not np.isfinite(score):
            continue
        item = dict(y0=y0, y1=y1, ymid=0.5 * (y0 + y1 - 1), xpk=xpk, peak=peak, width=width, score=score, profile=prof)
        if best is None or item["score"] > best["score"]:
            best = item
    return best


def estimate_halfmax_crossings(profile: np.ndarray):
    p = np.asarray(profile, float)
    if not np.isfinite(p).any():
        return np.nan, np.nan, np.nan, np.nan
    peak_idx = int(np.nanargmax(p))
    peak_val = float(p[peak_idx])
    if not np.isfinite(peak_val) or peak_val <= 0:
        return np.nan, np.nan, np.nan, np.nan
    half = 0.5 * peak_val
    x = np.arange(len(p), dtype=float)
    xl = np.nan
    for i in range(peak_idx - 1, -1, -1):
        if np.isfinite(p[i]) and np.isfinite(p[i + 1]) and p[i] <= half <= p[i + 1]:
            denom = p[i + 1] - p[i]
            t = (half - p[i]) / denom if denom != 0 else 0.0
            xl = x[i] + t
            break
    xr = np.nan
    for i in range(peak_idx, len(p) - 1):
        if np.isfinite(p[i]) and np.isfinite(p[i + 1]) and p[i] >= half >= p[i + 1]:
            denom = p[i + 1] - p[i]
            t = (half - p[i]) / denom if denom != 0 else 0.0
            xr = x[i] + t
            break
    return float(peak_idx), peak_val, xl, xr


def classify_seed_block(seed, img: np.ndarray):
    if seed is None:
        return False, "NOSEED", np.nan, np.nan, np.nan, np.nan, np.nan
    ymid = int(np.clip(round(seed["ymid"]), 0, img.shape[0] - 1))
    prof = np.asarray(seed["profile"], float)
    base = np.nanmedian(prof[np.isfinite(prof)]) if np.isfinite(prof).any() else 0.0
    prof0 = prof - base
    _, peak_val, xl_h, xr_h = estimate_halfmax_crossings(prof0)
    fwhm = xr_h - xl_h if np.isfinite(xl_h) and np.isfinite(xr_h) and xr_h > xl_h else np.nan
    row_valid = np.isfinite(img[ymid])
    xL, xR = valid_interval_from_row(row_valid)
    xC = 0.5 * (xL + xR) if np.isfinite(xL) and np.isfinite(xR) else np.nan
    dxcen = seed["xpk"] - xC if np.isfinite(seed["xpk"]) and np.isfinite(xC) else np.nan
    if not np.isfinite(fwhm):
        return False, "EMPTY", fwhm, xL, xR, xC, dxcen
    if np.isfinite(seed["width"]) and seed["width"] > EXTENDED_WIDTH_SIGMA_MAX:
        return False, "EXTENDED", fwhm, xL, xR, xC, dxcen
    if np.isfinite(xL) and np.isfinite(xR) and (xl_h < xL or xr_h > xR):
        return False, "EDGE", fwhm, xL, xR, xC, dxcen
    return True, "GOOD", fwhm, xL, xR, xC, dxcen


def build_binned_profiles(img: np.ndarray, ybin: int):
    ny, nx = img.shape
    yedges = list(range(0, ny, ybin))
    if yedges[-1] != ny:
        yedges.append(ny)
    profiles, ycenters = [], []
    for i in range(len(yedges) - 1):
        a, b = yedges[i], yedges[i + 1]
        block = img[a:b, :]
        if np.isfinite(block).sum() < 0.1 * block.size:
            continue
        with np.errstate(all="ignore"):
            p = np.nanmedian(block, axis=0)
        if not np.isfinite(p).any():
            continue
        profiles.append(p)
        ycenters.append(0.5 * (a + b))
    return np.asarray(profiles, float), np.asarray(ycenters, float)


def stabilize_ridge_with_global_model(x0, img, frac=0.2, poly_order=2):
    """
    Stabilize ridge using a global polynomial fit anchored on high-S/N rows.

    Parameters
    ----------
    x0 : array (ny)
        Initial ridge from bidirectional tracking
    img : 2D array (ny, nx)
        TRACECOORDS slit image
    frac : float
        Threshold fraction of max signal to define good rows (default 0.2)
    poly_order : int
        Polynomial order (2 is recommended)

    Returns
    -------
    x0_final : array (ny)
        Stabilized ridge
    """

    ny = img.shape[0]
    y = np.arange(ny, dtype=float)

    # --- estimate per-row signal strength ---
    peak_row = np.full(ny, np.nan, float)

    for iy in range(ny):
        row = img[iy]
        if not np.isfinite(row).any():
            continue
        base = np.nanmedian(row)
        peak_row[iy] = np.nanmax(row - base)

    if not np.isfinite(peak_row).any():
        return x0  # nothing to do

    # --- select good rows ---
    max_peak = np.nanmax(peak_row)
    threshold = frac * max_peak
    good = (peak_row > threshold) & np.isfinite(x0)

    # not enough points → skip
    if np.sum(good) < max(10, poly_order + 2):
        return x0

    # --- fit global polynomial ---
    try:
        coeff = np.polyfit(y[good], x0[good], poly_order)
        x_model = np.polyval(coeff, y)
    except Exception:
        return x0

    # --- blend model with data ---
    snr_weight = peak_row / max_peak
    snr_weight = np.clip(snr_weight, 0, 1)

    alpha = 1.0 - snr_weight  # stronger model at low S/N

    x0_final = (1 - alpha) * x0 + alpha * x_model

    # --- final gentle smoothing ---
    ok = np.isfinite(x0_final)
    if ok.sum() > 10:
        y_ok = y[ok]
        x_ok = x0_final[ok]
        x0_final[~ok] = np.interp(y[~ok], y_ok, x_ok)

    return x0_final


def _corridor_centroid(profile: np.ndarray, x_pred: float, halfwidth: int, x_center: float):
    if not np.isfinite(x_pred):
        return np.nan, np.nan
    p = np.asarray(profile, float)
    nx = p.size
    xc = int(np.round(x_pred))
    lo = max(0, xc - int(halfwidth))
    hi = min(nx, xc + int(halfwidth) + 1)
    if hi <= lo:
        return np.nan, np.nan
    x = np.arange(lo, hi, dtype=float)
    seg = p[lo:hi]
    base = np.nanmedian(seg)
    w = seg - base
    w[~np.isfinite(w)] = 0.0
    w[w < 0] = 0.0
    if w.sum() <= 0:
        return np.nan, np.nan
    contrast = float(np.nanmax(w)) if w.size else np.nan
    center_weight = np.exp(-0.5 * ((x - float(x_center)) / float(CENTER_SIGMA)) ** 2)
    w2 = w * center_weight
    if w2.sum() <= 0:
        return np.nan, np.nan
    xcen = float((x * w2).sum() / w2.sum())
    return xcen, contrast


def track_rows_bidirectional(img: np.ndarray, y_seed: int, x_seed: float, x_center: float) -> np.ndarray:
    ny, nx = img.shape
    x0 = np.full(ny, np.nan, float)
    y_seed = int(np.clip(y_seed, 0, ny - 1))
    x0[y_seed] = float(x_seed)

    def row_centroid(y: int, x_pred: float):
        row = np.asarray(img[y], float)
        if not np.isfinite(x_pred):
            return np.nan, np.nan
        xx = np.arange(nx, dtype=float)
        lo = max(0, int(round(x_pred)) - RIDGE_TRACK_GATE)
        hi = min(nx, int(round(x_pred)) + RIDGE_TRACK_GATE + 1)
        if hi <= lo:
            return np.nan, np.nan
        segx = xx[lo:hi]
        seg = row[lo:hi]
        base = np.nanmedian(seg)
        w = seg - base
        w[~np.isfinite(w)] = 0.0
        w[w < 0] = 0.0
        if w.sum() <= 0:
            return np.nan, np.nan
        center_weight = np.exp(-0.5 * ((segx - float(x_center)) / float(CENTER_SIGMA)) ** 2)
        w2 = w * center_weight
        if w2.sum() <= 0:
            return np.nan, np.nan
        xcen = float((segx * w2).sum() / w2.sum())
        contrast = float(np.nanmax(w)) if w.size else np.nan
        return xcen, contrast

    def walk(direction: int):
        x_prev = float(x_seed)
        slope = 0.0
        rng = range(y_seed + direction, ny, direction) if direction > 0 else range(y_seed - 1, -1, -1)
        for y in rng:
            x_pred = x_prev + slope
            xcen, contrast = row_centroid(y, x_pred)
            if (not np.isfinite(xcen)) or (not np.isfinite(contrast)) or (contrast < RIDGE_TRACK_MIN_CONTRAST):
                continue
            dx = np.clip(xcen - x_prev, -float(RIDGE_TRACK_JUMP), float(RIDGE_TRACK_JUMP))
            x_new = x_prev + dx
            x0[y] = np.clip(x_new, 0.0, nx - 1.0)
            slope = 0.7 * slope + 0.3 * (x0[y] - x_prev)
            x_prev = x0[y]

    walk(+1)
    walk(-1)
    ok = np.isfinite(x0)
    if ok.sum() >= max(5, RIDGE_MIN_BINS):
        y = np.arange(ny, dtype=float)
        x0[~ok] = np.interp(y[~ok], y[ok], x0[ok])
        x0 = savgol_smooth(x0, SMOOTH_WIN, SMOOTH_POLY)
    return x0


def gaussian_cdf_scalar(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1.0 + erf((x - mu) / (SQRT2 * sigma)))


def gaussian_integral(lo: float, hi: float, mu: float, sigma: float) -> float:
    return gaussian_cdf_scalar(hi, mu, sigma) - gaussian_cdf_scalar(lo, mu, sigma)


def aperture_capture_fraction_gaussian(x0: float, aper_halfwidth: float, sigma_psf: float, xL_valid: float, xR_valid: float):
    if not np.isfinite(x0) or not np.isfinite(aper_halfwidth) or not np.isfinite(sigma_psf):
        return np.nan
    if not np.isfinite(xL_valid) or not np.isfinite(xR_valid) or sigma_psf <= 0:
        return np.nan
    a0 = x0 - aper_halfwidth
    a1 = x0 + aper_halfwidth
    lo = max(a0, xL_valid)
    hi = min(a1, xR_valid)
    if hi <= lo:
        return 0.0
    nom = gaussian_integral(a0, a1, x0, sigma_psf)
    got = gaussian_integral(lo, hi, x0, sigma_psf)
    if nom <= 0:
        return np.nan
    return float(np.clip(got / nom, 0.0, 1.0))


def compute_aperture_loss_correction(img: np.ndarray, x0: np.ndarray):
    ny, nx = img.shape
    fin = np.isfinite(img)
    apfrac = np.full(ny, np.nan, float)
    edgeflag = np.zeros(ny, dtype=np.int16)
    xleft = np.full(ny, np.nan, float)
    xright = np.full(ny, np.nan, float)
    aper_halfwidth = float(W_OBJ) + 0.5
    for y in range(ny):
        if not np.isfinite(x0[y]):
            continue
        xL, xR = valid_interval_from_row(fin[y])
        xleft[y] = xL
        xright[y] = xR
        frac = aperture_capture_fraction_gaussian(x0[y], aper_halfwidth, PROFILE_SIGMA, xL, xR)
        apfrac[y] = frac
        a0 = x0[y] - aper_halfwidth
        a1 = x0[y] + aper_halfwidth
        clipped = (not np.isfinite(xL)) or (not np.isfinite(xR)) or (xL > a0) or (xR < a1)
        if clipped:
            edgeflag[y] = 1
        if np.isfinite(frac) and (frac < 0.55):
            edgeflag[y] = 2
    return apfrac, edgeflag, xleft, xright


def make_placeholder_table(ny: int, x0: np.ndarray, apfrac: np.ndarray, edgeflag: np.ndarray, xleft: np.ndarray, xright: np.ndarray):
    nanf = np.full(ny, np.nan, dtype=np.float32)
    zerof = np.zeros(ny, dtype=np.float32)
    zeroi = np.zeros(ny, dtype=np.int16)
    cols = [
        fits.Column(name="YPIX", format="J", array=np.arange(ny, dtype=np.int32)),
        fits.Column(name="FLUX", format="E", array=nanf.copy()),
        fits.Column(name="VAR", format="E", array=nanf.copy()),
        fits.Column(name="SKY", format="E", array=nanf.copy()),
        fits.Column(name="OBJ_PRESKY", format="E", array=nanf.copy()),
        fits.Column(name="X0", format="E", array=np.asarray(x0, np.float32)),
        fits.Column(name="NOBJ", format="J", array=zeroi.astype(np.int32)),
        fits.Column(name="NSKY", format="J", array=zeroi.astype(np.int32)),
        fits.Column(name="SKYSIG", format="E", array=nanf.copy()),
        fits.Column(name="APLOSS_FRAC", format="E", array=np.asarray(apfrac, np.float32)),
        fits.Column(name="FLUX_APCORR", format="E", array=nanf.copy()),
        fits.Column(name="VAR_APCORR", format="E", array=nanf.copy()),
        fits.Column(name="EDGEFLAG", format="J", array=edgeflag.astype(np.int32)),
        fits.Column(name="TRXLEFT", format="E", array=np.asarray(xleft, np.float32)),
        fits.Column(name="TRXRIGHT", format="E", array=np.asarray(xright, np.float32)),
    ]
    return fits.BinTableHDU.from_columns(cols)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
rows = []
hdus = [fits.PrimaryHDU()]
hdus[0].header["PIPESTEP"] = "STEP08"
hdus[0].header["STAGE"] = ("08a1", "Trace analysis stage")
hdus[0].header["TRACESET"] = SET_TAG
hdus[0].header["INFILE"] = IN_FITS.name
hdus[0].header["OUTCSV"] = OUT_CSV.name

with fits.open(IN_FITS, memmap=False) as hdul:
    for h in hdul[1:]:
        slit = (h.name or "").upper()
        if not slit.startswith("SLIT") or h.data is None:
            continue
        img = np.asarray(h.data, float)
        if img.ndim != 2:
            continue
        ny, nx = img.shape
        x_center = estimate_trace_center(img)
        seed = find_brightest_seed_block(img, block_rows=args.block_rows)
        good, cls, fwhm, slit_xl, slit_xr, center_x, dx_center = classify_seed_block(seed, img)
        if seed is not None and np.isfinite(seed["xpk"]):
            x_seed = float(seed["xpk"])
            y_seed = int(np.clip(round(seed["ymid"]), 0, ny - 1))
        else:
            x_seed = float(x_center)
            y_seed = ny // 2
        if np.isfinite(x_seed) and abs(x_seed - x_center) > float(MAX_CENTER_OFFSET):
            x_seed = float(x_center)
        if good:
            x0 = track_rows_bidirectional(img, y_seed, x_seed, x_center)
            # --- NEW: stabilize ridge ---
            x0 = stabilize_ridge_with_global_model(x0, img, frac=0.2, poly_order=2)
            mode = "BRIGHTBLOCK_BIDIR"
        else:
            x0 = np.full(ny, x_seed, float)
            mode = f"SEED_{cls}"
        valid_frac = float(np.mean(np.isfinite(x0))) if x0.size else 0.0
        if valid_frac < float(SKYONLY_MIN_VALID_FRAC):
            x0 = np.full(ny, x_seed, float)
            mode = "SKYONLY"
            
        apfrac, edgeflag, xleft, xright = compute_aperture_loss_correction(img, x0)
        tab = make_placeholder_table(ny, x0, apfrac, edgeflag, xleft, xright)
        tab.name = slit
        
        # Preserve geometry metadata needed downstream by 08c wavelength attachment.
        # These keywords were present in the old Step08 products and must survive
        # the new 08a1/08a2 split.
        for key in ["Y0DET", "YMIN", "SHIFT2M", "SLITID"]:
            if key in h.header:
                tab.header[key] = (h.header[key], h.header.comments[key])
                
        tab.header["RIDGEMOD"] = mode        
        tab.header["WOBJ"] = float(W_OBJ)
        tab.header["GAP"] = float(GAP)
        tab.header["S08GOOD"] = (int(bool(good)), "1 if slit passes automatic quality test")
        tab.header["S08CLAS"] = (str(cls), "Automatic slit class")
        
        set_hdr_float_safe(tab.header, "S08FWHM", fwhm, "Brightest-block FWHM")
        set_hdr_float_safe(tab.header, "S08YSED", y_seed, "Seed Y from brightest block")
        set_hdr_float_safe(tab.header, "S08XSED", x_seed, "Seed X from brightest block")
        set_hdr_float_safe(
            tab.header, "S08SCOR",
            (seed["score"] if seed is not None else np.nan),
            "Brightest-block compactness score",
        )
        set_hdr_float_safe(tab.header, "S08DXC", dx_center, "Peak minus slit center")
        
        tab.header["S08BAD"] = (0, "Sky/flux inconsistency flag")
        tab.header["S08EMP"] = (int(cls in {"EMPTY", "NOSEED"}), "1 if slit is empty/no-seed")
        hdus.append(tab)
        rows.append({
            "SLIT": slit,
            "SET": SET_TAG,
            "GOOD": int(bool(good)),
            "CLASS": cls,
            "SEED_Y": float(y_seed),
            "SEED_X": float(x_seed),
            "PEAK_X": float(seed["xpk"]) if seed is not None else np.nan,
            "CENTER_X": float(center_x) if np.isfinite(center_x) else np.nan,
            "DX_CENTER": float(dx_center) if np.isfinite(dx_center) else np.nan,
            "FWHM": float(fwhm) if np.isfinite(fwhm) else np.nan,
            "SLIT_XL": float(slit_xl) if np.isfinite(slit_xl) else np.nan,
            "SLIT_XR": float(slit_xr) if np.isfinite(slit_xr) else np.nan,
            "SCORE": float(seed["score"]) if seed is not None else np.nan,
            "USE": int(bool(good)),
            "COMMENT": "",
        })

fits.HDUList(hdus).writeto(OUT_FITS, overwrite=True)
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f"Wrote {OUT_FITS}")
print(f"Wrote {OUT_CSV}")
