#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step08a — ridge-guided optimal extraction from TRACECOORDS slit images.

PURPOSE
-------
Extract 1D spectra from rectified slit images (TRACECOORDS) using a
ridge-guided optimal extraction algorithm with robust sky subtraction
and aperture-loss correction.

This step converts 2D slit images into 1D spectra in pixel space.
Wavelength calibration is NOT applied here (see Step08c).

INPUT
-----
- Step06 TRACECOORDS MEF:
    *_{EVEN,ODD}_tracecoords.fits

Each SLIT extension contains a rectified slit image where:
- Y axis = dispersion direction (pixel space)
- X axis = spatial direction (slit width)

OUTPUT
------
- extract1d_optimal_ridge_even.fits
- extract1d_optimal_ridge_odd.fits

Each SLIT### extension is a binary table with columns:
    YPIX, FLUX, VAR, SKY, X0, NOBJ, NSKY, SKYSIG,
    APLOSS_FRAC, FLUX_APCORR, VAR_APCORR, EDGEFLAG,
    TRXLEFT, TRXRIGHT

SCIENTIFIC METHOD
-----------------

1) Ridge tracking
   - Identify object trace using binned profiles in Y
   - Seed near slit center (avoids edge bias)
   - Track using corridor-constrained centroiding
   - Enforce smoothness via polynomial fit + Savitzky–Golay smoothing
   - Automatic fallback to "stiff ridge" if tracking is unstable

2) Sky estimation (ROWFIRST_POOL)
   - Prefer sky from the same row (preserves OH line structure)
   - Use one-sided sky (left/right) to avoid contamination
   - Apply sigma-clipping and low-tail rejection
   - Fall back to pooled sky (Y±YSKYWIN) when needed
   - Interpolate missing rows to ensure continuity

3) Optimal extraction
   - Gaussian profile weighting (Horne-style)
   - Variance includes:
        Poisson term (object signal)
        read noise
        sky variance
   - Extraction performed per row

4) Aperture-loss correction
   - Compute fraction of Gaussian profile within valid TRACECOORDS region
   - Correct flux when truncation is mild
   - Flag strongly truncated rows (EDGEFLAG)

ROBUSTNESS FEATURES
-------------------
- Ridge fallback (CENTERSEEDED → STIFFRIDGE)
- Corridor tracking limits ridge jumps
- Sky fallback hierarchy (row → pooled → previous row)
- Sigma-clipping and tail rejection for sky
- Edge detection and correction
- Per-row flags for quality control

OUTPUT FLAGS
------------
Per row:
    EDGEFLAG:
        0 = OK
        1 = clipped but corrected
        2 = heavily truncated (unsafe)

Per slit (header):
    S08BAD : sky/flux inconsistency flag
    S08EMP : empty extraction flag

NOTES
-----
- Extraction is done entirely in pixel space
- Ridge is NOT constrained by quartz traces (allows flexure mismatch)
- Full TRACECOORDS width is used (no artificial truncation)
- SKY is designed to preserve narrow OH emission features

DOES NOT DO
-----------
- wavelength calibration (Step08c)
- telluric correction (Step09)
- wavelength refinement (Step10)
- flux calibration (Step11)

run
> PYTHONPATH=. python pipeline/step08_extract1d/step08a_extract_1d.py --set EVEN
> PYTHONPATH=. python pipeline/step08_extract1d/step08a_extract_1d.py --set 

"""
import numpy as np
from pathlib import Path
from astropy.io import fits
import config
import argparse
from math import erf, sqrt
from glob import glob 

ap = argparse.ArgumentParser(description="SAMOS Step08 optimal extraction")
ap.add_argument("--set", choices=["EVEN", "ODD"], required=True, help="Slit set to process")
args = ap.parse_args()
SET_TAG = args.set.upper()

pattern = str(Path(config.ST06_SCIENCE) / f"*_{SET_TAG}_tracecoords.fits")
matches = sorted(glob(pattern))

if not matches:
    raise FileNotFoundError(f"No TRACECOORDS file found for {SET_TAG}: {pattern}")

IN_FITS = Path(matches[-1])

OUT_FITS = Path(config.ST08_EXTRACT1D) / f"extract1d_optimal_ridge_{SET_TAG.lower()}.fits"

YBIN = 50
GATE = 2
MIN_CONTRAST = 5e-4
JUMP_MAX = 1.5

# --- center-seeded ridge parameters ---
RIDGE_SEED_CENTRAL_FRAC = 0.5   # fraction of slit height used to search for seed
RIDGE_TRACK_GATE = 3
RIDGE_TRACK_JUMP = 1.25
RIDGE_SEED_MIN_CONTRAST = 5e-4
RIDGE_TRACK_MIN_CONTRAST = 2e-4

# --- automatic fallback to stiff ridge when center-seeded tracking is weak ---
RIDGE_AUTO_MODE = True
RIDGE_MIN_TRACKED_FRAC = 0.35   # minimum fraction of coarse bins successfully tracked
RIDGE_SEED_REQUIRED = True

SMOOTH_WIN = 301
SMOOTH_POLY = 2
PRED_BINS = 35
RIDGE_FIT_ORDER = 3
RIDGE_OUTLIER_PIX = 1.5
RIDGE_MIN_BINS = 3
W_OBJ = 3
GAP = 1
YSKYWIN = 1     #numer of rows of pooled sky
PROFILE_SIGMA = 2.0
APMINFR = 0.55 # Fraction of PSF that can be recovered if missing due to ridgeline at the edge
SKY_CLIP_K = 3.0
SKY_CLIP_ITERS = 2
SKY_NMIN = 5
SKY_EDGE_EXCLUDE_LEFT = 2
SKY_EDGE_EXCLUDE_RIGHT = 0
SKY_LOW_FRAC_LEFT = 0.15#1.0
SKY_LOW_FRAC_RIGHT = 0.15#1.0

# --- ridge protection against neighbor stealing ---
CENTER_SIGMA = 2.0
MAX_CENTER_OFFSET = 3.0
SKY_SN_THRESHOLD = 2.0
RIDGE_SCATTER_MAX = 2.0
SKYONLY_MIN_VALID_FRAC = 0.20

GAIN_E_PER_ADU = config.GAIN_E_PER_ADU
READ_NOISE_E = config.READNOISE_E
SQRT2 = sqrt(2.0)

def mad_sigma(x):
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad

def sigma_clip_high_mask(x, k=3.0, iters=2):
    ok = np.isfinite(x)
    if ok.sum() < 5:
        return ok
    m = np.median(x[ok]); s = mad_sigma(x[ok])
    if not np.isfinite(s) or s <= 0:
        return ok
    for _ in range(iters):
        thr = m + k * s
        ok2 = ok & (x <= thr)
        if ok2.sum() < 5 or ok2.sum() == ok.sum():
            ok = ok2
            break
        ok = ok2
        m = np.median(x[ok]); s = mad_sigma(x[ok])
        if not np.isfinite(s) or s <= 0:
            break
    return ok

def savgol_smooth(y, window, polyorder):
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
            lo = max(0, i - half); hi = min(n, i + half + 1)
            xx = x[lo:hi]; yy = yfill[lo:hi]
            coeff = np.polyfit(xx, yy, polyorder)
            out[i] = np.polyval(coeff, x[i])
        return out

def gaussian_cdf_scalar(x, mu, sigma):
    return 0.5 * (1.0 + erf((x - mu) / (SQRT2 * sigma)))

def gaussian_integral(lo, hi, mu, sigma):
    return gaussian_cdf_scalar(hi, mu, sigma) - gaussian_cdf_scalar(lo, mu, sigma)

def valid_interval_from_row(row_valid: np.ndarray, x0: float):
    """
    Return the valid TRACECOORDS interval [xL,xR].

    In TRACECOORDS the trace is simply the first and last
    finite pixels in the row.
    """

    idx = np.where(row_valid)[0]

    if idx.size == 0:
        return np.nan, np.nan

    xL = float(idx[0]) - 0.5
    xR = float(idx[-1]) + 0.5

    return xL, xR

def estimate_trace_center(img):
    """Estimate the expected spatial center of the trace from valid TRACECOORDS rows."""
    ny, nx = img.shape
    fin = np.isfinite(img)
    centers = []
    for y in range(ny):
        xs = np.where(fin[y])[0]
        if xs.size > 5:
            centers.append(0.5 * (float(xs[0]) + float(xs[-1])))
    if len(centers) == 0:
        return 0.5 * float(nx - 1)
    return float(np.median(centers))


def estimate_center_signal(img, x_center):
    """Robust central signal estimate using a narrow core around the expected center."""
    ny, nx = img.shape
    x = np.arange(nx, dtype=float)
    vals = []
    for y in range(ny):
        if not np.isfinite(x_center):
            continue
        dx = x - float(x_center)
        mask = np.isfinite(img[y]) & (np.abs(dx) <= 2.0)
        if mask.sum() < 3:
            continue
        v = img[y, mask]
        if np.isfinite(v).any():
            vals.append(float(np.nanmedian(v)))
    if len(vals) < 10:
        return np.nan
    return float(np.nanmedian(vals))


def estimate_background(img):
    """Robust global background location and scale for a slit image."""
    v = img[np.isfinite(img)]
    if v.size < 100:
        return np.nan, np.nan
    med = float(np.nanmedian(v))
    sig = float(mad_sigma(v))
    return med, sig


def aperture_capture_fraction_gaussian(x0, aper_halfwidth, sigma_psf, xL_valid, xR_valid):
    if not np.isfinite(x0) or not np.isfinite(aper_halfwidth) or not np.isfinite(sigma_psf):
        return np.nan
    if not np.isfinite(xL_valid) or not np.isfinite(xR_valid) or sigma_psf <= 0:
        return np.nan
    a0 = x0 - aper_halfwidth; a1 = x0 + aper_halfwidth
    lo = max(a0, xL_valid); hi = min(a1, xR_valid)
    if hi <= lo:
        return 0.0
    nom = gaussian_integral(a0, a1, x0, sigma_psf)
    got = gaussian_integral(lo, hi, x0, sigma_psf)
    if nom <= 0:
        return np.nan
    return float(np.clip(got / nom, 0.0, 1.0))

def compute_aperture_loss_correction(img, x0):
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
        xL, xR = valid_interval_from_row(fin[y], x0[y])
        xleft[y] = xL; xright[y] = xR
        frac = aperture_capture_fraction_gaussian(x0[y], aper_halfwidth, PROFILE_SIGMA, xL, xR)
        apfrac[y] = frac
        a0 = x0[y] - aper_halfwidth; a1 = x0[y] + aper_halfwidth
        clipped = (not np.isfinite(xL)) or (not np.isfinite(xR)) or (xL > a0) or (xR < a1)
        if clipped:
            edgeflag[y] = 1
        if np.isfinite(frac) and (frac < APMINFR):
            edgeflag[y] = 2
    return apfrac, edgeflag, xleft, xright

def build_binned_profiles(img, ybin):
    ny, nx = img.shape
    yedges = list(range(0, ny, ybin))
    if yedges[-1] != ny:
        yedges.append(ny)
    profiles = []; ycenters = []
    for i in range(len(yedges) - 1):
        a, b = yedges[i], yedges[i + 1]
        block = img[a:b, :]
        if np.isfinite(block).sum() < 0.1 * block.size:
            continue
        with np.errstate(all="ignore"):
            p = np.nanmedian(block, axis=0)
        if not np.isfinite(p).any():
            continue
        profiles.append(p); ycenters.append(0.5 * (a + b))
    return np.asarray(profiles, float), np.asarray(ycenters, float)

def _profile_positive_weights(p):
    p = np.asarray(p, float)
    if not np.isfinite(p).any():
        return np.array([], dtype=float), np.array([], dtype=float), np.nan, np.nan
    x = np.arange(p.size, dtype=float)
    base = np.nanmedian(p)
    w = p - base
    w[~np.isfinite(w)] = 0.0
    w[w < 0] = 0.0
    contrast = float(np.nanmax(w)) if w.size else np.nan
    signal = float(np.sum(w)) if w.size else np.nan
    return x, w, contrast, signal


def find_center_seed_bin(profiles, ycenters):
    """
    Pick the brightest reliable seed bin near the center of the slit.

    The seed is chosen from the central RIDGE_SEED_CENTRAL_FRAC of the slit
    length, using strongly averaged Y-binned profiles. This avoids anchoring
    the ridge at a faint or clipped end of the slit.
    """
    if profiles.ndim != 2 or ycenters.size == 0:
        return None, np.nan
    nbin, nx = profiles.shape
    yc = np.asarray(ycenters, float)
    ylo = np.nanmin(yc)
    yhi = np.nanmax(yc)
    if not np.isfinite(ylo) or not np.isfinite(yhi):
        return None, np.nan
    frac = float(np.clip(RIDGE_SEED_CENTRAL_FRAC, 0.1, 1.0))
    span = yhi - ylo
    half = 0.5 * frac * span
    ymid = 0.5 * (ylo + yhi)
    central = np.where((yc >= ymid - half) & (yc <= ymid + half))[0]
    if central.size == 0:
        central = np.arange(nbin, dtype=int)

    best_i = None
    best_x = np.nan
    best_score = -np.inf
    for i in central:
        x, w, contrast, signal = _profile_positive_weights(profiles[i])
        if w.size == 0 or (not np.isfinite(signal)) or signal <= 0:
            continue
        if (not np.isfinite(contrast)) or (contrast < RIDGE_SEED_MIN_CONTRAST):
            continue
        xcen = float((x * w).sum() / w.sum())
        if (not np.isfinite(xcen)) or xcen < 0 or xcen > (nx - 1):
            continue
        score = signal * max(contrast, 1e-12)
        if score > best_score:
            best_score = score
            best_i = int(i)
            best_x = xcen

    if best_i is None:
        # Fallback: use strongest bin anywhere in the slit.
        for i in range(nbin):
            x, w, contrast, signal = _profile_positive_weights(profiles[i])
            if w.size == 0 or (not np.isfinite(signal)) or signal <= 0:
                continue
            xcen = float((x * w).sum() / w.sum())
            if (not np.isfinite(xcen)) or xcen < 0 or xcen > (nx - 1):
                continue
            score = signal * max(float(contrast) if np.isfinite(contrast) else 0.0, 1e-12)
            if score > best_score:
                best_score = score
                best_i = int(i)
                best_x = xcen

    return best_i, best_x


def _corridor_centroid(profile, x_pred, halfwidth, x_center):
    """Centroid of positive signal inside a narrow X corridor around x_pred, with a strong prior toward the slit center."""
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
    if abs(xcen - float(x_center)) > float(MAX_CENTER_OFFSET):
        return np.nan, np.nan
    return xcen, contrast


def track_from_seed(profiles, seed_i, seed_x, x_center):
    """
    Track the ridge in coarse Y bins starting from a central seed and moving
    both upward and downward. Local motion is tightly constrained; the final
    row-by-row ridge is obtained later from a stiff global fit.
    """
    if profiles.ndim != 2:
        return np.array([], dtype=float)
    nbin, nx = profiles.shape
    xb = np.full(nbin, np.nan, float)
    if seed_i is None or (not np.isfinite(seed_x)):
        return xb
    xb[int(seed_i)] = float(seed_x)

    def _walk(start, stop, step):
        prev = float(seed_x) if start == seed_i + step or start == seed_i - 1 else np.nan
        for i in range(start, stop, step):
            if not np.isfinite(prev):
                break
            xcen, contrast = _corridor_centroid(profiles[i], prev, RIDGE_TRACK_GATE, x_center)
            if (not np.isfinite(xcen)) or (not np.isfinite(contrast)) or (contrast < RIDGE_TRACK_MIN_CONTRAST):
                # keep this bin undefined and let the global fit bridge it
                continue
            if abs(xcen - prev) > float(RIDGE_TRACK_JUMP):
                xcen = prev + np.clip(xcen - prev, -float(RIDGE_TRACK_JUMP), float(RIDGE_TRACK_JUMP))
            xb[i] = float(np.clip(xcen, 0.0, nx - 1.0))
            prev = xb[i]

    _walk(seed_i + 1, nbin, +1)
    _walk(seed_i - 1, -1, -1)
    return xb


def measure_binned_centroids_stiff(profiles):
    """Independent centroid per coarse Y-bin (stiff-ridge fallback)."""
    if profiles.ndim != 2:
        return np.array([], dtype=float)
    nbin, nx = profiles.shape
    x = np.arange(nx, dtype=float)
    xb = np.full(nbin, np.nan, float)
    for i in range(nbin):
        p = np.asarray(profiles[i], float)
        if not np.isfinite(p).any():
            continue
        base = np.nanmedian(p)
        w = p - base
        w[~np.isfinite(w)] = 0.0
        w[w < 0] = 0.0
        if w.sum() <= 0:
            continue
        xb[i] = float((x * w).sum() / w.sum())
    return xb


def fit_stiff_ridge(ycenters, xb, ny):
    """
    Fit a stiff global ridge x(y) from coarse-bin tracked centers.

    The ridge is allowed to drift gently in X but is not clipped to the
    quartz-mask X boundaries. Outlying bins are rejected before the final fit.
    """
    x0 = np.full(ny, np.nan, float)
    ok = np.isfinite(xb) & np.isfinite(ycenters)
    if ok.sum() < max(2, RIDGE_MIN_BINS):
        if ok.sum() == 1:
            x0[:] = float(xb[ok][0])
        return x0

    order = int(min(RIDGE_FIT_ORDER, ok.sum() - 1))
    coeff = np.polyfit(ycenters[ok], xb[ok], order)
    model = np.polyval(coeff, ycenters)
    resid = xb - model
    ok2 = ok & (np.abs(resid) <= float(RIDGE_OUTLIER_PIX))

    if ok2.sum() >= max(2, RIDGE_MIN_BINS):
        order = int(min(RIDGE_FIT_ORDER, ok2.sum() - 1))
        coeff = np.polyfit(ycenters[ok2], xb[ok2], order)

    y = np.arange(ny, dtype=float)
    x0 = np.polyval(coeff, y)
    return x0


def ridge_x0_per_row(img, return_mode=False):
    ny, nx = img.shape
    x_center = estimate_trace_center(img)
    profiles, ycenters = build_binned_profiles(img, YBIN)
    if profiles.size == 0 or ycenters.size == 0:
        x0 = np.full(ny, x_center, float)
        mode = "SKYONLY"
        return (x0, mode) if return_mode else x0

    mode = "CENTERSEEDED"

    # First try center-seeded tracking
    seed_i, seed_x = find_center_seed_bin(profiles, ycenters)
    if np.isfinite(seed_x) and abs(seed_x - x_center) > float(MAX_CENTER_OFFSET):
        seed_x = np.nan
        seed_i = None
    xb = track_from_seed(profiles, seed_i, seed_x, x_center)
    tracked_frac = float(np.mean(np.isfinite(xb))) if xb.size else 0.0
    use_stiff = (not RIDGE_AUTO_MODE) and False
    if RIDGE_AUTO_MODE:
        if RIDGE_SEED_REQUIRED and (seed_i is None or (not np.isfinite(seed_x))):
            use_stiff = True
        elif tracked_frac < float(RIDGE_MIN_TRACKED_FRAC):
            use_stiff = True

    if use_stiff:
        xb = measure_binned_centroids_stiff(profiles)
        xb[np.abs(xb - x_center) > float(MAX_CENTER_OFFSET)] = np.nan
        mode = "STIFFRIDGE"

    x0 = fit_stiff_ridge(ycenters, xb, ny)
    x0 = savgol_smooth(x0, SMOOTH_WIN, SMOOTH_POLY)

    valid_frac = float(np.mean(np.isfinite(x0))) if x0.size else 0.0
    med_offset = float(np.nanmedian(np.abs(x0 - x_center))) if np.isfinite(x0).any() else np.inf

    if valid_frac < float(SKYONLY_MIN_VALID_FRAC):
        x0 = np.full(ny, x_center, float)
        mode = "SKYONLY"
    elif not np.isfinite(med_offset) or med_offset > float(MAX_CENTER_OFFSET):
        x0 = np.full(ny, x_center, float)
        mode = "CENTERLOCK"

    return (x0, mode) if return_mode else x0

"""
def _low_fraction_median(vals, frac):
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return np.nan
    frac = float(np.clip(frac, 0.1, 1.0))
    k = int(np.ceil(frac * v.size)); k = max(1, min(v.size, k))
    vv = np.partition(v, k - 1)[:k]
    return float(np.median(vv))
"""
def reject_low_tail(vals, frac=0.15, nmin_keep=8):
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return v

    frac = float(np.clip(frac, 0.0, 0.45))
    if frac <= 0 or v.size < max(2 * nmin_keep, 10):
        return v

    k = int(np.floor(frac * v.size))
    if k <= 0 or (v.size - k) < nmin_keep:
        return v

    idx = np.argsort(v)
    return v[idx[k:]]

def estimate_sky_pooled(img: np.ndarray, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hybrid sky estimator.

    Priority:
      1) Use current-row sky only (preserves narrow sky lines; no y-blurring)
      2) If too few valid sky pixels, fall back to pooled sky over y±YSKYWIN

    Returns:
      sky[y], sky_sigma[y], n_sky_samples[y]
    """
    ny, nx = img.shape
    x = np.arange(nx, dtype=float)

    sky = np.full(ny, np.nan, float)
    sky_sig = np.full(ny, np.nan, float)
    nsky = np.zeros(ny, int)

    fin = np.isfinite(img)
    edge = (W_OBJ + GAP)

    # Per-row side choice and masks
    side = np.zeros(ny, int)  # -1 = left, +1 = right, 0 = none
    base_mask_rows = np.zeros((ny, nx), dtype=bool)
    excl_mask_rows = np.zeros((ny, nx), dtype=bool)

    xmin = np.full(ny, np.nan, float)
    xmax = np.full(ny, np.nan, float)
    for y in range(ny):
        xs = np.where(fin[y])[0]
        if xs.size:
            xmin[y] = float(xs[0])
            xmax[y] = float(xs[-1])

    # Build row masks
    for y in range(ny):
        if not np.isfinite(x0[y]):
            continue

        dx = x - x0[y]
        left = fin[y] & (dx <= -edge)
        right = fin[y] & (dx >= edge)

        nL = int(left.sum())
        nR = int(right.sum())
        if nL == 0 and nR == 0:
            continue

        use_left = (nL >= nR)
        if use_left:
            side[y] = -1
            base = left
            excl = left.copy()
            if np.isfinite(xmin[y]) and SKY_EDGE_EXCLUDE_LEFT > 0:
                excl &= (x >= xmin[y] + SKY_EDGE_EXCLUDE_LEFT)
        else:
            side[y] = +1
            base = right
            excl = right.copy()
            if np.isfinite(xmax[y]) and SKY_EDGE_EXCLUDE_RIGHT > 0:
                excl &= (x <= xmax[y] - SKY_EDGE_EXCLUDE_RIGHT)

        base_mask_rows[y] = base
        excl_mask_rows[y] = excl if excl.any() else base

    def _robust_sky_from_vals(vals: np.ndarray, side_sign: int):
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return np.nan, np.nan, 0

        keep = sigma_clip_high_mask(vals, k=SKY_CLIP_K, iters=SKY_CLIP_ITERS)
        vals2 = vals[keep] if keep.sum() >= max(5, SKY_NMIN // 2) else vals
        """
        if side_sign < 0:
            sk = _low_fraction_median(vals2, SKY_LOW_FRAC_LEFT)
        else:
            sk = _low_fraction_median(vals2, SKY_LOW_FRAC_RIGHT)
        """ 
        frac = 0.15 if side_sign < 0 else 0.15
        vals3 = reject_low_tail(vals2, frac=frac)
        
        if vals3.size < max(3, SKY_NMIN // 2):
            vals3 = vals2
        
        sk = float(np.median(vals3))        
                
        return float(sk), float(mad_sigma(vals2)), int(vals2.size)

    def _pool_vals(y0: int, y1: int, mask_rows: np.ndarray) -> np.ndarray:
        m = mask_rows[y0:y1, :]
        if not m.any():
            return np.array([], dtype=float)
        v = img[y0:y1, :][m]
        return v[np.isfinite(v)]

    for y in range(ny):
        if side[y] == 0:
            continue

        # -------------------------------------------------
        # First choice: current row only (no skyline blur)
        # -------------------------------------------------
        vals_row = img[y, :][excl_mask_rows[y]]
        if vals_row.size < SKY_NMIN:
            vals_row = img[y, :][base_mask_rows[y]]

        sk, sig, nn = _robust_sky_from_vals(vals_row, side[y])

        if nn >= SKY_NMIN and np.isfinite(sk):
            sky[y] = sk
            sky_sig[y] = sig
            nsky[y] = nn
            continue

        # -------------------------------------------------
        # Fallback: pooled rows if row-only sky is too sparse
        # -------------------------------------------------
        y0 = max(0, y - YSKYWIN)
        y1 = min(ny, y + YSKYWIN + 1)

        vals = _pool_vals(y0, y1, excl_mask_rows)
        if vals.size < SKY_NMIN:
            vals = _pool_vals(y0, y1, base_mask_rows)

        if vals.size < SKY_NMIN:
            # continuity fallback
            if y > 0 and np.isfinite(sky[y - 1]):
                sky[y] = sky[y - 1]
                sky_sig[y] = sky_sig[y - 1]
                nsky[y] = nsky[y - 1]
            continue

        sk, sig, nn = _robust_sky_from_vals(vals, side[y])
        sky[y] = sk
        sky_sig[y] = sig
        nsky[y] = nn

    # Fill remaining holes by interpolation
    ok = np.isfinite(sky)
    if ok.sum() >= 2:
        idx = np.arange(ny, dtype=float)
        sky[~ok] = np.interp(idx[~ok], idx[ok], sky[ok])

        ok2 = np.isfinite(sky_sig)
        if ok2.sum() >= 2:
            sky_sig[~ok2] = np.interp(idx[~ok2], idx[ok2], sky_sig[ok2])

    return sky, sky_sig, nsky


def extract_optimal(img, x0, sky, sky_sig, TEXP):
    ny, nx = img.shape
    x = np.arange(nx, dtype=float); fin = np.isfinite(img)
    flux = np.full(ny, np.nan, float); varf = np.full(ny, np.nan, float); nobj = np.zeros(ny, int)
    g = float(GAIN_E_PER_ADU); rn = float(READ_NOISE_E); T = float(TEXP)
    rn_term = (rn / (g * T)) ** 2
    for y in range(ny):
        if (not np.isfinite(x0[y])) or (not np.isfinite(sky[y])) or fin[y].sum() < 5:
            continue
        row = img[y].astype(float)
        dx = x - x0[y]
        obj_mask = fin[y] & (np.abs(dx) <= W_OBJ)
        nobj[y] = int(obj_mask.sum())
        if nobj[y] < 3:
            continue
        D = row - sky[y]
        P = np.exp(-0.5 * (dx / float(PROFILE_SIGMA)) ** 2)
        P[~obj_mask] = 0.0
        psum = P[obj_mask].sum()
        if psum <= 0 or (not np.isfinite(psum)):
            continue
        P = P / psum
        R = np.maximum(row, 0.0)
        V = (R / (g * T)) + rn_term
        if np.isfinite(sky_sig[y]) and sky_sig[y] > 0:
            V = V + float(sky_sig[y]) ** 2
        invV = np.zeros(nx, float); invV[obj_mask] = 1.0 / V[obj_mask]
        denom = np.sum((P[obj_mask] ** 2) * invV[obj_mask])
        if denom <= 0 or (not np.isfinite(denom)):
            continue
        numer = np.sum(P[obj_mask] * D[obj_mask] * invV[obj_mask])
        flux[y] = numer / denom; varf[y] = 1.0 / denom
    return flux, varf, nobj

def is_slit_image_hdu(h):
    name = (h.name or "").upper()
    return name.startswith("SLIT") and (h.data is not None) and (np.asarray(h.data).ndim == 2)

def main():
    if not IN_FITS.exists():
        raise FileNotFoundError(f"Input not found: {IN_FITS}")
    with fits.open(IN_FITS, memmap=False) as hdul:
        phdr = hdul[0].header
        TEXP0 = hdul[0].header.get("EXPTIME", phdr.get("EXPTIME", np.nan))
        out = fits.HDUList([fits.PrimaryHDU()])
        out[0].header["PIPESTEP"] = "STEP08"
        out[0].header["GEOM"] = "TRACECOORDS"
        out[0].header["INFILE"] = IN_FITS.name
        out[0].header["SLITSET"] = SET_TAG
        out[0].header["STEP08"] = "RIDGEGUIDED_POOLSKY"
        out[0].header["RIDGEAUT"] = (1 if RIDGE_AUTO_MODE else 0, "Auto fallback CENTERSEEDED->STIFFRIDGE")
        out[0].header["APCORR"] = (1, "Aperture-loss correction computed")
        out[0].header["APMINFR"] = (float(APMINFR), "Min enclosed fraction to apply correction")
        if np.isfinite(TEXP0):
            out[0].header["EXPTIME"] = float(TEXP0)
        n_slits = 0; n_flag_bad = 0; n_flag_empty = 0
        for h in hdul[1:]:
            if not isinstance(h, fits.ImageHDU) or not is_slit_image_hdu(h):
                continue
            slit = h.name
            img = np.asarray(h.data, float)
            ny, nx = img.shape
            exptime = h.header.get("EXPTIME", TEXP0)
            exptime = float(exptime) if exptime is not None else np.nan
            if not np.isfinite(exptime) or exptime <= 0:
                raise KeyError(f"{slit}: EXPTIME not found or invalid.")
            x_center = estimate_trace_center(img)
            center_signal = estimate_center_signal(img, x_center)
            bkg, bkg_sig = estimate_background(img)
            if np.isfinite(center_signal) and np.isfinite(bkg) and np.isfinite(bkg_sig) and bkg_sig > 0:
                sn_center = float((center_signal - bkg) / bkg_sig)
            else:
                sn_center = np.nan

            if (not np.isfinite(sn_center)) or (sn_center < float(SKY_SN_THRESHOLD)):
                x0 = np.full(ny, x_center, float)
                ridge_mode = "SKYONLY"
            else:
                x0, ridge_mode = ridge_x0_per_row(img, return_mode=True)

            ridge_scatter = float(np.nanstd(x0)) if np.isfinite(x0).any() else np.nan
            if ridge_mode not in ("SKYONLY",) and np.isfinite(ridge_scatter) and ridge_scatter > float(RIDGE_SCATTER_MAX):
                x0 = np.full(ny, x_center, float)
                ridge_mode = "CENTERLOCK"

            sky, sky_sig, nsky = estimate_sky_pooled(img, x0)
            flux, varf, nobj = extract_optimal(img, x0, sky, sky_sig, exptime)
            apfrac, edgeflag, xleft, xright = compute_aperture_loss_correction(img, x0)
            flux_apcorr = np.array(flux, copy=True)
            var_apcorr = np.array(varf, copy=True)
            good_corr = np.isfinite(apfrac) & (apfrac > 0) & (apfrac >= APMINFR) & np.isfinite(flux) & np.isfinite(varf)
            flux_apcorr[good_corr] = flux[good_corr] / apfrac[good_corr]
            var_apcorr[good_corr] = varf[good_corr] / (apfrac[good_corr] ** 2)
            ypix = np.arange(ny, dtype=np.int32)
            cols = [
                fits.Column(name="YPIX", format="J", array=ypix),
                fits.Column(name="FLUX", format="E", array=flux.astype(np.float32)),
                fits.Column(name="VAR", format="E", array=varf.astype(np.float32)),
                fits.Column(name="SKY", format="E", array=sky.astype(np.float32)),
                fits.Column(name="X0", format="E", array=x0.astype(np.float32)),
                fits.Column(name="NOBJ", format="J", array=nobj.astype(np.int32)),
                fits.Column(name="NSKY", format="J", array=nsky.astype(np.int32)),
                fits.Column(name="SKYSIG", format="E", array=sky_sig.astype(np.float32)),
                fits.Column(name="APLOSS_FRAC", format="E", array=apfrac.astype(np.float32)),
                fits.Column(name="FLUX_APCORR", format="E", array=flux_apcorr.astype(np.float32)),
                fits.Column(name="VAR_APCORR", format="E", array=var_apcorr.astype(np.float32)),
                fits.Column(name="EDGEFLAG", format="I", array=edgeflag.astype(np.int16)),
                fits.Column(name="TRXLEFT", format="E", array=xleft.astype(np.float32)),
                fits.Column(name="TRXRIGHT", format="E", array=xright.astype(np.float32)),
            ]
            tab = fits.BinTableHDU.from_columns(cols, name=slit)
            for k in ("SLITID","INDEX","RA","DEC","TRACESET","Y0WIN","YMIN","YMAX","XREF","ROT180","BKGID","XLO","XHI"):
                if k in h.header:
                    tab.header[k] = h.header[k]
            if "EXPTIME" in h.header:
                tab.header["EXPTIME"] = h.header["EXPTIME"]
            tab.header["NY"] = ny; tab.header["NX"] = nx; tab.header["YBIN"] = YBIN
            tab.header["RIDGEMOD"] = (ridge_mode, "Ridge mode used for this slit")
            tab.header["CTRSIG"] = (float(CENTER_SIGMA), "Center-prior sigma in ridge fit")
            tab.header["CTROFF"] = (float(MAX_CENTER_OFFSET), "Max allowed offset from slit center")
            tab.header["S08SNR"] = (float(sn_center) if np.isfinite(sn_center) else -999.0, "Central S/N estimate")
            tab.header["RIDGESC"] = (float(ridge_scatter) if np.isfinite(ridge_scatter) else -999.0, "Ridge scatter in pixels")
            tab.header["YSKYWIN"] = int(YSKYWIN); tab.header["WOBJ"] = float(W_OBJ); tab.header["GAP"] = float(GAP)
            tab.header["PSFSIG"] = float(PROFILE_SIGMA); tab.header["APCORR"] = (1, "Aperture-loss correction computed")
            tab.header["APMINFR"] = (float(APMINFR), "Min enclosed fraction to apply correction")
            tab.header["NEDGE1"] = (int(np.sum(edgeflag == 1)), "Rows clipped but corrected")
            tab.header["NEDGE2"] = (int(np.sum(edgeflag == 2)), "Rows too clipped / unsafe")
            lo0, lo1 = int(0.15 * ny), int(0.35 * ny); hi0, hi1 = int(0.65 * ny), int(0.85 * ny)
            lo_mask = (ypix >= lo0) & (ypix <= lo1); hi_mask = (ypix >= hi0) & (ypix <= hi1)
            sky_lo = float(np.nanmedian(sky[lo_mask])) if np.any(lo_mask) else np.nan
            sky_hi = float(np.nanmedian(sky[hi_mask])) if np.any(hi_mask) else np.nan
            flx_lo = float(np.nanmedian(flux[lo_mask])) if np.any(lo_mask) else np.nan
            flx_hi = float(np.nanmedian(flux[hi_mask])) if np.any(hi_mask) else np.nan
            r_sky = (sky_hi / sky_lo) if (np.isfinite(sky_lo) and sky_lo != 0.0) else np.nan
            r_flux = (flx_hi / flx_lo) if (np.isfinite(flx_lo) and flx_lo != 0.0) else np.nan
            med_nobj = float(np.nanmedian(nobj)) if nobj.size else np.nan
            flag_empty = bool(np.isfinite(med_nobj) and med_nobj < 1.0)
            flag_bad = bool((np.isfinite(r_sky) and r_sky > 2.0) and (np.isfinite(r_flux) and r_flux < 0.6))
            tab.header["S08RSKY"] = (float(r_sky) if np.isfinite(r_sky) else -1.0, "median SKY_hi / SKY_lo; -1=undef")
            tab.header["S08RFLX"] = (float(r_flux) if np.isfinite(r_flux) else -1.0, "median FLUX_hi / FLUX_lo; -1=undef")
            tab.header["S08BAD"] = (int(flag_bad), "1=sky/flux inconsistency")
            tab.header["S08EMP"] = (int(flag_empty), "1=empty extraction")
            n_slits += 1; n_flag_bad += int(flag_bad); n_flag_empty += int(flag_empty)
            print(f"[OK] {slit}: good_rows={int(np.isfinite(flux).sum())} apcorr_rows={int(np.sum(good_corr))}")
            out.append(tab)

        out[0].header["N_SLITS"] = int(n_slits)
        out[0].header["N_BAD"] = int(n_flag_bad)
        out[0].header["N_EMPTY"] = int(n_flag_empty)

        # --- ensure output directory exists ---
        OUT_FITS.parent.mkdir(parents=True, exist_ok=True)

        out.writeto(OUT_FITS, overwrite=True)
        print("[DONE] Wrote", OUT_FITS)
if __name__ == "__main__":
    main()
