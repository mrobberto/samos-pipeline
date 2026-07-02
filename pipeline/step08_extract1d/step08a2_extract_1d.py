#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step08a2 — ridge-guided optimal extraction from TRACECOORDS slit images.

PURPOSE
-------
Perform the final 1D spectral extraction using the ridge and slit-quality
products derived in Step08a1.

This stage converts TRACECOORDS slit images into calibrated 1D spectra
in pixel space, using a ridge-centered optimal extraction with robust
sky estimation and aperture-loss correction.

This is the photometric counterpart to Step08a1, which provides the
geometric and quality-control inputs.

INPUT
-----
1. Step06 TRACECOORDS slit images:
   *_EVEN_tracecoords.fits, *_ODD_tracecoords.fits

2. Step08a1 analysis FITS:
   trace_analysis_optimal_ridge_{even,odd}.fits
   Provides:
     - ridge X0(y)
     - TRACECOORDS slit boundaries (TRXLEFT/TRXRIGHT)
     - per-slit metadata

3. Step08a1 slit-quality CSV:
   step08a_slit_quality_{even,odd}.csv
   Provides:
     - GOOD / USE flags
     - FWHM estimates
     - diagnostic metrics

OUTPUT
------
extract1d_optimal_ridge_{even,odd}.fits

For each slit:
  YPIX           : dispersion pixel index
  FLUX           : optimal extracted flux (sky-subtracted)
  VAR            : variance of FLUX
  SKY            : scalar sky level per row
  OBJ_PRESKY     : simple aperture sum before sky subtraction
  X0             : ridge position
  NOBJ, NSKY     : number of object/sky pixels used
  SKYSIG         : sky noise estimate
  APLOSS_FRAC    : fraction of PSF inside valid slit
  FLUX_APCORR    : aperture-loss corrected flux
  VAR_APCORR     : corrected variance
  EDGEFLAG       : truncation flag
  TRXLEFT/RIGHT  : slit boundaries

SCIENTIFIC METHOD
-----------------

1. Ridge-guided extraction
   - Uses X0(y) from Step08a1 (no re-fitting here)
   - Extraction is centered on the science-determined ridge
   - Ridge is allowed to deviate from quartz geometry

2. Sky estimation (row-first, with fallback)
   - Primary: row-by-row sky estimation excluding object aperture
   - Rejects bright pixels via sigma-clipping
   - Applies low-tail selection to avoid object contamination
   - Fallback: pooled sky from neighboring rows (Y ± YSKYWIN)
   - Final fallback: continuity from previous valid row

   This preserves narrow OH features while remaining robust in
   crowded or low-S/N conditions.

3. Optimal extraction (Horne-style)
   - Gaussian spatial weighting centered on X0(y)
   - Per-slit PSF width derived from Step08a1 FWHM
   - Variance includes:
       • Poisson noise
       • read noise
       • sky noise

4. Aperture-loss correction
   - Computes fraction of PSF inside valid TRACECOORDS region
   - Applies correction when truncation is mild
   - Flags heavily truncated rows (EDGEFLAG)

5. Slit selection
   - Extraction is performed only if USE==1 (or GOOD fallback)
   - Non-selected slits are propagated with NaN spectral columns

ROBUSTNESS FEATURES
-------------------
- Ridge fully decoupled from extraction (no re-centering)
- Row-first sky preserves spectral line structure
- Pooled sky fallback prevents data loss in narrow slits
- Continuity fallback stabilizes sparse regions
- Aperture correction handles edge clipping
- Per-row and per-slit quality flags

DESIGN NOTES
------------
- Operates entirely in TRACECOORDS
- Does NOT modify ridge (geometry fixed from Step08a1)
- Preserves compatibility with Step08b/08c
- Overwrites placeholder spectral columns from Step08a1
- Designed to work with manual intervention via CSV

DOES NOT DO
-----------
- Ridge detection (Step08a1)
- Wavelength calibration (Step08c)
- OH/telluric correction (Step09)
- Flux calibration (Step11)

PIPELINE ROLE
-------------
Step08a1 → detect ridge, classify slits
Step08a2 → extract flux along ridge

Together they replace the legacy monolithic Step08a extraction.

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

parser = argparse.ArgumentParser(description="SAMOS Step08a2 1D extraction")
parser.add_argument("--set", choices=["EVEN", "ODD"], required=True, help="Slit set to process")
parser.add_argument("--csv", default="", help="Optional explicit slit-quality CSV")
args = parser.parse_args()
SET_TAG = args.set.upper()

# Step06 TRACECOORDS slit images
pattern = str(Path(config.ST06_SCIENCE) / f"*_{SET_TAG}_tracecoords.fits")
matches = sorted(glob(pattern))
if not matches:
    raise FileNotFoundError(f"No TRACECOORDS file found for {SET_TAG}: {pattern}")
IN_FITS = Path(matches[-1])

ST08 = Path(config.ST08_EXTRACT1D)

# 08a1 output: ridge/trace analysis only
ANALYSIS_FITS = ST08 / f"trace_analysis_optimal_ridge_{SET_TAG.lower()}.fits"

# 08a2 output: real extracted spectra
OUT_FITS = ST08 / f"extract1d_optimal_ridge_{SET_TAG.lower()}.fits"

# editable slit-quality table from 08a1
CSV = Path(args.csv) if args.csv else ST08 / f"step08a_slit_quality_{SET_TAG.lower()}.csv"

if not ANALYSIS_FITS.exists():
    raise FileNotFoundError(f"Analysis FITS not found. Run 08a1 first: {ANALYSIS_FITS}")

if not CSV.exists():
    raise FileNotFoundError(f"Quality CSV not found. Run 08a1 first: {CSV}")
    
    
W_OBJ = 3.0
GAP = 0.0
YSKYWIN = 3
PROFILE_SIGMA = 2.0
SKY_CLIP_K = 3.0
SKY_CLIP_ITERS = 2
SKY_NMIN = 2
SKY_EDGE_EXCLUDE_LEFT = 0
SKY_EDGE_EXCLUDE_RIGHT = 0
SKY_LOW_FRAC_LEFT = 0.15
SKY_LOW_FRAC_RIGHT = 0.15
APMINFR = 0.55
GAIN_E_PER_ADU = float(config.GAIN_E_PER_ADU)
READ_NOISE_E = float(config.READNOISE_E)
SQRT2 = sqrt(2.0)


def mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def sigma_clip_high_mask(x: np.ndarray, k: float = 3.0, iters: int = 2) -> np.ndarray:
    ok = np.isfinite(x)
    if ok.sum() < 5:
        return ok
    m = np.median(x[ok])
    s = mad_sigma(x[ok])
    if not np.isfinite(s) or s <= 0:
        return ok
    for _ in range(iters):
        thr = m + k * s
        ok2 = ok & (x <= thr)
        if ok2.sum() < 5 or ok2.sum() == ok.sum():
            ok = ok2
            break
        ok = ok2
        m = np.median(x[ok])
        s = mad_sigma(x[ok])
        if not np.isfinite(s) or s <= 0:
            break
    return ok


def gaussian_cdf_scalar(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1.0 + erf((x - mu) / (SQRT2 * sigma)))


def gaussian_integral(lo: float, hi: float, mu: float, sigma: float) -> float:
    return gaussian_cdf_scalar(hi, mu, sigma) - gaussian_cdf_scalar(lo, mu, sigma)


def valid_interval_from_row(row_valid: np.ndarray) -> tuple[float, float]:
    idx = np.where(row_valid)[0]
    if idx.size == 0:
        return np.nan, np.nan
    return float(idx[0]) - 0.5, float(idx[-1]) + 0.5

def set_hdr_float_safe(hdr, key, value, comment=""):
    if value is None or not np.isfinite(value):
        hdr[key] = ("UNDEF", comment)
    else:
        hdr[key] = (float(value), comment)

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


def low_fraction_median(vals: np.ndarray, frac: float) -> float:
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    frac = float(np.clip(frac, 0.1, 1.0))
    k = int(np.ceil(frac * v.size))
    k = max(1, min(v.size, k))
    vv = np.partition(v, k - 1)[:k]
    return float(np.median(vv))


def estimate_sky_row(row: np.ndarray, x0: float) -> tuple[float, int, float]:
    nx = row.size
    xx = np.arange(nx, dtype=float)
    dx = xx - float(x0)
    valid = np.isfinite(row)

    obj_mask = valid & (np.abs(dx) <= W_OBJ)
    if obj_mask.sum() < 2:
        return np.nan, 0, np.nan

    # Candidate sky pixels outside object aperture
    sky_mask = valid & (~obj_mask)

    if SKY_EDGE_EXCLUDE_LEFT > 0:
        sky_mask[:SKY_EDGE_EXCLUDE_LEFT] = False
    if SKY_EDGE_EXCLUDE_RIGHT > 0:
        sky_mask[-SKY_EDGE_EXCLUDE_RIGHT:] = False

    vals = np.asarray(row[sky_mask], float)
    vals = vals[np.isfinite(vals)]

    if vals.size < SKY_NMIN:
        return np.nan, 0, np.nan

    # Use faintest valid outside-aperture pixels.
    # For very narrow slits, keep at least 2–3 pixels.
    frac = 0.35
    nkeep = max(SKY_NMIN, int(np.ceil(frac * vals.size)))
    nkeep = min(nkeep, vals.size)

    use = np.sort(vals)[:nkeep]

    if use.size < SKY_NMIN:
        return np.nan, 0, np.nan

    skylev = float(np.median(use))
    skysig = float(mad_sigma(use))

    # If MAD is undefined with very few pixels, use a conservative fallback.
    if not np.isfinite(skysig):
        skysig = float(np.std(use)) if use.size > 1 else np.nan

    return skylev, int(use.size), skysig

def extract_one_slit(img: np.ndarray, x0: np.ndarray, profile_sigma_slit: float):
    """
    Extract one slit in TRACECOORDS using a ridge-centered Gaussian-weighted extraction.

    Inputs
    ------
    img : 2D array
        TRACECOORDS slit image
    x0 : 1D array
        ridge center x(y)
    profile_sigma_slit : float
        Gaussian sigma for the optimal-extraction weights, derived per slit
        from the brightest-block FWHM in 08a1

    Notes
    -----
    - OBJ_PRESKY is a simple aperture sum before sky subtraction
    - FLUX is the weighted optimal extraction after sky subtraction
    - SKY stores the scalar sky level used for each row
    - The pooled-sky fallback is deliberately permissive: if left/right sky
      windows are too sparse, it uses all valid pixels outside the object aperture
    """
    ny, nx = img.shape
    xx = np.arange(nx, dtype=float)

    flux = np.full(ny, np.nan, float)
    var = np.full(ny, np.nan, float)
    sky = np.full(ny, np.nan, float)
    obj_presky = np.full(ny, np.nan, float)

    nobj = np.zeros(ny, dtype=np.int32)
    nsky = np.zeros(ny, dtype=np.int32)
    skysig = np.full(ny, np.nan, float)

    apfrac = np.full(ny, np.nan, float)
    flux_ap = np.full(ny, np.nan, float)
    var_ap = np.full(ny, np.nan, float)

    edgeflag = np.zeros(ny, dtype=np.int16)
    trxleft = np.full(ny, np.nan, float)
    trxright = np.full(ny, np.nan, float)

    aper_halfwidth = float(W_OBJ) + 0.5

    # Reuse the last valid sky estimate if an isolated row fails.
    last_good_sky = np.nan
    last_good_sig = np.nan

    for y in range(ny):
        # Skip rows where the ridge is undefined.
        if not np.isfinite(x0[y]):
            continue

        row = np.asarray(img[y], float)
        valid = np.isfinite(row)

        # Record the valid TRACECOORDS footprint for this row.
        xL, xR = valid_interval_from_row(valid)
        trxleft[y] = xL
        trxright[y] = xR

        # Aperture-loss estimate using the same Gaussian PSF width used for extraction.
        frac = aperture_capture_fraction_gaussian(
            x0[y], aper_halfwidth, profile_sigma_slit, xL, xR
        )
        apfrac[y] = frac

        a0 = x0[y] - aper_halfwidth
        a1 = x0[y] + aper_halfwidth
        clipped = (not np.isfinite(xL)) or (not np.isfinite(xR)) or (xL > a0) or (xR < a1)
        if clipped:
            edgeflag[y] = 1
        if np.isfinite(frac) and frac < APMINFR:
            edgeflag[y] = 2

        # Object aperture centered on the ridge.
        dx = xx - float(x0[y])
        obj_mask = valid & (np.abs(dx) <= W_OBJ)
        if obj_mask.sum() < 2:
            continue
        
        # Always save pre-sky object aperture sum, even if sky fails later.
        obj_presky[y] = float(np.nansum(row[obj_mask]))
        nobj[y] = int(obj_mask.sum())


        # --- First try the nominal row-by-row sky estimate ---
        skylev, nsky_y, skysig_y = estimate_sky_row(row, x0[y])

        # --- If that fails, pool sky from neighboring rows ---
        if not np.isfinite(skylev):
            ylo = max(0, y - YSKYWIN)
            yhi = min(ny, y + YSKYWIN + 1)

            pooled = []

            for yy in range(ylo, yhi):
                if yy == y:
                    continue

                rr = np.asarray(img[yy], float)
                v = np.isfinite(rr)

                # Use the neighboring row's own ridge if available.
                # This is important when the source drifts with y.
                xref = float(x0[yy]) if np.isfinite(x0[yy]) else float(x0[y])
                ddx = xx - xref

                obj_neigh = v & (np.abs(ddx) <= W_OBJ)
                left = v & (ddx <= -(W_OBJ + GAP))
                right = v & (ddx >= +(W_OBJ + GAP))

                if SKY_EDGE_EXCLUDE_LEFT > 0:
                    left[:SKY_EDGE_EXCLUDE_LEFT] = False
                if SKY_EDGE_EXCLUDE_RIGHT > 0:
                    right[-SKY_EDGE_EXCLUDE_RIGHT:] = False

                # If the side windows are too sparse, fall back to all valid pixels
                # outside the object aperture. This is intentionally permissive, so
                # narrow TRACECOORDS slits do not end up with no sky at all.
                if int(left.sum()) + int(right.sum()) < SKY_NMIN:
                    sky_mask = v & (~obj_neigh)
                    pooled.extend(rr[sky_mask].tolist())
                else:
                    pooled.extend(rr[left].tolist())
                    pooled.extend(rr[right].tolist())

            pooled = np.asarray(pooled, float)

            if pooled.size >= SKY_NMIN:
                ok = sigma_clip_high_mask(pooled, k=SKY_CLIP_K, iters=SKY_CLIP_ITERS)
                use = pooled[ok]
                if use.size >= SKY_NMIN:
                    skylev = float(np.median(use))
                    skysig_y = float(mad_sigma(use))
                    nsky_y = int(use.size)

        # --- Last-resort continuity fallback: use previous valid sky ---
        if not np.isfinite(skylev) and np.isfinite(last_good_sky):
            skylev, skysig_y, nsky_y = last_good_sky, last_good_sig, 0

        # Still no sky? Skip this row.
        if not np.isfinite(skylev):
            continue

        last_good_sky = skylev
        last_good_sig = skysig_y

        # Gaussian optimal-extraction profile for this slit.
        prof = np.exp(-0.5 * (dx / profile_sigma_slit) ** 2)
        prof[~obj_mask] = 0.0

        if prof.sum() <= 0:
            continue

        prof /= prof.sum()

        # Row after subtraction of the scalar sky level.
        row_sub = row - skylev

        nobj[y] = int(obj_mask.sum())
        nsky[y] = int(nsky_y)
        skysig[y] = float(skysig_y) if np.isfinite(skysig_y) else np.nan
        sky[y] = float(skylev)

        # Horne-style weighted extraction.
        var_pix = (
            np.abs(row[obj_mask]) * GAIN_E_PER_ADU
            + (READ_NOISE_E ** 2)
            + (skysig[y] * GAIN_E_PER_ADU) ** 2
        )
        var_pix = np.asarray(var_pix, float)

        prof_use = prof[obj_mask]
        den = np.nansum((prof_use ** 2) / np.maximum(var_pix, 1e-6))
        num = np.nansum((prof_use * row_sub[obj_mask]) / np.maximum(var_pix, 1e-6))

        if den <= 0:
            flux[y] = np.nan
            var[y] = np.nan
        else:
            flux[y] = float(num / den)
            var[y] = float(1.0 / den)

        # Aperture-loss correction.
        if np.isfinite(frac) and frac > 0:
            flux_ap[y] = flux[y] / frac if np.isfinite(flux[y]) else np.nan
            var_ap[y] = var[y] / (frac ** 2) if np.isfinite(var[y]) else np.nan
        else:
            flux_ap[y] = np.nan
            var_ap[y] = np.nan

    return dict(
        FLUX=flux,
        VAR=var,
        SKY=sky,
        OBJ_PRESKY=obj_presky,
        X0=x0,
        NOBJ=nobj,
        NSKY=nsky,
        SKYSIG=skysig,
        APLOSS_FRAC=apfrac,
        FLUX_APCORR=flux_ap,
        VAR_APCORR=var_ap,
        EDGEFLAG=edgeflag,
        TRXLEFT=trxleft,
        TRXRIGHT=trxright,
    )

def replace_table_data(tab_hdu: fits.BinTableHDU, arrays: dict[str, np.ndarray]) -> fits.BinTableHDU:
    cols = []
    names = list(tab_hdu.columns.names)
    for name in names:
        arr = arrays[name]
        if arr.dtype.kind in "iu":
            fmt = "J"
        else:
            fmt = "E"
        cols.append(fits.Column(name=name, format=fmt, array=arr))
    out = fits.BinTableHDU.from_columns(cols, header=tab_hdu.header.copy(), name=tab_hdu.name)
    return out


df = pd.read_csv(CSV)
df = df.set_index("SLIT")

with fits.open(IN_FITS, memmap=False) as h06, fits.open(ANALYSIS_FITS, memmap=False) as h08:
    imgs = { (h.name or "").upper(): np.asarray(h.data, float)
             for h in h06[1:] if (h.name or "").upper().startswith("SLIT") and h.data is not None and np.asarray(h.data).ndim == 2 }
    out_hdus = [fits.PrimaryHDU(header=h08[0].header.copy())]
    out_hdus[0].header["PIPESTEP"] = ("STEP08", "Pipeline step")
    out_hdus[0].header["STAGE"] = ("08a2", "Extraction stage")
    out_hdus[0].header["PRODUCT"] = ("EXTRACT1D", "Product type")
    out_hdus[0].header["TRACESET"] = (SET_TAG, "EVEN or ODD")
    out_hdus[0].header["ANAFITS"] = ANALYSIS_FITS.name
    out_hdus[0].header["QUALCSV"] = CSV.name
    for h in h08[1:]:
        slit = (h.name or "").upper()
        if slit not in imgs or slit not in df.index:
            out_hdus.append(h.copy())
            continue
        img = imgs[slit]
        tab = h.data
        hdr = h.header.copy()
        x0 = np.asarray(tab["X0"], float)
        row = df.loc[slit]
        
        fwhm_slit = float(row["FWHM"]) if pd.notna(row["FWHM"]) else np.nan
        #
        if np.isfinite(fwhm_slit) and fwhm_slit > 0:
            profile_sigma_slit = fwhm_slit / 2.355
        else:
            profile_sigma_slit = float(PROFILE_SIGMA)
    
        use = int(row["USE"]) if "USE" in row.index and pd.notna(row["USE"]) else int(row["GOOD"])
        
        # Always compute aperture-sum products so OBJ_PRESKY is available for all slits.
        arrays = extract_one_slit(img, x0, profile_sigma_slit)
        arrays["YPIX"] = np.arange(img.shape[0], dtype=np.int32)
        
        # If slit is not selected for local sky-subtracted extraction, preserve only
        # the geometry and pre-sky signal. Leave sky-subtracted products NaN.
        if not use:
            arrays["FLUX"][:] = np.nan
            arrays["VAR"][:] = np.nan
            arrays["SKY"][:] = np.nan
            arrays["NSKY"][:] = 0
            arrays["SKYSIG"][:] = np.nan
            arrays["FLUX_APCORR"][:] = np.nan
            arrays["VAR_APCORR"][:] = np.nan
        
        newh = replace_table_data(h, arrays)
        
        # Preserve geometry metadata needed downstream by 08c.
        for key in ["Y0DET", "YMIN", "SHIFT2M", "SLITID"]:
            if key in h.header:
                newh.header[key] = (h.header[key], h.header.comments[key])
        
        newh.header["S08GOOD"] = int(row["GOOD"])
        newh.header["S08USE"] = int(use)
        newh.header["S08CLAS"] = str(row["CLASS"])
        newh.header["PSFSIG"] = (float(profile_sigma_slit), "Gaussian sigma used for extraction")
        set_hdr_float_safe(newh.header, "S08FWHM", fwhm_slit, "Brightest-block FWHM from 08a1/CSV")
        set_hdr_float_safe(
            newh.header,
            "S08DXC",
            float(row["DX_CENTER"]) if pd.notna(row["DX_CENTER"]) else np.nan,
            "Peak minus slit center",
        )
        
        out_hdus.append(newh)

fits.HDUList(out_hdus).writeto(OUT_FITS, overwrite=True)
print(f"Wrote {OUT_FITS}")
