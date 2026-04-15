#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 05 — build pixflat (EVEN + ODD)

This step builds *pixel-to-pixel* correction images using quartz illumination,
following the "classic" proven logic:

    quartz_diff = QUARTZ_SLITS_ON_* - QUARTZ_SLITS_OFF
    illum2d     = heavily smoothed quartz_diff within the trace mask
    pixflat     = quartz_diff / illum2d   (clipped + renormalized to median ~1)

We build TWO pixflats (Option A):
  - Even_traces_*.fits using config.QUARTZ_SLITS_ON_EVEN - config.QUARTZ_SLITS_OFF
  - Odd_traces_*.fits  using config.QUARTZ_SLITS_ON_ODD  - config.QUARTZ_SLITS_OFF

Mask choice:
  Prefer geometry-regularized masks if present:
    Even_traces_mask_reg.fits / Odd_traces_mask_reg.fits
  Otherwise fall back to:
    Even_traces_mask.fits / Odd_traces_mask.fits

*** Input files are defined in the config.py ***

Outputs are written into config.ST05_PIXFLAT.

Products per set (TRACE_BASE = Even_traces or Odd_traces):
  - {TRACE_BASE}_QuartzDiff_for_pixflat.fits
  - {TRACE_BASE}_Illum2D_quartzdiff.fits
  - {TRACE_BASE}_PixelFlat_from_quartz_diff.fits   (main pixflat)

Run 
---
from repo root:
    PYTHONPATH=. python pipeline/step05_pixflat/step05_build_pixflat.py
    
Or in Spyder:

runfile("pipeline/step05_pixflat/step05_build_pixflat.py")    
"""

import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_erosion

import config

in_dir = Path(config.ST03P5_ROWSTRIPE)

logger = logging.getLogger("step05_pixflat")

# -----------------------------------------------------------------------------
# User-tunable parameters
# -----------------------------------------------------------------------------

# Smoothing for illum2d model. Larger => keep only low-frequency illumination.
SIGMA_Y = 75.0
SIGMA_X = 12.0

# Erode mask edges so illum2d fit ignores trace boundaries (recommended)
MASK_EROSION_ITERS = 2

# Clip pixflat to avoid extreme corrections (safe-guard)
CLIP_LO = 0.5
CLIP_HI = 2.0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _nan_gaussian_filter(img: np.ndarray, sigma_y: float, sigma_x: float) -> np.ndarray:
    """Gaussian filter that ignores NaNs by filtering numerator/denominator."""
    img = np.asarray(img, float)
    w = np.isfinite(img).astype(float)
    num = np.nan_to_num(img, nan=0.0)
    num_f = gaussian_filter(num, sigma=(sigma_y, sigma_x), mode="nearest")
    den_f = gaussian_filter(w,   sigma=(sigma_y, sigma_x), mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num_f / den_f
    out[den_f <= 0] = np.nan
    return out



def _pick_mask(st04: Path, trace_base: str) -> Path:
    """Prefer *_mask_reg.fits if present, else *_mask.fits."""
    reg = st04 / f"{trace_base}_mask_reg.fits"
    if reg.exists():
        return reg
    raw = st04 / f"{trace_base}_mask.fits"
    return raw


def _build_one(trace_set: str, trace_base: str, qa_name: str, qb_name: str, st04: Path, st05: Path) -> None:
    """Build pixflat products for one of {EVEN, ODD}."""
    mask_file = _pick_mask(st04, trace_base)
    if not mask_file.exists():
        raise FileNotFoundError(f"Missing {mask_file}. Run Step 04 for {trace_set} first.")

    file_a = in_dir / Path(qa_name).name
    file_b = in_dir / Path(qb_name).name
    if not file_a.exists():
        raise FileNotFoundError(f"Quartz A not found: {file_a}")
    if not file_b.exists():
        raise FileNotFoundError(f"Quartz B not found: {file_b}")

    logger.info("[%s] Quartz A: %s", trace_set, file_a)
    logger.info("[%s] Quartz B: %s", trace_set, file_b)
    logger.info("[%s] Mask:     %s", trace_set, mask_file)

    with fits.open(file_a) as hdul:
        img_a = hdul[0].data.astype(np.float32)
        hdr_a = hdul[0].header
    with fits.open(file_b) as hdul:
        img_b = hdul[0].data.astype(np.float32)
        hdr_b = hdul[0].header

    if img_a.shape != img_b.shape:
        raise ValueError(f"[{trace_set}] Quartz shape mismatch: {img_a.shape} vs {img_b.shape}")

    quartz = (img_b - img_a).astype(np.float32)

    # load mask (+ header for provenance)
    with fits.open(mask_file) as hdul:
        mask = (hdul[0].data > 0)
        hdr_mask = hdul[0].header

    # erode edges for illum2d model
    if MASK_EROSION_ITERS > 0:
        mask_er = binary_erosion(mask, iterations=int(MASK_EROSION_ITERS))
    else:
        mask_er = mask

    kept = float(mask_er.sum()) / float(max(mask.sum(), 1))
    logger.info("[%s] Mask erosion: %d iters | kept %.3f", trace_set, int(MASK_EROSION_ITERS), kept)

    # illum2d model inside traces
    q = quartz.astype(float)
    q[~mask_er] = np.nan
    illum2d = _nan_gaussian_filter(q, sigma_y=float(SIGMA_Y), sigma_x=float(SIGMA_X)).astype(np.float32)

    # pixflat: residual correction within mask
    pixflat = np.ones_like(quartz, dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        pf = (quartz / illum2d).astype(np.float32)

    # clip & renormalize (inside mask)
    m = mask & np.isfinite(pf)
    if np.any(m):
        pf = np.clip(pf, float(CLIP_LO), float(CLIP_HI))
        med = float(np.nanmedian(pf[m]))
        if np.isfinite(med) and med > 0:
            pf = pf / med
        pixflat[m] = pf[m]

    # Headers
    hdr = hdr_b.copy()
    # Propagate key provenance from Step04 mask (e.g. rotation/orientation bookkeeping)
    # NOTE: pixflat is not per-slit, so we only carry global provenance keywords.
    for k in ("ROT180", "XFLIP", "YFLIP", "BKGID", "NSLITS", "NSLDET", "NSLTAB", "SLTMISM"):
        if 'hdr_mask' in locals() and k in hdr_mask:
            hdr[k] = hdr_mask[k]
    hdr["ST04MASK"] = (mask_file.name, "Step04 mask source (trace footprint)")
    hdr.add_history(f"Mask for pixflat: {mask_file.name} (from Step04)")
    hdr.add_history(f"Pixflat ({trace_set}) built from quartz: {file_b.name} - {file_a.name}")
    hdr["TRACESET"] = (trace_set, "Trace set used for mask (EVEN/ODD)")
    hdr["QZOFF"] = (Path(file_a).name, "Quartz slits-off exposure")
    hdr["QZON"] = (Path(file_b).name, "Quartz slits-on exposure")
    hdr["MASKFILE"] = (mask_file.name, "Trace mask used")
    hdr["SY"] = (float(SIGMA_Y), "Illum2D smoothing sigma_y (px)")
    hdr["SX"] = (float(SIGMA_X), "Illum2D smoothing sigma_x (px)")
    hdr["ERODE"] = (int(MASK_EROSION_ITERS), "Mask erosion iterations")
    hdr["CLIPLO"] = (float(CLIP_LO), "Pixflat clip low")
    hdr["CLIPHI"] = (float(CLIP_HI), "Pixflat clip high")

    # Write products
    if trace_set == "EVEN":
        q_path = st05 / "quartz_diff_even.fits"
        i_path = st05 / "illum2d_even.fits"
        p_path = Path(getattr(config, "PIXFLAT_EVEN", st05 / "PixelFlat_from_quartz_diff_EVEN.fits"))
    elif trace_set == "ODD":
        q_path = st05 / "quartz_diff_odd.fits"
        i_path = st05 / "illum2d_odd.fits"
        p_path = Path(getattr(config, "PIXFLAT_ODD", st05 / "PixelFlat_from_quartz_diff_ODD.fits"))
    else:
        raise ValueError(f"Unknown trace_set: {trace_set}")

    fits.PrimaryHDU(quartz.astype(np.float32), header=hdr).writeto(q_path, overwrite=True)
    fits.PrimaryHDU(illum2d.astype(np.float32), header=hdr).writeto(i_path, overwrite=True)
    fits.PrimaryHDU(pixflat.astype(np.float32), header=hdr).writeto(p_path, overwrite=True)

    logger.info("[%s] Wrote %s", trace_set, q_path)
    logger.info("[%s] Wrote %s", trace_set, i_path)
    logger.info("[%s] Wrote %s", trace_set, p_path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    if hasattr(config, "ensure_directories"):
        config.ensure_directories()

    st04 = Path(config.ST04_TRACES).expanduser()
    st05 = Path(config.ST05_PIXFLAT).expanduser()
    st05.mkdir(parents=True, exist_ok=True)

    qa = getattr(config, "QUARTZ_SLITS_OFF", None)
    if qa is None:
        raise RuntimeError("Missing config.QUARTZ_SLITS_OFF")

    qb_even = getattr(config, "QUARTZ_SLITS_ON_EVEN", None)
    qb_odd  = getattr(config, "QUARTZ_SLITS_ON_ODD", None)

    if qb_even is None:
        raise RuntimeError("Missing config.QUARTZ_SLITS_ON_EVEN")
    if qb_odd is None:
        raise RuntimeError("Missing config.QUARTZ_SLITS_ON_ODD")

    # Build EVEN pixflat
    _build_one(
        trace_set="EVEN",
        trace_base="Even_traces",
        qa_name=str(qa),
        qb_name=str(qb_even),
        st04=st04,
        st05=st05,
    )

    # Build ODD pixflat
    _build_one(
        trace_set="ODD",
        trace_base="Odd_traces",
        qa_name=str(qa),
        qb_name=str(qb_odd),
        st04=st04,
        st05=st05,
    )


if __name__ == "__main__":
    main()
