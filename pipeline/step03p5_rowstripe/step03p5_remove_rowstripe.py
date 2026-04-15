#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step03.5 — Row-wise Stripe Removal and Quadrant Pedestal Matching
=================================================================

Purpose
-------
Remove low-level additive background structures present after CR-cleaning,
specifically:

1) Horizontal striping (1/f-like noise)
   - Appears as row-wise offsets across the detector
   - Modulates spectra along the dispersion direction

2) Residual quadrant pedestal mismatches
   - Small constant offsets between detector quadrants
   - Visible after row-wise correction as discontinuities between halves

Method
------
The correction is applied in two stages, using only signal-free regions:

A) Row-wise background subtraction
   - For each detector row (Y), estimate a scalar background level using
     off-spectrum sidebands in X:
         X ∈ [0 : X_LEFT_MAX]  ∪  [X_RIGHT_MIN : NX]
   - Use a robust estimator (median by default)
   - Subtract this value from the full row

B) Quadrant pedestal matching
   - Divide detector into four quadrants:
         Y ∈ [0 : Y_SPLIT] and [Y_SPLIT : NY]
         X ∈ [0 : XMID] and [XMID : NX]
   - Measure background medians in sidebands for each quadrant
   - Compute a global reference (median of quadrant medians)
   - Subtract constant offsets to match all quadrants to the reference

Key Assumptions
---------------
- Spectral signal is confined to the central X region
- Sidebands are representative of background (no strong signal contamination)
- Residual quadrant offsets are approximately constant (not gradients)

Inputs
------
- Step03 CR-cleaned frames:
    *_biascorr_cr.fits

Outputs
-------
- Row- and quadrant-corrected frames:
    *_biascorr_cr_rowcorr.fits

- Additional extensions:
    ROWOFFSET  : per-row background offsets
    QUADSTATS  : quadrant medians and applied corrections

- Header keywords document:
    ROWCORR, ROWXL, ROWXR, ROWEST
    QPCORR, QREF, DQ00–DQ11, QXMID, ROWSPLIT

Pipeline Position
-----------------
Step03.5 is inserted between:
    Step03 (CR cleaning)
    Step04 (trace detection)

Step04 and all downstream steps must consume the Step03.5 products.

Notes
-----
- This step is intentionally conservative and additive-only:
    no rescaling, no filtering of spectral signal
- Designed to be fast and re-runnable without repeating Step03
- Parameters (sidebands, Y split) can be tuned per dataset if needed
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]   # .../samos-pipeline
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib
import config
importlib.reload(config)

import numpy as np
from astropy.io import fits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(Path(__file__).stem)


# ---------------------------------------------------------------------
# User-tunable parameters
# ---------------------------------------------------------------------
# signal-free sidebands used to measure background / striping
X_LEFT_MAX = 1300
X_RIGHT_MIN = 2800

# Y split between upper/lower detector halves
Y_SPLIT = 2056

# robust estimator per row:
# "median" is safest; "mean" is slightly more aggressive
ROW_ESTIMATOR = "median"

# optional smoothing of the row-offset vector
SMOOTH_OFFSETS = False
SMOOTH_WIN = 21   # odd integer if smoothing enabled


def smooth_1d(x: np.ndarray, win: int) -> np.ndarray:
    win = max(3, int(win) | 1)
    k = np.ones(win, dtype=float) / win
    return np.convolve(x, k, mode="same")


def robust_region_median(arr: np.ndarray) -> float:
    vals = np.asarray(arr, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.median(vals))


def robust_row_offset(img: np.ndarray, x_left_max: int, x_right_min: int) -> np.ndarray:
    """
    Estimate one scalar offset per row from signal-free side regions.
    """
    ny, nx = img.shape

    xl = max(0, min(nx, int(x_left_max)))
    xr = max(0, min(nx, int(x_right_min)))

    if xl <= 0 and xr >= nx:
        raise ValueError("No valid background sidebands defined.")

    offsets = np.full(ny, np.nan, dtype=float)

    for y in range(ny):
        pieces = []
        if xl > 0:
            pieces.append(img[y, :xl])
        if xr < nx:
            pieces.append(img[y, xr:])

        if not pieces:
            continue

        vals = np.concatenate(pieces).astype(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        if ROW_ESTIMATOR.lower() == "mean":
            offsets[y] = np.mean(vals)
        else:
            offsets[y] = np.median(vals)

    good = np.isfinite(offsets)
    if good.sum() == 0:
        raise RuntimeError("Could not estimate any row offsets.")
    if good.sum() < ny:
        yy = np.arange(ny, dtype=float)
        offsets[~good] = np.interp(yy[~good], yy[good], offsets[good])

    if SMOOTH_OFFSETS:
        offsets = smooth_1d(offsets, SMOOTH_WIN)

    return offsets


def apply_quadrant_pedestal_match(
    img: np.ndarray,
    x_left_max: int,
    x_right_min: int,
    y_split: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Remove residual constant pedestal offsets between the four detector quadrants
    using only the off-spectrum sidebands.

    Measurement regions:
      Y: [0:y_split) and [y_split:ny)
      X sidebands: [0:x_left_max) and [x_right_min:nx)

    Application regions:
      full quadrants split at X midpoint and Y split.

    Returns
    -------
    img_corr : corrected image
    stats    : measured quadrant medians and applied deltas
    """
    ny, nx = img.shape

    xl = max(0, min(nx, int(x_left_max)))
    xr = max(0, min(nx, int(x_right_min)))
    ys = max(1, min(ny - 1, int(y_split)))

    if xl <= 0 or xr >= nx or xl >= xr:
        raise ValueError("Invalid X sideband configuration for quadrant matching.")

    # Measure background in sideband-only regions
    q00_bg = img[0:ys, 0:xl]
    q01_bg = img[0:ys, xr:nx]
    q10_bg = img[ys:ny, 0:xl]
    q11_bg = img[ys:ny, xr:nx]

    m00 = robust_region_median(q00_bg)
    m01 = robust_region_median(q01_bg)
    m10 = robust_region_median(q10_bg)
    m11 = robust_region_median(q11_bg)

    meds = np.array([m00, m01, m10, m11], dtype=float)
    good = np.isfinite(meds)
    if good.sum() == 0:
        raise RuntimeError("Could not measure any quadrant background medians.")

    ref = float(np.median(meds[good]))

    d00 = 0.0 if not np.isfinite(m00) else (m00 - ref)
    d01 = 0.0 if not np.isfinite(m01) else (m01 - ref)
    d10 = 0.0 if not np.isfinite(m10) else (m10 - ref)
    d11 = 0.0 if not np.isfinite(m11) else (m11 - ref)

    # Apply to full quadrants
    xmid = nx // 2

    out = img.copy()
    out[0:ys, 0:xmid] -= d00
    out[0:ys, xmid:nx] -= d01
    out[ys:ny, 0:xmid] -= d10
    out[ys:ny, xmid:nx] -= d11

    stats = {
        "Q00_MED": float(m00) if np.isfinite(m00) else np.nan,
        "Q01_MED": float(m01) if np.isfinite(m01) else np.nan,
        "Q10_MED": float(m10) if np.isfinite(m10) else np.nan,
        "Q11_MED": float(m11) if np.isfinite(m11) else np.nan,
        "QREF": float(ref),
        "DQ00": float(d00),
        "DQ01": float(d01),
        "DQ10": float(d10),
        "DQ11": float(d11),
        "XMID": int(xmid),
        "YSPLIT": int(ys),
    }
    return out, stats


def main():
    if hasattr(config, "ensure_directories"):
        config.ensure_directories()

    in_dir = Path(config.ST03_CRCLEAN)
    out_dir = Path(config.ST03P5_ROWSTRIPE)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*_biascorr_cr.fits"))
    if not files:
        raise FileNotFoundError(f"No *_biascorr_cr.fits found in {in_dir}. Run Step03 first.")

    logger.info("Input dir   = %s", in_dir)
    logger.info("Output dir  = %s", out_dir)
    logger.info("Found %d files", len(files))
    logger.info("Row background windows: [0:%d] and [%d:nx]", X_LEFT_MAX, X_RIGHT_MIN)
    logger.info("Quadrant Y split: %d", Y_SPLIT)

    wrote = 0

    for p in files:
        root = p.stem.replace("_biascorr_cr", "")
        out = out_dir / f"{root}_biascorr_cr_rowcorr.fits"

        with fits.open(p) as hdul:
            img = np.asarray(hdul[0].data, dtype=np.float32)
            hdr = hdul[0].header.copy()
            extra_hdus = [h.copy() for h in hdul[1:]]

        if img.ndim != 2:
            raise ValueError(f"{p.name}: expected 2D image, got shape {img.shape}")

        # -------------------------------------------------------------
        # 1) Remove row-wise striping
        # -------------------------------------------------------------
        offsets = robust_row_offset(img, X_LEFT_MAX, X_RIGHT_MIN)
        img_rowcorr = img - offsets[:, None]

        # -------------------------------------------------------------
        # 2) Match residual quadrant pedestal offsets
        # -------------------------------------------------------------
        img_corr, qstats = apply_quadrant_pedestal_match(
            img_rowcorr,
            x_left_max=X_LEFT_MAX,
            x_right_min=X_RIGHT_MIN,
            y_split=Y_SPLIT,
        )

        hdr["ROWCORR"] = (True, "Row-wise stripe correction applied (Step03.5)")
        hdr["ROWXL"] = (int(X_LEFT_MAX), "Left background region max X")
        hdr["ROWXR"] = (int(X_RIGHT_MIN), "Right background region min X")
        hdr["ROWEST"] = (ROW_ESTIMATOR.upper(), "Row background estimator")
        hdr["ROWSMTH"] = (bool(SMOOTH_OFFSETS), "Row offset smoothing applied")
        hdr["ROWSPLIT"] = (int(Y_SPLIT), "Y split for quadrant pedestal matching")
        hdr["QPCORR"] = (True, "Residual quadrant pedestal matching applied")
        hdr["QREF"] = (float(qstats["QREF"]), "Reference quadrant background level")
        hdr["DQ00"] = (float(qstats["DQ00"]), "Applied pedestal correction: upper-left")
        hdr["DQ01"] = (float(qstats["DQ01"]), "Applied pedestal correction: upper-right")
        hdr["DQ10"] = (float(qstats["DQ10"]), "Applied pedestal correction: lower-left")
        hdr["DQ11"] = (float(qstats["DQ11"]), "Applied pedestal correction: lower-right")
        hdr["QXMID"] = (int(qstats["XMID"]), "X split used for quadrant application")
        hdr["INSTEP"] = ("ST03", "Input stage")
        hdr["OUTSTEP"] = ("ST03.5", "Output stage")
        hdr.add_history("Step03.5: removed horizontal striping via row-wise background subtraction.")
        hdr.add_history("Step03.5: matched residual quadrant pedestal offsets using off-spectrum sidebands.")

        hdul_out = fits.HDUList([fits.PrimaryHDU(data=img_corr.astype(np.float32), header=hdr)])

        # preserve CR mask or any other extensions if present
        for h in extra_hdus:
            hdul_out.append(h)

        # Save the fitted row offsets for QC
        row_hdu = fits.ImageHDU(data=offsets.astype(np.float32), name="ROWOFFSET")
        row_hdu.header["COMMENT"] = "Per-row background offset subtracted in Step03.5"
        hdul_out.append(row_hdu)

        # Save quadrant stats for QC / provenance
        q_arr = np.array(
            [[qstats["Q00_MED"], qstats["Q01_MED"]],
             [qstats["Q10_MED"], qstats["Q11_MED"]]],
            dtype=np.float32,
        )
        q_hdu = fits.ImageHDU(data=q_arr, name="QUADSTATS")
        q_hdu.header["QREF"] = float(qstats["QREF"])
        q_hdu.header["DQ00"] = float(qstats["DQ00"])
        q_hdu.header["DQ01"] = float(qstats["DQ01"])
        q_hdu.header["DQ10"] = float(qstats["DQ10"])
        q_hdu.header["DQ11"] = float(qstats["DQ11"])
        q_hdu.header["XMID"] = int(qstats["XMID"])
        q_hdu.header["YSPLIT"] = int(qstats["YSPLIT"])
        q_hdu.header["COMMENT"] = "Quadrant background medians and applied corrections"
        hdul_out.append(q_hdu)

        hdul_out.writeto(out, overwrite=True)
        wrote += 1
        logger.info("Wrote: %s", out.name)

    logger.info("Done. Wrote %d row-corrected files -> %s", wrote, out_dir)


if __name__ == "__main__":
    main()