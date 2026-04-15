#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step06a — build the baseline FinalScience mosaic.

PURPOSE
-------
Combine the Step03.5 science exposures into a single 2D science mosaic and
optionally convert it to count-rate units (ADU/s).

This product defines the full-frame science image used by Step06b for pixel-flat
correction and by all later slit-based processing.

INPUT
-----
- config.ST03P5_ROWSTRIPE / *_biascorr_cr_rowcorr.fits

OUTPUT
------
Written to config.ST06_SCIENCE:

- FinalScience_<target>_ADUperS.fits   (default)
- FinalScience_<target>_ADU.fits       (if --no-normalize-to-rate)

METHOD
------
- exposure-time weighted coaddition
- optional sigma clipping
- optional normalization to ADU/s

NOTES
-----
- Inputs are already orientation-corrected, bias-corrected, CR-cleaned,
  row-stripe corrected, and quadrant-matched.
- Inputs are single-HDU 2D mosaics.
- This step does not apply pixel-flat or slit geometry information.

run:
> PYTHONPATH=. python pipeline/step06_science_rectify/step06a_make_final_science.py 

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
import config


in_dir = Path(config.ST03P5_ROWSTRIPE)
OUTDIR = Path(config.ST06_SCIENCE)
OUTDIR.mkdir(parents=True, exist_ok=True)


def _science_inputs() -> list[Path]:
    names = getattr(config, "SCIENCE_FILES", None)
    if not names:
        raise RuntimeError("Missing config.SCIENCE_FILES")

    files = [in_dir / Path(name).name for name in names]
    missing = [p for p in files if not p.exists()]
    if missing:
        msg = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Missing science files listed in config.SCIENCE_FILES:\n" + msg
        )
    return files


def _print_exptime_table(files: list[Path], exptime_key: str = "EXPTIME") -> float:
    exptimes = []
    print("\nInputs / exposure times:")
    print("------------------------------------------------------------")
    for f in files:
        with fits.open(f) as hdul:
            t = hdul[0].header.get(exptime_key, None)
        try:
            t = float(t)
        except Exception:
            t = np.nan
        exptimes.append(t)
        print(f"{f.name:40s}  {t:10.3f} s")
    print("------------------------------------------------------------")
    finite = [t for t in exptimes if np.isfinite(t) and t > 0]
    t_sum = float(np.sum(finite)) if finite else float("nan")
    print(f"Total integration time (sum {exptime_key}): {t_sum:.3f} s\n")
    return t_sum


def _sigma_clip_mask(stack: np.ndarray, sigma: float = 3.0, maxiters: int = 5) -> np.ndarray:
    """
    Return boolean good-pixel mask for a stack with shape (nimg, ny, nx).
    """
    good = np.isfinite(stack)

    for _ in range(maxiters):
        med = np.nanmedian(np.where(good, stack, np.nan), axis=0)
        std = np.nanstd(np.where(good, stack, np.nan), axis=0)
        std[~np.isfinite(std) | (std <= 0)] = np.nan

        new_good = good & (
            np.abs(stack - med[None, :, :]) <= sigma * std[None, :, :]
        )

        if np.array_equal(new_good, good):
            break
        good = new_good

    return good


def _combine_weighted(
    files: list[Path],
    exptime_key: str = "EXPTIME",
    normalize_to_rate: bool = True,
    sigma_clip: bool = True,
    sigma: float = 3.0,
    maxiters: int = 5,
) -> tuple[np.ndarray, fits.Header, float]:
    """
    Weighted combine plain 2D FITS images.
    """
    imgs = []
    weights = []
    hdr0 = None

    shape0 = None
    for p in files:
        with fits.open(p) as hdul:
            img = np.asarray(hdul[0].data, dtype=np.float32)
            hdr = hdul[0].header.copy()

        if img.ndim != 2:
            raise ValueError(f"{p.name}: expected 2D image, got shape {img.shape}")

        if shape0 is None:
            shape0 = img.shape
            hdr0 = hdr
        elif img.shape != shape0:
            raise ValueError(f"Shape mismatch: {p.name} has {img.shape}, expected {shape0}")

        t = hdr.get(exptime_key, None)
        try:
            t = float(t)
        except Exception:
            t = np.nan

        if not np.isfinite(t) or t <= 0:
            raise ValueError(f"{p.name}: invalid {exptime_key}={t}")

        if normalize_to_rate:
            img_use = img / t
            w = t
        else:
            img_use = img
            w = t

        imgs.append(img_use)
        weights.append(w)

    stack = np.stack(imgs, axis=0).astype(np.float32)
    w = np.asarray(weights, dtype=np.float32)

    if sigma_clip:
        good = _sigma_clip_mask(stack, sigma=float(sigma), maxiters=int(maxiters))
    else:
        good = np.isfinite(stack)

    w3 = w[:, None, None] * good.astype(np.float32)
    num = np.nansum(stack * w3, axis=0)
    den = np.nansum(w3, axis=0)

    out = np.full_like(num, np.nan, dtype=np.float32)
    m = den > 0
    out[m] = (num[m] / den[m]).astype(np.float32)

    hdr = hdr0.copy() if hdr0 is not None else fits.Header()
    hdr["NCOMBINE"] = (len(files), "Number of input science frames")
    hdr["WTMODE"] = ("EXPTIME", "Weighting mode")
    hdr["EXPTKEY"] = (exptime_key, "Exposure-time keyword")
    hdr["SIGCLIP"] = (bool(sigma_clip), "Sigma clipping applied")
    hdr["SIGMA"] = (float(sigma), "Sigma clipping threshold")
    hdr["MAXITER"] = (int(maxiters), "Max sigma-clip iterations")
    hdr["NORATE"] = (bool(normalize_to_rate), "Inputs normalized to ADU/s before combine")
    hdr["INSTEP"] = ("ST03.5", "Input stage")
    hdr["OUTSTEP"] = ("ST06a", "Output stage")

    for p in files[:20]:
        hdr.add_history(f"INFILE: {p.name}")
    if len(files) > 20:
        hdr.add_history(f"INFILE: ... ({len(files) - 20} more)")

    total_exptime = float(np.sum(w))
    return out, hdr, total_exptime


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FinalScience_* from Step03.5 science inputs.")
    ap.add_argument(
        "--target",
        default=getattr(config, "TARGET_FILE_STEM", "dolidze"),
        help="Target tag used in output filename: FinalScience_<target>_ADUperS.fits",
    )
    ap.add_argument(
        "--exptime-key",
        default="EXPTIME",
        help="Header keyword for exposure time (default: EXPTIME)",
    )
    ap.add_argument(
        "--normalize-to-rate",
        action="store_true",
        default=True,
        help="Normalize each input to ADU/s before combining (default: True)",
    )
    ap.add_argument(
        "--no-normalize-to-rate",
        dest="normalize_to_rate",
        action="store_false",
        help="Disable ADU/s normalization (combine in raw ADU units).",
    )
    ap.add_argument(
        "--sigma-clip",
        action="store_true",
        default=True,
        help="Apply sigma clipping during stack (default: True)",
    )
    ap.add_argument(
        "--no-sigma-clip",
        dest="sigma_clip",
        action="store_false",
        help="Disable sigma clipping.",
    )
    ap.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Sigma threshold for sigma clipping (default: 3.0)",
    )
    ap.add_argument(
        "--maxiters",
        type=int,
        default=5,
        help="Max iterations for sigma clipping (default: 5)",
    )
    args = ap.parse_args()

    if hasattr(config, "ensure_directories"):
        config.ensure_directories()

    st06 = Path(config.ST06_SCIENCE).expanduser()
    st06.mkdir(parents=True, exist_ok=True)

    files = _science_inputs()
    if not files:
        raise FileNotFoundError(f"No science target files found in {in_dir}")

    _print_exptime_table(files, exptime_key=args.exptime_key)

    data, hdr, total_exptime = _combine_weighted(
        files=files,
        exptime_key=args.exptime_key,
        normalize_to_rate=bool(args.normalize_to_rate),
        sigma_clip=bool(args.sigma_clip),
        sigma=float(args.sigma),
        maxiters=int(args.maxiters),
    )

    hdr["TOTEXP"] = (float(total_exptime), "Total integration time used")
    hdr.add_history(f"FinalScience combine from stage: {in_dir.name}")

    outname = (
        f"FinalScience_{args.target}_ADUperS.fits"
        if args.normalize_to_rate
        else f"FinalScience_{args.target}_ADU.fits"
    )
    outpath = st06 / outname

    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32), header=hdr).writeto(
        outpath, overwrite=True
    )
    print(f"Wrote: {outpath}")


if __name__ == "__main__":
    main()