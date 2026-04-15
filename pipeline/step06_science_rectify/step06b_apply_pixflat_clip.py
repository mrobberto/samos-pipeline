#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step06b — apply the Step05 pixel flat to the FinalScience mosaic.

Baseline-preserving port:
- keeps the classic Step06b correction behavior
- updates only plumbing to the new config / directory architecture

Inputs
------
science : config.ST06_SCIENCE / FinalScience_*_ADUperS.fits
flat    : config.PIXFLAT_EVEN or config.PIXFLAT_ODD
mask    : Step04 mask in config.ST04_TRACES:
          prefer *_mask_reg.fits, fallback *_mask.fits

Output
------
By default writes to config.ST06_SCIENCE using the historical naming style:
  FinalScience_<target>_ADUperS[_reg]_pixflatcorr_clipped_<TRACESET>.fits

Optional QC
-----------
If --write-masked-qc is given, also writes a masked copy with pixels outside
the active trace mask set to NaN.

Run:

PYTHONPATH=. python pipeline/step06_science_rectify/step06b_apply_pixflat_clip.py --traceset EVEN
PYTHONPATH=. python pipeline/step06_science_rectify/step06b_apply_pixflat_clip.py --traceset ODD

Spyder:

runfile("pipeline/step06_science_rectify/step06b_apply_pixflat_clip.py", args="--traceset EVEN")
runfile("pipeline/step06_science_rectify/step06b_apply_pixflat_clip.py", args="--traceset ODD")
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from astropy.io import fits

import sys

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib
import config
importlib.reload(config)

try:
    from scipy.ndimage import shift as imshift
except Exception:
    imshift = None


def _profile_x_sum(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Collapse along Y to get a robust spatial profile vs X."""
    im = np.where(mask, image, np.nan)
    lo, hi = np.nanpercentile(im, [5, 95])
    im = np.clip(im, lo, hi)
    return np.nansum(im, axis=0)


def _xshift_from_profiles(p_sci: np.ndarray, p_ref: np.ndarray) -> int:
    """Integer-pixel X shift via FFT cross-correlation (ref relative to sci)."""
    a = p_sci - np.nanmedian(p_sci)
    b = p_ref - np.nanmedian(p_ref)
    a = np.nan_to_num(a)
    b = np.nan_to_num(b)
    cc = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=a.size)
    dx = int(np.argmax(cc))
    if dx > a.size // 2:
        dx -= a.size
    return dx


def _merge_provenance(dst_hdr: fits.Header, src_hdr: fits.Header, keys) -> None:
    """Copy a whitelist of keys from src_hdr -> dst_hdr if present."""
    for k in keys:
        if k in src_hdr:
            dst_hdr[k] = src_hdr[k]


def _pick_mask(st04: Path, traceset: str) -> Path:
    if traceset == "EVEN":
        reg = st04 / "Even_traces_mask_reg.fits"
        raw = st04 / "Even_traces_mask.fits"
    else:
        reg = st04 / "Odd_traces_mask_reg.fits"
        raw = st04 / "Odd_traces_mask.fits"
    return reg if reg.exists() else raw


def _default_science(st06: Path) -> Path:
    default_science = next(st06.glob("FinalScience*_ADUperS.fits"), None)
    if default_science is None:
        raise FileNotFoundError(f"No FinalScience*_ADUperS.fits found in {st06}")
    return default_science


def _default_flat(traceset: str) -> Path:
    if traceset == "EVEN":
        p = Path(config.PIXFLAT_EVEN).expanduser()
    else:
        p = Path(config.PIXFLAT_ODD).expanduser()
    return p


def _infer_traceset(name: str) -> str | None:
    n = name.upper()
    if "EVEN" in n:
        return "EVEN"
    if "ODD" in n:
        return "ODD"
    return None


def _default_output_name(science_name: str, traceset: str, register_flat: bool) -> str:
    base = science_name[:-5] if science_name.lower().endswith(".fits") else science_name
    if register_flat:
        return f"{base}_reg_pixflatcorr_clipped_{traceset}.fits"
    return f"{base}_pixflatcorr_clipped_{traceset}.fits"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--traceset",
        type=str,
        choices=["EVEN", "ODD", "even", "odd"],
        required=True,
        help="Trace set to process (EVEN or ODD).",
    )
    ap.add_argument(
        "--science",
        type=str,
        default=None,
        help="Path to science FITS. Default: first FinalScience*_ADUperS.fits in config.ST06_SCIENCE.",
    )
    ap.add_argument(
        "--flat",
        type=str,
        default=None,
        help="Path to pixel flat FITS. Default: config.PIXFLAT_EVEN / config.PIXFLAT_ODD.",
    )
    ap.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to trace mask FITS. Default: prefer *_mask_reg.fits, fallback *_mask.fits.",
    )
    ap.add_argument("--clip-lo", type=float, default=0.70,
                    help="Lower clip bound for flat values inside mask.")
    ap.add_argument("--clip-hi", type=float, default=1.30,
                    help="Upper clip bound for flat values inside mask.")
    ap.add_argument("--register-flat", action="store_true",
                    help="Register flat+mask to science via integer X shift measured inside mask.")
    ap.add_argument("--out", type=str, default=None,
                    help="Explicit output filename. If omitted, uses historical naming in config.ST06_SCIENCE.")
    ap.add_argument("--write-masked-qc", action="store_true",
                    help="Also write a QC image with pixels outside mask set to NaN.")
    args = ap.parse_args(argv)

    traceset = str(args.traceset).upper()
    st06 = Path(config.ST06_SCIENCE).expanduser()
    st04 = Path(config.ST04_TRACES).expanduser()

    science_path = Path(args.science).expanduser() if args.science else _default_science(st06)
    if not science_path.exists():
        hits = sorted(st06.glob("FinalScience*_ADUperS*.fits"))
        if hits:
            science_path = hits[0]
        else:
            raise FileNotFoundError(f"Science file not found: {science_path} (and no FinalScience*_ADUperS*.fits in {st06})")

    flat_path = Path(args.flat).expanduser() if args.flat else _default_flat(traceset)
    if not flat_path.exists():
        raise FileNotFoundError(f"Pixel flat not found at: {flat_path}")

    mask_path = Path(args.mask).expanduser() if args.mask else _pick_mask(st04, traceset)
    if not mask_path.exists():
        raise FileNotFoundError(f"Trace mask not found at: {mask_path}")

    trace_set_guess = _infer_traceset(flat_path.name) or _infer_traceset(mask_path.name) or _infer_traceset(science_path.name)

    with fits.open(science_path) as hdul:
        sci = hdul[0].data.astype(np.float32)
        hdr = hdul[0].header.copy()

    with fits.open(flat_path) as hdul:
        flat = hdul[0].data.astype(np.float32)
        hdr_flat = hdul[0].header.copy()

    with fits.open(mask_path) as hdul:
        m = (hdul[0].data > 0)
        hdr_mask = hdul[0].header.copy()

    if sci.shape != flat.shape or sci.shape != m.shape:
        raise ValueError(f"Shape mismatch: sci={sci.shape}, flat={flat.shape}, mask={m.shape}")

    dx = 0
    flat_use = flat
    m_use = m
    if args.register_flat:
        if imshift is None:
            raise RuntimeError("scipy is required for --register-flat but is not available.")
        ps = _profile_x_sum(sci, m)
        pf = _profile_x_sum(flat, m)
        dx = _xshift_from_profiles(ps, pf)
        flat_use = imshift(flat, shift=(0, dx), order=1, mode="nearest")
        m_use = imshift(m.astype(np.float32), shift=(0, dx), order=0, mode="nearest") > 0.5
        print("dx (flat->science) =", dx)

    lo, hi = float(args.clip_lo), float(args.clip_hi)
    flat_c = flat_use.copy()
    flat_c[m_use] = np.clip(flat_c[m_use], lo, hi)

    corr = sci.copy()
    corr[m_use] = sci[m_use] / flat_c[m_use]

    corr_masked = corr.copy()
    corr_masked[~m_use] = np.nan

    st06.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out).expanduser()
    else:
        out_path = st06 / _default_output_name(science_path.name, traceset, bool(args.register_flat))

    _merge_provenance(hdr, hdr_flat, keys=("ROT180", "XFLIP", "YFLIP", "BKGID", "NSLITS", "NSLDET", "NSLTAB", "SLTMISM", "TRACESET", "TRCSET"))
    _merge_provenance(hdr, hdr_mask, keys=("ROT180", "XFLIP", "YFLIP", "BKGID", "NSLITS", "NSLDET", "NSLTAB", "SLTMISM", "TRACESET", "TRCSET"))

    hdr.add_history("Pixel-flat correction with clipping inside trace mask (Step06b)")
    hdr["SCIENCE"] = (science_path.name, "Input science file (Step06b)")
    hdr["PIXFLAT"] = (flat_path.name, "Pixel-flat file applied (Step05)")
    hdr["MASKFILE"] = (mask_path.name, "Trace mask file used (Step04)")
    hdr["TRACESET"] = (traceset, "Trace set used (EVEN/ODD)")
    if trace_set_guess is not None and trace_set_guess != traceset:
        hdr["TRCSET"] = (trace_set_guess, "Trace set inferred from filenames (for debugging)")

    hdr["PFLAT"] = (flat_path.name, "Pixel flat used")
    hdr["TMASK"] = (mask_path.name, "Trace mask used")

    hdr["FLCLIPLO"] = (lo, "Flat clip lower bound (inside mask)")
    hdr["FLCLIPHI"] = (hi, "Flat clip upper bound (inside mask)")
    hdr["CLIPLO"] = (lo, "Alias of FLCLIPLO")
    hdr["CLIPHI"] = (hi, "Alias of FLCLIPHI")

    hdr["REGFLAT"] = (bool(args.register_flat), "Flat registered to science via integer X shift")
    hdr["FLATDX"] = (int(dx), "X shift applied to flat/mask to align to science (pix)")

    fits.PrimaryHDU(corr.astype(np.float32), header=hdr).writeto(out_path, overwrite=True)
    print("Wrote:", out_path)

    if args.write_masked_qc:
        out_qc = out_path.with_name(out_path.stem + "_masked.fits")
        hq = hdr.copy()
        hq["QC_MASK"] = (True, "Pixels outside trace mask set to NaN")
        fits.PrimaryHDU(corr_masked.astype(np.float32), header=hq).writeto(out_qc, overwrite=True)
        print("Wrote:", out_qc)


if __name__ == "__main__":
    main()
