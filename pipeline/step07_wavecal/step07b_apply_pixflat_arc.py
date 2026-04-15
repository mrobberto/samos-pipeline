#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step07b — apply pixel-flat correction to the canonical ArcDiff image.

PURPOSE
-------
Create a single processed full-frame arc-difference image by applying the
Step05 pixel flats with clipping inside the illuminated trace masks only:

  - apply EVEN flat inside EVEN mask
  - apply ODD  flat inside ODD  mask

Outside the masks, the arc-difference image is unchanged.

This preserves the classic Step07.0b behavior and improves uniformity of arc
line brightness across slitlets, stabilizing cross-correlation and wavelength
solution propagation in later Step07 stages.

INPUTS
------
- config.MASTER_ARC_DIFF                       (Step07a canonical arc difference)
- config.ST04_TRACES / Even_traces_mask*.fits (Step04)
- config.ST04_TRACES / Odd_traces_mask*.fits  (Step04)
- config.PIXFLAT_EVEN                         (Step05)
- config.PIXFLAT_ODD                          (Step05)

OUTPUT
------
By default writes:

  <MASTER_ARC_DIFF stem>_pixflatcorr_clipped.fits

in config.ST07_WAVECAL.

NOTES
-----
- Flat values are clipped to [clip_lo, clip_hi] inside the masks only.
- No geometric transformation is performed here.
- This step is the arc analogue of Step06b.

RUN
---
    PYTHONPATH=. python pipeline/step07_wavecal/step07b_apply_pixflat_arc.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
import config


def _find_first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_image(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path) as h:
        return h[0].data.astype(np.float32), h[0].header.copy()


def _load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with fits.open(path) as h:
        m = (h[0].data > 0)
    if m.shape != shape:
        raise ValueError(f"Mask shape mismatch: {path} has {m.shape}, expected {shape}")
    return m


def _load_flat(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with fits.open(path) as h:
        f = h[0].data.astype(np.float32)
    if f.shape != shape:
        raise ValueError(f"Flat shape mismatch: {path} has {f.shape}, expected {shape}")
    return f


def _apply_pixflat_inside_mask(
    image: np.ndarray,
    flat: np.ndarray,
    mask: np.ndarray,
    clip_lo: float,
    clip_hi: float,
) -> np.ndarray:
    """Return a corrected copy: image/flat inside mask, unchanged outside."""
    out = image.copy()
    flat_c = flat.copy()
    flat_c[mask] = np.clip(flat_c[mask], clip_lo, clip_hi)
    out[mask] = out[mask] / flat_c[mask]
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arcdiff",
        type=str,
        default=None,
        help="Override input ArcDiff FITS. Default: config.MASTER_ARC_DIFF",
    )
    ap.add_argument("--clip-lo", type=float, default=0.70,
                    help="Lower clip bound for flat values inside masks.")
    ap.add_argument("--clip-hi", type=float, default=1.30,
                    help="Upper clip bound for flat values inside masks.")
    ap.add_argument("--mask-even", type=str, default=None,
                    help="Override EVEN trace mask path.")
    ap.add_argument("--mask-odd", type=str, default=None,
                    help="Override ODD trace mask path.")
    ap.add_argument("--flat-even", type=str, default=None,
                    help="Override EVEN pixel flat path.")
    ap.add_argument("--flat-odd", type=str, default=None,
                    help="Override ODD pixel flat path.")
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Explicit output FITS filename. Default: <MASTER_ARC_DIFF stem>_pixflatcorr_clipped.fits",
    )
    args = ap.parse_args(argv)

    if hasattr(config, "ensure_directories"):
        config.ensure_directories()

    st07 = Path(config.ST07_WAVECAL).expanduser()
    st04 = Path(config.ST04_TRACES).expanduser()

    arcdiff_path = Path(args.arcdiff).expanduser() if args.arcdiff else Path(config.MASTER_ARC_DIFF).expanduser()
    if not arcdiff_path.exists():
        raise FileNotFoundError(f"ArcDiff not found: {arcdiff_path}")

    img, hdr = _load_image(arcdiff_path)
    shape = img.shape

    mask_even = Path(args.mask_even).expanduser() if args.mask_even else _find_first_existing([
        st04 / "Even_traces_mask_reg.fits",
        st04 / "Even_traces_mask.fits",
    ])
    mask_odd = Path(args.mask_odd).expanduser() if args.mask_odd else _find_first_existing([
        st04 / "Odd_traces_mask_reg.fits",
        st04 / "Odd_traces_mask.fits",
    ])

    if mask_even is None or not mask_even.exists():
        raise FileNotFoundError(f"EVEN mask not found in {st04} (tried *_mask_reg.fits, *_mask.fits)")
    if mask_odd is None or not mask_odd.exists():
        raise FileNotFoundError(f"ODD mask not found in {st04} (tried *_mask_reg.fits, *_mask.fits)")

    flat_even = Path(args.flat_even).expanduser() if args.flat_even else Path(config.PIXFLAT_EVEN).expanduser()
    flat_odd = Path(args.flat_odd).expanduser() if args.flat_odd else Path(config.PIXFLAT_ODD).expanduser()

    if not flat_even.exists():
        raise FileNotFoundError(f"EVEN pixel flat not found: {flat_even}")
    if not flat_odd.exists():
        raise FileNotFoundError(f"ODD pixel flat not found: {flat_odd}")

    print("ST07_WAVECAL =", st07)
    print("Input ArcDiff =", arcdiff_path)
    print("EVEN mask =", mask_even)
    print("ODD  mask =", mask_odd)
    print("EVEN flat =", flat_even)
    print("ODD  flat =", flat_odd)

    m_even = _load_mask(mask_even, shape)
    m_odd = _load_mask(mask_odd, shape)
    f_even = _load_flat(flat_even, shape)
    f_odd = _load_flat(flat_odd, shape)

    lo, hi = float(args.clip_lo), float(args.clip_hi)

    out = _apply_pixflat_inside_mask(img, f_even, m_even, lo, hi)
    out = _apply_pixflat_inside_mask(out, f_odd, m_odd, lo, hi)

    st07.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out).expanduser()
    else:
        base = arcdiff_path.stem
        if "pixflatcorr" in base:
            out_name = base + "_clipped.fits"
        else:
            out_name = base + "_pixflatcorr_clipped.fits"
        out_path = st07 / out_name

    hdr.add_history("Step07b: pixel-flat correction with clipping inside EVEN+ODD trace masks")
    hdr["PFLTEVN"] = (flat_even.name, "EVEN pixel flat used")
    hdr["PFLTODD"] = (flat_odd.name, "ODD pixel flat used")
    hdr["TMSKEVN"] = (mask_even.name, "EVEN trace mask used")
    hdr["TMSKODD"] = (mask_odd.name, "ODD trace mask used")
    hdr["FLCLIPLO"] = (lo, "Flat clip lower bound (inside masks)")
    hdr["FLCLIPHI"] = (hi, "Flat clip upper bound (inside masks)")

    fits.PrimaryHDU(out.astype(np.float32), header=hdr).writeto(out_path, overwrite=True)
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
