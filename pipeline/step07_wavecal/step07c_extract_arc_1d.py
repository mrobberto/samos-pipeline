#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step07c — extract 1D arc spectra using the Step04 slitid map (global slit IDs)

PURPOSE
-------
Extract one 1D arc spectrum per slit directly from the full-frame pixflat-corrected
arc-difference image, using the Step04 slitid map as the authoritative slit
footprint definition.

This preserves the global slit numbering convention established in Step04.

INPUTS
------
- Step07b full-frame arc image:
    config.ST07_WAVECAL / ArcDiff_*_pixflatcorr_clipped.fits

- Step04 slitid map:
    EVEN: Even_traces_slitid.fits (fallback Even_traces_slitid_reg.fits)
    ODD : Odd_traces_slitid.fits  (fallback Odd_traces_slitid_reg.fits)

- Step04 geometry file (metadata only):
    Even_traces_geometry.fits / Odd_traces_geometry.fits

OUTPUTS
-------
- ArcDiff_*_pixflatcorr_clipped_1D_slitid_<EVEN|ODD>.fits   (MEF)

Each extension is named SLIT### and contains:
  data[0] = extracted arc flux as a function of detector Y
  data[1] = number of contributing pixels per row

METHOD
------
For each global slit ID present in the Step04 slitid map:
- select all pixels belonging to that slit
- collapse along X at each detector row Y
- store the 1D arc spectrum and the number of contributing pixels per row

NOTES
-----
- Extraction is driven by the Step04 slitid map, not by connected components
  or re-numbering.
- Step04 geometry is used only to propagate slit metadata (RA, DEC, INDEX, etc).
- This step preserves the detector-Y sampling for downstream shift measurement
  and wavelength solution propagation.

RUN
---
    PYTHONPATH=. python pipeline/step07_wavecal/step07c_extract_arc_1d.py --traceset EVEN
    PYTHONPATH=. python pipeline/step07_wavecal/step07c_extract_arc_1d.py --traceset ODD
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

import config


def extract_1d_spectrum(arc2d: np.ndarray, slit_mask: np.ndarray, collapse: str = "sum"):
    ny, nx = arc2d.shape
    f = np.full(ny, np.nan, dtype=np.float32)
    npix_per_row = np.zeros(ny, dtype=np.int32)

    for y in range(ny):
        row_mask = slit_mask[y, :]
        if not np.any(row_mask):
            continue
        vals = arc2d[y, row_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        npix_per_row[y] = int(vals.size)
        if collapse == "median":
            f[y] = np.float32(np.sum(vals))
        else:
            f[y] = np.float32(np.median(vals))

    out = np.vstack([f, npix_per_row.astype(np.float32)])
    return out


def _pick_first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--traceset", dest="trace_set", choices=["EVEN", "ODD"], default="EVEN",
                    help="Select which Step04 slitid map to use.")
    ap.add_argument("--arc", type=str, default=None,
                    help="Input pixflat-corrected ArcDiff FITS (Step07b). Default: inferred from config.MASTER_ARC_DIFF.")
    ap.add_argument("--slitid", type=str, default=None,
                    help="Explicit Step04 slitid map FITS (overrides default for EVEN/ODD).")
    ap.add_argument("--collapse", choices=["sum", "median"], default="sum")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    trace_set = args.trace_set.upper()

    st07 = Path(config.ST07_WAVECAL).expanduser()
    st04 = Path(config.ST04_TRACES).expanduser()

    # Canonical Step07b input
    if hasattr(config, "MASTER_ARC_DIFF"):
        base_arc = Path(config.MASTER_ARC_DIFF).expanduser()
        stem = base_arc.stem
        default_arc = st07 / f"{stem}_pixflatcorr_clipped.fits"
    else:
        default_arc = _pick_first_existing(sorted(st07.glob("ArcDiff*_pixflatcorr_clipped.fits")))
        if default_arc is None:
            default_arc = st07 / "ArcDiff_pixflatcorr_clipped.fits"

    if trace_set == "EVEN":
        default_slitid = _pick_first_existing([
            st04 / "Even_traces_slitid.fits",
            st04 / "Even_traces_slitid_reg.fits",
        ])
        geom_path = st04 / "Even_traces_geometry.fits"
    else:
        default_slitid = _pick_first_existing([
            st04 / "Odd_traces_slitid.fits",
            st04 / "Odd_traces_slitid_reg.fits",
        ])
        geom_path = st04 / "Odd_traces_geometry.fits"

    arc_path = Path(args.arc).expanduser() if args.arc else default_arc
    slitid_path = Path(args.slitid).expanduser() if args.slitid else default_slitid

    if arc_path is None or not arc_path.exists():
        raise FileNotFoundError(f"ArcDiff pixflat-corrected file not found: {arc_path}")
    if slitid_path is None or not slitid_path.exists():
        raise FileNotFoundError(f"slitid map not found: {slitid_path}")
    if not geom_path.exists():
        raise FileNotFoundError(f"geometry file not found: {geom_path}")

    # Build lookup: sid -> Step04 geometry header
    geom_meta = {}
    with fits.open(geom_path) as hg:
        for ext in hg[1:]:
            name = (ext.name or "").strip().upper()
            if not name.startswith("SLIT"):
                continue
            sid0 = ext.header.get("SLITID", None)
            if sid0 is None:
                continue
            geom_meta[int(sid0)] = ext.header.copy()

    arc = fits.getdata(arc_path).astype(np.float32)
    slitid = fits.getdata(slitid_path)

    if arc.shape != slitid.shape:
        raise RuntimeError(f"Shape mismatch: arc{arc.shape} vs slitid{slitid.shape}")

    # Global IDs present in Step04 map
    vals = np.unique(slitid).astype(int)
    bkgid = -1 if (-1 in vals) else int(max(vals, key=lambda v: np.sum(slitid == v)))
    ids = vals[vals != bkgid].astype(int)
    ids = ids[np.argsort(ids)]

    # Parity guard
    if trace_set == "EVEN":
        ids = ids[ids % 2 == 0]
    else:
        ids = ids[ids % 2 == 1]

    if ids.size == 0:
        raise RuntimeError("No slit IDs found after parity filtering. Check slitid map.")

    # Output name
    if args.out:
        out_path = Path(args.out).expanduser()
    else:
        stem = arc_path.with_suffix("").name
        out_path = st07 / f"{stem}_1D_slitid_{trace_set}.fits"

    hdus = [fits.PrimaryHDU()]
    hdus[0].header["STAGE"] = ("07c", "Pipeline stage")
    hdus[0].header["TRACESET"] = (trace_set, "EVEN/ODD")
    hdus[0].header["SRCARC"] = (arc_path.name, "Source ArcDiff file")
    hdus[0].header["SRCSLIT"] = (slitid_path.name, "Step04 slitid map used")
    hdus[0].header["SRCGEOM"] = (geom_path.name, "Step04 geometry file used")
    hdus[0].header["COLLAPSE"] = (args.collapse, "Collapse along X per Y")
    hdus[0].header["NSLITS"] = (int(ids.size), "Number of slits extracted (from slitid map)")

    for sid in ids:
        m = (slitid == sid)
        out = extract_1d_spectrum(arc, m, collapse=args.collapse)

        h = fits.Header()
        h["SLITID"] = (int(sid), "Global slit ID from Step04 map")
        h["NPX_TOT"] = (int(np.sum(m)), "Total pixels in this slit mask")

        gh = geom_meta.get(int(sid), None)
        if gh is not None:
            for k in ("INDEX", "RA", "DEC", "TRACESET", "OLDSID", "XREF", "YMIN", "YMAX"):
                if k in gh:
                    h[k] = gh[k]

        extname = f"SLIT{sid:03d}"
        hdus.append(fits.ImageHDU(data=out, header=h, name=extname))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(out_path, overwrite=True)
    print("Wrote:", out_path)
    print(f"Extracted {ids.size} slits:", f"{ids.min()}..{ids.max()}")


if __name__ == "__main__":
    main()
