#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step06c — rectify science slitlets into TRACECOORDS.

Purpose
-------
Generate a multi-extension TRACECOORDS representation from the canonical
Step06b EVEN or ODD pixel-flat-corrected science image.

The transformation uses the slit geometry derived in Step04. Each slitlet is
mapped from the curved detector geometry into a rectangular coordinate system
with dispersion along Y and cross-dispersion along X.

Processing
----------
For each slit in the selected trace set, the script:

1. reads the canonical Step06b science image;
2. reads the Step04 slit-ID map, geometry file, and slit-width table;
3. evaluates the center polynomial and, when available, the left and right
   slit-edge polynomials;
4. applies an edge-based, width-preserving coordinate transformation;
5. applies the constant-width slit mask in rectified space;
6. rejects rows with insufficient valid slit pixels;
7. writes the rectified slitlet as one extension in a multi-extension FITS file.

Configuration
-------------
The production defaults are defined by the active reduction profile:

- ``STEP06C_PADX``
- ``STEP06C_INTERP_ORDER``
- ``STEP06C_WIDTH_KEY``
- ``STEP06C_PAD_PIX``
- ``STEP06C_MIN_MASK_PIX_PER_ROW``

Inputs
------
Canonical Step06b products:

    config.SCI_EVEN_PIXFLATCORR
    config.SCI_ODD_PIXFLATCORR

Step04 geometry products from ``config.ST04_TRACES``:

    Even_traces_slitid.fits
    Odd_traces_slitid.fits

    Even_traces_geometry.fits
    Odd_traces_geometry.fits

    Even_traces_slit_table.csv
    Odd_traces_slit_table.csv

Canonical outputs
-----------------
    config.SCI_EVEN_TRACECOORDS
    config.SCI_ODD_TRACECOORDS

corresponding to:

    FinalScience_<target>_ADUperS_pixflatcorr_EVEN_tracecoords.fits
    FinalScience_<target>_ADUperS_pixflatcorr_ODD_tracecoords.fits

Each output is a multi-extension FITS file with one extension per surviving
slitlet, named ``SLIT###``.

Coordinate definition
---------------------
- Y: dispersion direction
- X: cross-dispersion direction within the slit
- pixels outside the rectified slit mask: NaN

Notes
-----
- Rectification is performed only after detector-level corrections and
  pixel-flat correction.
- No spectral centroid alignment or wavelength calibration is performed here.
- The Step04 slit geometry and edge models define the transformation.
- The production workflow uses canonical configured inputs and outputs.
- An explicit ``--science`` override remains available for testing.

Run
---
  PYTHONPATH=. python \
      pipeline/step06_science_rectify/step06c_TRACECOORDS_representation.py \
      --traceset EVEN

  PYTHONPATH=. python \
      pipeline/step06_science_rectify/step06c_TRACECOORDS_representation.py \
      --traceset ODD

Normally both trace sets are run through:

  PYTHONPATH=. python drivers/run_pipeline.py \
      --from-step 06c --to-step 06c
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("step06c_rectify_edgeswarp_clean")


def _read_halfwidths_from_slit_table(csv_path: Path, width_key: str = "width_med", pad_pix: float = 0.5):
    halfwidth_by_id = {}
    widths = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if width_key not in reader.fieldnames:
            raise KeyError(f"{csv_path} missing column '{width_key}'. Has: {reader.fieldnames}")
        for row in reader:
            sid = int(row["slit_id"])
            w = float(row[width_key])
            if np.isfinite(w) and w > 0:
                widths.append(w)
                hw = int(np.ceil((w + float(pad_pix)) / 2.0))
                hw = max(1, min(hw, 200))
                halfwidth_by_id[sid] = hw
    if not halfwidth_by_id:
        raise RuntimeError(f"No valid widths read from {csv_path} using column {width_key}")
    default_hw = int(np.ceil((np.nanmedian(widths) + float(pad_pix)) / 2.0))
    default_hw = max(1, min(default_hw, 200))
    return halfwidth_by_id, default_hw


def _poly_eval_powerbasis(coeff: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=np.float32)
    yy = y.astype(np.float32)
    for i, c in enumerate(coeff):
        out += np.float32(c) * (yy ** np.float32(i))
    return out


def _safe_key(s: str) -> str:
    s = str(s)
    return s[:68] if len(s) > 68 else s


def _default_step06b_input(trace_set: str) -> Path:
    """Return the canonical Step06b product for the selected trace set."""
    trace_set = trace_set.upper()

    if trace_set == "EVEN":
        path = Path(config.SCI_EVEN_PIXFLATCORR)
    elif trace_set == "ODD":
        path = Path(config.SCI_ODD_PIXFLATCORR)
    else:
        raise ValueError(f"Unknown trace set: {trace_set}")

    if not path.exists():
        raise FileNotFoundError(
            f"Canonical Step06b product not found: {path}. "
            f"Run Step06b for {trace_set} first."
        )

    return path


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="Baseline Step06c TRACECOORDS rectification")
    ap.add_argument("--traceset", required=True, choices=["EVEN", "ODD", "even", "odd"])
    ap.add_argument("--science", type=str, default=None, help="Override Step06b science input FITS")
    ap.add_argument("--padx", type=int, default=int(config.STEP06C_PADX))
    ap.add_argument("--interp-order", type=int, default=int(config.STEP06C_INTERP_ORDER))
    ap.add_argument("--width-key", type=str, default="width_med")
    ap.add_argument("--pad-pix", type=float, default=float(config.STEP06C_PAD_PIX))
    ap.add_argument("--min-mask-pix-per-row", type=int, default=int(config.STEP06C_MIN_MASK_PIX_PER_ROW))
    ap.add_argument("--want-regflat", choices=["true", "false"], default=None,
                    help="Require Step06b input with REGFLAT=true/false; default prefers non-registered")
    args = ap.parse_args(argv)

    if hasattr(config, "ensure_directories"):
        config.ensure_directories()

    trace_set = args.traceset.upper()
    st04 = Path(config.ST04_TRACES)
    st06 = Path(config.ST06_SCIENCE)

    if args.science:
        sci = Path(args.science).expanduser()
    else:
        if args.want_regflat is not None:
            raise ValueError(
                "--want-regflat is only supported with an explicit --science override. "
                "The production pipeline uses the canonical non-registered Step06b product."
            )
        sci = _default_step06b_input(trace_set)


    base = "Even_traces" if trace_set == "EVEN" else "Odd_traces"
    slitid_path = st04 / f"{base}_slitid.fits"
    if not slitid_path.exists():
        slitid_path = st04 / f"{base}_slitid_reg.fits"
    geom_path = st04 / f"{base}_geometry.fits"
    slit_table_csv = st04 / f"{base}_slit_table.csv"

    if not sci.exists():
        raise FileNotFoundError(sci)
    if not slitid_path.exists():
        raise FileNotFoundError(slitid_path)
    if not geom_path.exists():
        raise FileNotFoundError(geom_path)
    if not slit_table_csv.exists():
        raise FileNotFoundError(slit_table_csv)

    if args.science:
        stem = sci.stem
        if f"_{trace_set}" not in stem:
            stem = f"{stem}_{trace_set}"
        out_mef_tracecoords = sci.parent / f"{stem}_tracecoords.fits"
    else:
        out_mef_tracecoords = Path(
            config.SCI_EVEN_TRACECOORDS
            if trace_set == "EVEN"
            else config.SCI_ODD_TRACECOORDS
        )

    log.info("Input   : %s", sci)
    log.info("SlitID  : %s", slitid_path)
    log.info("Geometry: %s", geom_path)
    log.info("Widths  : %s", slit_table_csv)
    log.info("Output  : %s", out_mef_tracecoords)

    halfwidth_by_id, default_hw = _read_halfwidths_from_slit_table(
        slit_table_csv, width_key=args.width_key, pad_pix=args.pad_pix
    )
    log.info(
        "Per-slit halfwidths from %s (%s + %.1f px pad). Default hw=%d.",
        slit_table_csv.name, args.width_key, args.pad_pix, default_hw
    )

    with fits.open(sci) as hdul:
        if hdul[0].data is not None:
            img = hdul[0].data.astype(np.float32, copy=False)
            hdr0 = hdul[0].header.copy()
        else:
            idx = next((i for i, h in enumerate(hdul) if h.data is not None), None)
            if idx is None:
                raise RuntimeError(f"No image data found in {sci}")
            img = hdul[idx].data.astype(np.float32, copy=False)
            hdr0 = hdul[idx].header.copy()

    with fits.open(slitid_path) as hdul:
        slitid = hdul[0].data.astype(np.int32)
        hdr_slitid = hdul[0].header

    if img.shape != slitid.shape:
        raise ValueError(f"Image and slitid shapes differ: {img.shape} vs {slitid.shape}")

    ny, nx = img.shape
    y_full = np.arange(ny, dtype=np.float32)

    with fits.open(geom_path) as gh:
        geom_by_id = {}
        for ext in range(1, len(gh)):
            h = gh[ext].header
            sid = int(h.get("SLITID", 0))
            if sid >= 0:
                geom_by_id[sid] = h

    vals, cnt = np.unique(slitid, return_counts=True)
    bkgid = hdr_slitid.get("BKGID", None)
    if bkgid is None:
        bkgid = int(vals[int(np.argmax(cnt))]) if len(vals) else -1
    else:
        bkgid = int(bkgid)

    slit_ids = sorted(int(v) for v in vals if int(v) != bkgid)
    slit_ids = [sid for sid in slit_ids if sid % 2 == (0 if trace_set == "EVEN" else 1)]
    log.info(
        "SlitIDs present in slitid map for %s: %s",
        trace_set,
        f"{slit_ids[0]}..{slit_ids[-1]} ({len(slit_ids)} slits)" if slit_ids else "NONE",
    )

    phdr = hdr0.copy()
    phdr["GEOM"] = ("TRACECOORDS", "Rectified slit frame (y=dispersion, x=cross-disp)")
    phdr["DISPAXIS"] = ("Y", "Dispersion axis in TRACECOORDS")
    phdr["CROSSAX"] = ("X", "Cross-dispersion axis in TRACECOORDS")
    phdr["PADX"] = (int(args.padx), "Extra x padding on each side in TRACECOORDS")
    phdr["TRACESET"] = (trace_set, "Trace set used (EVEN/ODD)")
    phdr["SLITIDF"] = (slitid_path.name, "SlitID map used")
    phdr["GEOMF"] = (geom_path.name, "Geometry file used")
    phdr["WIDTHTAB"] = (slit_table_csv.name, "Slit table used")
    out_hdus_slit = [fits.PrimaryHDU(header=phdr)]

    for sid in slit_ids:
        if sid not in geom_by_id or not np.any(slitid == sid):
            continue
        hgeo = geom_by_id[sid]

        y_min = int(hgeo.get("YMIN", 0))
        y_max = int(hgeo.get("YMAX", ny - 1))
        y_min = max(0, min(ny - 1, y_min))
        y_max = max(0, min(ny - 1, y_max))
        if y_max <= y_min:
            y_min, y_max = 0, ny - 1

        hw = halfwidth_by_id.get(sid, default_hw)
        porder = int(hgeo.get("PORDER", 5))
        eporder = int(hgeo.get("EPORDER", porder))
        edge_pad = float(hgeo.get("EPAD", 0.5))
        pad_int = int(np.ceil(max(0.0, edge_pad)))

        x_ref = float(hgeo.get("XREF", np.nan))
        coeff_c = np.array([float(hgeo.get(f"PC{i}", 0.0)) for i in range(porder + 1)], dtype=np.float32)
        xC_model = _poly_eval_powerbasis(coeff_c, y_full)

        has_edges = ("LC0" in hgeo) and ("RC0" in hgeo)
        if has_edges:
            coeff_l = np.array([float(hgeo.get(f"LC{i}", 0.0)) for i in range(eporder + 1)], dtype=np.float32)
            coeff_r = np.array([float(hgeo.get(f"RC{i}", 0.0)) for i in range(eporder + 1)], dtype=np.float32)
            xL_model = _poly_eval_powerbasis(coeff_l, y_full)
            xR_model = _poly_eval_powerbasis(coeff_r, y_full)
        else:
            xL_model = xC_model - float(hw)
            xR_model = xC_model + float(hw)

        xMID_model = 0.5 * (xL_model + xR_model)
        if not np.isfinite(x_ref):
            x_ref = float(np.nanmedian(xMID_model[y_min:y_max + 1]))

        w_model = np.nanmedian(xR_model[y_min:y_max + 1] - xL_model[y_min:y_max + 1])
        if (not np.isfinite(w_model)) or (w_model <= 0):
            w_model = float(2 * hw)

        W0 = float(w_model)
        W0_int = int(np.round(W0)) + 1
        W0_int = max(1, min(W0_int, nx))

        mask_width_int = int(min(nx, W0_int + 2 * pad_int))
        mask_width_int = max(1, mask_width_int)

        lo_mask = int(np.round(x_ref - 0.5 * mask_width_int))
        hi_mask = int(lo_mask + mask_width_int - 1)
        if lo_mask < 0:
            hi_mask -= lo_mask
            lo_mask = 0
        if hi_mask > (nx - 1):
            shift_back = hi_mask - (nx - 1)
            lo_mask -= shift_back
            hi_mask = nx - 1
            lo_mask = max(0, lo_mask)

        half_mask = int(np.ceil(mask_width_int / 2.0))
        cut_hw = int(max(hw + pad_int, half_mask + 1))
        x_ref_int = int(np.round(x_ref))
        x_lo = max(0, x_ref_int - cut_hw)
        x_hi = min(nx, x_ref_int + cut_hw + 1)

        x_out_full = np.arange(x_lo, x_hi, dtype=np.float32)
        nx_cut = x_out_full.size
        rect_cut = np.full((y_max - y_min + 1, nx_cut), np.nan, dtype=np.float32)

        lo_cut = int(max(0, lo_mask - x_lo))
        hi_cut = int(min(nx_cut - 1, hi_mask - x_lo))
        mask_row = np.zeros(nx_cut, dtype=bool)
        if hi_cut >= lo_cut:
            mask_row[lo_cut:hi_cut + 1] = True

        for yy in range(y_min, y_max + 1):
            W_y = float(xR_model[yy] - xL_model[yy])
            if (not np.isfinite(W_y)) or (W_y <= 1e-3):
                W_y = float(W0)
            xmid = float(xMID_model[yy])
            x_in = xmid + (x_out_full - np.float32(x_ref)) * (np.float32(W_y) / np.float32(W0))
            coords = np.vstack([
                np.full(nx_cut, yy, dtype=np.float32),
                x_in.astype(np.float32),
            ])
            row_out = map_coordinates(
                img,
                coords,
                order=int(args.interp_order),
                prefilter=(int(args.interp_order) > 1),
                mode="constant",
                cval=np.nan,
            ).astype(np.float32)
            row_out[~mask_row] = np.nan
            rect_cut[yy - y_min, :] = row_out

        keep_row_data = np.isfinite(rect_cut).any(axis=1)
        if hi_cut >= lo_cut:
            keep_row_mask = (
                np.isfinite(rect_cut[:, lo_cut:hi_cut + 1]).sum(axis=1) >= int(args.min_mask_pix_per_row)
            )
        else:
            keep_row_mask = keep_row_data
        keep_row = keep_row_data & keep_row_mask
        if not np.any(keep_row):
            log.warning("SLIT%03d: no usable rows after screening; skipping.", sid)
            continue

        yy_idx = np.where(keep_row)[0]
        yy0, yy1 = int(yy_idx[0]), int(yy_idx[-1])
        rect_cut = rect_cut[yy0:yy1 + 1, :].copy()
        y_min_out = y_min + yy0
        y_max_out = y_min_out + rect_cut.shape[0] - 1

        hh = fits.Header()
        hh["EXTNAME"] = f"SLIT{sid:03d}"
        hh["SLITID"] = sid
        for key in ("INDEX", "RA", "DEC", "OLDSID"):
            if key in hgeo:
                hh[key] = hgeo[key]
        hh["SRCFILE"] = (_safe_key(sci.name), "Source 2D file")
        hh["GEOMREF"] = (_safe_key(geom_path.name), "Fixed geometry from quartz/flat Step04")
        hh["TRACESET"] = (_safe_key(trace_set), "Trace set used (EVEN/ODD)")
        hh["SLITIDF"] = (_safe_key(slitid_path.name), "SlitID map used")
        hh["WIDREF"] = (slit_table_csv.name, "Per-slit widths from step04 traces table")
        hh["WIDKEY"] = (args.width_key, "Width statistic used from traces table")
        hh["WIDPAD"] = (float(args.pad_pix), "Pad added to width before halving (px)")
        hh["HFWID"] = (int(hw), "Half-width from traces table (px)")
        hh["MSKWID"] = (int(mask_width_int), "Rectified mask width (px), constant")
        hh["MSKLO"] = (int(lo_mask), "Rectified mask left edge (full-frame x)")
        hh["MSKHI"] = (int(hi_mask), "Rectified mask right edge (full-frame x)")
        hh["YMIN"] = int(y_min_out)
        hh["YMAX"] = int(y_max_out)
        hh["XREF"] = float(x_ref)
        hh["XLO"] = int(x_lo)
        hh["XHI"] = int(x_hi)
        hh["XMIN"] = (int(x_lo), "Detector x at left edge of cutout (inclusive)")
        hh["XMAX"] = (int(x_hi - 1), "Detector x at right edge of cutout (inclusive)")
        hh["XWIN"] = (int(x_lo), "Alias for XMIN (detector x of cutout column 0)")
        hh["Y0DET"] = (int(y_min_out), "Detector y corresponding to row 0 of this cutout")
        hh["PORDER"] = int(porder)
        hh["EPORDER"] = int(eporder)
        hh["EPAD"] = float(edge_pad)
        hh["W0"] = (float(W0), "Median modeled slit width (edge separation, px)")
        hh["W0INT"] = (int(W0_int), "Median modeled slit width (inclusive px)")
        hh["RECTMAP"] = ("x_in=xmid+(x_out-xref)*W(y)/W0", "Rectification mapping")
        for i, c in enumerate(coeff_c):
            hh[f"PC{i}"] = (float(c), f"center x(y) coeff c{i} in px, power basis")
        if has_edges:
            for i, c in enumerate(coeff_l):
                hh[f"LC{i}"] = (float(c), f"left edge xL(y) coeff c{i} in px, power basis")
            for i, c in enumerate(coeff_r):
                hh[f"RC{i}"] = (float(c), f"right edge xR(y) coeff c{i} in px, power basis")

        if args.padx and int(args.padx) > 0:
            pad = int(args.padx)
            nyc, nxc = rect_cut.shape
            rect_pad = np.full((nyc, nxc + 2 * pad), np.nan, dtype=np.float32)
            rect_pad[:, pad:pad + nxc] = rect_cut.astype(np.float32)
            rect_cut = rect_pad
            hh["PADX"] = (pad, "Extra x padding on each side in TRACECOORDS")
        hh["GEOM"] = ("TRACECOORDS", "Rectified slit frame (y=dispersion, x=cross-disp)")
        hh["DISPAXIS"] = ("Y", "Dispersion axis in this image")
        hh["CROSSAX"] = ("X", "Cross-dispersion axis in this image")
        out_hdus_slit.append(fits.ImageHDU(data=rect_cut.astype(np.float32), header=hh))

        log.info(
            "SLIT%03d: W0_int=%d  mask_w=%d  cut_w=%d  x_lo=%d x_hi=%d  y=[%d,%d]",
            sid, W0_int, mask_width_int, nx_cut, x_lo, x_hi, y_min_out, y_max_out
        )

    fits.HDUList(out_hdus_slit).writeto(out_mef_tracecoords, overwrite=True)
    log.info("Wrote: %s", out_mef_tracecoords)


if __name__ == "__main__":
    main()
