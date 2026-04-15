#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step08_add_obj_presky_reference.py

Reference patch script for the SAMOS Step08 extractor.

Purpose
-------
This is a *drop-in reference implementation* showing the exact logic needed to
carry a true pre-sky-subtraction extracted spectrum into the Step08 output table.

It does NOT attempt to replace your full Step08 script, because that script was
not provided in this chat. Instead, it gives you the exact code block to insert
in the extraction stage, plus a small helper writer that you can call from your
existing Step08 code with your current arrays.

What must be preserved
----------------------
For each slit / row, Step08 currently computes something equivalent to:

    OBJ_PRESKY_1D = optimally extracted object+sky spectrum
    SKY_1D        = extracted sky model on the same weighting basis
    FLUX_1D       = OBJ_PRESKY_1D - SKY_1D

The critical requirement is that OBJ_PRESKY_1D must be written *before* any sky
subtraction and with the *same extraction weights* used for FLUX_1D.

Why this matters
----------------
OBJ_PRESKY (or FLUX_RAW) is the correct input to the alternative OH-removal path
in Step09. Reconstructing it later as FLUX + SKY is not reliable enough for the
purpose, because FLUX and SKY are already transformed products.

How to use this file
--------------------
1) Open your current Step08 script.
2) Find the place where you already compute:
      - extracted flux before subtraction
      - extracted sky
      - sky-subtracted flux
      - variance
3) Replace / augment the table-writing block with the helper below.
4) Keep all existing columns to preserve downstream compatibility.
5) Add the new column name exactly as:
      OBJ_PRESKY
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
from astropy.io import fits


# ----------------------------------------------------------------------
# Helper 1: build the Step08 output table with OBJ_PRESKY added
# ----------------------------------------------------------------------

def build_step08_table_with_obj_presky(
    *,
    ypix: np.ndarray,
    flux_sky_sub: np.ndarray,
    var_sky_sub: np.ndarray,
    sky_model: np.ndarray,
    obj_presky: np.ndarray,
    x0: Optional[np.ndarray] = None,
    nobj: Optional[np.ndarray] = None,
    nsky: Optional[np.ndarray] = None,
    skysig: Optional[np.ndarray] = None,
    aploss_frac: Optional[np.ndarray] = None,
    flux_apcorr: Optional[np.ndarray] = None,
    var_apcorr: Optional[np.ndarray] = None,
    edgeflag: Optional[np.ndarray] = None,
    trxleft: Optional[np.ndarray] = None,
    trxright: Optional[np.ndarray] = None,
    lambda_nm: Optional[np.ndarray] = None,
) -> fits.BinTableHDU:
    """
    Build a BinTableHDU for one slit including the new OBJ_PRESKY column.

    Required arrays must all have the same length.
    """
    arrays = {
        "YPIX": ypix,
        "FLUX": flux_sky_sub,
        "VAR": var_sky_sub,
        "SKY": sky_model,
        "OBJ_PRESKY": obj_presky,
    }
    n = len(ypix)

    for name, arr in arrays.items():
        if arr is None:
            raise ValueError(f"{name} is required")
        if len(arr) != n:
            raise ValueError(f"{name} length {len(arr)} != {n}")

    cols = [
        fits.Column(name="YPIX", format="J", array=np.asarray(ypix, dtype=np.int32)),
        fits.Column(name="FLUX", format="E", array=np.asarray(flux_sky_sub, dtype=np.float32)),
        fits.Column(name="VAR", format="E", array=np.asarray(var_sky_sub, dtype=np.float32)),
        fits.Column(name="SKY", format="E", array=np.asarray(sky_model, dtype=np.float32)),
        fits.Column(name="OBJ_PRESKY", format="E", array=np.asarray(obj_presky, dtype=np.float32)),
    ]

    def add_opt(name: str, arr: Optional[np.ndarray], fmt: str) -> None:
        if arr is not None:
            if len(arr) != n:
                raise ValueError(f"{name} length {len(arr)} != {n}")
            cols.append(fits.Column(name=name, format=fmt, array=np.asarray(arr)))

    add_opt("X0", x0, "E")
    add_opt("NOBJ", nobj, "J")
    add_opt("NSKY", nsky, "J")
    add_opt("SKYSIG", skysig, "E")
    add_opt("APLOSS_FRAC", aploss_frac, "E")
    add_opt("FLUX_APCORR", flux_apcorr, "E")
    add_opt("VAR_APCORR", var_apcorr, "E")
    add_opt("EDGEFLAG", edgeflag, "J")
    add_opt("TRXLEFT", trxleft, "E")
    add_opt("TRXRIGHT", trxright, "E")
    add_opt("LAMBDA_NM", lambda_nm, "E")

    hdu = fits.BinTableHDU.from_columns(cols)
    return hdu


# ----------------------------------------------------------------------
# Helper 2: exact extraction bookkeeping logic
# ----------------------------------------------------------------------

def compute_presky_and_flux(
    *,
    obj_presky_1d: np.ndarray,
    sky_1d: np.ndarray,
    var_presky_1d: Optional[np.ndarray] = None,
    var_sky_1d: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Canonical bookkeeping for Step08.

    Parameters
    ----------
    obj_presky_1d
        The extracted spectrum BEFORE sky subtraction, on the same weighting
        basis as the final 1D extraction. This is the quantity that must be
        stored as OBJ_PRESKY.
    sky_1d
        The extracted sky model on the same weighting basis.
    var_presky_1d
        Variance of obj_presky_1d, if available.
    var_sky_1d
        Variance contribution of sky_1d, if available.

    Returns
    -------
    dict with keys:
        OBJ_PRESKY, SKY, FLUX, VAR
    """
    obj_presky_1d = np.asarray(obj_presky_1d, dtype=float)
    sky_1d = np.asarray(sky_1d, dtype=float)

    if obj_presky_1d.shape != sky_1d.shape:
        raise ValueError("obj_presky_1d and sky_1d must have the same shape")

    flux_1d = obj_presky_1d - sky_1d

    if var_presky_1d is None and var_sky_1d is None:
        var_1d = np.full_like(flux_1d, np.nan, dtype=float)
    else:
        vp = np.zeros_like(flux_1d, dtype=float) if var_presky_1d is None else np.asarray(var_presky_1d, dtype=float)
        vs = np.zeros_like(flux_1d, dtype=float) if var_sky_1d is None else np.asarray(var_sky_1d, dtype=float)
        if vp.shape != flux_1d.shape or vs.shape != flux_1d.shape:
            raise ValueError("Variance arrays must match extracted spectrum shape")
        var_1d = vp + vs

    return {
        "OBJ_PRESKY": obj_presky_1d,
        "SKY": sky_1d,
        "FLUX": flux_1d,
        "VAR": var_1d,
    }


# ----------------------------------------------------------------------
# Example insertion block for your existing Step08 extraction loop
# ----------------------------------------------------------------------

EXAMPLE_PATCH = r"""
# ============================================================
# PATCH TO INSERT INSIDE YOUR EXISTING STEP08 SLIT EXTRACTION
# ============================================================

# Assume your current code already computes, per slit / per Y row:
#   ypix
#   x0
#   nobj
#   nsky
#   skysig
#   aploss_frac
#   flux_apcorr
#   var_apcorr
#   edgeflag
#   trxleft
#   trxright
#   lambda_nm
#
# and, critically, these two quantities on the SAME extraction basis:
#   obj_presky_1d   # extracted object+sky BEFORE subtraction
#   sky_1d          # extracted sky model
#
# If you currently only keep the sky-subtracted product:
#   flux_1d = obj_presky_1d - sky_1d
# then do NOT discard obj_presky_1d. Keep it and write it out.

book = compute_presky_and_flux(
    obj_presky_1d=obj_presky_1d,
    sky_1d=sky_1d,
    var_presky_1d=var_obj_presky_1d,   # if available
    var_sky_1d=var_sky_1d,             # if available
)

slit_hdu = build_step08_table_with_obj_presky(
    ypix=ypix,
    flux_sky_sub=book["FLUX"],
    var_sky_sub=book["VAR"],
    sky_model=book["SKY"],
    obj_presky=book["OBJ_PRESKY"],
    x0=x0,
    nobj=nobj,
    nsky=nsky,
    skysig=skysig,
    aploss_frac=aploss_frac,
    flux_apcorr=flux_apcorr,
    var_apcorr=var_apcorr,
    edgeflag=edgeflag,
    trxleft=trxleft,
    trxright=trxright,
    lambda_nm=lambda_nm,
)

slit_hdu.name = slit_extname   # e.g. "SLIT024"
hdus_out.append(slit_hdu)
"""


# ----------------------------------------------------------------------
# Step09 reader helper (optional)
# ----------------------------------------------------------------------

def read_step09_input_prefer_obj_presky(hdul: fits.HDUList, slit_name: str) -> Dict[str, np.ndarray]:
    """
    Reader logic for Step09: prefer OBJ_PRESKY if present, else fail loudly.

    This is intentionally strict so the pipeline does not silently fall back to
    FLUX+SKY after the Step08 patch is adopted.
    """
    slit_name = slit_name.strip().upper()
    hdu = None
    for ext in hdul[1:]:
        if ext.name.strip().upper() == slit_name:
            hdu = ext
            break
    if hdu is None:
        raise KeyError(f"{slit_name}: slit not found")

    data = hdu.data
    cols = {c.upper(): c for c in data.names}

    if "OBJ_PRESKY" not in cols:
        raise KeyError(
            f"{slit_name}: missing OBJ_PRESKY. "
            "Step08 must be patched to write the true pre-sky extracted spectrum."
        )

    out = {
        "LAMBDA_NM": np.asarray(data[cols["LAMBDA_NM"]], dtype=float) if "LAMBDA_NM" in cols else None,
        "OBJ_PRESKY": np.asarray(data[cols["OBJ_PRESKY"]], dtype=float),
        "SKY": np.asarray(data[cols["SKY"]], dtype=float) if "SKY" in cols else None,
        "FLUX": np.asarray(data[cols["FLUX"]], dtype=float) if "FLUX" in cols else None,
    }
    return out


if __name__ == "__main__":
    print(__doc__)
    print()
    print("=" * 78)
    print("EXAMPLE PATCH BLOCK")
    print("=" * 78)
    print(EXAMPLE_PATCH)
