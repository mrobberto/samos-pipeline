#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qc_step11_summary.py

Create a per-slit PDF summary for the final Step11 products.

For each slit, the summary page includes:
  A) detector-frame science cutout with slit-center / object-ridge overlays
  B) rectified slit panel with extraction regions
  C) WCS imaging postage stamp at the target position
  D) RA/Dec mini-map + SkyMapper r/i/z photometry table
  E) 1D spectrum panel

Display policy
--------------
The 1D spectrum panel prefers the final Step11 calibrated spectrum:
  1) FLUX_FLAM
  2) FLUX_TELLCOR_O2
  3) FLUX

This script is intended to serve as the final closeout QC for Step11.
It should be callable both:
- directly from the command line
- from the Step11 notebook final cell
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import astropy.units as u


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _pe(p):
    if p is None:
        return "None"
    try:
        return f"{p}  (exists={Path(p).exists()})"
    except Exception:
        return str(p)


def _norm_key(s: str) -> str:
    return s.strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def get_slit_hdu(hdul, slit: str):
    """Return the HDU for a given SLIT### even if EXTNAME is not exactly SLIT###."""
    s = str(slit).strip().upper()

    try:
        return hdul[s]
    except Exception:
        pass

    m = re.match(r"SLIT\s*(\d+)", s)
    sid = int(m.group(1)) if m else None

    if sid is not None:
        for hd in hdul[1:]:
            try:
                if int(hd.header.get("SLITID", -999)) == sid:
                    return hd
            except Exception:
                continue

    for hd in hdul[1:]:
        try:
            if hd.name.strip().upper() == s:
                return hd
        except Exception:
            continue

    raise KeyError(f"Could not find HDU for {s} (no EXTNAME or SLITID match)")


def _robust_limits(img: np.ndarray, stretch_sigma: float = 5.0) -> Tuple[float, float]:
    v = img[np.isfinite(img)]
    if v.size == 0:
        return 0.0, 1.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(v))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    vmin = med - stretch_sigma * sigma
    vmax = med + stretch_sigma * sigma
    return vmin, vmax


def _vbin(img: np.ndarray, fac: int) -> np.ndarray:
    if fac <= 1:
        return img
    ny, nx = img.shape
    n2 = (ny // fac) * fac
    if n2 <= 0:
        return img
    a = img[:n2, :]
    a = a.reshape(n2 // fac, fac, nx)
    return np.nanmean(a, axis=1)


def _abmag_to_fnu_jy(m: float) -> float:
    return 3631.0 * (10.0 ** (-0.4 * m))


def _fnu_jy_to_flambda(fnu_jy: float, lam_nm: float) -> float:
    cA = 2.99792458e18  # Angstrom/s
    lam_A = lam_nm * 10.0
    fnu_cgs = fnu_jy * 1e-23
    return fnu_cgs * cA / (lam_A ** 2)


def _choose_columns(tab: fits.FITS_rec, prefer_flux: Optional[str], prefer_lam: Optional[str]) -> Tuple[str, str]:
    cols = [c.name for c in tab.columns]
    ncols = {_norm_key(c): c for c in cols}

    if prefer_lam and prefer_lam in cols:
        lam_col = prefer_lam
    else:
        for cand in ["LAMBDA_NM", "lambda_nm", "LAMBDA", "WAVE_NM", "WAVELENGTH_NM"]:
            if _norm_key(cand) in ncols:
                lam_col = ncols[_norm_key(cand)]
                break
        else:
            raise KeyError(f"Could not find wavelength column in {cols}")

    if prefer_flux and prefer_flux in cols:
        flux_col = prefer_flux
    else:
        for cand in ["FLUX_FLAM", "FLUX_TELLCOR_O2", "FLUX_TELLCOR", "FLUX", "OBJ_SKYSUB", "OBJ_RAW"]:
            if _norm_key(cand) in ncols:
                flux_col = ncols[_norm_key(cand)]
                break
        else:
            for c in cols:
                if _norm_key(c) == _norm_key(lam_col):
                    continue
                if tab[c].dtype.kind in ("f", "i"):
                    flux_col = c
                    break
            else:
                raise KeyError(f"Could not auto-choose a flux column in {cols}")

    return lam_col, flux_col


def _choose_display_spectrum(tab: fits.FITS_rec, lam_col: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """
    Choose the best spectrum to display for the Step11 summary pages.

    Priority:
      1) FLUX_FLAM       (final flux-calibrated spectrum) in black
      2) FLUX_TELLCOR_O2 (telluric-corrected pre-fluxcal spectrum) in red
      3) FLUX            (raw extracted spectrum) in red
    """
    lam = np.asarray(tab[lam_col], float)
    cols = [c.name for c in tab.columns]

    def finite_array(colname: str):
        if colname not in cols:
            return None
        arr = np.asarray(tab[colname], float)
        if np.isfinite(arr).sum() == 0:
            return None
        return arr

    arr = finite_array("FLUX_FLAM")
    if arr is not None:
        return lam, arr, "FLUX_FLAM", "k"

    arr = finite_array("FLUX_TELLCOR_O2")
    if arr is not None:
        return lam, arr, "FLUX_TELLCOR_O2", "red"

    arr = finite_array("FLUX")
    if arr is not None:
        return lam, arr, "FLUX", "red"

    for c in cols:
        cu = c.upper()
        if c == lam_col or cu in {"VAR", "VAR_FLAM2", "SKY", "X0", "NOBJ", "NSKY", "SKYSIG", "OH_SHIFT_NM", "YPIX"}:
            continue
        try:
            arr = np.asarray(tab[c], float)
        except Exception:
            continue
        if np.isfinite(arr).sum() > 0:
            return lam, arr, c, "red"

    return lam, np.full_like(lam, np.nan, dtype=float), "Flux", "red"


def _read_photcat(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    colmap = {_norm_key(c): c for c in df.columns}

    def pick(*names):
        for n in names:
            k = _norm_key(n)
            if k in colmap:
                return colmap[k]
        return None

    slit_c = pick("slit", "slit_id", "slitnum", "slit_number", "slitname", "extname")
    ra_c = pick("ra", "ra_deg", "raj2000", "ra2000")
    dec_c = pick("dec", "dec_deg", "dej2000", "dec2000")
    r_c = pick("r_mag", "rmag", "r", "sm_r", "r_psf", "r_petro")
    i_c = pick("i_mag", "imag", "i", "sm_i", "i_psf", "i_petro")
    z_c = pick("z_mag", "zmag", "z", "sm_z", "z_psf", "z_petro")

    missing = [("SLIT", slit_c), ("RA", ra_c), ("DEC", dec_c), ("r_mag", r_c), ("i_mag", i_c), ("z_mag", z_c)]
    miss = [n for n, c in missing if c is None]
    if miss:
        raise KeyError(f"photcat missing columns: {miss}. Found={list(df.columns)}")

    out = pd.DataFrame({
        "SLIT": df[slit_c].astype(str),
        "RA": df[ra_c].astype(float),
        "DEC": df[dec_c].astype(float),
        "r_mag": pd.to_numeric(df[r_c], errors="coerce"),
        "i_mag": pd.to_numeric(df[i_c], errors="coerce"),
        "z_mag": pd.to_numeric(df[z_c], errors="coerce"),
    })

    def norm_slit(s: str) -> str:
        s2 = s.strip().upper()
        dig = "".join(ch for ch in s2 if ch.isdigit())
        if dig:
            return f"SLIT{int(dig):03d}"
        return s2

    out["SLIT"] = out["SLIT"].map(norm_slit)
    out = out.dropna(subset=["RA", "DEC"])
    return out


def _split_even_odd(path_str: Optional[str]) -> tuple[Optional[Path], Optional[Path]]:
    if not path_str:
        return None, None
    s = str(path_str)
    for sep in ["|", ";"]:
        if sep in s:
            a, b = [p.strip() for p in s.split(sep, 1)]
            return (Path(a) if a else None), (Path(b) if b else None)
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) == 2 and Path(parts[0]).exists() and Path(parts[1]).exists():
            return Path(parts[0]), Path(parts[1])
    p = Path(s)
    return p, p


def _pick_mef_for_slit(slit: str, even_path: Optional[Path], odd_path: Optional[Path]) -> Optional[Path]:
    if even_path is None and odd_path is None:
        return None
    m = re.match(r"SLIT(\d+)", slit.upper().strip())
    if not m:
        return even_path or odd_path
    sid = int(m.group(1))
    return even_path if (sid % 2 == 0) else odd_path


def _parse_slit_list(spec: Optional[str], available: list[str]) -> list[str]:
    if not spec:
        return available
    items = []
    for part in spec.split(","):
        p = part.strip()
        if not p:
            continue
        if p.upper().startswith("SLIT"):
            items.append(p.upper().strip())
        else:
            i = int(p)
            items.append(f"SLIT{i:03d}")
    avail_set = set(available)
    out = [s for s in items if s in avail_set]
    if not out:
        raise ValueError("None of the requested slits were found in the extract FITS.")
    return out


def _cutout_from_wcs(fits_path: Path, ra_deg: float, dec_deg: float, halfsize_px: int):
    h = _open_cached(fits_path)
    if h[0].data is not None and h[0].data.ndim >= 2:
        data = h[0].data
        w = WCS(h[0].header)
    else:
        img_hdu = None
        for k in range(1, len(h)):
            if h[k].data is not None and getattr(h[k], "is_image", False):
                img_hdu = h[k]
                break
        if img_hdu is None:
            raise RuntimeError(f"No image HDU found in {fits_path}")
        data = img_hdu.data
        w = WCS(img_hdu.header)

    data2 = data[0] if data.ndim > 2 else data
    pos = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    size = (2 * halfsize_px + 1, 2 * halfsize_px + 1)
    cut = Cutout2D(data2, position=pos, size=size, wcs=w, mode="partial", fill_value=np.nan)
    return cut.data, cut.wcs


def _add_stamp_axes(fig, gs_cell, stamp, stamp_wcs, title: str):
    try:
        ax = fig.add_subplot(gs_cell, projection=stamp_wcs)
        ax.imshow(stamp, origin="lower")
        ax.set_title(title)
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")
        return ax, "WCS"
    except Exception as e:
        ax = fig.add_subplot(gs_cell)
        ax.imshow(stamp, origin="lower")
        ax.set_title(title + " (pixel)")
        ax.set_xlabel("x (pix)")
        ax.set_ylabel("y (pix)")
        ax.text(
            0.01, 0.99, f"WCSAxes unavailable:\n{type(e).__name__}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", alpha=0.15),
        )
        return ax, "PIX"


def _read_slit_image_and_header(mef_path: Optional[Path], slit: str) -> Tuple[Optional[np.ndarray], Optional[fits.Header]]:
    if mef_path is None:
        return None, None
    mp = Path(mef_path)
    if not mp.exists():
        return None, None

    h = _open_cached(mp)
    if len(h) > 1:
        try:
            hd = get_slit_hdu(h, slit)
        except Exception:
            return None, None
        d = hd.data
        if d is None:
            return None, None
        if ("mask" in mp.name.lower()) and getattr(d, "ndim", 0) != 2:
            d2 = _read_slit_image(mp, slit)
            return (np.asarray(d2, float) if d2 is not None else None), hd.header
        if getattr(d, "ndim", 0) != 2:
            return None, hd.header
        return np.asarray(d, float), hd.header

    d0 = h[0].data
    if d0 is None:
        return None, h[0].header
    if getattr(d0, "ndim", 0) == 2:
        return np.asarray(d0, float), h[0].header
    if getattr(d0, "ndim", 0) == 3:
        return np.asarray(d0[0, :, :], float), h[0].header
    return None, h[0].header


def _pick_mask_plane(mask_data: np.ndarray, slit: str, filename: str) -> Optional[np.ndarray]:
    if mask_data is None:
        return None
    if mask_data.ndim == 2:
        return mask_data
    if mask_data.ndim != 3:
        return None

    m = re.match(r"SLIT(\d+)", str(slit).strip().upper())
    sid = int(m.group(1)) if m else None
    if sid is None:
        return mask_data[0]

    shape = mask_data.shape
    cand_axes = [ax for ax, n in enumerate(shape) if 1 < n <= 80]
    slit_ax = cand_axes[0] if cand_axes else 0

    nslit = shape[slit_ax]
    name = filename.lower()
    idx = None
    if "even" in name and sid % 2 == 0:
        idx = sid // 2 - 1
    elif "odd" in name and sid % 2 == 1:
        idx = (sid + 1) // 2 - 1

    if idx is None and nslit >= sid:
        idx = sid - 1
    if idx is None:
        idx = 0
    idx = int(np.clip(idx, 0, nslit - 1))

    if slit_ax == 0:
        return mask_data[idx, :, :]
    if slit_ax == 1:
        return mask_data[:, idx, :]
    return mask_data[:, :, idx]


def _read_slit_image(mef_path: Optional[Path], slit: str) -> Optional[np.ndarray]:
    if mef_path is None:
        return None
    mp = Path(mef_path)
    if not mp.exists():
        return None

    h = _open_cached(mp)
    if len(h) > 1:
        try:
            d = get_slit_hdu(h, slit).data
        except Exception:
            d = None
        if d is None or getattr(d, "ndim", 0) != 2:
            return None
        return np.asarray(d, float)

    d0 = h[0].data
    if d0 is None:
        return None

    if "mask" in mp.name.lower():
        plane = _pick_mask_plane(np.asarray(d0), slit, mp.name)
        if plane is None or getattr(plane, "ndim", 0) != 2:
            return None
        return np.asarray(plane, float)

    if getattr(d0, "ndim", 0) == 2:
        return np.asarray(d0, float)
    if getattr(d0, "ndim", 0) == 3:
        return np.asarray(d0[0, :, :], float)
    return None


def _object_ridge_constrained(cut: np.ndarray, geom1d: np.ndarray, X0: int, Y0: int, ridge_halfwidth: int = 6) -> np.ndarray:
    cut = np.asarray(cut, float)
    ny, nx = cut.shape
    x_obj = np.full(ny, np.nan, float)
    if geom1d is None or geom1d.size == 0:
        return x_obj

    for j in range(ny):
        y_det = Y0 + j
        if y_det < 0 or y_det >= geom1d.size:
            continue
        x0_det = float(geom1d[y_det])
        if not np.isfinite(x0_det):
            continue
        x0 = x0_det - X0

        lo = int(max(0, np.floor(x0 - ridge_halfwidth)))
        hi = int(min(nx, np.ceil(x0 + ridge_halfwidth + 1)))
        if hi - lo < 3:
            continue

        row = cut[j, lo:hi]
        tmp = np.where(np.isfinite(row), row, -np.inf)
        k = int(np.argmax(tmp))
        if not np.isfinite(tmp[k]):
            continue
        x_obj[j] = lo + k

    return x_obj


def _plot_detector_cutout_with_ridge(
    ax,
    sci2d: np.ndarray,
    hdr_trace: fits.Header,
    geom1d: np.ndarray,
    stretch_sigma: float = 5.0,
    pad_x: int = 6,
    pad_y: int = 20,
    show_obj_ridge: bool = True,
    ridge_halfwidth: int = 6,
):
    if sci2d is None or hdr_trace is None or geom1d is None:
        return None

    for k in ("XLO", "XHI", "YMIN", "YMAX"):
        if k not in hdr_trace:
            return None

    xlo = int(hdr_trace["XLO"])
    xhi = int(hdr_trace["XHI"])
    ymin = int(hdr_trace["YMIN"])
    ymax = int(hdr_trace["YMAX"])

    ny_det, nx_det = sci2d.shape
    X0 = max(0, xlo - pad_x)
    X1 = min(nx_det, xhi + pad_x)
    Y0 = max(0, ymin - pad_y)
    Y1 = min(ny_det, ymax + pad_y)

    if (X1 <= X0 + 2) or (Y1 <= Y0 + 2):
        return None

    cut = np.asarray(sci2d[Y0:Y1, X0:X1], float)
    vmin, vmax = _robust_limits(cut, float(stretch_sigma))
    ax.imshow(cut, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)

    y_det = np.arange(Y0, Y1)
    ok = (y_det >= 0) & (y_det < geom1d.size) & np.isfinite(geom1d[y_det])
    if np.any(ok):
        x_det = geom1d[y_det[ok]]
        y_det_ok = y_det[ok]
        ax.plot(x_det - X0, y_det_ok - Y0, lw=1.2, label="slit center (Step04)")

    if show_obj_ridge:
        x_obj = _object_ridge_constrained(cut, geom1d, X0, Y0, ridge_halfwidth=int(ridge_halfwidth))
        y_obj = np.arange(cut.shape[0], dtype=float)
        if np.any(np.isfinite(x_obj)):
            ax.plot(x_obj, y_obj, lw=1.0, label="object ridge (constrained)")

    ax.set_xlabel("Detector X (cutout px)")
    ax.set_ylabel("Detector Y (cutout px)")
    ax.legend(loc="upper right", fontsize=8)

    return dict(X0=X0, X1=X1, Y0=Y0, Y1=Y1, xlo=xlo, xhi=xhi, ymin=ymin, ymax=ymax)


def _scalar_from_tab(tab: fits.FITS_rec, col: str, default=np.nan) -> float:
    if tab is None or col is None:
        return float(default)
    if col not in tab.columns.names:
        return float(default)
    v = np.asarray(tab[col], float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float(default)
    return float(np.nanmedian(v))


def _plot_rectified_with_apertures(
    ax,
    img_rect: np.ndarray,
    tab: fits.FITS_rec,
    vbin: int = 6,
    stretch_sigma: float = 5.0,
    ridge_halfwidth: int = 6,
    sky_gap: int = 2,
    show_obj_ridge: bool = True,
):
    if img_rect is None:
        ax.text(0.5, 0.5, "Rectified image not available", ha="center", va="center")
        ax.set_axis_off()
        return

    im = _vbin(np.asarray(img_rect, float), int(vbin))
    vmin, vmax = _robust_limits(im, float(stretch_sigma))
    ax.imshow(im, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)

    ny, nx = im.shape
    x0 = _scalar_from_tab(tab, "X0", default=np.nan)
    nobj = _scalar_from_tab(tab, "NOBJ", default=np.nan)
    nsky = _scalar_from_tab(tab, "NSKY", default=np.nan)

    x_ridge = None
    if show_obj_ridge:
        x_ridge = np.full(ny, np.nan, float)
        for j in range(ny):
            row = im[j, :]
            if not np.any(np.isfinite(row)):
                continue
            if np.isfinite(x0):
                lo = int(max(0, np.floor(x0 - ridge_halfwidth)))
                hi = int(min(nx, np.ceil(x0 + ridge_halfwidth + 1)))
                if hi - lo < 3:
                    lo, hi = 0, nx
            else:
                lo, hi = 0, nx
            r = row[lo:hi]
            tmp = np.where(np.isfinite(r), r, -np.inf)
            k = int(np.argmax(tmp))
            if np.isfinite(tmp[k]):
                x_ridge[j] = lo + k

        ok = np.isfinite(x_ridge)
        if np.any(ok):
            ax.plot(x_ridge[ok], np.where(ok)[0], lw=1.0, label="object ridge")
            if not np.isfinite(x0):
                x0 = float(np.nanmedian(x_ridge[ok]))

    if x_ridge is None:
        x0_use = x0 if np.isfinite(x0) else 0.5 * (nx - 1)
        x_ridge = np.full(ny, float(x0_use), float)

    yy = np.arange(ny, dtype=float)
    okr = np.isfinite(x_ridge)
    if np.any(okr) and not show_obj_ridge:
        ax.plot(x_ridge[okr], yy[okr], lw=1.0, label="object ridge")

    half = 0.0
    if np.isfinite(nobj) and nobj > 0:
        half = 0.5 * float(nobj)
        ax.plot((x_ridge - half)[okr], yy[okr], lw=0.9, ls="--", label="obj aperture")
        ax.plot((x_ridge + half)[okr], yy[okr], lw=0.9, ls="--")

    if np.isfinite(nsky) and nsky > 0:
        g = float(sky_gap)
        nskyf = float(nsky)
        ax.plot((x_ridge - (half + g))[okr], yy[okr], lw=0.8, ls=":", label="sky regions")
        ax.plot((x_ridge - (half + g + nskyf))[okr], yy[okr], lw=0.8, ls=":")
        ax.plot((x_ridge + (half + g))[okr], yy[okr], lw=0.8, ls=":")
        ax.plot((x_ridge + (half + g + nskyf))[okr], yy[okr], lw=0.8, ls=":")

    ax.set_xlabel("x (rectified spatial pix)")
    ax.set_ylabel(f"y (binned rows, vbin={int(vbin)})")
    ax.legend(loc="upper right", fontsize=8)


# -----------------------------------------------------------------------------
# Lightweight FITS cache
# -----------------------------------------------------------------------------
_FITS_CACHE = {}


def _open_cached(path: Optional[Path]):
    if path is None:
        return None
    p = str(Path(path))
    hdul = _FITS_CACHE.get(p)
    if hdul is None:
        hdul = fits.open(p, memmap=True)
        _FITS_CACHE[p] = hdul
    return hdul


def _close_cache():
    for hdul in _FITS_CACHE.values():
        try:
            hdul.close()
        except Exception:
            pass
    _FITS_CACHE.clear()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Create per-slit Step11 summary PDF pages.")

    p.add_argument("--extract", required=True, help="Input Step11 flux-calibrated MEF FITS")
    p.add_argument("--photcat", required=True, help="Step11 photometry/RADEC catalog CSV")
    p.add_argument("--image", required=True, help="WCS imaging FITS for postage-stamp cutouts")
    p.add_argument("--detector", required=True, help="Detector-frame science image FITS")

    p.add_argument("--tracecoords", required=True, help="TRACECOORDS FITS, or EVEN|ODD pair")
    p.add_argument("--slitcoords", default="", help="Optional SLITCOORDS FITS, or EVEN|ODD pair")
    p.add_argument("--mask", default="", help="Optional mask FITS, or EVEN|ODD pair")
    p.add_argument("--geometry", default="", help="Optional geometry FITS, or EVEN|ODD pair")

    p.add_argument("--outpdf", required=True, help="Output summary PDF")
    p.add_argument("--slits", default=None, help="Optional comma-separated slit list")

    p.add_argument("--cutout", type=int, default=35)
    p.add_argument("--vbin", type=int, default=6)
    p.add_argument("--stretch", type=float, default=5.0)
    p.add_argument("--det-pad-x", type=int, default=6)
    p.add_argument("--det-pad-y", type=int, default=20)
    p.add_argument("--det-ridge-halfwidth", type=int, default=6)
    p.add_argument("--rect-ridge-halfwidth", type=int, default=6)
    p.add_argument("--sky-gap", type=int, default=2)

    p.add_argument("--lambda-r", type=float, default=616.0)
    p.add_argument("--lambda-i", type=float, default=779.0)
    p.add_argument("--lambda-z", type=float, default=916.0)

    p.add_argument(
        "--spec-col",
        default="FLUX_FLAM",
        help="Preferred spectrum column; display logic still falls back automatically",
    )
    p.add_argument("--lam-col", default="LAMBDA_NM")

    p.add_argument("--mask-mode", default="rectified", choices=["rectified", "panel", "geometry", "overlay"])
    p.add_argument("--mask-crop", default="match_trace", choices=["bbox", "match_trace"])
    p.add_argument("--show-edges", action="store_true", default=True)
    p.add_argument("--search-halfwidth", type=int, default=120)
    p.add_argument("--mask-pad", type=int, default=6)
    p.add_argument("--quiet", action="store_true")

    return p.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    extract_fits = Path(args.extract)
    photcat_csv = Path(args.photcat)
    wcs_fits = Path(args.image)
    detector_fits = Path(args.detector)
    out_pdf = Path(args.outpdf)

    if not extract_fits.exists():
        raise FileNotFoundError(extract_fits)
    if not photcat_csv.exists():
        raise FileNotFoundError(photcat_csv)
    if not wcs_fits.exists():
        raise FileNotFoundError(wcs_fits)
    if not detector_fits.exists():
        raise FileNotFoundError(detector_fits)

    phot = _read_photcat(photcat_csv)
    phot_idx = {r.SLIT: r for r in phot.itertuples(index=False)}

    ra_all = phot["RA"].to_numpy(float)
    dec_all = phot["DEC"].to_numpy(float)

    trace_even, trace_odd = _split_even_odd(args.tracecoords)
    slit_even, slit_odd = _split_even_odd(args.slitcoords)
    mask_even, mask_odd = _split_even_odd(args.mask)
    geom_even, geom_odd = _split_even_odd(args.geometry)

    with fits.open(detector_fits) as _hs:
        if _hs[0].data is None:
            raise RuntimeError(f"Detector science file has no PRIMARY image: {detector_fits}")
        sci_full = np.asarray(_hs[0].data, float)
        if sci_full.ndim != 2:
            sci_full = np.asarray(sci_full[0], float)

    with fits.open(extract_fits) as h:
        slits = []
        for hd in h[1:]:
            name = (hd.name or "").strip().upper()
            if name.startswith("SLIT"):
                slits.append(name)
        seen = set()
        slits = [s for s in slits if not (s in seen or seen.add(s))]
        slits = sorted(slits, key=lambda s: int("".join(ch for ch in s if ch.isdigit()) or 0))

        if not args.quiet:
            print(f"Found {len(slits)} SLIT extensions in extract file.")
            print("EXTRACT =", extract_fits)
            print("TRACECOORDS EVEN/ODD =", _pe(trace_even), _pe(trace_odd))
            print("SLITCOORDS  EVEN/ODD =", _pe(slit_even), _pe(slit_odd))
            print("MASK       EVEN/ODD =", _pe(mask_even), _pe(mask_odd))
            print("GEOMETRY   EVEN/ODD =", _pe(geom_even), _pe(geom_odd))
            print("WCS IMG =", wcs_fits)

        slits = _parse_slit_list(args.slits, slits)

        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        n_written = 0
        n_skipped_nophot = 0

        with PdfPages(out_pdf) as pdf:
            for slit in slits:
                hd_spec = get_slit_hdu(h, slit)
                hdr_spec = hd_spec.header
                tab = hd_spec.data

                prow = phot_idx.get(slit, None)
                ra_h = hdr_spec.get("RA", None)
                dec_h = hdr_spec.get("DEC", None)

                if (ra_h is not None) and (dec_h is not None):
                    ra, dec = float(ra_h), float(dec_h)
                elif prow is not None:
                    ra, dec = float(prow.RA), float(prow.DEC)
                else:
                    n_skipped_nophot += 1
                    continue

                lam_col, flux_col = _choose_columns(tab, args.spec_col, args.lam_col)
                trace_path = _pick_mef_for_slit(slit, trace_even, trace_odd)
                slit_path = _pick_mef_for_slit(slit, slit_even, slit_odd)

                img_trace, hdr_trace = _read_slit_image_and_header(trace_path, slit)
                img_slit, _ = _read_slit_image_and_header(slit_path, slit)

                stamp, stamp_wcs = _cutout_from_wcs(wcs_fits, ra, dec, int(args.cutout))

                fig = plt.figure(figsize=(11, 8.5))
                gs = fig.add_gridspec(
                    2, 4,
                    height_ratios=[1.05, 1.0],
                    width_ratios=[1.1, 1.1, 0.85, 1.0],
                    hspace=0.28, wspace=0.25,
                )

                # A1: detector-frame science cutout
                axA1 = fig.add_subplot(gs[0, 0])

                geom1d = None
                geom_path = _pick_mef_for_slit(slit, geom_even, geom_odd)
                if geom_path is not None and Path(geom_path).exists():
                    try:
                        _hg = _open_cached(Path(geom_path))
                        geom1d = np.asarray(get_slit_hdu(_hg, slit).data, float)
                    except Exception:
                        geom1d = None

                if (sci_full is not None) and (hdr_trace is not None) and (geom1d is not None):
                    bbox = _plot_detector_cutout_with_ridge(
                        axA1,
                        sci_full,
                        hdr_trace,
                        geom1d,
                        stretch_sigma=float(args.stretch),
                        pad_x=int(args.det_pad_x),
                        pad_y=int(args.det_pad_y),
                        show_obj_ridge=True,
                        ridge_halfwidth=int(args.det_ridge_halfwidth),
                    )
                    axA1.set_title(f"{slit} DETECTOR CUTOUT + RIDGES")
                    if bbox is not None:
                        axA1.text(
                            0.01, 0.01,
                            (
                                f"DET bbox: X[{bbox['X0']}:{bbox['X1']})  Y[{bbox['Y0']}:{bbox['Y1']})\n"
                                f"TRC bbox: X[{bbox['xlo']}:{bbox['xhi']})  Y[{bbox['ymin']}:{bbox['ymax']}]"
                            ),
                            transform=axA1.transAxes,
                            va="bottom",
                            ha="left",
                            fontsize=8,
                            family="monospace",
                            bbox=dict(boxstyle="round,pad=0.25", alpha=0.15),
                        )
                else:
                    msg = []
                    if sci_full is None:
                        msg.append("Detector science not loaded")
                    if hdr_trace is None:
                        msg.append("TRACECOORDS header missing")
                    if geom1d is None:
                        msg.append("Geometry 1D missing")
                    axA1.text(0.5, 0.5, "\n".join(msg) if msg else "Detector panel unavailable", ha="center", va="center")
                    axA1.set_axis_off()

                # A2: rectified image with extraction regions
                axA2 = fig.add_subplot(gs[0, 1])
                img_rect = img_slit if img_slit is not None else img_trace
                _plot_rectified_with_apertures(
                    axA2,
                    img_rect,
                    tab,
                    vbin=int(args.vbin),
                    stretch_sigma=float(args.stretch),
                    ridge_halfwidth=int(args.rect_ridge_halfwidth),
                    sky_gap=int(args.sky_gap),
                    show_obj_ridge=True,
                )
                axA2.set_title(f"{slit} RECTIFIED + EXTRACTION REGIONS")

                # D: sky map + photometry
                axD = fig.add_subplot(gs[0, 2])
                axD.scatter(ra_all, dec_all, s=6, alpha=0.35)
                axD.scatter([ra], [dec], s=60, marker="*", linewidths=0.8)
                axD.set_xlabel("RA (deg)")
                axD.set_ylabel("Dec (deg)")
                axD.set_title("SkyMapper positions")
                axD.grid(True, alpha=0.2)
                axD.set_aspect("equal", adjustable="box")
                axD.text(ra, dec, slit, fontsize=7)
                axD.invert_xaxis()

                if prow is not None:
                    mags = {
                        "r": float(prow.r_mag) if np.isfinite(prow.r_mag) else np.nan,
                        "i": float(prow.i_mag) if np.isfinite(prow.i_mag) else np.nan,
                        "z": float(prow.z_mag) if np.isfinite(prow.z_mag) else np.nan,
                    }
                else:
                    mags = {"r": np.nan, "i": np.nan, "z": np.nan}

                txt = (
                    f"RA  = {ra:.6f}\n"
                    f"Dec = {dec:.6f}\n\n"
                    f"r = {mags['r']:.3f}\n"
                    f"i = {mags['i']:.3f}\n"
                    f"z = {mags['z']:.3f}"
                )
                axD.text(
                    0.02, 0.02, txt,
                    transform=axD.transAxes,
                    va="bottom", ha="left",
                    fontsize=9, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", alpha=0.15),
                )

                # B: imaging cutout
                vmin, vmax = _robust_limits(stamp, float(args.stretch))
                axB, _ = _add_stamp_axes(fig, gs[0, 3], stamp, stamp_wcs, "Imaging cutout")
                axB.images[-1].set_clim(vmin, vmax)
                axB.scatter([args.cutout], [args.cutout], marker="+", s=80)
                axB.text(
                    0.02, 0.02,
                    f"center = ({ra:.6f}, {dec:.6f})",
                    transform=axB.transAxes,
                    va="bottom", ha="left",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", alpha=0.15),
                )

                # C: final 1D spectrum panel (prefer FLUX_FLAM when available) + photometry points
                axC = fig.add_subplot(gs[1, :])

                lam_disp, flux_disp, ylab, line_color = _choose_display_spectrum(tab, lam_col)
                ok = np.isfinite(lam_disp) & np.isfinite(flux_disp)
                if ok.sum() > 0:
                    axC.plot(lam_disp[ok], flux_disp[ok], lw=1.0, color=line_color)

                lam_r, lam_i, lam_z = float(args.lambda_r), float(args.lambda_i), float(args.lambda_z)
                lam_map = {"r": lam_r, "i": lam_i, "z": lam_z}

                if ylab == "FLUX_FLAM":
                    for band in ["r", "i", "z"]:
                        m = mags[band]
                        if not np.isfinite(m):
                            continue
                        fnu = _abmag_to_fnu_jy(m)
                        flam = _fnu_jy_to_flambda(fnu, lam_map[band])
                        axC.scatter([lam_map[band]], [flam], s=60, label=f"SkyMapper {band} (AB={m:.3f})")

                axC.set_xlabel("Wavelength (nm)")
                axC.set_ylabel(ylab)
                axC.grid(True, alpha=0.2)
                axC.text(
                    0.01, 0.98,
                    f"display={ylab}",
                    transform=axC.transAxes,
                    va="top", ha="left",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", alpha=0.15),
                )
                if ylab == "FLUX_FLAM":
                    axC.legend(loc="best", fontsize=9)

                yv = flux_disp[ok]
                if yv.size > 50:
                    p1, p99 = np.nanpercentile(yv, [1, 99])
                    if np.isfinite(p1) and np.isfinite(p99) and p99 > p1:
                        axC.set_ylim(p1, p99)

                fig.suptitle(f"{slit}  (Step11 final summary)", fontsize=12)
                pdf.savefig(fig)
                plt.close(fig)
                n_written += 1

        if not args.quiet:
            print("Wrote:", out_pdf)
            if n_skipped_nophot:
                print(f"Skipped {n_skipped_nophot} slits (no RA/Dec in photometry catalog).")
            print(f"Pages written: {n_written}")

    _close_cache()


if __name__ == "__main__":
    main()