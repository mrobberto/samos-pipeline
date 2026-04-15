#!/usr/bin/env python3
"""
qc_step11_summary_pages_v4_PATCHED.py

Create a per-slit PDF summary with:
  A) 2D rectified slit image (TRACECOORDS; spectrum vertical)
  B) optional SLITCOORDS panel
  C) WCS postage stamp cutout from imaging coadd at target RA/Dec
  D) RA/Dec mini-map + SkyMapper r/i/z photometry table
  E) 1D spectrum panel (prefer telluric-corrected spectrum; overplot photometry only for flux-calibrated view)

Patched for SAMOS Run-8 Step11 products:
- Fixes get_slit_hdu() availability (was defined after main).
- Robustly selects EVEN/ODD tracecoords/slitcoords files (if provided).
- Spyder-friendly defaults via parse_args() class Args block.
- Skips slits missing RA/Dec in photometry catalog (as before), but logs count.

Requirements:
  astropy, numpy, pandas, matplotlib

"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u

def _pe(p):
    if p is None:
        return "None"
    try:
        return f"{p}  (exists={Path(p).exists()})"
    except Exception:
        return str(p)



# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _norm_key(s: str) -> str:
    return s.strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def get_slit_hdu(hdul, slit: str):
    """Return the HDU for a given SLIT### even if EXTNAME is not exactly SLIT###.

    Tries:
      1) direct EXTNAME lookup (hdul['SLIT###'])
      2) match by header keyword SLITID (int)
      3) match by exact hd.name
    """
    s = str(slit).strip().upper()
    # direct extname
    try:
        return hdul[s]
    except Exception:
        pass

    m = re.match(r"SLIT\s*(\d+)", s)
    sid = int(m.group(1)) if m else None

    # match by SLITID keyword
    if sid is not None:
        for hd in hdul[1:]:
            try:
                if int(hd.header.get("SLITID", -999)) == sid:
                    return hd
            except Exception:
                continue

    # match by name
    for hd in hdul[1:]:
        try:
            if hd.name.strip().upper() == s:
                return hd
        except Exception:
            continue

    raise KeyError(f"Could not find HDU for {s} (no EXTNAME match and no SLITID match)")


def _first_existing(header: fits.Header, keys) -> Optional[str]:
    for k in keys:
        if k in header and str(header[k]).strip():
            return str(header[k]).strip()
    return None


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


def _vbin_vec(v: np.ndarray, fac: int) -> np.ndarray:
    """Vertical binning for 1D vectors: mean in contiguous blocks of length fac."""
    v = np.asarray(v, float)
    if fac <= 1 or v.size == 0:
        return v
    n2 = (v.size // fac) * fac
    if n2 <= 0:
        return v
    a = v[:n2].reshape(n2 // fac, fac)
    return np.nanmean(a, axis=1)


def _abmag_to_fnu_jy(m: float) -> float:
    return 3631.0 * (10.0 ** (-0.4 * m))


def _fnu_jy_to_flambda(fnu_jy: float, lam_nm: float) -> float:
    # Convert f_nu [Jy] to f_lambda [erg/s/cm^2/Angstrom]
    cA = 2.99792458e18  # Angstrom/s
    lam_A = lam_nm * 10.0
    fnu_cgs = fnu_jy * 1e-23
    return fnu_cgs * cA / (lam_A ** 2)


def _choose_columns(tab: fits.FITS_rec, prefer_flux: Optional[str], prefer_lam: Optional[str]) -> Tuple[str, str]:
    cols = [c.name for c in tab.columns]
    ncols = {_norm_key(c): c for c in cols}

    # wavelength
    if prefer_lam and prefer_lam in cols:
        lam_col = prefer_lam
    else:
        for cand in ["LAMBDA_NM", "lambda_nm", "LAMBDA", "WAVE_NM", "WAVELENGTH_NM"]:
            if _norm_key(cand) in ncols:
                lam_col = ncols[_norm_key(cand)]
                break
        else:
            raise KeyError(f"Could not find wavelength column in {cols}")

    # spectrum
    if prefer_flux and prefer_flux in cols:
        flux_col = prefer_flux
    else:
        for cand in ["FLUX_TELLCOR_O2", "FLUX_FLAM", "FLUX_CAL", "FLUX", "OBJ_SKYSUB", "OBJ_RAW"]:
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

    Returns
    -------
    lam : ndarray
        Wavelength array.
    flux : ndarray
        Flux array to display.
    ylabel : str
        Label identifying the displayed flux column.
    color : str
        Matplotlib line color.
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

    # Final Step11 product should be shown by default
    arr = finite_array("FLUX_FLAM")
    if arr is not None:
        return lam, arr, "FLUX_FLAM", "k"

    arr = finite_array("FLUX_TELLCOR_O2")
    if arr is not None:
        return lam, arr, "FLUX_TELLCOR_O2", "red"

    arr = finite_array("FLUX")
    if arr is not None:
        return lam, arr, "FLUX", "red"

    # last resort: any finite numeric column that is not wavelength or obvious variance/sky helpers
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
    ra_c   = pick("ra", "ra_deg", "raj2000", "ra2000")
    dec_c  = pick("dec", "dec_deg", "dej2000", "dec2000")
    r_c    = pick("r_mag", "rmag", "r", "sm_r", "r_psf", "r_petro")
    i_c    = pick("i_mag", "imag", "i", "sm_i", "i_psf", "i_petro")
    z_c    = pick("z_mag", "zmag", "z", "sm_z", "z_psf", "z_petro")

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
    """
    Create an axes for the imaging stamp.
    Tries WCSAxes (projection=stamp_wcs). If the environment has an older Matplotlib/Astropy
    combo missing AnchoredEllipse, fall back to a normal pixel axes.
    """
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
        # Keep the error string in case you want to log it
        ax.text(0.01, 0.99, f"WCSAxes unavailable:\n{type(e).__name__}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", alpha=0.15))
        return ax, "PIX"

def _read_slit_image_and_header(mef_path: Optional[Path], slit: str) -> Tuple[Optional[np.ndarray], Optional[fits.Header]]:
    """Like _read_slit_image, but also returns the HDU header (for XWIN/YMIN mapping)."""
    if mef_path is None:
        return None, None
    mp = Path(mef_path)
    if not mp.exists():
        return None, None

    h = _open_cached(mp)
    # MEF: per-slit EXTNAMEs
    if len(h) > 1:
        try:
            hd = get_slit_hdu(h, slit)
        except Exception:
            return None, None
        d = hd.data
        if d is None:
            return None, None
        # If mask file uses PRIMARY cube, fall back to _read_slit_image logic
        if ("mask" in mp.name.lower()) and getattr(d, "ndim", 0) != 2:
            d2 = _read_slit_image(mp, slit)
            return (np.asarray(d2, float) if d2 is not None else None), hd.header
        if getattr(d, "ndim", 0) != 2:
            return None, hd.header
        return np.asarray(d, float), hd.header

    # Single-HDU
    d0 = h[0].data
    if d0 is None:
        return None, h[0].header
    if getattr(d0, "ndim", 0) == 2:
        return np.asarray(d0, float), h[0].header
    if getattr(d0, "ndim", 0) == 3:
        return np.asarray(d0[0, :, :], float), h[0].header
    return None, h[0].header


def _read_slit_image(mef_path: Optional[Path], slit: str) -> Optional[np.ndarray]:
    """
    Read a 2D image for a given slit from a FITS file.

    Supports:
      (1) MEF with per-slit EXTNAMEs (SLIT###)  -> returns that extension's 2D image.
      (2) Single-HDU FITS where PRIMARY is:
            - 2D full-frame mask/image -> returns the 2D array (cannot be slit-specific).
            - 3D cube (nslit, ny, nx) -> selects a plane based on slit number and file parity (Even/Odd).
    """
    if mef_path is None:
        return None
    mp = Path(mef_path)
    if not mp.exists():
        return None

    # parse slit id
    m = re.match(r"SLIT(\d+)", str(slit).strip().upper())
    sid = int(m.group(1)) if m else None

    h = _open_cached(mp)
    # If this is a proper MEF, try to find the slit extension
    if len(h) > 1:
        try:
            d = get_slit_hdu(h, slit).data
        except Exception:
            # Fallback: match by digits in EXTNAME
            dig = f"{sid:03d}" if sid is not None else None
            d = None
            if dig:
                for hd in h[1:]:
                    nm = (hd.name or "").upper()
                    if dig in nm and hd.data is not None and getattr(hd.data, "ndim", 0) == 2:
                        d = hd.data
                        break
            if d is None:
                return None
        if d is None or getattr(d, "ndim", 0) != 2:
            return None
        return np.asarray(d, float)

    # Single-HDU FITS: use PRIMARY
    d0 = h[0].data
    if d0 is None:
        return None

    # If this looks like a mask file, try to extract a single-slit plane when possible
    if "mask" in mp.name.lower():
        plane = _pick_mask_plane(np.asarray(d0), slit, mp.name)
        if plane is None:
            return None
        if getattr(plane, "ndim", 0) != 2:
            return None
        return np.asarray(plane, float)

    # Otherwise return PRIMARY as-is (2D) or first plane (3D)
    if getattr(d0, "ndim", 0) == 2:
        return np.asarray(d0, float)
    if getattr(d0, "ndim", 0) == 3:
        return np.asarray(d0[0, :, :], float)

    return None


def _pick_mask_plane(mask_data: np.ndarray, slit: str, filename: str) -> np.ndarray:
    """
    Given mask_data from PRIMARY (2D or 3D), return a *single-slit* 2D mask if possible.
    - If 2D: return as-is (may be full-frame union mask).
    - If 3D: detect which axis is the slit axis (typically ~31/32 planes) and index it using slit number + parity.
    """
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

    # Identify slit-axis: choose axis whose length is "small" (<=80) and >1
    shape = mask_data.shape
    cand_axes = [ax for ax, n in enumerate(shape) if 1 < n <= 80]
    slit_ax = cand_axes[0] if cand_axes else 0

    nslit = shape[slit_ax]
    name = filename.lower()

    # Compute index within EVEN or ODD set if file name indicates parity
    idx = None
    if "even" in name and sid % 2 == 0:
        idx = sid // 2 - 1
    elif "odd" in name and sid % 2 == 1:
        idx = (sid + 1) // 2 - 1

    # Fallback: if nslit is large enough, assume global indexing sid-1
    if idx is None and nslit >= sid:
        idx = sid - 1
    if idx is None:
        idx = 0
    idx = int(np.clip(idx, 0, nslit - 1))

    # Slice along slit axis to get 2D
    if slit_ax == 0:
        return mask_data[idx, :, :]
    if slit_ax == 1:
        return mask_data[:, idx, :]
    return mask_data[:, :, idx]


def _mask_to_target(mask2d: np.ndarray, target_shape: tuple[int,int], pad: int = 5) -> np.ndarray:
    """
    Convert mask2d into an array with target_shape suitable for contour overlay.
    Strategy:
      - If already matches, return.
      - Else crop around nonzero bbox (pad pixels).
      - If still not matching, nearest-neighbor resample to target_shape.
    """
    if mask2d is None:
        return None
    m = (np.asarray(mask2d) > 0).astype(float)
    ty, tx = target_shape
    if m.shape == (ty, tx):
        return m

    # Crop around bbox if possible
    ys, xs = np.where(m > 0.5)
    if ys.size > 0:
        y0, y1 = max(0, ys.min() - pad), min(m.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(m.shape[1], xs.max() + pad + 1)
        m = m[y0:y1, x0:x1]

    if m.shape == (ty, tx):
        return m

    # Nearest-neighbor resample
    yi = (np.linspace(0, m.shape[0] - 1, ty)).astype(int)
    xi = (np.linspace(0, m.shape[1] - 1, tx)).astype(int)
    return m[np.ix_(yi, xi)]

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


def _split_even_odd(path_str: Optional[str]) -> tuple[Optional[Path], Optional[Path]]:
    """
    Allow passing 'even_path|odd_path' (or ';' or ',') for MEFs that are split by parity.
    Returns (even, odd).
    If only one path provided, returns (path, path).
    """
    if not path_str:
        return None, None
    s = str(path_str)
    for sep in ["|", ";"]:
        if sep in s:
            a, b = [p.strip() for p in s.split(sep, 1)]
            return (Path(a) if a else None), (Path(b) if b else None)
    # comma can appear in lists; only treat as sep if two existing files
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




def _slit_bbox_from_geom_using_mask(mask2d: np.ndarray, geom_path: Optional[Path], slit: str,
                                   search_halfwidth: int = 300, pad: int = 12):
    """
    Derive a detector-frame bbox for a slit when:
      - mask2d is a FULL-FRAME (ny,nx) union mask (all slits), and
      - geom_path contains per-slit 1D vectors (length ny) giving x_center(y) or similar.

    This matches your geometry file structure:
      HDU 'SLIT###' data shape (ny,) with ny~4112.

    Algorithm (per detector row y):
      - take x_center = geom[y]
      - within window x_center ± search_halfwidth, find mask segments (connected runs)
      - pick the segment whose midpoint is closest to x_center
      - accumulate min/max of the chosen segment; track y range where chosen segment exists
    """
    if mask2d is None or geom_path is None:
        return None
    gp = Path(geom_path)
    if not gp.exists():
        return None

    ny, nx = mask2d.shape
    slit = str(slit).strip().upper()

    try:
        with fits.open(gp) as h:
            hd = get_slit_hdu(h, slit)
            g = hd.data
    except Exception:
        return None

    if g is None:
        return None
    g = np.asarray(g, float)

    if g.ndim == 1 and g.shape[0] == ny:
        xs_min, xs_max, ys_hit = [], [], []

        for y in range(ny):
            xc = g[y]
            if not np.isfinite(xc):
                continue
            xci = int(round(float(xc)))
            x0 = max(0, xci - search_halfwidth)
            x1 = min(nx, xci + search_halfwidth + 1)

            row = mask2d[y, x0:x1]
            idx = np.where(row > 0)[0]
            if idx.size == 0:
                continue

            # split into connected segments
            breaks = np.where(np.diff(idx) > 1)[0]
            starts = np.r_[0, breaks + 1]
            ends   = np.r_[breaks, idx.size - 1]

            # choose segment whose midpoint is closest to x_center
            best = None
            best_dist = None
            for s,e in zip(starts, ends):
                seg = idx[s:e+1] + x0
                mid = 0.5 * (seg[0] + seg[-1])
                dist = abs(mid - xci)
                if (best_dist is None) or (dist < best_dist):
                    best_dist = dist
                    best = seg

            if best is None:
                continue

            xs_min.append(int(best[0]))
            xs_max.append(int(best[-1]))
            ys_hit.append(int(y))

        if ys_hit:
            y0 = min(ys_hit); y1 = max(ys_hit)
            x0 = int(min(xs_min)); x1 = int(max(xs_max))
        else:
            # No mask pixels found near the trace: fall back to bbox from finite geom range
            ys = np.where(np.isfinite(g))[0]
            if ys.size == 0:
                return None
            y0, y1 = int(ys.min()), int(ys.max())
            xc_med = float(np.nanmedian(g[ys]))
            x0 = int(round(xc_med - 50))
            x1 = int(round(xc_med + 50))

        # pad and clip; y1/x1 inclusive
        y0 = max(0, y0 - pad); y1 = min(ny - 1, y1 + pad)
        x0 = max(0, x0 - pad); x1 = min(nx - 1, x1 + pad)
        return (y0, y1, x0, x1)

    return None


def _cutout_bbox(img2d: np.ndarray, bbox, pad: int = 0):
    """Cut a detector-frame bbox; bbox=(y0,y1,x0,x1) with y1/x1 inclusive."""
    if img2d is None or bbox is None:
        return None
    y0,y1,x0,x1 = bbox
    ny, nx = img2d.shape
    y0 = max(0, int(y0) - pad); x0 = max(0, int(x0) - pad)
    y1 = min(ny - 1, int(y1) + pad); x1 = min(nx - 1, int(x1) + pad)
    if y1 <= y0 or x1 <= x0:
        return None
    return img2d[y0:y1+1, x0:x1+1]



def _object_ridge_constrained(cut: np.ndarray, geom1d: np.ndarray, X0: int, Y0: int,
                             ridge_halfwidth: int = 6) -> np.ndarray:
    """Measure an object ridge in a detector cutout, constrained near the slit-center ridge.

    For each cutout row j (detector y = Y0+j):
      - take expected center x0 = geom1d[y_det] (detector coords)
      - search only within [x0-ridge_halfwidth, x0+ridge_halfwidth] in cutout coords
      - pick argmax within that corridor

    This prevents the ridge from jumping to a nearby object when the cutout is wide.
    """
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
        x0 = x0_det - X0  # expected center in cutout coords

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


def _plot_detector_cutout_with_ridge(ax, sci2d: np.ndarray, hdr_trace: fits.Header,
                                    geom1d: np.ndarray, stretch_sigma: float = 5.0,
                                    pad_x: int = 6, pad_y: int = 20,
                                    show_obj_ridge: bool = True,
                                    ridge_halfwidth: int = 6):
    """
    Plot detector-frame cutout for a slit using bbox keywords from TRACECOORDS header
    (XLO/XHI/YMIN/YMAX), and overlay the Step04 geometry ridge (x_center(y)).
    Optionally overlay an "object ridge" measured from the detector cutout, *constrained*
    to stay near the slit-center ridge.

    Returns bbox dict or None.
    """
    if sci2d is None or hdr_trace is None or geom1d is None:
        return None

    for k in ("XLO", "XHI", "YMIN", "YMAX"):
        if k not in hdr_trace:
            return None

    xlo = int(hdr_trace["XLO"])
    xhi = int(hdr_trace["XHI"])  # exclusive
    ymin = int(hdr_trace["YMIN"])
    ymax = int(hdr_trace["YMAX"])

    ny_det, nx_det = sci2d.shape

    # keep the cutout tight in x; wide cutouts encourage the ridge to lock onto neighbors
    X0 = max(0, xlo - pad_x)
    X1 = min(nx_det, xhi + pad_x)
    Y0 = max(0, ymin - pad_y)
    Y1 = min(ny_det, ymax + pad_y)

    if (X1 <= X0 + 2) or (Y1 <= Y0 + 2):
        return None

    cut = np.asarray(sci2d[Y0:Y1, X0:X1], float)

    vmin, vmax = _robust_limits(cut, float(stretch_sigma))
    ax.imshow(cut, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)

    # ridge from geometry in detector coords -> cutout coords
    y_det = np.arange(Y0, Y1)
    ok = (y_det >= 0) & (y_det < geom1d.size) & np.isfinite(geom1d[y_det])
    if np.any(ok):
        x_det = geom1d[y_det[ok]]
        y_det_ok = y_det[ok]
        ax.plot(x_det - X0, y_det_ok - Y0, lw=1.2, label="slit center (Step04)")

    # object ridge from data, constrained near slit center
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
    """Extract a representative scalar value from a (possibly per-row) column."""
    if tab is None or col is None:
        return float(default)
    if col not in tab.columns.names:
        return float(default)
    v = np.asarray(tab[col], float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float(default)
    return float(np.nanmedian(v))


def _plot_rectified_with_apertures(ax, img_rect: np.ndarray, tab: fits.FITS_rec,
                                  vbin: int = 6, stretch_sigma: float = 5.0,
                                  ridge_halfwidth: int = 6, sky_gap: int = 2,
                                  show_obj_ridge: bool = True):
    """Plot a rectified slit image and annotate object/sky regions used for extraction.

    In rectified products, the spectrum should be approximately vertical. The most useful
    diagnostics here are:
      - measured object ridge (from the rectified image itself)
      - object aperture width (NOBJ) around the ridge
      - sky regions (NSKY) on both sides with a configurable gap
    """
    if img_rect is None:
        ax.text(0.5, 0.5, "Rectified image not available", ha="center", va="center")
        ax.set_axis_off()
        return

    im = _vbin(np.asarray(img_rect, float), int(vbin))
    vmin, vmax = _robust_limits(im, float(stretch_sigma))
    ax.imshow(im, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)

    ny, nx = im.shape

    # Prefer X0/NOBJ/NSKY from the extraction table if present
    x0 = _scalar_from_tab(tab, "X0", default=np.nan)
    nobj = _scalar_from_tab(tab, "NOBJ", default=np.nan)
    nsky = _scalar_from_tab(tab, "NSKY", default=np.nan)

    # Measured ridge in rectified coords (constrained near x0 if available)
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

            # If X0 missing, define a representative center from the ridge
            if not np.isfinite(x0):
                x0 = float(np.nanmedian(x_ridge[ok]))

    # Draw extraction regions following the *object ridge* as a function of y.
    # This works in both cases:
    #   - perfectly rectified spectrum (ridge ~ constant) -> bands appear vertical
    #   - residual tilt / off-centering / imperfect rectification -> bands are curved
    if x_ridge is None:
        # fall back to constant centerline from X0 (or image midpoint)
        x0_use = x0 if np.isfinite(x0) else 0.5 * (nx - 1)
        x_ridge = np.full(ny, float(x0_use), float)

    yy = np.arange(ny, dtype=float)

    # If we did not plot ridge above (e.g. show_obj_ridge=False), plot it here for context.
    okr = np.isfinite(x_ridge)
    if np.any(okr) and not show_obj_ridge:
        ax.plot(x_ridge[okr], yy[okr], lw=1.0, label="object ridge")

    # object aperture boundaries
    half = 0.0
    if np.isfinite(nobj) and nobj > 0:
        half = 0.5 * float(nobj)
        ax.plot((x_ridge - half)[okr], yy[okr], lw=0.9, ls="--", label="obj aperture")
        ax.plot((x_ridge + half)[okr], yy[okr], lw=0.9, ls="--")

    # sky regions
    if np.isfinite(nsky) and nsky > 0:
        g = float(sky_gap)
        nskyf = float(nsky)
        # left sky bounds
        ax.plot((x_ridge - (half + g))[okr], yy[okr], lw=0.8, ls=":", label="sky regions")
        ax.plot((x_ridge - (half + g + nskyf))[okr], yy[okr], lw=0.8, ls=":")
        # right sky bounds
        ax.plot((x_ridge + (half + g))[okr], yy[okr], lw=0.8, ls=":")
        ax.plot((x_ridge + (half + g + nskyf))[okr], yy[okr], lw=0.8, ls=":")

    ax.set_xlabel("x (rectified spatial pix)")
    ax.set_ylabel(f"y (binned rows, vbin={int(vbin)})")
    ax.legend(loc="upper right", fontsize=8)


def parse_args():
    class Args:
        extract = r"/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_RUN8_Science_2026_01/SAMI/20260113/reduced/11_fluxcal/Extract1D_fluxcal.fits"
        photcat = r"/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_RUN8_Science_2026_01/SAMI/20260113/reduced/11_fluxcal/slit_trace_RA_match_skymapper_ALL.csv"
        wcs     = r"/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_RUN8_Science_2026_01/SISI/SISI_2026-01-13/reduced/05_wcs/Coadd_i_median_078-082_ff_flipx_wcs_manual.fits"
        detector = r"/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_RUN8_Science_2026_01/SAMI/20260113/reduced/06_science/FinalScience_dolidze_ADUperS.fits"
        #detector = r"/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_RUN8_Science_2026_01/SAMI/20260113/reduced/07_wavecal/ArcDiff_036.arc_biascorr_cr_rev_minus_037.arc_biascorr_cr_rev_pixflatcorr_clipped.fits"

        # You can pass a single MEF, or "EVEN_path|ODD_path"
        tracecoords = (
            r"/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_RUN8_Science_2026_01/SAMI/20260113/reduced/06_science/FinalScience_dolidze_ADUperS_reg_pixflatcorr_clipped_EVEN_tracecoords.fits"
            + "|"
            + r"/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_RUN8_Science_2026_01/SAMI/20260113/reduced/06_science/FinalScience_dolidze_ADUperS_reg_pixflatcorr_clipped_ODD_tracecoords.fits"
        )
        slitcoords  = (
            r"/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_RUN8_Science_2026_01/SAMI/20260113/reduced/06_science/FinalScience_dolidze_ADUperS_reg_pixflatcorr_clipped_EVEN_slitcoords.fits"
            + "|"
            + r"/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_RUN8_Science_2026_01/SAMI/20260113/reduced/06_science/FinalScience_dolidze_ADUperS_reg_pixflatcorr_clipped_ODD_slitcoords.fits"
        )

        # Binary trace masks from Step04 pixflat (you have both *_mask.fits and *_mask_reg.fits).
        # Use the *_reg versions if you want masks in the same registered frame as the science tracecoords.
        mask = (
            r"/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_RUN8_Science_2026_01/SAMI/20260113/reduced/04_PIXFLAT/Even_traces_mask_reg.fits"
            + "|"
            + r"/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_RUN8_Science_2026_01/SAMI/20260113/reduced/04_PIXFLAT/Odd_traces_mask_reg.fits"
        )

        # Optional geometry products (if you want to inspect them as images)
        geometry = (
            r"/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_RUN8_Science_2026_01/SAMI/20260113/reduced/04_PIXFLAT/Even_traces_geometry.fits"
            + "|"
            + r"/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_RUN8_Science_2026_01/SAMI/20260113/reduced/04_PIXFLAT/Odd_traces_geometry.fits"
        )

        # How to display masks:
        #   "overlay"  = contour mask on TRACECOORDS panel (recommended)
        #   "panel"    = show mask as its own panel (replaces SLITCOORDS panel)
        #   "geometry" = show geometry as its own panel (replaces SLITCOORDS panel)
        mask_mode = "rectified"  # "rectified" shows extraction geometry on rectified slit image; use "panel" for binary masks
        mask_crop = "match_trace"  # "bbox" or "match_trace"
        show_edges = True
        search_halfwidth = 120
        mask_pad = 6


        out = r"/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_RUN8_Science_2026_01/SAMI/20260113/reduced/11_fluxcal/QC_summary_Run8.pdf"

        slits = None          # e.g. "SLIT001,SLIT010" or "1,10,24"
        cutout = 35
        vbin = 6
        stretch = 5.0
        det_pad_x = 6
        det_pad_y = 20
        det_ridge_halfwidth = 6
        rect_ridge_halfwidth = 6
        sky_gap = 2
        lambda_r = 616.0
        lambda_i = 779.0
        lambda_z = 916.0
        spec_col = "FLUX_TELLCOR_O2"   # default: telluric-corrected spectrum
        lam_col  = "LAMBDA_NM"
        quiet = False

    return Args()


def _segment_edges_near_xc(mask_row: np.ndarray, xc: int, x0: int) -> tuple[int,int] | None:
    """
    Given a boolean mask_row (already sliced), find connected segments and return (xL,xR)
    in full-frame coordinates for the segment whose midpoint is closest to xc.
    """
    idx = np.where(mask_row > 0)[0]
    if idx.size == 0:
        return None
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, breaks + 1]
    ends   = np.r_[breaks, idx.size - 1]
    best = None
    best_dist = None
    for s,e in zip(starts, ends):
        seg = idx[s:e+1] + x0
        mid = 0.5 * (seg[0] + seg[-1])
        dist = abs(mid - xc)
        if (best_dist is None) or (dist < best_dist):
            best_dist = dist
            best = (int(seg[0]), int(seg[-1]))
    return best


def _bbox_and_edges_from_fullmask(mask2d: np.ndarray, geom1d: np.ndarray,
                                 search_halfwidth: int = 120, pad: int = 6):
    """
    Use geom1d ~ x_center(y) to pick the nearest mask segment per row and return:
      bbox (y0,y1,x0,x1) inclusive
      edges arrays in detector coords (y, xL, xR) for rows with hits
      ridge array (y, xC) where xC from geom1d (finite)
    """
    ny, nx = mask2d.shape
    xsL, xsR, ys = [], [], []
    for y in range(ny):
        xc = geom1d[y]
        if not np.isfinite(xc):
            continue
        xci = int(round(float(xc)))
        a0 = max(0, xci - search_halfwidth)
        a1 = min(nx, xci + search_halfwidth + 1)
        seg = _segment_edges_near_xc(mask2d[y, a0:a1], xci, a0)
        if seg is None:
            continue
        xL, xR = seg
        xsL.append(xL); xsR.append(xR); ys.append(y)

    if len(ys) == 0:
        return None, None, None

    y0 = max(0, min(ys) - pad); y1 = min(ny - 1, max(ys) + pad)
    x0 = max(0, int(min(xsL)) - pad); x1 = min(nx - 1, int(max(xsR)) + pad)

    edges = (np.asarray(ys, int), np.asarray(xsL, int), np.asarray(xsR, int))
    ridge = (np.where(np.isfinite(geom1d))[0].astype(int),
             np.asarray(geom1d[np.isfinite(geom1d)], float))
    return (y0,y1,x0,x1), edges, ridge




# -----------------------------------------------------------------------------
# Lightweight FITS cache for speed
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
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    extract_fits = Path(args.extract)
    photcat_csv  = Path(args.photcat)
    wcs_fits     = Path(args.wcs)
    detector_fits = Path(getattr(args, 'detector', ''))
    out_pdf      = Path(args.out)

    if not extract_fits.exists():
        raise FileNotFoundError(extract_fits)
    if not photcat_csv.exists():
        raise FileNotFoundError(photcat_csv)
    if not wcs_fits.exists():
        raise FileNotFoundError(wcs_fits)
    if detector_fits and str(detector_fits).strip() and (not detector_fits.exists()):
        raise FileNotFoundError(detector_fits)

    # photometry lookup
    phot = _read_photcat(photcat_csv)
    phot_idx = {r.SLIT: r for r in phot.itertuples(index=False)}

    # Read all RA/Dec for mini-map
    ra_all = phot["RA"].to_numpy(float)
    dec_all = phot["DEC"].to_numpy(float)

    # Split even/odd image MEFs if provided
    trace_even, trace_odd = _split_even_odd(getattr(args, "tracecoords", None))
    slit_even, slit_odd = _split_even_odd(getattr(args, "slitcoords", None))
    mask_even, mask_odd = _split_even_odd(getattr(args, "mask", None))
    geom_even, geom_odd = _split_even_odd(getattr(args, "geometry", None))

    # detector-frame science image (PrimaryHDU)
    sci_full = None
    if detector_fits and str(detector_fits).strip():
        with fits.open(detector_fits) as _hs:
            if _hs[0].data is None:
                raise RuntimeError(f"Detector science file has no PRIMARY image: {detector_fits}")
            sci_full = np.asarray(_hs[0].data, float)
            if sci_full.ndim != 2:
                # allow a cube, take first plane
                sci_full = np.asarray(sci_full[0], float)


    with fits.open(extract_fits) as h:
        # list slits in extract file
        slits = []
        for hd in h[1:]:
            name = (hd.name or "").strip().upper()
            if name.startswith("SLIT"):
                slits.append(name)
        # de-dup and sort by numeric
        seen = set()
        slits = [s for s in slits if not (s in seen or seen.add(s))]
        slits = sorted(slits, key=lambda s: int("".join(ch for ch in s if ch.isdigit()) or 0))

        if not getattr(args, "quiet", False):
            print(f"Found {len(slits)} SLIT extensions in extract file.")
            print("EXTRACT =", extract_fits)
            print("TRACECOORDS EVEN/ODD =", _pe(trace_even), _pe(trace_odd))
            print("SLITCOORDS  EVEN/ODD =", _pe(slit_even), _pe(slit_odd))
            print("MASK       EVEN/ODD =", _pe(mask_even), _pe(mask_odd))
            print("GEOMETRY   EVEN/ODD =", _pe(geom_even), _pe(geom_odd))
            print("WCS IMG =", wcs_fits)

        raw_slits = slits
        slits = _parse_slit_list(getattr(args, "slits", None), raw_slits)

        from matplotlib.backends.backend_pdf import PdfPages
        out_pdf.parent.mkdir(parents=True, exist_ok=True)

        n_written = 0
        n_skipped_nophot = 0

        with PdfPages(out_pdf) as pdf:
            for slit in slits:
                # get slit HDU once, and trust its header first
                hd_spec = get_slit_hdu(h, slit)
                hdr_spec = hd_spec.header
                tab = hd_spec.data

                # photometry row is optional for position; only needed for mags
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

                # calibrated spectrum                tab = get_slit_hdu(h, slit).data
                lam_col, flux_col = _choose_columns(tab, getattr(args, "spec_col", None), getattr(args, "lam_col", None))
                lam = np.asarray(tab[lam_col], float)
                flux = np.asarray(tab[flux_col], float)

                # trace/slit images (pick even/odd)
                trace_path = _pick_mef_for_slit(slit, trace_even, trace_odd)
                slit_path  = _pick_mef_for_slit(slit, slit_even, slit_odd)
                mask_path  = _pick_mef_for_slit(slit, mask_even, mask_odd)
                geom_path  = _pick_mef_for_slit(slit, geom_even, geom_odd)

                img_trace, hdr_trace = _read_slit_image_and_header(trace_path, slit)
                img_slit,  hdr_slit  = _read_slit_image_and_header(slit_path, slit)
                img_mask  = _read_slit_image(mask_path, slit)
                img_geom  = _read_slit_image(geom_path, slit)
                mask_fullframe = (img_mask is not None and img_mask.shape[0] > 2000 and img_mask.shape[1] > 2000)

                # postage stamp
                stamp, stamp_wcs = _cutout_from_wcs(wcs_fits, ra, dec, int(args.cutout))

                # ----- Figure layout -----
                fig = plt.figure(figsize=(11, 8.5))  # landscape letter
                gs = fig.add_gridspec(
                    2, 4,
                    height_ratios=[1.05, 1.0],
                    width_ratios=[1.1, 1.1, 0.85, 1.0],
                    hspace=0.28, wspace=0.25
                )
                # A1: detector-frame science cutout + ridge overlays
                axA1 = fig.add_subplot(gs[0, 0])

                geom1d = None
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
                        pad_x=int(getattr(args,'det_pad_x',6)),
                        pad_y=int(getattr(args,'det_pad_y',20)),
                        show_obj_ridge=True,
                        ridge_halfwidth=int(getattr(args,'det_ridge_halfwidth',6)),
                    )
                    axA1.set_title(f"{slit} DETECTOR CUTOUT + RIDGES")
                    if bbox is not None:
                        axA1.text(
                            0.01, 0.01,
                            f"DET bbox: X[{bbox['X0']}:{bbox['X1']})  Y[{bbox['Y0']}:{bbox['Y1']})\n"f"TRC bbox: X[{bbox['xlo']}:{bbox['xhi']})  Y[{bbox['ymin']}:{bbox['ymax']}]",
                            transform=axA1.transAxes, va="bottom", ha="left",
                            fontsize=8, family="monospace",
                            bbox=dict(boxstyle="round,pad=0.25", alpha=0.15)
                        )
                else:
                    msg = []
                    if sci_full is None:
                        msg.append("Detector science not loaded")
                    if hdr_trace is None:
                        msg.append("TRACECOORDS header missing")
                    if geom1d is None:
                        msg.append("Geometry 1D missing")
                    axA1.text(0.5, 0.5, "\n".join(msg) if msg else "Detector panel unavailable",
                              ha="center", va="center")
                    axA1.set_axis_off()

                # A2: SLITCOORDS or MASK panel                
                axA2 = fig.add_subplot(gs[0, 1])
                mode = getattr(args, "mask_mode", "overlay").lower()
                if mode == "rectified":
                    # Rectified slit image diagnostic: show measured object ridge and object/sky extraction regions
                    # Prefer SLITCOORDS; fall back to TRACECOORDS.
                    img_rect = img_slit if img_slit is not None else img_trace
                    _plot_rectified_with_apertures(
                        axA2,
                        img_rect,
                        tab,
                        vbin=int(getattr(args, "vbin", 6)),
                        stretch_sigma=float(getattr(args, "stretch", 5.0)),
                        ridge_halfwidth=int(getattr(args, "rect_ridge_halfwidth", 6)),
                        sky_gap=int(getattr(args, "sky_gap", 2)),
                        show_obj_ridge=True,
                    )
                    axA2.set_title(f"{slit} RECTIFIED + EXTRACTION REGIONS")
                elif mode == "panel":
                    if img_mask is not None:
                        if mask_fullframe:
                            # Mask is detector-frame full image: cut out the slit region using geometry bbox
                            _hg = _open_cached(Path(geom_path))
                            _g1d = get_slit_hdu(_hg, slit).data
                            if _g1d is None:
                                bbox = None; edges=None; ridge=None
                            else:
                                bbox, edges, ridge = _bbox_and_edges_from_fullmask(np.asarray(img_mask, float), np.asarray(_g1d, float),
                                                                                   search_halfwidth=int(getattr(args,'search_halfwidth',120)),
                                                                                   pad=int(getattr(args,'mask_pad',6)))
                            # Optionally match the TRACECOORDS panel width for side-by-side comparison
                            if bbox is not None and getattr(args,'mask_crop','bbox') == 'match_trace' and img_trace is not None:
                                im_trace_v = _vbin(img_trace, int(args.vbin))
                                tw = im_trace_v.shape[1]
                                # center on median ridge (geom1d) within bbox y-range
                                y0,y1,x0,x1 = bbox
                                if _g1d is not None:
                                    ys = np.arange(y0, y1+1)
                                    xc_med = float(np.nanmedian(np.asarray(_g1d, float)[ys]))
                                else:
                                    xc_med = 0.5*(x0+x1)
                                xc = int(round(xc_med))
                                x0n = max(0, xc - tw//2)
                                x1n = min(np.asarray(img_mask).shape[1]-1, x0n + tw - 1)
                                bbox = (y0,y1,x0n,x1n)
                            cut = _cutout_bbox(np.asarray(img_mask, float), bbox, pad=0)
                            if cut is None:
                                axA2.text(0.5, 0.5, "MASK is full-frame; could not derive bbox from geometry 1D vectors", ha="center", va="center")
                                axA2.set_axis_off()
                            else:
                                axA2.imshow((cut > 0).astype(float), origin="lower", aspect="auto", vmin=0, vmax=1)
                                axA2.set_title(f"{slit} MASK CUTOUT (detector frame)")
                                axA2.set_xlabel("x (detector pix)")
                                axA2.set_ylabel("y (detector pix)")
                                if getattr(args,'show_edges', True) and edges is not None and bbox is not None:
                                    y0,y1,x0,x1 = bbox
                                    ys_e, xL, xR = edges
                                    msel = (ys_e>=y0) & (ys_e<=y1)
                                    axA2.plot(xL[msel]-x0, ys_e[msel]-y0, lw=0.8)
                                    axA2.plot(xR[msel]-x0, ys_e[msel]-y0, lw=0.8)
                                # ridge line from geometry (x_center)
                                if ridge is not None and bbox is not None:
                                    y0,y1,x0,x1 = bbox
                                    yr, xr = ridge
                                    msel = (yr>=y0) & (yr<=y1)
                                    axA2.plot(xr[msel]-x0, yr[msel]-y0, lw=0.8)
                        else:
                            m = _vbin(np.asarray(img_mask, float), int(args.vbin))
                            axA2.imshow((m > 0).astype(float), origin="lower", aspect="auto", vmin=0, vmax=1)
                            axA2.set_title(f"{slit} BINARY MASK (vbin={args.vbin})")
                        axA2.set_xlabel("x (spatial pix)")
                        axA2.set_ylabel("y (binned rows)")
                    else:
                        axA2.text(0.5, 0.5, "MASK not available", ha="center", va="center")
                        axA2.set_axis_off()
                elif mode == "geometry":
                    if img_geom is not None:
                        gg = np.asarray(img_geom, float)
                        if gg.ndim == 1:
                            y = np.arange(gg.size)
                            okg = np.isfinite(gg)
                            axA2.plot(gg[okg], y[okg], lw=1.0)
                            axA2.invert_yaxis()
                            axA2.set_title(f"{slit} GEOMETRY x_center(y)")
                            axA2.set_xlabel("x_center (detector pix)")
                            axA2.set_ylabel("y (detector pix)")
                        else:
                            g = _vbin(gg, int(args.vbin))
                            vmin, vmax = _robust_limits(g, float(args.stretch))
                            axA2.imshow(g, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
                            axA2.set_title(f"{slit} GEOMETRY (vbin={args.vbin})")
                            axA2.set_xlabel("x")
                            axA2.set_ylabel("y")
                    else:
                        axA2.text(0.5, 0.5, "GEOMETRY not available", ha="center", va="center")
                        axA2.set_axis_off()
                else:
                    if img_slit is not None:
                        im2 = _vbin(img_slit, int(args.vbin))
                        vmin, vmax = _robust_limits(im2, float(args.stretch))
                        axA2.imshow(im2, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
                        axA2.set_title(f"{slit} SLITCOORDS (vbin={args.vbin})")
                        axA2.set_xlabel("x (spatial pix)")
                        axA2.set_ylabel("y (binned rows)")
                    else:
                        axA2.text(0.5, 0.5, "SLITCOORDS not available", ha="center", va="center")
                        axA2.set_axis_off()
                """        
                # D: RA/Dec mini-map + photometry
                axD = fig.add_subplot(gs[0, 2])
                
                # Reflect RA around target so that East is left, with tick labels increasing leftward
                ra0 = ra
                ra_all_plot = 2*ra0 - ra_all
                ra_plot = 2*ra0 - ra
                axD.scatter(ra_all_plot, dec_all, s=6, alpha=0.35)
                axD.scatter([ra_plot], [dec], s=60, marker="*", linewidths=0.8)
                axD.set_xlabel("RA (deg)")
                axD.set_ylabel("Dec (deg)")
                axD.set_title("SkyMapper positions")
                axD.grid(True, alpha=0.2)

                # Make the mini-map square in degrees and use astronomical convention:
                # RA increases to the left. Do this AFTER plotting and any limit changes.
                axD.set_aspect('equal', adjustable='box')
                x0, x1 = axD.get_xlim()
                axD.set_xlim(max(x0, x1), min(x0, x1))
                """
                # D: RA/Dec mini-map + photometry
                axD = fig.add_subplot(gs[0, 2])

                # Plot literal sky coordinates
                axD.scatter(ra_all, dec_all, s=6, alpha=0.35)
                axD.scatter([ra], [dec], s=60, marker="*", linewidths=0.8)

                axD.set_xlabel("RA (deg)")
                axD.set_ylabel("Dec (deg)")
                axD.set_title("SkyMapper positions")
                axD.grid(True, alpha=0.2)
                axD.set_aspect("equal", adjustable="box")
                axD.text(ra, dec, slit, fontsize=7)
                
                # Astronomical convention: RA increases to the left
                axD.invert_xaxis()
                
                
                
                
                # photometry table as anchored text (under title area)
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
                axD.text(0.02, 0.02, txt, transform=axD.transAxes, va="bottom", ha="left",
                         fontsize=9, family="monospace",
                         bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))

                # B: postage stamp
                vmin, vmax = _robust_limits(stamp, float(args.stretch))
                axB, stamp_mode = _add_stamp_axes(fig, gs[0, 3], stamp, stamp_wcs, "Imaging cutout")
                axB.images[-1].set_clim(vmin, vmax)
                # mark the center pixel of the cutout
                axB.scatter([args.cutout], [args.cutout], marker="+", s=80)
                
                axB.text(
                    0.02, 0.02,
                    f"center = ({ra:.6f}, {dec:.6f})",
                    transform=axB.transAxes,
                    va="bottom", ha="left",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", alpha=0.15)
                )

                # C: spectrum + photometry points
                axC = fig.add_subplot(gs[1, :])

                lam_disp, flux_disp, ylab, line_color = _choose_display_spectrum(tab, lam_col)
                ok = np.isfinite(lam_disp) & np.isfinite(flux_disp)
                if ok.sum() > 0:
                    axC.plot(lam_disp[ok], flux_disp[ok], lw=1.0, color=line_color)

                lam_r, lam_i, lam_z = float(args.lambda_r), float(args.lambda_i), float(args.lambda_z)
                lam_map = {"r": lam_r, "i": lam_i, "z": lam_z}

                # Only overplot SkyMapper points when the displayed spectrum is flux calibrated
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
                    bbox=dict(boxstyle="round,pad=0.2", alpha=0.15)
                )
                if ylab == "FLUX_FLAM":
                    axC.legend(loc="best", fontsize=9)

                # spectrum y-limits: robust
                yv = flux_disp[ok]
                if yv.size > 50:
                    p1, p99 = np.nanpercentile(yv, [1, 99])
                    if np.isfinite(p1) and np.isfinite(p99) and p99 > p1:
                        axC.set_ylim(p1, p99)

                fig.suptitle(f"{slit}  (Step11 fluxcal summary)", fontsize=12)

                pdf.savefig(fig)
                plt.close(fig)
                n_written += 1

        if not args.quiet:
            print("Wrote:", out_pdf)
            if n_skipped_nophot:
                print(f"Skipped {n_skipped_nophot} slits (no RA/Dec in photometry catalog).")
            print(f"Pages written: {n_written}")


if __name__ == "__main__":
    main()