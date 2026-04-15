#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step11 QC — per-slit summary pages for the final flux-calibrated product.

PURPOSE
-------
Create a per-slit PDF summary that combines:
  A) 2D rectified slit image (TRACECOORDS)
  B) optional detector-frame cutout and/or slit geometry overlays when available
  C) imaging postage stamp cutout from the reference imaging coadd
  D) RA/Dec mini-context with SkyMapper r/i/z photometry
  E) 1D spectrum panel using the best available science column

This is the primary detailed QC script for Step11.

DEFAULT INPUTS
--------------
- final spectrum product:
    config.ST11_FLUXCAL / extract1d_fluxcal.fits
- photometric match table:
    config.ST11_FLUXCAL / slit_trace_radec_skymapper_all.csv

Optional auxiliary inputs may be supplied explicitly:
- TRACECOORDS EVEN|ODD MEFs
- detector/full-frame science image
- imaging coadd
- mask / geometry files

OUTPUT
------
- config.ST11_FLUXCAL / qc_step11 / qc_step11_summary_pages.pdf

NOTES
-----
- This script is read-only and does not modify pipeline data products.
- It is intended for science validation, not catalog repair.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u

import config


def _norm_key(s: str) -> str:
    return s.strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def norm_slit(s: str) -> str:
    s2 = str(s).strip().upper()
    m = re.search(r"(SLIT\d+)", s2)
    return m.group(1) if m else s2


def get_slit_hdu(hdul, slit: str):
    s = norm_slit(slit)
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
    raise KeyError(f"Could not find HDU for {s}")


_OPEN_CACHE = {}
def _open_cached(path: Path):
    path = Path(path)
    key = str(path.resolve())
    if key not in _OPEN_CACHE:
        _OPEN_CACHE[key] = fits.open(path, memmap=False)
    return _OPEN_CACHE[key]


def _robust_limits(img: np.ndarray, stretch_sigma: float = 5.0):
    v = img[np.isfinite(img)]
    if v.size == 0:
        return 0.0, 1.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(v))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return med - stretch_sigma * sigma, med + stretch_sigma * sigma


def _abmag_to_fnu_jy(m: float) -> float:
    return 3631.0 * (10.0 ** (-0.4 * m))


def _fnu_jy_to_flambda(fnu_jy: float, lam_nm: float) -> float:
    cA = 2.99792458e18
    lam_A = lam_nm * 10.0
    fnu_cgs = fnu_jy * 1e-23
    return fnu_cgs * cA / (lam_A ** 2)


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
        "SLIT": df[slit_c].astype(str).map(norm_slit),
        "RA": pd.to_numeric(df[ra_c], errors="coerce"),
        "DEC": pd.to_numeric(df[dec_c], errors="coerce"),
        "r_mag": pd.to_numeric(df[r_c], errors="coerce"),
        "i_mag": pd.to_numeric(df[i_c], errors="coerce"),
        "z_mag": pd.to_numeric(df[z_c], errors="coerce"),
    })
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
    try:
        ax = fig.add_subplot(gs_cell, projection=stamp_wcs)
        ax.imshow(stamp, origin="lower")
        ax.set_title(title)
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")
        return ax
    except Exception:
        ax = fig.add_subplot(gs_cell)
        ax.imshow(stamp, origin="lower")
        ax.set_title(title + " (pixel)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return ax


def _read_slit_image(mef_path: Optional[Path], slit: str):
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
            return None
        if d is None or getattr(d, "ndim", 0) != 2:
            return None
        return np.asarray(d, float)
    d0 = h[0].data
    if d0 is None:
        return None
    if getattr(d0, "ndim", 0) == 2:
        return np.asarray(d0, float)
    if getattr(d0, "ndim", 0) == 3:
        return np.asarray(d0[0], float)
    return None


def _choose_display_spectrum(tab) -> tuple[np.ndarray, np.ndarray, str, str]:
    cols = [c.name for c in tab.columns]
    lam_col = "LAMBDA_NM" if "LAMBDA_NM" in cols else None
    if lam_col is None:
        raise KeyError("Missing LAMBDA_NM")
    lam = np.asarray(tab[lam_col], float)

    for col, label, color in [
        ("FLUX_FLAM", "FLUX_FLAM", "k"),
        ("FLUX_TELLCOR_O2", "FLUX_TELLCOR_O2", "red"),
        ("FLUX_APCORR", "FLUX_APCORR", "red"),
        ("FLUX", "FLUX", "red"),
    ]:
        if col in cols:
            arr = np.asarray(tab[col], float)
            if np.isfinite(arr).sum() > 0:
                return lam, arr, label, color

    for c in cols:
        if c == lam_col:
            continue
        try:
            arr = np.asarray(tab[c], float)
        except Exception:
            continue
        if np.isfinite(arr).sum() > 0:
            return lam, arr, c, "red"

    return lam, np.full_like(lam, np.nan), "Flux", "red"


def _split_even_odd(path_str: Optional[str]):
    if not path_str:
        return None, None
    s = str(path_str)
    for sep in ["|", ";"]:
        if sep in s:
            a, b = [p.strip() for p in s.split(sep, 1)]
            return (Path(a) if a else None), (Path(b) if b else None)
    p = Path(s)
    return p, p


def _pick_mef_for_slit(slit: str, even_path: Optional[Path], odd_path: Optional[Path]):
    if even_path is None and odd_path is None:
        return None
    m = re.match(r"SLIT(\d+)", slit.upper().strip())
    if not m:
        return even_path or odd_path
    sid = int(m.group(1))
    return even_path if sid % 2 == 0 else odd_path


def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Step11 QC summary pages")
    ap.add_argument("--extract", default=str(Path(config.ST11_FLUXCAL) / "extract1d_fluxcal.fits"))
    ap.add_argument("--photcat", default=str(Path(config.ST11_FLUXCAL) / "slit_trace_radec_skymapper_all.csv"))
    ap.add_argument("--tracecoords", default="")
    ap.add_argument("--image", default="")
    ap.add_argument("--slits", default="")
    ap.add_argument("--halfsize", type=int, default=30)
    ap.add_argument("--outpdf", default=str(Path(config.ST11_FLUXCAL) / "qc_step11" / "qc_step11_summary_pages.pdf"))
    return ap.parse_args()


def main():
    args = parse_args()
    extract_fits = Path(args.extract)
    photcat = Path(args.photcat)
    outpdf = Path(args.outpdf)
    outpdf.parent.mkdir(parents=True, exist_ok=True)

    if not extract_fits.exists():
        raise FileNotFoundError(extract_fits)
    if not photcat.exists():
        raise FileNotFoundError(photcat)

    trace_even, trace_odd = _split_even_odd(args.tracecoords if args.tracecoords else None)
    image_path = Path(args.image) if args.image else None
    if image_path is not None and not image_path.exists():
        image_path = None

    phot = _read_photcat(photcat)
    phot_map = {r["SLIT"]: r for _, r in phot.iterrows()}

    with fits.open(extract_fits) as hspec, PdfPages(outpdf) as pdf:
        available = [norm_slit(h.name) for h in hspec[1:] if h.data is not None and norm_slit(h.name).startswith("SLIT")]
        slits = available if not args.slits else [s for s in [norm_slit(x) for x in args.slits.split(",")] if s in available]

        n_missing_phot = 0
        for slit in slits:
            if slit not in phot_map:
                n_missing_phot += 1
                continue

            hdu = get_slit_hdu(hspec, slit)
            tab = hdu.data
            hdr = hdu.header
            lam, flux, ylabel, color = _choose_display_spectrum(tab)
            row = phot_map[slit]
            ra = float(row["RA"])
            dec = float(row["DEC"])

            fig = plt.figure(figsize=(11, 8.5))
            gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])

            # TRACECOORDS
            ax0 = fig.add_subplot(gs[0, 0])
            tfile = _pick_mef_for_slit(slit, trace_even, trace_odd)
            timg = _read_slit_image(tfile, slit) if tfile else None
            if timg is not None:
                vmin, vmax = _robust_limits(timg)
                ax0.imshow(timg, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="gray")
                ax0.set_title(f"{slit} TRACECOORDS")
            else:
                ax0.axis("off")
                ax0.text(0.5, 0.5, "No TRACECOORDS", ha="center", va="center")

            # Imaging stamp
            if image_path is not None:
                try:
                    stamp, stamp_wcs = _cutout_from_wcs(image_path, ra, dec, args.halfsize)
                    _add_stamp_axes(fig, gs[0, 1], stamp, stamp_wcs, "Imaging cutout")
                except Exception:
                    ax = fig.add_subplot(gs[0, 1])
                    ax.axis("off")
                    ax.text(0.5, 0.5, "Imaging cutout failed", ha="center", va="center")
            else:
                ax = fig.add_subplot(gs[0, 1])
                ax.axis("off")
                ax.text(0.5, 0.5, "No imaging file", ha="center", va="center")

            # Photometry/metadata panel
            ax2 = fig.add_subplot(gs[0, 2]); ax2.axis("off")
            txt = (
                f"SLIT   : {slit}\n"
                f"RA     : {ra:.6f}\n"
                f"DEC    : {dec:.6f}\n"
                f"r_mag  : {row['r_mag']}\n"
                f"i_mag  : {row['i_mag']}\n"
                f"z_mag  : {row['z_mag']}\n"
                f"FLUXCAL: {hdr.get('FLUXCAL', 'NA')}\n"
                f"SCALE  : {hdr.get('SCALE', 'NA')}\n"
                f"ALPHA  : {hdr.get('ALPHA', 'NA')}\n"
                f"NBAND  : {hdr.get('NBAND', 'NA')}\n"
                f"QCFLAG : {hdr.get('QCFLAG', 'NA')}"
            )
            ax2.text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=10)

            # Spectrum panel
            ax3 = fig.add_subplot(gs[1, :])
            ax3.plot(lam, flux, color=color, lw=0.9, label=ylabel)
            # photometric points if flux calibrated
            if ylabel == "FLUX_FLAM":
                ppts = []
                for mag, lnm, lab in [(row["r_mag"], 617.0, "r"), (row["i_mag"], 748.0, "i"), (row["z_mag"], 894.0, "z")]:
                    if pd.notnull(mag):
                        ppts.append((lnm, _fnu_jy_to_flambda(_abmag_to_fnu_jy(float(mag)), lnm), lab))
                if ppts:
                    ax3.scatter([p[0] for p in ppts], [p[1] for p in ppts], marker="o", s=28, label="SkyMapper r/i/z")
                    for x, y, lab in ppts:
                        ax3.text(x, y, lab, fontsize=8)
            ax3.set_xlabel("Wavelength (nm)")
            ax3.set_ylabel(ylabel)
            ax3.set_title(f"{slit} spectrum")
            ax3.legend(fontsize=8)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Summary page
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111); ax.axis("off")
        ax.text(
            0.02, 0.98,
            f"Step11 QC summary pages\n\n"
            f"Extract file: {extract_fits.name}\n"
            f"Photcat    : {photcat.name}\n"
            f"N slits in extract: {len(available)}\n"
            f"N slits skipped (missing phot): {n_missing_phot}\n",
            va="top", ha="left", family="monospace", fontsize=12
        )
        pdf.savefig(fig)
        plt.close(fig)

    print("[DONE] Wrote", outpdf)


if __name__ == "__main__":
    main()
