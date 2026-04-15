#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step07h — propagate MASTER wavelength solution to all slit 1D arc spectra.

Corrected version:
- preserves the original Step 07.5 propagation logic
- uses the restored canonical Step07c slitid products
- uses config.MASTER_ARC_FITS
- uses config.WAVECAL_YWIN0 / config.WAVECAL_FIRSTLEN

Panel-B convention
------------------
For each slit:
    y_local = 0 .. FIRSTLEN-1
    y_master = y_local + SHIFT_TO_MASTER
    lambda_nm = poly(y_master)

Flux is windowed from detector rows [YWIN0 : YWIN0+FIRSTLEN).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("step07h_propagate")

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=str, default=None,
                    help="Optional explicit Step07f master arc FITS")
    ap.add_argument("--show-plots", action="store_true")
    return ap.parse_args()

def default_arc1d_path(trace_set: str) -> Path:
    wavecal_dir = Path(config.ST07_WAVECAL).expanduser()

    if hasattr(config, "MASTER_ARC_DIFF"):
        stem = Path(config.MASTER_ARC_DIFF).stem
        p = wavecal_dir / f"{stem}_pixflatcorr_clipped_1D_slitid_{trace_set}.fits"
        if p.exists():
            return p

    hits = sorted(wavecal_dir.glob(f"*_1D_slitid_{trace_set}.fits"))
    if hits:
        return hits[-1]

    return wavecal_dir / f"ArcDiff_pixflatcorr_clipped_1D_slitid_{trace_set}.fits"


def poly_from_header(hdr: fits.Header) -> np.poly1d:
    coeff = []
    for i in range(50):
        k = f"WVC{i}"
        if k in hdr:
            coeff.append(float(hdr[k]))
        else:
            break
    if len(coeff) < 2:
        raise ValueError("MASTER solution header lacks WVC0.. coefficients.")
    return np.poly1d(coeff[::-1])


def read_shift_to_master(master_all: Path) -> dict[str, int]:
    with fits.open(master_all) as h:
        if "SLITLIST" not in h:
            raise KeyError(f"{master_all} missing SLITLIST extension")
        tab = h["SLITLIST"].data

    out: dict[str, int] = {}
    for row in tab:
        slit = str(row["SLIT"]).strip().upper()
        if slit.startswith("SLIT"):
            out[slit] = int(row["SHIFT_TO_MASTER"])

    if not out:
        raise RuntimeError(f"No SHIFT_TO_MASTER entries found in {master_all}")

    return out


def open_arc_sources(even_path: Path, odd_path: Path) -> list[Path]:
    sources = []
    if even_path.exists():
        sources.append(even_path)
    if odd_path.exists():
        sources.append(odd_path)
    if not sources:
        raise FileNotFoundError(f"No arc slitid files found: {even_path} / {odd_path}")
    return sources


def main() -> None:
    args = parse_args()
    
    wavecal_dir = Path(config.ST07_WAVECAL)
    
    ARC1D_EVEN = wavecal_dir / "arc_1d_even.fits"
    ARC1D_ODD  = wavecal_dir / "arc_1d_odd.fits"
    
    MASTER_ALL = Path(args.master).expanduser() if args.master else (wavecal_dir / "arc_master.fits")
    MASTER_SOL = wavecal_dir / "arc_master_wavesol.fits"
    OUT_FITS   = wavecal_dir / "arc_1d_wavelength_all.fits"   # or your chosen local output name

    YWIN0 = int(config.WAVECAL_YWIN0)
    FIRSTLEN = int(config.WAVECAL_FIRSTLEN)

    log.info("ARC1D_EVEN = %s", ARC1D_EVEN)
    log.info("ARC1D_ODD  = %s", ARC1D_ODD)
    log.info("MASTER_ALL = %s", MASTER_ALL)
    log.info("MASTER_SOL = %s", MASTER_SOL)
    log.info("OUT_FITS   = %s", OUT_FITS)
    log.info("YWIN0=%d FIRSTLEN=%d", YWIN0, FIRSTLEN)

    if not MASTER_SOL.exists():
        raise FileNotFoundError(MASTER_SOL)
    if not MASTER_ALL.exists():
        raise FileNotFoundError(MASTER_ALL)

    shift_to_master = read_shift_to_master(MASTER_ALL)
    log.info("Loaded SHIFT_TO_MASTER for %d slits", len(shift_to_master))

    with fits.open(MASTER_SOL) as h:
        hdrm = h[0].header.copy()
        poly = poly_from_header(hdrm)

    wvc = []
    for i in range(50):
        k = f"WVC{i}"
        if k in hdrm:
            wvc.append((k, float(hdrm[k]), hdrm.comments[k] if k in hdrm.comments else ""))
        else:
            break

    ywin_local = np.arange(FIRSTLEN, dtype=np.float32)
    y_det_lo = int(YWIN0)
    y_det_hi = int(YWIN0 + FIRSTLEN - 1)

    phdr = fits.Header()
    phdr["STAGE"] = ("07h", "Pipeline stage")
    phdr["CONVENT"] = ("panel_B", "lambda(y_slit)=poly(y_local + SHIFT_TO_MASTER)")
    phdr["YWIN0"] = (int(YWIN0), "Detector Y start of master window")
    phdr["FIRSTLEN"] = (int(FIRSTLEN), "Length of master window")
    phdr["YDETLO"] = (y_det_lo, "Detector Y low (inclusive) for window")
    phdr["YDETHI"] = (y_det_hi, "Detector Y high (inclusive) for window")
    phdr["MASTER"] = (MASTER_SOL.name, "Master wavelength solution source")
    phdr["SHIFTSRC"] = (MASTER_ALL.name, "SHIFT_TO_MASTER source")
    phdr["SRCLEN"] = (4112, "Input arc1D grid is detector rows 0..4111")
    phdr.add_history("Step07h: propagated master wavelength solution using SHIFT_TO_MASTER.")
    for k, v, c in wvc:
        phdr[k] = (v, c)

    out_hdus = [fits.PrimaryHDU(header=phdr)]
    written: set[str] = set()

    sources = open_arc_sources(ARC1D_EVEN, ARC1D_ODD)

    for src in sources:
        log.info("Reading %s", src.name)
        with fits.open(src) as h:
            for ext in h[1:]:
                slit = ext.name.strip().upper()
                if not slit.startswith("SLIT"):
                    continue
                if slit in written:
                    continue
                if slit not in shift_to_master:
                    log.warning("Skipping %s (no SHIFT_TO_MASTER in SLITLIST)", slit)
                    continue

                d = ext.data
                if d is None:
                    log.warning("%s: no data, skipping", slit)
                    continue

                arr = np.asarray(d, dtype=np.float32)
                if not (arr.ndim == 2 and arr.shape[0] >= 2):
                    raise ValueError(f"{slit}: unexpected data shape {arr.shape}; expected 2-row image HDU")

                flux_full = arr[0].astype(np.float32)
                npix_full = arr[1].astype(np.float32)

                y0 = int(YWIN0)
                y1 = int(YWIN0 + FIRSTLEN)
                if flux_full.size < y1:
                    raise ValueError(f"{slit}: 1D length={flux_full.size} too short for window [{y0}:{y1}]")

                flux = flux_full[y0:y1].copy()
                npix = npix_full[y0:y1].copy()

                sh = int(shift_to_master[slit])
                if slit == "SLIT001":
                    log.info("DEBUG %s: SHIFT_TO_MASTER used = %s ; SHIFTSRC = %s", slit, sh, MASTER_ALL)

                y_eff = ywin_local + np.float32(sh)
                lam = poly(y_eff).astype(np.float32)

                outside = (y_eff < 0) | (y_eff > (FIRSTLEN - 1))
                lam[outside] = np.nan

                bad = (~np.isfinite(flux)) | (~np.isfinite(npix)) | (npix <= 0)
                lam[bad] = np.nan
                flux[bad] = np.nan

                out = np.vstack([flux, lam]).astype(np.float32)

                hh = fits.Header()
                hh["EXTNAME"] = slit
                hh["SLITID"] = slit
                hh["SHIFT2M"] = (int(sh), "SHIFT_TO_MASTER: y_master = y_slit + SHIFT2M")
                hh["YWIN0"] = (int(YWIN0), "Detector Y start of window")
                hh["FIRSTLEN"] = (int(FIRSTLEN), "Length of window")
                hh["LAMUNIT"] = ("nm", "Wavelength unit")
                hh["SRC1D"] = (src.name, "Source 1D arc spectra")
                hh["MASTRSOL"] = (MASTER_SOL.name, "Master polynomial source")
                hh["SHIFTSRC"] = (MASTER_ALL.name, "SHIFT_TO_MASTER source")

                for k, v, c in wvc:
                    hh[k] = (v, c)

                for k in ("INDEX", "RA", "DEC", "TRACESET", "ROT180", "BKGID"):
                    if k in ext.header:
                        hh[k] = ext.header[k]

                out_hdus.append(fits.ImageHDU(data=out, header=hh))
                written.add(slit)

    out_hdus[0].header["NSLITS"] = (len(written), "Number of slit extensions written")
    fits.HDUList(out_hdus).writeto(OUT_FITS, overwrite=True)
    log.info("Wrote: %s", OUT_FITS)

    with fits.open(OUT_FITS) as h:
        slit_names = [x.name for x in h[1:] if x.name.startswith("SLIT")]
        if not slit_names:
            raise RuntimeError("No SLIT extensions found in output FITS")

        s0 = slit_names[0]
        lam0 = h[s0].data[1]
        log.info("%s lambda nm: min/med/max = %.3f / %.3f / %.3f",
                 s0, np.nanmin(lam0), np.nanmedian(lam0), np.nanmax(lam0))
        log.info("Output wavelength FITS contains %d slits", len(slit_names))
        log.info("First/last slits: %s … %s", slit_names[0], slit_names[-1])

    log.info("Step07h complete.")


if __name__ == "__main__":
    main()
