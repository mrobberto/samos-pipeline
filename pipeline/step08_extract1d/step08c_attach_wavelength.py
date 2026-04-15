#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step08c — attach wavelength vectors to extracted spectra.

PURPOSE
-------
Convert pixel-space extracted spectra into wavelength-calibrated spectra
by applying the master wavelength solution derived in Step07.

This is the ONLY step where wavelength (LAMBDA_NM) is attached to Step08 data.

INPUTS
------
1) Step08 merged extraction:
    extract1d_optimal_ridge_all.fits

2) Step07 arc master file:
    arc_master.fits
    (contains SLITLIST with SHIFT_TO_MASTER)

3) Step07 wavelength solution:
    arc_master_wavesol.fits
    (contains polynomial coefficients WVC*)

OUTPUT
------
- extract1d_optimal_ridge_all_wav.fits

Each SLIT### table gains:
    LAMBDA_NM column (wavelength per row)

WAVELENGTH MAPPING
------------------
For each row:

    YDET   = Y0DET + YPIX
    y_eff  = (YDET - YWIN0) + SHIFT_TO_MASTER
    λ      = polynomial(y_eff)

Where:
- Y0DET: detector offset per slit
- YWIN0: master reference origin
- SHIFT_TO_MASTER: slit-dependent alignment shift
- polynomial: defined by WVC* coefficients

AUTHORITATIVE SOURCE
--------------------
The wavelength solution is defined exclusively by:
    WVC* coefficients in arc_master_wavesol.fits

SHIFT_TO_MASTER provides per-slit alignment relative to the master slit.

ROBUSTNESS FEATURES
-------------------
- Missing inputs → slit copied without wavelength
- Out-of-domain rows → LAMBDA_NM = NaN
- Header propagation ensures full provenance
- WVC* coefficients copied into output headers

OUTPUT METADATA
---------------
Per slit:
    LAMBDA_NM : wavelength vector (nm)
    SHIFT2M   : applied shift to master
    YMAP      : mapping description

Primary header:
    MASTERARC : arc master file
    MASTSOL   : wavelength solution file
    YWIN0     : reference origin
    FIRSTLEN  : valid polynomial domain

NOTES
-----
- Wavelength is strictly a function of detector Y coordinate
- No interpolation or resampling of flux is performed
- Pixel sampling is preserved
- Output is ready for telluric correction (Step09)

DOES NOT DO
-----------
- modify flux, variance, or sky
- perform telluric correction
- refine wavelength (Step10)
- flux calibration (Step11)

run:
    > PYTHONPATH=. python pipeline/step08_extract1d/step08c_attach_wavelength.py --overwrite

"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from astropy.io import fits

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("step08c_attach_wavelength")


def poly_from_header(hdr: fits.Header) -> np.poly1d:
    coeff = []
    for i in range(50):
        key = f"WVC{i}"
        if key in hdr:
            coeff.append(float(hdr[key]))
        else:
            break
    if len(coeff) < 2:
        raise ValueError("Master solution header lacks WVC0.. coefficients.")
    return np.poly1d(coeff[::-1])


def read_shift_to_master(master_arc: Path) -> dict[str, float]:
    with fits.open(master_arc) as hdul:
        if "SLITLIST" not in hdul:
            raise KeyError(f"{master_arc} is missing SLITLIST extension")
        tab = hdul["SLITLIST"].data

    out: dict[str, float] = {}
    for row in tab:
        slit = str(row["SLIT"]).strip().upper()
        if not slit.startswith("SLIT"):
            continue
        if "SHIFT_TO_MASTER" not in tab.names:
            raise KeyError("SLITLIST lacks SHIFT_TO_MASTER column")
        out[slit] = float(row["SHIFT_TO_MASTER"])

    if not out:
        raise RuntimeError(f"No SHIFT_TO_MASTER entries found in {master_arc}")
    return out


def is_slit_ext(hdu) -> bool:
    return (hdu.name or "").upper().startswith("SLIT")


def add_or_replace_column(tab_hdu: fits.BinTableHDU, name: str, arr: np.ndarray, fmt: str = "E") -> fits.BinTableHDU:
    cols = tab_hdu.columns
    names = [c.name.upper() for c in cols]
    newcol = fits.Column(name=name, format=fmt, array=arr)
    if name.upper() in names:
        idx = names.index(name.upper())
        col_list = list(cols)
        col_list[idx] = newcol
        newcols = fits.ColDefs(col_list)
    else:
        newcols = cols + newcol
    return fits.BinTableHDU.from_columns(newcols, header=tab_hdu.header.copy(), name=tab_hdu.name)


def parse_args():
    ap = argparse.ArgumentParser(description="SAMOS Step08c attach wavelength")
    ap.add_argument("--in", dest="infile", type=str, default="",
                    help="Input merged Extract1D FITS")
    ap.add_argument("--out", dest="outfile", type=str, default="",
                    help="Output Extract1D FITS with LAMBDA_NM")
    ap.add_argument("--master", dest="master_arc", type=str, default="",
                    help="Step07 arc_master.fits (contains SLITLIST and SHIFT_TO_MASTER)")
    ap.add_argument("--wavesol", dest="master_sol", type=str, default="",
                    help="Step07 arc_master_wavesol.fits (contains WVC* polynomial)")
    ap.add_argument("--ywin0", type=float, default=np.nan,
                    help="Override YWIN0 if not present in master solution header")
    ap.add_argument("--firstlen", type=int, default=-1,
                    help="Override FIRSTLEN if not present in master solution header")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output file")
    return ap.parse_args()


def main():
    args = parse_args()

    st07 = Path(config.ST07_WAVECAL)
    st08 = Path(config.ST08_EXTRACT1D)

    infile = Path(args.infile) if args.infile else (st08 / "extract1d_optimal_ridge_all.fits")
    outfile = Path(args.outfile) if args.outfile else (st08 / "extract1d_optimal_ridge_all_wav.fits")
    master_arc = Path(args.master_arc) if args.master_arc else (st07 / "arc_master.fits")
    master_sol = Path(args.master_sol) if args.master_sol else (st07 / "arc_master_wavesol.fits")

    if not infile.exists():
        raise FileNotFoundError(infile)
    if not master_arc.exists():
        raise FileNotFoundError(master_arc)
    if not master_sol.exists():
        raise FileNotFoundError(master_sol)
    if outfile.exists() and not args.overwrite:
        raise FileExistsError(f"{outfile} exists. Use --overwrite to replace it.")

    log.info("INFILE    = %s", infile)
    log.info("OUTFILE   = %s", outfile)
    log.info("MASTERARC = %s", master_arc)
    log.info("MASTERSOL = %s", master_sol)

    shift_to_master = read_shift_to_master(master_arc)
    log.info("Loaded SHIFT_TO_MASTER for %d slits", len(shift_to_master))

    with fits.open(master_sol) as hdul_sol:
        hdrm = hdul_sol[0].header.copy()
        poly = poly_from_header(hdrm)

    ywin0 = float(args.ywin0) if np.isfinite(args.ywin0) else float(hdrm.get("YWIN0", np.nan))
    firstlen = int(args.firstlen) if args.firstlen > 0 else int(hdrm.get("FIRSTLEN", -1))
    if not np.isfinite(ywin0):
        raise KeyError("YWIN0 not found in master solution header; provide --ywin0.")
    if firstlen <= 0:
        raise KeyError("FIRSTLEN not found in master solution header; provide --firstlen.")

    wvc_keys = []
    for i in range(50):
        key = f"WVC{i}"
        if key in hdrm:
            wvc_keys.append((key, float(hdrm[key]), hdrm.comments[key] if key in hdrm.comments else ""))
        else:
            break

    with fits.open(infile) as hin:
        phdr = hin[0].header.copy()
        phdr["PIPESTEP"] = "STEP08"
        phdr["STAGE"] = ("08c", "Pipeline stage")
        phdr["WAVATT"] = (True, "LAMBDA_NM column attached by Step08c")
        phdr["MASTERARC"] = (master_arc.name, "Step07 arc master file")
        phdr["MASTSOL"] = (master_sol.name, "Step07 master wavelength solution")
        phdr["YWIN0"] = (float(ywin0), "MASTER window start used for wavelength mapping")
        phdr["FIRSTLEN"] = (int(firstlen), "MASTER window length (poly domain)")
        phdr.add_history("Step08c: attached LAMBDA_NM using Step07 master polynomial and SHIFT_TO_MASTER.")
        phdr.add_history(f"INPUT_EXTRACT1D={infile}")
        phdr.add_history(f"MASTER_ARC={master_arc}")
        phdr.add_history(f"MASTER_SOL={master_sol}")

        for key, val, comment in wvc_keys:
            phdr[key] = (val, comment)

        hout = fits.HDUList([fits.PrimaryHDU(header=phdr)])

        n_slits = 0
        n_wavok = 0
        for ext in hin[1:]:
            if not is_slit_ext(ext):
                hout.append(ext.copy())
                continue

            slit = ext.name.strip().upper()
            n_slits += 1
            if ext.data is None:
                log.warning("%s: empty table, copying", slit)
                hout.append(ext.copy())
                continue

            if "YPIX" not in ext.columns.names:
                log.warning("%s: missing YPIX, copying without LAMBDA_NM", slit)
                hout.append(ext.copy())
                continue

            if slit not in shift_to_master:
                log.warning("%s: missing SHIFT_TO_MASTER, copying without LAMBDA_NM", slit)
                hout.append(ext.copy())
                continue

            y0det = ext.header.get("Y0DET", ext.header.get("YMIN", None))
            if y0det is None:
                log.warning("%s: missing Y0DET/YMIN, copying without LAMBDA_NM", slit)
                hout.append(ext.copy())
                continue

            y_local = np.asarray(ext.data["YPIX"], float)
            y_det = y_local + float(y0det)
            shift = float(shift_to_master[slit])
            y_eff = (y_det - ywin0) + shift
            lam = poly(y_eff).astype(np.float32)

            # Set outside-master-domain wavelengths to NaN for safety.
            bad = ~np.isfinite(y_eff) | (y_eff < 0) | (y_eff > (firstlen - 1))
            lam[bad] = np.nan

            new_hdu = add_or_replace_column(ext, "LAMBDA_NM", lam, fmt="E")
            new_hdu.header["Y0DET"] = (float(y0det), "Detector-row offset for Extract1D YPIX")
            new_hdu.header["YWIN0"] = (float(ywin0), "MASTER window start used for wavelength mapping")
            new_hdu.header["FIRSTLEN"] = (int(firstlen), "MASTER window length (poly domain)")
            new_hdu.header["SHIFT2M"] = (float(shift), "SHIFT_TO_MASTER used to attach wavelength")
            new_hdu.header["YMAP"] = ("YDET=(Y0DET+YPIX); y=(YDET-YWIN0)+SHIFT2M",
                                       "Step08c wavelength mapping")
            new_hdu.header.add_history("Added/updated LAMBDA_NM using Step07 master polynomial and SHIFT_TO_MASTER.")
            for key, val, comment in wvc_keys:
                if key not in new_hdu.header:
                    new_hdu.header[key] = (val, comment)

            hout.append(new_hdu)
            n_wavok += 1

        hout[0].header["N_SLITS"] = (n_slits, "Number of SLIT extensions seen")
        hout[0].header["N_WAVOK"] = (n_wavok, "Number of slits with LAMBDA_NM written")

        outfile.parent.mkdir(parents=True, exist_ok=True)
        hout.writeto(outfile, overwrite=args.overwrite)

    log.info("Wrote: %s", outfile)
    log.info("Slits processed: %d; wavelength attached: %d", n_slits, n_wavok)


if __name__ == "__main__":
    main()
