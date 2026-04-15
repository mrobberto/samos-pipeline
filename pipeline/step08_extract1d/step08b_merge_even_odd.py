#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step08b — merge EVEN and ODD extraction products into a single MEF.

PURPOSE
-------
Combine the two parity-specific extraction products (EVEN and ODD slits)
into a single merged Extract1D file with globally consistent SLIT###
identifiers.

This step does NOT modify the extracted spectra; it only reorganizes them.

INPUTS
------
- extract1d_optimal_ridge_even.fits
- extract1d_optimal_ridge_odd.fits

Each file contains:
    SLIT### binary tables in pixel space (no wavelength yet)

OUTPUT
------
- extract1d_optimal_ridge_all.fits

Contains:
    All SLIT### extensions, sorted by slit number

MERGING RULES
-------------
- SLIT extensions are merged by EXTNAME (SLIT###)
- Global slit numbering must already be unique (from Step04)
- Duplicates are NOT allowed by default

Duplicate handling:
    - default: raise error if duplicates found
    - optional (--allow-dup): keep EVEN version, drop ODD

METADATA HANDLING
-----------------
- Preserves all per-slit headers
- Adds SRCFILE keyword to track origin (EVEN/ODD)
- Ensures SLITID matches SLIT### naming

PRIMARY HEADER
--------------
Records:
    - input filenames
    - merge stage (STAGE=08b)
    - number of merged slits

NOTES
-----
- This is still a pixel-space product
- No wavelength information is present yet
- Step08c must be run next to attach LAMBDA_NM

DOES NOT DO
-----------
- modify flux, variance, or sky
- perform wavelength calibration
- reorder within slits (only across slits)

run:
    > PYTHONPATH=. python pipeline/step08_extract1d/step08b_merge_even_odd.py 
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from astropy.io import fits

import config


def default_dir() -> Path:
    return Path(config.ST08_EXTRACT1D)


def is_slit_ext(hdu: fits.hdu.base.ExtensionHDU) -> bool:
    name = (hdu.name or "").upper()
    return name.startswith("SLIT")


def slit_num(name: str) -> int:
    try:
        return int(name.upper().replace("SLIT", ""))
    except Exception:
        return 10**9


def collect_slits(hdul: fits.HDUList) -> list:
    return [h for h in hdul[1:] if is_slit_ext(h)]


def merge_even_odd(even_path: Path, odd_path: Path, out_path: Path, allow_dup: bool = False) -> None:
    if not even_path.exists():
        raise FileNotFoundError(even_path)
    if not odd_path.exists():
        raise FileNotFoundError(odd_path)

    with fits.open(even_path) as he, fits.open(odd_path) as ho:
        prihdr = he[0].header.copy()
        prihdr["PIPESTEP"] = "STEP08"
        prihdr["STAGE"] = ("08b", "Pipeline stage")
        prihdr["INEVEN"] = (even_path.name, "Input EVEN extraction file")
        prihdr["INODD"] = (odd_path.name, "Input ODD extraction file")
        prihdr["OUTTYPE"] = ("MERGED", "Merged EVEN+ODD extraction product")
        prihdr.add_history(f"Step08b merge created {datetime.utcnow().isoformat()}Z")
        prihdr.add_history(f"INPUT_EVEN={even_path}")
        prihdr.add_history(f"INPUT_ODD={odd_path}")

        hdul_out = fits.HDUList([fits.PrimaryHDU(header=prihdr)])

        e_slits = collect_slits(he)
        o_slits = collect_slits(ho)

        e_names = {h.name.upper() for h in e_slits}
        o_names = {h.name.upper() for h in o_slits}
        dup = sorted(e_names & o_names, key=slit_num)

        if dup and not allow_dup:
            raise RuntimeError(
                "Duplicate SLIT extensions found in BOTH EVEN and ODD inputs:\n"
                f"{dup}\n"
                "This should not happen if global slit IDs are unique. "
                "Use --allow-dup only if you intentionally want EVEN to win."
            )

        merged = {h.name.upper(): (h, even_path.name) for h in e_slits}
        for h in o_slits:
            key = h.name.upper()
            if key in merged and allow_dup:
                continue
            merged[key] = (h, odd_path.name)

        n_slits = 0
        for key in sorted(merged.keys(), key=slit_num):
            src, srcname = merged[key]
            new_hdu = src.copy()
            new_hdu.name = key
            if "SLITID" in new_hdu.header:
                try:
                    sid = int(new_hdu.header["SLITID"])
                    if sid != slit_num(key):
                        new_hdu.header["SLITID"] = slit_num(key)
                except Exception:
                    new_hdu.header["SLITID"] = slit_num(key)
            else:
                new_hdu.header["SLITID"] = slit_num(key)

            new_hdu.header["SRCFILE"] = (srcname, "Source parity product for this slit")
            hdul_out.append(new_hdu)
            n_slits += 1

        hdul_out[0].header["N_SLITS"] = (n_slits, "Number of merged SLIT extensions")
        hdul_out[0].header["ALLOWDUP"] = (1 if allow_dup else 0, "Duplicates allowed during merge")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        hdul_out.writeto(out_path, overwrite=True)

    with fits.open(out_path) as h:
        slits = [x.name for x in h[1:] if is_slit_ext(x)]
        if len(slits) != len(set(slits)):
            raise RuntimeError("Output contains duplicate SLIT EXTNAMEs.")

    print("[DONE] Wrote", out_path)
    print("[INFO] SLIT extensions:", len(slits))
    if slits:
        print("[INFO] first/last:", slits[0], slits[-1])


def parse_args():
    ap = argparse.ArgumentParser(description="SAMOS Step08b merge EVEN/ODD extractions")
    ap.add_argument("--even", type=str, default="", help="Input EVEN extraction FITS")
    ap.add_argument("--odd", type=str, default="", help="Input ODD extraction FITS")
    ap.add_argument("--out", type=str, default="", help="Output merged extraction FITS")
    ap.add_argument("--allow-dup", action="store_true",
                    help="Allow duplicate SLITs (keep EVEN version, drop ODD duplicate)")
    return ap.parse_args()


def main():
    args = parse_args()
    base_dir = default_dir()

    even_path = Path(args.even) if args.even else (base_dir / "extract1d_optimal_ridge_even.fits")
    odd_path = Path(args.odd) if args.odd else (base_dir / "extract1d_optimal_ridge_odd.fits")
    out_path = Path(args.out) if args.out else (base_dir / "extract1d_optimal_ridge_all.fits")

    merge_even_odd(even_path, odd_path, out_path, allow_dup=args.allow_dup)


if __name__ == "__main__":
    main()
