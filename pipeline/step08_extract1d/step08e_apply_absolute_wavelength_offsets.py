#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step08e — apply authoritative science-vs-arc wavelength zero-point offsets.

This stage applies only the offsets listed in an authoritative CSV table.
It does not measure offsets and does not perform the small ensemble-relative
OH refinement handled later by Step09a/09b.

Required CSV columns
--------------------
slit, correction_nm, status, evidence

Sign convention
---------------
lambda_out = lambda_in + correction_nm

Safety
------
- The input is never modified.
- Existing output is refused unless --overwrite is supplied.
- Inputs already tagged ABSWAPP=True are refused to prevent double application.
- Every SLIT extension in the FITS file must have exactly one table row.
- Non-zero corrections must have status ADOPTED.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


def norm_slit(x: str) -> str:
    return str(x).strip().upper()


def slit_num(x: str) -> int:
    try:
        return int(norm_slit(x).replace("SLIT", ""))
    except Exception:
        return 10**9


def parse_args():
    p = argparse.ArgumentParser(
        description="Apply authoritative Step08 absolute wavelength offsets"
    )
    p.add_argument("--infile", type=Path, required=True)
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--outfile", type=Path, required=True)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output file",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.infile.exists():
        raise FileNotFoundError(args.infile)
    if not args.table.exists():
        raise FileNotFoundError(args.table)
    if args.outfile.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.outfile} exists; use --overwrite only if replacement is intended"
        )

    tab = pd.read_csv(args.table)
    required = {"slit", "correction_nm", "status", "evidence"}
    missing = required - set(tab.columns)
    if missing:
        raise ValueError(
            "Authoritative table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    tab = tab.copy()
    tab["slit"] = tab["slit"].map(norm_slit)

    if tab["slit"].duplicated().any():
        dup = tab.loc[tab["slit"].duplicated(keep=False), "slit"].tolist()
        raise ValueError(f"Duplicate slit rows in authoritative table: {dup}")

    tab["correction_nm"] = pd.to_numeric(tab["correction_nm"], errors="raise")
    if not np.isfinite(tab["correction_nm"].to_numpy(float)).all():
        raise ValueError("Non-finite correction_nm found in authoritative table")

    # Only explicitly ADOPTED rows may carry a non-zero absolute correction.
    nonzero = np.abs(tab["correction_nm"].to_numpy(float)) > 1e-12
    bad_nonzero = tab.loc[
        nonzero & (tab["status"].astype(str).str.upper() != "ADOPTED")
    ]
    if len(bad_nonzero):
        raise ValueError(
            "Non-zero correction found for non-ADOPTED row(s): "
            + ", ".join(bad_nonzero["slit"].tolist())
        )

    rows = tab.set_index("slit")

    with fits.open(args.infile, memmap=False) as hdul:
        if bool(hdul[0].header.get("ABSWAPP", False)):
            raise RuntimeError(
                "Input already has ABSWAPP=True; refusing possible double application"
            )

        out = fits.HDUList([h.copy() for h in hdul])

    fits_slits = []
    for hdu in out[1:]:
        slit = norm_slit(hdu.name)
        if slit.startswith("SLIT"):
            fits_slits.append(slit)

    fits_slits = sorted(set(fits_slits), key=slit_num)

    missing_rows = [s for s in fits_slits if s not in rows.index]
    extra_rows = [s for s in rows.index if s not in set(fits_slits)]
    if missing_rows:
        raise ValueError(
            "Authoritative table has no row for FITS slit(s): "
            + " ".join(missing_rows)
        )
    if extra_rows:
        raise ValueError(
            "Authoritative table contains slit(s) absent from FITS input: "
            + " ".join(sorted(extra_rows, key=slit_num))
        )

    n_applied = 0

    for slit in fits_slits:
        hdu = out[slit]
        if hdu.data is None or not hasattr(hdu.data, "columns"):
            raise ValueError(f"{slit}: missing binary-table data")

        names = {n.upper(): n for n in hdu.columns.names}
        if "LAMBDA_NM" not in names:
            raise ValueError(f"{slit}: no LAMBDA_NM column")

        r = rows.loc[slit]
        corr = float(r["correction_nm"])
        status = str(r["status"]).strip()
        evidence = str(r["evidence"]).strip()

        lam_name = names["LAMBDA_NM"]
        lam = np.asarray(hdu.data[lam_name], float)

        if abs(corr) > 1e-12:
            hdu.data[lam_name] = lam + corr
            n_applied += 1

        hdu.header["ABSWCOR"] = (
            corr,
            "absolute wavelength correction added, nm",
        )
        hdu.header["ABSWSTAT"] = (status[:68], "absolute wavelength status")
        hdu.header["ABSWEVID"] = (evidence[:68], "absolute wavelength evidence")

    ph = out[0].header
    ph["ABSWAPP"] = (True, "authoritative absolute wavelength offsets applied")
    ph["ABSWN"] = (len(fits_slits), "number of slit rows checked")
    ph["ABSWNAD"] = (n_applied, "number of non-zero offsets applied")
    ph["ABSWTAB"] = (args.table.name[:68], "authoritative offset table")
    ph.add_history(
        "Step08e: applied authoritative science-vs-arc wavelength offsets."
    )
    ph.add_history(
        f"STEP08E_INPUT={args.infile}"
    )
    ph.add_history(
        f"STEP08E_TABLE={args.table}"
    )

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    out.writeto(args.outfile, overwrite=args.overwrite)

    adopted = tab[
        (tab["status"].astype(str).str.upper() == "ADOPTED")
        & (np.abs(tab["correction_nm"].to_numpy(float)) > 1e-12)
    ].copy()

    print("INFILE  =", args.infile)
    print("TABLE   =", args.table)
    print("OUTFILE =", args.outfile)
    print("Slits checked =", len(fits_slits))
    print("Non-zero authoritative corrections applied =", n_applied)

    if len(adopted):
        print("\nApplied:")
        for _, r in adopted.sort_values(
            "slit", key=lambda x: x.map(slit_num)
        ).iterrows():
            print(
                f"  {r['slit']}  {float(r['correction_nm']):+.4f} nm  "
                f"{r['status']}  {r['evidence']}"
            )
    else:
        print("\nApplied: none")


if __name__ == "__main__":
    main()
