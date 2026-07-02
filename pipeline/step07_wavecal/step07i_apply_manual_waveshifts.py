#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:33:36 2026

@author: robberto
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits

import config


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default=str(config.ARC_WAVELENGTH_BASE))
    ap.add_argument("--outfile", default=str(config.ARC_WAVELENGTH_TWEAKED))
    ap.add_argument(
        "--table",
        default=str(config.MANUAL_WAVESHIFT_TABLE)
        if config.MANUAL_WAVESHIFT_TABLE is not None
        else "",
    )
    ap.add_argument("--delete-stale-if-empty", action="store_true")
    return ap.parse_args()


def read_shift_table(path):
    path = Path(path)

    if not path.exists():
        return {}

    df = pd.read_csv(path)

    if len(df) == 0:
        return {}

    if "slit" not in df.columns or "shift_nm" not in df.columns:
        raise RuntimeError(
            f"{path} must contain columns: slit, shift_nm"
        )

    df["slit"] = df["slit"].astype(str).str.upper().str.strip()
    df["shift_nm"] = pd.to_numeric(df["shift_nm"], errors="coerce")

    df = df[np.isfinite(df["shift_nm"])]

    return dict(zip(df["slit"], df["shift_nm"]))


def main():
    args = parse_args()

    infile = Path(args.infile)
    outfile = Path(args.outfile)
    table = Path(args.table) if args.table else None

    if table is None:
        shifts = {}
    else:
        shifts = read_shift_table(table)

    print("Input :", infile)
    print("Output:", outfile)
    print("Table :", table)
    print("Manual wavelength shifts:", shifts)

    if not shifts:
        if args.delete_stale_if_empty and outfile.exists():
            outfile.unlink()
            print("No manual shifts requested; deleted stale file:")
            print(" ", outfile)

        print("No manual shifts requested.")
        print("Use baseline file:")
        print("ACTIVE_WAV_FITS =", infile)
        return

    with fits.open(infile) as hdul:
        new = fits.HDUList([h.copy() for h in hdul])

    changed = []

    for hdu in new[1:]:
        slit = (hdu.header.get("EXTNAME") or hdu.name or "").strip().upper()

        if slit not in shifts:
            continue

        arr = np.asarray(hdu.data, float)

        if arr.ndim < 2 or arr.shape[0] < 2:
            raise ValueError(f"{slit}: unexpected shape {arr.shape}")

        dlam = float(shifts[slit])

        arr[1] = arr[1] + dlam
        hdu.data = arr

        hdu.header["LAMSHIFT"] = (
            dlam,
            "Manual wavelength zero-point shift [nm]",
        )
        hdu.header["LAMSHSRC"] = (
            table.name,
            "Manual wavelength-shift table",
        )

        changed.append((slit, dlam))
        print(f"{slit}: applied {dlam:+.3f} nm")

    if not changed:
        raise RuntimeError("No requested slits were found; no tweaked file written.")

    new[0].header["HISTORY"] = (
        f"Manual slit wavelength zero-point shifts applied from {table}"
    )
    new[0].header["LAMSHSRC"] = (
        table.name,
        "Manual wavelength-shift table",
    )

    outfile.parent.mkdir(parents=True, exist_ok=True)
    new.writeto(outfile, overwrite=True)

    print()
    print("Wrote:", outfile)
    print("ACTIVE_WAV_FITS =", outfile)


if __name__ == "__main__":
    main()
