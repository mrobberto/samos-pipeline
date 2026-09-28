#!/usr/bin/env python3
"""
Production Step09: ensemble-relative OH wavelength registration.

This driver runs only the validated wavelength-registration sequence:

  09a  measure ensemble-relative OH shifts
  09b  apply accepted shifts to the wavelength arrays

No residual-sky subtraction or spectral-flux modification is performed in
Step09.  The input is the Step08e product after application of the
authoritative absolute science-spectrum wavelength offsets.
"""

import argparse
from pathlib import Path
import subprocess
import sys

import config


def main():
    ap = argparse.ArgumentParser(
        description="Production Step09: ensemble-relative OH wavelength registration"
    )
    ap.add_argument(
        "--in-fits",
        default=str(config.EXTRACT1D_ABSWAV),
        help="Input Step08e absolute-wavelength-corrected extraction",
    )
    ap.add_argument(
        "--outdir",
        default=str(config.ST09),
        help="Accepted for master-driver compatibility; canonical outputs come from config",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]

    inp = Path(args.in_fits)
    csv = Path(config.OH_SHIFT_CSV)
    ohref = Path(config.EXTRACT1D_OHREF)

    cmds = [
        [
            sys.executable,
            str(root / "pipeline/step09_oh_refine/step09a_measure_oh_shifts.py"),
            "--infile", str(inp),
            "--outcsv", str(csv),
        ],
        [
            sys.executable,
            str(root / "pipeline/step09_oh_refine/step09b_apply_oh_shifts.py"),
            "--in", str(inp),
            "--csv", str(csv),
            "--out", str(ohref),
            "--clip", "1.0",
        ],
    ]

    for cmd in cmds:
        print("[CMD]", " ".join(map(str, cmd)))
        subprocess.run(cmd, cwd=root, check=True)

    print("Step09 OH-refined product:", ohref)


if __name__ == "__main__":
    main()
