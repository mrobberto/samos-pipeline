#!/usr/bin/env python3

import argparse
from pathlib import Path
import subprocess
import sys

import config


def main():
    ap = argparse.ArgumentParser(
        description="Production Step09: OH registration + residual-sky cleanup"
    )
    ap.add_argument(
        "--in-fits",
        default=str(config.EXTRACT1D_WAV),
        help="Input Step08 wavelength-attached extraction",
    )
    ap.add_argument(
        "--outdir",
        default=str(config.ST09),
        help="Accepted for master-driver compatibility; canonical output paths come from config",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]

    inp = Path(args.in_fits)
    csv = Path(config.OH_SHIFT_CSV)
    ohref = Path(config.EXTRACT1D_OHREF)
    final = Path(config.EXTRACT1D_OHCLEAN)

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
        ],
        [
            sys.executable,
            str(root / "pipeline/step09_oh_refine/step09_sky_template_cleanup.py"),
            "--infile", str(ohref),
            "--outfile", str(final),
        ],
    ]

    for cmd in cmds:
        print("[CMD]", " ".join(map(str, cmd)))
        subprocess.run(cmd, cwd=root, check=True)

    print("Step09 production product:", final)


if __name__ == "__main__":
    main()
