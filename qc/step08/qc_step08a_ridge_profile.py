#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from glob import glob
import argparse

import numpy as np
from astropy.io import fits

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def main():
    ap = argparse.ArgumentParser(description="Step08 ridge profile QC")
    ap.add_argument("--set", choices=["EVEN", "ODD"], required=True)
    ap.add_argument("--slit", required=True)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    set_tag = args.set.upper()
    slit = args.slit.upper()
    if not slit.startswith("SLIT"):
        slit = f"SLIT{int(slit):03d}"

    st06 = Path(config.ST06_SCIENCE)
    st08 = Path(config.ST08_EXTRACT1D)

    pattern = str(st06 / f"*_{set_tag}_tracecoords.fits")
    matches = sorted(glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)

    tracecoords = Path(matches[-1])
    analysis = st08 / f"trace_analysis_optimal_ridge_{set_tag.lower()}.fits"

    if not analysis.exists():
        raise FileNotFoundError(analysis)

    outdir = Path(args.outdir) if args.outdir else st08 / "qc_step08"
    outdir.mkdir(parents=True, exist_ok=True)

    with fits.open(tracecoords) as h06, fits.open(analysis) as h08:
        if slit not in h06:
            raise KeyError(f"{slit} not found in {tracecoords.name}")
        if slit not in h08:
            raise KeyError(f"{slit} not found in {analysis.name}")

        img = np.asarray(h06[slit].data, float)
        x0 = np.asarray(h08[slit].data["X0"], float)

    row_peak = np.nanmax(img, axis=1)
    y = int(np.nanargmax(row_peak))

    row = img[y]
    xx = np.arange(row.size)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xx, row, "k-", lw=1)
    ax.axvline(x0[y], color="r", label="X0")
    ax.axvspan(x0[y] - 3, x0[y] + 3, color="r", alpha=0.15, label="object aperture")
    ax.set_title(f"{slit}, {set_tag}, row {y}")
    ax.set_xlabel("TRACECOORDS X")
    ax.set_ylabel("ADU/s")
    ax.legend()
    fig.tight_layout()

    out = outdir / f"qc_step08a_ridge_profile_{set_tag.lower()}_{slit.lower()}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)

    print("[OK] tracecoords:", tracecoords)
    print("[OK] analysis   :", analysis)
    print("[OK] wrote      :", out)


if __name__ == "__main__":
    main()