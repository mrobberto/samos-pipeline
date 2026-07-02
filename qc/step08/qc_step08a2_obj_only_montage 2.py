#!/usr/bin/env python3
"""
QC Step08a2 — OBJ-only spectral montage

This script generates a slit-by-slit mosaic of the extracted one-dimensional
spectra using the FLUX column from the Step08a2 products
(`extract1d_optimal_ridge_{even,odd}.fits`).

Despite the column name, this QC is intended to visualize the *object signal*
in a robust way, and is most useful when FLUX approximates OBJ_PRESKY (i.e.,
prior to high-fidelity sky subtraction in Step09).

Purpose
-------
- Provide a quick visual inspection of spectral shape and S/N across all slits
- Identify problematic extractions (e.g., failed sky subtraction, noisy spectra,
  empty slits, edge-truncated traces)
- Assess consistency of continuum shape and spectral features across the field

Key features
------------
- One panel per slit (sorted by slit number)
- Linear scale per slit using robust percentile limits (5–95%)
- Zero level indicated for reference
- No sky or diagnostic overlays (clean view of extracted signal)

Input
-----
Step08a2 output FITS file:
    extract1d_optimal_ridge_{even,odd}.fits

Each slit is expected to contain at least:
    YPIX  : row index (dispersion axis)
    FLUX  : extracted 1D spectrum (may include residual sky)

Notes
-----
- This QC is primarily intended to visualize the *signal morphology*, not the
  absolute flux calibration or sky-subtraction quality.
- In cases where local sky subtraction is unreliable (e.g., crowded slits),
  the spectra may appear biased or noisy; this is expected and addressed in Step09.
- Empty or low-S/N slits will appear flat or noise-dominated.

Output
------
PNG image:
    qc_step08/{even,odd}_obj_only_montage.png

Conceptual context
------------------
This plot represents the first fully extracted 1D spectra in the pipeline.
It bridges:
    Step08a1 → geometry and ridge determination
    Step08a2 → flux extraction
and serves as a diagnostic input to:
    Step09 → sky emission modeling and refinement
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import math
import re
from pathlib import Path
import config
import argparse

def slit_num(name):
    m = re.match(r"SLIT(\d+)", name.upper())
    return int(m.group(1)) if m else 9999

def main(set_tag="EVEN", outdir=None):

    infile = Path(config.ST08_EXTRACT1D) / f"extract1d_optimal_ridge_{set_tag.lower()}.fits"
    outdir = (
        Path(outdir)
        if outdir is not None
        else Path(config.ST08_EXTRACT1D) / "qc_step08"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    slit_data = {}

    with fits.open(infile) as hdul:
        for h in hdul[1:]:
            name = h.name.upper()
            if not name.startswith("SLIT"):
                continue

            d = h.data
            if d is None:
                continue

            names = [c.upper() for c in d.names]
            if "FLUX" not in names:
                continue

            y = np.asarray(d["YPIX"], float)
            obj = np.asarray(d["OBJ_PRESKY"], float)

            slit_data[name] = (y, obj, h.header)

    slits = sorted(slit_data.keys(), key=slit_num)

    ncols = 6
    nrows = math.ceil(len(slits) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5*ncols, 2.5*nrows))
    axes = axes.ravel()

    for ax in axes:
        ax.set_visible(False)

    for ax, slit in zip(axes, slits):
        ax.set_visible(True)

        y, obj, hdr = slit_data[slit]

        good = np.isfinite(obj)
        if np.any(good):
            lo, hi = np.nanpercentile(obj[good], [5, 95])
        else:
            lo, hi = -1, 1

        ax.plot(y, obj, 'k', lw=0.8)
        ax.axhline(0, color='0.7', lw=0.5)

        ax.set_ylim(lo, hi)
        ax.set_title(slit, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    outpng = outdir / f"{set_tag.lower()}_obj_presky_montage.png"
    fig.suptitle(f"{set_tag} OBJ_PRESKY spectra", fontsize=16)

    fig.suptitle(f"{set_tag} OBJ (linear scale)", fontsize=16)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(outpng, dpi=160)
    plt.close(fig)

    print("Wrote:", outpng)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QC Step08a2 OBJ-only mosaic")
    
    parser.add_argument(
        "--set",
        required=True,
        choices=["EVEN", "ODD", "even", "odd"]
    )
    
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for QC PNG"
    )
    
    args = parser.parse_args()
    
    main(
        args.set.upper(),
        outdir=args.outdir,
    )
