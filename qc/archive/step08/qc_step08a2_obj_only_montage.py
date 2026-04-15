#!/usr/bin/env python3

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

def main(set_tag="EVEN"):

    infile = Path(config.ST08_EXTRACT1D) / f"extract1d_optimal_ridge_{set_tag.lower()}.fits"
    outdir = Path(config.ST08_EXTRACT1D) / "qc_step08"
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
            obj = np.asarray(d["FLUX"], float)

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

    outpng = outdir / f"{set_tag.lower()}_obj_only_montage.png"

    fig.suptitle(f"{set_tag} OBJ (linear scale)", fontsize=16)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(outpng, dpi=160)
    plt.close(fig)

    print("Wrote:", outpng)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QC Step08a2 OBJ-only mosaic")
    parser.add_argument("--set", required=True, choices=["EVEN", "ODD", "even", "odd"])
    args = parser.parse_args()
    main(args.set.upper())