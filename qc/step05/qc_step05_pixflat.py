#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:23:00 2026

@author: robberto
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step05 QC — pixel flat validation

This script performs sanity checks and quick-look diagnostics for the
pixel flat products generated in Step05.

Outputs:
- histograms of pixflat values
- image panels (quartz_diff, illum2d, pixflat)
- summary statistics

Written to:
    config.ST05_PIXFLAT / "qc_step05_{even|odd}"

run from xterm, from samos-pipeline
PYTHONPATH=. python qc/step05/qc_step05_pixflat.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits

import config


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_mask(st04, trace_base):
    reg = st04 / f"{trace_base}_mask_reg.fits"
    if reg.exists():
        return fits.getdata(reg) > 0
    raw = st04 / f"{trace_base}_mask.fits"
    return fits.getdata(raw) > 0


def qc_one(trace_set):
    trace_set = trace_set.upper()
    suffix = trace_set.lower()

    st04 = Path(config.ST04_TRACES)
    st05 = Path(config.ST05_PIXFLAT)

    outdir = st05 / f"qc_step05_{suffix}"
    outdir.mkdir(parents=True, exist_ok=True)

    # files
    qfile = st05 / f"quartz_diff_{suffix}.fits"
    ifile = st05 / f"illum2d_{suffix}.fits"
    pfile = st05 / f"pixflat_{suffix}.fits"

    quartz = fits.getdata(qfile)
    illum2d = fits.getdata(ifile)
    pixflat = fits.getdata(pfile)

    trace_base = "Even_traces" if trace_set == "EVEN" else "Odd_traces"
    mask = load_mask(st04, trace_base)

    vals = pixflat[mask & np.isfinite(pixflat)]

    med = np.nanmedian(vals)
    std = np.nanstd(vals)
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)

    print(f"\n[{trace_set}] Pixflat stats:")
    print(f" median = {med:.4f}")
    print(f" std    = {std:.4f}")
    print(f" min/max= {vmin:.4f} / {vmax:.4f}")

    # -----------------------------------------------------------------------------
    # Histogram
    # -----------------------------------------------------------------------------
    plt.figure(figsize=(6,4))
    plt.hist(vals, bins=100)
    plt.axvline(1.0)
    plt.title(f"{trace_set} pixflat histogram")
    plt.xlabel("pixflat value")
    plt.ylabel("N")
    plt.tight_layout()
    plt.savefig(outdir / f"hist_{suffix}.png")
    plt.close()

    # -----------------------------------------------------------------------------
    # Image panels
    # -----------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15,5))

    ax[0].imshow(quartz, origin='lower')
    ax[0].set_title("quartz_diff")

    ax[1].imshow(illum2d, origin='lower')
    ax[1].set_title("illum2d")

    im = ax[2].imshow(pixflat, origin='lower')
    ax[2].set_title("pixflat")

    plt.colorbar(im, ax=ax[2], fraction=0.046)
    plt.tight_layout()
    plt.savefig(outdir / f"images_{suffix}.png")
    plt.close()

    # -----------------------------------------------------------------------------
    # Row medians (detect gradients)
    # -----------------------------------------------------------------------------
    row_med = np.nanmedian(pixflat, axis=1)

    plt.figure(figsize=(6,4))
    plt.plot(row_med)
    plt.axhline(1.0)
    plt.title(f"{trace_set} row median (pixflat)")
    plt.xlabel("row")
    plt.ylabel("median")
    plt.tight_layout()
    plt.savefig(outdir / f"row_median_{suffix}.png")
    plt.close()

    # -----------------------------------------------------------------------------
    # Save summary
    # -----------------------------------------------------------------------------
    txt = outdir / f"summary_{suffix}.txt"
    with open(txt, "w") as f:
        f.write(f"{trace_set} pixflat QC\n")
        f.write(f"median = {med:.6f}\n")
        f.write(f"std    = {std:.6f}\n")
        f.write(f"min    = {vmin:.6f}\n")
        f.write(f"max    = {vmax:.6f}\n")

    print(f"[DONE] QC written to {outdir}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    qc_one("EVEN")
    qc_one("ODD")