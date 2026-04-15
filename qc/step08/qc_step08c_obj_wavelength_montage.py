#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import math
import re
from pathlib import Path
import config


TELLURIC_LINES = [686.7, 760.5]
TELLURIC_WINDOWS = [685.3, 687.9, 758.5, 762.5]

def slit_num(name):
    m = re.match(r"SLIT(\d+)", name.upper())
    return int(m.group(1)) if m else 9999

def main():

    infile = Path(config.ST08_EXTRACT1D) / "extract1d_optimal_ridge_all_wav.fits"
    outdir = Path(config.ST08_EXTRACT1D) / "qc_step08"
    outdir.mkdir(parents=True, exist_ok=True)

    slit_data = {}

    # --- read data ---
    with fits.open(infile) as hdul:
        for h in hdul[1:]:
            name = h.name.upper()
            if not name.startswith("SLIT"):
                continue

            d = h.data
            if d is None:
                continue

            cols = [c.upper() for c in d.names]
            if "FLUX" not in cols or "LAMBDA_NM" not in cols:
                continue

            lam = np.asarray(d["LAMBDA_NM"], float)
            flux = np.asarray(d["FLUX"], float)

            slit_data[name] = (lam, flux)

    slits = sorted(slit_data.keys(), key=slit_num)

    # --- global scaling ---
    all_flux = []
    all_lam = []

    for lam, flux in slit_data.values():
        good = np.isfinite(lam) & np.isfinite(flux)
        if np.any(good):
            all_flux.append(flux[good])
            all_lam.append(lam[good])

    if not all_flux:
        print("No valid data")
        return

    all_flux = np.concatenate(all_flux)
    all_lam = np.concatenate(all_lam)

    # robust limits
    y_lo, y_hi = np.nanpercentile(all_flux, [5, 95])
    x_lo, x_hi = np.nanpercentile(all_lam, [1, 99])

    # --- mosaic ---
    ncols = 6
    nrows = math.ceil(len(slits) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5*ncols, 2.5*nrows))
    axes = axes.ravel()

    for ax in axes:
        ax.set_visible(False)

    for ax, slit in zip(axes, slits):
        ax.set_visible(True)

        lam, flux = slit_data[slit]

        good = np.isfinite(lam) & np.isfinite(flux)
        if not np.any(good):
            continue

        ax.plot(lam, flux, 'k', lw=0.8)
        
        # Telluric guides
        for x in [686.7, 760.5]:
            ax.axvline(x, color='cyan', lw=0.8, ls='--', alpha=0.7)
        
        # Optional band edges
        for x in [685.3, 687.9, 758.5, 762.5]:
            ax.axvline(x, color='cyan', lw=0.4, ls=':', alpha=0.45)
        

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

        ax.set_title(slit, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        # zero line helps interpretation
        ax.axhline(0, color='0.7', lw=0.5)
        
        for x in TELLURIC_LINES:
            ax.axvline(x, color='cyan', lw=0.8, ls='--', alpha=0.7)
        
        for x in TELLURIC_WINDOWS:
            ax.axvline(x, color='cyan', lw=0.4, ls=':', alpha=0.45)
    

    outpng = outdir / "all_obj_wavelength_montage.png"

    fig.suptitle("OBJ spectra (wavelength space, global scale)", fontsize=16)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(outpng, dpi=160)
    plt.close(fig)

    print("Wrote:", outpng)


if __name__ == "__main__":
    main()