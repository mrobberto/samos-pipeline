#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:45:40 2026

@author: robberto

What you should see
In a correct behavior:
illumination: smooth, broad curve
OBJ_PRESKY: follows illumination trend
corrected curve:
flatter
less curvature
not exploding

If something is wrong
Case A — explosion at high Y

→ illumination too close to zero
→ need floor (as we discussed)

Case B — corrected curve still curved

→ illumination underfitted (sigma too small)

Case C — corrected curve overcorrected

→ illumination too aggressive (sigma too large)
"""
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import config

def plot_three_curves(hdul, slit_index=1, y0=800, y1=1700):
    hdu = hdul[slit_index]
    name = hdu.name

    x = hdu.data["LAMBDA_NM"].astype(float)
    y = hdu.data["YPIX"].astype(float)
    
    flux = hdu.data["FLUX_FLAM"].astype(float)
    flux_corr = hdu.data["FLUX_FLAM_ILLUMCORR"].astype(float)
    illum = hdu.data["ILLUM_PROFILE"].astype(float)
    
    good = (
        np.isfinite(x) &
        np.isfinite(flux) &
        np.isfinite(flux_corr) &
        np.isfinite(illum) & (illum > 0)
    )
    
    x = x[good]
    flux = flux[good]
    flux_corr = flux_corr[good]
    illum = illum[good]
    
    order = np.argsort(x)
    x = x[order]
    flux = flux[order]
    flux_corr = flux_corr[order]
    illum = illum[order]
    
    m = (x >= 650) & (x <= 850)
    
    flux_n = flux / np.nanmedian(flux[m])
    illum_n = illum / np.nanmedian(illum[m])
    corr_n = flux_corr / np.nanmedian(flux_corr[m])
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, flux_n, label="FLUX_FLAM")
    plt.plot(x, illum_n, label="ILLUM_PROFILE")
    plt.plot(x, corr_n, label="FLUX_FLAM_ILLUMCORR")
    
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized signal")
    plt.title(f"{name} — Step12b illumination diagnostic")
    plt.legend()
    plt.grid(alpha=0.3)
#    plt.yscale('log')
#    plt.yrange(0.001,100)
    plt.show()


# usage
"""
with fits.open(config.EXTRACT1D_ILLUMCORR) as hdul:
    plot_three_curves(hdul, slit_index=1)
"""