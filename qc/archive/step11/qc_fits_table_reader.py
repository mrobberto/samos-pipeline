#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:27:24 2026

@author: robberto
"""
from astropy.io import fits
import matplotlib.pyplot as plt

fn_old = "../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/11_fluxcal/Extract1D_fluxcal.fits"
fn_new = "../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/11_fluxcal/Extract1D_fluxcal_refined_master.fits"

h_old = fits.open(fn_old)
h_new = fits.open(fn_new)

for slit in ["SLIT000", "SLIT001", "SLIT012", "SLIT015"]:
    t_old = h_old[slit].data
    t_new = h_new[slit].data

    lam = t_old["LAMBDA_NM"]
    f_old = t_old["FLUX_FLAM"]
    f_new = t_new["FLUX_FLAM_REFINED"]

    plt.figure(figsize=(7,4))
    plt.plot(lam, f_old, label="Step11c", alpha=0.8)
    plt.plot(lam, f_new, label="Step11d", alpha=0.8)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux")
    plt.title(slit)
    plt.legend()
    plt.grid()
    plt.show()