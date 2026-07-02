"""
QC Step12b — Illumination correction diagnostic.

This utility visualizes the effect of the Step12b illumination correction
on individual slit spectra. It is intended as a low-level sanity check
to verify that the empirical illumination profile derived in Step12a is
being applied consistently and without introducing artifacts.

Overview
--------
For a selected slit, the script compares three quantities as a function
of detector row (YPIX):

  1) FLUX_FLAM
     The input flux-calibrated spectrum (after Step11), still affected by
     large-scale instrumental response.

  2) ILLUM_PROFILE
     The empirical illumination profile derived from quartz flats (Step12a),
     representing the wavelength-dependent throughput of the instrument.

  3) FLUX_FLAM_ILLUMCORR
     The corrected spectrum obtained by dividing the input spectrum by the
     illumination profile (Step12b).

Two visualization modes are provided:

  - Normalized view:
      All curves are normalized over a user-defined Y range to compare shapes.
      This highlights how the illumination correction removes large-scale
      curvature from the spectrum.

  - Raw (absolute) view:
      The original and corrected spectra are plotted in physical units,
      together with a scaled version of the illumination profile. This
      confirms that the correction is purely multiplicative.

Interpretation
--------------
A successful illumination correction should show:

  - The illumination profile tracing the large-scale shape of the raw spectrum
  - The corrected spectrum (FLUX_FLAM_ILLUMCORR) being significantly flatter
  - No oscillatory behavior or discontinuities introduced by the correction
  - Consistent behavior across the full Y range (excluding low-S/N edges)

This diagnostic is particularly useful for:

  - Verifying profile orientation consistency (no hidden reversals)
  - Identifying problematic slits or edge effects
  - Validating the normalization region used in Step12a

Notes
-----
- The illumination profile is not a true flat field; it includes the spectral
  energy distribution of the quartz lamp and represents an empirical
  approximation to the instrumental response.

- The correction is performed in detector coordinates (YPIX) and must be
  consistent with the wavelength solution and extraction geometry.

- This script is intended for interactive inspection and is not part of the
  automated QC pipeline.
"""
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import config

def plot_three_curves(hdul, slit_index=1, y0=500, y1=2000):
    hdu = hdul[slit_index]
    name = hdu.name

    y = hdu.data["YPIX"].astype(float)
    flux = hdu.data["FLUX_FLAM"].astype(float)
    flux_corr = hdu.data["FLUX_FLAM_ILLUMCORR"].astype(float)
    illum = hdu.data["ILLUM_PROFILE"].astype(float)

    good = (
        np.isfinite(y) &
        np.isfinite(flux) & (flux > 0) &
        np.isfinite(flux_corr) & (flux_corr > 0) &
        np.isfinite(illum) & (illum > 0)
    )

    y = y[good]
    flux = flux[good]
    flux_corr = flux_corr[good]
    illum = illum[good]

    m = (y >= y0) & (y <= y1)

    flux_n = flux / np.nanmedian(flux[m])
    illum_n = illum / np.nanmedian(illum[m])
    corr_n = flux_corr / np.nanmedian(flux_corr[m])

    plt.figure(figsize=(10, 5))
    plt.plot(y, illum_n, label="illumination")
    plt.plot(y, flux_n, label="FLUX_FLAM")
    plt.plot(y, corr_n, label="FLUX_FLAM / illumination")

    plt.yscale("log")
    plt.xlabel("Detector row Y")
    plt.ylabel("Normalized signal")
    plt.title(f"{name} — Step12b diagnostic")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.show()

def plot_three_curves_raw(hdul, slit_index=1):
    hdu = hdul[slit_index]
    name = hdu.name
    
    y = hdu.data["YPIX"]
    flux = hdu.data["FLUX_FLAM"]
    flux_corr = hdu.data["FLUX_FLAM_ILLUMCORR"]
    illum = hdu.data["ILLUM_PROFILE"]

    good = (
        np.isfinite(y) &
        np.isfinite(flux) & (flux > 0) &
        np.isfinite(flux_corr) & (flux_corr > 0) &
        np.isfinite(illum) & (illum > 0)
    )

    plt.figure(figsize=(10,5))

    plt.plot(y[good], flux[good], label="FLUX_FLAM")
    plt.plot(y[good], flux_corr[good], label="FLUX_FLAM_ILLUMCORR")
    plt.plot(y[good], illum[good]*np.nanmedian(flux[good]), label="illum (scaled)")

    plt.yscale("log")
    plt.legend()
    plt.title(name)
    plt.show()
    print(illum[good])
# usage

def slit_index_from_name(hdul, slitname):
    for i, hdu in enumerate(hdul):
        if hdu.name == slitname:
            return i
    raise KeyError(f"{slitname} not found")

with fits.open(config.EXTRACT1D_ILLUMCORR) as hdul:
    idx = slit_index_from_name(hdul, "SLIT025")  # change this
    #plot_three_curves(hdul, slit_index=idx, y0=500, y1=2000)
    plot_three_curves_raw(hdul, slit_index=idx)