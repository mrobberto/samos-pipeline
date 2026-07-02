#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAMOS Step12e: Apply Ensemble Stellar-Response Correction
=========================================================

Purpose
-------
Apply the ensemble stellar-response correction derived in Step12d to all
flux-calibrated SAMOS spectra.

This stage produces the final refined science spectra:

    FLUX_FLAM_STELLARRESP

which represent the preferred final calibrated spectral products.

Method
------
1. Read the Step11 flux-calibrated spectra.
2. Read the ensemble median stellar-response curve derived in Step12d.
3. Interpolate the response correction onto each slit wavelength grid.
4. Apply the wavelength-dependent correction to:
       - FLUX_FLAM
       - VAR_FLAM2
5. Optionally apply a scalar photometric normalization using available
   SkyMapper photometry:
       preference order:
           i-band -> r-band -> z-band
6. Store the corrected spectra and provenance metadata.

Scientific Interpretation
-------------------------
The Step12d/e refinement sequence corrects residual continuum-shape
systematics remaining after Step11 flux calibration.

The correction is intended to improve:
- broad-band continuum realism,
- consistency with external photometry,
- large-scale spectrophotometric stability,
- relative spectral shape accuracy.

Importantly, the correction is ensemble-derived and physically motivated,
rather than an arbitrary polynomial adjustment.

Normalization Strategy
----------------------
If reliable photometry exists for a slit:

    synthetic photometry(corrected spectrum)
        -> matched to observed SkyMapper flux

using a scalar normalization factor.

If no reliable photometry exists:
- the spectral-shape correction is still applied,
- but no scalar normalization is performed.

These cases are flagged using:
    HAS_PHOTNORM
    NORM_BAND

Inputs
------
Step11 flux-calibrated spectra:
    extract1d_fluxcal.fits

Step12d master response:
    step12d_stellar_response_master.fits

SkyMapper photometric catalog:
    slit_trace_radec_skymapper_all.csv

Outputs
-------
extract1d_finalcal_stellarresp.fits

New FITS Columns
----------------
RESP_STELLAR_MASTER
    Ensemble response correction curve.

FLUX_FLAM_STELLARRESP
    Final stellar-response-corrected spectrum.

VAR_FLAM2_STELLARRESP
    Propagated variance spectrum.

NORM_STELLARRESP
    Scalar photometric normalization factor.

HAS_PHOTNORM
    1 if photometric normalization applied.

NORM_BAND
    Band used for normalization:
        I, R, Z, or NONE.

Notes
-----
- Outside the trusted wavelength range the response defaults to unity.
- Negative flux values are preserved and are considered valid after
  sky subtraction.
- This stage supersedes the earlier Step12c refinement approach.
"""
from pathlib import Path
import argparse
import numpy as np
from astropy.io import fits
import pandas as pd

import config

def synthetic_flux_density(lam_nm, flam, lam_eff_nm, width_nm=40.0):
    """
    Simple box synthetic photometry returning mean f_lambda.
    """

    m = (
        np.isfinite(lam_nm) &
        np.isfinite(flam) &
        (flam > 0) &
        (lam_nm >= lam_eff_nm - width_nm/2) &
        (lam_nm <= lam_eff_nm + width_nm/2)
    )

    if np.count_nonzero(m) < 5:
        return np.nan

    return float(np.nanmean(flam[m]))


def main():

    ap = argparse.ArgumentParser()

    default_input = config.EXTRACT1D_FLUXCAL

    ap.add_argument(
        "--input-fits",
        default=str(default_input)
    )

    ap.add_argument(
        "--master-response",
        default=str(
            config.ST12_FINALCAL /
            "step12d_stellar_response" /
            "step12d_stellar_response_master.fits"
        )
    )

    ap.add_argument(
        "--output-fits",
        default=str(
            config.ST12_FINALCAL /
            "extract1D_finalcal_stellarresp.fits"
        )
    )

    ap.add_argument("--trusted-min", type=float, default=590.0)
    ap.add_argument("--trusted-max", type=float, default=960.0)

    args = ap.parse_args()

    input_fits = Path(args.input_fits)
    master_fits = Path(args.master_response)
    output_fits = Path(args.output_fits)
    
    phot = pd.read_csv(config.STEP12_PHOTCAT)
    phot["slit"] = phot["slit"].astype(str).str.upper().str.strip()

    # ---------------------------------------------------------
    # Read master response
    # ---------------------------------------------------------

    with fits.open(master_fits) as hm:
        dm = hm["MASTER_RESPONSE"].data

        lam_master = np.asarray(dm["LAMBDA_NM"], float)
        resp_master = np.asarray(dm["RESP_MASTER"], float)

    mresp = (
        np.isfinite(lam_master) &
        np.isfinite(resp_master) &
        (resp_master > 0)
    )

    lam_master = lam_master[mresp]
    resp_master = resp_master[mresp]

    # ---------------------------------------------------------
    # Apply to all spectra
    # ---------------------------------------------------------

    hdus_out = []

    with fits.open(input_fits) as hin:

        # copy primary
        hdus_out.append(fits.PrimaryHDU(header=hin[0].header))

        for hdu in hin[1:]:

            if not isinstance(hdu, fits.BinTableHDU):
                hdus_out.append(hdu.copy())
                continue

            name = hdu.name

            if not name.startswith("SLIT"):
                hdus_out.append(hdu.copy())
                continue

            data = hdu.data
            cols = hdu.columns

            names = cols.names

            if "LAMBDA_NM" not in names:
                hdus_out.append(hdu.copy())
                continue

            if "FLUX_FLAM" not in names:
                hdus_out.append(hdu.copy())
                continue

            lam = np.asarray(data["LAMBDA_NM"], float)
            flux = np.asarray(data["FLUX_FLAM"], float)

            var = None
            if "VAR_FLAM2" in names:
                var = np.asarray(data["VAR_FLAM2"], float)

            # interpolate response
            resp = np.interp(
                lam,
                lam_master,
                resp_master,
                left=np.nan,
                right=np.nan
            )

            # outside trusted range → unity
            outside = (
                (lam < args.trusted_min) |
                (lam > args.trusted_max) |
                ~np.isfinite(resp) |
                (resp <= 0)
            )

            resp[outside] = 1.0

            # -------------------------------------------------
            # Apply master shape correction
            # -------------------------------------------------
            
            flux_shape = flux * resp
            
            if var is not None:
                var_shape = var * resp**2
            
            # -------------------------------------------------
            # Apply master shape correction to ALL pixels
            # -------------------------------------------------
            
            shape_factor = resp.copy()
            
            bad_resp = (
                ~np.isfinite(shape_factor) |
                (shape_factor <= 0)
            )
            
            shape_factor[bad_resp] = 1.0
            
            flux_shape = flux * shape_factor
            
            if var is not None:
                var_shape = var * shape_factor**2
            
            # -------------------------------------------------
            # Final scalar normalization to available photometry
            # Prefer i, then r, then z
            # -------------------------------------------------
            
            norm_scale = 1.0
            has_photnorm = False
            norm_band = "NONE"
            
            prow = phot.loc[phot["slit"] == name]
            
            band_info = {
                "i": (760.0, 40.0),
                "r": (620.0, 40.0),
                "z": (900.0, 40.0),
            }
            
            if len(prow) == 1:
                prow = prow.iloc[0]
            
                for band in ["i", "r", "z"]:
                    mag_col = f"{band}_mag"
            
                    if mag_col not in prow.index:
                        continue
            
                    if not np.isfinite(prow[mag_col]):
                        continue
            
                    lam_eff, width_nm = band_info[band]
                    mag = float(prow[mag_col])
            
                    f_phot = (
                        10 ** (-0.4 * (mag + 48.60))
                        * 2.99792458e18
                        / ((lam_eff * 10.0) ** 2)
                    )
            
                    f_syn = synthetic_flux_density(
                        lam,
                        flux_shape,
                        lam_eff,
                        width_nm=width_nm
                    )
            
                    if np.isfinite(f_syn) and (f_syn != 0):
                        norm_scale = f_phot / f_syn
                        has_photnorm = True
                        norm_band = band.upper()
                        break
            
            flux_corr = flux_shape * norm_scale
            
            if var is not None:
                var_corr = var_shape * norm_scale**2
                
            # -------------------------------------------------
            # append new columns
            # -------------------------------------------------

            newcols = list(cols)

            if "RESP_STELLAR_MASTER" not in names:
                newcols.append(
                    fits.Column(
                        name="RESP_STELLAR_MASTER",
                        array=resp.astype(np.float32),
                        format="E"
                    )
                )

            if "FLUX_FLAM_STELLARRESP" not in names:
                newcols.append(
                    fits.Column(
                        name="FLUX_FLAM_STELLARRESP",
                        array=flux_corr.astype(np.float32),
                        format="E"
                    )
                )

            if (var is not None) and ("VAR_FLAM2_STELLARRESP" not in names):
                newcols.append(
                    fits.Column(
                        name="VAR_FLAM2_STELLARRESP",
                        array=var_corr.astype(np.float32),
                        format="E"
                    )
                )

            if "NORM_STELLARRESP" not in names:
                newcols.append(
                    fits.Column(
                        name="NORM_STELLARRESP",
                        array=np.full_like(flux, norm_scale, dtype=np.float32),
                        format="E"
                    )
                )
                
            if "HAS_PHOTNORM" not in names:
                newcols.append(
                    fits.Column(
                        name="HAS_PHOTNORM",
                        array=np.full_like(flux, int(has_photnorm), dtype=np.int16),
                        format="I"
                    )
                )
                
            if "NORM_BAND" not in names:
                newcols.append(
                    fits.Column(
                        name="NORM_BAND",
                        array=np.full(len(flux), norm_band, dtype="S8"),
                        format="8A"
                    )
                )
    
            hnew = fits.BinTableHDU.from_columns(
                newcols,
                name=name
            )

            # preserve header
            for k, v in hdu.header.items():
                if k not in hnew.header:
                    hnew.header[k] = v

            hnew.header["HIERARCH STEP12E"] = True
            hnew.header["HIERARCH STEP12E RESP"] = "MASTER_STELLAR_RESPONSE"
            hnew.header["HIERARCH STEP12E NORM"] = (
                norm_band,
                "Band used for scalar photometric normalization"
            )
            hnew.header["HIERARCH STEP12E HASNORM"] = (int(has_photnorm), "1 if i-band photometric norm applied")
            hnew.header["HIERARCH STEP12E NORMBAND"] = norm_band

            hdus_out.append(hnew)

    fits.HDUList(hdus_out).writeto(
        output_fits,
        overwrite=True
    )

    print("[OK] Wrote", output_fits)


if __name__ == "__main__":
    main()
