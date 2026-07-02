#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAMOS Step12d: Build Ensemble Stellar-Response Correction
=========================================================

Purpose
-------
Derive an empirical wavelength-dependent response correction for SAMOS
spectra by comparing flux-calibrated stellar continua against physically
plausible stellar spectral energy distributions.

This stage supersedes the earlier Step12c polynomial refinement approach.
Rather than fitting arbitrary smooth residual curves, Step12d derives a
physically motivated spectrograph response function from the ensemble of
stellar spectra in the field.

Method
------
For each slit with valid SkyMapper r/i/z photometry:

1. Read the Step11 flux-calibrated spectrum (FLUX_FLAM).
2. Read the corresponding continuum estimate derived in Step09.
3. Convert the Step09 continuum into physical flux units by matching its
   median level to the Step11 calibrated spectrum.
4. Fit a reddened blackbody model using:
       - effective temperature (Teff)
       - extinction (Av)
       - multiplicative scale factor
   constrained by the SkyMapper r/i/z photometry.
5. Compute the ratio:

       response(lambda) =
           blackbody_model(lambda) /
           observed_continuum(lambda)

6. Normalize the response curve near the i-band region.
7. Reject pathological solutions using percentile and RMS criteria.
8. Build an ensemble median stellar-response correction from all accepted
   stellar spectra.

Scientific Rationale
--------------------
The resulting response curve represents residual large-scale throughput
errors remaining after the nominal Step11 flux calibration. These include:

- wavelength-dependent slit losses,
- imperfect flat-field illumination structure,
- residual instrumental throughput curvature,
- continuum-shape systematics.

Because the response is derived from an ensemble of stars constrained by
physical stellar continua and external photometry, the correction is more
stable and astrophysically meaningful than a purely empirical polynomial
fit.

Inputs
------
Step09 continuum product:
    extract1d_optimal_ridge_all_wav_step09_abab_preferred_consensus.fits

Step11 flux-calibrated spectra:
    extract1d_fluxcal.fits

SkyMapper photometric catalog:
    slit_trace_radec_skymapper_all.csv

Outputs
-------
step12d_stellar_response_per_slit.fits
    Per-slit fitted continua, blackbody models, and response curves.

step12d_stellar_response_master.fits
    Ensemble median response correction.

step12d_stellar_response_summary.csv
    Per-slit fit statistics and acceptance flags.

step12d_stellar_response_metadata.json
    Processing metadata and provenance.

Notes
-----
- The trusted wavelength range is typically 590--960 nm.
- Step12d derives only the spectral-shape correction.
- Absolute normalization is handled later in Step12e.
- Spectra with pathological fits are excluded from the ensemble response.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.optimize import least_squares
from scipy.ndimage import median_filter, gaussian_filter1d

import config

H = 6.62607015e-27
C = 2.99792458e10
K = 1.380649e-16

LAM_EFF = {"r": 620.0, "i": 760.0, "z": 900.0}


def abmag_to_flam(mag_ab, lam_nm):
    fnu = 10 ** (-0.4 * (mag_ab + 48.60))
    lam_A = lam_nm * 10.0
    return fnu * 2.99792458e18 / lam_A**2


def bb_flam_shape(lam_nm, teff):
    lam_cm = lam_nm * 1e-7
    x = H * C / (lam_cm * K * teff)
    x = np.clip(x, 1e-6, 700)
    b_cm = (2 * H * C**2 / lam_cm**5) / np.expm1(x)
    return b_cm * 1e-8  # per Angstrom


def ccm89_k_lambda(lam_nm, rv=3.1):
    """Approx CCM89 optical/NIR A_lambda / A_V."""
    lam_um = lam_nm / 1000.0
    x = 1.0 / lam_um
    y = x - 1.82

    a = (1
         + 0.17699*y - 0.50447*y**2 - 0.02427*y**3
         + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6
         + 0.32999*y**7)

    b = (1.41338*y + 2.28305*y**2 + 1.07233*y**3
         - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6
         - 2.09002*y**7)

    return a + b / rv


def reddened_bb(lam_nm, teff, av, scale):
    bb = bb_flam_shape(lam_nm, teff)
    ext = 10 ** (-0.4 * av * ccm89_k_lambda(lam_nm))
    return scale * bb * ext


def fit_bb_av_scale(lam_nm, flam):
    lam_nm = np.asarray(lam_nm, float)
    flam = np.asarray(flam, float)

    m = np.isfinite(lam_nm) & np.isfinite(flam) & (flam > 0)
    lam_nm, flam = lam_nm[m], flam[m]

    if len(lam_nm) < 3:
        raise RuntimeError("Need at least 3 photometric points")

    def resid(p):
        logT, av, logS = p
        teff = 10**logT
        scale = 10**logS
        model = reddened_bb(lam_nm, teff, av, scale)
        return np.log10(model) - np.log10(flam)

    p0 = [np.log10(6000.0), 1.0, -35.0]
    bounds = ([np.log10(2500), 0.0, -80], [np.log10(30000), 10.0, 20])

    res = least_squares(resid, p0, bounds=bounds, max_nfev=5000)

    logT, av, logS = res.x
    return 10**logT, av, 10**logS, float(np.sqrt(np.mean(res.fun**2)))


def choose_continuum_column(names):
    for c in ["CONTINUUM_STEP09", "CONTINUUM_P1", "CONT2", "CONT1"]:
        if c in names:
            return c
    return None


def norm_median(y, lam, lo=730, hi=780):
    m = np.isfinite(lam) & np.isfinite(y) & (y > 0) & (lam >= lo) & (lam <= hi)
    if np.count_nonzero(m) < 5:
        return np.nan
    return float(np.nanmedian(y[m]))


def main():
    ap = argparse.ArgumentParser()
    
    ap.add_argument(
        "--step09",
        default=str(config.EXTRACT1D_STEP09_CONSENSUS)
    )
    
    default_fluxcal = config.EXTRACT1D_FLUXCAL
    
    ap.add_argument("--fluxcal", default=str(default_fluxcal))
    
    ap.add_argument("--photcat", default=str(config.STEP12_PHOTCAT))
    
    ap.add_argument(
        "--outdir",
        default=str(config.ST12_FINALCAL / "step12d_stellar_response")
    )
    
    ap.add_argument("--trusted-min", type=float, default=590.0)
    ap.add_argument("--trusted-max", type=float, default=960.0)
    
    args = ap.parse_args()

    step09 = Path(args.step09)
    fluxcal = Path(args.fluxcal)
    photcat = Path(args.photcat)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    phot = pd.read_csv(photcat)
    phot["slit"] = phot["slit"].astype(str).str.upper().str.strip()

    rows = []
    hdus = [fits.PrimaryHDU()]
    response_stack = []
    common_lam = np.arange(args.trusted_min, args.trusted_max + 0.25, 0.25)

    with fits.open(step09) as h9, fits.open(fluxcal) as hf:
        for _, r in phot.iterrows():
            slit = r["slit"]
            if not all(np.isfinite(r.get(f"{b}_mag", np.nan)) for b in ["r", "i", "z"]):
                continue
            if slit not in h9 or slit not in hf:
                continue

            d9 = h9[slit].data
            df = hf[slit].data

            cont_col = choose_continuum_column(d9.names)
            if cont_col is None:
                continue

            lam = np.asarray(df["LAMBDA_NM"], float)
            flux = np.asarray(df["FLUX_FLAM"], float)

            lam9 = np.asarray(d9["LAMBDA_NM"], float)
            cont9 = np.asarray(d9[cont_col], float)

            s9 = np.argsort(lam9)
            cont_interp = np.interp(lam, lam9[s9], cont9[s9], left=np.nan, right=np.nan)

            trusted = (
                np.isfinite(lam) & np.isfinite(flux) & np.isfinite(cont_interp) &
                (flux > 0) & (cont_interp > 0) &
                (lam >= args.trusted_min) & (lam <= args.trusted_max)
            )

            if np.count_nonzero(trusted) < 100:
                continue

            scale09 = np.nanmedian(flux[trusted] / cont_interp[trusted])
            cont_flam = cont_interp * scale09

            # Mild cleanup of continuum, not raw flux.
            cont_smooth = median_filter(cont_flam, size=31)
            cont_smooth = gaussian_filter1d(cont_smooth, sigma=7)

            lam_phot = np.array([LAM_EFF["r"], LAM_EFF["i"], LAM_EFF["z"]], float)
            flam_phot = np.array([
                abmag_to_flam(r["r_mag"], LAM_EFF["r"]),
                abmag_to_flam(r["i_mag"], LAM_EFF["i"]),
                abmag_to_flam(r["z_mag"], LAM_EFF["z"]),
            ])

            try:
                teff, av, scale, rmsdex = fit_bb_av_scale(lam_phot, flam_phot)
            except Exception:
                continue

            bb = reddened_bb(lam, teff, av, scale)
            resp = bb / cont_smooth

            nrm = norm_median(resp, lam)
            if not np.isfinite(nrm) or nrm <= 0:
                continue
            resp_norm = resp / nrm

            q = np.nanpercentile(resp_norm[trusted], [1, 50, 99])
            accept = (
                np.all(np.isfinite(q)) and
                q[0] > 0.05 and
                q[2] < 20.0 and
                rmsdex < 0.15
            )

            rows.append({
                "slit": slit,
                "cont_col": cont_col,
                "teff": teff,
                "av": av,
                "scale": scale,
                "rmsdex": rmsdex,
                "scale09": scale09,
                "resp_p01": q[0],
                "resp_med": q[1],
                "resp_p99": q[2],
                "accepted": accept,
            })

            cols = [
                fits.Column(name="LAMBDA_NM", array=lam, format="D"),
                fits.Column(name="FLUX_FLAM", array=flux, format="D"),
                fits.Column(name="CONT09_FLAM", array=cont_flam, format="D"),
                fits.Column(name="CONT09_SMOOTH", array=cont_smooth, format="D"),
                fits.Column(name="BB_MODEL", array=bb, format="D"),
                fits.Column(name="RESP_BB", array=resp_norm, format="D"),
            ]
            hdu = fits.BinTableHDU.from_columns(cols, name=slit)
            hdu.header["TEFF"] = float(teff)
            hdu.header["AV"] = float(av)
            hdu.header["RMSDEX"] = float(rmsdex)
            hdu.header["ACCEPT"] = int(accept)
            hdus.append(hdu)

            if accept:
            
                mresp = (
                    np.isfinite(lam) &
                    np.isfinite(resp_norm) &
                    (resp_norm > 0) &
                    (lam >= args.trusted_min) &
                    (lam <= args.trusted_max)
                )
            
                if np.count_nonzero(mresp) > 50:
                    order = np.argsort(lam[mresp])
                    lam_good = lam[mresp][order]
                    resp_good = resp_norm[mresp][order]
            
                    keep = np.concatenate(([True], np.diff(lam_good) > 0))
                    lam_good = lam_good[keep]
                    resp_good = resp_good[keep]
            
                    resp_interp = np.interp(
                        common_lam,
                        lam_good,
                        resp_good,
                        left=np.nan,
                        right=np.nan,
                    )
            
                    response_stack.append(resp_interp)
                    
                    
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "step12d_stellar_response_summary.csv", index=False)

    fits.HDUList(hdus).writeto(outdir / "step12d_stellar_response_per_slit.fits", overwrite=True)

    if response_stack:
        stack = np.vstack(response_stack)
        
        n_used = np.sum(np.isfinite(stack), axis=0)
        valid = n_used > 0
        
        master = np.full(common_lam.shape, np.nan)
        p16 = np.full(common_lam.shape, np.nan)
        p84 = np.full(common_lam.shape, np.nan)
        
        master[valid] = np.nanmedian(stack[:, valid], axis=0)
        p16[valid] = np.nanpercentile(stack[:, valid], 16, axis=0)
        p84[valid] = np.nanpercentile(stack[:, valid], 84, axis=0)

        cols = [
            fits.Column(name="LAMBDA_NM", array=common_lam, format="D"),
            fits.Column(name="RESP_MASTER", array=master, format="D"),
            fits.Column(name="RESP_P16", array=p16, format="D"),
            fits.Column(name="RESP_P84", array=p84, format="D"),
            fits.Column(name="N_USED", array=n_used.astype(float), format="D"),
        ]
        fits.HDUList([
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(cols, name="MASTER_RESPONSE")
        ]).writeto(outdir / "step12d_stellar_response_master.fits", overwrite=True)

    with open(outdir / "step12d_stellar_response_metadata.json", "w") as f:
        json.dump({
            "step09": str(step09),
            "fluxcal": str(fluxcal),
            "photcat": str(photcat),
            "n_rows": int(len(rows)),
            "n_accepted": int(summary["accepted"].sum()) if len(summary) else 0,
        }, f, indent=2)

    print("[OK] Wrote", outdir)


if __name__ == "__main__":
    main()