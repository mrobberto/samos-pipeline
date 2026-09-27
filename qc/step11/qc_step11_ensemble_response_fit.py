#!/usr/bin/env python3
"""
QC-only ensemble broadband response fit for the SAMOS Step11/12 redesign.

This diagnostic derives a COMMON smooth multiplicative response-shape
correction directly from SkyMapper photometry, without fitting a stellar SED
or blackbody to each object.

Selection
---------
A slit contributes only if:
  * S08USE == 1
  * TELL_OK == 1
  * at least two SkyMapper r/i/z bands have finite photometry
  * the observed spectrum covers >= min_coverage of the corresponding
    photon-weighted passband response
  * the synthetic band integral is finite and positive

Model
-----
The common correction is parameterized as

    ln C(lambda) = a1 x + a2 x^2
    x = (lambda - lambda_i,pivot) / scale_nm

with C(lambda_i,pivot)=1 by construction.

For every star and usable band, the code computes the scalar S_b required to
match the catalog AB magnitude after applying C(lambda).  A correct common
response makes S_b independent of band for the same star.  The nuisance gray
normalization of each star is removed analytically by subtracting that star's
mean log-scale.

Three models are compared:
  gray      C(lambda)=1
  linear    ln C = a1 x
  quadratic ln C = a1 x + a2 x^2

The fit uses robust soft-L1 least squares.  Leave-one-star-out (LOO)
cross-validation is used to assess whether curvature genuinely generalizes.
For the quadratic model, the LOO coefficient distribution is also propagated
to a response-envelope CSV.

This is QC only.  No science spectra are modified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.optimize import least_squares

import config


C_AA_S = 2.99792458e18
AB_ZERO_JY = 3631.0


@dataclass
class BandDatum:
    band: str
    filt_lam_nm: np.ndarray
    filt_thr: np.ndarray
    target_fnu: float
    coverage: float


@dataclass
class StarDatum:
    slit: str
    lam_nm: np.ndarray
    flux: np.ndarray
    bands: list[BandDatum]


def load_filter(path: Path):
    a = np.genfromtxt(path, comments="#")
    if a.ndim != 2 or a.shape[1] < 2:
        raise RuntimeError(f"Invalid filter file: {path}")
    lam = np.asarray(a[:, 0], float)
    thr = np.asarray(a[:, 1], float)
    if np.nanmedian(lam) > 2000:
        lam = lam / 10.0
    good = np.isfinite(lam) & np.isfinite(thr)
    lam = lam[good]
    thr = np.clip(thr[good], 0.0, None)
    order = np.argsort(lam)
    return lam[order], thr[order]


def pivot_nm(lam_nm, thr):
    lam_nm = np.asarray(lam_nm, float)
    thr = np.asarray(thr, float)
    num = np.trapezoid(thr * lam_nm, lam_nm)
    den = np.trapezoid(thr / lam_nm, lam_nm)
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return np.nan
    return float(np.sqrt(num / den))


def photon_coverage(lam_min_nm, lam_max_nm, filt_lam_nm, filt_thr):
    lam_A = filt_lam_nm * 10.0
    weight = filt_thr / lam_A
    full = np.trapezoid(weight, lam_A)
    m = (filt_lam_nm >= lam_min_nm) & (filt_lam_nm <= lam_max_nm)
    if np.count_nonzero(m) < 2 or not np.isfinite(full) or full <= 0:
        return np.nan
    covered = np.trapezoid(weight[m], lam_A[m])
    return float(covered / full)


def target_fnu_cgs(mag_ab):
    return float(AB_ZERO_JY * 1e-23 * 10.0 ** (-0.4 * float(mag_ab)))


def correction(lam_nm, params, lambda0_nm, scale_nm):
    x = (np.asarray(lam_nm, float) - lambda0_nm) / scale_nm
    if len(params) == 0:
        ln_c = np.zeros_like(x)
    elif len(params) == 1:
        ln_c = params[0] * x
    else:
        ln_c = params[0] * x + params[1] * x * x
    return np.exp(ln_c)


def synthetic_fnu_per_gray(star, bd, params, lambda0_nm, scale_nm):
    lam = star.lam_nm
    flux = star.flux
    good = np.isfinite(lam) & np.isfinite(flux)
    if np.count_nonzero(good) < 10:
        return np.nan

    lam = lam[good]
    flux = flux[good]
    order = np.argsort(lam)
    lam = lam[order]
    flux = flux[order]

    thr = np.interp(
        lam,
        bd.filt_lam_nm,
        bd.filt_thr,
        left=0.0,
        right=0.0,
    )
    m = thr > 0
    if np.count_nonzero(m) < 10:
        return np.nan

    lam_A = lam[m] * 10.0
    rr = thr[m]
    ff = flux[m] * correction(lam[m], params, lambda0_nm, scale_nm)

    numer = np.trapezoid(ff * lam_A * rr, lam_A)
    denom = C_AA_S * np.trapezoid(rr / lam_A, lam_A)

    if (
        not np.isfinite(numer)
        or not np.isfinite(denom)
        or numer <= 0
        or denom <= 0
    ):
        return np.nan

    return float(numer / denom)


def star_band_log_scales(star, params, lambda0_nm, scale_nm):
    vals = []
    bands = []
    for bd in star.bands:
        model_fnu = synthetic_fnu_per_gray(
            star, bd, params, lambda0_nm, scale_nm
        )
        if not np.isfinite(model_fnu) or model_fnu <= 0:
            continue
        s = bd.target_fnu / model_fnu
        if not np.isfinite(s) or s <= 0:
            continue
        vals.append(np.log(s))
        bands.append(bd.band)
    return np.asarray(vals, float), bands


def residual_vector(params, stars, lambda0_nm, scale_nm):
    rr = []
    for star in stars:
        y, _ = star_band_log_scales(star, params, lambda0_nm, scale_nm)
        if len(y) < 2:
            continue
        y0 = np.mean(y)
        # Equalize the approximate total weight per star.
        rr.extend(((y - y0) / np.sqrt(len(y))).tolist())
    return np.asarray(rr, float)


def fit_model(stars, degree, lambda0_nm, scale_nm):
    if degree == 0:
        return np.zeros(0, float)
    p0 = np.zeros(degree, float)
    sol = least_squares(
        residual_vector,
        p0,
        args=(stars, lambda0_nm, scale_nm),
        loss="soft_l1",
        f_scale=0.10 / 1.0857362047581296,  # 0.10 mag in ln-flux
        max_nfev=400,
    )
    return np.asarray(sol.x, float)


def star_rms_mag(star, params, lambda0_nm, scale_nm):
    y, _ = star_band_log_scales(star, params, lambda0_nm, scale_nm)
    if len(y) < 2:
        return np.nan
    dm = 1.0857362047581296 * (y - np.mean(y))
    return float(np.sqrt(np.mean(dm * dm)))


def summarize_model(name, stars, params, lambda0_nm, scale_nm):
    vals = np.array(
        [star_rms_mag(s, params, lambda0_nm, scale_nm) for s in stars],
        float,
    )
    vals = vals[np.isfinite(vals)]
    print(
        f"{name:9s} in-sample star RMS: "
        f"N={len(vals)} median={np.nanmedian(vals):.4f} mag "
        f"[p16,p84]=[{np.nanpercentile(vals,16):.4f},"
        f"{np.nanpercentile(vals,84):.4f}]"
    )
    return vals


def loo_model(stars, degree, lambda0_nm, scale_nm):
    scores = []
    coeffs = []
    slits = []
    for i, star in enumerate(stars):
        train = stars[:i] + stars[i + 1 :]
        p = fit_model(train, degree, lambda0_nm, scale_nm)
        score = star_rms_mag(star, p, lambda0_nm, scale_nm)
        scores.append(score)
        coeffs.append(p)
        slits.append(star.slit)
    return (
        np.asarray(scores, float),
        coeffs,
        slits,
    )


def choose_flux_col(names):
    for c in [
        "FLUX_TELLCOR_O2",
        "STELLAR_CONSENSUS",
        "FLUX_APCORR",
        "FLUX",
    ]:
        if c in names:
            return c
    return None


def parse_args():
    p = argparse.ArgumentParser(
        description="QC ensemble SkyMapper response-shape fit"
    )
    p.add_argument("--infile", type=Path, required=True)
    p.add_argument(
        "--phot-csv",
        type=Path,
        default=Path(config.STEP11_PHOTCAT),
    )
    p.add_argument(
        "--filters-dir",
        type=Path,
        default=Path("calibration/reference_tables/filters"),
    )
    p.add_argument("--min-coverage", type=float, default=0.95)
    p.add_argument("--scale-nm", type=float, default=200.0)
    p.add_argument("--out-response-csv", type=Path, required=True)
    p.add_argument("--out-stars-csv", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    phot = pd.read_csv(args.phot_csv)
    phot = phot.copy()
    phot["slit"] = phot["slit"].astype(str).str.upper().str.strip()

    filters = {}
    pivots = {}
    for b in "riz":
        path = args.filters_dir / f"skymapper_{b}_nm.txt"
        filters[b] = load_filter(path)
        pivots[b] = pivot_nm(*filters[b])

    lambda0_nm = pivots["i"]

    stars = []
    selection_rows = []

    with fits.open(args.infile) as hdul:
        for hdu in hdul[1:]:
            slit = str(hdu.name or "").upper().strip()
            if not slit.startswith("SLIT") or hdu.data is None:
                continue

            s08use = int(hdu.header.get("S08USE", 0))
            tell_ok = bool(hdu.header.get("TELL_OK", False))
            if s08use != 1 or not tell_ok:
                continue

            names = list(hdu.columns.names)
            if "LAMBDA_NM" not in names:
                continue
            flux_col = choose_flux_col(names)
            if flux_col is None:
                continue

            lam = np.asarray(hdu.data["LAMBDA_NM"], float)
            flux = np.asarray(hdu.data[flux_col], float)
            good = np.isfinite(lam) & np.isfinite(flux)
            if np.count_nonzero(good) < 20:
                continue

            lam_good = lam[good]
            lo = float(np.nanmin(lam_good))
            hi = float(np.nanmax(lam_good))

            pr = phot.loc[phot["slit"] == slit]
            if len(pr) != 1:
                continue
            prow = pr.iloc[0]

            band_data = []
            sel = dict(slit=slit, flux_col=flux_col)

            for b in "riz":
                mag_col = f"{b}_mag"
                mag = (
                    pd.to_numeric(prow[mag_col], errors="coerce")
                    if mag_col in phot.columns
                    else np.nan
                )
                fl, ft = filters[b]
                cov = photon_coverage(lo, hi, fl, ft)

                usable = bool(
                    np.isfinite(mag)
                    and np.isfinite(cov)
                    and cov >= args.min_coverage
                )

                bd = None
                if usable:
                    candidate = BandDatum(
                        band=b,
                        filt_lam_nm=fl,
                        filt_thr=ft,
                        target_fnu=target_fnu_cgs(mag),
                        coverage=cov,
                    )
                    # Require a finite positive synthetic integral for C=1.
                    test_star = StarDatum(
                        slit=slit,
                        lam_nm=lam,
                        flux=flux,
                        bands=[candidate],
                    )
                    test = synthetic_fnu_per_gray(
                        test_star,
                        candidate,
                        np.zeros(0),
                        lambda0_nm,
                        args.scale_nm,
                    )
                    if np.isfinite(test) and test > 0:
                        bd = candidate
                        band_data.append(candidate)

                sel[f"cov_{b}"] = cov
                sel[f"use_{b}"] = int(bd is not None)

            sel["n_band"] = len(band_data)
            selection_rows.append(sel)

            if len(band_data) >= 2:
                stars.append(
                    StarDatum(
                        slit=slit,
                        lam_nm=lam,
                        flux=flux,
                        bands=band_data,
                    )
                )

    if len(stars) < 5:
        raise RuntimeError("Too few >=2-band stars for ensemble response fit.")

    print("Filter pivot wavelengths (nm):", pivots)
    print("Reference wavelength lambda0 =", f"{lambda0_nm:.4f}", "nm")
    print("Stars entering response fit:", len(stars))
    print(
        "Band-pattern counts:",
        pd.Series(
            ["".join(b.band for b in s.bands) for s in stars]
        ).value_counts().to_dict(),
    )

    params0 = fit_model(stars, 0, lambda0_nm, args.scale_nm)
    params1 = fit_model(stars, 1, lambda0_nm, args.scale_nm)
    params2 = fit_model(stars, 2, lambda0_nm, args.scale_nm)

    summarize_model("gray", stars, params0, lambda0_nm, args.scale_nm)
    summarize_model("linear", stars, params1, lambda0_nm, args.scale_nm)
    summarize_model("quadratic", stars, params2, lambda0_nm, args.scale_nm)

    loo0, _, loo_slits = loo_model(stars, 0, lambda0_nm, args.scale_nm)
    loo1, coeff1, _ = loo_model(stars, 1, lambda0_nm, args.scale_nm)
    loo2, coeff2, _ = loo_model(stars, 2, lambda0_nm, args.scale_nm)

    print()
    for name, arr in [
        ("gray", loo0),
        ("linear", loo1),
        ("quadratic", loo2),
    ]:
        q = arr[np.isfinite(arr)]
        print(
            f"{name:9s} LOO star RMS: "
            f"N={len(q)} median={np.nanmedian(q):.4f} mag "
            f"[p16,p84]=[{np.nanpercentile(q,16):.4f},"
            f"{np.nanpercentile(q,84):.4f}]"
        )

    print()
    print("Best-fit linear params:", params1.tolist())
    print("Best-fit quadratic params:", params2.tolist())

    def c_at(lam_nm, p):
        return float(
            correction(
                np.array([lam_nm]),
                p,
                lambda0_nm,
                args.scale_nm,
            )[0]
        )

    print("Quadratic correction at filter pivots:")
    for b in "riz":
        print(f"  {b}: lambda={pivots[b]:.3f} nm  C={c_at(pivots[b], params2):.4f}")

    coeff2_arr = np.array(
        [p for p in coeff2 if len(p) == 2 and np.all(np.isfinite(p))],
        float,
    )

    grid = np.arange(550.0, 1000.0 + 0.5, 0.5)
    master = correction(grid, params2, lambda0_nm, args.scale_nm)

    if len(coeff2_arr):
        stack = np.array(
            [
                correction(grid, p, lambda0_nm, args.scale_nm)
                for p in coeff2_arr
            ]
        )
        p16 = np.nanpercentile(stack, 16, axis=0)
        p84 = np.nanpercentile(stack, 84, axis=0)
    else:
        p16 = np.full_like(grid, np.nan)
        p84 = np.full_like(grid, np.nan)

    response = pd.DataFrame(
        dict(
            lambda_nm=grid,
            response_quadratic=master,
            loo_p16=p16,
            loo_p84=p84,
        )
    )
    args.out_response_csv.parent.mkdir(parents=True, exist_ok=True)
    response.to_csv(args.out_response_csv, index=False)

    star_rows = []
    for slit, r0, r1, r2 in zip(loo_slits, loo0, loo1, loo2):
        star = next(s for s in stars if s.slit == slit)
        star_rows.append(
            dict(
                slit=slit,
                bands="".join(b.band for b in star.bands),
                loo_rms_gray_mag=r0,
                loo_rms_linear_mag=r1,
                loo_rms_quadratic_mag=r2,
            )
        )
    pd.DataFrame(star_rows).to_csv(args.out_stars_csv, index=False)

    print("Wrote:", args.out_response_csv)
    print("Wrote:", args.out_stars_csv)
    print("QC only; no spectra were modified.")


if __name__ == "__main__":
    main()
