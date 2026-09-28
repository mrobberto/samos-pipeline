#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step12d — build the adopted ensemble spectrophotometric response
================================================================

This stage converts the validated Step11 ensemble shape correction into the
production Step12 response.

The response shape is supplied by the validated quadratic ensemble solution
(normalized at the SkyMapper i-band pivot).  Step12d then derives ONE global
normalization factor from the Step11 spectra themselves so that, on average,
the correction preserves the synthetic SkyMapper i-band flux.

Thus the production response is

    R_prod(lambda) = C_i * R_shape(lambda)

where C_i is the median, over usable spectra, of

    <f_nu>_i,Step11 / <f_nu>_i,Step11*R_shape .

No per-object catalog normalization is performed here or in Step12e.

Outputs
-------
step12d_stellar_response_master.fits
step12d_stellar_response_summary.csv
step12d_stellar_response_metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.optimize import least_squares

import config

log = logging.getLogger("step12d_build_stellar_response")

C_A_S = 2.99792458e18  # Angstrom/s
AB_ZERO_JY = 3631.0

PIVOT_I_NM = 776.79762950059
PIVOT_Z_NM = 914.5992987637427


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


def load_response_filter(path: Path):
    """Load a SkyMapper response curve using the validated QC convention."""
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
    lam_A = np.asarray(filt_lam_nm, float) * 10.0
    thr = np.asarray(filt_thr, float)

    weight = thr / lam_A
    full = np.trapezoid(weight, lam_A)

    m = (
        (filt_lam_nm >= lam_min_nm)
        & (filt_lam_nm <= lam_max_nm)
    )

    if (
        np.count_nonzero(m) < 2
        or not np.isfinite(full)
        or full <= 0
    ):
        return np.nan

    covered = np.trapezoid(weight[m], lam_A[m])
    return float(covered / full)


def target_fnu_cgs(mag_ab):
    return float(
        AB_ZERO_JY
        * 1e-23
        * 10.0 ** (-0.4 * float(mag_ab))
    )


def correction(lam_nm, params, lambda0_nm, scale_nm):
    x = (
        np.asarray(lam_nm, float) - lambda0_nm
    ) / scale_nm

    if len(params) == 0:
        ln_c = np.zeros_like(x)
    elif len(params) == 1:
        ln_c = params[0] * x
    else:
        raise ValueError(
            "Production Step12d supports only gray or linear response."
        )

    return np.exp(ln_c)


def synthetic_fnu_per_gray(
    star,
    bd,
    params,
    lambda0_nm,
    scale_nm,
):
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

    ff = (
        flux[m]
        * correction(
            lam[m],
            params,
            lambda0_nm,
            scale_nm,
        )
    )

    numer = np.trapezoid(
        ff * lam_A * rr,
        lam_A,
    )
    denom = (
        C_A_S
        * np.trapezoid(
            rr / lam_A,
            lam_A,
        )
    )

    if (
        not np.isfinite(numer)
        or not np.isfinite(denom)
        or numer <= 0
        or denom <= 0
    ):
        return np.nan

    return float(numer / denom)


def star_band_log_scales(
    star,
    params,
    lambda0_nm,
    scale_nm,
):
    vals = []
    bands = []

    for bd in star.bands:
        model_fnu = synthetic_fnu_per_gray(
            star,
            bd,
            params,
            lambda0_nm,
            scale_nm,
        )

        if not np.isfinite(model_fnu) or model_fnu <= 0:
            continue

        scale = bd.target_fnu / model_fnu

        if not np.isfinite(scale) or scale <= 0:
            continue

        vals.append(np.log(scale))
        bands.append(bd.band)

    return np.asarray(vals, float), bands


def residual_vector(
    params,
    stars,
    lambda0_nm,
    scale_nm,
):
    rr = []

    for star in stars:
        y, _ = star_band_log_scales(
            star,
            params,
            lambda0_nm,
            scale_nm,
        )

        if len(y) < 2:
            continue

        y0 = np.mean(y)

        # Same weighting convention as the validated QC fitter.
        rr.extend(
            (
                (y - y0)
                / np.sqrt(len(y))
            ).tolist()
        )

    return np.asarray(rr, float)


def fit_model(
    stars,
    degree,
    lambda0_nm,
    scale_nm,
):
    if degree == 0:
        return np.zeros(0, float)

    if degree != 1:
        raise ValueError(
            "Production Step12d adopts only the linear i-z model."
        )

    p0 = np.zeros(degree, float)

    sol = least_squares(
        residual_vector,
        p0,
        args=(
            stars,
            lambda0_nm,
            scale_nm,
        ),
        loss="soft_l1",
        f_scale=0.10 / 1.0857362047581296,
        max_nfev=400,
    )

    return np.asarray(sol.x, float)


def star_rms_mag(
    star,
    params,
    lambda0_nm,
    scale_nm,
):
    y, _ = star_band_log_scales(
        star,
        params,
        lambda0_nm,
        scale_nm,
    )

    if len(y) < 2:
        return np.nan

    dm = (
        1.0857362047581296
        * (y - np.mean(y))
    )

    return float(
        np.sqrt(
            np.mean(dm * dm)
        )
    )


def loo_model(
    stars,
    degree,
    lambda0_nm,
    scale_nm,
):
    scores = []
    coeffs = []
    slits = []

    for i, star in enumerate(stars):
        train = (
            stars[:i]
            + stars[i + 1 :]
        )

        params = fit_model(
            train,
            degree,
            lambda0_nm,
            scale_nm,
        )

        score = star_rms_mag(
            star,
            params,
            lambda0_nm,
            scale_nm,
        )

        scores.append(score)
        coeffs.append(params)
        slits.append(star.slit)

    return (
        np.asarray(scores, float),
        coeffs,
        slits,
    )


def _stats(arr):
    x = np.asarray(arr, float)
    x = x[np.isfinite(x)]

    if not len(x):
        return {
            "median": np.nan,
            "p16": np.nan,
            "p84": np.nan,
        }

    return {
        "median": float(np.median(x)),
        "p16": float(np.percentile(x, 16)),
        "p84": float(np.percentile(x, 84)),
    }


def build_linear_iz_shape(
    spectra_path: Path,
    phot_csv: Path,
    filter_i_path: Path,
    filter_z_path: Path,
    min_coverage: float = 0.95,
    scale_nm: float = 200.0,
):
    """
    Derive the common Step12 wavelength-dependent response shape.

    Only SkyMapper i and z are used.  SkyMapper r is deliberately excluded
    because a substantial fraction of that passband lies below the formal
    600-nm SAMOS cutoff.

    The response is normalized at the i-band pivot:

        ln C(lambda) = a1 * (lambda - lambda_i) / scale_nm

    with no quadratic term.
    """

    phot = pd.read_csv(phot_csv).copy()

    if "slit" not in phot.columns:
        raise KeyError(
            f"Photometry table has no 'slit' column: {phot_csv}"
        )

    phot["slit"] = (
        phot["slit"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    filters = {
        "i": load_response_filter(filter_i_path),
        "z": load_response_filter(filter_z_path),
    }

    pivots = {
        b: pivot_nm(*filters[b])
        for b in ("i", "z")
    }

    lambda0_nm = pivots["i"]

    if not np.isfinite(lambda0_nm):
        raise RuntimeError("Could not determine SkyMapper i pivot.")

    stars = []

    with fits.open(spectra_path, memmap=False) as hdul:
        for hdu in hdul[1:]:
            slit = str(
                hdu.name or ""
            ).upper().strip()

            if (
                not slit.startswith("SLIT")
                or hdu.data is None
            ):
                continue

            s08use = int(
                hdu.header.get(
                    "S08USE",
                    0,
                )
            )
            tell_ok = bool(
                hdu.header.get(
                    "TELL_OK",
                    False,
                )
            )

            if s08use != 1 or not tell_ok:
                continue

            names = list(hdu.columns.names)

            if "LAMBDA_NM" not in names:
                continue

            # Production shape derivation is pinned to the canonical
            # telluric-corrected spectrum.  Do not silently select some
            # other historical flux column.
            if "FLUX_TELLCOR_O2" not in names:
                continue

            lam = np.asarray(
                hdu.data["LAMBDA_NM"],
                float,
            )
            flux = np.asarray(
                hdu.data["FLUX_TELLCOR_O2"],
                float,
            )

            good = (
                np.isfinite(lam)
                & np.isfinite(flux)
            )

            if np.count_nonzero(good) < 20:
                continue

            lam_good = lam[good]
            lo = float(
                np.nanmin(lam_good)
            )
            hi = float(
                np.nanmax(lam_good)
            )

            pr = phot.loc[
                phot["slit"] == slit
            ]

            if len(pr) != 1:
                continue

            prow = pr.iloc[0]
            band_data = []

            for b in ("i", "z"):
                mag_col = f"{b}_mag"

                mag = (
                    pd.to_numeric(
                        prow[mag_col],
                        errors="coerce",
                    )
                    if mag_col in phot.columns
                    else np.nan
                )

                fl, ft = filters[b]

                cov = photon_coverage(
                    lo,
                    hi,
                    fl,
                    ft,
                )

                usable = bool(
                    np.isfinite(mag)
                    and np.isfinite(cov)
                    and cov >= min_coverage
                )

                if not usable:
                    continue

                candidate = BandDatum(
                    band=b,
                    filt_lam_nm=fl,
                    filt_thr=ft,
                    target_fnu=target_fnu_cgs(
                        mag
                    ),
                    coverage=cov,
                )

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
                    scale_nm,
                )

                if np.isfinite(test) and test > 0:
                    band_data.append(
                        candidate
                    )

            # The adopted production response is specifically an i-z color
            # constraint.  Both bands are required.
            if len(band_data) == 2:
                stars.append(
                    StarDatum(
                        slit=slit,
                        lam_nm=lam,
                        flux=flux,
                        bands=band_data,
                    )
                )

    if len(stars) < 5:
        raise RuntimeError(
            "Too few usable i-z stars for ensemble response fit."
        )

    params0 = fit_model(
        stars,
        0,
        lambda0_nm,
        scale_nm,
    )
    params1 = fit_model(
        stars,
        1,
        lambda0_nm,
        scale_nm,
    )

    insample0 = np.asarray(
        [
            star_rms_mag(
                star,
                params0,
                lambda0_nm,
                scale_nm,
            )
            for star in stars
        ],
        float,
    )

    insample1 = np.asarray(
        [
            star_rms_mag(
                star,
                params1,
                lambda0_nm,
                scale_nm,
            )
            for star in stars
        ],
        float,
    )

    loo0, _, loo_slits = loo_model(
        stars,
        0,
        lambda0_nm,
        scale_nm,
    )

    loo1, coeff1, _ = loo_model(
        stars,
        1,
        lambda0_nm,
        scale_nm,
    )

    coeff1_arr = np.asarray(
        [
            p
            for p in coeff1
            if (
                len(p) == 1
                and np.all(
                    np.isfinite(p)
                )
            )
        ],
        float,
    )

    # Reproduce the validated QC response grid first, then trim to the
    # scientifically trusted i-pivot to z-pivot interval.
    full_grid = np.arange(
        550.0,
        1000.0 + 0.5,
        0.5,
    )

    full_shape = correction(
        full_grid,
        params1,
        lambda0_nm,
        scale_nm,
    )

    if len(coeff1_arr):
        stack = np.asarray(
            [
                correction(
                    full_grid,
                    p,
                    lambda0_nm,
                    scale_nm,
                )
                for p in coeff1_arr
            ]
        )

        full_p16 = np.nanpercentile(
            stack,
            16,
            axis=0,
        )
        full_p84 = np.nanpercentile(
            stack,
            84,
            axis=0,
        )
    else:
        full_p16 = np.full_like(
            full_grid,
            np.nan,
        )
        full_p84 = np.full_like(
            full_grid,
            np.nan,
        )

    raw = pd.DataFrame(
        {
            "lambda_nm": full_grid,
            "response_linear": full_shape,
            "loo_p16": full_p16,
            "loo_p84": full_p84,
        }
    )

    trust_lo = float(
        pivots["i"]
    )
    trust_hi = float(
        pivots["z"]
    )

    inside = raw[
        (raw["lambda_nm"] >= trust_lo)
        & (raw["lambda_nm"] <= trust_hi)
    ].copy()

    edge_rows = []

    for w in (
        trust_lo,
        trust_hi,
    ):
        # Evaluate trusted endpoints directly from the fitted model.
        # In particular, the i-pivot normalization must be exactly C(i)=1,
        # rather than inheriting sub-ppm interpolation error from the
        # auxiliary 0.5-nm response grid.
        edge_shape = float(
            correction(
                np.array([w]),
                params1,
                lambda0_nm,
                scale_nm,
            )[0]
        )

        if len(coeff1_arr):
            edge_stack = np.asarray(
                [
                    correction(
                        np.array([w]),
                        coeff,
                        lambda0_nm,
                        scale_nm,
                    )[0]
                    for coeff in coeff1_arr
                ],
                float,
            )
            edge_p16 = float(
                np.nanpercentile(edge_stack, 16)
            )
            edge_p84 = float(
                np.nanpercentile(edge_stack, 84)
            )
        else:
            edge_p16 = np.nan
            edge_p84 = np.nan

        edge_rows.append(
            {
                "lambda_nm": w,
                "response_linear": edge_shape,
                "loo_p16": edge_p16,
                "loo_p84": edge_p84,
            }
        )

    master = pd.concat(
        [
            inside,
            pd.DataFrame(edge_rows),
        ],
        ignore_index=True,
    )

    master = (
        master
        .sort_values("lambda_nm")
        .drop_duplicates("lambda_nm")
        .reset_index(drop=True)
    )

    for c in (
        "response_linear",
        "loo_p16",
        "loo_p84",
    ):
        x = master[c].to_numpy(float)

        if not np.all(np.isfinite(x)):
            raise RuntimeError(
                f"Non-finite values in production {c}"
            )

        if np.any(x <= 0):
            raise RuntimeError(
                f"Non-positive values in production {c}"
            )

    info = {
        "n_stars": len(stars),
        "slits": [
            s.slit
            for s in stars
        ],
        "loo_slits": loo_slits,
        "pivot_i_nm": float(
            pivots["i"]
        ),
        "pivot_z_nm": float(
            pivots["z"]
        ),
        "linear_a1": float(
            params1[0]
        ),
        "scale_nm": float(
            scale_nm
        ),
        "min_coverage": float(
            min_coverage
        ),
        "insample_gray": _stats(
            insample0
        ),
        "insample_linear": _stats(
            insample1
        ),
        "loo_gray": _stats(
            loo0
        ),
        "loo_linear": _stats(
            loo1
        ),
    }

    return master, info


def load_filter_curve(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals = []
            for p in s.replace(",", " ").split():
                try:
                    vals.append(float(p))
                except ValueError:
                    continue
            if len(vals) >= 2:
                rows.append((vals[0], vals[1]))

    if len(rows) < 3:
        raise RuntimeError(f"Could not read two-column filter curve: {path}")

    a = np.asarray(rows, float)
    w = a[:, 0]
    t = a[:, 1]

    med = float(np.nanmedian(w))
    if med > 3000:      # Angstrom -> nm
        w = w / 10.0
    elif med < 10:      # micron -> nm
        w = w * 1000.0

    good = np.isfinite(w) & np.isfinite(t) & (t >= 0)
    w, t = w[good], t[good]
    order = np.argsort(w)
    return w[order], t[order]


def synthetic_fnu_and_coverage(lam_nm, flam_A, filt_nm, filt_t):
    lam = np.asarray(lam_nm, float)
    flam = np.asarray(flam_A, float)

    m = np.isfinite(lam) & np.isfinite(flam)
    if m.sum() < 2:
        return np.nan, 0.0

    lam = lam[m]
    flam = flam[m]
    order = np.argsort(lam)
    lam, flam = lam[order], flam[order]
    lam, idx = np.unique(lam, return_index=True)
    flam = flam[idx]

    fw = np.asarray(filt_nm, float)
    ft = np.asarray(filt_t, float)
    positive = np.isfinite(fw) & np.isfinite(ft) & (ft > 0)
    fw, ft = fw[positive], ft[positive]

    if fw.size < 3 or lam.size < 2:
        return np.nan, 0.0

    lamA_f = fw * 10.0
    denom_full = np.trapezoid(ft / lamA_f, lamA_f)
    if not np.isfinite(denom_full) or denom_full <= 0:
        return np.nan, 0.0

    overlap = (fw >= lam.min()) & (fw <= lam.max())
    if overlap.sum() < 3:
        return np.nan, 0.0

    fw_o = fw[overlap]
    ft_o = ft[overlap]
    lamA_o = fw_o * 10.0
    denom_cov = np.trapezoid(ft_o / lamA_o, lamA_o)
    coverage = float(denom_cov / denom_full)

    f_interp = np.interp(fw_o, lam, flam)
    numerator = np.trapezoid(f_interp * ft_o * lamA_o, lamA_o)

    if not np.isfinite(numerator) or denom_cov <= 0:
        return np.nan, coverage

    return float(numerator / (C_A_S * denom_cov)), coverage


def _response_on_grid(lam_nm, master_lam, master_resp):
    lam_nm = np.asarray(lam_nm, float)
    out = np.full(lam_nm.shape, np.nan, dtype=float)
    good = np.isfinite(lam_nm)
    if good.any():
        out[good] = np.interp(
            lam_nm[good],
            master_lam,
            master_resp,
            left=float(master_resp[0]),
            right=float(master_resp[-1]),
        )
    return out


def derive_global_i_normalization(
    spectra_path: Path,
    master_lam: np.ndarray,
    shape_resp: np.ndarray,
    filter_i_path: Path,
    min_coverage: float = 0.90,
):
    """
    Derive one global scalar that preserves Step11 synthetic i-band flux,
    in the median over usable slit spectra.
    """
    fw, ft = load_filter_curve(filter_i_path)
    rows = []

    with fits.open(spectra_path, memmap=False) as h:
        for ext in h[1:]:
            if not (ext.name or "").upper().startswith("SLIT"):
                continue
            if ext.data is None:
                continue

            # Independent source-validity gate. Step12d remains protected
            # even if an upstream file contains rejected slit HDUs.
            s08use = int(ext.header.get("S08USE", 0))
            s08good = int(ext.header.get("S08GOOD", s08use))
            s08clas = str(ext.header.get("S08CLAS", "")).strip().upper()

            fcaluse = int(ext.header.get("FCALUSE", 0))
            fluxcal = str(
                ext.header.get("FLUXCAL", "")
            ).strip().upper()

            if (
                s08use != 1
                or s08good != 1
                or s08clas in {"EMPTY", "NOSEED"}
                or fcaluse != 1
                or fluxcal != "I_ANCHOR"
            ):
                continue

            names = set(ext.columns.names)
            if "LAMBDA_NM" not in names or "FLUX_FLAM" not in names:
                continue

            lam = np.asarray(ext.data["LAMBDA_NM"], float)
            flux = np.asarray(ext.data["FLUX_FLAM"], float)
            resp = _response_on_grid(lam, master_lam, shape_resp)
            shaped = flux * resp

            fpre, c0 = synthetic_fnu_and_coverage(lam, flux, fw, ft)
            fshp, c1 = synthetic_fnu_and_coverage(lam, shaped, fw, ft)

            if (
                c0 >= min_coverage
                and c1 >= min_coverage
                and np.isfinite(fpre)
                and np.isfinite(fshp)
                and fpre > 0
                and fshp > 0
            ):
                rows.append((ext.name.upper(), float(fpre / fshp)))

    if not rows:
        raise RuntimeError("No spectra usable for global i-band response normalization.")

    ratios = np.asarray([x[1] for x in rows], float)
    factor = float(np.median(ratios))
    p16, p84 = np.percentile(ratios, [16, 84])

    return factor, float(p16), float(p84), rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build the Step12d production linear i-z ensemble response."
        )
    )

    ap.add_argument(
        "--shape-spectra",
        type=str,
        default="",
        help=(
            "Telluric-corrected spectra used to derive the common i-z shape. "
            "Default: config.EXTRACT1D_TELLCOR."
        ),
    )

    ap.add_argument(
        "--phot-csv",
        type=str,
        default="",
        help=(
            "SkyMapper photometry table. "
            "Default: config.STEP11_PHOTCAT."
        ),
    )

    ap.add_argument(
        "--spectra",
        type=str,
        default="",
        help=(
            "Step11 flux-calibrated spectra used for the global i normalization. "
            "Default: config.EXTRACT1D_FLUXCAL."
        ),
    )

    ap.add_argument(
        "--out-fits",
        type=str,
        default="",
    )

    ap.add_argument(
        "--summary-csv",
        type=str,
        default="",
    )

    ap.add_argument(
        "--metadata-json",
        type=str,
        default="",
    )

    ap.add_argument(
        "--min-shape-coverage",
        type=float,
        default=0.95,
    )

    ap.add_argument(
        "--scale-nm",
        type=float,
        default=200.0,
    )

    ap.add_argument(
        "--min-i-coverage",
        type=float,
        default=0.90,
    )

    return ap.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)-8s | "
            "%(name)s | %(message)s"
        ),
    )

    args = parse_args()

    shape_spectra = (
        Path(args.shape_spectra)
        if args.shape_spectra
        else Path(config.EXTRACT1D_TELLCOR)
    )

    phot_csv = (
        Path(args.phot_csv)
        if args.phot_csv
        else Path(config.STEP11_PHOTCAT)
    )

    spectra_path = (
        Path(args.spectra)
        if args.spectra
        else Path(config.EXTRACT1D_FLUXCAL)
    )

    out_fits = (
        Path(args.out_fits)
        if args.out_fits
        else Path(config.STEP12D_MASTER_FITS)
    )

    summary_csv = (
        Path(args.summary_csv)
        if args.summary_csv
        else Path(config.STEP12D_SUMMARY_CSV)
    )

    metadata_json = (
        Path(args.metadata_json)
        if args.metadata_json
        else Path(config.STEP12D_METADATA_JSON)
    )

    filter_i = Path(config.FILTER_I)
    filter_z = Path(config.FILTER_Z)

    for inp in (
        shape_spectra,
        phot_csv,
        spectra_path,
        filter_i,
        filter_z,
    ):
        if not inp.exists():
            raise FileNotFoundError(inp)

    for out in (
        out_fits,
        summary_csv,
        metadata_json,
    ):
        out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    log.info(
        "Shape spectra: %s",
        shape_spectra,
    )
    log.info(
        "SkyMapper photometry: %s",
        phot_csv,
    )
    log.info(
        "Step11 spectra for global i normalization: %s",
        spectra_path,
    )
    log.info(
        "Response bands: i,z only; SkyMapper r excluded"
    )

    master, fit = build_linear_iz_shape(
        shape_spectra,
        phot_csv,
        filter_i,
        filter_z,
        min_coverage=float(
            args.min_shape_coverage
        ),
        scale_nm=float(
            args.scale_nm
        ),
    )

    lam = master[
        "lambda_nm"
    ].to_numpy(np.float64)

    shape_resp = master[
        "response_linear"
    ].to_numpy(np.float64)

    shape_p16 = master[
        "loo_p16"
    ].to_numpy(np.float64)

    shape_p84 = master[
        "loo_p84"
    ].to_numpy(np.float64)

    trust_lo = float(
        fit["pivot_i_nm"]
    )
    trust_hi = float(
        fit["pivot_z_nm"]
    )

    global_i_norm, norm_p16, norm_p84, norm_rows = (
        derive_global_i_normalization(
            spectra_path,
            lam,
            shape_resp,
            filter_i,
            min_coverage=float(
                args.min_i_coverage
            ),
        )
    )

    resp = (
        shape_resp
        * global_i_norm
    )
    p16 = (
        shape_p16
        * global_i_norm
    )
    p84 = (
        shape_p84
        * global_i_norm
    )

    phdr = fits.Header()

    phdr["PIPESTEP"] = (
        "STEP12",
        "SAMOS pipeline step",
    )
    phdr["STAGE"] = (
        "12d",
        "Pipeline stage",
    )
    phdr["METHOD"] = (
        "ENS_LIN_IZ",
        "Robust ensemble linear i-z response",
    )
    phdr["SHAPEIN"] = (
        shape_spectra.name,
        "Spectra used for response shape",
    )
    phdr["PHOTCAT"] = (
        phot_csv.name,
        "SkyMapper photometry table",
    )
    phdr["TRUSTLO"] = (
        trust_lo,
        "Trusted response lower wavelength [nm]",
    )
    phdr["TRUSTHI"] = (
        trust_hi,
        "Trusted response upper wavelength [nm]",
    )
    phdr["SHAPENM"] = (
        fit["pivot_i_nm"],
        "Shape normalization pivot [nm]",
    )
    phdr["SHPA1"] = (
        fit["linear_a1"],
        "Linear ln-response coefficient",
    )
    phdr["SCLNM"] = (
        fit["scale_nm"],
        "Linear-response wavelength scale [nm]",
    )
    phdr["MINCOV"] = (
        fit["min_coverage"],
        "Minimum i,z passband coverage",
    )
    phdr["GINORM"] = (
        global_i_norm,
        "Global i-band normalization",
    )
    phdr["GINP16"] = (
        norm_p16,
        "16th percentile individual i norm",
    )
    phdr["GINP84"] = (
        norm_p84,
        "84th percentile individual i norm",
    )
    phdr["NGINORM"] = (
        len(norm_rows),
        "Spectra used for global i normalization",
    )
    phdr["PIVOTI"] = (
        fit["pivot_i_nm"],
        "SkyMapper i pivot [nm]",
    )
    phdr["PIVOTZ"] = (
        fit["pivot_z_nm"],
        "SkyMapper z pivot [nm]",
    )
    phdr["NSTARS"] = (
        fit["n_stars"],
        "Stars in ensemble i-z response fit",
    )
    phdr["LOOLIN"] = (
        fit["loo_linear"]["median"],
        "Median linear LOO RMS [mag]",
    )

    phdr.add_history(
        "Step12d common response shape from robust linear SkyMapper i-z ensemble fit."
    )
    phdr.add_history(
        "SkyMapper r excluded: substantial passband weight lies below formal 600-nm SAMOS cutoff."
    )
    phdr.add_history(
        "Trusted wavelength-dependent response interval is i pivot through z pivot."
    )
    phdr.add_history(
        "Step12e holds the nearest trusted boundary outside this interval."
    )
    phdr.add_history(
        "One global factor preserves median Step11 synthetic i-band flux."
    )
    phdr.add_history(
        "No per-object photometric normalization is part of production Step12."
    )

    cols = fits.ColDefs(
        [
            fits.Column(
                name="LAMBDA_NM",
                format="D",
                array=lam,
            ),
            fits.Column(
                name="RESP_MASTER",
                format="D",
                array=resp,
            ),
            fits.Column(
                name="RESP_P16",
                format="D",
                array=p16,
            ),
            fits.Column(
                name="RESP_P84",
                format="D",
                array=p84,
            ),
            fits.Column(
                name="RESP_SHAPE",
                format="D",
                array=shape_resp,
            ),
        ]
    )

    hdu = fits.BinTableHDU.from_columns(
        cols,
        name="MASTER_RESPONSE",
    )

    hdu.header["METHOD"] = "ENS_LIN_IZ"
    hdu.header["TRUSTLO"] = trust_lo
    hdu.header["TRUSTHI"] = trust_hi
    hdu.header["SHAPENM"] = fit["pivot_i_nm"]
    hdu.header["SHPA1"] = fit["linear_a1"]
    hdu.header["GINORM"] = global_i_norm

    fits.HDUList(
        [
            fits.PrimaryHDU(
                header=phdr
            ),
            hdu,
        ]
    ).writeto(
        out_fits,
        overwrite=True,
    )

    def interp(w, arr):
        return float(
            np.interp(
                w,
                lam,
                arr,
            )
        )

    summary = {
        "method": (
            "ensemble_linear_iz_plus_global_i_normalization"
        ),
        "shape_spectra_path": str(
            shape_spectra
        ),
        "shape_flux_column": (
            "FLUX_TELLCOR_O2"
        ),
        "phot_csv": str(
            phot_csv
        ),
        "spectra_path": str(
            spectra_path
        ),
        "r_band_used": 0,
        "r_exclusion": (
            "formal_600nm_cutoff"
        ),
        "trust_min_nm": trust_lo,
        "trust_max_nm": trust_hi,
        "pivot_i_nm": fit[
            "pivot_i_nm"
        ],
        "pivot_z_nm": fit[
            "pivot_z_nm"
        ],
        "scale_nm": fit[
            "scale_nm"
        ],
        "linear_a1": fit[
            "linear_a1"
        ],
        "shape_min_coverage": fit[
            "min_coverage"
        ],
        "n_shape_stars": fit[
            "n_stars"
        ],
        "shape_slits": ";".join(
            fit["slits"]
        ),
        "insample_gray_median_mag": fit[
            "insample_gray"
        ]["median"],
        "insample_linear_median_mag": fit[
            "insample_linear"
        ]["median"],
        "loo_gray_median_mag": fit[
            "loo_gray"
        ]["median"],
        "loo_gray_p16_mag": fit[
            "loo_gray"
        ]["p16"],
        "loo_gray_p84_mag": fit[
            "loo_gray"
        ]["p84"],
        "loo_linear_median_mag": fit[
            "loo_linear"
        ]["median"],
        "loo_linear_p16_mag": fit[
            "loo_linear"
        ]["p16"],
        "loo_linear_p84_mag": fit[
            "loo_linear"
        ]["p84"],
        "global_i_norm": global_i_norm,
        "global_i_norm_p16": norm_p16,
        "global_i_norm_p84": norm_p84,
        "global_i_norm_n": len(
            norm_rows
        ),
        "shape_response_i": interp(
            fit["pivot_i_nm"],
            shape_resp,
        ),
        "shape_response_z": interp(
            fit["pivot_z_nm"],
            shape_resp,
        ),
        "response_i": interp(
            fit["pivot_i_nm"],
            resp,
        ),
        "response_z": interp(
            fit["pivot_z_nm"],
            resp,
        ),
        "response_below_i_hold": float(
            resp[0]
        ),
        "response_above_z_hold": float(
            resp[-1]
        ),
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        summary_csv,
        index=False,
    )

    metadata = {
        **summary,
        "output_fits": str(
            out_fits
        ),
        "shape_fit_slits": fit[
            "slits"
        ],
        "edge_policy_for_step12e": (
            "hold nearest trusted boundary; "
            "never polynomial-extrapolate"
        ),
        "global_i_normalization_rows": [
            {
                "slit": slit,
                "factor": factor,
            }
            for slit, factor in norm_rows
        ],
        "notes": [
            (
                "SkyMapper r is diagnostic only and is not used "
                "to derive the production response."
            ),
            (
                "RESP_SHAPE is the robust linear i-z ensemble "
                "response normalized at the i-band pivot."
            ),
            (
                "RESP_MASTER = global_i_norm * RESP_SHAPE."
            ),
            (
                "Below the i pivot Step12e applies only the held "
                "global-i boundary normalization; no wavelength-"
                "dependent blue response is inferred."
            ),
            (
                "Above the z pivot Step12e holds the z-boundary "
                "response; no polynomial extrapolation is used."
            ),
            (
                "No per-object photometric normalization is used "
                "in production Step12."
            ),
        ],
    }

    metadata_json.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    log.info(
        "Linear i-z response: N=%d  a1=%.12f",
        fit["n_stars"],
        fit["linear_a1"],
    )

    log.info(
        "Linear LOO RMS: median=%.4f  [p16,p84]=[%.4f, %.4f] mag",
        fit["loo_linear"]["median"],
        fit["loo_linear"]["p16"],
        fit["loo_linear"]["p84"],
    )

    log.info(
        "Global i normalization: %.12f  "
        "(N=%d; p16=%.12f p84=%.12f)",
        global_i_norm,
        len(norm_rows),
        norm_p16,
        norm_p84,
    )

    log.info(
        "Production response: i=%.6f  z=%.6f",
        summary["response_i"],
        summary["response_z"],
    )

    log.info(
        "Wrote: %s",
        out_fits,
    )
    log.info(
        "Wrote: %s",
        summary_csv,
    )
    log.info(
        "Wrote: %s",
        metadata_json,
    )


if __name__ == "__main__":
    main()
