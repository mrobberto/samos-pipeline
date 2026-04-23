#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
step11c_continuum_snr.py

Prototype diagnostic script:
estimate continuum-based S/N for each slit after Step11c.

Intent
------
Use the Step11c flux-calibrated spectra to rank slits by continuum quality,
so the cleanest / highest-SNR spectra can be selected to build the Step11d
master refinement response.

Method
------
For each slit:
1) read LAMBDA_NM and FLUX_FLAM
2) estimate a smooth continuum with a running median
3) compute residuals = flux - continuum
4) estimate robust noise from MAD(residuals)
5) report continuum S/N in broad wavelength windows and globally

Outputs
-------
CSV with one row per slit and columns such as:
- slit
- n_good
- snr_600_720
- snr_730_820
- snr_830_900
- snr_global
- cont_median_global
- noise_mad_global
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

import config


def running_median_1d(y: np.ndarray, width: int) -> np.ndarray:
    """
    Simple running median with edge padding.
    Width should be odd; if even, it is incremented by 1.
    """
    y = np.asarray(y, dtype=float)
    if width < 3:
        return y.copy()
    if width % 2 == 0:
        width += 1

    pad = width // 2
    yp = np.pad(y, pad_width=pad, mode="edge")
    out = np.empty_like(y)

    for i in range(len(y)):
        out[i] = np.nanmedian(yp[i:i + width])

    return out


def robust_sigma_from_residuals(resid: np.ndarray) -> float:
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    if resid.size < 5:
        return np.nan
    med = np.nanmedian(resid)
    mad = np.nanmedian(np.abs(resid - med))
    return 1.4826 * mad


def continuum_snr_in_window(
    lam_nm: np.ndarray,
    flux: np.ndarray,
    wmin: float,
    wmax: float,
    smooth_pix: int = 51,
) -> dict:
    m = np.isfinite(lam_nm) & np.isfinite(flux) & (lam_nm >= wmin) & (lam_nm <= wmax)
    n = int(np.count_nonzero(m))
    if n < max(20, smooth_pix + 5):
        return {
            "npts": n,
            "cont_median": np.nan,
            "noise_sigma": np.nan,
            "snr": np.nan,
        }

    x = np.asarray(lam_nm[m], dtype=float)
    y = np.asarray(flux[m], dtype=float)

    cont = running_median_1d(y, smooth_pix)
    resid = y - cont

    sigma = robust_sigma_from_residuals(resid)
    cmed = np.nanmedian(cont)

    if not np.isfinite(sigma) or sigma <= 0:
        snr = np.nan
    else:
        snr = cmed / sigma

    return {
        "npts": n,
        "cont_median": float(cmed),
        "noise_sigma": float(sigma),
        "snr": float(snr) if np.isfinite(snr) else np.nan,
    }


def infer_slit_id(hdu, idx: int) -> str:
    hdr = hdu.header
    for key in ("EXTNAME", "NAME", "OBJECT"):
        if key in hdr:
            val = str(hdr[key]).strip()
            if val:
                return val
    return f"HDU{idx:03d}"


def parse_args():
    st11 = Path(config.ST11_FLUXCAL)
    p = argparse.ArgumentParser(description="Prototype continuum S/N ranking from Step11c spectra.")
    p.add_argument(
        "--infile",
        type=Path,
        default=Path(config.EXTRACT1D_FLUXCAL),
        help="Step11c FITS file containing FLUX_FLAM",
    )
    p.add_argument(
        "--outcsv",
        type=Path,
        default=st11 / "Extract1D_fluxcal_continuum_snr.csv",
        help="Output CSV with continuum S/N diagnostics",
    )
    p.add_argument(
        "--smooth-pix",
        type=int,
        default=51,
        help="Running-median width in pixels",
    )
    p.add_argument(
        "--global-min",
        type=float,
        default=600.0,
        help="Global S/N lower wavelength bound [nm]",
    )
    p.add_argument(
        "--global-max",
        type=float,
        default=900.0,
        help="Global S/N upper wavelength bound [nm]",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.infile.exists():
        raise FileNotFoundError(args.infile)

    args.outcsv.parent.mkdir(parents=True, exist_ok=True)

    windows = {
        "600_720": (600.0, 720.0),
        "730_820": (730.0, 820.0),
        "830_900": (830.0, 900.0),
    }

    rows = []

    with fits.open(args.infile) as hdul:
        for idx, hdu in enumerate(hdul[1:], start=1):
            if not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                continue

            slit = infer_slit_id(hdu, idx)
            tab = hdu.data
            if tab is None:
                continue

            cols = list(tab.columns.names)
            if "LAMBDA_NM" not in cols or "FLUX_FLAM" not in cols:
                continue

            lam = np.asarray(tab["LAMBDA_NM"], dtype=float)
            flux = np.asarray(tab["FLUX_FLAM"], dtype=float)

            good = np.isfinite(lam) & np.isfinite(flux)
            n_good = int(np.count_nonzero(good))

            out = {
                "slit": slit,
                "n_good": n_good,
            }

            global_stats = continuum_snr_in_window(
                lam, flux,
                wmin=args.global_min,
                wmax=args.global_max,
                smooth_pix=args.smooth_pix,
            )

            out["snr_global"] = global_stats["snr"]
            out["cont_median_global"] = global_stats["cont_median"]
            out["noise_mad_global"] = global_stats["noise_sigma"]

            band_snrs = []
            for tag, (wmin, wmax) in windows.items():
                stats = continuum_snr_in_window(
                    lam, flux,
                    wmin=wmin,
                    wmax=wmax,
                    smooth_pix=args.smooth_pix,
                )
                out[f"snr_{tag}"] = stats["snr"]
                out[f"cont_{tag}"] = stats["cont_median"]
                out[f"noise_{tag}"] = stats["noise_sigma"]
                out[f"npts_{tag}"] = stats["npts"]
                if np.isfinite(stats["snr"]):
                    band_snrs.append(stats["snr"])

            out["snr_median_bands"] = float(np.nanmedian(band_snrs)) if len(band_snrs) else np.nan
            out["snr_min_bands"] = float(np.nanmin(band_snrs)) if len(band_snrs) else np.nan

            rows.append(out)

    df = pd.DataFrame(rows)

    # Useful default ranking:
    # primary = median band S/N, secondary = global S/N
    if not df.empty:
        df = df.sort_values(
            by=["snr_median_bands", "snr_global"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)

    df.to_csv(args.outcsv, index=False)

    print("[OK] Wrote:", args.outcsv)
    print("Rows:", len(df))
    if len(df):
        print("\nTop 10 by snr_median_bands:")
        show = ["slit", "snr_600_720", "snr_730_820", "snr_830_900", "snr_median_bands", "snr_global"]
        print(df[show].head(10).to_string(index=False))


if __name__ == "__main__":
    main()