#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step12e — apply the production Step12 response
==============================================

The Step12d master already contains both:
  1. the common wavelength-dependent ensemble shape correction; and
  2. the single global normalization that preserves the median Step11
     synthetic SkyMapper i-band flux.

Step12e therefore performs only a deterministic multiplicative application:

    FLUX_FLAM_STELLARRESP = FLUX_FLAM * RESP_STELLAR_MASTER

    VAR_FLAM2_STELLARRESP = VAR_FLAM2 * RESP_STELLAR_MASTER**2

No per-object photometric normalization is performed.

Science invariants
------------------
- LAMBDA_NM is unchanged and never resampled.
- Negative sky-subtracted flux values are retained.
- Outside the trusted response interval the nearest boundary response is held
  constant; the quadratic is never extrapolated.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from astropy.io import fits

import config

log = logging.getLogger("step12e_apply_stellar_response")


def is_slit_ext(hdu) -> bool:
    return (hdu.name or "").upper().startswith("SLIT")


def _add_or_replace_column(
    hdu: fits.BinTableHDU,
    name: str,
    array,
    fmt: str,
) -> fits.BinTableHDU:
    cols = list(hdu.columns)
    names = [c.name.upper() for c in cols]
    new = fits.Column(name=name, format=fmt, array=array)

    if name.upper() in names:
        cols[names.index(name.upper())] = new
    else:
        cols.append(new)

    return fits.BinTableHDU.from_columns(
        fits.ColDefs(cols),
        header=hdu.header.copy(),
        name=hdu.name,
    )


def read_master_response(path: Path):
    with fits.open(path, memmap=False) as h:
        if "MASTER_RESPONSE" not in h:
            raise KeyError(f"{path} lacks MASTER_RESPONSE extension")

        d = h["MASTER_RESPONSE"].data
        needed = {"LAMBDA_NM", "RESP_MASTER"}
        if not needed <= set(d.names):
            raise KeyError(
                f"{path}: MASTER_RESPONSE lacks {sorted(needed - set(d.names))}"
            )

        lam = np.asarray(d["LAMBDA_NM"], float)
        resp = np.asarray(d["RESP_MASTER"], float)
        global_i_norm = float(h[0].header.get("GINORM", 1.0))

    m = np.isfinite(lam) & np.isfinite(resp) & (resp > 0)
    lam, resp = lam[m], resp[m]

    if lam.size < 2:
        raise RuntimeError(
            "Master response has fewer than two finite positive samples."
        )

    order = np.argsort(lam)
    lam, resp = lam[order], resp[order]

    lam, idx = np.unique(lam, return_index=True)
    resp = resp[idx]

    return lam, resp, global_i_norm


def response_on_spectrum(lam_nm, master_lam, master_resp):
    """
    Interpolate inside the trusted range and hold the nearest boundary outside.
    """
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

    inrange = (
        good
        & (lam_nm >= master_lam[0])
        & (lam_nm <= master_lam[-1])
    )

    return out, inrange


def parse_args():
    ap = argparse.ArgumentParser(
        description="Apply production Step12d response to all Step11 spectra."
    )
    ap.add_argument("--in", dest="infile", type=str, default="")
    ap.add_argument("--master", type=str, default="")
    ap.add_argument("--out", dest="outfile", type=str, default="")
    return ap.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    args = parse_args()

    infile = (
        Path(args.infile)
        if args.infile
        else Path(config.EXTRACT1D_STEP12_INPUT)
    )
    master = (
        Path(args.master)
        if args.master
        else Path(config.STEP12D_MASTER_FITS)
    )
    outfile = (
        Path(args.outfile)
        if args.outfile
        else Path(config.EXTRACT1D_FINALCAL)
    )

    if not infile.exists():
        raise FileNotFoundError(infile)
    if not master.exists():
        raise FileNotFoundError(master)

    master_lam, master_resp, global_i_norm = read_master_response(master)

    outfile.parent.mkdir(parents=True, exist_ok=True)

    n_slit = 0
    n_corrected = 0

    with fits.open(infile, memmap=False) as hin:
        phdr = hin[0].header.copy()

        phdr["PIPESTEP"] = ("STEP12", "SAMOS pipeline step")
        phdr["STAGE"] = ("12e", "Pipeline stage")
        phdr["S12RESP"] = (master.name, "Step12d production master response")
        phdr["S12EDGE"] = ("HOLD", "Outside trusted range, hold boundary response")
        phdr["S12NORM"] = ("GLOBAL_I", "Global i normalization already in response")
        phdr["S12GIN"] = (global_i_norm, "Global i-band factor folded into response")
        phdr["TRUSTLO"] = (
            float(master_lam[0]),
            "Master response lower trusted wavelength [nm]",
        )
        phdr["TRUSTHI"] = (
            float(master_lam[-1]),
            "Master response upper trusted wavelength [nm]",
        )

        phdr.add_history(
            "Step12e: applied production Step12d response to FLUX_FLAM."
        )
        phdr.add_history(
            "Step12e: no per-object photometric normalization."
        )
        phdr.add_history(
            "Step12e: LAMBDA_NM unchanged; no spectral resampling."
        )
        phdr.add_history(
            "Step12e: negative flux values retained."
        )
        phdr.add_history(
            "Step12e: outside trusted range uses nearest response boundary."
        )

        hout = fits.HDUList([fits.PrimaryHDU(header=phdr)])

        for ext in hin[1:]:
            if not is_slit_ext(ext) or ext.data is None:
                hout.append(ext.copy())
                continue

            slit = ext.name.strip().upper()
            n_slit += 1
            names = set(ext.columns.names)

            if "LAMBDA_NM" not in names or "FLUX_FLAM" not in names:
                log.warning(
                    "%s: missing LAMBDA_NM/FLUX_FLAM; copied unchanged",
                    slit,
                )
                hout.append(ext.copy())
                continue

            lam = np.asarray(ext.data["LAMBDA_NM"], float)
            flux = np.asarray(ext.data["FLUX_FLAM"], float)

            resp, inrange = response_on_spectrum(
                lam,
                master_lam,
                master_resp,
            )

            final_flux = flux * resp

            new = ext.copy()

            new = _add_or_replace_column(
                new,
                "RESP_STELLAR_MASTER",
                resp.astype(np.float32),
                "E",
            )

            new = _add_or_replace_column(
                new,
                "RESP_INRANGE",
                inrange.astype(np.int16),
                "I",
            )

            new = _add_or_replace_column(
                new,
                "FLUX_FLAM_STELLARRESP",
                final_flux.astype(np.float32),
                "E",
            )

            if "VAR_FLAM2" in names:
                var = np.asarray(ext.data["VAR_FLAM2"], float)
                final_var = var * resp**2

                new = _add_or_replace_column(
                    new,
                    "VAR_FLAM2_STELLARRESP",
                    final_var.astype(np.float32),
                    "E",
                )

            nrow = len(lam)

            # Retained for backward-compatible QC interface.
            # The production normalization is already folded into RESP_MASTER.
            new = _add_or_replace_column(
                new,
                "NORM_STELLARRESP",
                np.ones(nrow, dtype=np.float32),
                "E",
            )

            new = _add_or_replace_column(
                new,
                "HAS_PHOTNORM",
                np.zeros(nrow, dtype=np.int16),
                "I",
            )

            new = _add_or_replace_column(
                new,
                "NORM_BAND",
                np.full(nrow, b"", dtype="S1"),
                "1A",
            )

            new.header["S12RESP"] = (
                master.name,
                "Step12d production response",
            )
            new.header["S12EDGE"] = (
                "HOLD",
                "Response behavior outside trusted interval",
            )
            new.header["S12NORM"] = (
                "GLOBAL_I",
                "Global i normalization is folded into response",
            )
            new.header["S12GIN"] = (
                global_i_norm,
                "Global i factor folded into response",
            )
            new.header["TRUSTLO"] = (
                float(master_lam[0]),
                "Trusted lower wavelength [nm]",
            )
            new.header["TRUSTHI"] = (
                float(master_lam[-1]),
                "Trusted upper wavelength [nm]",
            )
            new.header["PHOTNORM"] = (
                False,
                "No per-object photometric normalization",
            )
            new.header["NORMBAND"] = (
                "NONE",
                "No per-object normalization band",
            )
            new.header["NORMFAC"] = (
                1.0,
                "No additional per-object scalar",
            )

            new.header.add_history(
                "Step12e: FLUX_FLAM_STELLARRESP = "
                "FLUX_FLAM * RESP_STELLAR_MASTER."
            )

            if "VAR_FLAM2" in names:
                new.header.add_history(
                    "Step12e: VAR_FLAM2_STELLARRESP = "
                    "VAR_FLAM2 * RESP_STELLAR_MASTER^2."
                )

            hout.append(new)
            n_corrected += 1

        hout[0].header["NSLITS"] = (
            n_slit,
            "SLIT extensions seen",
        )
        hout[0].header["NCORR"] = (
            n_corrected,
            "Slits with response applied",
        )

        hout.writeto(outfile, overwrite=True)

    log.info("Wrote: %s", outfile)
    log.info("Slits corrected: %d / %d", n_corrected, n_slit)
    log.info("Global i normalization already folded into master: %.8f", global_i_norm)
    log.info("Per-object photometric normalization: disabled")
    log.info(
        "Trusted response interval: %.3f .. %.3f nm; outside = boundary hold",
        master_lam[0],
        master_lam[-1],
    )


if __name__ == "__main__":
    main()
