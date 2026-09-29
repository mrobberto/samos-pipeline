#!/usr/bin/env python3

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

import config


GRID = np.arange(500.0, 1065.0001, 0.25)
NORM_LO = 760.0
NORM_HI = 790.0


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Apply geometry-aware relative Illum2D correction "
            "to Step10 telluric-corrected spectra."
        )
    )

    default_in = Path(config.EXTRACT1D_TELLCOR)

    default_out = Path(
        getattr(
            config,
            "EXTRACT1D_TELLCOR_ILLUMREL",
            Path(config.ST10_TELLURIC)
            / (default_in.stem + "_illumrel.fits"),
        )
    )

    default_ref = Path(
        getattr(
            config,
            "ILLUMREL_REFERENCE_CSV",
            Path(config.ST10_TELLURIC)
            / "illumrel_reference.csv",
        )
    )

    p.add_argument(
        "--infile",
        type=Path,
        default=default_in,
        help="Input Step10b telluric-corrected MEF.",
    )

    p.add_argument(
        "--outfile",
        type=Path,
        default=default_out,
        help="Output Step10c relative-illumination-corrected MEF.",
    )

    p.add_argument(
        "--reference-csv",
        type=Path,
        default=default_ref,
        help="Output median normalized Illum2D reference curve.",
    )

    return p.parse_args()


def norm_slit(s):
    s = str(s).strip().upper()
    d = "".join(c for c in s if c.isdigit())
    return f"SLIT{int(d):03d}" if d else s


def read_calibration():
    st04 = Path(config.ST04_TRACES)
    out = {}

    for tag in ("EVEN", "ODD"):

        illum_path = Path(
            config.ILLUM2D_EVEN
            if tag == "EVEN"
            else config.ILLUM2D_ODD
        )

        with fits.open(illum_path, memmap=False) as h:
            illum = np.asarray(h[0].data, float)

        base = (
            "Even_traces"
            if tag == "EVEN"
            else "Odd_traces"
        )

        sid_path = st04 / f"{base}_slitid.fits"

        if not sid_path.exists():
            sid_path = st04 / f"{base}_slitid_reg.fits"

        with fits.open(sid_path, memmap=False) as h:
            sidmap = np.asarray(h[0].data, int)

        out[tag] = (illum, sidmap)

    return out


def slit_illumination_vector(hd, cal):
    slit = norm_slit(hd.name)
    sid = int(slit[4:])
    tag = "EVEN" if sid % 2 == 0 else "ODD"

    illum, sidmap = cal[tag]

    lam = np.asarray(
        hd.data["LAMBDA_NM"],
        float,
    )

    yp = np.asarray(
        hd.data["YPIX"],
        float,
    )

    y0 = float(
        hd.header.get(
            "Y0DET",
            hd.header.get("YMIN", 0.0),
        )
    )

    yd = np.rint(y0 + yp).astype(int)

    ivec = np.full(len(yd), np.nan, float)

    for j, yy in enumerate(yd):

        if yy < 0 or yy >= illum.shape[0]:
            continue

        m = sidmap[yy] == sid

        if not np.any(m):
            continue

        v = illum[yy, m]
        v = v[np.isfinite(v)]

        if len(v):
            ivec[j] = np.median(v)

    return lam, ivec


def main():

    args = parse_args()

    infile = Path(args.infile)
    OUT = Path(args.outfile)
    REFCSV = Path(args.reference_csv)

    if not infile.exists():
        raise FileNotFoundError(infile)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    REFCSV.parent.mkdir(parents=True, exist_ok=True)

    cal = read_calibration()

    curves = {}

    # ------------------------------------------------------
    # Build normalized Illum2D curve for every usable slit.
    # ------------------------------------------------------

    with fits.open(
        infile,
        memmap=False,
    ) as h:

        for hd in h[1:]:

            if hd.data is None:
                continue

            names = set(hd.data.names)

            if not {
                "LAMBDA_NM",
                "YPIX",
            }.issubset(names):
                continue

            slit = norm_slit(hd.name)

            lam, ivec = slit_illumination_vector(
                hd,
                cal,
            )

            gn = (
                np.isfinite(lam)
                & np.isfinite(ivec)
                & (ivec > 0)
                & (lam >= NORM_LO)
                & (lam <= NORM_HI)
            )

            if gn.sum() < 20:
                continue

            norm = np.nanmedian(ivec[gn])

            if not np.isfinite(norm) or norm <= 0:
                continue

            iv = ivec / norm

            good = (
                np.isfinite(lam)
                & np.isfinite(iv)
                & (iv > 0)
            )

            if good.sum() < 50:
                continue

            ll = lam[good]
            ii = iv[good]

            order = np.argsort(ll)
            ll = ll[order]
            ii = ii[order]

            q = np.full_like(
                GRID,
                np.nan,
                dtype=float,
            )

            inside = (
                (GRID >= ll[0])
                & (GRID <= ll[-1])
            )

            q[inside] = np.interp(
                GRID[inside],
                ll,
                ii,
            )

            curves[slit] = q

    if not curves:
        raise RuntimeError(
            "No usable Illum2D slit curves."
        )

    M = np.vstack(list(curves.values()))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning,
        )
        ref = np.nanmedian(M, axis=0)

    nref = np.sum(np.isfinite(M), axis=0)

    pd.DataFrame({
        "lambda_nm": GRID,
        "illum_reference_norm": ref,
        "n_slits": nref,
    }).to_csv(
        REFCSV,
        index=False,
    )

    print(
        "Illumination curves used:",
        len(curves),
    )

    print(
        "Reference contributors "
        "[600,700,780,860,950,1000,1025] nm:",
        [
            int(
                nref[
                    np.argmin(
                        np.abs(GRID-x)
                    )
                ]
            )
            for x in (
                600,700,780,860,
                950,1000,1025
            )
        ],
    )

    # ------------------------------------------------------
    # Apply REF / slit illumination shape to Step10 spectra.
    # ------------------------------------------------------

    hout = []

    with fits.open(
        infile,
        memmap=False,
    ) as h:

        ph = h[0].copy()

        ph.header["PIPESTEP"] = (
            "STEP10C",
            "SAMOS relative illumination correction",
        )
        ph.header["STAGE"] = (
            "10c",
            "Pipeline stage",
        )

        ph.header["ILLRELC"] = (
            True,
            "Relative Illum2D correction",
        )
        ph.header["ILLREF"] = (
            "MEDIAN_ALL",
            "Normalized Illum2D reference",
        )
        ph.header["ILLNLO"] = (
            NORM_LO,
            "Illumination normalization lower nm",
        )
        ph.header["ILLNHI"] = (
            NORM_HI,
            "Illumination normalization upper nm",
        )

        hout.append(ph)

        n_corr = 0

        for hd in h[1:]:

            out = hd.copy()

            if (
                hd.data is None
                or norm_slit(hd.name) not in curves
            ):
                hout.append(out)
                continue

            names = set(hd.data.names)

            if not {
                "LAMBDA_NM",
                "FLUX_TELLCOR_O2",
            }.issubset(names):
                hout.append(out)
                continue

            slit = norm_slit(hd.name)

            lam = np.asarray(
                hd.data["LAMBDA_NM"],
                float,
            )

            flux = np.asarray(
                hd.data["FLUX_TELLCOR_O2"],
                float,
            )

            slit_grid = curves[slit]

            ref_lam = np.interp(
                lam,
                GRID,
                ref,
                left=np.nan,
                right=np.nan,
            )

            slit_lam = np.interp(
                lam,
                GRID,
                slit_grid,
                left=np.nan,
                right=np.nan,
            )

            corr = np.ones_like(
                lam,
                dtype=float,
            )

            good = (
                np.isfinite(ref_lam)
                & np.isfinite(slit_lam)
                & (ref_lam > 0)
                & (slit_lam > 0)
            )

            corr[good] = (
                ref_lam[good]
                / slit_lam[good]
            )

            fc = flux * corr

            out.data["FLUX_TELLCOR_O2"][:] = (
                fc.astype(np.float32)
            )

            if "VAR_TELLCOR_O2" in names:

                var = np.asarray(
                    hd.data["VAR_TELLCOR_O2"],
                    float,
                )

                out.data["VAR_TELLCOR_O2"][:] = (
                    var * corr**2
                ).astype(np.float32)

            out.header["ILLRELC"] = (
                True,
                "Relative Illum2D shape correction applied",
            )
            out.header["ILLREF"] = (
                "MEDIAN_ALL",
                "Reference illumination profile",
            )
            out.header["ILLNORM"] = (
                "760-790",
                "Normalization interval [nm]",
            )

            n_corr += 1
            hout.append(out)

    fits.HDUList(hout).writeto(
        OUT,
        overwrite=True,
    )

    print("Corrected slit extensions:", n_corr)
    print("Wrote:", OUT)
    print("Wrote:", REFCSV)


if __name__ == "__main__":
    main()
