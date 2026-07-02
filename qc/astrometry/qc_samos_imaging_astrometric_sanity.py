#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:23:48 2026

@author: robberto
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAMOS imaging-channel astrometric sanity check.

Correlates catalog RA/DEC with measured pixels in the imaging channel.
Assumes a DS9 pixel-region file containing one region per target, in the
same order as the RA/DEC table.

Compared to the spectroscopic CCD sanity check:
  - RA  is fitted against imaging X pixel
  - DEC is fitted against imaging Y pixel
  - no A-band proxy is needed

For the SAMOS imaging channel, increasing RA is expected to increase XIMG.
"""

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--radec",
        default=str(config.STEP11_RADEC),
        help="CSV table with columns slit, RA, DEC",
    )

    ap.add_argument(
        "--pixreg",
        default=str(
            Path(config.CALIB_ROOT)
            / "reference_tables/regions/Dolidze25-T00_0061_2026-01-13T22-54-59_pix.reg"
        ),
        help="DS9 region file with imaging-channel pixel positions",
    )
    
    ap.add_argument(
        "--outdir",
        default=str(config.REDUCED_DIR / "astrometry_qc"),
        help="Output directory",
    )

    ap.add_argument(
        "--clip",
        type=float,
        default=3.0,
        help="Sigma clipping threshold for robust line fits",
    )

    return ap.parse_args()


def robust_linefit(x, y, clip=3.0, niter=5):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    ok = np.isfinite(x) & np.isfinite(y)

    for _ in range(niter):
        if ok.sum() < 4:
            break

        p = np.polyfit(x[ok], y[ok], 1)
        r = y - np.polyval(p, x)

        sig = 1.4826 * np.nanmedian(
            np.abs(r[ok] - np.nanmedian(r[ok]))
        )

        if not np.isfinite(sig) or sig <= 0:
            break

        ok_new = ok & (np.abs(r) < clip * sig)

        if ok_new.sum() == ok.sum():
            ok = ok_new
            break

        ok = ok_new

    p = np.polyfit(x[ok], y[ok], 1)
    r = y - np.polyval(p, x)

    sig = 1.4826 * np.nanmedian(
        np.abs(r[ok] - np.nanmedian(r[ok]))
    )

    return p, r, sig, ok


def parse_ds9_pixel_regions(reg_path):
    """
    Parse simple DS9 pixel regions.

    Supported examples:
        point(x,y)
        circle(x,y,r)
        box(x,y,w,h,angle)

    Returns a table with XIMG, YIMG and optional shape information.
    """

    reg_path = Path(reg_path)
    rows = []

    point_pat = re.compile(r"point\(([^,]+),([^)]+)\)")
    circle_pat = re.compile(r"circle\(([^,]+),([^,]+),([^)]+)\)")
    box_pat = re.compile(r"box\(([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)")

    with open(reg_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if line.lower() in ["global", "image", "physical", "fk5"]:
                continue

            m = box_pat.search(line)
            if m:
                x, y, w, h, ang = map(float, m.groups())
                rows.append(
                    dict(
                        XIMG=x,
                        YIMG=y,
                        SHAPE="box",
                        W=w,
                        H=h,
                        R=np.nan,
                        ANG=ang,
                        RAW=line,
                    )
                )
                continue

            m = circle_pat.search(line)
            if m:
                x, y, r = map(float, m.groups())
                rows.append(
                    dict(
                        XIMG=x,
                        YIMG=y,
                        SHAPE="circle",
                        W=np.nan,
                        H=np.nan,
                        R=r,
                        ANG=np.nan,
                        RAW=line,
                    )
                )
                continue

            m = point_pat.search(line)
            if m:
                x, y = map(float, m.groups())
                rows.append(
                    dict(
                        XIMG=x,
                        YIMG=y,
                        SHAPE="point",
                        W=np.nan,
                        H=np.nan,
                        R=np.nan,
                        ANG=np.nan,
                        RAW=line,
                    )
                )
                continue

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(f"No usable pixel regions found in {reg_path}")

    return df


def main():
    args = parse_args()

    radec_path = Path(args.radec)
    pixreg_path = Path(args.pixreg)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    radec = pd.read_csv(radec_path)

    required = {"RA", "DEC"}
    missing = required - set(radec.columns)
    if missing:
        raise RuntimeError(f"Missing required RADEC columns: {missing}")

    if "slit" not in radec.columns:
        radec["slit"] = [f"OBJ{i + 1:03d}" for i in range(len(radec))]

    radec["slit"] = radec["slit"].astype(str).str.upper()

    pix = parse_ds9_pixel_regions(pixreg_path)

    if len(radec) != len(pix):
        raise RuntimeError(
            f"RADEC rows = {len(radec)} but pixel regions = {len(pix)}. "
            "For now this script assumes the same order."
        )

    df = pd.concat(
        [
            radec.reset_index(drop=True),
            pix.reset_index(drop=True),
        ],
        axis=1,
    )

    ra0 = np.nanmedian(df["RA"])
    dec0 = np.nanmedian(df["DEC"])

    df["RA_ARCSEC"] = (
        3600.0
        * (df["RA"] - ra0)
        * np.cos(np.deg2rad(dec0))
    )

    df["DEC_ARCSEC"] = 3600.0 * (df["DEC"] - dec0)

    p_ra, res_ra, sig_ra, ok_ra = robust_linefit(
        df["RA_ARCSEC"],
        df["XIMG"],
        clip=args.clip,
    )

    p_dec, res_dec, sig_dec, ok_dec = robust_linefit(
        df["DEC_ARCSEC"],
        df["YIMG"],
        clip=args.clip,
    )

    df["X_FIT_FROM_RA"] = np.polyval(p_ra, df["RA_ARCSEC"])
    df["Y_FIT_FROM_DEC"] = np.polyval(p_dec, df["DEC_ARCSEC"])

    df["DX_RA_FIT"] = res_ra
    df["DY_DEC_FIT"] = res_dec
    df["OK_RA"] = ok_ra
    df["OK_DEC"] = ok_dec

    df["RA_ARCSEC_FROM_XIMG"] = (df["XIMG"] - p_ra[1]) / p_ra[0]
    df["DEC_ARCSEC_FROM_YIMG"] = (df["YIMG"] - p_dec[1]) / p_dec[0]

    df["RA_FROM_XIMG"] = ra0 + df["RA_ARCSEC_FROM_XIMG"] / (
        3600.0 * np.cos(np.deg2rad(dec0))
    )

    df["DEC_FROM_YIMG"] = dec0 + df["DEC_ARCSEC_FROM_YIMG"] / 3600.0

    df["DRA_IMPLIED_ARCSEC"] = (
        3600.0
        * (df["RA_FROM_XIMG"] - df["RA"])
        * np.cos(np.deg2rad(dec0))
    )

    df["DDEC_IMPLIED_ARCSEC"] = (
        3600.0
        * (df["DEC_FROM_YIMG"] - df["DEC"])
    )

    bad_ra = ~df["OK_RA"]
    bad_dec = ~df["OK_DEC"]
    bad_any = bad_ra | bad_dec

    csv_out = outdir / "qc_samos_imaging_astrometric_sanity.csv"
    png_out = outdir / "qc_samos_imaging_astrometric_sanity.png"

    df.to_csv(csv_out, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    ax = axes[0]
    ax.scatter(
        df["RA_ARCSEC"],
        df["XIMG"],
        c=np.where(bad_ra, "r", "k"),
        s=35,
    )
    xx = np.linspace(df["RA_ARCSEC"].min(), df["RA_ARCSEC"].max(), 100)
    ax.plot(xx, np.polyval(p_ra, xx), "r--", lw=1.2)

    for _, r in df.loc[bad_ra].iterrows():
        ax.text(r["RA_ARCSEC"], r["XIMG"], r["slit"], fontsize=7)

    ax.set_xlabel("Catalog RA offset [arcsec]")
    ax.set_ylabel("Imaging-channel X pixel")
    ax.set_title(
        f"RA → XIMG: slope={p_ra[0]:.3f} pix/arcsec, rms={sig_ra:.2f} pix"
    )
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.scatter(
        df["DEC_ARCSEC"],
        df["YIMG"],
        c=np.where(bad_dec, "r", "k"),
        s=35,
    )
    yy = np.linspace(df["DEC_ARCSEC"].min(), df["DEC_ARCSEC"].max(), 100)
    ax.plot(yy, np.polyval(p_dec, yy), "r--", lw=1.2)

    for _, r in df.loc[bad_dec].iterrows():
        ax.text(r["DEC_ARCSEC"], r["YIMG"], r["slit"], fontsize=7)

    ax.set_xlabel("Catalog DEC offset [arcsec]")
    ax.set_ylabel("Imaging-channel Y pixel")
    ax.set_title(
        f"DEC → YIMG: slope={p_dec[0]:.3f} pix/arcsec, rms={sig_dec:.2f} pix"
    )
    ax.grid(alpha=0.25)

    ax = axes[2]

    ax.scatter(
        df["RA"],
        df["DEC"],
        c=np.where(bad_any, "r", "k"),
        s=35,
        label="catalog RA/DEC",
    )

    ax.scatter(
        df["RA_FROM_XIMG"],
        df["DEC_FROM_YIMG"],
        facecolors="none",
        edgecolors=np.where(bad_any, "r", "0.5"),
        s=80,
        linewidths=1.1,
        label="image-inferred RA/DEC",
    )

    for _, r in df.iterrows():
        color = "r" if (not r["OK_RA"] or not r["OK_DEC"]) else "0.6"
        lw = 0.9 if color == "r" else 0.5

        ax.plot(
            [r["RA"], r["RA_FROM_XIMG"]],
            [r["DEC"], r["DEC_FROM_YIMG"]],
            color=color,
            lw=lw,
            alpha=0.75,
        )

    for _, r in df.loc[bad_any].iterrows():
        ax.text(r["RA"], r["DEC"], r["slit"], fontsize=7, color="r")

    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("DEC [deg]")
    ax.set_title("Sky map: catalog → imaging-inferred")
    ax.grid(alpha=0.25)
    ax.invert_xaxis()
    ax.legend(fontsize=8)

    fig.suptitle("SAMOS imaging-channel astrometric sanity check")
    plt.tight_layout()
    fig.savefig(png_out, dpi=160)
    plt.close(fig)

    print("Wrote:")
    print(" ", csv_out)
    print(" ", png_out)
    print()

    print("Fit summary:")
    print(f"  RA0, DEC0             = {ra0:.8f}, {dec0:.8f}")
    print(f"  dXIMG/dRA_arcsec      = {p_ra[0]:.6f} pix/arcsec")
    print(f"  dYIMG/dDEC_arcsec     = {p_dec[0]:.6f} pix/arcsec")
    print(f"  RA fit RMS            = {sig_ra:.3f} pix")
    print(f"  DEC fit RMS           = {sig_dec:.3f} pix")
    print()

    if p_ra[0] > 0:
        print("RA orientation check: OK, increasing RA increases XIMG.")
    else:
        print("RA orientation check: WARNING, increasing RA decreases XIMG.")
    print()

    print("Likely outliers:")
    cols = [
        "slit",
        "RA",
        "DEC",
        "XIMG",
        "YIMG",
        "DX_RA_FIT",
        "DY_DEC_FIT",
        "RA_FROM_XIMG",
        "DEC_FROM_YIMG",
        "DRA_IMPLIED_ARCSEC",
        "DDEC_IMPLIED_ARCSEC",
    ]

    bad = df[bad_any].copy()

    if len(bad) == 0:
        print("  None")
    else:
        print(bad[cols].to_string(index=False))


if __name__ == "__main__":
    main()