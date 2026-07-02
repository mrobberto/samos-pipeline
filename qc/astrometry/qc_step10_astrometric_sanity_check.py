#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:02:32 2026

@author: robberto
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

import config


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectra", default=str(config.EXTRACT1D_TELLCOR))
    ap.add_argument("--radec", default=str(config.STEP11_RADEC))
    ap.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parent),
        help="Output directory; default is the directory containing this script.",
    )
    ap.add_argument("--flux-col", default="STELLAR_CONSENSUS")
    ap.add_argument("--a-lo", type=float, default=758.0)
    ap.add_argument("--a-hi", type=float, default=766.5)
    ap.add_argument("--clip", type=float, default=3.0)
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
        sig = 1.4826 * np.nanmedian(np.abs(r[ok] - np.nanmedian(r[ok])))
        if not np.isfinite(sig) or sig <= 0:
            break
        ok_new = ok & (np.abs(r) < clip * sig)
        if ok_new.sum() == ok.sum():
            ok = ok_new
            break
        ok = ok_new

    p = np.polyfit(x[ok], y[ok], 1)
    r = y - np.polyval(p, x)
    sig = 1.4826 * np.nanmedian(np.abs(r[ok] - np.nanmedian(r[ok])))
    return p, r, sig, ok


def local_norm(lam, flux):
    sb = (
        ((lam > 752.0) & (lam < 756.5)) |
        ((lam > 768.0) & (lam < 772.0))
    )
    ok = sb & np.isfinite(lam) & np.isfinite(flux)
    if ok.sum() >= 6:
        p = np.polyfit(lam[ok], flux[ok], 1)
        cont = np.polyval(p, lam)
    else:
        cont = np.full_like(flux, np.nanmedian(flux[np.isfinite(flux)]))

    if not np.isfinite(cont).any():
        return flux * np.nan

    return flux / cont

def load_tracecoords_headers():
    paths = [
        Path(config.SCI_EVEN_TRACECOORDS),
        Path(config.SCI_ODD_TRACECOORDS),
    ]

    lookup = {}

    for p in paths:
        if not p.exists():
            continue

        with fits.open(p) as h:
            for ext in h[1:]:
                slit = ext.name.upper()
                if not slit.startswith("SLIT"):
                    continue

                hdr = ext.header
                xwin = hdr.get("XWIN", hdr.get("XMIN", hdr.get("XLO", np.nan)))
                padx = hdr.get("PADX", 0)

                lookup[slit] = dict(
                    XWIN=float(xwin),
                    PADX=float(padx),
                )

    return lookup

def main():
    args = parse_args()

    spectra = Path(args.spectra)
    radec_path = Path(args.radec)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    radec = pd.read_csv(radec_path)
    radec["slit"] = radec["slit"].astype(str).str.upper()
    radec = radec.set_index("slit")

    rows = []

    with fits.open(spectra) as hdul:
        for h in hdul[1:]:
            slit = h.name.upper()
            if not slit.startswith("SLIT"):
                continue
            if slit not in radec.index:
                continue
            
            trace_hdr = load_tracecoords_headers()
            
            d = h.data
            cols = d.columns.names
            if "LAMBDA_NM" not in cols or "YPIX" not in cols or "X0" not in cols:
                continue

            flux_col = args.flux_col if args.flux_col in cols else None
            if flux_col is None:
                for c in ["STELLAR_CONSENSUS", "OBJ_PRESKY", "FLUX", "FLUX_APCORR"]:
                    if c in cols:
                        flux_col = c
                        break
            if flux_col is None:
                continue

            lam = np.asarray(d["LAMBDA_NM"], float)
            flux = np.asarray(d[flux_col], float)
            ypix = np.asarray(d["YPIX"], float)
            x0 = np.asarray(d["X0"], float)

            m_geom = (
                np.isfinite(lam)
                & np.isfinite(ypix)
                & np.isfinite(x0)
                & (lam > args.a_lo)
                & (lam < args.a_hi)
            )
            
            m_aband = m_geom & np.isfinite(flux)
            
            has_telluric = False
            
            if m_aband.sum() >= 10:
                lam_m = lam[m_aband]
                flux_n = local_norm(lam_m, flux[m_aband])
            
                if np.isfinite(flux_n).any():
                    i = int(np.nanargmin(flux_n))
                    idx = np.where(m_aband)[0][i]
                
                    lam_a = float(lam[idx])
                    depth_a = float(1.0 - flux_n[i])
                
                    local_scatter = 1.4826 * np.nanmedian(
                        np.abs(flux_n - np.nanmedian(flux_n))
                    )
                    snr_aband = depth_a / local_scatter if local_scatter > 0 else np.nan
                
                    has_telluric = (
                        np.isfinite(depth_a)
                        and np.isfinite(snr_aband)
                        and depth_a > 0.025
                        and snr_aband > 1.8
                    )
                else:
                    idx = None
                    
            else:
                idx = None
            
            # Fallback: no measurable A-band.
            # Place the slit at the nominal A-band wavelength position,
            # but mark it as not used for the DEC/A-band fit.
            if idx is None:
                if m_geom.sum() < 1:
                    continue
            
                lam_nom = 0.5 * (args.a_lo + args.a_hi)
                geom_idx = np.where(m_geom)[0]
                idx = geom_idx[np.nanargmin(np.abs(lam[geom_idx] - lam_nom))]
            
                lam_a = float(lam[idx])
                depth_a = np.nan
                has_telluric = False
                

            y0det = float(h.header.get("Y0DET", h.header.get("YMIN", 0.0)))

            if slit not in trace_hdr:
                raise KeyError(f"{slit}: missing Step06c TRACECOORDS header lookup")
            
            xwin = trace_hdr[slit]["XWIN"]
            padx = trace_hdr[slit]["PADX"]
            
            ydet_a = y0det + ypix[idx]
            xdet_obj = xwin + x0[idx] - padx
            
            if slit in ["SLIT024", "SLIT039", "SLIT044", "SLIT047"]:
                print(
                    slit,
                    "LAM_A_MIN =", lam_a,
                    "YDET_A =", ydet_a,
                    "HAS_TELLURIC =", has_telluric,
                )

            rr = radec.loc[slit]

            rows.append(dict(
                slit=slit,
                RA=float(rr["RA"]),
                DEC=float(rr["DEC"]),
                XDET_OBJ=float(xdet_obj),
                YDET_A=float(ydet_a),
                LAM_A_MIN=float(lam_a),
                DEPTH_A=float(depth_a) if np.isfinite(depth_a) else np.nan,
                HAS_TELLURIC=bool(has_telluric),
                FLUXCOL=flux_col,
            ))

    df = pd.DataFrame(rows)
    if len(df) < 6:
        raise RuntimeError("Too few usable slits for astrometric sanity fit.")
        
    ra0 = np.nanmedian(df["RA"])
    dec0 = np.nanmedian(df["DEC"])

    df["RA_ARCSEC"] = 3600.0 * (df["RA"] - ra0) * np.cos(np.deg2rad(dec0))
    df["DEC_ARCSEC"] = 3600.0 * (df["DEC"] - dec0)     

    p_ra, res_ra, sig_ra, ok_ra = robust_linefit(df["RA_ARCSEC"], df["XDET_OBJ"], clip=args.clip)
    fit_dec = df["HAS_TELLURIC"].astype(bool)
    
    p_dec, res_dec, sig_dec, ok_dec_fit = robust_linefit(
        df.loc[fit_dec, "DEC_ARCSEC"],
        df.loc[fit_dec, "YDET_A"],
        clip=args.clip,
    )
    
    df["Y_FIT_FROM_DEC"] = np.polyval(p_dec, df["DEC_ARCSEC"])
    df["DY_DEC_FIT"] = df["YDET_A"] - df["Y_FIT_FROM_DEC"]
    
    df["OK_DEC"] = False
    df.loc[fit_dec, "OK_DEC"] = ok_dec_fit

    df["X_FIT_FROM_RA"] = np.polyval(p_ra, df["RA_ARCSEC"])
    df["DX_RA_FIT"] = res_ra
    df["OK_RA"] = ok_ra

    # Invert fits: target-coordinate implied by measured detector position
    df["RA_ARCSEC_FROM_XDET"] = (df["XDET_OBJ"] - p_ra[1]) / p_ra[0]
    df["DEC_ARCSEC_FROM_YDET"] = (df["YDET_A"] - p_dec[1]) / p_dec[0]
    
    df["RA_FROM_XDET"] = ra0 + df["RA_ARCSEC_FROM_XDET"] / (
        3600.0 * np.cos(np.deg2rad(dec0))
    )
    
    df["DEC_FROM_YDET"] = dec0 + df["DEC_ARCSEC_FROM_YDET"] / 3600.0
    
    df["DRA_IMPLIED_ARCSEC"] = (
        3600.0
        * (df["RA_FROM_XDET"] - df["RA"])
        * np.cos(np.deg2rad(dec0))
    )
    
    df["DDEC_IMPLIED_ARCSEC"] = (
        3600.0
        * (df["DEC_FROM_YDET"] - df["DEC"])
    )


    bad_dec = ~df["OK_DEC"]
    bad_ra = ~df["OK_RA"]
    bad_any = bad_ra | bad_dec
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # Panel 1: catalog RA vs detector-inferred RA
    ax = axes[0]
    ax.scatter(df["RA"], df["RA_FROM_XDET"], c=np.where(bad_ra, "r", "k"), s=35)
    lo = min(df["RA"].min(), df["RA_FROM_XDET"].min())
    hi = max(df["RA"].max(), df["RA_FROM_XDET"].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
    for _, r in df.loc[bad_ra].iterrows():
        ax.text(r["RA"], r["RA_FROM_XDET"], r["slit"], fontsize=7)
    ax.set_xlabel("Catalog RA [deg]")
    ax.set_ylabel("Detector-inferred RA from target ridge [deg]")
    ax.set_title(f"RA sanity: rms={sig_ra:.2f} pix")
    ax.grid(alpha=0.25)
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    # Panel 2: catalog DEC vs detector-inferred DEC
    ax = axes[1]
    has_tell = df["HAS_TELLURIC"].astype(bool)
    
    ax.scatter(
        df.loc[has_tell, "DEC"],
        df.loc[has_tell, "DEC_FROM_YDET"],
        c=np.where(bad_dec[has_tell], "r", "k"),
        s=35,
        label="A-band measured",
    )
    
    ax.scatter(
        df.loc[~has_tell, "DEC"],
        df.loc[~has_tell, "DEC_FROM_YDET"],
        marker="x",
        c="0.5",
        s=45,
        label="nominal A-band position",
    )
    
    lo = min(df["DEC"].min(), df["DEC_FROM_YDET"].min())
    hi = max(df["DEC"].max(), df["DEC_FROM_YDET"].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
    for _, r in df.loc[bad_dec].iterrows():
        ax.text(r["DEC"], r["DEC_FROM_YDET"], r["slit"], fontsize=7)
    ax.set_xlabel("Catalog DEC [deg]")
    ax.set_ylabel("Detector-inferred DEC from A-band/CaT-Y [deg]")
    ax.set_title(f"DEC sanity: rms={sig_dec:.2f} pix")
    ax.grid(alpha=0.25)
    
    # ----------------------------
    # Panel 3: RA/DEC sky map
    # Catalog positions plus detector-inferred positions
    # ----------------------------
    ax = axes[2]
    
    bad_dec = ~df["OK_DEC"]
    bad_ra = ~df["OK_RA"]
    bad_any = bad_ra | bad_dec
    
    # catalog positions
    ax.scatter(
        df["RA"],
        df["DEC"],
        c=np.where(bad_any, "r", "k"),
        s=35,
        label="catalog RA/DEC",
    )
    
    has_tell = df["HAS_TELLURIC"].astype(bool)
    no_tell = ~has_tell
    
    # ---------------------------------------
    # GOOD/BAD telluric-derived positions
    # ---------------------------------------
    
    # normal telluric-based positions
    ax.scatter(
        df.loc[has_tell & ~bad_any, "RA_FROM_XDET"],
        df.loc[has_tell & ~bad_any, "DEC_FROM_YDET"],
        facecolors="none",
        edgecolors="0.5",
        s=70,
        linewidths=1.0,
        label="A-band inferred",
    )
    
    # outliers
    ax.scatter(
        df.loc[has_tell & bad_any, "RA_FROM_XDET"],
        df.loc[has_tell & bad_any, "DEC_FROM_YDET"],
        facecolors="none",
        edgecolors="r",
        s=90,
        linewidths=1.3,
        label="A-band outliers",
    )
    
    # ---------------------------------------
    # no telluric detection
    # ---------------------------------------
    
    ax.scatter(
        df.loc[no_tell, "RA_FROM_XDET"],
        df.loc[no_tell, "DEC_FROM_YDET"],
        marker="x",
        c="dodgerblue",   # or "limegreen"
        s=70,
        linewidths=1.5,
        label="no measurable A-band",
    )

    
    # draw catalog -> inferred vectors
    for _, r in df.loc[bad_any].iterrows():
        ax.plot(
            [r["RA"], r["RA_FROM_XDET"]],
            [r["DEC"], r["DEC_FROM_YDET"]],
            color="r",
            lw=0.8,
            alpha=0.7,
        )
        ax.text(r["RA"], r["DEC"], r["slit"], fontsize=7, color="r")
    
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("DEC [deg]")
    ax.set_title("Sky map: catalog → detector-inferred")
    ax.grid(alpha=0.25)
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    
    fig.suptitle("Step10 astrometric sanity: catalog vs detector-inferred coordinates")
    plt.tight_layout()
    
    png_out = outdir / "qc_step10_astrometric_sanity.png"
    fig.savefig(png_out, dpi=160)
    plt.close(fig)

    print("Wrote:")
    #print(" ", csv_out)
    print(" ", png_out)

    print()
    print("XDET_OBJ range:", df["XDET_OBJ"].min(), df["XDET_OBJ"].max())
    
    print("Likely outliers:")
    bad = df[(~df["OK_RA"]) | (~df["OK_DEC"])].copy()
    cols = [
        "slit", "RA", "DEC", "XDET_OBJ", "YDET_A",
        "DX_RA_FIT", "DY_DEC_FIT",
        "RA_FROM_XDET", "DEC_FROM_YDET",
        "DRA_IMPLIED_ARCSEC", "DDEC_IMPLIED_ARCSEC",
    ]
    print(bad[cols].to_string(index=False))

    print()

    print("A-band-implied celestial coordinates for DEC outliers:")
    cols2 = [
        "slit",
        "RA",
        "DEC",
        "RA_FROM_XDET",
        "DEC_FROM_YDET",
        "DRA_IMPLIED_ARCSEC",
        "DDEC_IMPLIED_ARCSEC",
    ]
    print(df.loc[bad_dec, cols2].to_string(index=False))

if __name__ == "__main__":
    main()
