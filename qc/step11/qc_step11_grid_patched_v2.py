#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step11 QC — multi-panel overview of flux-calibrated spectra.

Patched to restore the auxiliary context pages:
- a stellar-field page using the Step11 SkyMapper table (RA/DEC + magnitudes)
- a slit-trace overview page using x_center / xlo / xhi / YMIN when available
- the original flux-calibrated spectrum grid pages
- a legend explaining the stellar-field symbols
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import config

ST11 = Path(config.ST11_FLUXCAL)

FITS_IN = ST11 / "Extract1D_fluxcal.fits"
SUMCSV = ST11 / "Step11_fluxcal_summary.csv"
SKYCSV = ST11 / "slit_trace_radec_skymapper_all.csv"
RADECCSV = ST11 / "slit_trace_radec_all.csv"
OUTPDF = ST11 / "qc_step11" / "qc_step11_fluxcal_grid.pdf"

XLIM_NM = (600, 980)
NCOL = 4
NROW = 8
PER_PAGE = NCOL * NROW


def find_latest(path: Path, patterns):
    hits = []
    for pat in patterns:
        hits.extend(path.glob(pat))
    if not hits:
        return None
    hits = sorted(set(hits), key=lambda p: p.stat().st_mtime)
    return hits[-1]


def _get_col(tbl, *names):
    cols = tbl.columns.names
    for n in names:
        if n in cols:
            return n
    return None


def _read(hdu):
    d = hdu.data
    if d is None or not hasattr(d, "columns"):
        return None
    lamc = _get_col(d, "LAMBDA_NM")
    ycol = _get_col(d, "FLUX_FLAM", "FLAM", "FLUX_CAL_FLAM", "FLUX_TELLCOR_O2", "FLUX")
    if lamc is None or ycol is None:
        return None
    lam = np.asarray(d[lamc], float)
    y = np.asarray(d[ycol], float)
    m = np.isfinite(lam) & np.isfinite(y)
    if m.sum() < 2:
        return None
    return lam[m], y[m], (lamc, ycol)


def _norm_slit_series(s):
    return s.astype(str).str.strip().str.upper()


def _pick_df_col(df, *names):
    cols = {c.upper(): c for c in df.columns}
    for n in names:
        if n.upper() in cols:
            return cols[n.upper()]
    return None


def _load_optional_csv(path: Path):
    if path is not None and path.exists():
        return pd.read_csv(path)
    return None


def _robust_ylim(y):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (-1.0, 1.0)
    lo, hi = np.nanpercentile(y, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        med = np.nanmedian(y)
        return (med - 1.0, med + 1.0)
    pad = 0.08 * (hi - lo)
    return (lo - pad, hi + pad)


def _draw_field_page(pdf, sky_df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    ax1, ax2 = axes

    if sky_df is not None and {"RA", "DEC", "slit"}.issubset(sky_df.columns):
        sdf = sky_df.copy()
        sdf["slit"] = _norm_slit_series(sdf["slit"])
        rcol = _pick_df_col(sdf, "r_mag")
        sepcol = _pick_df_col(sdf, "match_sep_arcsec")

        x = pd.to_numeric(sdf["RA"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(sdf["DEC"], errors="coerce").to_numpy(float)

        legend_handles = []
        used_color = False
        if rcol is not None:
            c = pd.to_numeric(sdf[rcol], errors="coerce").to_numpy(float)
            goodc = np.isfinite(c) & np.isfinite(x) & np.isfinite(y)
            if goodc.any():
                sc = ax1.scatter(x[goodc], y[goodc], c=c[goodc], s=28)
                cbar = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
                cbar.set_label("r_mag")
                legend_handles.append(Line2D([0], [0], marker='o', linestyle='None',
                                             markersize=6, markerfacecolor='0.6',
                                             markeredgecolor='0.6',
                                             label='matched source, colored by r_mag'))
                used_color = True
            bad = (~np.isfinite(c)) & np.isfinite(x) & np.isfinite(y)
            if bad.any():
                ax1.scatter(x[bad], y[bad], marker="x", s=30)
                legend_handles.append(Line2D([0], [0], marker='x', linestyle='None',
                                             markersize=6, color='k',
                                             label='match exists, r_mag missing'))
        else:
            good = np.isfinite(x) & np.isfinite(y)
            if good.any():
                ax1.scatter(x[good], y[good], s=28)
                legend_handles.append(Line2D([0], [0], marker='o', linestyle='None',
                                             markersize=6, color='k',
                                             label='matched source'))

        for _, row in sdf.iterrows():
            try:
                ax1.text(float(row["RA"]), float(row["DEC"]), str(row["slit"])[-3:], fontsize=6)
            except Exception:
                pass
        legend_handles.append(Line2D([0], [0], linestyle='None', label='text label = slit number'))

        ax1.set_title("Step11 stellar field (SkyMapper matches)")
        ax1.set_xlabel("RA (deg)")
        ax1.set_ylabel("DEC (deg)")
        ax1.invert_xaxis()
        ax1.legend(handles=legend_handles, loc="best", fontsize=8, frameon=True)

        if sepcol is not None:
            sep = pd.to_numeric(sdf[sepcol], errors="coerce").to_numpy(float)
            sep = sep[np.isfinite(sep)]
            if sep.size:
                ax2.hist(sep, bins=20)
                ax2.set_title("SkyMapper match separation")
                ax2.set_xlabel("arcsec")
                ax2.set_ylabel("N")
                ax2.axvline(0.5, ls="--", lw=1)
                ax2.axvline(1.0, ls="--", lw=1)
                ax2.text(0.98, 0.95,
                         "dashed lines: 0.5, 1.0 arcsec",
                         transform=ax2.transAxes, ha="right", va="top", fontsize=8)
            else:
                ax2.text(0.5, 0.5, "No finite match separations", ha="center", va="center")
                ax2.set_axis_off()
        else:
            ax2.text(0.5, 0.5, "No match_sep_arcsec column", ha="center", va="center")
            ax2.set_axis_off()
    else:
        ax1.text(0.5, 0.5, "No stellar-field table found", ha="center", va="center")
        ax1.set_axis_off()
        ax2.text(0.5, 0.5, "No SkyMapper table found", ha="center", va="center")
        ax2.set_axis_off()

    fig.suptitle("Step11 QC — field / catalog context", y=0.98, fontsize=12)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def _draw_trace_page(pdf, radec_df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    ax1, ax2 = axes

    if radec_df is None or "slit" not in radec_df.columns:
        ax1.text(0.5, 0.5, "No slit trace table found", ha="center", va="center")
        ax1.set_axis_off()
        ax2.set_axis_off()
        pdf.savefig(fig)
        plt.close(fig)
        return

    df = radec_df.copy()
    df["slit"] = _norm_slit_series(df["slit"])

    slitnum_col = _pick_df_col(df, "SLITNUM")
    x_center_col = _pick_df_col(df, "x_center", "XREF", "X0")
    xlo_col = _pick_df_col(df, "xlo", "XLO")
    xhi_col = _pick_df_col(df, "xhi", "XHI")
    ymin_col = _pick_df_col(df, "YMIN")
    src_col = _pick_df_col(df, "SRC")

    if slitnum_col is not None and x_center_col is not None:
        sn = pd.to_numeric(df[slitnum_col], errors="coerce").to_numpy(float)
        xc = pd.to_numeric(df[x_center_col], errors="coerce").to_numpy(float)
        ok = np.isfinite(sn) & np.isfinite(xc)
        ax1.plot(sn[ok], xc[ok], marker="o", lw=0.8, ms=3)
        if xlo_col is not None and xhi_col is not None:
            xlo = pd.to_numeric(df[xlo_col], errors="coerce").to_numpy(float)
            xhi = pd.to_numeric(df[xhi_col], errors="coerce").to_numpy(float)
            ok2 = ok & np.isfinite(xlo) & np.isfinite(xhi)
            for s, lo, hi in zip(sn[ok2], xlo[ok2], xhi[ok2]):
                ax1.vlines(s, lo, hi, lw=0.7)
        ax1.set_title("Slit trace overview")
        ax1.set_xlabel("Slit number")
        ax1.set_ylabel("x center / bounds")
    else:
        ax1.text(0.5, 0.5, "No x_center/slitnum columns", ha="center", va="center")
        ax1.set_axis_off()

    if slitnum_col is not None and ymin_col is not None:
        sn = pd.to_numeric(df[slitnum_col], errors="coerce").to_numpy(float)
        yy = pd.to_numeric(df[ymin_col], errors="coerce").to_numpy(float)
        ok = np.isfinite(sn) & np.isfinite(yy)
        if src_col is not None:
            src = df[src_col].astype(str)
            for label in sorted(src.dropna().unique()):
                m = ok & (src == label).to_numpy()
                if m.any():
                    ax2.scatter(sn[m], yy[m], s=18, label=label)
            ax2.legend(loc="best", fontsize=8)
        else:
            ax2.scatter(sn[ok], yy[ok], s=18)
        ax2.set_title("Trace YMIN / provenance")
        ax2.set_xlabel("Slit number")
        ax2.set_ylabel("YMIN")
    else:
        ax2.text(0.5, 0.5, "No YMIN/slitnum columns", ha="center", va="center")
        ax2.set_axis_off()

    fig.suptitle("Step11 QC — slit trace context", y=0.98, fontsize=12)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def main():
    global FITS_IN, SUMCSV, SKYCSV, RADECCSV

    if not FITS_IN.exists():
        FITS_IN = find_latest(ST11, ["*fluxcal*.fits", "Extract1D_fluxcal.fits"])
    if not SUMCSV.exists():
        SUMCSV = find_latest(ST11, ["*fluxcal_summary*.csv", "Step11_fluxcal_summary.csv"])
    if not SKYCSV.exists():
        SKYCSV = find_latest(ST11, ["*skymapper*.csv"])
    if not RADECCSV.exists():
        RADECCSV = find_latest(ST11, ["*radec_all*.csv"])

    if FITS_IN is None or not FITS_IN.exists():
        raise FileNotFoundError("No Step11 fluxcal FITS found")

    OUTPDF.parent.mkdir(parents=True, exist_ok=True)
    summary = _load_optional_csv(SUMCSV)
    sky_df = _load_optional_csv(SKYCSV)
    radec_df = _load_optional_csv(RADECCSV)

    with fits.open(FITS_IN) as hdul:
        hmap = {h.name.strip().upper(): h for h in hdul[1:] if h.name}
        slits = list(hmap.keys())

        if summary is not None and "slit" in summary.columns:
            summary["slit"] = _norm_slit_series(summary["slit"])
            good = np.ones(len(summary), dtype=bool)
            if "qcflag" in summary.columns:
                good = summary["qcflag"].astype(str).str.upper().isin(["GOOD", "OK"])
            if "S" in summary.columns:
                sval = pd.to_numeric(summary["S"], errors="coerce").fillna(0).to_numpy(float)
                order = np.argsort(-np.abs(sval))
                summary = summary.iloc[order]
            slits = summary.loc[good, "slit"].tolist() or summary["slit"].tolist()

        with PdfPages(OUTPDF) as pdf:
            _draw_field_page(pdf, sky_df)
            _draw_trace_page(pdf, radec_df)

            for i0 in range(0, len(slits), PER_PAGE):
                batch = slits[i0:i0+PER_PAGE]
                fig, axes = plt.subplots(NROW, NCOL, figsize=(11, 14), sharex=True)
                axes = axes.ravel()

                for ax, slit in zip(axes, batch):
                    hdu = hmap.get(slit)
                    if hdu is None:
                        ax.axis("off")
                        continue
                    parsed = _read(hdu)
                    if parsed is None:
                        ax.axis("off")
                        continue

                    lam, y, cols = parsed
                    ax.plot(lam, y, lw=0.8)
                    ax.set_xlim(*XLIM_NM)
                    ax.text(0.02, 0.92, slit, transform=ax.transAxes, fontsize=8, va="top")
                    ax.axvspan(686.0, 689.0, alpha=0.08)
                    ax.axvspan(760.0, 770.0, alpha=0.08)
                    ax.set_ylim(*_robust_ylim(y))

                for ax in axes[len(batch):]:
                    ax.axis("off")

                fig.suptitle("Step11 QC — flux-calibrated spectra", y=0.995, fontsize=12)
                fig.text(0.5, 0.005, "Wavelength (nm)", ha="center")
                fig.tight_layout(rect=[0.03, 0.02, 0.98, 0.985])
                pdf.savefig(fig)
                plt.close(fig)

    print("[DONE] Wrote:", OUTPDF)


if __name__ == "__main__":
    main()
