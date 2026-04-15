#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
from astropy.table import Table


def arr(tab, col):
    x = np.asarray(tab[col], float)
    return x


def finite_ylim(*arrays, qlo=0.5, qhi=99.5):
    vals = []
    for a in arrays:
        a = np.asarray(a, float)
        m = np.isfinite(a)
        if np.any(m):
            vals.append(a[m])
    if not vals:
        return (-1, 1)
    x = np.concatenate(vals)
    return (np.nanpercentile(x, qlo), np.nanpercentile(x, qhi))


def robust_rms(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    med = np.nanmedian(x)
    return float(np.sqrt(np.nanmedian((x - med) ** 2)))


def read_slit(fits_path: Path, slit: str):
    with fits.open(fits_path) as hdul:
        return Table(hdul[slit].data)


def make_main_page(slit, lam, flux, c1, r1, oh1, st1, rp1, c2, r2, oh2, st2, rp2):
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(f"Step09 A/B/A/B comparison - {slit}", fontsize=16)

    ax = fig.add_subplot(221)
    ax.plot(lam, flux, lw=0.6, color="0.6", label="OBJ_PRESKY")
    ax.plot(lam, c1, lw=1.0, label="CONTINUUM_P1")
    ax.plot(lam, c2, lw=1.0, label="CONTINUUM_P2")
    ax.set_title("Continuum comparison")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Signal")
    ax.set_ylim(*finite_ylim(flux, c1, c2))
    ax.legend(fontsize=8)

    ax = fig.add_subplot(222)
    ax.plot(lam, r1, lw=0.7, label="RESID_P1")
    ax.plot(lam, r2, lw=0.7, label="RESID_P2")
    ax.axhline(0, lw=0.8, ls="--", color="0.5")
    ax.set_title(
        f"Continuum residuals   RMS P1={robust_rms(r1):.6f}   P2={robust_rms(r2):.6f}"
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Residual")
    ax.set_ylim(*finite_ylim(r1, r2))
    ax.legend(fontsize=8)

    ax = fig.add_subplot(223)
    ax.plot(lam, oh1, lw=0.8, label="OH_MODEL_P1")
    ax.plot(lam, oh2, lw=0.8, label="OH_MODEL_FINAL")
    ax.set_title("OH model comparison")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Model")
    ax.set_ylim(*finite_ylim(oh1, oh2))
    ax.legend(fontsize=8)

    ax = fig.add_subplot(224)
    ax.plot(lam, rp1, lw=0.7, label="RESID_POSTOH_P1")
    ax.plot(lam, rp2, lw=0.7, label="RESID_POSTOH_FINAL")
    ax.axhline(0, lw=0.8, ls="--", color="0.5")
    ax.set_title(
        f"Final residuals   RMS B1={robust_rms(rp1):.6f}   B2={robust_rms(rp2):.6f}"
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Residual")
    ax.set_ylim(*finite_ylim(rp1, rp2))
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def make_zoom_page(slit, lam, st1, st2, rp1, rp2, lo, hi):
    m = np.isfinite(lam) & (lam >= lo) & (lam <= hi)

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(f"{slit}   Zoom {lo}-{hi} nm", fontsize=15)

    ax = fig.add_subplot(121)
    ax.plot(lam[m], st1[m], lw=0.8, label="STELLAR_P1")
    ax.plot(lam[m], st2[m], lw=0.8, label="STELLAR_FINAL")
    ax.set_title("Line-cleaned stellar spectrum")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Signal")
    ax.set_ylim(*finite_ylim(st1[m], st2[m]))
    ax.legend(fontsize=8)

    ax = fig.add_subplot(122)
    ax.plot(lam[m], rp1[m], lw=0.8, label="RESID_POSTOH_P1")
    ax.plot(lam[m], rp2[m], lw=0.8, label="RESID_POSTOH_FINAL")
    ax.axhline(0, lw=0.8, ls="--", color="0.5")
    ax.set_title("Residual comparison")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Residual")
    ax.set_ylim(*finite_ylim(rp1[m], rp2[m]))
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def parse_args():
    p = argparse.ArgumentParser(description="QC for Step09 A/B/A/B one-slit comparison")
    p.add_argument("--pass1a", type=Path, required=True)
    p.add_argument("--pass1b", type=Path, required=True)
    p.add_argument("--pass2a", type=Path, required=True)
    p.add_argument("--final", type=Path, required=True)
    p.add_argument("--slit", type=str, required=True)
    p.add_argument("--out-pdf", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    slit = args.slit.strip().upper()

    t1a = read_slit(args.pass1a, slit)
    t1b = read_slit(args.pass1b, slit)
    t2a = read_slit(args.pass2a, slit)
    tf  = read_slit(args.final,  slit)

    lam  = arr(t1a, "LAMBDA_NM")
    flux = arr(t1a, "OBJ_PRESKY")
    c1   = arr(t1a, "CONTINUUM_P1")
    r1   = arr(t1a, "RESID_P1")

    oh1  = arr(t1b, "OH_MODEL_P1")
    st1  = arr(t1b, "STELLAR_P1")
    rp1  = arr(t1b, "RESID_POSTOH_P1")

    c2   = arr(t2a, "CONTINUUM_P2")
    r2   = arr(t2a, "RESID_P2")

    oh2  = arr(tf, "OH_MODEL_FINAL")
    st2  = arr(tf, "STELLAR_FINAL")
    rp2  = arr(tf, "RESID_POSTOH_FINAL")

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(args.out_pdf) as pdf:
        pdf.savefig(make_main_page(slit, lam, flux, c1, r1, oh1, st1, rp1, c2, r2, oh2, st2, rp2))
        for lo, hi in [(628, 636), (685, 690), (758, 770), (770, 900)]:
            pdf.savefig(make_zoom_page(slit, lam, st1, st2, rp1, rp2, lo, hi))

    print("WROTE", args.out_pdf)


if __name__ == "__main__":
    main()