#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC for the final SAMOS Step12d ensemble-response calibration.

Current production model
------------------------
RESP_MASTER(lambda) = C_i * RESP_SHAPE(lambda)

C_i is ONE global scalar chosen to preserve the median Step11 synthetic
SkyMapper i-band flux.  No per-object photometric renormalization is used.

The PDF contains:
  1) RESP_SHAPE, RESP_MASTER, P16--P84 envelope, pivots and trusted range.
  2) Distribution of the individual i-band preservation factors used only
     to derive the one global C_i.
  3) Gray/linear/quadratic LOO shape-fit diagnostics.

The LOO RMS values are explicitly shape-fit diagnostics, NOT measurements
of the external spectrophotometric accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import config

PR = 613.8440330950057
PI = 776.79762950059
PZ = 914.5992987637427


def getpath(name, fallback):
    v = getattr(config, name, None)
    return Path(v) if v is not None else Path(fallback)


def val(row, key, default=np.nan):
    try:
        x = float(row[key])
    except Exception:
        return float(default)
    return x if np.isfinite(x) else float(default)


def main():
    d0 = Path(getattr(config, "ST12_FINALCAL", ".")) / "step12d_stellar_response"

    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(getpath(
        "STEP12D_MASTER_FITS", d0 / "step12d_stellar_response_master.fits")))
    ap.add_argument("--summary", default=str(getpath(
        "STEP12D_SUMMARY_CSV", d0 / "step12d_stellar_response_summary.csv")))
    ap.add_argument("--metadata", default=str(getpath(
        "STEP12D_METADATA_JSON", d0 / "step12d_stellar_response_metadata.json")))
    ap.add_argument("--outpdf", default="")
    args = ap.parse_args()

    pm = Path(args.master)
    ps = Path(args.summary)
    pj = Path(args.metadata)
    po = Path(args.outpdf) if args.outpdf else pm.parent / "qc_step12d_stellar_response.pdf"

    for p in (pm, ps, pj):
        if not p.exists():
            raise FileNotFoundError(p)

    with fits.open(pm, memmap=False) as h:
        if "MASTER_RESPONSE" not in h:
            raise KeyError("MASTER_RESPONSE extension missing")
        d = h["MASTER_RESPONSE"].data
        required = {"LAMBDA_NM", "RESP_MASTER", "RESP_P16", "RESP_P84", "RESP_SHAPE"}
        missing = required - set(d.names)
        if missing:
            raise KeyError(f"MASTER_RESPONSE missing columns: {sorted(missing)}")
        lam = np.asarray(d["LAMBDA_NM"], float)
        master = np.asarray(d["RESP_MASTER"], float)
        p16 = np.asarray(d["RESP_P16"], float)
        p84 = np.asarray(d["RESP_P84"], float)
        shape = np.asarray(d["RESP_SHAPE"], float)

    s = pd.read_csv(ps)
    if len(s) != 1:
        raise RuntimeError(f"Expected one summary row, found {len(s)}")
    s = s.iloc[0]
    meta = json.loads(pj.read_text())

    # Numerical invariants.
    if not np.all(np.isfinite(lam)) or not np.all(np.diff(lam) > 0):
        raise RuntimeError("Invalid wavelength grid")
    for name, a in (
        ("RESP_MASTER", master), ("RESP_P16", p16),
        ("RESP_P84", p84), ("RESP_SHAPE", shape)
    ):
        if not np.all(np.isfinite(a)) or np.any(a <= 0):
            raise RuntimeError(f"Invalid {name}")
    if np.any(p16 > p84):
        raise RuntimeError("RESP_P16 > RESP_P84")

    ci = val(s, "global_i_norm", meta.get("global_i_norm", np.nan))
    if not np.isfinite(ci):
        raise RuntimeError("global_i_norm unavailable")

    ratio = master / shape
    ratio_med = float(np.median(ratio))
    ratio_spread = float(np.max(np.abs(ratio - ratio_med)))
    tol = 5e-7 * max(1.0, abs(ci))
    if ratio_spread > tol or abs(ratio_med - ci) > tol:
        raise RuntimeError(
            f"RESP_MASTER/RESP_SHAPE inconsistent with C_i: "
            f"median={ratio_med:.10f}, C_i={ci:.10f}, spread={ratio_spread:.3e}"
        )

    rows = []
    for r in meta.get("global_i_normalization_rows", []):
        try:
            slit = str(r["slit"])
            fac = float(r["factor"])
        except Exception:
            continue
        if np.isfinite(fac):
            rows.append((slit, fac))
    facs = np.asarray([x[1] for x in rows], float)

    po.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(po) as pdf:
        # --------------------------------------------------------------
        # Page 1: adopted response
        # --------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.fill_between(lam, p16, p84, alpha=0.22,
                        label="LOO 16--84% envelope")
        ax.plot(lam, shape, ls="--", lw=1.6,
                label="RESP_SHAPE (pivot-normalized)")
        ax.plot(lam, master, lw=2.2,
                label="RESP_MASTER = C_i x RESP_SHAPE")

        xlo, xhi = 590.0, 960.0
        ax.plot([xlo, lam[0]], [master[0], master[0]], ls=":", lw=1.4,
                label="Step12e boundary hold")
        ax.plot([lam[-1], xhi], [master[-1], master[-1]], ls=":", lw=1.4)

        for w, b in ((PR, "r"), (PI, "i"), (PZ, "z")):
            y = float(np.interp(w, lam, master))
            ax.axvline(w, lw=0.8, alpha=0.45)
            ax.scatter([w], [y], s=36, zorder=5)
            ax.annotate(f"{b}: R={y:.3f}", (w, y),
                        xytext=(5, 7), textcoords="offset points", fontsize=9)

        sr = float(np.interp(PR, lam, shape))
        si = float(np.interp(PI, lam, shape))
        sz = float(np.interp(PZ, lam, shape))
        mr = float(np.interp(PR, lam, master))
        mi = float(np.interp(PI, lam, master))
        mz = float(np.interp(PZ, lam, master))

        txt = (
            f"C_i = {ci:.8f}\n"
            f"Trusted interval: {lam[0]:.3f}--{lam[-1]:.3f} nm\n"
            f"Shape pivots: r={sr:.4f}, i={si:.4f}, z={sz:.4f}\n"
            f"Production pivots: r={mr:.4f}, i={mi:.4f}, z={mz:.4f}\n"
            "Outside trusted interval: nearest boundary response is held constant."
        )
        ax.text(0.02, 0.04, txt, transform=ax.transAxes, va="bottom",
            fontsize=10, bbox=dict(boxstyle="round", alpha=0.08))
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(0, np.nanpercentile(p84, 99) * 1.15)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Multiplicative response")
        ax.set_title("Step12d production ensemble response")
        ax.grid(alpha=0.20)
        ax.legend(fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --------------------------------------------------------------
        # Page 2: global normalization stability
        # --------------------------------------------------------------
        fig = plt.figure(figsize=(11, 8.5))
        gs = fig.add_gridspec(2, 1)
        a1 = fig.add_subplot(gs[0, 0])
        a2 = fig.add_subplot(gs[1, 0])

        if facs.size:
            q16, q50, q84 = np.percentile(facs, [16, 50, 84])
            a1.hist(facs, bins="auto", alpha=0.75)
            a1.axvline(ci, lw=2, label=f"adopted C_i={ci:.8f}")
            a1.axvspan(q16, q84, alpha=0.15,
                       label=f"16--84%={q16:.6f}--{q84:.6f}")
            a1.legend()
            a1.set_xlabel("Individual i-band preservation factor")
            a1.set_ylabel("N")
            a1.grid(alpha=0.20)

            x = np.arange(len(facs))
            a2.scatter(x, facs, s=24)
            a2.axhline(ci, lw=1.6)
            a2.axhspan(q16, q84, alpha=0.12)
            step = max(1, len(rows)//12)
            a2.set_xticks(x[::step])
            a2.set_xticklabels([r[0] for r in rows][::step],
                               rotation=45, ha="right", fontsize=8)
            a2.set_ylabel("i-band preservation factor")
            a2.set_xlabel("Spectrum")
            a2.grid(alpha=0.20)

            hw = 100.0 * (q84-q16) / (2.0*ci)
            fig.suptitle(
                f"One global normalization for all spectra: "
                f"N={len(facs)}, median={q50:.8f}, half-width~{hw:.2f}%",
                fontsize=13
            )
        else:
            for a in (a1, a2):
                a.axis("off")
            a1.text(0.5, 0.5, "No normalization rows found in metadata",
                    ha="center", va="center")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # --------------------------------------------------------------
        # Page 3: LOO shape-fit diagnostics
        # --------------------------------------------------------------
        labels = ["Gray", "Linear", "Quadratic"]
        loo = np.asarray([
            val(s, "loo_gray_median_mag"),
            val(s, "loo_linear_median_mag"),
            val(s, "loo_quadratic_median_mag"),
        ], float)

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.bar(np.arange(3), loo, alpha=0.78)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Median leave-one-out RMS [mag]")
        ax.set_title("Step12d ensemble shape-fit validation")
        ax.grid(axis="y", alpha=0.20)

        for b, y in zip(bars, loo):
            if np.isfinite(y):
                ax.text(b.get_x()+b.get_width()/2, y, f"{y:.4f}",
                        ha="center", va="bottom", fontsize=11)

        nstar = val(s, "n_stars")
        qbetter = val(s, "quadratic_better_than_gray_n")
        head = []
        if np.isfinite(nstar):
            head.append(f"LOO sample: N={int(round(nstar))}")
        if np.isfinite(nstar) and np.isfinite(qbetter):
            head.append(
                f"Quadratic lower RMS than gray: "
                f"{int(round(qbetter))}/{int(round(nstar))}"
            )
        note = (
            "\n".join(head) + ("\n\n" if head else "") +
            "IMPORTANT:\n"
            "These are ensemble SHAPE-FIT diagnostics.\n"
            "They are NOT the external spectrophotometric accuracy.\n"
            "Broad-band/external closure must be quoted separately."
        )
        ax.text(0.03, 0.96, note, transform=ax.transAxes, va="top",
                fontsize=10, bbox=dict(boxstyle="round", alpha=0.08))
        finite = loo[np.isfinite(loo)]
        if finite.size:
            ax.set_ylim(0, max(0.38, 1.30*np.max(finite)))
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print("[OK] Step12d QC validation")
    print("  master              :", pm)
    print("  summary             :", ps)
    print("  metadata            :", pj)
    print("  response samples    :", len(lam))
    print(f"  trusted interval    : {lam[0]:.6f} .. {lam[-1]:.6f} nm")
    print(f"  global i norm       : {ci:.8f}")
    print(f"  master/shape spread : {ratio_spread:.3e}")
    print("  normalization N     :", len(facs))
    if facs.size:
        print("  norm 16/50/84       :",
              " ".join(f"{x:.8f}" for x in np.percentile(facs, [16, 50, 84])))
    print("  LOO gray/lin/quad   :",
          f"{loo[0]:.4f}", f"{loo[1]:.4f}", f"{loo[2]:.4f}",
          "mag  [shape-fit diagnostic]")
    print("  wrote               :", po)


if __name__ == "__main__":
    main()
