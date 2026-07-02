#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QC Step08a2 — three-spectrum mosaic per slit.

For each slit, plot in the same panel:
A) OBJ_PRESKY  (black)
B) OBJ = FLUX  (red)
C) SKY = OBJ_PRESKY - OBJ  (blue)

The panel layout follows the usual slit mosaic ordering.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import config


def slit_num(name: str) -> int:
    m = re.match(r"SLIT(\d+)", str(name).upper())
    return int(m.group(1)) if m else 10**9


def pick_extract(set_tag: str) -> Path:
    path = Path(config.ST08_EXTRACT1D) / f"extract1d_optimal_ridge_{set_tag.lower()}.fits"
    if not path.exists():
        raise FileNotFoundError(f"No Step08a2 extract file found for {set_tag}: {path}")
    return path


def finite_limits(arrs, p_lo=1.0, p_hi=99.0):
    vals = []
    for a in arrs:
        a = np.asarray(a, float)
        vals.append(a[np.isfinite(a)])
    usable = [v for v in vals if v.size > 0]
    vals = np.concatenate(usable) if usable else np.array([])
    if vals.size == 0:
        return -1.0, 1.0
    lo = np.nanpercentile(vals, p_lo)
    hi = np.nanpercentile(vals, p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        med = np.nanmedian(vals)
        sig = np.nanstd(vals)
        if not np.isfinite(sig) or sig <= 0:
            sig = 1.0
        return med - 2 * sig, med + 4 * sig
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser(description="QC Step08a2 three-spectrum mosaic")
    ap.add_argument("--set", required=True, choices=["EVEN", "ODD", "even", "odd"])
    ap.add_argument("--file08", default=None, help="Explicit Step08a2 extract FITS")
    ap.add_argument("--slit-min", type=int, default=None)
    ap.add_argument("--slit-max", type=int, default=None)
    ap.add_argument("--ncols", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--normalize", action="store_true",
                    help="Normalize each spectrum by max(abs(OBJ_PRESKY)) for shape comparison")
    args = ap.parse_args()

    set_tag = args.set.upper()
    file08 = Path(args.file08) if args.file08 else pick_extract(set_tag)
    outdir = Path(args.outdir) if args.outdir else Path(config.ST08_EXTRACT1D) / "qc_step08"
    outdir.mkdir(parents=True, exist_ok=True)

    with fits.open(file08) as hdul:
        slit_names = []
        slit_data = {}
        for h in hdul[1:]:
            nm = (h.name or "").upper()
            if not nm.startswith("SLIT") or h.data is None:
                continue
            names = [n.upper() for n in h.data.names]
            if "FLUX" not in names or "OBJ_PRESKY" not in names:
                continue
            s = slit_num(nm)
            if args.slit_min is not None and s < args.slit_min:
                continue
            if args.slit_max is not None and s > args.slit_max:
                continue

            if "YPIX" in names:
                y = np.asarray(h.data["YPIX"], float)
            elif "YYPIX" in names:
                y = np.asarray(h.data["YYPIX"], float)
            else:
                y = np.arange(len(h.data), dtype=float)

            obj_presky = np.asarray(h.data["OBJ_PRESKY"], float)
            obj = np.asarray(h.data["FLUX"], float)
            sky = obj_presky - obj

            if args.normalize:
                good = np.isfinite(obj_presky)
                scale = np.nanmax(np.abs(obj_presky[good])) if np.any(good) else np.nan
                if np.isfinite(scale) and scale > 0:
                    obj_presky = obj_presky / scale
                    obj = obj / scale
                    sky = sky / scale

            slit_names.append(nm)
            slit_data[nm] = (y, obj_presky, obj, sky, h.header.copy())

    slit_names = sorted(slit_names, key=slit_num)
    if not slit_names:
        raise RuntimeError("No usable SLIT### spectra found with FLUX and OBJ_PRESKY columns")

    ncols = max(1, int(args.ncols))
    nrows = math.ceil(len(slit_names) / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.5 * ncols, 2.8 * nrows),
        squeeze=False
    )

    for ax in axes.ravel():
        ax.set_visible(False)

    for ax, slit in zip(axes.ravel(), slit_names):
        ax.set_visible(True)
        y, obj_presky, obj, sky, hdr = slit_data[slit]

        ymin, ymax = finite_limits([obj_presky, obj, sky], p_lo=1.0, p_hi=99.0)

        ax.plot(y, obj_presky, color="black", lw=0.9, label="OBJ_PRESKY")
        ax.plot(y, obj, color="red", lw=0.9, label="OBJ")
        ax.plot(y, sky, color="blue", lw=0.9, label="SKY")
        
        # choose threshold based on data scale
        vals = np.concatenate([
            obj_presky[np.isfinite(obj_presky)],
            obj[np.isfinite(obj)],
            sky[np.isfinite(sky)],
        ])
        
        if vals.size > 0:
            linthresh = 0.05 * np.nanpercentile(np.abs(vals), 90)
            linthresh = max(linthresh, 1e-6)
        else:
            linthresh = 1.0
        
        ax.set_yscale("symlog", linthresh=linthresh)

        ax.axhline(0.0, color="0.7", lw=0.5)

        cls = hdr.get("S08CLAS", "")
        use = hdr.get("S08USE", hdr.get("S08GOOD", ""))
        subtitle = f"{slit}"
        if cls != "":
            subtitle += f"  {cls}"
        if use != "":
            subtitle += f"  USE={use}"
        ax.set_title(subtitle, fontsize=8)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([])
        ax.set_yticks([])

    first_ax = None
    for ax in axes.ravel():
        if ax.get_visible():
            first_ax = ax
            break
    if first_ax is not None:
        first_ax.legend(loc="upper right", fontsize=6, framealpha=0.85)

    stem = f"{set_tag.lower()}_three_spectra_montage"
    if args.normalize:
        stem += "_norm"
    outpng = outdir / f"{stem}.png"

    fig.suptitle(stem, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpng, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"STEP08a2 : {file08}")
    print(f"Wrote    : {outpng}")


if __name__ == "__main__":
    main()
