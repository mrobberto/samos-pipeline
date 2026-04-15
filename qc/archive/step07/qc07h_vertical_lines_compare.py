#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QC75_VerticalLines_wl_All.py

Visual wavelength-space stack QC for Step07 outputs.

Improvements over the older version:
- supports an explicit baseline/reference FITS for before/after comparison
- can highlight one slit row so single-slit changes are visible
- can plot a difference image (current - reference)
- prints the plotted row index of the highlighted slit
- handles missing/degenerate rows more defensively
"""

import argparse
from pathlib import Path

from astropy.io import fits
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config


def first_existing(paths):
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            return p
    return None


def pick_input_file(explicit=None):
    st07 = Path(config.ST07_WAVECAL)
    return first_existing([
        Path(explicit).expanduser() if explicit else None,
        st07 / "arc_1d_wavelength_all.fits",
        *sorted(st07.glob("*_1D_wavelength_ALL.fits")),
        *sorted(st07.glob("*wavelength*ALL*.fits")),
    ])


def load_slit_data(fits_path: Path):
    slits = []
    waves = []
    fluxes = []

    with fits.open(fits_path) as h:
        for ext in h[1:]:
            slit = (ext.header.get("EXTNAME") or ext.name or "").strip().upper()
            if not slit.startswith("SLIT"):
                continue

            arr = np.asarray(ext.data, float)
            if arr.ndim < 2 or arr.shape[0] < 2:
                continue

            flux = arr[0].astype(float)
            lam = arr[1].astype(float)

            m = np.isfinite(flux) & np.isfinite(lam)
            if m.sum() < 10:
                continue

            lam_use = lam[m]
            flux_use = flux[m]

            order = np.argsort(lam_use)
            lam_use = lam_use[order]
            flux_use = flux_use[order]

            # remove duplicate x values for np.interp
            keep = np.diff(lam_use, prepend=lam_use[0] - 1e-9) > 0
            lam_use = lam_use[keep]
            flux_use = flux_use[keep]

            if lam_use.size < 10:
                continue

            slits.append(slit)
            waves.append(lam_use)
            fluxes.append(flux_use)

    if not waves:
        raise RuntimeError(f"No valid slit spectra found in {fits_path}")

    # sort according to slit names not index        
    # =====================================
    records = list(zip(slits, waves, fluxes))
    records.sort(key=lambda r: r[0])   # sort by slit name, e.g. SLIT000, SLIT001, ...
    #
    slits  = [r[0] for r in records]
    waves  = [r[1] for r in records]
    fluxes = [r[2] for r in records]        

    return slits, waves, fluxes


def choose_window(waves, xlim):
    lam_min = max(w[0] for w in waves)
    lam_max = min(w[-1] for w in waves)

    if xlim is not None:
        lam_min = max(lam_min, min(xlim))
        lam_max = min(lam_max, max(xlim))

    if not np.isfinite(lam_min) or not np.isfinite(lam_max) or lam_max <= lam_min:
        raise RuntimeError("No valid overlap in requested wavelength window.")

    return lam_min, lam_max


def build_stack(slits, waves, fluxes, lam_grid):
    stack = []
    kept_slits = []

    for slit, lam, flux in zip(slits, waves, fluxes):
        f_interp = np.interp(lam_grid, lam, flux, left=np.nan, right=np.nan)

        good = np.isfinite(f_interp)
        if np.sum(good) < 10:
            continue

        p99 = np.nanpercentile(f_interp[good], 99)
        if np.isfinite(p99) and p99 > 0:
            f_interp = f_interp / p99

        stack.append(f_interp)
        kept_slits.append(slit)

    if not stack:
        raise RuntimeError("No slit rows survived interpolation.")

    return np.asarray(stack, float), kept_slits


def robust_image_limits(img, plo=5, phi=99):
    v = img[np.isfinite(img)]
    if v.size == 0:
        return 0.0, 1.0
    lo = np.nanpercentile(v, plo)
    hi = np.nanpercentile(v, phi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float(np.nanmin(v)), float(np.nanmax(v))
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description="Visual wavelength-space stack QC")
    ap.add_argument("--fits", default=None, help="Explicit wavelength-ALL FITS to display")
    ap.add_argument("--ref-fits", default=None,
                    help="Optional baseline/reference wavelength-ALL FITS for side-by-side comparison")
    ap.add_argument("--show-plots", action="store_true")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--xlim", nargs=2, type=float, default=None,
                    help="Optional wavelength window, e.g. --xlim 740 830")
    ap.add_argument("--lines", nargs="*", type=float, default=[],
                    help="Optional vertical guide lines in nm")
    ap.add_argument("--highlight-slit", default=None,
                    help="Optional slit name to highlight, e.g. SLIT001")
    ap.add_argument("--ngrid", type=int, default=2000,
                    help="Number of wavelength grid points for the stack")
    args = ap.parse_args()

    fits_path = pick_input_file(args.fits)
    if fits_path is None:
        raise FileNotFoundError(f"No Step07 wavelength-ALL FITS found in {config.ST07_WAVECAL}")

    ref_path = pick_input_file(args.ref_fits) if args.ref_fits else None

    st07 = Path(config.ST07_WAVECAL)
    outdir = Path(args.outdir).expanduser() if args.outdir else (st07 / "qc_step07h")
    outdir.mkdir(parents=True, exist_ok=True)

    slits, waves, fluxes = load_slit_data(fits_path)
#    lam_min, lam_max = choose_window(waves, args.xlim)
    if args.xlim is not None:
        lam_min, lam_max = args.xlim
    else:
        lam_min, lam_max = choose_window(waves, args.xlim)
    lam_grid = np.linspace(lam_min, lam_max, int(args.ngrid))
    stack, kept_slits = build_stack(slits, waves, fluxes, lam_grid)

    highlight_row = None
    if args.highlight_slit:
        slit_u = args.highlight_slit.strip().upper()
        if slit_u in kept_slits:
            highlight_row = kept_slits.index(slit_u)
        else:
            print(f"Requested highlight slit not found in plotted stack: {slit_u}")

    out_png = outdir / f"{fits_path.stem}_vertical_lines"
    if ref_path is not None:
        out_png = out_png.with_name(out_png.name + "_compare")
    if args.xlim is not None:
        out_png = out_png.with_name(out_png.name + f"_{lam_min:.0f}_{lam_max:.0f}")
    out_png = out_png.with_suffix(".png")
    print(out_png)
    
    print("\n",lam_min,lam_max,"\n",args.xlim)
    
    if ref_path is None:
        vmin, vmax = robust_image_limits(stack, 5, 99)

        plt.figure(figsize=(11, 7))
        plt.imshow(
            stack,
            origin="lower",
            aspect="auto",
            extent=[lam_grid[0], lam_grid[-1], 0, stack.shape[0]],
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            cmap="gray",
        )

        for lam0 in args.lines:
            if lam_grid[0] <= lam0 <= lam_grid[-1]:
                plt.axvline(lam0, color="cyan", lw=0.8, alpha=0.8)

        if highlight_row is not None:
            plt.axhline(highlight_row + 0.5, color="red", lw=1.0, alpha=0.9)

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Slit index")
        plt.title("Step07 spectra aligned in wavelength space")
        plt.colorbar(label="normalized flux")
        plt.tight_layout()

    else:
        ref_slits, ref_waves, ref_fluxes = load_slit_data(ref_path)
        if args.xlim is None:
            ref_lam_min, ref_lam_max = choose_window(ref_waves, args.xlim)
        else:
            ref_lam_min, ref_lam_max = args.xlim
        
        # common display window across both files
        lam_lo = max(lam_min, ref_lam_min)
        lam_hi = min(lam_max, ref_lam_max)
       
        if not np.isfinite(lam_lo) or not np.isfinite(lam_hi) or lam_hi <= lam_lo:
            raise RuntimeError("No common wavelength overlap between current and reference files.")

        lam_grid = np.linspace(lam_lo, lam_hi, int(args.ngrid))

        stack, kept_slits = build_stack(slits, waves, fluxes, lam_grid)
        ref_stack, ref_kept_slits = build_stack(ref_slits, ref_waves, ref_fluxes, lam_grid)

        if kept_slits != ref_kept_slits:
            raise RuntimeError(
                "Current and reference slit order differ. "
                "Comparison mode expects matching slit sets and order."
            )

        highlight_row = None
        if args.highlight_slit:
            slit_u = args.highlight_slit.strip().upper()
            if slit_u in kept_slits:
                highlight_row = kept_slits.index(slit_u)

        diff = stack - ref_stack

        v1, v2 = robust_image_limits(stack, 5, 99)
        d1, d2 = robust_image_limits(diff, 2, 98)

        fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)

        panels = [
            (axes[0], ref_stack, "Reference", v1, v2),
            (axes[1], stack, "Current", v1, v2),
            (axes[2], diff, "Current - Reference", d1, d2),
        ]

        for ax, img, title, vmin, vmax in panels:
            im = ax.imshow(
                img,
                origin="lower",
                aspect="auto",
                extent=[lam_grid[0], lam_grid[-1], 0, img.shape[0]],
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                cmap="gray",
            )
            for lam0 in args.lines:
                if lam_grid[0] <= lam0 <= lam_grid[-1]:
                    ax.axvline(lam0, color="cyan", lw=0.8, alpha=0.8)
            if highlight_row is not None:
                ax.axhline(highlight_row + 0.5, color="red", lw=1.0, alpha=0.9)
            ax.set_title(title)
            ax.set_xlabel("Wavelength (nm)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        axes[0].set_ylabel("Slit index")
        fig.suptitle("Step07 wavelength-space barcode comparison")
        plt.tight_layout()

    plt.savefig(out_png, dpi=150, bbox_inches="tight")

    if args.show_plots:
        plt.show()
    else:
        plt.close()
        
        # --- save individual panels as separate files ---
    
    base = out_png.with_suffix("")  # remove .png
    
    names = ["reference", "current", "diff"]
    images = [ref_stack, stack, diff]
    limits = [(v1, v2), (v1, v2), (d1, d2)]
    
    for name, img, (vmin, vmax) in zip(names, images, limits):
    
        plt.figure(figsize=(8, 7))
    
        plt.imshow(
            img,
            origin="lower",
            aspect="auto",
            extent=[lam_grid[0], lam_grid[-1], 0, img.shape[0]],
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            cmap="gray",
        )
    
        for lam0 in args.lines:
            if lam_grid[0] <= lam0 <= lam_grid[-1]:
                plt.axvline(lam0, color="cyan", lw=0.8, alpha=0.8)
    
        if highlight_row is not None:
            plt.axhline(highlight_row + 0.5, color="red", lw=1.0, alpha=0.9)
    
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Slit index")
        plt.title(name.capitalize())
    
        out_single = base.with_name(base.name + f"_{name}.png")
        plt.tight_layout()
        plt.savefig(out_single, dpi=150, bbox_inches="tight")
        plt.close()
    
        print("Wrote:", out_single)    
    
        print("Input :", fits_path)
        if ref_path is not None:
            print("Ref   :", ref_path)
        print("Output:", out_png)
        out_png
        print(f"Window: {lam_grid[0]:.3f} – {lam_grid[-1]:.3f} nm")
        print(f"N slits plotted: {len(kept_slits)}")
        if highlight_row is not None:
            print(f"Highlighted slit: {args.highlight_slit.strip().upper()} at row {highlight_row}")


if __name__ == "__main__":
    main()
