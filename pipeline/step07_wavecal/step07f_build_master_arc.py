#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step07f — build the master arc from refined slit shifts.

This is the plumbing-updated version of the old Step07.3 script.

PURPOSE
-------
Build a master arc spectrum by aligning all slit spectra using the refined
per-slit shifts from Step07e, merging EVEN and ODD into a common master frame,
and preserving full shift provenance for later wavelength calibration.

This step keeps explicit track of:
- per-slit refined shifts
- global EVEN/ODD offset
- final shift to the master frame

INPUTS
------
- Step07c 1D arc MEFs:
    ArcDiff_*_pixflatcorr_clipped_1D_slitid_EVEN.fits
    ArcDiff_*_pixflatcorr_clipped_1D_slitid_ODD.fits

- Step07e refined shifts:
    Arc_shifts_final_EVEN.csv
    Arc_shifts_final_ODD.csv

OUTPUTS
-------
- config.MASTER_ARC_FITS
  containing:
    PRIMARY      : MASTER_MEDIAN
    EXT 1        : MASTER_MEAN
    EXT 2        : ALIGNED_STACK
    EXT 3        : COVERAGE
    EXT 4        : SLITLIST

- optional:
    Arc_shifts_final_ALL.csv

SHIFT BOOKKEEPING
-----------------
The SLITLIST table stores, for each slit:
- SHIFT_FINAL     : refined slit shift from Step07e
- SHIFT_GLOBAL    : global ODD->EVEN offset applied when merging parity sets
- SHIFT_TO_MASTER : final shift mapping that slit into the master-arc frame

RUN
---
    PYTHONPATH=. python pipeline/step07_wavecal/step07f_build_master_arc.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

import config


def norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def find_latest_or_fail(path: Path, pattern: str) -> Path:
    hits = sorted(path.glob(pattern))
    if not hits:
        sample = [p.name for p in sorted(path.glob("*"))[:60]]
        raise FileNotFoundError(
            f"No files match {pattern} in {path}\n"
            f"First items in directory:\n" + "\n".join(sample)
        )
    return hits[-1]


def list_slit_extnames(h: fits.HDUList) -> list[str]:
    out = []
    for ext in h[1:]:
        name = str(ext.header.get("EXTNAME", "")).strip()
        if name.upper().startswith("SLIT"):
            out.append(name.upper())
    return out


def slit_num(s: str) -> int:
    return int(s.replace("SLIT", ""))


def read_shifts_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read Arc_shifts_final CSV and return standardized dataframe with:
      SLIT (upper), SHIFT_FINAL (int), plus original columns preserved.
    """
    df = pd.read_csv(csv_path)
    nmap = {norm_col(c): c for c in df.columns}

    slit_col = None
    for cand in ["slit", "extname", "slitid"]:
        if cand in nmap:
            slit_col = nmap[cand]
            break
    if slit_col is None:
        raise RuntimeError(f"Missing SLIT column in {csv_path}. Columns={list(df.columns)}")

    shift_col = None
    for cand in [
        "shiftfinal",
        "shiftf",
        "shift",
        "shift0",
        "shiftvsrefpx",
        "shiftvsref",
        "dy",
        "dely",
    ]:
        if cand in nmap:
            shift_col = nmap[cand]
            break
    if shift_col is None:
        raise RuntimeError(f"Missing SHIFT column in {csv_path}. Columns={list(df.columns)}")

    df2 = df.copy()
    df2["SLIT"] = df2[slit_col].astype(str).str.upper()
    df2["SHIFT_FINAL"] = np.round(df2[shift_col].astype(float)).astype(int)
    return df2


def robust_valid_region(npix: np.ndarray, frac: float = 0.5) -> np.ndarray:
    """Rows with >= frac * median(npix>0) coverage (avoids edges)."""
    npix = np.asarray(npix, float)
    good = np.isfinite(npix) & (npix > 0)
    if not np.any(good):
        return np.zeros_like(npix, dtype=bool)
    med = np.nanmedian(npix[good])
    if not np.isfinite(med) or med <= 0:
        return good
    return good & (npix >= frac * med)


def shift_1d_with_nan(x: np.ndarray, shift: int) -> np.ndarray:
    """
    Roll a 1D array by integer shift, filling wrapped regions with NaN.
    Convention: positive shift moves features to larger index (right).
    """
    x = np.asarray(x, float)
    y = np.roll(x, shift)
    if shift > 0:
        y[:shift] = np.nan
    elif shift < 0:
        y[shift:] = np.nan
    return y


def default_arc1d_path(trace_set: str) -> Path:
    wavecal_dir = Path(config.ST07_WAVECAL).expanduser()
    if hasattr(config, "MASTER_ARC_DIFF"):
        stem = Path(config.MASTER_ARC_DIFF).stem
        p = wavecal_dir / f"{stem}_pixflatcorr_clipped_1D_slitid_{trace_set}.fits"
        if p.exists():
            return p
    hits = sorted(wavecal_dir.glob(f"*_1D_slitid_{trace_set}.fits"))
    if hits:
        return hits[-1]
    return wavecal_dir / f"ArcDiff_pixflatcorr_clipped_1D_slitid_{trace_set}.fits"


def build_aligned_stack(
    arc1d_fits: Path,
    shifts_csv: Path,
    set_name: str,
    min_coverage_frac: float = 0.5,
    edge_pad: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Returns:
      stack: (n_slits, n_pix)
      coverage: (n_pix,) number of finite contributors per pixel
      meta: list of dicts per slit in stack order
    """
    df = read_shifts_csv(shifts_csv)
    shift_map = {r.SLIT: int(r.SHIFT_FINAL) for r in df.itertuples(index=False)}

    with fits.open(arc1d_fits) as h:
        slits = list_slit_extnames(h)
        if not slits:
            raise RuntimeError(f"No SLIT* extensions found in {arc1d_fits}")

        first = slits[0]
        f0 = np.asarray(h[first].data[0], float)
        n_pix = f0.size

        stack_rows = []
        meta_rows = []

        for slit in slits:
            if slit not in shift_map:
                print(f"[WARN] No shift for {slit} in {shifts_csv.name}; skipping")
                continue

            flux = np.asarray(h[slit].data[0], float)
            npix = np.asarray(h[slit].data[1], float)

            if flux.size != n_pix:
                print(f"[WARN] {slit} length {flux.size} != {n_pix}; skipping")
                continue

            valid = robust_valid_region(npix, frac=min_coverage_frac)
            if edge_pad > 0:
                valid[:edge_pad] = False
                valid[-edge_pad:] = False

            x = flux.copy()
            x[~valid] = np.nan

            sh = int(shift_map[slit])
            x_al = shift_1d_with_nan(x, -sh)  # align to reference frame

            stack_rows.append(x_al)
            meta_rows.append({"SLIT": slit, "SET": set_name, "SHIFT_FINAL": sh})

    if not stack_rows:
        raise RuntimeError(f"No usable slits built from {arc1d_fits} with {shifts_csv}")

    stack = np.vstack(stack_rows).astype(np.float32)
    coverage = np.sum(np.isfinite(stack), axis=0).astype(np.int32)
    return stack, coverage, meta_rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include", type=str, default="EVEN,ODD",
                    help="Comma-separated sets to include: EVEN,ODD (default BOTH).")
    ap.add_argument("--min-coverage-frac", type=float, default=0.5,
                    help="Row validity threshold using npix coverage.")
    ap.add_argument("--edge-pad", type=int, default=0,
                    help="Extra invalid padding (pixels) at both ends after coverage mask.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output master FITS name (default config.MASTER_ARC_FITS).")
    ap.add_argument("--write-all-csv", action="store_true",
                    help="Also write Arc_shifts_final_ALL.csv merged table.")

    # Optional explicit inputs
    ap.add_argument("--arc1d-even", type=str, default=None)
    ap.add_argument("--arc1d-odd", type=str, default=None)
    ap.add_argument("--shifts-even", type=str, default=None)
    ap.add_argument("--shifts-odd", type=str, default=None)

    args = ap.parse_args(argv)

    wavecal_dir = Path(config.ST07_WAVECAL).expanduser()
    wavecal_dir.mkdir(parents=True, exist_ok=True)

    include = [s.strip().upper() for s in args.include.split(",") if s.strip()]
    do_even = "EVEN" in include
    do_odd = "ODD" in include

    if not do_even and not do_odd:
        raise RuntimeError("Nothing to do: --include must contain EVEN and/or ODD")

    if do_even:
        arc1d_even = Path(args.arc1d_even).expanduser() if args.arc1d_even else default_arc1d_path("EVEN")
        shifts_even = Path(args.shifts_even).expanduser() if args.shifts_even else (wavecal_dir / "Arc_shifts_final_EVEN.csv")
        if not arc1d_even.exists():
            raise FileNotFoundError(arc1d_even)
        if not shifts_even.exists():
            raise FileNotFoundError(shifts_even)

    if do_odd:
        arc1d_odd = Path(args.arc1d_odd).expanduser() if args.arc1d_odd else default_arc1d_path("ODD")
        shifts_odd = Path(args.shifts_odd).expanduser() if args.shifts_odd else (wavecal_dir / "Arc_shifts_final_ODD.csv")
        if not arc1d_odd.exists():
            raise FileNotFoundError(arc1d_odd)
        if not shifts_odd.exists():
            raise FileNotFoundError(shifts_odd)

    st_even = cov_even = meta_even = None
    st_odd = cov_odd = meta_odd = None

    if do_even:
        print("EVEN ARC1D  =", arc1d_even)
        print("EVEN SHIFTS =", shifts_even)
        st_even, cov_even, meta_even = build_aligned_stack(
            arc1d_even, shifts_even, "EVEN",
            min_coverage_frac=args.min_coverage_frac,
            edge_pad=args.edge_pad,
        )

    if do_odd:
        print("ODD  ARC1D  =", arc1d_odd)
        print("ODD  SHIFTS =", shifts_odd)
        st_odd, cov_odd, meta_odd = build_aligned_stack(
            arc1d_odd, shifts_odd, "ODD",
            min_coverage_frac=args.min_coverage_frac,
            edge_pad=args.edge_pad,
        )

    def stack_bry_median(stack):
        bry = []
        for r in stack:
            if np.all(~np.isfinite(r)):
                continue
            bry.append(int(np.nanargmax(r)))
        return float(np.median(bry)) if bry else np.nan

    def shift_stack(stack, d):
        out = np.empty_like(stack)
        for i in range(stack.shape[0]):
            out[i] = shift_1d_with_nan(stack[i], d)
        return out

    d_global = 0
    if (st_even is not None) and (st_odd is not None):
        b_even = stack_bry_median(st_even)
        b_odd = stack_bry_median(st_odd)

        if np.isfinite(b_even) and np.isfinite(b_odd):
            d_global = int(round(b_even - b_odd))
            print(f"[INFO] Global EVEN-ODD offset: {d_global:+d} px  (EVEN med={b_even:.1f}, ODD med={b_odd:.1f})")

            # Shift ODD aligned stack to match EVEN in the master frame
            st_odd = shift_stack(st_odd, d_global)
            for m in meta_odd:
                m["SHIFT_GLOBAL"] = d_global
        else:
            print("[WARN] Could not compute global EVEN/ODD offset; leaving as-is")

    stacks = []
    meta_all = []

    if st_even is not None:
        for m in meta_even:
            m["SHIFT_GLOBAL"] = 0
        stacks.append(st_even)
        meta_all += meta_even

    if st_odd is not None:
        for m in meta_odd:
            m.setdefault("SHIFT_GLOBAL", 0)
        stacks.append(st_odd)
        meta_all += meta_odd

    if not stacks:
        raise RuntimeError("No aligned stacks were built")

    stack_all = np.vstack(stacks).astype(np.float32)
    coverage_all = np.sum(np.isfinite(stack_all), axis=0).astype(np.int32)

    good = coverage_all > 0
    master_median = np.full(stack_all.shape[1], np.nan, dtype=np.float32)
    master_mean = np.full(stack_all.shape[1], np.nan, dtype=np.float32)

    master_median[good] = np.nanmedian(stack_all[:, good], axis=0).astype(np.float32)
    master_mean[good] = np.nanmean(stack_all[:, good], axis=0).astype(np.float32)

    out_fits = Path(args.out).expanduser() if args.out else Path(config.MASTER_ARC_FITS).expanduser()
    out_csv = wavecal_dir / "Arc_shifts_final_ALL.csv"

    for m in meta_all:
        m["SHIFT_TO_MASTER"] = int(-int(m["SHIFT_FINAL"]) + int(m.get("SHIFT_GLOBAL", 0)))

    slitlist = np.array(
        [(m["SLIT"], m["SET"], int(m["SHIFT_FINAL"]), int(m.get("SHIFT_GLOBAL", 0)), int(m["SHIFT_TO_MASTER"]))
         for m in meta_all],
        dtype=[("SLIT", "U16"), ("SET", "U8"), ("SHIFT_FINAL", "i4"), ("SHIFT_GLOBAL", "i4"), ("SHIFT_TO_MASTER", "i4")]
    )

    phdr = fits.Header()
    phdr.add_history("Step07f: master arc from Step07c 1D slitid MEFs")
    phdr.add_history("Aligned using Step07e final shifts; merged EVEN+ODD with global offset")
    phdr["STAGE"] = ("07f", "Pipeline stage")
    phdr["NSLITS"] = (stack_all.shape[0], "Number of slit spectra in ALIGNED_STACK")
    phdr["NPIX"] = (stack_all.shape[1], "Length of each 1D spectrum (pixels)")
    phdr["MCOVFR"] = (float(args.min_coverage_frac), "Min coverage fraction for valid rows")
    phdr["EDGEPAD"] = (int(args.edge_pad), "Extra pixels masked at both ends")
    phdr["SHFTGLOB"] = (int(d_global), "Global shift of ODD aligned stack to match EVEN master")
    if do_even:
        phdr["ARC1DEVN"] = (arc1d_even.name, "Input EVEN 1D arc file")
        phdr["SHFTEVN"] = (shifts_even.name, "Input EVEN refined shifts CSV")
    if do_odd:
        phdr["ARC1DODD"] = (arc1d_odd.name, "Input ODD 1D arc file")
        phdr["SHFTODD"] = (shifts_odd.name, "Input ODD refined shifts CSV")

    hdus = [
        fits.PrimaryHDU(master_median, header=phdr),
        fits.ImageHDU(master_mean, name="MASTER_MEAN"),
        fits.ImageHDU(stack_all, name="ALIGNED_STACK"),
        fits.ImageHDU(coverage_all.astype(np.int32), name="COVERAGE"),
        fits.BinTableHDU(slitlist, name="SLITLIST"),
    ]

    out_fits.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(out_fits, overwrite=True)
    print("Wrote:", out_fits)

    if args.write_all_csv:
        df_all = pd.DataFrame(meta_all)
        df_all.to_csv(out_csv, index=False)
        print("Wrote:", out_csv)

    bry = np.array([np.nanargmax(r) for r in stack_all if np.any(np.isfinite(r))], dtype=float)
    print("\nQC:")
    print("  stack shape:", stack_all.shape)
    if bry.size:
        print(f"  brightline median pix: {np.nanmedian(bry):.1f}")
        print(f"  brightline RMS scatter: {np.nanstd(bry):.2f} px")
    print(f"  coverage median: {np.median(coverage_all)}  min: {coverage_all.min()}  max: {coverage_all.max()}")


if __name__ == "__main__":
    main()
