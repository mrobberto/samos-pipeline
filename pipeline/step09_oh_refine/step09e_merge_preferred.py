#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step09e — merge preferred Step09 ABAB slit products into one canonical MEF.

Purpose
-------
Collect the per-slit products written by step09_abab_driver.py and build a
single downstream-ready FITS file. Each slit keeps the original columns from
its selected preferred product (B1 or B2), and the script also writes stable,
canonical aliases so later steps do not need to know whether the winning model
came from pass 1 or pass 2.

Input layout
------------
A Step09 root directory containing one subdirectory per slit, for example:

    09_oh_refine/
      SLIT000/
        step09_preferred.fits
        step09_selection.txt
      SLIT002/
        step09_preferred.fits
        step09_selection.txt
      ...
      step09_summary.csv

Each step09_preferred.fits is expected to contain a single matching SLIT HDU.
Preferred products may come from either:
- B1: columns such as OH_MODEL_P1, STELLAR_P1, RESID_POSTOH_P1
- B2: columns such as OH_MODEL_FINAL, STELLAR_FINAL, RESID_POSTOH_FINAL

Output
------
A merged MEF, by default:

    extract1d_optimal_ridge_all_wav_step09_abab_preferred.fits

Per slit the script adds or refreshes these canonical columns when available:
- OH_MODEL
- STELLAR
- RESID_POSTOH
- CONTINUUM_STEP09
- STEP09_PREF   (string scalar repeated per row: B1 or B2)

It also records slit-level provenance in the extension header.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from astropy.io import fits
from astropy.table import Table

try:
    import config  # type: ignore
except Exception:  # pragma: no cover
    config = None


DEFAULT_OUTPUT_NAME = "extract1d_optimal_ridge_all_wav_step09_abab_preferred.fits"


def default_root() -> Path:
    if config is not None and hasattr(config, "ST09_OH_REFINE"):
        return Path(config.ST09_OH_REFINE)
    return Path(".")


def default_output(root: Path) -> Path:
    if config is not None and hasattr(config, "ST09_OH_REFINE"):
        return Path(config.ST09_OH_REFINE) / DEFAULT_OUTPUT_NAME
    return root / DEFAULT_OUTPUT_NAME


def norm_slit(name: str) -> str:
    name = str(name).strip().upper()
    digits = "".join(ch for ch in name if ch.isdigit())
    return f"SLIT{int(digits):03d}" if digits else name


def slit_num(name: str) -> int:
    try:
        return int(norm_slit(name).replace("SLIT", ""))
    except Exception:
        return 10**9


def list_slit_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        name = p.name.strip().upper()
        if not name.startswith("SLIT"):
            continue
        try:
            _ = slit_num(name)
        except Exception:
            continue
        out.append(p)
    return sorted(out, key=lambda p: slit_num(p.name))


def read_selection_label(slit_dir: Path) -> str:
    sel = slit_dir / "step09_selection.txt"
    if not sel.exists():
        return "UNKNOWN"
    for line in sel.read_text(errors="ignore").splitlines():
        if line.startswith("PREFERRED ="):
            return line.split("=", 1)[1].strip().upper()
    return "UNKNOWN"


def add_or_replace_column(tab: Table, name: str, values: Iterable) -> None:
    arr = np.asarray(values)
    if name in tab.colnames:
        tab[name] = arr
    else:
        tab[name] = arr


def choose_first(tab: Table, candidates: list[str]) -> Optional[str]:
    names = {c.upper(): c for c in tab.colnames}
    for cand in candidates:
        if cand.upper() in names:
            return names[cand.upper()]
    return None


def broadcast_string(value: str, n: int, width: int = 16) -> np.ndarray:
    width = max(width, len(value), 1)
    return np.array([value] * n, dtype=f"U{width}")


def load_preferred_hdu(slit_dir: Path) -> tuple[str, fits.BinTableHDU, str]:
    slit = norm_slit(slit_dir.name)
    preferred = slit_dir / "step09_preferred.fits"
    if not preferred.exists():
        raise FileNotFoundError(f"{slit}: missing {preferred}")

    label = read_selection_label(slit_dir)

    with fits.open(preferred) as hdul:
        if slit in hdul:
            hdu = hdul[slit].copy()
        else:
            found = None
            for ext in hdul[1:]:
                if norm_slit(ext.name) == slit:
                    found = ext.copy()
                    break
            if found is None:
                raise KeyError(f"{slit}: matching HDU not found in {preferred.name}")
            hdu = found

    if not isinstance(hdu, fits.BinTableHDU):
        raise TypeError(f"{slit}: expected BinTableHDU, got {type(hdu).__name__}")

    return slit, hdu, label


def canonicalize_step09_columns(tab: Table, preferred_label: str) -> dict[str, str]:
    chosen: dict[str, str] = {}

    pref = preferred_label.upper()
    if pref == "B2":
        oh_name = choose_first(tab, ["OH_MODEL_FINAL", "OH_MODEL_P1"])
        stellar_name = choose_first(tab, ["STELLAR_FINAL", "STELLAR_P1"])
        resid_name = choose_first(tab, ["RESID_POSTOH_FINAL", "RESID_POSTOH_P1"])
        cont_name = choose_first(tab, ["CONTINUUM_P2", "CONTINUUM_P1", "CONTINUUM_FINAL"])
    else:
        oh_name = choose_first(tab, ["OH_MODEL_P1", "OH_MODEL_FINAL"])
        stellar_name = choose_first(tab, ["STELLAR_P1", "STELLAR_FINAL"])
        resid_name = choose_first(tab, ["RESID_POSTOH_P1", "RESID_POSTOH_FINAL"])
        cont_name = choose_first(tab, ["CONTINUUM_P1", "CONTINUUM_P2", "CONTINUUM_FINAL"])

    if oh_name is not None:
        add_or_replace_column(tab, "OH_MODEL", np.asarray(tab[oh_name]))
        chosen["OH_MODEL"] = oh_name
    if stellar_name is not None:
        add_or_replace_column(tab, "STELLAR", np.asarray(tab[stellar_name]))
        chosen["STELLAR"] = stellar_name
    if resid_name is not None:
        add_or_replace_column(tab, "RESID_POSTOH", np.asarray(tab[resid_name]))
        chosen["RESID_POSTOH"] = resid_name
    if cont_name is not None:
        add_or_replace_column(tab, "CONTINUUM_STEP09", np.asarray(tab[cont_name]))
        chosen["CONTINUUM_STEP09"] = cont_name

    nrow = len(tab)
    add_or_replace_column(tab, "STEP09_PREF", broadcast_string(pref, nrow, width=8))
    chosen["STEP09_PREF"] = pref
    return chosen


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge preferred Step09 ABAB slit products")
    ap.add_argument("--root", type=str, default="", help="Step09 root directory containing SLIT*/ subdirectories")
    ap.add_argument("--out", type=str, default="", help="Output merged FITS")
    ap.add_argument("--require-summary", action="store_true", help="Fail if step09_summary.csv is missing")
    ap.add_argument("--allow-missing", action="store_true", help="Skip slit dirs missing preferred FITS instead of failing")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root) if args.root else default_root()
    out_path = Path(args.out) if args.out else default_output(root)

    if not root.exists():
        raise FileNotFoundError(root)

    summary_csv = root / "step09_summary.csv"
    if args.require_summary and not summary_csv.exists():
        raise FileNotFoundError(summary_csv)

    slit_dirs = list_slit_dirs(root)
    if not slit_dirs:
        raise RuntimeError(f"No SLIT* directories found in {root}")

    hdus: list[fits.HDUBase] = []
    primary = fits.PrimaryHDU()
    phdr = primary.header
    phdr["PIPESTEP"] = "STEP09"
    phdr["STAGE"] = ("09e", "Pipeline stage")
    phdr["OUTTYPE"] = ("MERGED", "Merged preferred Step09 ABAB product")
    phdr["STEP09AB"] = (True, "Built from ABAB preferred slit products")
    phdr["INROOT"] = (str(root), "Root directory for per-slit Step09 products")
    phdr["HASCSV"] = (1 if summary_csv.exists() else 0, "step09_summary.csv present in root")
    phdr.add_history(f"Step09e merge created {datetime.utcnow().isoformat()}Z")
    phdr.add_history(f"INPUT_ROOT={root}")
    if summary_csv.exists():
        phdr.add_history(f"INPUT_SUMMARY={summary_csv}")
    hdus.append(primary)

    n_slits = 0
    n_b1 = 0
    n_b2 = 0

    for slit_dir in slit_dirs:
        slit = norm_slit(slit_dir.name)
        try:
            slit_name, hdu, preferred = load_preferred_hdu(slit_dir)
        except Exception:
            if args.allow_missing:
                continue
            raise

        tab = Table(hdu.data)
        chosen = canonicalize_step09_columns(tab, preferred)

        hdr = hdu.header.copy()
        hdr["SLITID"] = (slit_num(slit_name), "Global slit ID")
        hdr["STEP09PF"] = (preferred, "Preferred Step09 branch for this slit")
        hdr["STEP09DIR"] = (slit_dir.name, "Per-slit Step09 directory")
        hdr["STEP09SRC"] = ("step09_preferred.fits", "Merged source FITS inside slit directory")
        if "OH_MODEL" in chosen:
            hdr["S9OHCOL"] = (chosen["OH_MODEL"], "Source column mapped to OH_MODEL")
        if "STELLAR" in chosen:
            hdr["S9STCOL"] = (chosen["STELLAR"], "Source column mapped to STELLAR")
        if "RESID_POSTOH" in chosen:
            hdr["S9RSCOL"] = (chosen["RESID_POSTOH"], "Source column mapped to RESID_POSTOH")
        if "CONTINUUM_STEP09" in chosen:
            hdr["S9CTCOL"] = (chosen["CONTINUUM_STEP09"], "Source column mapped to CONTINUUM_STEP09")

        hdus.append(fits.BinTableHDU(tab, header=hdr, name=slit_name))
        n_slits += 1
        if preferred.upper() == "B1":
            n_b1 += 1
        elif preferred.upper() == "B2":
            n_b2 += 1

    if n_slits == 0:
        raise RuntimeError(f"No slit products merged from {root}")

    hdus[0].header["N_SLITS"] = (n_slits, "Number of merged preferred Step09 slit products")
    hdus[0].header["NB1"] = (n_b1, "Number of slits preferring B1")
    hdus[0].header["NB2"] = (n_b2, "Number of slits preferring B2")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(out_path, overwrite=True)

    with fits.open(out_path) as h:
        names = [norm_slit(ext.name) for ext in h[1:] if norm_slit(ext.name).startswith("SLIT")]
    if len(names) != len(set(names)):
        raise RuntimeError("Output contains duplicate SLIT EXTNAMEs.")

    print("[DONE] Wrote", out_path)
    print("[INFO] SLIT extensions:", len(names))
    if names:
        print("[INFO] first/last:", names[0], names[-1])
        print("[INFO] selection counts: B1=", n_b1, " B2=", n_b2)


if __name__ == "__main__":
    main()
