#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

#=============================
#    make step09_abab_driver.py able to run one slit or all slits
#=============================
def list_slit_extensions(fits_path: Path):
    slits = []
    with fits.open(fits_path) as h:
        for ext in h[1:]:
            name = str(ext.name).strip().upper()
            if name.startswith("SLIT"):
                slits.append(name)
    return sorted(slits)


def robust_rms(x):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    if m.sum() == 0:
        return np.nan
    x = x[m]
    med = np.nanmedian(x)
    return float(np.sqrt(np.nanmedian((x - med) ** 2)))

def compute_slit_rms(fits_path, slit, colname):
    with fits.open(fits_path) as hdul:
        tab = Table(hdul[slit].data)
        arr = np.asarray(tab[colname], float)
        return robust_rms(arr)
    
def run(cmd):
    print("RUN:", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser(description="Step09 A/B/A/B driver")
    p.add_argument("--in-fits", type=Path, required=True, 
                   help="Input extracted spectra FITS for ABAB cleanup "
                    "(normally the OH-aligned 09b or 09bmanual product)")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--slit", type=str, default=None)
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--cont-script", type=Path, default=Path("pipeline/step09_oh_refine/step09_twopass_continuum_driver.py"))
    p.add_argument("--oh-script", type=Path, default=Path("pipeline/step09_oh_refine/step09_iterative_oh_line_model.py"))
    p.add_argument("--n-bright", type=int, default=50)
    p.add_argument("--n-faint", type=int, default=80)
    p.add_argument("--max-cycles", type=int, default=7)
    p.add_argument("--a1-window-nm", type=float, default=20.0)
    p.add_argument("--a1-stride-nm", type=float, default=20.0)
    p.add_argument("--a1-passes", type=int, default=2)
    p.add_argument("--summary-csv", type=Path, default=None)
    
    p.add_argument("--a2-window-nm", type=float, default=35.0)
    p.add_argument("--a2-stride-nm", type=float, default=30.0)
    p.add_argument("--a2-passes", type=int, default=1)
    return p.parse_args()

def run_one_slit(args, slit_name: str):
    slit_outdir = args.outdir / slit_name
    slit_outdir.mkdir(parents=True, exist_ok=True)

    pass1a = slit_outdir / "step09_pass1a_continuum.fits"
    pass1b = slit_outdir / "step09_pass1b_oh.fits"
    pass2a = slit_outdir / "step09_pass2a_continuum.fits"
    final  = slit_outdir / "step09_final_abab.fits"

    common_slit = ["--slit", slit_name]

    # A1
    run([
        args.python, str(args.cont_script),
        "--in-fits", str(args.in_fits),
        "--out-fits", str(pass1a),
        "--source-col", "OBJ_PRESKY",
        "--continuum-col-name", "CONTINUUM_P1",
        "--resid-col-name", "RESID_P1",
        "--window-nm", str(args.a1_window_nm),
        "--stride-nm", str(args.a1_stride_nm),
        "--passes", str(args.a1_passes),
        *common_slit,
    ])

    # B1
    run([
        args.python, str(args.oh_script),
        "--extract", str(args.in_fits),
        "--in-fits", str(pass1a),
        "--out-fits", str(pass1b),
        "--source-col", "OBJ_PRESKY",
        "--continuum-col", "CONTINUUM_P1",
        "--oh-col-name", "OH_MODEL_P1",
        "--stellar-col-name", "STELLAR_P1",
        "--resid-col-name", "RESID_POSTOH_P1",
        "--n-bright", str(args.n_bright),
        "--n-faint", str(args.n_faint),
        "--max-cycles", str(args.max_cycles),
        *common_slit,
    ])

    # A2
    run([
        args.python, str(args.cont_script),
        "--in-fits", str(pass1b),
        "--out-fits", str(pass2a),
        "--source-col", "STELLAR_P1",
        "--continuum-col-name", "CONTINUUM_P2",
        "--resid-col-name", "RESID_P2",
        "--window-nm", str(args.a2_window_nm),
        "--stride-nm", str(args.a2_stride_nm),
        "--passes", str(args.a2_passes),
        *common_slit,
    ])

    # B2
    run([
        args.python, str(args.oh_script),
        "--extract", str(args.in_fits),
        "--in-fits", str(pass2a),
        "--out-fits", str(final),
        "--source-col", "OBJ_PRESKY",
        "--continuum-col", "CONTINUUM_P2",
        "--oh-col-name", "OH_MODEL_FINAL",
        "--stellar-col-name", "STELLAR_FINAL",
        "--resid-col-name", "RESID_POSTOH_FINAL",
        "--n-bright", str(args.n_bright),
        "--n-faint", str(args.n_faint),
        "--max-cycles", str(args.max_cycles),
        *common_slit,
    ])

    rms_b1 = compute_slit_rms(pass1b, slit_name, "RESID_POSTOH_P1")
    rms_b2 = compute_slit_rms(final,  slit_name, "RESID_POSTOH_FINAL")

    if np.isfinite(rms_b2) and (rms_b2 < rms_b1):
        preferred = final
        label = "B2"
    else:
        preferred = pass1b
        label = "B1"

    preferred_path = slit_outdir / "step09_preferred.fits"
    import shutil
    shutil.copy(preferred, preferred_path)

    log_path = slit_outdir / "step09_selection.txt"
    with open(log_path, "w") as f:
        f.write(f"SLIT = {slit_name}\n")
        f.write("RULE = choose lower robust RMS residual\n")
        f.write(f"RMS_B1 = {rms_b1:.8f}\n")
        f.write(f"RMS_B2 = {rms_b2:.8f}\n")
        f.write(f"DELTA_B2_MINUS_B1 = {rms_b2 - rms_b1:.8f}\n")
        f.write(f"PREFERRED = {label}\n")
        f.write(f"FILE = {preferred_path.name}\n")
    
    return {
        "slit": slit_name,
        "rms_b1": rms_b1,
        "rms_b2": rms_b2,
        "preferred": label,
        "preferred_file": str(preferred_path),
    }



def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.slit:
        slits = [args.slit.strip().upper()]
    else:
        slits = list_slit_extensions(args.in_fits)

    rows = []
    for slit_name in slits:
        print()
        print("=" * 70)
        print("RUNNING", slit_name)
        print("=" * 70)
        rows.append(run_one_slit(args, slit_name))

    summary_path = args.summary_csv if args.summary_csv else (args.outdir / "step09_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slit", "rms_b1", "rms_b2", "preferred", "preferred_file"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
            
    # selection count report for an immediate end-of-run diagnostic 
    n_b1 = sum(1 for row in rows if row["preferred"] == "B1")
    n_b2 = sum(1 for row in rows if row["preferred"] == "B2")
    print()
    print("SELECTION COUNTS")
    print("B1:", n_b1)
    print("B2:", n_b2)            

    print()
    print("DONE")
    print("SUMMARY:", summary_path)

if __name__ == "__main__":
    main()