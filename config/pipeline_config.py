#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(
    os.environ.get(
        "SAMOS_REPO_ROOT",
        str(Path(__file__).resolve().parents[1])
    )
).resolve()

CALIB_ROOT = REPO_ROOT / "calibration"
FILTERS_DIR = CALIB_ROOT / "filters"
REFERENCE_TABLES_DIR = CALIB_ROOT / "reference_tables"
THROUGHPUT_DIR = CALIB_ROOT / "throughput"
WAVECAL_CALIB_DIR = CALIB_ROOT / "wavecal"
SISI_DIR = CALIB_ROOT / "sisi"

THROUGHPUT_TABLE = THROUGHPUT_DIR / "throughput_total_SAMOS_SOAR_CCD.csv"

# -----------------------------------------------------------------------------
# Canonical pipeline product filenames
# -----------------------------------------------------------------------------
# These are GENERIC pipeline conventions.
#
# They should NOT depend on the target name. The target-specific config files
# (for example run8_dolidze25.py, run8_dolidze26.py, etc.) should build the
# full paths by combining these standard filenames with the target-specific
# step directories such as ST08_EXTRACT1D, ST09_OH_REFINE, ST10_TELLURIC, etc.
#
# Example:
#   EXTRACT1D_OHCLEAN = ST09_OH_REFINE / NAME_EXTRACT1D_OHCLEAN
#
# This keeps the naming convention in one central place while still allowing
# each target profile to resolve to its own filesystem location.
# -----------------------------------------------------------------------------
NAME_MASTER_BIAS = "MasterBias.fits"

# Step04
NAME_EVEN_TRACES_GEOM = "Even_traces_geometry.fits"
NAME_ODD_TRACES_GEOM = "Odd_traces_geometry.fits"
NAME_EVEN_TRACES_MASK = "Even_traces_mask.fits"
NAME_ODD_TRACES_MASK = "Odd_traces_mask.fits"
NAME_EVEN_TRACES_SLITID = "Even_traces_slitid.fits"
NAME_ODD_TRACES_SLITID = "Odd_traces_slitid.fits"
NAME_EVEN_TRACES_TABLE = "Even_traces_slit_table.csv"
NAME_ODD_TRACES_TABLE = "Odd_traces_slit_table.csv"

# Step05
NAME_PIXFLAT_EVEN = "PixelFlat_from_quartz_diff_EVEN.fits"
NAME_PIXFLAT_ODD = "PixelFlat_from_quartz_diff_ODD.fits"

# Step06
NAME_FINAL_SCIENCE_TEMPLATE = "FinalScience_{stem}_ADUperS.fits"

# Step07
NAME_MASTER_ARC_DIFF = "ArcDiff_036.arc_biascorr_cr_minus_037.arc_biascorr_cr.fits"
NAME_MASTER_ARC = "arc_master.fits"
NAME_WAVESOL = "arc_master_wavesol.fits"
NAME_WAVESOL_ALL = "arc_wavesol_per_slit.fits"
NAME_SHIFT2M_TABLE = "slit_shift2m_table.csv"
NAME_ARC_1D_WAVELENGTH_ALL = "arc_1d_wavelength_all.fits"

# Step08
NAME_EXTRACT1D_EVEN = "extract1d_optimal_ridge_even.fits"
NAME_EXTRACT1D_ODD = "extract1d_optimal_ridge_odd.fits"
NAME_EXTRACT1D_ALL = "extract1d_optimal_ridge_all.fits"
NAME_EXTRACT1D_WAV = "extract1d_optimal_ridge_all_wav.fits"

# Step09 = OH refine
NAME_OH_SHIFT_CSV = "oh_shift_table.csv"
NAME_OH_SHIFT_QC_CSV = "QC_OH_BG_registration.csv"
NAME_EXTRACT1D_OHCLEAN = "extract1d_optimal_ridge_all_wav_ohclean.fits"

# Step10 = telluric
NAME_TELLURIC_TEMPLATE = "telluric_O2_template.fits"
NAME_EXTRACT1D_TELLCOR = "extract1d_optimal_ridge_all_wav_ohclean_tellcorr.fits"

# Step11 = flux calibration
NAME_STEP11_RADEC = "slit_trace_radec_all.csv"
NAME_STEP11_PHOTCAT = "slit_trace_radec_skymapper_all.csv"
NAME_STEP11_QAPLOT = "Step11_fluxcal_QA.png"
NAME_EXTRACT1D_FLUXCAL = "Extract1D_fluxcal.fits"
NAME_FLUXCAL_SUMMARY_CSV = "step11_fluxcal_summary.csv"


PREPROC_SUBDIRS = {
    "00": "00_orient",
    "01": "01_bias",
    "02": "02_biascorr",
    "03": "03_crclean",
    "03.5": "03p5_rowstripe",
}

# Canonical science-stage directory map.
# IMPORTANT:
# - Step09 = OH refine
# - Step10 = telluric
REDUCED_SUBDIRS = {
    "04": "04_traces",
    "05": "05_pixflat",
    "06": "06_science",
    "07": "07_wavecal",
    "08": "08_extract1d",
    "09": "09_oh_refine",
    "10": "10_telluric",
    "11": "11_fluxcal",
}


def as_path(x) -> Path:
    return x if isinstance(x, Path) else Path(x)


def require_exists(pathlike, label: str | None = None) -> Path:
    p = as_path(pathlike)
    if not p.exists():
        name = label or str(p)
        raise FileNotFoundError(f"Required path does not exist: {name} -> {p}")
    return p


def preproc_step_dir(preproc_dir: Path, step_code: str) -> Path:
    return preproc_dir / PREPROC_SUBDIRS[step_code]


def reduced_step_dir(reduced_dir: Path, step_code: str) -> Path:
    return reduced_dir / REDUCED_SUBDIRS[step_code]


def qc_step_dir(qc_root: Path, step_tag: str) -> Path:
    return qc_root / step_tag


def science_tracecoords_name(target_file_stem: str, parity: str) -> str:
    parity = parity.upper()
    return f"FinalScience_{target_file_stem}_ADUperS_pixflatcorr_clipped_{parity}_tracecoords.fits"

def final_science_name(target_file_stem: str) -> str:
    """
    Canonical Step06 final science filename for a target.

    Notes
    -----
    The filename pattern is pipeline-generic; only the target stem varies.
    """
    return NAME_FINAL_SCIENCE_TEMPLATE.format(stem=target_file_stem)

def ensure_directories(dirs) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
