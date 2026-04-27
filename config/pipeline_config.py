#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path



# -----------------------------------------------------------------------------
#=== CANONICAL CONSTANTS ====
# -----------------------------------------------------------------------------
REPO_ROOT = Path(
    os.environ.get(
        "SAMOS_REPO_ROOT",
        str(Path(__file__).resolve().parents[1])
    )
).resolve()

CALIB_ROOT = REPO_ROOT / "calibration"
#
REFERENCE_TABLES_DIR = CALIB_ROOT / "reference_tables"
##
REFERENCE_FILTERS_DIR = REFERENCE_TABLES_DIR / "filters"
REFERENCE_NIST_DIR = REFERENCE_TABLES_DIR / "nist_list"
REFERENCE_PHOTOMETRY_DIR = REFERENCE_TABLES_DIR / "photometry"
#
SISI_DIR = CALIB_ROOT / "sisi"
#
THROUGHPUT_DIR = CALIB_ROOT / "throughput"

# LEGACY, to be remove
WAVECAL_CALIB_DIR = CALIB_ROOT / "wavecal"
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

# -----------------------------------------------------------------------------
# Step04
NAME_EVEN_TRACES_GEOM = "Even_traces_geometry.fits"
NAME_ODD_TRACES_GEOM = "Odd_traces_geometry.fits"
NAME_EVEN_TRACES_MASK = "Even_traces_mask.fits"
NAME_ODD_TRACES_MASK = "Odd_traces_mask.fits"
NAME_EVEN_TRACES_SLITID = "Even_traces_slitid.fits"
NAME_ODD_TRACES_SLITID = "Odd_traces_slitid.fits"
NAME_EVEN_TRACES_TABLE = "Even_traces_slit_table.csv"
NAME_ODD_TRACES_TABLE = "Odd_traces_slit_table.csv"

# -----------------------------------------------------------------------------
# Step05
NAME_PIXFLAT_EVEN = "PixelFlat_from_quartz_diff_EVEN.fits"
NAME_PIXFLAT_ODD = "PixelFlat_from_quartz_diff_ODD.fits"

# -----------------------------------------------------------------------------
# Step06
NAME_FINAL_SCIENCE_TEMPLATE = "FinalScience_{stem}_ADUperS.fits"

# -----------------------------------------------------------------------------
# Step07
NAME_MASTER_ARC_DIFF = "ArcDiff_036.arc_biascorr_cr_minus_037.arc_biascorr_cr.fits"
NAME_MASTER_ARC = "arc_master.fits"
NAME_WAVESOL = "arc_master_wavesol.fits"
NAME_WAVESOL_ALL = "arc_wavesol_per_slit.fits"
NAME_SHIFT2M_TABLE = "slit_shift2m_table.csv"
NAME_ARC_1D_WAVELENGTH_ALL = "arc_1d_wavelength_all.fits"

# -----------------------------------------------------------------------------
# Step08
NAME_EXTRACT1D_EVEN = "extract1d_optimal_ridge_even.fits"
NAME_EXTRACT1D_ODD = "extract1d_optimal_ridge_odd.fits"
NAME_EXTRACT1D_ALL = "extract1d_optimal_ridge_all.fits"
NAME_EXTRACT1D_WAV = "extract1d_optimal_ridge_all_wav.fits"

# -----------------------------------------------------------------------------
# Step09 = OH refine
NAME_OH_SHIFT_CSV = "oh_shift_table.csv"
NAME_OH_SHIFT_QC_CSV = "QC_OH_BG_registration.csv"
NAME_EXTRACT1D_OHCLEAN = "extract1d_optimal_ridge_all_wav_ohclean.fits"

# -----------------------------------------------------------------------------
# Step10 = telluric
NAME_TELLURIC_TEMPLATE = "telluric_O2_template.fits"
NAME_EXTRACT1D_TELLCOR = "extract1d_optimal_ridge_all_wav_ohclean_tellcorr.fits"

# -----------------------------------------------------------------------------
# Step11 = flux calibration
NAME_STEP11_RADEC = "slit_trace_radec_all.csv"
NAME_STEP11_PHOTCAT = "slit_trace_radec_skymapper_all.csv"

# Main Step11 science products
NAME_EXTRACT1D_FLUXCAL = "Extract1D_fluxcal.fits"
NAME_FLUXCAL_SUMMARY_CSV = "Step11_fluxcal_summary.csv"
NAME_STEP11_QAPLOT = "Step11_fluxcal_QA.png"

# Calibration-fit diagnostics / closeout products
NAME_ABSCAL_SUMMARY_CSV = "extract1d_optimal_ridge_all_wav_abscal_summary.csv"
NAME_MASTER_RESPONSE_FITS = "extract1d_optimal_ridge_all_wav_master_response.fits"

# QC products
NAME_QC_STEP11_GRID_PDF = "qc_step11_fluxcal_grid.pdf"
NAME_QC_STEP11_SUMMARY_PDF = "qc_step11_summary_pages.pdf"
NAME_QC_STEP11_RESPONSE_PDF = "qc_step11_response_summary.pdf"

# -----------------------------------------------------------------------------
# Step12 = final calibration / illumination correction

# Step12a illumination products
NAME_ILLUM1D_RAW_EVEN = "illum1d_raw_slits_even.fits"
NAME_ILLUM1D_RAW_ODD  = "illum1d_raw_slits_odd.fits"
NAME_ILLUM1D_PROFILE_EVEN = "illum1d_profile_even.fits"
NAME_ILLUM1D_PROFILE_ODD  = "illum1d_profile_odd.fits"

# Step12b illumination-corrected spectra
NAME_EXTRACT1D_ILLUMCORR = "extract1d_optimal_ridge_all_wav_ohclean_tellcorr_illumcorr.fits"

# Step12c final refined spectra
NAME_EXTRACT1D_FINALCAL = "extract1d_optimal_ridge_all_wav_ohclean_tellcorr_illumcorr_refined_perstar_edge_matched.fits"
#NAME_EXTRACT1D_STEP12C_REFINED_EDGE = "Extract1D_fluxcal_refined_perstar_edge_matched.fits"
NAME_STEP12C_SUMMARY_CSV = "Extract1D_fluxcal_step12c_summary.csv"
NAME_STEP12C_DEBUG_CSV = "Extract1D_fluxcal_step12c_debug.csv"
NAME_STEP12C_METADATA_JSON = "Extract1D_fluxcal_step12c_metadata.json"

# Step12 summary / metadata
NAME_STEP12_SUMMARY_CSV = "Step12_finalcal_summary.csv"
NAME_STEP12_QAPLOT = "Step12_finalcal_QA.png"

# Step12 QC products
NAME_QC_STEP12_PROFILE_PDF = "qc_step12_illum_profile.pdf"
NAME_QC_STEP12_SCIENCE_PDF = "qc_step12_science_compare.pdf"
NAME_QC_STEP12A_RAW_EVEN_PDF = "qc_step12a_raw_even.pdf"
NAME_QC_STEP12A_RAW_ODD_PDF  = "qc_step12a_raw_odd.pdf"

# -----------------------------------------------------------------------------




NAME_FILTER_G = "skymapper_g_nm.txt"
NAME_FILTER_R = "skymapper_r_nm.txt"
NAME_FILTER_I = "skymapper_i_nm.txt"
NAME_FILTER_Z = "skymapper_z_nm.txt"

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
    "09": "09_abab",
    "10": "10_telluric",
    "11": "11_fluxcal",
    "12": "12_finalcal",
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

def ensure_directories(dirs=None) -> None:
    if dirs is None:
        return
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        
def build_config(profile):
    """
    Given a target-specific reduction profile, attach all derived directories
    and canonical product paths.

    The profile supplies only editable reduction inputs.
    This function supplies deterministic pipeline paths.
    """

    profile.SAMI_ROOT = profile.RUN_ROOT / "SAMI"

    profile.NIGHT_ROOT = profile.SAMI_ROOT / profile.NIGHT_ID
    profile.RAW_DIR = profile.NIGHT_ROOT / "raw"
    profile.PREPROC_DIR = profile.NIGHT_ROOT / "preprocessed"
    profile.REGIONS_DIR = profile.NIGHT_ROOT / "regions"
    profile.NIGHT_LOGDIR = profile.NIGHT_ROOT / "logs"

    profile.TARGET_ROOT = profile.SAMI_ROOT / profile.TARGET_NAME
    profile.INPUT_DIR = profile.TARGET_ROOT / "input"
    profile.TABLES_DIR = profile.TARGET_ROOT / "tables"
    profile.TARGET_LOGDIR = profile.TARGET_ROOT / "logs"
    profile.REDUCED_DIR = profile.TARGET_ROOT / "reduced"

    profile.QC_DIR = profile.REDUCED_DIR / "qc"

    # Step directories
    profile.ST00_ORIENT = preproc_step_dir(profile.PREPROC_DIR, "00")
    profile.ST01_BIAS = preproc_step_dir(profile.PREPROC_DIR, "01")
    profile.ST02_BIASCORR = preproc_step_dir(profile.PREPROC_DIR, "02")
    profile.ST03_CRCLEAN = preproc_step_dir(profile.PREPROC_DIR, "03")
    profile.ST03P5_ROWSTRIPE = preproc_step_dir(profile.PREPROC_DIR, "03.5")

    profile.ST04_TRACES = reduced_step_dir(profile.REDUCED_DIR, "04")
    profile.ST05_PIXFLAT = reduced_step_dir(profile.REDUCED_DIR, "05")
    profile.ST06_SCIENCE = reduced_step_dir(profile.REDUCED_DIR, "06")
    profile.ST07_WAVECAL = reduced_step_dir(profile.REDUCED_DIR, "07")
    profile.ST08_EXTRACT1D = reduced_step_dir(profile.REDUCED_DIR, "08")

    # Canonical Step09 name should be standardized here.
    profile.ST09 = reduced_step_dir(profile.REDUCED_DIR, "09")
    profile.ST09_OH_REFINE = profile.ST09
    profile.ST09_ABAB = profile.ST09  # temporary alias, 
    profile.ST10_TELLURIC = reduced_step_dir(profile.REDUCED_DIR, "10")
    profile.ST11_FLUXCAL = reduced_step_dir(profile.REDUCED_DIR, "11")
    profile.ST12_FINALCAL = reduced_step_dir(profile.REDUCED_DIR, "12")

    # External region products
    profile.RADEC_EVEN_CSV = profile.REGIONS_DIR / "radec_Even.csv"
    profile.RADEC_ODD_CSV = profile.REGIONS_DIR / "radec_Odd.csv"
    profile.EVEN_REG_FILE = profile.REGIONS_DIR / "Even_traces_mask_reg.fits"
    profile.ODD_REG_FILE = profile.REGIONS_DIR / "Odd_traces_mask_reg.fits"

    # Calibration references
    profile.FILTER_R = REFERENCE_FILTERS_DIR / NAME_FILTER_R
    profile.FILTER_I = REFERENCE_FILTERS_DIR / NAME_FILTER_I
    profile.FILTER_Z = REFERENCE_FILTERS_DIR / NAME_FILTER_Z
    profile.NIST_DIR = REFERENCE_NIST_DIR   # use canonical constant
    profile.SISI_IMAGE_FITS = SISI_DIR / profile.SISI_IMAGE_NAME

    # Canonical products
    profile.MASTER_BIAS_FITS = profile.ST01_BIAS / NAME_MASTER_BIAS

    profile.EVEN_TRACES_GEOM = profile.ST04_TRACES / NAME_EVEN_TRACES_GEOM
    profile.ODD_TRACES_GEOM = profile.ST04_TRACES / NAME_ODD_TRACES_GEOM
    profile.EVEN_TRACES_MASK = profile.ST04_TRACES / NAME_EVEN_TRACES_MASK
    profile.ODD_TRACES_MASK = profile.ST04_TRACES / NAME_ODD_TRACES_MASK
    profile.EVEN_TRACES_SLITID = profile.ST04_TRACES / NAME_EVEN_TRACES_SLITID
    profile.ODD_TRACES_SLITID = profile.ST04_TRACES / NAME_ODD_TRACES_SLITID
    profile.EVEN_TRACES_TABLE = profile.ST04_TRACES / NAME_EVEN_TRACES_TABLE
    profile.ODD_TRACES_TABLE = profile.ST04_TRACES / NAME_ODD_TRACES_TABLE

    profile.PIXFLAT_EVEN = profile.ST05_PIXFLAT / NAME_PIXFLAT_EVEN
    profile.PIXFLAT_ODD = profile.ST05_PIXFLAT / NAME_PIXFLAT_ODD

    profile.FINAL_SCIENCE = profile.ST06_SCIENCE / final_science_name(profile.TARGET_FILE_STEM)
    profile.SCI_EVEN_TRACECOORDS = profile.ST06_SCIENCE / science_tracecoords_name(profile.TARGET_FILE_STEM, "EVEN")
    profile.SCI_ODD_TRACECOORDS = profile.ST06_SCIENCE / science_tracecoords_name(profile.TARGET_FILE_STEM, "ODD")

    profile.MASTER_ARC_DIFF = profile.ST07_WAVECAL / NAME_MASTER_ARC_DIFF
    profile.MASTER_ARC_FITS = profile.ST07_WAVECAL / NAME_MASTER_ARC
    profile.WAVESOL_FITS = profile.ST07_WAVECAL / NAME_WAVESOL
    profile.WAVESOL_ALL_FITS = profile.ST07_WAVECAL / NAME_WAVESOL_ALL
    profile.SHIFT2M_TABLE = profile.ST07_WAVECAL / NAME_SHIFT2M_TABLE
    profile.ARC_1D_WAVELENGTH_ALL = profile.ST07_WAVECAL / NAME_ARC_1D_WAVELENGTH_ALL

    profile.EXTRACT1D_EVEN = profile.ST08_EXTRACT1D / NAME_EXTRACT1D_EVEN
    profile.EXTRACT1D_ODD = profile.ST08_EXTRACT1D / NAME_EXTRACT1D_ODD
    profile.EXTRACT1D_ALL = profile.ST08_EXTRACT1D / NAME_EXTRACT1D_ALL
    profile.EXTRACT1D_WAV = profile.ST08_EXTRACT1D / NAME_EXTRACT1D_WAV

    profile.EXTRACT1D_OHCLEAN = profile.ST09_OH_REFINE / NAME_EXTRACT1D_OHCLEAN

    profile.TELLURIC_TEMPLATE = profile.ST10_TELLURIC / NAME_TELLURIC_TEMPLATE
    profile.EXTRACT1D_TELLCOR = profile.ST10_TELLURIC / NAME_EXTRACT1D_TELLCOR

    profile.STEP11_INPUT_SPECTRA = profile.EXTRACT1D_TELLCOR
    profile.EXTRACT1D_FLUXCAL = profile.ST11_FLUXCAL / NAME_EXTRACT1D_FLUXCAL
    profile.FLUXCAL_SUMMARY_CSV = profile.ST11_FLUXCAL / NAME_FLUXCAL_SUMMARY_CSV
    profile.MASTER_RESPONSE_FITS = profile.ST11_FLUXCAL / NAME_MASTER_RESPONSE_FITS

    profile.STEP12B_INPUT_SPECTRA = profile.EXTRACT1D_FLUXCAL
    profile.ILLUM1D_PROFILE_EVEN = profile.ST12_FINALCAL / NAME_ILLUM1D_PROFILE_EVEN
    profile.ILLUM1D_PROFILE_ODD = profile.ST12_FINALCAL / NAME_ILLUM1D_PROFILE_ODD
    profile.EXTRACT1D_ILLUMCORR = profile.ST12_FINALCAL / NAME_EXTRACT1D_ILLUMCORR
    profile.EXTRACT1D_FINALCAL = profile.ST12_FINALCAL / NAME_EXTRACT1D_FINALCAL
    
    required = [
        "EXTRACT1D_WAV",
        "EXTRACT1D_OHCLEAN",
        "EXTRACT1D_TELLCOR",
    ]
    
    for name in required:
        if not hasattr(profile, name):
            raise RuntimeError(f"Missing required config variable: {name}")

    return profile        