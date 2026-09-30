#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonical configuration helpers for the SAMI spectroscopic pipeline.

This module contains only generic pipeline conventions: repository roots,
calibration-reference locations, canonical product filenames, step-directory
names, and the ``build_config`` helper that attaches derived paths to a
reduction profile.

Target/run-specific information belongs in ``config/reductions/*.py``.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType
from typing import Iterable


# -----------------------------------------------------------------------------
# Repository and calibration roots
# -----------------------------------------------------------------------------
REPO_ROOT = Path(
    os.environ.get("SAMOS_REPO_ROOT", str(Path(__file__).resolve().parents[1]))
).resolve()

CALIB_ROOT = REPO_ROOT / "calibration"
REFERENCE_TABLES_DIR = CALIB_ROOT / "reference_tables"
REFERENCE_FILTERS_DIR = REFERENCE_TABLES_DIR / "filters"
REFERENCE_NIST_DIR = REFERENCE_TABLES_DIR / "nist_list"
REFERENCE_PHOTOMETRY_DIR = REFERENCE_TABLES_DIR / "photometry"
REFERENCE_REGIONS_DIR = REFERENCE_TABLES_DIR / "regions"
REFERENCE_WAVECAL_DIR = REFERENCE_TABLES_DIR / "wavecal"
SISI_DIR = CALIB_ROOT / "sisi"
THROUGHPUT_DIR = CALIB_ROOT / "throughput"
PRODUCTS_DIR = REPO_ROOT / "products"

# Backward-compatible constants still used by some scripts.
WAVECAL_CALIB_DIR = CALIB_ROOT / "wavecal"
THROUGHPUT_TABLE = THROUGHPUT_DIR / "throughput_total_SAMOS_SOAR_CCD.csv"


# -----------------------------------------------------------------------------
# Canonical product filenames
# -----------------------------------------------------------------------------
NAME_MASTER_BIAS = "MasterBias.fits"

# -----------------------------------------------------------------------------
# Step03.5 row-stripe / quadrant-pedestal correction defaults
# -----------------------------------------------------------------------------
ROWSTRIPE_X_LEFT_MAX = 1300
ROWSTRIPE_X_RIGHT_MIN = 2800
ROWSTRIPE_Y_SPLIT = 2056
ROWSTRIPE_ESTIMATOR = "median"
ROWSTRIPE_SMOOTH_OFFSETS = False
ROWSTRIPE_SMOOTH_WIN = 21

# -----------------------------------------------------------------------------
# Step04 trace-detection defaults
# -----------------------------------------------------------------------------
STEP04_ACTIVE_FRAC = 0.03
STEP04_ACTIVE_PAD = 20

STEP04_PROFILE_SMOOTH = 3.0
STEP04_MIN_PEAK_DIST = 18
STEP04_PEAK_PROMINENCE = 0.15
STEP04_PEAK_HEIGHT_FRAC = 0.10

STEP04_HALF_WINDOW = 18
STEP04_SIDEBAND = 6
STEP04_LOCAL_NSIG = 5.0
STEP04_MAX_WIDTH = 13
STEP04_EDGE_SHRINK = 1

STEP04_TRACE_CENTER_HW = 10
STEP04_TRACE_CENTER_JUMP = 2.0
STEP04_TRACE_CENTER_SMOOTH = 31

# Step04: traces and slit geometry
NAME_EVEN_TRACES_GEOM = "Even_traces_geometry.fits"
NAME_ODD_TRACES_GEOM = "Odd_traces_geometry.fits"
NAME_EVEN_TRACES_MASK = "Even_traces_mask.fits"
NAME_ODD_TRACES_MASK = "Odd_traces_mask.fits"
NAME_EVEN_TRACES_SLITID = "Even_traces_slitid.fits"
NAME_ODD_TRACES_SLITID = "Odd_traces_slitid.fits"
NAME_EVEN_TRACES_TABLE = "Even_traces_slit_table.csv"
NAME_ODD_TRACES_TABLE = "Odd_traces_slit_table.csv"



# Step05: pixel flats
NAME_PIXFLAT_EVEN = "PixelFlat_from_quartz_diff_EVEN.fits"
NAME_PIXFLAT_ODD = "PixelFlat_from_quartz_diff_ODD.fits"

NAME_QUARTZDIFF_EVEN = "quartz_diff_even.fits"
NAME_QUARTZDIFF_ODD = "quartz_diff_odd.fits"

NAME_ILLUM2D_EVEN = "illum2d_even.fits"
NAME_ILLUM2D_ODD = "illum2d_odd.fits"

# Step06: science products
NAME_FINAL_SCIENCE_TEMPLATE = "FinalScience_{stem}_ADUperS.fits"
NAME_SCI_PIXFLATCORR_TEMPLATE = (
    "FinalScience_{stem}_ADUperS_pixflatcorr_{parity}.fits"
)

# Step07: wavelength calibration
NAME_MASTER_ARC_DIFF = "arc_diff.fits"
NAME_MASTER_ARC_DIFF_PIXFLATCORR_CLIPPED = "arc_diff_pixflatcorr_clipped.fits"
NAME_MASTER_ARC_DIFF_PIXFLATCORR_CLIPPED_1D_SLITIT_EVEN = (
    "arc_diff_pixflatcorr_clipped_1d_slitid_EVEN.fits"
)
NAME_MASTER_ARC_DIFF_PIXFLATCORR_CLIPPED_1D_SLITIT_ODD = (
    "arc_diff_pixflatcorr_clipped_1d_slitid_ODD.fits"
)
NAME_MASTER_ARC = "arc_master.fits"
NAME_WAVESOL = "arc_master_wavesol.fits"
NAME_WAVESOL_ALL = "arc_wavesol_per_slit.fits"
NAME_SHIFT2M_TABLE = "slit_shift2m_table.csv"
NAME_ARC_1D_WAVELENGTH_ALL = "arc_1d_wavelength_all.fits"
NAME_ARC_WAVELENGTH_BASE = "arc_1d_wavelength_all.fits"
NAME_ARC_WAVELENGTH_TWEAKED = "arc_1d_wavelength_all_trial_tweak.fits"

# Step08: extracted spectra
NAME_EXTRACT1D_EVEN = "extract1d_optimal_ridge_even.fits"
NAME_EXTRACT1D_ODD = "extract1d_optimal_ridge_odd.fits"
NAME_EXTRACT1D_ALL = "extract1d_optimal_ridge_all.fits"
NAME_EXTRACT1D_WAV = "extract1d_optimal_ridge_all_wav.fits"
NAME_EXTRACT1D_ABSWAV = "extract1d_optimal_ridge_all_wav_abswav.fits"

# Step09: OH ensemble wavelength registration
NAME_OH_SHIFT_CSV = "oh_shifts.csv"
NAME_OH_SHIFT_QC_CSV = "QC_OH_BG_registration.csv"
NAME_EXTRACT1D_OHREF = "extract1d_optimal_ridge_all_wav_abswav_OHref.fits"

# Legacy Step09 products retained only for reproducibility of older reductions.
NAME_EXTRACT1D_OHCLEAN = "extract1d_optimal_ridge_all_wav_ohclean.fits"
NAME_EXTRACT1D_STEP09_ABAB = "extract1d_optimal_ridge_all_wav_step09_abab_preferred.fits"
NAME_EXTRACT1D_STEP09_CONSENSUS = (
    "extract1d_optimal_ridge_all_wav_step09_abab_preferred_consensus.fits"
)

# Step10: telluric correction
NAME_TELLURIC_TEMPLATE = "telluric_O2_template.fits"

NAME_EXTRACT1D_TELLCOR = (
    "extract1d_optimal_ridge_all_wav_abswav_OHref_tellcorr.fits"
)

# Step10c: geometry-aware relative illumination correction
NAME_EXTRACT1D_TELLCOR_ILLUMREL = (
    NAME_EXTRACT1D_TELLCOR.replace(".fits", "_illumrel.fits")
)
NAME_ILLUMREL_REFERENCE_CSV = "illumrel_reference.csv"

# Step12d: validated 24-star r>600/i/z quadratic response
NAME_STEP12_RESPONSE_CSV = (
    "validated_rtrunc_response_24_illumrel.csv"
)

# Step11: flux calibration
NAME_STEP11_RADEC = "slit_trace_radec_all.csv"
NAME_STEP11_PHOTCAT = "slit_trace_radec_skymapper_all.csv"
NAME_STEP11_SUMMARY_CSV = "Step11_fluxcal_summary.csv"
NAME_STEP11_QA_PNG = "Step11_fluxcal_QA.png"
NAME_EXTRACT1D_FLUXCAL = "extract1d_fluxcal.fits"
NAME_STEP11_CONTINUUM_SNR_CSV = "Extract1d_fluxcal_continuum_snr.csv"
NAME_STEP11_ENSEMBLE_RESPONSE_CSV = "ensemble_response.csv"
NAME_STEP11_ENSEMBLE_RESPONSE_LOO_CSV = "ensemble_response_loo.csv"
NAME_ABSCAL_SUMMARY_CSV = "extract1d_optimal_ridge_all_wav_abscal_summary.csv"
NAME_MASTER_RESPONSE_FITS = "extract1d_optimal_ridge_all_wav_master_response.fits"
NAME_QC_STEP11_GRID_PDF = "qc_step11_fluxcal_grid.pdf"
NAME_QC_STEP11_SUMMARY_PDF = "qc_step11_summary_pages.pdf"
NAME_QC_STEP11_RESPONSE_PDF = "qc_step11_response_summary.pdf"

# Backward-compatible Step11 aliases.
NAME_FLUXCAL_SUMMARY_CSV = NAME_STEP11_SUMMARY_CSV
NAME_STEP11_QAPLOT = NAME_STEP11_QA_PNG

# Step12: final calibration / stellar-response refinement
NAME_EXTRACT1D_STEP12_INPUT = NAME_EXTRACT1D_FLUXCAL
NAME_EXTRACT1D_FINALCAL = "extract1d_finalcal.fits"
NAME_EXTRACT1D_FINALCAL_STELLARRESP = NAME_EXTRACT1D_FINALCAL
NAME_STEP12_FINAL_MOSAIC_PDF = "qc_step12_final_mosaic.pdf"
NAME_STEP12_ILLUM_PROFILE_EVEN = "illum1d_profile_even.fits"
NAME_STEP12_ILLUM_PROFILE_ODD = "illum1d_profile_odd.fits"
NAME_STEP12_MASTER_RESPONSE = "step12_master_response.fits"
NAME_STEP12_QAPLOT = "Step12_finalcal_QA.png"
NAME_STEP12D_DIR = "step12d_stellar_response"
NAME_STEP12D_MASTER = "step12d_stellar_respon_response_per_slit.fits"
NAME_STEP12D_MASTER_FITS = "step12d_stellar_response_master.fits"
NAME_STEP12D_METADATA_JSON = "step12d_stellar_response_metadata.json"
NAME_STEP12D_PER_SLIT = "step12d_stellar_response_per_slit.fits"
NAME_STEP12D_PER_SLIT_FITS = "step12d_stellar_response_master.fits"
NAME_STEP12D_SUMMARY = "step12d_stellar_response_summary.csv"
NAME_STEP12D_SUMMARY_CSV = NAME_STEP12D_SUMMARY
NAME_QC_STEP12D_RESPONSE_PDF = "qc_step12d_stellar_response.pdf"
NAME_QC_STEP12DE_COMPREHENSIVE_PDF = "qc_step12de_comprehensive.pdf"
NAME_QC_STEP12_SUMMARY_PDF = "qc_step12_summary.pdf"

# Backward-compatible module-level aliases.
STEP12D_MASTER_RESPONSE = NAME_STEP12D_MASTER_FITS
EXTRACT1D_FINALCAL_STELLARRESP = "extract1d_finalcal_stellarresp.fits"

# SkyMapper filter files
NAME_FILTER_G = "skymapper_g_nm.txt"
NAME_FILTER_R = "skymapper_r_nm.txt"
NAME_FILTER_I = "skymapper_i_nm.txt"
NAME_FILTER_Z = "skymapper_z_nm.txt"


# -----------------------------------------------------------------------------
# Canonical directory maps
# -----------------------------------------------------------------------------
PREPROC_SUBDIRS = {
    "00": "00_orient",
    "01": "01_bias",
    "02": "02_biascorr",
    "03": "03_crclean",
    "03.5": "03p5_rowstripe",
}

REDUCED_SUBDIRS = {
    "04": "04_traces",
    "05": "05_pixflat",
    "06": "06_science",
    "07": "07_wavecal",
    "08": "08_extract1d",
    "09": "09_oh_refine",
    "10": "10_telluric",
    "11": "11_fluxcal",
    "12": "12_finalcal",
}


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
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


def science_pixflatcorr_name(target_file_stem: str, parity: str) -> str:
    parity = parity.upper()
    return NAME_SCI_PIXFLATCORR_TEMPLATE.format(
        stem=target_file_stem,
        parity=parity,
    )

def science_tracecoords_name(target_file_stem: str, parity: str) -> str:
    parity = parity.upper()
    return f"FinalScience_{target_file_stem}_ADUperS_pixflatcorr_{parity}_tracecoords.fits"


def final_science_name(target_file_stem: str) -> str:
    return NAME_FINAL_SCIENCE_TEMPLATE.format(stem=target_file_stem)


def ensure_directories(dirs: Iterable[Path] | None = None) -> None:
    if dirs is None:
        return
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def _assign(profile: ModuleType, **paths) -> None:
    for name, value in paths.items():
        setattr(profile, name, value)


# -----------------------------------------------------------------------------
# Derived path builders
# -----------------------------------------------------------------------------
def build_directory_tree(profile: ModuleType) -> None:
    """Attach canonical run, target, product, and step directories."""
    profile.SAMI_ROOT = profile.RUN_ROOT / "SAMI"
    profile.NIGHT_ROOT = profile.SAMI_ROOT / profile.NIGHT_ID
    profile.RAW_DIR = profile.NIGHT_ROOT / "raw"
    profile.PREPROC_DIR = profile.NIGHT_ROOT / "preprocessed"
    profile.REGIONS_DIR = profile.NIGHT_ROOT / "regions"
    profile.NIGHT_LOGDIR = profile.NIGHT_ROOT / "logs"

    profile.TARGET_ROOT = profile.SAMI_ROOT / profile.TARGET_NAME
    profile.INPUT_DIR = profile.TARGET_ROOT / "input"

    if not hasattr(profile, "PRODUCT_ROOT"):
        profile.PRODUCT_ROOT = profile.TARGET_ROOT

    profile.TABLES_DIR = profile.PRODUCT_ROOT / "tables"
    profile.TARGET_LOGDIR = profile.PRODUCT_ROOT / "logs"
    profile.REDUCED_DIR = profile.PRODUCT_ROOT / "reduced"
    profile.QC_DIR = profile.PRODUCT_ROOT / "qc"

    _assign(
        profile,
        ST00_ORIENT=preproc_step_dir(profile.PREPROC_DIR, "00"),
        ST01_BIAS=preproc_step_dir(profile.PREPROC_DIR, "01"),
        ST02_BIASCORR=preproc_step_dir(profile.PREPROC_DIR, "02"),
        ST03_CRCLEAN=preproc_step_dir(profile.PREPROC_DIR, "03"),
        ST03P5_ROWSTRIPE=preproc_step_dir(profile.PREPROC_DIR, "03.5"),
        ST04_TRACES=reduced_step_dir(profile.REDUCED_DIR, "04"),
        ST05_PIXFLAT=reduced_step_dir(profile.REDUCED_DIR, "05"),
        ST06_SCIENCE=reduced_step_dir(profile.REDUCED_DIR, "06"),
        ST07_WAVECAL=reduced_step_dir(profile.REDUCED_DIR, "07"),
        ST08_EXTRACT1D=reduced_step_dir(profile.REDUCED_DIR, "08"),
        ST09=reduced_step_dir(profile.REDUCED_DIR, "09"),
        ST10_TELLURIC=reduced_step_dir(profile.REDUCED_DIR, "10"),
        ST11_FLUXCAL=reduced_step_dir(profile.REDUCED_DIR, "11"),
        ST12_FINALCAL=reduced_step_dir(profile.REDUCED_DIR, "12"),
    )

    # Canonical Step09 directory.
    profile.ST09_OH_REFINE = profile.ST09

    # Legacy ABAB directory retained only for reproducibility of old reductions.
    profile.ST09_ABAB = profile.REDUCED_DIR / "09_abab"


def build_reference_files(profile: ModuleType) -> None:
    """Attach calibration/reference products used by multiple steps."""
    _assign(
        profile,
        RADEC_EVEN_CSV=REFERENCE_REGIONS_DIR / getattr(
            profile, "RADEC_EVEN_NAME", "radec_Even.csv"
        ),
        RADEC_ODD_CSV=REFERENCE_REGIONS_DIR / getattr(
            profile, "RADEC_ODD_NAME", "radec_Odd.csv"
        ),
        EVEN_REG_FILE=REFERENCE_REGIONS_DIR / "Even_traces_mask_reg.fits",
        ODD_REG_FILE=REFERENCE_REGIONS_DIR / "Odd_traces_mask_reg.fits",
        FILTER_R=REFERENCE_FILTERS_DIR / NAME_FILTER_R,
        FILTER_I=REFERENCE_FILTERS_DIR / NAME_FILTER_I,
        FILTER_Z=REFERENCE_FILTERS_DIR / NAME_FILTER_Z,
        NIST_DIR=REFERENCE_NIST_DIR,
        SISI_IMAGE_FITS=SISI_DIR / profile.SISI_IMAGE_NAME,
    )

    if hasattr(profile, "WAVESHIFT_TABLE"):
        profile.MANUAL_WAVESHIFT_TABLE = REFERENCE_WAVECAL_DIR / profile.WAVESHIFT_TABLE
    else:
        profile.MANUAL_WAVESHIFT_TABLE = None

    if hasattr(profile, "SCIENCE_WAVELENGTH_OFFSET_TABLE"):
        profile.SCIENCE_WAVELENGTH_OFFSET_TABLE = (
            REFERENCE_WAVECAL_DIR / profile.SCIENCE_WAVELENGTH_OFFSET_TABLE
        )
    else:
        profile.SCIENCE_WAVELENGTH_OFFSET_TABLE = None


def build_preprocessing_products(profile: ModuleType) -> None:
    profile.MASTER_BIAS_FITS = profile.ST01_BIAS / NAME_MASTER_BIAS


def build_trace_products(profile: ModuleType) -> None:
    _assign(
        profile,
        EVEN_TRACES_GEOM=profile.ST04_TRACES / NAME_EVEN_TRACES_GEOM,
        ODD_TRACES_GEOM=profile.ST04_TRACES / NAME_ODD_TRACES_GEOM,
        EVEN_TRACES_MASK=profile.ST04_TRACES / NAME_EVEN_TRACES_MASK,
        ODD_TRACES_MASK=profile.ST04_TRACES / NAME_ODD_TRACES_MASK,
        EVEN_TRACES_SLITID=profile.ST04_TRACES / NAME_EVEN_TRACES_SLITID,
        ODD_TRACES_SLITID=profile.ST04_TRACES / NAME_ODD_TRACES_SLITID,
        EVEN_TRACES_TABLE=profile.ST04_TRACES / NAME_EVEN_TRACES_TABLE,
        ODD_TRACES_TABLE=profile.ST04_TRACES / NAME_ODD_TRACES_TABLE,
    )


def build_pixflat_products(profile: ModuleType) -> None:
    _assign(
        profile,
        PIXFLAT_EVEN=profile.ST05_PIXFLAT / NAME_PIXFLAT_EVEN,
        PIXFLAT_ODD=profile.ST05_PIXFLAT / NAME_PIXFLAT_ODD,

        ILLUM2D_EVEN=profile.ST05_PIXFLAT / NAME_ILLUM2D_EVEN,
        ILLUM2D_ODD=profile.ST05_PIXFLAT / NAME_ILLUM2D_ODD,

        QUARTZDIFF_EVEN=profile.ST05_PIXFLAT / NAME_QUARTZDIFF_EVEN,
        QUARTZDIFF_ODD=profile.ST05_PIXFLAT / NAME_QUARTZDIFF_ODD,
    )

def build_science_products(profile: ModuleType) -> None:
    _assign(
        profile,
        FINAL_SCIENCE=(
            profile.ST06_SCIENCE
            / final_science_name(profile.TARGET_FILE_STEM)
        ),

        SCI_EVEN_PIXFLATCORR=(
            profile.ST06_SCIENCE
            / science_pixflatcorr_name(profile.TARGET_FILE_STEM, "EVEN")
        ),
        SCI_ODD_PIXFLATCORR=(
            profile.ST06_SCIENCE
            / science_pixflatcorr_name(profile.TARGET_FILE_STEM, "ODD")
        ),

        SCI_EVEN_TRACECOORDS=(
            profile.ST06_SCIENCE
            / science_tracecoords_name(profile.TARGET_FILE_STEM, "EVEN")
        ),
        SCI_ODD_TRACECOORDS=(
            profile.ST06_SCIENCE
            / science_tracecoords_name(profile.TARGET_FILE_STEM, "ODD")
        ),
    )

def build_wavecal_products(profile: ModuleType) -> None:
    _assign(
        profile,
        MASTER_ARC_DIFF=profile.ST07_WAVECAL / NAME_MASTER_ARC_DIFF,
        MASTER_ARC_DIFF_PIXFLATCORR_CLIPPED=(
            profile.ST07_WAVECAL / NAME_MASTER_ARC_DIFF_PIXFLATCORR_CLIPPED
        ),
        MASTER_ARC_FITS=profile.ST07_WAVECAL / NAME_MASTER_ARC,
        WAVESOL_FITS=profile.ST07_WAVECAL / NAME_WAVESOL,
        WAVESOL_ALL_FITS=profile.ST07_WAVECAL / NAME_WAVESOL_ALL,
        SHIFT2M_TABLE=profile.ST07_WAVECAL / NAME_SHIFT2M_TABLE,
        ARC_1D_WAVELENGTH_ALL=profile.ST07_WAVECAL / NAME_ARC_1D_WAVELENGTH_ALL,
        ARC_WAVELENGTH_BASE=profile.ST07_WAVECAL / NAME_ARC_WAVELENGTH_BASE,
        ARC_WAVELENGTH_TWEAKED=profile.ST07_WAVECAL / NAME_ARC_WAVELENGTH_TWEAKED,
    )
    profile.ARC_WAVELENGTH_ACTIVE = (
        profile.ARC_WAVELENGTH_TWEAKED
        if profile.ARC_WAVELENGTH_TWEAKED.exists()
        else profile.ARC_WAVELENGTH_BASE
    )


def build_extraction_products(profile: ModuleType) -> None:
    _assign(
        profile,
        EXTRACT1D_EVEN=profile.ST08_EXTRACT1D / NAME_EXTRACT1D_EVEN,
        EXTRACT1D_ODD=profile.ST08_EXTRACT1D / NAME_EXTRACT1D_ODD,
        EXTRACT1D_ALL=profile.ST08_EXTRACT1D / NAME_EXTRACT1D_ALL,
        EXTRACT1D_WAV=profile.ST08_EXTRACT1D / NAME_EXTRACT1D_WAV,
        EXTRACT1D_ABSWAV=profile.ST08_EXTRACT1D / NAME_EXTRACT1D_ABSWAV,
    )


def build_oh_products(profile: ModuleType) -> None:
    _assign(
        profile,
        # Legacy ABAB products retained only for reproducibility.
        EXTRACT1D_STEP09_ABAB=profile.ST09_ABAB / NAME_EXTRACT1D_STEP09_ABAB,
        EXTRACT1D_STEP09_CONSENSUS=profile.ST09_ABAB / NAME_EXTRACT1D_STEP09_CONSENSUS,

        # Active Step09 products.
        OH_SHIFT_CSV=profile.ST09 / NAME_OH_SHIFT_CSV,
        EXTRACT1D_OHREF=profile.ST09 / NAME_EXTRACT1D_OHREF,
    )

    # Compatibility alias only.  The active pipeline performs no residual-sky
    # cleanup after OH wavelength registration.
    profile.EXTRACT1D_OHCLEAN = profile.EXTRACT1D_OHREF


def build_telluric_products(profile: ModuleType) -> None:
    _assign(
        profile,
        TELLURIC_TEMPLATE=profile.ST10_TELLURIC / NAME_TELLURIC_TEMPLATE,
        EXTRACT1D_TELLCOR=profile.ST10_TELLURIC / NAME_EXTRACT1D_TELLCOR,
    )


def build_fluxcal_products(profile: ModuleType) -> None:
    _assign(
        profile,
        EXTRACT1D_TELLCOR_ILLUMREL=(
            profile.ST10_TELLURIC
            / NAME_EXTRACT1D_TELLCOR_ILLUMREL
        ),
        ILLUMREL_REFERENCE_CSV=(
            profile.ST10_TELLURIC
            / NAME_ILLUMREL_REFERENCE_CSV
        ),
        STEP11_INPUT_SPECTRA=(
            profile.ST10_TELLURIC
            / NAME_EXTRACT1D_TELLCOR_ILLUMREL
        ),
        STEP12_RESPONSE_CSV=(
            REFERENCE_TABLES_DIR
            / "spectrophotometry"
            / NAME_STEP12_RESPONSE_CSV
        ),
        STEP11_RADEC=profile.ST11_FLUXCAL / NAME_STEP11_RADEC,
        STEP11_PHOTCAT=profile.ST11_FLUXCAL / NAME_STEP11_PHOTCAT,
        EXTRACT1D_FLUXCAL=profile.ST11_FLUXCAL / NAME_EXTRACT1D_FLUXCAL,
        FLUXCAL_SUMMARY_CSV=profile.ST11_FLUXCAL / NAME_FLUXCAL_SUMMARY_CSV,
        MASTER_RESPONSE_FITS=profile.ST11_FLUXCAL / NAME_MASTER_RESPONSE_FITS,
        STEP11_SUMMARY_CSV=profile.ST11_FLUXCAL / NAME_STEP11_SUMMARY_CSV,
        STEP11_QA_PNG=profile.ST11_FLUXCAL / NAME_STEP11_QA_PNG,
        STEP11_CONTINUUM_SNR_CSV=profile.ST11_FLUXCAL / NAME_STEP11_CONTINUUM_SNR_CSV,
        STEP11_ENSEMBLE_RESPONSE_CSV=(
            profile.ST11_FLUXCAL / NAME_STEP11_ENSEMBLE_RESPONSE_CSV
        ),
        STEP11_ENSEMBLE_RESPONSE_LOO_CSV=(
            profile.ST11_FLUXCAL / NAME_STEP11_ENSEMBLE_RESPONSE_LOO_CSV
        ),
    )


def build_finalcal_products(profile: ModuleType) -> None:
    profile.QC12_DIR = profile.QC_DIR / "12_finalcal"
    profile.STEP12D_DIR = profile.ST12_FINALCAL / NAME_STEP12D_DIR

    _assign(
        profile,
        EXTRACT1D_STEP12_INPUT=profile.EXTRACT1D_FLUXCAL,
        EXTRACT1D_FINALCAL=profile.ST12_FINALCAL / NAME_EXTRACT1D_FINALCAL,
        EXTRACT1D_FINALCAL_STELLARRESP=(
            profile.ST12_FINALCAL / NAME_EXTRACT1D_FINALCAL_STELLARRESP
        ),
        ILLUM_PROFILE_EVEN=profile.ST12_FINALCAL / NAME_STEP12_ILLUM_PROFILE_EVEN,
        ILLUM_PROFILE_ODD=profile.ST12_FINALCAL / NAME_STEP12_ILLUM_PROFILE_ODD,
        STEP12_MASTER_RESPONSE=profile.ST12_FINALCAL / NAME_STEP12_MASTER_RESPONSE,
        STEP12_PHOTCAT=profile.STEP11_PHOTCAT,
        STEP12D_MASTER=profile.STEP12D_DIR / NAME_STEP12D_MASTER,
        STEP12D_MASTER_FITS=profile.STEP12D_DIR / NAME_STEP12D_MASTER_FITS,
        STEP12D_METADATA_JSON=profile.STEP12D_DIR / NAME_STEP12D_METADATA_JSON,
        STEP12D_PER_SLIT=profile.STEP12D_DIR / NAME_STEP12D_PER_SLIT,
        STEP12D_PER_SLIT_FITS=profile.STEP12D_DIR / NAME_STEP12D_PER_SLIT_FITS,
        STEP12D_SUMMARY=profile.STEP12D_DIR / NAME_STEP12D_SUMMARY,
        STEP12D_SUMMARY_CSV=profile.STEP12D_DIR / NAME_STEP12D_SUMMARY_CSV,
        QC_STEP12_FINAL_MOSAIC_PDF=profile.QC12_DIR / NAME_STEP12_FINAL_MOSAIC_PDF,
        QC_STEP12_SUMMARY_PDF=profile.QC12_DIR / NAME_QC_STEP12_SUMMARY_PDF,
        QC_STEP12D_RESPONSE_PDF=profile.QC12_DIR / NAME_QC_STEP12D_RESPONSE_PDF,
        QC_STEP12DE_COMPREHENSIVE_PDF=profile.QC12_DIR / NAME_QC_STEP12DE_COMPREHENSIVE_PDF,
    )


def validate_profile(profile: ModuleType) -> None:
    """Check that the derived profile exposes the minimum pipeline contract."""
    required = [
        "EXTRACT1D_WAV",
        "EXTRACT1D_OHCLEAN",
        "EXTRACT1D_TELLCOR",
    ]
    for name in required:
        if not hasattr(profile, name):
            raise RuntimeError(f"Missing required config variable: {name}")


def build_config(profile: ModuleType) -> ModuleType:
    """
    Attach derived directories and canonical product paths to a reduction profile.

    The profile supplies editable reduction inputs. This function supplies only
    deterministic paths and backward-compatible aliases required by current
    pipeline scripts.
    """
    build_directory_tree(profile)
    build_reference_files(profile)
    build_preprocessing_products(profile)
    build_trace_products(profile)
    build_pixflat_products(profile)
    build_science_products(profile)
    build_wavecal_products(profile)
    build_extraction_products(profile)
    build_oh_products(profile)
    build_telluric_products(profile)
    build_fluxcal_products(profile)
    build_finalcal_products(profile)
    validate_profile(profile)
    return profile
